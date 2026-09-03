"""Snakebot locomotion v2 environment configuration.

Key features:
- Actor-only proprioception + temporal history (3 frames)
- Strong goal-reaching reward shaping
- Stable reset for closed-chain snake model
- High-coverage domain randomization centered around friction ~0.9
"""

from __future__ import annotations

import math

import torch

from mjlab.asset_zoo.robots import get_snakebot_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.observations import joint_pos_rel, joint_vel_rel, last_action
from mjlab.envs.mdp.rewards import joint_pos_limits
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sim import MujocoCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from . import snake_locomotion_mdp
from .goal_pose_command import GoalPoseDebugCommandCfg

SNAKE_ROOT_BODY = "m1_bottom-base-plate-v1"
_MODULE_BODIES = "m[1-5]_bottom-base-plate-v1"
_ACTUATED_JOINTS = (".*Revolute-15", ".*Revolute-16")

GOAL_RADIUS_MIN = 0.50
GOAL_RADIUS_MAX = 3.00
GOAL_REACH_THRESHOLD = 0.13
GOAL_CURRICULUM_STEPS = 2_000_000


def _robot_com_xy(robot) -> torch.Tensor:
  """Compute full-robot center-of-mass XY position for each environment."""
  body_com_xy = robot.data.body_com_pos_w[:, :, :2]
  body_mass = robot.data.model.body_mass[:, robot.indexing.body_ids]
  body_mass = torch.nan_to_num(body_mass, nan=0.0, posinf=0.0, neginf=0.0).clamp(
    min=0.0
  )
  total_mass = body_mass.sum(dim=1, keepdim=True).clamp(min=1e-8)
  com_xy = (body_com_xy * body_mass.unsqueeze(-1)).sum(dim=1) / total_mass
  return torch.nan_to_num(com_xy, nan=0.0, posinf=0.0, neginf=0.0)


def _sample_goals(env, env_ids, **kwargs):
  """Sample COM-anchored XY goals with equal directional probability."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

  robot = env.scene["robot"]
  com_xy = _robot_com_xy(robot)[env_ids]
  n = len(env_ids)

  # Curriculum: start easier (shorter goals), then expand outwards.
  progress = min(float(env.common_step_counter) / GOAL_CURRICULUM_STEPS, 1.0)
  radius_max = 0.65 + progress * (GOAL_RADIUS_MAX - 0.65)

  radius = GOAL_RADIUS_MIN + torch.rand(n, device=env.device) * (
    radius_max - GOAL_RADIUS_MIN
  )

  # Sample directions with equal probabilities across the full XY plane.
  # This explicitly balances +X/-X and both sides (quadrants).
  quadrant = torch.randint(0, 4, (n,), device=env.device)
  local_angle = torch.rand(n, device=env.device) * (math.pi / 2.0)
  angle = local_angle + quadrant * (math.pi / 2.0)

  goal_xy = com_xy.clone()
  goal_xy[:, 0] += radius * torch.cos(angle)
  goal_xy[:, 1] += radius * torch.sin(angle)

  if not hasattr(env, "_loco_goal_pos"):
    env._loco_goal_pos = torch.zeros(env.num_envs, 2, device=env.device)
    env._loco_prev_dist = torch.ones(env.num_envs, device=env.device) * radius_max

  env._loco_goal_pos[env_ids] = goal_xy
  env._loco_prev_dist[env_ids] = torch.norm(goal_xy - com_xy, dim=1)


def _gentle_push_after_settle(
  env,
  env_ids,
  velocity_range,
  settle_steps: int = snake_locomotion_mdp.SETTLE_GRACE_STEPS,
  **kwargs,
):
  """Apply mild pushes only after settle period to avoid reset-time blowups."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

  valid = env_ids[env.episode_length_buf[env_ids] >= settle_steps]
  if len(valid) == 0:
    return

  envs_mdp.push_by_setting_velocity(
    env,
    valid,
    velocity_range=velocity_range,
  )


def snakebot_locomotion_v2_flat_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create snakebot locomotion v2 config."""
  cfg = make_velocity_env_cfg()

  cfg.scene.entities = {"robot": get_snakebot_robot_cfg()}

  # Flat terrain only.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )

  # Closed-chain stability.
  cfg.sim.njmax = 12000
  cfg.sim.nconmax = 1200
  cfg.sim.mujoco = MujocoCfg(
    timestep=0.0025,
    iterations=120,
    ls_iterations=40,
    impratio=12.0,
    solver="newton",
    integrator="implicitfast",
    ccd_iterations=120,
  )

  # 20 Hz control.
  cfg.decimation = 20
  cfg.episode_length_s = 75.0

  _module_body_cfg = SceneEntityCfg("robot", body_names=(_MODULE_BODIES,))
  _actuated_joint_cfg = SceneEntityCfg("robot", joint_names=_ACTUATED_JOINTS)
  _all_geom_cfg = SceneEntityCfg("robot", geom_names=(".*_geom",))
  _inertia_body_cfg = SceneEntityCfg(
    "robot",
    body_names=("m[1-5]_bottom-base-plate-v1", "m[1-5]_top-base-plate-v1"),
  )

  actor_terms: dict[str, ObservationTermCfg] = {
    "phase_clock": ObservationTermCfg(
      func=snake_locomotion_mdp.phase_clock,
    ),
    "goal_vector": ObservationTermCfg(
      func=snake_locomotion_mdp.goal_vector_body_frame,
      noise=Unoise(n_min=-0.03, n_max=0.03),
    ),
    "heading_to_goal": ObservationTermCfg(
      func=snake_locomotion_mdp.heading_to_goal,
      noise=Unoise(n_min=-0.03, n_max=0.03),
    ),
    "joint_pos_hist": ObservationTermCfg(
      func=joint_pos_rel,
      params={"asset_cfg": _actuated_joint_cfg},
      noise=Unoise(n_min=-0.02, n_max=0.02),
      history_length=3,
      flatten_history_dim=True,
    ),
    "joint_vel_hist": ObservationTermCfg(
      func=joint_vel_rel,
      params={"asset_cfg": _actuated_joint_cfg},
      noise=Unoise(n_min=-1.5, n_max=1.5),
      delay_min_lag=0,
      delay_max_lag=1,
      history_length=3,
      flatten_history_dim=True,
    ),
    "action_hist": ObservationTermCfg(
      func=last_action,
      history_length=3,
      flatten_history_dim=True,
    ),
    "module_lin_vel_root": ObservationTermCfg(
      func=snake_locomotion_mdp.module_lin_vel_root_frame,
      params={"asset_cfg": _module_body_cfg},
      noise=Unoise(n_min=-0.4, n_max=0.4),
      delay_min_lag=0,
      delay_max_lag=2,
    ),
    "module_ang_vel_root": ObservationTermCfg(
      func=snake_locomotion_mdp.module_ang_vel_root_frame,
      params={"asset_cfg": _module_body_cfg},
      noise=Unoise(n_min=-0.4, n_max=0.4),
      delay_min_lag=0,
      delay_max_lag=2,
    ),
  }

  critic_terms: dict[str, ObservationTermCfg] = {
    **actor_terms,
    "goal_distance": ObservationTermCfg(
      func=snake_locomotion_mdp.goal_distance_obs,
    ),
    "all_body_pos_rel": ObservationTermCfg(
      func=snake_locomotion_mdp.all_body_positions_rel,
      params={"asset_cfg": _module_body_cfg},
    ),
    "all_body_lin_vel": ObservationTermCfg(
      func=snake_locomotion_mdp.all_body_lin_velocities,
      params={"asset_cfg": _module_body_cfg},
    ),
    "all_body_ang_vel": ObservationTermCfg(
      func=snake_locomotion_mdp.all_body_ang_velocities,
      params={"asset_cfg": _module_body_cfg},
    ),
    "joint_efforts": ObservationTermCfg(
      func=snake_locomotion_mdp.joint_efforts,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
  }

  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = {
    ".*Revolute-15": 0.10,
    ".*Revolute-16": 0.10,
  }

  # Goal task: keep command manager only for goal marker debug visualization.
  cfg.commands = {
    "goal_pose_debug": GoalPoseDebugCommandCfg(
      resampling_time_range=(1.0e6, 1.0e6),
      debug_vis=True,
      entity_name="robot",
      radius=0.04,
      z_offset=0.025,
      color=(1.0, 0.2, 0.2, 0.95),
    )
  }

  # Stable reset.
  cfg.events["reset_base"].func = envs_mdp.reset_scene_to_default
  cfg.events["reset_base"].params = {}
  cfg.events.pop("reset_robot_joints", None)

  cfg.events["sample_goal"] = EventTermCfg(func=_sample_goals, mode="reset")

  # Replace base DR with stronger snake-specific DR.
  cfg.events.pop("foot_friction", None)
  cfg.events.pop("encoder_bias", None)
  cfg.events.pop("base_com", None)

  cfg.events["dr_geom_friction"] = EventTermCfg(
    mode="reset",
    func=dr.geom_friction,
    params={
      "asset_cfg": _all_geom_cfg,
      "operation": "abs",
      "ranges": (0.72, 1.18),
      "shared_random": True,
      "axes": [0],
    },
  )

  cfg.events["dr_pseudo_inertia"] = EventTermCfg(
    mode="reset",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": _inertia_body_cfg,
      "alpha_range": (-0.08, 0.08),
      "d_range": (-0.04, 0.04),
      "s12_range": (-0.012, 0.012),
      "s13_range": (-0.012, 0.012),
      "s23_range": (-0.012, 0.012),
      "t_range": (-0.006, 0.006),
    },
  )

  cfg.events["dr_joint_damping"] = EventTermCfg(
    mode="reset",
    func=dr.joint_damping,
    params={
      "asset_cfg": _actuated_joint_cfg,
      "operation": "scale",
      "ranges": (0.6, 1.6),
    },
  )

  cfg.events["dr_joint_friction"] = EventTermCfg(
    mode="reset",
    func=dr.joint_friction,
    params={
      "asset_cfg": _actuated_joint_cfg,
      "operation": "abs",
      "ranges": (0.0, 0.06),
    },
  )

  cfg.events["dr_encoder_bias"] = EventTermCfg(
    mode="reset",
    func=dr.encoder_bias,
    params={
      "asset_cfg": _actuated_joint_cfg,
      "bias_range": (-0.02, 0.02),
    },
  )

  cfg.events["dr_pd_gains"] = EventTermCfg(
    mode="reset",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "scale",
      "kp_range": (0.75, 1.30),
      "kd_range": (0.75, 1.30),
    },
  )

  cfg.events["dr_effort_limits"] = EventTermCfg(
    mode="reset",
    func=dr.effort_limits,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "scale",
      "effort_limit_range": (0.85, 1.15),
    },
  )

  cfg.events["push_robot"] = EventTermCfg(
    func=_gentle_push_after_settle,
    mode="interval",
    interval_range_s=(8.0, 12.0),
    params={
      "settle_steps": snake_locomotion_mdp.SETTLE_GRACE_STEPS,
      "velocity_range": {
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.02, 0.02),
        "roll": (-0.05, 0.05),
        "pitch": (-0.05, 0.05),
        "yaw": (-0.08, 0.08),
      },
    },
  )

  cfg.rewards = {
    "progress": RewardTermCfg(
      func=snake_locomotion_mdp.progress_reward,
      weight=33.0,
    ),
    "velocity_to_goal": RewardTermCfg(
      func=snake_locomotion_mdp.velocity_towards_goal_reward,
      weight=9.0,
    ),
    # COM-directional reward is heading-agnostic and encourages direct
    # sideward travel (sidewinding) when goals are lateral.
    "com_velocity_to_goal": RewardTermCfg(
      func=snake_locomotion_mdp.com_velocity_towards_goal_reward,
      weight=4.0,
    ),
    "distance_shaping": RewardTermCfg(
      func=snake_locomotion_mdp.distance_shaping_reward,
      weight=4.5,
    ),
    "goal_reached_bonus": RewardTermCfg(
      func=snake_locomotion_mdp.goal_reached_bonus,
      params={"threshold": GOAL_REACH_THRESHOLD},
      weight=66.0,
    ),
    "alive_bonus": RewardTermCfg(
      func=snake_locomotion_mdp.alive_bonus,
      weight=0.3,
    ),
    "stagnation": RewardTermCfg(
      func=snake_locomotion_mdp.stagnation_penalty,
      weight=-0.5,
    ),
    "lateral_slip": RewardTermCfg(
      func=snake_locomotion_mdp.lateral_slip_penalty,
      weight=-0.06,
    ),
    "vertical_velocity": RewardTermCfg(
      func=snake_locomotion_mdp.vertical_velocity_penalty,
      weight=-0.08,
    ),
    "yaw_rate": RewardTermCfg(
      func=snake_locomotion_mdp.yaw_rate_penalty,
      weight=-0.06,
    ),
    "action_smoothness": RewardTermCfg(
      func=snake_locomotion_mdp.action_smoothness_penalty,
      weight=-0.015,
    ),
    "control_cost": RewardTermCfg(
      func=snake_locomotion_mdp.control_cost,
      weight=-0.003,
    ),
    "dof_pos_limits": RewardTermCfg(
      func=joint_pos_limits,
      params={"asset_cfg": _actuated_joint_cfg},
      weight=-0.002,
    ),
  }

  cfg.terminations.pop("illegal_contact", None)
  cfg.terminations.pop("fell_over", None)
  cfg.terminations["goal_reached"] = TerminationTermCfg(
    func=snake_locomotion_mdp.goal_reached_termination,
    params={"threshold": GOAL_REACH_THRESHOLD},
    time_out=False,
  )
  cfg.terminations["too_high"] = TerminationTermCfg(
    func=snake_locomotion_mdp.root_too_high,
    params={"max_height": 0.65},
  )
  cfg.terminations["too_low"] = TerminationTermCfg(
    func=snake_locomotion_mdp.root_too_low,
    params={"min_height": -0.08},
  )
  cfg.terminations["too_far"] = TerminationTermCfg(
    func=snake_locomotion_mdp.too_far_from_goal,
    params={"max_dist": GOAL_RADIUS_MAX + 0.8},
  )
  cfg.terminations["unstable_motion"] = TerminationTermCfg(
    func=snake_locomotion_mdp.unstable_motion_termination,
    params={"max_lin_speed": 3.0, "max_ang_speed": 18.0},
  )
  cfg.terminations["nan_state"] = TerminationTermCfg(
    func=envs_mdp.nan_detection,
  )

  cfg.curriculum.pop("terrain_levels", None)
  cfg.curriculum.pop("command_vel", None)

  cfg.viewer.body_name = SNAKE_ROOT_BODY
  cfg.viewer.distance = 3.2
  cfg.viewer.elevation = -22.0

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.observations["critic"].enable_corruption = False

    # Deterministic visualization in play mode.
    for event_name in (
      "push_robot",
      "dr_geom_friction",
      "dr_pseudo_inertia",
      "dr_joint_damping",
      "dr_joint_friction",
      "dr_encoder_bias",
      "dr_pd_gains",
      "dr_effort_limits",
    ):
      cfg.events.pop(event_name, None)

    cfg.curriculum = {}

  return cfg

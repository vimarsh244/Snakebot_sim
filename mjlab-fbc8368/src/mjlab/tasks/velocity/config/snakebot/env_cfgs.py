"""Snakebot chain_5 velocity environment configurations (flat terrain only)."""

import math

from mjlab.asset_zoo.robots import (
  SNAKEBOT_ACTION_SCALE,
  get_snakebot_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

# reference body for viewer and orientation rewards (first module bottom plate)
SNAKE_ROOT_BODY = "m1_bottom-base-plate-v1"


def snakebot_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Snakebot flat terrain velocity configuration (no feet; snake uses full body contact)."""
  cfg = make_velocity_env_cfg()

  cfg.scene.entities = {"robot": get_snakebot_robot_cfg()}

  # flat terrain, no terrain scan
  cfg.sim.njmax = 2000
  cfg.sim.nconmax = 500
  cfg.sim.mujoco.ccd_iterations = 100
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  del cfg.observations["critic"].terms["foot_height"]
  del cfg.observations["critic"].terms["foot_air_time"]
  del cfg.observations["critic"].terms["foot_contact"]
  del cfg.observations["critic"].terms["foot_contact_forces"]
  # critic gets same obs as actor; enable corruption so critic sees same noisy obs
  cfg.observations["critic"].enable_corruption = True

  # heavy observation noise and domain randomization for sim-to-real
  cfg.observations["actor"].terms["base_lin_vel"].noise = Unoise(n_min=-1.5, n_max=1.5)
  cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.6, n_max=0.6)
  cfg.observations["actor"].terms["projected_gravity"].noise = Unoise(n_min=-0.12, n_max=0.12)
  cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.06, n_max=0.06)
  cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-4.0, n_max=4.0)
  cfg.observations["critic"].terms["base_lin_vel"].noise = Unoise(n_min=-1.5, n_max=1.5)
  cfg.observations["critic"].terms["base_ang_vel"].noise = Unoise(n_min=-0.6, n_max=0.6)
  cfg.observations["critic"].terms["projected_gravity"].noise = Unoise(n_min=-0.12, n_max=0.12)
  cfg.observations["critic"].terms["joint_pos"].noise = Unoise(n_min=-0.06, n_max=0.06)
  cfg.observations["critic"].terms["joint_vel"].noise = Unoise(n_min=-4.0, n_max=4.0)
  cfg.observations["actor"].terms["command"].noise = Unoise(n_min=-0.08, n_max=0.08)
  cfg.observations["critic"].terms["command"].noise = Unoise(n_min=-0.08, n_max=0.08)

  # domain randomization: encoder bias and base COM
  cfg.events["encoder_bias"].params["bias_range"] = (-0.04, 0.04)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.06, 0.06),
    1: (-0.06, 0.06),
    2: (-0.05, 0.05),
  }
  # stronger pushes during training for robustness
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-0.8, 0.8),
    "y": (-0.8, 0.8),
    "z": (-0.5, 0.5),
    "roll": (-0.6, 0.6),
    "pitch": (-0.6, 0.6),
    "yaw": (-0.4, 0.4),
  }

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = SNAKEBOT_ACTION_SCALE

  cfg.viewer.body_name = SNAKE_ROOT_BODY
  cfg.viewer.distance = 2.0
  cfg.viewer.elevation = -15.0

  cfg.events["base_com"].params["asset_cfg"].body_names = (SNAKE_ROOT_BODY,)
  cfg.events.pop("foot_friction", None)

  cfg.rewards["upright"].params["asset_cfg"].body_names = (SNAKE_ROOT_BODY,)
  cfg.rewards["upright"].params["std"] = math.sqrt(0.1)  # tighter: penalize tilt more so snake doesn't topple
  cfg.rewards["upright"].weight = 2.0
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (SNAKE_ROOT_BODY,)
  cfg.rewards["pose"].params["std_standing"] = {".*Revolute.*": 0.1}
  cfg.rewards["pose"].params["std_walking"] = {".*Revolute.*": 0.3}
  cfg.rewards["pose"].params["std_running"] = {".*Revolute.*": 0.3}

  # forward-only: no angular velocity reward or command
  cfg.rewards["track_linear_velocity"].weight = 4.0
  cfg.rewards["track_angular_velocity"].weight = 0.0
  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.0
  del cfg.rewards["foot_clearance"]
  del cfg.rewards["foot_swing_height"]
  del cfg.rewards["foot_slip"]
  del cfg.rewards["soft_landing"]

  cfg.terminations.pop("illegal_contact", None)

  cfg.curriculum.pop("terrain_levels", None)

  # forward-only velocity: no yaw/lateral command so snake stays aligned and doesn't topple
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (0.1, 0.5)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.ranges.heading = (-math.pi, math.pi)
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": 0, "lin_vel_x": (0.1, 0.35), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
    {"step": 3000 * 24, "lin_vel_x": (0.15, 0.45), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
    {"step": 8000 * 24, "lin_vel_x": (0.2, 0.55), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
  ]

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.observations["critic"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (0.1, 0.5)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

  return cfg

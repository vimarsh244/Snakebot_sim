"""Snakebot goal-reaching locomotion environment configuration (flat terrain).

Task: given a random goal on the XY ground plane (0.3–0.8 m away), navigate there.

Design (informed by snakebot-gym, COBRA thesis, Naish/EELS, sensors-22-09867):
- Actor:  phase_clock + goal_vector + heading + joint_pos + joint_vel + last_action (~36 dim)
- Critic: actor obs + full module pos/vel/ang-vel + efforts (~91 dim)
- 10 Hz control (decimation=20, dt=0.005s)
- Rewards: progress + heading + goal_bonus + distance + alive (penalties minimal)

Goal sampling:
- On each episode reset, a random XY goal is placed 0.3–0.8 m from the snake.
"""

import math

import torch

from mjlab.asset_zoo.robots import (
    get_snakebot_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.observations import (
    joint_pos_rel,
    joint_vel_rel,
    last_action,
)
from mjlab.envs.mdp.rewards import joint_pos_limits
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sim import MujocoCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from . import locomotion_mdp

# ── Body references ───────────────────────────────────────────────────────────
SNAKE_ROOT_BODY = "m1_bottom-base-plate-v1"
_MODULE_BODIES = "m[1-5]_bottom-base-plate-v1"
_ACTUATED_JOINTS = (".*Revolute-15", ".*Revolute-16")

# ── Goal parameters ───────────────────────────────────────────────────────────
# Forward goals: always along world +X, close enough to reach
GOAL_RADIUS_MIN = 0.20
GOAL_RADIUS_MAX = 0.45
GOAL_REACH_THRESHOLD = 0.15


# ── Goal sampling reset callback ──────────────────────────────────────────────
def _sample_goals(env, env_ids, **kwargs):
    """Sample goals along world +X direction, slightly randomized in Y.

    Ultra-simple: goals are placed 0.3-0.8m ahead in world +X direction
    with small Y offset (±0.15m). The snake starts facing roughly +X,
    so this places goals ahead of it.
    """
    device = env.device
    n = len(env_ids)

    robot = env.scene["robot"]
    root_pos = robot.data.root_link_pos_w[env_ids, :2]

    # Goal offset: mostly +X, small Y variation
    x_offset = GOAL_RADIUS_MIN + torch.rand(n, device=device) * (
        GOAL_RADIUS_MAX - GOAL_RADIUS_MIN
    )
    y_offset = (torch.rand(n, device=device) * 2 - 1) * 0.08

    goal_xy = root_pos.clone()
    goal_xy[:, 0] += x_offset
    goal_xy[:, 1] += y_offset

    # Initialise storage on first call
    if not hasattr(env, "_loco_goal_pos"):
        env._loco_goal_pos = torch.zeros(env.num_envs, 2, device=device)
        env._loco_prev_dist = torch.ones(env.num_envs, device=device) * 0.55

    env._loco_goal_pos[env_ids] = goal_xy
    env._loco_prev_dist[env_ids] = torch.norm(goal_xy - root_pos, dim=1)


def _update_prev_distance(env, env_ids=None, **kwargs):
    """Update prev_distance for ALL envs (called each interval step by EventManager).

    Despite receiving env_ids, we update all envs because the progress reward
    needs coherent prev_dist for every env.
    """
    if hasattr(env, "_loco_prev_dist"):
        head_xy = env.scene["robot"].data.root_link_pos_w[:, :2]
        env._loco_prev_dist = torch.norm(
            env._loco_goal_pos - head_xy, dim=1
        )


# ─────────────────────────────────────────────────────────────────────────────
def snakebot_locomotion_flat_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Snakebot flat terrain goal-reaching locomotion configuration.

    Reward design — literature-grounded (COBRA, snakebot-gym, Naish/EELS, Zhang 2024):
    ┌──────────────────────────┬────────┬──────────────────────────────────────────┐
    │ Term                     │ Weight │ Purpose                                  │
    ├──────────────────────────┼────────┼──────────────────────────────────────────┤
    │ progress_reward          │ +10.0  │ Δ distance to goal (primary signal)      │
    │ heading_alignment        │  +2.0  │ Face the goal (cosine, Naish/EELS)       │
    │ goal_reached_bonus       │ +50.0  │ Sparse bonus on arrival (<25 cm)         │
    │ distance_penalty         │  -1.0  │ Persistent pull toward goal              │
    │ alive_bonus              │  +0.2  │ Keep-alive baseline                      │
    │ action_smoothness        │  -0.05 │ Smooth gait → hardware transfer          │
    │ control_cost             │  -0.02 │ Energy efficiency                        │
    │ dof_pos_limits           │  -0.5  │ Soft joint-limit guard                   │
    └──────────────────────────┴────────┴──────────────────────────────────────┘
    """
    cfg = make_velocity_env_cfg()

    # ── Robot ─────────────────────────────────────────────────────────────────
    cfg.scene.entities = {"robot": get_snakebot_robot_cfg()}

    # ── Terrain: flat plane, no height scanner ────────────────────────────────
    cfg.sim.njmax = 5000
    cfg.sim.nconmax = 500
    cfg.sim.mujoco.ccd_iterations = 100
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
    )

    # ── Solver: override base factory for closed-chain weld constraints ────────
    # See velocity env_cfgs.py comment for rationale.
    cfg.sim.mujoco = MujocoCfg(
        timestep=0.0025,
        iterations=100,
        ls_iterations=30,
        impratio=10.0,
        solver="newton",
        integrator="implicitfast",
        ccd_iterations=100,
    )

    # ── 20 Hz control frequency ─────────────────────────────────────────
    # 0.0025 s × 20 = 0.05 s / step
    cfg.decimation = 20

    # ── Episode ────────────────────────────────────────────────────────────
    cfg.episode_length_s = 30.0

    # ── Observations ──────────────────────────────────────────────────────────
    _module_body_cfg = SceneEntityCfg("robot", body_names=(_MODULE_BODIES,))
    _actuated_joint_cfg = SceneEntityCfg("robot", joint_names=_ACTUATED_JOINTS)

    actor_terms: dict = {
        "phase_clock": ObservationTermCfg(
            func=locomotion_mdp.phase_clock,
        ),
        "goal_vector": ObservationTermCfg(
            func=locomotion_mdp.goal_vector_body_frame,
            noise=Unoise(n_min=-0.02, n_max=0.02),
        ),
        "heading_to_goal": ObservationTermCfg(
            func=locomotion_mdp.heading_to_goal,
            noise=Unoise(n_min=-0.02, n_max=0.02),
        ),
        "joint_pos": ObservationTermCfg(
            func=joint_pos_rel,
            params={"asset_cfg": _actuated_joint_cfg},
            noise=Unoise(n_min=-0.03, n_max=0.03),
        ),
        "joint_vel": ObservationTermCfg(
            func=joint_vel_rel,
            params={"asset_cfg": _actuated_joint_cfg},
            noise=Unoise(n_min=-1.0, n_max=1.0),
        ),
        "actions": ObservationTermCfg(func=last_action),
    }

    critic_terms: dict = {
        **actor_terms,
        "all_body_pos_rel": ObservationTermCfg(
            func=locomotion_mdp.all_body_positions_rel,
            params={"asset_cfg": _module_body_cfg},
        ),
        "all_body_lin_vel": ObservationTermCfg(
            func=locomotion_mdp.all_body_lin_velocities,
            params={"asset_cfg": _module_body_cfg},
        ),
        "all_body_ang_vel": ObservationTermCfg(
            func=locomotion_mdp.all_body_ang_velocities,
            params={"asset_cfg": _module_body_cfg},
        ),
        "joint_efforts": ObservationTermCfg(
            func=locomotion_mdp.joint_efforts,
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

    # ── Actions ──────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = {
        ".*Revolute-15": 0.08,
        ".*Revolute-16": 0.08,
    }

    # ── Commands: NONE (goal is the command, stored as env attributes) ──────
    cfg.commands = {}

    # ── Events ──────────────────────────────────────────────────────────────
    # Goal sampling on reset
    cfg.events["sample_goal"] = EventTermCfg(
        func=_sample_goals,
        mode="reset",
    )
    # Domain randomisation is disabled for early curriculum so the snake first
    # learns a stable gait before transfer-focused perturbations are added.
    cfg.events.pop("base_com", None)
    cfg.events.pop("encoder_bias", None)
    cfg.events.pop("push_robot", None)
    cfg.events.pop("foot_friction", None)
    # Use full-scene default reset for this closed-chain model.
    # It restores all free-joint and hinge states coherently before each episode.
    cfg.events["reset_base"].func = envs_mdp.reset_scene_to_default
    cfg.events["reset_base"].params = {}
    cfg.events.pop("reset_robot_joints", None)

    # ── Rewards — literature-grounded goal-reaching ──────────────────────────
    # Based on COBRA thesis, snakebot-gym, Naish/EELS, Zhang 2024.
    cfg.rewards = {
        # === Primary shaped navigation signal ===
        # Positive when snake reduces distance to goal (most important)
        "progress_reward": RewardTermCfg(
            func=locomotion_mdp.progress_reward,
            weight=25.0,
        ),
        # Cosine alignment — face the goal to move toward it (Naish/EELS style)
        "heading_alignment": RewardTermCfg(
            func=locomotion_mdp.heading_alignment_reward,
            weight=0.5,
        ),
        # Sparse terminal bonus on reaching the goal (strong incentive)
        "goal_reached_bonus": RewardTermCfg(
            func=locomotion_mdp.goal_reached_bonus,
            params={"threshold": GOAL_REACH_THRESHOLD},
            weight=40.0,
        ),
        # Persistent pull: keeps pressure even when progress is slow
        "distance_penalty": RewardTermCfg(
            func=locomotion_mdp.distance_penalty,
            weight=3.0,
        ),
        # === Keep-alive ===
        "alive_bonus": RewardTermCfg(
            func=locomotion_mdp.alive_bonus,
            weight=0.2,
        ),
        # === Regularisation ===
        "action_smoothness": RewardTermCfg(
            func=locomotion_mdp.action_smoothness_penalty,
            weight=-0.002,
        ),
        "control_cost": RewardTermCfg(
            func=locomotion_mdp.control_cost,
            weight=-0.001,
        ),
        "dof_pos_limits": RewardTermCfg(
            func=joint_pos_limits,
            params={"asset_cfg": _actuated_joint_cfg},
            weight=0.0,
        ),
    }

    # ── Terminations ────────────────────────────────────────────────────────
    cfg.terminations.pop("illegal_contact", None)
    cfg.terminations["goal_reached"] = TerminationTermCfg(
        func=locomotion_mdp.goal_reached_termination,
        params={"threshold": GOAL_REACH_THRESHOLD},
        time_out=False,
    )
    # bad_orientation uses projected_gravity_b[:, 2] which is ~0 for a snake
    # lying flat (body Z = world -X), so the check always gives ~90deg and
    # would immediately terminate every episode. remove it.
    cfg.terminations.pop("fell_over", None)
    cfg.terminations["too_high"] = TerminationTermCfg(
        func=locomotion_mdp.root_too_high,
        params={"max_height": 4.0},
    )

    # ── Curriculum ────────────────────────────────────────────────────────────
    cfg.curriculum.pop("terrain_levels", None)
    cfg.curriculum.pop("command_vel", None)

    # ── Viewer ────────────────────────────────────────────────────────────────
    cfg.viewer.body_name = SNAKE_ROOT_BODY
    cfg.viewer.distance = 3.0
    cfg.viewer.elevation = -25.0

    # ── Play mode ─────────────────────────────────────────────────────────────
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.observations["critic"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}

    return cfg

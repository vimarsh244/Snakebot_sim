"""Snakebot goal-reaching locomotion environment configuration (flat terrain).

Task: given a random goal on the XY ground plane (1–2 m away), navigate there.

Design (informed by snakebot-gym, COBRA thesis, Naish/EELS, sensors-22-09867):
- Actor:  goal_vector + heading + joint_pos + joint_vel + last_action (~34 dim)
- Critic: actor obs + full module pos/vel/ang-vel + efforts (~89 dim)
- 10 Hz control (decimation=20, dt=0.005s)
- Rewards: progress + distance + heading + alive + goal_bonus + smoothness

Goal sampling:
- On each episode reset, a random XY goal is placed 1–2 m from the snake.
"""

import math

import torch

from mjlab.asset_zoo.robots import (
    SNAKEBOT_ACTION_SCALE,
    get_snakebot_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.observations import (
    joint_pos_rel,
    joint_vel_rel,
    last_action,
)
from mjlab.envs.mdp.rewards import joint_pos_limits
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from . import locomotion_mdp

# ── Body references ───────────────────────────────────────────────────────────
SNAKE_ROOT_BODY = "m1_bottom-base-plate-v1"
_MODULE_BODIES = "m[1-5]_bottom-base-plate-v1"

# ── Goal parameters ───────────────────────────────────────────────────────────
GOAL_RADIUS_MIN = 1.0   # metres
GOAL_RADIUS_MAX = 2.0
GOAL_REACH_THRESHOLD = 0.15  # metres


# ── Goal sampling reset callback ──────────────────────────────────────────────
def _sample_goals(env, env_ids, **kwargs):
    """Sample random goals 1–2 m from the robot's reset position.

    Called as a reset event. Stores _loco_goal_pos (B,2) and
    _loco_prev_dist (B,) on the env for use by MDP functions.
    """
    device = env.device
    n = len(env_ids)

    # Random angle and radius
    angle = torch.rand(n, device=device) * 2 * math.pi
    radius = GOAL_RADIUS_MIN + torch.rand(n, device=device) * (GOAL_RADIUS_MAX - GOAL_RADIUS_MIN)

    # Robot's current XY position (after reset)
    robot = env.scene["robot"]
    root_pos = robot.data.root_link_pos_w[env_ids, :2]

    # Goal = robot_pos + polar offset
    goal_xy = root_pos.clone()
    goal_xy[:, 0] += radius * torch.cos(angle)
    goal_xy[:, 1] += radius * torch.sin(angle)

    # Initialise storage on first call
    if not hasattr(env, "_loco_goal_pos"):
        env._loco_goal_pos = torch.zeros(env.num_envs, 2, device=device)
        env._loco_prev_dist = torch.ones(env.num_envs, device=device) * 1.5

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

    Reward design (literature-grounded):
    ┌──────────────────────────┬────────┬──────────────────────────────────────┐
    │ Term                     │ Weight │ Purpose                              │
    ├──────────────────────────┼────────┼──────────────────────────────────────┤
    │ progress_reward          │  +5.0  │ Δ distance — KEY shaped signal       │
    │ distance_penalty         │  -0.5  │ Persist pull toward goal             │
    │ heading_alignment        │  +1.0  │ Face the goal                        │
    │ goal_reached_bonus       │ +100.0 │ Sparse big bonus on arrival          │
    │ alive_bonus              │  +0.5  │ Encourage long episodes              │
    │ control_cost             │  -0.05 │ Energy efficiency                    │
    │ action_smoothness        │  -0.02 │ Sim-to-real, no jitter               │
    │ dof_pos_limits           │  -1.0  │ Prevent self-collision via limits    │
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

    # ── 10 Hz control frequency ────────────────────────────────────────────
    cfg.decimation = 20

    # ── Episode ────────────────────────────────────────────────────────────
    cfg.episode_length_s = 30.0  # 300 steps at 10 Hz

    # ── Observations ──────────────────────────────────────────────────────────
    _module_body_cfg = SceneEntityCfg("robot", body_names=(_MODULE_BODIES,))

    actor_terms: dict = {
        "goal_vector": ObservationTermCfg(
            func=locomotion_mdp.goal_vector_body_frame,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "heading_to_goal": ObservationTermCfg(
            func=locomotion_mdp.heading_to_goal,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=joint_pos_rel,
            noise=Unoise(n_min=-0.06, n_max=0.06),
        ),
        "joint_vel": ObservationTermCfg(
            func=joint_vel_rel,
            noise=Unoise(n_min=-4.0, n_max=4.0),
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
    joint_pos_action.scale = SNAKEBOT_ACTION_SCALE

    # ── Commands: NONE (goal is the command, stored as env attributes) ──────
    cfg.commands = {}

    # ── Events ──────────────────────────────────────────────────────────────
    # Goal sampling on reset
    cfg.events["sample_goal"] = EventTermCfg(
        func=_sample_goals,
        mode="reset",
    )
    # Update prev_distance each step for progress reward
    cfg.events["update_prev_dist"] = EventTermCfg(
        func=_update_prev_distance,
        mode="interval",
        interval_range_s=(0.0, 0.0),
    )

    # Domain randomisation
    cfg.events["base_com"].params["asset_cfg"].body_names = (SNAKE_ROOT_BODY,)
    cfg.events["base_com"].params["ranges"] = {
        0: (-0.06, 0.06),
        1: (-0.06, 0.06),
        2: (-0.04, 0.04),
    }
    cfg.events["encoder_bias"].params["bias_range"] = (-0.04, 0.04)
    cfg.events["push_robot"].params["velocity_range"] = {
        "x": (-0.3, 0.3),
        "y": (-0.3, 0.3),
        "z": (-0.2, 0.2),
        "roll": (-0.3, 0.3),
        "pitch": (-0.3, 0.3),
        "yaw": (-0.2, 0.2),
    }
    cfg.events.pop("foot_friction", None)

    # ── Rewards ──────────────────────────────────────────────────────────────
    cfg.rewards = {
        "progress_reward": RewardTermCfg(
            func=locomotion_mdp.progress_reward,
            weight=5.0,
        ),
        "distance_penalty": RewardTermCfg(
            func=locomotion_mdp.distance_penalty,
            weight=0.5,
        ),
        "heading_alignment": RewardTermCfg(
            func=locomotion_mdp.heading_alignment_reward,
            weight=1.0,
        ),
        "goal_reached_bonus": RewardTermCfg(
            func=locomotion_mdp.goal_reached_bonus,
            params={"threshold": GOAL_REACH_THRESHOLD},
            weight=100.0,
        ),
        "alive_bonus": RewardTermCfg(
            func=locomotion_mdp.alive_bonus,
            weight=0.5,
        ),
        "control_cost": RewardTermCfg(
            func=locomotion_mdp.control_cost,
            weight=-0.05,
        ),
        "action_smoothness": RewardTermCfg(
            func=locomotion_mdp.action_smoothness_penalty,
            weight=-0.02,
        ),
        "dof_pos_limits": RewardTermCfg(
            func=joint_pos_limits,
            weight=-1.0,
        ),
    }

    # ── Terminations ────────────────────────────────────────────────────────
    cfg.terminations.pop("illegal_contact", None)
    cfg.terminations["goal_reached"] = TerminationTermCfg(
        func=locomotion_mdp.goal_reached_termination,
        params={"threshold": GOAL_REACH_THRESHOLD},
        time_out=False,
    )
    if "fell_over" in cfg.terminations:
        cfg.terminations["fell_over"].params["limit_angle"] = math.radians(80.0)

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

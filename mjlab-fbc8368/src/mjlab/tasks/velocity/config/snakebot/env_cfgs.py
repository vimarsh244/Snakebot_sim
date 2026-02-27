"""Snakebot chain_5 velocity environment configurations (flat terrain only).

Design philosophy:
- Actor:  joint_pos + joint_vel + last_action + command + root IMU (~42 dim).
          Deployable to hardware; no privileged information.
- Critic: actor obs + per-module positions, lin-vels, ang-vels, actuator effort
          (~139 dim, privileged simulation state, zero noise).
- 10 Hz control:   decimation=20  (MuJoCo dt=0.005 s × 20 = 0.10 s / step).
- Rewards:  physics-grounded locomotion rewards (no gait assumptions):
    forward_velocity (primary) + alive_bonus + lateral/yaw penalties +
    smoothness + control_cost + joint_limits.
"""

import math

from mjlab.asset_zoo.robots import (
    SNAKEBOT_ACTION_SCALE,
    get_snakebot_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.observations import (
    builtin_sensor,
    generated_commands,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
    projected_gravity,
)
from mjlab.envs.mdp.rewards import action_rate_l2, joint_pos_limits
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import (
    UniformVelocityCommandCfg,
)
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from . import snakebot_mdp

# ── Body references ───────────────────────────────────────────────────────────
SNAKE_ROOT_BODY  = "m1_bottom-base-plate-v1"
_MODULE_BODIES   = "m[1-5]_bottom-base-plate-v1"


# ─────────────────────────────────────────────────────────────────────────────
def snakebot_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Snakebot flat terrain velocity configuration.

    Reward design (literature-grounded):
    ┌──────────────────────────┬────────┬─────────────────────────────────────┐
    │ Term                     │ Weight │ Purpose                             │
    ├──────────────────────────┼────────┼─────────────────────────────────────┤
    │ forward_velocity         │  +3.0  │ Primary locomotion signal           │
    │ alive_bonus              │  +0.5  │ Encourage long episodes             │
    │ lateral_velocity_penalty │  -0.5  │ Prevent crab-walk / spin            │
    │ yaw_rate_penalty         │  -0.2  │ Keep straight heading               │
    │ control_cost             │  -0.1  │ Energy efficiency, no launch        │
    │ action_smoothness        │  -0.05 │ Sim-to-real, no jitter              │
    │ dof_pos_limits           │  -1.0  │ Prevent self-collision via limits   │
    └──────────────────────────┴────────┴─────────────────────────────────────┘
    """
    cfg = make_velocity_env_cfg()

    # ── Robot ─────────────────────────────────────────────────────────────────
    cfg.scene.entities = {"robot": get_snakebot_robot_cfg()}

    # ── Terrain: flat plane, no height scanner ────────────────────────────────
    cfg.sim.njmax = 2000
    cfg.sim.nconmax = 500
    cfg.sim.mujoco.ccd_iterations = 100
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
    )

    # ── 10 Hz control frequency ───────────────────────────────────────────────
    cfg.decimation = 20   # 0.005 s × 20 = 0.10 s / step

    # ── Observations (fully replaced) ─────────────────────────────────────────
    _module_body_cfg = SceneEntityCfg("robot", body_names=(_MODULE_BODIES,))

    actor_terms: dict = {
        "joint_pos": ObservationTermCfg(
            func=joint_pos_rel,
            noise=Unoise(n_min=-0.06, n_max=0.06),
        ),
        "joint_vel": ObservationTermCfg(
            func=joint_vel_rel,
            noise=Unoise(n_min=-4.0, n_max=4.0),
        ),
        "actions": ObservationTermCfg(func=last_action),
        "command": ObservationTermCfg(
            func=generated_commands,
            params={"command_name": "twist"},
            noise=Unoise(n_min=-0.08, n_max=0.08),
        ),
        # Root-module IMU (hardware-deployable)
        "base_lin_vel": ObservationTermCfg(
            func=builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.6, n_max=0.6),
        ),
        "projected_gravity": ObservationTermCfg(
            func=projected_gravity,
            noise=Unoise(n_min=-0.12, n_max=0.12),
        ),
    }

    # Critic: actor + privileged full-chain state (noise stripped by enable_corruption=False)
    critic_terms: dict = {
        **actor_terms,
        # Full kinematic chain: positions relative to root
        "all_body_pos_rel": ObservationTermCfg(
            func=snakebot_mdp.all_body_positions_rel,
            params={"asset_cfg": _module_body_cfg},
        ),
        # Full chain linear velocities
        "all_body_lin_vel": ObservationTermCfg(
            func=snakebot_mdp.all_body_lin_velocities,
            params={"asset_cfg": _module_body_cfg},
        ),
        # Per-module angular velocities: simulated IMU on every module
        "all_body_ang_vel": ObservationTermCfg(
            func=snakebot_mdp.all_body_ang_velocities,
            params={"asset_cfg": _module_body_cfg},
        ),
        # Actuator forces: energy / torque load awareness
        "joint_efforts": ObservationTermCfg(
            func=snakebot_mdp.joint_efforts,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
    }

    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,    # noise during training only
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,   # privileged info: zero noise
        ),
    }

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = SNAKEBOT_ACTION_SCALE

    # ── Commands: forward-only ────────────────────────────────────────────────
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x   = (0.10, 0.30)  # forward velocity target
    twist_cmd.ranges.lin_vel_y   = (0.00, 0.00)  # no lateral commands
    twist_cmd.ranges.ang_vel_z   = (0.00, 0.00)  # no turn commands
    twist_cmd.ranges.heading     = (-math.pi, math.pi)
    twist_cmd.rel_standing_envs  = 0.10           # 10 % zero-command envs

    # ── Events / Domain randomisation ────────────────────────────────────────
    cfg.events["base_com"].params["asset_cfg"].body_names = (SNAKE_ROOT_BODY,)
    cfg.events["base_com"].params["ranges"] = {
        0: (-0.06, 0.06),
        1: (-0.06, 0.06),
        2: (-0.04, 0.04),
    }
    cfg.events["encoder_bias"].params["bias_range"] = (-0.04, 0.04)
    cfg.events["push_robot"].params["velocity_range"] = {
        "x": (-0.4, 0.4),
        "y": (-0.4, 0.4),
        "z": (-0.3, 0.3),
        "roll":  (-0.4, 0.4),
        "pitch": (-0.4, 0.4),
        "yaw":   (-0.3, 0.3),
    }
    cfg.events.pop("foot_friction", None)

    # ── Rewards (physics-grounded locomotion, no gait assumptions) ────────────
    cfg.rewards = {
        # ── Primary: forward CoM velocity in body frame ──
        # This is a LINEAR reward — always informative regardless of speed,
        # and naturally aligns with commands without an explicit cmd-tracking exp term.
        "forward_velocity": RewardTermCfg(
            func=snakebot_mdp.forward_velocity,
            weight=3.0,
        ),
        # ── Alive bonus: incentivise the episode to continue ──
        "alive_bonus": RewardTermCfg(
            func=snakebot_mdp.alive_bonus,
            weight=0.5,
        ),
        # ── Keep going straight: penalise lateral drift ──
        "lateral_velocity_penalty": RewardTermCfg(
            func=snakebot_mdp.lateral_velocity_penalty,
            weight=-0.5,
        ),
        # ── Keep straight heading: penalise yaw rate ──
        "yaw_rate_penalty": RewardTermCfg(
            func=snakebot_mdp.yaw_rate_penalty,
            weight=-0.2,
        ),
        # ── Energy efficiency: penalise large actions (prevents "launch" behaviour) ──
        "control_cost": RewardTermCfg(
            func=snakebot_mdp.control_cost,
            weight=-0.1,
        ),
        # ── Smoothness: penalise jerk / high-freq oscillations ──
        "action_smoothness": RewardTermCfg(
            func=snakebot_mdp.action_smoothness_penalty,
            weight=-0.05,
        ),
        # ── Stay in joint limits ──
        "dof_pos_limits": RewardTermCfg(
            func=joint_pos_limits,
            weight=-1.0,
        ),
    }

    # ── Terminations ──────────────────────────────────────────────────────────
    cfg.terminations.pop("illegal_contact", None)
    # Raise orientation limit: snake DOES tilt during locomotion, only terminate
    # if it fully tumbles over 120°.
    if "fell_over" in cfg.terminations:
        cfg.terminations["fell_over"].params["limit_angle"] = math.radians(120.0)

    # ── Curriculum: start slow, add lateral/yaw penalties later ──────────────
    cfg.curriculum.pop("terrain_levels", None)
    # Simple velocity curriculum (no lateral/turn commands — stay straight)
    cfg.curriculum.pop("command_vel", None)

    # ── Viewer ────────────────────────────────────────────────────────────────
    cfg.viewer.body_name = SNAKE_ROOT_BODY
    cfg.viewer.distance  = 2.0
    cfg.viewer.elevation = -15.0

    # ── Play-mode ─────────────────────────────────────────────────────────────
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption  = False
        cfg.observations["critic"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (0.15, 0.30)
        twist_cmd.ranges.lin_vel_y = (0.00, 0.00)
        twist_cmd.ranges.ang_vel_z = (0.00, 0.00)

    return cfg

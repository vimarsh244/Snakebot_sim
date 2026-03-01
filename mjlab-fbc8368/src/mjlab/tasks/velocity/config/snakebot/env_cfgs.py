"""Snakebot chain_5 velocity environment configurations (flat terrain only).

Design philosophy:
- Actor:  phase_clock + joint_pos + joint_vel + last_action + command + root IMU (~44 dim).
          Only actuated joints (10) are observed, not all 60 mechanism joints.
- Critic: actor obs + per-module positions, lin-vels, ang-vels, actuator effort
          (~99 dim, privileged simulation state, zero noise).
- 10 Hz control:   decimation=20  (MuJoCo dt=0.005 s × 20 = 0.10 s / step).
- Rewards use world-frame velocities (body-frame X points downward, not forward).
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
from mjlab.envs.mdp.rewards import joint_pos_limits
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity.mdp import (
    UniformVelocityCommandCfg,
)
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from . import snakebot_mdp

# ── Body / joint references ───────────────────────────────────────────────────
SNAKE_ROOT_BODY  = "m1_bottom-base-plate-v1"
_MODULE_BODIES   = "m[1-5]_bottom-base-plate-v1"
_ACTUATED_JOINTS = (".*Revolute-15", ".*Revolute-16")


# ─────────────────────────────────────────────────────────────────────────────
def snakebot_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Snakebot flat terrain velocity configuration.

    Reward design — forward velocity dominates, penalties minimal:
    ┌──────────────────────────┬────────┬─────────────────────────────────────┐
    │ Term                     │ Weight │ Purpose                             │
    ├──────────────────────────┼────────┼─────────────────────────────────────┤
    │ forward_velocity_track   │ +12.0  │ Track commanded forward speed        │
    │ alive_bonus              │  +0.05 │ Small keep-alive baseline           │
    │ lateral_velocity_penalty │ -0.01  │ Discourage crab-walk                │
    │ yaw_rate_penalty         │ -0.01  │ Keep heading stable                 │
    │ control_cost             │ -0.004 │ Avoid high-effort thrashing         │
    │ action_smoothness        │ -0.002 │ Reduce jitter                       │
    │ vertical_velocity        │ -0.08  │ Discourage launching/bouncing       │
    │ dof_pos_limits           │  -0.1  │ Soft joint-limit guard              │
    └──────────────────────────┴────────┴─────────────────────────────────────┘
    """
    cfg = make_velocity_env_cfg()

    # ── Robot ─────────────────────────────────────────────────────────────────
    cfg.scene.entities = {"robot": get_snakebot_robot_cfg()}

    # ── Terrain: flat plane, no height scanner ────────────────────────────────
    cfg.sim.njmax = 8000
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
    cfg.episode_length_s = 30.0

    # ── Observations (fully replaced) ─────────────────────────────────────────
    _module_body_cfg = SceneEntityCfg("robot", body_names=(_MODULE_BODIES,))
    _actuated_joint_cfg = SceneEntityCfg("robot", joint_names=_ACTUATED_JOINTS)

    actor_terms: dict = {
        "phase_clock": ObservationTermCfg(
            func=snakebot_mdp.phase_clock,
        ),
        "joint_pos": ObservationTermCfg(
            func=joint_pos_rel,
            params={"asset_cfg": _actuated_joint_cfg},
            noise=Unoise(n_min=-0.06, n_max=0.06),
        ),
        "joint_vel": ObservationTermCfg(
            func=joint_vel_rel,
            params={"asset_cfg": _actuated_joint_cfg},
            noise=Unoise(n_min=-4.0, n_max=4.0),
        ),
        "actions": ObservationTermCfg(func=last_action),
        "command": ObservationTermCfg(
            func=generated_commands,
            params={"command_name": "twist"},
            noise=Unoise(n_min=-0.08, n_max=0.08),
        ),
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
    twist_cmd.ranges.lin_vel_x   = (0.05, 0.20)
    twist_cmd.ranges.lin_vel_y   = (0.00, 0.00)
    twist_cmd.ranges.ang_vel_z   = (0.00, 0.00)
    twist_cmd.ranges.heading     = (-math.pi, math.pi)
    twist_cmd.rel_standing_envs  = 0.05

    # ── Events / Domain randomisation ────────────────────────────────────────
    cfg.events["base_com"].params["asset_cfg"].body_names = (SNAKE_ROOT_BODY,)
    cfg.events["base_com"].params["ranges"] = {
        0: (-0.03, 0.03),
        1: (-0.03, 0.03),
        2: (-0.02, 0.02),
    }
    cfg.events.pop("encoder_bias", None)
    cfg.events["push_robot"].params["velocity_range"] = {
        "x": (-0.15, 0.15),
        "y": (-0.15, 0.15),
        "z": (-0.1, 0.1),
        "roll":  (-0.15, 0.15),
        "pitch": (-0.15, 0.15),
        "yaw":   (-0.1, 0.1),
    }
    cfg.events.pop("foot_friction", None)

    # ── Rewards — forward velocity dominates, penalties near-zero ─────────────
    cfg.rewards = {
        "forward_velocity_track": RewardTermCfg(
            func=snakebot_mdp.track_forward_velocity_command,
            params={"command_name": "twist", "std": 0.08},
            weight=12.0,
        ),
        "alive_bonus": RewardTermCfg(
            func=snakebot_mdp.alive_bonus,
            weight=0.05,
        ),
        "lateral_velocity_penalty": RewardTermCfg(
            func=snakebot_mdp.lateral_velocity_penalty,
            weight=-0.01,
        ),
        "yaw_rate_penalty": RewardTermCfg(
            func=snakebot_mdp.yaw_rate_penalty,
            weight=-0.01,
        ),
        "control_cost": RewardTermCfg(
            func=snakebot_mdp.control_cost,
            weight=-0.004,
        ),
        "action_smoothness": RewardTermCfg(
            func=snakebot_mdp.action_smoothness_penalty,
            weight=-0.002,
        ),
        "vertical_velocity": RewardTermCfg(
            func=snakebot_mdp.vertical_velocity_penalty,
            weight=-0.08,
        ),
        "dof_pos_limits": RewardTermCfg(
            func=joint_pos_limits,
            params={"asset_cfg": _actuated_joint_cfg},
            weight=-0.1,
        ),
    }

    # ── Terminations ──────────────────────────────────────────────────────────
    cfg.terminations.pop("illegal_contact", None)
    # bad_orientation uses projected_gravity_b[:, 2] which is ~0 for a snake
    # lying flat, so the check always gives ~90deg — remove it.
    cfg.terminations.pop("fell_over", None)
    # terminate if the snake gets launched into the air (physics blowup)
    cfg.terminations["too_high"] = TerminationTermCfg(
        func=snakebot_mdp.root_too_high,
        params={"max_height": 0.3},
    )

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

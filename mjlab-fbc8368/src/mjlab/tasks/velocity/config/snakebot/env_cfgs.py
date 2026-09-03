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
    get_snakebot_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
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
from mjlab.sim import MujocoCfg
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

    Reward design — literature-grounded for serpentine locomotion:
    ┌──────────────────────────┬────────┬──────────────────────────────────────────┐
    │ Term                     │ Weight │ Purpose                                  │
    ├──────────────────────────┼────────┼──────────────────────────────────────────┤
    │ forward_velocity         │  +2.0  │ World +X CoM velocity (primary drive)    │
    │ alive_bonus              │  +0.2  │ Keep-alive baseline                      │
    │ lateral_velocity         │  -0.3  │ Penalise Y drift (Bing 2020, COBRA)      │
    │ vertical_velocity        │  -0.3  │ Penalise vertical bounce (Qiu 2021)      │
    │ yaw_rate                 │  -0.2  │ Penalise spin-in-place (Shi 2020)        │
    │ action_smoothness        │  -0.05 │ Smooth gait → hardware transfer          │
    │ control_cost             │  -0.02 │ Energy efficiency (Singh 2022)           │
    │ dof_pos_limits           │  -0.5  │ Soft joint-limit guard                   │
    └──────────────────────────┴────────┴──────────────────────────────────────────┘
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

    # ── Solver: override base factory defaults for closed-chain weld constraints ──
    # The base factory uses iterations=10 which cannot reliably resolve the
    # snakebot's equality-weld constraints. impratio=10 matches chain_5.xml and
    # prevents constraint drift under friction.  50 Newton iterations is enough
    # for weld convergence at this timestep.
    cfg.sim.mujoco = MujocoCfg(
        timestep=0.0025,
        iterations=100,
        ls_iterations=30,
        impratio=10.0,
        solver="newton",
        integrator="implicitfast",
        ccd_iterations=100,
    )

    # ── 20 Hz control frequency ───────────────────────────────────────────────
    # 0.0025 s × 20 = 0.05 s / step
    cfg.decimation = 20
    cfg.episode_length_s = 20.0

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
            noise=Unoise(n_min=-0.03, n_max=0.03),
        ),
        "joint_vel": ObservationTermCfg(
            func=joint_vel_rel,
            params={"asset_cfg": _actuated_joint_cfg},
            noise=Unoise(n_min=-1.0, n_max=1.0),
        ),
        "actions": ObservationTermCfg(func=last_action),
        "command": ObservationTermCfg(
            func=generated_commands,
            params={"command_name": "twist"},
            noise=Unoise(n_min=-0.03, n_max=0.03),
        ),
        "base_lin_vel": ObservationTermCfg(
            func=builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=Unoise(n_min=-0.8, n_max=0.8),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.3, n_max=0.3),
        ),
        "projected_gravity": ObservationTermCfg(
            func=projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
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
    joint_pos_action.scale = {
        ".*Revolute-15": 0.08,
        ".*Revolute-16": 0.08,
    }

    # ── Commands: forward-only ────────────────────────────────────────────────
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.heading_command    = False   # snake goes straight, no heading control
    twist_cmd.ranges.lin_vel_x   = (0.005, 0.03)
    twist_cmd.ranges.lin_vel_y   = (0.00, 0.00)
    twist_cmd.ranges.ang_vel_z   = (0.00, 0.00)
    twist_cmd.ranges.heading     = None
    twist_cmd.rel_standing_envs  = 0.05
    twist_cmd.viz.world_frame_viz = False

    # ── Events / Domain randomisation ────────────────────────────────────────
    # --- Base resets (inherited from velocity factory, adjusted for snake) ---
    cfg.events.pop("base_com", None)
    cfg.events.pop("encoder_bias", None)
    cfg.events.pop("push_robot", None)
    cfg.events.pop("foot_friction", None)
    # Use full-scene default reset for this closed-chain model.
    # It restores all free-joint and hinge states coherently before each episode.
    cfg.events["reset_base"].func = envs_mdp.reset_scene_to_default
    cfg.events["reset_base"].params = {}
    cfg.events.pop("reset_robot_joints", None)

    # ── Rewards — literature-grounded for snake locomotion ──────────────────────
    # Primary: forward world-X velocity (Bing 2020, Shi 2020, Qiu 2021)
    # Penalties: lateral drift, vertical bounce, yaw spin, action jerk, energy
    # (Singh 2022, Bing 2020, Liu 2023 — all highlight these four penalty terms)
    cfg.rewards = {
        # === Primary forward drive ===
        "forward_velocity": RewardTermCfg(
            func=snakebot_mdp.forward_velocity_reward,
            params={"max_vel": 0.05},
            weight=6.0,
        ),
        "alive_bonus": RewardTermCfg(func=snakebot_mdp.alive_bonus, weight=0.3),
        # === Penalties to shape a clean serpentine gait ===
        # Penalise world-Y drift so the snake travels along its body axis
        "lateral_velocity": RewardTermCfg(
            func=snakebot_mdp.lateral_velocity_penalty,
            weight=-0.03,
        ),
        # Penalise vertical CoM motion — snake should stay flat on ground
        "vertical_velocity": RewardTermCfg(
            func=snakebot_mdp.vertical_velocity_penalty,
            weight=-0.03,
        ),
        # Penalise spinning in place instead of translating
        "yaw_rate": RewardTermCfg(
            func=snakebot_mdp.yaw_rate_penalty,
            weight=-0.01,
        ),
        # Penalise high-frequency joint oscillations (smooth gait transfers to HW)
        "action_smoothness": RewardTermCfg(
            func=snakebot_mdp.action_smoothness_penalty,
            weight=-0.005,
        ),
        # Penalise large torques as energy proxy (Bing 2020 energy efficiency)
        "control_cost": RewardTermCfg(
            func=snakebot_mdp.control_cost,
            weight=-0.002,
        ),
        # Soft penalty for hitting actuated joint limits only (not all 60 mechanism joints)
        "dof_pos_limits": RewardTermCfg(
            func=joint_pos_limits,
            params={"asset_cfg": _actuated_joint_cfg},
            weight=0.0,
        ),
    }

    # ── Terminations ──────────────────────────────────────────────────────────
    cfg.terminations.pop("illegal_contact", None)
    # bad_orientation uses projected_gravity_b[:, 2] which is ~0 for a snake
    # lying flat, so the check always gives ~90deg — remove it.
    cfg.terminations.pop("fell_over", None)
    cfg.terminations["too_high"] = TerminationTermCfg(
        func=snakebot_mdp.root_too_high,
        params={"max_height": 2.5},
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
        twist_cmd.ranges.lin_vel_x = (0.01, 0.04)
        twist_cmd.ranges.lin_vel_y = (0.00, 0.00)
        twist_cmd.ranges.ang_vel_z = (0.00, 0.00)

    return cfg

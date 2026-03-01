"""Snakebot-specific MDP observation and reward functions.

Actor observations:  joint states + root IMU only (sim-to-real deployable).
Critic observations: actor obs + full per-module positions / velocities / ang-vel (privileged).

All velocity-based rewards use WORLD-FRAME velocities because the snake's
body frame X-axis points downward (world -Z) due to the compound frame
rotations in the MJCF. Using body-frame velocities would reward vertical
motion instead of horizontal forward travel.

Reward design philosophy:
  1. Forward velocity    — world-X CoM velocity (snake lies along world X)
  2. Alive bonus         — constant positive signal to encourage long episodes
  3. Control cost        — penalise large squared actions
  4. Smoothness          — penalise action changes each step
  5. Lateral drift       — penalise world-Y CoM velocity
  6. Yaw rate            — penalise rotation around world Z
  7. Joint pos limits    — soft penalty for hitting joint limits
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# gait period in env steps (15 steps * 0.1s = 1.5s cycle at 10 Hz)
GAIT_PERIOD_STEPS = 15


# ---------------------------------------------------------------------------
# Module body regex — one representative body per snake module (5 total)
# ---------------------------------------------------------------------------
MODULE_BODY_PATTERN = "m[1-5]_bottom-base-plate-v1"
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_MODULE_ASSET_CFG = SceneEntityCfg(
    "robot", body_names=(MODULE_BODY_PATTERN,)
)


# ---------------------------------------------------------------------------
# Critic-only observation functions (privileged state)
# ---------------------------------------------------------------------------

def all_body_positions_rel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_MODULE_ASSET_CFG,
) -> torch.Tensor:
    """Positions of all module bodies relative to the root body, shape (B, N*3)."""
    asset: Entity = env.scene[asset_cfg.name]
    body_pos_w  = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # (B, N, 3)
    root_pos_w  = asset.data.root_link_pos_w.unsqueeze(1)               # (B, 1, 3)
    return (body_pos_w - root_pos_w).flatten(start_dim=1)               # (B, N*3)


def all_body_lin_velocities(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_MODULE_ASSET_CFG,
) -> torch.Tensor:
    """World-frame linear velocities of all module bodies, shape (B, N*3)."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :].flatten(start_dim=1)


def all_body_ang_velocities(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_MODULE_ASSET_CFG,
) -> torch.Tensor:
    """Angular velocities of all module bodies — simulated IMU per module, shape (B, N*3)."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].flatten(start_dim=1)


def joint_efforts(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Raw actuator forces for all 10 servos, shape (B, 10)."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.actuator_force


# ---------------------------------------------------------------------------
# Actor observation: periodic clock for gait coordination
# ---------------------------------------------------------------------------

def phase_clock(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Sin/cos of normalised episode time — gives the policy a time reference
    for learning periodic gaits. Shape (B, 2)."""
    phase = (
        2.0 * math.pi * env.episode_length_buf.float() / GAIT_PERIOD_STEPS
    )
    return torch.stack([torch.sin(phase), torch.cos(phase)], dim=1)


# ---------------------------------------------------------------------------
# Reward functions (literature-grounded for wheelless serpentine locomotion)
# ---------------------------------------------------------------------------

def track_forward_velocity_command(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    std: float = 0.08,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Track commanded forward speed in world X (snake's physical forward axis).

    Velocity commands in this task are sampled as a scalar `lin_vel_x`.
    For snakebot, we intentionally interpret that scalar as desired *world-X*
    speed because the snake root body frame is rotated (body X points down).
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    target_vx = command[:, 0].clamp(0.0, 0.4)
    actual_vx = asset.data.root_link_lin_vel_w[:, 0].clamp(-1.0, 1.0)
    error = target_vx - actual_vx
    return torch.exp(-torch.square(error) / (std**2))


def lateral_velocity_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Squared lateral (world Y) CoM velocity, clamped before squaring.

    Penalises sideways drift so the snake travels along its body axis (world X).
    Clamped to ±3 m/s before squaring to avoid blowup from tumbling.
    """
    asset: Entity = env.scene[asset_cfg.name]
    v_lat = asset.data.root_link_lin_vel_w[:, 1].clamp(-3.0, 3.0)
    return torch.square(v_lat)   # (B,)  max penalty = 9 per step


def yaw_rate_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Squared yaw rate (rotation around world Z), clamped before squaring.

    Penalises heading changes so the snake travels in a straight line.
    Uses world-frame angular velocity because the body frame's Z-axis
    points along world -X (not upward), so body-frame Z rotation would
    penalise pitch instead of yaw.
    """
    asset: Entity = env.scene[asset_cfg.name]
    omega_z = asset.data.root_link_ang_vel_w[:, 2].clamp(-5.0, 5.0)
    return torch.square(omega_z)   # (B,)  max penalty = 25 per step


def vertical_velocity_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Squared vertical (world Z) CoM velocity, clamped before squaring.

    Discourages the snake from launching itself upward or bouncing.
    A snake doing proper serpentine locomotion should have near-zero Z velocity.
    """
    asset: Entity = env.scene[asset_cfg.name]
    v_z = asset.data.root_link_lin_vel_w[:, 2].clamp(-3.0, 3.0)
    return torch.square(v_z)   # (B,)


def action_smoothness_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Squared difference between the current and previous action (jerk penalty).

    Prevents high-frequency oscillations and produces gaits that transfer to
    hardware.  Complements `action_rate_l2` which is already in the base mdp.
    """
    delta = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(torch.square(delta), dim=1)   # (B,)


def alive_bonus(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Constant +1 each step the episode has NOT been terminated early.

    Encourages the episode to last so that forward velocity has time to
    accumulate.  Returns 0 on terminal steps.
    """
    return (~env.termination_manager.terminated).float()   # (B,)


def control_cost(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Squared sum of all actions.

    Prevents the policy from using excessive joint forces as a degenerate
    shortcut (e.g. 'launch' behaviour).  Equivalent to an L2 action penalty.
    """
    return torch.sum(torch.square(env.action_manager.action), dim=1)   # (B,)


# ---------------------------------------------------------------------------
# Termination: catch flying / physics blowup
# ---------------------------------------------------------------------------

def root_too_high(
    env: ManagerBasedRlEnv,
    max_height: float = 0.3,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate if the root body rises above max_height.

    A snake doing serpentine locomotion stays near z=0.08 (its initial height).
    If z exceeds max_height the snake has been launched and further simulation
    is wasted compute.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2] > max_height

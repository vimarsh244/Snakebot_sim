"""Snakebot-specific MDP observation and reward functions.

Actor observations:  joint states + root IMU only (sim-to-real deployable).
Critic observations: actor obs + full per-module positions / velocities / ang-vel (privileged).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# Helper: one representative body per module (used for per-module IMU / pos)
# ---------------------------------------------------------------------------
# We select the bottom base-plate of each module as the "module centre" body.
# Regex pattern used in SceneEntityCfg:  "m[1-5]_bottom-base-plate-v1"
MODULE_BODY_PATTERN = "m[1-5]_bottom-base-plate-v1"


# ---------------------------------------------------------------------------
# Critic-only observation functions (privileged state)
# ---------------------------------------------------------------------------


def all_body_positions_rel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=(MODULE_BODY_PATTERN,)),
) -> torch.Tensor:
    """Positions of all module bodies relative to the root body.

    Returns (B, N*3) where N = number of selected bodies.
    Provides the critic with full kinematic-chain pose information.
    """
    asset: Entity = env.scene[asset_cfg.name]
    # body_link_pos_w: (B, N_all_bodies, 3)
    body_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # (B, N, 3)
    root_pos_w = asset.data.root_link_pos_w.unsqueeze(1)  # (B, 1, 3)
    rel_pos = body_pos_w - root_pos_w  # (B, N, 3)
    return rel_pos.flatten(start_dim=1)  # (B, N*3)


def all_body_lin_velocities(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=(MODULE_BODY_PATTERN,)),
) -> torch.Tensor:
    """Linear velocities of all module bodies in world frame.

    Returns (B, N*3).  Gives the critic full chain dynamics.
    """
    asset: Entity = env.scene[asset_cfg.name]
    body_lin_vel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]  # (B, N, 3)
    return body_lin_vel_w.flatten(start_dim=1)  # (B, N*3)


def all_body_ang_velocities(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=(MODULE_BODY_PATTERN,)),
) -> torch.Tensor:
    """Angular velocities of all module bodies — simulated IMU per module.

    Returns (B, N*3).  Equivalent to having an IMU sensor on every module.
    """
    asset: Entity = env.scene[asset_cfg.name]
    body_ang_vel_w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]  # (B, N, 3)
    return body_ang_vel_w.flatten(start_dim=1)  # (B, N*3)


def joint_efforts(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Scalar actuator forces for all 10 servos.  Shape (B, nu).

    Useful for the critic to estimate energy consumption and effort limits.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.actuator_force  # (B, nu=10)


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def snake_forward_progress(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Bonus reward for CoM displacement in the commanded direction.

    Complements ``track_linear_velocity`` with an extra signal that directly
    measures how fast the snake is moving toward its goal.  Uses exponential
    kernel so the reward saturates gracefully at high speed.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    # Actual linear velocity in body frame, projected onto commanded xy direction.
    lin_vel_b = asset.data.root_link_lin_vel_b  # (B, 3)
    cmd_xy = command[:, :2]  # (B, 2)
    cmd_norm = torch.norm(cmd_xy, dim=1, keepdim=True).clamp(min=1e-3)  # (B, 1)
    cmd_dir = cmd_xy / cmd_norm  # (B, 2)
    forward_vel = torch.sum(lin_vel_b[:, :2] * cmd_dir, dim=1)  # (B,)
    # exp reward: peaks at 1.0 when forward_vel == cmd_norm
    vel_error = torch.square(forward_vel - cmd_norm.squeeze(1))
    std = 0.25
    return torch.exp(-vel_error / std**2)


def snake_body_height(
    env: ManagerBasedRlEnv,
    target_height: float = 0.08,
    std: float = 0.04,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=(MODULE_BODY_PATTERN,)),
) -> torch.Tensor:
    """Reward module bodies staying near ground-clearance height (target_height).

    Penalises the snake from lifting all modules high or scraping the ground,
    which destabilises locomotion.  Returns mean exp-reward over all modules.
    """
    asset: Entity = env.scene[asset_cfg.name]
    body_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]  # (B, N)
    height_error = torch.square(body_z - target_height)  # (B, N)
    return torch.mean(torch.exp(-height_error / std**2), dim=1)  # (B,)


def snake_undulation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*",)),
) -> torch.Tensor:
    """Reward serpentine wave pattern by encouraging alternating joint signs.

    A classical serpentine gait has adjacent joints with opposite polarity.
    We compute the mean of -sign(q_i) * sign(q_{i+1}) across all adjacent
    pairs; this is +1 when all adjacent joints alternate perfectly and -1
    when all have the same sign.  Returned as [0, 1]-normalised.
    """
    asset: Entity = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]  # (B, nj)
    if joint_pos.shape[1] < 2:
        return torch.zeros(env.num_envs, device=env.device)
    # Compare sign of adjacent joints
    q_even = joint_pos[:, :-1]  # (B, nj-1)
    q_odd  = joint_pos[:, 1:]   # (B, nj-1)
    # -1 when same sign (bad), +1 when opposite sign (serpentine)
    alternation = -torch.sign(q_even) * torch.sign(q_odd)  # (B, nj-1)
    return (torch.mean(alternation, dim=1) + 1.0) * 0.5  # map to [0, 1]


def snake_joint_torque_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise high actuator forces (energy cost / torque limit protection).

    Returns a bounded [0, 1) penalty based on mean squared actuator force.
    Bounding prevents rare force spikes from destabilizing PPO updates.
    """
    asset: Entity = env.scene[asset_cfg.name]
    mean_sq_force = torch.mean(torch.square(asset.data.actuator_force), dim=1)  # (B,)
    force_scale = 20.0
    return torch.tanh(mean_sq_force / (force_scale**2))


def snake_lateral_deviation(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise deviation perpendicular to the commanded direction.

    Encourages the snake to travel in a straight line rather than drifting
    sideways, which improves forward efficiency.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    lin_vel_b = asset.data.root_link_lin_vel_b  # (B, 3)
    # Lateral velocity (y in body frame when heading forward)
    lateral_vel = lin_vel_b[:, 1]  # (B,)
    lat_vel_scale = 0.5
    return torch.tanh(torch.square(lateral_vel) / (lat_vel_scale**2))

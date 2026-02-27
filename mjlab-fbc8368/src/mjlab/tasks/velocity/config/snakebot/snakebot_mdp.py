"""Snakebot-specific MDP observation and reward functions.

Actor observations:  joint states + root IMU only (sim-to-real deployable).
Critic observations: actor obs + full per-module positions / velocities / ang-vel (privileged).

Reward design philosophy (based on locomotion literature for wheelless snakes):
  1. Forward velocity    — primary locomotion signal (CoM velocity along heading)
  2. Alive bonus         — constant positive signal to encourage long episodes
  3. Control cost        — penalise large squared actions (energy & prevent degenerate solutions)
  4. Smoothness          — penalise action *changes* each step (sim-to-real, prevents high-freq jitter)
  5. Lateral drift       — penalise Y-axis CoM velocity (no crab-walking or spinning)
  6. Yaw rate            — penalise angular velocity around Z (stay straight)
  7. Joint pos limits    — soft penalty for hitting joint limits (prevent self-collision)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# Module body regex — one representative body per snake module (5 total)
# ---------------------------------------------------------------------------
MODULE_BODY_PATTERN = "m[1-5]_bottom-base-plate-v1"


# ---------------------------------------------------------------------------
# Critic-only observation functions (privileged state)
# ---------------------------------------------------------------------------

def all_body_positions_rel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=(MODULE_BODY_PATTERN,)),
) -> torch.Tensor:
    """Positions of all module bodies relative to the root body, shape (B, N*3)."""
    asset: Entity = env.scene[asset_cfg.name]
    body_pos_w  = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # (B, N, 3)
    root_pos_w  = asset.data.root_link_pos_w.unsqueeze(1)               # (B, 1, 3)
    return (body_pos_w - root_pos_w).flatten(start_dim=1)               # (B, N*3)


def all_body_lin_velocities(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=(MODULE_BODY_PATTERN,)),
) -> torch.Tensor:
    """World-frame linear velocities of all module bodies, shape (B, N*3)."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :].flatten(start_dim=1)


def all_body_ang_velocities(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=(MODULE_BODY_PATTERN,)),
) -> torch.Tensor:
    """Angular velocities of all module bodies — simulated IMU per module, shape (B, N*3)."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].flatten(start_dim=1)


def joint_efforts(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Raw actuator forces for all 10 servos, shape (B, 10)."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.actuator_force


# ---------------------------------------------------------------------------
# Reward functions (literature-grounded for wheelless serpentine locomotion)
# ---------------------------------------------------------------------------

def forward_velocity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Forward CoM velocity in the body frame (x-component).

    This is the **primary locomotion reward** — directly measuring the rate at
    which the snake moves forward.  Using body-frame velocity means the reward
    is agnostic to world-frame heading, which pairs well with the yaw penalty.

    Returns velocity in m/s (raw, not exp-transformed — keep it linear so the
    gradient signal is always informative regardless of current speed).
    """
    asset: Entity = env.scene[asset_cfg.name]
    # root_link_lin_vel_b: linear velocity expressed in the robot body frame
    return asset.data.root_link_lin_vel_b[:, 0]   # (B,)  x = forward


def lateral_velocity_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared lateral (Y-body-frame) CoM velocity.

    Without this, the policy may learn to "crab-walk" sideways or spin in
    place — exploiting anisotropic friction in the wrong direction.  This
    penalty forces truly forward motion.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 1])   # (B,)


def yaw_rate_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared yaw angular velocity (Z-body-frame).

    Without this the snake tends to curve and circle.  Penalising Z angular
    velocity keeps the snake tracking in a straight line.
    """
    asset: Entity = env.scene[asset_cfg.name]
    # root_link_ang_vel_b: angular velocity in body frame; index 2 = yaw (Z)
    return torch.square(asset.data.root_link_ang_vel_b[:, 2])   # (B,)


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
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared sum of all actions.

    Prevents the policy from using excessive joint forces as a degenerate
    shortcut (e.g. 'launch' behaviour).  Equivalent to an L2 action penalty.
    """
    return torch.sum(torch.square(env.action_manager.action), dim=1)   # (B,)

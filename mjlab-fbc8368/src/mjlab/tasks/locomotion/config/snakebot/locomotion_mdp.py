"""Snakebot locomotion MDP: observation and reward functions for goal-reaching.

Body-frame convention for this snake:
  The MJCF has compound frame rotations that make body-frame X = world -Z
  (downward), body Y = world -Y, and body Z = world -X. The snake's forward
  axis is therefore body -Z, and its lateral axis is body -Y. All body-frame
  computations use full quaternion rotation to avoid gimbal-lock issues.

Observation design:
  Actor (~38 dim):
    - phase_clock (2), goal_vector (2, forward/lateral in body frame),
      heading_to_goal (2), joint_pos (10), joint_vel (10), last_action (10)
  Critic (~93 dim, privileged):
    - actor obs + per-module positions/velocities/angular-velocities + efforts

Goal data is stored as env._loco_goal_pos (B, 2) and env._loco_prev_dist (B,)
by the event callbacks in env_cfg.py.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# Module body regex — one representative body per snake module (5 total)
MODULE_BODY_PATTERN = "m[1-5]_bottom-base-plate-v1"
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_MODULE_ASSET_CFG = SceneEntityCfg(
    "robot", body_names=(MODULE_BODY_PATTERN,)
)

# gait period in env steps (15 steps * 0.1s = 1.5s cycle at 10 Hz)
GAIT_PERIOD_STEPS = 15


# ---------------------------------------------------------------------------
# Goal management helpers
# ---------------------------------------------------------------------------

def _get_goal_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Get goal XY position, shape (B, 2). Lazily initializes on first call."""
    env_any = cast(Any, env)
    if not hasattr(env_any, "_loco_goal_pos"):
        # Initialize with random goals spread around the origin to give
        # the obs normalizer diverse initial statistics (prevents NaN from
        # zero-variance normalization on the first forward pass).
        import math
        n = env.num_envs
        angle = torch.rand(n, device=env.device) * 2 * math.pi
        radius = 0.3 + torch.rand(n, device=env.device) * 0.5  # 0.3-0.8m
        env_any._loco_goal_pos = torch.stack([
            radius * torch.cos(angle),
            radius * torch.sin(angle),
        ], dim=1)
        env_any._loco_prev_dist = radius.clone()
    return env_any._loco_goal_pos  # set by reset event


def _get_head_pos_xy(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Get head/root module XY world position, shape (B, 2)."""
    asset: Entity = env.scene["robot"]
    head_xy = asset.data.root_link_pos_w[:, :2]
    return torch.nan_to_num(head_xy, nan=0.0, posinf=0.0, neginf=0.0)


def _distance_to_goal(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Euclidean XY distance from root to goal, shape (B,)."""
    diff = _get_goal_pos(env) - _get_head_pos_xy(env)
    return torch.norm(diff, dim=1)


# ---------------------------------------------------------------------------
# Actor observation functions (sim-to-real deployable)
# ---------------------------------------------------------------------------

def goal_vector_body_frame(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Goal direction in the snake's local frame, shape (B, 2).

    Uses full quaternion rotation instead of yaw extraction because the
    snake's initial pitch is ~90deg (body X points down), which puts the
    standard yaw formula in gimbal lock.

    The snake's forward axis is body -Z (maps to world +X initially) and
    lateral axis is body -Y (maps to world +Y). We return (forward, lateral)
    so that a positive first component means "goal is ahead."
    """
    asset: Entity = env.scene[asset_cfg.name]
    goal_w_2d = _get_goal_pos(env) - _get_head_pos_xy(env)  # (B, 2)
    # pad to 3D (goal is on XY plane, z=0)
    goal_w_3d = torch.cat([
        goal_w_2d,
        torch.zeros_like(goal_w_2d[:, :1]),
    ], dim=1)  # (B, 3)

    quat = asset.data.root_link_quat_w  # (B, 4) wxyz
    quat = torch.nan_to_num(quat, nan=0.0, posinf=0.0, neginf=0.0)
    quat_norm = torch.norm(quat, dim=1, keepdim=True).clamp(min=1e-8)
    quat = quat / quat_norm

    goal_b = quat_apply_inverse(quat, goal_w_3d)  # (B, 3)

    # body -Z = forward along snake axis, body -Y = lateral
    forward = -goal_b[:, 2]
    lateral = -goal_b[:, 1]
    result = torch.stack([forward, lateral], dim=1)
    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)  # (B, 2)


def heading_to_goal(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Sin/cos of angle between robot heading and goal direction, shape (B, 2).

    Provides heading-invariant goal direction info.
    """
    goal_body = goal_vector_body_frame(env, asset_cfg)  # (B, 2)
    angle = torch.atan2(goal_body[:, 1], goal_body[:, 0])  # (B,)
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=1)  # (B, 2)


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
# Critic-only observation functions (privileged state)
# ---------------------------------------------------------------------------

def all_body_positions_rel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_MODULE_ASSET_CFG,
) -> torch.Tensor:
    """Positions of all module bodies relative to the root body, shape (B, N*3)."""
    asset: Entity = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    root_pos_w = asset.data.root_link_pos_w.unsqueeze(1)
    return (body_pos_w - root_pos_w).flatten(start_dim=1)


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
    """Angular velocities of all module bodies, shape (B, N*3)."""
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
# Reward functions (literature-grounded goal-reaching)
# ---------------------------------------------------------------------------

def progress_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Change in distance to goal: prev_dist - curr_dist.

    Positive when the snake gets closer to the goal.
    Key shaped reward from COBRA thesis and snakebot-gym.
    Clamped to [-1, 1] to prevent blowup during resets.
    """
    env_any = cast(Any, env)
    _get_goal_pos(env)  # ensure lazy init
    curr_dist = _distance_to_goal(env)
    prev_dist = env_any._loco_prev_dist
    progress = (prev_dist - curr_dist).clamp(-1.0, 1.0)
    # update history here so reward is independent of event callback ordering
    env_any._loco_prev_dist = curr_dist.detach()
    return progress


def distance_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Negative distance to goal (snakebot-gym style).

    Provides a persistent pull toward the goal.
    Clamped to prevent huge values at large distances.
    """
    return -_distance_to_goal(env).clamp(0.0, 3.0)


def goal_reached_bonus(
    env: ManagerBasedRlEnv,
    threshold: float = 0.15,
) -> torch.Tensor:
    """Sparse +1 when head is within `threshold` meters of goal.

    Weight in config controls magnitude (we set weight=100).
    """
    dist = _distance_to_goal(env)
    return (dist < threshold).float()


def heading_alignment_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Cosine similarity between body heading and goal direction.

    +1 when facing directly at goal, -1 when facing away.
    Inspired by Naish/EELS heading reward.
    """
    goal_body = goal_vector_body_frame(env, asset_cfg)  # (B, 2)
    dist = torch.norm(goal_body, dim=1, keepdim=True).clamp(min=0.01)
    direction = goal_body / dist
    # reward alignment but do not add extra penalty when facing away; progress
    # and distance terms already provide directional pressure.
    return torch.clamp(direction[:, 0], min=0.0)


def alive_bonus(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Constant +1 each step the episode has NOT been terminated early."""
    return (~env.termination_manager.terminated).float()


def control_cost(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Squared sum of all actions (energy penalty)."""
    return torch.sum(torch.square(env.action_manager.action), dim=1)


def action_smoothness_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Squared difference between current and previous action (jerk penalty)."""
    delta = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(torch.square(delta), dim=1)


# ---------------------------------------------------------------------------
# Termination functions
# ---------------------------------------------------------------------------

def goal_reached_termination(
    env: ManagerBasedRlEnv,
    threshold: float = 0.15,
) -> torch.Tensor:
    """Terminate episode when the snake reaches the goal."""
    return _distance_to_goal(env) < threshold


def root_too_high(
    env: ManagerBasedRlEnv,
    max_height: float = 0.3,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate if the root body rises above max_height (physics blowup)."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2] > max_height

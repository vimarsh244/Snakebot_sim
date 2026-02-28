"""Snakebot locomotion MDP: observation and reward functions for goal-reaching.

Reward design (informed by 5 reference papers):
  1. Progress reward — change in distance to goal per step (COBRA thesis,
     Naish/EELS, snakebot-gym). THE key shaped signal.
  2. Distance penalty — negative current distance to goal (snakebot-gym).
  3. Goal bonus — large sparse reward when goal reached (snakebot-gym, +100).
  4. Alive bonus — constant +1 to encourage long episodes.
  5. Control cost — L2 action penalty for energy efficiency.
  6. Action smoothness — L2 action-delta penalty for sim-to-real.
  7. Heading reward — reward cos(angle_to_goal) to encourage facing goal
     (Naish/EELS heading reward, sensors paper direction reward).
  8. Joint limits — soft penalty near limits (prevent self-collision).

Observation design:
  Actor (~34 dim, hardware-deployable):
    - goal_vector_body_frame (2): XY vector from root to goal in body frame
    - heading_to_goal (2): sin/cos of angle between heading and goal direction
    - joint_pos (10): actuated joint angles
    - joint_vel (10): actuated joint velocities
    - last_action (10): previous actions

  Critic (~89 dim, privileged):
    - All actor obs + per-module positions/velocities/angular-velocities + efforts

Goal data is stored as env._loco_goal_pos (B, 2) and env._loco_prev_dist (B,)
by the event callbacks in env_cfg.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# Module body regex — one representative body per snake module (5 total)
MODULE_BODY_PATTERN = "m[1-5]_bottom-base-plate-v1"


# ---------------------------------------------------------------------------
# Goal management helpers
# ---------------------------------------------------------------------------

def _get_goal_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Get goal XY position, shape (B, 2). Lazily initializes on first call."""
    if not hasattr(env, "_loco_goal_pos"):
        # Initialize with random goals spread around the origin to give
        # the obs normalizer diverse initial statistics (prevents NaN from
        # zero-variance normalization on the first forward pass).
        import math
        n = env.num_envs
        angle = torch.rand(n, device=env.device) * 2 * math.pi
        radius = 1.0 + torch.rand(n, device=env.device)  # 1-2m
        env._loco_goal_pos = torch.stack([
            radius * torch.cos(angle),
            radius * torch.sin(angle),
        ], dim=1)
        env._loco_prev_dist = radius.clone()
    return env._loco_goal_pos  # set by reset event


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
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """XY vector from root to goal expressed in body frame, shape (B, 2).

    On hardware this would come from onboard localisation (SLAM, UWB, etc).
    """
    asset: Entity = env.scene[asset_cfg.name]
    # Goal vector in world frame
    goal_w = _get_goal_pos(env) - _get_head_pos_xy(env)  # (B, 2)

    # Get robot heading from root quaternion
    quat = asset.data.root_link_quat_w  # (B, 4) wxyz
    quat = torch.nan_to_num(quat, nan=0.0, posinf=0.0, neginf=0.0)
    quat_norm = torch.norm(quat, dim=1, keepdim=True).clamp(min=1e-8)
    quat = quat / quat_norm
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # Yaw angle from quaternion
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    # Rotate goal vector into body frame
    cos_yaw = torch.cos(-yaw)
    sin_yaw = torch.sin(-yaw)
    goal_body_x = cos_yaw * goal_w[:, 0] - sin_yaw * goal_w[:, 1]
    goal_body_y = sin_yaw * goal_w[:, 0] + cos_yaw * goal_w[:, 1]

    goal_body = torch.stack([goal_body_x, goal_body_y], dim=1)
    return torch.nan_to_num(goal_body, nan=0.0, posinf=0.0, neginf=0.0)  # (B, 2)


def heading_to_goal(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sin/cos of angle between robot heading and goal direction, shape (B, 2).

    Provides heading-invariant goal direction info.
    """
    goal_body = goal_vector_body_frame(env, asset_cfg)  # (B, 2)
    angle = torch.atan2(goal_body[:, 1], goal_body[:, 0])  # (B,)
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=1)  # (B, 2)


# ---------------------------------------------------------------------------
# Critic-only observation functions (privileged state)
# ---------------------------------------------------------------------------

def all_body_positions_rel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=(MODULE_BODY_PATTERN,)),
) -> torch.Tensor:
    """Positions of all module bodies relative to the root body, shape (B, N*3)."""
    asset: Entity = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    root_pos_w = asset.data.root_link_pos_w.unsqueeze(1)
    return (body_pos_w - root_pos_w).flatten(start_dim=1)


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
    """Angular velocities of all module bodies, shape (B, N*3)."""
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
# Reward functions (literature-grounded goal-reaching)
# ---------------------------------------------------------------------------

def progress_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Change in distance to goal: prev_dist - curr_dist.

    Positive when the snake gets closer to the goal.
    Key shaped reward from COBRA thesis and snakebot-gym.
    Clamped to [-1, 1] to prevent blowup during resets.
    """
    _get_goal_pos(env)  # ensure lazy init
    curr_dist = _distance_to_goal(env)
    prev_dist = env._loco_prev_dist
    return (prev_dist - curr_dist).clamp(-1.0, 1.0)


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
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Cosine similarity between body heading and goal direction.

    +1 when facing directly at goal, -1 when facing away.
    Inspired by Naish/EELS heading reward.
    """
    goal_body = goal_vector_body_frame(env, asset_cfg)  # (B, 2)
    dist = torch.norm(goal_body, dim=1, keepdim=True).clamp(min=0.01)
    direction = goal_body / dist
    return direction[:, 0]  # cos(angle)


def alive_bonus(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Constant +1 each step the episode has NOT been terminated early."""
    return (~env.termination_manager.terminated).float()


def control_cost(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
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

"""Snakebot locomotion v2 MDP functions.

This v2 design is fully proprioceptive for the actor:
- Goal in body frame
- Temporal history of joint pos/vel/actions
- Per-module inertial-style signals (lin/ang velocity in root frame)

No global world positions are exposed to the actor.
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


MODULE_BODY_PATTERN = "m[1-5]_bottom-base-plate-v1"
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_MODULE_ASSET_CFG = SceneEntityCfg("robot", body_names=(MODULE_BODY_PATTERN,))

GAIT_PERIOD_STEPS = 30
SETTLE_GRACE_STEPS = 12


def _grace_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
  return (env.episode_length_buf >= SETTLE_GRACE_STEPS).float()


def _root_quat(asset: Entity) -> torch.Tensor:
  quat = torch.nan_to_num(asset.data.root_link_quat_w, nan=0.0, posinf=0.0, neginf=0.0)
  norm = torch.norm(quat, dim=1, keepdim=True).clamp(min=1e-8)
  return quat / norm


def _get_goal_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
  env_any = cast(Any, env)
  if not hasattr(env_any, "_loco_goal_pos"):
    n = env.num_envs
    angle = torch.rand(n, device=env.device) * (2.0 * math.pi)
    radius = 0.25 + torch.rand(n, device=env.device) * 0.35
    env_any._loco_goal_pos = torch.stack(
      [radius * torch.cos(angle), radius * torch.sin(angle)], dim=1
    )
    env_any._loco_prev_dist = radius.clone()
  return env_any._loco_goal_pos


def _get_robot_com_xy(env: ManagerBasedRlEnv) -> torch.Tensor:
  asset: Entity = env.scene["robot"]
  body_com_xy = asset.data.body_com_pos_w[:, :, :2]
  body_mass = asset.data.model.body_mass[:, asset.indexing.body_ids]
  body_mass = torch.nan_to_num(body_mass, nan=0.0, posinf=0.0, neginf=0.0).clamp(
    min=0.0
  )
  total_mass = body_mass.sum(dim=1, keepdim=True).clamp(min=1e-8)
  com_xy = (body_com_xy * body_mass.unsqueeze(-1)).sum(dim=1) / total_mass
  return torch.nan_to_num(com_xy, nan=0.0, posinf=0.0, neginf=0.0)


def _goal_delta_xy(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _get_goal_pos(env) - _get_robot_com_xy(env)


def _distance_to_goal(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.norm(_goal_delta_xy(env), dim=1).clamp(0.0, 10.0)


def _goal_direction_xy(env: ManagerBasedRlEnv) -> torch.Tensor:
  delta = _goal_delta_xy(env)
  denom = torch.norm(delta, dim=1, keepdim=True).clamp(min=1e-6)
  return delta / denom


def phase_clock(env: ManagerBasedRlEnv) -> torch.Tensor:
  phase = 2.0 * math.pi * env.episode_length_buf.float() / GAIT_PERIOD_STEPS
  return torch.stack([torch.sin(phase), torch.cos(phase)], dim=1)


def goal_vector_body_frame(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Goal direction in body frame as [forward, lateral]."""
  asset: Entity = env.scene[asset_cfg.name]
  goal_w_2d = _goal_delta_xy(env)
  goal_w_3d = torch.cat(
    [goal_w_2d, torch.zeros_like(goal_w_2d[:, :1])],
    dim=1,
  )

  goal_b = quat_apply_inverse(_root_quat(asset), goal_w_3d)

  # For this robot orientation, goal in world +X maps to body +Z.
  forward = goal_b[:, 2]
  lateral = -goal_b[:, 1]
  result = torch.stack([forward, lateral], dim=1)
  result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
  return result.clamp(-3.0, 3.0)


def heading_to_goal(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  goal_body = goal_vector_body_frame(env, asset_cfg)
  angle = torch.atan2(goal_body[:, 1], goal_body[:, 0])
  return torch.stack([torch.sin(angle), torch.cos(angle)], dim=1)


def _rotate_vectors_world_to_root(
  root_quat_w: torch.Tensor,
  vectors_w: torch.Tensor,
) -> torch.Tensor:
  batch_size, num_vectors, _ = vectors_w.shape
  quat_rep = root_quat_w.unsqueeze(1).expand(-1, num_vectors, -1).reshape(-1, 4)
  vectors_flat = vectors_w.reshape(-1, 3)
  vectors_root = quat_apply_inverse(quat_rep, vectors_flat)
  return vectors_root.reshape(batch_size, num_vectors, 3)


def module_lin_vel_root_frame(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_MODULE_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  body_lin_vel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]
  lin_root = _rotate_vectors_world_to_root(_root_quat(asset), body_lin_vel_w)
  return torch.nan_to_num(
    lin_root.flatten(start_dim=1), nan=0.0, posinf=0.0, neginf=0.0
  ).clamp(-20.0, 20.0)


def module_ang_vel_root_frame(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_MODULE_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  body_ang_vel_w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_root = _rotate_vectors_world_to_root(_root_quat(asset), body_ang_vel_w)
  return torch.nan_to_num(
    ang_root.flatten(start_dim=1), nan=0.0, posinf=0.0, neginf=0.0
  ).clamp(-50.0, 50.0)


def all_body_positions_rel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_MODULE_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  body_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
  root_pos_w = asset.data.root_link_pos_w.unsqueeze(1)
  return torch.nan_to_num(
    (body_pos_w - root_pos_w).flatten(start_dim=1),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
  ).clamp(-3.0, 3.0)


def all_body_lin_velocities(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_MODULE_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.nan_to_num(
    asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :].flatten(start_dim=1),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
  ).clamp(-20.0, 20.0)


def all_body_ang_velocities(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_MODULE_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.nan_to_num(
    asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].flatten(start_dim=1),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
  ).clamp(-50.0, 50.0)


def joint_efforts(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.nan_to_num(
    asset.data.actuator_force, nan=0.0, posinf=0.0, neginf=0.0
  ).clamp(-120.0, 120.0)


def goal_distance_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _distance_to_goal(env).unsqueeze(1)


def progress_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Distance progress (prev - current)."""
  env_any = cast(Any, env)
  _get_goal_pos(env)
  curr_dist = _distance_to_goal(env)
  prev_dist = env_any._loco_prev_dist
  progress = (prev_dist - curr_dist).clamp(-0.20, 0.20)
  env_any._loco_prev_dist = curr_dist.detach()
  return progress * _grace_mask(env)


def velocity_towards_goal_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  v_xy = torch.nan_to_num(
    asset.data.root_link_lin_vel_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0
  ).clamp(-2.0, 2.0)
  v_toward = torch.sum(v_xy * _goal_direction_xy(env), dim=1).clamp(-1.0, 1.0)
  return torch.clamp(v_toward, min=0.0) * _grace_mask(env)


def com_velocity_towards_goal_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  v_xy = torch.nan_to_num(
    asset.data.root_com_lin_vel_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0
  ).clamp(-2.0, 2.0)
  v_toward = torch.sum(v_xy * _goal_direction_xy(env), dim=1).clamp(-1.0, 1.0)
  return torch.clamp(v_toward, min=0.0) * _grace_mask(env)


def distance_shaping_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  dist = _distance_to_goal(env).clamp(0.0, 4.0)
  return torch.exp(-2.5 * dist) * _grace_mask(env)


def heading_alignment_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  goal_body = goal_vector_body_frame(env, asset_cfg)
  denom = torch.norm(goal_body, dim=1, keepdim=True).clamp(min=0.05)
  direction = goal_body / denom
  return torch.clamp(direction[:, 0], min=0.0) * _grace_mask(env)


def goal_reached_bonus(
  env: ManagerBasedRlEnv,
  threshold: float = 0.16,
) -> torch.Tensor:
  return (_distance_to_goal(env) < threshold).float()


def stagnation_penalty(
  env: ManagerBasedRlEnv,
  speed_threshold: float = 0.015,
  dist_threshold: float = 0.35,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  v_xy = torch.nan_to_num(
    asset.data.root_link_lin_vel_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0
  ).clamp(-2.0, 2.0)
  v_toward = torch.sum(v_xy * _goal_direction_xy(env), dim=1)
  dist = _distance_to_goal(env)
  stuck = (dist > dist_threshold) & (v_toward < speed_threshold)
  return stuck.float() * _grace_mask(env)


def lateral_slip_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  v_xy = torch.nan_to_num(
    asset.data.root_link_lin_vel_w[:, :2], nan=0.0, posinf=0.0, neginf=0.0
  ).clamp(-2.0, 2.0)
  goal_dir = _goal_direction_xy(env)
  v_toward = torch.sum(v_xy * goal_dir, dim=1, keepdim=True)
  v_lateral = v_xy - v_toward * goal_dir
  lat_sq = torch.sum(torch.square(v_lateral), dim=1)
  return lat_sq.clamp(0.0, 2.0) * _grace_mask(env)


def vertical_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  v_z = asset.data.root_link_lin_vel_w[:, 2].clamp(-3.0, 3.0)
  return torch.square(v_z) * _grace_mask(env)


def yaw_rate_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  yaw_rate = asset.data.root_link_ang_vel_w[:, 2].clamp(-6.0, 6.0)
  return torch.square(yaw_rate) * _grace_mask(env)


def action_smoothness_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  delta = env.action_manager.action - env.action_manager.prev_action
  return torch.sum(torch.square(delta), dim=1).clamp(0.0, 8.0) * _grace_mask(env)


def control_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.sum(torch.square(env.action_manager.action), dim=1).clamp(
    0.0, 8.0
  ) * _grace_mask(env)


def alive_bonus(env: ManagerBasedRlEnv) -> torch.Tensor:
  return (~env.termination_manager.terminated).float()


def goal_reached_termination(
  env: ManagerBasedRlEnv,
  threshold: float = 0.16,
) -> torch.Tensor:
  return _distance_to_goal(env) < threshold


def root_too_high(
  env: ManagerBasedRlEnv,
  max_height: float = 0.65,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  too_high = asset.data.root_link_pos_w[:, 2] > max_height
  settled = env.episode_length_buf >= SETTLE_GRACE_STEPS
  return too_high & settled


def root_too_low(
  env: ManagerBasedRlEnv,
  min_height: float = -0.08,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  too_low = asset.data.root_link_pos_w[:, 2] < min_height
  settled = env.episode_length_buf >= SETTLE_GRACE_STEPS
  return too_low & settled


def unstable_motion_termination(
  env: ManagerBasedRlEnv,
  max_lin_speed: float = 3.0,
  max_ang_speed: float = 18.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  lin_vel = asset.data.root_link_lin_vel_w
  ang_vel = asset.data.root_link_ang_vel_w
  root_pos = asset.data.root_link_pos_w
  lin_speed = torch.norm(
    torch.nan_to_num(lin_vel, nan=0.0, posinf=0.0, neginf=0.0), dim=1
  )
  ang_speed = torch.norm(
    torch.nan_to_num(ang_vel, nan=0.0, posinf=0.0, neginf=0.0), dim=1
  )
  non_finite = (
    (~torch.isfinite(root_pos)).any(dim=1)
    | (~torch.isfinite(lin_vel)).any(dim=1)
    | (~torch.isfinite(ang_vel)).any(dim=1)
  )
  unstable = non_finite | (lin_speed > max_lin_speed) | (ang_speed > max_ang_speed)
  settled = env.episode_length_buf >= SETTLE_GRACE_STEPS
  return unstable & settled


def too_far_from_goal(
  env: ManagerBasedRlEnv,
  max_dist: float = 1.8,
) -> torch.Tensor:
  too_far = _distance_to_goal(env) > max_dist
  settled = env.episode_length_buf >= SETTLE_GRACE_STEPS
  return too_far & settled

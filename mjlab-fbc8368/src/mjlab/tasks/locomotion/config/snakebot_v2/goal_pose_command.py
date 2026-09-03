from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class GoalPoseDebugCommand(CommandTerm):
  cfg: GoalPoseDebugCommandCfg

  def __init__(self, cfg: GoalPoseDebugCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self._goal_pose_w = torch.zeros(self.num_envs, 3, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._goal_pose_w

  def _update_metrics(self) -> None:
    return

  def _robot_com_z(self) -> torch.Tensor:
    body_com_z = self.robot.data.body_com_pos_w[:, :, 2]
    body_mass = self.robot.data.model.body_mass[:, self.robot.indexing.body_ids]
    body_mass = torch.nan_to_num(body_mass, nan=0.0, posinf=0.0, neginf=0.0).clamp(
      min=0.0
    )
    total_mass = body_mass.sum(dim=1).clamp(min=1e-8)
    com_z = (body_com_z * body_mass).sum(dim=1) / total_mass
    return torch.nan_to_num(com_z, nan=0.0, posinf=0.0, neginf=0.0)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    goal_xy = getattr(self._env, "_loco_goal_pos", None)
    if goal_xy is None:
      self._goal_pose_w[env_ids] = 0.0
      return

    self._goal_pose_w[env_ids, 0:2] = goal_xy[env_ids]
    com_z = self._robot_com_z()
    self._goal_pose_w[env_ids, 2] = com_z[env_ids] + self.cfg.z_offset

  def _update_command(self) -> None:
    goal_xy = getattr(self._env, "_loco_goal_pos", None)
    if goal_xy is None:
      self._goal_pose_w.zero_()
      return

    self._goal_pose_w[:, 0:2] = goal_xy
    com_z = self._robot_com_z()
    self._goal_pose_w[:, 2] = com_z + self.cfg.z_offset

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    goal_xy = getattr(self._env, "_loco_goal_pos", None)
    if goal_xy is None:
      return

    self._update_command()
    for env_idx in visualizer.get_env_indices(self.num_envs):
      visualizer.add_sphere(
        center=self._goal_pose_w[env_idx],
        radius=self.cfg.radius,
        color=self.cfg.color,
      )


@dataclass(kw_only=True)
class GoalPoseDebugCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  radius: float = 0.04
  z_offset: float = 0.025
  color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.95)

  def build(self, env: ManagerBasedRlEnv) -> GoalPoseDebugCommand:
    return GoalPoseDebugCommand(self, env)

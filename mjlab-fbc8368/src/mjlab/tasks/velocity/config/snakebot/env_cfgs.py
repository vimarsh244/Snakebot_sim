"""Snakebot chain_5 velocity environment configurations (flat terrain only)."""

from mjlab.asset_zoo.robots import (
  SNAKEBOT_ACTION_SCALE,
  get_snakebot_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

# reference body for viewer and orientation rewards (first module bottom plate)
SNAKE_ROOT_BODY = "m1_bottom-base-plate-v1"


def snakebot_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Snakebot flat terrain velocity configuration (no feet; snake uses full body contact)."""
  cfg = make_velocity_env_cfg()

  cfg.scene.entities = {"robot": get_snakebot_robot_cfg()}

  # flat terrain, no terrain scan
  cfg.sim.njmax = 2000
  cfg.sim.nconmax = 500
  cfg.sim.mujoco.ccd_iterations = 100
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  del cfg.observations["critic"].terms["foot_height"]
  del cfg.observations["critic"].terms["foot_air_time"]
  del cfg.observations["critic"].terms["foot_contact"]
  del cfg.observations["critic"].terms["foot_contact_forces"]

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = SNAKEBOT_ACTION_SCALE

  cfg.viewer.body_name = SNAKE_ROOT_BODY
  cfg.viewer.distance = 2.0
  cfg.viewer.elevation = -15.0

  cfg.events["base_com"].params["asset_cfg"].body_names = (SNAKE_ROOT_BODY,)
  cfg.events.pop("foot_friction", None)

  cfg.rewards["upright"].params["asset_cfg"].body_names = (SNAKE_ROOT_BODY,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (SNAKE_ROOT_BODY,)
  cfg.rewards["pose"].params["std_standing"] = {".*Revolute.*": 0.1}
  cfg.rewards["pose"].params["std_walking"] = {".*Revolute.*": 0.3}
  cfg.rewards["pose"].params["std_running"] = {".*Revolute.*": 0.3}

  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.0
  del cfg.rewards["foot_clearance"]
  del cfg.rewards["foot_swing_height"]
  del cfg.rewards["foot_slip"]
  del cfg.rewards["soft_landing"]

  cfg.terminations.pop("illegal_contact", None)

  cfg.curriculum.pop("terrain_levels", None)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.3, 0.3)

  return cfg

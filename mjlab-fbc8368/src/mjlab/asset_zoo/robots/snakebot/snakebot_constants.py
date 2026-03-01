"""Snakebot chain_5 constants (5-module closed-chain snake)."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import XmlPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

##
# MJCF and assets.
##

SNAKEBOT_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "snakebot" / "xmls" / "chain_5.xml"
)
assert SNAKEBOT_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, SNAKEBOT_XML.parent / "meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(SNAKEBOT_XML))
  meshdir = getattr(spec, "meshdir", None) or "meshes"
  spec.assets = get_assets(meshdir)
  return spec


##
# Actuator config (XML already defines position actuators; we wrap them).
##

# actuated joints: m1_Revolute-15, m1_Revolute-16, ... m5_Revolute-16 (10 total)
SNAKEBOT_ACTUATOR_CFG = XmlPositionActuatorCfg(
  target_names_expr=(".*Revolute-15", ".*Revolute-16"),
)

SNAKEBOT_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(SNAKEBOT_ACTUATOR_CFG,),
  soft_joint_pos_limit_factor=0.9,
)

##
# Initial state (frame at 0 0 0.08, snake horizontal; all joints at 0).
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.08),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)

##
# Action scale for position control (ctrlrange in XML is +/- 0.5236 rad).
##

SNAKEBOT_ACTION_SCALE: dict[str, float] = {
  ".*Revolute-15": 0.5,
  ".*Revolute-16": 0.5,
}

##
# Final config.
##


def get_snakebot_robot_cfg() -> EntityCfg:
  """Get a fresh Snakebot robot configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    spec_fn=get_spec,
    articulation=SNAKEBOT_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_snakebot_robot_cfg())
  viewer.launch(robot.spec.compile())

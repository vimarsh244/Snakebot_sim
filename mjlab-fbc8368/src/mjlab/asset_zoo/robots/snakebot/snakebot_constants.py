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
#
# DS3235 servo specifications (6.8V):
#   Stall torque:  35 kg·cm (3.43 N·m)
#   No-load speed: 0.11 s/60° (~9.5 rad/s)
#   Travel:        270° (limited to ±30° by ctrlrange in MJCF)
#
# MuJoCo position actuator:  force = kp*(ctrl - q) - kv*qdot
#   kp = 20.0  → saturates at ~10° error (servo is stiff)
#   kv = 0.5   → ~0.5× critical damping (fast, minimal overshoot)
#   forcerange = ±3.43 N·m (DS3235 stall torque)
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
#
# rot: 90° rotation about world Y axis so the snake's chain axis lies along
# world +X (matching the <frame euler="0 1.5707963267948966 0"> in chain_5.xml).
# quaternion (w, x, y, z) = (cos45°, 0, sin45°, 0) ≈ (0.7071068, 0, 0.7071068, 0).
# Without this, reset_root_state_uniform writes a (1,0,0,0) identity quat and
# the snake spawns with its chain axis vertical (on its side).
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.08),
  rot=(0.7071068, 0.0, 0.7071068, 0.0),  # 90° Y — snake horizontal along world +X
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)

##
# Action scale for position control (ctrlrange in XML is +/- 0.5236 rad).
# With kp=20, even small position deltas produce strong torque.  Scale=0.25
# maps network output ±1 → ±0.25 rad (±14°) target offset — well within the
# ±30° ctrlrange, giving the policy headroom without saturating.
##

SNAKEBOT_ACTION_SCALE: dict[str, float] = {
  ".*Revolute-15": 0.25,
  ".*Revolute-16": 0.25,
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

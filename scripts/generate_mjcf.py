from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from lxml import etree


@dataclass(frozen=True)
class SnakeConfig:
    num_modules: int = 5
    module_height: float = 0.11
    plate_mass: float = 0.05
    servo_mass: float = 0.07
    rod_mass: float = 0.02
    support_mass: float = 0.02
    plate_half_extents: tuple[float, float, float] = (0.06, 0.06, 0.008)
    rod_radius: float = 0.006
    rod_length: float = 0.11
    joint_limit_rad: float = 1.0471975512  # +-60 deg
    damping: float = 0.15
    frictionloss: float = 0.03
    armature: float = 0.003
    top_main_offset: tuple[float, float, float] = (0.028, -0.018, 0.0)
    top_support_offset: tuple[float, float, float] = (-0.028, 0.018, 0.0)


def _vec(v: tuple[float, float, float]) -> str:
    return f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"


def _add_mesh_asset(asset: etree._Element, mesh_name: str, filename: str) -> None:
    etree.SubElement(asset, "mesh", name=mesh_name, file=filename)


def _add_module(
    parent_plate: etree._Element,
    module_index: int,
    cfg: SnakeConfig,
    actuator_names: list[str],
    equality_node: etree._Element,
) -> etree._Element:
    main_joint_name = f"module_{module_index}_servo_a"
    second_joint_name = f"module_{module_index}_servo_b"
    actuator_names.extend([main_joint_name, second_joint_name])

    # module main branch: plate_i -> rod_main_i -> plate_{i+1}
    rod_main = etree.SubElement(
        parent_plate,
        "body",
        name=f"module_{module_index}_rod_main",
        pos=_vec(cfg.top_main_offset),
    )
    etree.SubElement(
        rod_main,
        "joint",
        name=main_joint_name,
        type="hinge",
        axis="0 0 1",
        range=f"{-cfg.joint_limit_rad:.6f} {cfg.joint_limit_rad:.6f}",
    )
    etree.SubElement(
        rod_main,
        "inertial",
        mass=f"{cfg.rod_mass:.6f}",
        pos=f"0 0 {cfg.rod_length / 2:.6f}",
        diaginertia="0.00001 0.00001 0.00001",
    )
    etree.SubElement(
        rod_main,
        "geom",
        name=f"module_{module_index}_rod_main_collision",
        type="capsule",
        fromto=f"0 0 0 0 0 {cfg.rod_length:.6f}",
        size=f"{cfg.rod_radius:.6f}",
        **{"class": "collision"},
    )
    etree.SubElement(
        rod_main,
        "geom",
        name=f"module_{module_index}_rod_main_visual",
        type="mesh",
        mesh="danda_v4",
        **{"class": "visual"},
    )
    etree.SubElement(
        rod_main,
        "geom",
        name=f"module_{module_index}_rod_main_bracket_visual",
        type="mesh",
        mesh="fusioncomponent__2",
        **{"class": "visual"},
    )

    plate_next = etree.SubElement(
        rod_main,
        "body",
        name=f"plate_{module_index + 1}",
        pos=f"0 0 {cfg.module_height:.6f}",
    )
    etree.SubElement(
        plate_next,
        "joint",
        name=second_joint_name,
        type="hinge",
        axis="1 0 0",
        range=f"{-cfg.joint_limit_rad:.6f} {cfg.joint_limit_rad:.6f}",
    )
    etree.SubElement(
        plate_next,
        "inertial",
        mass=f"{(cfg.plate_mass + cfg.servo_mass):.6f}",
        pos="0 0 0",
        diaginertia="0.00030 0.00030 0.00030",
    )
    etree.SubElement(
        plate_next,
        "geom",
        name=f"plate_{module_index + 1}_collision",
        type="box",
        size=_vec(cfg.plate_half_extents),
        **{"class": "collision"},
    )
    euler_val = "-1.5707963 0 0" if module_index % 2 == 0 else "1.5707963 0 0"
    etree.SubElement(
        plate_next,
        "geom",
        name=f"plate_{module_index + 1}_base_servo_visual",
        type="mesh",
        mesh="Servo_snake_base_plate",
        euler=euler_val,
        **{"class": "visual"},
    )
    etree.SubElement(
        plate_next,
        "geom",
        name=f"plate_{module_index + 1}_horn_visual",
        type="mesh",
        mesh="default",
        **{"class": "visual"},
    )

    # support branch with closed-loop equality constraint to plate_{i+1}
    support = etree.SubElement(
        parent_plate,
        "body",
        name=f"module_{module_index}_rod_support",
        pos=_vec(cfg.top_support_offset),
    )
    etree.SubElement(
        support,
        "joint",
        name=f"module_{module_index}_support_passive",
        type="hinge",
        axis="0 0 1",
        range=f"{-cfg.joint_limit_rad:.6f} {cfg.joint_limit_rad:.6f}",
    )
    etree.SubElement(
        support,
        "inertial",
        mass=f"{cfg.support_mass:.6f}",
        pos=f"0 0 {cfg.rod_length / 2:.6f}",
        diaginertia="0.00001 0.00001 0.00001",
    )
    etree.SubElement(
        support,
        "geom",
        name=f"module_{module_index}_rod_support_collision",
        type="capsule",
        fromto=f"0 0 0 0 0 {cfg.rod_length:.6f}",
        size=f"{cfg.rod_radius:.6f}",
        **{"class": "collision"},
    )
    etree.SubElement(
        support,
        "geom",
        name=f"module_{module_index}_rod_support_visual",
        type="mesh",
        mesh="danda_support_v5",
        **{"class": "visual"},
    )
    etree.SubElement(
        support,
        "geom",
        name=f"module_{module_index}_rod_support_bracket_visual",
        type="mesh",
        mesh="fusioncomponent__2",
        **{"class": "visual"},
    )

    # connect equality keeps support rod endpoint attached to the next plate.
    anchor = (
        cfg.top_support_offset[0],
        cfg.top_support_offset[1],
        cfg.module_height,
    )
    etree.SubElement(
        equality_node,
        "connect",
        name=f"module_{module_index}_loop_connect",
        body1=f"module_{module_index}_rod_support",
        body2=f"plate_{module_index + 1}",
        anchor=_vec(anchor),
        solref="0.005 1",
        solimp="0.95 0.99 0.001",
    )

    return plate_next


def build_model_xml(cfg: SnakeConfig) -> etree._ElementTree:
    mujoco = etree.Element("mujoco", model="snakebot")
    etree.SubElement(mujoco, "compiler", angle="radian", meshdir="assets", autolimits="true")
    etree.SubElement(
        mujoco,
        "option",
        timestep="0.002",
        gravity="0 0 -9.81",
        iterations="80",
        integrator="implicitfast",
    )

    default = etree.SubElement(mujoco, "default")
    robot_default = etree.SubElement(default, "default", **{"class": "snake"})
    etree.SubElement(
        robot_default,
        "joint",
        damping=f"{cfg.damping:.6f}",
        frictionloss=f"{cfg.frictionloss:.6f}",
        armature=f"{cfg.armature:.6f}",
    )
    etree.SubElement(robot_default, "position", kp="55", dampratio="1")
    visual_default = etree.SubElement(robot_default, "default", **{"class": "visual"})
    etree.SubElement(visual_default, "geom", type="mesh", contype="0", conaffinity="0", group="2")
    collision_default = etree.SubElement(robot_default, "default", **{"class": "collision"})
    etree.SubElement(collision_default, "geom", group="3", friction="1.2 0.03 0.01")

    asset = etree.SubElement(mujoco, "asset")
    _add_mesh_asset(asset, "fusioncomponent__2", "fusioncomponent__2.stl")
    _add_mesh_asset(asset, "danda_v4", "danda_v4.stl")
    _add_mesh_asset(asset, "danda_support_v5", "danda_support_v5.stl")
    _add_mesh_asset(asset, "Servo_snake_base_plate", "Servo_snake_base_plate.stl")
    _add_mesh_asset(asset, "default", "default.stl")
    etree.SubElement(asset, "material", name="plate_mat", rgba="0.9 0.9 0.9 1")
    etree.SubElement(asset, "material", name="rod_mat", rgba="0.95 0.95 0.95 1")
    etree.SubElement(asset, "material", name="servo_mat", rgba="0.1 0.1 0.15 1")

    worldbody = etree.SubElement(mujoco, "worldbody")
    root = etree.SubElement(worldbody, "body", name="plate_0", pos="0 0 0.13", childclass="snake")
    etree.SubElement(root, "freejoint", name="snake_freejoint")
    etree.SubElement(
        root,
        "inertial",
        mass=f"{(cfg.plate_mass + cfg.servo_mass):.6f}",
        pos="0 0 0",
        diaginertia="0.00030 0.00030 0.00030",
    )
    etree.SubElement(root, "site", name="head_imu", pos="0 0 0.01", size="0.003")
    etree.SubElement(
        root,
        "geom",
        name="plate_0_collision",
        type="box",
        size=_vec(cfg.plate_half_extents),
        **{"class": "collision"},
    )
    etree.SubElement(root, "geom", name="plate_0_base_servo_visual", type="mesh", mesh="Servo_snake_base_plate", euler="1.5707963 0 0", **{"class": "visual"})
    etree.SubElement(root, "geom", name="plate_0_horn_visual", type="mesh", mesh="default", **{"class": "visual"})

    equality = etree.SubElement(mujoco, "equality")
    actuator_names: list[str] = []

    parent_plate = root
    for i in range(cfg.num_modules):
        parent_plate = _add_module(
            parent_plate=parent_plate,
            module_index=i,
            cfg=cfg,
            actuator_names=actuator_names,
            equality_node=equality,
        )

    actuator = etree.SubElement(mujoco, "actuator")
    for name in actuator_names:
        etree.SubElement(actuator, "position", name=f"{name}_act", joint=name, **{"class": "snake"})

    sensor = etree.SubElement(mujoco, "sensor")
    etree.SubElement(sensor, "accelerometer", name="head_acc", site="head_imu")
    etree.SubElement(sensor, "gyro", name="head_gyro", site="head_imu")
    for name in actuator_names:
        etree.SubElement(sensor, "jointpos", name=f"{name}_pos", joint=name)
        etree.SubElement(sensor, "jointvel", name=f"{name}_vel", joint=name)

    return etree.ElementTree(mujoco)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MJCF model for Snakebot.")
    parser.add_argument("--modules", type=int, default=5, help="Number of snake modules.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("robot_desc") / "snakebot.xml",
        help="Output MJCF path.",
    )
    parser.add_argument(
        "--chain5",
        action="store_true",
        help="Use chain_5 style (real mesh bodies with exact CAD constants).",
    )
    args = parser.parse_args()

    if args.modules < 1:
        raise ValueError("modules must be >= 1")

    if args.chain5:
        from generate_chain5_style import build_chain5_xml  # type: ignore

        tree = build_chain5_xml(args.modules)
    else:
        cfg = SnakeConfig(num_modules=args.modules)
        tree = build_model_xml(cfg)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(args.output), pretty_print=True, xml_declaration=True, encoding="utf-8")
    print(f"generated {args.output} with {args.modules} modules")


if __name__ == "__main__":
    main()

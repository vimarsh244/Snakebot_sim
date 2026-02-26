"""
Generate fused_module.xml by loading robot.xml via MuJoCo, extracting
world-space geom transforms at zero configuration, and writing a single
rigid body that contains all those geoms at their correct positions.

Only geoms belonging to the kinematic chain rooted at 'fusioncomponent'
(i.e. NOT the separate fusioncomponent_7 / fusioncomponent_8 floating bodies)
are included, so the one assembled module is captured faithfully.
"""

import math
import mujoco
import numpy as np
from pathlib import Path

SRC_XML  = "robot_desc/robot.xml"
OUT_XML  = "robot_desc/fused_module.xml"
SCENE_XML = "robot_desc/fused_module_scene.xml"

# Bodies belonging to the main 'fusioncomponent' subtree (all child bodies of
# the first freejoint body).  The two extra floating bodies are excluded.
MAIN_ROOT = "fusioncomponent"
EXCLUDE_ROOTS = {"fusioncomponent_7", "fusioncomponent_8"}

# Mesh files that already exist for servo variants –
# each one is the same geometry, used as separate instances.
SERVO_STL_VARIANTS = [
    "servo_motor_35kg_motor_v1__2.stl",
    "servo_motor_35kg_motor_v1__3.stl",
    "servo_motor_35kg_motor_v1__4.stl",
    "servo_motor_35kg_motor_v1__5.stl",
]


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to MuJoCo quaternion (w, x, y, z)."""
    # Uses Shepperd's method
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def get_main_body_ids(model) -> set:
    """Return geom ids that belong to the main fusioncomponent subtree."""
    # Build parent map: body_id -> parent_body_id
    parent = {}
    for bid in range(model.nbody):
        parent[bid] = model.body_parentid[bid]

    # Get id of each root body we care about
    def body_id(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    main_root_id = body_id(MAIN_ROOT)
    exclude_ids = {body_id(n) for n in EXCLUDE_ROOTS}

    # Collect all body ids in main subtree via BFS
    in_main = set()
    queue = [main_root_id]
    while queue:
        bid = queue.pop()
        if bid in exclude_ids:
            continue
        in_main.add(bid)
        for child in range(model.nbody):
            if parent[child] == bid and child not in in_main:
                queue.append(child)

    # Collect geom ids whose body is in the main subtree
    geom_ids = set()
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] in in_main:
            geom_ids.add(gid)
    return geom_ids


def format_pos(v) -> str:
    return " ".join(f"{x:.8g}" for x in v)


def format_quat(q) -> str:
    return " ".join(f"{x:.8g}" for x in q)


def main():
    model = mujoco.MjModel.from_xml_path(SRC_XML)
    data  = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)   # compute xpos / xmat at zero config

    geom_ids = get_main_body_ids(model)

    print(f"Main subtree geom count: {len(geom_ids)}")

    # Collect per-mesh info: which meshes are in the main subtree?
    mesh_set = set()
    geom_infos = []   # list of (mesh_name, pos_world, mat_world, group)
    for gid in sorted(geom_ids):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh_id = model.geom_dataid[gid]
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        pos  = data.geom_xpos[gid].copy()
        mat  = data.geom_xmat[gid].reshape(3, 3).copy()
        quat = mat_to_quat(mat)
        group = int(model.geom_group[gid])
        mesh_set.add(mesh_name)
        geom_infos.append((mesh_name, pos, quat, group))

    print("Meshes used:", mesh_set)

    # Build XML manually (no lxml dependency in this helper)
    lines = []
    lines.append('<?xml version="1.0" ?>')
    lines.append('<!-- Auto-generated fused module: all parts in a single rigid body -->')
    lines.append('<mujoco model="fused-module">')
    lines.append('  <compiler angle="radian" meshdir="assets" autolimits="true"/>')
    lines.append('  <default>')
    lines.append('    <default class="fused-module">')
    lines.append('      <joint damping="0.01" frictionloss="0.01" armature="0.002"/>')
    lines.append('      <default class="visual">')
    lines.append('        <geom type="mesh" contype="0" conaffinity="0" group="2"/>')
    lines.append('      </default>')
    lines.append('      <default class="collision">')
    lines.append('        <geom group="3"/>')
    lines.append('      </default>')
    lines.append('    </default>')
    lines.append('  </default>')
    lines.append('  <worldbody>')
    lines.append('    <!-- Single fused body: all servo+fusioncomponent parts rigidly fixed -->')
    lines.append('    <body name="fused_module" pos="0 0 0.05" quat="1 0 0 0" childclass="fused-module">')
    lines.append('      <freejoint name="fused_module_freejoint"/>')

    # Calculate total mass
    total_mass = 0.0
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        total_mass += model.body_mass[bid]

    lines.append(f'      <inertial pos="0 0 0" mass="{total_mass:.4f}" fullinertia="0.001 0.001 0.001 0 0 0"/>')
    lines.append('')

    # Emit geoms in world-space positions
    for i, (mesh_name, pos, quat, group) in enumerate(geom_infos):
        cls = "visual" if group == 2 else "collision"
        lines.append(f'      <!-- {mesh_name} -->')
        lines.append(
            f'      <geom type="mesh" class="{cls}"'
            f' pos="{format_pos(pos)}"'
            f' quat="{format_quat(quat)}"'
            f' mesh="{mesh_name}"/>'
        )
    lines.append('')

    lines.append('    </body>')
    lines.append('  </worldbody>')
    lines.append('  <asset>')

    # Primary meshes
    meshes_declared = set()
    for mesh_name in sorted(mesh_set):
        lines.append(f'    <mesh name="{mesh_name}" file="{mesh_name}.stl"/>')
        meshes_declared.add(mesh_name)

    # Extra servo variant stl files (same geometry, additional instances)
    for stl in SERVO_STL_VARIANTS:
        mesh_name = stl.replace(".stl", "")
        if mesh_name not in meshes_declared:
            lines.append(f'    <!-- extra servo variant: {stl} -->')
            lines.append(f'    <mesh name="{mesh_name}" file="{stl}"/>')
            meshes_declared.add(mesh_name)

    lines.append('    <!-- Materials -->')
    lines.append('    <material name="servo_motor_35kg_motor_v1_material" rgba="0.101961 0.101961 0.101961 1"/>')
    lines.append('    <material name="fusioncomponent_material"   rgba="0.627451 0.627451 0.627451 1"/>')
    lines.append('    <material name="fusioncomponent__2_material" rgba="0.627451 0.627451 0.627451 1"/>')
    lines.append('    <material name="danda_v4_material"          rgba="0.627451 0.627451 0.627451 1"/>')
    lines.append('    <material name="danda_support_v5_material"  rgba="0.627451 0.627451 0.627451 1"/>')
    lines.append('    <material name="default_material" rgba="0.866667 0.909804 1 1"/>')
    lines.append('  </asset>')
    lines.append('</mujoco>')

    xml_str = "\n".join(lines)

    out = Path(OUT_XML)
    out.write_text(xml_str, encoding="utf-8")
    print(f"Written: {out}")

    # Quick validation
    try:
        m2 = mujoco.MjModel.from_xml_path(OUT_XML)
        print(f"Validation OK — bodies: {m2.nbody}, geoms: {m2.ngeom}, joints: {m2.njnt}")
    except Exception as e:
        print(f"Validation FAILED: {e}")


if __name__ == "__main__":
    main()

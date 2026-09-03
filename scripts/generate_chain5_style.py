"""
Generate a chain-N MJCF model using the exact CAD-derived constants from chain_5.xml.

Each module is identical in geometry/inertia – only the name prefix (m{i}_) and
the base Z position change. The chain stacks along the local Z axis inside a
frame that is rotated 90° around Y, so the snake lies horizontally in the world.

Key constants:
  MODULE_STEP_Z   = 0.136 m   (center-to-center along local Z inside the frame)
  BASE_Z_BOTTOM_M1 = 0.0145310940675635
  BASE_Z_TOP_M1    = 0.09490695867371797

Usage:
  from generate_chain5_style import build_chain5_xml
  tree = build_chain5_xml(20)
  tree.write("snake_description/chain_20.xml", pretty_print=True,
             xml_declaration=True, encoding="utf-8")
"""

from __future__ import annotations

from pathlib import Path

from lxml import etree

# ── Constants derived from chain_5.xml ──────────────────────────────────────
MODULE_STEP_Z = 0.136          # metres between modules
BASE_Z_BOTTOM = 0.0145310940675635
BASE_Z_TOP    = 0.09490695867371797


# ── Per-module body builders ─────────────────────────────────────────────────

def _bottom_body(parent: etree._Element, idx: int, z_b: float, is_first: bool) -> None:
    """Append the bottom-plate body (with all its children) for module idx."""
    body = etree.SubElement(
        parent, "body",
        name=f"m{idx}_bottom-base-plate-v1",
        pos=f"0.0 0.033146846605945184 {z_b:.16f}",
        euler="-3.141592653589787 -0.0 0.0",
    )
    etree.SubElement(body, "freejoint", name=f"m{idx}_bottom_plate_free")
    if is_first:
        etree.SubElement(body, "site", name="imu", pos="0 0 0", group="5")
    etree.SubElement(
        body, "geom",
        name=f"m{idx}_bottom-base-plate-v1_geom",
        type="mesh", mesh="bottom-base-plate-v1",
        pos="0 0 0", euler="0 0 0",
    )
    etree.SubElement(
        body, "inertial",
        mass="0.11423549467877546",
        pos="-0.005347637848628755 0.017762619548364243 0.005846593790389482",
        fullinertia="6.343319058534959e-05 7.281520644550472e-05 0.00012434394985893243 3.7136264751377626e-06 2.47114130118783e-06 3.8448439909348267e-07",
    )

    # ── bottom-diagonal-chintu-v1 ──
    bdc = etree.SubElement(
        body, "body",
        name=f"m{idx}_bottom-diagonal-chintu-v1",
        pos="0.04926263546510042 -0.04140998343068075 0.009999457199693682",
        euler="-1.5603770637091834 3.3923831869601494e-14 -0.7853981633974351",
    )
    etree.SubElement(bdc, "joint", name=f"m{idx}_Revolute-8", type="hinge",
        axis="-1.0000003094489522 3.8658312662143146e-14 1.2434497875801753e-14",
        pos="-0.0425002751358633 0.009999904807428087 -1.0747999264540658e-07")
    etree.SubElement(bdc, "geom", name=f"m{idx}_bottom-diagonal-chintu-v1_geom",
        type="mesh", mesh="bottom-diagonal-chintu-v1", pos="0 0 0", euler="0 0 0")
    etree.SubElement(bdc, "inertial",
        mass="0.004829931580386949",
        pos="-0.03749999999999955 0.011871044300627112 -7.670762931010015e-05",
        fullinertia="4.650328956622811e-07 2.982637335367364e-07 2.4558056057872756e-07 -1.401062240615398e-19 -4.2146220485210394e-20 9.58260291458006e-10")

    # ── bottom-side-chintu-v1 ──
    bsc = etree.SubElement(
        body, "body",
        name=f"m{idx}_bottom-side-chintu-v1",
        pos="-0.00799999999999995 0.016966177185297776 0.009999942800696776",
        euler="-1.5674140388758464 9.144902985720049e-17 -8.872539658711765e-15",
    )
    etree.SubElement(bsc, "joint", name=f"m{idx}_Revolute-9", type="hinge",
        axis="1.0 -6.143908029357409e-17 8.872798214762224e-15",
        pos="-0.03250000000000005 0.009999905414152896 -1.5307501481837577e-07")
    etree.SubElement(bsc, "geom", name=f"m{idx}_bottom-side-chintu-v1_geom",
        type="mesh", mesh="bottom-side-chintu-v1", pos="0 0 0", euler="0 0 0")
    etree.SubElement(bsc, "inertial",
        mass="0.004829931580386949",
        pos="-0.03749999999999956 0.011871044300627118 -7.670762931009596e-05",
        fullinertia="4.650328956622802e-07 2.9826373353673726e-07 2.45580560578726e-07 -1.41658964876519e-19 -4.2703888590353796e-20 9.582602914587785e-10")
    # danda-top-servo child of bsc
    dts = etree.SubElement(
        bsc, "body",
        name=f"m{idx}_danda-top-servo-v1",
        pos="-0.001039225735187278 0.030188428968929436 -0.0001000000000005552",
        euler="-3.141592653589793 -3.094550169498861e-15 -3.136427452099049",
    )
    etree.SubElement(dts, "joint", name=f"m{idx}_Revolute-13", type="hinge",
        axis="1.4489308988763791e-09 -2.805176106089602e-07 0.9999997189799212",
        pos="0.036499999974189734 -0.007499995003343646 -0.007499610395922852")
    etree.SubElement(dts, "geom", name=f"m{idx}_danda-top-servo-v1_geom",
        type="mesh", mesh="danda-top-servo-v1", pos="0 0 0", euler="0 0 0")
    etree.SubElement(dts, "inertial",
        mass="0.016605113423866974",
        pos="-2.3784152288882894e-05 0.02001791956679029 -0.01786461691933594",
        fullinertia="7.623057128479143e-06 1.7163411694359828e-05 1.8058312150822906e-05 7.320182465442226e-06 7.0554258389466934e-09 -5.315731795654608e-09")

    # ── bottom-chintu-and-servo-head-v1 ──
    bcs = etree.SubElement(
        body, "body",
        name=f"m{idx}_bottom-chintu-and-servo-head-v1",
        pos="-0.00909307013060929 0.01600000000000013 0.009999693148195961",
        euler="-1.5629623830476533 -1.0643858845815612e-16 -1.5707963267949052",
    )
    etree.SubElement(bcs, "joint", name=f"m{idx}_Revolute-15", type="hinge",
        axis="-1.0 6.33541605345897e-15 -8.555451894091773e-15",
        pos="-0.031599846605945045 0.009999903831887062 -2.6776042357766494e-07")
    etree.SubElement(bcs, "geom", name=f"m{idx}_bottom-chintu-and-servo-head-v1_geom",
        type="mesh", mesh="bottom-chintu-and-servo-head-v1", pos="0 0 0", euler="0 0 0")
    etree.SubElement(bcs, "inertial",
        mass="0.005733695228341913",
        pos="-0.03650460388444623 0.011576204158460035 -6.457349709795574e-05",
        fullinertia="5.102605793878946e-07 3.508074233995106e-07 3.0080962417111126e-07 8.993010602147903e-09 -3.6963826731627576e-10 1.0676610542722085e-09")
    proxy6 = etree.SubElement(bcs, "body", name=f"m{idx}_proxy_Rev6",
        pos="-0.0374998466 0.0225002604 0.0074000332")
    etree.SubElement(proxy6, "joint", name=f"m{idx}_Revolute-6", type="hinge", axis="0 0 -1")
    etree.SubElement(proxy6, "inertial", mass="1.8e-07", pos="0 0 0",
        diaginertia="1.8e-13 1.8e-13 1.8e-13")


def _top_body(parent: etree._Element, idx: int, z_t: float) -> None:
    """Append the top-plate body (with all its children) for module idx."""
    body = etree.SubElement(
        parent, "body",
        name=f"m{idx}_top-base-plate-v1",
        pos=f"-0.025385058611262133 0.006860259759015023 {z_t:.16f}",
        euler="-4.031953765759422e-12 -2.2077461834844432e-13 -1.5707963267948715",
    )
    etree.SubElement(body, "freejoint", name=f"m{idx}_top_plate_free")
    etree.SubElement(body, "geom", name=f"m{idx}_top-base-plate-v1_geom",
        type="mesh", mesh="top-base-plate-v1", pos="0 0 0", euler="0 0 0")
    etree.SubElement(body, "inertial",
        mass="0.11423549467877546",
        pos="-0.005347637848628756 0.01776261954836422 0.005846593790389466",
        fullinertia="6.343319058534974e-05 7.281520644550476e-05 0.00012434394985893216 3.713626475137759e-06 2.471141301187818e-06 3.8448439909350347e-07")

    # ── top-side-chintu-v1 ──
    tsc = etree.SubElement(
        body, "body",
        name=f"m{idx}_top-side-chintu-v1",
        pos="-0.00799999999999995 0.017078338636143844 0.00999969314819694",
        euler="-1.578630270538108 -1.306137021765825e-16 -8.880599230995793e-15",
    )
    etree.SubElement(tsc, "joint", name=f"m{idx}_Revolute-4", type="hinge",
        axis="1.0 2.2102639645820695e-13 -1.4600991065996208e-14",
        pos="-0.042499740240984595 0.009999958215754213 5.828571949048492e-08")
    etree.SubElement(tsc, "geom", name=f"m{idx}_top-side-chintu-v1_geom",
        type="mesh", mesh="top-side-chintu-v1", pos="0 0 0", euler="0 0 0")
    etree.SubElement(tsc, "inertial",
        mass="0.004829931580386949",
        pos="-0.03749999999999955 0.011871044300627109 -7.670762931009832e-05",
        fullinertia="4.6503289566228293e-07 2.982637335367454e-07 2.4558056057874233e-07 -1.3846477334768132e-19 -4.2602358610788335e-20 9.58260291458075e-10")
    # danda-bottom-servo child of tsc
    dbs = etree.SubElement(
        tsc, "body",
        name=f"m{idx}_danda-bottom-servo-v1-v1",
        pos="-0.0010394379249127627 0.030189435165771963 -0.00010000000000055742",
        euler="3.141592653589793 -2.9614914578805773e-15 -3.136399855324415",
    )
    etree.SubElement(dbs, "joint", name=f"m{idx}_Revolute-5", type="hinge",
        axis="7.20995457049534e-10 -1.388417767162614e-07 0.9999996862584414",
        pos="0.03649973832316588 -0.007499631354740782 -0.007500211881852813")
    etree.SubElement(dbs, "geom", name=f"m{idx}_danda-bottom-servo-v1-v1_geom",
        type="mesh", mesh="danda-bottom-servo-v1-v1", pos="0 0 0", euler="0 0 0")
    etree.SubElement(dbs, "inertial",
        mass="0.016605113423866974",
        pos="-2.3784152288883613e-05 0.02001791956679029 -0.017864616919335944",
        fullinertia="7.623057128479126e-06 1.7163411694359828e-05 1.8058312150822902e-05 7.320182465442222e-06 7.055425838946612e-09 -5.315731795651702e-09")

    # ── top-diagonal-chintu-v1 ──
    tdc = etree.SubElement(
        body, "body",
        name=f"m{idx}_top-diagonal-chintu-v1",
        pos="-0.0036230251583287965 0.011770373123872933 0.009999457199694799",
        euler="-1.5603770637118737 6.644630040307432e-15 2.356194490192345",
    )
    etree.SubElement(tdc, "joint", name=f"m{idx}_Revolute-7", type="hinge",
        axis="1.0000003094489522 -3.013877342139537e-12 6.2727600891321345e-15",
        pos="-0.03249975863110075 0.009999956330450638 -2.246789806976512e-07")
    etree.SubElement(tdc, "geom", name=f"m{idx}_top-diagonal-chintu-v1_geom",
        type="mesh", mesh="top-diagonal-chintu-v1", pos="0 0 0", euler="0 0 0")
    etree.SubElement(tdc, "inertial",
        mass="0.004829931580386949",
        pos="-0.03749999999999956 0.011871044300627121 -7.67076293101015e-05",
        fullinertia="4.650328956622769e-07 2.9826373353673753e-07 2.4558056057872316e-07 -1.4261443553100522e-19 -4.5825481478176893e-20 9.58260291460098e-10")
    # danda-support child of tdc
    ds = etree.SubElement(
        tdc, "body",
        name=f"m{idx}_danda-support-v1",
        pos="0.007299769040579557 0.07018889486988192 -0.00010000000000057963",
        euler="3.141592653589793 3.0759154446592663e-15 0.004207017307401544",
    )
    etree.SubElement(ds, "joint", name=f"m{idx}_Revolute-11", type="hinge",
        axis="-2.8891292943510017e-10 6.86758660859571e-08 0.999999434404338",
        pos="-0.044999711787780526 0.04750026372664425 -0.007499920962069091")
    etree.SubElement(ds, "geom", name=f"m{idx}_danda-support-v1_geom",
        type="mesh", mesh="danda-support-v1", pos="0 0 0", euler="0 0 0")
    etree.SubElement(ds, "inertial",
        mass="0.019527465720087768",
        pos="-2.9218977497941223e-15 0.019999999999999987 -0.023051676135472255",
        fullinertia="1.0376506423676357e-05 2.9562585312891514e-05 2.849558323957432e-05 1.0062330029436117e-05 1.3103303232492487e-18 -1.1130012926921505e-20")
    proxy10 = etree.SubElement(ds, "body", name=f"m{idx}_proxy_Rev10",
        pos="0.0449996782 -0.0075003154 0.0075000382")
    etree.SubElement(proxy10, "joint", name=f"m{idx}_Revolute-10", type="hinge", axis="0 0 1")
    etree.SubElement(proxy10, "inertial", mass="1.8e-07", pos="0 0 0",
        diaginertia="1.8e-13 1.8e-13 1.8e-13")

    # ── top-chintu-and-servo-head-v1 ──
    tcs = etree.SubElement(
        body, "body",
        name=f"m{idx}_top-chintu-and-servo-head-v1",
        pos="-0.00898090867971997 0.015099999999999631 0.009999942800695454",
        euler="-1.5741786147141739 4.058338165747889e-14 -1.5707963267948932",
    )
    etree.SubElement(tcs, "joint", name=f"m{idx}_Revolute-16", type="hinge",
        axis="-1.0 -3.991444155681031e-12 1.5061075503405146e-14",
        pos="-0.03250005861126232 0.009999958703343846 -8.886286865848563e-09")
    etree.SubElement(tcs, "geom", name=f"m{idx}_top-chintu-and-servo-head-v1_geom",
        type="mesh", mesh="top-chintu-and-servo-head-v1", pos="0 0 0", euler="0 0 0")
    etree.SubElement(tcs, "inertial",
        mass="0.005733695228341913",
        pos="-0.03650460388444623 0.011576204158460028 -6.457349709795596e-05",
        fullinertia="5.102605793879037e-07 3.5080742339951084e-07 3.008096241711188e-07 8.993010602150155e-09 -3.696382673168742e-10 1.067661054271072e-09")
    proxy14 = etree.SubElement(tcs, "body", name=f"m{idx}_proxy_Rev14",
        pos="-0.0375000586 0.0224997347 -0.0075997746")
    etree.SubElement(proxy14, "joint", name=f"m{idx}_Revolute-14", type="hinge", axis="0 0 -1")
    etree.SubElement(proxy14, "inertial", mass="1.8e-07", pos="0 0 0",
        diaginertia="1.8e-13 1.8e-13 1.8e-13")

    # ── battery ──
    bat = etree.SubElement(body, "body", name=f"battery_{idx}",
        pos="0.07386 0.049385 -0.03101", euler="0 0 4.71238898038469")
    etree.SubElement(bat, "geom", name=f"battery_{idx}_geom",
        type="mesh", mesh="battery_middle_module_v2",
        rgba="0.3 0.3 0.35 1")
    etree.SubElement(bat, "inertial", mass="0.027",
        pos="0.033 -0.083 0.059",
        diaginertia="3.096e-05 2.754e-05 5.508e-05")


# ── Top-level builder ─────────────────────────────────────────────────────────

def build_chain5_xml(num_modules: int) -> etree._ElementTree:
    """Build a chain_N MJCF using exact chain_5 CAD constants with real meshes."""
    mujoco = etree.Element("mujoco", model=f"snakebot_chain_{num_modules}")

    etree.SubElement(mujoco, "compiler", angle="radian", eulerseq="XYZ")
    etree.SubElement(mujoco, "option",
        timestep="0.0005", iterations="500", solver="Newton",
        impratio="10", tolerance="1e-14")

    # Defaults
    default = etree.SubElement(mujoco, "default")
    etree.SubElement(default, "joint", damping="0.5", armature="0.01")
    etree.SubElement(default, "geom",
        contype="1", conaffinity="1", condim="6",
        friction="0.3 1.05 0.001")

    # Assets – all meshes used by chain_5
    asset = etree.SubElement(mujoco, "asset")
    meshes = [
        ("bottom-base-plate-v1",            "meshes/bottom-base-plate-v1.stl"),
        ("top-base-plate-v1",               "meshes/top-base-plate-v1.stl"),
        ("danda-support-v1",                "meshes/danda-support-v1.stl"),
        ("danda-top-servo-v1",              "meshes/danda-top-servo-v1.stl"),
        ("danda-bottom-servo-v1-v1",        "meshes/danda-bottom-servo-v1-v1.stl"),
        ("bottom-chintu-and-servo-head-v1", "meshes/bottom-chintu-and-servo-head-v1.stl"),
        ("top-chintu-and-servo-head-v1",    "meshes/top-chintu-and-servo-head-v1.stl"),
        ("top-diagonal-chintu-v1",          "meshes/top-diagonal-chintu-v1.stl"),
        ("bottom-side-chintu-v1",           "meshes/bottom-side-chintu-v1.stl"),
        ("top-side-chintu-v1",              "meshes/top-side-chintu-v1.stl"),
        ("bottom-diagonal-chintu-v1",       "meshes/bottom-diagonal-chintu-v1.stl"),
        ("battery_middle_module_v2",        "meshes/battery_middle_module_v2.stl"),
    ]
    for name, filepath in meshes:
        etree.SubElement(asset, "mesh", name=name, file=filepath,
                         scale="0.001 0.001 0.001")

    # World body
    worldbody = etree.SubElement(mujoco, "worldbody")
    # Directional sun
    etree.SubElement(worldbody, "light",
        name="sun", pos="0 0 5", dir="0 -0.3 -1",
        diffuse="0.9 0.9 0.85", specular="0.3 0.3 0.3",
        directional="true", castshadow="true")
    # Fill / ambient light from the side
    etree.SubElement(worldbody, "light",
        name="fill", pos="3 3 3", dir="-1 -1 -1",
        diffuse="0.35 0.38 0.45", specular="0.05 0.05 0.05")
    # Visible ground plane – checkerboard grey
    etree.SubElement(worldbody, "geom", name="ground", type="plane",
        size="10 10 0.1", pos="0 0 0", euler="0 0 0",
        contype="1", conaffinity="1", condim="6",
        friction="0.3 1.05 0.001", rgba="0.55 0.62 0.55 1")

    frame = etree.SubElement(worldbody, "frame",
        pos="0 0 0.08",
        euler="0 1.5707963267948966 0")

    # Generate N modules
    for i in range(1, num_modules + 1):
        z_b = BASE_Z_BOTTOM + (i - 1) * MODULE_STEP_Z
        z_t = BASE_Z_TOP    + (i - 1) * MODULE_STEP_Z
        _bottom_body(frame, i, z_b, is_first=(i == 1))
        _top_body(frame, i, z_t)

    # Equality constraints
    equality = etree.SubElement(mujoco, "equality")
    weld_params = dict(solref="0.001 1", solimp="0.9999 0.99999 0.00001")
    for i in range(1, num_modules + 1):
        etree.SubElement(equality, "weld", name=f"m{i}_loop_Rev6",
            body1=f"m{i}_proxy_Rev6", body2=f"m{i}_danda-bottom-servo-v1-v1",
            **weld_params)
        etree.SubElement(equality, "weld", name=f"m{i}_loop_Rev10",
            body1=f"m{i}_proxy_Rev10", body2=f"m{i}_bottom-diagonal-chintu-v1",
            **weld_params)
        etree.SubElement(equality, "weld", name=f"m{i}_loop_Rev14",
            body1=f"m{i}_proxy_Rev14", body2=f"m{i}_danda-top-servo-v1",
            **weld_params)
    # Chain links: battery_i welds to m(i+1) bottom plate
    for i in range(1, num_modules):
        etree.SubElement(equality, "weld", name=f"chain_link_{i}",
            body1=f"battery_{i}", body2=f"m{i + 1}_bottom-base-plate-v1",
            **weld_params)

    # Actuators
    actuator = etree.SubElement(mujoco, "actuator")
    act_params = dict(kp="4.0", kv="0.12",
                      ctrlrange="-0.5236 0.5236", forcerange="-3.432 3.432")
    for i in range(1, num_modules + 1):
        etree.SubElement(actuator, "position",
            name=f"module_{i}_servo_bottom",
            joint=f"m{i}_Revolute-15", **act_params)
        etree.SubElement(actuator, "position",
            name=f"module_{i}_servo_top",
            joint=f"m{i}_Revolute-16", **act_params)

    # Sensors (same as chain_5)
    sensor = etree.SubElement(mujoco, "sensor")
    etree.SubElement(sensor, "gyro",         name="imu_ang_vel", site="imu")
    etree.SubElement(sensor, "velocimeter",  name="imu_lin_vel", site="imu")

    return etree.ElementTree(mujoco)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate chain_N MJCF with exact chain_5 CAD constants.")
    parser.add_argument("--modules", type=int, default=20,
                        help="Number of modules (default: 20)")
    parser.add_argument("--output", type=Path,
                        default=None,
                        help="Output path (default: snake_description/chain_<N>.xml)")
    args = parser.parse_args()

    out = args.output or (
        Path(__file__).resolve().parent.parent
        / "snake_description"
        / f"chain_{args.modules}.xml"
    )
    tree = build_chain5_xml(args.modules)
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out), pretty_print=True, xml_declaration=True, encoding="utf-8")
    print(f"Generated {out}  ({args.modules} modules)")


if __name__ == "__main__":
    main()

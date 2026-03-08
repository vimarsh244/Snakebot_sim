"""
circle_lifted.py  –  Lifted vertical loop / 3D circle for chain_20 snake robot.

Two modes:

  --mode loop    (default)
    Curl all joints in the PITCH (vertical) plane so the snake forms a full
    vertical circle.  The bottom of the circle rests on the ground; the head
    and tail meet in the air.  N * α = 2π  →  α = 2π/N ≈ 18° per module.

  --mode helix
    Apply the turn angle to BOTH servos simultaneously – one servo bends yaw,
    the other bends pitch – producing a 3D helix.  The head and tail also
    eventually meet in the air at the apex of the spiral.

Usage:
  python scripts/circle_lifted.py                     # vertical loop, viewer
  python scripts/circle_lifted.py --mode helix        # 3D helix
  python scripts/circle_lifted.py --servo bottom      # use only bottom servo
  python scripts/circle_lifted.py --modules 5         # test with chain_5.xml
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
from mujoco import viewer

HERE   = Path(__file__).resolve().parent
MODELS = HERE.parent / "snake_description"


def apply_torque_boost(model: mujoco.MjModel, multiplier: float) -> None:
    """Multiply kp and forcerange of every actuator by `multiplier`."""
    if multiplier <= 1.0:
        return
    for i in range(model.nu):
        kp_old = model.actuator_gainprm[i, 0]
        kv_old = -model.actuator_biasprm[i, 2]
        fl_old = model.actuator_forcerange[i, 1]

        kp_new = kp_old * multiplier
        fl_new = fl_old * multiplier

        model.actuator_gainprm[i, 0]  =  kp_new
        model.actuator_biasprm[i, 1]  = -kp_new
        model.actuator_biasprm[i, 2]  = -kv_old * multiplier
        model.actuator_forcerange[i]  = [-fl_new, fl_new]

    print(f"  [torque-boost] {multiplier}×  →  kp {kp_old:.1f}→{kp_new:.1f}  "
          f"forcerange ±{fl_old:.2f}→±{fl_new:.2f} N")


def run_circle_lifted(
    xml_path: Path,
    num_modules: int,
    mode: str = "loop",         # "loop" | "helix"
    servo: str = "bottom",      # which servo drives the main bend: "bottom" | "top"
    headless: bool = False,
    steps: int = 12_000,
    ramp_steps: int = 3_000,
    settle_steps: int = 1_000,
    torque_boost: float = 1.0,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    apply_torque_boost(model, torque_boost)

    turn_angle = 2.0 * math.pi / num_modules  # ≈ 0.3142 rad for N=20

    print(f"[circle_lifted] mode={mode}  modules={num_modules}")
    print(f"  turn_angle = {math.degrees(turn_angle):.2f}°/module  "
          f"  joint_limit = ±{math.degrees(0.5236):.1f}°")
    print(f"  Primary servo: {'servo_bottom (Revolute-15)' if servo=='bottom' else 'servo_top (Revolute-16)'}")

    def set_ctrl(step: int) -> None:
        if step < settle_steps:
            fraction = 0.0          # let the snake settle flat first
        else:
            fraction = min(1.0, (step - settle_steps) / max(1, ramp_steps))

        alpha = fraction * turn_angle

        for i in range(num_modules):
            bottom_idx = 2 * i       # module_i_servo_bottom  (Revolute-15)
            top_idx    = 2 * i + 1   # module_i_servo_top     (Revolute-16)

            if mode == "loop":
                # Vertical loop: one servo bends pitch, other stays flat
                if servo == "bottom":
                    data.ctrl[bottom_idx] = alpha
                    data.ctrl[top_idx]    = 0.0
                else:
                    data.ctrl[bottom_idx] = 0.0
                    data.ctrl[top_idx]    = alpha

            elif mode == "helix":
                # 3D helix: both servos active
                # Bottom bends pitch, top bends yaw → helix in 3D
                data.ctrl[bottom_idx] = alpha
                data.ctrl[top_idx]    = alpha

    print(f"  settle_steps={settle_steps}  ramp_steps={ramp_steps}  "
          f"total_steps={steps}")
    print("  The snake will settle flat, then gradually curl into a loop/helix.")

    if headless:
        for step in range(steps):
            set_ctrl(step)
            mujoco.mj_step(model, data)
        print(f"[circle_lifted] headless done at t={data.time:.3f}s")
        return

    with viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=False) as gui:
        step = 0
        while gui.is_running():
            set_ctrl(step)
            mujoco.mj_step(model, data)
            gui.sync()
            step += 1
            time.sleep(model.opt.timestep * 0.5)



def main() -> None:
    p = argparse.ArgumentParser(description="Lifted vertical loop / helix for chain snake.")
    p.add_argument("--modules",  type=int,  default=20,
                   help="Number of modules.")
    p.add_argument("--model",    type=Path, default=None,
                   help="Override XML path.")
    p.add_argument("--mode",     choices=["loop", "helix"], default="loop",
                   help="'loop' = vertical circle; 'helix' = 3D spiral (default: loop).")
    p.add_argument("--servo",    choices=["bottom", "top"], default="bottom",
                   help="Primary servo for the main bend axis (default: bottom).")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--steps",    type=int, default=12_000)
    p.add_argument("--ramp",     type=int, default=3_000)
    p.add_argument("--settle",   type=int, default=1_000)
    p.add_argument("--torque-boost", type=float, default=100.0,
                   help="Multiply every actuator's kp and forcerange by this factor. "
                        "Default 100× gives plenty of torque to form the loop. "
                        "Use 1 for original chain_5 limits.")
    args = p.parse_args()

    xml = args.model or (MODELS / f"chain_{args.modules}.xml")
    if not xml.exists():
        raise FileNotFoundError(
            f"{xml} not found – run: python scripts/generate_chain5_style.py --modules {args.modules}")

    run_circle_lifted(
        xml_path     = xml,
        num_modules  = args.modules,
        mode         = args.mode,
        servo        = args.servo,
        headless     = args.headless,
        steps        = args.steps,
        ramp_steps   = args.ramp,
        settle_steps = args.settle,
        torque_boost = args.torque_boost,
    )


if __name__ == "__main__":
    main()

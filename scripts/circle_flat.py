"""
circle_flat.py  –  Flat horizontal circle configuration for chain_20 snake robot.

The goal: 20 modules lie flat on the ground, bent so the snake forms a complete
circle (head touching tail) in the horizontal plane.

Math:
  Full circle → N modules each turn 2π/N = π/10 ≈ 18° ≈ 0.3142 rad.
  Joint limit is ±0.5236 rad (±30°), so 18° per module is well within range.

Usage:
  python scripts/circle_flat.py                    # uses chain_20.xml, viewer
  python scripts/circle_flat.py --servo bottom     # only bottom servos
  python scripts/circle_flat.py --servo top        # only top servos (default)
  python scripts/circle_flat.py --servo both       # both servos simultaneously
  python scripts/circle_flat.py --modules 5        # try with chain_5.xml
  python scripts/circle_flat.py --headless --steps 5000
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
    """Multiply kp and forcerange of every actuator by `multiplier`.

    For a MuJoCo `position` actuator the internal layout is:
      gainprm[0]   = kp
      biasprm[1]   = -kp   (must stay equal in magnitude for correct position control)
      biasprm[2]   = -kv
      forcerange   = [−limit, +limit]
    """
    if multiplier <= 1.0:
        return
    for i in range(model.nu):
        kp_old = model.actuator_gainprm[i, 0]
        kv_old = -model.actuator_biasprm[i, 2]          # kv stored as negative
        fl_old = model.actuator_forcerange[i, 1]         # upper limit

        kp_new = kp_old * multiplier
        fl_new = fl_old * multiplier

        model.actuator_gainprm[i, 0]  =  kp_new
        model.actuator_biasprm[i, 1]  = -kp_new          # position term
        model.actuator_biasprm[i, 2]  = -kv_old * multiplier  # velocity term
        model.actuator_forcerange[i]  = [-fl_new, fl_new]

    print(f"  [torque-boost] {multiplier}×  →  kp {kp_old:.1f}→{kp_new:.1f}  "
          f"forcerange ±{fl_old:.2f}→±{fl_new:.2f} N")


def run_circle_flat(
    xml_path: Path,
    num_modules: int,
    servo: str = "top",        # "bottom" | "top" | "both"
    headless: bool = False,
    steps: int = 10_000,
    ramp_steps: int = 2_000,   # gradually increase from 0 to target over this many steps
    torque_boost: float = 1.0, # multiply kp + forcerange by this factor
) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    apply_torque_boost(model, torque_boost)

    # Target angle per module so N modules close a full 2π circle.
    turn_angle = 2.0 * math.pi / num_modules  # ≈ 0.3142 rad for N=20

    print(f"[circle_flat] {num_modules} modules  |  turn angle per module = "
          f"{math.degrees(turn_angle):.2f}°  |  servo = {servo}")
    print(f"[circle_flat] joint limit = ±{math.degrees(0.5236):.1f}°  "
          f"  target = {math.degrees(turn_angle):.1f}°  ✓ within limit")

    def set_ctrl(t_step: int) -> None:
        # Ramp from 0 → turn_angle over ramp_steps
        fraction = min(1.0, t_step / max(1, ramp_steps))
        target   = fraction * turn_angle
        for i in range(num_modules):
            bottom_idx = 2 * i       # module_i_servo_bottom  (Revolute-15)
            top_idx    = 2 * i + 1   # module_i_servo_top     (Revolute-16)
            if servo in ("bottom", "both"):
                data.ctrl[bottom_idx] = target
            if servo in ("top", "both"):
                data.ctrl[top_idx] = target

    if headless:
        for step in range(steps):
            set_ctrl(step)
            mujoco.mj_step(model, data)
        print(f"[circle_flat] headless done at t={data.time:.3f}s")
        return

    # Interactive viewer
    with viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=False) as gui:
        step = 0
        while gui.is_running():
            set_ctrl(step)
            mujoco.mj_step(model, data)
            gui.sync()
            step += 1
            time.sleep(model.opt.timestep * 0.5)  # run at ~2× real-time


def main() -> None:
    p = argparse.ArgumentParser(description="Flat horizontal circle for chain snake.")
    p.add_argument("--modules",  type=int,  default=20,
                   help="Number of modules (corresponds to chain_<N>.xml).")
    p.add_argument("--model",    type=Path, default=None,
                   help="Override XML path (default: snake_description/chain_<N>.xml).")
    p.add_argument("--servo",    choices=["bottom", "top", "both"], default="top",
                   help="Which servo to use for horizontal bending (default: top).")
    p.add_argument("--headless", action="store_true",
                   help="Run without viewer.")
    p.add_argument("--steps",    type=int, default=10_000,
                   help="Steps for headless run.")
    p.add_argument("--ramp",     type=int, default=2_000,
                   help="Steps over which to ramp up to target angle.")
    p.add_argument("--torque-boost", type=float, default=100.0,
                   help="Multiply every actuator's kp and forcerange by this factor. "
                        "Default 100× gives plenty of torque to form the circle. "
                        "Use 1 to run with original chain_5 torque limits.")
    args = p.parse_args()

    xml = args.model or (MODELS / f"chain_{args.modules}.xml")
    if not xml.exists():
        raise FileNotFoundError(
            f"{xml} not found – run: python scripts/generate_chain5_style.py --modules {args.modules}")

    run_circle_flat(
        xml_path     = xml,
        num_modules  = args.modules,
        servo        = args.servo,
        headless     = args.headless,
        steps        = args.steps,
        ramp_steps   = args.ramp,
        torque_boost = args.torque_boost,
    )


if __name__ == "__main__":
    main()

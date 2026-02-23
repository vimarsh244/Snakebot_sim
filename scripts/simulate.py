from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
from mujoco import viewer


def collect_servo_joint_ids(model: mujoco.MjModel) -> list[int]:
    ids: list[int] = []
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and (name.endswith("_servo_a") or name.endswith("_servo_b")):
            ids.append(i)
    return ids


def run_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    module_count: int,
    sim_time: float,
    amp_yaw: float,
    amp_pitch: float,
    frequency: float,
    phase_step: float,
) -> None:
    dt = model.opt.timestep
    steps = max(1, int(sim_time / dt))
    for _ in range(steps):
        t = data.time
        for i in range(module_count):
            data.ctrl[2 * i] = amp_yaw * math.sin(2.0 * math.pi * frequency * t + phase_step * i)
            data.ctrl[2 * i + 1] = amp_pitch * math.sin(
                2.0 * math.pi * frequency * t + phase_step * i + math.pi / 2.0
            )
        mujoco.mj_step(model, data)
    print(f"headless run finished at sim time {data.time:.3f}s")


def run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    module_count: int,
    amp_yaw: float,
    amp_pitch: float,
    frequency: float,
    phase_step: float,
) -> None:
    with viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=False) as gui:
        while gui.is_running():
            t = data.time
            for i in range(module_count):
                data.ctrl[2 * i] = amp_yaw * math.sin(2.0 * math.pi * frequency * t + phase_step * i)
                data.ctrl[2 * i + 1] = amp_pitch * math.sin(
                    2.0 * math.pi * frequency * t + phase_step * i + math.pi / 2.0
                )
            mujoco.mj_step(model, data)
            gui.sync()
            time.sleep(model.opt.timestep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run snakebot MJCF simulation with serpentine control.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("robot_desc") / "snakebot_scene.xml",
        help="Path to MJCF scene file.",
    )
    parser.add_argument("--modules", type=int, default=5, help="Number of modules.")
    parser.add_argument("--amp-yaw", type=float, default=0.55, help="Yaw amplitude in radians.")
    parser.add_argument("--amp-pitch", type=float, default=0.35, help="Pitch amplitude in radians.")
    parser.add_argument("--frequency", type=float, default=0.6, help="Wave frequency in Hz.")
    parser.add_argument("--phase-step", type=float, default=0.7, help="Phase offset between neighboring modules.")
    parser.add_argument("--headless", action="store_true", help="Run without launching the viewer.")
    parser.add_argument("--sim-time", type=float, default=10.0, help="Simulation horizon for headless mode.")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

    expected_ctrl = args.modules * 2
    if model.nu < expected_ctrl:
        raise RuntimeError(f"model has {model.nu} actuators but {expected_ctrl} are expected for {args.modules} modules")

    joint_ids = collect_servo_joint_ids(model)
    print(f"loaded {args.model}")
    print(f"nbody={model.nbody} njnt={model.njnt} neq={model.neq} nu={model.nu}")
    print(f"servo joints discovered: {len(joint_ids)}")
    print(f"control settings: amp_yaw={args.amp_yaw}, amp_pitch={args.amp_pitch}, f={args.frequency}, phase={args.phase_step}")

    if args.headless:
        run_headless(
            model=model,
            data=data,
            module_count=args.modules,
            sim_time=args.sim_time,
            amp_yaw=args.amp_yaw,
            amp_pitch=args.amp_pitch,
            frequency=args.frequency,
            phase_step=args.phase_step,
        )
    else:
        run_viewer(
            model=model,
            data=data,
            module_count=args.modules,
            amp_yaw=args.amp_yaw,
            amp_pitch=args.amp_pitch,
            frequency=args.frequency,
            phase_step=args.phase_step,
        )


if __name__ == "__main__":
    main()

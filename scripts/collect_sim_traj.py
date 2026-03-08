#!/usr/bin/env python3
"""
collect_sim_traj.py
===================
Run the best gait (Az=Ay=0.6, f=3.0 Hz) on chain_5.xml in MuJoCo and record
the head-module IMU site trajectory to traj_output.csv.

The IMU site (named "imu") is defined on m1_bottom-base-plate-v1 at its local
origin (pos="0 0 0").  This is the exact point tracked by the T265 camera in
the real hardware experiments.

Output CSV columns (same schema as the T265 data):
    t, x, y, z, qw, qx, qy, qz, vx, vy, vz

All positions/orientations are in the MuJoCo world frame.
Timestamps start from 0 (settle period is stripped before saving).

Usage
-----
    python scripts/collect_sim_traj.py                  # writes traj_output.csv
    python scripts/collect_sim_traj.py --out my_traj.csv --duration 45
"""

import os
import sys
import csv
import argparse
import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from evaluate_gait import compute_angles, angles_to_ctrl, build_scene

# ── Default configuration (6/6/3 = Az=0.6, Ay=0.6, f=3.0) ──────────────────
NUM_MODULES   = 5
AMP_Z         = 0.6
AMP_Y         = 0.6
FREQUENCY     = 3.0
REQ_LENGTH    = 1.14
NUM_FRAMES    = 200
DURATION      = 30.0    # total simulation time [s]
SETTLE_TIME   = 2.0     # strip this many seconds from the start of the output
RECORD_DT     = 0.01    # recording interval [s]  -> 100 Hz output
GAIT_INTERVAL = 0.03    # how often to advance the gait frame [s]
OUTPUT_CSV    = "traj_output.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def rotmat_to_quat(mat9):
    """Convert a MuJoCo site_xmat (9-element row-major) to [w, x, y, z]."""
    R = mat9.reshape(3, 3)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=float)


def get_imu_row(model, data, site_id):
    """
    Return [t, x, y, z, qw, qx, qy, qz, vx, vy, vz] for the IMU site.

    - position / orientation : world frame
    - linear velocity        : world-frame translational velocity of the site
                               via mujoco.mj_objectVelocity
    """
    pos  = data.site_xpos[site_id].copy()           # (3,) world pos
    quat = rotmat_to_quat(data.site_xmat[site_id])  # (4,) [w,x,y,z]

    vel6 = np.zeros(6)
    mujoco.mj_objectVelocity(
        model, data,
        mujoco.mjtObj.mjOBJ_SITE, site_id,
        vel6, 0,   # flg_local=0 -> world frame
    )
    lin_vel = vel6[3:6]   # translational part of the 6-DOF spatial velocity

    return np.array([
        data.time,
        pos[0],  pos[1],  pos[2],
        quat[0], quat[1], quat[2], quat[3],
        lin_vel[0], lin_vel[1], lin_vel[2],
    ], dtype=float)


# ── Main simulation ───────────────────────────────────────────────────────────

def run(amp_z=AMP_Z, amp_y=AMP_Y, freq=FREQUENCY,
        duration=DURATION, settle=SETTLE_TIME,
        output=OUTPUT_CSV, num_modules=NUM_MODULES):

    print("=" * 60)
    print("  collect_sim_traj.py")
    print("=" * 60)
    print(f"  Gait : Az={amp_z}, Ay={amp_y}, f={freq} Hz")
    print(f"  Model: chain_{num_modules}.xml  (IMU site on head module m1)")
    print(f"  Sim  : {duration}s total, first {settle}s stripped")
    print(f"  Out  : {output}")
    print()

    # ── Gait ─────────────────────────────────────────────────────────────────
    print("Computing gait angles ...")
    angles_real, answers, R, approx_dist = compute_angles(
        num_segments    = num_modules,
        req_length      = REQ_LENGTH,
        req_amplitude_z = amp_z,
        req_amplitude_y = amp_y,
        frequency       = freq,
        num_frames      = NUM_FRAMES,
    )
    if not angles_real:
        raise RuntimeError("Gait computation returned no frames.")
    ctrl_frames = angles_to_ctrl(angles_real, num_modules, R, approx_dist)
    print(f"  {len(ctrl_frames)} gait frames.")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Loading chain_{num_modules}.xml ...")
    scene_path = build_scene(num_modules)
    model = mujoco.MjModel.from_xml_path(scene_path)
    data  = mujoco.MjData(model)
    print(f"  nq={model.nq}  nu={model.nu}  dt={model.opt.timestep*1000:.2f}ms")

    # ── Locate IMU site ───────────────────────────────────────────────────────
    try:
        imu_id = model.site("imu").id
        print(f"  IMU site id = {imu_id}  (m1_bottom-base-plate-v1, local origin)")
    except KeyError:
        imu_id = None
        head_bid = model.body("m1_bottom-base-plate-v1").id
        print("  WARNING: 'imu' site not found; falling back to body COM.")

    # ── Simulation loop ───────────────────────────────────────────────────────
    dt             = model.opt.timestep
    steps_per_gait = max(1, int(GAIT_INTERVAL / dt))
    record_every   = max(1, int(RECORD_DT / dt))
    total_steps    = int(duration / dt)
    progress_step  = max(1, total_steps // 20)

    rows      = []
    frame_idx = 0

    print(f"Running simulation ({total_steps} steps) ...")
    for step in range(total_steps):
        # advance gait frame
        if step % steps_per_gait == 0:
            ctrl = ctrl_frames[frame_idx % len(ctrl_frames)]
            data.ctrl[:len(ctrl)] = ctrl
            frame_idx += 1

        mujoco.mj_step(model, data)

        # record after settle period
        if step % record_every == 0 and data.time >= settle:
            if imu_id is not None:
                row = get_imu_row(model, data, imu_id)
            else:
                # fallback: body COM + body quaternion + body COM velocity
                pos  = data.xpos[head_bid].copy()
                quat = data.xquat[head_bid].copy()
                vel  = data.cvel[head_bid, 3:6].copy()
                row  = np.array([data.time,
                                 pos[0],  pos[1],  pos[2],
                                 quat[0], quat[1], quat[2], quat[3],
                                 vel[0],  vel[1],  vel[2]], dtype=float)
            rows.append(row)

        if step % progress_step == 0:
            print(f"  {100*step/total_steps:4.0f}%  t={data.time:.2f}s", end="\r", flush=True)

    print(f"\n  Recorded {len(rows)} samples at {1/RECORD_DT:.0f} Hz.")

    # ── Post-process & save ───────────────────────────────────────────────────
    arr = np.array(rows, dtype=float)

    # zero-origin timestamps so t[0] == 0
    arr[:, 0] -= arr[0, 0]

    cols = ["t", "x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz"]
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for row in arr:
            writer.writerow([f"{v:.8g}" for v in row])

    print(f"  Saved -> {output}  ({len(arr)} rows)")
    t_span = arr[-1, 0] - arr[0, 0]
    fwd    = arr[-1, 1] - arr[0, 1]   # crude forward (X axis in sim)
    print(f"  Duration : {t_span:.2f} s")
    print(f"  Net X    : {fwd*100:.1f} cm  (sim forward axis)")
    print("Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Collect head-IMU trajectory from MuJoCo sim (6/6/3 gait).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--amp-z",    type=float, default=AMP_Z,
                   help="Vertical amplitude Az")
    p.add_argument("--amp-y",    type=float, default=AMP_Y,
                   help="Horizontal amplitude Ay")
    p.add_argument("--freq",     type=float, default=FREQUENCY,
                   help="Spatial frequency [Hz]")
    p.add_argument("--duration", type=float, default=DURATION,
                   help="Simulation duration [s]")
    p.add_argument("--settle",   type=float, default=SETTLE_TIME,
                   help="Settle period stripped from output [s]")
    p.add_argument("--out",      type=str,   default=OUTPUT_CSV,
                   help="Output CSV path")
    p.add_argument("--num",      type=int,   default=NUM_MODULES,
                   help="Number of snake modules")
    args = p.parse_args()

    run(amp_z=args.amp_z, amp_y=args.amp_y, freq=args.freq,
        duration=args.duration, settle=args.settle,
        output=args.out, num_modules=args.num)


if __name__ == "__main__":
    main()

"""
Intel RealSense T265 Bag File Trajectory Plotter
=================================================
Reads recorded .bag files from the T265 tracking camera and plots
the XYZ trajectory using matplotlib.

Usage:
    python scripts/plot_trajectory.py bags/20260302_160408.bag
    python scripts/plot_trajectory.py bags/*.bag --swap yz
    python scripts/plot_trajectory.py bags/20260302_160408.bag --flip x z --transpose xzy
    python scripts/plot_trajectory.py bags/20260302_160408.bag --csv output.csv
"""

import argparse
import sys
import os
import glob
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("ERROR: pyrealsense2 not installed. Install with:")
    print("  pip install pyrealsense2")
    sys.exit(1)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ─────────────────────────────────────────────────────────────────────
# Core extraction
# ─────────────────────────────────────────────────────────────────────

def extract_trajectory(bag_path: str) -> dict:
    """
    Extract pose trajectory from a T265 .bag file.

    Returns a dict with numpy arrays:
        t   – timestamps (seconds, relative to first frame)
        x, y, z – translation in metres
        qw, qx, qy, qz – orientation quaternion
        vx, vy, vz – velocity
    """
    if not os.path.isfile(bag_path):
        raise FileNotFoundError(f"Bag file not found: {bag_path}")

    pipeline = rs.pipeline()
    config = rs.config()

    # Load the bag file; disable looping
    rs.config.enable_device_from_file(config, bag_path, repeat_playback=False)

    # Request the pose stream (T265 tracking data)
    config.enable_stream(rs.stream.pose)

    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        # If there's no pose stream, the bag may be from a D4xx camera
        raise RuntimeError(
            f"Could not open pose stream in '{bag_path}'. "
            "Make sure this .bag was recorded from a T265 tracking camera.\n"
            f"  Inner error: {e}"
        )

    # Process as fast as possible (don't wait for real-time playback)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    timestamps = []
    positions = []       # (x, y, z)
    orientations = []    # (qw, qx, qy, qz)
    velocities = []      # (vx, vy, vz)

    consecutive_failures = 0
    max_consecutive_failures = 30  # stop after this many timeouts in a row

    while True:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=100)
            consecutive_failures = 0
        except RuntimeError:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                break
            continue

        pose_frame = frames.first_or_default(rs.stream.pose)
        if not pose_frame:
            continue

        pose = pose_frame.as_pose_frame().get_pose_data()
        ts = pose_frame.get_timestamp() / 1000.0  # ms -> s

        timestamps.append(ts)
        positions.append((pose.translation.x,
                          pose.translation.y,
                          pose.translation.z))
        orientations.append((pose.rotation.w,
                             pose.rotation.x,
                             pose.rotation.y,
                             pose.rotation.z))
        velocities.append((pose.velocity.x,
                           pose.velocity.y,
                           pose.velocity.z))

    pipeline.stop()

    if len(timestamps) == 0:
        raise RuntimeError(f"No pose frames found in '{bag_path}'.")

    t = np.array(timestamps)
    t -= t[0]  # make relative
    pos = np.array(positions)
    ori = np.array(orientations)
    vel = np.array(velocities)

    return {
        "t": t,
        "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2],
        "qw": ori[:, 0], "qx": ori[:, 1], "qy": ori[:, 2], "qz": ori[:, 3],
        "vx": vel[:, 0], "vy": vel[:, 1], "vz": vel[:, 2],
    }


# ─────────────────────────────────────────────────────────────────────
# Coordinate transforms
# ─────────────────────────────────────────────────────────────────────

AXIS_MAP = {"x": 0, "y": 1, "z": 2}


def apply_flips(x, y, z, flip_axes: list):
    """Negate the specified axes.  e.g. flip_axes=['x','z']"""
    out = [x.copy(), y.copy(), z.copy()]
    for ax in flip_axes:
        idx = AXIS_MAP[ax.lower()]
        out[idx] = -out[idx]
    return out[0], out[1], out[2]


def apply_transpose(x, y, z, order: str):
    """
    Reorder axes.  order is a 3-char string like 'xzy', 'zxy', etc.
    Each character says which *source* axis goes into the 1st, 2nd, 3rd
    position of the result.
    e.g. 'xzy' → result_x=source_x, result_y=source_z, result_z=source_y
    """
    src = {"x": x, "y": y, "z": z}
    order = order.lower()
    if sorted(order) != ["x", "y", "z"] or len(order) != 3:
        raise ValueError(f"Invalid transpose order '{order}'. Must be a permutation of 'xyz'.")
    return src[order[0]], src[order[1]], src[order[2]]


def apply_swap(x, y, z, pair: str):
    """
    Swap two axes. pair is a 2-char string like 'xy', 'xz', 'yz'.
    """
    pair = pair.lower()
    if sorted(pair) not in [["x", "y"], ["x", "z"], ["y", "z"]]:
        raise ValueError(f"Invalid swap pair '{pair}'. Use 'xy', 'xz', or 'yz'.")
    src = {"x": x.copy(), "y": y.copy(), "z": z.copy()}
    a, b = pair[0], pair[1]
    src[a], src[b] = src[b], src[a]
    return src["x"], src["y"], src["z"]


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────

def plot_trajectories(datasets: list[dict], labels: list[str], show_velocity: bool = False):
    """
    Plot one or more trajectories.
    Each dataset is a dict returned by extract_trajectory (after transforms).
    """
    n = len(datasets)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 1)))

    # ── Figure 1: 3D trajectory ──
    fig3d = plt.figure("3D Trajectory", figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection="3d")
    for i, (d, label) in enumerate(zip(datasets, labels)):
        ax3d.plot(d["x"], d["y"], d["z"], color=colors[i], label=label, linewidth=1.2)
        ax3d.scatter(d["x"][0], d["y"][0], d["z"][0],
                     color=colors[i], marker="o", s=60, zorder=5)
        ax3d.scatter(d["x"][-1], d["y"][-1], d["z"][-1],
                     color=colors[i], marker="x", s=60, zorder=5)
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("T265 Pose Trajectory (3D)")
    ax3d.legend()

    # ── Figure 2: XYZ vs time ──
    fig_ts, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig_ts.canvas.manager.set_window_title("XYZ vs Time")
    axis_names = ["X", "Y", "Z"]
    axis_keys = ["x", "y", "z"]
    for i, (d, label) in enumerate(zip(datasets, labels)):
        for j, (key, name) in enumerate(zip(axis_keys, axis_names)):
            axes[j].plot(d["t"], d[key], color=colors[i], label=label, linewidth=1)
            axes[j].set_ylabel(f"{name} (m)")
            axes[j].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("Position vs Time")
    axes[0].legend()
    fig_ts.tight_layout()

    # ── Figure 3: 2D top-down (XZ plane) ──
    fig2d, ax2d = plt.subplots(figsize=(10, 8))
    fig2d.canvas.manager.set_window_title("2D Top-Down (XZ)")
    for i, (d, label) in enumerate(zip(datasets, labels)):
        ax2d.plot(d["x"], d["z"], color=colors[i], label=label, linewidth=1.2)
        ax2d.plot(d["x"][0], d["z"][0], "o", color=colors[i], ms=8)
        ax2d.plot(d["x"][-1], d["z"][-1], "x", color=colors[i], ms=8)
    ax2d.set_xlabel("X (m)")
    ax2d.set_ylabel("Z (m)")
    ax2d.set_title("Top-Down View (XZ Plane)")
    ax2d.set_aspect("equal")
    ax2d.grid(True, alpha=0.3)
    ax2d.legend()

    # ── Optional: velocity ──
    if show_velocity:
        fig_vel, vaxes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig_vel.canvas.manager.set_window_title("Velocity vs Time")
        vel_keys = ["vx", "vy", "vz"]
        for i, (d, label) in enumerate(zip(datasets, labels)):
            for j, (key, name) in enumerate(zip(vel_keys, axis_names)):
                vaxes[j].plot(d["t"], d[key], color=colors[i], label=label, linewidth=1)
                vaxes[j].set_ylabel(f"V{name} (m/s)")
                vaxes[j].grid(True, alpha=0.3)
        vaxes[-1].set_xlabel("Time (s)")
        vaxes[0].set_title("Velocity vs Time")
        vaxes[0].legend()
        fig_vel.tight_layout()

    plt.show()


# ─────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────

def export_csv(data: dict, path: str):
    """Write trajectory to CSV."""
    header = "t,x,y,z,qw,qx,qy,qz,vx,vy,vz"
    arr = np.column_stack([data[k] for k in header.split(",")])
    np.savetxt(path, arr, delimiter=",", header=header, comments="", fmt="%.8f")
    print(f"Exported {len(data['t'])} samples -> {path}")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot T265 tracking trajectory from .bag files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_trajectory.py bags/20260302_160408.bag
  python plot_trajectory.py bags/*.bag
  python plot_trajectory.py bags/20260302_160408.bag --flip x z
  python plot_trajectory.py bags/20260302_160408.bag --swap yz
  python plot_trajectory.py bags/20260302_160408.bag --transpose xzy
  python plot_trajectory.py bags/20260302_160408.bag --csv trajectory.csv
  python plot_trajectory.py bags/20260302_160408.bag --velocity
""",
    )
    parser.add_argument(
        "bags", nargs="+",
        help="One or more .bag file paths (supports glob on Windows too)")
    parser.add_argument(
        "--flip", nargs="+", choices=["x", "y", "z"], default=[],
        help="Negate (mirror) one or more axes.  e.g. --flip x z")
    parser.add_argument(
        "--swap", type=str, default=None,
        help="Swap two axes. ''. e.g. --swap yz  or  --swap xz")
    parser.add_argument(
        "--transpose", type=str, default=None,
        help="Reorder axes with a 3-letter permutation of xyz. "
             "e.g. --transpose xzy  swaps Y<->Z")
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Export first trajectory to CSV file")
    parser.add_argument(
        "--velocity", action="store_true",
        help="Also plot velocity vs time")

    args = parser.parse_args()

    # Expand globs (Windows doesn't expand * in cmd)
    bag_files = []
    for pattern in args.bags:
        expanded = glob.glob(pattern)
        if expanded:
            bag_files.extend(expanded)
        else:
            bag_files.append(pattern)  # let it fail with a clear message later

    if not bag_files:
        parser.error("No .bag files specified.")

    print(f"Processing {len(bag_files)} bag file(s)...\n")

    datasets = []
    labels = []

    for bag_path in bag_files:
        name = os.path.basename(bag_path)
        print(f"  Reading: {name} ... ", end="", flush=True)
        try:
            data = extract_trajectory(bag_path)
        except Exception as e:
            print(f"FAILED\n    → {e}")
            continue

        # Apply coordinate transforms in order: flip → swap → transpose
        x, y, z = data["x"], data["y"], data["z"]
        vx, vy, vz = data["vx"], data["vy"], data["vz"]

        if args.flip:
            x, y, z = apply_flips(x, y, z, args.flip)
            vx, vy, vz = apply_flips(vx, vy, vz, args.flip)
        if args.swap:
            x, y, z = apply_swap(x, y, z, args.swap)
            vx, vy, vz = apply_swap(vx, vy, vz, args.swap)
        if args.transpose:
            x, y, z = apply_transpose(x, y, z, args.transpose)
            vx, vy, vz = apply_transpose(vx, vy, vz, args.transpose)

        data["x"], data["y"], data["z"] = x, y, z
        data["vx"], data["vy"], data["vz"] = vx, vy, vz

        duration = data["t"][-1] - data["t"][0]
        print(f"OK  ({len(data['t'])} frames, {duration:.1f}s)")

        datasets.append(data)
        labels.append(name)

    if not datasets:
        print("\nNo valid trajectories found.")
        sys.exit(1)

    # CSV export (first file only)
    if args.csv:
        export_csv(datasets[0], args.csv)

    # Print summary
    print("\n-- Summary --")
    for label, d in zip(labels, datasets):
        rng = lambda k: (d[k].min(), d[k].max())
        print(f"  {label}:")
        print(f"    Frames : {len(d['t'])}")
        print(f"    Duration: {d['t'][-1]:.2f}s")
        for ax in ["x", "y", "z"]:
            lo, hi = rng(ax)
            print(f"    {ax.upper()} range : [{lo:+.4f}, {hi:+.4f}] m  (span {hi-lo:.4f})")
        # Total distance traveled
        dx = np.diff(d["x"])
        dy = np.diff(d["y"])
        dz = np.diff(d["z"])
        dist = np.sum(np.sqrt(dx**2 + dy**2 + dz**2))
        print(f"    Distance: {dist:.4f} m")

    # Plot
    transforms_desc = []
    if args.flip:
        transforms_desc.append(f"flip({','.join(args.flip)})")
    if args.swap:
        transforms_desc.append(f"swap({args.swap})")
    if args.transpose:
        transforms_desc.append(f"transpose({args.transpose})")
    if transforms_desc:
        print(f"\n  Coordinate transforms applied: {' → '.join(transforms_desc)}")

    plot_trajectories(datasets, labels, show_velocity=args.velocity)


if __name__ == "__main__":
    main()

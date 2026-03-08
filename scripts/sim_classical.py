# -*- coding: utf-8 -*-
"""
sim_classical.py  –  Run the classical serpentine gait from classical.py
                     on the MuJoCo snakebot simulation (no real hardware needed).

Usage (from repo root):
    python scripts/sim_classical.py          # uses chain_5 (5-module default)
    python scripts/sim_classical.py --num 3  # 3-module chain

Controls while the viewer is open:
    ESC / close window  →  quit
    SPACE               →  pause / resume stepping (built-in MuJoCo viewer)

Angle mapping
─────────────
classical.py computes, for each inter-module joint pair (horizontal, vertical):

    angle_1 = atan(R·sin(θ_h) / (R·cos(θ_h) − d)) * 180/π    [degrees]
    angle_2 = atan(R·sin(θ_v) / (R·cos(θ_v) − d)) * 180/π    [degrees]

and then sends to Arduino:
    servo_bottom = clamp(90 − int(angle_1), 70, 110)   ← horizontal servo
    servo_top    = clamp(90 − int(angle_2), 70, 110)   ← vertical   servo

The servo value sits in [70, 110]°, with neutral at 90°, so the actual
deflection is  (servo − 90)° = −int(angle) degrees.

For MuJoCo the actuators are position-controlled in *radians*, ctrlrange ±0.5236 rad.
We therefore convert:
    ctrl_bottom = clamp(−angle_1 * π/180, −0.5236, 0.5236)
    ctrl_top    = clamp(−angle_2 * π/180, −0.5236, 0.5236)
"""

import sys
import os
import math
import time
import argparse
import numpy as np
from copy import deepcopy
from math import atan, sin, cos, sqrt
from scipy.interpolate import splprep, splev

import mujoco
from mujoco import viewer

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



def compute_angles(num_segments=5):
    """
    Reproduce the angle-computation loop from classical.py.

    Returns
    -------
    angles_real  : list of lists of (horiz, vert) tuples per gait frame
    answers      : list of lists of [x, y, z] keypoints per gait frame
    x_new, y_new : spline backbone arrays
    x_sine_wave, y_sine_wave, z_wave : 3-D sine-wave path arrays
    R, approximation_circle_dist     : geometry constants
    """
    req_length        = 1.35
    req_amplitude_z   = 0.8
    req_amplitude_y   = 0.0
    R                 = req_length / 2
    approximation_circle_dist = 0.17
    accuracylevel     = 100_000
    frequency         = 2.0

    # Path spline
    path_points_x = [0, 3, 4, 6]
    path_points_y = [0, 3, 4, 6]

    # path_points_x *= 5
    # path_points_y *= 5

    points = np.array([path_points_x, path_points_y])
    tck, u = splprep(points, s=0)
    u_new  = np.linspace(0, 1, accuracylevel)
    x_new, y_new = splev(u_new, tck)
    dx, dy = splev(u_new, tck, der=1)

    magnitude         = np.sqrt(dx**2 + dy**2)
    dx_normalized     = -dy / magnitude
    dy_normalized     =  dx / magnitude

    A1 = 0
    x_sine_wave = A1 + x_new + req_amplitude_y * np.sin(2 * np.pi * frequency * u_new) * dx_normalized
    y_sine_wave = A1 + y_new + req_amplitude_y * np.sin(2 * np.pi * frequency * u_new) * dy_normalized
    z_wave      = req_amplitude_z * np.abs(np.sin(2 * np.pi * frequency * u_new))

    angles_ground_apparent        = []
    angles_ground_real            = []
    angles_relative               = []
    angles_real                   = []
    answers                       = []
    answers_per_iteration         = [[x_sine_wave[0], y_sine_wave[0], z_wave[0]]]
    angles_real_per_iteration     = []
    angles_relative_per_iteration = []

    offset = 0
    while offset < accuracylevel:
        for i in range(offset, len(x_sine_wave)):
            prev = answers_per_iteration[-1]
            if sqrt((x_sine_wave[i]-prev[0])**2 +
                    (y_sine_wave[i]-prev[1])**2 +
                    (z_wave[i]-prev[2])**2) > req_length:
                answers_per_iteration.append([x_sine_wave[i], y_sine_wave[i], z_wave[i]])
            if len(answers_per_iteration) == num_segments + 1:
                for j in range(len(answers_per_iteration) - 1):
                    p0 = answers_per_iteration[j]
                    p1 = answers_per_iteration[j+1]
                    dx01 = p1[0] - p0[0]
                    dy01 = p1[1] - p0[1]
                    dz01 = p1[2] - p0[2]
                    angles_ground_apparent.append([
                        atan(dy01 / dx01) * 180 / np.pi,
                        atan(dz01 / dx01) * 180 / np.pi,
                    ])
                    angles_ground_real.append([
                        atan(dy01 / sqrt(dx01**2 + dz01**2)) * 180 / np.pi,
                        atan(dz01 / sqrt(dx01**2 + dy01**2)) * 180 / np.pi,
                    ])
                    if j:
                        angles_real_per_iteration.append((
                            round(180 + angles_ground_real[j][0] - angles_ground_real[j-1][0], 3),
                            round(180 + angles_ground_real[j][1] - angles_ground_real[j-1][1], 3),
                        ))
                        angles_relative_per_iteration.append((
                            round(180 + angles_ground_apparent[j][0] - angles_ground_apparent[j-1][0], 3),
                            round(180 + angles_ground_apparent[j][1] - angles_ground_apparent[j-1][1], 3),
                        ))
                break

        if len(answers_per_iteration) < num_segments + 1:
            break

        answers.append(deepcopy(answers_per_iteration))
        angles_real.append(deepcopy(angles_real_per_iteration))
        angles_relative.append(deepcopy(angles_relative_per_iteration))

        answers_per_iteration.clear()
        angles_real_per_iteration.clear()
        angles_ground_apparent.clear()
        angles_ground_real.clear()
        angles_relative_per_iteration.clear()

        offset += int(accuracylevel / 150)
        answers_per_iteration.append([x_sine_wave[offset], y_sine_wave[offset], z_wave[offset]])

    # Overwrite all angles with the first module's angles for every frame
    # if angles_real:
    #     for frame in angles_real:
    #         if frame:
    #             first_angle = frame[0]
    #             for i in range(len(frame)):
    #                 # Add small random noise to each angle (in degrees)
    #                 noise = np.random.uniform(-3, 3, size=2)  # e.g., ±2 degrees noise
    #                 frame[i] = (first_angle[0] + noise[0], first_angle[1] + noise[1])

    return (angles_real, answers,
            x_new, y_new, x_sine_wave, y_sine_wave, z_wave,
            R, approximation_circle_dist)


def show_gait_graph(answers, x_new, y_new, x_sine_wave, y_sine_wave, z_wave):
    """Exact graph rendering from classical.py — blocks until window closed."""
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1 in x:y:z

    # Plot the original spline in 3D (XY plane)
    ax.plot(x_new, y_new, np.zeros_like(x_new), label='Original Spline', color='red')

    # Initial sine wave plot in 3D
    sine_wave_line, = ax.plot(x_sine_wave, y_sine_wave, z_wave,
                              label='3D Spline with Sine Waves', color='green')

    max_dist = max(x_new.max(), y_new.max())
    ax.set_xlim(0, max_dist)
    ax.set_ylim(0, max_dist)
    ax.set_zlim(0, max_dist)

    # plotting animation wali cheez
    line, = ax.plot([], [], [], 'o-', lw=2)

    def update(frame):
        """Update function for animation"""
        data = np.array(answers[frame])  # Get points for the current frame
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        # Update line data
        line.set_data(x, y)
        line.set_3d_properties(z)
        return line,

    ani = FuncAnimation(fig, update, frames=len(answers), interval=50, blit=False)

    plt.legend()
    plt.grid(True)
    print("  [graph window open – close it to launch the MuJoCo viewer]")
    plt.show()


def angles_to_ctrl(angles_real, num_modules, R, approximation_circle_dist):
    """
    Convert the angles_real frames from classical.py into MuJoCo ctrl arrays.

    Each frame in angles_real has (num_modules−1) tuples (horiz_angle, vert_angle)
    in the 180+Δ convention.

    Returns list of np.ndarray, each of length num_modules*2 (bottom, top interleaved).
    """
    CTRL_LIMIT = 0.5236  # ±30° in radians  (matches ctrlrange in chain xml)
    ctrl_frames = []
    for angles_set in angles_real:
        ctrl = np.zeros(num_modules * 2)
        for idx, angle_pair in enumerate(angles_set):
            # Replicate the servo formula from classical.py
            # angle_pair[0] = horizontal (y-axis in classical)
            # angle_pair[1] = vertical   (z-axis in classical)
            # MuJoCo axes are flipped: servo_bottom → z, servo_top → y
            # so we swap: theta_h ↔ angle_pair[1], theta_v ↔ angle_pair[0]
            theta_h = (180 - angle_pair[1]) * np.pi / 180   # z classical → bottom servo
            theta_v = (180 - angle_pair[0]) * np.pi / 180   # y classical → top servo
            angle_1 = atan(R * sin(theta_h) / (R * cos(theta_h) - approximation_circle_dist)) * 180 / np.pi
            angle_2 = atan(R * sin(theta_v) / (R * cos(theta_v) - approximation_circle_dist)) * 180 / np.pi

            ctrl_bottom = np.clip(-angle_1 * np.pi / 180, -CTRL_LIMIT, CTRL_LIMIT)
            ctrl_top    = np.clip(-angle_2 * np.pi / 180, -CTRL_LIMIT, CTRL_LIMIT)

            # There are (num_modules-1) joint pairs for the inter-module joints.
            # Pair idx drives the joint between module[idx] and module[idx+1].
            # We assign the angles to module idx (its top servo bends toward idx+1)
            # AND to module idx+1 (its bottom servo bends toward idx).
            # This ensures every module, including module 0, receives a non-zero command.
            mod = idx  # "source" module of this joint (0-indexed)
            ctrl[2 * mod]     = ctrl_bottom   # servo_bottom of module[idx]
            ctrl[2 * mod + 1] = ctrl_top       # servo_top    of module[idx]
            # Also propagate to the next module so both sides of the joint move
            next_mod = idx + 1
            if next_mod < num_modules:
                ctrl[2 * next_mod]     = ctrl_bottom
                ctrl[2 * next_mod + 1] = ctrl_top

        ctrl_frames.append(ctrl)
    return ctrl_frames


def build_scene(num: int) -> str:
    """Write a temporary scene XML that includes chain_{num}.xml and return its path."""
    scene_path = os.path.join("snake_description", "chain_scene.xml")
    chain_path = os.path.join("snake_description", f"chain_{num}.xml")

    if not os.path.exists(chain_path):
        from generate_chain import generate_chain
        generate_chain(num, chain_path)

    scene_xml = f"""<mujoco model="chain_scene">
    <include file="chain_{num}.xml" />
    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
        <rgba haze="0.15 0.25 0.35 1" />
        <global azimuth="160" elevation="-20" />
    </visual>
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072" />
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2" />
    </asset>
    <worldbody>
        <light pos="0 0 3.5" dir="0 0 -1" directional="true" />
        <geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="groundplane" contype="1" conaffinity="2" />
    </worldbody>
</mujoco>"""

    with open(scene_path, "w") as f:
        f.write(scene_xml)

    return scene_path


def run_simulation(model, data, ctrl_frames, step_interval: float, loop: bool):
    """
    Drive the MuJoCo model through the computed ctrl_frames sequence.

    Parameters
    ----------
    model, data       : MuJoCo model and data objects
    ctrl_frames       : list of np.ndarray, one per time-step of the gait
    step_interval     : real-time pause between gait frames (seconds)
    loop              : if True, restart the gait sequence after reaching the end
    """
    frame_idx = [0]
    paused    = [False]

    print(f"\n{'─'*60}")
    print(f"  Running {len(ctrl_frames)} gait frames, loop={loop}")
    print(f"  Actuators: {model.nu}   Modules: {model.nu // 2}")
    print(f"  Frame interval: {step_interval*1000:.1f} ms")
    print(f"  Press ESC or close the window to quit.")
    print(f"{'─'*60}\n")

    with viewer.launch_passive(model, data) as gui:
        while gui.is_running():
            if not paused[0]:
                ctrl = ctrl_frames[frame_idx[0]]
                data.ctrl[:len(ctrl)] = ctrl
                # Advance physics several steps per gait frame for stability
                physics_substeps = max(1, int(step_interval / model.opt.timestep))
                for _ in range(physics_substeps):
                    mujoco.mj_step(model, data)
                gui.sync()
                frame_idx[0] += 1
                if frame_idx[0] >= len(ctrl_frames):
                    if loop:
                        frame_idx[0] = 0
                        print("  [gait loop restart]")
                    else:
                        print("  [gait sequence complete – holding last pose]")
                        paused[0] = True

            else:
                # Keep physics alive but hold pose
                mujoco.mj_step(model, data)
                gui.sync()

            time.sleep(model.opt.timestep)


def main():
    parser = argparse.ArgumentParser(
        description="Simulate the classical serpentine gait on the MuJoCo snakebot chain."
    )
    parser.add_argument(
        "--num", type=int, default=5,
        help="Number of snake modules (default: 5, must match chain_<num>.xml)."
    )
    parser.add_argument(
        "--interval", type=float, default=0.055,
        help="Time (seconds) between successive gait frames (mirrors classical.py's 0.055 s sleep)."
    )
    parser.add_argument(
        "--loop", action="store_true", default=True,
        help="Loop the gait sequence continuously (default: True)."
    )
    parser.add_argument(
        "--no-loop", dest="loop", action="store_false",
        help="Play the gait sequence once and hold the last pose."
    )
    args = parser.parse_args()

    # Must run from repo root so relative paths to snake_description/* work
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    print("Computing gait angles (classical method) …")
    (angles_real, answers,
     x_new, y_new, x_sine_wave, y_sine_wave, z_wave,
     R, approx_dist) = compute_angles(num_segments=args.num)
    print(f"  Generated {len(angles_real)} gait frames, "
          f"{len(angles_real[0]) if angles_real else 0} joint pairs each.")

    show_gait_graph(answers, x_new, y_new, x_sine_wave, y_sine_wave, z_wave)

    ctrl_frames = angles_to_ctrl(angles_real, num_modules=args.num, R=R,
                                  approximation_circle_dist=approx_dist)

    print(f"Building MuJoCo scene for chain_{args.num} …")
    scene_path = build_scene(args.num)

    model = mujoco.MjModel.from_xml_path(scene_path)
    data  = mujoco.MjData(model)

    print(f"  Joints: {model.njnt}  Actuators: {model.nu}")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"    [{i}] {name}")

    run_simulation(model, data, ctrl_frames,
                   step_interval=args.interval, loop=args.loop)


if __name__ == "__main__":
    main()

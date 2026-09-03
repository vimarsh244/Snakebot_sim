# -*- coding: utf-8 -*-
"""
serpentine_sim.py  –  Run the phase-shifting serpentine gait on the MuJoCo
                      snakebot simulation (no real hardware needed).

New estimation method: sacrifices some accuracy for simpler and faster code.
NOTE: angles are now in (x, y) order – i.e. (horizontal, vertical) –
      unlike the older scripts which used (y, x) / (vertical, horizontal).

Usage (from repo root):
    python scripts/serpentine_sim.py                 # 5-module default
    python scripts/serpentine_sim.py --num 6         # 6-module chain
    python scripts/serpentine_sim.py --num 6 \\
        --req-length 1.14 --amp-z 0.4 --amp-y 0.4 \\
        --frequency 3 --num-frames 200

Controls while the viewer is open:
    ESC / close window  →  quit
    SPACE               →  pause / resume stepping (built-in MuJoCo viewer)

Gait generation (phase-shifting method)
────────────────────────────────────────
Instead of sliding a window along the spline, the body of the snake stays
centred on a fixed spline and the sine-wave PHASE is incremented each frame.
The multiplier  1.5·u  causes the amplitude to grow along the body length,
giving a more natural travelling-wave appearance.

Angle convention
────────────────
angles_real[frame][joint] = (horiz_angle, vert_angle) in degrees,
in the 180+Δ convention (neutral joint = 180°).

MuJoCo ctrl mapping
────────────────────
angle_pair[0] = horizontal → ctrl_bottom (servo bends around z-axis)
angle_pair[1] = vertical   → ctrl_top    (servo bends around y-axis)

    θ_h = (180 − angle_pair[0]) × π/180
    θ_v = (180 − angle_pair[1]) × π/180

    angle_1 = atan(R·sin(θ_h) / (R·cos(θ_h) − d)) × 180/π
    angle_2 = atan(R·sin(θ_v) / (R·cos(θ_v) − d)) × 180/π

    ctrl_bottom = clamp(−angle_1 × π/180, −0.5236, 0.5236)
    ctrl_top    = clamp(−angle_2 × π/180, −0.5236, 0.5236)
"""

import os
import time
import argparse
import numpy as np
from copy import deepcopy
from math import atan, sin, cos, sqrt

import mujoco
from mujoco import viewer

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import splprep, splev


#
# Gait computation (phase-shifting method)
#

def compute_angles(
    num_segments: int = 5,
    req_length: float = 1.14,
    req_amplitude_z: float = 0.4,
    req_amplitude_y: float = 0.4,
    frequency: float = 3.0,
    num_frames: int = 200,
    path_points_x: list = None,
    path_points_y: list = None,
):
    """
    Generate gait frames using the phase-shifting serpentine method.

    The sine-wave phase is incremented by  5 × (2π / num_frames)  each frame
    so that exactly num_frames frames complete 5 full wave cycles.

    Returns
    -------
    angles_real   : list of lists of (horiz, vert) tuples per frame
                    Convention: (x, y) = (horizontal, vertical) [degrees, 180+Δ]
    answers       : list of lists of [x, y, z] backbone keypoints per frame
    x_new, y_new              : base spline arrays
    x_sine_wave_init,
    y_sine_wave_init, z_init  : initial 3-D sine-wave path arrays (for plotting)
    R, approximation_circle_dist : geometry constants
    """
    if path_points_x is None:
        path_points_x = [0, 3, 4, 6]
    if path_points_y is None:
        path_points_y = [0, 3, 4, 6]

    R                        = req_length / 2
    approximation_circle_dist = 0.17
    accuracylevel             = 100_000

    # ── Base spline ──────────────────────────────────────────────────────────
    points  = np.array([path_points_x, path_points_y])
    tck, _  = splprep(points, s=0)
    u_new   = np.linspace(0, 1, accuracylevel)
    x_new, y_new = splev(u_new, tck)

    dx, dy   = splev(u_new, tck, der=1)
    magnitude     = np.sqrt(dx**2 + dy**2)
    dx_normalized = -dy / magnitude
    dy_normalized =  dx / magnitude

    multiplier = 1.5 * u_new   # growing amplitude along body

    # ── Initial sine wave (phase = 0) for display ─────────────────────────
    phase = 0.0
    x_sine_wave_init = (x_new
                        + multiplier * req_amplitude_y
                        * np.sin(2 * np.pi * frequency * u_new + 1 + phase)
                        * dx_normalized)
    y_sine_wave_init = (y_new
                        + multiplier * req_amplitude_y
                        * np.sin(2 * np.pi * frequency * u_new + 1 + phase)
                        * dy_normalized)
    z_init = req_amplitude_z * np.abs(np.sin(2 * np.pi * frequency * u_new + 1 + phase))

    # ── Phase-shifting loop ───────────────────────────────────────────────
    angles_real   = []
    answers       = []

    phase_step = 5.0 * (2 * np.pi / num_frames)

    for frame_idx in range(num_frames):
        # Recompute wave for current phase
        x_sw = (x_new
                + multiplier * req_amplitude_y
                * np.sin(2 * np.pi * frequency * u_new + 1 + phase)
                * dx_normalized)
        y_sw = (y_new
                + multiplier * req_amplitude_y
                * np.sin(2 * np.pi * frequency * u_new + 1 + phase)
                * dy_normalized)
        z_sw = req_amplitude_z * np.abs(np.sin(2 * np.pi * frequency * u_new + 1 + phase))

        # Walk along the current wave to find evenly-spaced keypoints
        answers_per_iter          = [[x_sw[0], y_sw[0], z_sw[0]]]
        angles_real_per_iter      = []
        angles_ground_apparent    = []
        angles_ground_real        = []

        for i in range(len(x_sw)):
            prev = answers_per_iter[-1]
            dist = sqrt((x_sw[i] - prev[0])**2
                        + (y_sw[i] - prev[1])**2
                        + (z_sw[i] - prev[2])**2)
            if dist > req_length:
                answers_per_iter.append([x_sw[i], y_sw[i], z_sw[i]])

            if len(answers_per_iter) == num_segments + 1:
                for j in range(len(answers_per_iter) - 1):
                    p0 = answers_per_iter[j]
                    p1 = answers_per_iter[j + 1]
                    dx01, dy01, dz01 = (p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2])

                    # Angles are now (x, y) = (horizontal, vertical)
                    angles_ground_apparent.append([
                        atan(dy01 / dx01) * 180 / np.pi,
                        atan(dz01 / dx01) * 180 / np.pi,
                    ])
                    angles_ground_real.append([
                        atan(dy01 / sqrt(dx01**2 + dz01**2)) * 180 / np.pi,
                        atan(dz01 / sqrt(dx01**2 + dy01**2)) * 180 / np.pi,
                    ])
                    if j:
                        angles_real_per_iter.append((
                            round(180 + angles_ground_real[j][0]
                                  - angles_ground_real[j - 1][0], 3),
                            round(180 + angles_ground_real[j][1]
                                  - angles_ground_real[j - 1][1], 3),
                        ))
                break

        if len(answers_per_iter) < num_segments + 1:
            print(f"  [frame {frame_idx}] not enough points on wave – skipping")
            phase += phase_step
            continue

        answers.append(deepcopy(answers_per_iter))
        angles_real.append(deepcopy(angles_real_per_iter))
        phase += phase_step

    return (angles_real, answers,
            x_new, y_new,
            x_sine_wave_init, y_sine_wave_init, z_init,
            R, approximation_circle_dist)


#
# 3-D gait graph
#

def show_gait_graph(answers, x_new, y_new, x_sine_wave, y_sine_wave, z_wave,
                    path_points_x, path_points_y):
    """Display the backbone animation – blocks until the window is closed."""
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.1, bottom=0.25)
    ax.set_box_aspect([1, 1, 1])

    ax.plot(x_sine_wave, y_sine_wave, z_wave,
            label='3-D sine wave (initial frame)', color='green')

    max_dist = max(max(path_points_x), max(path_points_y))
    ax.set_xlim(0, max_dist)
    ax.set_ylim(0, max_dist)
    ax.set_zlim(0, max_dist)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    line, = ax.plot([], [], [], 'o-', lw=2)

    def update(frame):
        data    = np.array(answers[frame])
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        line.set_data(x, y)
        line.set_3d_properties(z)
        return line,

    ani = FuncAnimation(fig, update, frames=len(answers), interval=50, blit=False)  # must be kept alive

    plt.legend()
    plt.grid(True)
    print("  [graph window open – close it to launch the MuJoCo viewer]")
    plt.show()


#
# Angle → MuJoCo ctrl conversion
#

def angles_to_ctrl(angles_real, num_modules, R, approximation_circle_dist):
    """
    Convert phase-shifted angles_real frames into MuJoCo ctrl arrays.

    Angle convention: (x, y) = (horizontal, vertical), 180+Δ degrees.

    Mapping (matches sim_classical.py and confirmed by chain_N.xml joint axes):
      angle_pair[1] (vertical)   → theta_h → bottom servo (Revolute-15, axis ≈ −X in local frame)
      angle_pair[0] (horizontal) → theta_v → top    servo (Revolute-16, axis ≈ −X in local frame)
    The two joints appear identical in isolation but their parent bodies are rotated
    relative to each other in the VRJ design, making them effectively orthogonal.

    Returns list of np.ndarray, each of length num_modules * 2.
    """
    CTRL_LIMIT = 0.5236 * 25/30  # ±30° in radians (matches ctrlrange in chain xml)
    ctrl_frames = []

    for angles_set in angles_real:
        ctrl = np.zeros(num_modules * 2)

        for idx, angle_pair in enumerate(angles_set):
            # (x, y) = (horizontal, vertical)
            # Mirror of sim_classical.py axis mapping (confirmed by XML joint axes):
            #   vertical   (angle_pair[1]) → theta_h → bottom servo (Revolute-15)
            #   horizontal (angle_pair[0]) → theta_v → top    servo (Revolute-16)
            theta_h = (180 - angle_pair[1]) * np.pi / 180   # vertical   → bottom servo
            theta_v = (180 - angle_pair[0]) * np.pi / 180   # horizontal → top    servo

            angle_1 = atan(R * sin(theta_h) / (R * cos(theta_h) - approximation_circle_dist)) * 180 / np.pi
            angle_2 = atan(R * sin(theta_v) / (R * cos(theta_v) - approximation_circle_dist)) * 180 / np.pi

            ctrl_bottom = np.clip(-angle_1 * np.pi / 180, -CTRL_LIMIT, CTRL_LIMIT)
            ctrl_top    = np.clip(-angle_2 * np.pi / 180, -CTRL_LIMIT, CTRL_LIMIT)

            # Write to source module
            mod = idx
            ctrl[2 * mod]     = ctrl_bottom
            ctrl[2 * mod + 1] = ctrl_top

            # Propagate to next module so both sides of joint move
            next_mod = idx + 1
            if next_mod < num_modules:
                ctrl[2 * next_mod]     = ctrl_bottom
                ctrl[2 * next_mod + 1] = ctrl_top

        ctrl_frames.append(ctrl)

    return ctrl_frames


#
# Scene builder
#

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


#
# MuJoCo simulation loop
#

def run_simulation(model, data, ctrl_frames, step_interval: float, loop: bool):
    """
    Drive the MuJoCo model through the computed ctrl_frames sequence.

    Parameters
    ----------
    model, data    : MuJoCo model and data objects
    ctrl_frames    : list of np.ndarray, one per gait frame
    step_interval  : real-time pause between gait frames (seconds)
    loop           : if True, restart the gait sequence after the last frame
    """
    frame_idx = [0]
    paused    = [False]

    print(f"\n{'─'*60}")
    print(f"  Running {len(ctrl_frames)} gait frames, loop={loop}")
    print(f"  Actuators : {model.nu}   Modules: {model.nu // 2}")
    print(f"  Frame interval : {step_interval * 1000:.1f} ms")
    print(f"  Press ESC or close the window to quit.")
    print(f"{'─'*60}\n")

    with viewer.launch_passive(model, data) as gui:
        while gui.is_running():
            if not paused[0]:
                ctrl = ctrl_frames[frame_idx[0]]
                data.ctrl[:len(ctrl)] = ctrl

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
                mujoco.mj_step(model, data)
                gui.sync()

            time.sleep(model.opt.timestep)


# Entry point

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Simulate the phase-shifting serpentine gait on the MuJoCo snakebot.\n"
            "Angles are in (x, y) = (horizontal, vertical) order."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num",        type=int,   default=5,    help="Number of snake modules.")
    parser.add_argument("--req-length", type=float, default=1.14, help="Required segment length (m).")
    parser.add_argument("--amp-z",      type=float, default=0.4,  help="Vertical sine amplitude.")
    parser.add_argument("--amp-y",      type=float, default=0.4,  help="Horizontal sine amplitude.")
    parser.add_argument("--frequency",  type=float, default=3.0,  help="Sine-wave spatial frequency.")
    parser.add_argument("--num-frames", type=int,   default=200,  help="Number of gait frames to generate.")
    parser.add_argument("--interval",   type=float, default=0.03, help="Seconds between gait frames.")
    parser.add_argument("--loop",       action="store_true",  default=True,
                        help="Loop the gait sequence continuously.")
    parser.add_argument("--no-loop",    dest="loop", action="store_false",
                        help="Play the gait sequence once and hold the last pose.")
    parser.add_argument("--no-graph",   action="store_true", default=False,
                        help="Skip the matplotlib gait preview and go straight to MuJoCo.")
    args = parser.parse_args()

    # Must run from repo root so relative paths to snake_description/* work
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    print("Computing gait angles (phase-shifting method) …")
    print(f"  Modules={args.num}  req_length={args.req_length}  "
          f"amp_z={args.amp_z}  amp_y={args.amp_y}  "
          f"frequency={args.frequency}  frames={args.num_frames}")

    (angles_real, answers,
     x_new, y_new,
     x_sw_init, y_sw_init, z_init,
     R, approx_dist) = compute_angles(
        num_segments   = args.num,
        req_length     = args.req_length,
        req_amplitude_z= args.amp_z,
        req_amplitude_y= args.amp_y,
        frequency      = args.frequency,
        num_frames     = args.num_frames,
    )

    print(f"  Generated {len(angles_real)} gait frames, "
          f"{len(angles_real[0]) if angles_real else 0} joint pairs each.")

    if not angles_real:
        print("ERROR: No gait frames were generated.  "
              "Try reducing --req-length or increasing --num-frames.")
        return

    # Print a sample of the angles
    print("  Sample angles (first 3 frames, (horizontal, vertical) in degrees):")
    for fi, frame in enumerate(angles_real[:3]):
        print(f"    frame {fi}: {frame}")

    if not args.no_graph:
        path_points_x = [0, 3, 4, 6]
        path_points_y = [0, 3, 4, 6]
        show_gait_graph(answers, x_new, y_new,
                        x_sw_init, y_sw_init, z_init,
                        path_points_x, path_points_y)

    ctrl_frames = angles_to_ctrl(angles_real, num_modules=args.num,
                                  R=R, approximation_circle_dist=approx_dist)

    print(f"\nBuilding MuJoCo scene for chain_{args.num} …")
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

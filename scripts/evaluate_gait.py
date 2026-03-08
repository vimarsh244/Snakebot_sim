#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_gait.py
================
Headless gait evaluator for the MuJoCo snakebot.

Given gait parameters (Az, Ay, frequency, …), runs the phase-shifting
serpentine gait for a specified wall-time duration, tracks the snake's
centre-of-mass trajectory, and reports:

    • V_forward  - mean speed along the snake's primary travel axis [m/s]
    • V_lateral  - mean lateral drift speed perpendicular to travel [m/s]
    • heading    - travel direction in the XY-plane [degrees, world frame]
    • net displacement (forward & lateral)

Orientation note
----------------
The chain XML places the snake horizontally.  The frame euler "0 π/2 0"
rotates local-Z into world-X, so the initial chain axis is approximately
aligned with the world -X direction.  After letting the snake settle, all
velocity estimates are in the world XY-plane.

Usage examples
--------------
    # 5-module default, headless, 30 s
    python scripts/evaluate_gait.py

    # 6-module, custom gait, with viewer + trajectory plot
    python scripts/evaluate_gait.py --num 6 --amp-z 0.35 --amp-y 0.45 \\
        --frequency 2.5 --duration 30 --viewer --plot

    # With live matplotlib plot updating during simulation
    python scripts/evaluate_gait.py --live-plot

    # Save full head-module trajectory (t,x,y,z,qw,qx,qy,qz,vx,vy,vz)
    python scripts/evaluate_gait.py --traj-csv head_traj.csv

    # Parameter sweep helper (headless, CSV output)
    python scripts/evaluate_gait.py --num 5 --amp-z 0.3 --amp-y 0.3 \\
        --frequency 3 --duration 20 --csv results.csv

    # Record top-view + side-view MP4 videos (offscreen, no GUI needed)
    python scripts/evaluate_gait.py --record
    python scripts/evaluate_gait.py --record --video-fps 60 \\
        --video-width 1280 --video-height 720 --video-prefix my_run
"""

import os
import sys
import time
import argparse
import csv
import numpy as np
from copy import deepcopy
from math import atan, sin, cos, sqrt, atan2, degrees

# Force UTF-8 output on Windows (avoids cp1252 encode errors for box-drawing chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import mujoco
from scipy.interpolate import splprep, splev


# -----------------------------------------------------------------------------
# Re-use gait logic from serpentine_sim (same repo)
# -----------------------------------------------------------------------------

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
    """Phase-shifting serpentine gait - see serpentine_sim.py for full docs."""
    if path_points_x is None:
        path_points_x = [0, 3, 4, 6]
    if path_points_y is None:
        path_points_y = [0, 3, 4, 6]

    R                        = req_length / 2
    approximation_circle_dist = 0.17
    accuracylevel             = 100_000

    points  = np.array([path_points_x, path_points_y])
    tck, _  = splprep(points, s=0)
    u_new   = np.linspace(0, 1, accuracylevel)
    x_new, y_new = splev(u_new, tck)

    dx, dy   = splev(u_new, tck, der=1)
    magnitude     = np.sqrt(dx**2 + dy**2)
    dx_normalized = -dy / magnitude
    dy_normalized =  dx / magnitude

    multiplier  = 1.5 * u_new
    phase_step  = 5.0 * (2 * np.pi / num_frames)

    angles_real = []
    answers     = []
    phase       = 0.0

    for frame_idx in range(num_frames):
        x_sw = (x_new
                + multiplier * req_amplitude_y
                * np.sin(2 * np.pi * frequency * u_new + 1 + phase)
                * dx_normalized)
        y_sw = (y_new
                + multiplier * req_amplitude_y
                * np.sin(2 * np.pi * frequency * u_new + 1 + phase)
                * dy_normalized)
        z_sw = req_amplitude_z * np.abs(
                    np.sin(2 * np.pi * frequency * u_new + 1 + phase))

        answers_per_iter       = [[x_sw[0], y_sw[0], z_sw[0]]]
        angles_real_per_iter   = []
        angles_ground_real     = []

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
                    dx01 = p1[0] - p0[0]
                    dy01 = p1[1] - p0[1]
                    dz01 = p1[2] - p0[2]
                    angles_ground_real.append([
                        atan(dy01 / sqrt(dx01**2 + dz01**2)) * 180 / np.pi,
                        atan(dz01 / sqrt(dx01**2 + dy01**2)) * 180 / np.pi,
                    ])
                    if j:
                        angles_real_per_iter.append((
                            round(180 + angles_ground_real[j][0]
                                  - angles_ground_real[j-1][0], 3),
                            round(180 + angles_ground_real[j][1]
                                  - angles_ground_real[j-1][1], 3),
                        ))
                break

        if len(answers_per_iter) < num_segments + 1:
            phase += phase_step
            continue

        answers.append(deepcopy(answers_per_iter))
        angles_real.append(deepcopy(angles_real_per_iter))
        phase += phase_step

    return angles_real, answers, R, approximation_circle_dist


def angles_to_ctrl(angles_real, num_modules, R, approximation_circle_dist):
    """
    Convert phase-shifted angles_real frames → MuJoCo ctrl arrays.
    Matches the convention in serpentine_sim.py.
    """
    CTRL_LIMIT  = 0.5236 * 25 / 30
    ctrl_frames = []

    for angles_set in angles_real:
        ctrl = np.zeros(num_modules * 2)

        for idx, angle_pair in enumerate(angles_set):
            theta_h = (180 - angle_pair[1]) * np.pi / 180
            theta_v = (180 - angle_pair[0]) * np.pi / 180

            angle_1 = atan(R * sin(theta_h) / (R * cos(theta_h) - approximation_circle_dist)) * 180 / np.pi
            angle_2 = atan(R * sin(theta_v) / (R * cos(theta_v) - approximation_circle_dist)) * 180 / np.pi

            ctrl_bottom = np.clip(-angle_1 * np.pi / 180, -CTRL_LIMIT, CTRL_LIMIT)
            ctrl_top    = np.clip(-angle_2 * np.pi / 180, -CTRL_LIMIT, CTRL_LIMIT)

            mod = idx
            ctrl[2 * mod]     = ctrl_bottom
            ctrl[2 * mod + 1] = ctrl_top

            next_mod = idx + 1
            if next_mod < num_modules:
                ctrl[2 * next_mod]     = ctrl_bottom
                ctrl[2 * next_mod + 1] = ctrl_top

        ctrl_frames.append(ctrl)

    return ctrl_frames


# -----------------------------------------------------------------------------
# Scene helpers
# -----------------------------------------------------------------------------

def _patch_chain_xml(src: str, dest: str,
                     body_slide: float, floor_slide: float) -> None:
    """
    Write a copy of chain XML with overridden friction values.
      body_slide  – slide friction for all body mesh geoms (<default><geom …/>)
      floor_slide – slide friction for the chain's embedded ground plane geom
    """
    import re
    with open(src) as f:
        xml = f.read()
    # Default body geom (contype=2 means body mesh geoms)
    xml = re.sub(
        r'(contype="2"[^/]*friction=")[^"]*(")',
        rf'\g<1>{body_slide} 1.05 0.001\g<2>',
        xml,
    )
    # Chain embedded ground plane (name="ground")
    xml = re.sub(
        r'(name="ground"[^/]*friction=")[^"]*(")',
        rf'\g<1>{floor_slide} 0.005 0.0001\g<2>',
        xml,
    )
    with open(dest, "w") as f:
        f.write(xml)


def build_scene(num: int,
                floor_friction: float = 0.3,
                body_friction:  float = 0.3,
                offwidth:       int   = 0,
                offheight:      int   = 0) -> str:
    """Write a temporary scene XML that includes chain_{num}.xml.

    If body_friction differs from the default (0.3) a patched copy of the
    chain XML is written alongside as chain_{num}_patched.xml so the original
    is never modified.

    offwidth / offheight: when non-zero, sets the MuJoCo offscreen framebuffer
    via <visual><global offwidth=... offheight=.../></visual>.  Required when
    using mujoco.Renderer at resolutions above the 640 x 480 default.
    """
    scene_path = os.path.join("snake_description", "chain_scene.xml")
    chain_path = os.path.join("snake_description", f"chain_{num}.xml")

    if not os.path.exists(chain_path):
        sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
        from generate_chain import generate_chain
        generate_chain(num, chain_path)

    if abs(body_friction - 0.3) > 1e-9:
        patched = os.path.join("snake_description", f"chain_{num}_patched.xml")
        _patch_chain_xml(chain_path, patched,
                         body_slide=body_friction,
                         floor_slide=floor_friction)
        include_name = f"chain_{num}_patched.xml"
    else:
        include_name = f"chain_{num}.xml"

    # Build offscreen size line only when needed
    _off = (f'\n        <global offwidth="{offwidth}" offheight="{offheight}" />'
            if offwidth > 0 and offheight > 0 else "")

    scene_xml = f"""<mujoco model="chain_scene">
    <include file="{include_name}" />
    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
        <rgba haze="0.15 0.25 0.35 1" />{_off}
    </visual>
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7"
                 rgb2="0 0 0" width="512" height="3072" />
        <texture type="2d" name="groundplane" builtin="checker" mark="edge"
                 rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
                 width="300" height="300" />
        <material name="groundplane" texture="groundplane" texuniform="true"
                  texrepeat="5 5" reflectance="0.2" />
    </asset>
    <worldbody>
        <light pos="0 0 3.5" dir="0 0 -1" directional="true" />
        <geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane"
              material="groundplane" contype="1" conaffinity="2"
              friction="{floor_friction} 0.005 0.0001" />
    </worldbody>
</mujoco>"""

    with open(scene_path, "w") as f:
        f.write(scene_xml)
    return scene_path


# -----------------------------------------------------------------------------
# COM & head-state trackers
# -----------------------------------------------------------------------------

def get_snake_com(model, data, num_modules):
    """
    Return the mean XYZ position of the num_modules bottom-base-plate bodies.
    These carry the majority of the mass and are the canonical module origins.
    """
    positions = []
    for n in range(1, num_modules + 1):
        body_name = f"m{n}_bottom-base-plate-v1"
        try:
            bid = model.body(body_name).id
            positions.append(data.xpos[bid].copy())
        except KeyError:
            pass
    if positions:
        return np.mean(positions, axis=0)
    # Fallback: use MuJoCo's world subtree COM
    return data.subtree_com[0].copy()


def get_head_state(model, data):
    """
    Return the full state of the head module (m1_bottom-base-plate-v1).

    Returns a 1-D array of 11 values:
        [t, x, y, z, qw, qx, qy, qz, vx, vy, vz]

    Position and quaternion are in the world frame.
    Linear velocity is the world-frame velocity of the body's COM
    from MuJoCo's cvel (indices [3:6], world-aligned axes).
    """
    bid = model.body("m1_bottom-base-plate-v1").id
    pos  = data.xpos[bid]               # (3,)  world frame
    quat = data.xquat[bid]              # (4,)  [w, x, y, z] world frame
    vel  = data.cvel[bid, 3:6]          # (3,)  linear vel, world-aligned, at COM
    return np.array([
        data.time,
        pos[0],  pos[1],  pos[2],
        quat[0], quat[1], quat[2], quat[3],
        vel[0],  vel[1],  vel[2],
    ], dtype=float)


# -----------------------------------------------------------------------------
# Offscreen video recording helpers
# -----------------------------------------------------------------------------

class _VideoWriter:
    """
    Thin wrapper around an imageio FFMPEG writer.

    Parameters
    ----------
    path   : output MP4 file path
    fps    : video frame rate
    width  : frame width  in pixels (must be even for yuv420p)
    height : frame height in pixels (must be even for yuv420p)
    """
    def __init__(self, path: str, fps: int, width: int, height: int):
        import imageio
        # yuv420p requires even dimensions
        w = width  + (width  % 2)
        h = height + (height % 2)
        self._writer = imageio.get_writer(
            path, fps=fps, codec="libx264",
            quality=8, pixelformat="yuv420p",
            output_params=["-preset", "fast"],
        )
        self.path  = path
        self.count = 0
        self.w     = w
        self.h     = h

    def write(self, frame_rgb: np.ndarray):
        """Accept an (H, W, 3) uint8 RGB array and append it."""
        self._writer.append_data(frame_rgb)
        self.count += 1

    def close(self):
        self._writer.close()
        print(f"  Video saved -> {self.path}  ({self.count} frames)")


def _make_tracking_cam(elevation: float, azimuth: float,
                       distance: float, zoom_factor: float = 1.0) -> "mujoco.MjvCamera":
    """
    Return a free MjvCamera with given elevation / azimuth / distance.
    Update `cam.lookat[:]` each frame to track the snake's COM.

    zoom_factor: multiplier for distance (< 1.0 zooms in, > 1.0 zooms out)
    """
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.3]
    cam.elevation = elevation
    cam.azimuth   = azimuth
    cam.distance  = distance * zoom_factor
    return cam


# -----------------------------------------------------------------------------
# Main simulation + evaluation
# -----------------------------------------------------------------------------

def run_evaluation(
    num_modules:    int   = 5,
    req_length:     float = 1.14,
    amp_z:          float = 0.4,
    amp_y:          float = 0.4,
    frequency:      float = 3.0,
    num_frames:     int   = 200,
    duration:       float = 30.0,
    gait_interval:  float = 0.03,
    settle_time:    float = 2.0,
    use_viewer:     bool  = False,
    show_plot:      bool  = False,
    live_plot:      bool  = False,
    traj_csv:       str   = None,
    verbose:        bool  = True,
    floor_friction: float = 0.3,
    body_friction:  float = 0.3,
    record_video:   bool  = False,
    video_fps:      int   = 30,
    video_width:    int   = 960,
    video_height:   int   = 540,
    video_prefix:   str   = "snake",
    zoom_factor:    float = 1.0,
    show_contacts:  bool  = False,
) -> dict:
    """
    Run the serpentine gait for `duration` seconds and return a results dict.

    Returns
    -------
    dict with keys:
        v_forward    - mean forward speed [m/s] (along primary travel axis)
        v_lateral    - mean lateral drift speed [m/s]
        heading_deg  - travel heading in world XY plane [degrees from +X axis]
        disp_forward - net forward displacement [m]
        disp_lateral - net lateral displacement [m]
        com_x_arr    - ndarray of COM X positions vs time
        com_y_arr    - ndarray of COM Y positions vs time
        time_arr     - ndarray of simulation timestamps
        head_traj    - ndarray shape (N,11): t,x,y,z,qw,qx,qy,qz,vx,vy,vz
        params       - dict of input parameters

    Video recording (when record_video=True)
    ----------------------------------------
    Two MP4 files are written:
        <video_prefix>_top.mp4   - overhead (bird's-eye) camera, tracks COM
        <video_prefix>_side.mp4  - side camera (elevation -12 deg), tracks COM

    Parameters
    ----------
    zoom_factor: camera distance multiplier (0.5 = 50% zoom in, 2.0 = 100% zoom out)
    show_contacts: enable contact force/point visualization in the viewer (when --viewer flag used);
                   note: not supported in offline video rendering currently
    Requires: imageio + imageio-ffmpeg  (pip install imageio imageio-ffmpeg)
    """
    # -- Repository root (so relative XML paths resolve) ------------------
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    if verbose:
        print("=" * 65)
        print("  Serpentine Gait Evaluator")
        print("=" * 65)
        print(f"  Modules   : {num_modules}")
        print(f"  Az        : {amp_z}   Ay : {amp_y}")
        print(f"  Frequency : {frequency}")
        print(f"  Duration  : {duration} s   (settle: {settle_time} s)")
        print(f"  req_length: {req_length}")
        print(f"  num_frames: {num_frames}")
        print(f"  floor_friction: {floor_friction}")
        print(f"  body_friction : {body_friction}")
        if record_video:
            print(f"  Video output  : {video_prefix}_top.mp4 / {video_prefix}_side.mp4")
            print(f"  Video spec    : {video_width}x{video_height} @ {video_fps} fps")
            print(f"  Zoom factor   : {zoom_factor}x")
            print(f"  Show contacts : {show_contacts}")
        print()

    # -- Gait computation -------------------------------------------------
    if verbose:
        print("Computing gait angles …")

    angles_real, answers, R, approx_dist = compute_angles(
        num_segments    = num_modules,
        req_length      = req_length,
        req_amplitude_z = amp_z,
        req_amplitude_y = amp_y,
        frequency       = frequency,
        num_frames      = num_frames,
    )

    if not angles_real:
        raise RuntimeError(
            "Gait computation produced no frames.  "
            "Try reducing --req-length or increasing --num-frames."
        )

    ctrl_frames = angles_to_ctrl(angles_real, num_modules, R, approx_dist)
    if verbose:
        print(f"  {len(ctrl_frames)} gait frames, "
              f"{len(angles_real[0])} joint pairs each.")

    # -- Build and load model ---------------------------------------------
    if verbose:
        print(f"Building MuJoCo scene for chain_{num_modules} …")

    scene_path = build_scene(num_modules,
                             floor_friction=floor_friction,
                             body_friction=body_friction,
                             offwidth=video_width  if record_video else 0,
                             offheight=video_height if record_video else 0)
    model = mujoco.MjModel.from_xml_path(scene_path)
    data  = mujoco.MjData(model)

    if verbose:
        print(f"  Joints: {model.njnt}   Actuators: {model.nu}")

    # -- Offscreen renderers & video writers (optional) -------------------
    if record_video:
        vw = video_width  + (video_width  % 2)   # ensure even (yuv420p)
        vh = video_height + (video_height % 2)
        _renderer_top  = mujoco.Renderer(model, height=vh, width=vw)
        _renderer_side = mujoco.Renderer(model, height=vh, width=vw)
        _cam_top  = _make_tracking_cam(elevation=-90, azimuth=0,   distance=2.5,
                                        zoom_factor=zoom_factor)
        _cam_side = _make_tracking_cam(elevation=-12, azimuth=90,  distance=2.5,
                                        zoom_factor=zoom_factor)
        _vid_top  = _VideoWriter(f"{video_prefix}_top.mp4",  video_fps, vw, vh)
        _vid_side = _VideoWriter(f"{video_prefix}_side.mp4", video_fps, vw, vh)
        if verbose:
            print(f"  Offscreen renderers: {vw}x{vh} px")
            if show_contacts:
                print(f"  (Note: contact viz enabled in viewer, not in video)")
            if show_contacts:
                print(f"  (Note: contact viz enabled in viewer, not in video)")
    else:
        _renderer_top = _renderer_side = None
        _cam_top = _cam_side = None
        _vid_top = _vid_side = None

    dt             = model.opt.timestep          # physics timestep (0.0005 s)
    steps_per_gait   = max(1, int(gait_interval / dt))  # physics steps per gait frame
    total_steps      = int(duration / dt)
    settle_steps     = int(settle_time / dt)
    record_every     = max(1, int(0.01 / dt))      # record COM ~100 Hz
    video_step_every = max(1, int(1.0 / (video_fps * dt))) if record_video else None

    if verbose:
        print(f"  Physics dt: {dt*1000:.2f} ms")
        print(f"  Steps per gait frame : {steps_per_gait}")
        print(f"  Total physics steps  : {total_steps}")
        print()

    # -- Simulation loop --------------------------------------------------
    com_x_list  = []
    com_y_list  = []
    time_list   = []
    head_rows   = []          # list of 11-element arrays for head trajectory
    frame_idx   = 0

    if use_viewer:
        from mujoco import viewer as mj_viewer
        gui_ctx = mj_viewer.launch_passive(model, data)
    else:
        gui_ctx = None

    # -- Live matplotlib plot setup ----------------------------------------
    if live_plot:
        import matplotlib
        import matplotlib.pyplot as plt
        plt.ion()
        fig_live, axes_live = plt.subplots(1, 2, figsize=(11, 4))
        fig_live.suptitle("Live: Serpentine Gait Evaluation", fontsize=11)

        ax_xy   = axes_live[0]
        ax_time = axes_live[1]

        ax_xy.set_xlabel("X [m]");   ax_xy.set_ylabel("Y [m]")
        ax_xy.set_title("COM trajectory");  ax_xy.grid(True, alpha=0.4)
        ax_time.set_xlabel("Sim time [s]"); ax_time.set_ylabel("Position [m]")
        ax_time.set_title("COM X / Y vs time"); ax_time.grid(True, alpha=0.4)

        (ln_xy,)   = ax_xy.plot([], [], lw=1.2, color="steelblue")
        (ln_head,) = ax_xy.plot([], [], "o", ms=6, color="tomato",   label="head")
        (ln_com,)  = ax_xy.plot([], [], "s", ms=5, color="limegreen", label="COM")
        ax_xy.legend(fontsize=7)

        (ln_x,) = ax_time.plot([], [], lw=1, color="steelblue", label="COM X")
        (ln_y,) = ax_time.plot([], [], lw=1, color="tomato",    label="COM Y")
        (ln_hx,) = ax_time.plot([], [], lw=0.8, color="steelblue", ls="--", label="head X")
        (ln_hy,) = ax_time.plot([], [], lw=0.8, color="tomato",    ls="--", label="head Y")
        ax_time.legend(fontsize=7)

        info_text = ax_xy.text(
            0.02, 0.98, "", transform=ax_xy.transAxes,
            fontsize=7.5, va="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )
        plt.tight_layout()
        plt.pause(0.01)

        live_update_every = max(1, int(0.5 / dt))   # update plot every ~0.5 s sim
    else:
        live_update_every = None

    sim_start = time.perf_counter()

    if verbose:
        print("Running simulation …")
        progress_interval = max(1, total_steps // 20)

    try:
        for step in range(total_steps):
            # -- Apply current gait frame ------------------------------
            if step % steps_per_gait == 0:
                ctrl = ctrl_frames[frame_idx % len(ctrl_frames)]
                data.ctrl[:len(ctrl)] = ctrl
                frame_idx += 1

            mujoco.mj_step(model, data)

            # -- Record COM + head state -------------------------------
            if step % record_every == 0:
                com = get_snake_com(model, data, num_modules)
                com_x_list.append(com[0])
                com_y_list.append(com[1])
                time_list.append(data.time)
                head_rows.append(get_head_state(model, data))

            # -- Viewer sync -------------------------------------------
            if gui_ctx is not None and step % steps_per_gait == 0:
                if not gui_ctx.is_running():
                    if verbose:
                        print("  [viewer closed - stopping early]")
                    break
                gui_ctx.sync()

            # -- Offscreen video capture --------------------------------
            if record_video and step % video_step_every == 0:
                com_now = get_snake_com(model, data, num_modules)
                lookat  = [com_now[0], com_now[1], 0.3]
                _cam_top.lookat[:]  = lookat
                _cam_side.lookat[:] = lookat
                _renderer_top.update_scene(data, camera=_cam_top)
                _renderer_side.update_scene(data, camera=_cam_side)
                _vid_top.write(_renderer_top.render())
                _vid_side.write(_renderer_side.render())

            # -- Live plot update --------------------------------------
            if live_plot and step % live_update_every == 0 and len(com_x_list) > 1:
                cx = np.array(com_x_list)
                cy = np.array(com_y_list)
                ct = np.array(time_list)
                ht = np.array(head_rows)   # shape (N, 11)

                ln_xy.set_data(cx, cy)
                ln_com.set_data([cx[-1]], [cy[-1]])
                ln_head.set_data([ht[-1, 1]], [ht[-1, 2]])

                ln_x.set_data(ct, cx)
                ln_y.set_data(ct, cy)
                ln_hx.set_data(ht[:, 0], ht[:, 1])
                ln_hy.set_data(ht[:, 0], ht[:, 2])

                for ax in (ax_xy, ax_time):
                    ax.relim(); ax.autoscale_view()

                elapsed = time.perf_counter() - sim_start
                pct = 100 * step / total_steps
                info_text.set_text(
                    f"t={data.time:.1f}/{duration:.1f}s  ({pct:.0f}%)\n"
                    f"COM  ({cx[-1]:.3f}, {cy[-1]:.3f})\n"
                    f"head ({ht[-1,1]:.3f}, {ht[-1,2]:.3f}, {ht[-1,3]:.3f})\n"
                    f"wall {elapsed:.1f}s"
                )
                fig_live.canvas.draw_idle()
                fig_live.canvas.flush_events()

            # -- Progress ----------------------------------------------
            if verbose and step % progress_interval == 0:
                pct = 100 * step / total_steps
                elapsed = time.perf_counter() - sim_start
                print(f"  {pct:5.1f}%  sim_time={data.time:.2f}s  "
                      f"wall={elapsed:.1f}s", end="\r", flush=True)

    finally:
        if gui_ctx is not None:
            gui_ctx.close()
        if live_plot:
            plt.ioff()
        if record_video:
            _renderer_top.close()
            _renderer_side.close()
            _vid_top.close()
            _vid_side.close()

    if verbose:
        elapsed = time.perf_counter() - sim_start
        print(f"\n  Simulation complete in {elapsed:.1f}s wall-time.")
        print()

    # -- Build arrays -----------------------------------------------------
    time_arr      = np.array(time_list)
    com_x_arr     = np.array(com_x_list)
    com_y_arr     = np.array(com_y_list)
    head_traj_arr = np.array(head_rows)   # shape (N, 11)

    # -- Save head trajectory CSV ------------------------------------------
    if traj_csv:
        _save_traj_csv(traj_csv, head_traj_arr)

    # -- Velocity analysis ------------------------------------------------

    # Mask out the settle period for statistics
    mask = time_arr >= settle_time
    if mask.sum() < 10:
        mask = np.ones_like(time_arr, dtype=bool)   # fallback

    t_eval  = time_arr[mask]
    x_eval  = com_x_arr[mask]
    y_eval  = com_y_arr[mask]

    # Linear regression for X and Y independently
    #   position(t) = V * t + C   → V = slope
    vx_fit = np.polyfit(t_eval, x_eval, 1)    # [slope, intercept]
    vy_fit = np.polyfit(t_eval, y_eval, 1)

    vx = vx_fit[0]   # world +X velocity [m/s]
    vy = vy_fit[0]   # world +Y velocity [m/s]

    # Primary travel direction (heading) in XY plane
    heading_rad = atan2(vy, vx)
    heading_deg = degrees(heading_rad)

    # Forward speed = component along primary travel direction
    v_forward = sqrt(vx**2 + vy**2)   # speed magnitude

    # Lateral drift = how much the perpendicular component deviates
    #   If heading is (cos θ, sin θ), lateral is (-sin θ, cos θ)
    if v_forward > 1e-9:
        fwd_unit = np.array([vx, vy]) / v_forward
    else:
        fwd_unit = np.array([1.0, 0.0])

    lat_unit = np.array([-fwd_unit[1], fwd_unit[0]])

    # Project COM positions onto forward / lateral axes
    fwd_pos = (x_eval * fwd_unit[0] + y_eval * fwd_unit[1])   # forward position
    lat_pos = (x_eval * lat_unit[0] + y_eval * lat_unit[1])   # lateral position

    # Net drift rates via linear fit on projected positions
    fwd_fit = np.polyfit(t_eval, fwd_pos, 1)   # [slope = speed, intercept]
    lat_fit = np.polyfit(t_eval, lat_pos, 1)

    v_fwd_mean = float(fwd_fit[0])             # signed mean forward speed [m/s]
    v_lateral  = float(lat_fit[0])             # net lateral drift rate    [m/s]

    # Instantaneous lateral oscillation amplitude (separate from drift)
    dt_rec   = np.diff(t_eval)
    dx_inst  = np.diff(x_eval) / dt_rec
    dy_inst  = np.diff(y_eval) / dt_rec
    vel_xy   = np.stack([dx_inst, dy_inst], axis=1)
    v_lat_osc = float(np.mean(np.abs(vel_xy @ lat_unit)))   # oscillation mag

    # Net displacement over the evaluation window
    disp_x = x_eval[-1] - x_eval[0]
    disp_y = y_eval[-1] - y_eval[0]

    t_span       = t_eval[-1] - t_eval[0]
    disp_forward = float(np.dot([disp_x, disp_y], fwd_unit))
    disp_lateral = float(np.dot([disp_x, disp_y], lat_unit))

    # -- Print results ----------------------------------------------------
    if verbose:
        print("-" * 65)
        print("  RESULTS")
        print("-" * 65)
        print(f"  Evaluation window    : {t_eval[0]:.2f}s - {t_eval[-1]:.2f}s  "
              f"(Dt = {t_span:.2f}s)")
        print(f"  COM start            : ({x_eval[0]:.4f}, {y_eval[0]:.4f}) m")
        print(f"  COM end              : ({x_eval[-1]:.4f}, {y_eval[-1]:.4f}) m")
        print()
        print(f"  Travel heading       : {heading_deg:+.1f}deg  (world XY, +X = 0deg)")
        print(f"  V_x  (world)         : {vx:+.4f} m/s")
        print(f"  V_y  (world)         : {vy:+.4f} m/s")
        print(f"  +--------------------------------------------")
        print(f"  |  V_forward          : {v_fwd_mean:+.4f} m/s  ({v_fwd_mean*100:+.2f} cm/s)")
        print(f"  |  V_lateral drift    : {v_lateral:+.4f} m/s  ({v_lateral*100:+.2f} cm/s)  [net drift rate]")
        print(f"  |  V_lateral osc      : {v_lat_osc:+.4f} m/s  ({v_lat_osc*100:+.2f} cm/s)  [body oscillation]")
        print(f"  |  Disp forward       : {disp_forward:+.4f} m  ({disp_forward*100:+.2f} cm)")
        print(f"  |  Disp lateral       : {disp_lateral:+.4f} m  ({disp_lateral*100:+.2f} cm)")
        print(f"  +--------------------------------------------")
        print()

    result = dict(
        v_forward       = v_fwd_mean,
        v_lateral       = v_lateral,
        v_lateral_osc   = v_lat_osc,
        vx_world        = vx,
        vy_world        = vy,
        heading_deg     = heading_deg,
        disp_forward    = disp_forward,
        disp_lateral    = disp_lateral,
        com_x_arr    = com_x_arr,
        com_y_arr    = com_y_arr,
        time_arr     = time_arr,
        head_traj    = head_traj_arr,
        params = dict(
            num_modules = num_modules,
            req_length  = req_length,
            amp_z       = amp_z,
            amp_y       = amp_y,
            frequency   = frequency,
            num_frames  = num_frames,
            duration    = duration,
        ),
    )

    # -- Optional post-run trajectory plot -------------------------------
    if show_plot:
        _plot_trajectory(result, settle_time)
    elif live_plot:
        # Keep the live window open after simulation finishes
        import matplotlib.pyplot as plt
        plt.ioff()
        print("  [live plot - close the window to continue]")
        plt.show()

    return result


# -----------------------------------------------------------------------------
# Head trajectory CSV
# -----------------------------------------------------------------------------

TRAJ_COLS = ["t", "x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz"]

def _save_traj_csv(path: str, arr: np.ndarray):
    """
    Save head-module trajectory array (N, 11) to a CSV file.
    Columns: t, x, y, z, qw, qx, qy, qz, vx, vy, vz
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(TRAJ_COLS)
        for row in arr:
            writer.writerow([f"{v:.8g}" for v in row])
    print(f"  Head trajectory saved -> {path}  ({len(arr)} rows)")


# -----------------------------------------------------------------------------
# Post-run trajectory plot
# -----------------------------------------------------------------------------

def _plot_trajectory(result: dict, settle_time: float = 2.0):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    time_arr  = result["time_arr"]
    com_x_arr = result["com_x_arr"]
    com_y_arr = result["com_y_arr"]
    mask      = time_arr >= settle_time

    fig = plt.figure(figsize=(13, 5))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.35)

    # -- XY trajectory --
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(com_x_arr[~mask], com_y_arr[~mask], color="lightgray", lw=1, label="settle")
    ax1.plot(com_x_arr[mask],  com_y_arr[mask],  color="steelblue", lw=1.5, label="eval")
    ax1.scatter([com_x_arr[mask][0]],  [com_y_arr[mask][0]],  color="green", zorder=5, label="start")
    ax1.scatter([com_x_arr[mask][-1]], [com_y_arr[mask][-1]], color="red",   zorder=5, label="end")
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.set_title("COM trajectory (world XY)")
    ax1.legend(fontsize=7)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.4)

    # -- X and Y vs time --
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_arr, com_x_arr, color="steelblue", lw=1, label="COM X")
    ax2.plot(time_arr, com_y_arr, color="tomato",    lw=1, label="COM Y")
    ax2.axvline(settle_time, color="gray", ls="--", lw=0.8, label="settle end")

    # Overlay linear fits
    mask_fit  = time_arr[mask]
    vx        = result["vx_world"]
    vy        = result["vy_world"]
    x_eval0   = com_x_arr[mask][0] - vx * mask_fit[0]
    y_eval0   = com_y_arr[mask][0] - vy * mask_fit[0]
    ax2.plot(mask_fit, vx * mask_fit + x_eval0, "b--", lw=0.8, label=f"fit X (V={vx*100:.2f} cm/s)")
    ax2.plot(mask_fit, vy * mask_fit + y_eval0, "r--", lw=0.8, label=f"fit Y (V={vy*100:.2f} cm/s)")
    ax2.set_xlabel("Simulation time [s]")
    ax2.set_ylabel("Position [m]")
    ax2.set_title("COM position vs time")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.4)

    # -- Head trajectory mini-plot (XY head path) --
    ht = result.get("head_traj")
    if ht is not None and len(ht) > 0:
        ax1.plot(ht[:, 1], ht[:, 2], color="orange", lw=0.8, ls="--", label="head")
        ax1.legend(fontsize=7)

    # -- Summary text --
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    p   = result["params"]
    lines = [
        "--- Parameters ---",
        f"  Modules   : {p['num_modules']}",
        f"  Az        : {p['amp_z']}",
        f"  Ay        : {p['amp_y']}",
        f"  Frequency : {p['frequency']}",
        f"  Duration  : {p['duration']} s",
        "",
        "--- Results ---",
        f"  Heading       : {result['heading_deg']:+.1f}deg",
        f"  V_forward     : {result['v_forward']*100:+.2f} cm/s",
        f"  V_lat drift   : {result['v_lateral']*100:+.2f} cm/s",
        f"  V_lat osc     : {result['v_lateral_osc']*100:+.2f} cm/s",
        f"  Disp fwd      : {result['disp_forward']*100:+.2f} cm",
        f"  Disp lat      : {result['disp_lateral']*100:+.2f} cm",
    ]
    ax3.text(0.05, 0.95, "\n".join(lines),
             transform=ax3.transAxes,
             fontsize=9, verticalalignment="top", family="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("Serpentine Gait Evaluation", fontsize=12, fontweight="bold")
    plt.savefig("gait_evaluation.png", dpi=150, bbox_inches="tight")
    print("  Trajectory plot saved -> gait_evaluation.png")
    plt.show()


# -----------------------------------------------------------------------------
# CSV helper
# -----------------------------------------------------------------------------

def append_csv(path: str, result: dict):
    fieldnames = [
        "num_modules", "req_length", "amp_z", "amp_y", "frequency",
        "num_frames", "duration",
        "v_forward_cm_s", "v_lateral_cm_s", "v_lateral_osc_cm_s",
        "heading_deg", "disp_forward_cm", "disp_lateral_cm",
    ]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        p = result["params"]
        writer.writerow({
            "num_modules":      p["num_modules"],
            "req_length":       p["req_length"],
            "amp_z":            p["amp_z"],
            "amp_y":            p["amp_y"],
            "frequency":        p["frequency"],
            "num_frames":       p["num_frames"],
            "duration":         p["duration"],
            "v_forward_cm_s":       round(result["v_forward"] * 100, 4),
            "v_lateral_cm_s":       round(result["v_lateral"] * 100, 4),
            "v_lateral_osc_cm_s":   round(result["v_lateral_osc"] * 100, 4),
            "heading_deg":          round(result["heading_deg"], 2),
            "disp_forward_cm":      round(result["disp_forward"] * 100, 4),
            "disp_lateral_cm":      round(result["disp_lateral"] * 100, 4),
            "v_lateral_osc_cm_s":   round(result["v_lateral_osc"] * 100, 4),
        })
    print(f"  Results appended -> {path}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Headless serpentine-gait evaluator — reports V_forward and "
            "V_lateral for a given parameter set."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num",        type=int,   default=5,     help="Number of snake modules.")
    parser.add_argument("--req-length", type=float, default=1.14,  help="Required segment length (m).")
    parser.add_argument("--amp-z",      type=float, default=0.4,   help="Vertical sine amplitude Az.")
    parser.add_argument("--amp-y",      type=float, default=0.4,   help="Horizontal sine amplitude Ay.")
    parser.add_argument("--frequency",  type=float, default=3.0,   help="Spatial frequency of the sine wave.")
    parser.add_argument("--num-frames", type=int,   default=200,   help="Gait frames per cycle.")
    parser.add_argument("--duration",   type=float, default=30.0,  help="Total simulation duration (s).")
    parser.add_argument("--interval",   type=float, default=0.03,  help="Gait-frame interval (s).")
    parser.add_argument("--settle",     type=float, default=2.0,   help="Settle time excluded from stats (s).")
    parser.add_argument("--viewer",     action="store_true", default=False,
                        help="Open the interactive MuJoCo viewer while simulating.")
    parser.add_argument("--plot",       action="store_true", default=False,
                        help="Show trajectory plot and save gait_evaluation.png.")
    parser.add_argument("--live-plot",  action="store_true", default=False,
                        help="Show a live matplotlib plot that updates during simulation.")
    parser.add_argument("--traj-csv",   type=str,   default=None,
                        help="Save head-module trajectory (t,x,y,z,qw,qx,qy,qz,vx,vy,vz) to this CSV.")
    parser.add_argument("--csv",        type=str,   default=None,
                        help="Append summary results to this CSV file.")
    parser.add_argument("--floor-friction", type=float, default=0.3,
                        help="Floor geom slide-friction coefficient.")
    parser.add_argument("--body-friction",  type=float, default=0.3,
                        help="Body geom slide-friction coefficient (patches chain XML default).")
    parser.add_argument("--quiet",      action="store_true", default=False,
                        help="Suppress verbose output.")
    parser.add_argument("--record",       action="store_true", default=False,
                        help="Record top-view and side-view MP4 videos during simulation.")
    parser.add_argument("--video-fps",    type=int,   default=30,
                        help="Video frame rate for recorded MP4 files.")
    parser.add_argument("--video-width",  type=int,   default=960,
                        help="Video frame width in pixels (even number recommended).")
    parser.add_argument("--video-height", type=int,   default=540,
                        help="Video frame height in pixels (even number recommended).")
    parser.add_argument("--video-prefix", type=str,   default="snake",
                        help="Output filename prefix (produces <prefix>_top.mp4 and <prefix>_side.mp4).")
    parser.add_argument("--zoom-factor",  type=float, default=1.0,
                        help="Camera zoom factor (0.5 = 50%% zoom in, 2.0 = zoom out).")
    parser.add_argument("--show-contacts", action="store_true", default=False,
                        help="Visualize contact forces and contact points in video.")
    args = parser.parse_args()

    result = run_evaluation(
        num_modules   = args.num,
        req_length    = args.req_length,
        amp_z         = args.amp_z,
        amp_y         = args.amp_y,
        frequency     = args.frequency,
        num_frames    = args.num_frames,
        duration      = args.duration,
        gait_interval = args.interval,
        settle_time   = args.settle,
        use_viewer    = args.viewer,
        show_plot     = args.plot,
        live_plot     = args.live_plot,
        traj_csv      = args.traj_csv,
        verbose       = not args.quiet,
        floor_friction= args.floor_friction,
        body_friction = args.body_friction,
        record_video  = args.record,
        video_fps     = args.video_fps,
        video_width   = args.video_width,
        video_height  = args.video_height,
        video_prefix  = args.video_prefix,
        zoom_factor   = args.zoom_factor,
        show_contacts = args.show_contacts,
    )

    if args.csv:
        append_csv(args.csv, result)

    # Machine-readable one-liner for scripting
    print(
        f"[RESULT] "
        f"v_forward={result['v_forward']*100:.3f}cm/s  "
        f"v_lateral_drift={result['v_lateral']*100:.3f}cm/s  "
        f"v_lateral_osc={result['v_lateral_osc']*100:.3f}cm/s  "
        f"heading={result['heading_deg']:+.1f}deg  "
        f"disp_fwd={result['disp_forward']*100:.2f}cm  "
        f"disp_lat={result['disp_lateral']*100:.2f}cm"
    )


if __name__ == "__main__":
    main()

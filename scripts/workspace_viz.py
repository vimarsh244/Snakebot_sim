#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
workspace_viz.py  –  Map the reachable workspace of a single snakebot module.

Freezes the bottom base plate to the world frame, sweeps both servos
(servo_bottom and servo_top) through ±max_angle degrees, lets the
closed-loop mechanism settle with gravity=0 so it converges to the
pure geometric equilibrium, and records the top base-plate position.

Layout (4 panels):
  ┌──────────────────┬──────────────────┐
  │  Rendered Top    │  Rendered Side   │
  │  (robot view)    │  (robot view)    │
  ├──────────────────┼──────────────────┤
  │  Workspace Top   │  Workspace Side  │
  │  (X-Y scatter)   │  (Y-Z scatter)   │
  └──────────────────┴──────────────────┘

Usage (from repo root, PowerShell):
    python scripts/workspace_viz.py
    python scripts/workspace_viz.py --max-angle 25 --resolution 30 --save workspace.png
"""

import os
import re
import argparse
import numpy as np
import mujoco
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEST_XML  = os.path.join(REPO_ROOT, "snake_description", "best.xml")
TMP_XML   = os.path.join(REPO_ROOT, "snake_description", "_ws_tmp.xml")


# ── XML helpers ───────────────────────────────────────────────────────────────

def _patched_xml(gravity_off: bool) -> str:
    """
    Return best.xml with:
      - bottom-plate freejoint removed  (bottom plate fixed to world)
      - top-plate   freejoint kept      (top plate positioned by equality constraints)
      - gravity optionally zeroed       (mechanism settles to pure geometry)
    """
    with open(BEST_XML, "r") as fh:
        xml = fh.read()

    # Fix bottom plate
    xml = xml.replace(
        '<freejoint name="bottom_plate_free" />',
        "<!-- bottom plate fixed to world -->",
    )

    # Inject gravity="0 0 0" into the <option> tag
    if gravity_off:
        xml = re.sub(
            r'(<option\b[^>]*?)(\s*/>)',
            r'\1 gravity="0 0 0"\2',
            xml, count=1,
        )

    return xml


def _load_model(gravity_off: bool) -> mujoco.MjModel:
    xml = _patched_xml(gravity_off)
    old_cwd = os.getcwd()
    os.chdir(os.path.join(REPO_ROOT, "snake_description"))
    try:
        with open(os.path.basename(TMP_XML), "w") as fh:
            fh.write(xml)
        model = mujoco.MjModel.from_xml_path(os.path.basename(TMP_XML))
    finally:
        os.chdir(old_cwd)
        try:
            os.remove(TMP_XML)
        except OSError:
            pass
    return model


# ── Workspace sweep ───────────────────────────────────────────────────────────

def compute_workspace(
    max_angle_deg: float = 25.0,
    resolution:    int   = 30,
    settle_steps:  int   = 3000,
    verbose:       bool  = True,
) -> np.ndarray:
    """
    Sweep servo_bottom × servo_top through a (resolution × resolution) grid
    of ±max_angle_deg, settle physics with gravity=0, record top-plate pos.

    Returns ndarray shape (N, 5): [a_bot, a_top, x_m, y_m, z_m]
    """
    model = _load_model(gravity_off=True)
    data  = mujoco.MjData(model)

    act_bot  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "servo_bottom")
    act_top  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "servo_top")
    body_top = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "top-base-plate-v1")

    if act_bot < 0 or act_top < 0:
        raise RuntimeError("Cannot find servo actuators – check names in best.xml")
    if body_top < 0:
        raise RuntimeError("Cannot find top-base-plate-v1 body")

    angles = np.linspace(np.radians(-max_angle_deg),
                         np.radians( max_angle_deg),
                         resolution)

    total = resolution * resolution
    rows  = []

    if verbose:
        print(f"  servo_bottom [id {act_bot}], servo_top [id {act_top}]")
        print(f"  tracking   : top-base-plate-v1 (body id {body_top})")
        print(f"  grid       : {resolution}×{resolution} = {total} samples")
        print(f"  settle     : {settle_steps} steps/pose  (gravity = 0)")
        print()

    for i, a_bot in enumerate(angles):
        for j, a_top in enumerate(angles):
            mujoco.mj_resetData(model, data)
            data.ctrl[act_bot] = float(a_bot)
            data.ctrl[act_top] = float(a_top)

            for _ in range(settle_steps):
                mujoco.mj_step(model, data)

            pos = data.xpos[body_top].copy()   # world-frame XYZ in metres
            rows.append([a_bot, a_top, pos[0], pos[1], pos[2]])

        if verbose:
            print(f"\r  {(i+1)*resolution}/{total}  "
                  f"({(i+1)/resolution*100:.0f}%)", end="", flush=True)

    if verbose:
        print("\n  sweep complete.")

    return np.array(rows, dtype=np.float64)


# ── Offscreen rendering ───────────────────────────────────────────────────────

def _render_view(
    model:     mujoco.MjModel,
    data:      mujoco.MjData,
    azimuth:   float,
    elevation: float,
    distance:  float,
    lookat:    np.ndarray,
    width:     int = 600,
    height:    int = 480,
) -> np.ndarray:
    """Return an RGB array for one offscreen camera pose."""
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam               = mujoco.MjvCamera()
    cam.type          = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth       = azimuth
    cam.elevation     = elevation
    cam.distance      = distance
    cam.lookat[:]     = lookat
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    img = renderer.render()
    renderer.close()
    return img


def render_robot_views(settle_steps: int = 3000):
    """
    Load model WITH gravity, settle at neutral pose, return
    (top_img, side_img) as RGB arrays.
    """
    model = _load_model(gravity_off=False)
    data  = mujoco.MjData(model)

    # Settle neutral pose
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    body_top = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "top-base-plate-v1")
    body_bot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bottom-base-plate-v1")
    lookat   = (data.xpos[body_top] + data.xpos[body_bot]) / 2.0

    # Top-down view
    top_img = _render_view(
        model, data,
        azimuth=90, elevation=-89, distance=0.32, lookat=lookat,
        width=600, height=480,
    )

    # Side view (from the front/side)
    side_img = _render_view(
        model, data,
        azimuth=180, elevation=-15, distance=0.40, lookat=lookat,
        width=600, height=480,
    )

    return top_img, side_img


# ── Dimension arrow helper ────────────────────────────────────────────────────

def _dim_arrow(ax, x1, y1, x2, y2, label,
               color="dimgray", fontsize=9, offset=(0, 0)):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.2))
    mx = (x1 + x2) / 2 + offset[0]
    my = (y1 + y2) / 2 + offset[1]
    ax.text(mx, my, label, ha="center", va="center",
            fontsize=fontsize, color=color,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


# ── Main figure ───────────────────────────────────────────────────────────────

def plot_all(
    positions:     np.ndarray,
    max_angle_deg: float = 25.0,
    settle_steps:  int   = 3000,
    save_path:     str   = None,
):
    # ── Render robot views ──────────────────────────────────────────────────────
    print("\n  Rendering neutral-pose robot views…")
    has_render = False
    top_img = side_img = None
    try:
        top_img, side_img = render_robot_views(settle_steps=settle_steps)
        has_render = True
        print("  Render OK.")
    except Exception as exc:
        print(f"  Warning: render failed ({exc}), skipping render panels.")

    # ── Workspace data ──────────────────────────────────────────────────────────
    x_mm = positions[:, 2] * 1000
    y_mm = positions[:, 3] * 1000
    z_mm = positions[:, 4] * 1000

    x_rng = x_mm.max() - x_mm.min()
    y_rng = y_mm.max() - y_mm.min()
    z_rng = z_mm.max() - z_mm.min()

    print(f"\n  Workspace extents (top-plate centre):")
    print(f"    X : [{x_mm.min():.1f}, {x_mm.max():.1f}] mm  →  span {x_rng:.1f} mm")
    print(f"    Y : [{y_mm.min():.1f}, {y_mm.max():.1f}] mm  →  span {y_rng:.1f} mm")
    print(f"    Z : [{z_mm.min():.1f}, {z_mm.max():.1f}] mm  →  span {z_rng:.1f} mm")

    # ── Figure layout ───────────────────────────────────────────────────────────
    if has_render:
        fig = plt.figure(figsize=(16, 12))
        gs  = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.35,
                               left=0.06, right=0.97, top=0.93, bottom=0.07)
        ax_rt = fig.add_subplot(gs[0, 0])
        ax_rs = fig.add_subplot(gs[0, 1])
        ax_wt = fig.add_subplot(gs[1, 0])
        ax_ws = fig.add_subplot(gs[1, 1])

        ax_rt.imshow(top_img)
        ax_rt.set_title("Robot – Top View (neutral pose)", fontsize=11, fontweight="bold")
        ax_rt.axis("off")

        ax_rs.imshow(side_img)
        ax_rs.set_title("Robot – Side View (neutral pose)", fontsize=11, fontweight="bold")
        ax_rs.axis("off")
    else:
        fig, (ax_wt, ax_ws) = plt.subplots(1, 2, figsize=(14, 6.5),
                                            gridspec_kw={"wspace": 0.35})

    fig.suptitle(
        f"Single Module Workspace  (servo sweep ±{max_angle_deg:.0f}°)",
        fontsize=15, fontweight="bold",
    )

    # ── Scatter style ───────────────────────────────────────────────────────────
    DOT_C = "#cc2222"
    DOT_S = 22
    DOT_A = 0.70

    # helper: safe padding
    def _pad(val): return max(val * 0.18, 1.5)

    # ── Top-view scatter  X vs Y ────────────────────────────────────────────────
    ax_wt.scatter(x_mm, y_mm, s=DOT_S, c=DOT_C, alpha=DOT_A,
                  linewidths=0, zorder=3)
    ax_wt.set_title("Workspace – Top View  (X‑Y)", fontsize=11, fontweight="bold")
    ax_wt.set_xlabel("x (mm)", fontsize=10)
    ax_wt.set_ylabel("y (mm)", fontsize=10)
    ax_wt.set_aspect("equal")
    ax_wt.grid(True, alpha=0.22)

    px, py = _pad(x_rng), _pad(y_rng)
    if x_rng > 0.5:
        _dim_arrow(ax_wt,
                   x_mm.min(), y_mm.min() - py,
                   x_mm.max(), y_mm.min() - py,
                   f"{x_rng:.1f} mm", offset=(0, -py * 0.45))
    if y_rng > 0.5:
        _dim_arrow(ax_wt,
                   x_mm.min() - px, y_mm.min(),
                   x_mm.min() - px, y_mm.max(),
                   f"{y_rng:.1f} mm", offset=(-px * 0.55, 0))

    # ── Side-view scatter  Y vs Z ───────────────────────────────────────────────
    ax_ws.scatter(y_mm, z_mm, s=DOT_S, c=DOT_C, alpha=DOT_A,
                  linewidths=0, zorder=3)
    ax_ws.set_title("Workspace – Side View  (Y‑Z)", fontsize=11, fontweight="bold")
    ax_ws.set_xlabel("y (mm)", fontsize=10)
    ax_ws.set_ylabel("z (mm)", fontsize=10)
    ax_ws.set_aspect("equal")
    ax_ws.grid(True, alpha=0.22)

    py2, pz = _pad(y_rng), _pad(z_rng)
    if y_rng > 0.5:
        _dim_arrow(ax_ws,
                   y_mm.min(), z_mm.min() - pz,
                   y_mm.max(), z_mm.min() - pz,
                   f"{y_rng:.1f} mm", offset=(0, -pz * 0.45))
    if z_rng > 0.5:
        _dim_arrow(ax_ws,
                   y_mm.min() - py2, z_mm.min(),
                   y_mm.min() - py2, z_mm.max(),
                   f"{z_rng:.1f} mm", offset=(-py2 * 0.55, 0))

    # ── Legend ──────────────────────────────────────────────────────────────────
    fig.legend(
        handles=[mpatches.Patch(color=DOT_C, label="Sample points")],
        loc="lower center", ncol=1, fontsize=10,
        frameon=True, framealpha=0.9,
        bbox_to_anchor=(0.5, 0.005),
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Figure saved → {save_path}")

    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Workspace visualisation – single snakebot module.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--max-angle",    type=float, default=25.0,  metavar="DEG",
                        help="Servo range in degrees (sweep ±max_angle)")
    parser.add_argument("--resolution",   type=int,   default=30,    metavar="N",
                        help="Samples per servo axis (total = N²)")
    parser.add_argument("--settle-steps", type=int,   default=3000,  metavar="STEPS",
                        help="Physics steps to settle each pose (gravity=0)")
    parser.add_argument("--save",         type=str,   default=None,  metavar="PATH",
                        help="Save figure to this path (e.g. workspace.png)")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    print("=" * 60)
    print("  Single-Module Workspace Visualisation")
    print("=" * 60)
    print(f"  Servo range : ±{args.max_angle}°")
    print(f"  Resolution  : {args.resolution}² = {args.resolution**2} samples")
    print(f"  Settle steps: {args.settle_steps}  (gravity = 0 → geometric settle)")
    print()

    positions = compute_workspace(
        max_angle_deg=args.max_angle,
        resolution=args.resolution,
        settle_steps=args.settle_steps,
    )

    plot_all(
        positions,
        max_angle_deg=args.max_angle,
        settle_steps=args.settle_steps,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()

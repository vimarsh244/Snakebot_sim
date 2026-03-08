#!/usr/bin/env python3
"""
plot_sim2real.py
================
1. Grouped bar chart  – sim vs real v_x across all plywood configurations,
   with signed-error labels and a MAPE annotation per gait.
2. Error-direction strip  – signed error (%) per config, coloured by gait.
3. Head-module XY trajectories – for the three key configurations:
      (a) best-real serpentine   Az=0.6, Ay=0.6, f=3.0
      (b) best-real caterpillar  Az=0.6, Ay=0.0, f=2.5
      (c) best-sim-match serp   Az=0.4, Ay=0.4, f=2.5  (lowest gap, -8 %)
4. Head velocity (vx) time-series for those same three configs.

Saves  sim2real_overview.pdf  and  sim2real_trajectories.pdf
"""

import os, sys, csv, subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── repo root ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
PYTHON = sys.executable

# ─────────────────────────────────────────────────────────────────────────────
# Hard-coded real results (plywood, from experimental table)
# key: (ay, az, f)  →  ('S'|'C', vx_real_m_per_min)
# ─────────────────────────────────────────────────────────────────────────────
REAL = {
    (0.4, 0.4, 2.5): ("S", 0.93),
    (0.4, 0.4, 3.0): ("S", 0.70),
    (0.4, 0.6, 3.0): ("S", 1.46),
    (0.6, 0.4, 3.0): ("S", 0.65),
    (0.6, 0.6, 3.0): ("S", 1.60),
    (0.0, 0.4, 2.0): ("C", 0.62),
    (0.0, 0.4, 2.5): ("C", 0.72),
    (0.0, 0.4, 3.0): ("C", 0.27),
    (0.0, 0.6, 2.0): ("C", 0.47),
    (0.0, 0.6, 2.5): ("C", 1.42),
    (0.0, 0.6, 3.0): ("C", 0.97),
}

# Trajectory configs: (label, az, ay, f)
TRAJ_CONFIGS = [
    ("Serp best\n$A_z{=}A_y{=}0.6,f{=}3$",   0.6, 0.6, 3.0),
    ("Carp best\n$A_z{=}0.6,f{=}2.5$",         0.6, 0.0, 2.5),
    ("Serp best fit\n$A_z{=}A_y{=}0.4,f{=}2.5$", 0.4, 0.4, 2.5),
]
TRAJ_FILES = ["traj_sim_key0.csv", "traj_sim_key1.csv", "traj_sim_key2.csv"]

# Distinct color pairs: light (sim) and vivid (real) per gait type
SIM_COLORS  = {"S": "#90CAF9", "C": "#FFCC80"}   # pastel – simulation bars
REAL_COLORS = {"S": "#1565C0", "C": "#E65100"}   # vivid – hardware bars
# Keep a single alias for scatter / line plots that still need one colour per gait
COLORS = REAL_COLORS


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: load sim_sweep.csv
# ─────────────────────────────────────────────────────────────────────────────
def load_sweep():
    path = "sim_sweep.csv"
    if not os.path.exists(path):
        raise FileNotFoundError("sim_sweep.csv not found – run run_sim_sweep.py first.")
    rows = list(csv.DictReader(open(path)))
    configs = []
    for r in rows:
        ay = float(r["amp_y"]); az = float(r["amp_z"]); f = float(r["frequency"])
        key = (ay, az, f)
        if key not in REAL:
            continue
        gait, rv = REAL[key]
        sim_vx = float(r["v_forward_cm_s"]) * 0.6   # cm/s -> m/min
        err = (rv - sim_vx) / rv * 100  # +ve = hardware faster than sim (under-prediction)
        label = (f"$A_y{ay},A_z{az}$\n$f{f}$"
                 if gait == "S"
                 else f"$A_z{az},f{f}$")
        short = (f"{ay}/{az}/{f}" if gait == "S" else f"-/{az}/{f}")
        configs.append(dict(gait=gait, ay=ay, az=az, f=f,
                            sim=sim_vx, real=rv, err=err,
                            label=short))
    # Sort: serpentine first, then caterpillar, within each by real vx
    configs.sort(key=lambda c: (0 if c["gait"] == "S" else 1, c["real"]))
    return configs


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: generate head-module traj CSVs for key configs
# ─────────────────────────────────────────────────────────────────────────────
def generate_trajs():
    for (label, az, ay, f), fp in zip(TRAJ_CONFIGS, TRAJ_FILES):
        if os.path.exists(fp):
            print(f"  [traj] {fp} already exists, skipping.")
            continue
        print(f"  [traj] Generating {fp}  Az={az} Ay={ay} f={f}...")
        subprocess.run([
            PYTHON, "scripts/evaluate_gait.py",
            "--amp-z", str(az), "--amp-y", str(ay),
            "--frequency", str(f),
            "--duration", "30",
            "--traj-csv", fp,
            "--quiet",
        ], check=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Figure 1 – overview (bar chart + error strip)
# ─────────────────────────────────────────────────────────────────────────────
def plot_overview(configs):
    fig = plt.figure(figsize=(7.16, 4.6))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[3, 1.2], hspace=0.44)
    ax_bar = fig.add_subplot(gs[0])
    ax_err = fig.add_subplot(gs[1])

    n = len(configs)
    x = np.arange(n)
    w = 0.35

    sim_vals  = [c["sim"]  for c in configs]
    real_vals = [c["real"] for c in configs]
    errs      = [c["err"]  for c in configs]
    gaits     = [c["gait"] for c in configs]
    labels    = [c["label"] for c in configs]

    bar_sim  = ax_bar.bar(x - w/2, sim_vals,  w, label="Simulation",
                          color=[SIM_COLORS[g]  for g in gaits],
                          edgecolor=[REAL_COLORS[g] for g in gaits], lw=0.8)
    bar_real = ax_bar.bar(x + w/2, real_vals, w, label="Hardware",
                          color=[REAL_COLORS[g] for g in gaits],
                          edgecolor="none")

    # Annotate error on top of each sim bar  (+ve = hardware faster)
    for i, (bsim, breal, err) in enumerate(zip(bar_sim, bar_real, errs)):
        top = max(bsim.get_height(), breal.get_height()) + 0.04
        color = "#2e7d32" if err > 20 else ("#d32f2f" if err < -20 else "#555")
        ax_bar.text(x[i], top, f"{err:+.0f}%", ha="center", va="bottom",
                    fontsize=6.8, color=color, fontweight="bold")

    # Divider between serpentine and caterpillar
    n_serp = sum(1 for c in configs if c["gait"] == "S")
    ax_bar.axvline(n_serp - 0.5, color="gray", lw=1.2, ls="--", alpha=0.7)
    ax_bar.set_ylim(0, 2.25)   # set early so text placement is correct
    ax_bar.text(n_serp/2 - 0.5, 2.10,
                "Serpentine", ha="center", va="bottom", fontsize=9,
                color=COLORS["S"], fontweight="bold")
    ax_bar.text(n_serp + (n - n_serp)/2 - 0.5, 2.10,
                "Caterpillar", ha="center", va="bottom", fontsize=9,
                color=COLORS["C"], fontweight="bold")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=7.5)
    ax_bar.set_ylabel("$v_x$ (m/min)", fontsize=10)
    ax_bar.set_title("Sim-to-Real Velocity Comparison (Plywood)",
                     fontsize=9, fontweight="bold")
    ax_bar.yaxis.grid(True, alpha=0.35)
    ax_bar.set_axisbelow(True)

    # Legend and MAPE annotations
    serp_errs = [abs(c["err"]) for c in configs if c["gait"] == "S"]
    carp_errs = [abs(c["err"]) for c in configs if c["gait"] == "C"]
    mape_s = sum(serp_errs) / len(serp_errs)
    mape_c = sum(carp_errs) / len(carp_errs)

    sim_serp_patch  = mpatches.Patch(facecolor=SIM_COLORS["S"],  edgecolor=REAL_COLORS["S"], lw=0.8,
                                      label=f"Sim — Serpentine (MAPE {mape_s:.0f}%)")
    real_serp_patch = mpatches.Patch(facecolor=REAL_COLORS["S"],
                                      label="Real — Serpentine")
    sim_carp_patch  = mpatches.Patch(facecolor=SIM_COLORS["C"],  edgecolor=REAL_COLORS["C"], lw=0.8,
                                      label=f"Sim — Caterpillar (MAPE {mape_c:.0f}%)")
    real_carp_patch = mpatches.Patch(facecolor=REAL_COLORS["C"],
                                      label="Real — Caterpillar")
    ax_bar.legend(handles=[sim_serp_patch, real_serp_patch,
                            sim_carp_patch, real_carp_patch],
                  fontsize=8, loc="upper left", ncol=2)

    # ── Error strip ──────────────────────────────────────────────────────────
    dot_colors = [COLORS[c["gait"]] for c in configs]
    ax_err.scatter(x, errs, c=dot_colors, s=60, zorder=3)
    ax_err.axhline(0,   color="black", lw=0.8)
    ax_err.axhline(20,  color="#cccccc", lw=0.6, ls="--")
    ax_err.axhline(-20, color="#cccccc", lw=0.6, ls="--")
    ax_err.fill_between([-0.5, n-0.5], -20, 20, color="lightgray", alpha=0.25,
                        label="±20 % band")
    ax_err.axvline(n_serp - 0.5, color="gray", lw=1.2, ls="--", alpha=0.7)
    for i, (xi, err, g) in enumerate(zip(x, errs, gaits)):
        ax_err.text(xi, err + (3 if err >= 0 else -5), f"{err:+.0f}%",
                    ha="center", va="bottom", fontsize=6.5,
                    color=COLORS[g])
    ax_err.set_xlim(-0.5, n - 0.5)
    ax_err.set_xticks(x)
    ax_err.set_xticklabels(labels, fontsize=7.5)
    ax_err.set_ylabel("$\\varepsilon$ (%)", fontsize=9)
    ax_err.set_title("Signed Gap: $(v^{\\mathrm{real}}{-}v^{\\mathrm{sim}})/v^{\\mathrm{real}}$  "
                     "[+ = hardware exceeds sim]",
                     fontsize=8)
    ax_err.legend(fontsize=8, loc="upper right")
    ax_err.yaxis.grid(True, alpha=0.3)
    ax_err.set_axisbelow(True)

    plt.savefig("sim2real_overview.pdf", bbox_inches="tight")
    plt.savefig("sim2real_overview.png", dpi=180, bbox_inches="tight")
    print("  Saved sim2real_overview.pdf / .png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Figure 2 – Head-module trajectory + velocity plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_trajectories():
    traj_colors = ["#1565C0", "#E65100", "#2E7D32"]
    traj_names  = [t[0].replace("\n", " ") for t in TRAJ_CONFIGS]

    fig = plt.figure(figsize=(7.16, 4.8))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.28, wspace=0.32)

    for col, (fp, color, name, cfg) in enumerate(
            zip(TRAJ_FILES, traj_colors, traj_names, TRAJ_CONFIGS)):
        label_full, az, ay, f = cfg
        rows = list(csv.DictReader(open(fp)))
        t  = np.array([float(r["t"])  for r in rows])
        x  = np.array([float(r["x"])  for r in rows])
        y  = np.array([float(r["y"])  for r in rows])
        z  = np.array([float(r["z"])  for r in rows])
        # Forward speed = dx/dt  (world-X only; avoids vertical oscillation in cvel)
        fwd_speed = np.gradient(x, t) * 60   # m/s -> m/min

        # Real vx for this config
        key  = (ay, az, f)
        gait, rv = REAL.get(key, ("?", None))
        sim_vx_mpm = float(
            next(r for r in csv.DictReader(open("sim_sweep.csv"))
                 if abs(float(r["amp_y"])-ay)<0.001
                 and abs(float(r["amp_z"])-az)<0.001
                 and abs(float(r["frequency"])-f)<0.001
                 )["v_forward_cm_s"]) * 0.6
        err_str = f"{(sim_vx_mpm-rv)/rv*100:+.0f}%" if rv else "N/A"

        settle_mask = t >= 2.0

        # ── Row 0: XY trajectory ──────────────────────────────────────
        ax_xy = fig.add_subplot(gs[0, col])
        ax_xy.plot(x[~settle_mask], y[~settle_mask],
                   color="lightgray", lw=0.8, label="settle")
        ax_xy.plot(x[settle_mask],  y[settle_mask],
                   color=color, lw=1.5, label="steady-state")
        ax_xy.scatter(x[settle_mask][[0]],  y[settle_mask][[0]],
                      color="green", s=40, zorder=5, label="start")
        ax_xy.scatter(x[settle_mask][[-1]], y[settle_mask][[-1]],
                      color="red",   s=40, zorder=5, label="end")

        # Commanded straight-line reference (from start to end along x-axis)
        xs, xe = x[settle_mask][0], x[settle_mask][-1]
        ys_ref  = y[settle_mask][0]
        ax_xy.plot([xs, xe], [ys_ref, ys_ref],
                   color="black", ls="--", lw=0.9, label="reference")

        ax_xy.set_xlabel("X [m]", fontsize=8)
        ax_xy.set_ylabel("Y [m]", fontsize=8)
        ax_xy.set_title(
            f"{name}\nSim {sim_vx_mpm:.2f} | Real {rv:.2f} m/min  ($\\varepsilon$={err_str})",
            fontsize=8, fontweight="bold")
        if col == 0: ax_xy.legend(fontsize=6.5, loc="upper left")
        ax_xy.set_aspect("equal")
        ax_xy.grid(True, alpha=0.35)
        ax_xy.tick_params(labelsize=7)

        # Lateral deviation (signed Y-distance from reference line)
        y_ref_val = ys_ref
        lat_dev = y[settle_mask] - y_ref_val
        rms_lat = np.sqrt(np.mean(lat_dev**2)) * 100  # cm
        ax_xy.text(0.98, 0.04,
                   f"RMS lat. dev. = {rms_lat:.1f} cm",
                   transform=ax_xy.transAxes, fontsize=6.5,
                   ha="right", va="bottom",
                   bbox=dict(boxstyle="round", fc="white", alpha=0.75))

        # ── Row 1: Forward speed time-series (dx/dt) ─────────────────
        ax_v = fig.add_subplot(gs[1, col])
        # ~1 s rolling mean at 100 Hz
        win = 100
        fs_m = fwd_speed[settle_mask]
        speed_smooth = np.convolve(fs_m, np.ones(win)/win, mode="same")
        ax_v.plot(t[settle_mask], fs_m,
                  color=color, lw=0.5, alpha=0.25)
        ax_v.plot(t[settle_mask], speed_smooth,
                  color=color, lw=1.8, label="smoothed $v_x$")
        ax_v.axhline(sim_vx_mpm, color=color,   ls="--", lw=1.0,
                     label=f"sim mean {sim_vx_mpm:.2f}")
        if rv:
            ax_v.axhline(rv, color="black", ls=":", lw=1.2,
                         label=f"real {rv:.2f}")
        ax_v.axvline(2.0, color="gray", lw=0.7, ls="--", alpha=0.6)
        ax_v.set_ylim(-1.0, max(3.0, sim_vx_mpm * 2.5))
        ax_v.set_xlabel("Sim time [s]", fontsize=8)
        ax_v.set_ylabel("$v_x$ [m/min]  (fwd only)", fontsize=8)
        ax_v.set_title("Head forward speed vs. time", fontsize=8)
        ax_v.legend(fontsize=6.5)
        ax_v.grid(True, alpha=0.35)
        ax_v.tick_params(labelsize=7)

    fig.suptitle("Head-Module Trajectories — MuJoCo Simulation",
                 fontsize=8.5, fontweight="bold")
    plt.savefig("sim2real_trajectories.pdf", bbox_inches="tight")
    plt.savefig("sim2real_trajectories.png", dpi=180, bbox_inches="tight")
    print("  Saved sim2real_trajectories.pdf / .png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.dpi":  130,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    print("Loading sweep data...")
    configs = load_sweep()

    print("Generating head-trajectory CSVs for 3 key configs...")
    generate_trajs()

    print("Plotting Figure 1: sim-to-real overview...")
    plot_overview(configs)

    print("Plotting Figure 2: head-module trajectories...")
    plot_trajectories()

    print("Done.")

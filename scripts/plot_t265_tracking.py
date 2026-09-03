#!/usr/bin/env python3
"""
plot_t265_tracking.py
=====================
Properly aligns and overlays the T265 real trajectory against the simulation
for the best-performing gait (serpentine, Ay=Az=0.6, f=3.0).

Alignment pipeline
------------------
Step 1  Raw T265 xy-plane: find net displacement vector (start -> end).
Step 2  Rotate all T265 points onto that vector.
        forward = projection onto net-displacement unit vector
        lateral = component perpendicular to net-displacement (true tracking error)
        This removes ALL yaw-offset drift; what remains is true oscillation.
Step 3  Detrend lateral (remove any residual slow VIO drift) with a linear fit.
Step 4  FFT of detrended lateral vs time for both real & sim.
        Extract dominant frequency, amplitude.
Step 5  Overlay plots with:
          - normalised forward distance (shape comparison)
          - absolute scale with sim distance/amplitude scaled to real
Step 6  Report everything in m/min.
"""

import os, sys, csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})
C_REAL = "#D32F2F"
C_SIM  = "#1565C0"
C_REF  = "#444444"

# =============================================================================
# 1. Load CSVs
# =============================================================================
def load_csv(path):
    rows = list(csv.DictReader(open(path)))
    return {k: np.array([float(r[k]) for r in rows])
            for k in ("t", "x", "y", "z", "vx", "vy", "vz")}

real_raw = load_csv("concrete_663_serp_trajectory_tracked.csv")
sim_raw  = load_csv("traj_output.csv")

# =============================================================================
# 1b. Remove big positional jumps from T265 data
#     At ~200 Hz the snake travels ~0.025 cm per step (2.1 m / 58.7 s / 200 Hz).
#     Any step larger than JUMP_THRESH is a sensor glitch; remove those samples
#     and time-shift all subsequent data to close the resulting gap.
# =============================================================================
JUMP_THRESH = 0.03   # metres – anything above this between consecutive samples

def remove_jumps(d, thresh=JUMP_THRESH):
    """Remove T265 glitch jumps and reconnect the timeline without gaps.

    For each pair of consecutive samples where the 3-D position distance
    exceeds `thresh`, the sample at the larger-displacement end is deleted
    and all later timestamps are shifted down to close the gap.

    Returns a new dict with the same keys.
    """
    x, y, z, t = d["x"].copy(), d["y"].copy(), d["z"].copy(), d["t"].copy()
    other = {k: d[k].copy() for k in d if k not in ("x", "y", "z", "t")}

    n_before = len(t)
    i = 1
    n_removed = 0
    while i < len(x):
        step = np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2 + (z[i]-z[i-1])**2)
        if step > thresh:
            # time gap that this jump occupied
            gap = t[i] - t[i-1]
            # delete sample i and close the time gap for all subsequent samples
            x = np.delete(x, i);  y = np.delete(y, i)
            z = np.delete(z, i);  t = np.delete(t, i)
            for k in other:
                other[k] = np.delete(other[k], i)
            t[i:] -= gap          # shift later times to remove the gap
            n_removed += 1
        else:
            i += 1

    if n_removed:
        print(f"  [T265 de-jump] removed {n_removed} glitch sample(s) "
              f"({n_before} -> {len(t)}), threshold = {thresh*100:.1f} cm")
    else:
        print("  [T265 de-jump] no jumps detected above "
              f"{thresh*100:.1f} cm threshold")

    out = {"x": x, "y": y, "z": z, "t": t}
    out.update(other)
    return out

real_raw = remove_jumps(real_raw)

# =============================================================================
# 2. Zero-origin (subtract first sample position & time)
# =============================================================================
def zero_origin(d):
    out = dict(d)
    for k in ("t", "x", "y", "z"):
        out[k] = d[k] - d[k][0]
    return out

real_z = zero_origin(real_raw)
sim_z  = zero_origin(sim_raw)

rt = real_z["t"]
sim_t = sim_z["t"]

# =============================================================================
# 3. Align T265 to straight-line reference via start-to-end vector rotation
# =============================================================================
# Work in T265 horizontal plane (x, y).  Z is vertical / VIO drift – skip.
rx_raw = real_z["x"]
ry_raw = real_z["y"]

# Net displacement vector
dx, dy = float(rx_raw[-1]), float(ry_raw[-1])
D_real = np.hypot(dx, dy)

fwd_hat = np.array([dx, dy]) / D_real          # unit forward
lat_hat = np.array([-dy, dx]) / D_real         # unit left-perpendicular

pts       = np.column_stack([rx_raw, ry_raw])
real_fwd  = pts @ fwd_hat                       # (N,) metres along reference
real_lat  = pts @ lat_hat                       # (N,) metres off reference

real_vert = real_z["z"]                         # for 3-D display only

# Simulation: forward = +X, lateral = Y
sim_fwd  = sim_z["x"]
sim_lat  = sim_z["y"]
sim_vert = sim_z["z"]
D_sim    = float(sim_fwd[-1])

# =============================================================================
# 4. Detrend lateral signals (remove residual linear VIO drift)
# =============================================================================
def detrend(sig, t):
    coef = np.polyfit(t, sig, 1)
    return sig - np.polyval(coef, t)

real_lat_dt = detrend(real_lat, rt)
sim_lat_dt  = detrend(sim_lat,  sim_t)

# =============================================================================
# 5. Speeds in m/min
# =============================================================================
real_speed_mpm = D_real / rt[-1] * 60
sim_speed_mpm  = D_sim  / sim_t[-1] * 60

print("=" * 60)
print("Real (T265, concrete):")
print(f"  Distance : {D_real*100:.1f} cm  in  {rt[-1]:.1f} s")
print(f"  Speed    : {real_speed_mpm:.3f} m/min  "
      f"({real_speed_mpm/60*100:.2f} cm/s)")
print("Simulation:")
print(f"  Distance : {D_sim*100:.1f} cm  in  {sim_t[-1]:.1f} s")
print(f"  Speed    : {sim_speed_mpm:.3f} m/min  "
      f"({sim_speed_mpm/60*100:.2f} cm/s)")
print("=" * 60)

# =============================================================================
# 6. FFT – extract dominant oscillation frequency and amplitude
# =============================================================================
def dominant_fft(signal, dt, f_min=0.5, f_max=10.0):
    """Return (freq_Hz, one-sided_amplitude_m) of dominant component in band."""
    n      = len(signal)
    freqs  = np.fft.rfftfreq(n, d=dt)
    spec   = np.fft.rfft(signal)
    mag    = np.abs(spec) * 2.0 / n            # one-sided amplitude
    mag[0] = 0.0                                # zero DC
    mask   = (freqs >= f_min) & (freqs <= f_max)
    idx    = np.argmax(mag * mask)
    return float(freqs[idx]), float(mag[idx])

dt_real = float(np.mean(np.diff(rt)))
dt_sim  = float(np.mean(np.diff(sim_t)))

f_real, A_real = dominant_fft(real_lat_dt, dt_real)
f_sim,  A_sim  = dominant_fft(sim_lat_dt,  dt_sim)

print(f"\nFFT lateral oscillation:")
print(f"  Real : f = {f_real:.3f} Hz,  amplitude = {A_real*100:.2f} cm")
print(f"  Sim  : f = {f_sim:.3f} Hz,   amplitude = {A_sim*100:.2f} cm")

# =============================================================================
# 7. Match oscillation cycles: take min(N_real, N_sim) cycles from both,
#    then resample both onto a shared 1000-point normalised [0,1] grid.
# =============================================================================
N_real_cyc = f_real * rt[-1]        # ~33
N_sim_cyc  = f_sim  * sim_t[-1]     # ~25
N_use      = 10.0                         # show 10 clean cycles for clarity

# Trim real to hold exactly N_use cycles
T_real_trim = N_use / f_real
mask_r = rt <= T_real_trim
rt_c   = rt[mask_r]
rfwd_c = real_fwd[mask_r]
rlat_c = real_lat_dt[mask_r]
D_real_c = float(rfwd_c[-1])

# Trim sim to hold exactly N_use cycles
T_sim_trim = N_use / f_sim
mask_s = sim_t <= T_sim_trim
st_c   = sim_t[mask_s]
sfwd_c = sim_fwd[mask_s]
slat_c = sim_lat_dt[mask_s]
D_sim_c = float(sfwd_c[-1])

print(f"\nCycle matching: using {N_use:.1f} cycles")
print(f"  Real trimmed: {T_real_trim:.1f} s, {D_real_c*100:.1f} cm")
print(f"  Sim  trimmed: {T_sim_trim:.1f} s, {D_sim_c*100:.1f} cm")

# Normalised forward distance (0 -> 1) for each trimmed segment
real_fwd_n_c = rfwd_c / D_real_c
sim_fwd_n_c  = sfwd_c / D_sim_c

# Common 1000-pt grid — resample both onto it so shapes overlap cleanly
N_GRID = 1000
grid   = np.linspace(0.0, 1.0, N_GRID)
real_on_grid = np.interp(grid, real_fwd_n_c, rlat_c)
sim_on_grid  = np.interp(grid, sim_fwd_n_c,  slat_c)

# Amplitude scale (real / sim) so lateral amplitudes match for overlay
amp_ratio_c = A_real / A_sim

# Full-run normalised (for absolute-scale panel) using ALL real data
real_fwd_n = real_fwd / D_real
sim_fwd_n  = sim_fwd  / D_sim

# =============================================================================
# 8. L2 tracking error (using detrended lateral, trimmed segment)
# =============================================================================
mean_l2_real = np.trapezoid(np.abs(rlat_c), rt_c) / rt_c[-1]
mean_l2_sim  = np.trapezoid(np.abs(slat_c),  st_c) / st_c[-1]
pct_real     = mean_l2_real / D_real_c * 100

print(f"\nMean L2 tracking error  (real, detrended) : "
      f"{mean_l2_real*100:.2f} cm  ({pct_real:.1f}% of {D_real_c*100:.0f} cm segment)")
print(f"Mean L2 tracking error  (sim)             : {mean_l2_sim*100:.2f} cm")

# =============================================================================
# 9. Rolling ±1-sigma band for full real lateral (0.5 s window)
# =============================================================================
win = max(1, int(0.5 / dt_real))
kern = np.ones(win) / win
real_sm  = np.convolve(real_lat_dt, kern, mode="same")
real_std = np.sqrt(np.convolve((real_lat_dt - real_sm)**2, kern, mode="same"))

# band on the trimmed/grid segment
real_sm_c  = np.interp(grid, real_fwd_n_c,
                        np.convolve(rlat_c, np.ones(win)/win, mode="same"))
real_std_c = np.interp(grid, real_fwd_n_c,
                        np.sqrt(np.convolve((rlat_c - np.convolve(rlat_c, np.ones(win)/win,
                        mode="same"))**2, np.ones(win)/win, mode="same")))

# =============================================================================
# 10.  FIGURE 1 – Diagnostic three-panel
# =============================================================================
fig1 = plt.figure(figsize=(15, 5))
gs   = gridspec.GridSpec(1, 3, width_ratios=[1.9, 1.6, 1.5],
                         wspace=0.44, figure=fig1)

# --- 10a. 3-D view (trimmed segments, detrended lateral) ---
ax3 = fig1.add_subplot(gs[0], projection="3d")
vert_clip = np.clip(real_vert[mask_r], -0.10, 0.10)
ax3.plot(sfwd_c,  slat_c * amp_ratio_c,  sim_vert[mask_s], color=C_SIM,  lw=1.4, alpha=0.88,
         label="Simulation")
ax3.plot(rfwd_c,  rlat_c,               vert_clip,         color=C_REAL, lw=1.0, alpha=0.75,
         label="Real (T265)")
ref_end = max(float(rfwd_c.max()), float(sfwd_c.max()))
ax3.plot([0, ref_end], [0, 0], [0, 0],
         color=C_REF, lw=1.4, ls="--", alpha=0.55, label="Reference path")
ax3.scatter([0], [0], [0], color="green", s=50, zorder=6)
ax3.set_xlabel("Forward (m)", labelpad=6)
ax3.set_ylabel("Lateral (m)", labelpad=6)
ax3.set_zlabel("Vertical (m)", labelpad=6)
ax3.set_title(r"3-D head trajectory, serpentine $A_y{=}A_z{=}0.6$, $f{=}3.0\,$Hz",
              fontsize=9, pad=8)
ax3.legend(fontsize=7.5, framealpha=0.85)
ax3.view_init(elev=22, azim=-55)

# --- 10b. Cycle-matched shape overlay on common grid ---
ax_n = fig1.add_subplot(gs[1])
ax_n.plot(grid, sim_on_grid  * amp_ratio_c * 100, color=C_SIM,  lw=1.8, alpha=0.9,
          label="Simulation")
ax_n.plot(grid, real_on_grid * 100,               color=C_REAL, lw=1.2, alpha=0.85,
          label="Real (T265)")
ax_n.fill_between(grid,
                  (real_sm_c - real_std_c)*100,
                  (real_sm_c + real_std_c)*100,
                  color=C_REAL, alpha=0.18, label=r"$\pm1\sigma$")
ax_n.axhline(0, color=C_REF, lw=1.1, ls="--", alpha=0.55, label="Reference path")
ax_n.set_xlabel("Normalised forward distance")
ax_n.set_ylabel("Lateral displacement (cm)")
ax_n.set_title(f"Lateral displacement, {N_use:.0f} cycles", fontsize=9)
ax_n.legend(fontsize=8, framealpha=0.9)
ax_n.grid(True, alpha=0.30)

# --- 10c. FFT spectra ---
ax_f = fig1.add_subplot(gs[2])
n_r = len(real_lat_dt);  n_s = len(sim_lat_dt)
fr_r = np.fft.rfftfreq(n_r, d=dt_real)
fr_s = np.fft.rfftfreq(n_s, d=dt_sim)
mag_r = np.abs(np.fft.rfft(real_lat_dt)) * 2 / n_r * 100   # cm
mag_s = np.abs(np.fft.rfft(sim_lat_dt))  * 2 / n_s * 100
ax_f.plot(fr_s[1:], mag_s[1:], color=C_SIM,  lw=1.6, alpha=0.9,
          label=f"Simulation ({f_sim:.2f} Hz)")
ax_f.plot(fr_r[1:], mag_r[1:], color=C_REAL, lw=1.2, alpha=0.85,
          label=f"Real (T265) ({f_real:.2f} Hz)")
ax_f.axvline(f_sim,  color=C_SIM,  lw=1.0, ls=":", alpha=0.7)
ax_f.axvline(f_real, color=C_REAL, lw=1.0, ls=":", alpha=0.7)
ax_f.set_xlim(0, 9);  ax_f.set_ylim(bottom=0)
ax_f.set_xlabel("Frequency (Hz)")
ax_f.set_ylabel("Amplitude (cm)")
ax_f.set_title("FFT of lateral oscillation", fontsize=9)
ax_f.legend(fontsize=8, framealpha=0.9)
ax_f.grid(True, alpha=0.30)

fig1.suptitle(
    r"T265 head trajectory vs simulation, serpentine $A_y{=}A_z{=}0.6$, $f{=}3.0\,$Hz"
    f"  |  Mean $L_2$ = {mean_l2_real*100:.2f} cm ({pct_real:.1f}%)",
    fontsize=9, y=1.01,
)
fig1.savefig("t265_tracking_3d.pdf", bbox_inches="tight")
fig1.savefig("t265_tracking_3d.png", bbox_inches="tight", dpi=200)
print("\nSaved: t265_tracking_3d.pdf / .png")

# =============================================================================
# 11.  FIGURE 2 – Paper figure (two panels)
# =============================================================================
fig2, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

# ── 11a. Cycle-matched, common-grid overlay (paper panel) ─────────────────────
ax = axes[0]
ax.plot(grid, sim_on_grid  * amp_ratio_c * 100, color=C_SIM,  lw=2.0,
        alpha=0.9, zorder=3, label="Simulation")
ax.plot(grid, real_on_grid * 100,               color=C_REAL, lw=1.5,
        alpha=0.85, zorder=2, label="Real (T265)")
ax.fill_between(grid,
                (real_sm_c - real_std_c)*100,
                (real_sm_c + real_std_c)*100,
                color=C_REAL, alpha=0.18, zorder=0, label=r"$\pm1\sigma$")
ax.axhline(0, color=C_REF, lw=1.4, ls="--", alpha=0.65, label="Reference path")
ax.set_xlabel("Normalised forward displacement", fontsize=11)
ax.set_ylabel("Lateral displacement (cm)", fontsize=11)
ax.set_title(r"T265 vs Simulation, serpentine $A_y{=}A_z{=}0.6$, $f{=}3.0\,$Hz",
             fontsize=10)
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.30)
ax.annotate(
    f"Real:  $f$ = {f_real:.2f} Hz,  $A$ = {A_real*100:.1f} cm\n"
    f"Sim:   $f$ = {f_sim:.2f} Hz,   $A$ = {A_sim*100:.1f} cm\n"
    f"Mean $L_2$ = {mean_l2_real*100:.2f} cm  ({pct_real:.1f}% of path)",
    xy=(0.02, 0.98), xycoords="axes fraction",
    va="top", ha="left", fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#aaaaaa", alpha=0.90),
)

# ── 11b. Absolute scale: sim resampled to full real run length ─────────────────
# Resample sim_on_grid (covering D_sim_c) stretched to D_real metres.
# Forward axis = grid * D_real, lateral = sim_on_grid * amp_ratio_c
ax2 = axes[1]
ax2.plot(grid * D_real,  real_on_grid * 100,               color=C_REAL,
         lw=1.5, alpha=0.85, label="Real (T265)")
ax2.fill_between(grid * D_real,
                 (real_sm_c - real_std_c)*100,
                 (real_sm_c + real_std_c)*100,
                 color=C_REAL, alpha=0.18, label=r"$\pm1\sigma$")
ax2.axhline(0, color=C_REF, lw=1.4, ls="--", alpha=0.65, label="Reference path")
ax2.plot(grid * D_real,  sim_on_grid  * amp_ratio_c * 100, color=C_SIM,
         lw=1.8, alpha=0.9, label="Simulation")
ax2.set_xlabel("Forward displacement (m)", fontsize=11)
ax2.set_ylabel("Lateral displacement (cm)", fontsize=11)
ax2.set_title(f"Absolute scale, {N_use:.0f} matched cycles", fontsize=10)
ax2.legend(fontsize=8.5, framealpha=0.9)
ax2.grid(True, alpha=0.30)

fig2.suptitle(
    r"T265 head trajectory vs simulation, serpentine $A_y{=}A_z{=}0.6$, $f{=}3.0\,$Hz",
    fontsize=11,
)
fig2.savefig("t265_tracking_2d.pdf", bbox_inches="tight")
fig2.savefig("t265_tracking_2d.png", bbox_inches="tight", dpi=200)
print("Saved: t265_tracking_2d.pdf / .png")

# =============================================================================
# 12. Paper fill-in summary
# =============================================================================
print("\n" + "=" * 60)
print("PAPER FILL-IN SUMMARY")
print("=" * 60)
print(f"  Full run length     : {D_real:.3f} m  ({D_real*100:.1f} cm)")
print(f"  Matched segment     : {D_real_c:.3f} m  ({N_use:.0f} cycles)")
print(f"  Speed (real)        : {real_speed_mpm:.3f} m/min")
print(f"  Speed (sim)         : {sim_speed_mpm:.3f} m/min")
print(f"  Oscillation (real)  : {f_real:.3f} Hz,  A = {A_real*100:.2f} cm")
print(f"  Oscillation (sim)   : {f_sim:.3f} Hz,   A = {A_sim*100:.2f} cm")
print(f"  Amplitude ratio     : {amp_ratio_c:.2f}x")
print(f"  Mean L2 error       : {mean_l2_real*100:.2f} cm  ({pct_real:.1f}%  of segment)")
print("=" * 60)

plt.show()

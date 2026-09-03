#!/usr/bin/env python3
"""
plot_friction.py
================
Analysis figures for the friction sweep (friction_sweep_results.csv).

Produces three IEEE-column-width figures:
  friction_mape_curve.pdf   – MAPE vs μ for both sweep types and both gaits
  friction_per_config.pdf   – Per-config |ε| at baseline μ=0.3 vs best tuned μ
  friction_best_bars.pdf    – Grouped sim/real bar chart at the best μ

Run friction_sweep.py first to generate friction_sweep_results.csv.
"""

import os, sys, csv
# Force UTF-8 output so Greek letters in labels don't crash on Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

DATA_CSV = "friction_sweep_results.csv"
matplotlib.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          8,
    "axes.titlesize":     8.5,
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "figure.dpi":         160,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "lines.linewidth":    1.4,
})

IEEE_W1 = 3.5    # single column
IEEE_W2 = 7.16   # double column

REAL = {
    (0.4, 0.4, 2.5): ("S", 0.93),
    (0.4, 0.4, 3.0): ("S", 0.70),
    (0.4, 0.6, 3.0): ("S", 1.46),
    (0.6, 0.4, 3.0): ("S", 0.65),
    (0.6, 0.6, 3.0): ("S", 1.60),
    (0.4, 0.0, 2.0): ("C", 0.62),
    (0.4, 0.0, 2.5): ("C", 0.72),
    (0.4, 0.0, 3.0): ("C", 0.27),
    (0.6, 0.0, 2.0): ("C", 0.47),
    (0.6, 0.0, 2.5): ("C", 1.42),
    (0.6, 0.0, 3.0): ("C", 0.97),
}

C_SERP  = "#1565C0"
C_CARP  = "#E65100"
C_ISO   = "#2E7D32"   # green  – isotropic sweep line
C_FLR   = "#6A1B9A"   # purple – floor-only sweep line
C_BASE  = "#9E9E9E"   # grey   – baseline


# ─────────────────────────────────────────────────────────────────────────────
def load():
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(
            f"{DATA_CSV} not found.\nRun:  python scripts/friction_sweep.py"
        )
    rows = []
    with open(DATA_CSV, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "stype": r["sweep_type"],
                "mu_fl": float(r["floor_friction"]),
                "mu_bd": float(r["body_friction"]),
                "az":    float(r["amp_z"]),
                "ay":    float(r["amp_y"]),
                "f":     float(r["frequency"]),
                "gait":  r["gait"],
                "v_sim": float(r["v_sim_mpm"]),
                "v_real":float(r["v_real_mpm"]),
                "aerr":  float(r["abs_err_pct"]),
                "gap":   float(r["gap_pct"]),
            })
    return rows


def mape(rows, stype=None, mu_fl=None, mu_bd=None, gait=None):
    sub = [r for r in rows
           if (stype is None or r["stype"] == stype)
           and (mu_fl is None or abs(r["mu_fl"] - mu_fl) < 1e-9)
           and (mu_bd is None or abs(r["mu_bd"] - mu_bd) < 1e-9)
           and (gait  is None or r["gait"]  == gait)]
    return np.mean([r["aerr"] for r in sub]) if sub else np.nan


def pivot_mape(rows, stype, gait=None):
    """Return sorted lists (mu_vals, mape_vals) for one sweep type + optional gait filter."""
    mu_set = sorted({r["mu_fl"] for r in rows if r["stype"] == stype})
    mus, mapes = [], []
    for mu in mu_set:
        mu_bd = mu if stype == "isotropic" else 0.3
        m = mape(rows, stype=stype, mu_fl=mu, mu_bd=mu_bd, gait=gait)
        if not np.isnan(m):
            mus.append(mu)
            mapes.append(m)
    return np.array(mus), np.array(mapes)


def best_mu(rows, stype="isotropic"):
    mus, mapes = pivot_mape(rows, stype)
    if len(mus) == 0:
        return None, None
    idx = np.argmin(mapes)
    return float(mus[idx]), float(mapes[idx])


def config_label(az, ay, f, gait):
    if gait == "S":
        return f"S {az}/{ay}@{f}"
    return f"C {az}@{f}"


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: MAPE vs μ curve  (double-column width, 2 rows × 2 cols)
# ─────────────────────────────────────────────────────────────────────────────
def fig_mape_curve(rows):
    fig, ax = plt.subplots(1, 1, figsize=(IEEE_W1 + 0.6, 2.8),
                           constrained_layout=True)

    stype, color = "isotropic", C_ISO
    mus_all, mr_all = pivot_mape(rows, stype)
    mus_s,   mr_s   = pivot_mape(rows, stype, gait="S")
    mus_c,   mr_c   = pivot_mape(rows, stype, gait="C")

    l1, = ax.plot(mus_all, mr_all, "o-",  color=color,  label="All configs", lw=1.8, ms=4)
    l2, = ax.plot(mus_s,   mr_s,   "s--", color=C_SERP, label="Serpentine",  lw=1.2, ms=3.5)
    l3, = ax.plot(mus_c,   mr_c,   "^--", color=C_CARP, label="Caterpillar", lw=1.2, ms=3.5)

    if len(mus_all):
        best_idx = np.argmin(mr_all)
        l4 = ax.axvline(mus_all[best_idx], color=color, lw=0.9, ls=":",
                        alpha=0.8, label=f"best $\\mu$={mus_all[best_idx]:.2f}")
        ax.scatter([mus_all[best_idx]], [mr_all[best_idx]], s=60, color=color, zorder=5)
        ax.legend(handles=[l1, l2, l3, l4], fontsize=7, loc="lower left", frameon=True)

    ax.set_xlabel("Friction coefficient  $\\mu$")
    ax.set_ylabel("MAPE  (%)")
    ax.set_title("Sim-to-Real MAPE vs. Contact Friction (Plywood Surface)",
                 fontsize=8.5, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=0)

    _save(fig, "friction_mape_curve")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Per-config |ε| at baseline vs best μ  (horizontal bar comparison)
# ─────────────────────────────────────────────────────────────────────────────
def fig_per_config(rows):
    bmu, _ = best_mu(rows, "isotropic")
    if bmu is None:
        print("  No isotropic data yet – skipping per-config figure.")
        return

    base_mu = 0.3
    configs_ordered = sorted(REAL.keys(),
                             key=lambda k: (0 if REAL[k][0] == "S" else 1,
                                            REAL[k][1]))

    labels, errs_base, errs_best, colors = [], [], [], []
    for az, ay, f in configs_ordered:
        gait, _ = REAL[(az, ay, f)]
        # baseline row
        def get_err(mu, stype):
            mu_bd = mu if stype == "isotropic" else 0.3
            sub = [r for r in rows
                   if r["stype"] == stype
                   and abs(r["mu_fl"] - mu) < 1e-9
                   and abs(r["mu_bd"] - mu_bd) < 1e-9
                   and abs(r["az"] - az) < 1e-9
                   and abs(r["ay"] - ay) < 1e-9
                   and abs(r["f"]  - f)  < 1e-9]
            return sub[0]["aerr"] if sub else np.nan

        eb = get_err(base_mu, "isotropic")
        eo = get_err(bmu,     "isotropic")
        if np.isnan(eb) or np.isnan(eo):
            continue
        labels.append(config_label(az, ay, f, gait))
        errs_base.append(eb)
        errs_best.append(eo)
        colors.append(C_SERP if gait == "S" else C_CARP)

    n    = len(labels)
    y    = np.arange(n)
    h    = 0.35
    fig, ax = plt.subplots(figsize=(IEEE_W1 + 0.5, 0.42 * n + 0.8),
                           constrained_layout=True)

    bars_base = ax.barh(y + h/2, errs_base, h,
                        color=C_BASE, edgecolor="none", label=f"Baseline  μ={base_mu:.2f}")
    bars_best = ax.barh(y - h/2, errs_best, h,
                        color=colors, edgecolor="none", alpha=0.85,
                        label=f"Tuned     μ={bmu:.2f}")

    ax.axvline(20, color="lightgray", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.8)
    ax.set_xlabel("Absolute error  |ε|  (%)")
    ax.set_title(f"Per-config Error: Baseline vs Tuned  (μ={bmu:.2f})",
                 fontsize=8.5, fontweight="bold")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(axis="x", alpha=0.25)

    # Improvement arrows
    for i, (eb, eo) in enumerate(zip(errs_base, errs_best)):
        delta = eb - eo
        sign  = "▼" if delta > 0 else "▲"
        col   = "#2E7D32" if delta > 0 else "#C62828"
        ax.text(max(eb, eo) + 0.5, i, f"{sign}{abs(delta):.0f}%",
                va="center", ha="left", fontsize=5.5, color=col)

    _save(fig, "friction_per_config")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Grouped bar chart at best μ  (compare baseline vs tuned vs real)
# ─────────────────────────────────────────────────────────────────────────────
def fig_best_bars(rows):
    bmu, bmape = best_mu(rows, "isotropic")
    if bmu is None:
        print("  No isotropic data yet – skipping best-bars figure.")
        return

    configs_ordered = sorted(REAL.keys(),
                             key=lambda k: (0 if REAL[k][0] == "S" else 1,
                                            REAL[k][1]))

    SIM_CLR  = {"S": "#42A5F5", "C": "#FFA726"}   # tuned sim bars
    REAL_CLR = {"S": "#1565C0", "C": "#E65100"}   # hardware bars

    labels, gaits, v_tuned, v_real = [], [], [], []

    for az, ay, f in configs_ordered:
        gait, vr = REAL[(az, ay, f)]
        mu_bd = bmu
        sub = [r for r in rows
               if r["stype"] == "isotropic"
               and abs(r["mu_fl"] - bmu) < 1e-9
               and abs(r["mu_bd"] - mu_bd) < 1e-9
               and abs(r["az"] - az) < 1e-9
               and abs(r["ay"] - ay) < 1e-9
               and abs(r["f"]  - f)  < 1e-9]
        if not sub:
            continue
        v_tuned.append(sub[0]["v_sim"])
        v_real.append(vr)
        labels.append(config_label(az, ay, f, gait))
        gaits.append(gait)

    n = len(labels)
    x = np.arange(n)
    w = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(IEEE_W2, 3.2))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.38)

    # ── Bar chart ────────────────────────────────────────────────────────────
    ax.bar(x - w/2, v_tuned, w,
           color=[SIM_CLR[g]  for g in gaits], edgecolor="none")
    ax.bar(x + w/2, v_real,  w,
           color=[REAL_CLR[g] for g in gaits], edgecolor="none")

    # Gap % labels above the taller bar
    for i, (vt, vr) in enumerate(zip(v_tuned, v_real)):
        gap = (vr - vt) / vr * 100
        top = max(vt, vr) + 0.06
        ax.text(x[i], top, f"{gap:+.0f}%", ha="center", va="bottom",
                fontsize=5.8, color="#444444", fontweight="bold")

    n_serp = sum(1 for g in gaits if g == "S")
    ax.axvline(n_serp - 0.5, color="gray", lw=1, ls="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5, rotation=30, ha="right")
    ax.set_ylabel("$v_x$  (m/min)")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.5, n - 0.5)

    # ── Legend below axes ─────────────────────────────────────────────────────
    sim_serp  = mpatches.Patch(facecolor=SIM_CLR["S"],  label="Sim  (Serpentine)")
    sim_carp  = mpatches.Patch(facecolor=SIM_CLR["C"],  label="Sim  (Caterpillar)")
    hw_serp   = mpatches.Patch(facecolor=REAL_CLR["S"], label="Hardware  (Serpentine)")
    hw_carp   = mpatches.Patch(facecolor=REAL_CLR["C"], label="Hardware  (Caterpillar)")
    fig.legend(handles=[sim_serp, hw_serp, sim_carp, hw_carp],
               ncol=4, fontsize=7, frameon=True,
               loc="lower center",
               bbox_to_anchor=(0.5, 0.1),
               bbox_transform=fig.transFigure)

    _save(fig, "friction_best_bars")


# ─────────────────────────────────────────────────────────────────────────────
def _save(fig, name):
    for ext in ("pdf", "png"):
        path = f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading friction sweep results …")
    rows = load()
    print(f"  {len(rows)} rows loaded from {DATA_CSV}")

    print("\nFigure 1: MAPE vs μ curve …")
    fig_mape_curve(rows)

    print("\nFigure 2: Per-config error comparison …")
    fig_per_config(rows)

    print("\nFigure 3: Best-friction bar chart …")
    fig_best_bars(rows)

    print("\nDone.")

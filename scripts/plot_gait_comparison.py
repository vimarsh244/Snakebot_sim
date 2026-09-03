"""Generate gait-comparison bar charts (velocity & lateral shift) and save as PNG/PDF."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Style ──────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.4,
    "axes.axisbelow": True,
})

BLUE = "#4A90D9"
ORANGE = "#E8833A"
BLUE_EDGE = "#2A5A8C"
ORANGE_EDGE = "#A85A1A"

surfaces = ["Plywood", "Incline", "Concrete", "Rough/Dirt"]
x = np.arange(len(surfaces))
bar_w = 0.30

# ── Data ───────────────────────────────────────────────────────────────
# Panel 1: Best v_x (m/min)
serp_vx = [1.60, 1.24, 1.43, 1.52]
cat_vx  = [1.42, 1.33, 1.38, 1.30]

# Panel 2: Lateral shift velocity at best-v_x config
serp_lat = [0.00, 0.17, 0.29, 0.17]
cat_lat  = [0.00, 0.76, 0.67, 0.00]
# "n/a" flags for Plywood (index 0)
na_mask = [True, False, False, False]

# ── Figure ─────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8.5), constrained_layout=True)

# ===== Panel 1: Peak Forward Velocity =====
bars1_s = ax1.bar(x - bar_w / 2, serp_vx, bar_w,
                  color=BLUE, edgecolor=BLUE_EDGE, linewidth=0.8,
                  label="Serpentine", zorder=3)
bars1_c = ax1.bar(x + bar_w / 2, cat_vx, bar_w,
                  color=ORANGE, edgecolor=ORANGE_EDGE, linewidth=0.8,
                  label="Caterpillar", zorder=3)

# bar labels
ax1.bar_label(bars1_s, fmt="%.2f", fontsize=8, padding=2)
ax1.bar_label(bars1_c, fmt="%.2f", fontsize=8, padding=2)

ax1.set_xticks(x)
ax1.set_xticklabels(surfaces)
ax1.set_ylabel(r"Best $v_x$ (m/min)")
ax1.set_ylim(0, 1.95)
ax1.set_yticks([0, 0.4, 0.8, 1.2, 1.6])
ax1.set_title("Peak Forward Velocity by Surface", fontweight="bold")
ax1.legend(loc="upper right", framealpha=0.9, edgecolor="gray")

# ===== Panel 2: Lateral Shift Velocity =====
bars2_s = ax2.bar(x - bar_w / 2, serp_lat, bar_w,
                  color=BLUE, edgecolor=BLUE_EDGE, linewidth=0.8,
                  label="Serpentine", zorder=3)
bars2_c = ax2.bar(x + bar_w / 2, cat_lat, bar_w,
                  color=ORANGE, edgecolor=ORANGE_EDGE, linewidth=0.8,
                  label="Caterpillar", zorder=3)

# Custom bar labels: show "n/a" for Plywood, value otherwise
def label_lat_bars(bars, values, na_mask):
    for bar, val, is_na in zip(bars, values, na_mask):
        txt = "n/a" if is_na else f"{val:.2f}"
        ax2.annotate(txt,
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8)

label_lat_bars(bars2_s, serp_lat, na_mask)
label_lat_bars(bars2_c, cat_lat, na_mask)

ax2.set_xticks(x)
ax2.set_xticklabels(surfaces)
ax2.set_ylabel(r"$\dot{\Delta}_\perp$ (m/min)")
ax2.set_ylim(0, 0.95)
ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax2.set_title("Lateral Shift Velocity at Best Forward-Velocity Configuration",
              fontweight="bold")
ax2.legend(loc="upper right", framealpha=0.9, edgecolor="gray")

# ── Save ───────────────────────────────────────────────────────────────
fig.savefig("gait_comparison.png", dpi=300)
fig.savefig("gait_comparison.pdf")
print("Saved  gait_comparison.png  and  gait_comparison.pdf")
plt.show()

"""Generate peak forward velocity table graph and save as PNG/PDF."""

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

surfaces = ["Plywood", "5° Incline", "Concrete", "Rough/Dirt"]
x = np.arange(len(surfaces))
bar_w = 0.35

# ── Data ───────────────────────────────────────────────────────────────
# Peak forward velocity (v_x^max in m/min)
serp_vx = [1.60, 1.24, 1.43, 1.52]
cat_vx  = [1.42, 0.94, 1.38, 1.30]

# ── Figure ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)

bars_s = ax.bar(x - bar_w / 2, serp_vx, bar_w,
                color=BLUE, edgecolor=BLUE_EDGE, linewidth=0.8,
                label="Serpentine", zorder=3)
bars_c = ax.bar(x + bar_w / 2, cat_vx, bar_w,
                color=ORANGE, edgecolor=ORANGE_EDGE, linewidth=0.8,
                label="Caterpillar", zorder=3)

# bar labels
ax.bar_label(bars_s, fmt="%.2f", fontsize=9, padding=2)
ax.bar_label(bars_c, fmt="%.2f", fontsize=9, padding=2)

ax.set_xticks(x)
ax.set_xticklabels(surfaces)
ax.set_ylabel(r"Peak Forward Velocity $v_x^{\max}$ (m/min)", fontsize=12)
ax.set_ylim(0, 1.95)
ax.set_yticks([0, 0.4, 0.8, 1.2, 1.6])
ax.set_title("Peak Forward Velocity by Gait and Surface", fontweight="bold", fontsize=13)
ax.legend(loc="upper right", framealpha=0.9, edgecolor="gray", fontsize=11)

# ── Save ───────────────────────────────────────────────────────────────
fig.savefig("peak_velocity.png", dpi=300)
fig.savefig("peak_velocity.pdf")
print("Saved  peak_velocity.png  and  peak_velocity.pdf")
plt.show()

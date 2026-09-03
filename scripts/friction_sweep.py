#!/usr/bin/env python3
"""
friction_sweep.py
=================
Sweep effective contact friction across the 11 plywood configurations to find
the μ value (and body/floor split) that minimises simulation-to-real MAPE.

Two sweeps are run in sequence:
  1. Combined isotropic   – body_friction = floor_friction = μ
                            μ ∈ MU_VALS   (changes effective friction)
  2. Floor-only           – body_friction = 0.3 (default), floor_friction = μ
                            Equivalent for μ > 0.3 since MuJoCo takes max(floor, body)
                            but included for completeness / verification.

Results saved to  friction_sweep_results.csv
Usage:
    python scripts/friction_sweep.py            # full sweep (≈ 10–15 min)
    python scripts/friction_sweep.py --quick    # shorter 15 s runs
"""

import os, sys, csv, subprocess, argparse
from itertools import product

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable

# ── Real-hardware results (plywood) ──────────────────────────────────────────
REAL = {
    # key: (amp_z, amp_y, frequency)  →  (gait, v_real_m_per_min)
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

# All 11 configs: (label, amp_z, amp_y, frequency)
ALL_CONFIGS = [
    ("serp_0.4_0.4_2.5", 0.4, 0.4, 2.5),
    ("serp_0.4_0.4_3.0", 0.4, 0.4, 3.0),
    ("serp_0.4_0.6_3.0", 0.4, 0.6, 3.0),
    ("serp_0.6_0.4_3.0", 0.6, 0.4, 3.0),
    ("serp_0.6_0.6_3.0", 0.6, 0.6, 3.0),
    ("carp_0.4_2.0",     0.4, 0.0, 2.0),
    ("carp_0.4_2.5",     0.4, 0.0, 2.5),
    ("carp_0.4_3.0",     0.4, 0.0, 3.0),
    ("carp_0.6_2.0",     0.6, 0.0, 2.0),
    ("carp_0.6_2.5",     0.6, 0.0, 2.5),
    ("carp_0.6_3.0",     0.6, 0.0, 3.0),
]

# Friction values to sweep
MU_VALS = [0.20, 0.30, 0.45, 0.60, 0.75, 0.90, 1.10, 1.35, 1.60]

OUTPUT_CSV = "friction_sweep_results.csv"
FIELDNAMES = [
    "sweep_type", "floor_friction", "body_friction",
    "amp_z", "amp_y", "frequency", "gait",
    "v_sim_mpm", "v_real_mpm", "abs_err_pct", "gap_pct",
]


def run_one(az, ay, f, floor_mu, body_mu, duration):
    """Run evaluate_gait for one config+friction, return v_forward in m/min."""
    cmd = [
        PYTHON, "scripts/evaluate_gait.py",
        "--amp-z", str(az), "--amp-y", str(ay), "--frequency", str(f),
        "--duration", str(duration),
        "--floor-friction", str(floor_mu),
        "--body-friction",  str(body_mu),
        "--quiet",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL,
                                      text=True, cwd=ROOT)
        for line in out.splitlines():
            if line.startswith("[RESULT]"):
                tok = {p.split("=")[0]: p.split("=")[1]
                       for p in line.split()[1:] if "=" in p}
                return float(tok["v_forward"].rstrip("cm/s")) * 0.6  # m/min
    except Exception as e:
        print(f"    ERROR: {e}")
    return None


def existing_keys(path):
    """Return set of (sweep_type, floor_friction, body_friction, az, ay, f)
    already present in the CSV so we can resume a interrupted sweep."""
    keys = set()
    if not os.path.exists(path):
        return keys
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            keys.add((row["sweep_type"],
                      float(row["floor_friction"]),
                      float(row["body_friction"]),
                      float(row["amp_z"]),
                      float(row["amp_y"]),
                      float(row["frequency"])))
    return keys


def append_row(path, row):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow(row)


def mape_for(rows, sweep_type, floor_mu, body_mu):
    errs = [abs(r["abs_err_pct"]) for r in rows
            if r["sweep_type"] == sweep_type
            and abs(r["floor_friction"] - floor_mu) < 1e-9
            and abs(r["body_friction"]  - body_mu)  < 1e-9]
    return sum(errs) / len(errs) if errs else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Use 15 s runs instead of 25 s for speed.")
    parser.add_argument("--output", default=OUTPUT_CSV)
    args = parser.parse_args()

    duration  = 15 if args.quick else 25
    out_path  = args.output
    done_keys = existing_keys(out_path)

    # ── Sweep definitions: (sweep_type, floor_mu, body_mu) ──────────────────
    sweeps = []
    for mu in MU_VALS:
        sweeps.append(("isotropic", mu,   mu))        # body = floor = μ
        sweeps.append(("floor_only", mu,  0.30))      # body fixed @ 0.30
    # Remove duplicates where mu == 0.30 (isotropic(0.3) == floor_only(0.3))

    total = len(sweeps) * len(ALL_CONFIGS)
    run_n = 0

    print(f"Friction sweep  |  {len(sweeps)} friction settings x "
          f"{len(ALL_CONFIGS)} configs = {total} runs  |  "
          f"duration={duration}s  |  saving -> {out_path}")
    print("=" * 70)

    all_rows = []  # collect for in-memory MAPE summary

    for stype, flr, bdy in sweeps:
        for label, az, ay, f in ALL_CONFIGS:
            run_n += 1
            key = (stype, flr, bdy, az, ay, f)
            if key in done_keys:
                print(f"  [{run_n:3d}/{total}] SKIP  {stype:12s} mu_fl={flr:.2f} "
                      f"mu_bd={bdy:.2f}  {label}")
                continue

            gait, v_real = REAL[(az, ay, f)]
            print(f"  [{run_n:3d}/{total}] {stype:12s} mu_fl={flr:.2f} "
                  f"mu_bd={bdy:.2f}  {label} ...", end="", flush=True)

            v_sim = run_one(az, ay, f, flr, bdy, duration)
            if v_sim is None:
                print("  FAILED")
                continue

            abs_err = abs(v_sim - v_real) / v_real * 100
            gap     = (v_real - v_sim)    / v_real * 100  # +ve = real faster

            row = dict(sweep_type=stype,
                       floor_friction=flr, body_friction=bdy,
                       amp_z=az, amp_y=ay, frequency=f, gait=gait,
                       v_sim_mpm=round(v_sim, 4), v_real_mpm=v_real,
                       abs_err_pct=round(abs_err, 2),
                       gap_pct=round(gap, 2))
            append_row(out_path, row)
            all_rows.append(row)
            done_keys.add(key)
            print(f"  sim={v_sim:.3f}  real={v_real:.3f}  err={abs_err:.1f}%")

    # ── Print MAPE summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Sweep type':<14} {'mu_floor':>8} {'mu_body':>8} {'MAPE all':>10} "
          f"{'MAPE S':>8} {'MAPE C':>8}")
    print("-" * 70)
    # Re-load all rows from file for the summary (includes previously cached)
    with open(out_path, newline="") as f:
        all_saved = list(csv.DictReader(f))

    seen = set()
    summary_rows = []
    for r in all_saved:
        k = (r["sweep_type"], float(r["floor_friction"]), float(r["body_friction"]))
        if k in seen:
            continue
        seen.add(k)
        rows_k = [x for x in all_saved
                  if x["sweep_type"] == r["sweep_type"]
                  and abs(float(x["floor_friction"]) - float(r["floor_friction"])) < 1e-9
                  and abs(float(x["body_friction"])  - float(r["body_friction"]))  < 1e-9]
        if not rows_k:
            continue
        mape_all = sum(abs(float(x["abs_err_pct"])) for x in rows_k) / len(rows_k)
        mape_s   = sum(abs(float(x["abs_err_pct"])) for x in rows_k
                       if x["gait"] == "S") or None
        n_s      = sum(1 for x in rows_k if x["gait"] == "S")
        mape_c   = sum(abs(float(x["abs_err_pct"])) for x in rows_k
                       if x["gait"] == "C") or None
        n_c      = sum(1 for x in rows_k if x["gait"] == "C")
        mape_s   = (mape_s / n_s  if n_s  else float("nan"))
        mape_c   = (mape_c / n_c  if n_c  else float("nan"))
        summary_rows.append((mape_all, r["sweep_type"],
                              float(r["floor_friction"]),
                              float(r["body_friction"]),
                              mape_s, mape_c))

    for mape_all, stype, flr, bdy, ms, mc in sorted(summary_rows):
        print(f"  {stype:<14} {flr:7.2f} {bdy:7.2f} {mape_all:9.2f}%"
              f"  {ms:7.2f}%  {mc:7.2f}%")

    if summary_rows:
        best = min(summary_rows, key=lambda x: x[0])
        print(f"\n  Best: {best[1]}  mu_floor={best[2]:.2f}  mu_body={best[3]:.2f}"
              f"  MAPE={best[0]:.2f}%")
    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()

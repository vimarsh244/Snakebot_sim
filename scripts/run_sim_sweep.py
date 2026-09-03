"""
run_sim_sweep.py
Run all plywood-table configurations and collect results into sim_sweep.csv
"""
import subprocess, sys

python = sys.executable

# (label, amp_z, amp_y, frequency)
serpentine_configs = [
    ("serp_0.4_0.4_2.5", 0.4, 0.4, 2.5),
    ("serp_0.4_0.4_3.0", 0.4, 0.4, 3.0),
    ("serp_0.4_0.6_3.0", 0.4, 0.6, 3.0),
    ("serp_0.6_0.4_2.5", 0.6, 0.4, 2.5),
    ("serp_0.6_0.4_3.0", 0.6, 0.4, 3.0),
    ("serp_0.6_0.6_3.0", 0.6, 0.6, 3.0),   # already done but re-run for CSV
]

caterpillar_configs = [
    ("carp_0.4_2.0", 0.4, 0.0, 2.0),
    ("carp_0.4_2.5", 0.4, 0.0, 2.5),
    ("carp_0.4_3.0", 0.4, 0.0, 3.0),
    ("carp_0.6_2.0", 0.6, 0.0, 2.0),
    ("carp_0.6_2.5", 0.6, 0.0, 2.5),   # already done
    ("carp_0.6_3.0", 0.6, 0.0, 3.0),
]

all_configs = serpentine_configs + caterpillar_configs
total = len(all_configs)

for i, (label, az, ay, freq) in enumerate(all_configs, 1):
    print(f"\n[{i}/{total}] {label}  Az={az}  Ay={ay}  f={freq}", flush=True)
    result = subprocess.run(
        [python, "scripts/evaluate_gait.py",
         "--amp-z", str(az),
         "--amp-y", str(ay),
         "--frequency", str(freq),
         "--duration", "30",
         "--csv", "sim_sweep.csv",
         "--quiet"],
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR (exit {result.returncode})")

print("\nDone. Results in sim_sweep.csv")

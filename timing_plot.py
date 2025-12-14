"""
Plots total preprocessing time versus landmark budget K for multiple strategies.
It scans a dataset folder for timing JSON files, extracts total runtime values, and
writes a single comparison figure (timing_comparison.png).

Reads (from FOLDER)
  - *_timing_*.json

Writes (to FOLDER)
  - timing_comparison.png
"""

import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re

# Dataset folder containing timing JSON files and where the plot will be saved
FOLDER = "Facebook"

# Plot style configuration per strategy key (matches the timing JSON naming convention)
STRATEGY_INFO = {
    "random":      {"name": "Random",      "color": "#808080", "marker": "o"},
    "degree":      {"name": "Degree",      "color": "#E9E910", "marker": "s"},
    "degree_h":    {"name": "Degree/1",    "color": "#00FFFF", "marker": "^"},
    "closeness":   {"name": "Closeness",   "color": "#15B01A", "marker": "D"},
    "closeness_h": {"name": "Closeness/1", "color": "#f97306", "marker": "v"},
    "brandes":     {"name": "Brandes",     "color": "#d62728", "marker": "P", "lw": 3, "ms": 11},
    "brandes_h":   {"name": "Brandes/1",   "color": "#0f13ff", "marker": "X", "lw": 3, "ms": 11},
}

print(f"\nScanning for timing files in: {os.path.abspath(FOLDER)}\n")

# Timing files are expected to follow: <K>_timing_<strategy>.json
pattern = os.path.join(FOLDER, "*_timing_*.json")
files = sorted(glob.glob(pattern))

if not files:
    print("No *_timing_*.json files found!")
    exit(1)

data = defaultdict(dict)

for filepath in files:
    filename = os.path.basename(filepath)

    # Extract K and strategy from the filename
    match = re.match(r"(\d+)_timing_([a-zA-Z_]+)\.json", filename)
    if not match:
        print(f"Skipped (bad name): {filename}")
        continue

    K, strategy = int(match.group(1)), match.group(2)
    if strategy not in STRATEGY_INFO:
        print(f"Unknown strategy: {strategy}")
        continue
    
    # Load timing JSON
    try:
        with open(filepath) as f:
            res = json.load(f)
    except Exception as e:
        print(f"Failed to read {filename}: {e}")
        continue
    
    # Brandes scripts store total time under a slightly different key
    if strategy.startswith("brandes"):
        t_total = res.get("T_total_brandes", None)
    else:
        t_total = res.get("T_total", None)

    if t_total is None:
        print(f"No suitable T_total in {filename}")
        continue

    data[strategy][K] = t_total
    print(f"{STRATEGY_INFO[strategy]['name']:<15} K={K:4d} → T_total={t_total:.6f} s")

# --------------------------
# Plotting
# --------------------------
plt.figure(figsize=(11, 7))
plt.rcParams.update({"font.size": 13})

all_ks = sorted({k for strat in data.values() for k in strat.keys()})
if not all_ks:
    print("No data to plot!")
    exit(1)

# Plot each strategy only if we have data for it
for strategy, info in STRATEGY_INFO.items():
    if strategy not in data:
        continue

    ks = sorted(data[strategy].keys())
    times = [data[strategy][k] for k in ks]

    plt.plot(
        ks,
        times,
        label=info["name"],
        color=info["color"],
        marker=info["marker"],
        markersize=info.get("ms", 9),
        linewidth=info.get("lw", 2.4),
        linestyle="-",
        markerfacecolor=info["color"],
        markeredgecolor="black",
        markeredgewidth=0.9,
    )

plt.xscale("log", base=10)
plt.xticks([1, 10, 100, 1000], ["$10^0$", "$10^1$", "$10^2$", "$10^3$"])
plt.gca().xaxis.set_minor_formatter(plt.NullFormatter())
plt.gca().xaxis.set_minor_locator(
    plt.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
)

plt.xlabel("Size of landmark set", fontsize=20)
plt.ylabel("Precomputation time T_total (s)", fontsize=20)

plt.title(
    f"Landmark-based Precomputation Time\n"
    f"Dataset: {os.path.basename(os.path.abspath(FOLDER))}",
    fontsize=22,
    pad=20,
)
plt.grid(True, which="major", ls="-", alpha=0.7)
plt.grid(True, which="minor", ls=":", alpha=0.4)
plt.legend(fontsize=16, loc="upper left")
plt.tight_layout()

# Save plot
png = os.path.join(FOLDER, "timing_comparison.png")
plt.savefig(png, dpi=350, bbox_inches="tight")
plt.show()
print("\nTiming plot saved successfully")

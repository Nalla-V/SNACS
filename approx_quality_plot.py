"""
Plots approximation quality (mean relative error) versus landmark budget K.
It scans a dataset folder for per-budget JSON outputs from evaluate.py, aggregates
the mean error per strategy, and saves a single comparison plot.

Reads (from FOLDER)
  - *_approx_quality_*.json

Writes (to FOLDER)
  - approx_quality_comparison.png
"""

import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re

# Dataset folder containing approximation-quality JSON files and where the plot will be saved
FOLDER = "Facebook"

# Plot style configuration per strategy key (must match the JSON filename suffixes)
STRATEGY_INFO = {
    "random":       {"name": "Random",        "color": "#808080", "marker": "o"},
    "degree":       {"name": "Degree",        "color": "#E9E910", "marker": "s"},
    "degree_h":     {"name": "Degree/1",      "color": "#00FFFF", "marker": "^"},
    "closeness":    {"name": "Closeness",     "color": "#15B01A", "marker": "D"},
    "closeness_h":  {"name": "Closeness/1",   "color": "#f97306", "marker": "v"},
    "brandes":      {"name": "Brandes",       "color": "#d62728", "marker": "P", "lw": 3, "ms": 11},
    "brandes_h":    {"name": "Brandes/1",     "color": "#0f13ff", "marker": "X", "lw": 3, "ms": 11},
}

print(f"\nScanning: {os.path.abspath(FOLDER)}\n")

# --------------------------
# Find and parse result files
# --------------------------
pattern = os.path.join(FOLDER, "*_approx_quality_*.json")
files = sorted(glob.glob(pattern))

if not files:
    print("No *_approx_quality_*.json files found!")
    exit(1)

data = defaultdict(dict)  

for filepath in files:
    filename = os.path.basename(filepath)

    # Extract K and strategy from the filename
    match = re.match(r"(\d+)_approx_quality_([a-zA-Z_]+)\.json", filename)
    if not match:
        print(f"Skipped (bad name): {filename}")
        continue

    K, strategy = int(match.group(1)), match.group(2)
    if strategy not in STRATEGY_INFO:
        print(f"Unknown strategy: {strategy}")
        continue
    
    # Load evaluate.py output (list of per-query entries)
    try:
        with open(filepath) as f:
            results = json.load(f)
    except:
        print(f"Failed to read: {filename}")
        continue
    
    # Only use queries with a valid positive true distance to avoid divide-by-zero error
    errors = [r["approx_error"] for r in results if r.get("true", 0) > 0]
    if not errors:
        print(f"No valid errors in {filename}")
        continue

    avg = sum(errors) / len(errors)
    data[strategy][K] = avg
    print(f"{STRATEGY_INFO[strategy]['name']:<15} K={K:4d} → {avg:.4f}")

# --------------------------
# Save Plot
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
    errs = [data[strategy][k] for k in ks]

    plt.plot(
        ks,
        errs,
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

plt.xlabel("Size of landmark set", fontsize=15)
plt.ylabel("Approximation error", fontsize=15)

plt.title(
    f"Landmark-based Distance Approximation Quality\n"
    f"Dataset: {os.path.basename(os.path.abspath(FOLDER))}",
    fontsize=16,
    pad=20,
)
plt.grid(True, which="major", ls="-", alpha=0.7)
plt.grid(True, which="minor", ls=":", alpha=0.4)
plt.ylim(0, 1.8)
plt.legend(fontsize=12, loc="upper right")
plt.tight_layout()

png = os.path.join(FOLDER, "approx_quality_comparison.png")
plt.savefig(png, dpi=350, bbox_inches="tight")
plt.show()
print(f"\nPlot saved successfully")
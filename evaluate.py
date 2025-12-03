"""
evaluate.py — FAST VERSION (uses precomputed true distances)
===========
Computes approximation quality for each query using precomputed true distances.

Output: approx_quality_<strategy>_<h_min>.json (dynamic)
"""

import os
import yaml
import pandas as pd
import pyarrow.parquet as pq
import json

with open("config.yaml") as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config["output_dir"]
DISTANCES_PARQUET = os.path.join(OUTPUT_DIR, "distances.parquet")
NODE_MAP_JSON = os.path.join(OUTPUT_DIR, "node_map.json")
TRUE_DIST_JSON = os.path.join(OUTPUT_DIR, "true_distances.json")

# DYNAMIC OUTPUT FILENAME
LM_SEL = config.get("lm_sel", "degree").lower()
H_MIN = config.get("h_min", 2)
LM = config.get("k", 4)

if LM_SEL in ["degree_h", "closeness_h"]:
    base_name = LM_SEL.replace("_h", "")
    OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"{LM}_approx_quality_{base_name}_{H_MIN}.json")
else:
    OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"{LM}_approx_quality_{LM_SEL}.json")

QUERIES_RAW = config["queries"]

print("\n# 4_evaluate.py: Running approximation evaluation (using precomputed true distances)")
print(f"   Strategy: {LM_SEL}, h_min: {H_MIN} → Output: {os.path.basename(OUTPUT_JSON)}")

# Load precomputed true distances
with open(TRUE_DIST_JSON) as f:
    true_dist_data = json.load(f)

# Map queries to true distances (assumes same order as QUERIES_RAW)
true_map = {tuple(d["query_labels"]): d["true"] for d in true_dist_data}

# Load label mappings (for IDs if needed)
with open(NODE_MAP_JSON) as f:
    id_to_label = json.load(f)
    label_to_id = {str(v): int(k) for k, v in id_to_label.items()}

# Load landmark distances
df_dist = pd.read_parquet(DISTANCES_PARQUET)

dist_map = {(int(r.node), int(r.landmark)): int(r.distance)
            for r in df_dist.itertuples()}

landmarks = sorted({r.landmark for r in df_dist.itertuples()})

# Distance estimation
def estimate(s, t):
    if s == t:
        return 0

    min_d = float("inf")
    for lm in landmarks:
        ds = dist_map.get((s, lm), float("inf"))
        dt = dist_map.get((t, lm), float("inf"))
        if ds < float("inf") and dt < float("inf"):
            min_d = min(min_d, ds + dt)

    return int(min_d) if min_d < float("inf") else -1

# Evaluate queries
print(f"   Evaluating {len(QUERIES_RAW)} queries")

results = []
serial_no = 1

for s_label, t_label in QUERIES_RAW:
    s_id = label_to_id.get(str(s_label))
    t_id = label_to_id.get(str(t_label))
    if s_id is None or t_id is None:
        continue

    est = estimate(s_id, t_id)
    true = true_map.get(tuple(sorted([s_label, t_label])), -1)  # sorted to match precompute

    if true > 0:
        approx_err = round(abs(est - true) / true, 4)
    else:
        approx_err = -1

    results.append({
        "query_serial_no": serial_no,
        "query_ids": [s_id, t_id],
        "query_labels": [s_label, t_label],
        "estimated": est,
        "true": true,
        "approx_error": approx_err
    })

    serial_no += 1

# Save output
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved approximation quality → {OUTPUT_JSON}")
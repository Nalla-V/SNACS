"""
Evaluates landmark-based distance estimates by comparing them with true shortest-path
distances for a fixed set of query pairs and saving the relative errors to disk. 

Inputs (read from output_dir in config.yaml)
  - config.yaml
  - distances.parquet
  - node_map.json
  - true_distances.json

Output (written to output_dir)
  - <k>_approx_quality_<strategy>.json

"""

import os
import yaml
import pandas as pd
import pyarrow.parquet as pq
import json


# Configuration 

with open("config.yaml") as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config["output_dir"]
DISTANCES_PARQUET = os.path.join(OUTPUT_DIR, "distances.parquet")
NODE_MAP_JSON = os.path.join(OUTPUT_DIR, "node_map.json")
TRUE_DIST_JSON = os.path.join(OUTPUT_DIR, "true_distances.json")

LM_SEL = config.get("lm_sel", "degree").lower()
H_MIN = config.get("h_min", 2)
LM = config.get("k", 4)

OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"{LM}_approx_quality_{LM_SEL}.json")

# Raw query pairs from config (original node IDs)
QUERIES_RAW = config["queries"]

print(f"\n# 4_evaluate.py: Evaluating approximation quality")
print(f"   Strategy: {LM_SEL}, h_min: {H_MIN} → {os.path.basename(OUTPUT_JSON)}")

# Load true distances
with open(TRUE_DIST_JSON) as f:
    true_dist_data = json.load(f)

true_map = {tuple(d["query_labels"]): d["true"] for d in true_dist_data}


# Load node ID mapping

with open(NODE_MAP_JSON, "r") as f:
    internal_to_orig_str = json.load(f)

orig_to_internal = {int(v): int(k) for k, v in internal_to_orig_str.items()}

# Load landmark distances
df_dist = pd.read_parquet(DISTANCES_PARQUET)

# Build a fast lookup: (node_id, landmark_id) -> distance
dist_map = {(int(r.node), int(r.landmark)): int(r.distance)
            for r in df_dist.itertuples()}

# Landmark set used for estimation
landmarks = sorted(df_dist["landmark"].unique())

# Estimate distance between two internal node IDs using landmark upper bound
def estimate(s_internal, t_internal):
    if s_internal == t_internal:
        return 0
    min_d = float("inf")
    for lm in landmarks:
        ds = dist_map.get((s_internal, lm), float("inf"))
        dt = dist_map.get((t_internal, lm), float("inf"))
        if ds < float("inf") and dt < float("inf"):
            min_d = min(min_d, ds + dt)
    return int(min_d) if min_d < float("inf") else -1

# Evaluate
print(f"   Evaluating {len(QUERIES_RAW)} queries...")
results = []
serial_no = 1

for s_label, t_label in QUERIES_RAW:
    s_id = orig_to_internal.get(s_label)
    t_id = orig_to_internal.get(t_label)
    if s_id is None or t_id is None:
        continue  

    est = estimate(s_id, t_id)
    true = true_map.get(tuple(sorted([str(s_label), str(t_label)])), -1)
    approx_err = round(abs(est - true) / true, 4) if true > 0 else -1

    results.append({
        "query_serial_no": serial_no,
        "query": [s_label, t_label],           
        "estimated": est,
        "true": true,
        "approx_error": approx_err
    })
    serial_no += 1

# Save results
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(f"   Saved → {OUTPUT_JSON}")
print(f"   Done! {len(results)} queries evaluated.\n")

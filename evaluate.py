"""
evaluate.py
===========
Computes approximation quality for each query:
- estimated distance
- true BFS distance
- approximation error abs(est - true) / true

Output: approx_quality_<strategy>_<h_min>.json (dynamic)
"""

import os
import yaml
import pandas as pd
import pyarrow.parquet as pq
import json
from collections import deque

with open("config.yaml") as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config["output_dir"]
EDGES_PARQUET = os.path.join(OUTPUT_DIR, "edges.parquet")
DISTANCES_PARQUET = os.path.join(OUTPUT_DIR, "distances.parquet")
NODE_MAP_JSON = os.path.join(OUTPUT_DIR, "node_map.json")

# === DYNAMIC OUTPUT FILENAME ===
LM_SEL = config.get("lm_sel", "degree").lower()
H_MIN = config.get("h_min", 2)

if LM_SEL in ["degree_h", "closeness_h"]:
    base_name = LM_SEL.replace("_h", "")
    OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"approx_quality_{base_name}_{H_MIN}.json")
else:
    OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"approx_quality_{LM_SEL}.json")

QUERIES_RAW = config["queries"]

print("\n# 4_evaluate.py: Running approximation evaluation")
print(f"   Strategy: {LM_SEL}, h_min: {H_MIN} → Output: {os.path.basename(OUTPUT_JSON)}")

# === 1. Load label mappings ===
with open(NODE_MAP_JSON) as f:
    id_to_label = json.load(f)
    label_to_id = {str(v): int(k) for k, v in id_to_label.items()}

# === 2. Map queries ===
QUERIES = []
for s_label, t_label in QUERIES_RAW:
    s_id = label_to_id.get(str(s_label))
    t_id = label_to_id.get(str(t_label))
    if s_id is None or t_id is None:
        print(f"   WARNING: Node {s_label} or {t_label} not in graph!")
        continue
    QUERIES.append((s_id, t_id))

print(f"   Queries mapped: {QUERIES}")

# === 3. Load graph adjacency ===
df_edges = pd.read_parquet(EDGES_PARQUET)
n = len(id_to_label)

adj = {i: [] for i in range(n)}
for _, row in df_edges.iterrows():
    u, v = int(row["source"]), int(row["target"])
    adj[u].append(v)
    adj[v].append(u)

# === 4. Load landmark distances ===
df_dist = pd.read_parquet(DISTANCES_PARQUET)

dist_map = {(int(r.node), int(r.landmark)): int(r.distance)
            for r in df_dist.itertuples()}

landmarks = sorted({r.landmark for r in df_dist.itertuples()})

# === 5. Distance estimation using triangle inequality ===
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

# === 6. Exact BFS ===
def exact_bfs(s, t):
    if s == t:
        return 0
    visited = {s: 0}
    q = deque([s])
    while q:
        u = q.popleft()
        if u == t:
            return visited[u]
        for v in adj[u]:
            if v not in visited:
                visited[v] = visited[u] + 1
                q.append(v)
    return -1

# === 7. Evaluate queries ===
print(f"   Evaluating {len(QUERIES)} queries...")

results = []
serial_no = 1

for s, t in QUERIES:
    est = estimate(s, t)
    true = exact_bfs(s, t)

    if true > 0:
        approx_err = round(abs(est - true) / true, 4)
    else:
        approx_err = -1  # undefined

    results.append({
        "query_serial_no": serial_no,
        "query_ids": [s, t],
        "query_labels": [id_to_label[str(s)], id_to_label[str(t)]],
        "estimated": est,
        "true": true,
        "approx_error": approx_err
    })

    serial_no += 1

# === 8. Save output with dynamic name ===
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved approximation quality → {OUTPUT_JSON}")

# Pretty print
print("\n" + " APPROXIMATION QUALITY ".center(60, "="))
print(f"{'Q#':<4} {'Query':<12} {'Est':>4} {'True':>5} {'Error':>8}")
print("-" * 60)
for r in results:
    q = f"{r['query_labels'][0]}-{r['query_labels'][1]}"
    err = f"{r['approx_error']:.4f}" if r['approx_error'] >= 0 else "N/A"
    print(f"{r['query_serial_no']:<4} {q:<12} {r['estimated']:>4} {r['true']:>5} {err:>8}")
print("=" * 60)
# inspect_query.py
# Enhanced version — finds exactly why a query gets a certain estimate

import os
import yaml
import json
import pandas as pd
from collections import deque
import sys

# ===================== CONFIG =====================
# Change these to match your setup
CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config["output_dir"]
EDGES_PARQUET = os.path.join(OUTPUT_DIR, "edges.parquet")
DISTANCES_PARQUET = os.path.join(OUTPUT_DIR, "distances.parquet")  
NODE_MAP_JSON = os.path.join(OUTPUT_DIR, "node_map.json")

# SET YOUR QUERY HERE (original labels!)
QUERY_LABEL_S = "61469"
QUERY_LABEL_T = "99828"
# =================================================

print(f"Inspecting query: label {QUERY_LABEL_S} -> {QUERY_LABEL_T}")

# 1. Load node mappings
with open(NODE_MAP_JSON) as f:
    id_to_label = json.load(f)
    label_to_id = {v: int(k) for k, v in id_to_label.items()}

s_label = str(QUERY_LABEL_S)
t_label = str(QUERY_LABEL_T)

if s_label not in label_to_id or t_label not in label_to_id:
    print("ERROR: One or both labels not found in node_map.json!")
    sys.exit(1)

s_id = label_to_id[s_label]
t_id = label_to_id[t_label]
print(f"           → internal IDs: {s_id} -> {t_id}")

# 2. Load precomputed landmark distances
print("Loading distances parquet (this may be large)...")
df_dist = pd.read_parquet(DISTANCES_PARQUET)

# Build dist_map: (node, landmark) → distance
dist_map = {}
for row in df_dist.itertuples():
    node = int(row.node)
    lm = int(row.landmark)
    dist = int(row.distance)
    dist_map[(node, lm)] = dist

landmarks = sorted({int(lm) for lm in df_dist["landmark"].unique()})
print(f"Loaded {len(landmarks)} landmarks: {landmarks}")

# 3. Per-landmark analysis
print("\nPer-landmark ds, dt, ds+dt (sorted by ds+dt):")
results = []

for lm in landmarks:
    ds = dist_map.get((s_id, lm), None)
    dt = dist_map.get((t_id, lm), None)

    if ds is None or dt is None:
        total = None
        status = "MISSING"
    else:
        total = ds + dt
        status = "OK"

    results.append((lm, ds, dt, total, status))

# Sort by total (ascending), then by landmark ID
results.sort(key=lambda x: (x[3] if x[3] is not None else 999, x[0]))

for lm, ds, dt, total, status in results:
    ds_str = str(ds) if ds is not None else "MISS"
    dt_str = str(dt) if dt is not None else "MISS"
    total_str = str(total) if total is not None else "—"
    marker = " ← BEST" if total == min(t for t in [r[3] for r in results] if t is not None) else ""
    print(f"  lm={lm:6d}  ds={ds_str:>4}  dt={dt_str:>4}  ds+dt={total_str:>4}  [{status}]{marker}")

best_total = min((t for t in [r[3] for r in results] if t is not None), default=None)
best_lms = [r[0] for r in results if r[3] == best_total]

print(f"\nBest ds+dt: {best_total}")
print(f"Best landmark(s): {best_lms}")

# 4. Load graph and reconstruct true shortest path
print("\nLoading edges to build adjacency (for exact path reconstruction)...")
df_edges = pd.read_parquet(EDGES_PARQUET)
adj = {}
for _, row in df_edges.iterrows():
    u, v = int(row["source"]), int(row["target"])
    adj.setdefault(u, []).append(v)
    adj.setdefault(v, []).append(u)

def bfs_path(src, dst):
    if src == dst:
        return [src], 0
    parent = {src: None}
    q = deque([src])
    found = False
    while q and not found:
        u = q.popleft()
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                q.append(v)
                if v == dst:
                    found = True
                    break
    if not found:
        return None, -1

    # reconstruct path
    path = []
    current = dst
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path, len(path) - 1

true_path_ids, true_dist = bfs_path(s_id, t_id)
if true_path_ids is None:
    print("No path exists between nodes!")
    sys.exit(1)

print(f"True shortest distance: {true_dist}")
print(f"One shortest path (ids): {true_path_ids}")
print(f"One shortest path (labels): {[id_to_label[str(nid)] for nid in true_path_ids]}")

# 5. Check which landmarks are on the true path
print("\nLandmarks on true shortest path:")
on_path = []
for lm in landmarks:
    if lm in true_path_ids:
        pos = true_path_ids.index(lm)
        dist_s_lm = pos
        dist_lm_t = true_dist - pos
        print(f"  lm={lm} at position {pos} → ds={dist_s_lm}, dt={dist_lm_t}, ds+dt={dist_s_lm + dist_lm_t} (perfect bound!)")
        on_path.append(lm)

if not on_path:
    print("  None! ← This explains overestimation")

# 6. Final summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Query:          {QUERY_LABEL_S} → {QUERY_LABEL_T}")
print(f"True distance:  {true_dist}")
print(f"Estimated:      {best_total}")
print(f"Error:          {abs(best_total - true_dist)/true_dist:.4f} (rel)")
print(f"Best landmark:  {best_lms}")
print(f"On true path?   {'Yes' if any(lm in true_path_ids for lm in best_lms) else 'No'}")
if on_path:
    print(f"Landmarks on path: {on_path} ← these should give perfect estimates!")
else:
    print("No landmark on true path → overestimation expected with small K")
print("="*70)
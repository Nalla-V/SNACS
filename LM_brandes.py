#!/usr/bin/env python3
"""
landmark_strategy_brandes.py — with timing instrumentation that isolates:
  - T_LM_brandes      (landmark-selection time only)
  - T_precompute_brandes  (BFS + Brandes accumulation time only)
  - T_total_brandes   (sum of the two)
  - bfs_times         (per-landmark BFS durations)

This excludes graph-loading time entirely.
"""

import os
import json
import yaml
import time
import pyarrow.parquet as pq
import pyarrow as pa
from collections import deque
import pandas as pd

# --------------------------
# Load config
# --------------------------
with open("config.yaml") as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config["output_dir"]
K = int(config["k"])
H_MIN = int(config.get("h_min", 2))

EDGES_PARQUET = os.path.join(OUTPUT_DIR, "edges.parquet")
NODE_MAP_JSON = os.path.join(OUTPUT_DIR, "node_map.json")

LANDMARKS_JSON = os.path.join(OUTPUT_DIR, "landmarks.json")
DISTANCES_PQ = os.path.join(OUTPUT_DIR, "distances.parquet")
TIMING_JSON = os.path.join(OUTPUT_DIR, "timing_brandes.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n# landmark_strategy_brandes.py starting")
print(f"K={K}, h_min={H_MIN}")
print(f"Output → {OUTPUT_DIR}\n")

# --------------------------
# LOAD NODE MAP
# --------------------------
with open(NODE_MAP_JSON, "r") as f:
    id_to_node = json.load(f)
n = len(id_to_node)
print(f"Loaded node_map.json → {n} nodes")

# --------------------------
# LOAD EDGES + BUILD ADJ
# --------------------------
print("Loading edges parquet + building adjacency …")
table = pq.read_table(EDGES_PARQUET, columns=["source", "target"])
src_col = table.column("source").to_numpy()
tgt_col = table.column("target").to_numpy()

adj = [[] for _ in range(n)]
for u, v in zip(src_col, tgt_col):
    u = int(u); v = int(v)
    adj[u].append(v)
    adj[v].append(u)

print("Adjacency built.\n")

# =====================================================
# Brandes-supporting helper functions (same as before)
# =====================================================

def bfs_brandes(source):
    dist = [-1] * n
    pred = [[] for _ in range(n)]
    sigma = [0] * n
    stack = []

    dist[source] = 0
    sigma[source] = 1
    q = deque([source])

    while q:
        v = q.popleft()
        stack.append(v)
        for w in adj[v]:
            if dist[w] < 0:
                dist[w] = dist[v] + 1
                q.append(w)
            if dist[w] == dist[v] + 1:
                sigma[w] += sigma[v]
                pred[w].append(v)

    return dist, stack, pred, sigma


def brandes_accumulate(stack, pred, sigma, source):
    delta = [0.0] * n
    while stack:
        w = stack.pop()
        coeff = 1.0 + delta[w]
        for v in pred[w]:
            if sigma[w] != 0:
                delta[v] += (sigma[v] / sigma[w]) * coeff
    return delta


def nodes_within_hops(start, h):
    if h <= 0:
        return {start}
    dist = [-1] * n
    q = deque([start])
    dist[start] = 0
    out = {start}
    while q:
        u = q.popleft()
        if dist[u] >= h:
            continue
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                if dist[v] <= h:
                    out.add(v)
                q.append(v)
    return out

# =====================================================
# TIMING BLOCK BEGINS HERE — excludes all loading above
# =====================================================

T_LM = 0.0                     # landmark selection time
T_PRE = 0.0                    # BFS+Brandes time
bfs_times = []                 # list of tuples: (landmark, time)

records = []
participation = [0.0] * n
deg = [len(adj[i]) for i in range(n)]

# --------------------------
# Step 1 — Select L1
# --------------------------
t0_LM = time.time()
L1 = max(range(n), key=lambda x: deg[x])
landmarks = [int(L1)]
t1_LM = time.time()
T_LM += (t1_LM - t0_LM)

print(f"Selected L1 = {L1} (degree={deg[L1]})")

# --------------------------
# Run BFS+Brandes from L1
# --------------------------
def run_from_landmark(lm):
    """Perform BFS + Brandes accumulate and update global participation."""
    dist, stack, pred, sigma = bfs_brandes(lm)

    # record distances
    for node_idx, d in enumerate(dist):
        if d != -1:
            records.append({
                "node": int(node_idx),
                "landmark": int(lm),
                "distance": int(d)
            })

    stack_copy = list(stack)
    delta = brandes_accumulate(stack_copy, pred, sigma, lm)

    # update participation
    for i in range(n):
        if i != lm:
            participation[i] += delta[i]

# BFS time for L1
t0_B = time.time()
run_from_landmark(L1)
t1_B = time.time()
dt = t1_B - t0_B
T_PRE += dt
bfs_times.append((int(L1), dt))

# --------------------------
# SELECT NEXT LANDMARKS
# --------------------------
for idx in range(2, K + 1):

    # --- timing: landmark-selection only ---
    t0_LM = time.time()

    # forbidden nodes = within H_MIN hops of already selected
    forbidden = set()
    for lm in landmarks:
        forbidden |= nodes_within_hops(lm, H_MIN)

    best_node = None
    best_score = -1.0

    for v in range(n):
        if v in forbidden: continue
        if v in landmarks: continue
        score = participation[v]
        if score > best_score or (score == best_score and (best_node is None or deg[v] > deg[best_node])): 
            best_node = v
            best_score = score

    # fallback if participation all zero
    if best_node is None or best_score <= 0:
        print("Warning: falling back to degree-based selection")
        best_deg = -1
        best_node = None
        for v in range(n):
            if v in forbidden or v in landmarks:
                continue
            if deg[v] > best_deg:
                best_node = v
                best_deg = deg[v]

    landmarks.append(int(best_node))

    t1_LM = time.time()
    T_LM += (t1_LM - t0_LM)

    print(f"Selected L{idx} = {best_node} (score={best_score:.3f}, deg={deg[best_node]})")

    # --- timing: BFS+Brandes only ---
    t0_B = time.time()
    run_from_landmark(best_node)
    t1_B = time.time()
    dt = t1_B - t0_B

    T_PRE += dt
    bfs_times.append((int(best_node), dt))

# =====================================================
# SAVE OUTPUTS
# =====================================================

print("\nSaving outputs…")

# landmarks
with open(LANDMARKS_JSON, "w") as f:
    json.dump(landmarks, f, indent=2)

# distances
df_out = pd.DataFrame.from_records(records)
df_out = df_out.astype({"node": "int32", "landmark": "int32", "distance": "int32"})
pq.write_table(pa.Table.from_pandas(df_out), DISTANCES_PQ)

# timing
T_TOTAL = T_LM + T_PRE
timing_data = {
    "T_LM_brandes": T_LM,
    "T_precompute_brandes": T_PRE,
    "T_total_brandes": T_TOTAL,
    "bfs_times": bfs_times
}
with open(TIMING_JSON, "w") as f:
    json.dump(timing_data, f, indent=2)

print("Saved:")
print(f"  Landmarks → {LANDMARKS_JSON}")
print(f"  Distances → {DISTANCES_PQ}")
print(f"  Timing    → {TIMING_JSON}")

print("\nCompleted BRANDES strategy.\n")

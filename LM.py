"""
Selects landmarks using simple baseline strategies (random/degree/sampled closeness) and builds
a landmark distance index. It selects K landmarks, runs BFS from each landmark to all nodes,
and writes the landmark-to-node distances to a Parquet file.

Reads (from output_dir in config.yaml)
  - config.yaml
  - edges.parquet
  - node_map.json

Writes (to output_dir)
  - landmarks.json
  - distances.parquet
  - <k>_timing_<lm_sel>.json
"""

import os
import time
import json
import yaml
import random
from collections import deque
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd

# --------------------------
# Configuration 
# --------------------------
with open("config.yaml") as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config["output_dir"]
K = int(config["k"])
H_MIN = int(config.get("h_min", 0))
LM_SEL = config.get("lm_sel", "degree").lower()
LM = config.get("k", 4)
CLOSENESS_SAMPLES = int(config.get("closeness_samples", 200))
RANDOM_SEED = int(config.get("random_seed", 42))

EDGES_PARQUET = os.path.join(OUTPUT_DIR, "edges.parquet")
NODE_MAP_JSON = os.path.join(OUTPUT_DIR, "node_map.json")
LANDMARKS_JSON = os.path.join(OUTPUT_DIR, "landmarks.json")
DISTANCES_PARQUET = os.path.join(OUTPUT_DIR, "distances.parquet")

TIMING_JSON = os.path.join(OUTPUT_DIR, f"{LM}_timing_{LM_SEL}.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n# Landmark selection (sampled) starting — method={LM_SEL}, K={K}, h_min={H_MIN}")
print(f"Output → {OUTPUT_DIR}")

# --------------------------
# Load node map
# --------------------------
with open(NODE_MAP_JSON, "r") as f:
    id_to_label = json.load(f)
n = len(id_to_label)
print(f"Loaded node_map.json: {n} nodes (0..{n-1})")


# Build adjacency list 
t0 = time.time()
pf = pq.ParquetFile(EDGES_PARQUET)

adj = [[] for _ in range(n)]
for batch in pf.iter_batches(batch_size=1_000_000, columns=["source", "target"]):
    d = batch.to_pydict()
    srcs = d["source"]
    tgts = d["target"]
    for u, v in zip(srcs, tgts):
        u = int(u); v = int(v)
        adj[u].append(v)
        adj[v].append(u)

t1 = time.time()


# BFS helpers
def bfs_full(source):
    """Standard BFS returning distances from `source` to all nodes (-1 if unreachable)."""

    dist = [-1] * n
    q = deque([source])
    dist[source] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def nodes_within_hops(start, h):
    """Return set of nodes within <= h hops from `start` (including `start`)."""

    if h <= 0:
        return {start}
    dist = [-1] * n
    q = deque([start])
    dist[start] = 0
    res = {start}
    while q:
        u = q.popleft()
        if dist[u] >= h: continue
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                if dist[v] <= h: res.add(v)
                q.append(v)
    return res

# --------------------------
# Random sampling for Closeness
# --------------------------
random.seed(RANDOM_SEED)

def sample_pivots(sample_size):
    sample_size = min(sample_size, n)
    return random.sample(range(n), sample_size)

# Landmark selection timing starts here

t_lm_start = time.time()
landmarks = []

# Degree strategies

def compute_degrees_streaming():
    deg = [0] * n
    pf = pq.ParquetFile(EDGES_PARQUET)
    seen_edges = set()  
    for batch in pf.iter_batches(batch_size=1_000_000, columns=["source", "target"]):
        d = batch.to_pydict()
        srcs, tgts = d["source"], d["target"]
        for u, v in zip(srcs, tgts):
            u, v = int(u), int(v)
            edge = tuple(sorted([u, v]))  
            if edge not in seen_edges:
                seen_edges.add(edge)
                deg[u] += 1
                deg[v] += 1
    return deg


if LM_SEL == "random":
    # Uniform random landmarks

    print("Selecting Random Landmarks ")
    landmarks = random.sample(range(n), K)

elif LM_SEL == "degree":
    # Top-K nodes by degree

    print("Computing degree ")
    deg = compute_degrees_streaming()
    landmarks = sorted(range(n), key=lambda x: deg[x], reverse=True)[:K]

elif LM_SEL == "degree_h":
    # Degree ranking with hop-exclusion

    print("Computing degree for degree_h")
    deg = compute_degrees_streaming()

    candidates = sorted(range(n), key=lambda x: deg[x], reverse=True)
    selected = []
    forbidden = set()

    for node in candidates:
        if node in forbidden: continue
        selected.append(node)
        if len(selected) == K: break
        if H_MIN > 0:
            forbidden.update(nodes_within_hops(node, H_MIN))

    landmarks = selected


# Sampled Closeness strategy

elif LM_SEL == "closeness":
    print("Computing closeness on Sample size ")
    S = CLOSENESS_SAMPLES
    pivots = sample_pivots(S)
    farness = [0.0] * n
    reach_counts = [0] * n

    for p in pivots:
        dist = bfs_full(p)
        for v, d in enumerate(dist):
            if d != -1:
                farness[v] += d
                reach_counts[v] += 1

    import math
    avg_farness = [
        (farness[v] / reach_counts[v]) if reach_counts[v] > 0 else math.inf
        for v in range(n)
    ]

    landmarks = sorted(range(n), key=lambda x: avg_farness[x])[:K]

elif LM_SEL == "closeness_h":
    # Sampled closeness with hop-exclusion

    print("Computing closeness on Sample size ")
    S = CLOSENESS_SAMPLES
    pivots = sample_pivots(S)
    farness = [0.0] * n
    reach_counts = [0] * n

    for p in pivots:
        dist = bfs_full(p)
        for v, d in enumerate(dist):
            if d != -1:
                farness[v] += d
                reach_counts[v] += 1

    import math
    avg_farness = [
        (farness[v] / reach_counts[v]) if reach_counts[v] > 0 else math.inf
        for v in range(n)
    ]

    candidates = sorted(range(n), key=lambda x: avg_farness[x])
    selected = []
    forbidden = set()

    for node in candidates:
        if node in forbidden: continue
        selected.append(node)
        if len(selected) == K: break
        if H_MIN > 0:
            forbidden.update(nodes_within_hops(node, H_MIN))

    landmarks = selected

else:
    raise ValueError(f"Unknown lm_sel: {LM_SEL}")

t_lm_end = time.time()
T_LM = t_lm_end - t_lm_start
    
print(f"T_LM = {T_LM:.3f}s\n")

# Save selected Landmarks
with open(LANDMARKS_JSON, "w") as f:
    json.dump(landmarks, f, indent=2)


# Precompute BFS from landmarks

print("Precomputing distance using BFS from selected landmarks to all other nodes")
records_node = []
records_lm = []
records_dist = []
T_pre = 0.0
bfs_times = []

# For each landmark, store (node, landmark, distance) for all reachable nodes
for lm in landmarks:
    t0 = time.time()
    dist = bfs_full(lm)
    t1 = time.time()

    bfs_times.append({"landmark": lm, "time": t1 - t0})
    T_pre += (t1 - t0)

    for v, d in enumerate(dist):
        if d != -1:
            records_node.append(v)
            records_lm.append(lm)
            records_dist.append(d)

print("\nWriting distances in distances.parquet file")
df_out = pd.DataFrame({
    "node": records_node,
    "landmark": records_lm,
    "distance": records_dist
}).astype({"node":"int32","landmark":"int32","distance":"int32"})

pq.write_table(pa.Table.from_pandas(df_out), DISTANCES_PARQUET)

# Save timing
with open(TIMING_JSON, "w") as f:
    json.dump({
        "T_LM": T_LM,
        "T_precompute": T_pre,
        "T_total": T_LM + T_pre,
        "bfs_times": bfs_times,
        "landmarks": landmarks
    }, f, indent=2)

print(f"Done — timings saved to {TIMING_JSON}")

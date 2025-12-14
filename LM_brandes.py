"""
Selects landmarks using a Brandes-inspired participation score and builds a landmark distance index.
It iteratively picks landmarks, runs BFS to compute distances, accumulates dependency scores, and
writes landmark-to-node distances incrementally to a Parquet file.

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
import json
import yaml
import time
from collections import deque
import pyarrow.parquet as pq
import pyarrow as pa

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

EDGES_PARQUET = os.path.join(OUTPUT_DIR, "edges.parquet")
NODE_MAP_JSON = os.path.join(OUTPUT_DIR, "node_map.json")

LANDMARKS_JSON = os.path.join(OUTPUT_DIR, "landmarks.json")
DISTANCES_PQ = os.path.join(OUTPUT_DIR, "distances.parquet")

# dynamic timing file
TIMING_JSON = os.path.join(OUTPUT_DIR, f"{LM}_timing_{LM_SEL}.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------
# Load node map (dense ids)
# --------------------------
with open(NODE_MAP_JSON, "r") as f:
    id_to_node = json.load(f)  
n = len(id_to_node)
print(f"Loaded node_map.json → {n} nodes")

# --------------------------
# Build adjacency list by streaming Parquet
# --------------------------
t_build_start = time.time()

pf = pq.ParquetFile(EDGES_PARQUET)
adj = [[] for _ in range(n)]
rows_seen = 0
batch_no = 0

for batch in pf.iter_batches(batch_size=1_000_000, columns=["source", "target"]):
    batch_no += 1
    d = batch.to_pydict()
    srcs = d["source"]
    tgts = d["target"]
    for u, v in zip(srcs, tgts):
        u = int(u); v = int(v)
        adj[u].append(v)
        adj[v].append(u)
        rows_seen += 1

t_build_end = time.time()
print(f"Adjacency built: {n} nodes from {rows_seen} edges\n")

# --------------------------
# Basic helpers (BFS + Brandes accumulation)
# --------------------------
def bfs_brandes(source):
    """
    BFS from `source` and collect the structures needed for Brandes-style accumulation.

    Returns:
      dist  : list[int] shortest-path distances from source (-1 if unreachable)
      stack : list[int] nodes in BFS visitation order (used for reverse accumulation)
      pred  : list[list[int]] predecessors on shortest paths from source
      sigma : list[int] number of shortest paths from source to each node
    """
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
        dv = dist[v]
        for w in adj[v]:
            if dist[w] < 0:
                dist[w] = dv + 1
                q.append(w)
            if dist[w] == dv + 1:
                # shortest path via v
                sigma[w] += sigma[v]
                pred[w].append(v)

    return dist, stack, pred, sigma

def brandes_accumulate(stack, pred, sigma, source):
    """
    Reverse dependency accumulation (Brandes recurrence).

    Returns:
      delta : list[float] where delta[v] is the dependency contribution from `source`.
    """
    delta = [0.0] * n
    # process nodes in reverse order of distance (stack is in non-decreasing dist)
    while stack:
        w = stack.pop()
        coeff = 1.0 + delta[w]
        for v in pred[w]:
            if sigma[w] != 0:
                delta[v] += (sigma[v] / sigma[w]) * coeff
        # source is not accumulated into participation
    return delta

def nodes_within_hops(start, h):
    """Return set of nodes within <= h hops from start (including start)."""
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
        du = dist[u]
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = du + 1
                if dist[v] <= h:
                    out.add(v)
                q.append(v)
    return out

# --------------------------
# Degree heuristic (used for initial seed and tie-breaking)
# --------------------------
deg = [len(adj[i]) for i in range(n)]

# --------------------------
# Prepare Parquet writer for distances (incremental)
# --------------------------
schema = pa.schema([
    pa.field("node", pa.int32()),
    pa.field("landmark", pa.int32()),
    pa.field("distance", pa.int32())
])

if os.path.exists(DISTANCES_PQ):
    os.remove(DISTANCES_PQ)
parquet_writer = None 

def write_distances_batch(nodes, lm_id, dists):
    """
    Append a batch of distances for a single landmark to distances.parquet.

    nodes: list of internal node ids
    lm_id: internal id of the landmark
    dists: corresponding distances aligned with `nodes`
    """
    global parquet_writer
    if not nodes:
        return
    tbl = pa.Table.from_pydict({
        "node": pa.array(nodes, type=pa.int32()),
        "landmark": pa.array([int(lm_id)] * len(nodes), type=pa.int32()),
        "distance": pa.array(dists, type=pa.int32())
    }, schema=schema)
    if parquet_writer is None:
        parquet_writer = pq.ParquetWriter(DISTANCES_PQ, schema)
    parquet_writer.write_table(tbl)

# --------------------------
# Main algorithm: iterative landmark selection + index construction
# --------------------------
print("Select landmark iteratively and precompute distance\n")

T_LM = 0.0
T_PRE = 0.0
bfs_times = []

participation = [0.0] * n
landmarks = []

# Step 1: choose the first landmark by highest degree
t0 = time.time()
L1 = max(range(n), key=lambda x: deg[x])
landmarks.append(int(L1))
t1 = time.time()
T_LM += (t1 - t0)

def run_from_landmark(lm):
    """
    Run BFS + Brandes accumulation from `lm`, and write distances to disk.

    Returns:
      delta      : dependency contributions from this source
      reachable  : number of reachable nodes (should be n for connected graphs)
      elapsed    : wall-clock time for this BFS + accumulation + write
    """
    t_start = time.time()
    dist, stack, pred, sigma = bfs_brandes(lm)

    # Collect distances for reachable nodes
    nodes_batch = []
    dists_batch = []
    reachable = 0
    for node_idx, d in enumerate(dist):
        if d != -1:
            nodes_batch.append(int(node_idx))
            dists_batch.append(int(d))
            reachable += 1

    write_distances_batch(nodes_batch, lm, dists_batch)

    # Accumulate dependency scores (Brandes reverse pass)
    stack_copy = list(stack)
    delta = brandes_accumulate(stack_copy, pred, sigma, lm)

    elapsed = time.time() - t_start
    return delta, reachable, elapsed

# Process the first landmark
delta, reachable, elapsed = run_from_landmark(L1)

for i in range(n):
    if i == L1: continue
    participation[i] += delta[i]
T_PRE += elapsed
bfs_times.append({"landmark": int(L1), "time": elapsed, "reachable": int(reachable)})

# Select remaining landmarks
for idx in range(2, K + 1):
    t0 = time.time()

    # Hop-exclusion: avoid selecting landmarks too close to any already selected landmark
    forbidden = set()
    for lm in landmarks:
        forbidden |= nodes_within_hops(lm, H_MIN)

    best_node = None
    best_score = -1.0

    # Choose the candidate with highest participation score (tie-break by degree)
    for v in range(n):
        if v in forbidden: 
            continue
        if v in landmarks:
            continue
        sc = participation[v]
        if sc > best_score or (sc == best_score and (best_node is None or deg[v] > deg[best_node])):
            best_score = sc
            best_node = v

    # Fallback if participation signal is not informative
    if best_node is None or best_score <= 0:
        print("  Warning: falling back to degree-based selection")
        best_node = None
        best_deg = -1
        for v in range(n):
            if v in forbidden or v in landmarks:
                continue
            if deg[v] > best_deg:
                best_node = v
                best_deg = deg[v]
        if best_node is None:
            raise RuntimeError("Unable to find a new landmark")

    landmarks.append(int(best_node))
    t1 = time.time()
    T_LM += (t1 - t0)

    # Compute and store distances for the new landmark, then update participation
    delta, reachable, elapsed = run_from_landmark(best_node)
   
    for i in range(n):
        if i == best_node: continue
        participation[i] += delta[i]
    T_PRE += elapsed
    bfs_times.append({"landmark": int(best_node), "time": elapsed, "reachable": int(reachable)})

if 'parquet_writer' in globals() and parquet_writer is not None:
    parquet_writer.close()

# --------------------------
# Save outputs
# --------------------------
with open(LANDMARKS_JSON, "w") as f:
    json.dump(landmarks, f, indent=2)

T_TOTAL = T_LM + T_PRE
timing_data = {
    "T_LM_brandes": T_LM,
    "T_precompute_brandes": T_PRE,
    "T_total_brandes": T_TOTAL,
    "bfs_times": bfs_times,
    "landmarks": landmarks
}
with open(TIMING_JSON, "w") as f:
    json.dump(timing_data, f, indent=2)

print("Saved:")
print(f"  Landmarks → {LANDMARKS_JSON}")
print(f"  Distances → {DISTANCES_PQ}")
print(f"  Timing    → {TIMING_JSON}")
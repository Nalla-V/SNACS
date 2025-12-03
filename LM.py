import os
import time
import yaml
import json
import pandas as pd
from collections import deque
import pyarrow.parquet as pq
import pyarrow as pa

# --------------------------
# Load config (not timed)
# --------------------------
with open("config.yaml") as f:
    config = yaml.safe_load(f)

OUTPUT_DIR  = config["output_dir"]
K           = config["k"]
H_MIN       = config.get("h_min", 2)
LM_SEL      = config.get("lm_sel", "degree").lower()

EDGES_PARQUET     = os.path.join(OUTPUT_DIR, "edges.parquet")
DISTANCES_PARQUET = os.path.join(OUTPUT_DIR, "distances.parquet")
LANDMARKS_JSON    = os.path.join(OUTPUT_DIR, "landmarks.json")

# --------------------------
# DYNAMIC TIMING FILENAME
# --------------------------
if LM_SEL in ["degree_h", "closeness_h"]:
    # Replace '_h' with actual h_min value
    base_name = LM_SEL.replace("_h", "")
    TIMING_JSON = os.path.join(OUTPUT_DIR, f"timing_{base_name}_{H_MIN}.json")
else:
    TIMING_JSON = os.path.join(OUTPUT_DIR, f"timing_{LM_SEL}.json")

print(f"\n# OLD STRATEGY RUN: lm_sel={LM_SEL}, k={K}, h_min={H_MIN}")

# --------------------------
# Load graph (not timed)
# --------------------------
df_edges = pd.read_parquet(EDGES_PARQUET)

all_nodes = sorted(set(df_edges["source"]) | set(df_edges["target"]))
n = len(all_nodes)

# adjacency list
adj = {i: [] for i in range(n)}
for _, row in df_edges.iterrows():
    u, v = int(row["source"]), int(row["target"])
    adj[u].append(v)
    adj[v].append(u)

print(f"  Loaded graph: {n} nodes, {len(df_edges)} edges")

# --------------------------
# Timing containers
# --------------------------
T_LM_old = 0
T_precompute_old = 0
T_BFS_list = []

# ==========================================================
#                 LANDMARK SELECTION TIMING
# ==========================================================
t_lm_start = time.time()

landmarks = []

if LM_SEL == "random":
    import random
    landmarks = random.sample(range(n), K)

elif LM_SEL == "degree":
    degrees = pd.concat([
        df_edges['source'].value_counts(),
        df_edges['target'].value_counts()
    ]).groupby(level=0).sum()
    landmarks = degrees.nlargest(K).index.astype(int).tolist()

elif LM_SEL == "degree_h":
    degrees = pd.concat([
        df_edges['source'].value_counts(),
        df_edges['target'].value_counts()
    ]).groupby(level=0).sum()

    candidates = degrees.index.astype(int).tolist()
    candidates.sort(key=lambda x: degrees[x], reverse=True)

    selected = []
    forbidden = set()

    def nodes_within(start, h):
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        result = {start}
        while q:
            u = q.popleft()
            if dist[u] >= h:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    result.add(v)
                    q.append(v)
        return result

    for node in candidates:
        if node in forbidden:
            continue
        selected.append(node)
        if len(selected) == K:
            break
        forbidden.update(nodes_within(node, H_MIN))

    landmarks = selected

elif LM_SEL == "closeness":
    avg_dist = {}
    for node in range(n):
        dist = [-1] * n
        dist[node] = 0
        q = deque([node])
        dist_sum = 0
        reachable = 0

        while q:
            u = q.popleft()
            dist_sum += dist[u]
            reachable += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)

        avg_dist[node] = dist_sum / max(1, (reachable - 1))

    landmarks = sorted(avg_dist, key=avg_dist.get)[:K]

elif LM_SEL == "closeness_h":
    print("  Computing closeness centrality with hop separation (closeness_h)...")

    # Step 1: Compute farness for every node
    farness = {}
    for node in range(n):
        if node % 1000 == 0 or node == n-1:
            print(f"    closeness BFS {node+1}/{n}")
        dist = [-1] * n
        dist[node] = 0
        q = deque([node])
        total_dist = 0
        reachable = 0
        while q:
            u = q.popleft()
            total_dist += dist[u]
            reachable += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        farness[node] = total_dist if reachable > 1 else float('inf')

    # Step 2: Sort by farness (lower = better closeness)
    candidates = sorted(farness.items(), key=lambda x: x[1])

    selected = []
    forbidden = set()

    def nodes_within(start, h):
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        result = {start}
        while q:
            u = q.popleft()
            if dist[u] >= h:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] <= h:
                        result.add(v)
                    q.append(v)
        return result

    for node, f in candidates:
        if node in forbidden:
            continue
        selected.append(node)
        if len(selected) == K:
            break
        forbidden.update(nodes_within(node, H_MIN))

    landmarks = selected
    print(f"  Selected closeness_h landmarks: {landmarks}")

else:
    raise ValueError(f"Unknown selection method: {LM_SEL}")

t_lm_end = time.time()
T_LM_old = t_lm_end - t_lm_start

print(f"  Selected landmarks: {landmarks}")
print(f"  T_LM_old = {T_LM_old:.6f} sec")

# Save LM
with open(LANDMARKS_JSON, "w") as f:
    json.dump(landmarks, f)

# ==========================================================
#                 BFS PRECOMPUTATION TIMING
# ==========================================================
records = []

t_pre_start = time.time()

print("\n  Running BFS for selected landmarks...")

for lm in landmarks:
    t_bfs_start = time.time()

    dist = [-1] * n
    dist[lm] = 0
    q = deque([lm])

    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)

    # store results
    for v in range(n):
        if dist[v] != -1:
            records.append({"node": v, "landmark": lm, "distance": dist[v]})

    t_bfs_end = time.time()
    bfs_time = t_bfs_end - t_bfs_start
    T_BFS_list.append({"landmark": lm, "time": bfs_time})

    print(f"    LM {lm}: BFS = {bfs_time:.6f} sec")

t_pre_end = time.time()
T_precompute_old = t_pre_end - t_pre_start

# T_total = selection + BFS
T_total_old = T_LM_old + T_precompute_old

# save distances
df_out = pd.DataFrame(records)
pq.write_table(pa.Table.from_pandas(df_out), DISTANCES_PARQUET)

print(f"\n  Saved embeddings → {DISTANCES_PARQUET}")

# ==========================================================
# Save timing JSON with dynamic name
# ==========================================================
timing_data = {
    "T_LM_old": T_LM_old,
    "T_precompute_old": T_precompute_old,
    "T_total_old": T_total_old,
    "T_BFS_each": T_BFS_list,
    "landmarks": landmarks
}

with open(TIMING_JSON, "w") as f:
    json.dump(timing_data, f, indent=2)

print(f"\n  T_total_old = {T_total_old:.6f} sec")
print(f"  Timing saved → {TIMING_JSON}")
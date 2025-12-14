"""
Computes exact shortest-path distances for the configured query pairs.
It maps query node labels to internal IDs, groups queries by source, runs BFS per unique
source with early stopping, and saves the true distances for later evaluation.

Reads (from output_dir in config.yaml)
  - config.yaml (queries, output_dir)
  - edges.parquet
  - node_map.json

Writes (to output_dir)
  - true_distances.json
"""

import os
import time
import json
import yaml
from collections import deque, defaultdict
import pyarrow.parquet as pq

# --------------------------
# Configurations
# --------------------------
with open("config.yaml") as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config["output_dir"]
EDGES_PARQUET = os.path.join(OUTPUT_DIR, "edges.parquet")
NODE_MAP_JSON = os.path.join(OUTPUT_DIR, "node_map.json")
TRUE_DIST_JSON = os.path.join(OUTPUT_DIR, "true_distances.json")
QUERIES_RAW = config.get("queries", [])

if not QUERIES_RAW:
    raise SystemExit("No queries found in config.yaml under 'queries' key.")

print(f"\n# precompute_true_distances.py: Computing exact distances for {len(QUERIES_RAW)} queries")

# --------------------------
# Load node map 
# --------------------------
with open(NODE_MAP_JSON, "r") as f:
    internal_to_orig_str = json.load(f)

internal_to_orig = {int(k): int(v) for k, v in internal_to_orig_str.items()}
orig_to_internal = {orig: internal for internal, orig in internal_to_orig.items()}

n_nodes = len(internal_to_orig)
print(f"   Loaded {n_nodes:,} nodes")

# --------------------------
# Convert queries to internal IDs
# --------------------------
queries = []
for idx, (s_label, t_label) in enumerate(QUERIES_RAW, start=1):
    s_int = orig_to_internal.get(s_label)
    t_int = orig_to_internal.get(t_label)
    if s_int is None or t_int is None:
        continue
    queries.append((idx, s_int, t_int, str(s_label), str(t_label)))

if not queries:
    raise SystemExit("No valid queries after mapping!")

# Group queries by source so we can reuse one BFS per unique source node
queries_by_source = defaultdict(list)
for serial, s_int, t_int, s_lab, t_lab in queries:
    queries_by_source[s_int].append((serial, t_int, s_lab, t_lab))

unique_sources = sorted(queries_by_source.keys())
print(f"   Valid queries: {len(queries)}, unique sources: {len(unique_sources)}")

# --------------------------
# Build adjacency list
# --------------------------
print("   Building adjacency list")
t0 = time.perf_counter()

adj = [[] for _ in range(n_nodes)]
pf = pq.ParquetFile(EDGES_PARQUET)

for batch in pf.iter_batches(batch_size=1_000_000, columns=["source", "target"]):
    data = batch.to_pydict()
    for u, v in zip(data["source"], data["target"]):
        u, v = int(u), int(v)
        adj[u].append(v)
        adj[v].append(u)

print(f"   Adjacency built in {time.perf_counter() - t0:.2f}s")

# --------------------------
# BFS with early stopping
# --------------------------
def bfs_early_stop(source: int, targets: set):
    """
    BFS from `source` but stop once all nodes in `targets` have been reached.
    Returns a dict {target_node: distance}
    """
    if not targets:
        return {}
    if source in targets:
        targets = targets - {source}
        if not targets:
            return {source: 0}

    dist = [-1] * n_nodes
    dist[source] = 0
    q = deque([source])
    found = {}

    while q and targets:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                if v in targets:
                    found[v] = dist[v]
                    targets.remove(v)
                    if not targets:
                        return found
                q.append(v)
    return found

# -----------------------------------------------------
# Run BFS for each unique source and collect results
# -----------------------------------------------------
results = []
t_start = time.perf_counter()

for i, src in enumerate(unique_sources, 1):
    target_list = queries_by_source[src]
    target_set = {t_int for _, t_int, _, _ in target_list}

    found = bfs_early_stop(src, target_set.copy())

    for serial_no, t_int, s_lab, t_lab in target_list:
        if src == t_int:
            dist = 0
        else:
            dist = found.get(t_int, -1)

        labels_sorted = sorted([s_lab, t_lab])

        results.append({
            "query_serial_no": serial_no,
            "query_labels": labels_sorted,   
            "true": dist
        })

    if i % 50 == 0:
        print(f"   Processed {i}/{len(unique_sources)} sources")

total_time = time.perf_counter() - t_start
print(f"   All done in {total_time:.1f}s")

# --------------------------
# Save results
# --------------------------
results_sorted = sorted(results, key=lambda x: x["query_serial_no"])
with open(TRUE_DIST_JSON, "w") as f:
    json.dump(results_sorted, f, indent=2)

print(f"   Saved → {TRUE_DIST_JSON}")
print(f"   {len(results_sorted)} queries with correct true distances.\n")
#!/usr/bin/env python3
"""
precompute_true_distances.py

Efficiently compute true (exact) distances for the queries listed in config.yaml.

- Streams edges.parquet in batches to build adjacency (undirected).
- Groups queries by unique source and runs one BFS per source.
- Each BFS stops early when all target nodes for that source are discovered.
- Saves results to output/true_distances.json

Safe for large graphs (hundreds of thousands of nodes, millions of edges).
"""

import os
import time
import json
import yaml
from collections import deque, defaultdict
import pyarrow.parquet as pq

# --------------------------
# Config + paths
# --------------------------
with open("config.yaml") as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config["output_dir"]
EDGES_PARQUET = os.path.join(OUTPUT_DIR, "edges.parquet")
NODE_MAP_JSON = os.path.join(OUTPUT_DIR, "node_map.json")
TRUE_DIST_JSON = os.path.join(OUTPUT_DIR, "true_distances.json")
QUERIES_RAW = config.get("queries", [])

# --------------------------
# Load node map
# --------------------------
with open(NODE_MAP_JSON, "r") as f:
    id_to_label = json.load(f)  # keys are stringified ints
label_to_id = {str(v): int(k) for k, v in id_to_label.items()}
n_nodes = len(id_to_label)

# --------------------------
# Map queries to internal IDs and create serial numbers
# --------------------------
queries = []          # (serial_no, s_id, t_id, s_label, t_label)
serial = 1
for s_label, t_label in QUERIES_RAW:
    s_id = label_to_id.get(str(s_label))
    t_id = label_to_id.get(str(t_label))
    if s_id is None or t_id is None:
        # skip missing nodes but warn once
        print(f"WARNING: query labels {s_label},{t_label} not found in node_map.json — skipping")
        continue
    queries.append((serial, s_id, t_id, str(s_label), str(t_label)))
    serial += 1

if not queries:
    raise SystemExit("No valid queries found in config.yaml. Exiting.")

# Group queries by source to minimize BFS runs
queries_by_source = defaultdict(list)  # s_id -> list of (serial, t_id, s_label, t_label)
for serial_no, s_id, t_id, s_label, t_label in queries:
    queries_by_source[s_id].append((serial_no, t_id, s_label, t_label))

unique_sources = list(queries_by_source.keys())
print(f"Total queries: {len(queries)}, unique sources: {len(unique_sources)}")

# --------------------------
# Build adjacency (stream Parquet in batches)
# --------------------------
print("Building adjacency list from Parquet (streaming batches)...")
t0 = time.perf_counter()

# Preallocate adjacency list (list of lists)
adj = [[] for _ in range(n_nodes)]

pf = pq.ParquetFile(EDGES_PARQUET)
batch_count = 0
for batch in pf.iter_batches(batch_size=1_000_000, columns=["source", "target"]):
    batch_count += 1
    tb = batch.to_pydict()
    srcs = tb["source"]
    tgts = tb["target"]
    # Add edges undirected
    for u, v in zip(srcs, tgts):
        u = int(u); v = int(v)
        adj[u].append(v)
        adj[v].append(u)
# lightweight dedup can be applied if memory/time allows; skipping it for speed

t1 = time.perf_counter()
print(f"Adjacency built: {n_nodes} nodes, batches={batch_count}, time={t1-t0:.2f}s")

# --------------------------
# BFS that stops when all targets are found
# --------------------------
def bfs_find_targets(source: int, target_set: set):
    """
    BFS from source; stops early when all nodes in target_set are found.
    Returns dict: target_node -> distance (only for found targets).
    """
    if not target_set:
        return {}
    if source in target_set:
        # if source is also a target, distance 0
        found = {source: 0}
        remaining = set(target_set)
        remaining.discard(source)
        if not remaining:
            return found
        target_set = remaining

    dist = [-1] * n_nodes
    dist[source] = 0
    q = deque([source])
    found_map = {}
    remaining = set(target_set)

    # loop
    while q and remaining:
        u = q.popleft()
        du = dist[u]
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = du + 1
                if v in remaining:
                    found_map[v] = dist[v]
                    remaining.remove(v)
                    # if no remaining targets left we can return immediately
                    if not remaining:
                        return found_map
                q.append(v)
    # return whatever we found (missing targets are considered unreachable -> not present)
    return found_map

# --------------------------
# Run BFS for each unique source, collect results
# --------------------------
results = []  # list of dicts with serial_no, query_labels, true

total_sources = len(unique_sources)
t_all_start = time.perf_counter()
last_print = time.time()

# Iterate deterministically (sorted) to make runs reproducible
for i, src in enumerate(unique_sources, start=1):
    # targets for this source
    qlist = queries_by_source[src]
    target_nodes = {t for (_, t, _, _) in qlist}

    t_src_start = time.perf_counter()
    found = bfs_find_targets(src, target_nodes)
    t_src_end = time.perf_counter()

    # For each query belonging to this source, fetch true distance or -1
    for serial_no, t_id, s_label, t_label in qlist:
        # if source==target, distance 0
        if src == t_id:
            true_dist = 0
        else:
            true_dist = found.get(t_id, -1)
        results.append({
            "query_serial_no": serial_no,
            "query_labels": [s_label, t_label],
            "true": true_dist
        })

    # periodic progress (every 50 sources or every 30s)
    if (i % 50 == 0) or (time.time() - last_print > 30):
        elapsed = time.perf_counter() - t_all_start
        pct = (i / total_sources) * 100
        print(f"  processed sources {i}/{total_sources} ({pct:.1f}%)  elapsed={elapsed:.1f}s  src={src}  bfs_time={(t_src_end-t_src_start):.2f}s")
        last_print = time.time()

t_all_end = time.perf_counter()
print(f"Completed BFS for {len(unique_sources)} sources in {t_all_end - t_all_start:.1f}s")

# --------------------------
# Save results (sorted by serial_no for convenience)
# --------------------------
results_sorted = sorted(results, key=lambda r: r["query_serial_no"])
with open(TRUE_DIST_JSON, "w") as f:
    json.dump(results_sorted, f, indent=2)

print(f"Saved true distances → {TRUE_DIST_JSON} (queries: {len(results_sorted)})")
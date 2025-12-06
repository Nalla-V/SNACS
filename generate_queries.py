# generate_queries.py — FINAL, FAST & 100% CORRECT (matches your preprocess perfectly)
import os
import random
import yaml
import json
import numpy as np
import pandas as pd
from collections import deque

with open("config.yaml") as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config["output_dir"]
EDGES_PARQUET = os.path.join(OUTPUT_DIR, "edges.parquet")
NODE_MAP_JSON = os.path.join(OUTPUT_DIR, "node_map.json")

print("\n# generate_queries.py: Generating 500 diverse queries (FAST + CORRECT)")

# ---------------- Load node_map: {"0": "1001", "1": "54321", ...} ----------------
with open(NODE_MAP_JSON) as f:
    internal_to_orig_str = json.load(f)   # This is exactly what your preprocess saves!

# Build correct dense ↔ original mappings
dense_to_orig = {int(internal): int(orig_node) 
                 for internal, orig_node in internal_to_orig_str.items()}
orig_to_dense = {orig: dense for dense, orig in dense_to_orig.items()}

num_nodes = len(dense_to_orig)
label_arr = [dense_to_orig[i] for i in range(num_nodes)]  # ← REAL original node IDs

print(f"   Loaded {num_nodes:,} nodes → dense internal IDs 0..{num_nodes-1}")

# ---------------- Load edges (already use dense internal IDs!) ----------------
df = pd.read_parquet(EDGES_PARQUET)
sources = df["source"].astype(np.int32).to_numpy()
targets = df["target"].astype(np.int32).to_numpy()

print(f"   Loaded {len(sources):,} edges (already in dense internal format)")

# ---------------- Build NumPy adjacency list ----------------
adj = [[] for _ in range(num_nodes)]
for u, v in zip(sources, targets):
    adj[u].append(v)
    adj[v].append(u)

# Convert to NumPy arrays for max speed
adj = [np.array(nei, dtype=np.int32) for nei in adj]
print("   Adjacency list built (NumPy optimized)")

# ---------------- Ultra-fast vectorized BFS ----------------
def bfs_np(src: int) -> np.ndarray:
    dist = np.full(num_nodes, -1, dtype=np.int32)
    dist[src] = 0
    q = deque([src])

    while q:
        u = q.popleft()
        neighbors = adj[u]
        candidates = neighbors[dist[neighbors] == -1]
        if len(candidates) > 0:
            dist[candidates] = dist[u] + 1
            q.extend(candidates)
    return dist

# ---------------- Stratified pair collection ----------------
short, medium, long = set(), set(), set()
TARGET_S = 180
TARGET_M = 220
TARGET_L = 120

seeds = random.sample(range(num_nodes), min(120, num_nodes))
print(f"   Running BFS from {len(seeds)} random seeds...")

for i, s in enumerate(seeds):
    dist = bfs_np(s)
    reachable = np.flatnonzero(dist > 0)

    s_orig = label_arr[s]

    for t in reachable:
        d = dist[t]
        t_orig = label_arr[t]
        a, b = s_orig, t_orig
        pair = (a, b) if a < b else (b, a)

        if d <= 2 and len(short) < TARGET_S * 2:
            short.add(pair)
        elif d <= 5 and len(medium) < TARGET_M * 2:
            medium.add(pair)
        elif d >= 6 and len(long) < TARGET_L * 2:
            long.add(pair)

    print(f"   → S={len(short):4d}  M={len(medium):4d}  L={len(long):4d}", end="\r")

    if len(short) >= TARGET_S and len(medium) >= TARGET_M and len(long) >= TARGET_L:
        break

print(f"\n   Collection complete: Short={len(short)}, Medium={len(medium)}, Long={len(long)}")

# ---------------- Build final 500 unique queries ----------------
final_pairs = set()

final_pairs.update(random.sample(list(short), min(TARGET_S, len(short))))
final_pairs.update(random.sample(list(medium), min(TARGET_M, len(medium))))
final_pairs.update(random.sample(list(long), min(TARGET_L, len(long))))

# Fill up to exactly 500 with random unique pairs (deduped)
while len(final_pairs) < 500:
    a, b = random.sample(label_arr, 2)
    pair = (min(a, b), max(a, b))
    final_pairs.add(pair)

queries = [list(p) for p in final_pairs]
random.shuffle(queries)
queries = queries[:500]

print(f"   Final: {len(queries)} diverse queries (real original node IDs)")

# ---------------- Save in your preferred clean YAML format ----------------
class FlowPair(list):
    pass

def flow_pair_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(FlowPair, flow_pair_representer)

OUT_PATH = os.path.join(OUTPUT_DIR, "generated_queries.yaml")
with open(OUT_PATH, "w") as f:
    f.write("queries:\n")
    for u, v in queries:
        f.write(f"- [{u}, {v}]\n")

print(f"   Saved → {OUT_PATH}")
print("   All done! Queries are correct, diverse, and ready for evaluation.")
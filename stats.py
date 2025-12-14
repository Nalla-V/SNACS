"""
Computes basic graph statistics from the preprocessed graph.
It reports |V|, |E|, average degree, and an approximate average clustering coefficient.

Reads (from output_dir in config.yaml)
  - config.yaml
  - edges.parquet
  - node_map.json

Output
  - prints statistics to stdout (no files written)
"""

import os
import json
import random
import yaml
import numpy as np
import pandas as pd
from itertools import combinations

# -------------------------------
# Configuration 
# -------------------------------
with open("config.yaml") as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config["output_dir"]
EDGES_PARQUET = os.path.join(OUTPUT_DIR, "edges.parquet")
NODE_MAP_JSON = os.path.join(OUTPUT_DIR, "node_map.json")

random.seed(42)

# -------------------------------
# Load node map
# -------------------------------
with open(NODE_MAP_JSON) as f:
    id_to_node = json.load(f)

num_nodes = len(id_to_node)

# -------------------------------
# Load edges
# -------------------------------
df = pd.read_parquet(EDGES_PARQUET)
src = df["source"].astype(int).to_numpy()
tgt = df["target"].astype(int).to_numpy()

num_edges = len(df) 

# -------------------------------
# Compute Average degree (exact)
# -------------------------------
avg_degree = 2.0 * num_edges / num_nodes

print("BASIC STATS")
print(f"  |V| (nodes)  : {num_nodes}")
print(f"  |E| (edges)  : {num_edges}")
print(f"  <k> (avg deg): {avg_degree:.4f}")

# --------------------------------------
# Build adjacency (needed for clustering)
# --------------------------------------
adj = [set() for _ in range(num_nodes)]
for u, v in zip(src, tgt):
    adj[u].add(v)
    adj[v].add(u)

# -------------------------------
# Local clustering coefficient
# -------------------------------
def local_clustering(node_id: int) -> float:
    """
    Local clustering coefficient for one node:
      (# of edges between neighbors) / (deg * (deg - 1) / 2)

    For high-degree nodes, we approximate by sampling neighbor pairs to keep runtime bounded.
    """
    neighbors = list(adj[node_id])
    deg = len(neighbors)
    if deg < 2:
        return 0.0

    possible = deg * (deg - 1) // 2

    # For high degree nodes, sample neighbor pairs
    if deg > 100:
        sample_size = min(500, possible)
        pairs = random.sample(list(combinations(range(deg), 2)), sample_size)
        hits = 0
        for i, j in pairs:
            u, v = neighbors[i], neighbors[j]
            if v in adj[u]:
                hits += 1

        return hits / sample_size

    # Exact counting for smaller degree
    triangles = 0
    for i in range(deg):
        u = neighbors[i]
        for j in range(i + 1, deg):
            v = neighbors[j]
            if v in adj[u]:
                triangles += 1

    return triangles / possible if possible > 0 else 0.0

# ------------------------------------------
# Compute Average Clustering (approximate)
# ------------------------------------------
CLUSTERING_SAMPLES = 1000
sample_size = min(CLUSTERING_SAMPLES, num_nodes)
sample_nodes = random.sample(range(num_nodes), sample_size)

clustering_vals = [local_clustering(i) for i in sample_nodes]
avg_clustering = float(np.mean(clustering_vals))

print(f"  c (avg clustering, ~{sample_size} samples): {avg_clustering:.4f}")

"""
Compute a rough distance distribution from the preprocessed graph with sample size = 50.

Reads:
  - edges.parquet
  - node_map.json

The script samples a handful of source nodes, runs BFS, and counts how often each
hop distance occurs. It then saves a compact bar chart in the dataset output folder.
"""
import os
import json
import random
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque, Counter
import matplotlib.ticker as mtick

# -------------------------------
# Plot title
# -------------------------------
TITLE = "Twitch Gamers Dataset"

# -------------------------------
# Configuration 
# -------------------------------
with open("config.yaml") as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config["output_dir"]
EDGES_PARQUET = os.path.join(OUTPUT_DIR, "edges.parquet")
NODE_MAP_JSON = os.path.join(OUTPUT_DIR, "node_map.json")

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

# -------------------------------
# Build adjacency
# -------------------------------
adj = [[] for _ in range(num_nodes)]
for u, v in zip(src, tgt):
    adj[u].append(v)
    adj[v].append(u)

adj = [np.array(nei, dtype=np.int32) for nei in adj]

# -------------------------------
# BFS helper
# -------------------------------
def bfs(src):
    dist = [-1] * num_nodes
    q = deque([src])
    dist[src] = 0

    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

# -------------------------------
# Sampling: run BFS from a small set of nodes
# -------------------------------
SAMPLES = 50
sample_nodes = random.sample(range(num_nodes), min(SAMPLES, num_nodes))

distance_counter = Counter()

for s in sample_nodes:
    dist = bfs(s)
    for d in dist:
        if d > 0:
            distance_counter[d] += 1

# Sort keys
distances = sorted(distance_counter.keys())
frequencies = [distance_counter[d] for d in distances]

# -------------------------------
# Plot
# -------------------------------
plt.figure(figsize=(4, 3))
plt.bar(distances, frequencies)

plt.xlabel("Distance", fontsize=15)
plt.ylabel("Frequency", fontsize=15)
plt.title(TITLE, fontsize=15)

max_d = 10
plt.xticks(range(0, max_d + 5, 5), fontsize=10)

max_freq = max(frequencies)
plt.gca().yaxis.set_major_formatter(
    mtick.FuncFormatter(lambda val, pos: f"{val / max_freq:.1f}")
)

plt.yticks([i * max_freq * 0.2 for i in range(6)])

# Save plot
plt.tight_layout()
OUT_PNG = os.path.join(OUTPUT_DIR, "twitch_distance_distribution.png")
plt.savefig(OUT_PNG, dpi=200)
plt.close()

print(f"Saved small distance distribution graph → {OUT_PNG}")

# generate_queries.py
import os
import random
import yaml
import pandas as pd
from collections import deque
import json

CONFIG_PATH = "config.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config["output_dir"]
EDGES_PARQUET = os.path.join(OUTPUT_DIR, "edges.parquet")
NODE_MAP_JSON = os.path.join(OUTPUT_DIR, "node_map.json")

print("\n# generate_queries.py: generating evaluation queries...")

# Load node map: internal_id (str) → original label
with open(NODE_MAP_JSON) as f:
    id_to_label = json.load(f)           # e.g. {"0": "alice", "1": "bob", ...}
    label_to_id = {v: int(k) for k, v in id_to_label.items()}  # reverse map for BFS

all_internal_ids = [int(x) for x in id_to_label.keys()]

# Load edges (still in internal IDs)
df = pd.read_parquet(EDGES_PARQUET)
n = len(all_internal_ids)

adj = {i: [] for i in all_internal_ids}
for _, row in df.iterrows():
    u, v = int(row["source"]), int(row["target"])
    adj[u].append(v)
    adj[v].append(u)

print(f"   Loaded graph with {n} nodes, {len(df)} edges")

# BFS using internal IDs
def bfs(source_internal):
    dist = {source_internal: 0}
    q = deque([source_internal])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

short = []
medium = []
long = []
random_pairs = []

seed_nodes = random.sample(all_internal_ids, min(30, len(all_internal_ids)))

print("   Running BFS from sample nodes to discover distances...")
for s_internal in seed_nodes:
    dist = bfs(s_internal)
    s_label = id_to_label[str(s_internal)]

    for t_internal, d in dist.items():
        if d == 0:
            continue
        t_label = id_to_label[str(t_internal)]

        pair = (s_label, t_label)  # ← Store original labels directly

        if 1 <= d <= 3:
            short.append(pair)
        elif 4 <= d <= 10:
            medium.append(pair)
        elif d > 10:
            long.append(pair)

# Add some random label pairs (still using original labels)
random_label_pairs = []
for _ in range(20):
    s_label = random.choice(list(id_to_label.values()))
    t_label = random.choice(list(id_to_label.values()))
    random_label_pairs.append((s_label, t_label))

def pick(lst, k):
    return random.sample(lst, min(k, len(lst)))

# Final queries — all in original labels!
queries_with_labels = []
queries_with_labels += pick(short, 10)
queries_with_labels += pick(medium, 15)
queries_with_labels += pick(long, 15)
queries_with_labels += pick(random_label_pairs, 10)

# Convert to list of lists for clean YAML output: [label_s, label_t]
queries = [list(pair) for pair in queries_with_labels]

print(f"   Selected {len(queries)} queries (using original node labels)")

# Force clean flow style: - [label1, label2]
def list_flow_representer(dumper, data):
    if len(data) == 2:
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)

yaml.add_representer(list, list_flow_representer)

# Save
OUT_YAML = os.path.join(OUTPUT_DIR, "generated_queries.yaml")

with open(OUT_YAML, "w") as f:
    yaml.dump(
        {"queries": queries},
        f,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True  # in case labels have special chars
    )

print(f"   Saved queries → {OUT_YAML}")
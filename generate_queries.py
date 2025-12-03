# generate_queries.py — FINAL: Your exact style + perfect logic + guaranteed 500 queries
import os
import random
import yaml
import json
import pandas as pd
from collections import deque

with open("config.yaml") as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config["output_dir"]
EDGES_PARQUET = os.path.join(OUTPUT_DIR, "edges.parquet")
NODE_MAP_JSON = os.path.join(OUTPUT_DIR, "node_map.json")

print("\n# generate_queries.py: Generating 500 diverse queries for evaluation")

# Load node map
with open(NODE_MAP_JSON) as f:
    id_to_label = json.load(f)
    label_to_id = {v: int(k) for k, v in id_to_label.items()}

all_internal_ids = [int(x) for x in id_to_label.keys()]
all_labels = list(id_to_label.values())

# Build adjacency list
df = pd.read_parquet(EDGES_PARQUET)
adj = {i: [] for i in all_internal_ids}
for _, row in df.iterrows():
    u, v = int(row["source"]), int(row["target"])
    adj[u].append(v)
    adj[v].append(u)

print(f"   Graph loaded: {len(all_internal_ids):,} nodes, {len(df):,} edges")

def bfs(source):
    dist = {source: 0}
    q = deque([source])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

# === Collect stratified pairs (deduplicated) ===
short = set()
medium = set()
long = set()

seeds = random.sample(all_internal_ids, min(80, len(all_internal_ids)))
print(f"   Running BFS from {len(seeds)} seeds...")

for s in seeds:
    dists = bfs(s)
    s_label = id_to_label[str(s)]
    for t, d in dists.items():
        if d == 0: continue
        t_label = id_to_label[str(t)]
        pair = tuple(sorted([s_label, t_label]))
        if d <= 3:
            short.add(pair)
        elif d <= 7:
            medium.add(pair)
        elif d >= 9:
            long.add(pair)

print(f"   Collected: {len(short)} short, {len(medium)} medium, {len(long)} long")

# === Build final query set ===
final_pairs = set()

# Add as many as possible from each bucket
final_pairs.update(random.sample(list(short), min(150, len(short))))
final_pairs.update(random.sample(list(medium), min(200, len(medium))))
final_pairs.update(random.sample(list(long), min(100, len(long))))

# Fill up to 500 with random pairs (deduplicated)
while len(final_pairs) < 500:
    s = random.choice(all_labels)
    t = random.choice(all_labels)
    if s != t:
        final_pairs.add(tuple(sorted([s, t])))

# Convert to list and shuffle
queries_list = list(final_pairs)
random.shuffle(queries_list)
queries_list = queries_list[:500]  # exactly 500

# Convert to [label1, label2] format
queries = [list(pair) for pair in queries_list]

print(f"   Generated {len(queries)} unique queries")

# === YOUR EXACT OUTPUT FORMAT ===
def list_flow_representer(dumper, data):
    if len(data) == 2:  # only for query pairs
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
        allow_unicode=True
    )

print(f"   Saved queries → {OUT_YAML}")
import os
import yaml
import pandas as pd
import json
import time

with open("config.yaml") as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config["output_dir"]
DISTANCES_PARQUET = os.path.join(OUTPUT_DIR, "distances.parquet")
NODE_MAP_JSON = os.path.join(OUTPUT_DIR, "node_map.json")
QUERIES_RAW = config["queries"]

print("\n# 3_query_service.py: Fast estimation")

# === 1. Load node map (label → id) ===
with open(NODE_MAP_JSON) as f:
    id_to_label = json.load(f)
    # Reverse mapping: external label → internal ID
    label_to_id = {str(v): int(k) for k, v in id_to_label.items()}

# === 2. Load precomputed landmark distances ===
df = pd.read_parquet(DISTANCES_PARQUET)
dist_map = df.set_index(['node', 'landmark'])['distance'].to_dict()
landmarks = sorted(df['landmark'].unique())

# === 3. Map queries to internal IDs ===
QUERIES = []
for s_label, t_label in QUERIES_RAW:
    s_id = label_to_id.get(str(s_label))
    t_id = label_to_id.get(str(t_label))
    if s_id is None or t_id is None:
        print(f"   WARNING: Query ({s_label}, {t_label}) → node not in graph, skipping.")
        continue
    QUERIES.append((s_id, t_id))

print(f"   Mapped queries: {QUERIES}")

# === 4. Define estimation function ===
def estimate(s, t):
    if s == t:
        return 0
    min_d = float('inf')
    for u in landmarks:
        ds = dist_map.get((s, u), float('inf'))
        dt = dist_map.get((t, u), float('inf'))
        if ds < float('inf') and dt < float('inf'):
            min_d = min(min_d, ds + dt)
    return int(min_d) if min_d < float('inf') else -1

print("   Ready for queries!\n")

# === 5. Run queries ===
for s, t in QUERIES:
    result = estimate(s, t)
    print(f"   Query ({id_to_label[str(s)]}, {id_to_label[str(t)]}) → estimate: {result}")
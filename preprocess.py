import os
import json
import yaml
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

INPUT_TSV = config["input_tsv"]
OUTPUT_DIR = config["output_dir"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

EDGES_PARQUET = os.path.join(OUTPUT_DIR, "edges.parquet")
NODE_MAP_JSON = os.path.join(OUTPUT_DIR, "node_map.json")

print("\n# 1_preprocess.py: Starting")

# Read as strings
df = pd.read_csv(INPUT_TSV, sep=r'\s+', header=None, names=['u', 'v'], dtype=str)

# Make undirected
df_rev = df.rename(columns={'u': 'v', 'v': 'u'})
df = pd.concat([df, df_rev]).drop_duplicates().reset_index(drop=True)

# Map to dense int32 IDs
all_nodes = sorted(set(df['u']) | set(df['v']))
node_to_id = {node: i for i, node in enumerate(all_nodes)}
id_to_node = {i: node for node, i in node_to_id.items()}

df['source'] = df['u'].map(node_to_id).astype('int32')
df['target'] = df['v'].map(node_to_id).astype('int32')
df = df[['source', 'target']]

# Save
pq.write_table(pa.Table.from_pandas(df), EDGES_PARQUET)
with open(NODE_MAP_JSON, 'w') as f:
    json.dump(id_to_node, f, indent=2)

print(f"→ {len(df)} edges → {EDGES_PARQUET}")
print(f"→ {len(id_to_node)} nodes → {NODE_MAP_JSON}")
print("Done.\n")
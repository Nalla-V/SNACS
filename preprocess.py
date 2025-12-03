import os
import json
import yaml
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

INPUT_FILE = config["input_tsv"]        # can be .tsv, .csv, .txt — any text edge list
OUTPUT_DIR = config["output_dir"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

EDGES_PARQUET = os.path.join(OUTPUT_DIR, "edges.parquet")
NODE_MAP_JSON = os.path.join(OUTPUT_DIR, "node_map.json")

print("\n# 1_preprocess.py: Converting undirected edge list → parquet + node mapping")

# === AUTO-DETECT: does the file have a header? ===
# Try reading first row to check if it's numeric
try:
    first_row = pd.read_csv(INPUT_FILE, nrows=1, header=None)
    is_header = not (pd.to_numeric(first_row.iloc[0], errors='coerce').notnull().all())
except:
    is_header = True  # fallback

header_param = 0 if is_header else None

# === READ EDGE LIST ===
# Works for:
# - CSV with header ("source,target" or "u,v")
# - CSV without header
# - TSV (tab or space separated)
df = pd.read_csv(
    INPUT_FILE,
    sep=None,              # auto-detect separator (comma, tab, space)
    engine='python',
    header=header_param,
    names=['u', 'v'],
    dtype=str,
    comment='#',           # skip comment lines
    skip_blank_lines=True
)

# Drop any accidental self-loops or invalid rows
df = df[df['u'] != df['v']].dropna()

print(f"   Loaded {len(df):,} undirected edges")

# === MAP NODES TO DENSE 0..N-1 IDs ===
all_nodes = sorted(set(df['u']) | set(df['v']))
node_to_id = {node: i for i, node in enumerate(all_nodes)}
id_to_node = {i: str(node) for node, i in node_to_id.items()}  # keep as string

df['source'] = df['u'].map(node_to_id).astype('int32')
df['target'] = df['v'].map(node_to_id).astype('int32')
df = df[['source', 'target']]

# Optional: remove duplicate edges (u,v) and (v,u) if input had both
df = df.drop_duplicates()

# === SAVE ===
pq.write_table(pa.Table.from_pandas(df), EDGES_PARQUET)
with open(NODE_MAP_JSON, 'w') as f:
    json.dump(id_to_node, f, indent=2)

print(f"→ {len(df):,} final undirected edges → {EDGES_PARQUET}")
print(f"→ {len(id_to_node):,} nodes (dense IDs 0..{len(id_to_node)-1}) → {NODE_MAP_JSON}")
print("Preprocessing complete.\n")
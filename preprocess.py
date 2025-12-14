"""
Converts an edge-list file (CSV/TSV/whitespace) into a compact internal representation.
It streams the input, maps node labels to contiguous integer IDs, removes self-loops,
and writes the resulting edges to `edges.parquet` plus a `node_map.json`.

Reads
  - config.yaml (input_tsv, output_dir)
  - input_tsv (edge list)

Writes (to output_dir)
  - edges.parquet
  - node_map.json
"""

import os
import json
import yaml
import pyarrow as pa
import pyarrow.parquet as pq

# --------------------------
# Configuration 
# --------------------------
with open("config.yaml") as f:
    config = yaml.safe_load(f)

INPUT_FILE = config["input_tsv"]
OUTPUT_DIR = config["output_dir"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

EDGES_PARQUET = os.path.join(OUTPUT_DIR, "edges.parquet")
NODE_MAP_JSON = os.path.join(OUTPUT_DIR, "node_map.json")
TMP_EDGES = os.path.join(OUTPUT_DIR, "_edges.tmp")  # temporary numeric edge list

print("\n# 1_preprocess: streaming edge list → parquet + node mapping")

# --------------------------
# Header detection
# --------------------------
def looks_like_header(tokens):
    """Heuristic: return True if the first two fields look like a header row."""
    if len(tokens) < 2:
        return False

    common_headers = {
        "source", "target", "u", "v", "node1", "node2",
        "from", "to", "src", "dst", "id1", "id2", "head", "tail"
    }

    if any(t.lower() in common_headers for t in tokens):
        return True

    both_numeric = all(
        t.strip().replace("-", "").isdigit() and t.strip() != ""
        for t in tokens
    )
    if both_numeric:
        return False

    if any(any(c.isalpha() for c in t) for t in tokens):
        return True

    return False


# --------------------------
# Pass 1: Stream input, build node mapping, write numeric edges
# --------------------------
node_to_id = {}
id_to_node = []
next_id = 0
edges_processed = 0
skipped_header = False

# Open temp file for numeric edge pairs and input once
with open(INPUT_FILE, "r", encoding="utf-8", errors="replace") as fin, \
     open(TMP_EDGES, "w", encoding="utf-8") as tmp_f:

    while True:
        pos = fin.tell()
        line = fin.readline()
        if not line:  # EOF
            break

        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Split: prefer comma-separated; otherwise fall back to whitespace
        if "," in line:
            parts = [p.strip() for p in stripped.split(",") if p.strip()]
        else:
            parts = stripped.split()

        if len(parts) < 2:
            continue

        tokens = parts[:2]

        if not skipped_header and looks_like_header(tokens):
            skipped_header = True
            continue
        else:
            if not skipped_header:
                fin.seek(pos)
            break  

    # Process all remaining lines as edges
    for line in fin:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if "," in line:
            parts = [p.strip() for p in stripped.split(",") if p.strip()]
        else:
            parts = stripped.split()

        if len(parts) < 2:
            continue

        u_raw, v_raw = parts[0], parts[1]

        # Skip self-loops
        if u_raw == v_raw:
            continue

        # Map raw node labels to contiguous internal IDs
        for node in (u_raw, v_raw):
            if node not in node_to_id:
                node_to_id[node] = next_id
                id_to_node.append(node)
                next_id += 1

        u_id = node_to_id[u_raw]
        v_id = node_to_id[v_raw]

        tmp_f.write(f"{u_id},{v_id}\n")
        edges_processed += 1


print(f"  Pass 1 complete: {edges_processed:,} edges processed, {next_id:,} unique nodes found (header skipped={skipped_header})")

# --------------------------
# Saving node mapping
# --------------------------
node_map = {str(i): node for i, node in enumerate(id_to_node)}
with open(NODE_MAP_JSON, "w", encoding="utf-8") as f:
    json.dump(node_map, f, indent=2, ensure_ascii=False)
print(f"  Saved node map → {NODE_MAP_JSON}")

# --------------------------
# Pass 2: write edges.parquet in batches
# --------------------------
schema = pa.schema([
    pa.field("source", pa.int32()),
    pa.field("target", pa.int32())
])

writer = pq.ParquetWriter(EDGES_PARQUET, schema, compression="SNAPPY")

batch_size = 500_000
src_batch = []
tgt_batch = []
edges_written = 0

with open(TMP_EDGES, "r", encoding="utf-8") as f:
    for line in f:
        s_str, t_str = line.strip().split(",")
        src_batch.append(int(s_str))
        tgt_batch.append(int(t_str))

        if len(src_batch) >= batch_size:
            table = pa.Table.from_arrays(
                [pa.array(src_batch, type=pa.int32()),
                 pa.array(tgt_batch, type=pa.int32())],
                schema=schema
            )
            writer.write_table(table)
            edges_written += len(src_batch)
            src_batch.clear()
            tgt_batch.clear()

    # Final batch
    if src_batch:
        table = pa.Table.from_arrays(
            [pa.array(src_batch, type=pa.int32()),
             pa.array(tgt_batch, type=pa.int32())],
            schema=schema
        )
        writer.write_table(table)
        edges_written += len(src_batch)

writer.close()
os.remove(TMP_EDGES)  # Remove temporary files

print(f"  Saved edges parquet → {EDGES_PARQUET}")
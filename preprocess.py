#!/usr/bin/env python3
"""
Streaming preprocess: TSV/CSV edge list → edges.parquet + node_map.json

Fixes:
- No more double file opening → eliminates +1 edge / +2 nodes bug
- Single pass over input file
- Robust header detection
- Handles comma-separated or whitespace-separated files
- Skips comments (#) and empty lines
- Ignores self-loops
- Memory efficient (batched Parquet writing)
"""

import os
import json
import yaml
import pyarrow as pa
import pyarrow.parquet as pq

# ---------- config ----------
with open("config.yaml") as f:
    config = yaml.safe_load(f)

INPUT_FILE = config["input_tsv"]
OUTPUT_DIR = config["output_dir"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

EDGES_PARQUET = os.path.join(OUTPUT_DIR, "edges.parquet")
NODE_MAP_JSON = os.path.join(OUTPUT_DIR, "node_map.json")
TMP_EDGES = os.path.join(OUTPUT_DIR, "_edges.tmp")  # temporary numeric edge list

print("\n# 1_preprocess: streaming edge list → parquet + node mapping")

# ---------- Robust header detection ----------
def looks_like_header(tokens):
    """Return True if the token pair looks like a header row."""
    if len(tokens) < 2:
        return False

    common_headers = {
        "source", "target", "u", "v", "node1", "node2",
        "from", "to", "src", "dst", "id1", "id2", "head", "tail"
    }

    # If either token is a known header keyword → definitely header
    if any(t.lower() in common_headers for t in tokens):
        return True

    # If both tokens are purely numeric (or negative integers) → data, not header
    both_numeric = all(
        t.strip().replace("-", "").isdigit() and t.strip() != ""
        for t in tokens
    )
    if both_numeric:
        return False

    # If any token contains letters → likely header (node IDs are usually numeric or UUIDs)
    if any(any(c.isalpha() for c in t) for t in tokens):
        return True

    return False


# ---------- Single streaming pass ----------
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

        # Split: prefer comma, fallback to whitespace
        if "," in line:
            parts = [p.strip() for p in stripped.split(",") if p.strip()]
        else:
            parts = stripped.split()

        if len(parts) < 2:
            continue

        tokens = parts[:2]

        # Header detection on first valid line
        if not skipped_header and looks_like_header(tokens):
            skipped_header = True
            # Header consumed → continue to next line
            continue
        else:
            # Not a header → if we hadn't decided yet, rewind this line
            if not skipped_header:
                fin.seek(pos)
            break  # exit header detection loop

    # Now process all remaining lines (header already skipped if present)
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

        # Extra safety: skip any line if it looks exactly like a header (very rare)
        if u_raw.lower() in {"source", "target", "u", "v", "from", "to"} and \
           v_raw.lower() in {"source", "target", "u", "v", "from", "to"}:
            continue

        # Skip self-loops
        if u_raw == v_raw:
            continue

        # Assign incremental IDs
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

# ---------- Save node mapping ----------
node_map = {str(i): node for i, node in enumerate(id_to_node)}
with open(NODE_MAP_JSON, "w", encoding="utf-8") as f:
    json.dump(node_map, f, indent=2, ensure_ascii=False)
print(f"  Saved node map → {NODE_MAP_JSON}")

# ---------- Pass 2: Write Parquet in batches ----------
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
os.remove(TMP_EDGES)  # clean up

print(f"  Saved edges parquet → {EDGES_PARQUET}")
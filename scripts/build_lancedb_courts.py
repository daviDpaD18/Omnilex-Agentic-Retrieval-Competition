"""
Build LanceDB vector index for Swiss court decisions.

What this does:
  1. Reads court_considerations.csv (~2.5M rows, ~2.3GB)
  2. Embeds each row's text with multilingual-e5-base (768-dim) on GPU
  3. Writes [citation, text, vector] to LanceDB at data/processed/lancedb_courts/

Why LanceDB:
  - Files are memory-mapped on disk — no RAM cost at query time
  - HyDE agent loads the table with lancedb.connect() in ~1 second
  - ANN search (IVF-PQ) at query time costs ~50ms per call
  - Total disk space: ~5-8 GB for 2.5M x 768 float32 vectors

Runtime estimate on RTX 4050:
  - Embedding throughput: ~2,500 rows/sec at batch_size=256
  - 2.5M rows / 2,500 = ~17 minutes total

Usage:
  python scripts/build_lancedb_courts.py
  python scripts/build_lancedb_courts.py --batch-size 512   # faster if VRAM allows
  python scripts/build_lancedb_courts.py --sample 10000     # quick test run
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_PATH   = REPO_ROOT / "data"
COURTS_CSV  = DATA_PATH / "court_considerations.csv"
OUTPUT_DIR  = REPO_ROOT / "data" / "processed" / "lancedb_courts"

EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"
EMBED_DIM        = 768

# ── Args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=256,
                    help="Rows per GPU embedding batch (default 256)")
parser.add_argument("--sample", type=int, default=None,
                    help="Only embed first N rows (for testing)")
parser.add_argument("--chunk-size", type=int, default=50_000,
                    help="CSV read chunk size (default 50000)")
parser.add_argument("--courts-csv", type=Path, default=COURTS_CSV,
                    help="Path to court_considerations.csv")
parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                    help="LanceDB output directory")
parser.add_argument("--overwrite", action="store_true",
                    help="Delete existing index and rebuild from scratch")
args = parser.parse_args()

# ── Validate inputs ───────────────────────────────────────────────────────
if not args.courts_csv.exists():
    print(f"ERROR: Courts CSV not found: {args.courts_csv}")
    sys.exit(1)

csv_size_gb = args.courts_csv.stat().st_size / 1e9
print(f"Courts CSV: {args.courts_csv}  ({csv_size_gb:.2f} GB)")

if args.output_dir.exists() and not args.overwrite:
    print(f"Output already exists: {args.output_dir}")
    print("Pass --overwrite to rebuild. Exiting.")
    sys.exit(0)

if args.overwrite and args.output_dir.exists():
    import shutil
    shutil.rmtree(args.output_dir)
    print(f"Deleted existing index at {args.output_dir}")

args.output_dir.mkdir(parents=True, exist_ok=True)

# ── Load embedding model ──────────────────────────────────────────────────
print(f"\nLoading embedding model: {EMBED_MODEL_NAME}")
from sentence_transformers import SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")
if device == "cpu":
    print("  WARNING: running on CPU, this will take ~10x longer")

embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
embed_model.max_seq_length = 512
print(f"  Model loaded  (dim={EMBED_DIM}, max_seq=512)")

# ── Connect to LanceDB ────────────────────────────────────────────────────
print(f"\nConnecting to LanceDB at: {args.output_dir}")
import lancedb
import pyarrow as pa

db = lancedb.connect(str(args.output_dir))

# Define schema
schema = pa.schema([
    pa.field("citation", pa.string()),
    pa.field("text",     pa.string()),
    pa.field("vector",   pa.list_(pa.float32(), EMBED_DIM)),
])

# ── Stream CSV → embed → write ────────────────────────────────────────────
print(f"\nStreaming CSV in chunks of {args.chunk_size:,} rows...")
print(f"Embedding batch size: {args.batch_size}")
if args.sample:
    print(f"SAMPLE MODE: first {args.sample:,} rows only")

table = None
total_rows = 0
total_written = 0
t0 = time.time()
last_report = t0

reader = pd.read_csv(
    args.courts_csv,
    usecols=["citation", "text"],
    chunksize=args.chunk_size,
    dtype=str,
)

for chunk_num, chunk in enumerate(reader):
    chunk = chunk.fillna("")
    chunk = chunk[chunk["citation"].str.strip() != ""]
    chunk = chunk[chunk["text"].str.strip() != ""]

    if args.sample and total_rows + len(chunk) > args.sample:
        chunk = chunk.iloc[: args.sample - total_rows]

    total_rows += len(chunk)

    citations = chunk["citation"].tolist()
    texts     = chunk["text"].tolist()

    # Embed in sub-batches
    all_vectors = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i : i + args.batch_size]
        # multilingual-e5-base expects "query: " or "passage: " prefix
        prefixed = ["passage: " + t[:512] for t in batch]
        vecs = embed_model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        all_vectors.append(vecs)

    vectors = np.vstack(all_vectors).astype(np.float32)

    # Build PyArrow batch
    pa_batch = pa.table({
        "citation": pa.array(citations, type=pa.string()),
        "text":     pa.array(texts,     type=pa.string()),
        "vector":   pa.array(
            [v.tolist() for v in vectors],
            type=pa.list_(pa.float32(), EMBED_DIM),
        ),
    })

    if table is None:
        table = db.create_table("courts", data=pa_batch, schema=schema, mode="overwrite")
    else:
        table.add(pa_batch)

    total_written += len(pa_batch)

    # Progress report every 30s
    now = time.time()
    if now - last_report >= 30 or chunk_num == 0:
        elapsed = now - t0
        rate = total_written / elapsed if elapsed > 0 else 0
        eta_s  = (2_500_000 - total_written) / rate if rate > 0 else 0
        print(
            f"  [{elapsed/60:5.1f} min]  {total_written:>8,} rows written"
            f"  ({rate:.0f} rows/s)"
            f"  ETA {eta_s/60:.1f} min"
        )
        last_report = now

    if args.sample and total_rows >= args.sample:
        break

# ── Build ANN index (IVF-PQ) ─────────────────────────────────────────────
print(f"\nBuilding ANN index (IVF-PQ) on {total_written:,} vectors...")
print("  This takes 2-5 minutes...")
t_idx = time.time()
table.create_index(
    metric="cosine",
    num_partitions=256,    # IVF clusters
    num_sub_vectors=96,    # PQ sub-vectors (768 / 96 = 8-dim each)
)
print(f"  ANN index built in {time.time() - t_idx:.1f}s")

# ── Summary ───────────────────────────────────────────────────────────────
elapsed = time.time() - t0
disk_gb = sum(f.stat().st_size for f in args.output_dir.rglob("*") if f.is_file()) / 1e9

print(f"\n{'='*60}")
print(f"  Done!")
print(f"  Rows written  : {total_written:,}")
print(f"  Total time    : {elapsed/60:.1f} minutes")
print(f"  Disk usage    : {disk_gb:.2f} GB")
print(f"  Output        : {args.output_dir}")
print(f"{'='*60}")
print(f"\nVerification:")
db2    = lancedb.connect(str(args.output_dir))
tbl2   = db2.open_table("courts")
print(f"  Table row count : {tbl2.count_rows():,}")
sample = tbl2.head(2).to_pandas()
print("  Sample rows:")
print(sample[["citation"]])

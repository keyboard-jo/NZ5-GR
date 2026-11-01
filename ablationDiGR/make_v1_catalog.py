#!/usr/bin/env python
"""Rebuild the v1 SID as a parquet catalogue, from an archive's .npy pair.

    python make_v1_catalog.py --data_dir ~/NZ5/.../digr_data --out v1_catalog.parquet

Why this exists: the archived `digr_data` directories carry the SID codes as
`catalog_sid_codes.npy` plus `catalog_item_ids.npy`, and the preprocessor reads a
parquet or an RQ-VAE json, never that pair. The pair is also the authority on what
the archived checkpoints were trained against, so an ablation arm that wants "the
v1 SID" has to use these bytes and not some other artefact that happens to exist
in the same archive tree.

An RQ-VAE json is not a substitute even when it comes from the same run: the
quantiser can be exported with its raw codebook indices while the archive holds a
pruned or remapped codebook, and the two then disagree on both the number of codes
per level and the code assigned to a given item. Verified arms are built from the
arrays, because those are the bytes the numbers came from.

Row order is preserved exactly: the catalogue row index is the label `y`, so
sorting or reindexing here would silently relabel every training target.

Needs numpy and pyarrow only.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List

import numpy as np


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data_dir", required=True,
                   help="archive digr_data holding catalog_item_ids.npy and "
                        "catalog_sid_codes.npy")
    p.add_argument("--out", required=True, help="parquet to write")
    p.add_argument("--item_col", default="item_id")
    p.add_argument("--expect_sizes", default="",
                   help="comma separated, e.g. 256,256,256,256,731. Warns on a "
                        "difference and exits 1. Default: just report them.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def load_arrays(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    codes = np.load(f"{data_dir}/catalog_sid_codes.npy", mmap_mode="r")
    ids = np.load(f"{data_dir}/catalog_item_ids.npy", mmap_mode="r")
    if codes.ndim != 2:
        raise SystemExit(f"catalog_sid_codes.npy must be 2-D, got {codes.shape}")
    if ids.ndim != 1:
        raise SystemExit(f"catalog_item_ids.npy must be 1-D, got {ids.shape}")
    if len(ids) != len(codes):
        raise SystemExit(
            f"length mismatch: {len(ids):,} item ids vs {len(codes):,} code rows")
    return codes, ids


def code_sizes(codes: np.ndarray) -> List[int]:
    """max(code) + 1 per level. Reported, never used to reshape the codes."""
    return [int(codes[:, l].max()) + 1 for l in range(codes.shape[1])]


def main(argv=None) -> int:
    args = parse_args(argv)

    import pyarrow as pa
    import pyarrow.parquet as pq

    codes, ids = load_arrays(args.data_dir)
    sizes = code_sizes(codes)
    n, levels = codes.shape

    dup = len(ids) - len(np.unique(np.asarray(ids)))
    print(f"items         {n:,}")
    print(f"levels        {levels}")
    print(f"code sizes    {sizes}  (max code + 1 per level)")
    print(f"duplicate ids {dup}")
    if dup:
        print("warn: duplicate item ids; the catalogue row is the label, so a "
              "duplicate id is not fatal but the ids are not a key", file=sys.stderr)

    # unused codes are normal for a residual quantiser, and they are the reason a
    # json from the same run can look different: say it rather than hide it
    for l in range(levels):
        used = np.unique(np.asarray(codes[:, l])).size
        if used != sizes[l]:
            print(f"  level {l}: {used:,} distinct codes of {sizes[l]:,} "
                  f"(codes {used}..{sizes[l] - 1} unused)")

    want = [int(t) for t in args.expect_sizes.split(",") if t.strip()]
    if want and want != sizes:
        print(f"MISMATCH code sizes {sizes} != expected {want}", file=sys.stderr)
        print("this is not the SID you think it is; do not build an arm on it",
              file=sys.stderr)
        return 1

    if ids.dtype == object or ids.dtype.kind in {"U", "S", "O"}:
        # archived ids can be python object arrays of strings; pyarrow wants a
        # concrete type, and ints are what the preprocessor normalises to anyway
        try:
            item_ids = np.asarray([int(v) for v in ids], dtype=np.int64)
        except (TypeError, ValueError):
            item_ids = np.asarray([str(v) for v in ids], dtype=object)
    else:
        item_ids = np.asarray(ids)

    cols: Dict[str, pa.Array] = {args.item_col: pa.array(item_ids)}
    for l in range(levels):
        cols[f"sid_{l + 1}"] = pa.array(np.asarray(codes[:, l], dtype=np.int32))

    table = pa.table(cols)
    import os
    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"{args.out} exists; pass --overwrite to replace it")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pq.write_table(table, args.out)
    print(f"wrote         {args.out}  ({table.num_rows:,} rows, "
          f"{table.num_columns} columns: {table.column_names[:3]}...)")

    meta = {
        "source": args.data_dir,
        "size": int(n),
        "num_levels": int(levels),
        "sid_sizes": sizes,
        "duplicate_ids": int(dup),
    }
    with open(args.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

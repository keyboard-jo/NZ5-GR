"""preprocess_candidate_digr.py -- turn a candidate pool of OIDs into catalog SIDs.

The candidate pool (`candidate.parquet`) lists OIDs, while training predicts rows
of the catalog built by `preprocess_digr.py` (whose ids are RIDs).  This script
makes the conversion explicit and reports every row that cannot be converted:

    1. path: candidate_pool/candidate.parquet  is on this machine at data/candidate.parquet
    2. data/exported_indexer/i.json maps OID -> RID (4.78M string keys, int values)
    3. digr_data/catalog_sid_codes.npy maps the RID -> its SID codes.
       Reading the catalog arrays instead of re-parsing index_rqvae_prod.json
       guarantees the same code space (and the same row order) as training.

Two hops can fail, and both are counted, logged and written out:

    no_rid             the OID is not a key of i.json (new / cold-start item)
    rid_not_in_catalog the RID is absent from the catalog used for training

Outputs written to `--output_dir` (default `digr_candidates/`):

    candidate_sid_codes.npy     int32  [K, L]  SID codes of the kept candidates
    candidate_catalog_rows.npy  int64  [K]     row in digr_data/catalog_item_ids.npy
    candidate_item_ids.npy      int64  [K]     the original OID
    candidate_rids.npy          int64  [K]     RID from i.json
    candidate_source_rows.npy   int64  [K]     row index in the candidate parquet
    is_candidate.npy            uint8  [|I|]   1 when that catalog row is a candidate
    candidate_missing_item_ids.npy  int64 [M]  OIDs that produced no SID
    candidate_missing_reasons.npy   int8  [M]  1 = no_rid, 2 = rid_not_in_catalog
    candidate_meta.json         counts, reasons, coverage

Every kept array is in source order, so row i of `candidate_sid_codes.npy` is
row `candidate_source_rows.npy[i]` of the parquet.  Rows are never dropped
silently: `candidate_meta.json["counts"]` adds up to `rows_read`.

Usage
-----
    python preprocess_candidate_digr.py \
        --candidate_path data/candidate.parquet \
        --i_json_path data/exported_indexer/i.json \
        --data_dir digr_data \
        --output_dir digr_candidates
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:
    from preprocess_digr import (  # noqa: E402  (path fix above)
        _INT64_SENTINEL,
        ItemCatalog,
        _col_to_numpy,
        _find_col,
        human_int,
        log_banner,
        log_card,
        normalize_ids,
        setup_logger,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "preprocess_digr.py must live next to preprocess_candidate_digr.py "
        "(shared catalog types)."
    ) from exc

MANIFEST_NAME = "candidate_meta.json"
OID_COL_NAMES = ("item_id", "oid", "anonymous_cid", "cid", "candidate_id", "id")
RID_NOT_FOUND = 1
RID_NOT_IN_CATALOG = 2


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Map a candidate pool (OID) to catalog SIDs (OID -> RID -> SID).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--candidate_path", "--candidates_path", dest="candidate_path",
                   default="data/candidate.parquet",
                   help="parquet holding one row per candidate item")
    p.add_argument("--i_json_path", default="data/exported_indexer/i.json",
                   help="OID -> RID json (string keys, int values)")
    p.add_argument("--data_dir", default="digr_data",
                   help="preprocessed dataset dir; supplies catalog_item_ids.npy and "
                        "catalog_sid_codes.npy (run preprocess_digr.py first)")
    p.add_argument("--output_dir", default="digr_candidates",
                   help="where the SID arrays and candidate_meta.json are written")
    p.add_argument("--item_id_col", default=None,
                   help="candidate column holding the OID "
                        f"(default: auto-detect from {', '.join(OID_COL_NAMES)})")
    p.add_argument("--max_rows", type=int, default=None,
                   help="only read the first N candidate rows (a quick trial run)")
    p.add_argument("--overwrite", action="store_true",
                   help="replace --output_dir if it already holds a run")
    p.add_argument("--log_dir", default=None,
                   help="also write the log to <log_dir>/preprocess_candidate.log")
    args = p.parse_args(argv)
    if args.max_rows is not None and int(args.max_rows) < 1:
        p.error("--max_rows must be >= 1")
    return args


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _validate_paths(args: argparse.Namespace) -> None:
    for flag, path in (("--candidate_path", args.candidate_path),
                       ("--i_json_path", args.i_json_path)):
        if not path:
            raise SystemExit(f"{flag} is required")
        if not os.path.isfile(path):
            raise SystemExit(f"{flag}: file not found: {path}")
    if not args.data_dir or not os.path.isdir(args.data_dir):
        raise SystemExit(
            f"--data_dir: directory not found: {args.data_dir} "
            "(run preprocess_digr.py first)"
        )
    for name in ("catalog_item_ids.npy", "catalog_sid_codes.npy", "catalog_meta.json"):
        if not os.path.isfile(os.path.join(args.data_dir, name)):
            raise SystemExit(
                f"--data_dir {args.data_dir} has no {name}; the catalog comes from "
                "preprocess_digr.py, so build the dataset before the candidate pool"
            )


def _check_output_dir(out_dir: str, overwrite: bool) -> None:
    if overwrite:
        if os.path.isdir(out_dir):
            import shutil
            shutil.rmtree(out_dir)
        return
    if os.path.isdir(out_dir):
        existing = [f for f in (MANIFEST_NAME, "candidate_sid_codes.npy")
                    if os.path.exists(os.path.join(out_dir, f))]
        if existing:
            raise SystemExit(
                f"{out_dir} already contains candidate data ({', '.join(existing)}); "
                "pass --overwrite to replace it"
            )


def read_candidate_ids(path: str, item_id_col: Optional[str], max_rows: Optional[int],
                       logger) -> Tuple[np.ndarray, str]:
    """-> (OID int64 array with _INT64_SENTINEL for nulls, column name used)."""
    logger.info(f"  > reading candidate ids: {path}")
    names = pq.ParquetFile(path).schema_arrow.names
    col = item_id_col or _find_col(names, OID_COL_NAMES)
    if col is None:
        raise SystemExit(
            f"could not find an item-id column in {path}; columns={names}. "
            "Pass --item_id_col."
        )
    table = pq.read_table(path, columns=[col])
    if max_rows is not None:
        table = table.slice(0, int(max_rows))
    ids, is_int = normalize_ids(_col_to_numpy(table.column(col)))
    if not is_int:
        raise SystemExit(
            f"{path}[{col}] holds non-integer ids; i.json keys are integers, so the "
            "candidate ids must be integers too"
        )
    logger.info(f"  + {human_int(len(ids))} candidate rows from column '{col}' "
                f"(null ids: {human_int(int((ids == _INT64_SENTINEL).sum()))})")
    return ids, col


def load_oid_to_rid(path: str, logger) -> Tuple[np.ndarray, np.ndarray]:
    """-> (sorted int64 OID keys, RID values) so a lookup is one searchsorted."""
    logger.info(f"  > reading OID -> RID map: {path}")
    t0 = time.time()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a JSON object of {{OID: RID}}, "
                         f"got {type(data).__name__}")
    try:
        keys = np.fromiter((int(k) for k in data.keys()), dtype=np.int64, count=len(data))
        vals = np.fromiter((int(v) for v in data.values()), dtype=np.int64, count=len(data))
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{path}: every key and value must be an integer ({exc})") from exc
    if keys.size == 0:
        raise SystemExit(f"{path} is empty")
    order = np.argsort(keys, kind="stable")
    keys, vals = keys[order], vals[order]
    dup = int((keys[1:] == keys[:-1]).sum()) if keys.size > 1 else 0
    if dup:
        raise SystemExit(f"{path}: {human_int(dup)} duplicated OID key(s); "
                         "the map must be one RID per OID")
    logger.info(f"  + {human_int(keys.size)} OID -> RID entries "
                f"(RIDs {human_int(int(vals.min()))} .. {human_int(int(vals.max()))}) "
                f"in {time.time() - t0:.1f}s")
    return keys, vals


def _lookup_sorted(keys: np.ndarray, vals: np.ndarray, query: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """-> (value or -1 when absent, found mask) for a sorted key array."""
    if keys.size == 0:
        return np.full(len(query), -1, dtype=np.int64), np.zeros(len(query), dtype=bool)
    pos = np.searchsorted(keys, query)
    pos_clip = np.clip(pos, 0, keys.size - 1)
    found = keys[pos_clip] == query
    out = np.where(found, vals[pos_clip], np.int64(-1))
    return out.astype(np.int64), found


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert(args, logger) -> Dict[str, Any]:
    t_all = time.time()
    _validate_paths(args)
    out_dir = args.output_dir
    _check_output_dir(out_dir, args.overwrite)
    os.makedirs(out_dir, exist_ok=True)

    catalog = ItemCatalog.load(args.data_dir)
    logger.info(f"  + catalog: {human_int(catalog.size)} items x "
                f"{catalog.num_levels} SID levels (from {args.data_dir})")

    oids, id_col = read_candidate_ids(args.candidate_path, args.item_id_col,
                                      args.max_rows, logger)
    keys, vals = load_oid_to_rid(args.i_json_path, logger)

    rows_read = int(len(oids))
    null_ids = oids == _INT64_SENTINEL
    n_null = int(null_ids.sum())

    # hop 1: OID -> RID
    rids, has_rid = _lookup_sorted(keys, vals, oids)
    has_rid &= ~null_ids
    # hop 2: RID -> catalog row
    cat_rows = catalog.lookup(rids)
    has_row = has_rid & (cat_rows >= 0)

    # one candidate per catalog row: keep the first source row, count the rest
    keep = np.flatnonzero(has_row)
    seen: Dict[int, int] = {}
    for r, item in zip(keep.tolist(), cat_rows[keep].tolist()):
        if item not in seen:
            seen[item] = r
    kept = np.asarray(sorted(seen.values()), dtype=np.int64)
    duplicates = int(len(keep) - len(kept))

    missing = np.flatnonzero(~has_row & ~null_ids)
    reasons = np.where(has_rid[missing], RID_NOT_IN_CATALOG, RID_NOT_FOUND).astype(np.int8)

    kept_rows = cat_rows[kept].astype(np.int64)
    sid_codes = np.asarray(catalog.sid_codes[kept_rows], dtype=np.int32)

    np.save(os.path.join(out_dir, "candidate_sid_codes.npy"), sid_codes)
    np.save(os.path.join(out_dir, "candidate_catalog_rows.npy"), kept_rows)
    np.save(os.path.join(out_dir, "candidate_item_ids.npy"), oids[kept].astype(np.int64))
    np.save(os.path.join(out_dir, "candidate_rids.npy"), rids[kept].astype(np.int64))
    np.save(os.path.join(out_dir, "candidate_source_rows.npy"), kept)
    is_candidate = np.zeros(catalog.size, dtype=np.uint8)
    is_candidate[kept_rows] = 1
    np.save(os.path.join(out_dir, "is_candidate.npy"), is_candidate)
    np.save(os.path.join(out_dir, "candidate_missing_item_ids.npy"),
            oids[missing].astype(np.int64))
    np.save(os.path.join(out_dir, "candidate_missing_reasons.npy"), reasons)

    counts = {
        "rows_read": rows_read,
        "null_ids": n_null,
        "no_rid": int((reasons == RID_NOT_FOUND).sum()),
        "rid_not_in_catalog": int((reasons == RID_NOT_IN_CATALOG).sum()),
        "duplicate_rids": duplicates,
        "kept": int(len(kept)),
    }
    if sum(counts[k] for k in ("null_ids", "no_rid", "rid_not_in_catalog",
                               "duplicate_rids", "kept")) != rows_read:
        raise AssertionError(f"candidate accounting does not add up: {counts}")

    meta = {
        "created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "sources": {
            "candidate_path": os.path.abspath(args.candidate_path),
            "i_json_path": os.path.abspath(args.i_json_path),
            "data_dir": os.path.abspath(args.data_dir),
        },
        "item_id_column": id_col,
        "catalog_size": int(catalog.size),
        "num_levels": int(catalog.num_levels),
        "counts": counts,
        "coverage": {
            "of_candidate_rows": round(len(kept) / max(1, rows_read), 6),
            "of_catalog": round(len(kept) / max(1, catalog.size), 6),
        },
        "reason_codes": {RID_NOT_FOUND: "no_rid",
                         RID_NOT_IN_CATALOG: "rid_not_in_catalog"},
        "elapsed_seconds": round(time.time() - t_all, 1),
    }
    with open(os.path.join(out_dir, MANIFEST_NAME), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log_card(logger, "candidate pool -> SID", {
        "Candidate rows": human_int(rows_read),
        "Kept (has a SID)": f"{human_int(len(kept))} "
                            f"({100.0 * len(kept) / max(1, rows_read):.2f}%)",
        "No RID in i.json": human_int(counts["no_rid"]),
        "RID not in catalog": human_int(counts["rid_not_in_catalog"]),
        "Duplicate RIDs": human_int(duplicates),
        "Null candidate ids": human_int(n_null),
        "Coverage of catalog": f"{100.0 * len(kept) / max(1, catalog.size):.2f}%",
        "Output": out_dir,
        "Elapsed": f"{time.time() - t_all:.1f}s",
    })
    if counts["no_rid"]:
        logger.info(f"  ! {human_int(counts['no_rid'])} OIDs are not keys of "
                    f"{args.i_json_path}; they get no SID and stay in "
                    "candidate_missing_item_ids.npy (reasons code 1)")
    if counts["rid_not_in_catalog"]:
        logger.info(f"  ! {human_int(counts['rid_not_in_catalog'])} RIDs are absent from "
                    f"the catalog in {args.data_dir}; reasons code 2")
    return meta


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = setup_logger(args.log_dir)
    log_banner(logger, "DiGR CANDIDATE POOL")
    try:
        convert(args, logger)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error(f"candidate preprocessing failed: {exc}")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())

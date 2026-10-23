#!/usr/bin/env python3
"""
Unit tests for `preprocess_candidate_digr.py` (numpy + pyarrow only, no torch).

Run from the repo root:

    python -m pytest "Model Training/test_preprocess_candidate_digr.py" -q

The synthetic pool is built so every rule has exactly one obvious outcome.
OIDs are deliberately large and unrelated to the RIDs, and the candidate rows
are deliberately not in catalog order:

    source row 0   OID ...003 -> RID 3  -> catalog row 2      kept
    source row 1   OID ...999 -> absent from i.json           no_rid
    source row 2   OID ...001 -> RID 1  -> catalog row 0      kept
    source row 3   OID ...007 -> RID 99 -> not in catalog     rid_not_in_catalog
    source row 4   OID ...003 -> RID 3  -> already seen       duplicate
    source row 5   null      -> nothing to look up            null_ids
    source row 6   OID ...005 -> RID 5  -> catalog row 4      kept

so 7 rows in, 3 kept, and every dropped row has its own reason.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import preprocess_candidate_digr as pc  # noqa: E402
import preprocess_digr as pp  # noqa: E402

DATA_ROOT = os.path.join(HERE, "_pytest_data_candidates")
NUM_LEVELS = 5

# RID -> SID codes, 4 semantic levels + 1 collision rank like the production index.
CATALOG = {
    "1": [0, 1, 2, 0, 0],
    "2": [1, 0, 2, 3, 0],
    "3": [2, 1, 0, 3, 1],
    "4": [3, 3, 3, 3, 0],
    "5": [4, 2, 1, 0, 0],
    "6": [0, 4, 4, 4, 2],
}

# OID -> RID.  OID ...007 points at a RID that the catalog does not contain.
I_JSON = {
    "90000000001": 1,
    "90000000002": 2,
    "90000000003": 3,
    "90000000004": 4,
    "90000000005": 5,
    "90000000006": 6,
    "90000000007": 99,
}

OID_UNKNOWN = 90000000999          # not a key of I_JSON
CANDIDATE_IDS = [
    90000000003,      # -> RID 3 -> catalog row 2   kept
    OID_UNKNOWN,      # no_rid
    90000000001,      # -> RID 1 -> catalog row 0   kept
    90000000007,      # -> RID 99, absent from catalog
    90000000003,      # duplicate of source row 0
    None,             # null id
    90000000005,      # -> RID 5 -> catalog row 4   kept
]
KEPT_SOURCE_ROWS = [0, 2, 6]
KEPT_CATALOG_ROWS = [2, 0, 4]
KEPT_OIDS = [90000000003, 90000000001, 90000000005]


# ---------------------------------------------------------------------------
# mock data
# ---------------------------------------------------------------------------

def _rmtree(path: str) -> None:
    for _ in range(20):
        gc.collect()
        shutil.rmtree(path, ignore_errors=True)
        if not os.path.exists(path):
            return
        time.sleep(0.05)
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def workspace():
    """Fresh `_pytest_data_candidates/` tree inside the project (never /tmp)."""
    _rmtree(DATA_ROOT)
    os.makedirs(DATA_ROOT, exist_ok=True)
    yield DATA_ROOT
    _rmtree(DATA_ROOT)


def _write_parquet(path: str, rows, schema=None) -> None:
    if schema is None:
        schema = pa.schema([("item_id", pa.int64()), ("decoy", pa.int64())])
        table = pa.table({"item_id": [r[0] for r in rows],
                          "decoy": [r[1] for r in rows]}, schema=schema)
    else:
        table = pa.table(rows, schema=schema)
    pq.write_table(table, path)


def _setup(workspace: str) -> dict:
    """catalog npy + candidate parquet + i.json, all under `workspace`."""
    data_dir = os.path.join(workspace, "digr_data")
    os.makedirs(data_dir, exist_ok=True)
    rids = np.array([int(k) for k in CATALOG], dtype=np.int64)
    codes = np.array([CATALOG[str(int(r))] for r in rids], dtype=np.int32)
    pp.ItemCatalog(rids, True, codes).save(data_dir)

    candidate_path = os.path.join(workspace, "candidate.parquet")
    _write_parquet(candidate_path, [(oid, i) for i, oid in enumerate(CANDIDATE_IDS)])

    i_json_path = os.path.join(workspace, "i.json")
    with open(i_json_path, "w", encoding="utf-8") as f:
        json.dump(I_JSON, f)

    return {"data_dir": data_dir, "candidate": candidate_path, "i_json": i_json_path}


def _argv(paths: dict, workspace: str, extra=(), overwrite: bool = True) -> list:
    argv = [
        "--candidate_path", paths["candidate"],
        "--i_json_path", paths["i_json"],
        "--data_dir", paths["data_dir"],
        "--output_dir", os.path.join(workspace, "out"),
    ] + list(extra)
    if overwrite:
        argv.append("--overwrite")
    return argv


def _run(paths: dict, workspace: str, extra=(), overwrite: bool = True):
    args = pc.parse_args(_argv(paths, workspace, extra, overwrite))
    log = pc.setup_logger()
    try:
        meta = pc.convert(args, log)
    finally:
        # Windows: an open FileHandler keeps the log file locked.
        for handler in list(log.handlers):
            handler.close()
            log.removeHandler(handler)
    return os.path.join(workspace, "out"), meta


def _load(out_dir: str, name: str):
    return np.load(os.path.join(out_dir, name))


def _meta(out_dir: str) -> dict:
    with open(os.path.join(out_dir, pc.MANIFEST_NAME), "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. the happy path
# ---------------------------------------------------------------------------

def test_kept_candidates_carry_the_catalog_sid(workspace):
    paths = _setup(workspace)
    out_dir, meta = _run(paths, workspace)

    sids = _load(out_dir, "candidate_sid_codes.npy")
    assert sids.shape == (3, NUM_LEVELS)
    assert sids.tolist() == [CATALOG["3"], CATALOG["1"], CATALOG["5"]]


def test_kept_candidates_keep_their_source_order(workspace):
    """Row i of every kept array describes source row `candidate_source_rows[i]`."""
    paths = _setup(workspace)
    out_dir, _ = _run(paths, workspace)

    assert _load(out_dir, "candidate_source_rows.npy").tolist() == KEPT_SOURCE_ROWS
    assert _load(out_dir, "candidate_catalog_rows.npy").tolist() == KEPT_CATALOG_ROWS
    assert _load(out_dir, "candidate_item_ids.npy").tolist() == KEPT_OIDS
    assert _load(out_dir, "candidate_rids.npy").tolist() == [3, 1, 5]


def test_catalog_row_is_the_label_of_the_sid(workspace):
    """The SID must be `catalog_sid_codes[catalog_row]`, the label space of training."""
    paths = _setup(workspace)
    out_dir, _ = _run(paths, workspace)

    catalog_sids = np.load(os.path.join(paths["data_dir"], "catalog_sid_codes.npy"))
    rows = _load(out_dir, "candidate_catalog_rows.npy")
    sids = _load(out_dir, "candidate_sid_codes.npy")
    assert np.array_equal(sids, catalog_sids[rows])


def test_is_candidate_marks_the_kept_catalog_rows(workspace):
    paths = _setup(workspace)
    out_dir, _ = _run(paths, workspace)

    mask = _load(out_dir, "is_candidate.npy")
    assert mask.dtype == np.uint8
    assert len(mask) == len(CATALOG)
    assert np.flatnonzero(mask).tolist() == sorted(KEPT_CATALOG_ROWS)


# ---------------------------------------------------------------------------
# 2. every dropped row has a reason
# ---------------------------------------------------------------------------

def test_counts_add_up(workspace):
    paths = _setup(workspace)
    out_dir, meta = _run(paths, workspace)

    counts = meta["counts"]
    assert counts == {
        "rows_read": 7,
        "null_ids": 1,
        "no_rid": 1,
        "rid_not_in_catalog": 1,
        "duplicate_rids": 1,
        "kept": 3,
    }
    assert sum(v for k, v in counts.items() if k != "rows_read") == counts["rows_read"]
    assert _meta(out_dir)["counts"] == counts


def test_unknown_oid_and_unknown_rid_are_kept_apart(workspace):
    paths = _setup(workspace)
    out_dir, _ = _run(paths, workspace)

    missing = _load(out_dir, "candidate_missing_item_ids.npy")
    reasons = _load(out_dir, "candidate_missing_reasons.npy")
    assert missing.tolist() == [OID_UNKNOWN, 90000000007]
    assert reasons.tolist() == [pc.RID_NOT_FOUND, pc.RID_NOT_IN_CATALOG]
    assert pc.RID_NOT_FOUND == 1 and pc.RID_NOT_IN_CATALOG == 2


def test_duplicate_rid_keeps_the_first_row_and_is_counted(workspace):
    paths = _setup(workspace)
    out_dir, meta = _run(paths, workspace)

    assert meta["counts"]["duplicate_rids"] == 1
    # source row 0 wins over source row 4, both point at catalog row 2
    assert _load(out_dir, "candidate_source_rows.npy").tolist() == KEPT_SOURCE_ROWS
    assert _load(out_dir, "candidate_catalog_rows.npy").tolist().count(2) == 1


def test_null_candidate_id_is_not_a_lookup(workspace):
    """A null OID is reported as `null_ids`, never as a missing i.json entry."""
    paths = _setup(workspace)
    out_dir, meta = _run(paths, workspace)

    assert meta["counts"]["null_ids"] == 1
    assert OID_UNKNOWN in _load(out_dir, "candidate_missing_item_ids.npy").tolist()


def test_coverage_is_reported(workspace):
    paths = _setup(workspace)
    out_dir, meta = _run(paths, workspace)

    assert meta["coverage"]["of_candidate_rows"] == round(3 / 7, 6)
    assert meta["coverage"]["of_catalog"] == round(3 / 6, 6)
    assert meta["catalog_size"] == 6
    assert meta["num_levels"] == NUM_LEVELS
    assert meta["item_id_column"] == "item_id"


# ---------------------------------------------------------------------------
# 3. the two hops in isolation
# ---------------------------------------------------------------------------

def test_i_json_lookup_is_an_oid_lookup_not_a_row_number(workspace):
    """OIDs are huge and unrelated to RIDs, so a positional read would be wrong."""
    paths = _setup(workspace)
    keys, vals = pc.load_oid_to_rid(paths["i_json"], pc.setup_logger())
    assert keys.tolist() == sorted(int(k) for k in I_JSON)
    rids, found = pc._lookup_sorted(keys, vals, np.array([90000000003, 90000000007,
                                                          OID_UNKNOWN], dtype=np.int64))
    assert rids.tolist() == [3, 99, -1]
    assert found.tolist() == [True, True, False]


def test_read_candidate_ids_uses_the_declared_column(workspace):
    paths = _setup(workspace)
    ids, col = pc.read_candidate_ids(paths["candidate"], None, None, pc.setup_logger())
    assert col == "item_id"
    raw = ids.tolist()
    assert raw[:3] == [90000000003, OID_UNKNOWN, 90000000001]
    assert raw[5] == pp._INT64_SENTINEL     # null id keeps the sentinel


def test_item_id_column_can_be_named(workspace):
    paths = _setup(workspace)
    custom = os.path.join(workspace, "candidate_custom.parquet")
    pq.write_table(pa.table({"candidate_key": [90000000003, 90000000001]},
                            schema=pa.schema([("candidate_key", pa.int64())])), custom)
    paths["candidate"] = custom
    out_dir, meta = _run(paths, workspace, ["--item_id_col", "candidate_key"])
    assert meta["item_id_column"] == "candidate_key"
    assert meta["counts"]["kept"] == 2
    assert _load(out_dir, "candidate_catalog_rows.npy").tolist() == [2, 0]


def test_missing_item_id_column_is_an_error(workspace):
    paths = _setup(workspace)
    bad = os.path.join(workspace, "candidate_bad.parquet")
    pq.write_table(pa.table({"foo": [1, 2]}, schema=pa.schema([("foo", pa.int64())])), bad)
    paths["candidate"] = bad
    with pytest.raises(SystemExit, match="could not find an item-id column"):
        _run(paths, workspace)


def test_duplicate_oids_in_i_json_are_rejected(workspace):
    """'1' and '01' are the same integer, so the map would be ambiguous."""
    paths = _setup(workspace)
    with open(paths["i_json"], "w", encoding="utf-8") as f:
        json.dump({"1": 1, "01": 2}, f)
    with pytest.raises(SystemExit, match="duplicated OID key"):
        pc.load_oid_to_rid(paths["i_json"], pc.setup_logger())


def test_non_integer_i_json_value_is_an_error(workspace):
    paths = _setup(workspace)
    with open(paths["i_json"], "w", encoding="utf-8") as f:
        json.dump({"90000000001": "not-a-rid"}, f)
    with pytest.raises(SystemExit, match="must be an integer"):
        pc.load_oid_to_rid(paths["i_json"], pc.setup_logger())


# ---------------------------------------------------------------------------
# 4. args, paths, reruns
# ---------------------------------------------------------------------------

def test_max_rows_reads_a_prefix(workspace):
    paths = _setup(workspace)
    out_dir, meta = _run(paths, workspace, ["--max_rows", "3"])

    assert meta["counts"]["rows_read"] == 3
    assert meta["counts"]["kept"] == 2
    assert meta["counts"]["no_rid"] == 1
    assert _load(out_dir, "candidate_catalog_rows.npy").tolist() == [2, 0]


def test_existing_output_dir_needs_overwrite(workspace):
    paths = _setup(workspace)
    _run(paths, workspace)
    with pytest.raises(SystemExit, match="already contains candidate data"):
        _run(paths, workspace, overwrite=False)


def test_missing_data_dir_is_an_error(workspace):
    paths = _setup(workspace)
    paths["data_dir"] = os.path.join(workspace, "not_built")
    with pytest.raises(SystemExit, match="directory not found"):
        _run(paths, workspace)


def test_empty_data_dir_points_at_preprocess_digr(workspace):
    paths = _setup(workspace)
    empty = os.path.join(workspace, "empty")
    os.makedirs(empty, exist_ok=True)
    paths["data_dir"] = empty
    with pytest.raises(SystemExit, match="preprocess_digr.py"):
        _run(paths, workspace)


def test_missing_candidate_file_is_an_error(workspace):
    paths = _setup(workspace)
    paths["candidate"] = os.path.join(workspace, "nope.parquet")
    with pytest.raises(SystemExit, match="file not found"):
        _run(paths, workspace)


def test_parse_args_defaults():
    args = pc.parse_args([])
    assert args.candidate_path == "data/candidate.parquet"
    assert args.i_json_path == "data/exported_indexer/i.json"
    assert args.data_dir == "digr_data"
    assert args.output_dir == "digr_candidates"
    assert args.item_id_col is None
    assert args.max_rows is None
    assert args.overwrite is False

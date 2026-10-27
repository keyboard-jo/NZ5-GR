#!/usr/bin/env python3
"""
Unit tests for `Plan 2/preprocess_digr_v2.py` (numpy + pyarrow only, no GPU).

Run from the repo root:

    python -m pytest "Plan 2/test_preprocess_digr_v2.py" -q

The point of v2 is one line of layout, so the tests are about that line:

  v1  (T, S1..SL)                 1 + L tokens per event
  v2  (T, A, [C], S1..SL)         1 + A + C + L tokens per event

What is pinned here:
  * the row length is exactly `1 + M + T * event_width`;
  * the action token is in the slot right after the gap token, with
    exposure = 0, click = 1, unknown = 2, and no other value;
  * the codes stay last inside the event slot, so "the last L tokens of a row
    are the last event's codes" still holds and train_digr.py needs no change;
  * `--no-add_action --n_temporal_bins 64` reproduces v1 token for token;
  * the clock token is 1 + weekday*24 + hour UTC, and 0 when the timestamp is
    missing;
  * a SID with duplicate codes reports the share of items that share one.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
V1_DIR = os.path.join(REPO, "Model Training")
for p in (HERE, V1_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import preprocess_digr_v2 as v2  # noqa: E402

DATA_ROOT = os.path.join(HERE, "_pytest_v2")

# 4 semantic levels + 1 collision rank, like `index_rqvae_prod.json --add_unique_id`.
# Items 1 and 2 share the top four codes and differ only in the collision rank.
CATALOG = {
    "1": [1, 0, 2, 3, 0],
    "2": [1, 0, 2, 3, 1],
    "3": [2, 1, 0, 0, 0],
    "4": [3, 3, 3, 3, 0],
    "5": [4, 2, 1, 0, 0],
    "6": [0, 4, 4, 4, 0],
}
NUM_LEVELS = 5

CLICK, OTHER, NULL = 1, 2, None

SEQ_SCHEMA = pa.schema(
    [
        ("user_id", pa.int64()),
        ("seq", pa.list_(pa.struct([("item_id", pa.int64()),
                                    ("action_type", pa.int64()),
                                    ("timestamp", pa.int64())]))),
    ]
)


def _seq(*rows):
    return [{"item_id": int(i), "action_type": a, "timestamp": int(t)} for i, a, t in rows]


def _train_rows():
    return [
        # exposure at index 2 of the history, two clicks
        {"user_id": 100, "seq": _seq((1, CLICK, 1000), (2, CLICK, 1060),
                                     (2, OTHER, 1120), (3, CLICK, 1180))},
        # null action at index 1 of the history
        {"user_id": 101, "seq": _seq((4, CLICK, 1000), (5, NULL, 1010), (6, CLICK, 1020))},
        # no click at all -> no row
        {"user_id": 102, "seq": _seq((6, OTHER, 1000), (6, OTHER, 1001))},
    ]


def _val_rows():
    return [
        {"user_id": 200, "seq": _seq((3, CLICK, 1000), (4, CLICK, 1060), (1, CLICK, 1120))},
        {"user_id": 201, "seq": _seq((5, CLICK, 1000), (2, CLICK, 1060))},
    ]


def _write_parquet(path: str, rows):
    pq.write_table(pa.Table.from_pylist(rows, schema=SEQ_SCHEMA), path)


def _rmtree(path: str) -> None:
    for _ in range(20):
        gc.collect()
        shutil.rmtree(path, ignore_errors=True)
        if not os.path.exists(path):
            return
        time.sleep(0.05)
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="function")
def workspace():
    _rmtree(DATA_ROOT)
    os.makedirs(DATA_ROOT, exist_ok=True)
    catalog_path = os.path.join(DATA_ROOT, "index.json")
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(CATALOG, f)
    train = os.path.join(DATA_ROOT, "train.parquet")
    val = os.path.join(DATA_ROOT, "val.parquet")
    _write_parquet(train, _train_rows())
    _write_parquet(val, _val_rows())
    yield {"catalog": catalog_path, "train": train, "val": val}
    _rmtree(DATA_ROOT)


def _run(paths, module=v2, extra=(), out_name="out"):
    out_dir = os.path.join(DATA_ROOT, out_name)
    argv = [
        "--sequence_path", paths["train"],
        "--val_sequence_path", paths["val"],
        "--catalog_path", paths["catalog"],
        "--output_dir", out_dir,
        "--overwrite",
        "--min_history", "2",
    ] + (["--allow_new_split"] if hasattr(module, "_require_split_decision") else [])       + list(extra)
    args = module.parse_args(argv)
    log = module.setup_logger()
    try:
        manifest = module.preprocess(args, log)
    finally:
        # Windows: an open FileHandler keeps the log file locked, and the fixture
        # could not delete the tree afterwards.
        for handler in list(log.handlers):
            handler.close()
            log.removeHandler(handler)
    return out_dir, manifest, args


def _split(out_dir: str, name: str = "train"):
    d = os.path.join(out_dir, name)
    return (
        np.load(os.path.join(d, "tokens.npy")),
        np.load(os.path.join(d, "offsets.npy")),
        np.load(os.path.join(d, "targets.npy")),
    )


def _vocab(out_dir: str) -> dict:
    with open(os.path.join(out_dir, "vocab.json"), encoding="utf-8") as f:
        return json.load(f)


def _catalog(out_dir: str):
    codes = np.load(os.path.join(out_dir, "catalog_sid_codes.npy"))
    items = np.load(os.path.join(out_dir, "catalog_item_ids.npy"))
    by_item = {int(items[i]): i for i in range(len(items))}
    by_codes = {tuple(int(c) for c in codes[i]): int(items[i]) for i in range(len(items))}
    return codes, by_item, by_codes


def _row_items(tokens, offsets, row: int, vocab: dict, by_codes: dict) -> list:
    """Item ids of the history of `row`, read out of the event slots."""
    slot = vocab["event_width"]
    levels = vocab["num_levels"]
    start = int(offsets[row]) + 1              # skip SEP (no user features here)
    n = int((int(offsets[row + 1]) - start) // slot)
    out = []
    for i in range(n):
        base = start + i * slot + slot - levels
        codes = tuple(int(tokens[base + lvl]) - vocab["sid_offsets"][lvl]
                      for lvl in range(levels))
        out.append(by_codes[codes])
    return out


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------

def test_event_width_and_row_length(workspace):
    out_dir, _, args = _run(workspace)
    vocab = _vocab(out_dir)
    assert vocab["n_action"] == v2.N_ACTION_TOKENS == 3
    assert vocab["n_clock"] == 0
    assert vocab["event_width"] == 1 + 1 + 0 + NUM_LEVELS
    assert args.n_temporal_bins == 128          # v2 default, v1 used 64

    tokens, offsets, _ = _split(out_dir)
    lengths = np.diff(offsets)
    # no --user_feat_path: SEP + one slot per event
    assert lengths[0] == 1 + 3 * vocab["event_width"]
    assert tokens[offsets[0]] == vocab["sep_id"]
    assert tokens.max() < vocab["total_size"]


def test_action_token_slots(workspace):
    out_dir, _, _ = _run(workspace)
    vocab = _vocab(out_dir)
    tokens, offsets, targets = _split(out_dir)
    _codes, _by_item, by_codes = _catalog(out_dir)
    slot = vocab["event_width"]
    levels = vocab["num_levels"]

    # user 100: history [1, 2, 2] where the 3rd event is an exposure
    assert _row_items(tokens, offsets, 0, vocab, by_codes) == [1, 2, 2]
    start = int(offsets[0]) + 1
    actions = [int(tokens[start + i * slot + 1]) - vocab["offset_action"]
               for i in range(3)]
    assert actions == [v2.ACTION_CLICK_TOKEN, v2.ACTION_CLICK_TOKEN,
                       v2.ACTION_EXPOSURE_TOKEN]
    assert actions[2] == v2.ACTION_EXPOSURE_TOKEN == 0

    # user 101: history [4, 5] where the 2nd event has a null action
    assert _row_items(tokens, offsets, 1, vocab, by_codes) == [4, 5]
    start = int(offsets[1]) + 1
    actions = [int(tokens[start + i * slot + 1]) - vocab["offset_action"]
               for i in range(2)]
    assert actions == [v2.ACTION_CLICK_TOKEN, v2.ACTION_UNKNOWN_TOKEN]
    assert actions[1] == v2.ACTION_UNKNOWN_TOKEN == 2

    # every action token in the file is one of the three published values
    seen = set()
    for row, length in enumerate(np.diff(offsets)):
        n = int((int(length) - 1) // slot)
        for i in range(n):
            seen.add(int(tokens[offsets[row] + 1 + i * slot + 1]) - vocab["offset_action"])
    assert seen == {0, 1, 2}
    assert vocab["action_token_order"] == ["exposure", "click", "unknown"]

    # the codes stay last inside the slot, so the last L tokens of a row are the
    # last history event's codes (the anchor's codes are never in the row)
    for row in range(len(targets)):
        items = _row_items(tokens, offsets, row, vocab, by_codes)
        tail = tuple(int(t) - vocab["sid_offsets"][lvl]
                     for lvl, t in enumerate(tokens[offsets[row + 1] - levels:offsets[row + 1]]))
        assert by_codes[tail] == items[-1]
        assert int(targets[row]) >= 0


def test_no_action_reproduces_v1(workspace):
    try:
        import preprocess_digr as v1
    except Exception as exc:                     # pragma: no cover
        pytest.skip(f"v1 not importable: {exc}")

    new_dir, _, new_args = _run(
        workspace, extra=["--no-add_action", "--n_temporal_bins", "64"], out_name="out_v2"
    )
    old_dir, _, old_args = _run(workspace, module=v1, out_name="out_v1")

    assert new_args.n_temporal_bins == old_args.n_temporal_bins == 64
    assert _vocab(new_dir)["n_action"] == 0
    assert _vocab(new_dir)["event_width"] == 1 + NUM_LEVELS

    for split in ("train", "val"):
        new_tokens, new_offsets, new_targets = _split(new_dir, split)
        old_tokens, old_offsets, old_targets = _split(old_dir, split)
        assert np.array_equal(new_offsets, old_offsets)
        assert np.array_equal(new_targets, old_targets)
        assert np.array_equal(new_tokens, old_tokens)


def test_add_clock_widens_the_slot(workspace):
    out_dir, _, _ = _run(workspace, extra=["--add_clock"])
    vocab = _vocab(out_dir)
    assert vocab["n_clock"] == v2.N_CLOCK_TOKENS == 1 + 7 * 24
    assert vocab["event_width"] == 1 + 1 + 1 + NUM_LEVELS

    tokens, offsets, _ = _split(out_dir)
    slot = vocab["event_width"]
    lo = vocab["offset_clock"]
    hi = lo + vocab["n_clock"]
    for row, length in enumerate(np.diff(offsets)):
        for i in range(int((length - 1) // slot)):
            clock = int(tokens[offsets[row] + 1 + i * slot + 2])
            assert lo <= clock < hi
            # the fixture timestamps are ~1000s, i.e. before 2000: unknown
            assert clock == lo


def test_clock_codes_values():
    # 2021-01-04 00:00:00 UTC was a Monday -> weekday 0, hour 0
    monday = 1609718400.0
    assert v2._clock_codes(np.array([monday]))[0] == 1 + 0 * 24 + 0
    # +13h
    assert v2._clock_codes(np.array([monday + 13 * 3600.0]))[0] == 1 + 0 * 24 + 13
    # 2021-01-10 was a Sunday
    sunday = monday + 6 * 86400.0
    assert v2._clock_codes(np.array([sunday]))[0] == 1 + 6 * 24 + 0
    # missing / pre-2000 timestamps are slot 0, never midnight in 1970
    assert v2._clock_codes(np.array([0.0, 1000.0, np.nan]))[:3].tolist() == [0, 0, 0]
    assert v2._clock_codes(np.empty(0)).size == 0
    # every value a real timestamp can produce is inside the vocabulary
    rng = np.random.default_rng(0)
    ts = rng.integers(946684800, 1900000000, size=5000).astype(np.float64)
    out = v2._clock_codes(ts)
    assert out.min() == 1 and out.max() == v2.N_CLOCK_TOKENS - 1


# ---------------------------------------------------------------------------
# vocab round-trip and the SID collision report
# ---------------------------------------------------------------------------

def test_vocab_roundtrip():
    vocab = v2.Vocab(sid_sizes=[8192, 8192], n_temporal=128,
                     feature_sizes=[4, 3], n_action=3, n_clock=169)
    d = vocab.to_dict()
    assert d["event_width"] == 1 + 1 + 1 + 2
    assert d["offset_action"] == 2 + 16384 + 128
    assert d["offset_clock"] == d["offset_action"] + 3
    assert d["offset_features"] == d["offset_clock"] + 169
    assert d["total_size"] == d["offset_features"] + 7

    back = v2.Vocab.from_dict(d)
    assert back.to_dict() == d
    assert back.event_width == vocab.event_width


def test_vocab_from_dict_defaults_to_v1_layout():
    """A v1 vocab.json has no n_action / n_clock keys and must still load."""
    old = {"sid_sizes": [256, 256, 256, 256, 731], "n_temporal": 64,
           "feature_sizes": [87, 3, 8, 15, 20, 5, 4, 3]}
    vocab = v2.Vocab.from_dict(old)
    assert vocab.n_action == 0 and vocab.n_clock == 0
    assert vocab.event_width == 1 + 5
    assert vocab.offset_features == 2 + 1755 + 64


def test_collision_share_counts_duplicate_codes():
    codes = np.array([[0, 0], [0, 0], [1, 0], [2, 0]], dtype=np.int32)
    catalog = v2.ItemCatalog(np.array([10, 11, 12, 13]), True, codes)
    stats = v2.catalog_collision_stats(catalog)
    assert stats["levels"] == 2
    assert stats["distinct_sids"] == 3
    assert stats["items_sharing_a_sid"] == 2
    assert stats["collision_share"] == pytest.approx(0.5)
    assert stats["max_items_per_sid"] == 2


def test_manifest_records_the_new_tokens(workspace):
    _, manifest, _ = _run(workspace)
    assert manifest["n_action"] == 3
    assert manifest["n_clock"] == 0
    assert manifest["event_width"] == 1 + 1 + NUM_LEVELS
    assert manifest["action_token_order"] == ["exposure", "click", "unknown"]
    assert manifest["catalog_collisions"]["items_sharing_a_sid"] == 0


def test_expect_val_split_flag_parses():
    args = v2.parse_args(["--sequence_path", "s.parquet",
                          "--catalog_path", "c.json",
                          "--expect_val_split", "/tmp/ref"])
    assert args.expect_val_split == "/tmp/ref"
    assert v2.parse_args(["--sequence_path", "s", "--catalog_path", "c"]).expect_val_split is None


def test_expect_val_split_accepts_the_same_users_in_any_order(tmp_path):
    ref = tmp_path / "ref" / "val"
    ref.mkdir(parents=True)
    np.save(ref / "users.npy", np.asarray([7, 8, 9], dtype=object))
    logger = logging.getLogger("test")
    for got in ([7, 8, 9], [9, 7, 8]):
        v2._check_val_users(np.asarray(got, dtype=object), str(tmp_path / "ref"), 30, logger)


def test_expect_val_split_rejects_a_different_membership(tmp_path):
    ref = tmp_path / "ref" / "val"
    ref.mkdir(parents=True)
    np.save(ref / "users.npy", np.asarray([7, 8, 9], dtype=object))
    logger = logging.getLogger("test")
    with pytest.raises(SystemExit) as exc:
        v2._check_val_users(np.asarray([7, 8, 10], dtype=object),
                            str(tmp_path / "ref"), 30, logger)
    msg = str(exc.value)
    assert "2 of 3 shared" in msg
    assert "--val_ratio" in msg
    assert "10.00%" in msg          # 3 of 30 records


def test_expect_val_split_requires_the_reference_file(tmp_path):
    logger = logging.getLogger("test")
    with pytest.raises(SystemExit) as exc:
        v2._check_val_users(np.asarray([1], dtype=object), str(tmp_path), 10, logger)
    assert "--expect_val_split" in str(exc.value)


def test_expect_val_split_checks_the_saved_split(tmp_path):
    ref = tmp_path / "ref"
    (ref / "val").mkdir(parents=True)
    np.save(ref / "val" / "users.npy", np.asarray([7, 8, 9], dtype=object))
    np.save(ref / "val" / "targets.npy", np.asarray([100, 101, 102], dtype=np.int64))
    ids = np.asarray([10, 11, 12], dtype=np.int64)
    np.save(ref / "catalog_item_ids.npy", ids)
    catalog = v2.ItemCatalog(ids, True, np.zeros((3, 2), dtype=np.int32))
    logger = logging.getLogger("test")

    class Split:
        users = np.asarray([9, 8, 7], dtype=object)
        targets = np.asarray([102, 101, 100], dtype=np.int64)

    v2._check_val_split(Split(), catalog, str(ref), logger)     # order is free

    Split.targets = np.asarray([102, 101, 99], dtype=np.int64)
    with pytest.raises(SystemExit) as exc:
        v2._check_val_split(Split(), catalog, str(ref), logger)
    assert "1 of 3 different" in str(exc.value)

    Split.targets = np.asarray([102, 101, 100], dtype=np.int64)
    Split.users = np.asarray([9, 8, 6], dtype=object)
    with pytest.raises(SystemExit) as exc:
        v2._check_val_split(Split(), catalog, str(ref), logger)
    assert "1 missing, 1 unexpected" in str(exc.value)

    Split.users = np.asarray([9, 8, 7], dtype=object)
    other = v2.ItemCatalog(np.asarray([10, 11, 13], dtype=np.int64), True,
                           np.zeros((3, 2), dtype=np.int32))
    with pytest.raises(SystemExit) as exc:
        v2._check_val_split(Split(), other, str(ref), logger)
    assert "catalog_item_ids differ" in str(exc.value)


def test_split_decision_is_mandatory():
    base = ["--sequence_path", "s.parquet", "--catalog_path", "c.json"]
    with pytest.raises(SystemExit) as exc:
        v2._require_split_decision(v2.parse_args(base))
    assert "--allow_new_split" in str(exc.value)
    assert "--expect_val_split" in str(exc.value)
    v2._require_split_decision(v2.parse_args(base + ["--allow_new_split"]))
    v2._require_split_decision(v2.parse_args(base + ["--expect_val_split", "d"]))


def test_mismatched_reference_stops_before_tokenising(workspace):
    paths = workspace
    ref = os.path.join(DATA_ROOT, "ref")
    os.makedirs(os.path.join(ref, "val"), exist_ok=True)
    np.save(os.path.join(ref, "val", "users.npy"), np.asarray([1, 2, 3], dtype=object))
    np.save(os.path.join(ref, "val", "targets.npy"), np.asarray([0, 0, 0], dtype=np.int64))
    args = v2.parse_args([
        "--sequence_path", paths["train"], "--val_sequence_path", paths["val"],
        "--catalog_path", paths["catalog"],
        "--output_dir", os.path.join(DATA_ROOT, "refout"),
        "--overwrite", "--min_history", "2", "--expect_val_split", ref,
    ])
    log = v2.setup_logger()
    try:
        with pytest.raises(SystemExit) as exc:
            v2.preprocess(args, log)
    finally:
        for handler in list(log.handlers):
            handler.close()
            log.removeHandler(handler)
    assert "val users do not match" in str(exc.value)

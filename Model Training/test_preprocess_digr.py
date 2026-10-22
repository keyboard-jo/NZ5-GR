#!/usr/bin/env python3
"""
Unit tests for `preprocess_digr.py` (numpy + pyarrow only, no torch, no GPU).

Run from the repo root:

    python -m pytest "Model Training/test_preprocess_digr.py" -q

Everything is built from a tiny synthetic dataset written under
`Model Training/_pytest_data/` (created and deleted by the `workspace` fixture).
No real data, no /tmp, no model download.

The synthetic data is built so that each rule has exactly one obvious answer:

  * `user 100` clicks at event 0, 1 and 3, with an exposure at event 2
  * `user 101` clicks at event 0 and 2, with a null action at event 1
  * `user 102` has no click at all
  * `user 103` has a single click at event 0 (no history)
  * `user 105` clicks only at event 1 (one event of history)
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

import preprocess_digr as pp  # noqa: E402

DATA_ROOT = os.path.join(HERE, "_pytest_data")

# 4 semantic levels + 1 collision rank, like `index_rqvae_prod.json --add_unique_id`.
# Items 1 and 2 share the same semantic SID and differ only in the collision rank.
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

# Every run keeps the default window / anchors / gap, only the history length
# barrier is lowered to 2 so the tiny records survive it.
BASE = ["--min_history", "2"]


# ---------------------------------------------------------------------------
# mock dataset
# ---------------------------------------------------------------------------

SEQ_SCHEMA = pa.schema(
    [
        ("user_id", pa.int64()),
        ("seq", pa.list_(pa.struct([("item_id", pa.int64()),
                                    ("action_type", pa.int64()),
                                    ("timestamp", pa.int64())]))),
    ]
)


def _seq(*rows):
    """rows of (item_id, action_type, timestamp) tuples."""
    return [{"item_id": int(i), "action_type": a, "timestamp": int(t)} for i, a, t in rows]


def _cascade(n: int, action=CLICK, start_ts: int = 1000, first_item: int = 1):
    """`n` consecutive events, cycling through items 1..6 from `first_item`."""
    return tuple((1 + ((first_item - 1 + i) % 6), action, start_ts + i) for i in range(n))


def _write_parquet(path: str, rows):
    pq.write_table(pa.Table.from_pylist(rows, schema=SEQ_SCHEMA), path)


def _train_rows():
    return [
        # clicks at 0, 1, 3 (one exposure at 2) -> anchor 3, history [1, 2, 2]
        {"user_id": 100, "seq": _seq((1, CLICK, 1000), (2, CLICK, 1060),
                                     (2, OTHER, 1120), (3, CLICK, 1180))},
        # clicks at 0 and 2 (a null action at 1) -> anchor 2, history [4, 5]
        {"user_id": 101, "seq": _seq((4, CLICK, 1000), (5, NULL, 1010), (6, CLICK, 1020))},
        # no click at all -> no target -> no row
        {"user_id": 102, "seq": _seq((6, OTHER, 1000), (6, OTHER, 1001))},
        # single click, no history -> dropped by --min_history
        {"user_id": 103, "seq": _seq((1, CLICK, 1000))},
        # one event of history -> dropped by --min_history 2, kept by --min_history 1
        {"user_id": 105, "seq": _seq((1, CLICK, 1000), (2, CLICK, 1001))},
    ]


def _val_rows():
    return [
        {"user_id": 200, "seq": _seq((3, CLICK, 1000), (4, CLICK, 1060), (1, CLICK, 1120))},
        {"user_id": 201, "seq": _seq((5, CLICK, 1000), (2, CLICK, 1060))},
    ]


@pytest.fixture(scope="function")
def workspace():
    """Fresh `_pytest_data/` tree inside the project (never /tmp)."""
    _rmtree(DATA_ROOT)
    os.makedirs(DATA_ROOT, exist_ok=True)
    yield DATA_ROOT
    _rmtree(DATA_ROOT)


def _rmtree(path: str) -> None:
    """Remove a tree; token arrays are memory-mapped, so Windows may still hold
    the handle for a moment after the split object is dropped."""
    for _ in range(20):
        gc.collect()
        shutil.rmtree(path, ignore_errors=True)
        if not os.path.exists(path):
            return
        time.sleep(0.05)
    shutil.rmtree(path, ignore_errors=True)


def _setup(workspace) -> dict:
    catalog_path = os.path.join(workspace, "index.json")
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(CATALOG, f)
    train_path = os.path.join(workspace, "train.parquet")
    val_path = os.path.join(workspace, "val.parquet")
    _write_parquet(train_path, _train_rows())
    _write_parquet(val_path, _val_rows())
    return {"catalog": catalog_path, "train": train_path, "val": val_path}


def _run(workspace, paths, extra=(), base=BASE):
    out_dir = os.path.join(workspace, "out")
    argv = [
        "--sequence_path", paths["train"],
        "--val_sequence_path", paths["val"],
        "--catalog_path", paths["catalog"],
        "--output_dir", out_dir,
        "--overwrite",
    ] + list(base) + list(extra)
    args = pp.parse_args(argv)
    log = pp.setup_logger()
    try:
        manifest = pp.preprocess(args, log)
    finally:
        # Windows: an open FileHandler keeps the log file locked, and the
        # `workspace` fixture could not delete the tree afterwards.
        for handler in list(log.handlers):
            handler.close()
            log.removeHandler(handler)
    return out_dir, manifest, args


def _tokens(out_dir: str, split: str):
    d = os.path.join(out_dir, split)
    tokens = np.load(os.path.join(d, "tokens.npy"))
    offsets = np.load(os.path.join(d, "offsets.npy"))
    targets = np.load(os.path.join(d, "targets.npy"))
    return tokens, offsets, targets


def _maps(out_dir: str):
    codes = np.load(os.path.join(out_dir, "catalog_sid_codes.npy"))
    items = np.load(os.path.join(out_dir, "catalog_item_ids.npy"))
    by_codes = {tuple(int(c) for c in codes[i]): int(items[i]) for i in range(len(items))}
    by_item = {int(items[i]): i for i in range(len(items))}
    return codes, by_codes, by_item


def _vocab(out_dir: str) -> dict:
    with open(os.path.join(out_dir, "vocab.json"), encoding="utf-8") as f:
        return json.load(f)


def _decode_row(tokens, offsets, r: int, vocab: dict, by_codes: dict) -> list:
    """Item ids of the history of row `r`, in order (the target is not in it)."""
    sid_offsets = vocab["sid_offsets"]
    L = len(sid_offsets)
    stride = 1 + L
    start, end = int(offsets[r]), int(offsets[r + 1])
    hist = []
    pos = start + 1
    while pos + stride <= end:
        codes = tuple(int(tokens[pos + 1 + lvl]) - sid_offsets[lvl] for lvl in range(L))
        hist.append(by_codes[codes])
        pos += stride
    return hist


def _row_len(n_events: int, num_levels: int = NUM_LEVELS) -> int:
    return 1 + n_events * (1 + num_levels)


# ---------------------------------------------------------------------------
# 1. defaults and CLI
# ---------------------------------------------------------------------------

def test_anchor_defaults():
    args = pp.parse_args([])
    assert args.window == 20
    assert args.min_history == 5
    assert args.anchors_per_record == 3
    assert args.anchor_gap == 5
    assert args.clicks_only is False


def test_window_has_a_max_seq_len_alias():
    args = pp.parse_args(["--max_seq_len", "7"])
    assert args.window == 7


@pytest.mark.parametrize("flag", ["--window", "--anchors_per_record", "--min_history"])
def test_invalid_anchor_flags_are_rejected(flag):
    with pytest.raises(SystemExit):
        pp.parse_args([flag, "0"])


def test_anchor_gap_zero_is_allowed_but_negative_is_not():
    assert pp.parse_args(["--anchor_gap", "0"]).anchor_gap == 0
    with pytest.raises(SystemExit):
        pp.parse_args(["--anchor_gap", "-1"])


# ---------------------------------------------------------------------------
# 2. collision index (catalog side)
# ---------------------------------------------------------------------------

def test_collision_index_is_in_the_output(workspace):
    paths = _setup(workspace)
    out_dir, manifest, _ = _run(workspace, paths)

    codes = np.load(os.path.join(out_dir, "catalog_sid_codes.npy"))
    assert codes.shape == (len(CATALOG), NUM_LEVELS)

    # items 1 and 2 keep distinct rows: identical semantic prefix, different rank
    _, _, by_item = _maps(out_dir)
    assert np.array_equal(codes[by_item[1]][:4], codes[by_item[2]][:4])
    assert codes[by_item[1]][4] != codes[by_item[2]][4]

    stats = manifest["catalog_collisions"]
    assert stats["semantic_levels"] == 4
    assert stats["collision_level"] == 1
    assert stats["items_with_duplicate_sid"] == 2
    assert stats["max_items_per_sid"] == 2

    # and they really are different tokens in the sequence vocabulary
    vocab = _vocab(out_dir)
    sid_offsets = vocab["sid_offsets"]
    tok1 = [sid_offsets[lvl] + int(c) for lvl, c in enumerate(codes[by_item[1]])]
    tok2 = [sid_offsets[lvl] + int(c) for lvl, c in enumerate(codes[by_item[2]])]
    assert tok1 != tok2


def test_dropping_the_collision_level(workspace):
    paths = _setup(workspace)
    out_dir, manifest, _ = _run(workspace, paths, ["--num_sid_levels", "4"])
    codes = np.load(os.path.join(out_dir, "catalog_sid_codes.npy"))
    assert codes.shape[1] == 4
    assert manifest["num_levels"] == 4
    assert manifest["catalog_collisions"]["collision_level"] == 1  # levels[:-1] = 3 semantic


# ---------------------------------------------------------------------------
# 3. action_type normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, np.nan, "null", "None", "", "  ", float("nan"), object()])
def test_null_action_is_unknown(value):
    assert pp._action_code(value, 1.0) == pp.ACTION_UNKNOWN


@pytest.mark.parametrize("value", [1, 1.0, "1", " 1 "])
def test_click_detection(value):
    assert pp._action_code(value, 1.0) == pp.ACTION_CLICK


def test_non_click_action():
    assert pp._action_code(2, 1.0) == pp.ACTION_OTHER


# ---------------------------------------------------------------------------
# 4. anchors: which events become a target
# ---------------------------------------------------------------------------

def test_only_clicks_become_targets(workspace):
    paths = _setup(workspace)
    out_dir, manifest, _ = _run(workspace, paths)

    _, offsets, targets = _tokens(out_dir, "train")
    _, _, by_item = _maps(out_dir)
    # user 100 -> its last click (item 3), user 101 -> its last click (item 6)
    assert [int(t) for t in targets] == [by_item[3], by_item[6]]
    assert manifest["splits"]["train"] == 2
    assert manifest["split_stats"]["train"]["rows_without_click"] == 1   # user 102


def test_history_is_every_event_not_only_clicks(workspace):
    paths = _setup(workspace)
    out_dir, manifest, _ = _run(workspace, paths)

    tokens, offsets, _ = _tokens(out_dir, "train")
    vocab = _vocab(out_dir)
    _, by_codes, _ = _maps(out_dir)

    # user 100 keeps its exposure (item 2 twice), user 101 keeps its null action (item 5)
    assert _decode_row(tokens, offsets, 0, vocab, by_codes) == [1, 2, 2]
    assert _decode_row(tokens, offsets, 1, vocab, by_codes) == [4, 5]
    assert int(offsets[1] - offsets[0]) == _row_len(3)
    assert int(offsets[2] - offsets[1]) == _row_len(2)

    stats = manifest["split_stats"]["train"]
    assert stats["unknown_actions"] == 1          # the NULL of user 101
    assert stats["non_click_actions"] == 3        # user 100 exposure + 2 of user 102


def test_clicks_only_flag_filters_the_history(workspace):
    paths = _setup(workspace)
    out_dir, _, _ = _run(workspace, paths, ["--clicks_only"])

    tokens, offsets, _ = _tokens(out_dir, "train")
    vocab = _vocab(out_dir)
    _, by_codes, _ = _maps(out_dir)
    # user 100: history is only its two earlier clicks; the exposure is gone
    assert _decode_row(tokens, offsets, 0, vocab, by_codes) == [1, 2]
    assert int(offsets[1] - offsets[0]) == _row_len(2)


def test_min_history_counts_events_not_clicks(workspace):
    """Two exposures before the click are two events of history."""
    paths = _setup(workspace)
    _write_parquet(paths["train"], [
        {"user_id": 110, "seq": _seq((1, OTHER, 1000), (2, OTHER, 1010), (3, CLICK, 1020))},
    ])
    out_dir, manifest, _ = _run(workspace, paths)
    tokens, offsets, targets = _tokens(out_dir, "train")
    _, _, by_item = _maps(out_dir)
    assert int(targets[0]) == by_item[3]
    assert int(offsets[1] - offsets[0]) == _row_len(2)
    assert manifest["split_stats"]["train"]["rows_short"] == 0

    with pytest.raises(ValueError, match="no usable sequences"):
        _run(workspace, paths, ["--min_history", "3"])


def test_single_click_record_needs_enough_history(workspace):
    paths = _setup(workspace)
    # a click with nothing before it can never produce a row
    _write_parquet(paths["train"], [{"user_id": 103, "seq": _seq((1, CLICK, 1000))}])
    with pytest.raises(ValueError, match="no usable sequences"):
        _run(workspace, paths, ["--min_history", "1"])

    # one event of history is enough once the barrier is 1
    _write_parquet(paths["train"], [
        {"user_id": 105, "seq": _seq((1, CLICK, 1000), (2, CLICK, 1001))},
    ])
    out_dir, manifest, _ = _run(workspace, paths, ["--min_history", "1"])
    tokens, offsets, targets = _tokens(out_dir, "train")
    vocab = _vocab(out_dir)
    _, by_codes, by_item = _maps(out_dir)
    assert [int(t) for t in targets] == [by_item[2]]
    assert _decode_row(tokens, offsets, 0, vocab, by_codes) == [1]
    assert manifest["split_stats"]["train"]["rows_short"] == 0

    # ... and not when it is 2
    with pytest.raises(ValueError, match="no usable sequences"):
        _run(workspace, paths, ["--min_history", "2"])


def test_anchor_gap_keeps_rows_apart(workspace):
    """Clicks 1 event apart never produce two rows, clicks 5 apart do."""
    paths = _setup(workspace)
    rows = [{"user_id": 110, "seq": _seq((1, OTHER, 1000), (2, OTHER, 1010),
                                        *_cascade(6, start_ts=1020, first_item=3))}]
    _write_parquet(paths["train"], rows)

    # clicks sit at event 2..7; gap 5 -> anchors 7 and 2 (oldest first)
    out_dir, manifest, _ = _run(workspace, paths, ["--anchor_gap", "5"])
    tokens, offsets, targets = _tokens(out_dir, "train")
    _, _, by_item = _maps(out_dir)
    assert len(targets) == 2
    assert [int(t) for t in targets] == [by_item[3], by_item[2]]
    stats = manifest["split_stats"]["train"]
    # 6 clicks but only 2 anchors fit: the cap of 3 was not reached
    assert stats["rows_capped_by_anchors"] == 0
    assert stats["rows_gap_limited"] == 1

    # gap 6 -> only anchor 7 (event 2 is too close to 7)
    out_dir, manifest, _ = _run(workspace, paths, ["--anchor_gap", "6"])
    _, _, targets = _tokens(out_dir, "train")
    assert len(targets) == 1
    assert manifest["split_stats"]["train"]["rows_gap_limited"] == 1

    # gap 0 -> three anchors (event cap 3) and nothing is gap limited
    out_dir, manifest, _ = _run(workspace, paths, ["--anchor_gap", "0"])
    _, _, targets = _tokens(out_dir, "train")
    assert len(targets) == 3
    assert manifest["split_stats"]["train"]["rows_gap_limited"] == 0


def test_anchors_per_record_is_an_upper_bound(workspace):
    paths = _setup(workspace)
    _write_parquet(paths["train"], [
        {"user_id": 110, "seq": _seq(*_cascade(10))},      # 10 clicks, 1 event apart
    ])
    out_dir, manifest, _ = _run(workspace, paths, ["--anchor_gap", "1"])
    _, _, targets = _tokens(out_dir, "train")
    assert len(targets) == 3
    stats = manifest["split_stats"]["train"]
    assert stats["rows_capped_by_anchors"] == 1     # a fourth anchor existed
    assert stats["anchors_per_record"] == 3.0

    out_dir, _, _ = _run(workspace, paths, ["--anchor_gap", "1", "--anchors_per_record", "5"])
    _, _, targets = _tokens(out_dir, "train")
    assert len(targets) == 5

    # fewer clicks than the cap -> fewer rows, nothing reported as capped.
    # Three clicks one event apart are two valid anchors (2 and 1), but the
    # older one has a single event of history -> dropped by --min_history 2.
    _write_parquet(paths["train"], [
        {"user_id": 110, "seq": _seq(*_cascade(3))},
    ])
    out_dir, manifest, _ = _run(workspace, paths, ["--anchor_gap", "1"])
    _, _, targets = _tokens(out_dir, "train")
    assert len(targets) == 1
    stats = manifest["split_stats"]["train"]
    assert stats["rows_capped_by_anchors"] == 0
    assert stats["rows_kept"] == 1
    assert stats["rows_short"] == stats["rows_proposed"] - 1


def test_window_truncates_the_history(workspace):
    paths = _setup(workspace)
    _write_parquet(paths["train"], [{"user_id": 110, "seq": _seq(*_cascade(30))}])

    out_dir, manifest, _ = _run(workspace, paths)          # window 20
    _, offsets, targets = _tokens(out_dir, "train")
    assert manifest["window"] == 20
    assert manifest["max_seq_len"] == 20                  # kept for train_digr.py
    # anchors are events 29, 24 and 19, so the oldest row has only 19 events
    lengths = np.diff(offsets)
    assert lengths[0] == _row_len(19)
    assert np.all(lengths[1:] == _row_len(20))
    assert manifest["split_stats"]["train"]["rows_window_truncated"] == 2

    out_dir, _, _ = _run(workspace, paths, ["--window", "5"])
    _, offsets, _ = _tokens(out_dir, "train")
    assert np.all(np.diff(offsets) == _row_len(5))


def test_records_are_sorted_by_timestamp(workspace):
    paths = _setup(workspace)
    _write_parquet(paths["train"], [
        # stored out of order: item 1 is the newest event
        {"user_id": 110, "seq": _seq((1, CLICK, 900), (2, CLICK, 100), (3, CLICK, 500))},
    ])
    out_dir, manifest, _ = _run(workspace, paths)
    tokens, offsets, targets = _tokens(out_dir, "train")
    vocab = _vocab(out_dir)
    _, by_codes, by_item = _maps(out_dir)
    assert int(targets[0]) == by_item[1]                     # newest click
    assert _decode_row(tokens, offsets, 0, vocab, by_codes) == [2, 3]
    assert manifest["split_stats"]["train"]["rows_non_monotonic"] == 1


def test_unknown_history_items_are_dropped(workspace):
    paths = _setup(workspace)
    _write_parquet(paths["train"], [
        {"user_id": 110, "seq": _seq((1, CLICK, 1000), (999, OTHER, 1010), (2, CLICK, 1020))},
    ])
    out_dir, manifest, _ = _run(workspace, paths)
    tokens, offsets, targets = _tokens(out_dir, "train")
    vocab = _vocab(out_dir)
    _, by_codes, by_item = _maps(out_dir)
    assert int(targets[0]) == by_item[2]
    assert _decode_row(tokens, offsets, 0, vocab, by_codes) == [1]
    assert manifest["split_stats"]["train"]["unknown_items"] == 1


def test_target_outside_the_catalog_is_dropped(workspace):
    paths = _setup(workspace)
    _write_parquet(paths["train"], [
        {"user_id": 110, "seq": _seq((1, CLICK, 1000), (2, OTHER, 1010), (999, CLICK, 1020))},
    ])
    with pytest.raises(ValueError, match="no usable sequences"):
        _run(workspace, paths)


# ---------------------------------------------------------------------------
# 5. val / infer modes
# ---------------------------------------------------------------------------

def test_val_keeps_one_anchor_per_record(workspace):
    paths = _setup(workspace)
    out_dir, manifest, _ = _run(workspace, paths)
    tokens, offsets, targets = _tokens(out_dir, "val")
    vocab = _vocab(out_dir)
    _, by_codes, by_item = _maps(out_dir)

    # user 200 has three clicks but only its last one becomes a row;
    # user 201's last click has a single event of history -> dropped
    assert len(targets) == 1
    assert int(targets[0]) == by_item[1]
    assert _decode_row(tokens, offsets, 0, vocab, by_codes) == [3, 4]
    stats = manifest["split_stats"]["val"]
    assert stats["rows_proposed"] == 2            # exactly one anchor per record
    assert stats["rows_kept"] == 1
    assert stats["rows_short"] == 1               # user 201


def test_infer_split_has_no_target(workspace):
    paths = _setup(workspace)
    infer_path = os.path.join(workspace, "infer.parquet")
    _write_parquet(infer_path, [{"user_id": 400, "seq": _seq(*_cascade(25))}])
    out_dir, manifest, _ = _run(workspace, paths, ["--infer_sequence_path", infer_path])

    tokens, offsets, targets = _tokens(out_dir, "infer")
    vocab = _vocab(out_dir)
    _, by_codes, _ = _maps(out_dir)
    assert len(targets) == 1
    assert int(targets[0]) == -1
    # items cycle 1..6, so the last 20 of 25 events start at item 6
    assert _decode_row(tokens, offsets, 0, vocab, by_codes) == [6, 1, 2, 3, 4, 5, 6, 1, 2, 3,
                                                                4, 5, 6, 1, 2, 3, 4, 5, 6, 1]
    assert manifest["split_stats"]["infer"]["rows_window_truncated"] == 1


# ---------------------------------------------------------------------------
# 6. token layout
# ---------------------------------------------------------------------------

def test_token_layout(workspace):
    paths = _setup(workspace)
    out_dir, manifest, _ = _run(workspace, paths)

    vocab = _vocab(out_dir)
    assert vocab["total_size"] == manifest["vocab_size"]
    n_special = vocab["n_special"]
    sid_sizes = vocab["sid_sizes"]
    offsets = vocab["sid_offsets"]
    assert len(sid_sizes) == NUM_LEVELS
    assert offsets[0] == n_special
    for lvl in range(1, NUM_LEVELS):
        assert offsets[lvl] == offsets[lvl - 1] + sid_sizes[lvl - 1]

    tokens, row_offsets, _ = _tokens(out_dir, "train")
    assert tokens.max() < vocab["total_size"]
    assert tokens.min() >= 0
    profile = tokens[row_offsets[:-1]]          # first token of every row
    assert np.all(profile >= vocab["offset_profile"])
    assert np.all(profile < vocab["offset_profile"] + vocab["n_profile"])


def test_unknown_user_falls_back_to_unk(workspace):
    paths = _setup(workspace)
    _write_parquet(paths["train"], [
        {"user_id": None, "seq": _seq((1, CLICK, 1000), (2, CLICK, 1060))},
    ])
    out_dir, _, _ = _run(workspace, paths, ["--min_history", "1"])
    tokens, row_offsets, _ = _tokens(out_dir, "train")
    vocab = _vocab(out_dir)
    assert int(tokens[row_offsets[0]]) == vocab["offset_profile"]  # dense index 0 == UNK


# ---------------------------------------------------------------------------
# 7. flat schema
# ---------------------------------------------------------------------------

FLAT_SCHEMA = pa.schema(
    [
        ("user_id", pa.int64()),
        ("history_items", pa.list_(pa.int64())),
        ("history_actions", pa.list_(pa.int64())),
        ("history_timestamps", pa.list_(pa.int64())),
        ("target", pa.int64()),
    ]
)


def _flat(rows):
    return pa.Table.from_pylist(rows, schema=FLAT_SCHEMA)


def test_explicit_target_yields_one_row(workspace):
    paths = _setup(workspace)
    pq.write_table(_flat([
        {"user_id": 300, "history_items": [1, 2, 3, 4],
         "history_actions": [1, None, 2, 1],
         "history_timestamps": [1000, 1010, 1020, 1030], "target": 5},
        {"user_id": 301, "history_items": [6, 6],
         "history_actions": [2, None],
         "history_timestamps": [1000, 1010], "target": 1},
    ]), paths["train"])
    out_dir, manifest, _ = _run(workspace, paths)

    stats = manifest["split_stats"]["train"]
    assert stats["unknown_actions"] == 2          # one null in each row
    assert stats["non_click_actions"] == 2
    # `rows_without_click` only counts the anchor mode: with an explicit target
    # column the anchor rules (and therefore the click scan) are not applied.

    tokens, offsets, targets = _tokens(out_dir, "train")
    vocab = _vocab(out_dir)
    _, by_codes, by_item = _maps(out_dir)
    # an explicit target wins over the anchor rules: one row per record
    assert [int(t) for t in targets] == [by_item[5], by_item[1]]
    # every event of the history is kept, exposures and nulls included
    assert _decode_row(tokens, offsets, 0, vocab, by_codes) == [1, 2, 3, 4]
    assert int(offsets[1] - offsets[0]) == _row_len(4)

    # ... unless --clicks_only asks for clicks only
    out_dir, _, _ = _run(workspace, paths, ["--clicks_only"])
    tokens, offsets, _ = _tokens(out_dir, "train")
    vocab = _vocab(out_dir)
    _, by_codes, _ = _maps(out_dir)
    assert _decode_row(tokens, offsets, 0, vocab, by_codes) == [1, 4]


def test_flat_actions_must_align(workspace):
    paths = _setup(workspace)
    pq.write_table(_flat([
        {"user_id": 300, "history_items": [1, 2],
         "history_actions": [1],
         "history_timestamps": [1000, 1010], "target": 5},
    ]), paths["train"])
    with pytest.raises(ValueError, match="must be aligned"):
        _run(workspace, paths)


def test_flat_without_actions_uses_the_last_event_as_target(workspace):
    paths = _setup(workspace)
    schema = pa.schema([("user_id", pa.int64()),
                        ("history_items", pa.list_(pa.int64())),
                        ("target", pa.int64())])
    pq.write_table(pa.Table.from_pylist(
        [{"user_id": 300, "history_items": [1, 2, 3], "target": 4}], schema=schema,
    ), paths["train"])
    out_dir, manifest, _ = _run(workspace, paths)
    tokens, offsets, targets = _tokens(out_dir, "train")
    vocab = _vocab(out_dir)
    _, by_codes, by_item = _maps(out_dir)
    assert [int(t) for t in targets] == [by_item[4]]
    assert _decode_row(tokens, offsets, 0, vocab, by_codes) == [1, 2, 3]


def test_flat_without_actions_or_target_uses_the_last_event(workspace):
    paths = _setup(workspace)
    schema = pa.schema([("user_id", pa.int64()),
                        ("history_items", pa.list_(pa.int64()))])
    pq.write_table(pa.Table.from_pylist(
        [{"user_id": 300, "history_items": [1, 2, 3]}], schema=schema,
    ), paths["train"])
    out_dir, manifest, _ = _run(workspace, paths)
    tokens, offsets, targets = _tokens(out_dir, "train")
    vocab = _vocab(out_dir)
    _, by_codes, by_item = _maps(out_dir)
    assert [int(t) for t in targets] == [by_item[3]]
    assert _decode_row(tokens, offsets, 0, vocab, by_codes) == [1, 2]
    assert manifest["split_stats"]["train"]["non_click_actions"] == 0

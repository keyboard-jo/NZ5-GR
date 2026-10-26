"""Tests for the pure-SID full-catalog baselines (B3 transition counts, B4 co-click).

Everything here runs on a synthetic split written to tmp_path: 8 catalog items,
two SID levels, a hand-built transition structure. No torch, no model, no real
data. The point of the fixture is that popularity and the SID transition
deliberately disagree, so a baseline that accidentally ranks by frequency fails
the test.
"""

import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import baselines_digr as B


# ---------------------------------------------------------------------------
# Fixture: a tiny catalog with a known transition structure
# ---------------------------------------------------------------------------

SID_SIZES = (4, 4)
SID_OFFSETS = (2, 6)
NUM_LEVELS = 2
SEP_ID = 1
TEMPORAL_OFFSET = 10          # above total_size, so it can never look like a code

#           level0, level1
CATALOG = np.array([
    [0, 0],   # 0
    [0, 1],   # 1   <- predicted by the transition from item 0
    [0, 2],   # 2
    [1, 0],   # 3
    [1, 1],   # 4
    [0, 3],   # 5   <- the popular item, but not the transition successor of 0
    [1, 2],   # 6
    [1, 3],   # 7
], dtype=np.int32)


def _event_tokens(item, temporal_index):
    codes = CATALOG[item]
    return [TEMPORAL_OFFSET + temporal_index,
            int(SID_OFFSETS[0] + codes[0]),
            int(SID_OFFSETS[1] + codes[1])]


def _row(items, n_features=2):
    """[features][SEP][(temporal, code0, code1) * len(items)]."""
    tokens = [0] * n_features + [SEP_ID]
    for t, item in enumerate(items):
        tokens.extend(_event_tokens(item, t))
    return tokens


# 20 rows whose target is item 1, reached from a last event of item 0.
# 25 rows whose target is item 5 (more popular than item 1 on purpose).
# 1 row that adds a lower-count edge item 4 -> item 1 for the co-click table.
TRAIN_ROWS = ([_row([0])] * 20) + ([_row([4])] * 25) + [_row([3, 4, 1])]
TRAIN_TARGETS = [1] * 20 + [5] * 25 + [1]

# A: plain transition query, popularity ranks the target second.
# B: the popular item, reached from item 4.
# C: two events, so the history window has something older to decay away.
VAL_ROWS = [_row([0]), _row([4]), _row([3, 0], n_features=3)]
VAL_TARGETS = [1, 5, 1]


def _write_split(root, name, rows, targets):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    flat = np.asarray([t for row in rows for t in row], dtype=np.int32)
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum([len(row) for row in rows], out=offsets[1:])
    np.save(os.path.join(d, "tokens.npy"), flat)
    np.save(os.path.join(d, "offsets.npy"), offsets)
    np.save(os.path.join(d, "targets.npy"), np.asarray(targets, dtype=np.int64))
    np.save(os.path.join(d, "users.npy"), np.arange(len(rows), dtype=np.int64))
    with open(os.path.join(d, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": len(rows), "tokens": int(flat.size)}, f)


@pytest.fixture()
def data_dir(tmp_path):
    root = str(tmp_path)
    vocab = {
        "sid_sizes": list(SID_SIZES),
        "sid_offsets": list(SID_OFFSETS),
        "num_levels": NUM_LEVELS,
        "sep_id": SEP_ID,
        "pad_id": 0,
        "total_size": int(sum(SID_SIZES) + 2),
    }
    with open(os.path.join(root, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    np.save(os.path.join(root, "catalog_sid_codes.npy"), CATALOG)
    _write_split(root, "train", TRAIN_ROWS, TRAIN_TARGETS)
    _write_split(root, "val", VAL_ROWS, VAL_TARGETS)
    return root


@pytest.fixture()
def logger():
    return B.setup_logger()


# ---------------------------------------------------------------------------
# Code packing / row layout
# ---------------------------------------------------------------------------

def test_pack_codes_is_injective_and_sorted():
    keys = B.pack_codes(CATALOG, SID_SIZES)
    assert len(np.unique(keys)) == len(CATALOG)
    # Mixed radix: level 0 dominates.
    assert keys[3] > keys[0]

    index = B.build_code_index(CATALOG, SID_SIZES)
    rows, found = index.lookup(CATALOG)
    assert found.all()
    np.testing.assert_array_equal(rows, np.arange(len(CATALOG)))


def test_build_code_index_rejects_duplicate_tuples():
    duplicated = np.vstack([CATALOG, CATALOG[:1]])
    with pytest.raises(ValueError, match="duplicate full SID tuples"):
        B.build_code_index(duplicated, SID_SIZES)


def test_row_feature_lengths_reads_each_row_boundary(data_dir, logger):
    layout = B.build_layout(data_dir, "val", B.load_vocab(data_dir), logger,
                            mmap=False)
    assert len(layout) == len(VAL_ROWS)
    # Two rows carry two feature tokens, the last one carries three.
    np.testing.assert_array_equal(layout.feat_len, [2, 2, 3])
    np.testing.assert_array_equal(layout.n_events, [1, 1, 2])


def test_build_layout_rejects_a_row_that_is_not_event_aligned(data_dir, logger):
    bad = os.path.join(data_dir, "bad")
    os.makedirs(bad, exist_ok=True)
    row = _row([0])
    row.append(TEMPORAL_OFFSET)          # trailing token, breaks the stride
    flat = np.asarray(row, dtype=np.int32)
    np.save(os.path.join(bad, "tokens.npy"), flat)
    np.save(os.path.join(bad, "offsets.npy"),
            np.asarray([0, flat.size], dtype=np.int64))
    with pytest.raises(ValueError, match="row length is not"):
        B.build_layout(data_dir, "bad", B.load_vocab(data_dir), logger,
                       mmap=False)


def test_iter_code_chunks_returns_events_in_row_order(data_dir, logger):
    vocab = B.load_vocab(data_dir)
    layout = B.build_layout(data_dir, "val", vocab, logger, mmap=False)
    chunks = list(B.iter_code_chunks(layout, vocab["sid_offsets"],
                                     vocab["sid_sizes"], chunk_rows=2))
    codes = np.concatenate([c.codes for c in chunks])
    expected = np.asarray([CATALOG[item] for item in [0, 4, 3, 0]], dtype=np.int32)
    np.testing.assert_array_equal(codes, expected)


def test_iter_code_chunks_rejects_a_code_outside_its_level(data_dir, logger):
    vocab = B.load_vocab(data_dir)
    layout = B.build_layout(data_dir, "val", vocab, logger, mmap=False)
    tokens = np.array(layout.tokens, copy=True)
    tokens[layout.base0[0] + 1] = int(SID_OFFSETS[0] + SID_SIZES[0])  # one past the end
    bad = B.SplitLayout(tokens=tokens, offsets=layout.offsets,
                        feat_len=layout.feat_len, n_events=layout.n_events,
                        base0=layout.base0, num_levels=NUM_LEVELS, sep_id=SEP_ID)
    with pytest.raises(ValueError, match="code out of range"):
        list(B.iter_code_chunks(bad, vocab["sid_offsets"], vocab["sid_sizes"]))


def test_last_event_codes_window_and_valid_mask(data_dir, logger):
    vocab = B.load_vocab(data_dir)
    layout = B.build_layout(data_dir, "val", vocab, logger, mmap=False)

    codes, valid = B.last_event_codes(layout, vocab["sid_offsets"], 1)
    assert codes.shape == (3, 1, NUM_LEVELS)
    assert valid.all()
    np.testing.assert_array_equal(codes[:, 0, :], CATALOG[[0, 4, 0]])

    codes, valid = B.last_event_codes(layout, vocab["sid_offsets"], 2)
    # Rows with a single event leave the older slot invalid.
    np.testing.assert_array_equal(valid, [[True, False], [True, False], [True, True]])
    np.testing.assert_array_equal(codes[2, 0, :], CATALOG[0])   # most recent first
    np.testing.assert_array_equal(codes[2, 1, :], CATALOG[3])


# ---------------------------------------------------------------------------
# B3: transition counts
# ---------------------------------------------------------------------------

def _tables(data_dir, logger, anchor_pairs=True):
    vocab = B.load_vocab(data_dir)
    catalog = B.load_catalog_codes(data_dir)
    layout = B.build_layout(data_dir, "train", vocab, logger, mmap=False)
    targets = B.load_targets(data_dir, "train")
    return B.build_transition_counts(layout, vocab, catalog, targets, logger,
                                     chunk_rows=10,
                                     with_anchor_pairs=anchor_pairs)


def test_transition_counts_are_directional(data_dir, logger):
    tables = _tables(data_dir, logger)
    level0, level1 = tables
    # Anchors: item 0 -> item 1 twenty times, item 4 -> item 5 twenty-five times.
    assert level0[0, 0] == 21      # 20 anchors + the item1 -> item1 anchor
    assert level0[1, 0] == 26      # 25 anchors + the item4 -> item1 pair
    assert level1[0, 1] == 21
    assert level1[1, 3] == 25
    # The transition is not symmetric.
    assert level0[0, 1] == 0


def test_transition_counts_can_skip_anchor_pairs(data_dir, logger):
    without = _tables(data_dir, logger, anchor_pairs=False)
    # Only the single row [3, 4, 1] contributes consecutive pairs.
    assert int(without[0].sum()) == 2
    assert int(without[1].sum()) == 2


def test_sid_transition_ranks_the_successor_first(data_dir, logger):
    vocab = B.load_vocab(data_dir)
    catalog = B.load_catalog_codes(data_dir)
    layout = B.build_layout(data_dir, "val", vocab, logger, mmap=False)
    codes, valid = B.last_event_codes(layout, vocab["sid_offsets"], 1)
    tables = _tables(data_dir, logger)
    level_codes = [np.ascontiguousarray(catalog[:, l]) for l in range(NUM_LEVELS)]

    score = B.score_sid_transition(codes[0], valid[0], tables, level_codes,
                                   B.DEFAULT_SMOOTHING, B.DEFAULT_HISTORY_DECAY,
                                   "uniform", None)
    assert score.shape == (len(CATALOG),)
    assert int(np.argmax(score)) == 1


def test_sid_transition_beats_popularity_on_the_same_row(data_dir, logger):
    """The whole point: a SID transition the frequency table cannot express."""
    vocab = B.load_vocab(data_dir)
    catalog = B.load_catalog_codes(data_dir)
    layout = B.build_layout(data_dir, "val", vocab, logger, mmap=False)
    codes, valid = B.last_event_codes(layout, vocab["sid_offsets"], 1)
    tables = _tables(data_dir, logger)
    level_codes = [np.ascontiguousarray(catalog[:, l]) for l in range(NUM_LEVELS)]

    popularity = np.bincount(B.load_targets(data_dir, "train"),
                             minlength=len(CATALOG))
    assert int(np.argmax(popularity)) == 5          # frequency says item 5
    assert int(np.argmax(popularity)) != VAL_TARGETS[0]

    score = B.score_sid_transition(codes[0], valid[0], tables, level_codes,
                                   B.DEFAULT_SMOOTHING, B.DEFAULT_HISTORY_DECAY,
                                   "uniform", None)
    assert int(np.argmax(score)) == VAL_TARGETS[0]  # the transition says item 1


def test_sid_transition_history_window_still_finds_the_target(data_dir, logger):
    vocab = B.load_vocab(data_dir)
    catalog = B.load_catalog_codes(data_dir)
    layout = B.build_layout(data_dir, "val", vocab, logger, mmap=False)
    codes, valid = B.last_event_codes(layout, vocab["sid_offsets"], 2)
    tables = _tables(data_dir, logger)
    level_codes = [np.ascontiguousarray(catalog[:, l]) for l in range(NUM_LEVELS)]
    score = B.score_sid_transition(codes[2], valid[2], tables, level_codes,
                                   B.DEFAULT_SMOOTHING, B.DEFAULT_HISTORY_DECAY,
                                   "uniform", None)
    assert int(np.argmax(score)) == VAL_TARGETS[2]


def test_sid_transition_smoothing_keeps_cold_codes_finite(data_dir, logger):
    vocab = B.load_vocab(data_dir)
    catalog = B.load_catalog_codes(data_dir)
    tables = _tables(data_dir, logger)
    level_codes = [np.ascontiguousarray(catalog[:, l]) for l in range(NUM_LEVELS)]
    # Codes with no observed transition at all.
    cold_codes = np.asarray([[3, 3]], dtype=np.int32)
    cold_valid = np.asarray([True])
    score = B.score_sid_transition(cold_codes, cold_valid, tables, level_codes,
                                   B.DEFAULT_SMOOTHING, B.DEFAULT_HISTORY_DECAY,
                                   "uniform", None)
    assert np.isfinite(score).all()
    # Pure smoothing means every item scores the same, so nothing is preferred.
    assert float(score.max() - score.min()) < 1e-6


def test_sid_transition_inv_entropy_matches_shape_and_is_finite(data_dir, logger):
    vocab = B.load_vocab(data_dir)
    catalog = B.load_catalog_codes(data_dir)
    layout = B.build_layout(data_dir, "val", vocab, logger, mmap=False)
    codes, valid = B.last_event_codes(layout, vocab["sid_offsets"], 1)
    tables = _tables(data_dir, logger)
    level_codes = [np.ascontiguousarray(catalog[:, l]) for l in range(NUM_LEVELS)]
    score = B.score_sid_transition(codes[0], valid[0], tables, level_codes,
                                   B.DEFAULT_SMOOTHING, B.DEFAULT_HISTORY_DECAY,
                                   "inv_entropy", None)
    assert score.shape == (len(CATALOG),)
    assert np.isfinite(score).all()
    assert int(np.argmax(score)) == 1


def test_sid_transition_rejects_unknown_weighting(data_dir, logger):
    vocab = B.load_vocab(data_dir)
    catalog = B.load_catalog_codes(data_dir)
    tables = _tables(data_dir, logger)
    level_codes = [np.ascontiguousarray(catalog[:, l]) for l in range(NUM_LEVELS)]
    with pytest.raises(ValueError, match="unknown weighting"):
        B.score_sid_transition(np.asarray([[0, 0]]), np.asarray([True]), tables,
                               level_codes, B.DEFAULT_SMOOTHING,
                               B.DEFAULT_HISTORY_DECAY, "bogus", None)


def test_tie_break_cannot_overturn_a_real_gap(data_dir, logger):
    vocab = B.load_vocab(data_dir)
    catalog = B.load_catalog_codes(data_dir)
    layout = B.build_layout(data_dir, "val", vocab, logger, mmap=False)
    codes, valid = B.last_event_codes(layout, vocab["sid_offsets"], 1)
    tables = _tables(data_dir, logger)
    level_codes = [np.ascontiguousarray(catalog[:, l]) for l in range(NUM_LEVELS)]
    popularity = np.bincount(B.load_targets(data_dir, "train"),
                             minlength=len(CATALOG)).astype(np.float32)
    tie_break = np.log1p(popularity)
    score = B.score_sid_transition(codes[0], valid[0], tables, level_codes,
                                   B.DEFAULT_SMOOTHING, B.DEFAULT_HISTORY_DECAY,
                                   "uniform", tie_break)
    assert int(np.argmax(score)) == 1


# ---------------------------------------------------------------------------
# B4: co-click kNN
# ---------------------------------------------------------------------------

def _coclick(data_dir, logger, anchor_pairs=True):
    vocab = B.load_vocab(data_dir)
    catalog = B.load_catalog_codes(data_dir)
    layout = B.build_layout(data_dir, "train", vocab, logger, mmap=False)
    index = B.build_code_index(catalog, SID_SIZES)
    targets = B.load_targets(data_dir, "train")
    return B.build_coclick(layout, vocab, catalog, index, targets, len(CATALOG),
                           logger, chunk_rows=10,
                           with_anchor_pairs=anchor_pairs)


def test_coclick_table_lookup(data_dir, logger):
    table = _coclick(data_dir, logger)
    items, counts = table.row(4)
    mapping = dict(zip(items.tolist(), counts.tolist()))
    assert mapping[5] == 25          # 25 anchors item 4 -> item 5
    assert mapping[1] == 1           # the single consecutive pair item 4 -> item 1
    assert table.row(0)[0].tolist() == [1]


def test_coclick_prefers_the_higher_count_successor(data_dir, logger):
    table = _coclick(data_dir, logger)
    popularity_order = np.argsort(
        -np.bincount(B.load_targets(data_dir, "train"), minlength=len(CATALOG)),
        kind="stable").astype(np.int64)
    top = B.top_from_coclick(table, np.asarray([4]), np.asarray([True]),
                             B.DEFAULT_HISTORY_DECAY, popularity_order)
    assert int(top[0]) == 5          # count 25 beats count 1, not catalog order


def test_coclick_cold_query_falls_back_to_popularity(data_dir, logger):
    table = _coclick(data_dir, logger)
    popularity_order = np.argsort(
        -np.bincount(B.load_targets(data_dir, "train"), minlength=len(CATALOG)),
        kind="stable").astype(np.int64)
    top = B.top_from_coclick(table, np.asarray([6]), np.asarray([True]),
                             B.DEFAULT_HISTORY_DECAY, popularity_order, kmax=4)
    assert len(top) == 4
    assert int(top[0]) == int(popularity_order[0])
    assert len(set(top.tolist())) == 4


def test_coclick_window_sums_decayed_counts(data_dir, logger):
    table = _coclick(data_dir, logger)
    popularity_order = np.arange(len(CATALOG), dtype=np.int64)
    # t=0 item 0 (only successor item 1, count 20); t=1 item 3 (successor item 4).
    top = B.top_from_coclick(table, np.asarray([0, 3]), np.asarray([True, True]),
                             1.0, popularity_order, kmax=2)
    assert sorted(top.tolist()) == [1, 4]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_accumulate_never_gives_a_miss_rank_zero():
    metrics = B.empty_metrics()
    top = np.arange(1, B.KMAX + 1)           # target 0 is absent
    B.accumulate(metrics, top, 0)
    assert metrics["n"] == 1.0
    assert metrics["recall@10"] == 0.0
    assert metrics["ndcg@10"] == 0.0          # not inf
    assert metrics["mrr"] == 0.0


def test_accumulate_grades_a_rank_two_hit():
    metrics = B.empty_metrics()
    B.accumulate(metrics, np.asarray([7, 3, 5]), 3)
    assert metrics["recall@1"] == 0.0
    assert metrics["recall@5"] == 1.0
    assert metrics["ndcg@5"] == pytest.approx(1.0 / math.log2(3.0))
    assert metrics["mrr"] == pytest.approx(0.5)


def test_finalize_divides_every_metric_by_n():
    metrics = B.empty_metrics()
    B.accumulate(metrics, np.asarray([0]), 0)
    B.accumulate(metrics, np.asarray([1]), 0)
    out = B.finalize(metrics)
    assert out["n"] == 2
    assert out["recall@1"] == pytest.approx(0.5)
    assert out["mrr"] == pytest.approx(0.5)


def test_make_ranges_covers_every_row_once():
    ranges = B.make_ranges(10, 3)
    covered = [i for lo, hi in ranges for i in range(lo, hi)]
    assert covered == list(range(10))
    assert B.make_ranges(0, 4) == []


def test_reference_constants_are_the_measured_baselines():
    assert B.REF_POPULARITY_R1 == pytest.approx(0.00439453125)
    # Re-measured by this script; an earlier ad-hoc run's 0.9644% did not
    # reproduce because it used a different frequency vector.  Exact, not
    # approx: the two modules used to disagree in the 8th digit, and every
    # lift ratio in the reports divides by this number.
    assert B.REF_POPULARITY_R10 == 1321 / 98304
    assert B.REF_DIGR["R@1"] > B.REF_POPULARITY_R1
    # The headline of the whole exercise: at the strongest window the count
    # table wins at R@10 and R@100, and DiGR only holds the far tail. Encode
    # that so a future edit cannot quietly record a flattering comparison.
    assert B.REF_COCLICK["R@10"] > B.REF_DIGR["R@10"]
    assert B.REF_COCLICK["R@100"] > B.REF_DIGR["R@100"]
    assert B.REF_DIGR["R@10000"] > B.REF_COCLICK["R@10000"]
    # R@1 stopped being a tie once the window reached 20: the count table takes
    # it outright, so DiGR now holds exactly one cut-off, the far tail.
    assert B.REF_COCLICK["R@1"] > B.REF_DIGR["R@1"]
    assert B.REF_COCLICK["R@1"] - B.REF_DIGR["R@1"] < 2e-3
    # The window sweep has to stay ordered, or the recorded bar is not the best
    # measured one.
    assert (B.REF_COCLICK_H1["R@10"] < B.REF_COCLICK_H5["R@10"]
            < B.REF_COCLICK_H10["R@10"] < B.REF_COCLICK["R@10"])


def test_both_modules_quote_the_same_popularity_r10():
    """eval_full_catalog.py carries its own copy; keep the copies identical."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import eval_full_catalog as E

    assert E.REF_POPULARITY_R10 == B.REF_POPULARITY_R10
    assert E.REF_POPULARITY[10] == B.REF_POPULARITY_R10


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_parser_defaults():
    args = B.build_parser().parse_args(["--data_dir", "x"])
    # sid_transition is opt-in: 3,805 s for R@10 0.1801%, 7.5x worse than
    # popularity, so it is a recorded negative result rather than a default.
    assert args.baselines == "popularity,coclick"
    assert args.history_window == 1
    assert args.weighting == "uniform"
    assert args.no_anchor_pairs is False
    assert args.no_popularity_tie_break is False


def test_cli_rejects_an_unknown_baseline(data_dir):
    with pytest.raises(SystemExit):
        B.main(["--data_dir", data_dir, "--baselines", "tiger"])


def test_cli_rejects_remapped_targets(data_dir):
    np.save(os.path.join(data_dir, "val", "targets.npy"),
            np.asarray([len(CATALOG)], dtype=np.int64))
    with pytest.raises(SystemExit):
        B.main(["--data_dir", data_dir])


def test_cli_popularity_reproduces_the_frequency_ranking(data_dir, tmp_path):
    """Popularity ranks the same top-k for every row, so R@1 must be the hit rate."""
    out_json = str(tmp_path / "metrics.json")
    assert B.main(["--data_dir", data_dir, "--baselines", "popularity",
                   "--workers", "1", "--out_json", out_json]) == 0
    with open(out_json, "r", encoding="utf-8") as f:
        metrics = json.load(f)["baselines"]["popularity"]
    # Target 5 is the most frequent train target, so only row B is a rank-1 hit.
    assert metrics["n"] == len(VAL_TARGETS)
    assert metrics["recall@1"] == pytest.approx(1.0 / 3.0)
    assert metrics["recall@10"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx((0.5 + 1.0 + 0.5) / 3.0)


@pytest.mark.parametrize("baselines,expected,recall_at_1", [
    ("popularity", ["popularity"], 1.0 / 3.0),
    ("sid_transition", ["sid_transition"], 1.0),
    ("coclick", ["coclick"], 1.0),
    ("popularity,sid_transition,coclick",
     ["popularity", "sid_transition", "coclick"], None),
])
def test_cli_end_to_end(data_dir, tmp_path, baselines, expected, recall_at_1):
    out_json = str(tmp_path / "out" / "metrics.json")
    code = B.main(["--data_dir", data_dir, "--baselines", baselines,
                   "--workers", "1", "--out_json", out_json])
    assert code == 0
    with open(out_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["catalog_size"] == len(CATALOG)
    assert payload["n_rows"] == len(VAL_TARGETS)
    assert sorted(payload["baselines"]) == sorted(expected)
    for kind, metrics in payload["baselines"].items():
        assert metrics["n"] == len(VAL_TARGETS)
        # Every val target sits in the top 10 of an 8 item catalog.
        assert metrics["recall@10"] == pytest.approx(1.0)
        if recall_at_1 is not None:
            assert metrics["recall@1"] == pytest.approx(recall_at_1)
            assert metrics["mrr"] >= recall_at_1


def test_sid_baselines_beat_popularity_on_the_fixture(data_dir, tmp_path):
    """The fixture is built so the transition - not frequency - finds the target."""
    out_json = str(tmp_path / "metrics.json")
    assert B.main(["--data_dir", data_dir, "--workers", "1",
                   "--baselines", "popularity,sid_transition,coclick",
                   "--out_json", out_json]) == 0
    with open(out_json, "r", encoding="utf-8") as f:
        res = json.load(f)["baselines"]
    assert res["sid_transition"]["recall@1"] > res["popularity"]["recall@1"]
    assert res["coclick"]["recall@1"] > res["popularity"]["recall@1"]


def test_cli_max_rows_limits_the_evaluation(data_dir, tmp_path):
    out_json = str(tmp_path / "metrics.json")
    assert B.main(["--data_dir", data_dir, "--baselines", "sid_transition",
                   "--workers", "1", "--max_rows", "1",
                   "--out_json", out_json]) == 0
    with open(out_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["n_rows"] == 1
    assert payload["baselines"]["sid_transition"]["n"] == 1


def test_cli_window_two_runs(data_dir, tmp_path):
    out_json = str(tmp_path / "metrics.json")
    assert B.main(["--data_dir", data_dir, "--history_window", "2",
                   "--baselines", "sid_transition,coclick",
                   "--workers", "1", "--out_json", out_json]) == 0
    with open(out_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["settings"]["history_window"] == 2
    for metrics in payload["baselines"].values():
        assert metrics["recall@1"] == pytest.approx(1.0)


def test_accumulate_keeps_mrr_at_ten_when_the_ks_are_wide():
    # Same invariant as train_digr: widening KMAX to 1000 for a recall@N curve
    # must not turn a rank-60 hit into MRR credit.
    metrics = B.empty_metrics()
    top = np.arange(120, dtype=np.int64)
    top[:59] = np.arange(1, 60)
    top[59] = 0                      # target 0 lands at rank 60
    top[60:] = np.arange(60, 120)
    B.accumulate(metrics, top, 0)
    assert metrics["recall@10"] == 0.0
    assert metrics["recall@100"] == 1.0
    assert metrics["ndcg@100"] == pytest.approx(1 / math.log2(61))
    assert metrics["mrr"] == 0.0


def test_accumulate_misses_stay_misses_when_ks_are_raised_by_cli(monkeypatch):
    # main() reassigns the module globals from --ks. accumulate() must resolve
    # its kmax default at call time, otherwise a miss would get rank 1001 and
    # count as a hit at k=10000.
    monkeypatch.setattr(B, "EVAL_KS", (1, 10, 10000))
    monkeypatch.setattr(B, "KMAX", 10000)
    metrics = B.empty_metrics()
    assert "recall@10000" in metrics
    B.accumulate(metrics, np.arange(1, 11, dtype=np.int64), 0)   # miss
    assert metrics["recall@10000"] == 0.0
    assert metrics["mrr"] == 0.0
    # and a hit inside the wider window still counts
    metrics = B.empty_metrics()
    B.accumulate(metrics, np.arange(1, 11, dtype=np.int64), 3)   # rank 3
    assert metrics["recall@10000"] == 1.0
    assert metrics["recall@1"] == 0.0


def test_coclick_cold_fill_is_bounded_by_the_catalog(data_dir, logger):
    # kmax far larger than the catalog must return every item once: no duplicate
    # entries and no per-query copy of the whole popularity order.
    table = _coclick(data_dir, logger)
    popularity_order = np.arange(len(CATALOG), dtype=np.int64)
    top = B.top_from_coclick(table, np.asarray([6]), np.asarray([True]),
                             B.DEFAULT_HISTORY_DECAY, popularity_order,
                             kmax=1000)
    assert len(top) == len(CATALOG)
    assert len(set(top.tolist())) == len(CATALOG)

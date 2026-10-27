"""Tests for Plan 2/sid_diagnose.py.

Mock data only. No GPU, no real catalogue, no network.
"""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "sid_diagnose", os.path.join(_HERE, "sid_diagnose.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sd = _load_module()


# ---------------------------------------------------------------------------
# pack_keys
# ---------------------------------------------------------------------------
def test_pack_keys_puts_level_zero_at_the_least_significant_digit():
    codes = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [1, 1]], dtype=np.int64)
    keys = sd.pack_keys(codes, [2, 2])
    assert keys.tolist() == [0, 1, 2, 3, 3]


def test_pack_keys_is_injective_on_full_radix():
    keys = sd.pack_keys(np.array([[2, 1], [1, 2]], dtype=np.int64), [3, 3])
    assert keys.tolist() == [5, 7]
    assert len(set(keys.tolist())) == 2


def test_pack_keys_does_not_overflow_on_int32_input():
    # 255 * 256**3 = 4_278_190_080, past int32. On Windows a C long is 32 bits, so
    # an int32 array times a Python int of that size raises or wraps. The int64 cast
    # is what keeps depth 4 and 5 honest.
    codes = np.array([[255, 255, 255, 255]], dtype=np.int32)
    keys = sd.pack_keys(codes, [256, 256, 256, 256])
    assert keys.dtype == np.int64
    assert int(keys[0]) == 2 ** 32 - 1


def test_pack_keys_rejects_codes_outside_the_codebook():
    with pytest.raises(ValueError, match="codebook size"):
        sd.pack_keys(np.array([[0, 5]], dtype=np.int64), [4, 4])


def test_pack_keys_rejects_negative_codes():
    with pytest.raises(ValueError, match="codebook size"):
        sd.pack_keys(np.array([[-1, 0]], dtype=np.int64), [4, 4])


def test_pack_keys_rejects_a_radix_product_that_would_overflow():
    with pytest.raises(ValueError, match="overflow"):
        sd.pack_keys(np.zeros((1, 7), dtype=np.int64), [2048] * 7)


def test_pack_keys_rejects_a_level_count_mismatch():
    with pytest.raises(ValueError, match="levels"):
        sd.pack_keys(np.zeros((3, 2), dtype=np.int64), [4, 4, 4])


# ---------------------------------------------------------------------------
# PrefixIndex
# ---------------------------------------------------------------------------
def test_prefix_index_counts_and_shared_items():
    codes = np.array([[0, 0], [0, 0], [0, 0], [1, 0]], dtype=np.int64)
    index = sd.PrefixIndex(codes, [2, 2], 1)
    assert index.n_buckets == 2
    assert index.counts.tolist() == [3, 1]
    # four items in two buckets, so only two items have a prefix of their own
    assert index.shared_items == 2
    assert index.largest_bucket == 3
    assert index.items_per_bucket == pytest.approx(2.0)


def test_prefix_index_is_injective_at_the_last_level():
    codes = np.array(
        [[a, b, c] for a in range(8) for b in range(8) for c in range(8)], dtype=np.int64
    )
    index = sd.PrefixIndex(codes, [8, 8, 8], 3)
    assert index.shared_items == 0
    assert index.largest_bucket == 1


def test_prefix_index_lookup_matches_bucket_of_item():
    codes = np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=np.int64)
    index = sd.PrefixIndex(codes, [2, 2], 2)
    bucket, found = index.lookup(codes)
    assert found.all()
    assert bucket.tolist() == index.bucket_of_item.tolist()


def test_prefix_index_lookup_flags_a_prefix_absent_from_the_catalogue():
    codes = np.array([[0, 0], [0, 1]], dtype=np.int64)
    index = sd.PrefixIndex(codes, [2, 2], 1)
    _, found = index.lookup(np.array([[1, 0]], dtype=np.int64))  # full width is fine
    assert found.tolist() == [False]


def test_prefix_index_shares_bucket_numbering_with_lookup():
    # Bucket ids are positions in the sorted unique key array for both paths, so a
    # query and a catalogue item with the same prefix get the same id.
    rng = np.random.default_rng(3)
    codes = rng.integers(0, 4, size=(40, 2), dtype=np.int64)
    index = sd.PrefixIndex(codes, [4, 4], 1)
    bucket, _ = index.lookup(codes)
    assert bucket.tolist() == index.bucket_of_item.tolist()


# ---------------------------------------------------------------------------
# codebook_report
# ---------------------------------------------------------------------------
def test_codebook_report_pegs_a_uniform_codebook_at_its_ceiling():
    codes = np.tile(np.arange(4, dtype=np.int64).reshape(4, 1), (1, 4))
    rows = sd.codebook_report(codes, np.array([4, 4, 4, 4]))
    for row in rows:
        assert row["used"] == 4
        assert row["entropy_bits"] == pytest.approx(2.0)
        assert row["max_bits"] == pytest.approx(2.0)
        assert row["utilisation"] == pytest.approx(1.0)
        assert row["entropy_ratio"] == pytest.approx(1.0)


def test_codebook_report_reports_zero_entropy_for_a_single_code():
    codes = np.zeros((6, 2), dtype=np.int64)
    rows = sd.codebook_report(codes, np.array([4, 4]))
    assert rows[0]["used"] == 1
    assert rows[0]["entropy_bits"] == pytest.approx(0.0)
    assert rows[0]["utilisation"] == pytest.approx(0.25)


def test_codebook_report_detects_an_unused_code():
    codes = np.array([[0], [0], [0], [1]], dtype=np.int64)
    rows = sd.codebook_report(codes, np.array([4]))
    assert rows[0]["used"] == 2
    assert rows[0]["entropy_bits"] == pytest.approx(0.811278, abs=1e-6)


# ---------------------------------------------------------------------------
# occupancy_report
# ---------------------------------------------------------------------------
def test_occupancy_report_reaches_one_item_per_bucket_at_the_last_level():
    codes = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    rows = sd.occupancy_report(codes, np.array([2, 2]))
    assert rows[0]["buckets"] == 2
    assert rows[0]["items_per_bucket"] == pytest.approx(2.0)
    assert rows[-1]["buckets"] == 4
    assert rows[-1]["shared_items"] == 0
    assert rows[-1]["catalog_share"] == pytest.approx(1.0)


def test_occupancy_report_buckets_never_increase_with_depth():
    rng = np.random.default_rng(7)
    codes = rng.integers(0, 8, size=(200, 3), dtype=np.int64)
    rows = sd.occupancy_report(codes, np.array([8, 8, 8]))
    counts = [row["buckets"] for row in rows]
    assert counts == sorted(counts)


# ---------------------------------------------------------------------------
# last_event_codes
# ---------------------------------------------------------------------------
def _toy_split(root, rows, targets):
    """Two levels, sid_offsets [2, 8], sizes [4, 4]. One event per row."""
    os.makedirs(root, exist_ok=True)
    tokens = []
    offsets = [0]
    for codes in rows:
        row = [1, 99] + [0] + [VOCAB["sid_offsets"][i] + int(c) for i, c in enumerate(codes)]
        tokens.extend(row)
        offsets.append(len(tokens))
    np.save(os.path.join(root, "tokens.npy"), np.asarray(tokens, dtype=np.uint16))
    np.save(os.path.join(root, "offsets.npy"), np.asarray(offsets, dtype=np.int64))
    np.save(os.path.join(root, "targets.npy"), np.asarray(targets, dtype=np.int64))
    return root


VOCAB = {
    "num_levels": 2,
    "sid_offsets": [2, 8],
    "sid_sizes": [4, 4],
}


def test_last_event_codes_reads_the_final_levels_of_each_row(tmp_path):
    rows = [[0, 1], [3, 2]]
    root = _toy_split(str(tmp_path), rows, [0, 0])
    tokens = np.load(os.path.join(root, "tokens.npy"))
    offsets = np.load(os.path.join(root, "offsets.npy"))
    codes, inside, n_rows = sd.last_event_codes(tokens, offsets, VOCAB)
    assert n_rows == 2
    assert inside.tolist() == [True, True]
    assert codes.tolist() == rows


def test_last_event_codes_marks_a_row_shorter_than_the_level_count():
    tokens = np.array([1, 99, 0, 2], dtype=np.uint16)  # one level only
    offsets = np.array([0, 4], dtype=np.int64)
    codes, inside, n_rows = sd.last_event_codes(tokens, offsets, VOCAB)
    assert n_rows == 1
    assert inside.tolist() == [False]
    assert codes.tolist() == [[-1, -1]]


# ---------------------------------------------------------------------------
# lift_report
# ---------------------------------------------------------------------------
def test_lift_report_scores_a_mixed_split(tmp_path):
    root = str(tmp_path)
    codes = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    np.save(os.path.join(root, "codes.npy"), codes)
    # Both rows are asked for item 1. Row 0 sits in a bucket that holds item 1, row
    # 1 does not, so depth 1 hits half the time out of buckets holding half the
    # catalogue: recall 50% and lift exactly 1, which is no information at all.
    split = _toy_split(os.path.join(root, "val"), [[0, 1], [1, 0]], [1, 1])
    vocab = {**VOCAB, "sid_sizes": [2, 2]}

    report = sd.lift_report(codes, np.array([2, 2]), vocab, split)
    depth1 = report["depths"][0]
    assert depth1["recall"] == pytest.approx(0.5)
    assert depth1["buckets"] == 2
    assert depth1["uniform_base"] == pytest.approx(0.5)
    assert depth1["lift"] == pytest.approx(1.0)
    # One level down the same recall comes out of buckets holding a quarter of the
    # catalogue, so the same 50% is now worth 2x chance.
    depth2 = report["depths"][1]
    assert depth2["recall"] == pytest.approx(0.5)
    assert depth2["uniform_base"] == pytest.approx(0.25)
    assert depth2["lift"] == pytest.approx(2.0)


def test_lift_report_is_one_when_the_bucket_carries_no_information(tmp_path):
    # One bucket holds everything, so containing the target is guaranteed and the
    # lift is exactly one. Recall is 100% and worth nothing.
    root = str(tmp_path)
    codes = np.zeros((4, 2), dtype=np.int64)
    np.save(os.path.join(root, "codes.npy"), codes)
    split = _toy_split(os.path.join(root, "val"), [[0, 0]], [3])
    report = sd.lift_report(codes, np.array([2, 2]), {**VOCAB, "sid_sizes": [2, 2]}, split)
    depth1 = report["depths"][0]
    assert depth1["recall"] == pytest.approx(1.0)
    assert depth1["lift"] == pytest.approx(1.0)


def test_lift_report_keeps_lift_flat_when_buckets_only_shrink(tmp_path):
    # Two buckets, then four by splitting each in half. The queries are all in one
    # coarse bucket and the targets are split across items, so recall halves as the
    # bucket halves and lift must not move. This is the trap the script exists to
    # catch: a deeper SID that halves recall has bought nothing.
    root = str(tmp_path)
    fine = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    coarse = fine[:, :1]
    np.save(os.path.join(root, "fine.npy"), fine)
    split = _toy_split(os.path.join(root, "val"), [[0, 0]] * 4, [0, 1, 0, 1])
    vocab = {"num_levels": 2, "sid_offsets": [2, 8], "sid_sizes": [2, 4]}

    coarse_report = sd.lift_report(coarse, np.array([2]), vocab, split)
    fine_report = sd.lift_report(fine, np.array([2, 4]), vocab, split)
    assert coarse_report["depths"][0]["buckets"] == 2
    assert coarse_report["depths"][0]["recall"] == pytest.approx(1.0)
    assert fine_report["depths"][1]["buckets"] == 4
    assert fine_report["depths"][1]["recall"] == pytest.approx(0.5)
    assert fine_report["depths"][1]["lift"] == pytest.approx(
        coarse_report["depths"][0]["lift"]
    )
    assert coarse_report["depths"][0]["lift"] == pytest.approx(2.0)


def test_lift_report_counts_targets_outside_the_catalogue(tmp_path):
    root = str(tmp_path)
    codes = np.array([[0, 0], [0, 1]], dtype=np.int64)
    np.save(os.path.join(root, "codes.npy"), codes)
    split = _toy_split(os.path.join(root, "val"), [[0, 1]], [99])
    report = sd.lift_report(codes, np.array([2, 2]), {**VOCAB, "sid_sizes": [2, 2]}, split)
    assert report["targets_outside_catalog"] == 1
    assert report["depths"][0]["recall"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------
def test_human_pct_keeps_small_values_readable():
    assert sd.human_pct(0.5) == "50.0000%"
    assert sd.human_pct(0.0183156) == "1.8316%"
    assert sd.human_pct(0.0) == "0%"
    assert "e-05%" in sd.human_pct(6.64e-7)


def test_human_int_groups_thousands():
    assert sd.human_int(4783154) == "4,783,154"
    assert sd.human_int(19_308.02) == "19,308.02"


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def test_main_runs_without_a_split(tmp_path, capsys):
    codes_dir = str(tmp_path)
    np.save(
        os.path.join(codes_dir, "catalog_sid_codes.npy"),
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32),
    )
    with open(os.path.join(codes_dir, "vocab.json"), "w", encoding="utf-8") as handle:
        json.dump({"sid_sizes": [2, 2], "num_levels": 2, "sid_offsets": [2, 8]}, handle)

    code = sd.main(["--data_dir", codes_dir])
    out = capsys.readouterr().out
    assert code == 0
    assert "every level uses every code" in out
    assert "items   4" in out
    assert "Bucket lift skipped" in out


def test_main_writes_a_json_report(tmp_path):
    codes_dir = str(tmp_path)
    np.save(
        os.path.join(codes_dir, "catalog_sid_codes.npy"),
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32),
    )
    with open(os.path.join(codes_dir, "vocab.json"), "w", encoding="utf-8") as handle:
        json.dump({"sid_sizes": [2, 2], "num_levels": 2, "sid_offsets": [2, 8]}, handle)
    _toy_split(os.path.join(codes_dir, "val"), [[0, 1], [1, 0]], [1, 1])

    out_path = os.path.join(codes_dir, "report.json")
    assert sd.main(["--data_dir", codes_dir, "--json", out_path]) == 0
    with open(out_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["n_items"] == 4
    assert report["n_levels"] == 2
    assert len(report["occupancy"]) == 2
    assert report["lift"]["depths"][0]["recall"] == pytest.approx(0.5)


def test_main_rejects_missing_inputs(tmp_path):
    with pytest.raises(SystemExit):
        sd.main(["--data_dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        sd.main([])


def test_resolve_paths_prefers_explicit_overrides(tmp_path):
    codes_dir = str(tmp_path)
    np.save(os.path.join(codes_dir, "catalog_sid_codes.npy"), np.zeros((2, 2), np.int32))
    with open(os.path.join(codes_dir, "vocab.json"), "w", encoding="utf-8") as handle:
        json.dump({"sid_sizes": [2, 2]}, handle)

    class Args:
        data_dir = codes_dir
        codes = os.path.join(codes_dir, "catalog_sid_codes.npy")
        vocab = os.path.join(codes_dir, "vocab.json")

    resolved = sd.resolve_paths(Args)
    assert resolved["codes"].endswith("catalog_sid_codes.npy")
    assert resolved["vocab"].endswith("vocab.json")

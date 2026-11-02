"""Tests for `dump_metrics.py`.

Fixtures are hand-sized so every number in the assertions is checkable by hand.
Ranks are chosen to cover the cases that have bitten this project: a miss stored
at the sentinel, a cut-off wider than the cap, an arm that is a beam, and two arms
that disagree on disjoint rows.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dump_metrics as dm  # noqa: E402


# n = 8 rows. Two misses, stored at 10001, so the inferred cap is 10000.
RANK_A = np.array([1, 2, 11, 4, 10001, 3, 1, 10001], dtype=np.int64)
RANK_B = np.array([1, 3, 10001, 10001, 2, 10001, 1, 10001], dtype=np.int64)
RANK_C = np.array([10001, 10001, 2, 10001, 10001, 10001, 1, 3], dtype=np.int64)
# hits only row 0, which A also hits, so A wins on four rows and loses on none
RANK_D = np.array([1, 10001, 10001, 10001, 10001, 10001, 10001, 10001], dtype=np.int64)
TARGETS = np.arange(8, dtype=np.int64)
CAP = 10000


def _write(tmp_path, name, rank, targets=None, rows=None, extra=None):
    targets = TARGETS if targets is None else targets
    rows = np.arange(len(rank)) if rows is None else rows
    payload = {"rows": rows, "targets": targets, "rank": rank}
    if extra:
        payload.update(extra)
    path = tmp_path / name
    np.savez(path, **payload)
    return str(path)


# --------------------------------------------------------------------------- #
# the arithmetic
# --------------------------------------------------------------------------- #
def test_recall_at_known_cut_offs():
    # ranks <= 1: rows 0 and 6. ranks <= 10: 0,1,3,5,6.
    assert dm.recall(RANK_A, 1, CAP) == pytest.approx(2 / 8)
    assert dm.recall(RANK_A, 10, CAP) == pytest.approx(5 / 8)
    assert dm.recall(RANK_A, 10000, CAP) == pytest.approx(6 / 8)


def test_ndcg_and_mrr_match_the_hand_calculation():
    # gains at ranks 1,2,4,3,1 -> 1 + 1/log2(3) + 1/log2(5) + 1/log2(4) + 1
    expected = (
        1.0
        + 1.0 / math.log2(3.0)
        + 1.0 / math.log2(5.0)
        + 1.0 / math.log2(4.0)
        + 1.0
    ) / 8.0
    assert dm.ndcg(RANK_A, 10, CAP) == pytest.approx(expected)

    expected_mrr = (1.0 + 0.5 + 1.0 / 4.0 + 1.0 / 3.0 + 1.0) / 8.0
    assert dm.mrr(RANK_A, 10, CAP) == pytest.approx(expected_mrr)


def test_ndcg_at_one_equals_recall_at_one():
    # With IDCG@1 = 1 the two must agree exactly, the invariant the docs rely on.
    assert dm.ndcg(RANK_A, 1, CAP) == pytest.approx(dm.recall(RANK_A, 1, CAP))


def test_effective_rank_ignores_the_sentinel():
    # hits are 1,2,11,4,3,1 -> geometric mean, and the two misses must not enter it
    hits = np.array([1, 2, 11, 4, 3, 1], dtype=np.float64)
    expected = 2.0 ** np.log2(hits).mean()
    assert dm.effective_rank(RANK_A, CAP) == pytest.approx(expected)
    # substituting the sentinel instead would give a much larger number: 20.1 here
    wrong = 2.0 ** np.log2(RANK_A.astype(np.float64)).mean()
    assert wrong > expected * 5


def test_miss_share():
    assert dm.miss_share(RANK_A, CAP) == pytest.approx(2 / 8)


# --------------------------------------------------------------------------- #
# the cap, which is where a beam must not be flattered
# --------------------------------------------------------------------------- #
def test_cap_is_the_beam_width_so_a_wide_cut_off_cannot_print_one():
    rank = np.array([1, 3, 2, 1], dtype=np.int64)
    arm = {"name": "beam", "rank": rank, "beam_width": np.array(2)}
    cap = dm.infer_cap(arm)
    assert cap == 2
    # rank 3 is a miss for a width-2 beam even though 3 <= 10 and 3 <= 10000
    assert dm.recall(rank, 10000, cap) == pytest.approx(3 / 4)
    assert dm.recall(rank, 10, cap) == pytest.approx(3 / 4)


def test_cap_is_inferred_from_a_sentinel_shaped_maximum():
    assert dm.infer_cap({"rank": RANK_A}) == 10000


def test_cap_override_wins():
    assert dm.infer_cap({"rank": RANK_A}, 100) == 100


def test_clamp_never_exceeds_the_cap():
    assert dm.clamp(10000, 20) == 20
    assert dm.clamp(10, 10000) == 10


# --------------------------------------------------------------------------- #
# pairs
# --------------------------------------------------------------------------- #
def test_pair_union_and_the_disagreement_split():
    arms = [
        {"name": "A", "rank": RANK_A, "n": 8},
        {"name": "B", "rank": RANK_B, "n": 8},
    ]
    row = dm.union_beats(arms, 10, {"A": CAP, "B": CAP})[0]
    # A hits {0,1,3,5,6}, B hits {0,1,4,6}
    assert row["a_only"] == pytest.approx(2 / 8)
    assert row["b_only"] == pytest.approx(1 / 8)
    assert row["neither"] == pytest.approx(2 / 8)
    assert row["union"] == pytest.approx(6 / 8)
    assert row["union"] == pytest.approx(
        row["a_only"] + row["b_only"] + dm.recall(RANK_A, 10, CAP) - row["a_only"]
    )


def test_mcnemar_is_one_when_the_arms_swap_the_same_number_of_rows():
    # b = 2 (A only), c = 1 (B only): |2-1| - 1 = 0, so chi2 = 0 and p = 1
    s = dm.mcnemar(RANK_A, RANK_B, 10, CAP, CAP)
    assert s["b"] == 2 and s["c"] == 1
    assert s["chi2"] == pytest.approx(0.0)
    assert s["p"] == pytest.approx(1.0)


def test_mcnemar_counts_both_directions_of_disagreement():
    # A hits {0,1,3,5,6}, C hits {2,6,7}: b = 4, c = 2 -> chi2 = (2-1)^2/6
    s = dm.mcnemar(RANK_A, RANK_C, 10, CAP, CAP)
    assert s["b"] == 4 and s["c"] == 2
    assert s["chi2"] == pytest.approx(1 / 6)
    assert s["p"] == pytest.approx(math.erfc(math.sqrt(1 / 12)))
    assert s["p"] > 0.5


def test_mcnemar_rewards_a_lopsided_pair():
    # A hits {0,1,3,5,6}, D hits {0}: b = 4, c = 0 -> chi2 = (4-1)^2/4 = 2.25
    s = dm.mcnemar(RANK_A, RANK_D, 10, CAP, CAP)
    assert s["b"] == 4 and s["c"] == 0
    assert s["chi2"] == pytest.approx(2.25)
    assert s["p"] == pytest.approx(math.erfc(math.sqrt(1.125)))
    assert s["p"] < 0.15


def test_mcnemar_on_identical_arms_is_not_significant():
    s = dm.mcnemar(RANK_A, RANK_A, 10, CAP, CAP)
    assert s == {"b": 0, "c": 0, "chi2": 0.0, "p": 1.0}


# --------------------------------------------------------------------------- #
# strata and groups
# --------------------------------------------------------------------------- #
def test_popularity_strata_partition_every_row():
    freq = np.array([10, 10, 10, 1, 1, 1, 0, 0], dtype=np.int64)
    table = dm.strata(RANK_A, TARGETS, freq, CAP, 10)
    assert sum(row["n"] for row in table) == 8
    hottest = max(table, key=lambda r: r["mean_freq"])
    assert hottest["n"] == 3
    assert hottest["mean_freq"] == pytest.approx(10.0)
    # the labels must run cold to hot, so "head" is the most popular bin
    assert hottest["bin"] == "head"
    assert min(table, key=lambda r: r["mean_freq"])["bin"] == "tail"
    # the three hot rows are 0,1,2 and only row 1 of those is a hit at rank 10
    assert hottest["recall"] == pytest.approx(2 / 3)


def test_strata_are_flat_for_an_arm_that_ignores_popularity():
    freq = np.array([5, 5, 5, 5, 5, 5, 5, 5], dtype=np.int64)
    table = dm.strata(RANK_A, TARGETS, freq, CAP, 10)
    for row in table:
        assert row["recall"] == pytest.approx(dm.recall(RANK_A, 10, CAP))


def test_per_group_recall():
    gids = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    table = {row["group"]: row for row in dm.per_group(RANK_A, gids, CAP, 1)}
    assert table["0"]["n"] == 4 and table["0"]["recall"] == pytest.approx(1 / 4)
    assert table["1"]["n"] == 4 and table["1"]["recall"] == pytest.approx(1 / 4)


# --------------------------------------------------------------------------- #
# cells
# --------------------------------------------------------------------------- #
def test_cell_stats_count_cells_not_items():
    tuples = np.array(
        [
            [[1, 0], [2, 0]],
            [[1, 0], [-1, -1]],
            [[1, 0], [2, 0]],
            [[2, 1], [-1, -1]],
        ],
        dtype=np.int64,
    )
    s = dm.cell_stats(tuples)
    assert s["beam_width"] == 2
    assert s["levels"] == 2
    assert s["named_per_row_mean"] == pytest.approx(1.5)
    assert s["named_per_row_min"] == 1
    assert s["named_per_row_max"] == 2
    # packed cells (0,0), (1,0), (0,1) -> three distinct
    assert s["distinct_cells"] == 3
    assert s["cells_per_row_mean"] == pytest.approx(1.5)
    assert 0.0 <= s["pairwise_jaccard_mean"] <= 1.0


# --------------------------------------------------------------------------- #
# loading, alignment, refusals
# --------------------------------------------------------------------------- #
def test_glob_and_directory_resolution(tmp_path):
    for lo in (0, 4):
        _write(tmp_path, f"coclick_{lo:06d}.npz", RANK_A[lo:lo + 4], TARGETS[lo:lo + 4],
               np.arange(lo, lo + 4))
    files = dm.resolve(str(tmp_path / "coclick_*.npz"), "cc")
    assert len(files) == 2
    arm = dm.load_arm(files, "cc")
    assert arm["n"] == 8
    np.testing.assert_array_equal(arm["rank"], RANK_A)

    # a directory holding one prefix is fine, two is refused
    arm2 = dm.load_arm(dm.resolve(str(tmp_path), "cc"), "cc")
    np.testing.assert_array_equal(arm2["rank"], RANK_A)


def test_chunks_are_reassembled_in_row_order(tmp_path):
    # written backwards on purpose: a fork pool can deliver chunks out of order
    hi = _write(tmp_path, "cc_000004.npz", RANK_A[4:], TARGETS[4:], np.arange(4, 8))
    lo = _write(tmp_path, "cc_000000.npz", RANK_A[:4], TARGETS[:4], np.arange(0, 4))
    arm = dm.load_arm([hi, lo], "cc")
    np.testing.assert_array_equal(arm["rank"], RANK_A)


def test_a_directory_with_two_prefixes_is_refused(tmp_path):
    _write(tmp_path, "coclick_000000.npz", RANK_A, TARGETS)
    _write(tmp_path, "popularity_000000.npz", RANK_B, TARGETS)
    with pytest.raises(SystemExit, match="more than one baseline"):
        dm.resolve(str(tmp_path), "both")


def test_a_glob_that_matches_nothing_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="matched no files"):
        dm.resolve(str(tmp_path / "nothing_*.npz"), "cc")


def test_a_gap_in_the_rows_is_refused(tmp_path):
    path = _write(tmp_path, "a.npz", RANK_A[:4], TARGETS[:4], np.array([0, 1, 3, 4]))
    with pytest.raises(SystemExit, match="does not cover rows"):
        dm.load_arm([path], "a")


def test_a_duplicated_row_is_refused(tmp_path):
    path = _write(tmp_path, "a.npz", RANK_A[:4], TARGETS[:4], np.array([0, 1, 1, 2]))
    with pytest.raises(SystemExit, match="does not cover rows"):
        dm.load_arm([path], "a")


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def test_main_writes_the_table_and_the_json(tmp_path):
    a = _write(tmp_path, "a.npz", RANK_A)
    b = _write(tmp_path, "b.npz", RANK_B)
    out = tmp_path / "out"
    dm.main([
        "--arm", f"A={a}",
        "--arm", f"B={b}",
        "--ks", "1,10",
        "--pair_k", "10",
        "--bootstrap", "0",
        "--out_dir", str(out),
    ])
    md = (out / "dump_metrics.md").read_text(encoding="utf-8")
    assert "R@1" in md and "R@10" in md
    assert "25.0000%" in md  # R@1 for A
    assert "Pair union at k = 10" in md
    assert "Paired significance" in md

    payload = json.loads((out / "dump_metrics.json").read_text(encoding="utf-8"))
    assert payload["n_rows"] == 8
    assert payload["caps"] == {"A": 10000, "B": 10000}
    assert payload["arms"]["A"]["recall"]["1"] == pytest.approx(0.25)
    assert payload["arms"]["A"]["recall"]["10"] == pytest.approx(0.625)
    assert payload["union"][0]["union"] == pytest.approx(0.75)


def test_main_refuses_arms_with_different_targets(tmp_path):
    a = _write(tmp_path, "a.npz", RANK_A)
    b = _write(tmp_path, "b.npz", RANK_B, targets=TARGETS + 1)
    with pytest.raises(SystemExit, match="disagrees"):
        dm.main(["--arm", f"A={a}", "--arm", f"B={b}", "--out_dir", str(tmp_path / "o")])


def test_main_refuses_a_groups_array_of_the_wrong_length(tmp_path):
    a = _write(tmp_path, "a.npz", RANK_A)
    g = tmp_path / "g.npy"
    np.save(g, np.zeros(3, dtype=np.int64))
    with pytest.raises(SystemExit, match="entries"):
        dm.main(["--arm", f"A={a}", "--groups", str(g), "--out_dir", str(tmp_path / "o")])


def test_main_with_popularity_strata_needs_a_data_dir(tmp_path):
    a = _write(tmp_path, "a.npz", RANK_A)
    data = tmp_path / "digr_data"
    (data / "train").mkdir(parents=True)
    np.save(data / "train" / "targets.npy", np.array([0, 0, 1, 1, 1, 2, 3, 3]))
    np.save(data / "catalog_item_ids.npy", np.arange(8))
    out = tmp_path / "out"
    dm.main([
        "--arm", f"A={a}",
        "--data_dir", str(data),
        "--bootstrap", "0",
        "--out_dir", str(out),
    ])
    md = (out / "dump_metrics.md").read_text(encoding="utf-8")
    assert "by target popularity" in md

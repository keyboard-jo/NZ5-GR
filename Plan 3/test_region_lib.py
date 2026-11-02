#!/usr/bin/env python
"""Tests for region_lib. numpy only, no torch, no GPU, runs in well under a second.

The point is not coverage for its own sake: every assertion here is a place where
a silent mis-join would produce a plausible table. A tuple packed into the wrong
bucket, a rank off by one, an "items above" count that forgets to weight by the
bucket size -- each of those turns 9.3x into nonsense without raising anything.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import region_lib as R  # noqa: E402


def _brute_regions(codes, sizes):
    """Dict from tuple -> list of item rows, built without any packing."""
    out = {}
    for item in range(len(codes)):
        out.setdefault(tuple(int(c) for c in codes[item]), []).append(item)
    return out


def test_pack_keys_matches_a_hand_built_key():
    codes = np.array([[0, 0], [0, 2], [2, 0], [2, 2]], dtype=np.int64)
    # base 3: key = c0 * 3 + c1
    assert R.pack_keys(codes, [3, 3]).tolist() == [0, 2, 6, 8]
    # base 4 changes the grouping even though the codes are the same
    assert R.pack_keys(codes, [4, 4]).tolist() == [0, 2, 8, 10]


def test_pack_keys_refuses_a_code_outside_its_width():
    with pytest.raises(ValueError, match="outside"):
        R.pack_keys(np.array([[0, 3]]), [3, 3])


def test_pack_keys_refuses_the_wrong_number_of_levels():
    with pytest.raises(ValueError, match="levels"):
        R.pack_keys(np.array([[0, 1, 2]]), [3, 3])


def test_pack_keys_refuses_a_radix_that_stops_being_exact():
    with pytest.raises(ValueError, match="radix product"):
        R.pack_keys(np.zeros((1, 3), dtype=np.int64), [1 << 30, 1 << 30, 1 << 30])


def test_two_different_tuples_never_share_a_key():
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 5, size=(400, 3))
    keys = R.pack_keys(codes, [5, 5, 5])
    by_tuple = _brute_regions(codes, [5, 5, 5])
    assert len(np.unique(keys)) == len(by_tuple)


def test_regions_agree_with_a_dict_grouping():
    codes = np.array([[0, 0], [0, 1], [1, 0], [1, 0], [1, 0], [2, 2]], dtype=np.int64)
    reg = R.Regions(codes, [3, 3])
    by_tuple = _brute_regions(codes, [3, 3])

    assert reg.n_regions == len(by_tuple)
    assert reg.n_items == 6
    # counts and first-member agree, matched by tuple
    for i, key_tuple in enumerate(by_tuple):
        members = by_tuple[key_tuple]
        j = i  # np.unique sorts keys, and these keys are already ascending
        assert int(reg.counts[j]) == len(members)
        assert int(reg.first[j]) == members[0]
        assert [int(c) for c in reg.codes[j]] == list(key_tuple)
    # region_of_item is the inverse map
    for item in range(len(codes)):
        assert int(reg.codes[reg.region_of_item[item]].tolist()[0]) == int(codes[item][0])
    assert reg.items_per_region == pytest.approx(6 / 4)
    # 3 of 6 items sit in the (1,0) tuple
    assert reg.shared_fraction == pytest.approx(3 / 6)
    assert not reg.injective


def test_regions_report_injectivity():
    codes = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    reg = R.Regions(codes, [2, 2])
    assert reg.injective
    assert reg.items_per_region == pytest.approx(1.0)
    assert reg.shared_fraction == pytest.approx(0.0)


def test_item_rank_weights_each_region_above_by_its_size():
    """The whole experiment rests on this arithmetic, so check it by brute force.

    Five regions with sizes 1, 2, 3, 1, 1. A target in the region of size 1 at
    score 0.5 is beaten by the size-3 region (score 0.9) and the size-2 region
    (score 0.7): two regions above, but 5 items above, so rank 6 rather than 3.
    """
    sizes = np.array([1, 2, 3, 1, 1])
    scores = np.array([0.9, 0.7, 0.8, 0.5, 0.1])   # target is index 3
    target = 3
    t = scores[target]
    above = np.flatnonzero(scores > t)
    region_rank = int(above.size) + 1
    item_rank = int(sizes[above].sum()) + 1
    assert region_rank == 4
    assert item_rank == 7
    # and the inflation is exactly sum(size - 1) over the regions above
    assert item_rank - region_rank == int((sizes[above] - 1).sum())


def test_an_equal_score_is_not_counted_as_above():
    """Ties are favourable: `rank = 1 + #{i : s_i > s_t}`, strict.

    Two regions both at 0.5 means neither is above, so the target is rank 1. If
    this ever became `>=`, every shared bucket would be counted against itself and
    the measured ranks would collapse.
    """
    scores = np.array([0.5, 0.5, 0.0])
    sizes = np.array([4, 4, 1])
    target = 1
    above = np.flatnonzero(scores > scores[target])
    assert above.size == 0
    assert int(above.size) + 1 == 1


def test_recall_at_counts_rows_at_or_inside_the_cut_off():
    rank = np.array([1, 2, 10, 11, 10001])
    assert R.recall_at(rank, 1) == pytest.approx(1 / 5)
    assert R.recall_at(rank, 10) == pytest.approx(3 / 5)
    assert R.recall_at(rank, 100) == pytest.approx(4 / 5)


def test_summarise_reports_the_counterfactual_gain():
    # 4 rows: two alone in their tuple and hit at rank 1, two in a fat tuple and
    # pushed to rank 40 by the size of the regions above them.
    item = np.array([1, 1, 40, 40], dtype=np.int64)
    region = np.array([1, 1, 3, 3], dtype=np.int64)
    g = np.array([1, 1, 12, 12], dtype=np.int64)
    s = R.summarise(item, region, g, ks=(1, 10))
    assert s["n_rows"] == 4
    assert s["item_r1"] == pytest.approx(0.5)
    assert s["region_r1"] == pytest.approx(0.5)      # 3 > 1, so no change at R@1
    assert s["item_r10"] == pytest.approx(0.5)
    assert s["region_r10"] == pytest.approx(1.0)     # ranks 3 are inside 10
    assert s["counterfactual_gain_10"] == pytest.approx(2.0)
    assert s["inflated_share"] == pytest.approx(0.5)
    labels = [row["group"] for row in s["groups"]]
    assert labels == ["1", "9-16"]
    fat = [row for row in s["groups"] if row["group"] == "9-16"][0]
    assert fat["inflation"] == pytest.approx(37.0)
    assert fat["g_mean"] == pytest.approx(12.0)


def test_summarise_refuses_ragged_input():
    with pytest.raises(ValueError):
        R.summarise(np.array([1, 2]), np.array([1]), np.array([1, 1]))


def test_verdict_names_counting_when_injective_ranking_recovers_the_head():
    s = {"counterfactual_gain_1": 2.30, "counterfactual_gain_10": 1.56}
    assert R.verdict(s).startswith("counting")


def test_verdict_names_scoring_when_injective_ranking_recovers_nothing():
    s = {"counterfactual_gain_1": 1.02, "counterfactual_gain_10": 1.05}
    assert R.verdict(s).startswith("scoring")


def test_verdict_admits_a_mixture():
    s = {"counterfactual_gain_1": 1.30, "counterfactual_gain_10": 1.25}
    assert R.verdict(s).startswith("mixed")


def test_describe_reports_the_numbers_the_docs_quote():
    """The v2 catalogue's collision figures, on a fixture built to reproduce them.

    Ten items: one tuple holding 6, one holding 3, one alone. So the shared
    fraction is 9/10 and the size seen by a random ITEM averages 4.5, not the
    3.33 a random ITEM-TUPLE averages -- that size bias is the quantity the
    scoring loss is quadratic in, so it gets its own column.
    """
    codes = np.zeros((10, 2), dtype=np.int64)
    codes[0:6] = [1, 1]
    codes[6:9] = [2, 2]
    codes[9] = [3, 3]
    d = R.Regions(codes, [4, 4]).describe()
    assert d["n_regions"] == pytest.approx(3)
    assert d["items_per_region"] == pytest.approx(10 / 3)
    assert d["shared_fraction"] == pytest.approx(0.9)
    assert d["largest"] == pytest.approx(6)
    # 6 items see 6, 3 see 3, 1 sees 1 -> (36 + 9 + 1) / 10
    assert d["mean_size_seen_by_item"] == pytest.approx(4.6)
    assert d["injective"] == pytest.approx(0.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

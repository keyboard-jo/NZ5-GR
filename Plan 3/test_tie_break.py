"""Tests for tie_break.py, on a catalogue small enough to check by hand."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tie_break as tb
from region_lib import Regions

# 9 items, 3x3 SID. Blocks (row, col) by packed key:
#   (0,0) -> items 0,1,2      g = 3
#   (1,0) -> items 3,4        g = 2
#   (2,2) -> items 5          g = 1
#   (0,1) -> items 6,7,8      g = 3
CATALOG = np.array(
    [
        [0, 0],  # 0
        [0, 0],  # 1
        [0, 0],  # 2
        [1, 0],  # 3
        [1, 0],  # 4
        [2, 2],  # 5
        [0, 1],  # 6
        [0, 1],  # 7
        [0, 1],  # 8
    ],
    dtype=np.int64,
)
SIZES = [3, 3]

# Popularity: 0 is the most clicked item in its block, 2 the least.
FREQ = np.array([90, 10, 5, 1, 80, 7, 60, 4, 3], dtype=np.int64)


def regions() -> Regions:
    return Regions(CATALOG, SIZES)


def test_block_layout_puts_members_in_ascending_row_order():
    r = regions()
    order, starts, pos = tb.block_layout(r)
    block = order[starts[r.region_of_item[0]]:][:r.counts[r.region_of_item[0]]]
    assert list(block) == [0, 1, 2]
    assert pos[0] == starts[r.region_of_item[0]]


def test_offset_by_row_counts_members_before_the_target():
    r = regions()
    order, starts, pos = tb.block_layout(r)
    targets = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    assert list(tb.offset_by_row(r, starts, pos, targets)) == [0, 1, 2, 0, 1, 0]


def test_offset_by_freq_ranks_the_most_clicked_first():
    r = regions()
    order, starts, pos = tb.block_layout(r)
    targets = np.array([0, 1, 2], dtype=np.int64)  # freqs 90, 10, 5
    g = np.array([3, 3, 3], dtype=np.int64)
    off, undecided = tb.offset_by_freq(r, starts, order, FREQ, targets, g)
    assert list(off) == [0, 1, 2]  # already popularity order
    assert not undecided.any()


def test_offset_by_freq_reorders_when_the_catalogue_order_is_wrong():
    r = regions()
    order, starts, pos = tb.block_layout(r)
    # Items 6,7,8 share a block; freq 60, 4, 3, so item 8 should be last and 6 first.
    targets = np.array([6, 7, 8], dtype=np.int64)
    g = np.array([3, 3, 3], dtype=np.int64)
    off, _ = tb.offset_by_freq(r, starts, order, FREQ, targets, g)
    assert list(off) == [0, 1, 2]


def test_equal_popularity_is_reported_as_undecided():
    r = regions()
    order, starts, pos = tb.block_layout(r)
    freq = np.array([5, 5, 5, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    targets = np.array([0, 1, 2], dtype=np.int64)
    g = np.array([3, 3, 3], dtype=np.int64)
    off, undecided = tb.offset_by_freq(r, starts, order, freq, targets, g)
    assert list(off) == [0, 1, 2]  # falls back to catalogue row
    assert undecided.all()


def test_ceiling_is_base_plus_one_and_never_worse_than_current():
    dump = {"targets": np.array([0, 1, 2, 5], dtype=np.int64),
            "rank": np.array([1, 4, 5, 40], dtype=np.int64),
            "fav_item_rank": np.array([1, 3, 3, 40], dtype=np.int64)}
    f = tb.fold_ranks(dump, regions(), FREQ, kmax=10000)
    # base = fav - 1 = 0, 2, 2, 39; the row offsets are 0, 1, 2, 0.
    assert list(f["base"]) == [0, 2, 2, 39]
    assert list(f["off_current"]) == [0, 1, 2, 0]
    assert list(f["current"]) == [1, 4, 5, 40]
    assert list(f["ceiling"]) == [1, 3, 3, 40]
    assert (f["ceiling"] <= f["current"]).all()


def test_popularity_can_lose_as_well_as_win():
    dump = {"targets": np.array([6, 7, 8], dtype=np.int64),
            "rank": np.array([10, 11, 12], dtype=np.int64),
            "fav_item_rank": np.array([10, 10, 10], dtype=np.int64)}
    # Nine identical items score above, so base = 9 for all three and only the
    # within-block order can move. Here item 8 is the most clicked, item 6 the
    # least, which is the reverse of the catalogue row order.
    freq = np.array([90, 10, 5, 1, 80, 7, 3, 4, 60], dtype=np.int64)
    f = tb.fold_ranks(dump, regions(), freq, kmax=10000)
    assert list(f["base"]) == [9, 9, 9]
    assert list(f["current"]) == [10, 11, 12]
    assert list(f["popularity"]) == [12, 11, 10]  # 8 first now, 6 last
    assert list(f["ceiling"]) == [10, 10, 10]


def test_without_a_strict_count_the_base_is_derived_from_the_assumed_row_order():
    dump = {"targets": np.array([1], dtype=np.int64),
            "rank": np.array([27], dtype=np.int64)}
    f = tb.fold_ranks(dump, regions(), FREQ, kmax=10000)
    assert bool(f["assumed"]) is True
    assert int(f["off_current"][0]) == 1  # item 1 is second in its block by row
    assert int(f["base"][0]) == 25
    assert int(f["ceiling"][0]) == 26


def test_a_miss_is_capped_for_every_column():
    dump = {"targets": np.array([1], dtype=np.int64),
            "rank": np.array([10001], dtype=np.int64),
            "fav_item_rank": np.array([10001], dtype=np.int64)}
    f = tb.fold_ranks(dump, regions(), FREQ, kmax=10000)
    assert int(f["current"][0]) == 10001
    assert int(f["popularity"][0]) == 10001
    assert int(f["ceiling"][0]) == 10001


def test_load_freq_accepts_a_target_list_and_a_count_array(tmp_path):
    targets = np.array([0, 0, 2, 5], dtype=np.int64)
    path = tmp_path / "targets.npy"
    np.save(path, targets)
    counts = tb.load_freq(str(path), str(tmp_path), n_items=9)
    assert list(counts) == [2, 0, 1, 0, 0, 1, 0, 0, 0]
    # The same array read as a per-item count vector passes through untouched.
    same = tb.load_freq(str(path), str(tmp_path), n_items=4)
    assert list(same) == list(targets)


def test_load_freq_refuses_indices_past_the_catalogue(tmp_path):
    np.save(tmp_path / "targets.npy", np.array([0, 99], dtype=np.int64))
    with pytest.raises(SystemExit, match="catalogue has"):
        tb.load_freq(str(tmp_path / "targets.npy"), str(tmp_path), n_items=9)


def test_by_band_labels_the_tail_explicitly():
    g = np.array([1, 2, 3, 6, 9, 30], dtype=np.int64)
    ranks = {"current": np.array([1, 2, 3, 6, 9, 30], dtype=np.int64)}
    rows = tb.by_band(g, ranks, (1,))
    assert [r["band"] for r in rows] == ["1-1", "2-2", "3-4", "5-8", "9-16", "17+"]
    assert [r["rows"] for r in rows] == [1, 1, 1, 1, 1, 1]

"""Tests for `budget_match.py`, the matched-candidate-budget comparison.

The fixture is deliberately small enough to check by hand: 8 catalogue items over a
3x3 SID, so the four tuples used below carry 2, 1, 3 and 2 items. Every expected
number in these tests is a count, not a model output.

The load-bearing test is `test_tiger_reach_never_counts_a_miss_as_a_hit`. A beam miss
is stored as `beam_width + 1`, so `rank <= N` scores a miss as a hit on every row whose
shortlist is at least `beam_width + 1` items long, which on this data is most of them.
That turns 12.58% into 100% and looks like a discovery.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import budget_match as BM


SID_SIZES = (3, 3)
# item -> codes. Buckets: (0,0) = items 0,1 (2 items); (0,1) = item 2 (1);
# (1,0) = items 3,4,5 (3); (2,2) = items 6,7 (2).
CATALOG_CODES = np.asarray([
    [0, 0], [0, 0], [0, 1], [1, 0], [1, 0], [1, 0], [2, 2], [2, 2],
], dtype=np.int32)
VAL_TARGETS = np.asarray([3, 2, 5, 7], dtype=np.int64)   # (1,0) (0,1) (1,0) (2,2)
TRAIN_TARGETS = np.asarray([0, 1, 2], dtype=np.int64)    # makes items 0,1,2 "seen"
BEAM_WIDTH = 2
# Row 0: rank 1, tuples (1,0)+(0,0) -> 3 + 2 = 5 items named
# Row 1: MISS (target (0,1) is not in the beam), same 5 items named
# Row 2: rank 2, tuples (2,2)+(1,0) -> 2 + 3 = 5 items named
# Row 3: rank 1, a SHORT beam (one tuple) -> 2 items named
BEAM_RANK = np.asarray([1, BEAM_WIDTH + 1, 2, 1], dtype=np.int64)
BEAM_TUPLES = np.asarray([
    [[1, 0], [0, 0]],
    [[0, 0], [1, 0]],
    [[2, 2], [1, 0]],
    [[2, 2], [-1, -1]],
], dtype=np.int32)
ARM_RANK = np.asarray([1, 10001, 2, 10001], dtype=np.int64)

NAMED_ALL = np.asarray([5, 5, 5, 2], dtype=np.int64)
# Seen = train+val targets = items {0,1,2,3,5,7}: (0,0) holds 0 and 1 (2 of them),
# (0,1) holds 2 (1), (1,0) holds 3,4,5 (2 seen), (2,2) holds 6,7 (1 seen).
NAMED_SEEN = np.asarray([4, 4, 3, 1], dtype=np.int64)


def _write_dataset(tmp_path):
    data = tmp_path / "digr_data"
    for split in ("train", "val"):
        (data / split).mkdir(parents=True)
    with open(data / "vocab.json", "w", encoding="utf-8") as f:
        json.dump({"sid_sizes": list(SID_SIZES), "num_levels": 2,
                   "sid_offsets": [0, 0], "sep_id": 1}, f)
    np.save(data / "catalog_sid_codes.npy", CATALOG_CODES)
    np.save(data / "train" / "targets.npy", TRAIN_TARGETS)
    np.save(data / "val" / "targets.npy", VAL_TARGETS)
    return data


def _write_beam(path, sid_sizes=SID_SIZES, rank=BEAM_RANK, targets=VAL_TARGETS):
    np.savez_compressed(
        path, rows=np.arange(len(targets), dtype=np.int64), rank=rank,
        targets=targets, covered0=np.ones(len(targets), dtype=bool),
        tuples=BEAM_TUPLES, beam_width=np.int64(BEAM_WIDTH),
        sid_sizes=np.asarray(sid_sizes, dtype=np.int64),
        n_rows=np.int64(len(targets)), n_split=np.int64(len(targets)),
    )


def _write_arm(path, rank=ARM_RANK, targets=VAL_TARGETS):
    np.savez_compressed(path, rows=np.arange(len(targets), dtype=np.int64),
                        targets=targets, rank=rank)


def _run(tmp_path, data, beam, arms, extra=()):
    out = tmp_path / "out"
    argv = ["--data_dir", str(data), "--beam", str(beam),
            "--out_dir", str(out), *extra]
    for name, path in arms:
        argv += ["--arm", f"{name}={path}"]
    return BM.main(argv)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_pack_matches_mixed_radix():
    keys = BM.pack(np.asarray([[0, 1], [1, 0], [2, 2]]), [3, 3])
    assert keys.tolist() == [1, 3, 8]


def test_pack_rejects_a_code_outside_its_width():
    with pytest.raises(ValueError, match="out of range"):
        BM.pack(np.asarray([[0, 3]]), [3, 3])


def test_bucket_multiplicities_are_the_catalogue_counts(tmp_path):
    buckets = BM.Buckets(CATALOG_CODES, list(SID_SIZES))
    assert len(buckets) == 4                      # four distinct tuples
    mult, found = buckets.lookup(BM.pack(np.asarray([[1, 0], [0, 1], [2, 2]]),
                                         list(SID_SIZES)))
    assert found.tolist() == [True, True, True]
    assert mult.tolist() == [3, 1, 2]


def test_an_absent_tuple_is_reported_not_silently_zero(tmp_path):
    buckets = BM.Buckets(CATALOG_CODES, list(SID_SIZES))
    mult, found = buckets.lookup(BM.pack(np.asarray([[2, 0]]), list(SID_SIZES)))
    assert found.tolist() == [False]
    assert mult.tolist() == [0]


# ---------------------------------------------------------------------------
# The whole comparison
# ---------------------------------------------------------------------------

def test_matched_budget_uses_each_rows_named_items(tmp_path):
    data = _write_dataset(tmp_path)
    beam, arm = tmp_path / "beam.npz", tmp_path / "arm.npz"
    _write_beam(beam)
    _write_arm(arm)
    assert _run(tmp_path, data, beam, [("arm", arm)]) == 0

    with open(tmp_path / "out" / "budget_match.json", encoding="utf-8") as f:
        got = json.load(f)

    assert got["n_rows"] == 4
    assert got["k_min"] == 2 and got["k_max"] == 5
    assert got["k_mean"] == pytest.approx(float(NAMED_ALL.mean()))
    assert got["k_median"] == 5
    assert got["k_mean_seen"] == pytest.approx(float(NAMED_SEEN.mean()))

    arm_row = got["arms"][1]
    # rank <= N per row: 1<=5, 10001<=5 (no), 2<=5, 10001<=2 (no)
    assert arm_row["match_all"] == pytest.approx(0.5)
    # seen budget per row is 4, 4, 3, 1: rows 0 and 2 can be hits
    assert arm_row["match_seen"] == pytest.approx(0.5)
    assert arm_row["own_at_10"] == pytest.approx(0.5)
    # nDCG@N_r over ranks 1, 10001, 2, 10001 with budgets 5, 5, 5, 2:
    # hits at rank 1 and rank 2, weighted by 1/log2(r+1).
    assert arm_row["ndcg_all"] == pytest.approx((1.0 + 1.0 / np.log2(3.0)) / 4)
    assert arm_row["mrr10"] == pytest.approx((1.0 + 0.5) / 4)
    assert got["seen_label"] == "target-set"


def test_rank_aware_metrics_are_blank_for_a_tuple_axis_arm(tmp_path):
    """TIGER's positions are tuple positions, so nDCG and MRR are not reportable."""
    data = _write_dataset(tmp_path)
    beam, arm = tmp_path / "beam.npz", tmp_path / "arm.npz"
    _write_beam(beam)
    _write_arm(arm)
    _run(tmp_path, data, beam, [("arm", arm)])
    with open(tmp_path / "out" / "budget_match.json", encoding="utf-8") as f:
        got = json.load(f)

    tiger_row = got["arms"][0]
    assert np.isnan(tiger_row["ndcg_all"])
    assert np.isnan(tiger_row["mrr10"])
    # Its reach is still reportable, because that needs no axis conversion.
    assert tiger_row["match_all"] == pytest.approx(got["tiger_reach"])
    # And the item-axis arm still gets both, so the blank is per-arm, not global.
    assert not np.isnan(got["arms"][1]["ndcg_all"])


def test_tiger_reach_never_counts_a_miss_as_a_hit(tmp_path):
    data = _write_dataset(tmp_path)
    beam, arm = tmp_path / "beam.npz", tmp_path / "arm.npz"
    _write_beam(beam)
    _write_arm(arm)
    _run(tmp_path, data, beam, [("arm", arm)])
    with open(tmp_path / "out" / "budget_match.json", encoding="utf-8") as f:
        got = json.load(f)

    assert got["tiger_decoded"] == 3               # rows 0, 2 and 3
    assert got["tiger_reach"] == pytest.approx(0.75)
    # Row 1 is a miss with 5 items named, so `rank <= N` would call it a hit. That is
    # the number this script exists to avoid printing.
    naive = float(((BEAM_RANK <= NAMED_ALL)).mean())
    assert naive == 1.0
    assert got["tiger_reach"] < naive


def test_arm_targets_must_align_with_the_beam(tmp_path):
    data = _write_dataset(tmp_path)
    beam, arm = tmp_path / "beam.npz", tmp_path / "arm.npz"
    _write_beam(beam)
    _write_arm(arm, targets=np.asarray([3, 5, 2, 7]))   # rows 1 and 2 swapped
    with pytest.raises(SystemExit, match="different target"):
        _run(tmp_path, data, beam, [("arm", arm)])


def test_beam_dump_on_a_different_sid_is_refused(tmp_path):
    data = _write_dataset(tmp_path)
    beam, arm = tmp_path / "beam.npz", tmp_path / "arm.npz"
    _write_beam(beam, sid_sizes=(4, 4))
    _write_arm(arm)
    with pytest.raises(SystemExit, match="different SID"):
        _run(tmp_path, data, beam, [("arm", arm)])


def test_chunked_arm_directory_reassembles_in_row_order(tmp_path):
    data = _write_dataset(tmp_path)
    beam = tmp_path / "beam.npz"
    _write_beam(beam)
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    np.savez_compressed(chunk_dir / "arm_000000.npz",
                        rows=np.asarray([0, 1]), targets=VAL_TARGETS[:2],
                        rank=ARM_RANK[:2])
    np.savez_compressed(chunk_dir / "arm_000002.npz",
                        rows=np.asarray([2, 3]), targets=VAL_TARGETS[2:],
                        rank=ARM_RANK[2:])
    # No stray file in this directory on purpose: `--arm` reads every *.npz it finds,
    # so a second baseline here is refused rather than ignored (see the mixed test).
    assert _run(tmp_path, data, beam, [("arm", chunk_dir)]) == 0

    with open(tmp_path / "out" / "budget_match.json", encoding="utf-8") as f:
        got = json.load(f)
    arm_row = got["arms"][1]
    assert arm_row["match_all"] == pytest.approx(0.5)
    assert arm_row["own_at_1"] == pytest.approx(0.25)


def test_an_explicit_seen_set_replaces_the_target_set(tmp_path):
    data = _write_dataset(tmp_path)
    beam, arm = tmp_path / "beam.npz", tmp_path / "arm.npz"
    _write_beam(beam)
    _write_arm(arm)
    items = tmp_path / "clicked.npy"
    # Only items 3 and 4 (both (1,0)) are "clicked", so the tuple budget collapses:
    # row 0 names (1,0)+(0,0) -> 2, row 1 the same 2, row 2 (2,2)+(1,0) -> 2, row 3 -> 0.
    np.save(items, np.asarray([3, 4], dtype=np.int64))
    assert _run(tmp_path, data, beam, [("arm", arm)],
                extra=["--seen_items", str(items)]) == 0
    with open(tmp_path / "out" / "budget_match.json", encoding="utf-8") as f:
        got = json.load(f)
    assert got["seen_label"] == "ever-clicked"
    assert got["k_mean_seen"] == pytest.approx(1.5)
    # 1<=2, 10001<=2 (no), 2<=2, 10001<=0 (no)
    assert got["arms"][1]["match_seen"] == pytest.approx(0.5)


def test_a_partial_arm_dump_is_refused(tmp_path):
    data = _write_dataset(tmp_path)
    beam, arm = tmp_path / "beam.npz", tmp_path / "arm.npz"
    _write_beam(beam)
    np.savez_compressed(arm, rows=np.asarray([0, 1]), targets=VAL_TARGETS[:2],
                        rank=ARM_RANK[:2])
    with pytest.raises(SystemExit, match="have no rank"):
        _run(tmp_path, data, beam, [("arm", arm)])


def test_a_directory_of_one_baseline_loads_under_any_arm_name(tmp_path):
    """The directory is already scoped to one arm, so the label is free.

    That is what lets two windows of the same baseline be compared side by side, which
    the baseline script needs because co-click's ranks depend on the window.
    """
    data = _write_dataset(tmp_path)
    beam = tmp_path / "beam.npz"
    _write_beam(beam)
    chunk = tmp_path / "ranks" / "h20" / "coclick"
    chunk.mkdir(parents=True)
    np.savez(chunk / "coclick_000000.npz",
             rows=np.arange(4, dtype=np.int64),
             targets=np.asarray(VAL_TARGETS, dtype=np.int64),
             rank=np.asarray(ARM_RANK, dtype=np.int64))
    assert _run(tmp_path, data, beam, [("coclick_h20", chunk)]) == 0
    with open(tmp_path / "out" / "budget_match.json", encoding="utf-8") as f:
        got = json.load(f)
    assert [a["arm"] for a in got["arms"]] == ["TIGER (beam)", "coclick_h20"]
    assert got["arms"][1]["match_all"] == pytest.approx(0.5)


def test_a_directory_mixing_baselines_is_refused(tmp_path):
    data = _write_dataset(tmp_path)
    beam = tmp_path / "beam.npz"
    _write_beam(beam)
    chunk = tmp_path / "ranks"
    chunk.mkdir()
    for kind in ("coclick", "popularity"):
        np.savez(chunk / f"{kind}_000000.npz",
                 rows=np.arange(4, dtype=np.int64),
                 targets=np.asarray(VAL_TARGETS, dtype=np.int64),
                 rank=np.asarray(ARM_RANK, dtype=np.int64))
    with pytest.raises(SystemExit) as exc:
        _run(tmp_path, data, beam, [("mixed", chunk)])
    assert "more than one baseline" in str(exc.value)
    # and the refusal points at the form that does work
    assert "_*.npz" in str(exc.value)


def test_a_glob_picks_one_baseline_out_of_a_shared_window_directory(tmp_path):
    """The layout run_baselines.sh actually produces: both kinds in one directory.

    A directory of chunks cannot be read there, so the glob is the supported form and
    it has to select by name rather than by position.
    """
    data = _write_dataset(tmp_path)
    beam = tmp_path / "beam.npz"
    _write_beam(beam)
    shared = tmp_path / "baselines" / "h20"
    shared.mkdir(parents=True)
    for kind, rank in (("coclick", ARM_RANK), ("popularity", [10001] * 4)):
        for lo in (0, 2):
            np.savez(shared / f"{kind}_{lo:06d}.npz",
                     rows=np.arange(lo, lo + 2, dtype=np.int64),
                     targets=np.asarray(VAL_TARGETS[lo:lo + 2], dtype=np.int64),
                     rank=np.asarray(rank[lo:lo + 2], dtype=np.int64))

    out = BM.main(["--data_dir", str(data), "--beam", str(beam),
                   "--out_dir", str(tmp_path / "out"),
                   "--arm", f"coclick_h20={shared}/coclick_*.npz",
                   "--arm", f"popularity_h20={shared}/popularity_*.npz"])
    assert out == 0
    got = json.loads((tmp_path / "out" / "budget_match.json").read_text())
    arms = {a["arm"]: a for a in got["arms"]}
    assert set(arms) == {"TIGER (beam)", "coclick_h20", "popularity_h20"}
    # the two chunks are reassembled in row order, so coclick scores and popularity does not
    assert arms["coclick_h20"]["match_all"] == pytest.approx(0.5)
    assert arms["popularity_h20"]["match_all"] == pytest.approx(0.0)


def test_a_glob_that_matches_nothing_is_refused(tmp_path):
    data = _write_dataset(tmp_path)
    beam = tmp_path / "beam.npz"
    _write_beam(beam)
    with pytest.raises(SystemExit) as exc:
        _run(tmp_path, data, beam, [("gone", tmp_path / "nope_*.npz")])
    assert "matched no files" in str(exc.value)

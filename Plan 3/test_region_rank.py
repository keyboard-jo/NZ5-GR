#!/usr/bin/env python
"""Tests for the two functions in region_rank.py that do arithmetic.

`run()` itself needs a real checkpoint and a real preprocessed dir, so it cannot
run here. What CAN run here is the part that would silently produce a plausible
table: the count of tuples and items above a threshold, and the brute-force rank
that count is checked against.

`count_above` is compared with a literal double loop, and over chunk sizes that
cut the tuple list at uneven boundaries, because an accumulator that is reset or
mis-sliced at a boundary still returns a monotone-looking number.

Two mistakes in this file have already produced a full, plausible, wrong table,
so both now have a test that fails on them:

1. Scoring tuples from an averaged embedding vector built out of raw SID codes
   instead of token ids. The old test asserted the raw-code answer, which is how
   it survived. The scores now come from `model.item_logits`, and
   `test_brute_force_agrees_with_the_counted_rank` is what pins that down.
2. Treating `item_rank` and `region_rank` as independent at rank 1. They are not:
   rank 1 needs *no* tuple above, so a tie can never separate them.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Model Training"))

torch = pytest.importorskip("torch")

import region_lib as RL  # noqa: E402
import region_rank as RR  # noqa: E402


def _fixture(n_rows=37, n_reg=53, seed=0):
    """Region scores, target scores and tuple sizes on a small integer grid.

    Integer values keep every comparison exact, so a tie is a real tie and not a
    one-ulp coincidence that flips between chunkings. `counts` sums to a
    plausible catalogue size and contains ones, twos and a fat tuple.
    """
    g = torch.Generator().manual_seed(seed)
    S = torch.randint(-4, 5, (n_rows, n_reg), generator=g).float()
    tgt_score = torch.randint(-4, 5, (n_rows,), generator=g).float()
    counts = torch.randint(1, 5, (n_reg,), generator=g)
    counts[:3] = torch.tensor([1, 1, 7], dtype=counts.dtype)
    return S, tgt_score, counts


def _brute_force(S, tgt_score, counts):
    """The definition, written out: loop over every row and every tuple."""
    S = np.asarray(S)
    tgt_score = np.asarray(tgt_score)
    counts = np.asarray(counts, dtype=np.int64)
    ar = np.zeros(S.shape[0], dtype=np.int64)
    ai = np.zeros(S.shape[0], dtype=np.int64)
    for r in range(S.shape[0]):
        for c in range(S.shape[1]):
            if S[r, c] > tgt_score[r]:
                ar[r] += 1
                ai[r] += counts[c]
    return ar, ai


def test_count_above_matches_a_brute_force_double_loop():
    S, tgt_score, counts = _fixture()
    ar, ai = RR.count_above(S, tgt_score, counts, region_chunk=53)
    want_ar, want_ai = _brute_force(S, tgt_score, counts)
    assert np.array_equal(ar.cpu().numpy(), want_ar)
    assert np.array_equal(ai.cpu().numpy(), want_ai)


def test_count_above_is_unchanged_by_chunk_boundaries():
    S, tgt_score, counts = _fixture(n_rows=9, n_reg=97, seed=4)
    want_ar, want_ai = _brute_force(S, tgt_score, counts)
    for chunk in (1, 2, 3, 5, 31, 96, 97, 1000):
        ar, ai = RR.count_above(S, tgt_score, counts, region_chunk=chunk)
        assert np.array_equal(ar.cpu().numpy(), want_ar), f"chunk {chunk}"
        assert np.array_equal(ai.cpu().numpy(), want_ai), f"chunk {chunk}"


def test_a_tie_is_never_charged_to_the_target():
    """Strict `>`, the favourable convention every number here is reported under.

    A tuple scoring exactly the target's score is not above it, so the target
    keeps the first position of its own tie group. `>=` would move the target to
    the end of the group and, with a 315-item group, could cost tens of positions.
    """
    S = torch.tensor([[5.0, 5.0, 5.0, 4.0]])
    tgt_score = torch.tensor([5.0])
    counts = torch.tensor([1, 3, 9, 1])
    ar, ai = RR.count_above(S, tgt_score, counts, region_chunk=4)
    assert int(ar[0]) == 0
    assert int(ai[0]) == 0
    # tying at the bottom of the range: the two 5.0 tuples and nothing else
    ar, ai = RR.count_above(S, torch.tensor([4.0]), counts, region_chunk=4)
    assert int(ar[0]) == 3
    assert int(ai[0]) == 13


def test_the_item_count_is_weighted_by_tuple_size():
    S = torch.tensor([[4.0, 3.0, 2.0]])
    counts = torch.tensor([1, 1, 250])
    ar, ai = RR.count_above(S, torch.tensor([1.0]), counts, region_chunk=3)
    assert int(ar[0]) == 3
    assert int(ai[0]) == 252, "the fat tuple must contribute 250 items, not 1"


def test_item_rank_is_never_below_the_region_rank():
    S, tgt_score, counts = _fixture(n_rows=200, n_reg=80)
    ar, ai = RR.count_above(S, tgt_score, counts, region_chunk=13)
    assert bool((ar <= ai).all())


def test_rank_one_is_identical_for_both_axes():
    """A tie cannot separate the two ranks at rank 1, so `R@1` cannot show a gain.

    This is not a bug, it is the shape of the question: rank 1 requires that no
    tuple scores above the target's, which is the same condition whether a tuple
    holds one item or 315. Any verdict that reads a counterfactual gain at `R@1`
    is reading the same number twice.
    """
    S, tgt_score, counts = _fixture(n_rows=120, n_reg=64, seed=7)
    ar, ai = RR.count_above(S, tgt_score, counts, region_chunk=17)
    region_rank = (ar + 1).numpy()
    item_rank = (ai + 1).numpy()
    assert np.array_equal(region_rank == 1, item_rank == 1)


def test_brute_force_item_rank_agrees_with_the_counted_rank():
    """The end-to-end invariant, on a tiny catalogue with real collisions.

    `item_rank` is `1 + sum of sizes above`, while `brute_force_item_rank` walks a
    materialised item-level logits row. They share nothing but the scores, so
    agreement means the item->tuple map, the sizes and the weights all line up.
    This is the check that catches a mis-aligned catalogue, and it needs no
    reference file.
    """
    codes = np.array([[1, 2], [1, 2], [3, 4], [1, 5], [3, 4], [1, 2]], dtype=np.int64)
    reg = RL.Regions(codes, [8, 8])
    rows, n_reg = 5, reg.n_regions
    S = torch.randint(-3, 4, (rows, n_reg), generator=torch.Generator().manual_seed(11)).float()
    # A region's score is its members' score, which is why the tuple factorisation
    # is exact rather than approximate.
    item_logits = S.index_select(1, torch.as_tensor(reg.region_of_item))
    tgt_items = np.array([0, 3, 5, 1, 4])
    tgt_score = item_logits[torch.arange(rows), torch.as_tensor(tgt_items)]
    ar, ai = RR.count_above(S, tgt_score, torch.as_tensor(reg.counts), region_chunk=2)
    for r in range(rows):
        want = RR.brute_force_item_rank(item_logits[r], int(tgt_items[r]))
        assert int(ai[r]) + 1 == want, f"row {r}"
        assert int(ar[r]) + 1 <= want, f"row {r}"


def test_brute_force_item_rank_counts_strictly_above():
    logits = torch.tensor([0.5, 2.0, 2.0, -1.0])
    assert RR.brute_force_item_rank(logits, 1) == 1
    assert RR.brute_force_item_rank(logits, 2) == 1
    assert RR.brute_force_item_rank(logits, 0) == 3
    assert RR.brute_force_item_rank(logits, 3) == 4


def test_summarise_accepts_the_ranks_this_produces():
    """The two modules agree on conventions: rank arrays, 1-based, >= 1."""
    S, tgt_score, counts = _fixture(n_rows=120, n_reg=60)
    ar, ai = RR.count_above(S, tgt_score, counts, region_chunk=11)
    g = counts[torch.as_tensor(tgt_score.long().abs() % int(counts.numel()))]
    out = RL.summarise((ai + 1).numpy(), (ar + 1).numpy(), g.numpy(), ks=(1, 10))
    assert out["n_rows"] == 120
    assert 0.0 <= out["item_r1"] <= 1.0
    assert all(row["inflation"] >= 0 for row in out["groups"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_topk_rank_matches_the_evaluators_own_expression():
    """`topk_rank` must be the same operation `evaluate()` performs, not a rewrite."""
    scores = torch.tensor([[0.5, 3.0, 1.0, 2.0, -1.0]])
    col = torch.tensor([3])
    # Written out the way `Model Training/train_digr.py` writes it.
    top = scores.topk(scores.shape[1], dim=1).indices
    match = top == col.unsqueeze(1)
    expected = int(match.to(torch.int64).argmax(dim=1)) + 1
    got = int(RR.topk_rank(scores, col, kmax=10)[0])
    assert got == expected == 2


def test_topk_rank_charges_the_target_for_its_own_tie():
    """The published convention is `topk` position, so a tie-mate can land ahead.

    With two columns at exactly the same score and a column index order that puts
    the mate first, the target is rank 2 here even though nothing scores strictly
    above it. This is the difference that made run 3 disagree with the dump, so it
    is pinned rather than left to whichever `topk` the GPU happens to have.
    """
    scores = torch.tensor([[1.0, 1.0, 0.0]])
    assert int(RR.topk_rank(scores, torch.tensor([1]), kmax=10)[0]) == 2
    assert int(RR.topk_rank(scores, torch.tensor([0]), kmax=10)[0]) == 1
    # `count_above` reports 0 for both columns, which is why the two are not
    # interchangeable: it cannot see the tie, and the evaluator charges for it.
    ar, ai = RR.count_above(scores, torch.tensor([1.0]), torch.tensor([1, 1, 1]),
                             region_chunk=64)
    assert int(ar[0]) == 0 and int(ai[0]) == 0


def test_topk_rank_stores_a_miss_as_kmax_plus_one():
    scores = torch.tensor([[9.0, 8.0, 7.0, 6.0]])
    # Column 1 is inside the top 2 (second), column 3 falls outside it.
    assert int(RR.topk_rank(scores, torch.tensor([1]), kmax=2)[0]) == 2
    assert int(RR.topk_rank(scores, torch.tensor([3]), kmax=2)[0]) == 3

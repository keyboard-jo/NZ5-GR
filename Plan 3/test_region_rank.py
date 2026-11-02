#!/usr/bin/env python
"""Tests for the two functions in region_rank.py that do arithmetic.

`run()` itself needs a real checkpoint and a real preprocessed dir, so it cannot
run here. What CAN run here is the part that would silently produce a plausible
table: the count of tuples and items above a threshold, and the per-tuple vector.

`count_above` is checked against a brute-force full matrix, including chunk sizes
that cut the tuple list and the row list at uneven boundaries, because an
accumulator that is reset or mis-sliced at a boundary would still return a
monotone-looking number.
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


class _Backbone(torch.nn.Module):
    def __init__(self, vocab: int, d: int) -> None:
        super().__init__()
        self.tok = torch.nn.Embedding(vocab, d)

    def get_input_embeddings(self):
        return self.tok


class _Model(torch.nn.Module):
    def __init__(self, vocab: int, d: int, l2_normalize: bool = False,
                 logit_chunk: int = 8) -> None:
        super().__init__()
        self.backbone = _Backbone(vocab, d)
        self.l2_normalize = l2_normalize
        self.logit_chunk = logit_chunk
        self.temperature = 0.07


def _fixture(n_rows=37, n_reg=53, d=6, seed=0):
    """Integer-valued H and V so every dot product is exact in fp32.

    With continuous values a few pairs land within one ulp of the threshold and
    flip between GEMM blockings, which is a property of float ordering and not of
    this code. Keeping the values on a small integer grid removes that noise so
    the test can demand exact equality across chunk sizes.
    """
    g = torch.Generator().manual_seed(seed)
    H = torch.randint(-2, 3, (n_rows, d), generator=g).float()
    V = torch.randint(-2, 3, (n_reg, d), generator=g).float()
    tgt = torch.randint(0, n_reg, (n_rows,), generator=g)
    counts = torch.randint(1, 6, (n_reg,), generator=g).float()
    temp = 1.0
    tgt_score = (H @ V.t())[torch.arange(n_rows), tgt] / temp
    return H, V, tgt_score, counts, temp


def _brute(H, V, tgt_score, counts, temp):
    scores = (H @ V.t()) / temp
    hit = scores > tgt_score.unsqueeze(1)
    above_regions = hit.sum(dim=1)
    above_items = (hit.to(counts.dtype) @ counts).to(torch.int64)
    return above_regions, above_items


def test_count_above_matches_a_brute_force_full_matrix():
    H, V, tgt_score, counts, temp = _fixture()
    got_r, got_i = RR.count_above(H, V, tgt_score, counts, temp,
                                  region_chunk=53, row_batch=37)
    want_r, want_i = _brute(H, V, tgt_score, counts, temp)
    assert torch.equal(got_r, want_r)
    assert torch.equal(got_i, want_i)


def test_count_above_is_unchanged_by_chunk_boundaries():
    H, V, tgt_score, counts, temp = _fixture(n_rows=37, n_reg=53)
    want_r, want_i = _brute(H, V, tgt_score, counts, temp)
    for ck in (1, 5, 17, 53, 200):
        for rb in (1, 4, 37, 100):
            got_r, got_i = RR.count_above(H, V, tgt_score, counts, temp,
                                          region_chunk=ck, row_batch=rb)
            assert torch.equal(got_r, want_r), f"chunk {ck} rows {rb}"
            assert torch.equal(got_i, want_i), f"chunk {ck} rows {rb}"


def test_count_above_never_charges_a_tie_to_the_target():
    """Every tuple scores identically, so nothing is above and every rank is 1.

    This is the property the project reports under: `rank = 1 + #{s_i > s_t}`.
    If the comparison ever became `>=`, a shared tuple would count itself and the
    measured ranks would collapse, which is the single easiest way to get this
    wrong.
    """
    H, V, _, counts, temp = _fixture()
    V = torch.ones_like(V)
    # Same arithmetic as the scorer, so "identical" is identical and not one ulp
    # apart: the point of the test is the comparison, not the rounding.
    tgt_score = (H @ V.t())[:, 0] / temp
    got_r, got_i = RR.count_above(H, V, tgt_score, counts, temp,
                                  region_chunk=17, row_batch=8)
    assert int(got_r.max()) == 0
    assert int(got_i.max()) == 0


def test_count_above_weights_the_item_count_by_tuple_size():
    """A single tuple above the target must add its whole size, not 1."""
    H = torch.tensor([[0.0, 0.0, 1.0]])   # picks the third coordinate of each row
    V = torch.tensor([[0.0, 0.0, -1.0],   # the target sits here, score -1
                      [0.0, 0.0, 2.0]])   # this one scores +2, so it is above
    counts = torch.tensor([4.0, 7.0])
    tgt_score = torch.zeros(1)            # threshold 0, so only V[1] clears it
    got_r, got_i = RR.count_above(H, V, tgt_score, counts, 1.0,
                                  region_chunk=2, row_batch=1)
    assert int(got_r[0]) == 1             # one tuple above
    assert int(got_i[0]) == 7             # but seven items above


def test_region_vectors_average_the_codes_of_the_first_member():
    vocab, d = 40, 5
    model = _Model(vocab, d, l2_normalize=False)
    with torch.no_grad():
        model.backbone.tok.weight.copy_(
            torch.arange(vocab * d, dtype=torch.float32).reshape(vocab, d) / 100.0
        )
    codes = np.array([[1, 2], [1, 2], [3, 4], [1, 5]], dtype=np.int64)
    reg = RL.Regions(codes, [8, 8])
    V = RR.region_vectors(model, reg, torch.float32)
    w = model.backbone.tok.weight
    for i in range(reg.n_regions):
        c = reg.codes[i]
        want = (w[int(c[0])] + w[int(c[1])]) / 2.0
        assert torch.allclose(V[i], want, atol=1e-6), f"region {i}"


def test_region_vectors_honour_l2_normalize():
    """With `l2_normalize` the codes are normalised BEFORE the sum.

    That is what `precompute_vectors` does, and the item ranks this script has to
    reproduce were produced by that path, so normalising after the sum instead
    would quietly disagree with the dump.
    """
    vocab, d = 40, 5
    model = _Model(vocab, d, l2_normalize=True)
    with torch.no_grad():
        model.backbone.tok.weight.copy_(
            torch.randn(vocab, d, generator=torch.Generator().manual_seed(3))
        )
    codes = np.array([[1, 2], [3, 4]], dtype=np.int64)
    reg = RL.Regions(codes, [8, 8])
    V = RR.region_vectors(model, reg, torch.float32)
    w = model.backbone.tok.weight
    for i in range(reg.n_regions):
        c = reg.codes[i]
        want = (torch.nn.functional.normalize(w[int(c[0])], dim=-1)
                + torch.nn.functional.normalize(w[int(c[1])], dim=-1)) / 2.0
        assert torch.allclose(V[i], want, atol=1e-6)


def test_item_rank_is_never_below_the_region_rank():
    """`region_rank <= item_rank` always, since every region above adds >=1 item."""
    H, V, tgt_score, counts, temp = _fixture(n_rows=200, n_reg=80)
    a_r, a_i = RR.count_above(H, V, tgt_score, counts, temp,
                              region_chunk=13, row_batch=64)
    assert bool((a_r <= a_i).all())


def test_summarise_accepts_the_ranks_this_produces():
    """The two modules agree on conventions: rank arrays, 1-based, >= 1."""
    H, V, tgt_score, counts, temp = _fixture(n_rows=120, n_reg=60)
    a_r, a_i = RR.count_above(H, V, tgt_score, counts, temp,
                              region_chunk=11, row_batch=32)
    item_rank = (a_i + 1).numpy()
    region_rank = (a_r + 1).numpy()
    g = torch.randint(1, 6, (120,),
                      generator=torch.Generator().manual_seed(9)).numpy().astype(np.int64)
    s = RL.summarise(item_rank, region_rank, g, ks=(1, 10))
    assert s["n_rows"] == 120
    assert s["item_r1"] <= s["region_r1"] + 1e-12
    # nan when no row is at rank 1, which this integer fixture often is; the
    # invariant worth pinning is that the ceiling is never below the measured
    # value, never that it is above zero.
    gain = s["counterfactual_gain_1"]
    assert gain != gain or gain >= 1.0 - 1e-12


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

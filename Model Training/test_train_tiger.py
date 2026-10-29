"""Tests for `train_tiger.py`: prefix keys, the prefix tree, beam search, the
batching layout, and the ranking maths.

CPU only, no backbone, no dataset on disk. Everything here is a pure function or a
stub model, so the whole file runs in well under a second.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import train_tiger as tt


# ---------------------------------------------------------------------------
# Prefix keys
# ---------------------------------------------------------------------------

def test_pack_prefix_is_injective_and_ordered():
    radix = [4, 3, 2]
    codes = np.array([[c0, c1, c2]
                      for c0 in range(4) for c1 in range(3) for c2 in range(2)])
    keys = tt.pack_prefix(codes, radix)
    assert len(np.unique(keys)) == len(codes), "every tuple needs its own key"
    # Lexicographic order of the tuple must equal numeric order of the key, which is
    # what lets `searchsorted` find the run of rows sharing a prefix.
    order = np.lexsort([codes[:, 2], codes[:, 1], codes[:, 0]])
    assert np.all(np.diff(keys[order]) > 0)


def test_pack_prefix_rejects_overflow():
    with pytest.raises(ValueError, match="does not fit in int64"):
        tt.pack_prefix(np.zeros((1, 5), dtype=np.int64), [65536] * 5)


def test_pack_prefix_rejects_codes_out_of_range():
    with pytest.raises(ValueError, match="out of range at level 0"):
        tt.pack_prefix(np.array([[4, 0]], dtype=np.int64), [4, 3])


def test_pack_prefix_rejects_one_dimensional_input():
    with pytest.raises(ValueError, match="expects a 2-d array"):
        tt.pack_prefix(np.array([0, 1, 2]), [4, 3])


# ---------------------------------------------------------------------------
# Prefix tree
# ---------------------------------------------------------------------------

def _toy_codes() -> np.ndarray:
    # 6 items, 2 levels of 4. Level 1 codes 2 and 3 are never used, so they must
    # never come out of `children`.
    return np.array([[0, 0], [0, 1], [0, 1], [1, 2], [1, 3], [2, 0]], dtype=np.int32)


def test_sid_index_children_are_the_real_continuations():
    index = tt.SidIndex(_toy_codes(), [4, 4])
    assert list(index.children(())) == [0, 1, 2, 3], "level 0 is unrestricted"
    assert list(index.children((0,))) == [0, 1]
    assert list(index.children((1,))) == [2, 3]
    assert list(index.children((2,))) == [0]
    assert list(index.children((3,))) == [], "prefix nobody owns has no children"
    assert list(index.children((0, 1))) == [], "past the last level"


def test_sid_index_contains():
    index = tt.SidIndex(_toy_codes(), [4, 4])
    assert index.contains((0, 1))
    assert index.contains((2, 0))
    assert not index.contains((0, 2)), "0-2 is a valid prefix but no item owns it"
    assert not index.contains((3, 0))


def test_sid_index_agrees_with_brute_force():
    rng = np.random.default_rng(0)
    radix = [5, 4, 3]
    codes = np.stack([rng.integers(0, r, size=200) for r in radix], axis=1).astype(np.int32)
    index = tt.SidIndex(codes, radix)
    owned = {tuple(int(v) for v in row) for row in codes}
    for _ in range(50):
        row = tuple(int(rng.integers(0, r)) for r in radix)
        assert index.contains(row) is (row in owned)
    for prefix_len in range(len(radix)):
        for _ in range(20):
            prefix = tuple(int(rng.integers(0, radix[l])) for l in range(prefix_len))
            expected = sorted({int(r[prefix_len]) for r in codes
                               if tuple(int(v) for v in r[:prefix_len]) == prefix})
            assert list(index.children(prefix)) == expected


def test_sid_index_rejects_a_shorter_width_list():
    with pytest.raises(ValueError, match="only 1 widths"):
        tt.SidIndex(_toy_codes(), [4])


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------

def test_beam_search_takes_the_best_tuple_and_respects_the_tree():
    # Level 0 prefers code 0, level 1 prefers code 3, but (0, 3) is not a real item,
    # so the tree has to push the answer to (0, 1).
    level_zero = np.log([0.6, 0.3, 0.1])
    level_two = {(0,): np.log([0.4, 0.6, 1e-9, 1e-9]), (1,): np.log([0.1, 0.2, 0.3, 0.4])}

    def step_fn(prefixes):
        if not prefixes[0]:
            return np.stack([level_zero])
        return np.stack([level_two[p] for p in prefixes])

    allowed_table = {(): np.array([0, 1]), (0,): np.array([0, 1]), (1,): np.array([2, 3])}

    beam = tt.beam_search(step_fn, [3, 4], beam_width=2,
                          allowed=lambda prefix, level: allowed_table[prefix])
    assert beam[0][1] == (0, 1), "best total = log .6 + log .6"
    assert {p for _, p in beam} <= {(0, 0), (0, 1), (1, 2), (1, 3)}


def test_beam_search_without_a_mask_keeps_every_code():
    def step_fn(prefixes):
        return np.zeros((len(prefixes), 3))

    beam = tt.beam_search(step_fn, [3, 3], beam_width=9)
    assert len(beam) == 9, "3 x 3 candidates, beam of 9 keeps them all"
    assert beam[0][1] == (0, 0), "ties keep candidate order"


def test_beam_search_rejects_a_bad_step_fn_shape():
    def step_fn(prefixes):
        return np.zeros((len(prefixes), 2))

    with pytest.raises(ValueError, match="expected"):
        tt.beam_search(step_fn, [3, 3])


def test_beam_search_stops_when_every_branch_dies():
    def step_fn(prefixes):
        return np.zeros((len(prefixes), 2))

    beam = tt.beam_search(step_fn, [2, 2], beam_width=2,
                          allowed=lambda prefix, level: np.array([], dtype=np.int64))
    assert beam == []


# ---------------------------------------------------------------------------
# Batching layout
# ---------------------------------------------------------------------------

class _Split:
    """The slice of `TokenizedSplit` that the dataset touches."""

    def __init__(self, rows, targets, pad=0):
        self.offsets = np.array([0] + [len(r) for r in rows], dtype=np.int64)
        self.offsets = np.cumsum(self.offsets)
        self.tokens = np.array([t for r in rows for t in r], dtype=np.int32)
        self.targets = np.asarray(targets, dtype=np.int64)
        self.lengths = np.array([len(r) for r in rows], dtype=np.int64)

    def __len__(self) -> int:
        """`run_beam` sizes its row count from the split, so the slice needs this."""
        return int(self.targets.size)


def _batch():
    split = _Split([[5, 6, 7], [8, 9, 10, 11, 12]], [0, 5])
    sid_input = np.array([[100, 200], [101, 201], [102, 202],
                          [103, 203], [104, 204], [105, 205]], dtype=np.int64)
    sid_codes = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [0, 1], [1, 0]],
                         dtype=np.int64)
    indices = np.arange(2, dtype=np.int64)
    return split, sid_input, sid_codes, indices


def test_dataset_returns_history_and_target_codes():
    split, sid_input, sid_codes, indices = _batch()
    ds = tt.TigerDataset(split, indices, sid_input, sid_codes, sep_id=1,
                         prefix_conditioning=False)
    history, codes_in, codes_out, target_row, row = ds[1]
    assert list(history) == [8, 9, 10, 11, 12]
    assert list(codes_in) == [105, 205], "target item 5"
    assert list(codes_out) == [1, 0]
    assert target_row == 5, "the catalogue row, which is what ranking needs"
    assert row == 1
    assert list(ds.lengths) == [3, 5]


def test_collate_without_prefix_conditioning_reads_the_separator():
    split, sid_input, sid_codes, indices = _batch()
    ds = tt.TigerDataset(split, indices, sid_input, sid_codes, sep_id=1,
                         prefix_conditioning=False)
    collate = tt.TigerCollate(sep_id=1, num_levels=2, prefix_conditioning=False)
    ids, mask, labels, positions, _targets, rows = collate([ds[0], ds[1]])

    assert ids.shape == (2, 6), "max history 5 + one separator"
    assert list(ids[0].tolist()) == [5, 6, 7, 1, 0, 0], "history, SEP, then padding"
    assert list(mask[0].tolist()) == [1, 1, 1, 1, 0, 0]
    assert list(mask[1].tolist()) == [1, 1, 1, 1, 1, 1]
    assert labels.tolist() == [[0, 0], [1, 0]]
    assert positions.tolist() == [[3, 3], [5, 5]], "every level reads the separator"
    assert rows.tolist() == [0, 1]


def test_collate_with_prefix_conditioning_appends_the_codes():
    split, sid_input, sid_codes, indices = _batch()
    ds = tt.TigerDataset(split, indices, sid_input, sid_codes, sep_id=1,
                         prefix_conditioning=True)
    collate = tt.TigerCollate(sep_id=1, num_levels=2, prefix_conditioning=True)
    ids, mask, labels, positions, _targets, _rows = collate([ds[0], ds[1]])

    assert ids.shape == (2, 8), "max history 5 + separator + 2 codes"
    assert list(ids[0].tolist()) == [5, 6, 7, 1, 100, 200, 0, 0]
    assert list(mask[0].tolist()) == [1, 1, 1, 1, 1, 1, 0, 0]
    # level 0 predicts from the separator, level 1 from the first code token.
    assert positions.tolist() == [[3, 4], [5, 6]]
    assert labels.tolist() == [[0, 0], [1, 0]]
    for row in range(2):
        assert int(positions[row, 1]) < ids.shape[1], "positions must be in range"


def test_collate_positions_stay_inside_the_padded_row():
    # The longest row is the one with the smallest history, which is where an
    # off-by-one would put a position past the end of the tensor.
    split = _Split([[1, 2, 3, 4, 5, 6, 7], [8]], [0, 1])
    sid_input = np.array([[10, 20], [30, 40]], dtype=np.int64)
    sid_codes = np.array([[0, 1], [1, 0]], dtype=np.int64)
    ds = tt.TigerDataset(split, np.arange(2, dtype=np.int64), sid_input, sid_codes,
                         sep_id=9, prefix_conditioning=True)
    collate = tt.TigerCollate(sep_id=9, num_levels=2, prefix_conditioning=True)
    ids, _mask, _labels, positions, _t, _r = collate([ds[0], ds[1]])
    assert int(positions.max()) < ids.shape[1]


# ---------------------------------------------------------------------------
# Ranking maths
# ---------------------------------------------------------------------------

def test_rank_metrics_known_values():
    ranks = np.array([1, 2, 3, 11, 200, 20000], dtype=np.int64)
    out = tt.rank_metrics(ranks, ks=(1, 10, 100), mrr_k=10)
    assert out["n"] == 6
    assert out["recall@1"] == pytest.approx(1 / 6)
    assert out["recall@10"] == pytest.approx(3 / 6)
    assert out["recall@100"] == pytest.approx(4 / 6)
    assert out["ndcg@10"] == pytest.approx(
        (1.0 + 1.0 / np.log2(3) + 1.0 / np.log2(4)) / 6
    )
    # MRR is truncated at 10 even though the cut-offs go to 100, so the rank-11 row
    # contributes nothing.
    assert out["mrr"] == pytest.approx((1.0 + 1 / 2 + 1 / 3) / 6)


def test_rank_metrics_empty():
    out = tt.rank_metrics(np.zeros(0, dtype=np.int64), ks=(1,), mrr_k=10)
    assert out["n"] == 0
    assert out["recall@1"] == 0.0


class _StubModel(torch.nn.Module):
    """Enough of `TigerRetriever` for `target_ranks`: fixed logits, fixed codes."""

    def __init__(self, logits, codes):
        super().__init__()
        self._logits = [torch.as_tensor(l, dtype=torch.float32) for l in logits]
        self.register_buffer("catalog_codes", torch.as_tensor(codes, dtype=torch.long))
        self.num_levels = int(codes.shape[1])

    @staticmethod
    def _fit(logits, rows):
        return [lg if lg.shape[0] == rows else lg.repeat(rows, 1) for lg in logits]

    def logits_from_state(self, states):
        return self._fit(self._logits, states.shape[0])

    def separator_states(self, input_ids, attention_mask):
        return torch.zeros(input_ids.shape[0], 4)

    def forward(self, input_ids, attention_mask, positions):
        return self._fit(self._logits, input_ids.shape[0])


@torch.inference_mode()
def _stub_eval(logits, codes, targets_rows, chunk=2):
    model = _StubModel(logits, codes)
    split = _Split([[5, 6] for _ in targets_rows], list(targets_rows))
    ds = tt.TigerDataset(split, np.arange(len(targets_rows), dtype=np.int64),
                         np.zeros((len(codes), 2), dtype=np.int64),
                         np.asarray(codes, dtype=np.int64), sep_id=1,
                         prefix_conditioning=False)
    collate = tt.TigerCollate(sep_id=1, num_levels=len(logits), prefix_conditioning=False)
    loader = DataLoader(ds, batch_size=len(targets_rows), shuffle=False,
                        collate_fn=collate)
    return tt.evaluate_tiger(model, loader, torch.device("cpu"), torch.float32, False,
                             catalog_chunk=chunk), loader


def test_target_ranks_matches_brute_force():
    codes = np.array([[0, 0], [1, 0], [0, 1], [2, 2]], dtype=np.int64)
    # Two levels of three classes, one row of logits.
    logits = [np.array([[0.0, 1.0, 2.0]]), np.array([[2.0, 1.0, 0.0]])]
    model = _StubModel(logits, codes)

    logp0 = torch.log_softmax(torch.tensor(logits[0][0]), dim=-1).numpy()
    logp1 = torch.log_softmax(torch.tensor(logits[1][0]), dim=-1).numpy()
    scores = np.array([logp0[c0] + logp1[c1] for c0, c1 in codes])

    for target in range(len(codes)):
        ranks = tt.target_ranks(model, torch.zeros(1, 4), torch.tensor([target]),
                                chunk=2)
        expected = 1 + int((scores > scores[target]).sum())
        assert int(ranks[0]) == expected, f"target {target}"


def test_target_ranks_counts_chunked_and_unchunked_identically():
    rng = np.random.default_rng(3)
    codes = rng.integers(0, 5, size=(40, 2)).astype(np.int64)
    logits = [rng.normal(size=(2, 5)), rng.normal(size=(2, 5))]
    model = _StubModel(logits, codes)
    states = torch.zeros(2, 4)
    targets = torch.tensor([0, 7])
    wide = tt.target_ranks(model, states, targets, chunk=40)
    narrow = tt.target_ranks(model, states, targets, chunk=3)
    assert wide.tolist() == narrow.tolist()


def test_target_ranks_breaks_ties_in_favour_of_the_target():
    # Every item has the same score, so nothing scores strictly higher.
    codes = np.array([[0, 0], [0, 0], [0, 0]], dtype=np.int64)
    logits = [np.zeros((1, 2)), np.zeros((1, 2))]
    model = _StubModel(logits, codes)
    ranks = tt.target_ranks(model, torch.zeros(1, 4), torch.tensor([1]), chunk=2)
    assert int(ranks[0]) == 1


# ---------------------------------------------------------------------------
# Head and losses
# ---------------------------------------------------------------------------

def test_head_parameter_count_and_shapes():
    head = tt.TigerHead(hidden_size=8, sid_sizes=[5, 3], init_std=0.02)
    assert head.parameter_count() == 8 * 5 + 8 * 3
    out = head(torch.zeros(4, 8))
    assert [tuple(o.shape) for o in out] == [(4, 5), (4, 3)]
    assert all(lin.bias is None for lin in head.layers)


def test_level_cross_entropies_sum_to_the_row_loss():
    torch.manual_seed(0)
    logits = [torch.randn(3, 4), torch.randn(3, 2)]
    labels = torch.tensor([[0, 1], [1, 0], [3, 1]])
    parts = tt.level_cross_entropies(logits, labels)
    assert len(parts) == 2
    expected = (torch.nn.functional.cross_entropy(logits[0], labels[:, 0])
                + torch.nn.functional.cross_entropy(logits[1], labels[:, 1]))
    assert float(parts[0] + parts[1]) == pytest.approx(float(expected), rel=1e-6)


def test_sid_matrices_offsets_the_input_but_not_the_labels():
    class _Catalog:
        sid_codes = np.array([[0, 1], [2, 3]], dtype=np.int32)
        num_levels = 2
        size = 2

    class _Vocab:
        sid_offsets = [50, 60]
        num_levels = 2

    tokens, codes = tt.sid_matrices(_Catalog(), _Vocab())
    assert tokens.tolist() == [[50, 61], [52, 63]]
    assert codes.tolist() == [[0, 1], [2, 3]]


def test_sid_matrices_rejects_a_level_mismatch():
    class _Catalog:
        sid_codes = np.zeros((2, 3), dtype=np.int32)
        num_levels = 3
        size = 2

    class _Vocab:
        sid_offsets = [50, 60]
        num_levels = 2

    with pytest.raises(ValueError, match="SID levels"):
        tt.sid_matrices(_Catalog(), _Vocab())


# ---------------------------------------------------------------------------
# End to end on a stub backbone (no weights, no GPU)
# ---------------------------------------------------------------------------

class _Backbone(torch.nn.Module):
    """A table lookup standing in for the LLM: same call signature, no attention."""

    def __init__(self, vocab: int = 64, hidden: int = 8):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.config = type("Cfg", (), {"hidden_size": hidden})()

    def forward(self, input_ids, attention_mask, use_cache=False, return_dict=True):
        out = type("Out", (), {})()
        out.last_hidden_state = self.embed(input_ids)
        return out


def _tiny_retriever(num_levels: int = 2, widths=(4, 4), catalog: int = 5):
    torch.manual_seed(0)
    codes = np.stack([np.arange(catalog) % w for w in widths], axis=1)
    return tt.TigerRetriever(_Backbone(), hidden_size=8, sid_sizes=list(widths),
                             catalog_codes=codes)


def test_retriever_separator_state_is_the_last_real_token():
    model = _tiny_retriever()
    ids = torch.tensor([[3, 4, 5, 9, 0], [3, 9, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]])
    states = model.separator_states(ids, mask)
    hidden = model.backbone.embed(ids)
    assert torch.allclose(states[0], hidden[0, 3])
    assert torch.allclose(states[1], hidden[1, 1])


def test_retriever_logits_at_reads_the_requested_positions():
    model = _tiny_retriever()
    ids = torch.tensor([[3, 4, 9, 1, 2]])
    mask = torch.ones_like(ids)
    positions = torch.tensor([[2, 3]])
    logits = model(ids, mask, positions)
    hidden = model.backbone.embed(ids)
    assert len(logits) == 2
    for level, part in enumerate(logits):
        expected = model.head.layers[level](hidden[:, positions[0, level]])
        assert torch.allclose(part, expected, atol=1e-6)


def test_state_for_prefixes_matches_the_training_logits_on_a_padded_row():
    """The beam must score from the same position training read.

    `_state_for_prefixes` appends the prefix to a row taken out of a padded batch. The
    separator is located as `attention_mask.sum(-1) - 1`, so leftover padding pushes
    that index back into the pad and the level reads a state that predicts nothing.
    A cluster run with exactly that fault decoded 0.29% of targets against a 43.30%
    ceiling while every other beam test passed, because the stub models there are not
    position sensitive.
    """
    sep = 1
    codes = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.int64)
    sid_input = codes + np.array([10, 20], dtype=np.int64)
    split = _Split([[5, 6], [7, 8, 9, 10]], [0, 3])
    ds = tt.TigerDataset(split, np.arange(2, dtype=np.int64), sid_input, codes, sep,
                         prefix_conditioning=False)
    collate = tt.TigerCollate(sep, 2, prefix_conditioning=False)
    ids, mask, _labels, _positions, _targets, rows = collate([ds[0], ds[1]])
    assert int(mask[0].sum()) < ids.shape[1], "row 0 must be padded or this is vacuous"

    model = _tiny_retriever(num_levels=2, widths=(8, 8), catalog=5)
    vocab = type("V", (), dict(sid_offsets=[10, 20], sid_sizes=[8, 8], num_levels=2,
                               sep_id=sep, total_size=32))()
    meta = type("M", (), dict(vocab=vocab))()

    with torch.inference_mode():
        for row in range(2):
            target = codes[int(rows[row])]
            id_row, mask_row = ids[row:row + 1], mask[row:row + 1]
            n_real = int(mask_row.sum())
            # Training read its logits from `history + [SEP] + c1 + c2` at [sep, sep + 1].
            train_ids = torch.cat(
                [id_row[:, :n_real],
                 torch.as_tensor(sid_input[int(rows[row])]).reshape(1, -1)], dim=1)
            expected = model(train_ids, torch.ones_like(train_ids),
                             torch.tensor([[n_real - 1, n_real]]))
            for level in range(2):
                state = tt._state_for_prefixes(
                    model, id_row, mask_row,
                    [tuple(int(c) for c in target[:level])], meta, torch.device("cpu"))
                got = model.logits_from_state(state)[level]
                assert torch.allclose(got, expected[level], atol=1e-6), (
                    f"row {row} level {level} scored from the wrong position")

    # Multi-row input is the mistake the single-row contract exists to prevent.
    with pytest.raises(ValueError):
        tt._state_for_prefixes(model, ids, mask, [()], meta, torch.device("cpu"))


def test_evaluate_tiger_runs_and_reports_every_cut_off():
    model = _tiny_retriever()
    split = _Split([[5, 6, 7], [8, 9, 10, 11, 12]], [0, 4])
    sid_input = np.array([[20, 30]] * 5, dtype=np.int64)
    sid_codes = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [0, 1]], dtype=np.int64)
    ds = tt.TigerDataset(split, np.arange(2, dtype=np.int64), sid_input, sid_codes,
                         sep_id=1, prefix_conditioning=False)
    collate = tt.TigerCollate(sep_id=1, num_levels=2, prefix_conditioning=False)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate)
    metrics = tt.evaluate_tiger(model, loader, torch.device("cpu"), torch.float32,
                               False, catalog_chunk=2)

    for k in tt.EVAL_KS:
        assert f"recall@{k}" in metrics
        assert 0.0 <= metrics[f"recall@{k}"] <= 1.0
    assert metrics["n"] == 2
    assert metrics["recall@10000"] >= metrics["recall@10"] >= metrics["recall@1"]
    assert metrics["loss"] > 0
    assert "loss@1" in metrics and "loss@2" in metrics
    assert metrics["loss"] == pytest.approx(metrics["loss@1"] + metrics["loss@2"], rel=1e-5)


def test_beam_search_over_the_real_tree_finds_an_owned_tuple():
    model = _tiny_retriever(catalog=6)
    codes = model.catalog_codes.numpy().astype(np.int32)
    index = tt.SidIndex(codes, [4, 4])
    ids = torch.tensor([[3, 4, 9]])
    mask = torch.ones_like(ids)

    def step_fn(prefixes):
        states = tt._state_for_prefixes(model, ids, mask, prefixes,
                                        type("M", (), {"vocab": type("V", (), {
                                            "sid_offsets": [10, 20]})()})(), 
                                        torch.device("cpu"))
        logits = model.logits_from_state(states)[len(prefixes[0])]
        return torch.log_softmax(logits, dim=-1).detach().numpy()

    beam = tt.beam_search(step_fn, [4, 4], beam_width=4,
                          allowed=lambda prefix, level: index.children(prefix))
    assert beam, "the beam must not be empty"
    for _, prefix in beam:
        assert index.contains(prefix)


def test_evaluate_tiger_ranks_the_catalogue_row_not_the_first_code():
    """The first code is not a row index, and the evaluator must not treat it as one.

    Every item here has first code 0, so if the evaluator ranked `code0` it would
    rank item 0 for every row. Item 0 is the best-scoring item under these fixed
    logits, so that mistake reports recall@1 of 1.0 instead of 0.0.
    """
    codes = np.array([[0, 0], [0, 1], [0, 2], [3, 3]], dtype=np.int64)
    logits = [np.array([[5.0, 0.0, 0.0, 0.0]]), np.array([[5.0, 0.0, 0.0, 0.0]])]
    metrics, loader = _stub_eval(logits, codes, [1, 2])

    _ids, _mask, labels, _p, targets, _r = next(iter(loader))
    assert targets.tolist() == [1, 2], "the loader must carry catalogue rows"
    assert labels[:, 0].tolist() == [0, 0], "the code is 0 for both, which is the trap"

    # Rows 1 and 2 rank 2nd and 3rd: the scores are log p1[0] + log p2[c1], and c1 is
    # 1 and 2 against row 0's 0.
    assert metrics["recall@1"] == pytest.approx(0.0)
    assert metrics["recall@10"] == pytest.approx(1.0)


def test_run_eval_refuses_a_prefix_conditioned_checkpoint(monkeypatch, tmp_path):
    args = tt.parse_args(["--mode", "eval", "--prefix_conditioning",
                          "--data_dir", str(tmp_path)])
    monkeypatch.setattr(tt, "_load_for_eval", lambda a, l: (None, None, None, None, None))
    with pytest.raises(SystemExit, match="beam"):
        tt.run_eval(args, logging.getLogger("test"))


def test_level_zero_coverage_is_reported_and_bounded_by_one():
    # Catalogue codes 0,1,2,3 at level 0 with four classes: a beam of 4 covers
    # everything, a beam of 1 covers only the model's single best code.
    model = _tiny_retriever(widths=(4, 4), catalog=4)
    split = _Split([[5, 6, 7]], [0])
    sid_input = np.array([[20, 30]] * 4, dtype=np.int64)
    sid_codes = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int64)
    ds = tt.TigerDataset(split, np.arange(1, dtype=np.int64), sid_input, sid_codes,
                         sep_id=1, prefix_conditioning=False)
    collate = tt.TigerCollate(sep_id=1, num_levels=2, prefix_conditioning=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
    metrics = tt.evaluate_tiger(model, loader, torch.device("cpu"), torch.float32,
                               False, catalog_chunk=2)

    ids, mask, labels, _p, _t, _r = next(iter(loader))
    with torch.no_grad():
        logits = model(ids, mask, _p)
    best = int(torch.argmax(logits[0][0]))
    for k in tt.BEAM_COVER_KS:
        assert 0.0 <= metrics[f"cover@{k}"] <= 1.0
    assert metrics["cover@1000"] == 1.0, "4 classes cannot hide from a beam of 1000"
    assert metrics["cover@1"] == float(best == int(labels[0, 0]))
    assert metrics["cover@1"] <= metrics["cover@1000"]


def test_head_accepts_bf16_states_against_fp32_weights():
    """Evaluation runs outside autocast, where nothing casts the state for us.

    The backbone is loaded in bf16 and the head keeps fp32 weights, so the val loop
    handed `F.linear` a bf16 mat1 against an fp32 mat2, which raises
    "expected mat1 and mat2 to have the same dtype". Training never hit it because
    autocast was still active there.
    """
    head = tt.TigerHead(hidden_size=8, sid_sizes=[5, 3])
    out = head(torch.zeros(2, 8, dtype=torch.bfloat16))
    assert [tuple(o.shape) for o in out] == [(2, 5), (2, 3)]
    assert all(o.dtype == torch.float32 for o in out), "the weights decide the dtype"


def test_target_ranks_accepts_bf16_states():
    model = _tiny_retriever()
    states = torch.zeros(2, 8, dtype=torch.bfloat16)
    ranks = tt.target_ranks(model, states, torch.tensor([0, 1]), chunk=2)
    assert ranks.tolist() == [ranks[0].item(), ranks[1].item()]
    assert 1 <= int(ranks.min()) <= 5


# ---------------------------------------------------------------------------
# Reporting: nothing the evaluator computes may be dropped
# ---------------------------------------------------------------------------

def _eval_fixture():
    model = _tiny_retriever()
    split = _Split([[5, 6, 7], [8, 9, 10, 11, 12]], [0, 4])
    sid_input = np.array([[20, 30]] * 5, dtype=np.int64)
    sid_codes = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [0, 1]], dtype=np.int64)
    ds = tt.TigerDataset(split, np.arange(2, dtype=np.int64), sid_input, sid_codes,
                         sep_id=1, prefix_conditioning=False)
    collate = tt.TigerCollate(sep_id=1, num_levels=2, prefix_conditioning=False)
    return model, DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate)


def test_evaluate_tiger_can_skip_ranking_without_touching_loss_or_coverage():
    model, loader = _eval_fixture()
    ranked = tt.evaluate_tiger(model, loader, torch.device("cpu"), torch.float32,
                               False, catalog_chunk=2)
    skipped = tt.evaluate_tiger(model, loader, torch.device("cpu"), torch.float32,
                                False, catalog_chunk=2, rank_items=False)
    assert ranked["ranked"] == 1.0 and skipped["ranked"] == 0.0
    assert "recall@1" not in skipped and "ndcg@10" not in skipped
    assert skipped["n"] == ranked["n"]
    assert skipped["loss"] == pytest.approx(ranked["loss"], rel=1e-6)
    for k in tt.BEAM_COVER_KS:
        assert skipped["cover@%d" % k] == pytest.approx(ranked["cover@%d" % k])


def test_format_metrics_prints_r5_and_every_cover():
    metrics = {"loss": 1.0, "loss@1": 0.5, "loss@2": 0.5, "ranked": 1.0,
               "recall@1": 0.1, "recall@5": 0.2, "recall@10": 0.3,
               "ndcg@10": 0.15, "mrr": 0.12}
    for k in tt.BEAM_COVER_KS:
        metrics["cover@%d" % k] = 0.5
    line = tt.format_metrics(metrics, 2)
    assert "R@5 0.2000" in line
    assert "cover 1:50.00% 10:50.00% 20:50.00% 100:50.00% 1000:50.00%" in line
    assert "nDCG@10 0.1500" in line


def test_format_metrics_says_ranks_were_skipped():
    metrics = {"loss": 1.0, "loss@1": 0.5, "loss@2": 0.5, "ranked": 0.0,
               "cover@10": 0.4}
    line = tt.format_metrics(metrics, 2)
    assert "ranks skipped" in line
    assert "R@1" not in line
    assert "cover 10:40.00%" in line


def test_write_metrics_keeps_every_key_and_replaces_a_rerun(tmp_path):
    tt._write_metrics(str(tmp_path), 1, {"loss": 3.0, "cover@10": 0.25, "ranked": 0.0})
    tt._write_metrics(str(tmp_path), 2, {"loss": 2.0, "cover@10": 0.35, "ranked": 0.0})
    tt._write_metrics(str(tmp_path), 2, {"loss": 2.5, "cover@10": 0.45, "ranked": 0.0})
    rows = json.loads((tmp_path / "metrics.json").read_text())
    assert [r["epoch"] for r in rows] == [1, 2], "a rerun of an epoch replaces it"
    assert rows[1]["loss"] == 2.5
    assert rows[1]["cover@10"] == 0.45, "cover must survive to disk"
    assert len((tmp_path / "metrics.jsonl").read_text().strip().splitlines()) == 3


def test_head_config_file_is_written_and_read_back(tmp_path):
    assert tt._read_head_config(str(tmp_path)) == {}
    payload = {"num_levels": 2, "sid_sizes": [4, 4], "prefix_conditioning": True}
    (tmp_path / tt.HEAD_CONFIG_FILE).write_text(json.dumps(payload))
    assert tt._read_head_config(str(tmp_path))["prefix_conditioning"] is True


def test_run_eval_refuses_when_the_checkpoint_says_prefix_conditioned(monkeypatch,
                                                                     tmp_path):
    """The flag can be forgotten; the checkpoint cannot lie about itself."""
    (tmp_path / tt.HEAD_CONFIG_FILE).write_text(
        json.dumps({"num_levels": 2, "prefix_conditioning": True}))
    args = tt.parse_args(["--mode", "eval", "--no-prefix_conditioning",
                          "--data_dir", str(tmp_path),
                          "--checkpoint", str(tmp_path)])
    monkeypatch.setattr(tt, "_load_for_eval", lambda a, l: (None, None, None, None, None))
    with pytest.raises(SystemExit, match="beam"):
        tt.run_eval(args, logging.getLogger("test"))


def test_checkpoint_config_decides_the_arm_not_the_flag(tmp_path):
    """A forgotten flag must not turn a prefix-conditioned model into a free one."""
    def args_for(flag, ckpt=None):
        return type("A", (), {"prefix_conditioning": flag, "checkpoint": ckpt})()

    empty = tmp_path / "empty"
    empty.mkdir()
    assert tt._checkpoint_prefix_conditioning(args_for(False, str(empty))) is False
    assert tt._checkpoint_prefix_conditioning(args_for(True, str(empty))) is True

    for recorded in (True, False):
        d = tmp_path / ("ckpt_%s" % recorded)
        d.mkdir()
        (d / tt.HEAD_CONFIG_FILE).write_text(
            json.dumps({"num_levels": 2, "prefix_conditioning": recorded}))
        assert tt._checkpoint_prefix_conditioning(args_for(not recorded, str(d))) is recorded


# ---------------------------------------------------------------------------
# Beam evaluation end to end
# ---------------------------------------------------------------------------

class _GradModel(torch.nn.Module):
    """A loaded retriever, as far as the beam can tell: its logits still need grad.

    A stub built out of buffers would hide the failure this covers, because
    `Tensor.numpy()` only refuses a tensor that requires grad.
    """

    def __init__(self, weights):
        super().__init__()
        self.w = torch.nn.Parameter(torch.as_tensor(weights, dtype=torch.float32))
        self.num_levels = int(len(weights))
        self.grad_enabled: list = []
        self.rows: list = []

    def logits_from_state(self, states):
        self.grad_enabled.append(torch.is_grad_enabled())
        return [self.w[level].expand(states.shape[0], -1)
                for level in range(self.num_levels)]

    def separator_states(self, input_ids, attention_mask):
        self.rows.append(input_ids)
        return torch.zeros(input_ids.shape[0], 4)


def _beam_meta():
    vocab = type("V", (), dict(sid_sizes=[3, 3], num_levels=2, sid_offsets=[0, 3],
                               sep_id=99, total_size=100))()
    catalog = type("C", (), dict(size=6))()
    return type("M", (), dict(vocab=vocab, catalog=catalog))()


def test_run_beam_decodes_under_inference_mode_and_reports_ndcg(monkeypatch, caplog):
    """The beam pass must not build a graph, and must report nDCG beside recall.

    Regression: `step_fn` converts the head output with `.numpy()`, which raises
    `Can't call numpy() on Tensor that requires grad` on a loaded model, because
    nothing had turned autograd off. `--mode beam` had no test, so the first real
    run was the first time that line executed.
    """
    codes = np.array([[0, 0], [1, 1], [2, 2], [0, 1], [1, 0], [2, 0]], dtype=np.int64)
    sid_in = (codes + 10).astype(np.int64)
    split = _Split([[5, 6], [7, 8]], [3, 3])
    # Level 0 prefers code 0 and level 1 prefers code 1, so item 3 (0, 1) wins.
    model = _GradModel([[0.0, -6.0, -6.0], [-6.0, 0.0, -6.0]])
    meta = _beam_meta()

    monkeypatch.setattr(tt, "_load_for_eval",
                        lambda a, l: (meta, model, sid_in, codes, torch.device("cpu")))
    monkeypatch.setattr(tt, "_load_splits", lambda a, m, l: (split, split))

    args = tt.parse_args(["--mode", "beam", "--beam_width", "2", "--beam_rows", "0",
                          "--prefix_conditioning"])
    with caplog.at_level(logging.INFO, logger="test-beam"):
        tt.run_beam(args, logging.getLogger("test-beam"))

    text = caplog.text
    assert "2/2 targets decoded" in text
    assert "R@1 100.00%" in text, "both rows put their own tuple first"
    assert "nDCG@1 1.00000" in text, "nDCG comes from the same rank array as R@k"
    assert "MRR@10 1.00000" in text
    assert "level-0 cover@2: 2/2 rows" in text, "cover caps what the beam can decode"
    assert model.grad_enabled and not any(model.grad_enabled), \
        "beam_search calls .numpy() on the head output, so autograd must be off"

    # The row the beam scores has to be `history + [SEP]` and nothing else. The first
    # run of this pass built the loader with prefix_conditioning=True, so the row
    # already ended with the target's own codes and level 0 read its state from after
    # the answer. That showed up as a 0.29% decode rate and cost 53 minutes of GPU.
    shortest = min(model.rows, key=lambda t: t.shape[1])
    assert int(shortest[0, -1]) == meta.vocab.sep_id, \
        "level 0 must be read at the separator, not after the target's codes"
    longest = max(model.rows, key=lambda t: t.shape[1])
    assert int(longest[0, -1]) != meta.vocab.sep_id, "the prefix token was appended"


def test_run_beam_reports_no_cut_off_wider_than_the_beam(monkeypatch, caplog):
    """A row that misses the beam is a miss, and R@100 is not invented at width 2."""
    codes = np.array([[0, 0], [1, 1]], dtype=np.int64)
    # Level 0 prefers code 1, and a beam of 1 follows it, so item 0 (0, 0) never
    # enters the beam at all.
    model = _GradModel([[0.0, 6.0, -6.0], [0.0, 0.0, 0.0]])
    split = _Split([[5, 6]], [0])

    monkeypatch.setattr(tt, "_load_for_eval",
                        lambda a, l: (_beam_meta(), model, (codes + 10).astype(np.int64),
                                      codes, torch.device("cpu")))
    monkeypatch.setattr(tt, "_load_splits", lambda a, m, l: (split, split))

    args = tt.parse_args(["--mode", "beam", "--beam_width", "1", "--beam_rows", "0",
                          "--prefix_conditioning"])
    with caplog.at_level(logging.INFO, logger="test-beam"):
        tt.run_beam(args, logging.getLogger("test-beam"))

    text = caplog.text
    assert "0/1 targets decoded" in text
    assert "level-0 cover@1: 0/1 rows" in text, "the target code is not in the top 1"
    assert "R@1 0.00%" in text, "a target outside the beam counts as a miss here"
    assert "R@100" not in text, "the beam cannot report a cut-off wider than itself"
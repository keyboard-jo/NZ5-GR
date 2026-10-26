#!/usr/bin/env python3
"""Tests for `embedding_knn_nextclick.py`, mock data only, no GPU, no parquet.

The month-long risk in this script is the row layout: it reads
`<split>/tokens.npy` directly instead of going through `TokenDataset`, so a
mismatch with `preprocess_digr.build_tokenized_split` would produce confident
nonsense rather than a crash. `_make_rows` therefore rebuilds that layout the
same way the preprocessor does, and the decoder is tested against it.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import embedding_knn_nextclick as E


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
SID_SIZES = [4, 3, 2, 2, 2]          # 96 possible 5-code tuples
SID_OFFSETS = [2, 6, 9, 11, 13]      # cumulative, first offset 2 so SEP=1 is free
OFFSET_TEMPORAL = 15
OFFSET_FEATURES = 17
SEP_ID = 1
NUM_LEVELS = len(SID_SIZES)


def _all_tuples():
    """Every 5-code tuple in mixed-radix order, i.e. a catalog of 96 items."""
    out = []
    for a in range(SID_SIZES[0]):
        for b in range(SID_SIZES[1]):
            for c in range(SID_SIZES[2]):
                for d in range(SID_SIZES[3]):
                    for e in range(SID_SIZES[4]):
                        out.append((a, b, c, d, e))
    return np.asarray(out, dtype=np.int32)


def _make_rows(histories, feat_lens=None):
    """Build tokens/offsets the way `build_tokenized_split` does.

    `histories` is a list of lists of catalog rows, oldest first. The row is
    `[feature block][SEP][(temporal, 5 codes) x m]`.
    """
    if feat_lens is None:
        feat_lens = [2] * len(histories)
    catalog = _all_tuples()
    tokens = []
    offsets = [0]
    for hist, feat_len in zip(histories, feat_lens):
        row = [OFFSET_FEATURES + j for j in range(int(feat_len))]
        row.append(SEP_ID)
        for position, item in enumerate(hist):
            row.append(OFFSET_TEMPORAL + (position % 4))
            row.extend(int(c) + SID_OFFSETS[l]
                       for l, c in enumerate(catalog[item]))
        tokens.extend(row)
        offsets.append(len(tokens))
    return (np.asarray(tokens, dtype=np.int32),
            np.asarray(offsets, dtype=np.int64))


def _vocab():
    return {
        "sid_sizes": SID_SIZES,
        "sid_offsets": SID_OFFSETS,
        "num_levels": NUM_LEVELS,
        "sep_id": SEP_ID,
        "pad_id": 0,
        "offset_temporal": OFFSET_TEMPORAL,
        "offset_features": OFFSET_FEATURES,
        "total_size": 64,
    }


# --------------------------------------------------------------------------- #
# SID keys
# --------------------------------------------------------------------------- #
def test_pack_sid_keys_is_injective_and_matches_a_dict():
    codes = _all_tuples()
    keys = E.pack_sid_keys(codes, SID_SIZES)
    assert len(np.unique(keys)) == len(codes)

    reference = {tuple(int(x) for x in row): i for i, row in enumerate(codes)}
    for i in (0, 7, 55, len(codes) - 1):
        assert reference[tuple(int(x) for x in codes[i])] == i
        # round-trip through the packed key
        assert keys[i] == E.pack_sid_keys(codes[i][None, :], SID_SIZES)[0]


def test_pack_sid_keys_rejects_out_of_range_codes():
    with pytest.raises(ValueError, match="codebook holds"):
        E.pack_sid_keys(np.asarray([[0, 0, 0, 0, 99]]), SID_SIZES)


def test_pack_sid_keys_depth_changes_the_key():
    codes = np.asarray([[1, 2, 1, 0, 0]])
    assert E.pack_sid_keys(codes, SID_SIZES, depth=1)[0] == 1
    assert E.pack_sid_keys(codes, SID_SIZES, depth=2)[0] == 1 * 3 + 2


def test_build_key_lookup_keeps_the_first_catalog_row_of_a_prefix():
    catalog = _all_tuples()
    keys, rows = E.build_key_lookup(catalog, SID_SIZES, depth=2)
    assert len(keys) == SID_SIZES[0] * SID_SIZES[1]
    # every depth-2 prefix owns SID_SIZES[2] * SID_SIZES[3] * SID_SIZES[4] items
    # and the winner is the lowest catalog row in that run
    assert rows[0] == 0
    assert rows[1] == 8


def test_lookup_catalog_rows_returns_minus_one_for_an_unknown_key():
    catalog = _all_tuples()
    keys, rows = E.build_key_lookup(catalog, SID_SIZES, NUM_LEVELS)
    got = E.lookup_catalog_rows(keys, rows, np.asarray([keys[5], 10 ** 9]))
    assert got[0] == 5
    assert got[1] == -1


def test_bucket_slice_covers_a_prefix():
    catalog = _all_tuples()
    keys, rows = E.build_prefix_index(catalog, SID_SIZES, depth=2)
    lo, hi = E.bucket_slice(keys, int(keys[3]))
    assert hi - lo == 8
    assert np.all(E.pack_sid_keys(catalog[rows[lo:hi]], SID_SIZES, depth=2)
                  == keys[3])


def test_build_prefix_index_keeps_every_row_of_a_bucket():
    catalog = _all_tuples()
    keys, rows = E.build_prefix_index(catalog, SID_SIZES, depth=2)
    assert len(keys) == len(catalog)
    assert np.all(np.diff(keys) >= 0)
    # every catalog row appears exactly once
    assert sorted(rows.tolist()) == list(range(len(catalog)))


# --------------------------------------------------------------------------- #
# row decoding
# --------------------------------------------------------------------------- #
def test_decode_row_windows_reads_the_last_events_oldest_to_newest():
    histories = [[0, 10, 21], [3], [7, 9]]
    tokens, offsets = _make_rows(histories)
    codes, valid = E.decode_row_windows(
        tokens, offsets, 0, len(histories), SID_OFFSETS, NUM_LEVELS, SEP_ID,
        width=3)

    assert valid.shape == (3, 3)
    # j = 0 is the most recent event
    assert list(codes[0, 0]) == list(_all_tuples()[21])
    assert list(codes[0, 1]) == list(_all_tuples()[10])
    assert list(codes[0, 2]) == list(_all_tuples()[0])
    assert valid[0].all()

    # a one-event row can only fill j = 0
    assert list(codes[1, 0]) == list(_all_tuples()[3])
    assert valid[1].tolist() == [True, False, False]


def test_decode_row_windows_handles_variable_feature_lengths():
    histories = [[5, 6]] * 2
    tokens, offsets = _make_rows(histories, feat_lens=[0, 5])
    codes, valid = E.decode_row_windows(
        tokens, offsets, 0, 2, SID_OFFSETS, NUM_LEVELS, SEP_ID, width=2)
    assert valid.all()
    assert list(codes[0, 0]) == list(_all_tuples()[6])
    assert list(codes[1, 0]) == list(_all_tuples()[6])


def test_decode_row_windows_rejects_a_level_count_mismatch():
    histories = [[0]]
    tokens, offsets = _make_rows(histories)
    # claim 4 levels: the row no longer divides into whole events
    with pytest.raises(ValueError, match="1 \\+ feat_len"):
        E.decode_row_windows(tokens, offsets, 0, 1, SID_OFFSETS, 4, SEP_ID,
                             width=1)


def test_decode_row_windows_rejects_a_missing_sep():
    histories = [[0]]
    tokens, offsets = _make_rows(histories)
    tokens = tokens.copy()
    tokens[2] = 0                       # the SEP was at offset[0] + feat_len = 2
    with pytest.raises(ValueError, match="SEP"):
        E.decode_row_windows(tokens, offsets, 0, 1, SID_OFFSETS, NUM_LEVELS,
                             SEP_ID, width=1)


# --------------------------------------------------------------------------- #
# retrieval helpers
# --------------------------------------------------------------------------- #
def test_topk_over_matches_a_brute_force_scan():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    cand = rng.normal(size=(500, 8)).astype(np.float32)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    q = rng.normal(size=(3, 8)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)

    vals, idx = E.topk_over(torch.from_numpy(q), torch.from_numpy(cand), topk=5,
                            chunk=64)
    expect = np.argsort(-(q @ cand.T), axis=1)[:, :5]
    assert np.array_equal(idx.numpy(), expect)
    assert np.allclose(vals.numpy(), np.take_along_axis(q @ cand.T, expect, 1))


def test_topk_over_clamps_to_the_candidate_count():
    torch = pytest.importorskip("torch")
    cand = torch.eye(4, dtype=torch.float32)
    q = torch.ones(1, 4, dtype=torch.float32)
    _, idx = E.topk_over(q, cand, topk=1000, chunk=3)
    assert idx.shape == (1, 4)
    assert sorted(idx[0].tolist()) == [0, 1, 2, 3]


def test_ranks_give_a_miss_topk_plus_one_never_zero():
    torch = pytest.importorskip("torch")
    idx = torch.as_tensor([[4, 7, 9], [1, 2, 3]], dtype=torch.int64)
    targets = torch.as_tensor([7, 5], dtype=torch.int64)
    ranks = E.ranks_from_topk(idx, targets, topk=3)
    assert ranks.tolist() == [2, 4]


def test_metrics_do_not_count_a_miss_as_a_hit():
    metrics = E.new_metrics((1, 2))
    E.update_metrics(metrics, np.asarray([1, 2, 3]), (1, 2))
    out = E.finalise_metrics(metrics, (1, 2))
    assert out["recall@1"] == pytest.approx(1 / 3)
    assert out["recall@2"] == pytest.approx(2 / 3)
    # rank 3 is still inside MRR_K, so MRR keeps it
    assert out["mrr"] == pytest.approx((1.0 + 0.5 + 1.0 / 3.0) / 3)
    assert np.isfinite(out["ndcg@2"])


def test_metrics_pin_mrr_to_ten_even_when_the_cut_offs_are_wider():
    # otherwise recall@1000 runs would silently report MRR@1000
    metrics = E.new_metrics((1, 1000))
    E.update_metrics(metrics, np.asarray([500, 11]), (1, 1000))
    out = E.finalise_metrics(metrics, (1, 1000))
    assert out["mrr"] == pytest.approx(0.0)          # both ranks are past 10
    assert out["recall@1000"] == pytest.approx(1.0)


def test_align_to_catalog_reorders_by_item_id():
    emb = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    emb_ids = np.asarray([30, 10, 20])
    catalog_ids = np.asarray([10, 20, 30])
    got = E.align_to_catalog(emb, emb_ids, catalog_ids)
    assert np.allclose(got[0], [0.0, 1.0])
    assert np.allclose(got[1], [0.5, 0.5])
    assert np.allclose(got[2], [1.0, 0.0])


def test_align_to_catalog_leaves_zero_rows_for_items_without_an_embedding():
    emb = np.asarray([[1.0, 0.0]], dtype=np.float32)
    got = E.align_to_catalog(emb, np.asarray([20]), np.asarray([10, 20, 30]))
    assert np.allclose(got[1], [1.0, 0.0])
    assert np.allclose(got[0], 0.0)
    assert np.allclose(got[2], 0.0)


def test_normalise_rows_makes_unit_norm():
    x = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    got = E.normalise_rows(x, block=1)
    assert np.allclose(np.linalg.norm(got, axis=1), 1.0)


def test_verdict_bands():
    assert "SID is destroying it" in E.verdict(0.06)
    assert "exonerated" in E.verdict(0.03)
    assert "sequence tower" in E.verdict(0.01)


# --------------------------------------------------------------------------- #
# end to end on a synthetic world where the answer is known
# --------------------------------------------------------------------------- #
def _write_synthetic(tmp_path, n_rows=12):
    catalog = _all_tuples()
    n_catalog, dim = len(catalog), 8

    # orthogonal one-hot embeddings, so the nearest neighbour is exact and the
    # expected recall is 1.0 by construction
    emb = np.eye(n_catalog, dim, dtype=np.float32)[:, :dim].copy()
    rng = np.random.default_rng(1)
    emb += 0.01 * rng.normal(size=emb.shape).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb.astype(np.float32)

    # one event per row, and that event IS the target: kNN w1 must hit at rank 1
    targets = np.asarray([(7 * i) % n_catalog for i in range(n_rows)], dtype=np.int64)
    histories = [[int(t)] for t in targets]
    tokens, offsets = _make_rows(histories, feat_lens=[1] * n_rows)

    os.makedirs(os.path.join(tmp_path, "val"))
    os.makedirs(os.path.join(tmp_path, "train"))
    with open(os.path.join(tmp_path, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(_vocab(), f)
    np.save(os.path.join(tmp_path, "catalog_sid_codes.npy"), catalog)
    np.save(os.path.join(tmp_path, "catalog_item_ids.npy"),
            np.arange(1, n_catalog + 1, dtype=np.int64))
    np.save(os.path.join(tmp_path, "val", "tokens.npy"), tokens)
    np.save(os.path.join(tmp_path, "val", "offsets.npy"), offsets)
    np.save(os.path.join(tmp_path, "val", "targets.npy"), targets)
    np.save(os.path.join(tmp_path, "train", "targets.npy"), targets)

    np.save(os.path.join(tmp_path, "item_embeddings.npy"), emb)
    np.save(os.path.join(tmp_path, "item_ids.npy"),
            np.arange(1, n_catalog + 1, dtype=np.int64))
    return tmp_path


def test_main_end_to_end_on_a_synthetic_catalog(tmp_path, capsys):
    pytest.importorskip("torch")
    data_dir = _write_synthetic(tmp_path)
    out_json = os.path.join(tmp_path, "out", "metrics.json")

    rc = E.main([
        "--data_dir", str(data_dir),
        "--embeddings", str(data_dir / "item_embeddings.npy"),
        "--embedding_ids", str(data_dir / "item_ids.npy"),
        "--split", "val",
        "--ks", "1,3",
        "--windows", "1",
        "--sid_depths", "1,2",
        "--topk", "8",
        "--row_chunk", "5",
        "--query_chunk", "4",
        "--catalog_chunk", "16",
        "--device", "cpu",
        "--out_json", out_json,
    ])
    assert rc == 0

    with open(out_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["rows"] == 12
    assert payload["knn"]["w1"]["recall@1"] == pytest.approx(1.0)
    assert payload["knn"]["w1"]["mrr"] == pytest.approx(1.0)
    # the row's only event is the target itself
    assert payload["repeat_rate"] == pytest.approx(1.0)
    assert payload["last_item_unmapped"] == pytest.approx(0.0)
    # the depth-1 bucket always contains the target, and cosine rerank finds it
    assert payload["sid_prefix"]["depth1"]["candidate_recall"] == pytest.approx(1.0)
    assert payload["sid_prefix"]["depth1"]["recall@1"] == pytest.approx(1.0)
    assert payload["verdict"]

    log = capsys.readouterr().out
    assert "VERDICT" in log
    assert "candidate recall" in log


def test_main_rejects_a_topk_narrower_than_the_cut_offs(tmp_path):
    data_dir = _write_synthetic(tmp_path)
    with pytest.raises(SystemExit):
        E.parse_cli(["--data_dir", str(data_dir),
                     "--embeddings", "x.npy", "--ks", "100", "--topk", "10"])


def test_main_rejects_an_embedding_matrix_of_the_wrong_height(tmp_path):
    pytest.importorskip("torch")
    data_dir = _write_synthetic(tmp_path)
    np.save(os.path.join(tmp_path, "short.npy"),
            np.eye(5, 8, dtype=np.float32))
    with pytest.raises(ValueError, match="rows but the catalog has"):
        E.main(["--data_dir", str(data_dir),
                "--embeddings", str(tmp_path / "short.npy"),
                "--split", "val", "--device", "cpu"])

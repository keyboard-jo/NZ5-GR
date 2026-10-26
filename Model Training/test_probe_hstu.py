"""Tests for probe_hstu.py (HSTU tower scored against the DiGR val split).

Mock data only: a tiny 8-item / 2-level synthetic catalog, a tiny ItemDNN +
HSTUSequenceEncoder, and a few val rows. No GPU, no real checkpoint.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_SID_DIR = os.path.join(_REPO_ROOT, "SID Generation")
for _d in (_SCRIPT_DIR, _SID_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import probe_hstu  # noqa: E402
import train_item_embedding_cli as stage1  # noqa: E402

SID_SIZES = (4, 4)
SID_OFFSETS = (2, 6)
NUM_LEVELS = 2
SEP_ID = 1
TEMPORAL = 10

# 8 items, ids 100..107, codes (level0, level1)
CAT_CODES = np.array([
    [0, 0], [0, 1], [0, 2], [0, 3],
    [1, 0], [1, 1], [1, 2], [1, 3],
], dtype=np.int64)
CAT_IDS = np.arange(100, 108, dtype=np.int64)


def _write_items_parquet(path, item_ids, mm_emb):
    import pandas as pd
    df = pd.DataFrame({
        "item_id": np.asarray(item_ids, dtype=np.int64),
        "mm_emb": [np.asarray(e, dtype=np.float32) for e in mm_emb],
        "has_mm_emb": [True] * len(item_ids),
    })
    df.to_parquet(path, index=False)


def _encode_row(history_codes):
    """history_codes: chronological list of (l0, l1) code pairs."""
    toks = [SEP_ID]
    for c0, c1 in history_codes:
        toks.append(TEMPORAL)
        toks.append(SID_OFFSETS[0] + c0)
        toks.append(SID_OFFSETS[1] + c1)
    return toks


def _write_digr_data(data_dir, rows):
    """rows: list of (history_codes_chronological, target_catalog_row)."""
    split_dir = os.path.join(data_dir, "val")
    os.makedirs(split_dir, exist_ok=True)
    vocab = {
        "sid_sizes": list(SID_SIZES),
        "sid_offsets": list(SID_OFFSETS),
        "num_levels": NUM_LEVELS,
        "sep_id": SEP_ID,
    }
    with open(os.path.join(data_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    np.save(os.path.join(data_dir, "catalog_sid_codes.npy"), CAT_CODES)
    np.save(os.path.join(data_dir, "catalog_item_ids.npy"), CAT_IDS)

    tokens, offsets = [], [0]
    targets = []
    for history, target in rows:
        row = _encode_row(history)
        tokens.extend(row)
        offsets.append(len(tokens))
        targets.append(target)
    np.save(os.path.join(split_dir, "tokens.npy"),
            np.asarray(tokens, dtype=np.int64))
    np.save(os.path.join(split_dir, "offsets.npy"),
            np.asarray(offsets, dtype=np.int64))
    np.save(os.path.join(split_dir, "targets.npy"),
            np.asarray(targets, dtype=np.int64))


def build_fixture(tmp_path):
    """Build the tiny checkpoint, parquet, exported embeddings and DiGR data."""
    torch.manual_seed(0)
    np.random.seed(0)

    mm_emb = np.random.randn(8, 8).astype(np.float32)
    items_path = os.path.join(tmp_path, "canonical_items.parquet")
    _write_items_parquet(items_path, CAT_IDS, mm_emb)

    import pandas as pd
    items_df = pd.read_parquet(items_path)
    store, item_to_idx, idx_to_item, cardinalities, cat_cols = \
        stage1.build_feature_store(
            items_df, item_id_col="item_id", embedding_col="mm_emb",
            has_emb_col="has_mm_emb", embedding_dim=8,
            cat_col_candidates=[], dtype=torch.float32)

    item_dnn = stage1.ItemDNN(
        sem_dim=8, cat_cardinalities=cardinalities, cat_cols=cat_cols,
        cat_emb_dim=2, output_dim=4, hidden_dims=(8, 4), dropout=0.0)
    seq_encoder = stage1.HSTUSequenceEncoder(
        hidden_dim=4, max_seq_len=8, num_blocks=1, num_heads=2, dropout=0.0)
    model = stage1.SequentialRecModel(
        feature_store=store, item_dnn=item_dnn, seq_encoder=seq_encoder,
        temperature=0.05)

    args = {
        "item_id_col": "item_id", "embedding_col": "mm_emb",
        "has_emb_col": "has_mm_emb", "raw_emb_dim": 8, "output_dim": 4,
        "cat_emb_dim": 2, "item_dnn_hidden_dims": "8,4", "num_blocks": 1,
        "num_heads": 2, "dropout": 0.0, "max_seq_len": 8,
        "temperature": 0.05, "learnable_temperature": False,
        "no_log_q": False, "log_q_weight": 1.0, "cross_gpu_negatives": False,
        "fp16": False, "cat_cols": None, "items_path": items_path,
    }
    ckpt_path = os.path.join(tmp_path, "stage1_rec_model.pt")
    torch.save({
        "item_dnn_state_dict": model.item_dnn.state_dict(),
        "seq_encoder_state_dict": model.seq_encoder.state_dict(),
        "action_emb_state_dict": model.action_emb.state_dict(),
        "input_layer_norm_state_dict": model.input_layer_norm.state_dict(),
        "cardinalities": cardinalities, "cat_cols": cat_cols, "args": args,
    }, ckpt_path)

    with torch.no_grad():
        embs = torch.cat([model.get_item_embeddings(torch.arange(1, 9))])
    emb_path = os.path.join(tmp_path, "item_embeddings.npy")
    ids_path = os.path.join(tmp_path, "item_ids.npy")
    np.save(emb_path, embs.numpy().astype(np.float32))
    np.save(ids_path, CAT_IDS)

    data_dir = os.path.join(tmp_path, "digr_data")
    _write_digr_data(data_dir, [
        ([(0, 0)], 1),           # item0 -> target item1 (row 1)
        ([(1, 0)], 5),           # item4 -> target item5 (row 5)
        ([(0, 2), (1, 2)], 7),   # item2,item6 -> target item7 (row 7)
    ])
    return ckpt_path, items_path, emb_path, ids_path, data_dir


# --------------------------------------------------------------------------- #
# unit tests
# --------------------------------------------------------------------------- #
def test_parse_cli_defaults():
    a = probe_hstu.parse_cli(["--checkpoint", "x.pt", "--data_dir", "d"])
    assert a.ks == (1, 10, 100, 1000)
    assert a.actions == "approx"
    assert a.history_window == 20
    assert a.topk == 1000


def test_parse_cli_exact_requires_seq_path():
    with pytest.raises(SystemExit):
        probe_hstu.parse_cli(["--checkpoint", "x.pt", "--data_dir", "d",
                              "--actions", "exact"])


def test_verdict_bands():
    assert "SID is destroying" in probe_hstu.verdict(0.06)
    assert "worth doing" in probe_hstu.verdict(0.03)
    assert "needs retraining" in probe_hstu.verdict(0.01)


def test_to_chronological_left_aligns_and_keeps_newest_at_m_minus_1():
    codes = np.full((2, 5, 2), -1, dtype=np.int64)
    valid = np.zeros((2, 5), dtype=bool)
    codes[0, 0] = [10, 11]
    codes[0, 1] = [20, 21]
    codes[0, 2] = [30, 31]
    valid[0, :3] = True
    codes[1, 0] = [40, 41]
    valid[1, :1] = True

    chrono, keep = probe_hstu.to_chronological(codes, valid)
    # valid must be a left-aligned prefix
    assert keep[0].tolist() == [True, True, True, False, False]
    assert keep[1].tolist() == [True, False, False, False, False]
    # oldest first, newest at index m-1 (what HSTUSequenceEncoder reads)
    assert chrono[0, :3, 0].tolist() == [30, 20, 10]
    assert chrono[1, 0, 0] == 40


def test_padded_row_matches_the_unpadded_encoding(tmp_path):
    """A row with fewer events than the window must encode identically to the
    same row at its true length. This is the regression for the right-align bug:
    right-aligned input puts padding at the front, so the encoder's
    `x[:, length - 1]` read lands on a pad slot."""
    ckpt_path, items_path, emb_path, ids_path, data_dir = build_fixture(tmp_path)
    ckpt = probe_hstu.load_checkpoint(ckpt_path, torch.device("cpu"))
    ckpt_args = dict(ckpt["args"])
    store, _, _, _ = probe_hstu.rebuild_feature_store(
        ckpt_args, items_path, torch.device("cpu"))
    model = probe_hstu.rebuild_model(
        ckpt_args, dict(ckpt["cardinalities"]), list(ckpt["cat_cols"]), store)
    probe_hstu.load_state_dicts(model, ckpt)
    model.eval()

    # one event (store index 3) at its true length, then padded into width 5
    hi_short = torch.tensor([[3]], dtype=torch.long)
    ha_short = torch.tensor([[2]], dtype=torch.long)
    m_short = torch.tensor([[True]])
    hi_pad = torch.tensor([[3, 0, 0, 0, 0]], dtype=torch.long)
    ha_pad = torch.tensor([[2, 0, 0, 0, 0]], dtype=torch.long)
    m_pad = torch.tensor([[True, False, False, False, False]])

    with torch.no_grad():
        a = model.encode_user(hi_short, ha_short, m_short)
        b = model.encode_user(hi_pad, ha_pad, m_pad)
    assert torch.allclose(a, b, atol=1e-5), (a, b)


def test_rebuild_model_roundtrip(tmp_path):
    ckpt_path, items_path, emb_path, ids_path, data_dir = build_fixture(tmp_path)
    ckpt = probe_hstu.load_checkpoint(ckpt_path, torch.device("cpu"))
    ckpt_args = dict(ckpt["args"])
    store, item_to_idx, idx_to_item, derived = probe_hstu.rebuild_feature_store(
        ckpt_args, items_path, torch.device("cpu"))
    model = probe_hstu.rebuild_model(
        ckpt_args, dict(ckpt["cardinalities"]), list(ckpt["cat_cols"]), store)
    probe_hstu.load_state_dicts(model, ckpt)

    # a forward on the rebuilt model must match the saved state dict's shape
    hi = torch.tensor([[1, 0]], dtype=torch.long)
    ha = torch.tensor([[2, 0]], dtype=torch.long)
    m = torch.tensor([[True, False]])
    with torch.no_grad():
        user = model.encode_user(hi, ha, m)
    assert user.shape == (1, 4)
    assert torch.isfinite(user).all()


# --------------------------------------------------------------------------- #
# end-to-end smoke test
# --------------------------------------------------------------------------- #
def test_probe_end_to_end_smoke(tmp_path):
    ckpt_path, items_path, emb_path, ids_path, data_dir = build_fixture(tmp_path)
    out_json = os.path.join(tmp_path, "out.json")
    rc = probe_hstu.main([
        "--checkpoint", ckpt_path,
        "--items_path", items_path,
        "--catalog_embeddings", emb_path,
        "--catalog_ids", ids_path,
        "--data_dir", data_dir,
        "--split", "val",
        "--history_window", "2",
        "--actions", "approx",
        "--device", "cpu",
        "--dtype", "float32",
        "--batch_size", "2",
        "--out_json", out_json,
    ])
    assert rc == 0
    with open(out_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["n_rows"] == 3
    for k in ("recall@1", "recall@10", "recall@100", "recall@1000", "mrr"):
        assert k in payload["metrics"]
        assert 0.0 <= payload["metrics"][k] <= 1.0
    # no history event should be unmapped (all catalog ids are in the store)
    assert payload["diagnostics"]["unmapped_history_event_rate"] == 0.0
    assert "verdict" in payload

"""Tests for the guard that stops a mismatch between a checkpoint and its data.

The case these exist for is real. On 2026-10-28 the SID v2 checkpoint was scored
against the frozen v1 `digr_data` and reported R@10 0.0437% with loss 26.3504,
against 3.6600% and 11.0133 from the same checkpoint on its own directory. Every
id was in range, because a v1 token stream read as v2 is still a plausible set of
codes, so the run completed and the number looked like a weak model. The guard
compares the two layouts and refuses.

Mock data only: no torch, no checkpoint, no GPU. `eval_full_catalog` is imported
through its module, so `train_digr` has to import, which it does on CPU.
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import eval_full_catalog as E  # noqa: E402
import train_digr as T  # noqa: E402

# A two-level SID over a 16,662-token vocabulary, which is the shape the v2 run used.
V2_VOCAB = {
    "sid_sizes": [8192, 8192],
    "n_temporal": 128,
    "n_action": 3,
    "n_clock": 0,
    "feature_sizes": [87, 3],
    "n_special": 2,
    "pad_id": 0,
    "sep_id": 1,
}

# The frozen five-level layout: 4 x 256 plus a 731 collision level, 64 gap bins,
# no action token. Six tokens per event against the v2 build's four.
V1_VOCAB = {
    "sid_sizes": [256, 256, 256, 256, 731],
    "n_temporal": 64,
    "feature_sizes": [87, 3],
    "n_special": 2,
    "pad_id": 0,
    "sep_id": 1,
}

ITEM_IDS = np.array([1, 2, 3, 4], dtype=np.int64)
SIDS = np.array([[0, 5], [1, 5], [2, 6], [3, 6]], dtype=np.int32)


def _write_dir(path, vocab_dict, item_ids=ITEM_IDS, sids=SIDS):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f)
    with open(os.path.join(path, "catalog_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"is_int": True}, f)
    np.save(os.path.join(path, "catalog_item_ids.npy"), np.asarray(item_ids))
    np.save(os.path.join(path, "catalog_sid_codes.npy"), np.asarray(sids))
    return path


def _meta(vocab_dict=V2_VOCAB, item_ids=ITEM_IDS, sids=SIDS):
    return SimpleNamespace(
        catalog=T.ItemCatalog(np.asarray(item_ids), True, np.asarray(sids), prepared=True),
        vocab=T.Vocab.from_dict(vocab_dict),
    )


def test_matching_dir_is_accepted(tmp_path):
    data_dir = _write_dir(tmp_path / "digr_data", V2_VOCAB)
    E.check_data_dir_matches_checkpoint(_meta(), str(data_dir))


def test_v1_directory_is_rejected_for_a_v2_checkpoint(tmp_path):
    """The incident. Same catalogue, same val rows, wrong token layout."""
    data_dir = _write_dir(tmp_path / "digr_data", V1_VOCAB)
    with pytest.raises(SystemExit) as exc:
        E.check_data_dir_matches_checkpoint(_meta(), str(data_dir))
    message = str(exc.value)
    # Both layouts named, so the log says which side is which.
    assert "different token layouts" in message
    assert "[8192, 8192]" in message
    assert "[256, 256, 256, 256, 731]" in message
    assert "SID sizes" in message
    assert "tokens per event" in message


def test_gap_bin_change_alone_is_rejected(tmp_path):
    """64 versus 128 gap bins is enough on its own: the temporal offset moves."""
    other = dict(V2_VOCAB, n_temporal=64)
    data_dir = _write_dir(tmp_path / "digr_data", other)
    with pytest.raises(SystemExit) as exc:
        E.check_data_dir_matches_checkpoint(_meta(), str(data_dir))
    assert "gap bins" in str(exc.value)


def test_action_token_change_alone_is_rejected(tmp_path):
    """Dropping the action token shortens the event slot from four to three."""
    other = dict(V2_VOCAB, n_action=0)
    data_dir = _write_dir(tmp_path / "digr_data", other)
    with pytest.raises(SystemExit) as exc:
        E.check_data_dir_matches_checkpoint(_meta(), str(data_dir))
    assert "tokens per event" in str(exc.value)


def test_catalog_order_mismatch_is_rejected(tmp_path):
    shuffled = ITEM_IDS[::-1].copy()
    data_dir = _write_dir(tmp_path / "digr_data", V2_VOCAB, item_ids=shuffled)
    with pytest.raises(SystemExit) as exc:
        E.check_data_dir_matches_checkpoint(_meta(), str(data_dir))
    assert "catalog order" in str(exc.value)


def test_catalog_size_mismatch_is_rejected(tmp_path):
    shorter = ITEM_IDS[:3]
    short_sids = SIDS[:3]
    data_dir = _write_dir(tmp_path / "digr_data", V2_VOCAB,
                          item_ids=shorter, sids=short_sids)
    with pytest.raises(SystemExit) as exc:
        E.check_data_dir_matches_checkpoint(_meta(), str(data_dir))
    assert "catalog items" in str(exc.value)


def test_sid_code_change_is_rejected(tmp_path):
    """Identical order and size, one code different. Scores would come from the
    wrong SID while the labels were counted on the right one."""
    other_sids = SIDS.copy()
    other_sids[2, 0] = 99
    data_dir = _write_dir(tmp_path / "digr_data", V2_VOCAB, sids=other_sids)
    with pytest.raises(SystemExit) as exc:
        E.check_data_dir_matches_checkpoint(_meta(), str(data_dir))
    assert "SID codes" in str(exc.value)


def test_missing_vocab_is_rejected(tmp_path):
    data_dir = tmp_path / "not_a_dataset"
    os.makedirs(data_dir)
    with pytest.raises(SystemExit) as exc:
        E.check_data_dir_matches_checkpoint(_meta(), str(data_dir))
    assert "vocab.json" in str(exc.value)

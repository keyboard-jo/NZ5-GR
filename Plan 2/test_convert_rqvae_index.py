"""Tests for convert_rqvae_index.py. Mock data only."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import convert_rqvae_index as conv


def test_align_follows_item_id_order_not_file_order():
    # file order is deliberately scrambled relative to the catalogue
    index = {"11": [3, 1], "9": [2, 0], "10": [1, 2]}
    ids = np.array([9, 10, 11], dtype=np.int64)
    codes, sizes = conv.align(index, ids, verbose=False)
    assert codes.dtype == np.int32
    assert codes.tolist() == [[2, 0], [1, 2], [3, 1]]
    assert sizes == [4, 3]


def test_align_rejects_a_missing_item():
    index = {"1": [0], "3": [1]}
    with pytest.raises(KeyError, match="absent from the index"):
        conv.align(index, np.array([1, 2, 3]), verbose=False)


def test_align_rejects_ragged_lengths():
    index = {"1": [0], "2": [0, 1]}
    with pytest.raises(ValueError, match="ragged"):
        conv.align(index, np.array([1, 2]), verbose=False)


def test_align_rejects_negative_codes():
    index = {"1": [-1], "2": [0]}
    with pytest.raises(ValueError, match="negative"):
        conv.align(index, np.array([1, 2]), verbose=False)


def test_width_is_observed_max_not_declared_size():
    # a codebook that is half used must report its observed width, not the
    # 8192 it was built with, because nothing in the index carries that number
    index = {str(i): [i % 5, 0] for i in range(1, 11)}
    _codes, sizes = conv.align(index, np.arange(1, 11), verbose=False)
    assert sizes == [5, 1]


def test_vocab_shape():
    assert conv.vocab_of([8192, 8192]) == {"sid_sizes": [8192, 8192], "num_levels": 2}


def test_cli_round_trip(tmp_path):
    index = {str(i): [i % 4, i % 2] for i in range(1, 9)}
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index))
    ids_path = tmp_path / "ids.npy"
    np.save(ids_path, np.arange(1, 9))

    rc = conv.main(
        ["--index", str(index_path), "--item_ids", str(ids_path),
         "--output_dir", str(tmp_path), "--tag", "tiny", "--quiet"]
    )
    assert rc == 0
    codes = np.load(tmp_path / "codes_tiny_aligned.npy")
    vocab = json.loads((tmp_path / "vocab_tiny.json").read_text())
    assert codes.shape == (8, 2)
    assert vocab == {"sid_sizes": [4, 2], "num_levels": 2}
    # and the result feeds the diagnostics without a cast
    assert codes.dtype == np.int32

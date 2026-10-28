"""Small unit tests for the Tier 1/2/3 patch to train_digr.py.

Covers, without importing transformers weights or touching a GPU:
  * the pre-allocated exact full-catalog `item_logits` (bit-identical to the
    old chunk-and-cat version),
  * the new sampled-softmax `item_logits_sampled` (uniform proposal + log(K/N)),
  * `resolve_label_subset` (full identity vs clicked remap),
  * the new CLI flags and their defaults / validation.
"""
import math
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

import train_digr as td


class _FakeBackbone(nn.Module):
    def __init__(self, embed: nn.Embedding):
        super().__init__()
        self._embed = embed

    def get_input_embeddings(self) -> nn.Embedding:
        return self._embed


def _retriever(cat: np.ndarray, temperature=0.07, l2_normalize=False,
               logit_chunk=2):
    embed = nn.Embedding(8, 4)
    torch.nn.init.normal_(embed.weight, 0, 0.2)
    return td.DiGRRetriever(
        _FakeBackbone(embed),
        catalog_token_ids=cat,
        temperature=temperature,
        l2_normalize=l2_normalize,
        logit_chunk=logit_chunk,
    )


CAT4 = np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int64)  # 4 items, L=2


def test_item_logits_shape_and_values_match_manual():
    r = _retriever(CAT4, logit_chunk=2)  # 2 chunks -> exercises the chunk loop
    h = torch.randn(3, 4)
    logits = r.item_logits(h)

    w = r.backbone.get_input_embeddings().weight
    g = h @ w.t()
    scale = 1.0 / (2 * 0.07)
    expected = torch.stack(
        [scale * (g[:, CAT4[i, 0]] + g[:, CAT4[i, 1]]) for i in range(4)], dim=1
    )
    assert logits.shape == (3, 4)
    assert torch.allclose(logits, expected, atol=1e-6)


def test_item_logits_l2_normalize_finite_and_shaped():
    r = _retriever(CAT4, l2_normalize=True)
    logits = r.item_logits(torch.randn(3, 4))
    assert logits.shape == (3, 4)
    assert torch.isfinite(logits).all()


def test_item_logits_sampled_matches_manual():
    r = _retriever(CAT4)
    h = torch.randn(2, 4)
    targets = torch.tensor([0, 2])
    neg_tokens = torch.tensor([[1, 3], [5, 7]])  # N=2, L=2

    logits = r.item_logits_sampled(h, targets, neg_tokens)

    w = r.backbone.get_input_embeddings().weight
    g = h @ w.t()
    scale = 1.0 / (2 * 0.07)

    pos_toks = CAT4[targets.numpy()]
    pos = g.gather(1, torch.as_tensor(pos_toks[:, 0:1])) \
        + g.gather(1, torch.as_tensor(pos_toks[:, 1:2]))
    pos = pos * scale

    neg = g.index_select(1, neg_tokens[:, 0]) + g.index_select(1, neg_tokens[:, 1])
    log_k_over_n = math.log(r.catalog_size / neg_tokens.shape[0])
    neg = neg * scale + log_k_over_n

    expected = torch.cat([pos, neg], dim=1)
    assert logits.shape == (2, 3)
    assert torch.allclose(logits, expected, atol=1e-6)


def test_item_logits_sampled_log_k_over_n_uses_catalog_size():
    # 6-item catalog, 3 negatives => log(6/3) = log(2) added to every negative.
    cat6 = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [1, 2], [3, 4]], dtype=np.int64)
    r = _retriever(cat6)
    h = torch.zeros(1, 4)
    targets = torch.tensor([0])
    neg_tokens = torch.tensor([[1, 3], [5, 7], [2, 6]])

    logits = r.item_logits_sampled(h, targets, neg_tokens)
    pos, neg = logits[0, 0], logits[0, 1:]

    # h = 0 => g = 0 => neg is exactly log(K/N) on every level-sum, i.e. log(2).
    assert torch.allclose(neg, torch.full_like(neg, math.log(2.0)), atol=1e-6)
    assert torch.allclose(pos, torch.zeros_like(pos), atol=1e-6)


def test_resolve_label_subset_full_identity():
    cat = np.arange(12).reshape(4, 3)
    mat, remap = td.resolve_label_subset(cat, np.array([0, 2, 3]),
                                         np.array([1, 3]), "full")
    assert np.array_equal(mat, cat)
    assert np.array_equal(remap, np.arange(4))


def test_resolve_label_subset_clicked_remap():
    cat = np.arange(15).reshape(5, 3)
    train = np.array([0, 4, 4, 2])
    val = np.array([2, 3])
    mat, remap = td.resolve_label_subset(cat, train, val, "clicked")

    clicked = np.array([0, 2, 3, 4])  # unique union, ascending
    assert np.array_equal(mat, cat[clicked])
    assert np.array_equal(remap, np.array([0, -1, 1, 2, 3]))
    # item 1 (never a target) is dropped; every target maps to a valid new id.
    assert np.array_equal(remap[train], np.array([0, 3, 3, 1]))
    assert np.array_equal(remap[val], np.array([1, 2]))


def test_resolve_label_subset_unknown_raises():
    cat = np.arange(12).reshape(4, 3)
    with pytest.raises(ValueError):
        td.resolve_label_subset(cat, np.array([0]), np.array([1]), "bogus")


def test_constants_updated():
    assert td.DEFAULT_BATCH_SIZE == 512
    assert td.DEFAULT_NUM_NEGATIVES == 100_000


def test_cli_defaults_tier1():
    args = td.parse_args([])
    assert args.batch_size == 512
    assert args.use_lora is False
    assert args.label_subset == "full"
    assert args.sampled_softmax is False
    assert args.num_negatives == 100_000


def test_cli_toggles_tier2_tier3():
    args = td.parse_args(
        ["--use_lora", "--label_subset", "clicked", "--sampled_softmax",
         "--num_negatives", "1234"]
    )
    assert args.use_lora is True
    assert args.label_subset == "clicked"
    assert args.sampled_softmax is True
    assert args.num_negatives == 1234


def test_cli_num_negatives_must_be_positive_when_sampled():
    with pytest.raises(SystemExit):
        td.parse_args(["--sampled_softmax", "--num_negatives", "0"])


# ---------------------------------------------------------------------------
# mid-epoch checkpointing / resume
# ---------------------------------------------------------------------------

def test_resolve_resume_position_mid_epoch():
    state = {"epoch": 1, "batch": 400}
    assert td.resolve_resume_position(state, steps_per_epoch=3000) == (1, 400)


def test_resolve_resume_position_epoch_finished():
    # the epoch-end save writes batch == steps_per_epoch
    state = {"epoch": 1, "batch": 3000}
    assert td.resolve_resume_position(state, steps_per_epoch=3000) == (2, 0)


def test_resolve_resume_position_legacy_format():
    # no "batch" key: "epoch" is the last completed epoch
    assert td.resolve_resume_position({"epoch": 0}, steps_per_epoch=3000) == (1, 0)
    assert td.resolve_resume_position({}, steps_per_epoch=3000) == (1, 0)


def test_save_steps_default_and_disabled():
    assert td.DEFAULT_SAVE_STEPS == 1000
    assert td.parse_args([]).save_steps == 1000
    assert td.parse_args(["--save_steps", "0"]).save_steps == 0  # disables it


def test_save_steps_negative_rejected():
    with pytest.raises(SystemExit):
        td.parse_args(["--save_steps", "-1"])


def test_save_checkpoint_is_atomic(tmp_path):
    """A checkpoint dir is only ever replaced by a complete tmp dir."""
    class _Model:
        def __init__(self):
            self.n = 0

        def save_pretrained(self, out, safe_serialization=True):
            assert safe_serialization is True
            self.n += 1
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "weights.bin"), "w") as fh:
                fh.write("w%d" % self.n)

    class _Meta:
        def save(self, out):
            with open(os.path.join(out, "head_config.json"), "w") as fh:
                fh.write("{}")

    out = str(tmp_path / "final")
    model = _Model()
    td.save_checkpoint(model, _Meta(), out, state={"epoch": 0, "batch": 7})

    assert os.path.isfile(os.path.join(out, "weights.bin"))
    assert os.path.isfile(os.path.join(out, "head_config.json"))
    assert os.path.isfile(os.path.join(out, "train_state.pt"))
    assert not os.path.exists(out + ".tmp")
    assert torch.load(os.path.join(out, "train_state.pt"))["batch"] == 7

    # a second save fully replaces the first (no stale files, no .tmp left)
    td.save_checkpoint(model, _Meta(), out, state={"epoch": 0, "batch": 14})
    assert sorted(os.listdir(out)) == ["head_config.json", "train_state.pt", "weights.bin"]
    assert not os.path.exists(out + ".tmp")
    assert torch.load(os.path.join(out, "train_state.pt"))["batch"] == 14


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
class _FakeScorer(nn.Module):
    """Returns a hand-written logit matrix so each row's rank is known exactly."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self._logits = logits

    def encode(self, ids, mask):
        return torch.zeros(ids.shape[0], 1)

    def item_logits(self, h):
        return self._logits


def _loader(targets, n_rows, n_items=12):
    ids = torch.zeros(n_rows, 2, dtype=torch.long)
    mask = torch.ones(n_rows, 2, dtype=torch.long)
    return [(ids, mask, torch.as_tensor(targets), torch.arange(n_rows))]


def test_evaluate_counts_a_miss_as_a_miss_at_every_k():
    # 12 items, best score first.  Rows hit at rank 1, at rank 2, and never
    # (target 2 is scored lowest of all 12).
    logits = torch.zeros(3, 12)
    logits[0] = torch.arange(12, 0, -1).float()              # target 0 -> rank 1
    logits[1] = torch.arange(12, 0, -1).float()
    logits[1, 1] = 11.5                                        # target 1 -> rank 2
    logits[2] = torch.arange(12, 0, -1).float()              # target 2 -> rank 12
    logits[2, 2] = -100.0

    out = td.evaluate(_FakeScorer(logits), _loader([0, 1, 2], 3),
                      torch.device("cpu"), torch.float32, False)

    assert out["n"] == 3
    assert out["recall@1"] == pytest.approx(1 / 3)
    assert out["recall@5"] == pytest.approx(2 / 3)
    assert out["recall@10"] == pytest.approx(2 / 3)   # the miss must not count
    assert out["ndcg@1"] == pytest.approx(1 / 3)
    assert out["ndcg@10"] == pytest.approx((1 + 1 / math.log2(3)) / 3)
    assert out["mrr"] == pytest.approx((1 + 0.5) / 3)
    assert math.isfinite(out["ndcg@10"])


def test_evaluate_ndcg_is_finite_when_every_row_misses():
    logits = torch.zeros(2, 12)
    logits[0, 0] = -100.0
    logits[1, 1] = -100.0
    out = td.evaluate(_FakeScorer(logits), _loader([0, 1], 2),
                      torch.device("cpu"), torch.float32, False)
    assert out["recall@1"] == 0.0
    assert out["recall@10"] == 0.0
    assert out["ndcg@10"] == 0.0
    assert out["mrr"] == 0.0
    assert math.isfinite(out["loss"])


def test_widening_eval_ks_does_not_inflate_mrr(monkeypatch):
    # A recall@N curve needs EVAL_KS widened to 100/1000.  MRR must stay at
    # MRR_K, otherwise a rank-60 hit silently turns into a large MRR gain and
    # the widened run reports a "better" MRR than the @10 run it is compared to.
    monkeypatch.setattr(td, "EVAL_KS", (1, 5, 10, 100))
    logits = torch.zeros(1, 120)
    logits[0, 0] = 1.0            # target, weak score
    logits[0, 1:60] = 2.0         # 59 items outrank it
    logits[0, 60:] = 0.0          # the rest do not

    out = td.evaluate(_FakeScorer(logits), _loader([0], 1),
                      torch.device("cpu"), torch.float32, False)

    assert out["recall@1"] == 0.0
    assert out["recall@10"] == 0.0
    assert out["recall@100"] == 1.0            # rank 60 is inside the wide cut-off
    assert out["ndcg@100"] == pytest.approx(1 / math.log2(61))
    assert out["mrr"] == 0.0                   # ... but outside MRR_K
    assert "recall@5" in out and "ndcg@1000" not in out


def test_vocab_reads_the_action_block_from_a_v2_vocab_file():
    """A v2 vocab.json must round-trip, action tokens included.

    The v2 preprocessor writes total_size = 2 + sum(sid_sizes) + n_temporal +
    n_action + sum(feature_sizes). The v1 class knows nothing about n_action, so
    reading a v2 file with it shortens the embedding table by three rows and every
    id in the action block goes out of range (152 instead of 155 on the fixture
    below). This pins that train_digr imports the module that understands it.
    """
    payload = {
        "sid_sizes": [4, 4, 4, 4, 2],
        "n_temporal": 8,
        "n_action": 3,
        "feature_sizes": [5, 2],
        "total_size": 2 + 18 + 8 + 3 + 7,  # 38
    }
    vocab = td.Vocab.from_dict(payload)
    assert vocab.n_action == 3
    assert vocab.total_size == 38
    assert vocab.event_width == 1 + 1 + 5  # gap, action, five SID levels
    assert td.Vocab.__module__ in ("preprocess_digr_v2", "preprocess_digr")


def test_vocab_manifest_mismatch_is_fatal():
    """Reading a v2 file with a reader that drops the action block must fail loudly."""
    vocab = td.Vocab.from_dict({
        "sid_sizes": [4, 4], "n_temporal": 8, "feature_sizes": [],
    })
    td._check_vocab_matches_manifest(vocab, {"vocab_size": int(vocab.total_size)})
    with pytest.raises(SystemExit) as err:
        td._check_vocab_matches_manifest(vocab, {"vocab_size": int(vocab.total_size) + 3})
    assert "token layout" in str(err.value)
    # A v1 manifest has no vocab_size and must not be rejected.
    td._check_vocab_matches_manifest(vocab, {})


def test_temperature_default_keeps_the_score_scale_fixed():
    """1/(L*tau) is what the head sees, so the default tau has to follow L."""
    ref = 1.0 / (td.REFERENCE_NUM_LEVELS * td.DEFAULT_TEMPERATURE)
    for levels in (1, 2, 3, 5, 8):
        tau = td.resolve_temperature(None, levels)
        assert abs(1.0 / (levels * tau) - ref) < 1e-12
    # An explicit value always wins, so a hand-tuned run is untouched.
    assert td.resolve_temperature(0.3, 2) == 0.3
    # The 5-level reference reproduces the frozen run exactly.
    assert abs(td.resolve_temperature(None, 5) - 0.07) < 1e-12


def test_event_width_falls_back_to_the_v1_slot():
    """A v1 vocabulary has no event_width; the slot is then 1 gap + L codes."""
    v1 = td.Vocab.from_dict({"sid_sizes": [4] * 5, "n_temporal": 8, "feature_sizes": []})
    assert td._event_width(v1) == 6
    v2 = td.Vocab.from_dict({"sid_sizes": [8, 8], "n_temporal": 128, "n_action": 3,
                             "feature_sizes": []})
    assert td._event_width(v2) == 4

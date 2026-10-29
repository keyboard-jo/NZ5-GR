"""TIGER training and evaluation over a preprocessed `digr_data` directory.

TIGER is the comparison arm for DiGR: instead of scoring an item by a mean of its
code embeddings, it predicts the semantic ID token by token, one softmax per level,
and ranks items by the product of the level conditionals.

Two SID releases are supported, and the script reads whichever one `--data_dir`
points at, because the level count, the level widths and the token layout all come
out of the vocabulary written by the preprocessing phase:

  v1  `digr_data`       4 x 256 plus a 731-wide collision suffix, 5 levels, 64 gap
                        bins, no action token, 132 tokens per row
  v2  `digr_data_v2`    2 x 8,192, 2 levels, 128 gap bins, action token, 89 tokens
                        per row

INPUTS (all produced by the preprocessing phase, nothing is read from parquet here)
  manifest.json, vocab.json, catalog_item_ids.npy, catalog_sid_codes.npy,
  users.npy, user_feat.npy, train/{tokens,offsets,targets,users}.npy,
  val/{tokens,offsets,targets,users}.npy

THE PREFIX QUESTION
  The paper decodes level by level, so level `l` is conditioned on the codes chosen
  before it, and a prefix tree is needed at inference to keep the beam inside
  tuples that some item actually owns. That matters most for v1, where 256^4
  prefixes exist and only 1.5M of them are real, and it barely matters for v2,
  where a level-1 code has 584 items behind it on average and every one of the 8,192
  codes is real.

  Prefix conditioning and exhaustive full-catalogue scoring cannot both hold: a
  prefix-conditioned score for item `i` needs `i`'s own prefix in the input, so
  scoring all 4,783,154 items costs one forward pass per item. Two arms therefore:

  `--prefix_conditioning off` trains the L heads on the separator state with no
  prefix in the input, so all L logits come from one forward and the exact product
  score of every catalogue item is L gathers. This is the cheap arm: it is scored by
  the same full-catalogue evaluator as DiGR, co-click and popularity, which is why
  it was the first one run, but its score is additive over the code rows, so it does
  not test the paper's factorisation.

  `--prefix_conditioning on` trains with the true prefix fed back, which is the
  paper's TIGER, and evaluates with `--mode beam` over the prefix tree, because
  exhaustive scoring is not available. This is the arm `Plan 2/train_tiger.sh` runs
  by default (`PREFIX=1`). Its reachable cut-offs are capped at the beam width, so
  it reports `cover@b` beside them: `R@k(beam b) <= cover@b`.

QUICK START
  python preprocess_digr_v2.py --output_dir digr_data_v2 ...
  python train_tiger.py --mode train --data_dir digr_data_v2 --epochs 2 \
      --batch_size 512 --lr 5e-5 --embed_lr_mult 5.0
  python train_tiger.py --mode eval --checkpoint runs/tiger_v2/final --data_dir digr_data_v2
  python train_tiger.py --mode beam --checkpoint runs/tiger_v2/final --data_dir digr_data \
      --prefix_conditioning on --beam_width 20 --beam_rows 512

WHY THERE IS NO SAMPLED SOFTMAX
  A level has 256 to 8,192 classes, so its full softmax costs `hidden x classes`
  (8,192 x 1,024 is 16.8M multiply-adds per row) and needs no approximation. That is
  the opposite of DiGR, whose single softmax runs over all 4,783,154 items and is the
  reason its training needed 300,000 sampled negatives.

H800 NOTES
  v2: 92 tokens per training row (89 + separator + 2 codes), batch 512, bf16,
  gradient checkpointing on, about 63 minutes per epoch for 1,719,430 rows. The head
  is two [1,024, 8,192] matrices, so it costs the same as DiGR's code table.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None

from transformers import AutoModel

# The preprocessing phase owns the token layout. `train_digr.py` sits next to this
# file and already resolves the v2-vs-v1 import (same directory first, then the
# "Plan 2" sibling), so reuse its resolution instead of duplicating it.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import train_digr as digr  # noqa: E402  (path fix above)
from train_digr import (  # noqa: E402
    ItemCatalog,
    LengthBucketBatchSampler,
    RunMeta,
    Vocab,
    _enable_gradient_checkpointing,
    _load_full_checkpoint,
    _load_splits,
    _load_static_meta,
    _require_peft,
    build_optimizer,
    build_peft_backbone,
    catalog_token_matrix,
    human_int,
    load_backbone,
    log_banner,
    log_card,
    resolve_resume_position,
    setup_logger,
)

SEED = digr.SEED
DEFAULT_MODEL_NAME = digr.DEFAULT_MODEL_NAME
DEFAULT_DATA_DIR = digr.DEFAULT_DATA_DIR
DEFAULT_OUTPUT_DIR = digr.DEFAULT_OUTPUT_DIR
DEFAULT_BATCH_SIZE = 512
DEFAULT_EVAL_BATCH_SIZE = 128
DEFAULT_LR = 5e-5
DEFAULT_EMBED_LR_MULT = 5.0
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_MIN_LR_RATIO = 0.1
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_LABEL_SMOOTHING = 0.0
DEFAULT_SAVE_STEPS = 300
DEFAULT_HEAD_INIT_STD = 0.02
DEFAULT_BEAM_WIDTH = 20
DEFAULT_CATALOG_CHUNK = 200_000

# Rank cut-offs reported by `--mode eval`. All five come out of one pass because
# the rank of the target is counted directly, so there is no top-k tensor to bound.
EVAL_KS = (1, 5, 10, 100, 1000, 10000)
MRR_K = 10

# A mixed-radix prefix key must stay inside int64. 256**4 = 4.29e9 is fine, 731
# levels wrapped in int32 are not, and this ceiling is the same guard used by the
# SID diagnostics script.
MAX_KEY_PRODUCT = 2 ** 62


# ---------------------------------------------------------------------------
# Prefix keys and the prefix tree
# ---------------------------------------------------------------------------

def pack_prefix(codes: np.ndarray, radix: Sequence[int]) -> np.ndarray:
    """Mixed-radix key for every row of `codes` (one key per row).

    `codes` is [n, k] with values in `[0, radix[l])`. Keys are only compared for
    equality and ordering, so the exact map does not matter as long as it is
    injective: `key = c_0 * r_1 * r_2 * ... + c_1 * r_2 * ... + c_{k-1}`.
    """
    codes = np.asarray(codes, dtype=np.int64)
    if codes.ndim != 2:
        raise ValueError(f"pack_prefix expects a 2-d array, got shape {codes.shape}")
    k = codes.shape[1]
    radix = [int(r) for r in radix[:k]]
    product = 1
    for r in radix[1:]:
        product *= r
        if product >= MAX_KEY_PRODUCT:
            raise ValueError(
                f"prefix of {k} levels over radix {radix} does not fit in int64 "
                f"(product {product}); score this level without packing keys"
            )
    key = np.zeros(codes.shape[0], dtype=np.int64)
    for level in range(k):
        stride = 1
        for r in radix[level + 1:]:
            stride *= r
        if (codes[:, level] < 0).any() or (codes[:, level] >= radix[level]).any():
            raise ValueError(
                f"code out of range at level {level}: expected [0, {radix[level]})"
            )
        key += codes[:, level] * stride
    return key


class SidIndex:
    """Which codes are valid continuations of a prefix, without a dict per node.

    Rows are sorted lexicographically, so the items sharing a prefix of length `l`
    occupy one contiguous run. `keys[l - 1]` holds the packed prefix key of every
    sorted row, which turns "children of this prefix" into two `searchsorted` calls.
    Memory is `L - 1` int64 arrays of the catalogue size, not a node object per
    distinct prefix (v1 has 3.46M distinct 4-code prefixes).
    """

    def __init__(self, codes: np.ndarray, sid_sizes: Sequence[int]):
        codes = np.asarray(codes, dtype=np.int32)
        if codes.ndim != 2:
            raise ValueError(f"SidIndex expects [n, levels] codes, got {codes.shape}")
        self.sid_sizes = [int(s) for s in sid_sizes]
        self.num_levels = int(codes.shape[1])
        if len(self.sid_sizes) < self.num_levels:
            raise ValueError(
                f"{self.num_levels} code levels but only {len(self.sid_sizes)} widths"
            )
        # lexsort sorts by the last key first, so pass the columns reversed.
        order = np.lexsort([codes[:, l] for l in range(self.num_levels - 1, -1, -1)])
        self.codes = codes[order]
        self.order = order
        self.keys: List[np.ndarray] = []
        for level in range(1, self.num_levels):
            key = pack_prefix(self.codes[:, :level], self.sid_sizes)
            # pack_prefix is monotone in the leading codes, and the rows are already
            # sorted by those codes, so the keys come out non-decreasing.
            self.keys.append(key)

    def children(self, prefix: Sequence[int]) -> np.ndarray:
        """Valid codes at `len(prefix)` given a prefix of raw codes."""
        level = len(prefix)
        if level == 0:
            return np.arange(self.sid_sizes[0], dtype=np.int32)
        if level >= self.num_levels:
            return np.empty(0, dtype=np.int32)
        key = pack_prefix(np.asarray([prefix], dtype=np.int64), self.sid_sizes)
        keys = self.keys[level - 1]
        lo = int(np.searchsorted(keys, key[0], side="left"))
        hi = int(np.searchsorted(keys, key[0], side="right"))
        if lo == hi:
            return np.empty(0, dtype=np.int32)
        return np.unique(self.codes[lo:hi, level])

    def contains(self, codes_row: Sequence[int]) -> bool:
        """True when the catalogue owns this exact tuple."""
        if len(codes_row) != self.num_levels:
            return False
        prefix = np.asarray([list(codes_row)[:-1]], dtype=np.int64)
        key = pack_prefix(prefix, self.sid_sizes)
        keys = self.keys[self.num_levels - 2]
        lo = int(np.searchsorted(keys, key[0], side="left"))
        hi = int(np.searchsorted(keys, key[0], side="right"))
        if lo == hi:
            return False
        return bool((self.codes[lo:hi, -1] == int(codes_row[-1])).any())


def beam_search(step_fn: Callable[[List[Tuple[int, ...]]], np.ndarray],
                sid_sizes: Sequence[int],
                beam_width: int = DEFAULT_BEAM_WIDTH,
                allowed: Optional[Callable[[Tuple[int, ...], int], np.ndarray]] = None,
                ) -> List[Tuple[float, Tuple[int, ...]]]:
    """Breadth-first beam over `len(sid_sizes)` levels, best tuple first.

    `step_fn(prefixes)` returns log-probabilities `[len(prefixes), sid_sizes[level]]`
    for every prefix in the list, where `level == len(prefixes[0])`. `allowed(prefix, code)`
    masks the codes that no catalogue item continues with; the prefix tree drops in
    here. Kept free of torch so it can be tested with a stub scorer.
    """
    beam: List[Tuple[float, Tuple[int, ...]]] = [(0.0, ())]
    for level, width in enumerate(sid_sizes):
        if not beam:
            break
        prefixes = [p for _, p in beam]
        logp = np.asarray(step_fn(prefixes), dtype=np.float64)
        if logp.shape != (len(prefixes), int(width)):
            raise ValueError(
                f"step_fn returned {logp.shape} at level {level}, expected "
                f"({len(prefixes)}, {int(width)})"
            )
        candidates: List[Tuple[float, Tuple[int, ...]]] = []
        for row, (base, prefix) in enumerate(beam):
            keep = (np.arange(int(width)) if allowed is None
                    else np.asarray(allowed(prefix, level), dtype=np.int64))
            if keep.size == 0:
                continue
            scores = logp[row, keep] + base
            top = np.argsort(-scores)[:int(beam_width)]
            for t in top:
                candidates.append((float(scores[t]), prefix + (int(keep[t]),)))
        candidates.sort(key=lambda item: -item[0])
        beam = candidates[:int(beam_width)]
    return beam


# ---------------------------------------------------------------------------
# Data: the row, plus a separator, plus the target's code tokens
# ---------------------------------------------------------------------------

class TigerDataset(Dataset):
    """One training row: the history tokens and the target's SID tokens.

    `sid_input[j]` are the vocabulary ids of item `j`'s codes (offset + code) and go
    into the sequence, `sid_codes[j]` are the raw codes and are the head's labels.
    """

    def __init__(self, split, indices: np.ndarray, sid_input: np.ndarray,
                 sid_codes: np.ndarray, sep_id: int, prefix_conditioning: bool):
        self.tokens = split.tokens
        self.offsets = split.offsets
        self.targets = split.targets
        self.indices = np.asarray(indices, dtype=np.int64)
        self.lengths = split.lengths[self.indices]
        self.sid_input = sid_input
        self.sid_codes = sid_codes
        self.sep_id = int(sep_id)
        self.prefix_conditioning = bool(prefix_conditioning)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = int(self.indices[i])
        target = int(self.targets[j])
        s = int(self.offsets[j])
        e = int(self.offsets[j + 1])
        # `target` is the catalogue row index, which is what the evaluator needs to
        # rank the right item. It is not recoverable from the target's codes: the
        # first code takes only `sid_sizes[0]` distinct values, so using it as a row
        # index would rank some unrelated item.
        return (self.tokens[s:e], self.sid_input[target],
                self.sid_codes[target], target, j)


class TigerCollate:
    """Right-pad to `[sep] + codes`, and record where each code's prediction sits.

    Row tokens are `history` with length `n`. The appended block is
    `[SEP, c_1, ..., c_L]`, so the separator is at index `n` and the code that
    predicts `c_l` sits at `n + l`. Positions are indices into the padded row, which
    is why they are computed here and not in the dataset.

    With `prefix_conditioning` off only the separator is appended, every level reads
    the same hidden state, and the code tokens are not in the input at all.
    """

    def __init__(self, sep_id: int, num_levels: int, prefix_conditioning: bool):
        self.sep_id = int(sep_id)
        self.num_levels = int(num_levels)
        self.prefix_conditioning = bool(prefix_conditioning)

    def __call__(self, batch):
        b = len(batch)
        n_levels = self.num_levels
        tail = 1 + (n_levels if self.prefix_conditioning else 0)
        max_len = max(len(item[0]) for item in batch)
        ids = np.zeros((b, max_len + tail), dtype=np.int64)
        mask = np.zeros((b, max_len + tail), dtype=np.int64)
        labels = np.zeros((b, n_levels), dtype=np.int64)
        positions = np.zeros((b, n_levels), dtype=np.int64)
        targets = np.zeros(b, dtype=np.int64)
        rows = np.zeros(b, dtype=np.int64)

        for i, (history, sid_in, sid_code, target_row, j) in enumerate(batch):
            n = len(history)
            ids[i, :n] = history
            mask[i, :n] = 1
            ids[i, n] = self.sep_id
            mask[i, n] = 1
            if self.prefix_conditioning:
                ids[i, n + 1:n + 1 + n_levels] = sid_in
                mask[i, n + 1:n + 1 + n_levels] = 1
            labels[i] = sid_code
            # With prefix conditioning the level-`l` prediction is read from the
            # position of the code before it: the separator for level 0, then the
            # code token itself. Without it every level reads the separator, because
            # no code is in the input at all.
            positions[i] = (n + np.arange(n_levels, dtype=np.int64)
                            if self.prefix_conditioning
                            else np.full(n_levels, n, dtype=np.int64))
            targets[i] = target_row           # catalogue row, not a code
            rows[i] = j

        return (
            torch.from_numpy(ids),
            torch.from_numpy(mask),
            torch.from_numpy(labels),
            torch.from_numpy(positions),
            torch.from_numpy(targets),
            torch.from_numpy(rows),
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TigerHead(torch.nn.Module):
    """One linear map per level, `[hidden, sid_sizes[l]]`, no bias."""

    def __init__(self, hidden_size: int, sid_sizes: Sequence[int],
                 init_std: float = DEFAULT_HEAD_INIT_STD):
        super().__init__()
        self.sid_sizes = [int(s) for s in sid_sizes]
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(int(hidden_size), s, bias=False) for s in self.sid_sizes]
        )
        with torch.no_grad():
            for lin in self.layers:
                lin.weight.normal_(mean=0.0, std=float(init_std))

    def forward(self, states: torch.Tensor) -> List[torch.Tensor]:
        # Cast per call so the head works with or without autocast, and whichever
        # dtype the backbone ended up in. The backbone is loaded in bf16 while these
        # weights stay fp32, so a hidden state handed to the head outside an autocast
        # block is bf16 against fp32 weights and `F.linear` refuses it. Autocast did
        # the cast during training, which is why training ran and evaluation did not.
        return [lin(states.to(lin.weight.dtype)) for lin in self.layers]

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TigerRetriever(torch.nn.Module):
    """Backbone plus per-level heads, with both scoring paths on one object."""

    # Set by the trainer and written into `head_config.json`, so a checkpoint says
    # which arm it is instead of the evaluator having to be told.
    prefix_conditioning: bool = False

    def __init__(self, backbone: torch.nn.Module, hidden_size: int,
                 sid_sizes: Sequence[int], catalog_codes: np.ndarray,
                 head_init_std: float = DEFAULT_HEAD_INIT_STD):
        super().__init__()
        self.backbone = backbone
        self.head = TigerHead(int(hidden_size), sid_sizes, head_init_std)
        codes = torch.as_tensor(np.asarray(catalog_codes, dtype=np.int64),
                                dtype=torch.long)
        self.register_buffer("catalog_codes", codes, persistent=False)
        self.sid_sizes = [int(s) for s in sid_sizes]
        self.num_levels = len(self.sid_sizes)

    @property
    def catalog_size(self) -> int:
        return int(self.catalog_codes.shape[0])

    def hidden_states(self, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return out.last_hidden_state

    @staticmethod
    def gather_states(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """[B, T, d] + [B, k] indices -> [B, k, d]."""
        b, k = positions.shape
        d = hidden.shape[-1]
        idx = positions.reshape(b, k, 1).expand(b, k, d)
        return hidden.gather(1, idx)

    def logits_at(self, hidden: torch.Tensor, positions: torch.Tensor) -> List[torch.Tensor]:
        """Per-level logits at each of the `k` prediction positions."""
        states = self.gather_states(hidden, positions)          # [B, k, d]
        out: List[torch.Tensor] = []
        for level, lin in enumerate(self.head.layers):
            out.append(lin(states[:, level]))
        return out

    def logits_from_state(self, states: torch.Tensor) -> List[torch.Tensor]:
        """Per-level logits from one hidden state per row (no prefix feedback)."""
        return self.head(states)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                positions: torch.Tensor) -> List[torch.Tensor]:
        return self.logits_at(self.hidden_states(input_ids, attention_mask), positions)

    def separator_states(self, input_ids: torch.Tensor,
                         attention_mask: torch.Tensor) -> torch.Tensor:
        """Hidden state at the separator, the position that scores the item."""
        hidden = self.hidden_states(input_ids, attention_mask)
        lengths = attention_mask.sum(dim=1).clamp(min=1)
        # The separator is the last real token only when no codes were appended,
        # which is exactly the prefix-free layout used for scoring.
        idx = lengths - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]


def level_cross_entropies(logits: Sequence[torch.Tensor], labels: torch.Tensor,
                          label_smoothing: float = 0.0) -> List[torch.Tensor]:
    return [
        F.cross_entropy(logits[level], labels[:, level],
                        label_smoothing=float(label_smoothing))
        for level in range(len(logits))
    ]


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def rank_metrics(ranks: np.ndarray, ks: Sequence[int] = EVAL_KS,
                 mrr_k: int = MRR_K) -> Dict[str, float]:
    """Ranking metrics from 1-based ranks (a miss is `max(ks) + 1`)."""
    ranks = np.asarray(ranks, dtype=np.float64)
    n = max(1, ranks.size)
    out: Dict[str, float] = {"n": int(ranks.size)}
    for k in ks:
        hit = ranks <= k
        out[f"recall@{k}"] = float(hit.sum()) / n
        out[f"ndcg@{k}"] = float(
            np.where(hit, 1.0 / np.log2(ranks + 1.0), 0.0).sum()
        ) / n
    out["mrr"] = float(
        np.where(ranks <= mrr_k, 1.0 / np.maximum(ranks, 1.0), 0.0).sum()
    ) / n
    return out


@torch.inference_mode()
def target_ranks(model: TigerRetriever, states: torch.Tensor, targets: torch.Tensor,
                 chunk: int = DEFAULT_CATALOG_CHUNK) -> torch.Tensor:
    """Rank of each target under the product score, counted not sorted.

    `score(i) = sum_l log p_l(code_il)`, so the rank of a target is one plus the
    number of catalogue items that score strictly higher. Counting avoids a
    `[B, 4,783,154]` top-k tensor entirely, and it is exact. Ties (items sharing a
    tuple) count as not-higher, so a tied target is reported at the best rank it
    could hold.
    """
    logps = [F.log_softmax(lg.float(), dim=-1) for lg in model.logits_from_state(states)]
    codes = model.catalog_codes
    n = int(codes.shape[0])
    tgt = codes.index_select(0, targets)
    tgt_score = torch.zeros(states.shape[0], dtype=torch.float32, device=codes.device)
    for level in range(model.num_levels):
        tgt_score = tgt_score + logps[level].gather(1, tgt[:, level:level + 1]).squeeze(1)

    ranks = torch.ones(states.shape[0], dtype=torch.int64, device=codes.device)
    for start in range(0, n, int(chunk)):
        end = min(n, start + int(chunk))
        acc = torch.zeros(states.shape[0], end - start, dtype=torch.float32,
                          device=codes.device)
        for level in range(model.num_levels):
            acc = acc + logps[level].index_select(1, codes[start:end, level])
        ranks = ranks + (acc > tgt_score.unsqueeze(1)).sum(dim=1)
    return ranks


# Beam widths at which the level-0 coverage is reported. A beam of `b` can only
# return items whose first code the model placed in its own top `b`, so this is the
# hard ceiling on any prefix-conditioned arm's recall, at every cut-off.
BEAM_COVER_KS = (1, 10, 20, 100, 1000)


@torch.inference_mode()
def evaluate_tiger(model: TigerRetriever, loader: DataLoader, device: torch.device,
                   amp_dtype: torch.dtype, use_amp: bool,
                   catalog_chunk: int = DEFAULT_CATALOG_CHUNK,
                   rank_items: bool = True) -> Dict[str, float]:
    """Teacher-forced loss, level-0 coverage, and optionally full-catalogue ranks.

    `rank_items=False` is for prefix-conditioned training. The exhaustive ranking
    path reads every level from the separator state, which is the prefix-free
    score, so ranking a prefix-conditioned model through it reports a rule the
    model was never trained on. Loss and `cover@k` stay valid in both arms, and
    `--mode beam` is the prefix-conditioned ranking path.
    """
    model.eval()
    kmax = max(EVAL_KS)
    total_loss = 0.0
    total_n = 0
    per_level: List[float] = [0.0] * model.num_levels
    ranks: List[np.ndarray] = []
    cover = {k: 0.0 for k in BEAM_COVER_KS}

    for ids, mask, labels, positions, targets, _rows in loader:
        ids = ids.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        positions = positions.to(device, non_blocking=True)
        b = ids.shape[0]

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(ids, mask, positions)
            losses = level_cross_entropies(logits, labels)
            loss = sum(losses)
            states = model.separator_states(ids, mask) if rank_items else None

        total_loss += float(loss.item()) * b
        for level, part in enumerate(losses):
            per_level[level] += float(part.item()) * b
        total_n += b
        if rank_items:
            ranks.append(target_ranks(model, states, targets.to(device),
                                     chunk=catalog_chunk).cpu().numpy())
        # Level-0 coverage: how often the target's first code is among the model's
        # top `k` first codes. Cheap, and it bounds every beam arm.
        first = F.log_softmax(logits[0].float(), dim=-1)
        width = first.shape[1]
        for k in BEAM_COVER_KS:
            kk = min(int(k), width)
            top = first.topk(kk, dim=-1).indices
            cover[k] += float((top == labels[:, 0:1]).any(dim=-1).sum().item())

    n = max(1, total_n)
    if rank_items:
        rank_array = np.concatenate(ranks) if ranks else np.zeros(0, dtype=np.int64)
        # A target outside the top `kmax` is stored as `kmax + 1`, the same
        # convention as the DiGR evaluator, so it fails every cut-off instead of
        # passing them all.
        rank_array = np.where(rank_array > kmax, kmax + 1, rank_array)
        out = rank_metrics(rank_array)
    else:
        out = {"n": int(total_n)}
    out["ranked"] = 1.0 if rank_items else 0.0
    out["loss"] = total_loss / n
    out["loss@1"] = per_level[0] / n
    for level in range(1, model.num_levels):
        out[f"loss@{level + 1}"] = per_level[level] / n
    for k in BEAM_COVER_KS:
        out[f"cover@{k}"] = cover[k] / n
    return out


def format_metrics(metrics: Dict[str, float], levels: int) -> str:
    """One log line, with nothing the evaluator computed left out.

    `cover@k` is the share of rows whose target's first code is inside the model's
    own top `k` at level 0. Every beam arm's recall is bounded by it, so it is the
    number that sizes the prefix-conditioned arm, and it is printed on both arms.
    """
    parts = [f"loss {metrics['loss']:.4f}"]
    for level in range(levels):
        key = "loss@1" if level == 0 else f"loss@{level + 1}"
        parts.append(f"L{level + 1} {metrics[key]:.4f}")
    if metrics.get("ranked", 1.0):
        for k in EVAL_KS:
            key = f"recall@{k}"
            if key in metrics:
                parts.append(f"R@{k} {metrics[key]:.4f}")
        parts.append(f"nDCG@10 {metrics['ndcg@10']:.4f}")
        parts.append(f"MRR {metrics['mrr']:.4f}")
    else:
        parts.append("ranks skipped (prefix-conditioned, use --mode beam)")
    covers = [f"{k:g}:{metrics[f'cover@{k}'] * 100:.2f}%"
              for k in BEAM_COVER_KS if f"cover@{k}" in metrics]
    if covers:
        parts.append("cover " + " ".join(covers))
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def sid_matrices(catalog: ItemCatalog, vocab: Vocab) -> Tuple[np.ndarray, np.ndarray]:
    """(input token ids, raw codes), both [catalog_size, num_levels]."""
    codes = np.asarray(catalog.sid_codes, dtype=np.int64)
    return catalog_token_matrix(catalog, vocab), codes


def _tiger_meta(args, logger: logging.Logger) -> RunMeta:
    """RunMeta for a TIGER run. Temperature is DiGR-only, so it stays at 1.0."""
    if args.resume:
        if not os.path.isdir(args.resume):
            raise FileNotFoundError(f"--resume must be a checkpoint directory: {args.resume}")
        meta = RunMeta.load(args.resume)
        if args.model_name:
            meta.model_name = args.model_name
        args.dtype = str(meta.dtype)
        head_cfg = _read_head_config(args.resume)
        if "num_levels" in head_cfg and int(head_cfg["num_levels"]) != meta.vocab.num_levels:
            raise ValueError(
                f"checkpoint expects {head_cfg['num_levels']} SID levels but its vocab "
                f"says {meta.vocab.num_levels}; the checkpoint is inconsistent"
            )
        logger.info(f"  + resuming with checkpoint metadata from {args.resume}")
        return meta

    vocab, catalog, user_map, gap_edges, manifest = _load_static_meta(args.data_dir, logger)
    n_levels = int(vocab.num_levels)
    if len(vocab.sid_sizes) != n_levels:
        raise ValueError(
            f"vocabulary lists {len(vocab.sid_sizes)} SID widths but {n_levels} levels"
        )
    logger.info(
        f"  + TIGER head: {n_levels} level(s) over {vocab.sid_sizes}, "
        f"prefix_conditioning={bool(args.prefix_conditioning)}"
    )
    return RunMeta(
        vocab=vocab,
        catalog=catalog,
        user_map=user_map,
        gap_edges=gap_edges,
        model_name=args.model_name or DEFAULT_MODEL_NAME,
        temperature=1.0,
        l2_normalize=False,
        max_seq_len=int(manifest.get("max_seq_len") or 0),
        dtype=args.dtype,
        max_feat_tokens=int(manifest.get("max_feat_tokens") or 0),
        use_lora=bool(args.use_lora),
    )


def _read_head_config(ckpt_dir: Optional[str]) -> Dict[str, Any]:
    if not ckpt_dir:
        return {}
    path = os.path.join(ckpt_dir, HEAD_CONFIG_FILE)
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_model(args, meta: RunMeta, device: torch.device, logger: logging.Logger):
    """Backbone (fresh or resumed) plus the TIGER head."""
    sid_in, sid_codes = sid_matrices(meta.catalog, meta.vocab)
    hidden_size = int(getattr(args, "hidden_size", 0) or 0)
    # Checkpointing trades compute for memory by recomputing activations on the
    # backward pass. Eval never runs one, so it only slows the run down, and a
    # reentrant checkpoint under `torch.inference_mode` has no input that requires
    # grad, which some torch versions reject outright.
    checkpointing = bool(args.gradient_checkpointing) and args.mode == "train"

    if args.resume:
        if meta.use_lora:
            backbone = load_backbone(meta.model_name, meta.vocab.total_size, args, logger)
            _, PeftModel, _ = _require_peft()
            backbone_model = PeftModel.from_pretrained(backbone, args.resume, is_trainable=True)
            if checkpointing:
                _enable_gradient_checkpointing(backbone_model, logger)
        else:
            backbone_model = _load_full_checkpoint(meta.model_name, args.resume, args, logger)
            if checkpointing:
                _enable_gradient_checkpointing(backbone_model, logger)
    else:
        backbone = load_backbone(meta.model_name, meta.vocab.total_size, args, logger)
        if meta.use_lora:
            backbone_model = build_peft_backbone(backbone, args, logger)
        else:
            backbone_model = backbone
            backbone_model.train()
            if checkpointing:
                _enable_gradient_checkpointing(backbone_model, logger)

    if hidden_size <= 0:
        hidden_size = int(getattr(backbone_model.config, "hidden_size", 0) or 0)
    if hidden_size <= 0:
        raise ValueError("could not determine the hidden size; pass --hidden_size")

    model = TigerRetriever(
        backbone_model,
        hidden_size=hidden_size,
        sid_sizes=meta.vocab.sid_sizes[:meta.vocab.num_levels],
        catalog_codes=sid_codes,
        head_init_std=float(args.head_init_std),
    ).to(device)

    if args.resume:
        _load_head_weights(model, args.resume, logger, strict=not meta.use_lora)
    return model, sid_in, sid_codes


HEAD_FILE = "tiger_head.pt"
HEAD_CONFIG_FILE = "head_config.json"


def _load_head_weights(model: TigerRetriever, ckpt_dir: str,
                       logger: logging.Logger, strict: bool = True) -> None:
    path = os.path.join(ckpt_dir, HEAD_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{ckpt_dir} has no {HEAD_FILE}; it was not written by train_tiger.py"
        )
    state = torch.load(path, map_location="cpu")
    missing, unexpected = model.head.load_state_dict(state, strict=False)
    if strict and (missing or unexpected):
        raise ValueError(
            f"head weights do not match this vocabulary: missing={list(missing)}, "
            f"unexpected={list(unexpected)}"
        )
    logger.info(f"  + TIGER head restored from {path}")


def _save_head(model: TigerRetriever, out_dir: str) -> None:
    torch.save(model.head.state_dict(), os.path.join(out_dir, HEAD_FILE))


def _save_tiger_checkpoint(model: TigerRetriever, meta: RunMeta, out_dir: str,
                           state: Optional[Dict[str, Any]] = None) -> None:
    """Backbone, head and run metadata, swapped into `out_dir` atomically.

    `train_digr.save_checkpoint` cannot be reused: the head is not part of the
    backbone, so `backbone.save_pretrained` would not write it, and a job that dies
    between two separate writes would leave a checkpoint whose head belongs to a
    different step. The staged directory `<dir>.tmp` therefore gets all three pieces
    before the rename, exactly as the DiGR version does.
    """
    tmp = out_dir.rstrip("/" + chr(92)) + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    model.backbone.save_pretrained(tmp, safe_serialization=True)
    meta.save(tmp)
    torch.save(model.head.state_dict(), os.path.join(tmp, HEAD_FILE))
    # `meta.save` has already written this file with the keys `RunMeta.load` reads.
    # Update it rather than replacing it: a fresh dict here dropped `model_name`,
    # `temperature`, `l2_normalize` and `max_seq_len`, which made every TIGER
    # checkpoint unloadable by this file's own evaluator.
    head_path = os.path.join(tmp, HEAD_CONFIG_FILE)
    head_cfg = _read_head_config(tmp)
    head_cfg.update({
        "num_levels": int(model.num_levels),
        "sid_sizes": [int(w) for w in model.sid_sizes],
        "hidden_size": int(model.head.layers[0].in_features),
        "prefix_conditioning": bool(getattr(model, "prefix_conditioning", False)),
    })
    with open(head_path, "w", encoding="utf-8") as fh:
        json.dump(head_cfg, fh, indent=2, sort_keys=True)
    if state is not None:
        torch.save(state, os.path.join(tmp, "train_state.pt"))
    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(tmp, out_dir)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def _make_state(epoch: int, batch: int, batch_size: int, global_step: int,
                best_val: float, optimizer, scheduler, scaler) -> Dict[str, Any]:
    return {
        "epoch": epoch,
        "batch": batch,
        "batch_size": batch_size,
        "global_step": global_step,
        "best_val": best_val,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }


def _write_metrics(run_dir: str, epoch: int, metrics: Dict[str, float]) -> None:
    """Persist one epoch's metrics beside the checkpoints.

    The log line used to be the only record, so anything the formatter dropped was
    gone: `cover@1/10/20/100/1000` and `recall@5` were computed every epoch and
    never stored. `metrics.jsonl` is append-only, `metrics.json` is the whole run
    as a list, rewritten atomically so a crash cannot leave half a file.
    """
    os.makedirs(run_dir, exist_ok=True)
    row = {"epoch": int(epoch),
           **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}}
    with open(os.path.join(run_dir, "metrics.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    path = os.path.join(run_dir, "metrics.json")
    rows: List[Dict[str, Any]] = []
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, list):
                rows = [r for r in loaded if isinstance(r, dict)]
        except (json.JSONDecodeError, OSError):
            rows = []
    rows = [r for r in rows if int(r.get("epoch", -1)) != int(epoch)] + [row]
    rows.sort(key=lambda r: int(r.get("epoch", -1)))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def run_train(args, logger: logging.Logger) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    amp_dtype = dtype_map[args.dtype]
    use_amp = device.type == "cuda" and args.dtype != "fp32"
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    meta = _tiger_meta(args, logger)
    log_banner(logger, "TIGER TRAINING")
    log_card(logger, "runtime configuration", {
        "Device": str(device),
        "Precision": f"{args.dtype} (autocast={use_amp})",
        "Backbone": meta.model_name,
        "Catalog": f"{human_int(meta.catalog.size)} items x {meta.catalog.num_levels} levels",
        "Level widths": str(meta.vocab.sid_sizes[:meta.vocab.num_levels]),
        "Prefix conditioning": bool(args.prefix_conditioning),
        "Vocab": human_int(meta.vocab.total_size),
        "Max history": f"{meta.max_seq_len} events",
        "Batch / eval": f"{args.batch_size} / {args.eval_batch_size}",
        "Epochs / max steps": f"{args.epochs} / {args.max_steps}",
        "Resume": args.resume or "fresh",
        "Data dir": args.data_dir,
    })

    train_split, val_split = _load_splits(args, meta, logger)
    model, sid_in, sid_codes = _build_model(args, meta, device, logger)
    model.prefix_conditioning = bool(args.prefix_conditioning)

    head_params = model.head.parameter_count()
    total_params = sum(p.numel() for p in model.parameters())
    widths = " + ".join(str(l.out_features) for l in model.head.layers)
    logger.info(
        f"  + head: {human_int(head_params)} parameters ({widths} classes over "
        f"{model.head.layers[0].in_features}-d states), "
        f"{head_params / max(1, meta.catalog.size):.4f} per item, "
        f"backbone+head {human_int(total_params)}"
    )

    train_indices = np.arange(len(train_split), dtype=np.int64)
    val_indices = np.arange(len(val_split), dtype=np.int64)
    train_ds = TigerDataset(train_split, train_indices, sid_in, sid_codes,
                            meta.vocab.sep_id, args.prefix_conditioning)
    val_ds = TigerDataset(val_split, val_indices, sid_in, sid_codes,
                          meta.vocab.sep_id, args.prefix_conditioning)
    if len(train_ds) < args.batch_size:
        raise ValueError(
            f"training split has {len(train_ds)} rows but --batch_size={args.batch_size}"
        )

    # Row length grows by a constant, so length bucketing on the history lengths
    # still minimises padding.
    collate = TigerCollate(meta.vocab.sep_id, meta.vocab.num_levels,
                           args.prefix_conditioning)
    sampler = LengthBucketBatchSampler(
        train_split.lengths, args.batch_size, shuffle=True, seed=args.seed,
        bucket_factor=args.bucket_factor, drop_last=True,
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=sampler, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(4 if args.num_workers > 0 else None),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(2 if args.num_workers > 0 else None),
    )

    train_forward = model
    if args.compile:
        logger.info("  > torch.compile(dynamic=True)")
        train_forward = torch.compile(model, dynamic=True)

    optimizer = build_optimizer(model, args)
    steps_per_epoch = max(1, len(train_ds) // args.batch_size)
    total_steps = args.max_steps if args.max_steps else args.epochs * steps_per_epoch
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        return max(float(args.min_lr_ratio), 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = None
    if use_amp and amp_dtype == torch.float16:
        try:
            scaler = torch.amp.GradScaler(device.type, enabled=True)
        except TypeError:  # pragma: no cover
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    start_epoch, start_batch, global_step, best_val = 0, 0, 0, float("inf")
    state_path = os.path.join(args.resume, "train_state.pt") if args.resume else None
    if state_path and os.path.isfile(state_path):
        state = torch.load(state_path, map_location="cpu")
        global_step = int(state.get("global_step", 0))
        best_val = float(state.get("best_val", float("inf")))
        start_epoch, start_batch = resolve_resume_position(state, steps_per_epoch)
        saved_bs = state.get("batch_size")
        if saved_bs is not None and int(saved_bs) != int(args.batch_size):
            logger.warning(
                f"  ! checkpoint was written with --batch_size {int(saved_bs)} but this "
                f"run uses {args.batch_size}; the mid-epoch skip offset is approximate"
            )
        try:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            if scaler is not None and state.get("scaler") is not None:
                scaler.load_state_dict(state["scaler"])
            logger.info(
                f"  + training state restored: epoch={start_epoch}, "
                f"skip {start_batch}/{steps_per_epoch} batches, step={global_step}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"  ! could not restore optimizer/scheduler state: {exc}")
    if start_epoch >= args.epochs:
        logger.warning(
            f"  ! checkpoint already at epoch {start_epoch}/{args.epochs}; nothing to "
            "train (raise --epochs to extend the run)"
        )

    best_dir = os.path.join(args.run_dir, "best")
    final_dir = os.path.join(args.run_dir, "final")
    os.makedirs(args.run_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.run_dir, "tb")) if SummaryWriter else None

    log_banner(logger, "TRAINING LOOP")
    logger.info(
        f"  steps/epoch={steps_per_epoch} total_steps={total_steps} "
        f"warmup={warmup_steps} grad_accum={args.grad_accum}"
    )

    from itertools import islice

    stop = False
    final_saved_epoch = -1
    epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs):
        if stop:
            break
        sampler.set_epoch(epoch)
        active = train_forward
        active.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_t0 = time.time()

        resume_skip = start_batch if epoch == start_epoch else 0
        if resume_skip:
            logger.info(
                f"  > epoch {epoch + 1}: skipping {human_int(resume_skip)} consumption "
                "already done (resume)"
            )
        epoch_iter = islice(train_loader, resume_skip, None) if resume_skip else train_loader
        batches_done = resume_skip
        micro = 0
        running_loss = 0.0
        running_hit = 0.0
        window = 0
        tokens_seen = 0
        pending = False
        pending_micros = 0

        for ids, mask, labels, positions, _targets, _rows in epoch_iter:
            ids = ids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            positions = positions.to(device, non_blocking=True)
            tokens_seen += int(mask.sum().item())
            batches_done += 1

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = active(ids, mask, positions)
                losses = level_cross_entropies(logits, labels,
                                               float(args.label_smoothing))
                loss = sum(losses) / float(len(losses))

            scaled = loss / float(args.grad_accum)
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            micro += 1
            pending = True
            pending_micros += 1
            with torch.no_grad():
                running_loss += float(loss.item())
                running_hit += float(
                    (logits[0].argmax(dim=-1) == labels[:, 0]).float().mean().item()
                )
                window += 1

            if micro % args.grad_accum != 0:
                continue

            if scaler is not None:
                scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.grad_clip
                )
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            pending = False
            pending_micros = 0
            global_step += 1

            if args.save_steps and batches_done % args.save_steps == 0:
                _save_tiger_checkpoint(
                    model, meta, final_dir,
                    _make_state(epoch, batches_done, args.batch_size, global_step,
                                best_val, optimizer, scheduler, scaler),
                )
                logger.info(
                    f"  + mid-epoch checkpoint @ ep {epoch + 1} "
                    f"batch {human_int(batches_done)}/{human_int(steps_per_epoch)} "
                    f"-> {final_dir}"
                )

            if global_step % args.log_interval == 0:
                elapsed = max(1e-6, time.time() - epoch_t0)
                logger.info(
                    f"  ep {epoch + 1}/{args.epochs} | step {global_step} | "
                    f"loss {running_loss / max(1, window):.4f} | "
                    f"L1 acc {running_hit / max(1, window):.4f} | "
                    f"lr {scheduler.get_last_lr()[0]:.2e} | "
                    f"{tokens_seen / elapsed:,.0f} tok/s"
                )
                if writer is not None:
                    writer.add_scalar("train/loss", running_loss / max(1, window), global_step)
                    writer.add_scalar("train/l1_acc", running_hit / max(1, window), global_step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                running_loss = 0.0
                running_hit = 0.0
                window = 0

            if args.max_steps and global_step >= args.max_steps:
                stop = True
                break

        if pending:
            if pending_micros < args.grad_accum:
                tail_scale = float(args.grad_accum) / float(max(1, pending_micros))
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(tail_scale)
            if scaler is not None:
                scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.grad_clip
                )
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        if window:
            logger.info(
                f"  ep {epoch + 1} train tail | loss {running_loss / window:.4f} | "
                f"L1 acc {running_hit / window:.4f}"
            )
        val = evaluate_tiger(model, val_loader, device, amp_dtype, use_amp,
                             catalog_chunk=args.catalog_chunk,
                             rank_items=not bool(args.prefix_conditioning))
        logger.info(
            f"  == epoch {epoch + 1} done in {(time.time() - epoch_t0) / 60:.1f} min | "
            f"val {format_metrics(val, model.num_levels)}"
        )
        _write_metrics(args.run_dir, epoch + 1, val)
        if writer is not None:
            writer.add_scalar("val/loss", val["loss"], epoch + 1)
            for k in BEAM_COVER_KS:
                key = f"cover@{k}"
                if key in val:
                    writer.add_scalar(f"val/cover@{k}", val[key], epoch + 1)
            if val.get("ranked", 1.0):
                for k in EVAL_KS:
                    writer.add_scalar(f"val/recall@{k}", val[f"recall@{k}"], epoch + 1)
                writer.add_scalar("val/mrr", val["mrr"], epoch + 1)

        if val["loss"] < best_val:
            best_val = val["loss"]
            _save_tiger_checkpoint(
                model, meta, best_dir,
                _make_state(epoch, steps_per_epoch, args.batch_size, global_step,
                            best_val, optimizer, scheduler, scaler),
            )
            logger.info(f"  + new best checkpoint -> {best_dir} (val loss {best_val:.4f})")

        if (epoch + 1) % max(1, args.save_interval) == 0 or stop:
            _save_tiger_checkpoint(
                model, meta, final_dir,
                _make_state(epoch, steps_per_epoch, args.batch_size, global_step,
                            best_val, optimizer, scheduler, scaler),
            )
            logger.info(f"  + checkpoint -> {final_dir}")
            final_saved_epoch = epoch

    if final_saved_epoch != epoch or not os.path.isdir(final_dir):
        _save_tiger_checkpoint(
            model, meta, final_dir,
            _make_state(epoch, steps_per_epoch, args.batch_size, global_step,
                        best_val, optimizer, scheduler, scaler),
        )
    if writer is not None:
        writer.flush()
        writer.close()
    log_banner(logger, f"DONE - best val loss {best_val:.4f}")


# ---------------------------------------------------------------------------
# Evaluate / beam
# ---------------------------------------------------------------------------

def _load_for_eval(args, logger: logging.Logger):
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for --mode eval/beam")
    args.resume = args.checkpoint
    meta = _tiger_meta(args, logger)
    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, sid_in, sid_codes = _build_model(args, meta, device, logger)
    model.eval()
    return meta, model, sid_in, sid_codes, device


def _checkpoint_prefix_conditioning(args) -> bool:
    """Whether the checkpoint itself was trained with prefix conditioning.

    The command-line flag is only a fallback, for checkpoints written before
    `head_config.json` existed. Keying the decision off the flag alone meant that
    forgetting it silently scored a prefix-conditioned model with the prefix-free
    rule, which reports a model nobody trained.
    """
    cfg = _read_head_config(getattr(args, "checkpoint", None))
    return bool(cfg.get("prefix_conditioning", args.prefix_conditioning))


def run_eval(args, logger: logging.Logger) -> None:
    meta, model, sid_in, sid_codes, device = _load_for_eval(args, logger)
    # The checkpoint is the authority on which arm it is. Before this, the refusal
    # keyed off the command-line flag, so the flag being forgotten silently scored a
    # prefix-conditioned model with the prefix-free rule.
    if _checkpoint_prefix_conditioning(args):
        # Exhaustive scoring reads every level from the separator state, which is the
        # prefix-free arm's score. A prefix-conditioned head gives a different number
        # at every level after the first, so scoring it this way would report a model
        # that was never trained. Refuse rather than print a plausible-looking row.
        raise SystemExit(
            "--mode eval is exhaustive and therefore prefix-free, but this checkpoint "
            "was trained with prefix conditioning. Its score depends on the path, so "
            "it cannot be computed for every item in one pass. Evaluate it with "
            "--mode beam, and read `cover@b` from the training log as its ceiling."
        )
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    amp_dtype = dtype_map[args.dtype]
    use_amp = device.type == "cuda" and args.dtype != "fp32"

    split = _load_splits(args, meta, logger)[1]
    indices = np.arange(len(split), dtype=np.int64)
    dataset = TigerDataset(split, indices, sid_in, sid_codes, meta.vocab.sep_id,
                           prefix_conditioning=False)
    collate = TigerCollate(meta.vocab.sep_id, meta.vocab.num_levels,
                           prefix_conditioning=False)
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False,
                        collate_fn=collate, num_workers=args.num_workers)
    metrics = evaluate_tiger(model, loader, device, amp_dtype, use_amp,
                             catalog_chunk=args.catalog_chunk)
    log_banner(logger, "TIGER FULL-CATALOG EVALUATION")
    logger.info(f"  data_dir={args.data_dir} checkpoint={args.checkpoint}")
    logger.info(f"  levels={model.num_levels} widths={model.sid_sizes} "
                f"catalog={human_int(model.catalog_size)} rows={human_int(len(dataset))}")
    logger.info(f"  {format_metrics(metrics, model.num_levels)}")
    logger.info(f"  nDCG@100 {metrics['ndcg@100']:.4f} | R@5 {metrics['recall@5']:.4f}")
    cover = " | ".join(f"beam {k} {100.0 * metrics[f'cover@{k}']:.2f}%"
                       for k in BEAM_COVER_KS)
    logger.info(f"  level-0 coverage {cover}")
    if metrics["recall@10"] > 0:
        logger.info(
            f"  ceiling check: exhaustive R@10 {100.0 * metrics['recall@10']:.4f}% "
            f"against a beam of 20 at {100.0 * metrics['cover@20']:.2f}%, so the "
            f"best a beam-20 prefix-conditioned arm can reach is "
            f"{100.0 * metrics['cover@20']:.2f}%"
        )


@torch.inference_mode()
def run_beam(args, logger: logging.Logger) -> None:
    """Prefix-conditioned beam decoding on a val subset, with the prefix tree.

    This is the paper's inference path. It cannot score the whole catalogue, which
    is why `--mode eval` uses the prefix-free head, and its purpose is to measure how
    much the prefix actually buys on this data.

    Inference mode is load-bearing, not an optimisation. `beam_search` calls
    `.numpy()` on the head's output, which refuses a tensor that still requires grad,
    and a loaded model's parameters require grad until something turns it off.
    """
    meta, model, sid_in, sid_codes, device = _load_for_eval(args, logger)
    if not args.prefix_conditioning:
        logger.warning(
            "  ! --mode beam without --prefix_conditioning trains heads that never saw "
            "a prefix; the beam will be scored from a state the head did not learn on"
        )
    split = _load_splits(args, meta, logger)[1]
    rows = len(split) if not args.beam_rows else min(int(args.beam_rows), len(split))
    indices = np.arange(rows, dtype=np.int64)
    dataset = TigerDataset(split, indices, sid_in, sid_codes, meta.vocab.sep_id,
                           prefix_conditioning=False)
    collate = TigerCollate(meta.vocab.sep_id, meta.vocab.num_levels,
                           prefix_conditioning=False)
    # The row must be `history + [SEP]` here, not the training row. With
    # `prefix_conditioning=True` the loader appends the target's own codes, and
    # `_state_for_prefixes` then appends the predicted prefix after them, so level 0
    # read its state from after the true codes and every score was noise. The prefix
    # is built by the beam, so the loader must leave the row bare.
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)

    index = SidIndex(sid_codes, meta.vocab.sid_sizes[:meta.vocab.num_levels])
    logger.info(
        f"  prefix tree over {human_int(meta.catalog.size)} items, "
        f"{index.num_levels} levels, widths {index.sid_sizes}; "
        f"beam {args.beam_width} x {human_int(rows)} rows"
    )
    hit = 0
    total = 0
    cover = 0
    ranks: List[int] = []
    for ids, mask, labels, _positions, _targets, _rows in loader:
        ids = ids.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        # decode one row at a time through the same helper, batched over beam
        for row in range(ids.shape[0]):
            target = tuple(int(c) for c in labels[row].tolist())
            first: Dict[str, np.ndarray] = {}

            def step_fn(prefixes, _row=row, _target=target):
                states = _state_for_prefixes(model, ids[_row:_row + 1],
                                             mask[_row:_row + 1], prefixes, meta, device)
                logits = model.logits_from_state(states)[len(prefixes[0])]
                logp = F.log_softmax(logits.float(), dim=-1).cpu().numpy()
                if not prefixes[0]:
                    first["logp"] = logp[0]
                return logp

            def allowed(prefix, level, _index=index):
                return _index.children(prefix)

            beam = beam_search(step_fn, meta.vocab.sid_sizes[:meta.vocab.num_levels],
                               beam_width=args.beam_width, allowed=allowed)
            total += 1
            # Level-0 cover, the same quantity the training log prints as `cover@k`
            # and the hard cap on what the whole beam can decode. Without it a broken
            # row and a broken beam look identical, because both show a near-zero
            # decode rate, and one of those costs a 53-minute round trip to find.
            if "logp" in first:
                top = np.argsort(-first["logp"])[:args.beam_width]
                cover += int(int(target[0]) in top)
            ordered = [p for _, p in beam]
            if target in ordered:
                hit += 1
                # 1-based rank of the target inside the beam, which is the only
                # ranking this arm can report. Anything past the beam width is a
                # miss by construction, and that miss rate is the beam ceiling.
                ranks.append(ordered.index(target) + 1)
            else:
                ranks.append(args.beam_width + 1)
    ceiling = 100.0 * hit / max(1, total)
    logger.info(
        f"  beam@{args.beam_width}: {hit}/{total} targets decoded ({ceiling:.2f}%)"
    )
    logger.info(
        f"  level-0 cover@{args.beam_width}: {cover}/{total} rows "
        f"({100.0 * cover / max(1, total):.2f}%), which caps targets decoded"
    )
    if ranks:
        # `ranks` holds a 1-based position for every row, and a row whose target
        # never entered the beam is stored as `beam_width + 1`. So every recall
        # below is already end to end: the rows the beam missed count as misses.
        # Multiplying by the ceiling again would count those rows twice.
        arr = np.asarray(ranks, dtype=np.int64)
        # A cut-off above the beam width is unreachable by construction, and a miss
        # sits at `beam_width + 1`, so keep only k the beam can actually satisfy.
        ks = [k for k in EVAL_KS if k <= args.beam_width]
        m = rank_metrics(arr, ks=ks, mrr_k=MRR_K)
        for k in ks:
            logger.info(f"  beam R@{k} {100.0 * m[f'recall@{k}']:.2f}%  "
                        f"nDCG@{k} {m[f'ndcg@{k}']:.5f}")
        logger.info(f"  beam MRR@{MRR_K} {m['mrr']:.5f}")
        logger.info(f"  (beam@{args.beam_width} decoded the target for "
                    f"{ceiling:.2f}% of rows; the R@k above already count the "
                    f"rest as misses)")


def _state_for_prefixes(model: TigerRetriever, ids: torch.Tensor, mask: torch.Tensor,
                        prefixes: Sequence[Tuple[int, ...]], meta: RunMeta,
                        device: torch.device) -> torch.Tensor:
    """Hidden state at the separator for each prefix, one batched forward.

    The row is `history + [SEP] + prefix tokens`, so the state that predicts the next
    code is the last real token of that sequence.
    """
    if ids.shape[0] != 1:
        raise ValueError("expects one row; the caller slices ids[row:row + 1]")
    n = len(prefixes)
    # Crop the trailing padding before appending. The separator is located as
    # `attention_mask.sum(-1) - 1`, which only lands on the last appended token if no
    # padding sits between the row and the appended prefix. Callers pass a batch row of
    # the batch's padded width, so crop to this row's own length.
    n_real = int(mask[0].sum())
    row = ids[:, :n_real].repeat(n, 1)
    row_mask = mask[:, :n_real].repeat(n, 1)
    if prefixes and prefixes[0]:
        offsets = meta.vocab.sid_offsets
        extra = torch.as_tensor(
            [[int(offsets[level]) + int(code) for level, code in enumerate(prefix)]
             for prefix in prefixes],
            dtype=torch.long, device=device,
        )
        row = torch.cat([row, extra], dim=1)
        row_mask = torch.cat(
            [row_mask, torch.ones(n, extra.shape[1], dtype=row_mask.dtype,
                                  device=device)], dim=1
        )
    return model.separator_states(row, row_mask)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TIGER over preprocessed semantic-ID data (v1 and v2 SIDs)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["train", "eval", "beam"], default="train")
    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)

    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--bucket_factor", type=int, default=50)
    p.add_argument("--seed", type=int, default=SEED)

    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--attn_implementation", type=str, default=None)
    p.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--hidden_size", type=int, default=0,
                   help="override the backbone's hidden size for the head")

    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--reinit_embeddings", action=argparse.BooleanOptionalAction,
                   default=True)

    p.add_argument("--prefix_conditioning", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="feed the target's own codes back as input (paper factorisation); "
                        "off keeps every level's logits on one forward so the whole "
                        "catalogue can be scored exactly")
    p.add_argument("--head_init_std", type=float, default=DEFAULT_HEAD_INIT_STD)
    p.add_argument("--catalog_chunk", type=int, default=DEFAULT_CATALOG_CHUNK)
    p.add_argument("--compile", action="store_true")

    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--eval_batch_size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--embed_lr_mult", type=float, default=DEFAULT_EMBED_LR_MULT)
    p.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    p.add_argument("--min_lr_ratio", type=float, default=DEFAULT_MIN_LR_RATIO)
    p.add_argument("--grad_clip", type=float, default=DEFAULT_GRAD_CLIP)
    p.add_argument("--label_smoothing", type=float, default=DEFAULT_LABEL_SMOOTHING)
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--save_interval", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=DEFAULT_SAVE_STEPS)

    p.add_argument("--beam_width", type=int, default=DEFAULT_BEAM_WIDTH)
    p.add_argument("--beam_rows", type=int, default=512,
                   help="val rows to decode in --mode beam (0 = all)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    args.run_name = args.run_name or f"tiger_{stamp}"
    args.run_dir = (args.output_dir if args.mode != "train"
                    else os.path.join(args.output_dir, args.run_name))
    os.makedirs(args.run_dir, exist_ok=True)
    logger = setup_logger(args.run_dir)
    logger.info(f"train_tiger.py | mode={args.mode} | run={args.run_name}")

    if args.mode == "train":
        run_train(args, logger)
    elif args.mode == "eval":
        run_eval(args, logger)
    else:
        run_beam(args, logger)


if __name__ == "__main__":
    main()

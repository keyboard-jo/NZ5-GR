#!/usr/bin/env python3
"""Full-catalog baselines for DiGR. CPU only, no model, no GPU.

Two baselines, both scored over the whole catalog (all items in
`catalog_sid_codes.npy`) against the same val rows and the same targets as
`train_digr.evaluate`, so their numbers are directly comparable to DiGR's.

  sid_transition
      First-order transition counts over SID codes, per level, combined as a
      naive-Bayes product of per-level probabilities. Uses only the codes, so it
      is the honest "generative retrieval without the LLM" arm: it asks whether
      a count table over the same semantic IDs already explains DiGR's score.

          A_l[prev, next] += 1        for consecutive events in each train row
                                      and for (last event -> target)
          score(j) = sum_l w_l * log( A_l[q_l, c_jl] + k )

      The smoothing constant k makes unseen transitions finite, and the
      normaliser log Z_l[q_l] is dropped because it is constant over j and
      cannot change the ranking.

  coclick
      Item-to-item co-occurrence counts from the same train histories, in item
      space, so it never looks at a SID code. This is the classical next-item
      baseline and the one that can embarrass a generative-retrieval approach on
      e-commerce data.

Both use the last event of the history as the query (that is the event DiGR's
final token sees), and `--history_window` lets them aggregate the last K events
with a decay, which is the fairer comparison since DiGR reads the whole history.

Ties. Coarse SID codes cannot separate items in the same bucket, so both
baselines produce many exact ties. They are broken by training popularity
(`log1p(count)` scaled below any real score gap), because an arbitrary
tie-break would make the reported numbers noise. `--no_popularity_tie_break`
disables that.

Example:
    python baselines_digr.py \
        --data_dir /scratch/mscbdtspod2026/jksulaiman/data/digr_data \
        --out_json baselines.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# 100 and 1000 are here because @10 only answers "is the answer on the first
# page".  @100/@1000 answer "could this feed a reranker", which is the question
# that matters for a 4.78M-item catalog.
EVAL_KS = (1, 5, 10, 100, 1000)
KMAX = max(EVAL_KS)
MRR_K = 10          # MRR stays at 10 even though KMAX is 1000, otherwise
#                     recall@1000 runs report MRR@1000 and the number jumps.

DEFAULT_SMOOTHING = 0.5
DEFAULT_HISTORY_DECAY = 0.5
DEFAULT_CHUNK_ROWS = 200_000
DEFAULT_WORKERS = 8

# The tie-break perturbation must stay far below the smallest real score gap.
# Consecutive integer counts differ by log(1 + 1/k) >= 0.4 at k = 0.5, so 1e-6 is
# three orders of magnitude of headroom.
TIE_EPS = 1e-6

# Reference numbers already measured, printed next to the baseline table.
# popularity was re-measured by this script on the val split, same catalog, same
# metric code -- these are the reproducing numbers.  An earlier ad-hoc
# popularity run reported R@10 0.9644%; its R@1 matched to 8 digits but ranks
# 2-10 did not, so that run used a different frequency vector and is retired.
REF_POPULARITY_R1 = 0.00439453125          # 0.4395%
# 1321 hits out of 98,304 rows. Kept as a fraction so the value stays exact; a
# rounded decimal shifts every lift ratio printed from it.
REF_POPULARITY_R10 = 1321 / 98304          # 1.3438%
REF_POPULARITY_R100 = 0.047862             # 4.7862%
REF_POPULARITY_R1000 = 0.147288            # 14.7288%
REF_POPULARITY_R10000 = 0.368215           # 36.8215%
REF_POPULARITY_MRR = 0.00648
# B4 co-click. Window 20 is the bar, not window 1 and not window 5. The sweep
# 5 -> 10 -> 20 kept rising (R@10 3.6011 -> 3.6794 -> 3.7069) in decaying steps
# (+2.2% then +0.7%), so the curve has essentially closed. Window 20 also takes
# rank 1, which window 5 only tied, so DiGR now holds exactly one cut-off: the
# far tail at R@10000.  Any TIGER number is measured against these.
REF_COCLICK = {"R@1": 0.008057, "R@10": 0.037069, "R@100": 0.109873,
               "R@1000": 0.253021, "R@10000": 0.468882, "mrr": 0.01539}
REF_COCLICK_NDCG10 = 0.02044
# Earlier windows, kept so the sweep can be read without rerunning it.
REF_COCLICK_H10 = {"R@1": 0.007985, "R@10": 0.036794, "R@100": 0.108531,
                   "R@1000": 0.248311, "R@10000": 0.462148, "mrr": 0.01529}
REF_COCLICK_H5 = {"R@1": 0.007853, "R@10": 0.036011, "R@100": 0.104492,
                  "R@1000": 0.236176, "R@10000": 0.441905, "mrr": 0.014990}
REF_COCLICK_H1 = {"R@1": 0.005686, "R@10": 0.025238, "R@100": 0.074646,
                  "R@1000": 0.185822, "R@10000": 0.394318, "mrr": 0.01065}
# B3 sid_transition is kept for the record but is off by default: it took 3,805 s
# and scored R@10 0.1801%, 7.5x worse than plain popularity.
REF_SID_TRANSITION_R10 = 0.001801
# DiGR, full catalog, epoch 3 of digr_0p6b_e3_full_sampled300k (the `final`
# checkpoint). Epoch 2 has the lower val loss (11.2342) but worse ranking metrics
# (R@1 0.75%, R@10 2.42%), so `final` is the deployment checkpoint: rank the
# model by R@10, not by val loss.
REF_DIGR = {"loss": 11.2431, "R@1": 0.007924, "R@10": 0.025940,
            "R@100": 0.075094, "R@1000": 0.216003, "R@10000": 0.497691,
            "mrr": 0.01258}
REF_DIGR_NDCG10 = 0.0157

BANNER_WIDTH = 78


# ---------------------------------------------------------------------------
# Logging / small helpers
# ---------------------------------------------------------------------------

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("DiGR-baselines")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def human_int(n: Optional[int]) -> str:
    if n is None:
        return "n/a"
    return f"{int(n):,}"


def banner(logger: logging.Logger, text: str) -> None:
    logger.info("=" * BANNER_WIDTH)
    logger.info(f"  {text}")
    logger.info("=" * BANNER_WIDTH)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_vocab(data_dir: str) -> Dict[str, Any]:
    path = os.path.join(data_dir, "vocab.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"vocab.json not found in {data_dir}")
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    for key in ("sid_sizes", "sid_offsets", "num_levels", "sep_id"):
        if key not in vocab:
            raise KeyError(f"vocab.json is missing '{key}'")
    if len(vocab["sid_sizes"]) != int(vocab["num_levels"]):
        raise ValueError("vocab.json has inconsistent sid_sizes / num_levels")
    if len(vocab["sid_offsets"]) != int(vocab["num_levels"]):
        raise ValueError("vocab.json has inconsistent sid_offsets / num_levels")
    return vocab


def load_catalog_codes(data_dir: str) -> np.ndarray:
    path = os.path.join(data_dir, "catalog_sid_codes.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"catalog_sid_codes.npy not found in {data_dir}")
    codes = np.load(path)
    if codes.ndim != 2:
        raise ValueError("catalog_sid_codes.npy must be [n_items, num_levels]")
    return np.ascontiguousarray(codes, dtype=np.int32)


def load_targets(data_dir: str, name: str) -> np.ndarray:
    path = os.path.join(data_dir, name, "targets.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"split '{name}' has no targets.npy in {data_dir}")
    return np.asarray(np.load(path), dtype=np.int64)


def pack_codes(codes: np.ndarray, sizes: Sequence[int]) -> np.ndarray:
    """Mixed-radix pack of the SID codes into one int64 key.

    With sizes 256,256,256,256,731 the largest key is 256^4 * 731 = 3.1e12, so
    int64 is safe and the pack is injective over the whole catalog.
    """
    if codes.ndim != 2:
        raise ValueError("codes must be [n, levels]")
    if codes.shape[1] != len(sizes):
        raise ValueError(f"codes has {codes.shape[1]} levels, sizes has {len(sizes)}")
    key = np.zeros(codes.shape[0], dtype=np.int64)
    for level, size in enumerate(sizes):
        key = key * int(size) + codes[:, level].astype(np.int64)
    return key


@dataclass
class CodeIndex:
    """Injective map from a full SID tuple to its catalog row."""

    keys: np.ndarray      # sorted keys
    rows: np.ndarray      # catalog row for each sorted key
    sizes: List[int]

    @property
    def size(self) -> int:
        return len(self.rows)

    def lookup(self, codes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Catalog rows for each tuple. Returns (rows, found)."""
        key = pack_codes(codes, self.sizes)
        pos = np.searchsorted(self.keys, key)
        pos_clipped = np.minimum(pos, len(self.keys) - 1)
        found = (pos < len(self.keys)) & (self.keys[pos_clipped] == key)
        return self.rows[pos_clipped], found


def build_code_index(catalog_codes: np.ndarray, sizes: Sequence[int]) -> CodeIndex:
    keys = pack_codes(catalog_codes, sizes)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    if len(sorted_keys) > 1 and not (np.diff(sorted_keys) > 0).all():
        raise ValueError(
            "duplicate full SID tuples in the catalog; the code -> row map would "
            "not be injective (5-code SIDs are expected to be unique)"
        )
    return CodeIndex(keys=sorted_keys, rows=order.astype(np.int64), sizes=list(sizes))


# ---------------------------------------------------------------------------
# Row layout: where the events live inside a tokenized row
# ---------------------------------------------------------------------------

@dataclass
class SplitLayout:
    tokens: np.ndarray        # int32, flat (memmap for train)
    offsets: np.ndarray       # int64 [n_rows + 1]
    feat_len: np.ndarray      # int64 [n_rows], tokens before the SEP
    n_events: np.ndarray      # int64 [n_rows], history events per row
    base0: np.ndarray         # int64 [n_rows], offset of the first event
    num_levels: int
    sep_id: int

    def __len__(self) -> int:
        return len(self.n_events)

    @property
    def total_events(self) -> int:
        return int(self.n_events.sum())


def row_feature_lengths(tokens: np.ndarray, offsets: np.ndarray, sep_id: int) -> np.ndarray:
    """Token count before the SEP for each row.

    Row layout is `[features][SEP][(temporal, SID_l...) * m]` and every code
    token is >= 2 while pad is 0, so the SEP id is unique inside a row and its
    first occurrence is the boundary.
    """
    n_rows = len(offsets) - 1
    if n_rows <= 0:
        return np.zeros(0, dtype=np.int64)
    is_sep = np.asarray(tokens) == int(sep_id)
    pos = np.flatnonzero(is_sep).astype(np.int64)
    if len(pos) != n_rows:
        raise ValueError(
            f"found {human_int(len(pos))} SEP tokens but expected one per row "
            f"({human_int(n_rows)} rows); the split is not in the expected layout"
        )
    return pos - offsets[:-1]


def build_layout(data_dir: str, name: str, vocab: Dict[str, Any],
                 logger: logging.Logger, mmap: bool = True) -> SplitLayout:
    d = os.path.join(data_dir, name)
    for fname in ("tokens.npy", "offsets.npy"):
        if not os.path.isfile(os.path.join(d, fname)):
            raise FileNotFoundError(f"split '{name}' is missing {fname} in {data_dir}")
    tokens = np.load(os.path.join(d, "tokens.npy"),
                     mmap_mode="r" if mmap else None)
    offsets = np.asarray(np.load(os.path.join(d, "offsets.npy")), dtype=np.int64)
    num_levels = int(vocab["num_levels"])
    sep_id = int(vocab["sep_id"])

    feat_len = row_feature_lengths(tokens, offsets, sep_id)
    row_len = np.diff(offsets)
    event_tokens = row_len - 1 - feat_len
    stride = 1 + num_levels
    if (event_tokens % stride).any() or (event_tokens < 0).any():
        raise ValueError(
            "row length is not [features][SEP][(temporal, SID...)] * m; "
            "the split does not match this vocab"
        )
    n_events = event_tokens // stride
    base0 = offsets[:-1] + feat_len + 1
    if n_events.shape != feat_len.shape:
        raise AssertionError("layout arrays disagree in length")
    logger.info(
        f"  + split '{name}': {human_int(len(n_events))} rows, "
        f"{human_int(n_events.sum())} events, "
        f"{n_events.mean():.2f} events/row, "
        f"feature tokens {int(feat_len.min())}-{int(feat_len.max())}"
    )
    return SplitLayout(tokens=tokens, offsets=offsets, feat_len=feat_len,
                       n_events=n_events, base0=base0, num_levels=num_levels,
                       sep_id=sep_id)


@dataclass
class CodeChunk:
    """Codes of a contiguous block of rows, flattened over events."""

    lo: int
    hi: int
    codes: np.ndarray       # [n_events, L] int32, codes without offsets
    local: np.ndarray       # [n_events] int64, index of the event inside its row
    row_local: np.ndarray   # [n_events] int64, row index relative to `lo`
    n_events: np.ndarray    # [hi - lo] int64


def iter_code_chunks(layout: SplitLayout, sid_offsets: Sequence[int],
                     sid_sizes: Sequence[int],
                     chunk_rows: int = DEFAULT_CHUNK_ROWS):
    """Yield CodeChunk for consecutive blocks of rows."""
    n_rows = len(layout)
    level_count = layout.num_levels
    stride = 1 + level_count
    for lo in range(0, n_rows, chunk_rows):
        hi = min(n_rows, lo + chunk_rows)
        m = layout.n_events[lo:hi]
        total = int(m.sum())
        if total == 0:
            continue
        # Contiguous read of just this block keeps the token access sequential.
        o_lo = int(layout.offsets[lo])
        tok = np.asarray(layout.tokens[int(layout.offsets[lo]):int(layout.offsets[hi])])
        bound = np.cumsum(m) - m
        local = np.arange(total, dtype=np.int64) - np.repeat(bound, m)
        row_local = np.repeat(np.arange(hi - lo, dtype=np.int64), m)
        base = (layout.base0[lo:hi] - o_lo)[row_local]
        pos = base + local * stride
        codes = np.empty((total, level_count), dtype=np.int32)
        for level in range(level_count):
            raw = tok[pos + 1 + level].astype(np.int32) - int(sid_offsets[level])
            size = int(sid_sizes[level])
            if raw.size and (raw.min() < 0 or raw.max() >= size):
                raise ValueError(
                    f"level {level}: code out of range [0, {size}) in rows "
                    f"{lo}-{hi} ({int(raw.min())}..{int(raw.max())})"
                )
            codes[:, level] = raw
        yield CodeChunk(lo=lo, hi=hi, codes=codes, local=local,
                        row_local=row_local, n_events=m)


def last_event_codes(layout: SplitLayout, sid_offsets: Sequence[int],
                     history_window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Codes of the last `history_window` events, most recent first.

    Returns (codes [n_rows, K, L] int32, valid [n_rows, K] bool). Rows with fewer
    than K events leave the tail invalid and the scorer skips those entries.
    """
    n_rows = len(layout)
    level_count = layout.num_levels
    stride = 1 + level_count
    window = max(1, int(history_window))
    tok = np.asarray(layout.tokens)
    m = layout.n_events
    codes = np.zeros((n_rows, window, level_count), dtype=np.int32)
    valid = np.zeros((n_rows, window), dtype=bool)
    for t in range(window):
        has = m > t
        if not has.any():
            break
        base = layout.base0[has] + (m[has] - 1 - t) * stride
        for level in range(level_count):
            codes[has, t, level] = (
                tok[base + 1 + level].astype(np.int32) - int(sid_offsets[level])
            )
        valid[has, t] = True
    return codes, valid


# ---------------------------------------------------------------------------
# B3: SID transition counts
# ---------------------------------------------------------------------------

def build_transition_counts(layout: SplitLayout, vocab: Dict[str, Any],
                            catalog_codes: np.ndarray, targets: np.ndarray,
                            logger: logging.Logger,
                            chunk_rows: int = DEFAULT_CHUNK_ROWS,
                            with_anchor_pairs: bool = True
                            ) -> List[np.ndarray]:
    """A_l[prev_code, next_code] over consecutive events, plus (last event -> target)."""
    sizes = [int(s) for s in vocab["sid_sizes"]]
    offsets = [int(o) for o in vocab["sid_offsets"]]
    level_count = layout.num_levels
    tables = [np.zeros((s, s), dtype=np.int64) for s in sizes]
    pairs = 0
    anchor_pairs = 0
    start = time.time()

    for chunk in iter_code_chunks(layout, offsets, sizes, chunk_rows):
        # Within-row consecutive events.
        has_next = (chunk.local + 1) < chunk.n_events[chunk.row_local]
        prev = np.flatnonzero(has_next)
        if prev.size:
            nxt = prev + 1
            for level in range(level_count):
                key = (chunk.codes[prev, level].astype(np.int64) * sizes[level]
                       + chunk.codes[nxt, level])
                tables[level] += np.bincount(key, minlength=sizes[level] ** 2
                                             ).reshape(sizes[level], sizes[level])
            pairs += int(prev.size)
        if with_anchor_pairs:
            # (last event of the row -> the held-out target)
            bound = np.cumsum(chunk.n_events) - chunk.n_events
            last = bound + chunk.n_events - 1
            tgt_codes = catalog_codes[targets[chunk.lo:chunk.hi]]
            for level in range(level_count):
                key = (chunk.codes[last, level].astype(np.int64) * sizes[level]
                       + tgt_codes[:, level].astype(np.int64))
                tables[level] += np.bincount(key, minlength=sizes[level] ** 2
                                             ).reshape(sizes[level], sizes[level])
            anchor_pairs += int(len(last))
        logger.info(
            f"    transition rows {human_int(chunk.hi)}/{human_int(len(layout))}"
            f"  pairs {human_int(pairs)}  anchors {human_int(anchor_pairs)}"
            f"  {time.time() - start:.0f}s"
        )

    for level, table in enumerate(tables):
        density = float((table > 0).sum()) / float(table.size)
        logger.info(
            f"  + level {level}: {human_int(int(table.sum()))} transitions over "
            f"{human_int(table.size)} cells ({density * 100:.1f}% dense), "
            f"mean successor mass {table.sum(axis=1).mean():.1f}"
        )
    return tables


def aggregate_transition_rows(query_codes: np.ndarray, query_valid: np.ndarray,
                              tables: List[np.ndarray], history_decay: float
                              ) -> List[np.ndarray]:
    """agg_l = sum_t decay^t * A_l[q_t, l], one small row per level.

    Aggregating the *counts* before the log keeps the cost independent of the
    history window: the tables are 256 x 256, so this is a few hundred floats.
    """
    aggs: List[np.ndarray] = []
    for level, table in enumerate(tables):
        agg = np.zeros(table.shape[1], dtype=np.float32)
        for t in range(query_codes.shape[0]):
            if not query_valid[t]:
                continue
            agg += np.float32(history_decay ** t) * table[query_codes[t, level]].astype(np.float32)
        aggs.append(agg)
    return aggs


def score_sid_transition(query_codes: np.ndarray, query_valid: np.ndarray,
                         tables: List[np.ndarray], catalog_level_codes: List[np.ndarray],
                         smoothing: float, history_decay: float,
                         weighting: str, tie_break: Optional[np.ndarray]
                         ) -> np.ndarray:
    """Dense score over the whole catalog for one query.

    score(j) = sum_l w_l * log( agg_l[q_l, c_jl] + k )

    The per-query normaliser log Z_l[q_l] is dropped: it is constant over j and
    cannot change the ranking.
    """
    aggs = aggregate_transition_rows(query_codes, query_valid, tables, history_decay)
    n_levels = len(aggs)
    if weighting == "uniform":
        weights = np.ones(n_levels, dtype=np.float32)
    elif weighting == "inv_entropy":
        weights = np.empty(n_levels, dtype=np.float32)
        for level, agg in enumerate(aggs):
            prob = agg.astype(np.float64) + float(smoothing)
            prob /= prob.sum()
            ent = float(-(prob * np.log(np.maximum(prob, 1e-12))).sum())
            weights[level] = 1.0 / max(ent, 1e-6)
    else:
        raise ValueError(f"unknown weighting '{weighting}'")

    n_items = catalog_level_codes[0].shape[0]
    score = np.zeros(n_items, dtype=np.float32)
    tmp = np.empty(n_items, dtype=np.float32)
    for level, agg in enumerate(aggs):
        src = np.log(agg + np.float32(smoothing)).astype(np.float32)
        np.take(src, catalog_level_codes[level], out=tmp)
        score += weights[level] * tmp
    if tie_break is not None:
        score += np.float32(TIE_EPS) * tie_break
    return score


# ---------------------------------------------------------------------------
# B4: co-click kNN
# ---------------------------------------------------------------------------

@dataclass
class CoClickTable:
    indptr: np.ndarray      # int64 [n_items + 1]
    next_item: np.ndarray   # int32 [nnz]
    count: np.ndarray       # int32 [nnz]

    @property
    def nnz(self) -> int:
        return int(self.next_item.shape[0])

    def row(self, item: int) -> Tuple[np.ndarray, np.ndarray]:
        lo, hi = int(self.indptr[item]), int(self.indptr[item + 1])
        return self.next_item[lo:hi], self.count[lo:hi]


def build_coclick(layout: SplitLayout, vocab: Dict[str, Any],
                  catalog_codes: np.ndarray, index: CodeIndex,
                  targets: np.ndarray, n_items: int,
                  logger: logging.Logger, chunk_rows: int = DEFAULT_CHUNK_ROWS,
                  with_anchor_pairs: bool = True) -> CoClickTable:
    """Item -> item counts over consecutive events, plus (last event -> target)."""
    sizes = [int(s) for s in vocab["sid_sizes"]]
    offsets = [int(o) for o in vocab["sid_offsets"]]
    level_count = layout.num_levels
    keys: List[np.ndarray] = []
    start = time.time()
    pairs = 0

    for chunk in iter_code_chunks(layout, offsets, sizes, chunk_rows):
        rows, found = index.lookup(chunk.codes)
        if not found.all():
            raise ValueError(
                f"{human_int(int((~found).sum()))} history events map to a SID tuple "
                "that is not in the catalog; the split and the catalog disagree"
            )
        has_next = (chunk.local + 1) < chunk.n_events[chunk.row_local]
        prev = np.flatnonzero(has_next)
        if prev.size:
            keys.append(rows[prev] * np.int64(n_items) + rows[prev + 1])
            pairs += int(prev.size)
        if with_anchor_pairs:
            bound = np.cumsum(chunk.n_events) - chunk.n_events
            last = bound + chunk.n_events - 1
            keys.append(rows[last] * np.int64(n_items)
                        + targets[chunk.lo:chunk.hi].astype(np.int64))
        logger.info(
            f"    coclick rows {human_int(chunk.hi)}/{human_int(len(layout))}"
            f"  pairs {human_int(pairs)}  {time.time() - start:.0f}s"
        )
    if level_count <= 0:
        raise AssertionError("layout has no SID levels")

    all_keys = np.concatenate(keys) if keys else np.zeros(0, dtype=np.int64)
    del keys
    logger.info(f"  + sorting {human_int(all_keys.size)} raw item pairs ...")
    unique, counts = np.unique(all_keys, return_counts=True)
    del all_keys
    prev_item = (unique // np.int64(n_items)).astype(np.int64)
    next_item = (unique % np.int64(n_items)).astype(np.int32)
    # np.unique returns sorted keys, so prev_item is already non-decreasing.
    indptr = np.searchsorted(prev_item, np.arange(n_items + 1, dtype=np.int64))
    logger.info(
        f"  + coclick: {human_int(len(unique))} distinct pairs, "
        f"{len(unique) / max(1, n_items):.2f} successors per item on average, "
        f"max {int(counts.max()) if counts.size else 0}"
    )
    return CoClickTable(indptr=indptr, next_item=next_item,
                        count=np.ascontiguousarray(counts, dtype=np.int32))


def top_from_coclick(table: CoClickTable, query_items: np.ndarray,
                     query_valid: np.ndarray, history_decay: float,
                     popularity_order: np.ndarray, kmax: Optional[int] = None
                     ) -> np.ndarray:
    """Top-kmax items for one query, count-weighted over the history window."""
    kmax = KMAX if kmax is None else int(kmax)
    cand_parts: List[np.ndarray] = []
    cnt_parts: List[np.ndarray] = []
    for t in range(len(query_items)):
        if query_valid is not None and not query_valid[t]:
            continue
        item = int(query_items[t])
        if item < 0:
            continue
        cand, cnt = table.row(item)
        if cand.size:
            cand_parts.append(cand.astype(np.int64))
            cnt_parts.append(cnt.astype(np.float32) * np.float32(history_decay ** t))
    if cand_parts:
        cand = np.concatenate(cand_parts)
        cnt = np.concatenate(cnt_parts)
        uniq, inv = np.unique(cand, return_inverse=True)
        total = np.bincount(inv, weights=cnt, minlength=len(uniq))
        take = min(kmax, len(uniq))
        # np.argpartition is O(n) and is all we need before the final k-sort.
        part = np.argpartition(-total, take - 1)[:take]
        part = part[np.argsort(-total[part], kind="stable")]
        out = [int(x) for x in uniq[part]]
    else:
        out = []
    if len(out) < kmax:
        # Cold query: fall back to popularity for the empty slots.  Only the
        # prefix that can possibly be needed is materialised -- at kmax=1000
        # `popularity_order.tolist()` would copy all 4.78M entries per query.
        chosen = set(out)
        need = kmax - len(out)
        prefix = popularity_order[:min(popularity_order.shape[0],
                                       len(chosen) + need)]
        for item in prefix.tolist():
            if need <= 0:
                break
            if item not in chosen:
                out.append(int(item))
                chosen.add(int(item))
                need -= 1
    return np.asarray(out[:kmax], dtype=np.int64)


# ---------------------------------------------------------------------------
# Metrics (identical convention to train_digr.evaluate)
# ---------------------------------------------------------------------------

def empty_metrics() -> Dict[str, float]:
    out: Dict[str, float] = {"n": 0.0, "mrr": 0.0}
    for k in EVAL_KS:
        out[f"recall@{k}"] = 0.0
        out[f"ndcg@{k}"] = 0.0
    return out


def ndcg_head(metrics: Dict[str, float]) -> str:
    """`nDCG@{MRR_K}` for the log lines, or a dash when that cut-off was not scored.

    `--ks` decides which cut-offs exist, so a run with `--ks 1,100` has no
    `ndcg@10` and indexing it raised KeyError after the whole evaluation had run.
    """
    value = metrics.get(f"ndcg@{MRR_K}")
    return "n/a" if value is None else f"{value:.5f}"


def accumulate(metrics: Dict[str, float], top: np.ndarray, target: int,
               kmax: Optional[int] = None,
               rank_out: Optional[List[int]] = None) -> None:
    """Add one query. A miss gets rank kmax + 1, never rank 0.

    Rank 0 would pass every `rank <= k` test and turn 1/log2(rank+1) into a
    division by zero, which is the bug that made an earlier DiGR run report
    infinite nDCG.

    `rank_out` collects the exact rank, which the metric reduction throws away. It is
    the only way to answer "was the target inside the first N items" for an N that
    differs per row, which is what `budget_match.py` needs to compare these baselines
    against a beam that names a variable number of items.
    """
    kmax = KMAX if kmax is None else int(kmax)
    hit = np.flatnonzero(top == target)
    rank = int(hit[0]) + 1 if hit.size else kmax + 1
    if rank_out is not None:
        rank_out.append(rank)
    metrics["n"] += 1.0
    for k in EVAL_KS:
        if rank <= k:
            metrics[f"recall@{k}"] += 1.0
            metrics[f"ndcg@{k}"] += 1.0 / math.log2(rank + 1.0)
    metrics["mrr"] += 1.0 / float(rank) if rank <= MRR_K else 0.0


def finalize(metrics: Dict[str, float]) -> Dict[str, float]:
    n = max(1.0, metrics["n"])
    out = {k: (v / n if k != "n" else v) for k, v in metrics.items()}
    out["n"] = int(metrics["n"])
    return out


# ---------------------------------------------------------------------------
# Evaluation drivers
# ---------------------------------------------------------------------------

_G: Dict[str, Any] = {}


def _chunk_range(args: Tuple[int, int]) -> Dict[str, float]:
    lo, hi = args
    g = _G
    metrics = empty_metrics()
    targets = g["targets"]
    dump_dir = g.get("dump_dir")
    rank_out: Optional[List[int]] = [] if dump_dir else None
    for i in range(lo, hi):
        target = int(targets[i])
        if g["kind"] == "popularity":
            top = g["popularity_order"][:KMAX]
        elif g["kind"] == "sid_transition":
            score = score_sid_transition(
                g["query_codes"][i], g["query_valid"][i], g["tables"],
                g["catalog_level_codes"], g["smoothing"], g["history_decay"],
                g["weighting"], g["tie_break"],
            )
            top = np.argpartition(score, -min(KMAX, score.shape[0]))[-KMAX:]
            top = top[np.argsort(-score[top], kind="stable")]
        else:
            top = top_from_coclick(
                g["coclick"], g["query_items"][i], g["query_valid"][i],
                g["history_decay"], g["popularity_order"], kmax=KMAX,
            )
        accumulate(metrics, top, target, kmax=KMAX, rank_out=rank_out)
    if dump_dir and rank_out is not None:
        if len(rank_out) != hi - lo:
            raise AssertionError(
                f"collected {len(rank_out)} ranks for rows [{lo}, {hi})"
            )
        # One file per chunk, named by the chunk's first row. `budget_match.py`
        # concatenates them in name order, so the rank array it builds is in split
        # row order by construction rather than by luck of pool completion order.
        os.makedirs(dump_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(dump_dir, f"{g['kind']}_{lo:06d}.npz"),
            rows=np.arange(lo, hi, dtype=np.int64),
            targets=np.asarray(targets[lo:hi], dtype=np.int64),
            rank=np.asarray(rank_out, dtype=np.int64),
        )
    return metrics


def run_chunks(fn, ranges: List[Tuple[int, int]], workers: int) -> List[Dict[str, float]]:
    workers = max(1, int(workers))
    if workers == 1 or "fork" not in mp.get_all_start_methods():
        if workers > 1:
            print("  ! parallel workers need fork; running serially", flush=True)
        return [fn(r) for r in ranges]
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        return pool.map(fn, ranges, chunksize=1)


def make_ranges(n_rows: int, workers: int) -> List[Tuple[int, int]]:
    if n_rows <= 0:
        return []
    chunks = max(1, min(n_rows, workers * 8))
    step = int(math.ceil(n_rows / chunks))
    return [(lo, min(n_rows, lo + step)) for lo in range(0, n_rows, step)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Full-catalog SID baselines (transition counts, co-click kNN).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", type=str, required=True,
                   help="preprocessed digr_data directory")
    p.add_argument("--eval_split", type=str, default="val")
    p.add_argument("--ks", type=str, default=",".join(str(k) for k in EVAL_KS),
                   help="comma-separated cut-offs. Raising the maximum costs one "
                        "argpartition per query, so 10000 is affordable")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--baselines", type=str,
                   default="popularity,coclick",
                   help="comma separated subset of "
                        "{popularity, sid_transition, coclick}")
    p.add_argument("--history_window", type=int, default=1,
                   help="number of trailing events used as the query")
    p.add_argument("--history_decay", type=float, default=DEFAULT_HISTORY_DECAY,
                   help="decay applied to older events in the window")
    p.add_argument("--smoothing", type=float, default=DEFAULT_SMOOTHING,
                   help="add-k smoothing for unseen SID transitions")
    p.add_argument("--weighting", choices=["uniform", "inv_entropy"], default="uniform",
                   help="how per-level scores are combined")
    p.add_argument("--no_anchor_pairs", action="store_true",
                   help="skip the (last event -> target) pairs when building counts")
    p.add_argument("--no_popularity_tie_break", action="store_true",
                   help="break ties by catalog index instead of train popularity")
    p.add_argument("--max_rows", type=int, default=None,
                   help="evaluate on the first N val rows only")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--chunk_rows", type=int, default=DEFAULT_CHUNK_ROWS)
    p.add_argument("--out_json", type=str, default=None)
    p.add_argument("--dump_ranks", type=str, default=None,
                   help="directory for per-chunk rank files (npz: rows, targets, "
                        "rank). One file per baseline per chunk, so a parallel run "
                        "cannot deliver rows out of order. `budget_match.py` reads "
                        "`<dir>/<baseline>_*.npz`")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    global EVAL_KS, KMAX

    args = build_parser().parse_args(argv)
    logger = setup_logger()

    ks = tuple(sorted({int(x) for x in args.ks.split(",") if x.strip()}))
    if not ks or ks[0] < 1:
        raise SystemExit(f"--ks must be positive integers, got {args.ks!r}")
    EVAL_KS = ks
    KMAX = max(ks)

    if args.history_window < 1:
        raise SystemExit("--history_window must be >= 1")
    if args.smoothing <= 0:
        raise SystemExit("--smoothing must be > 0")
    requested = [b.strip() for b in args.baselines.split(",") if b.strip()]
    unknown = [b for b in requested
               if b not in ("popularity", "sid_transition", "coclick")]
    if unknown:
        raise SystemExit(f"unknown baseline(s): {', '.join(unknown)}")

    banner(logger, "DiGR baselines over the FULL catalog")
    vocab = load_vocab(args.data_dir)
    catalog_codes = load_catalog_codes(args.data_dir)
    n_items = int(catalog_codes.shape[0])
    if catalog_codes.shape[1] != int(vocab["num_levels"]):
        raise SystemExit("catalog_sid_codes.npy disagrees with vocab num_levels")
    sizes = [int(s) for s in vocab["sid_sizes"]]
    for level, size in enumerate(sizes):
        if int(catalog_codes[:, level].max()) >= size:
            raise SystemExit(f"catalog code out of range at level {level}")
    logger.info(f"  + catalog {human_int(n_items)} items x {catalog_codes.shape[1]} levels")
    logger.info(f"  + SID sizes {sizes}")
    logger.info(f"  + cut-offs {list(EVAL_KS)} (topk {KMAX}, MRR@{MRR_K})")

    train_targets = load_targets(args.data_dir, args.train_split)
    val_targets = load_targets(args.data_dir, args.eval_split)
    for name, tg in ((args.train_split, train_targets), (args.eval_split, val_targets)):
        if tg.size and (tg.min() < 0 or tg.max() >= n_items):
            raise SystemExit(
                f"split '{name}' has targets outside [0, {n_items}); "
                "targets.npy must hold catalog row indices, not a label subset remap"
            )
    popularity = np.bincount(train_targets, minlength=n_items).astype(np.float32)
    logger.info(
        f"  + train targets {human_int(len(train_targets))}, "
        f"val targets {human_int(len(val_targets))}, "
        f"distinct train targets {human_int(int((popularity > 0).sum()))}"
    )

    tie_break = None
    if not args.no_popularity_tie_break:
        tie_break = np.log1p(popularity).astype(np.float32)
        logger.info("  + ties broken by train popularity (log1p counts)")

    n_eval = len(val_targets) if args.max_rows is None else min(int(args.max_rows),
                                                               len(val_targets))
    val_layout = build_layout(args.data_dir, args.eval_split, vocab, logger)
    query_codes, query_valid = last_event_codes(
        val_layout, vocab["sid_offsets"], args.history_window)
    query_codes = query_codes[:n_eval]
    query_valid = query_valid[:n_eval]
    del val_layout

    results: Dict[str, Dict[str, float]] = {}
    catalog_level_codes = [np.ascontiguousarray(catalog_codes[:, l]) for l in range(catalog_codes.shape[1])]
    popularity_order = np.argsort(-popularity, kind="stable").astype(np.int64)

    need_counts = "sid_transition" in requested
    need_coclick = "coclick" in requested
    tables: List[np.ndarray] = []
    coclick: Optional[CoClickTable] = None
    index: Optional[CodeIndex] = None

    if need_coclick:
        logger.info("  building the SID -> catalog row index ...")
        index = build_code_index(catalog_codes, sizes)
        logger.info(f"  + index covers {human_int(index.size)} unique tuples")

    if need_counts or need_coclick:
        train_layout = build_layout(args.data_dir, args.train_split, vocab, logger)
        if need_counts:
            logger.info("  B3: counting SID transitions over the train split ...")
            tables = build_transition_counts(
                train_layout, vocab, catalog_codes, train_targets, logger,
                chunk_rows=args.chunk_rows,
                with_anchor_pairs=not args.no_anchor_pairs)
        if need_coclick:
            logger.info("  B4: counting item co-clicks over the train split ...")
            coclick = build_coclick(
                train_layout, vocab, catalog_codes, index, train_targets, n_items,
                logger, chunk_rows=args.chunk_rows,
                with_anchor_pairs=not args.no_anchor_pairs)
        del train_layout

    for kind in requested:
        start = time.time()
        logger.info("")
        if kind == "popularity":
            logger.info(
                f"  scoring popularity over {human_int(n_eval)} rows "
                f"(one global ranking by train target frequency) ..."
            )
            _G.clear()
            _G.update(kind="popularity", targets=val_targets[:n_eval],
                      popularity_order=popularity_order,
                      dump_dir=args.dump_ranks)
            parts = run_chunks(_chunk_range, make_ranges(n_eval, args.workers),
                               args.workers)
        elif kind == "sid_transition":
            logger.info(
                f"  scoring sid_transition over {human_int(n_eval)} rows "
                f"(smoothing {args.smoothing}, window {args.history_window}, "
                f"decay {args.history_decay}, weighting {args.weighting}) ..."
            )
            _G.clear()
            _G.update(kind="sid_transition", targets=val_targets[:n_eval],
                      query_codes=query_codes, query_valid=query_valid,
                      tables=tables, catalog_level_codes=catalog_level_codes,
                      smoothing=float(args.smoothing),
                      history_decay=float(args.history_decay),
                      weighting=args.weighting, tie_break=tie_break,
                      dump_dir=args.dump_ranks)
            parts = run_chunks(_chunk_range, make_ranges(n_eval, args.workers),
                               args.workers)
        else:
            if coclick is None:
                raise SystemExit("internal: coclick table was not built")
            query_items = np.full((n_eval, args.history_window), -1, dtype=np.int64)
            for t in range(args.history_window):
                rows, found = index.lookup(query_codes[:, t, :])
                query_items[:, t] = np.where(found & query_valid[:, t], rows, -1)
            logger.info(
                f"  scoring coclick over {human_int(n_eval)} rows "
                f"(window {args.history_window}, decay {args.history_decay}) ..."
            )
            _G.clear()
            _G.update(kind="coclick", targets=val_targets[:n_eval],
                      coclick=coclick, query_items=query_items,
                      query_valid=query_valid,
                      history_decay=float(args.history_decay),
                      popularity_order=popularity_order,
                      dump_dir=args.dump_ranks)
            parts = run_chunks(_chunk_range, make_ranges(n_eval, args.workers),
                               args.workers)

        merged = empty_metrics()
        for part in parts:
            for key, value in part.items():
                merged[key] += value
        metrics = finalize(merged)
        results[kind] = metrics
        logger.info(
            f"  rows {human_int(int(metrics['n']))}   "
            + "   ".join(f"R@{k} {metrics[f'recall@{k}'] * 100:.4f}%"
                         for k in EVAL_KS)
            + f"   nDCG@10 {ndcg_head(metrics)}"
              f"   MRR {metrics['mrr']:.5f}   ({time.time() - start:.0f}s)"
        )

    logger.info("")
    banner(logger, "summary, full catalog")
    for kind, metrics in results.items():
        logger.info(
            f"  {kind:<16} "
            + "  ".join(f"R@{k} {metrics[f'recall@{k}'] * 100:7.4f}%"
                        for k in EVAL_KS)
            + f"  nDCG@10 {ndcg_head(metrics)}  MRR {metrics['mrr']:.5f}"
        )
    if "popularity" not in results:
        logger.info(
            f"  {'popularity':<16} R@1 {REF_POPULARITY_R1 * 100:7.4f}%  "
            f"{'':>16}  R@10 {REF_POPULARITY_R10 * 100:7.4f}%"
            f"   (recorded earlier, not re-scored)"
        )
    logger.info(
        f"  {'DiGR ckpt':<16} "
        + "  ".join(f"R@{k} {REF_DIGR[f'R@{k}'] * 100:7.4f}%"
                    for k in EVAL_KS if f"R@{k}" in REF_DIGR)
        + f"  nDCG@10 {REF_DIGR_NDCG10:.5f}  MRR {REF_DIGR['mrr']:.5f}"
    )
    for kind, metrics in results.items():
        logger.info(
            f"  DiGR / {kind:<12} "
            + "  ".join(
                f"R@{k} "
                f"{REF_DIGR[f'R@{k}'] / max(1e-12, metrics[f'recall@{k}']):.2f}x"
                for k in EVAL_KS if f"R@{k}" in REF_DIGR
            )
            + "   (< 1.00x means the baseline wins)"
        )
    if "popularity" in results:
        got = results["popularity"]
        drift = abs(got["recall@1"] - REF_POPULARITY_R1) > 1e-9
        logger.info(
            f"  popularity cross-check: recorded R@1 {REF_POPULARITY_R1 * 100:.4f}% "
            f"/ R@10 {REF_POPULARITY_R10 * 100:.4f}%  ->  measured "
            f"{got['recall@1'] * 100:.4f}% / {got['recall@10'] * 100:.4f}%"
        )
        if drift:
            logger.warning(
                "  ! measured popularity differs from the recorded reference; "
                "the val split, the catalog or the train targets changed, so the "
                "DiGR comparison above is not apples to apples"
            )

    if args.out_json:
        payload = {
            "data_dir": os.path.abspath(args.data_dir),
            "eval_split": args.eval_split,
            "n_rows": int(n_eval),
            "catalog_size": n_items,
            "settings": {
                "history_window": args.history_window,
                "history_decay": args.history_decay,
                "smoothing": args.smoothing,
                "weighting": args.weighting,
                "anchor_pairs": not args.no_anchor_pairs,
                "popularity_tie_break": not args.no_popularity_tie_break,
            },
            "baselines": results,
            "reference": {"popularity": {"R@1": REF_POPULARITY_R1,
                                         "R@10": REF_POPULARITY_R10},
                          "digr": REF_DIGR},
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"  wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Region (SID-tuple) bookkeeping and the multiplicity analysis. numpy only.

Why this is a separate module
-----------------------------
`region_rank.py` needs torch to load a checkpoint. The part that can be wrong in
an interesting way -- packing codes into tuples, counting how many items share a
tuple, and turning per-row ranks into the by-bucket-size table -- needs nothing
but numpy, so it lives here and `test_region_lib.py` can hit it in under a second
on a fixture. That split is why `budget_match.py` is trustworthy, and the same
traps apply: a mixed-radix key that overflows, a tuple that silently maps to the
wrong bucket, a group boundary off by one.

The question this answers
-------------------------
DiGR scores an item as `s_i = (1/tau) h . V_i` with `V_i = (1/L) sum_l E_l[c_il]`,
so every item that shares a tuple has **exactly** the same score. 53.51% of the
catalogue shares a 2-code tuple. Measured on the v2 dump, `R@1` is 2.427% on rows
whose target is alone in its tuple and 0.262% otherwise, a 9.3x gap that survives
popularity controls and that co-click, which has no codebook, does not show (1.1x).

That gap can arise two ways and they call for opposite fixes:

  counting   `rank = 1 + #{i : s_i > s_t}` counts ITEMS, so every tuple above the
             target donates its whole size to the count. A target in a crowded
             code space is pushed down by the fat tuples above it. Fix: smaller
             tuples, i.e. a deeper or wider SID.

  scoring    the model simply fails to put the target's region on top when that
             region is crowded. Fix: better item scoring (per-level weights, a
             non-linear composition), not a finer codebook.

The two are separable because the region rank is computable on its own:
`region_rank = 1 + #{regions with a higher score}`, which is exactly the item
rank an *injective* codebook with the same region scores would produce. So the
counterfactual "what if every tuple held one item" costs nothing extra: it is the
same accumulation with the sizes set to 1.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

# int64 is the container, but a radix product past 2**53 is not exactly
# representable as a float and the key is printed in a few places, so refuse
# before the arithmetic quietly stops being exact.
MAX_KEY_PRODUCT = 2 ** 62

# Bucket-size groups. The DiGR v2 distribution over val targets is
# 36.64% g=1, 17.57% g=2, 16.73% g=3-4, 12.66% g=5-8, 7.77% g=9-16, 8.63% g>=17,
# so these edges keep every group above 7% of rows and still resolve the
# dose-response (R@1 falls 2.427 -> 0.590 -> 0.310 -> 0.040 -> 0.039 -> 0.024).
GROUP_EDGES: Tuple[Tuple[int, int], ...] = (
    (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 1 << 30),
)


def pack_keys(codes: np.ndarray, sizes: Sequence[int]) -> np.ndarray:
    """Mixed-radix key per row: `key = key * size_l + code_l`.

    Same map as `baselines_digr.pack_codes` and `train_tiger.pack_prefix`, kept
    local so this module imports no torch. Two different tuples must get two
    different keys, and that is exactly what a radix with the right widths gives.
    """
    codes = np.asarray(codes)
    if codes.ndim != 2:
        raise ValueError(f"codes must be [n, levels], got {codes.shape}")
    if codes.shape[1] != len(sizes):
        raise ValueError(
            f"codes has {codes.shape[1]} levels but {len(sizes)} widths were given"
        )
    prod = 1
    for s in sizes:
        prod *= int(s)
    if prod > MAX_KEY_PRODUCT:
        raise ValueError(
            f"radix product {prod} exceeds {MAX_KEY_PRODUCT}; the key would stop "
            "being exact and two different tuples could collide"
        )
    key = np.zeros(codes.shape[0], dtype=np.int64)
    for level, size in enumerate(sizes):
        col = codes[:, level].astype(np.int64)
        if col.size and (int(col.min()) < 0 or int(col.max()) >= int(size)):
            raise ValueError(
                f"level {level} holds a code outside [0, {int(size)}): "
                f"min {int(col.min())}, max {int(col.max())}. The widths and the "
                "codes come from different codebooks."
            )
        key = key * np.int64(size) + col
    return key


class Regions:
    """The distinct tuples of a SID, with their membership counts.

    `first[i]` is one catalogue row owning tuple `i`, so a per-region vector can
    be fetched by index. `counts[i]` is how many items own it. `region_of_item`
    maps a catalogue row to its region.
    """

    def __init__(self, codes: np.ndarray, sizes: Sequence[int]) -> None:
        codes = np.asarray(codes)
        keys = pack_keys(codes, sizes)
        uniq, first_index, inverse, counts = np.unique(
            keys, return_index=True, return_inverse=True, return_counts=True
        )
        self.keys = uniq
        self.first = first_index.astype(np.int64)
        self.region_of_item = inverse.astype(np.int64)
        self.counts = counts.astype(np.int64)
        self.codes = codes[self.first]
        self.sizes = [int(s) for s in sizes]
        self.n_items = int(codes.shape[0])
        self.n_regions = int(uniq.size)
        if int(self.counts.sum()) != self.n_items:
            raise AssertionError(
                f"region counts sum to {int(self.counts.sum()):,} but the catalogue "
                f"has {self.n_items:,} items"
            )

    @property
    def items_per_region(self) -> float:
        return self.n_items / max(1, self.n_regions)

    @property
    def shared_fraction(self) -> float:
        """Share of ITEMS sitting in a region that holds more than one item."""
        return float(self.counts[self.counts > 1].sum()) / max(1, self.n_items)

    @property
    def injective(self) -> bool:
        return self.n_regions == self.n_items

    def describe(self) -> Dict[str, float]:
        c = self.counts
        return {
            "n_items": float(self.n_items),
            "n_regions": float(self.n_regions),
            "items_per_region": float(self.items_per_region),
            "shared_fraction": float(self.shared_fraction),
            "largest": float(c.max()),
            "mean_size_seen_by_item": float((c.astype(np.float64) ** 2).sum() / self.n_items),
            "median_size_seen_by_item": float(np.median(np.repeat(c, c))),
            "injective": 1.0 if self.injective else 0.0,
            "redundant_bits": float(math.log2(self.n_regions)),
        }


def ranks_with_counts(scores_above_target: np.ndarray,
                      sizes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Turn a per-row count of regions above the target into both ranks.

    `scores_above_target` is the number of regions scoring strictly higher than
    the target's own region. Because every member of a region carries the same
    score, the item count above is not that number but the sum of those regions'
    sizes, which is what `rank = 1 + #{i : s_i > s_t}` really computes. The two
    differ by exactly `sum(size - 1)` over the regions above, and that difference
    is the quantity the whole experiment is about.
    """
    above = np.asarray(scores_above_target)
    sizes = np.asarray(sizes)
    if above.shape != sizes.shape:
        raise ValueError(f"{above.shape} vs {sizes.shape}")
    region_rank = above.astype(np.int64) + 1
    item_rank = above.astype(np.int64)  # replaced by the weighted count by caller
    return region_rank, item_rank


def recall_at(rank: np.ndarray, k: int) -> float:
    return float((np.asarray(rank) <= int(k)).sum()) / max(1, int(np.asarray(rank).size))


def group_label(lo: int, hi: int) -> str:
    return f"{lo}" if lo == hi else (f">={lo}" if hi >= (1 << 30) else f"{lo}-{hi}")


def summarise(item_rank: np.ndarray, region_rank: np.ndarray, g: np.ndarray,
              ks: Iterable[int] = (1, 10, 100, 1000)) -> Dict[str, object]:
    """The by-bucket-size table, plus the overall injective counterfactual.

    `region_rank` is what the SAME region scores would rank if every tuple held
    exactly one item, so `recall_at(region_rank, k)` is the ceiling a finer
    codebook could reach without changing a single score.
    """
    item_rank = np.asarray(item_rank, dtype=np.int64)
    region_rank = np.asarray(region_rank, dtype=np.int64)
    g = np.asarray(g, dtype=np.int64)
    ks = [int(k) for k in ks]
    if not (item_rank.shape == region_rank.shape == g.shape):
        raise ValueError("item_rank, region_rank and g must have the same shape")
    if (g < 1).any():
        raise ValueError("g must be at least 1")
    n = int(item_rank.size)
    if n == 0:
        raise ValueError("no rows")

    rows: List[Dict[str, object]] = []
    for lo, hi in GROUP_EDGES:
        m = (g >= lo) & (g <= hi)
        if not m.any():
            continue
        r = {"group": group_label(lo, hi), "rows": int(m.sum()),
             "share": float(m.sum()) / n,
             "g_mean": float(g[m].mean()),
             "item_rank_mean": float(item_rank[m].mean()),
             "region_rank_mean": float(region_rank[m].mean()),
             "inflation": float((item_rank[m] - region_rank[m]).mean())}
        for k in ks:
            r[f"item_r{k}"] = recall_at(item_rank[m], k)
            r[f"region_r{k}"] = recall_at(region_rank[m], k)
        rows.append(r)

    out: Dict[str, object] = {
        "n_rows": n,
        "ks": ks,
        "g_mean": float(g.mean()),
        "inflated_share": float((item_rank > region_rank).mean()),
        "groups": rows,
    }
    for k in ks:
        out[f"item_r{k}"] = recall_at(item_rank, k)
        out[f"region_r{k}"] = recall_at(region_rank, k)
        out[f"counterfactual_gain_{k}"] = (
            recall_at(region_rank, k) / recall_at(item_rank, k)
            if recall_at(item_rank, k) > 0 else float("nan")
        )
    for lo, hi in GROUP_EDGES:
        m = (g >= lo) & (g <= hi) & (g > 1)
        if m.sum() >= 200:
            out["shared_gap_r1"] = (
                recall_at(item_rank[g == 1], 1) / recall_at(item_rank[m], 1)
                if recall_at(item_rank[m], 1) > 0 else float("nan")
            )
            break
    return out


def verdict(summary: Dict[str, object], tolerance: float = 0.15) -> str:
    """Say which of the two causes the numbers point at, in one sentence.

    The test is the counterfactual gain: if ranking the same regions injectively
    recovers most of the early-k loss, the arithmetic of counting was the problem
    and a finer codebook is the fix. If it recovers almost nothing, the scores
    themselves are wrong on crowded rows and no amount of splitting helps.
    """
    g1 = float(summary.get("counterfactual_gain_1", float("nan")))
    g10 = float(summary.get("counterfactual_gain_10", float("nan")))
    if not (g1 == g1) or not (g10 == g10):
        return "not enough shared-bucket rows to separate the two causes"
    best = max(g1, g10)
    if best < 1.0 + tolerance:
        return (
            f"scoring: an injective codebook with these same region scores would "
            f"gain only {g1:.2f}x at R@1 and {g10:.2f}x at R@10, so the loss on "
            "crowded rows is in the scores, not in the counting. A finer SID will "
            "not fix it."
        )
    if best >= 1.0 + 3 * tolerance:
        return (
            f"counting: the same region scores ranked injectively would gain "
            f"{g1:.2f}x at R@1 and {g10:.2f}x at R@10, so most of the loss is fat "
            "tuples donating their size to every rank. A finer codebook is the fix; "
            "better item scoring will not find it."
        )
    return (
        f"mixed: an injective codebook would gain {g1:.2f}x at R@1 and {g10:.2f}x "
        f"at R@10. Part of the loss is counting and part is scoring, so both moves "
        "are worth something and neither is worth everything."
    )

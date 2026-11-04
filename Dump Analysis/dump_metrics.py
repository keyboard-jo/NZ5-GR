"""Metrics for the retrieval dumps that a single cut-off hides.

Reads the per-row rank arrays that `eval_full_catalog.py --dump_ranks`,
`baselines_digr.py --dump_ranks` and `train_tiger.py --dump_rows` write, and
reports the things a `R@10` column cannot show:

  * the recall curve at every cut-off, plus nDCG@k and MRR@k from the same ranks
  * the effective rank, `2 ** mean(log2(rank))`, which uses the whole list
  * pair-union recall: how much of the query space two arms fail on differently
  * paired significance (McNemar) between every pair of arms, on the same rows
  * popularity strata (head / body / tail), so a content-only claim can be
    checked against "it is just riding popularity"
  * optional per-group breakdown for any per-row grouping (anchor, user, ...)
  * cell statistics when the dump carries TIGER beam tuples

Everything is numpy. No torch, no GPU, no model. A dump and an npy array are
the whole input.

Conventions, all inherited from the evaluators rather than invented here
------------------------------------------------------------------------
`rank` is a 1-based position, and an arm that could not reach the target stores
a sentinel `cap + 1`. So a row is a hit at cut-off `k` when `rank <= k` and
`k <= cap`. Cut-offs wider than `cap` are clamped, because no dump can say
anything about a position past its own cap. This matters for a TIGER beam, whose
`cap` is the beam width: asking a width-20 beam for `R@100` must return its
`R@20`, not 100%.

The cap is `beam_width` when the dump has one, otherwise `rank.max() - 1` when
`rank.max()` looks like a sentinel, otherwise `rank.max()`. Override with
`--cap NAME=VALUE` when you know better.

A miss is charged as a miss. Whether the arm could have reached the item given a
wider list is a different question, and `budget_match.py` answers it.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

EVAL_KS = (1, 5, 10, 100, 1000, 10000)
PAIR_K = 10
BOOTSTRAP = 400
SEED = 20261102
SENTINEL_LOOKING = (1001, 10001, 100001, 1000001)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def resolve(path, name):
    """Turn `--arm name=PATH` into a list of files. A glob, a dir or one file."""
    if glob.has_magic(path):
        files = sorted(glob.glob(path))
        if not files:
            raise SystemExit(f"--arm {name}={path} matched no files")
        return files
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.npz")))
        if not files:
            raise SystemExit(f"--arm {name}={path} holds no .npz")
        prefixes = sorted({os.path.basename(f).rsplit("_", 1)[0] for f in files})
        if len(prefixes) > 1:
            raise SystemExit(
                f"--arm {name}={path} holds more than one baseline ({', '.join(prefixes)}); "
                f"point at one with --arm {name}={os.path.join(path, prefixes[0])}_*.npz"
            )
        return files
    if not os.path.exists(path):
        raise SystemExit(f"--arm {name}={path} does not exist")
    return [path]


def load_arm(files, name):
    """Concatenate chunks, sort by row, and keep the rank/targets arrays aligned."""
    rows, targets, ranks, extra = [], [], [], {}
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            missing = [k for k in ("rows", "targets", "rank") if k not in z.files]
            if missing:
                raise SystemExit(f"{f} is not a rank dump, no {missing[0]}")
            rows.append(np.asarray(z["rows"], dtype=np.int64))
            targets.append(np.asarray(z["targets"], dtype=np.int64))
            ranks.append(np.asarray(z["rank"], dtype=np.int64))
            for key in ("beam_width", "catalog_size", "n_rows", "n_split", "split", "kind"):
                if key in z.files:
                    extra.setdefault(key, z[key])
            if len(files) == 1 and "tuples" in z.files:
                extra["tuples"] = np.asarray(z["tuples"], dtype=np.int64)

    rows = np.concatenate(rows)
    targets = np.concatenate(targets)
    ranks = np.concatenate(ranks)

    order = np.argsort(rows, kind="stable")
    rows, targets, ranks = rows[order], targets[order], ranks[order]
    if "tuples" in extra:
        extra["tuples"] = extra["tuples"][order]

    n = int(rows.size)
    if not np.array_equal(rows, np.arange(n)):
        dup = n - int(np.unique(rows).size)
        raise SystemExit(
            f"--arm {name} does not cover rows 0..{n - 1} exactly once "
            f"({n} rows, {dup} duplicated, min {int(rows.min())}, max {int(rows.max())})"
        )
    return {"name": name, "n": n, "targets": targets, "rank": ranks, **extra}


def infer_cap(arm, override=None):
    if override is not None:
        return int(override)
    if "beam_width" in arm:
        return int(np.asarray(arm["beam_width"]).reshape(-1)[0])
    top = int(arm["rank"].max())
    if top in SENTINEL_LOOKING:
        return top - 1
    return top


def clamp(k, cap):
    """No dump can answer a cut-off past its own cap."""
    return int(min(k, cap))


def pack_keys(codes, sizes):
    """Mixed radix, the same map `budget_match.pack` and `train_tiger.pack_prefix` use."""
    key = np.zeros(len(codes), dtype=np.int64)
    for level in range(codes.shape[1]):
        key = key * int(sizes[level]) + codes[:, level]
    return key


def load_catalogue(data_dir):
    path = os.path.join(data_dir, "catalog_sid_codes.npy")
    if not os.path.exists(path):
        raise SystemExit(f"no catalogue codes for the item axis at {path}")
    codes = np.asarray(np.load(path, mmap_mode="r"), dtype=np.int64)
    return codes, codes.max(axis=0).astype(np.int64) + 1


def as_is_item_rank(tuples, rank, targets, data_dir):
    """Tuple positions to item positions, members of a cell in catalogue order.

    A beam orders cells and cannot order the items inside one, so the members are
    taken in catalogue order, which is the tie-break `topk` already applies inside
    `train_digr.evaluate`. An item's rank is then the summed size of every cell
    ranked above it, plus its own position inside its cell. There is no best case
    and no worst case to choose between: this is the ranking as it stands.

    Returns ``(item_rank, named)`` where ``named`` is how many items the row's
    cells actually hold, i.e. the arm's own candidate list length per row.
    """
    codes, sizes = load_catalogue(data_dir)
    key = pack_keys(codes, sizes)
    order = np.argsort(key, kind="stable")
    uniq, start, counts = np.unique(key[order], return_index=True, return_counts=True)
    within = np.empty(len(codes), dtype=np.int64)
    within[order] = np.arange(len(codes)) - np.repeat(start, counts) + 1

    n, width, levels = tuples.shape
    if levels != sizes.size:
        raise SystemExit(
            f"the dump's tuples have {levels} levels, {data_dir} has {sizes.size}; "
            "these are different SIDs and the item axis would be wrong"
        )
    per_cell = np.zeros((n, width), dtype=np.int64)
    for j in range(width):
        k = np.zeros(n, dtype=np.int64)
        for level in range(levels):
            k = k * int(sizes[level]) + np.clip(tuples[:, j, level], 0, None)
        pos = np.clip(np.searchsorted(uniq, k), 0, uniq.size - 1)
        owned = (tuples[:, j, 0] >= 0) & (uniq[pos] == k)
        per_cell[:, j] = np.where(owned, counts[pos], 0)

    named = per_cell.sum(axis=1)
    cum = np.cumsum(per_cell, axis=1)
    rows = np.arange(n)
    hit = rank <= width
    slot = np.where(hit, rank - 1, 0)
    above = np.concatenate([np.zeros((n, 1), dtype=np.int64), cum[:, :-1]], axis=1)[rows, slot]
    item_rank = np.where(hit, above + within[targets], len(codes) + 1)
    return item_rank, named


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def hits_at(rank, k, cap):
    return rank <= clamp(k, cap)


def recall(rank, k, cap):
    return float(hits_at(rank, k, cap).mean())


def ndcg(rank, k, cap):
    kk = clamp(k, cap)
    gain = np.where(rank <= kk, 1.0 / np.log2(rank + 1.0), 0.0)
    return float(gain.mean())


def mrr(rank, k, cap):
    kk = clamp(k, cap)
    return float(np.where(rank <= kk, 1.0 / rank, 0.0).mean())


def effective_rank(rank, cap):
    """2 ** mean(log2(rank)) over the rows that hit, i.e. the typical hit position.

    Rows stored at the sentinel are left out: their true rank is unknown and
    substituting the sentinel would flatter the arm. `miss_share` says how much
    of the split that leaves out.
    """
    hit = rank[rank <= cap]
    if hit.size == 0:
        return float("nan")
    return float(2.0 ** np.log2(hit.astype(np.float64)).mean())


def miss_share(rank, cap):
    return float((rank > cap).mean())


def union_beats(arms, k, caps):
    """|hit_A u hit_B| / n for every pair. The one metric that shows coverage."""
    n = arms[0]["n"]
    hit = {a["name"]: hits_at(a["rank"], k, caps[a["name"]]) for a in arms}
    out = []
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            a, b = arms[i]["name"], arms[j]["name"]
            union = float((hit[a] | hit[b]).mean())
            either = float(np.mean(hit[a] ^ hit[b]))
            out.append(
                {
                    "a": a,
                    "b": b,
                    "a_only": float((hit[a] & ~hit[b]).mean()),
                    "b_only": float((~hit[a] & hit[b]).mean()),
                    "neither": float((~hit[a] & ~hit[b]).mean()),
                    "union": union,
                    "disagree": either,
                }
            )
    return out


def mcnemar(rank_a, rank_b, k, cap_a, cap_b):
    """Paired test on the same rows. Much sharper than an unpaired proportion test."""
    ha = hits_at(rank_a, k, cap_a)
    hb = hits_at(rank_b, k, cap_b)
    b = int(np.sum(ha & ~hb))
    c = int(np.sum(~ha & hb))
    if b + c == 0:
        return {"b": 0, "c": 0, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / float(b + c)
    chi2 = max(chi2, 0.0)
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return {"b": b, "c": c, "chi2": chi2, "p": p}


def bootstrap_ci(rank, k, cap, n_boot=BOOTSTRAP, seed=SEED):
    """Percentile interval on recall@k. Paired intervals are not attempted here."""
    if n_boot <= 0:
        return (float("nan"), float("nan"))
    hit = hits_at(rank, k, cap).astype(np.float64)
    n = hit.size
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = hit[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return (float(lo), float(hi))


def strata(rank, targets, freq, cap, k, n_bins=5):
    """Recall@k by target popularity. Answers "is this just popularity"."""
    f = freq[targets]
    level = np.log1p(f.astype(np.float64))
    edges = np.quantile(level, np.linspace(0.0, 1.0, n_bins + 1))
    inner = edges[1:-1]
    which = np.searchsorted(inner, level, side="right")
    # ordered cold to hot, so index 0 is the tail of the popularity distribution
    names = ["tail", "lower", "middle", "upper", "head"][:n_bins]
    if n_bins != 5:
        names = [f"q{i + 1}" for i in range(n_bins)]
    hit = hits_at(rank, k, cap)
    out = []
    for i in range(n_bins):
        m = which == i
        if not m.any():
            continue
        out.append(
            {
                "bin": names[i],
                "n": int(m.sum()),
                "share": float(m.mean()),
                "mean_freq": float(f[m].mean()),
                "recall": float(hit[m].mean()),
                "recall_all": float(hit.mean()),
            }
        )
    return out


def per_group(rank, gids, cap, k, top=20):
    hit = hits_at(rank, k, cap)
    out = []
    for g in np.unique(gids)[:top]:
        m = gids == g
        out.append({"group": str(g), "n": int(m.sum()), "recall": float(hit[m].mean())})
    return out


def cell_stats(tuples):
    """Beam tuples are cells of the catalogue, not items. Say what they cover."""
    n, width, levels = tuples.shape
    valid = tuples[:, :, 0] >= 0
    widths = [int(tuples[:, :, l].max()) + 1 for l in range(levels)]
    packed = np.zeros((n, width), dtype=np.int64)
    for l in range(levels):
        packed = packed * widths[l] + np.clip(tuples[:, :, l], 0, None)
    packed = np.where(valid, packed, -1)

    named = valid.sum(axis=1)
    all_cells = np.unique(packed[valid])
    rng = np.random.default_rng(SEED)
    sample = min(2000, n)
    ia = rng.integers(0, n, sample)
    ib = rng.integers(0, n, sample)
    jac = []
    for a, b in zip(ia, ib):
        sa = set(packed[a][packed[a] >= 0].tolist())
        sb = set(packed[b][packed[b] >= 0].tolist())
        if sa or sb:
            jac.append(len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0)
    return {
        "rows": n,
        "beam_width": width,
        "levels": levels,
        "named_per_row_mean": float(named.mean()),
        "named_per_row_min": int(named.min()),
        "named_per_row_max": int(named.max()),
        "distinct_cells": int(all_cells.size),
        "cells_per_row_mean": float(np.mean([len(set(r[r >= 0].tolist())) for r in packed[:2000]])),
        "pairwise_jaccard_mean": float(np.mean(jac)) if jac else float("nan"),
    }


def load_freq(data_dir, n_items):
    """Train-target frequency per catalogue row, the popularity every arm claims not to use."""
    path = os.path.join(data_dir, "train", "targets.npy")
    if not os.path.exists(path):
        raise SystemExit(f"no train targets for the popularity strata at {path}")
    t = np.asarray(np.load(path, mmap_mode="r"), dtype=np.int64)
    freq = np.bincount(t, minlength=n_items).astype(np.int64)
    if freq.size < n_items:
        freq = np.pad(freq, (0, n_items - freq.size))
    return freq[:n_items]


def n_items_of(data_dir, targets):
    ids = os.path.join(data_dir, "catalog_item_ids.npy")
    if os.path.exists(ids):
        return int(np.load(ids, mmap_mode="r").shape[0])
    return int(targets.max()) + 1


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def pct(x, digits=4):
    return "n/a" if x != x else f"{100.0 * x:.{digits}f}%"


def report(arms, caps, ks, pair_k, freq, groups, boot, cands=None):
    n = arms[0]["n"]
    cands = cands or {}
    lines = []
    add = lines.append

    add("# Dump analysis")
    add("")
    add(f"{n:,} rows, {len(arms)} arm(s), same row set and same targets for every arm.")
    add("")
    add("| arm | cap | list | misses | effective rank |")
    add("|---|---|---|---|---|")
    for a in arms:
        cap = caps[a["name"]]
        listed = cands.get(a["name"], float(cap))
        add(
            f"| {a['name']} | {cap:,} | {listed:,.1f} | {pct(miss_share(a['rank'], cap))} | "
            f"{effective_rank(a['rank'], cap):.2f} |"
        )
    add("")
    add("`cap` is the deepest position the dump can express. A miss is stored at `cap + 1`,")
    add("so `effective rank` is computed over hits only and is not flattered by sentinels.")
    add("")
    add("`list` is how many candidates the arm ranked on an average row. For a beam that is")
    add("the summed size of the cells it named, counted from the catalogue; for a")
    add("full-catalogue arm it is the catalogue. A rank dump cannot tell 10,000 from")
    add("4,783,154, so pass `--candidates NAME=N` when the difference matters. A cut-off wider")
    add("than `list` reports the arm's reach, because nothing can be found past the end of a list.")
    if any(a.get("axis") == "item" for a in arms):
        add("")
        add("Arms marked below were converted to **item** ranks: a beam orders cells and")
        add("cannot order the items inside one, so the members are taken in catalogue order,")
        add("which is the tie-break `topk` applies inside `train_digr.evaluate`. No best case,")
        add("no worst case. That conversion is what lets a beam report `R@1` and `nDCG@10` at all.")
    add("")

    add("## The curve")
    add("")
    head = "| arm | " + " | ".join(f"R@{k}" for k in ks) + " |"
    add(head)
    add("|" + "---|" * (len(ks) + 1))
    for a in arms:
        cells = " | ".join(pct(recall(a["rank"], k, caps[a["name"]])) for k in ks)
        add(f"| {a['name']} | {cells} |")
    add("")
    add("### nDCG (top of the list only)")
    add("")
    add("| arm | " + " | ".join(f"nDCG@{k}" for k in ks) + " |")
    add("|" + "---|" * (len(ks) + 1))
    for a in arms:
        cells = " | ".join(f"{ndcg(a['rank'], k, caps[a['name']]):.5f}" for k in ks)
        add(f"| {a['name']} | {cells} |")
    add("")
    add("### MRR")
    add("")
    add("| arm | " + " | ".join(f"MRR@{k}" for k in ks) + " |")
    add("|" + "---|" * (len(ks) + 1))
    for a in arms:
        cells = " | ".join(f"{mrr(a['rank'], k, caps[a['name']]):.5f}" for k in ks)
        add(f"| {a['name']} | {cells} |")
    add("")
    add("`nDCG@k / R@k` is the mean weight of a hit, so it says where inside the top k")
    add("the hits land. 1.0 means every hit is at rank 1.")
    add("")
    add("| arm | nDCG@%d / R@%d |" % (pair_k, pair_k))
    add("|---|---|")
    for a in arms:
        r = recall(a["rank"], pair_k, caps[a["name"]])
        g = ndcg(a["rank"], pair_k, caps[a["name"]])
        add(f"| {a['name']} | {g / r if r else float('nan'):.4f} |")
    add("")

    if boot > 0 and len(arms) == 1:
        lo, hi = bootstrap_ci(arms[0]["rank"], pair_k, caps[arms[0]["name"]], boot)
        add(f"95% bootstrap interval on R@{pair_k} for {arms[0]['name']}: {pct(lo)} to {pct(hi)}")
        add("")

    if len(arms) > 1:
        add(f"## Pair union at k = {pair_k}")
        add("")
        add("The share of rows some arm reaches, which no single-arm number shows.")
        add("")
        add("| pair | A only | B only | union | disagreement |")
        add("|---|---|---|---|---|")
        for row in union_beats(arms, pair_k, caps):
            add(
                f"| {row['a']} vs {row['b']} | {pct(row['a_only'])} | {pct(row['b_only'])} | "
                f"**{pct(row['union'])}** | {pct(row['disagree'])} |"
            )
        add("")
        add(f"## Paired significance at k = {pair_k} (McNemar)")
        add("")
        add("Both arms see the same rows, so this is the right test. `p` is a chi-square")
        add("with one degree of freedom and a continuity correction.")
        add("")
        add("| pair | A hit B miss | A miss B hit | chi2 | p |")
        add("|---|---|---|---|---|")
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                a, b = arms[i], arms[j]
                s = mcnemar(a["rank"], b["rank"], pair_k, caps[a["name"]], caps[b["name"]])
                add(
                    f"| {a['name']} vs {b['name']} | {s['b']:,} | {s['c']:,} | "
                    f"{s['chi2']:.3f} | {s['p']:.3g} |"
                )
        add("")

    if freq is not None:
        add(f"## Recall@{pair_k} by target popularity")
        add("")
        add("Quintiles over `log1p(train target count)`, ordered cold to hot. A content-only")
        add("arm should hold up toward the tail; a popularity rider collapses there.")
        add("")
        add("| arm | " + " | ".join(s["bin"] for s in strata(arms[0]["rank"], arms[0]["targets"], freq, caps[arms[0]["name"]], pair_k)) + " |")
        add("|" + "---|" * 6)
        nrow = strata(arms[0]["rank"], arms[0]["targets"], freq, caps[arms[0]["name"]], pair_k)
        add("| rows | " + " | ".join(f"{s['n']:,}" for s in nrow) + " |")
        add("| mean freq | " + " | ".join(f"{s['mean_freq']:.1f}" for s in nrow) + " |")
        for a in arms:
            s = strata(a["rank"], a["targets"], freq, caps[a["name"]], pair_k)
            add(f"| {a['name']} | " + " | ".join(pct(x["recall"]) for x in s) + " |")
        add("")

    if groups is not None:
        add(f"## Recall@{pair_k} by group")
        add("")
        add("| arm | " + " | ".join(str(g["group"]) for g in per_group(arms[0]["rank"], groups, caps[arms[0]["name"]], pair_k)) + " |")
        add("|" + "---|" * 12)
        g0 = per_group(arms[0]["rank"], groups, caps[arms[0]["name"]], pair_k)
        add("| rows | " + " | ".join(f"{g['n']:,}" for g in g0) + " |")
        for a in arms:
            gs = per_group(a["rank"], groups, caps[a["name"]], pair_k)
            add(f"| {a['name']} | " + " | ".join(pct(g["recall"]) for g in gs) + " |")
        add("")

    for a in arms:
        if "tuples" not in a:
            continue
        s = cell_stats(a["tuples"])
        add(f"## Cells named by {a['name']}")
        add("")
        add("These are cell counts, kept because the cell structure is what makes a beam")
        add("different from an item ranking. The metrics above are on the item axis.")
        add("")
        add("| quantity | value |")
        add("|---|---|")
        add(f"| beam width | {s['beam_width']} |")
        add(f"| levels | {s['levels']} |")
        add(f"| cells named per row | {s['named_per_row_mean']:.1f} (min {s['named_per_row_min']}, max {s['named_per_row_max']}) |")
        add(f"| distinct cells per row | {s['cells_per_row_mean']:.1f} |")
        add(f"| distinct cells overall | {s['distinct_cells']:,} |")
        add(f"| mean pairwise Jaccard between two rows | {s['pairwise_jaccard_mean']:.3f} |")
        add("")
        add("Item-level catalogue coverage is not recoverable from a cell dump. It needs the")
        add("bucket sizes, which `budget_match.py` reads from `catalog_sid_codes.npy`.")
        add("")

    add("## Not reported, and why")
    add("")
    add("- precision@k, MAP, HR@k: with one relevant item per row they are R@k relabelled.")
    add("- diversity and serendipity: they need item content, which this project excludes.")
    add("- per-row costs: a rank dump does not record how many items were scored.")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_arm(raw):
    if "=" not in raw:
        raise SystemExit(f"--arm wants NAME=PATH, got {raw!r}")
    name, path = raw.split("=", 1)
    return name.strip(), path.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Metrics over the retrieval rank dumps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=PATH",
                    help="a rank dump: one npz, a glob, or a directory of chunks")
    ap.add_argument("--cap", action="append", default=[], metavar="NAME=N",
                    help="override the inferred cap for one arm")
    ap.add_argument("--candidates", action="append", default=[], metavar="NAME=N",
                    help="override how many candidates one arm ranked per row; a rank "
                         "dump cannot tell a 10,000-item list from the whole catalogue")
    ap.add_argument("--item_axis", choices=("auto", "off"), default="auto",
                    help="convert a beam's tuple ranks to item ranks using the catalogue "
                         "(default auto: needs --data_dir); off keeps the tuple axis")
    ap.add_argument("--ks", default=",".join(str(k) for k in EVAL_KS),
                    help="cut-offs to report, default %(default)s")
    ap.add_argument("--pair_k", type=int, default=PAIR_K)
    ap.add_argument("--data_dir", default=None,
                    help="popularity strata (train/targets.npy) and, with it, the "
                         "catalogue the item-axis conversion needs")
    ap.add_argument("--groups", default=None,
                    help="npy of per-row group ids, one per row of the dump")
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP,
                    help="bootstrap draws for a single-arm interval, 0 to skip")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args(argv)

    if not args.arm:
        raise SystemExit("--arm is required, e.g. --arm DiGR=dumps/digr_val.npz")

    ks = sorted({int(x) for x in args.ks.split(",") if x.strip()})
    overrides = {}
    for raw in args.cap:
        name, val = raw.split("=", 1)
        overrides[name.strip()] = int(val)

    arms = []
    for raw in args.arm:
        name, path = parse_arm(raw)
        arms.append(load_arm(resolve(path, name), name))

    n = arms[0]["n"]
    for a in arms[1:]:
        if a["n"] != n:
            raise SystemExit(f"--arm {a['name']} has {a['n']} rows, {arms[0]['name']} has {n}")
        if not np.array_equal(a["targets"], arms[0]["targets"]):
            bad = int((a["targets"] != arms[0]["targets"]).sum())
            raise SystemExit(
                f"--arm {a['name']} disagrees with {arms[0]['name']} on {bad} targets; "
                "the rows are not the same rows"
            )

    caps = {}
    for a in arms:
        caps[a["name"]] = infer_cap(a, overrides.get(a["name"]))

    cands = {}
    for raw in args.candidates:
        name, val = raw.split("=", 1)
        cands[name.strip()] = float(val)

    if args.item_axis == "auto":
        for a in arms:
            if "tuples" not in a:
                continue
            if not args.data_dir:
                a["axis"] = "tuple"
                continue
            beam_width = int(np.asarray(a["tuples"]).shape[1])
            item_rank, named = as_is_item_rank(
                np.asarray(a["tuples"], dtype=np.int64), a["rank"], a["targets"], args.data_dir
            )
            n_items = int(np.load(os.path.join(args.data_dir, "catalog_sid_codes.npy"),
                                  mmap_mode="r").shape[0])
            a["tuple_rank"], a["rank"] = a["rank"], item_rank
            a["axis"] = "item"
            caps[a["name"]] = n_items
            cands.setdefault(a["name"], float(named.mean()))
            print(
                f"note: {a['name']} converted from {beam_width} cell positions to item "
                f"ranks over {named.mean():,.1f} named items per row",
                file=sys.stderr,
            )
    for k in ks:
        for a in arms:
            if k > caps[a["name"]]:
                print(
                    f"note: R@{k} for {a['name']} is clamped to its cap {caps[a['name']]:,}",
                    file=sys.stderr,
                )

    freq = None
    if args.data_dir:
        n_items = n_items_of(args.data_dir, arms[0]["targets"])
        freq = load_freq(args.data_dir, n_items)

    groups = None
    if args.groups:
        groups = np.asarray(np.load(args.groups), dtype=np.int64)
        if groups.size != n:
            raise SystemExit(f"--groups has {groups.size} entries, the dump has {n}")

    text = report(arms, caps, ks, args.pair_k, freq, groups, args.bootstrap, cands)

    os.makedirs(args.out_dir, exist_ok=True)
    md = os.path.join(args.out_dir, f"dump_metrics{args.tag}.md")
    with open(md, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(text)
    print(f"wrote {md}")

    payload = {
        "n_rows": n,
        "caps": caps,
        "ks": ks,
        "pair_k": args.pair_k,
        "arms": {
            a["name"]: {
                "recall": {str(k): recall(a["rank"], k, caps[a["name"]]) for k in ks},
                "ndcg": {str(k): ndcg(a["rank"], k, caps[a["name"]]) for k in ks},
                "mrr": {str(k): mrr(a["rank"], k, caps[a["name"]]) for k in ks},
                "effective_rank": effective_rank(a["rank"], caps[a["name"]]),
                "miss_share": miss_share(a["rank"], caps[a["name"]]),
                "axis": a.get("axis", "item") if "tuples" not in a else a.get("axis", "tuple"),
                "candidates": cands.get(a["name"], float(caps[a["name"]])),
                **({"tuple_rank": {str(k): recall(a["tuple_rank"], k, int(np.asarray(a["tuples"]).shape[1]))
                                    for k in ks}} if "tuple_rank" in a else {}),
            }
            for a in arms
        },
    }
    if len(arms) > 1:
        payload["union"] = union_beats(arms, args.pair_k, caps)
        payload["mcnemar"] = [
            {
                "a": arms[i]["name"],
                "b": arms[j]["name"],
                **mcnemar(arms[i]["rank"], arms[j]["rank"], args.pair_k,
                          caps[arms[i]["name"]], caps[arms[j]["name"]]),
            }
            for i in range(len(arms))
            for j in range(i + 1, len(arms))
        ]
    if freq is not None:
        payload["strata"] = {a["name"]: strata(a["rank"], a["targets"], freq,
                                              caps[a["name"]], args.pair_k)
                             for a in arms}

    out_json = args.out_json or os.path.join(args.out_dir, f"dump_metrics{args.tag}.json")
    with open(out_json, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

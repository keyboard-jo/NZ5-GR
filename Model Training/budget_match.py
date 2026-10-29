#!/usr/bin/env python
"""Compare every arm at the candidate budget TIGER's beam actually named per row.

Why this exists
---------------
A beam of 20 over a 2-level SID names 20 two-code tuples. On this catalogue 53.51%
of items share a tuple with another item and the largest group is 315, so those 20
tuples name roughly 30 items, and the exact count differs on every row. DiGR ranks
all 4,783,154 items, so its R@10 answers "is the target in the top 10 of 4.78M" while
the beam answers "is the target inside the ~30 items I chose to name". The two are
not the same quantity, and interpolating DiGR's curve to k=30 hides that behind a
modelling assumption.

This script removes the assumption. It pairs each row's shortlist size `N_r` with
that same row's rank in every other arm, so the question becomes identical for all of
them: "is the target inside the N_r items this arm would have named".

    matched_reach = mean(rank_r <= N_r)

`N_r` is the exact number of catalogue items carried by the 20 tuples the beam kept
on row `r`, counted with `np.unique` over `catalog_sid_codes.npy`. Everything here is
a count. Nothing is fitted, interpolated, or extrapolated.

The one thing this cannot do is give the beam more than 20 tuples: on a row where the
beam missed, the target's rank is unknown beyond "worse than 20", so TIGER's reach is
always `mean(rank <= beam_width)` and never `mean(rank <= N_r)`. Scoring a miss as a
hit because `N_r >= 21 >= the miss marker` is the trap this file exists to avoid; it
would print 100%.

Inputs
------
`--beam`  the npz `train_tiger.py --mode beam --dump_rows` writes: `rows`, `rank`,
          `targets`, `tuples` [n, beam_width, levels] padded with -1, `beam_width`.
`--arm`   `NAME=PATH`, repeated. PATH is one npz (`eval_full_catalog.py
          --dump_ranks`) or a directory of per-chunk npz files (`baselines_digr.py
          --dump_ranks DIR`, which writes `<baseline>_<lo:06d>.npz`). The rank file
          supplies `rows`, `targets`, `rank`; the arm's NAME is also the chunk-file
          prefix when PATH is a directory.
`--data_dir` the directory holding `vocab.json` and `catalog_sid_codes.npy`. It must
          be the SAME SID the beam used (2x8192), because the whole point is to count
          that SID's tuples. It is also read for `train/targets.npy` and
          `val/targets.npy` when the ever-clicked-only view is wanted.

Refusals, on purpose
--------------------
* Any arm whose `targets` array differs from the beam's, positionally. That is the
  only cheap proof that two runs scored the same rows in the same order.
* Any tuple the beam named that the catalogue does not own.
* Any row set that is not the full split, or a length that disagrees.
* Any rank of -1 (a row that was never filled).

Usage on the cluster (after the three dumps exist)
--------------------------------------------------
    python budget_match.py \
        --data_dir "$HOME/NZ5/plan2/digr_data" \
        --beam  dumps/beam_v2_prefix.npz \
        --arm   DiGR=dumps/digr_ranks.npz \
        --arm   coclick=dumps/baselines \
        --arm   popularity=dumps/baselines \
        --out_dir dumps
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np


REF_USED = "python budget_match.py"

# Published values, so a dump that has drifted is visible instead of silent. Compared
# at every own-cut-off, warning on a difference larger than `DRIFT_PP`.
DRIFT_PP = 0.02
PUBLISHED = {
    "DiGR": {"@1": 0.010549, "@10": 0.036570, "@100": 0.109182},
    "coclick": {"@1": 0.008057, "@10": 0.037069, "@100": 0.109873},
    "popularity": {"@1": 0.004395, "@10": 0.013438, "@100": 0.047862},
    "TIGER": {"decoded": 12367.0 / 98304.0},
}


# ---------------------------------------------------------------------------
# Catalog geometry
# ---------------------------------------------------------------------------

def load_vocab(data_dir: str) -> Dict[str, object]:
    path = os.path.join(data_dir, "vocab.json")
    if not os.path.isfile(path):
        raise SystemExit(f"{data_dir} has no vocab.json, so it is not a dataset directory")
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    for key in ("sid_sizes", "num_levels"):
        if key not in vocab:
            raise SystemExit(f"vocab.json in {data_dir} is missing '{key}'")
    if len(vocab["sid_sizes"]) != int(vocab["num_levels"]):
        raise SystemExit("vocab.json has inconsistent sid_sizes / num_levels")
    return vocab


def load_catalog_codes(data_dir: str) -> np.ndarray:
    path = os.path.join(data_dir, "catalog_sid_codes.npy")
    if not os.path.isfile(path):
        raise SystemExit(f"{data_dir} has no catalog_sid_codes.npy")
    codes = np.load(path)
    if codes.ndim != 2:
        raise SystemExit(f"catalog_sid_codes.npy is {codes.shape}, expected [n, levels]")
    return np.ascontiguousarray(codes, dtype=np.int64)


def pack(codes: np.ndarray, sizes: List[int]) -> np.ndarray:
    """Mixed-radix key, `key = c0 * r1 * r2 + c1 * r2 + c2 ...`.

    Same map as `baselines_digr.pack_codes` and `train_tiger.pack_prefix`; kept local
    so this file needs no torch and no module with global state. Only equality is
    used, so any injective pack would do.
    """
    codes = np.asarray(codes, dtype=np.int64)
    if codes.ndim != 2:
        raise ValueError(f"expected [n, levels] codes, got {codes.shape}")
    if codes.shape[1] != len(sizes):
        raise ValueError(f"{codes.shape[1]} code levels against {len(sizes)} widths")
    if (codes < 0).any():
        raise ValueError("negative code: -1 padding must be removed before packing")
    key = np.zeros(codes.shape[0], dtype=np.int64)
    for level, size in enumerate(sizes):
        if (codes[:, level] >= int(size)).any():
            raise ValueError(f"code out of range at level {level} for width {size}")
        key = key * int(size) + codes[:, level]
    return key


class Buckets:
    """Item multiplicity per SID tuple, looked up by packed key."""

    def __init__(self, codes: np.ndarray, sizes: List[int]):
        self.sizes = [int(s) for s in sizes]
        self.key_of_item = pack(codes, self.sizes)
        self.uniq, self.counts = np.unique(self.key_of_item, return_counts=True)
        # Position of each catalog item's key, so a per-item boolean (ever clicked)
        # becomes a per-tuple count.
        self.item_key_pos = np.searchsorted(self.uniq, self.key_of_item)

    def __len__(self) -> int:
        return int(self.uniq.size)

    def lookup(self, keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Multiplicities for `keys`, and whether each key is catalogue-owned."""
        pos = np.searchsorted(self.uniq, keys)
        pos_c = np.clip(pos, 0, max(0, self.uniq.size - 1))
        found = (pos < self.uniq.size) & (self.uniq[pos_c] == keys)
        mult = np.where(found, self.counts[pos_c], 0).astype(np.int64)
        return mult, found

    def counts_over(self, mask: np.ndarray) -> np.ndarray:
        """Multiplicity per distinct key, counting only catalog items in `mask`."""
        return np.bincount(self.item_key_pos[mask], minlength=self.uniq.size)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def _as_int(a) -> np.ndarray:
    return np.asarray(a).astype(np.int64, copy=False)


class Arm:
    def __init__(self, name: str, rows, targets, rank):
        self.name = name
        self.rows = _as_int(rows)
        self.targets = _as_int(targets)
        self.rank = _as_int(rank)


def load_rank_file(path: str, name: str) -> Arm:
    with np.load(path, allow_pickle=False) as z:
        for key in ("rows", "targets", "rank"):
            if key not in z:
                raise SystemExit(f"{path} has no '{key}'; it is not a rank dump")
        return Arm(name, z["rows"], z["targets"], z["rank"])


def load_arm(path: str, name: str, n_rows: int) -> Arm:
    """One npz, a glob of per-chunk npz, or a directory of them for one baseline.

    The glob is the form the slurm scripts actually produce: `run_baselines.sh` writes
    every kind into the same window directory, so `baselines/h20/` holds 64
    `coclick_*.npz` beside 64 `popularity_*.npz` and the pattern is what keeps them
    apart. A directory is also accepted, but only when it holds a single baseline, so a
    mixed one is refused instead of silently merged.
    """
    if glob.has_magic(path):
        files = sorted(glob.glob(path))
        if not files:
            raise SystemExit(f"--arm {name}={path} matched no files")
        return _stack_chunks(files, name)
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.npz")))
        if not files:
            raise SystemExit(f"{path} holds no *.npz chunk files")
        prefixes = sorted({os.path.basename(f).split("_")[0] for f in files})
        if len(prefixes) > 1:
            raise SystemExit(
                f"{path} holds more than one baseline ({', '.join(prefixes)}), so one "
                "`--arm` would mix them. Pass a glob instead, e.g. "
                f"`--arm {name}={os.path.join(path, prefixes[0])}_*.npz`."
            )
        return _stack_chunks(files, name)
    if not os.path.isfile(path):
        raise SystemExit(f"--arm {name}={path} does not exist")
    return load_rank_file(path, name)


def _stack_chunks(files: List[str], name: str) -> Arm:
    rows = np.concatenate([_as_int(np.load(f)["rows"]) for f in files])
    targets = np.concatenate([_as_int(np.load(f)["targets"]) for f in files])
    rank = np.concatenate([_as_int(np.load(f)["rank"]) for f in files])
    order = np.argsort(rows, kind="stable")
    return Arm(name, rows[order], targets[order], rank[order])


def scatter(arm: Arm, n_rows: int, label: str) -> Tuple[np.ndarray, np.ndarray]:
    """Row-indexed arrays, verifying every row was written exactly once."""
    for field in ("rows", "targets", "rank"):
        if getattr(arm, field).shape != arm.rows.shape:
            raise SystemExit(
                f"{label}: '{field}' has shape {getattr(arm, field).shape} but "
                f"'rows' has {arm.rows.shape}"
            )
    if arm.rows.size == 0:
        raise SystemExit(f"{label}: the dump is empty")
    if int(arm.rows.min()) < 0 or int(arm.rows.max()) >= n_rows:
        raise SystemExit(
            f"{label}: row indices span [{int(arm.rows.min())}, {int(arm.rows.max())}] "
            f"but the split has {n_rows:,} rows"
        )
    if np.unique(arm.rows).size != arm.rows.size:
        raise SystemExit(f"{label}: a row index appears twice, so chunks overlap")
    targets = np.full(n_rows, -1, dtype=np.int64)
    rank = np.full(n_rows, -1, dtype=np.int64)
    targets[arm.rows] = arm.targets
    rank[arm.rows] = arm.rank
    missing = int((rank < 0).sum())
    if missing:
        raise SystemExit(
            f"{label}: {missing:,} of {n_rows:,} rows have no rank. The dump is "
            "partial (--max_rows / --beam_rows?), so it cannot be paired row by row "
            "with the others."
        )
    if (targets < 0).any():
        raise SystemExit(f"{label}: a row carries no target")
    return targets, rank


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_arm(text: str) -> Tuple[str, str]:
    if "=" not in text:
        raise SystemExit(f"--arm wants NAME=PATH, got {text!r}")
    name, path = text.split("=", 1)
    if not name.strip() or not path.strip():
        raise SystemExit(f"--arm wants NAME=PATH, got {text!r}")
    return name.strip(), path.strip()


def own_metric(rank: np.ndarray, k: int) -> float:
    return float((rank <= k).mean())


def ndcg_matched(rank: np.ndarray, budget: np.ndarray) -> float:
    """nDCG with the cut-off replaced by the row's own budget.

    One target per row, so IDCG is 1 at every cut-off and the metric is just the
    weighted hit rate.  The budget only moves the indicator threshold; nothing is
    divided by it.  This is the rank-aware twin of `match_all` and is only defined
    on an item axis, so it is NaN for TIGER, whose positions are tuple positions.
    """
    r = np.asarray(rank, dtype=np.float64)
    gain = np.where(r <= budget, 1.0 / np.log2(np.maximum(r, 1.0) + 1.0), 0.0)
    return float(gain.mean())


def mrr_at(rank: np.ndarray, k: int) -> float:
    """MRR truncated at a fixed k.  No budget involved, so this one is comparable
    for any arm that ranks items, and NaN for an arm that ranks tuples."""
    r = np.asarray(rank, dtype=np.float64)
    return float(np.where(r <= k, 1.0 / np.maximum(r, 1.0), 0.0).mean())


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Compare arms at the candidate budget each row's beam actually named",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--beam", required=True, help="npz from train_tiger.py --dump_rows")
    p.add_argument("--arm", action="append", default=[], metavar="NAME=PATH",
                   help="a rank dump (npz file, or dir of per-chunk npz). Repeatable")
    p.add_argument("--data_dir", required=True,
                   help="directory with vocab.json + catalog_sid_codes.npy for the "
                        "SID the beam used (2x8192), e.g. $HOME/NZ5/plan2/digr_data")
    p.add_argument("--split", default="val", help="split name, for the targets file")
    p.add_argument("--seen_items", default=None,
                   help="npy/npz of the catalogue row indices that were ever clicked "
                        "(npz key 'items'). Defaults to the train+val target set, "
                        "which is 10.5%% of the catalogue against 33%% for the "
                        "clicked set, so the default budget is the tighter one")
    p.add_argument("--out_dir", default=None, help="where to write the table")
    p.add_argument("--tag", default="", help="suffix for the output filenames")
    p.add_argument("--strict", action="store_true",
                   help="turn drift warnings into failures")
    cli = p.parse_args(argv)

    sizes_vocab = load_vocab(cli.data_dir)
    sizes = [int(s) for s in sizes_vocab["sid_sizes"]]      # type: ignore[arg-type]
    num_levels = int(sizes_vocab["num_levels"])             # type: ignore[arg-type]

    # --- the beam, which defines the budget -------------------------------
    if not os.path.isfile(cli.beam):
        raise SystemExit(f"--beam {cli.beam} does not exist")
    with np.load(cli.beam, allow_pickle=False) as z:
        for key in ("rows", "rank", "targets", "tuples", "beam_width"):
            if key not in z:
                raise SystemExit(f"{cli.beam} has no '{key}'; not a beam dump")
        beam_rows, beam_rank = _as_int(z["rows"]), _as_int(z["rank"])
        beam_targets = _as_int(z["targets"])
        tuples = np.asarray(z["tuples"]).astype(np.int64, copy=False)
        beam_width = int(z["beam_width"])
        dump_sizes = ([int(s) for s in np.asarray(z["sid_sizes"]).tolist()]
                      if "sid_sizes" in z else None)
        n_split = int(z["n_split"]) if "n_split" in z else int(beam_rows.max()) + 1

    if dump_sizes is not None and dump_sizes != sizes:
        raise SystemExit(
            f"the beam dump was produced with SID widths {dump_sizes} but "
            f"{cli.data_dir}/vocab.json says {sizes}. The bucket sizes would be "
            "counted on a different SID, which is the whole quantity being measured."
        )
    if tuples.ndim != 3 or tuples.shape[2] != num_levels:
        raise SystemExit(
            f"beam tuples have shape {tuples.shape}; expected [rows, {beam_width}, "
            f"{num_levels}] for this vocab"
        )
    n_rows = n_split
    if beam_rows.size != n_rows or np.unique(beam_rows).size != n_rows:
        raise SystemExit(
            f"the beam dump covers {beam_rows.size:,} rows of a {n_rows:,}-row split, "
            "or repeats a row. A per-row budget from one row set cannot be paired "
            "with a rank from another; rerun with --beam_rows 0."
        )
    if int(beam_rows.min()) < 0 or int(beam_rows.max()) >= n_rows:
        raise SystemExit("the beam dump's row indices are outside the split")

    assert_ok = []

    def complain(message: str) -> None:
        assert_ok.append(message)
        print(f"  ! {message}", file=sys.stderr)
        if cli.strict:
            raise SystemExit(f"refusing to write with --strict: {message}")

    # --- the budget, exactly ----------------------------------------------
    codes = load_catalog_codes(cli.data_dir)
    if codes.shape[1] != num_levels:
        raise SystemExit(
            f"catalog_sid_codes.npy has {codes.shape[1]} levels, vocab says {num_levels}"
        )
    buckets = Buckets(codes, sizes)
    print(f"catalogue {codes.shape[0]:,} items, {num_levels} levels, widths {sizes}")
    print(f"distinct tuples {len(buckets):,}, mean {codes.shape[0] / len(buckets):.3f} "
          f"items per tuple, max {int(buckets.counts.max()):,}")

    present = tuples[:, :, 0] >= 0
    safe = np.where(tuples < 0, 0, tuples)
    tuple_keys = pack(safe.reshape(-1, num_levels), sizes).reshape(tuples.shape[:2])
    mult, found = buckets.lookup(tuple_keys.reshape(-1))
    mult = mult.reshape(tuples.shape[:2])
    found = found.reshape(tuples.shape[:2])
    # Position of each named tuple among the distinct tuples, so a per-tuple count
    # (ever-clicked items) can be read with one gather instead of a second lookup.
    tuple_pos = np.clip(np.searchsorted(buckets.uniq, tuple_keys), 0,
                        max(0, len(buckets) - 1))
    named = int(present.sum())
    if named == 0:
        raise SystemExit("the beam dump names no tuples at all")
    if not bool(found[present].all()):
        bad = int((present & ~found).sum())
        example = tuple(tuples.reshape(-1, num_levels)[np.flatnonzero(
            (present & ~found).reshape(-1))[0]].tolist())
        raise SystemExit(
            f"{bad:,} tuples the beam named are not in {cli.data_dir}'s catalogue "
            f"(first: {example}). The dump and the SID directory disagree."
        )

    seen_item = np.zeros(codes.shape[0], dtype=bool)
    if cli.seen_items:
        raw = np.load(cli.seen_items)
        if isinstance(raw, np.lib.npyio.NpzFile):
            if "items" not in raw:
                raise SystemExit(f"{cli.seen_items} has no 'items' array")
            raw = raw["items"]
        ids = _as_int(raw)
        if ids.size == 0:
            raise SystemExit(f"{cli.seen_items} is empty")
        if int(ids.min()) < 0 or int(ids.max()) >= codes.shape[0]:
            raise SystemExit(
                f"{cli.seen_items} holds catalogue row indices outside "
                f"[0, {codes.shape[0]}); it is not a set of catalogue items"
            )
        seen_item[ids] = True
        seen_label = "ever-clicked"
        seen_source = os.path.abspath(cli.seen_items)
    else:
        train_t = os.path.join(cli.data_dir, "train", "targets.npy")
        split_t = os.path.join(cli.data_dir, cli.split, "targets.npy")
        if not (os.path.isfile(train_t) and os.path.isfile(split_t)):
            raise SystemExit(
                "no train/val targets.npy in --data_dir and no --seen_items, so the "
                "second budget view cannot be computed"
            )
        # Only the items that appear as a TARGET. That is 10.5% of the catalogue, not
        # the 33% that appear anywhere in a history, and the difference is not
        # cosmetic: the target set makes the second budget much tighter. Pass
        # --seen_items with the clicked set to get the larger view.
        seen_item[_as_int(np.load(train_t))] = True
        seen_item[_as_int(np.load(split_t))] = True
        seen_label = "target-set"
        seen_source = "train+val targets"
    if not seen_item.any():
        raise SystemExit("the seen set is empty, so the second budget view is undefined")
    seen_counts = buckets.counts_over(seen_item)
    print(f"{seen_label} items {int(seen_item.sum()):,} "
          f"({100.0 * seen_item.mean():.2f}% of the catalogue) from {seen_source}")

    n_all = np.where(present, mult, 0).sum(axis=1)
    n_seen = np.where(present, seen_counts[tuple_pos], 0).sum(axis=1)
    if (n_all <= 0).any():
        raise SystemExit("a row named tuples that carry no catalogue items")

    # The target must sit in the tuple the beam ranked it at. Cheap, and it turns a
    # row/tuple mix-up into a message instead of a plausible number.
    hit = beam_rank <= beam_width
    if hit.any():
        target_keys = pack(codes[beam_targets[hit]], sizes)
        at_rank = tuple_keys[np.flatnonzero(hit), beam_rank[hit] - 1]
        wrong = int((at_rank != target_keys).sum())
        if wrong:
            raise SystemExit(
                f"on {wrong:,} decoded rows the target is not the tuple the beam "
                "reported it at, so the dump's rank and tuples disagree"
            )

    # --- arms -------------------------------------------------------------
    if not cli.arm:
        raise SystemExit("nothing to compare: pass at least one --arm NAME=PATH")
    arms: List[Tuple[str, np.ndarray]] = []
    for text in cli.arm:
        name, path = parse_arm(text)
        arm = load_arm(path, name, n_rows)
        arm_targets, arm_rank = scatter(arm, n_rows, f"arm '{name}'")
        if not np.array_equal(arm_targets, beam_targets):
            bad = int((arm_targets != beam_targets).sum())
            first = int(np.flatnonzero(arm_targets != beam_targets)[0])
            raise SystemExit(
                f"arm '{name}' scored {bad:,} rows against a different target than the "
                f"beam (first at row {first}: {int(arm_targets[first])} vs "
                f"{int(beam_targets[first])}). The two runs are not over the same rows "
                "in the same order, so pairing their ranks would be meaningless."
            )
        arms.append((name, arm_rank))

    tiger_reach = float(hit.mean())
    k_med = int(round(float(np.median(n_all))))
    k_mean = float(n_all.mean())

    print("")
    print(f"beam width {beam_width}, decoded {int(hit.sum()):,}/{n_rows:,} "
          f"({100 * tiger_reach:.4f}%)")
    print(f"items named per row: mean {k_mean:.2f}, median {k_med}, "
          f"min {int(n_all.min())}, max {int(n_all.max())}")
    print(f"{seen_label} only: mean {n_seen.mean():.2f}, median "
          f"{int(np.median(n_seen))}, min {int(n_seen.min())}, max {int(n_seen.max())}")

    rows_out: List[Dict[str, float]] = []
    for name, arm_rank in arms:
        row = {
            "arm": name,
            "named_mean": k_mean,
            "match_all": float((arm_rank <= n_all).mean()),
            "match_seen": float((arm_rank <= n_seen).mean()),
            "match_kmed": float((arm_rank <= k_med).mean()),
            "ndcg_all": ndcg_matched(arm_rank, n_all),
            "mrr10": mrr_at(arm_rank, 10),
            "own_at_1": own_metric(arm_rank, 1),
            "own_at_10": own_metric(arm_rank, 10),
            "own_at_100": own_metric(arm_rank, 100),
        }
        rows_out.append(row)
        for tag, value in (("@1", row["own_at_1"]), ("@10", row["own_at_10"]),
                           ("@100", row["own_at_100"])):
            ref = PUBLISHED.get(name, {}).get(tag)
            if ref is not None and abs(100 * value - 100 * ref) > DRIFT_PP:
                complain(
                    f"{name} own {tag} is {100 * value:.4f}% but {100 * ref:.4f}% is "
                    "published; this dump may come from a different run"
                )
    if "TIGER" in PUBLISHED and abs(100 * tiger_reach
                                    - 100 * PUBLISHED["TIGER"]["decoded"]) > DRIFT_PP:
        complain(
            f"TIGER decoded {100 * tiger_reach:.4f}% against the published "
            f"{100 * PUBLISHED['TIGER']['decoded']:.4f}%; a rerun should match, so "
            "either the checkpoint, the rows, or a near-tie changed"
        )

    tiger_row = {"arm": "TIGER (beam)", "named_mean": k_mean, "match_all": tiger_reach,
                 "match_seen": tiger_reach, "match_kmed": tiger_reach,
                 "ndcg_all": float("nan"), "mrr10": float("nan"),
                 "own_at_1": float((beam_rank <= 1).mean()),
                 "own_at_10": float((beam_rank <= 10).mean()),
                 "own_at_100": float("nan")}
    table = [tiger_row] + rows_out
    for row in table[1:]:
        if tiger_reach > 0:
            row["ratio_to_tiger"] = row["match_all"] / tiger_reach

    header = ["arm", "named_mean", "match_all", "match_seen", "match_kmed",
              "ndcg_all", "mrr10", "own_at_1", "own_at_10", "own_at_100"]
    lines = [f"| arm | named/row | matched (all items) | matched ({seen_label}) | "
             f"matched at k={k_med} | nDCG@N_r | MRR@10 | own R@1 | own R@10 | own R@100 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in table:
        cells = [row["arm"], f"{row['named_mean']:.1f}"]
        for key in ("match_all", "match_seen", "match_kmed"):
            value = row[key]
            cells.append("n/a" if value != value else f"{100 * value:.4f}%")
        for key in ("ndcg_all", "mrr10"):
            value = row[key]
            cells.append("n/a" if value != value else f"{value:.5f}")
        for key in ("own_at_1", "own_at_10", "own_at_100"):
            value = row[key]
            cells.append("n/a" if value != value else f"{100 * value:.4f}%")
        lines.append("| " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    print("")
    print(text)
    print("")
    print(f"TIGER's reach is `mean(rank <= {beam_width})`, never `rank <= N`: a miss is "
          f"stored as {beam_width + 1} and would otherwise pass whenever N >= "
          f"{beam_width + 1}.")
    print("n/a in nDCG@N_r or MRR@10 means the arm ranks tuples, not items, so the metric "
          "would be measured on a different axis and is not reportable beside the others.")

    out_dir = cli.out_dir or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"budget_match{cli.tag}")
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(stem + ".csv", "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in table:
            f.write(",".join(
                [row["arm"]] + ["nan" if row[k] != row[k] else f"{row[k]:.6f}"
                                for k in header[1:]]) + "\n")
    payload = {
        "beam_width": beam_width,
        "n_rows": n_rows,
        "tiger_decoded": int(hit.sum()),
        "tiger_reach": tiger_reach,
        "k_mean": k_mean,
        "k_median": k_med,
        "k_min": int(n_all.min()),
        "k_max": int(n_all.max()),
        "k_mean_seen": float(n_seen.mean()),
        "seen_label": seen_label,
        "seen_items": int(seen_item.sum()),
        "catalog_items": int(codes.shape[0]),
        "distinct_tuples": len(buckets),
        "items_per_tuple": float(codes.shape[0] / len(buckets)),
        "arms": table,
        "warnings": assert_ok,
        "produced_by": REF_USED,
    }
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {stem}.md, {stem}.csv, {stem}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

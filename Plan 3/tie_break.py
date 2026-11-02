"""Tie breaking inside a SID tuple: what popularity buys, and what perfection buys.

The model has one score per *tuple*, so every catalogue item sharing a tuple
gets the same score. `train_digr.evaluate()` resolves that by taking the
target's position in `topk`, which for equal scores is whatever the kernel
happens to do. This script answers two questions on any rank dump, with numpy
only, in seconds:

  1. if ties were broken by train popularity instead, what would R@k be?
  2. what is the ceiling of ANY tie break, popularity or otherwise?

The second one is the number to look at. It is not the injective column that
`region_rank.py` prints. That column puts a whole tuple in one position, so it
credits the fix for skipping every other item in a tuple ranked above the
target, which no within-block reordering can do. Here the rank of a row is
written as

    rank = base + offset + 1

where `base` is how many items sit in tuples that scored strictly higher, and
`offset` is the target's position inside its own block. `base` is fixed by the
scores and no tie break can touch it. A tie break can only drive `offset` from
its current value down to 0, so the true ceiling is `base + 1`. That can be far
below 10, which is why the ceiling here is worth more than the injective one.

`base` comes straight from a dump that carries `fav_item_rank` (the strict
count of items above, written by `region_rank.py`). Without it the script
assumes the tie order is ascending catalogue row, recovers `base` from the
measured rank, and says so, along with how often the assumption is even
self-consistent.

Read only. Writes nothing unless `--json` is given.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from region_lib import Regions, recall_at  # noqa: E402

DEFAULT_KS = (1, 5, 10, 100, 1000)
BANDS = ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 10 ** 9))


# ---------------------------------------------------------------- inputs --


def load_vocab(data_dir: str) -> Dict[str, object]:
    path = os.path.join(data_dir, "vocab.json")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def expand(path: str) -> List[str]:
    """One dump file, a glob, or a directory of `<arm>_<row>.npz` chunks."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.npz")))
    elif glob.has_magic(path):
        files = sorted(glob.glob(path))
    else:
        files = [path]
    if not files:
        raise SystemExit(f"{path} matched no .npz files")
    return files


def load_dump(path: str) -> Dict[str, np.ndarray]:
    """Read one rank dump, or a chunked arm, into `rows/targets/rank`.

    `rank` is what `eval_full_catalog.py --dump_ranks` writes; `fav_item_rank`
    and `region_rank` are extra and come from `region_rank.py`. Chunks are
    concatenated and put back into row order, because a fork pool does not
    deliver them in order.
    """
    files = expand(path)
    parts = []
    for f in files:
        with np.load(f, allow_pickle=False) as npz:
            if "targets" not in npz:
                raise SystemExit(f"{f} has no 'targets' array")
            part = {
                "rows": npz["rows"] if "rows" in npz else np.arange(len(npz["targets"])),
                "targets": npz["targets"],
            }
            for key in ("rank", "item_rank", "region_rank", "fav_item_rank", "g"):
                if key in npz:
                    part[key] = npz[key]
            if "sid_sizes" in npz:
                part["sid_sizes"] = np.asarray(npz["sid_sizes"]).astype(np.int64)
        parts.append(part)

    if len(parts) == 1:
        out = parts[0]
    else:
        keys = set(parts[0])
        for part in parts[1:]:
            keys &= set(part)
        for optional in ("rank", "item_rank", "fav_item_rank", "region_rank", "g"):
            keys.discard(optional)
            if any(optional in part for part in parts):
                keys.add(optional)
        out = {k: np.concatenate([part[k] for part in parts]) for k in keys}
        order = np.argsort(out["rows"], kind="stable")
        out = {k: v[order] for k, v in out.items()}
        rows = out["rows"]
        if rows.size != np.unique(rows).size:
            raise SystemExit(f"{path}: a row appears in more than one chunk")
        if rows[-1] - rows[0] + 1 != rows.size:
            missing = (rows[-1] - rows[0] + 1) - rows.size
            raise SystemExit(f"{path}: {missing:,} rows missing between chunk 0 and the last")

    for key in ("rows", "targets"):
        out[key] = out[key].astype(np.int64)
    rank = out.get("rank", out.get("item_rank"))
    if rank is None:
        raise SystemExit(f"{path} has neither 'rank' nor 'item_rank'")
    out["rank"] = rank.astype(np.int64)
    return out


def load_freq(path: Optional[str], data_dir: str, n_items: int,
              split: str = "train") -> np.ndarray:
    """Per-item train click count, the popularity tie-break's only input."""
    if path is None:
        path = os.path.join(data_dir, split, "targets.npy")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; pass --freq")
    arr = np.load(path, mmap_mode="r").astype(np.int64)
    if arr.size == n_items:
        return np.asarray(arr)
    if arr.max() >= n_items:
        raise SystemExit(
            f"{path} holds {arr.size:,} indices, max {int(arr.max()):,}, but the "
            f"catalogue has {n_items:,} items"
        )
    return np.bincount(arr, minlength=n_items).astype(np.int64)


# ------------------------------------------------------------- the ranks --


def block_layout(regions: Regions) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Members of each region in ascending catalogue row, as one flat order.

    Within a region the members are the catalogue rows, ascending, because a
    stable sort on `region_of_item` keeps the original order inside each block.
    The third return value inverts that order, so a row's place in the flat
    layout is one gather rather than a search.
    """
    order = np.argsort(regions.region_of_item, kind="stable").astype(np.int64)
    starts = np.concatenate(([0], np.cumsum(regions.counts)[:-1])).astype(np.int64)
    pos_of_item = np.empty(regions.n_items, dtype=np.int64)
    pos_of_item[order] = np.arange(regions.n_items, dtype=np.int64)
    return order, starts, pos_of_item


def offset_by_row(regions: Regions, starts: np.ndarray, pos_of_item: np.ndarray,
                  targets: np.ndarray) -> np.ndarray:
    """How many members of the target's own block come before it by catalogue row."""
    region = regions.region_of_item[targets]
    return (pos_of_item[targets] - starts[region]).astype(np.int64)


def offset_by_freq(regions: Regions, starts: np.ndarray, order: np.ndarray,
                   freq: np.ndarray, targets: np.ndarray,
                   g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Position inside the block if members were ordered by popularity, then row.

    Ties in popularity are broken by catalogue row, which is what a stable sort
    of the whole block by frequency would give. `undecided` marks the rows where
    at least one mate has exactly the target's frequency, i.e. where popularity
    alone cannot place the target.
    """
    off = np.zeros(targets.size, dtype=np.int64)
    undecided = np.zeros(targets.size, dtype=bool)
    # Only rows in a shared block need work; the rest have offset 0 by definition.
    for i in np.flatnonzero(g > 1):
        region = regions.region_of_item[targets[i]]
        lo = starts[region]
        members = order[lo:lo + g[i]]
        mf = freq[members]
        ti = int(freq[targets[i]])
        off[i] = int((mf > ti).sum() + ((mf == ti) & (members < targets[i])).sum())
        undecided[i] = bool((mf == ti).sum() > 1)
    return off, undecided


def fold_ranks(dump: Dict[str, np.ndarray], regions: Regions, freq: np.ndarray,
               kmax: int) -> Dict[str, np.ndarray]:
    """Current rank, popularity rank, and the ceiling any tie break can reach."""
    targets = dump["targets"]
    rank = dump["rank"]
    order, starts, pos_of_item = block_layout(regions)
    off_row = offset_by_row(regions, starts, pos_of_item, targets)
    g = regions.counts[regions.region_of_item[targets]].astype(np.int64)
    off_pop, undecided = offset_by_freq(regions, starts, order, freq, targets, g)

    capped = rank > kmax
    if "fav_item_rank" in dump:
        base = np.where(capped, kmax, np.asarray(dump["fav_item_rank"]) - 1)
        assumed = False
    else:
        # rank = base + off_row + 1, so this is forced rather than measured.
        base = np.where(capped, kmax - off_row, rank - 1 - off_row)
        assumed = True

    out = {
        "current": rank.astype(np.int64),
        "popularity": np.minimum(base + off_pop + 1, kmax + 1).astype(np.int64),
        "ceiling": np.minimum(base + 1, kmax + 1).astype(np.int64),
        "base": base.astype(np.int64),
        "off_current": off_row,
        "off_pop": off_pop,
        "g": g,
        "undecided": undecided,
        "assumed": assumed,
    }
    # Self-consistency of the assumed tie order, when there is anything to test.
    if "region_rank" in dump:
        out["region_rank"] = np.asarray(dump["region_rank"]).astype(np.int64)
    return out


# --------------------------------------------------------------- report --


def by_band(g: np.ndarray, columns: Dict[str, np.ndarray], ks: Sequence[int]) -> List[dict]:
    rows = []
    for lo, hi in BANDS:
        m = (g >= lo) & (g <= hi)
        if not m.any():
            continue
        row = {"band": f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+", "rows": int(m.sum())}
        for name, rank in columns.items():
            row[name] = {k: recall_at(rank[m], k) for k in ks}
        rows.append(row)
    return rows


def fmt_block(title: str, ranks: Dict[str, np.ndarray], ks: Sequence[int]) -> str:
    names = list(ranks)
    head = f"  {title:<12}" + "".join(f"{('R@' + str(k)):>11}" for k in ks)
    lines = [head, "  " + "-" * (len(head) - 2)]
    for name in names:
        line = f"  {name:<12}" + "".join(f"{recall_at(ranks[name], k):>11.4%}" for k in ks)
        lines.append(line)
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="What a tie break inside a SID tuple would buy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--arm", action="append", default=[],
                    metavar="NAME=PATH", help="a rank dump to evaluate, repeatable")
    ap.add_argument("--data_dir", required=True,
                    help="holds vocab.json and catalog_sid_codes.npy")
    ap.add_argument("--freq", default=None,
                    help="train/targets.npy, or a per-item count array")
    ap.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    ap.add_argument("--kmax", type=int, default=10000,
                    help="the cut-off the dump was capped at")
    ap.add_argument("--json", default=None, help="write a summary here")
    args = ap.parse_args(argv)

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    if not args.arm:
        raise SystemExit("pass at least one --arm NAME=PATH")

    vocab = load_vocab(args.data_dir)
    sid_sizes = [int(s) for s in vocab["sid_sizes"]]
    codes_path = os.path.join(args.data_dir, "catalog_sid_codes.npy")
    if not os.path.exists(codes_path):
        raise SystemExit(f"missing {codes_path}")
    codes = np.load(codes_path, mmap_mode="r")
    if codes.shape[1] != len(sid_sizes):
        raise SystemExit(
            f"catalog_sid_codes.npy has {codes.shape[1]} levels but vocab.json "
            f"declares {len(sid_sizes)}"
        )
    regions = Regions(np.asarray(codes), sid_sizes)
    freq = load_freq(args.freq, args.data_dir, regions.n_items)

    print("=" * 88)
    print("  tie breaking inside a SID tuple")
    print("=" * 88)
    print(f"  data_dir      : {args.data_dir}")
    print(f"  SID           : widths {sid_sizes}, {regions.n_regions:,} distinct "
          f"tuples over {regions.n_items:,} items ({regions.items_per_region:.2f}/tuple)")
    print(f"  popularity    : {args.freq or os.path.join(args.data_dir, 'train', 'targets.npy')}")
    print(f"  kmax          : {args.kmax:,}")
    print(f"  ties to break : {100.0 * regions.shared_fraction:.4f}% of items sit in a "
          f"tuple that holds more than one, largest {int(regions.counts.max()):,}")

    payload: Dict[str, object] = {"data_dir": os.path.abspath(args.data_dir),
                                  "sid_sizes": sid_sizes, "arms": {}}
    first = True
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm wants NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        dump = load_dump(path)
        if "sid_sizes" in dump and list(dump["sid_sizes"]) != sid_sizes:
            raise SystemExit(
                f"{name}: the dump was built on widths {list(dump['sid_sizes'])} but "
                f"vocab.json declares {sid_sizes}; the blocks would not match"
            )
        f = fold_ranks(dump, regions, freq, args.kmax)
        if (f["base"] < 0).any():
            share = 100.0 * (f["base"] < 0).mean()
            raise SystemExit(
                f"{name}: {share:.4f}% of rows imply a negative number of items above the "
                f"target, which cannot happen if this arm's scores are one value per SID "
                f"tuple.\nSuch an arm has no ties in this SID to break, so ranking it by "
                f"popularity would reorder the whole list rather than resolve a tie. Only "
                f"tuple-scored arms (DiGR) are valid inputs."
            )

        print()
        print(f"  --- arm {name}  ({len(dump['targets']):,} rows) ---")
        if bool(f["assumed"]):
            print("  note: no 'fav_item_rank' in this dump, so the tie order is ASSUMED to be")
            print("        ascending catalogue row. Re-run region_rank.py to get the exact base.")
            print("        base >= 0 on 100.0000% of rows, which the assumption needs and which")
            print("        an arm without tuple ties fails. See the note at the end.")
            if "region_rank" in dump:
                ok = f["base"] >= f["region_rank"] - 1
                print(f"        base >= region_rank - 1 on {100.0 * ok.mean():.4f}% of rows "
                      f"(a real check: every tuple above contributes at least one item)")
        else:
            print("  base taken from the dump's own strict count, so nothing is assumed")

        print()
        print(fmt_block("R@k", {"current": f["current"],
                                "popularity": f["popularity"],
                                "ceiling": f["ceiling"]}, ks))
        print()
        g = f["g"]
        shift = f["current"] - f["popularity"]
        if not first:
            pass
        else:
            print(f"  rows in a shared tuple   : {100.0 * (g > 1).mean():.4f}%, and the target "
                  f"is the most")
            print(f"    clicked of its block    : {100.0 * (f['off_pop'] == 0).mean():.4f}% of "
                  f"all rows ({100.0 * (f['off_pop'][g > 1] == 0).mean():.4f}% of shared rows)")
            print(f"  popularity is also tied  : {100.0 * f['undecided'].mean():.4f}% of rows, so "
                  f"it cannot order those")
        print(f"  popularity moves it up   : {100.0 * (shift > 0).mean():.4f}% of rows, "
              f"mean {shift[shift > 0].mean() if (shift > 0).any() else 0.0:.1f} positions")
        print(f"  still movable (offset>0): {100.0 * (f['off_pop'] > 0).mean():.4f}% of rows")

        print()
        print("  by tuple size")
        bands = by_band(g, {"current": f["current"],
                            "popularity": f["popularity"],
                            "ceiling": f["ceiling"]}, ks)
        best = ks[-1] if ks[-1] <= 100 else 100
        print(f"  {'band':<9}{'rows':>8}" + "".join(
            f"{f'{n}@{best}':>16}" for n in ("current", "popularity", "ceiling")))
        for row in bands:
            print(f"  {row['band']:<9}{row['rows']:>8}" + "".join(
                f"{row[n][best]:>16.4%}" for n in ("current", "popularity", "ceiling")))

        payload["arms"][name] = {
            "rows": int(len(dump["targets"])),
            "tie_order_assumed": bool(f["assumed"]),
            "current": {f"r@{k}": recall_at(f["current"], k) for k in ks},
            "popularity": {f"r@{k}": recall_at(f["popularity"], k) for k in ks},
            "ceiling": {f"r@{k}": recall_at(f["ceiling"], k) for k in ks},
            "shared_tuple_share": float((g > 1).mean()),
            "most_popular_share": float((f["off_pop"] == 0).mean()),
            "popularity_also_tied": float(f["undecided"].mean()),
            "bands": bands,
        }
        first = False

    print()
    print("=" * 88)
    print("  'ceiling' is base + 1: no tie break, however good, can rank an item above")
    print("  any item that scored strictly higher. Read it before believing 'popularity'.")
    print("  Only arms scored one-value-per-tuple are valid input. An arm whose ranks do")
    print("  not imply a non-negative base is refused, because reordering it would move")
    print("  every row rather than resolve a tie.")
    print("=" * 88)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

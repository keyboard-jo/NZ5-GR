#!/usr/bin/env python
"""Diagnose one Semantic ID assignment.

Answers four questions about a set of RQ-VAE codes, with no model and no GPU:

  1. Are the codebooks healthy? Codes used, entropy in bits, against the ceiling.
  2. How top heavy is the partition? Items per bucket at each prefix depth.
  3. How many items collide? Items that share their prefix with another item.
  4. How much better than chance is a bucket at holding the real next click?

Question 4 is the one that compares two SIDs. Raw bucket recall is not comparable
across SIDs, because a smaller bucket is mechanically less likely to contain the
target. So the number to read is `lift`, which is recall divided by the bucket's
share of the catalogue. A SID that shrinks every bucket by 4x and leaves lift
unchanged has bought nothing.

Usage
-----
  # full diagnosis from a preprocessed digr_data directory
  python "Plan 2/sid_diagnose.py" --data_dir local/2026-20-10-2-pre/digr_data

  # codes only, no lift (for a fresh RQ-VAE output before preprocessing)
  python "Plan 2/sid_diagnose.py" --codes new_catalog_sid_codes.npy --vocab vocab.json

  # compare two SIDs
  python "Plan 2/sid_diagnose.py" --data_dir a/digr_data --json a.json
  python "Plan 2/sid_diagnose.py" --data_dir b/digr_data --json b.json

Only numpy is needed. The catalogue does not have to be the one the val split was
built from, as long as `catalog_sid_codes.npy` and `vocab.json` describe the same
item ordering as `catalog_item_ids.npy`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# The mixed-radix key is packed into int64. Keeping the product below 2**62 leaves
# room for the largest single code and avoids silent wraparound.
MAX_KEY_PRODUCT = 2 ** 62

CATALOG_CODES = "catalog_sid_codes.npy"
VOCAB_NAME = "vocab.json"
CATALOG_META = "catalog_meta.json"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_paths(args) -> dict:
    """Work out which files to read, allowing --codes / --vocab to override."""
    data_dir = args.data_dir
    codes_path = args.codes or (os.path.join(data_dir, CATALOG_CODES) if data_dir else None)
    vocab_path = args.vocab or (os.path.join(data_dir, VOCAB_NAME) if data_dir else None)
    if codes_path is None:
        raise SystemExit("need --codes, or --data_dir holding " + CATALOG_CODES)
    if vocab_path is None:
        raise SystemExit("need --vocab, or --data_dir holding " + VOCAB_NAME)
    for path in (codes_path, vocab_path):
        if not os.path.isfile(path):
            raise SystemExit("missing file: " + path)
    return {"codes": codes_path, "vocab": vocab_path}


def load_codes(path: str) -> np.ndarray:
    codes = np.load(path)
    if codes.ndim != 2:
        raise SystemExit("codes must be 2-D (items, levels), got shape " + str(codes.shape))
    # int32 in the file, but the mixed-radix packing needs int64. On Windows a C
    # long is 32 bits, so leaving this as int32 silently overflows from depth 4 on.
    return np.ascontiguousarray(codes, dtype=np.int64)


def load_sizes(vocab: dict, n_levels: int, path: str) -> np.ndarray:
    sizes = vocab.get("sid_sizes")
    if sizes is None:
        raise SystemExit("vocab has no sid_sizes: " + path)
    sizes = [int(s) for s in sizes]
    if len(sizes) < n_levels:
        raise SystemExit(
            "codes have %d levels but vocab lists %d codebook sizes" % (n_levels, len(sizes))
        )
    return np.asarray(sizes[:n_levels], dtype=np.int64)


# ---------------------------------------------------------------------------
# Packing and prefix indexing
# ---------------------------------------------------------------------------
def pack_keys(codes: np.ndarray, sizes) -> np.ndarray:
    """Mixed-radix key per row. Level 0 is the least significant digit."""
    codes = np.asarray(codes, dtype=np.int64)
    sizes = [int(s) for s in sizes]
    if codes.ndim != 2:
        raise ValueError("codes must be 2-D, got shape " + str(codes.shape))
    if codes.shape[1] != len(sizes):
        raise ValueError("codes have %d levels, sizes cover %d" % (codes.shape[1], len(sizes)))
    product = 1
    for size in sizes:
        product *= size
    if product >= MAX_KEY_PRODUCT:
        raise ValueError(
            "mixed-radix product %d exceeds %d, so keys would overflow int64; "
            "diagnose this SID a depth at a time instead" % (product, MAX_KEY_PRODUCT)
        )
    key = np.zeros(codes.shape[0], dtype=np.int64)
    radix = 1
    for level, size in enumerate(sizes):
        column = codes[:, level]
        if column.size:
            low, high = int(column.min()), int(column.max())
            if low < 0 or high >= size:
                raise ValueError(
                    "level %d holds codes in [%d, %d] but its codebook size is %d"
                    % (level, low, high, size)
                )
        key += column * radix
        radix *= size
    return key


class PrefixIndex:
    """Buckets the catalogue by the first `depth` codes."""

    def __init__(self, codes: np.ndarray, sizes, depth: int):
        self.depth = int(depth)
        self.sizes = [int(s) for s in sizes[: self.depth]]
        keys = pack_keys(codes[:, : self.depth], self.sizes)
        unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        self.unique_keys = unique
        self.bucket_of_item = inverse
        self.counts = counts
        self.n_items = int(codes.shape[0])

    @property
    def n_buckets(self) -> int:
        return int(self.counts.size)

    @property
    def items_per_bucket(self) -> float:
        return self.n_items / self.n_buckets if self.n_buckets else float("nan")

    @property
    def shared_items(self) -> int:
        """Items whose prefix is shared with at least one other item."""
        return int(self.n_items - self.n_buckets)

    @property
    def largest_bucket(self) -> int:
        return int(self.counts.max()) if self.counts.size else 0

    def lookup(self, codes: np.ndarray):
        """Bucket id per query, plus whether the prefix exists in the catalogue.

        Bucket ids are positions in the sorted unique key array, which is the same
        numbering `bucket_of_item` uses, so the two are directly comparable.
        """
        codes = np.asarray(codes, dtype=np.int64)
        keys = pack_keys(codes[:, : self.depth], self.sizes)
        if self.unique_keys.size == 0:
            return np.zeros(keys.size, dtype=np.int64), np.zeros(keys.size, dtype=bool)
        position = np.searchsorted(self.unique_keys, keys)
        position = np.clip(position, 0, self.unique_keys.size - 1)
        return position, self.unique_keys[position] == keys


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def codebook_report(codes: np.ndarray, sizes: np.ndarray) -> list:
    rows = []
    for level, size in enumerate(sizes):
        counts = np.bincount(codes[:, level], minlength=int(size)).astype(np.float64)
        used = int((counts > 0).sum())
        total = counts.sum()
        share = counts[counts > 0] / total
        entropy = float(-(share * np.log2(share)).sum()) if total else 0.0
        ceiling = float(np.log2(size)) if size > 1 else 0.0
        rows.append(
            {
                "level": level,
                "size": int(size),
                "used": used,
                "entropy_bits": entropy,
                "max_bits": ceiling,
                "utilisation": used / size if size else 0.0,
                "entropy_ratio": entropy / ceiling if ceiling else 0.0,
            }
        )
    return rows


def occupancy_report(codes: np.ndarray, sizes: np.ndarray) -> list:
    rows = []
    for depth in range(1, codes.shape[1] + 1):
        index = PrefixIndex(codes, sizes, depth)
        rows.append(
            {
                "depth": depth,
                "buckets": index.n_buckets,
                "items_per_bucket": index.items_per_bucket,
                "catalog_share": index.n_buckets / index.n_items,
                "shared_items": index.shared_items,
                "shared_fraction": index.shared_items / index.n_items,
                "largest_bucket": index.largest_bucket,
            }
        )
    return rows


def last_event_codes(tokens: np.ndarray, offsets: np.ndarray, vocab: dict) -> tuple:
    """SID codes of the final event in each row.

    Token layout per row is `[F_1 .. F_M, SEP, (T_i, S_i_1 .. S_i_L), ...]`, so the
    last `L` tokens are the last event's codes, level `L-1` last.
    """
    levels = int(vocab["num_levels"])
    sid_offsets = np.asarray([int(v) for v in vocab["sid_offsets"]], dtype=np.int64)
    sizes = np.asarray([int(v) for v in vocab["sid_sizes"]], dtype=np.int64)
    rows = int(offsets.size - 1)
    lengths = np.diff(offsets.astype(np.int64))
    starts = offsets[1:].astype(np.int64) - levels
    valid = lengths >= levels
    starts = np.where(valid, starts, 0)
    gather = starts[:, None] + np.arange(levels)[None, :]
    codes = np.asarray(tokens, dtype=np.int64)[gather] - sid_offsets[None, :]
    inside = ((codes >= 0) & (codes < sizes[None, :])).all(axis=1) & valid
    codes = np.where(inside[:, None], codes, -1)
    return codes, inside, rows


def lift_report(codes, sizes, vocab, split_dir, max_rows=None) -> dict:
    tokens_path = os.path.join(split_dir, "tokens.npy")
    offsets_path = os.path.join(split_dir, "offsets.npy")
    targets_path = os.path.join(split_dir, "targets.npy")
    for path in (tokens_path, offsets_path, targets_path):
        if not os.path.isfile(path):
            raise SystemExit("missing split file: " + path)

    tokens = np.load(tokens_path)
    offsets = np.load(offsets_path)
    targets = np.asarray(np.load(targets_path), dtype=np.int64).ravel()
    query, inside, rows = last_event_codes(tokens, offsets, vocab)

    if targets.size != rows:
        raise SystemExit(
            "split has %d target rows but %d token rows" % (targets.size, rows)
        )
    take = np.arange(rows)
    if max_rows and rows > max_rows:
        take = np.linspace(0, rows - 1, num=int(max_rows)).astype(np.int64)

    query = query[take]
    inside = inside[take]
    target = targets[take]
    target_in_catalog = (target >= 0) & (target < codes.shape[0])

    results = []
    for depth in range(1, codes.shape[1] + 1):
        index = PrefixIndex(codes, sizes, depth)
        bucket, found = index.lookup(np.where(inside[:, None], query, 0)[:, :depth])
        found = found & inside
        target_bucket = index.bucket_of_item[np.where(target_in_catalog, target, 0)]
        hit = found & target_in_catalog & (bucket == target_bucket)
        recall = float(hit.mean()) if hit.size else 0.0
        uniform_base = float(index.counts.mean()) / index.n_items
        weighted_base = (
            float(index.counts[bucket].mean()) / index.n_items if bucket.size else float("nan")
        )
        results.append(
            {
                "depth": depth,
                "rows": int(hit.size),
                "recall": recall,
                "uniform_base": uniform_base,
                "lift": recall / uniform_base if uniform_base else float("nan"),
                "size_biased_bucket": weighted_base * index.n_items,
                "largest_bucket": index.largest_bucket,
                "buckets": index.n_buckets,
            }
        )

    return {
        "split_dir": split_dir,
        "rows": int(take.size),
        "rows_available": rows,
        "queries_unmapped": int((~inside).sum()),
        "targets_outside_catalog": int((~target_in_catalog).sum()),
        "depths": results,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def human_int(value) -> str:
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e15:
            return "{:,}".format(int(value))
        return "{:,.2f}".format(value)
    return "{:,}".format(int(value))


def human_pct(value) -> str:
    if value == 0:
        return "0%"
    if abs(value) >= 0.01:
        return "{:.4f}%".format(value * 100.0)
    return "{:.3e}%".format(value * 100.0)


def print_report(report: dict) -> None:
    head = report["inputs"]
    print("SID diagnosis")
    print("  codes   " + head["codes"])
    print("  vocab   " + head["vocab"])
    print("  items   " + human_int(report["n_items"]))
    print("  levels  " + str(report["n_levels"]))

    print("")
    print("Codebooks")
    print("  level      size      used  entropy      max   utilis  entropy/size")
    for row in report["codebooks"]:
        print(
            "  {:5d}  {:8s}  {:8s}  {:7.4f}  {:7.3f}  {:6.2f}%  {:9.4f}".format(
                row["level"],
                human_int(row["size"]),
                human_int(row["used"]),
                row["entropy_bits"],
                row["max_bits"],
                row["utilisation"] * 100.0,
                row["entropy_ratio"],
            )
        )
    dead = [row["level"] for row in report["codebooks"] if row["used"] < row["size"]]
    if dead:
        print("  unused codes at levels: " + ", ".join(str(d) for d in dead))
    else:
        print("  every level uses every code")

    print("")
    print("Occupancy by prefix depth")
    print("  depth        buckets   items/bucket  catalogue share    shared items  largest")
    for row in report["occupancy"]:
        print(
            "  {:5d}  {:12s}  {:12s}  {:15s}  {:13s}  {:7s}".format(
                row["depth"],
                human_int(row["buckets"]),
                human_int(row["items_per_bucket"]),
                human_pct(row["catalog_share"]),
                "{} ({:.2f}%)".format(
                    human_int(row["shared_items"]), row["shared_fraction"] * 100.0
                ),
                human_int(row["largest_bucket"]),
            )
        )
    print("  shared items are items whose prefix is held by at least one other item")

    lift = report.get("lift")
    if not lift:
        print("")
        print("Bucket lift skipped, no split directory was read")
        return

    print("")
    print("Bucket lift against real next clicks")
    print(
        "  rows {} of {} | unmapped queries {} | targets outside catalogue {}".format(
            human_int(lift["rows"]),
            human_int(lift["rows_available"]),
            human_int(lift["queries_unmapped"]),
            human_int(lift["targets_outside_catalog"]),
        )
    )
    print("  depth      recall   uniform base        lift  query bucket  largest")
    for row in lift["depths"]:
        print(
            "  {:5d}  {:9s}  {:13s}  {:9s}  {:12s}  {:9s}".format(
                row["depth"],
                human_pct(row["recall"]),
                human_pct(row["uniform_base"]),
                "{:.1f}x".format(row["lift"]),
                human_int(row["size_biased_bucket"]),
                human_int(row["largest_bucket"]),
            )
        )
    print("  lift is recall divided by the bucket share, so it is not inflated by small buckets")
    print("  query bucket is the mean size of the buckets the queries actually land in")

    best = max(lift["depths"], key=lambda row: row["lift"])
    print("")
    print(
        "  peak lift {:.1f}x at depth {}, holding {:.2f} items per bucket".format(
            best["lift"], best["depth"], best["size_biased_bucket"]
        )
    )
    if best["depth"] < lift["depths"][-1]["depth"]:
        print(
            "  levels past depth {} add no discrimination, lift falls from {:.1f}x to {:.1f}x".format(
                best["depth"],
                best["lift"],
                lift["depths"][-1]["lift"],
            )
        )
    print(
        "  a wider SID passes this test only if it holds {:.1f}x lift at depth 1,".format(
            lift["depths"][0]["lift"]
        )
    )
    print(
        "  and raw recall is expected to fall there because the bucket is smaller"
    )


def build_report(args) -> dict:
    paths = resolve_paths(args)
    codes = load_codes(paths["codes"])
    vocab = load_json(paths["vocab"])
    sizes = load_sizes(vocab, codes.shape[1], paths["vocab"])

    report = {
        "inputs": {"codes": paths["codes"], "vocab": paths["vocab"], "data_dir": args.data_dir},
        "n_items": int(codes.shape[0]),
        "n_levels": int(codes.shape[1]),
        "codebooks": codebook_report(codes, sizes),
        "occupancy": occupancy_report(codes, sizes),
    }

    if args.split_dir and not args.no_lift:
        report["lift"] = lift_report(
            codes, sizes, vocab, args.split_dir, max_rows=args.max_rows
        )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose a Semantic ID assignment: codebook health, occupancy, "
        "collisions, and bucket lift against real next clicks."
    )
    parser.add_argument("--data_dir", default=None, help="preprocessed digr_data directory")
    parser.add_argument("--codes", default=None, help="override catalog_sid_codes.npy")
    parser.add_argument("--vocab", default=None, help="override vocab.json")
    parser.add_argument(
        "--split_dir",
        default=None,
        help="directory holding tokens.npy, offsets.npy and targets.npy; "
        "defaults to <data_dir>/val when data_dir is given",
    )
    parser.add_argument("--max_rows", type=int, default=None, help="subsample the split")
    parser.add_argument("--no_lift", action="store_true", help="skip the split entirely")
    parser.add_argument("--json", default=None, help="write the full report as JSON")
    args = parser.parse_args(argv)

    if args.split_dir is None and args.data_dir:
        args.split_dir = os.path.join(args.data_dir, "val")
    if args.split_dir and not os.path.isdir(args.split_dir):
        if args.data_dir:
            print("no val split at " + args.split_dir + ", skipping lift", file=sys.stderr)
            args.split_dir = None
        else:
            raise SystemExit("no such split_dir: " + args.split_dir)

    report = build_report(args)
    print_report(report)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print("")
        print("wrote " + args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

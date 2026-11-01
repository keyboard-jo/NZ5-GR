#!/usr/bin/env python
"""Diagnose a semantic-id assignment straight from an RQ-VAE index json.

The playbook is `docs/archive/sid_diagnostics.md`. It assumes the codes have already
been turned into an aligned `.npy`, which meant the diagnosis could only be run on a
data directory someone had built. This script reads the artefact the quantiser
actually writes, `{item_id: [c_0 .. c_{L-1}]}`, so a SID can be judged before any
preprocessing, any tokenisation and any GPU time.

    python "SID Diagnostics/diagnose_sid.py" \
        --sid 2x8192="SID Diagnostics/index_rqvae_2x8192.json" \
        --sid 4x256+731="SID Diagnostics/index_rqvae_4x256_731.json"

Answers the three questions that need no model:

  Q1  is the code space fully used, and how many bits does it spend?
  Q3  does each level add information, or only split?
  Q4  is the partition fine enough for the head, given the catalogue size?

and, when a tokenised split is supplied, the one that needs labels:

  Q2  is a shared code informative about what users do?  (bucket lift)

Lift needs the split that was built from the *same* SID, because the val rows carry
codes, not item ids. Bounding the loss: the query item is decoded to a catalogue row
through that split's catalogue, then re-read under the SID under test. If the SID
under test is the split's own SID the translation is the identity, so passing the
matching split is also how the archived numbers are reproduced.

Two behaviours worth knowing:

  * `--sid` may repeat. Every SID gets its own report and they all land in one
    comparison table, which is the only honest way to compare widths.
  * A level whose alphabet is small, flat and ordered is reported as a *suffix*, not
    a code. That is the `--add_unique_id` trap: it is fully utilised, its entropy
    ratio is fine as a fraction of its own log2(size), and it carries no semantics.

Exit status is 0 unless an input is unreadable or a codes array is ragged.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

CATALOG_CODES = "catalog_sid_codes.npy"
CATALOG_IDS = "catalog_item_ids.npy"
VOCAB_NAME = "vocab.json"

# how many items per level-0 code before a code says almost nothing about an item
REFERENCE_ITEMS_PER_CODE = (39, 195)  # DiGR paper's Amazon subsets


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
class Sid:
    """One SID under test: ids, codes, and the sizes implied by the codes."""

    def __init__(self, name: str, path: str, ids: np.ndarray, codes: np.ndarray):
        self.name = name
        self.path = path
        self.ids = ids
        self.codes = codes
        self.n_items = int(codes.shape[0])
        self.levels = int(codes.shape[1])
        self.sizes = np.asarray(
            [int(codes[:, level].max()) + 1 for level in range(self.levels)],
            dtype=np.int64,
        )


def _entries_body(text: bytes):
    """Rewrite `{"id": [a, b], ...}` as one line per entry, `id,c_0,..,c_{L-1}`.

    A dict of 4,783,154 short lists costs well over a gigabyte, which makes the
    diagnosis awkward on a laptop. Presenting the same text as a fixed-width table
    lets a C parser read it directly. One line per entry matters: the same numbers on
    a single row measure 55 s with numpy and 595 s with pandas, against 1.2 s for
    4,783,154 rows of three columns.

    Returns (body, n_entries) or None when the text is not in that shape or the
    entries are ragged, in which case the caller falls back to a real parser.
    """
    if not text.startswith(b'{"'):
        return None
    body = text[2:]
    if body.endswith(b"}"):
        body = body[:-1]
    n_entries = body.count(b'": [')
    if n_entries == 0:
        return None
    body = body.replace(b'": [', b",")
    body = body.replace(b'], "', b"\n")
    if body.endswith(b"]"):
        body = body[:-1]
    if any(symbol in body for symbol in (b"[", b"]", b"{")):
        return None
    return body, n_entries


def _parse_entries(body: bytes, n_entries: int) -> np.ndarray:
    """Parse the one-line-per-entry body into an (n, width + 1) int64 table."""
    table = None
    try:
        import io

        import pandas as pd

        table = pd.read_csv(
            io.BytesIO(body), header=None, dtype="int64", engine="c", sep=","
        ).to_numpy()
    except Exception:  # noqa: BLE001  pandas missing, or the table is ragged
        table = None
    if table is None:
        flat = np.fromstring(body.replace(b"\n", b",").decode("ascii"),
                             dtype=np.int64, sep=",")
        if n_entries == 0 or flat.size % n_entries:
            raise ValueError(
                f"{flat.size} integers do not split into {n_entries} entries"
            )
        table = flat.reshape(n_entries, flat.size // n_entries)
    if table.ndim != 2 or table.shape[0] != n_entries or table.shape[1] < 2:
        raise ValueError(f"parsed shape {table.shape} against {n_entries} entries")
    return np.ascontiguousarray(table, dtype=np.int64)


def load_sid_json(path: str) -> tuple[np.ndarray, np.ndarray]:
    """(ids, codes) from an RQ-VAE index json, in file order."""
    with open(path, "rb") as handle:
        text = handle.read()

    shaped = _entries_body(text)
    if shaped is not None:
        body, n_entries = shaped
        try:
            table = _parse_entries(body, n_entries)
            return np.ascontiguousarray(table[:, 0]), np.ascontiguousarray(table[:, 1:])
        except ValueError:
            pass  # ragged: fall through to the dict parser, which reports the widths

    ids, codes = [], []
    for match in _SLOW.finditer(text):
        ids.append(int(match.group(1)))
        codes.append([int(v) for v in match.group(2).split(b",")])
    if not codes:
        raise ValueError("no `\"id\": [codes]` entries found; is this an index json?")
    widths = {len(row) for row in codes}
    if len(widths) != 1:
        raise ValueError(f"ragged code vectors, widths {sorted(widths)}")
    return (
        np.asarray(ids, dtype=np.int64),
        np.asarray(codes, dtype=np.int64).reshape(len(codes), widths.pop()),
    )


def catalog_file(dirname: str, name: str) -> str:
    """Find a catalogue-level file in `dirname`, else in its parent.

    Tokens live in `<digr_data>/val/` while `vocab.json` and the catalogue arrays live
    in `<digr_data>`, so a split is passed as the `val` directory and the catalogue is
    one level up.
    """
    for base in (dirname, os.path.dirname(os.path.abspath(dirname))):
        candidate = os.path.join(base, name)
        if os.path.isfile(candidate):
            return candidate
    raise SystemExit(f"missing {name} in {dirname} or its parent")


def pack_keys(codes: np.ndarray, sizes) -> np.ndarray:
    """Mixed-radix key per row, level 0 least significant, int64 throughout."""
    sizes = [int(s) for s in sizes]
    if codes.shape[1] != len(sizes):
        raise ValueError(f"{codes.shape[1]} code columns against {len(sizes)} sizes")
    keys = np.zeros(codes.shape[0], dtype=np.int64)
    for level, size in enumerate(sizes):
        column = codes[:, level].astype(np.int64, copy=False)
        if column.size and (column.min() < 0 or column.max() >= size):
            raise ValueError(f"level {level} has codes outside [0, {size})")
        keys = keys * np.int64(size) + column
    if float(np.prod(sizes, dtype=float)) >= 2.0 ** 62:
        raise ValueError("radix product exceeds 2**62, which int64 cannot index")
    return keys


class Buckets:
    """Prefix buckets at one depth, with per-item multiplicities."""

    def __init__(self, codes: np.ndarray, sizes, depth: int):
        keys = pack_keys(codes[:, :depth], sizes[:depth])
        self.uniq, inverse, self.counts = np.unique(
            keys, return_inverse=True, return_counts=True
        )
        self.bucket_of_item = inverse.astype(np.int64)
        self.n_items = int(codes.shape[0])
        self.n_buckets = int(self.uniq.size)
        self.largest = int(self.counts.max()) if self.counts.size else 0
        self.items_per_bucket = self.n_items / self.n_buckets if self.n_buckets else 0.0
        self.shared_items = int((self.counts[self.bucket_of_item] > 1).sum())


# ---------------------------------------------------------------------------
# Q1 coverage, and what the bits are spent on
# ---------------------------------------------------------------------------
def coverage_report(sid: Sid) -> dict:
    rows = []
    for level in range(sid.levels):
        size = int(sid.sizes[level])
        counts = np.bincount(sid.codes[:, level], minlength=size).astype(np.float64)
        used = int((counts > 0).sum())
        share = counts[counts > 0] / counts.sum()
        entropy = float(-(share * np.log2(share)).sum()) if share.size else 0.0
        ceiling = float(np.log2(size)) if size > 1 else 0.0
        ratio = entropy / ceiling if ceiling else 0.0
        # A collision rank is a dense small alphabet with far less entropy than its own
        # log2(size): the archived unique-id level measured 0.1855. A real code shares
        # the shape 0..k-1 but not the entropy, so do not test for ordered counts.
        rows.append(
            {
                "level": level,
                "size": size,
                "used": used,
                "utilisation": used / size if size else 0.0,
                "entropy_bits": entropy,
                "max_bits": ceiling,
                "entropy_ratio": ratio,
                "looks_like_suffix": bool(
                    used == size and size <= 1024 and size < sid.n_items / 1000 and ratio < 0.5
                ),
            }
        )

    total_bits = float(sum(np.log2(s) for s in sid.sizes if s > 1))
    identity_bits = float(np.log2(sid.n_items)) if sid.n_items else 0.0
    injective_at = None
    for depth in range(1, sid.levels + 1):
        if Buckets(sid.codes, sid.sizes, depth).n_buckets == sid.n_items:
            injective_at = depth
            break
    return {
        "levels": rows,
        "total_bits": total_bits,
        "identity_bits": identity_bits,
        "redundant_bits": total_bits - identity_bits,
        "injective_at": injective_at,
        "items_per_level0_code": sid.n_items / sid.sizes[0],
    }


# ---------------------------------------------------------------------------
# Q3/Q4 occupancy and the tie ceiling
# ---------------------------------------------------------------------------
def occupancy_report(sid: Sid, ks=(1, 10, 100, 1000)) -> list:
    rows = []
    for depth in range(1, sid.levels + 1):
        buckets = Buckets(sid.codes, sid.sizes, depth)
        group = buckets.counts[buckets.bucket_of_item].astype(np.float64)
        ceiling = {
            f"ceiling@{k}": float(np.minimum(1.0, k / group).mean())
            for k in ks
        }
        rows.append(
            {
                "depth": depth,
                "buckets": buckets.n_buckets,
                "items_per_bucket": buckets.items_per_bucket,
                "catalog_share": buckets.n_buckets / buckets.n_items,
                "shared_items": buckets.shared_items,
                "shared_fraction": buckets.shared_items / buckets.n_items,
                "largest_bucket": buckets.largest,
                "mean_group": float(group.mean()),
                "median_group": float(np.median(group)),
                "p90_group": float(np.percentile(group, 90)),
                "ceiling": ceiling,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Q2 lift, when a tokenised split is available
# ---------------------------------------------------------------------------
def last_event_codes(tokens, offsets, vocab) -> tuple:
    """SID codes of the final event in each row: the last `L` tokens, level 0 first."""
    levels = int(vocab["num_levels"])
    sid_offsets = np.asarray([int(v) for v in vocab["sid_offsets"]], dtype=np.int64)
    sizes = np.asarray([int(v) for v in vocab["sid_sizes"]], dtype=np.int64)
    lengths = np.diff(np.asarray(offsets, dtype=np.int64))
    starts = np.asarray(offsets, dtype=np.int64)[1:] - levels
    valid = lengths >= levels
    starts = np.where(valid, starts, 0)
    gather = starts[:, None] + np.arange(levels)[None, :]
    codes = np.asarray(tokens, dtype=np.int64)[gather] - sid_offsets[None, :]
    inside = ((codes >= 0) & (codes < sizes[None, :])).all(axis=1) & valid
    return np.where(inside[:, None], codes, -1), inside


def split_identity(split_dir: str) -> tuple:
    """(sorted catalogue keys, order) so a decoded tuple can become a row index."""
    codes = np.load(catalog_file(split_dir, CATALOG_CODES))
    vocab = json.load(open(catalog_file(split_dir, VOCAB_NAME)))
    sizes = np.asarray(vocab["sid_sizes"], dtype=np.int64)[: codes.shape[1]]
    keys = pack_keys(np.asarray(codes, dtype=np.int64), sizes)
    order = np.argsort(keys, kind="stable")
    return keys[order], order, codes


def lift_report(sid: Sid, split_dir: str, max_rows=None) -> dict:
    for name in ("tokens.npy", "offsets.npy", "targets.npy"):
        path = os.path.join(split_dir, name)
        if not os.path.isfile(path):
            raise SystemExit("missing split file: " + path)
    ids_path = catalog_file(split_dir, CATALOG_IDS)
    vocab = json.load(open(catalog_file(split_dir, VOCAB_NAME)))
    tokens = np.load(os.path.join(split_dir, "tokens.npy"))
    offsets = np.load(os.path.join(split_dir, "offsets.npy"))
    targets = np.asarray(np.load(os.path.join(split_dir, "targets.npy")),
                         dtype=np.int64).ravel()

    # The json is in the quantiser's file order. Targets are catalogue row indexs, so
    # the codes have to be placed on catalogue rows before anything is compared. Skip
    # this and every metric becomes noise that still looks plausible.
    catalog_ids = np.asarray(np.load(ids_path), dtype=np.int64)
    row_order = np.argsort(catalog_ids, kind="stable")
    sorted_ids = catalog_ids[row_order]
    pos = np.clip(np.searchsorted(sorted_ids, sid.ids), 0, max(sorted_ids.size - 1, 0))
    matched = sorted_ids[pos] == sid.ids
    codes = np.zeros((catalog_ids.size, sid.levels), dtype=np.int64)
    codes[row_order[pos[matched]]] = sid.codes[matched]
    alignment = {
        "json_entries": int(sid.ids.size),
        "catalog_items": int(catalog_ids.size),
        "placed": int(matched.sum()),
        "unique_ids": int(np.unique(sid.ids).size),
    }

    query, inside = last_event_codes(tokens, offsets, vocab)
    keys, order, _old_codes = split_identity(split_dir)

    stream_sizes = np.asarray(vocab["sid_sizes"], dtype=np.int64)[: query.shape[1]]
    query_keys = pack_keys(np.where(inside[:, None], query, 0), stream_sizes)
    at = np.clip(np.searchsorted(keys, query_keys), 0, max(keys.size - 1, 0))
    resolved = keys[at] == query_keys
    item = order[at]
    inside = inside & resolved
    query = codes[np.where(inside, item, 0)]

    rows = int(offsets.size - 1)
    if targets.size != rows:
        raise SystemExit(f"split has {targets.size} target rows, {rows} token rows")
    take = np.arange(rows)
    if max_rows and rows > max_rows:
        take = np.linspace(0, rows - 1, num=int(max_rows)).astype(np.int64)
    query, inside, target = query[take], inside[take], targets[take]
    in_catalog = (target >= 0) & (target < codes.shape[0])

    depths = []
    for depth in range(1, sid.levels + 1):
        buckets = Buckets(codes, sid.sizes, depth)
        keys_q = pack_keys(np.where(inside[:, None], query, 0)[:, :depth], sid.sizes[:depth])
        found = np.clip(np.searchsorted(buckets.uniq, keys_q), 0,
                        max(buckets.uniq.size - 1, 0))
        weight = buckets.counts[found]
        target_bucket = buckets.bucket_of_item[np.where(in_catalog, target, 0)]
        hit = inside & in_catalog & (found == target_bucket)
        recall = float(hit.mean()) if hit.size else 0.0
        uniform = float(buckets.counts.mean()) / codes.shape[0]
        weighted = float(weight[inside].mean()) if inside.any() else float("nan")
        depths.append(
            {
                "depth": depth,
                "rows": int(hit.size),
                "recall": recall,
                "uniform_base": uniform,
                "lift": recall / uniform if uniform else float("nan"),
                "size_biased_bucket": weighted,
                "lift_per_1000_candidates": (
                    recall / (weighted / 1000.0) if weighted and weighted == weighted else float("nan")
                ),
                "buckets": buckets.n_buckets,
            }
        )
    return {
        "split_dir": split_dir,
        "alignment": alignment,
        "rows": int(take.size),
        "queries_unmapped": int((~inside).sum()),
        "targets_outside_catalog": int((~in_catalog).sum()),
        "depths": depths,
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def fmt_int(value) -> str:
    return f"{int(value):,}"


def print_one(sid: Sid, coverage: dict, occupancy: list, lift: dict | None) -> None:
    print()
    print("=" * 78)
    print(f"SID {sid.name}   ({os.path.basename(sid.path)})")
    print("=" * 78)
    print(f"items           : {fmt_int(sid.n_items)}")
    print(f"levels          : {sid.levels}   sizes {sid.sizes.tolist()}")
    print(f"bits            : {coverage['total_bits']:.2f} total, "
          f"identity needs {coverage['identity_bits']:.2f}, "
          f"redundancy {coverage['redundant_bits']:+.2f}")
    print(f"injective at    : depth {coverage['injective_at'] or 'never'}")
    print(f"items per level-0 code : {coverage['items_per_level0_code']:,.2f}"
          f"   (paper's range {REFERENCE_ITEMS_PER_CODE[0]}-{REFERENCE_ITEMS_PER_CODE[1]})")

    print("\nQ1 coverage")
    print(f"  {'level':>5} {'size':>7} {'used':>7} {'util':>6} {'entropy':>9} "
          f"{'of log2':>8} {'ratio':>6}  note")
    for row in coverage["levels"]:
        note = "SUFFIX, not a semantic code" if row["looks_like_suffix"] else ""
        print(f"  {row['level']:>5} {fmt_int(row['size']):>7} {fmt_int(row['used']):>7} "
              f"{row['utilisation'] * 100:>5.1f}% {row['entropy_bits']:>9.4f} "
              f"{row['max_bits']:>8.4f} {row['entropy_ratio'] * 100:>5.1f}%  {note}")

    print("\nQ3/Q4 occupancy and tie ceiling")
    print(f"  {'depth':>5} {'buckets':>10} {'items/bkt':>10} {'mean g':>7} {'med g':>6} "
          f"{'p90 g':>6} {'max g':>7} {'ceil@1':>7} {'ceil@10':>8} {'ceil@100':>9}")
    for row in occupancy:
        print(f"  {row['depth']:>5} {fmt_int(row['buckets']):>10} "
              f"{row['items_per_bucket']:>10.2f} {row['mean_group']:>7.2f} "
              f"{row['median_group']:>6.0f} {row['p90_group']:>6.0f} "
              f"{fmt_int(row['largest_bucket']):>7} "
              f"{row['ceiling']['ceiling@1'] * 100:>6.2f}% "
              f"{row['ceiling']['ceiling@10'] * 100:>7.2f}% "
              f"{row['ceiling']['ceiling@100'] * 100:>8.2f}%")

    if lift is None:
        print("\nQ2 lift : skipped (no tokenised split for this SID)")
        return
    print(f"\nQ2 lift   split {lift['split_dir']}")
    align = lift["alignment"]
    print(f"  alignment: {fmt_int(align['placed'])} of {fmt_int(align['json_entries'])} "
          f"json entries placed on {fmt_int(align['catalog_items'])} catalogue rows "
          f"({fmt_int(align['unique_ids'])} unique ids)")
    print(f"  rows {fmt_int(lift['rows'])}, unmapped queries {fmt_int(lift['queries_unmapped'])}, "
          f"targets outside catalogue {fmt_int(lift['targets_outside_catalog'])}")
    print(f"  {'depth':>5} {'recall':>9} {'uniform':>9} {'lift':>10} "
          f"{'query bkt':>10} {'lift/1k':>8}")
    for row in lift["depths"]:
        per_k = row["lift_per_1000_candidates"] * 100.0
        print(f"  {row['depth']:>5} {row['recall'] * 100:>8.4f}% "
              f"{row['uniform_base'] * 100:>8.4f}% {row['lift']:>9.1f}x "
              f"{row['size_biased_bucket']:>10.2f} {per_k:>8.3f}")
    print("  lift/1k is percent of rows hit per 1,000 candidates, "
          "recall / (query_bkt / 1000) x 100;")


def verdict(sid: Sid, coverage: dict, occupancy: list, lift: dict | None) -> list:
    """Plain sentences, including the unwelcome ones."""
    out = []
    ceilings = occupancy[-1]["ceiling"] if occupancy else {}
    suffixes = [r for r in coverage["levels"] if r["looks_like_suffix"]]
    for row in suffixes:
        out.append(
            f"level {row['level']} is suffix-like, not a code: {fmt_int(row['size'])} "
            f"dense values but only {row['entropy_ratio'] * 100:.1f}% of its own "
            f"log2(size). That is the shape of a collision rank, and giving it equal "
            "weight in the mean vector adds noise."
        )
    if coverage["injective_at"] == sid.levels and sid.levels > 1:
        out.append(
            f"the last level makes the catalogue injective, so it is very close to an "
            f"identity code: {occupancy[-1]['ceiling']['ceiling@1'] * 100:.2f}% of items "
            "sit alone in their bucket and R@1 there is unconstrained by ties."
        )
    if coverage["injective_at"] and coverage["injective_at"] < sid.levels:
        out.append(
            f"the catalogue is fully identified at depth {coverage['injective_at']}, "
            f"so levels after it buy identity only."
        )
    if lift:
        lifts = [r["lift"] for r in lift["depths"]]
        peak = int(np.nanargmax(lifts)) + 1
        if peak < sid.levels:
            drop = (lifts[peak - 1] - lifts[-1]) / lifts[peak - 1] * 100
            out.append(
                f"lift peaks at depth {peak} ({lifts[peak - 1]:.1f}x) and ends at "
                f"{lifts[-1]:.1f}x, a {drop:.0f}% fall: the later levels split density, "
                "not behaviour."
            )
        else:
            out.append(f"lift is still rising at the last depth ({lifts[-1]:.1f}x).")
        c1 = ceilings.get("ceiling@1")
        if c1 and c1 < 0.05:
            out.append(
                f"R@1 ceiling is {c1 * 100:.2f}%, close enough to any plausible measured "
                "R@1 that the codes, not the head, would be the constraint."
            )
    if coverage["items_per_level0_code"] > REFERENCE_ITEMS_PER_CODE[1]:
        out.append(
            f"{coverage['items_per_level0_code']:,.0f} items per level-0 code is "
            f"{coverage['items_per_level0_code'] / REFERENCE_ITEMS_PER_CODE[1]:.0f}x the "
            "coarsest catalogue in the DiGR paper, so a first code says very little."
        )
    return out


def comparison(sids, coverages, occupancies, lifts) -> None:
    print()
    print("=" * 78)
    print("comparison at matched depth")
    print("=" * 78)
    print(f"  {'SID':<14} {'depth':>5} {'items/bkt':>10} {'lift':>9} {'lift/1k':>9} "
          f"{'ceil@1':>8}")
    for sid, occ, lift in zip(sids, occupancies, lifts):
        for row in occ:
            entry = None
            if lift is not None:
                entry = next(d for d in lift["depths"] if d["depth"] == row["depth"])
            lift_text = f"{entry['lift']:.1f}x" if entry else "-"
            per_k = f"{entry['lift_per_1000_candidates'] * 100:.3f}" if entry else "-"
            print(f"  {sid.name:<14} {row['depth']:>5} {row['items_per_bucket']:>10.2f} "
                  f"{lift_text:>9} {per_k:>9} "
                  f"{row['ceiling']['ceiling@1'] * 100:>7.2f}%")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sid", action="append", required=True,
                        metavar="NAME=PATH.json",
                        help="SID under test, repeatable")
    parser.add_argument("--split", action="append", default=[], metavar="NAME=DIR",
                        help="tokenised digr_data for this SID, repeatable, enables lift")
    parser.add_argument("--max_rows", type=int, default=None,
                        help="subsample the split; only for smoke tests, it moves lift")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="write the full report as JSON")
    args = parser.parse_args(argv)

    def pairs(values, what):
        out = []
        for value in values:
            if "=" not in value:
                raise SystemExit(f"--{what} wants NAME=PATH, got {value!r}")
            name, rest = value.split("=", 1)
            out.append((name.strip(), rest))
        return out

    sids, coverages, occupancies, lifts = [], [], [], []
    for name, path in pairs(args.sid, "sid"):
        try:
            ids, codes = load_sid_json(path)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"{path}: {exc}") from exc
        sid = Sid(name, path, ids, codes)
        sids.append(sid)
        coverage = coverage_report(sid)
        occupancy = occupancy_report(sid)
        coverages.append(coverage)
        occupancies.append(occupancy)

    splits = dict(pairs(args.split, "split"))
    for sid in sids:
        split_dir = splits.get(sid.name)
        lift = None
        if split_dir:
            lift = lift_report(sid, split_dir, max_rows=args.max_rows)
        lifts.append(lift)

    for sid, coverage, occupancy, lift in zip(sids, coverages, occupancies, lifts):
        print_one(sid, coverage, occupancy, lift)
        print("\nwhat this says")
        for line in verdict(sid, coverage, occupancy, lift):
            print(f"  - {line}")

    if len(sids) > 1:
        comparison(sids, coverages, occupancies, lifts)

    unknown = set(splits) - {sid.name for sid in sids}
    if unknown:
        print(f"\nnote: --split given for {sorted(unknown)} but no such --sid", file=sys.stderr)

    if args.json_out:
        payload = {
            "sids": [
                {
                    "name": sid.name,
                    "path": sid.path,
                    "n_items": sid.n_items,
                    "levels": sid.levels,
                    "sizes": sid.sizes.tolist(),
                    "coverage": coverage,
                    "occupancy": occupancy,
                    "lift": lift,
                    "verdict": verdict(sid, coverage, occupancy, lift),
                }
                for sid, coverage, occupancy, lift in zip(sids, coverages, occupancies, lifts)
            ]
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

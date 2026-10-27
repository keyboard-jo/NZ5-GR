"""Turn an RQ-VAE ``index_rqvae_*.json`` into the arrays the diagnostics want.

``rq_vae_cli.py`` writes one JSON object mapping item id to a list of codes. Every
other tool in this repo wants a dense ``int32`` array indexed by catalogue row, plus
a small vocab saying how wide each level is. This does the conversion and refuses to
guess when the shapes disagree.

    python "Plan 2/convert_rqvae_index.py" \
        --index local/2026-27-10-2-rq/test_output/index_rqvae_3x1024.json \
        --item_ids local/2026-20-10-2-pre/digr_data/catalog_item_ids.npy \
        --output_dir local/2026-27-10-2-rq --tag 2x8192

Writes ``codes_<tag>_aligned.npy`` (items x levels, int32) and
``vocab_<tag>.json``. The row order is the order of ``--item_ids``, which is the
catalogue order every other artefact uses, so row ``r`` is item id ``ids[r]``.

numpy and the standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def load_index(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def align(index: dict, item_ids: np.ndarray, verbose: bool = True) -> tuple:
    """Dense code array in catalogue order, plus the declared level widths."""
    ids = np.asarray(item_ids)
    if ids.ndim != 1:
        raise ValueError(f"item_ids must be 1-d, got shape {ids.shape}")
    if ids.size == 0:
        raise ValueError("item_ids is empty")

    levels = {len(v) for v in index.values()}
    if len(levels) != 1:
        raise ValueError(f"index holds ragged code lengths {sorted(levels)}")
    num_levels = levels.pop()
    if num_levels == 0:
        raise ValueError("index holds empty code lists")

    keyed = {int(k): v for k, v in index.items()}
    present = ids.size
    if not all(int(i) in keyed for i in ids):
        missing = [int(i) for i in ids if int(i) not in keyed][:5]
        raise KeyError(
            f"{present - sum(1 for i in ids if int(i) in keyed)} of {present} item ids "
            f"are absent from the index, first few: {missing}"
        )
    if len(keyed) != present:
        # extra keys are harmless as long as every catalogue item is covered, but say so
        if verbose:
            print(
                f"[align] index has {len(keyed)} keys for {present} catalogue items; "
                "the extra keys are ignored",
                file=sys.stderr,
            )

    codes = np.empty((ids.size, num_levels), dtype=np.int32)
    for row, item in enumerate(ids):
        codes[row] = np.asarray(keyed[int(item)], dtype=np.int32)

    if int(codes.min()) < 0:
        raise ValueError("index holds negative codes")

    sid_sizes = []
    for level in range(num_levels):
        observed = int(codes[:, level].max()) + 1
        used = int(np.unique(codes[:, level]).size)
        sid_sizes.append(observed)
        if verbose:
            print(
                f"[align] level {level}: max code {observed - 1}, "
                f"{used} of {observed} codes used ({100.0 * used / observed:.4f}%)"
            )
    return codes, sid_sizes


def vocab_of(sid_sizes: list) -> dict:
    return {"sid_sizes": [int(s) for s in sid_sizes], "num_levels": len(sid_sizes)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="index_rqvae_*.json from rq_vae_cli.py")
    parser.add_argument(
        "--item_ids",
        required=True,
        help="catalog_item_ids.npy, the catalogue order to align against",
    )
    parser.add_argument("--output_dir", default=None, help="defaults to the index's directory")
    parser.add_argument(
        "--tag",
        required=True,
        help="name for the outputs, so 2x8192 gives codes_2x8192_aligned.npy",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.index))
    os.makedirs(out_dir, exist_ok=True)

    codes, sid_sizes = align(load_index(args.index), np.load(args.item_ids), not args.quiet)
    codes_path = os.path.join(out_dir, f"codes_{args.tag}_aligned.npy")
    vocab_path = os.path.join(out_dir, f"vocab_{args.tag}.json")
    np.save(codes_path, codes)
    with open(vocab_path, "w", encoding="utf-8") as handle:
        json.dump(vocab_of(sid_sizes), handle)

    if not args.quiet:
        distinct = int(np.unique(codes, axis=0).shape[0])
        print(f"[align] {codes.shape[0]:,} items x {codes.shape[1]} levels, int32")
        print(
            f"[align] {distinct:,} distinct tuples "
            f"({100.0 * distinct / codes.shape[0]:.2f}% of items sit on their own)"
        )
        print(f"[align] wrote {codes_path}")
        print(f"[align] wrote {vocab_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

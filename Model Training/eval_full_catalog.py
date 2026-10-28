#!/usr/bin/env python
"""Score a DiGR checkpoint over the FULL catalog (all 4,783,154 items).

Why this exists
---------------
``--label_subset clicked`` trains and validates the softmax over only the items
that appear as a train/val target (504,404 in the current slice).  Recall from
that space is not comparable to a full-catalog number: the same target has
9.5x fewer distractors to outrank, so it looks better than it is.

``--mode infer`` already ranks over the full catalog, but it writes
``predictions.parquet`` instead of metrics.  This script reuses
``train_digr.evaluate()`` -- the exact fp32 ``item_logits`` path used during
training -- with a retriever built on all 4,783,154 catalog rows.  The reported
R@k / nDCG@k / MRR therefore land in the same space as the popularity baseline
and as a ``--label_subset full`` run.

Usage
-----
    python eval_full_catalog.py \
      --checkpoint /scratch/.../digr_0p6b_e3_full_sampled300k/final \
      --data_dir   /scratch/.../digr_data \
      --local_files_only

`--ks` defaults to `1,5,10,100,1000`.  The 100/1000 cut-offs are the ones that
say whether DiGR works as a candidate generator (4.78M -> some smaller pool that
a cheap reranker can handle) rather than as a standalone top-10 retriever.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_digr as T  # noqa: E402

# Reference numbers measured on the same val split, quoted so the comparison is
# in one place.  These are inputs, not results of this script.  Popularity was
# re-measured with the same metric code as everything else (see
# baselines_digr.py): R@10 1.3438%, not the 0.9644% an earlier ad-hoc run
# reported -- that run's R@1 matched but its ranks 2-10 did not.
REF_POPULARITY_R1 = 0.00439453125          # 0.4395%
# Written as a fraction on purpose: 1321 rows out of 98,304. A printed decimal
# of this value silently loses the last digit, and every lift ratio here divides
# by it.
REF_POPULARITY_R10 = 1321 / 98304          # 0.0134379069, i.e. 1.3438%
# co-click kNN at --history_window 20, the strongest form measured. The sweep
# 5 -> 10 -> 20 kept rising and window 20 also wins rank 1, so this is the bar.
REF_COCLICK_WINDOW = 20
REF_COCLICK = {1: 0.008057, 10: 0.037069, 100: 0.109873,
               1000: 0.253021, 10000: 0.468882}
REF_COCLICK_R10 = REF_COCLICK[10]
REF_POPULARITY = {1: 0.00439453125, 10: 1321 / 98304, 100: 0.047862,
                  1000: 0.147288, 10000: 0.368215}
# The frozen SID v1 run (`digr_0p6b_e3_full_sampled300k/final`), same 98,304 val
# rows, same catalogue, same metric code. This is the row a new configuration has
# to beat, and unlike the 504,404-label subset numbers below it IS comparable.
REF_FROZEN_V1 = {1: 0.007924, 10: 0.025940, 100: 0.075094,
                 1000: 0.216003, 10000: 0.497691}
# Retired: an earlier run scored on a 504,404-label subset. Kept only so a reader
# who saw the number in a log can identify it as different.



def build_logger() -> logging.Logger:
    logger = logging.getLogger("eval_full_catalog")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    return logger


def parse_cli() -> argparse.Namespace:
    # Start from train_digr's own defaults so load_for_inference() sees every
    # attribute it expects (attn_implementation, reinit_embeddings, lora_*, ...).
    args = T.parse_args([])
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="checkpoint dir with head_config.json + weights. Use the "
                        "`final` dir: the val-loss-best epoch is not necessarily "
                        "the best-ranking one")
    p.add_argument("--data_dir", required=True,
                   help="preprocessed dir from preprocess_digr.py")
    p.add_argument("--split", default="val", choices=["val", "train", "infer"])
    p.add_argument("--ks", default="1,5,10,100,1000",
                   help="comma-separated ranking cut-offs. 100 and 1000 answer a "
                        "different question from @10: not \"is the answer on the "
                        "first page\" but \"is it cheap to rerank from here\", i.e. "
                        "whether this model is usable as a candidate generator")
    p.add_argument("--batch_size", type=int, default=128,
                   help="eval batch; the [B, |I|] fp32 logit tensor is "
                        "B * 4.78M * 4 bytes (128 -> ~2.4 GB)")
    p.add_argument("--max_rows", type=int, default=None,
                   help="score only the first N rows (smoke test)")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--out_json", default=None, help="write the metrics here as JSON")
    cli = p.parse_args()

    args.checkpoint = cli.checkpoint
    args.data_dir = cli.data_dir
    args.eval_batch_size = cli.batch_size
    args.num_workers = cli.num_workers
    args.local_files_only = cli.local_files_only
    # `load_for_inference` does `args.model_name or meta.model_name`. Starting from
    # train_digr's defaults makes args.model_name truthy, so the checkpoint's own
    # model name would never be used and a mismatched backbone would be loaded over
    # the weights. Clearing it here lets the checkpoint speak for itself.
    args.model_name = None
    if cli.device:
        args.device = cli.device
    return args, cli


def check_data_dir_matches_checkpoint(meta, data_dir: str) -> None:
    """The checkpoint and `--data_dir` must describe the same dataset.

    Two different files couple them and nothing else validates the coupling. The
    token layout comes from `vocab.json` in `--data_dir`; the embedding table, the
    SID offsets and the item vectors come from the checkpoint. Score a v2
    checkpoint on a v1 directory and every id is in range but means something
    else, so the model reads a plausible sequence of the wrong tokens, predicts
    nearly the same item for every row, and reports a loss above ln K. That looks
    like a weak model, which is the worst thing a metric can look like, so this
    fails instead.

    Confirmed on 2026-10-28: the SID v2 checkpoint scored against the frozen v1
    `digr_data` returned R@10 0.0437% and loss 26.3504, against 3.6600% and 11.0133
    from the same checkpoint on its own directory.
    """
    path = os.path.join(data_dir, "vocab.json")
    if not os.path.isfile(path):
        raise SystemExit(f"{data_dir} has no vocab.json, so it is not a "
                         "preprocessed dataset directory.")
    with open(path, "r", encoding="utf-8") as f:
        here_vocab = T.Vocab.from_dict(json.load(f))

    def layout(v):
        return {
            "tokens": int(v.total_size),
            "SID sizes": list(v.sid_sizes),
            "gap bins": int(v.n_temporal),
            "feature blocks": len(v.feature_sizes),
            "tokens per event": int(v.event_width),
        }

    want, got = layout(meta.vocab), layout(here_vocab)
    differ = [k for k in want if want[k] != got[k]]
    if differ:
        raise SystemExit(
            "the checkpoint and --data_dir use different token layouts, so the "
            "model would read the wrong tokens:\n"
            f"  checkpoint  : {want}\n"
            f"  {data_dir}  : {got}\n"
            f"  differing   : {', '.join(differ)}\n"
            "Point --data_dir at the directory this checkpoint was trained on."
        )

    here = T.ItemCatalog.load(data_dir)
    if here.size != meta.catalog.size:
        raise SystemExit(
            f"{data_dir} has {here.size:,} catalog items but the checkpoint was "
            f"built over {meta.catalog.size:,}. Same items, different files."
        )
    if not np.array_equal(np.asarray(here.item_ids), np.asarray(meta.catalog.item_ids)):
        same = np.asarray(here.item_ids) == np.asarray(meta.catalog.item_ids)
        first = int(np.argmin(same)) if not bool(same.all()) else -1
        raise SystemExit(
            f"{data_dir} and the checkpoint disagree on catalog order "
            f"({int((~same).sum()):,} rows differ, first at row {first}). "
            "Targets are row indices, so this would score one catalog's labels "
            "against another catalog's items. Point --data_dir at the directory "
            "the checkpoint was trained on, or at one whose catalog_item_ids.npy "
            "is identical."
        )
    if not np.array_equal(np.asarray(here.sid_codes), np.asarray(meta.catalog.sid_codes)):
        raise SystemExit(
            f"{data_dir} and the checkpoint disagree on SID codes for at least one "
            "item. The item vector is built from the codes, so the scores would "
            "come from a different SID than the labels were counted on."
        )


@torch.inference_mode()
def main() -> int:
    args, cli = parse_cli()
    logger = build_logger()

    # evaluate() reads the module global at call time, so widening it here is the
    # whole change: same fp32 item_logits path, same conventions, wider topk.
    ks = tuple(sorted({int(x) for x in cli.ks.split(",") if x.strip()}))
    if not ks or ks[0] < 1:
        raise ValueError(f"--ks must be positive integers, got {cli.ks!r}")
    T.EVAL_KS = ks

    t0 = time.time()
    meta, retriever, device, amp_dtype, use_amp = T.load_for_inference(args, logger)
    full_catalog = int(retriever.catalog_size)
    expected = int(meta.catalog.size)
    if full_catalog != expected:
        raise RuntimeError(
            f"retriever built over {full_catalog} rows but the catalog has "
            f"{expected}; refusing to report a number that is not full-catalog"
        )

    split = T.load_split(args.data_dir, cli.split)
    if len(split) == 0:
        raise ValueError(f"split '{cli.split}' in {args.data_dir} is empty")
    check_data_dir_matches_checkpoint(meta, args.data_dir)

    lengths = np.asarray(split.lengths)
    empty = int((lengths == 0).sum())
    if empty:
        raise SystemExit(
            f"{empty:,} of {len(lengths):,} rows in '{cli.split}' carry no tokens. "
            "Those rows have no history for the model to read, so they score as "
            "one fixed prediction and drag every cut-off toward uniform."
        )

    indices = np.arange(len(split), dtype=np.int64)
    if cli.max_rows is not None:
        indices = indices[: max(1, int(cli.max_rows))]

    ds = T.TokenDataset(split, indices)
    loader = DataLoader(
        ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=T.make_collate(meta.vocab.pad_id),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    logger.info("=" * 78)
    logger.info("  DiGR evaluation over the FULL catalog")
    logger.info("=" * 78)
    logger.info(f"  checkpoint   : {args.checkpoint}")
    logger.info(f"  data_dir     : {args.data_dir}")
    logger.info(f"  model        : {args.model_name or meta.model_name} "
                f"({args.dtype}, lora={meta.use_lora})")
    logger.info(f"  vocab        : {int(meta.vocab.total_size):,} tokens, "
                f"SID {meta.vocab.sid_sizes} at offsets {meta.vocab.sid_offsets}")
    logger.info(f"  score scale  : 1/(L*tau) = 1/({int(meta.vocab.num_levels)} * "
                f"{meta.temperature}) = "
                f"{1.0 / (meta.vocab.num_levels * meta.temperature):.3f}")
    logger.info(f"  split        : {cli.split} ({len(indices):,} rows), "
                f"tokens/row mean {lengths[indices].mean():.2f}, "
                f"min {int(lengths[indices].min())}, "
                f"max {int(lengths[indices].max())}")
    logger.info(f"  catalog      : {full_catalog:,} items  "
                f"(uniform CE = ln K = {math.log(full_catalog):.3f})")
    logger.info(f"  batch / dev  : {args.eval_batch_size} / {device}")
    logger.info(f"  cut-offs     : {list(ks)}  (topk {max(ks)}, MRR@{T.MRR_K})")
    logger.info("  scoring full-catalog logits per row (no candidate list) ...")

    m = T.evaluate(retriever, loader, device, amp_dtype, use_amp)
    elapsed = time.time() - t0

    uniform = math.log(full_catalog)
    logger.info("-" * 78)
    logger.info(f"  rows {int(m['n']):,}   loss {m['loss']:.4f}   "
                f"uniform CE {uniform:.3f}   "
                f"({uniform - m['loss']:+.3f} nats, lower loss is better)")
    if m["loss"] > uniform:
        raise SystemExit(
            f"loss {m['loss']:.4f} is above ln K = {uniform:.3f}, so the model is "
            "worse than a uniform guess and the checkpoint is not predicting these "
            "rows at all. That is a broken comparison, not a weak model (see "
            "docs/results.md section 12). Nothing has been written."
        )
    for k in ks:
        rec = m[f"recall@{k}"]
        uni = k / full_catalog          # what a uniform guess would score
        logger.info(f"  R@{k:<5d} {100 * rec:8.4f}%   "
                    f"nDCG@{k:<5d} {m[f'ndcg@{k}']:.5f}   "
                    f"{rec / uni:7.1f}x uniform")
    logger.info(f"  MRR@{T.MRR_K:<5d}{m['mrr']:.5f}")
    logger.info("-" * 78)
    logger.info(f"  DiGR / co-click(window {REF_COCLICK_WINDOW}) / popularity, "
                "same rows and cut-offs:")
    logger.info(f"    {'k':>6}  {'DiGR':>9}  {'co-click':>9}  {'pop':>9}  "
                f"{'vs co-click':>11}")
    for k in ks:
        rec = m[f"recall@{k}"]
        cc = REF_COCLICK.get(k)
        pop = REF_POPULARITY.get(k)
        logger.info(
            f"    {k:>6}  {100 * rec:8.4f}%  "
            + (f"{100 * cc:8.4f}%  " if cc else f"{'':>9}  ")
            + (f"{100 * pop:8.4f}%  " if pop else f"{'':>9}  ")
            + (f"{rec / cc:10.2f}x" if cc else "")
        )
    logger.info("  vs the frozen SID v1 run (final), same rows and cut-offs:")
    logger.info("    " + "  ".join(
        f"R@{k} {m[f'recall@{k}'] / REF_FROZEN_V1[k]:.2f}x"
        for k in ks if REF_FROZEN_V1.get(k)))
    logger.info("-" * 78)
    lift1 = m["recall@1"] / REF_POPULARITY_R1
    lift10 = m["recall@10"] / REF_POPULARITY_R10
    beats = [k for k in ks if REF_COCLICK.get(k) and m[f"recall@{k}"] > REF_COCLICK[k]]
    loses = [k for k in ks if REF_COCLICK.get(k) and m[f"recall@{k}"] <= REF_COCLICK[k]]
    logger.info(f"  lift over popularity: R@1 {lift1:.2f}x  R@10 {lift10:.2f}x")
    logger.info(f"  beats co-click(window {REF_COCLICK_WINDOW}) at "
                f"k={beats or 'none'}; loses at k={loses or 'none'}")
    logger.info(f"  best band: "
                + ", ".join(
                    f"R@{k} {m[f'recall@{k}'] / REF_COCLICK[k]:.2f}x"
                    for k in ks if REF_COCLICK.get(k)))
    logger.info(f"  elapsed {elapsed / 60:.1f} min")

    if cli.out_json:
        vs_coclick = m["recall@10"] / REF_COCLICK_R10
        payload = {k: float(v) for k, v in m.items()}
        payload.update(
            {
                "checkpoint": args.checkpoint,
                "split": cli.split,
                "catalog_size": full_catalog,
                "ks": list(ks),
                "mrr_k": T.MRR_K,
                "lift_r1_vs_popularity": lift1,
                "lift_r10_vs_popularity": lift10,
                "lift_r10_vs_coclick": vs_coclick,
            }
        )
        os.makedirs(os.path.dirname(os.path.abspath(cli.out_json)), exist_ok=True)
        with open(cli.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"  wrote {cli.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

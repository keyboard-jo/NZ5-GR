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
      --checkpoint /scratch/.../digr_resume_20261025_113432/best \
      --data_dir   /scratch/.../digr_data \
      --local_files_only
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
# in one place.  These are inputs, not results of this script.
REF_POPULARITY_R1 = 0.00439453125          # 0.4394%, popularity over 4,783,154
REF_POPULARITY_R10 = 0.0096435546875       # 0.9644%, popularity over 4,783,154
REF_SUBSET_R1 = 0.0083                     # 0.83%, clicked run over 504,404
REF_SUBSET_R10 = 0.0327                    # 3.27%, clicked run over 504,404
REF_SUBSET_LOSS = 9.7457


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
                   help="checkpoint dir with head_config.json + weights (use 'best')")
    p.add_argument("--data_dir", required=True,
                   help="preprocessed dir from preprocess_digr.py")
    p.add_argument("--split", default="val", choices=["val", "train", "infer"])
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
    if cli.device:
        args.device = cli.device
    return args, cli


@torch.inference_mode()
def main() -> int:
    args, cli = parse_cli()
    logger = build_logger()

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
    logger.info(f"  split        : {cli.split} ({len(indices):,} rows)")
    logger.info(f"  catalog      : {full_catalog:,} items  "
                f"(uniform CE = ln K = {math.log(full_catalog):.3f})")
    logger.info(f"  batch / dev  : {args.eval_batch_size} / {device}")
    logger.info("  scoring full-catalog logits per row (no candidate list) ...")

    m = T.evaluate(retriever, loader, device, amp_dtype, use_amp)
    elapsed = time.time() - t0

    uniform = math.log(full_catalog)
    logger.info("-" * 78)
    logger.info(f"  rows {int(m['n']):,}   loss {m['loss']:.4f}   "
                f"({uniform / m['loss']:.3f}x better than uniform)")
    logger.info(f"  R@1    {100 * m['recall@1']:.4f}%      nDCG@1  {m['ndcg@1']:.5f}")
    logger.info(f"  R@5    {100 * m['recall@5']:.4f}%      nDCG@5  {m['ndcg@5']:.5f}")
    logger.info(f"  R@10   {100 * m['recall@10']:.4f}%     nDCG@10 {m['ndcg@10']:.5f}")
    logger.info(f"  MRR    {m['mrr']:.5f}")
    logger.info("-" * 78)
    logger.info("  reference, same val split:")
    logger.info(f"    popularity, full catalog   R@1 {100 * REF_POPULARITY_R1:.4f}%  "
                f"R@10 {100 * REF_POPULARITY_R10:.4f}%")
    logger.info(f"    this ckpt, 504,404 subset  R@1 {100 * REF_SUBSET_R1:.4f}%  "
                f"R@10 {100 * REF_SUBSET_R10:.4f}%  loss {REF_SUBSET_LOSS:.4f}")
    logger.info(f"    (subset numbers came from the training log and are NOT "
                f"comparable; the lines above are)")
    logger.info("-" * 78)
    lift1 = m["recall@1"] / REF_POPULARITY_R1
    lift10 = m["recall@10"] / REF_POPULARITY_R10
    verdict = "BEATS popularity" if lift10 > 1.0 else "does NOT beat popularity"
    logger.info(f"  lift over popularity: R@1 {lift1:.2f}x  R@10 {lift10:.2f}x  -> {verdict}")
    logger.info(f"  elapsed {elapsed / 60:.1f} min")

    if cli.out_json:
        payload = {k: float(v) for k, v in m.items()}
        payload.update(
            {
                "checkpoint": args.checkpoint,
                "split": cli.split,
                "catalog_size": full_catalog,
                "lift_r1_vs_popularity": lift1,
                "lift_r10_vs_popularity": lift10,
            }
        )
        os.makedirs(os.path.dirname(os.path.abspath(cli.out_json)), exist_ok=True)
        with open(cli.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"  wrote {cli.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
preprocess_sequences_cli.py
=============================================================================
Production-Grade Offline Sequence Preprocessing for Stage 1 HSTU + ItemDNN.
Transforms raw nested interaction sequences (seq.parquet) into flat, high-performance
training and validation sample tables for PyTorch DataLoaders.
=============================================================================
"""

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("preprocess_sequences")


def parse_action_token(act: Any) -> int:
    """Maps raw action_type to 0: PAD, 1: Exposure, 2: Click, 3: UNK."""
    if act is None:
        return 3
    try:
        if isinstance(act, (int, np.integer)):
            v = int(act)
        elif isinstance(act, float):
            if np.isnan(act):
                return 3
            v = int(act)
        elif isinstance(act, str):
            act_clean = act.strip()
            if not act_clean or act_clean.lower() in ("none", "null", "nan"):
                return 3
            v = int(float(act_clean))
        else:
            return 3

        if v == 1:
            return 2  # Click
        if v == 0:
            return 1  # Exposure
        return 3      # Unexpected int -> UNK
    except (ValueError, TypeError):
        return 3


def process_sequence_chunk(
    chunk_sequences: List[Any],
    item_to_idx: Dict[int, int],
    max_seq_len: int,
    min_history: int,
    max_examples_per_user: int,
    split_mode: str,
    val_click_only: bool,
    is_valid_chunk: bool,
) -> Tuple[List[List[int]], List[List[int]], List[int], Dict[str, int]]:
    hist_items_list: List[List[int]] = []
    hist_acts_list: List[List[int]] = []
    targets_list: List[int] = []

    stats = {
        "sequences_checked": 0,
        "sequences_valid": 0,
        "sequences_skipped_short": 0,
        "total_interactions": 0,
        "total_clicks": 0,
        "total_exposures": 0,
        "total_nulls": 0,
        "samples_generated": 0,
    }

    for raw_seq in chunk_sequences:
        stats["sequences_checked"] += 1
        if raw_seq is None or not isinstance(raw_seq, (list, np.ndarray)) or len(raw_seq) == 0:
            continue

        item_seq: List[int] = []
        act_seq: List[int] = []

        for interaction in raw_seq:
            if isinstance(interaction, dict):
                iid = interaction.get("item_id")
                act = interaction.get("action_type")
            else:
                iid, act = interaction, 0

            if iid is None:
                continue

            try:
                mapped = item_to_idx.get(int(iid))
            except (TypeError, ValueError):
                continue

            if mapped is None:
                continue

            token = parse_action_token(act)
            item_seq.append(mapped)
            act_seq.append(token)

            stats["total_interactions"] += 1
            if token == 2:
                stats["total_clicks"] += 1
            elif token == 1:
                stats["total_exposures"] += 1
            elif token == 3:
                stats["total_nulls"] += 1

        seq_len = len(item_seq)
        if seq_len < min_history + 1:
            stats["sequences_skipped_short"] += 1
            continue

        stats["sequences_valid"] += 1

        if split_mode == "leave_one_out":
            if is_valid_chunk:
                last_idx = seq_len - 1
                if (not val_click_only) or (act_seq[last_idx] == 2):
                    start = max(0, last_idx - max_seq_len)
                    hist_items_list.append(item_seq[start:last_idx])
                    hist_acts_list.append(act_seq[start:last_idx])
                    targets_list.append(item_seq[last_idx])
                    stats["samples_generated"] += 1
            else:
                candidate_ends = [i for i in range(min_history, seq_len - 1) if act_seq[i] == 2]
                if max_examples_per_user > 0 and len(candidate_ends) > max_examples_per_user:
                    candidate_ends = candidate_ends[-max_examples_per_user:]

                for end in candidate_ends:
                    start = max(0, end - max_seq_len)
                    hist_items_list.append(item_seq[start:end])
                    hist_acts_list.append(act_seq[start:end])
                    targets_list.append(item_seq[end])
                    stats["samples_generated"] += 1
        else:
            candidate_ends = [
                i for i in range(min_history, seq_len)
                if (not is_valid_chunk or not val_click_only or act_seq[i] == 2)
                and (is_valid_chunk or act_seq[i] == 2)
            ]
            if is_valid_chunk:
                if candidate_ends:
                    end = candidate_ends[-1]
                    start = max(0, end - max_seq_len)
                    hist_items_list.append(item_seq[start:end])
                    hist_acts_list.append(act_seq[start:end])
                    targets_list.append(item_seq[end])
                    stats["samples_generated"] += 1
            else:
                if max_examples_per_user > 0 and len(candidate_ends) > max_examples_per_user:
                    candidate_ends = candidate_ends[-max_examples_per_user:]

                for end in candidate_ends:
                    start = max(0, end - max_seq_len)
                    hist_items_list.append(item_seq[start:end])
                    hist_acts_list.append(act_seq[start:end])
                    targets_list.append(item_seq[end])
                    stats["samples_generated"] += 1

    return hist_items_list, hist_acts_list, targets_list, stats


def _worker_wrapper(payload: Tuple) -> Tuple:
    return process_sequence_chunk(*payload)


def export_to_parquet(
    hist_items: List[List[int]],
    hist_actions: List[List[int]],
    targets: List[int],
    output_path: str,
    row_group_size: int = 100_000,
    compression: str = "zstd",
):
    logger.info("Writing %s samples to %s (compression=%s)...",
                f"{len(targets):,}", output_path, compression)
    t0 = time.time()
    if HAS_PYARROW:
        item_arrow = pa.array(hist_items, type=pa.list_(pa.int32()))
        action_arrow = pa.array(hist_actions, type=pa.list_(pa.int8()))
        target_arrow = pa.array(targets, type=pa.int32())

        table = pa.Table.from_arrays(
            [item_arrow, action_arrow, target_arrow],
            names=["history_items", "history_actions", "target"],
        )
        pq.write_table(table, output_path, row_group_size=row_group_size, compression=compression)
    else:
        df = pd.DataFrame({
            "history_items": hist_items,
            "history_actions": hist_actions,
            "target": np.array(targets, dtype=np.int32),
        })
        df.to_parquet(output_path, index=False, compression=compression)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("Wrote %s (%.2f MB) in %.2f seconds.", output_path, file_size_mb, time.time() - t0)


def main():
    parser = argparse.ArgumentParser(description="Preprocess user sequences for Stage 1 ItemDNN + HSTU.")
    parser.add_argument("--items_path", type=str, default="canonical_items.parquet", help="Path to item catalog parquet")
    parser.add_argument("--seq_path", type=str, default="seq.parquet", help="Path to raw sequence parquet")
    parser.add_argument("--output_dir", type=str, default="./preprocessed_data", help="Output directory")
    parser.add_argument("--item_id_col", type=str, default="item_id")
    parser.add_argument("--seq_col", type=str, default="seq")
    parser.add_argument("--max_seq_len", type=int, default=100)
    parser.add_argument("--min_history", type=int, default=1)
    parser.add_argument("--max_examples_per_user", type=int, default=10, help="Cap click targets per user (0=all)")
    parser.add_argument("--split_mode", type=str, default="leave_one_out", choices=["leave_one_out", "user_split"])
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--val_click_only", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--compression", type=str, default="zstd", choices=["zstd", "snappy", "gzip", "none"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isfile(args.items_path):
        logger.error("Items file not found: %s", args.items_path)
        sys.exit(1)
    if not os.path.isfile(args.seq_path):
        logger.error("Sequence file not found: %s", args.seq_path)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    logger.info("Indexing item catalog from: %s", args.items_path)
    items_df = pd.read_parquet(args.items_path, columns=[args.item_id_col])
    unique_ids = items_df[args.item_id_col].drop_duplicates().to_numpy(dtype=np.int64)
    item_to_idx = {int(iid): int(i + 1) for i, iid in enumerate(unique_ids)}
    num_items = len(unique_ids)

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump({
            "num_items": num_items,
            "max_seq_len": args.max_seq_len,
            "min_history": args.min_history,
            "max_examples_per_user": args.max_examples_per_user,
            "action_vocab": {0: "PAD", 1: "Exposure", 2: "Click", 3: "UNK"},
            "split_mode": args.split_mode,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)

    logger.info("Loading user sequences from: %s...", args.seq_path)
    seq_df = pd.read_parquet(args.seq_path, columns=[args.seq_col])
    raw_seqs = seq_df[args.seq_col].tolist()
    total_seqs = len(raw_seqs)

    if args.split_mode == "user_split":
        indices = np.arange(total_seqs)
        np.random.shuffle(indices)
        split_point = int(total_seqs * (1.0 - args.val_ratio))
        train_seqs = [raw_seqs[i] for i in indices[:split_point]]
        val_seqs = [raw_seqs[i] for i in indices[split_point:]]
    else:
        train_seqs = raw_seqs
        val_seqs = raw_seqs

    def run_parallel(seqs: List[Any], is_val: bool, desc: str):
        chunk_size = math.ceil(len(seqs) / max(1, args.num_workers))
        chunks = [seqs[i: i + chunk_size] for i in range(0, len(seqs), chunk_size)]
        payloads = [
            (c, item_to_idx, args.max_seq_len, args.min_history, args.max_examples_per_user,
             args.split_mode, args.val_click_only, is_val)
            for c in chunks
        ]
        logger.info("[%s] Processing %d chunks with %d CPU workers...", desc, len(payloads), args.num_workers)
        t_start = time.time()

        all_items, all_acts, all_targets = [], [], []
        agg_stats = {k: 0 for k in ["sequences_checked", "sequences_valid", "sequences_skipped_short",
                                   "total_interactions", "total_clicks", "total_exposures", "total_nulls", "samples_generated"]}

        with mp.Pool(processes=args.num_workers) as pool:
            for items, acts, targets, stats in pool.map(_worker_wrapper, payloads):
                all_items.extend(items)
                all_acts.extend(acts)
                all_targets.extend(targets)
                for k in agg_stats:
                    agg_stats[k] += stats[k]

        logger.info("[%s] Done in %.2f s | Produced %s samples.", desc, time.time() - t_start, f"{len(all_targets):,}")
        return all_items, all_acts, all_targets, agg_stats

    train_items, train_acts, train_targets, train_stats = run_parallel(train_seqs, is_val=False, desc="TRAIN")
    val_items, val_acts, val_targets, val_stats = run_parallel(val_seqs, is_val=True, desc="VALIDATION")

    export_to_parquet(train_items, train_acts, train_targets, os.path.join(args.output_dir, "train_samples.parquet"), compression=args.compression)
    export_to_parquet(val_items, val_acts, val_targets, os.path.join(args.output_dir, "val_samples.parquet"), compression=args.compression)

    print("\n" + "=" * 60)
    print("      DATASET PREPROCESSING & TELEMETRY SUMMARY")
    print("=" * 60)
    print(f"Total Unique Items in Catalog:   {num_items:>15,}")
    print(f"Total User Sequences Checked:    {total_seqs:>15,}")
    print(f"Total Interactions Encountered:  {train_stats['total_interactions']:>15,}")
    print(f"  - Confirmed Exposures (1):     {train_stats['total_exposures']:>15,} ({100 * train_stats['total_exposures'] / max(1, train_stats['total_interactions']):.2f}%)")
    print(f"  - Confirmed Clicks (2):        {train_stats['total_clicks']:>15,} ({100 * train_stats['total_clicks'] / max(1, train_stats['total_interactions']):.2f}%)")
    print(f"  - Telemetry UNK / Drops (3):   {train_stats['total_nulls']:>15,} ({100 * train_stats['total_nulls'] / max(1, train_stats['total_interactions']):.2f}%)")
    print("-" * 60)
    print(f"Final Train Samples (Clicks):    {len(train_targets):>15,}")
    print(f"Final Validation Samples:        {len(val_targets):>15,}")
    print(f"Output Artifacts Saved In:       {args.output_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""Is DiGR's early-k loss on crowded rows a COUNTING effect or a SCORING effect?

The measurement
---------------
DiGR scores `s_i = (1/tau) h . V_i` with `V_i = (1/L) sum_l E_l[c_il]`, so every
item sharing a SID tuple has **exactly** the same score. 53.51% of the catalogue
shares a 2-code tuple, and the measured `R@1` on the v2 dump is 2.427% when the
target is alone in its tuple against 0.262% when it is not (9.3x). Co-click, which
has no codebook at all, shows 1.1x on the same rows, so the gap belongs to DiGR's
item representation rather than to the rows.

Two very different fixes fit that fact:

  counting  `rank = 1 + #{i : s_i > s_t}` counts items. Every tuple ranked above
            the target donates its whole size to the count, so a crowded code
            space inflates the rank arithmetically. Fix: finer tuples, i.e. a
            deeper or wider SID. Item scoring is irrelevant.

  scoring   the model does not put the target's region on top often enough when
            the region is crowded. Fix: better item scoring (per-level weights, a
            non-linear composition). A finer codebook will not help.

They are separable because the region rank is computable on its own. With
`s_r` the score of region r, `region_rank = 1 + #{r : s_r > s_rt}` is exactly the
item rank an INJECTIVE codebook would produce from the same scores. So the
counterfactual "what if every tuple held one item" is free: same scores, sizes set
to 1. `region_lib` does that arithmetic and is unit tested against brute force.

What this prints
----------------
Per bucket-size group: rows, share, item `R@k`, region `R@k`, mean ranks, and the
mean inflation `item_rank - region_rank`. Then the overall counterfactual gain and
a verdict naming counting, scoring, or a mixture.

Usage
-----
    python "Plan 3/region_rank.py" \
      --checkpoint ~/NZ5/ablation/runs/digr_abl_sidv2_featv1_e2/best \
      --data_dir   ~/NZ5/ablation/digr_data_a \
      --out_json   ~/NZ5/ablation/region_rank_a.json \
      --dump_path  ~/NZ5/ablation/region_rank_a.npz

Re-print the table from a previous dump, no GPU needed:

    python "Plan 3/region_rank.py" --report_only ~/NZ5/ablation/region_rank_a.npz

Cost is one pass over the distinct tuples rather than over the catalogue: 3.0M
rows instead of 4.78M for the 2x8192 SID, a few minutes on one GPU, against the
1.8 min a full-catalogue eval takes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Model Training"))

import region_lib as RL  # noqa: E402

import train_digr as T  # noqa: E402
from eval_full_catalog import check_data_dir_matches_checkpoint  # noqa: E402

DEFAULT_KS = (1, 10, 100, 1000)


def build_logger():
    import logging

    logger = logging.getLogger("region_rank")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    return logger


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", default=None,
                   help="DiGR checkpoint dir (head_config.json + weights)")
    p.add_argument("--data_dir", default=None,
                   help="the preprocessed dir the checkpoint was trained on")
    p.add_argument("--split", default="val", choices=["val", "train"])
    p.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    p.add_argument("--batch_size", type=int, default=256,
                   help="rows per score block; the [B, region_chunk] fp32 tensor "
                        "is B * chunk * 4 bytes (256 x 131072 -> ~134 MB)")
    p.add_argument("--region_chunk", type=int, default=131072,
                   help="distinct tuples per GEMM block")
    p.add_argument("--max_rows", type=int, default=None,
                   help="score only the first N rows (smoke test; makes the "
                        "numbers not comparable to anything)")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--region_dtype", default="fp32", choices=["fp32", "bf16"],
                   help="bf16 halves the tuple matrix (6.1 GB instead of 12.3 for "
                        "3.0M x 1024) and is what the inferences path uses, but it "
                        "rounds near-ties between distinct tuples; fp32 matches the "
                        "fp32 item_logits path that produced the dumps")
    p.add_argument("--out_json", default=None)
    p.add_argument("--dump_path", default=None,
                   help="write per-row item_rank / region_rank / g here as npz, so "
                        "the table can be rebuilt without a GPU")
    p.add_argument("--check_dump", default=None,
                   help="an existing DiGR --dump_ranks npz. The item rank computed "
                        "here must reproduce it, which is the end-to-end check that "
                        "the region sizes are aligned with the catalogue")
    p.add_argument("--report_only", default=None,
                   help="skip the GPU entirely and re-print the table from a dump "
                        "written by --dump_path")
    cli = p.parse_args()
    if cli.report_only is None and (not cli.checkpoint or not cli.data_dir):
        p.error("--checkpoint and --data_dir are required unless --report_only is given")
    cli.ks = tuple(sorted({int(x) for x in str(cli.ks).split(",") if x.strip()}))
    if not cli.ks or cli.ks[0] < 1:
        p.error(f"--ks must be positive integers, got {cli.ks!r}")
    return cli


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def print_table(summary: Dict[str, object], logger) -> None:
    ks: Tuple[int, ...] = tuple(summary["ks"])  # type: ignore[arg-type]
    logger.info("-" * 96)
    logger.info(f"  {int(summary['n_rows']):,} rows, mean tuple size seen by the "
                f"target {float(summary['g_mean']):.2f}, "
                f"{100 * float(summary['inflated_share']):.2f}% of rows have at "
                "least one tuple ranked above the target's")
    logger.info("-" * 96)
    head = (f"  {'tuple size':>10}  {'rows':>7}  {'share':>6}  {'g':>6}  "
            + "  ".join(f"{'item R@' + str(k):>9}" for k in ks[:3])
            + "  " + "  ".join(f"{'inj R@' + str(k):>9}" for k in ks[:3])
            + f"  {'inflation':>9}")
    logger.info(head)
    for row in summary["groups"]:  # type: ignore[union-attr]
        line = (f"  {row['group']:>10}  {row['rows']:>7,}  "
                f"{100 * row['share']:>5.2f}%  {row['g_mean']:>6.2f}  ")
        line += "  ".join(f"{100 * row[f'item_r{k}']:>8.3f}%" for k in ks[:3])
        line += "  " + "  ".join(f"{100 * row[f'region_r{k}']:>8.3f}%" for k in ks[:3])
        line += f"  {row['inflation']:>9.1f}"
        logger.info(line)
    logger.info("-" * 96)
    logger.info("  item R@k is measured. inj R@k is the SAME tuple scores ranked as "
                "if every tuple held one item,")
    logger.info("  i.e. what an injective codebook would reach without improving a "
                "single score.")
    logger.info("-" * 96)
    for k in ks:
        item = float(summary[f"item_r{k}"])
        inj = float(summary[f"region_r{k}"])
        gain = float(summary[f"counterfactual_gain_{k}"])
        logger.info(f"    R@{k:<5d} measured {100 * item:8.4f}%   "
                    f"injective ceiling {100 * inj:8.4f}%   "
                    f"({gain:.2f}x, +{100 * (inj - item):.2f} pp)")
    logger.info("-" * 96)


def report(summary: Dict[str, object], logger) -> None:
    print_table(summary, logger)
    logger.info("  " + RL.verdict(summary))
    logger.info("-" * 96)


# --------------------------------------------------------------------------
# region vectors
# --------------------------------------------------------------------------
def region_vectors(model, regions: RL.Regions, dtype: torch.dtype) -> torch.Tensor:
    """`V` over the DISTINCT tuples, so a 3.0M-row GEMM instead of 4.78M.

    Same arithmetic as `SequentialRecModel.precompute_vectors`, including the
    per-level normalisation when the checkpoint used `l2_normalize`, because the
    item rank this script reproduces was produced by that exact path. The only
    change is which rows the codes are taken from.
    """
    w = T._embedding_weight(model.backbone)
    codes = torch.as_tensor(regions.codes, device=w.device, dtype=torch.long)
    d = int(w.shape[1])
    out = torch.zeros(int(regions.n_regions), d, dtype=dtype, device=w.device)
    step = max(1, min(int(model.logit_chunk), 65536))
    n = int(regions.n_regions)
    for level in range(codes.shape[1]):
        idx = codes[:, level]
        for start in range(0, n, step):
            block = w.index_select(0, idx[start:start + step]).float()
            if getattr(model, "l2_normalize", False):
                block = torch.nn.functional.normalize(block, dim=-1)
            out[start:start + step] += block.to(dtype)
    out /= float(codes.shape[1])
    return out


def count_above(H: torch.Tensor, V: torch.Tensor, tgt_score: torch.Tensor,
                counts: torch.Tensor, temperature: float,
                region_chunk: int, row_batch: int, logger=None):
    """Count tuples, and items, scoring strictly above each row's target.

    Two counts from one pass, because they differ only in the weight:

        above_regions = #{r : s_r > s_t}
        above_items   = sum over those r of size_r

    `rank = 1 + above_items` is the rank `item_logits` computes, since every
    member of a tuple carries the target's own score and so is never counted as
    above. `rank = 1 + above_regions` is the same quantity for an injective
    codebook, which is the counterfactual this whole script exists to measure.

    Strict `>` on both: the favourable tie convention every number in this
    project is reported under, where a tie is never charged to the target.

    Chunked over tuples and rows so the largest live tensor is
    `row_batch * region_chunk` floats.
    """
    n_rows = int(H.shape[0])
    n_reg = int(V.shape[0])
    above_regions = torch.zeros(n_rows, dtype=torch.int64, device=H.device)
    above_items = torch.zeros(n_rows, dtype=torch.int64, device=H.device)
    chunk = max(1, int(region_chunk))
    for c, start in enumerate(range(0, n_reg, chunk)):
        stop = min(start + chunk, n_reg)
        block = V[start:stop]
        blk_counts = counts[start:stop]
        for b0 in range(0, n_rows, row_batch):
            b1 = min(b0 + row_batch, n_rows)
            scores = (H[b0:b1] @ block.t()) / temperature
            hit = scores > tgt_score[b0:b1].unsqueeze(1)
            above_regions[b0:b1] += hit.sum(dim=1).to(torch.int64)
            above_items[b0:b1] += (hit.to(block.dtype) @ blk_counts).to(torch.int64)
        if logger is not None and (c % 4 == 0 or stop == n_reg):
            logger.info(f"  tuples {stop:,}/{n_reg:,}")
    return above_regions, above_items


@torch.inference_mode()
def run(cli: argparse.Namespace, logger) -> Dict[str, object]:
    if cli.device:
        torch.device(cli.device)

    args = T.parse_args([])
    args.checkpoint = cli.checkpoint
    args.data_dir = cli.data_dir
    args.eval_batch_size = cli.batch_size
    args.num_workers = cli.num_workers
    args.local_files_only = cli.local_files_only
    # Let the checkpoint speak for itself, exactly as eval_full_catalog does.
    args.model_name = None
    if cli.device:
        args.device = cli.device

    meta, model, device, amp_dtype, use_amp = T.load_for_inference(args, logger)
    check_data_dir_matches_checkpoint(meta, cli.data_dir)

    codes = np.asarray(meta.catalog.sid_codes)
    sizes = list(meta.vocab.sid_sizes)
    if codes.shape[1] != len(sizes):
        raise SystemExit(
            f"the catalogue has {codes.shape[1]} code columns but vocab.json "
            f"declares {len(sizes)} levels; these are different SID vintages"
        )
    regions = RL.Regions(codes, sizes)
    logger.info("=" * 96)
    logger.info("  DiGR region multiplicity: is the crowded-row loss counting or scoring?")
    logger.info("=" * 96)
    logger.info(f"  checkpoint   : {cli.checkpoint}")
    logger.info(f"  data_dir     : {cli.data_dir}")
    logger.info(f"  SID          : widths {sizes}, {int(regions.n_regions):,} distinct "
                f"tuples over {int(regions.n_items):,} items "
                f"({regions.items_per_region:.2f} items/tuple)")
    logger.info(f"  collisions   : {100 * regions.shared_fraction:.2f}% of items share a "
                f"tuple, largest {int(regions.counts.max())}, "
                f"{'injective' if regions.injective else 'NOT injective'}")

    split = T.load_split(cli.data_dir, cli.split)
    n_rows = len(split)
    indices = np.arange(n_rows, dtype=np.int64)
    if cli.max_rows is not None:
        indices = indices[: max(1, int(cli.max_rows))]
    targets = np.asarray(split.targets)[indices].astype(np.int64)
    if (targets < 0).any() or (targets >= regions.n_items).any():
        raise SystemExit("a target is outside the catalogue; wrong --data_dir")
    target_region = regions.region_of_item[targets]
    g = regions.counts[target_region]
    if not (g >= 1).all():
        raise SystemExit("a target mapped to a region with no members")

    t0 = time.time()
    logger.info(f"  split        : {cli.split} ({len(indices):,} rows), "
                f"mean tuple size seen by the target {g.mean():.2f}")
    logger.info("  building region vectors ...")
    vdtype = torch.float32 if cli.region_dtype == "fp32" else torch.bfloat16
    V = region_vectors(model, regions, vdtype)
    logger.info(f"  region matrix: {tuple(V.shape)} {V.dtype} "
                f"({V.numel() * V.element_size() / 1e9:.1f} GB)")

    # Pass 1: encode every row once and keep the separator states. 98,304 x 1024
    # fp32 is 0.4 GB, which buys an outer loop over region chunks instead of
    # re-reading the region matrix for every batch.
    ds = T.TokenDataset(split, indices)
    loader = DataLoader(
        ds, batch_size=cli.batch_size, shuffle=False,
        collate_fn=T.make_collate(meta.vocab.pad_id),
        num_workers=cli.num_workers, pin_memory=(device.type == "cuda"),
    )
    d = int(V.shape[1])
    H = torch.zeros(len(indices), d, dtype=torch.float32, device=device)
    tgt_score = torch.zeros(len(indices), dtype=torch.float32, device=device)
    tgt_region = torch.as_tensor(target_region, device=device, dtype=torch.long)
    vf = V.float()
    done = 0
    for ids, mask, tgts, rows in loader:
        ids = ids.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            h = model.encode(ids, mask)
        h = h.float()
        if getattr(model, "l2_normalize", False):
            h = torch.nn.functional.normalize(h, dim=-1)
        H[rows] = h
        # The target's own region score defines the threshold for every count.
        tgt_score[rows] = (h * vf.index_select(0, tgt_region[rows])).sum(-1) / model.temperature
        done += int(ids.shape[0])
        if done % (cli.batch_size * 16) == 0 or done == len(indices):
            logger.info(f"  encoded {done:,}/{len(indices):,} rows "
                        f"({(time.time() - t0) / 60:.1f} min)")

    # Pass 2: count regions and items above each row's threshold.
    counts = torch.as_tensor(regions.counts, device=device, dtype=torch.float32)
    logger.info(f"  counting over {int(regions.n_regions):,} tuples in "
                f"{math.ceil(int(regions.n_regions) / max(1, cli.region_chunk))} "
                f"chunks of {int(cli.region_chunk):,} ...")
    above_regions, above_items = count_above(
        H, vf, tgt_score, counts, float(model.temperature),
        cli.region_chunk, cli.batch_size, logger)
    del vf, V

    region_rank = (above_regions + 1).cpu().numpy().astype(np.int64)
    item_rank = (above_items + 1).cpu().numpy().astype(np.int64)
    if (region_rank > item_rank).any():
        raise SystemExit(
            "a region rank came out larger than the item rank, which cannot "
            "happen: every region ranked above the target adds at least one item "
            "to the count. The sizes are not aligned with the tuples."
        )

    summary = RL.summarise(item_rank, region_rank, g, cli.ks)
    return {
        "summary": summary,
        "item_rank": item_rank,
        "region_rank": region_rank,
        "g": g,
        "targets": targets,
        "n_rows": n_rows,
        "sid_sizes": sizes,
        "n_regions": int(regions.n_regions),
        "n_items": int(regions.n_items),
        "items_per_region": float(regions.items_per_region),
        "shared_fraction": float(regions.shared_fraction),
        "largest": int(regions.counts.max()),
        "elapsed_min": (time.time() - t0) / 60.0,
        "checkpoint": os.path.abspath(cli.checkpoint),
        "data_dir": os.path.abspath(cli.data_dir),
        "split": cli.split,
    }


def main() -> int:
    cli = parse_cli()
    logger = build_logger()

    if cli.report_only:
        z = np.load(cli.report_only)
        summary = RL.summarise(z["item_rank"], z["region_rank"], z["g"], cli.ks)
        logger.info(f"  re-analysing {cli.report_only} "
                    f"({int(z['n_rows']):,} rows, SID {z['sid_sizes'].tolist()})")
        report(summary, logger)
        return 0

    out = run(cli, logger)
    summary = out["summary"]
    report(summary, logger)
    logger.info(f"  elapsed {out['elapsed_min']:.1f} min")

    if cli.check_dump:
        d = np.load(cli.check_dump)
        ref_rank, ref_tgt = d["rank"].astype(np.int64), d["targets"].astype(np.int64)
        if ref_rank.shape[0] != out["item_rank"].shape[0]:
            raise SystemExit(
                f"--check_dump has {ref_rank.shape[0]:,} rows but this run scored "
                f"{out['item_rank'].shape[0]:,}; use the same --split and no --max_rows"
            )
        if not np.array_equal(ref_tgt, out["targets"]):
            raise SystemExit(
                "--check_dump is for different rows (the target arrays differ), so it "
                "cannot verify this run"
            )
        same = ref_rank == out["item_rank"]
        frac = float(same.mean())
        logger.info("-" * 96)
        logger.info(f"  reproduction: item rank matches the dumped ranks on "
                    f"{100 * frac:.5f}% of rows ({int((~same).sum()):,} differ)")
        if not same.all():
            diff = np.abs(ref_rank[~same] - out["item_rank"][~same])
            logger.info(f"  the differing rows sit within {int(diff.max())} positions, "
                        "which is fp32 rounding between a summed tuple vector and a "
                        "summed projection; exact ties are unaffected because every "
                        "member of a tuple gets the same vector")
        if frac < 0.95:
            raise SystemExit(
                f"only {100 * frac:.2f}% of item ranks reproduce, which is too few to "
                "be rounding. The region sizes are probably misaligned with the "
                "catalogue, and every number above is then wrong."
            )

    if cli.out_json:
        payload = {"summary": summary,
                   "n_regions": out["n_regions"], "n_items": out["n_items"],
                   "items_per_region": out["items_per_region"],
                   "shared_fraction": out["shared_fraction"],
                   "largest": int(out["largest"]),
                   "sid_sizes": out["sid_sizes"],
                   "checkpoint": out["checkpoint"], "data_dir": out["data_dir"],
                   "split": out["split"],
                   "verdict": RL.verdict(summary)}
        os.makedirs(os.path.dirname(os.path.abspath(cli.out_json)), exist_ok=True)
        with open(cli.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"  wrote {cli.out_json}")
    if cli.dump_path:
        os.makedirs(os.path.dirname(os.path.abspath(cli.dump_path)), exist_ok=True)
        np.savez_compressed(
            cli.dump_path,
            rows=np.arange(len(out["item_rank"]), dtype=np.int64),
            targets=out["targets"], item_rank=out["item_rank"],
            region_rank=out["region_rank"], g=out["g"],
            n_rows=np.int64(out["n_rows"]), sid_sizes=np.asarray(out["sid_sizes"]),
            n_regions=np.int64(out["n_regions"]), split=out["split"],
            checkpoint=out["checkpoint"],
        )
        logger.info(f"  wrote {cli.dump_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""probe_hstu.py

Score the trained Stage-1 HSTU tower (ItemDNN + HSTUSequenceEncoder) as a
retriever against the exact DiGR validation split.

The question it answers: does the Stage-1 sequential model encode *behaviour*
(the next click), or only item content? The pooled-content kNN probe
(`embedding_knn_nextclick.py`) found no behaviour, but that probe averaged raw
content embeddings and ignored the HSTU, the action embeddings and the
ItemDNN's learned transformation. This probe runs the real encoder.

Inputs (all on the cluster):
  --checkpoint         stage1_rec_model.pt  (state dicts + args + cardinalities)
  --items_path         canonical_items_cleaned.parquet (rebuilds the feature store)
  --catalog_embeddings item_embeddings.npy   (default: next to the checkpoint)
  --catalog_ids        item_ids.npy          (default: next to the checkpoint)
  --data_dir           DiGR preprocessed dir (vocab.json, catalog_*.npy, val/*.npy)

The DiGR tokens deliberately drop the action type (actions are never tokens),
so history actions are approximated as all-click (the 4-token vocab is
0=PAD 1=Exposure 2=Click 3=UNK). This is a lower bound on the tower's signal;
exact actions from seq.parquet are a follow-up (`--actions exact`).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_SID_DIR = os.path.join(_REPO_ROOT, "SID Generation")
for _d in (_SCRIPT_DIR, _SID_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import torch  # noqa: E402

import embedding_knn_nextclick as knn  # noqa: E402
import train_item_embedding_cli as stage1  # noqa: E402


logger = logging.getLogger("probe_hstu")

DEFAULT_KS: Tuple[int, ...] = (1, 10, 100, 1000)
MRR_K = 10
CLICK = 2  # action vocabulary: 0=PAD, 1=Exposure, 2=Click, 3=UNK


def build_logger(level: int = logging.INFO) -> logging.Logger:
    """Console logger matching the other scripts in this directory."""
    if not logger.handlers:
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except (AttributeError, OSError):                   # pragma: no cover
            pass
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(h)
    logger.setLevel(level)
    return logger


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score the Stage-1 HSTU tower against the DiGR val split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="path to stage1_rec_model.pt")
    p.add_argument("--items_path", default=None,
                   help="canonical_items parquet; default: the checkpoint's "
                        "recorded args.items_path")
    p.add_argument("--catalog_embeddings", default=None,
                   help="item_embeddings.npy; default: <checkpoint dir>/"
                        "item_embeddings.npy")
    p.add_argument("--catalog_ids", default=None,
                   help="item_ids.npy; default: <checkpoint dir>/item_ids.npy")
    p.add_argument("--data_dir", required=True,
                   help="preprocessed DiGR data dir (vocab.json, "
                        "catalog_*.npy, <split>/{tokens,offsets,targets}.npy)")
    p.add_argument("--split", default="val")
    p.add_argument("--history_window", type=int, default=20,
                   help="history events per row fed to the HSTU (<= max_seq_len)")
    p.add_argument("--actions", default="approx", choices=("approx", "exact"),
                   help="approx: treat every history event as a click; "
                        "exact: read action_type from seq.parquet (not yet used)")
    p.add_argument("--seq_path", default=None,
                   help="raw seq.parquet (only for --actions exact)")
    p.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS),
                   help="comma-separated recall cut-offs")
    p.add_argument("--topk", type=int, default=1000,
                   help="top-k the catalog is scanned for")
    p.add_argument("--batch_size", type=int, default=512,
                   help="rows per HSTU forward")
    p.add_argument("--catalog_chunk", type=int, default=262_144,
                   help="catalog rows per score block")
    p.add_argument("--max_rows", type=int, default=0,
                   help="debug: score only the first N rows (0 = all)")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16"),
                   help="autocast dtype for the HSTU forward (scoring is always "
                        "float32); bfloat16 matches training/export")
    p.add_argument("--out_json", default=None)
    args = p.parse_args(argv)
    args.ks = tuple(int(k) for k in args.ks.split(",") if k.strip())
    if args.actions == "exact" and not args.seq_path:
        p.error("--actions exact requires --seq_path")
    return args


def resolve_device(request: str) -> torch.device:
    if request == "cuda":
        return torch.device("cuda")
    if request == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# checkpoint / feature store / model
# --------------------------------------------------------------------------- #
def load_checkpoint(path: str, map_location) -> Dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def rebuild_feature_store(ckpt_args: Dict, items_path: str, device: torch.device):
    """Rebuild the non-persistent ItemFeatureStore and its id mapping.

    Returns (store, item_to_idx, idx_to_item). The feature store is not in the
    checkpoint (buffers are registered persistent=False), so it must be rebuilt
    from the items parquet, exactly as training did.
    """
    import pandas as pd
    cat_candidates = (
        [c.strip() for c in ckpt_args["cat_cols"].split(",")]
        if ckpt_args.get("cat_cols") else stage1.DEFAULT_CAT_COLS
    )
    items_df = pd.read_parquet(items_path)
    store_dtype = (
        torch.bfloat16 if (device.type == "cuda" and not ckpt_args.get("fp16", False))
        else torch.float32
    )
    store, item_to_idx, idx_to_item, _, derived_cols = stage1.build_feature_store(
        items_df,
        item_id_col=ckpt_args["item_id_col"],
        embedding_col=ckpt_args["embedding_col"],
        has_emb_col=ckpt_args["has_emb_col"],
        embedding_dim=ckpt_args["raw_emb_dim"],
        cat_col_candidates=cat_candidates,
        dtype=store_dtype,
    )
    return store, item_to_idx, idx_to_item, derived_cols


def rebuild_model(ckpt_args: Dict, cardinalities: Dict, cat_cols: Sequence[str],
                  store) -> stage1.SequentialRecModel:
    hidden_dims = tuple(
        int(x.strip()) for x in str(ckpt_args["item_dnn_hidden_dims"]).split(","))
    item_dnn = stage1.ItemDNN(
        sem_dim=ckpt_args["raw_emb_dim"], cat_cardinalities=cardinalities,
        cat_cols=list(cat_cols), cat_emb_dim=ckpt_args["cat_emb_dim"],
        output_dim=ckpt_args["output_dim"], hidden_dims=hidden_dims,
        dropout=ckpt_args["dropout"],
    )
    seq_encoder = stage1.HSTUSequenceEncoder(
        hidden_dim=ckpt_args["output_dim"], max_seq_len=ckpt_args["max_seq_len"],
        num_blocks=ckpt_args["num_blocks"], num_heads=ckpt_args["num_heads"],
        dropout=ckpt_args["dropout"],
    )
    return stage1.SequentialRecModel(
        feature_store=store, item_dnn=item_dnn, seq_encoder=seq_encoder,
        temperature=float(ckpt_args.get("temperature", 0.05)),
        learnable_temperature=bool(ckpt_args.get("learnable_temperature", False)),
        use_log_q=(not bool(ckpt_args.get("no_log_q", False))),
        log_q_weight=float(ckpt_args.get("log_q_weight", 1.0)),
        cross_gpu_negatives=bool(ckpt_args.get("cross_gpu_negatives", False)),
    )


def load_state_dicts(model, ckpt: Dict) -> None:
    model.item_dnn.load_state_dict(ckpt["item_dnn_state_dict"])
    model.seq_encoder.load_state_dict(ckpt["seq_encoder_state_dict"])
    model.action_emb.load_state_dict(ckpt["action_emb_state_dict"])
    model.input_layer_norm.load_state_dict(ckpt["input_layer_norm_state_dict"])


# --------------------------------------------------------------------------- #
# history reconstruction
# --------------------------------------------------------------------------- #
def to_chronological(codes: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Re-order `decode_row_windows` output into the layout Stage-1 trained on.

    `decode_row_windows` returns j=0 as the most recent event and j growing back
    in time, with `valid[:, :m]` true and `valid[:, m:]` false.

    Stage-1 pads at the END. Both of its datasets do `padded_items[:n] =
    h_items[:n]`, so it expects the oldest event at position 0 and the newest at
    position m-1, left-aligned. `HSTUSequenceEncoder.forward` then reads
    `x[:, length - 1]`, which is the newest event only for a left-aligned
    sequence. Reversing into a fixed-width buffer (what this used to do)
    right-aligns instead, so for every row with fewer than `width` events that
    read lands on a pad slot and the user vector is built from padding.
    """
    n, width = valid.shape[0], valid.shape[1]
    m = valid.sum(axis=1)
    src = m[:, None] - 1 - np.arange(width)[None, :]
    keep = src >= 0
    idx = np.clip(src, 0, width - 1)
    rows = np.arange(n)[:, None]
    return codes[rows, idx, :], keep


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_cli(argv)
    log = build_logger()
    device = resolve_device(args.device)
    amp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    t_start = time.time()

    # ---- checkpoint ------------------------------------------------------ #
    ckpt = load_checkpoint(args.checkpoint, torch.device("cpu"))
    ckpt_args = dict(ckpt["args"])
    cardinalities = dict(ckpt["cardinalities"])
    cat_cols = list(ckpt["cat_cols"])
    log.info(f"checkpoint {args.checkpoint}")
    log.info(f"  items_path (recorded) {ckpt_args.get('items_path')}")
    log.info(f"  output_dim {ckpt_args['output_dim']}  max_seq_len "
             f"{ckpt_args['max_seq_len']}  num_blocks {ckpt_args['num_blocks']}")

    # ---- feature store ---------------------------------------------------- #
    items_path = args.items_path or ckpt_args.get("items_path")
    if not items_path or not os.path.isfile(items_path):
        raise FileNotFoundError(
            f"items parquet not found: {items_path!r}; pass --items_path")
    store, item_to_idx, idx_to_item, derived_cols = rebuild_feature_store(
        ckpt_args, items_path, device)
    if set(cat_cols) != set(derived_cols):
        log.warning("checkpoint cat_cols %s differ from the parquet's %s; "
                    "using the checkpoint's for the ItemDNN", cat_cols, derived_cols)
    n_store_items = len(idx_to_item) - 1
    log.info(f"feature store {n_store_items:,} items rebuilt from {items_path}")

    # ---- model ------------------------------------------------------------ #
    model = rebuild_model(ckpt_args, cardinalities, cat_cols, store)
    load_state_dicts(model, ckpt)
    model = model.to(device).eval()
    log.info(f"model rebuilt on {device} (dtype {args.dtype})")

    # ---- DiGR vocab + catalog -------------------------------------------- #
    vocab_path = os.path.join(args.data_dir, "vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    sid_sizes = [int(s) for s in vocab["sid_sizes"]]
    sid_offsets = [int(o) for o in vocab["sid_offsets"]]
    num_levels = int(vocab.get("num_levels", len(sid_sizes)))
    sep_id = int(vocab.get("sep_id", 1))
    if len(sid_offsets) != num_levels:
        raise ValueError("vocab.json sid_offsets and num_levels disagree")

    cat_codes = np.load(os.path.join(args.data_dir, "catalog_sid_codes.npy"))
    cat_ids = np.load(os.path.join(args.data_dir, "catalog_item_ids.npy"),
                      allow_pickle=True)
    n_catalog = len(cat_codes)
    if len(cat_ids) != n_catalog:
        raise ValueError("catalog_item_ids.npy and catalog_sid_codes.npy disagree")
    full_keys, full_rows = knn.build_key_lookup(cat_codes, sid_sizes, num_levels)
    log.info(f"catalog {n_catalog:,} items, {num_levels} SID levels, "
             f"sizes {sid_sizes}")

    # ---- catalog embeddings for scoring ---------------------------------- #
    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    emb_path = args.catalog_embeddings or os.path.join(ckpt_dir, "item_embeddings.npy")
    ids_path = args.catalog_ids or os.path.join(ckpt_dir, "item_ids.npy")
    if not os.path.isfile(emb_path):
        raise FileNotFoundError(f"catalog embeddings not found: {emb_path}")
    emb = knn.load_embeddings(emb_path, cat_ids, ids_path, None, None)
    if emb.shape[0] != n_catalog:
        raise ValueError(f"embeddings have {emb.shape[0]} rows, catalog {n_catalog}")
    norms = np.linalg.norm(emb[:min(len(emb), 50_000)], axis=1)
    if np.abs(norms - 1.0).max() > 1e-3:
        log.warning("embedding rows not unit norm; renormalising")
        emb = knn.normalise_rows(emb)
    dim = emb.shape[1]
    if dim != ckpt_args["output_dim"]:
        raise ValueError(f"embedding dim {dim} != model output_dim "
                         f"{ckpt_args['output_dim']}")
    catalog_t = torch.from_numpy(emb).to(device)
    log.info(f"catalog matrix {emb.shape[0]} x {dim} float32 on {device}")

    # ---- catalog row -> store index lookup (computed once) --------------- #
    import pandas as pd
    s = pd.Series(np.arange(1, len(idx_to_item), dtype=np.int64),
                  index=idx_to_item[1:], dtype=np.int64)
    store_of_catalog_row = s.reindex(cat_ids.astype(np.int64)).fillna(0) \
                            .to_numpy(dtype=np.int64)
    unmapped = int((store_of_catalog_row == 0).sum())
    log.info(f"catalog->store: {unmapped:,} of {n_catalog:,} items unmapped "
             f"(not in the feature store)")

    # ---- split ------------------------------------------------------------ #
    split_dir = os.path.join(args.data_dir, args.split)
    tokens = np.load(os.path.join(split_dir, "tokens.npy"), mmap_mode="r")
    offsets = np.load(os.path.join(split_dir, "offsets.npy"))
    targets = np.load(os.path.join(split_dir, "targets.npy"))
    n_rows = len(targets)
    if args.max_rows:
        n_rows = min(n_rows, int(args.max_rows))
    if len(offsets) != len(targets) + 1:
        raise ValueError("offsets.npy and targets.npy disagree on the row count")
    log.info(f"split {args.split}: {n_rows:,} rows, {len(tokens):,} tokens")

    width = min(int(args.history_window), int(ckpt_args["max_seq_len"]))
    if width < 1:
        raise ValueError("--history_window must be >= 1")
    log.info("=" * 78)
    log.info(f"  split         : {args.split} ({n_rows:,} rows)")
    log.info(f"  history window: {width} events, actions={args.actions}")
    log.info(f"  device/dtype  : {device} / {args.dtype}")
    log.info(f"  cut-offs      : {list(args.ks)}  topk {args.topk}")
    log.info("=" * 78)

    metrics = knn.new_metrics(args.ks)
    diag = {"rows": 0.0, "unmapped_history_events": 0.0,
            "repeat_rate": 0.0}

    tgt_all = torch.from_numpy(np.asarray(targets[:n_rows], dtype=np.int64)) \
        .to(device)

    for r0 in range(0, n_rows, args.batch_size):
        r1 = min(r0 + args.batch_size, n_rows)
        n = r1 - r0
        codes, valid = knn.decode_row_windows(
            tokens, offsets, r0, r1, sid_offsets, num_levels, sep_id, width)
        chrono_codes, chrono_valid = to_chronological(codes, valid)

        safe = np.where(chrono_valid[:, :, None], chrono_codes, 0)
        row_keys = knn.pack_sid_keys(
            safe.reshape(-1, num_levels), sid_sizes).reshape(n, width)
        cat_of_event = np.where(
            chrono_valid.reshape(-1),
            knn.lookup_catalog_rows(full_keys, full_rows, row_keys.reshape(-1)),
            -1,
        ).reshape(n, width)
        tgt_np = np.asarray(targets[r0:r1], dtype=np.int64)

        diag["rows"] += float(n)
        diag["unmapped_history_events"] += float(
            ((cat_of_event >= 0) & chrono_valid
             & (store_of_catalog_row[np.clip(cat_of_event, 0, n_catalog - 1)] == 0)).sum())
        diag["repeat_rate"] += float(
            ((cat_of_event == tgt_np[:, None]) & chrono_valid).any(axis=1).sum())

        cat_safe = np.clip(cat_of_event, 0, n_catalog - 1)
        store_idx = np.where(cat_of_event >= 0,
                             store_of_catalog_row[cat_safe], 0)
        actions = np.where(chrono_valid, CLICK, 0)

        hi = torch.from_numpy(np.ascontiguousarray(store_idx)).to(device)
        ha = torch.from_numpy(np.ascontiguousarray(actions)).to(device)
        m = torch.from_numpy(np.ascontiguousarray(chrono_valid)).to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=(device.type == "cuda")):
            user = model.encode_user(hi, ha, m)
        user_f = user.float()

        _, idx = knn.topk_over(user_f, catalog_t, args.topk, args.catalog_chunk)
        ranks = knn.ranks_from_topk(idx, tgt_all[r0:r1], args.topk)
        knn.update_metrics(metrics, ranks.cpu().numpy(), args.ks)

        if (r1 // args.batch_size) % 50 == 0 or r1 == n_rows:
            log.info(f"  rows {r1:,}/{n_rows:,} ({time.time() - t_start:.0f}s)")

    result = knn.finalise_metrics(metrics, args.ks)
    rows = max(diag["rows"], 1.0)

    # ---- report ------------------------------------------------------------ #
    log.info("=" * 78)
    log.info(f"  HSTU tower (actions={args.actions}, window {width})")
    for k in args.ks:
        log.info(f"    recall@{k:<4} {result[f'recall@{k}']:.4%}")
    log.info(f"    mrr@10       {result['mrr']:.4%}")
    log.info(f"    nDCG@10      {result['ndcg@10']:.4%}")
    log.info(f"  diagnostics: {diag['unmapped_history_events']/max(1.0, diag['rows']*width):.4%} "
             f"unmapped history events, {diag['repeat_rate']/rows:.4%} rows whose "
             f"history already contains the target")
    log.info("  reference (same val rows, same space):")
    for name, ref in knn.REFERENCE.items():
        log.info(f"    {name:<11} R@1 {ref['R@1']:.4%}  R@10 {ref['R@10']:.4%}  "
                 f"R@100 {ref['R@100']:.4%}  R@1000 {ref['R@1000']:.4%}")
    log.info(f"  verdict: {verdict(result['recall@10'])}")
    log.info("=" * 78)

    if args.out_json:
        payload = {
            "split": args.split,
            "n_rows": int(rows),
            "history_window": width,
            "actions": args.actions,
            "checkpoint": os.path.abspath(args.checkpoint),
            "metrics": result,
            "diagnostics": {
                "unmapped_history_event_rate":
                    diag["unmapped_history_events"] / max(1.0, diag["rows"] * width),
                "repeat_rate": diag["repeat_rate"] / rows,
            },
            "reference": knn.REFERENCE,
            "verdict": verdict(result["recall@10"]),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        log.info(f"wrote {args.out_json}")
    return 0


def verdict(r10: float) -> str:
    if r10 >= 0.05:
        return ("behaviour is in the HSTU tower; the pooled-content probe was "
                "the wrong query and the SID is destroying the signal. Regenerate "
                "the item embedding (concat user/item or retrain)")
    if r10 >= 0.025:
        return ("some behaviour in the tower (clears co-click/DiGR territory); "
                "regenerating the item embedding is worth doing")
    return ("no behaviour in the tower either; the Stage-1 model itself needs "
            "retraining before the embedding is worth changing")


if __name__ == "__main__":                      # pragma: no cover
    raise SystemExit(main())

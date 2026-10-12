#!/usr/bin/env python3
"""
Stage 1: ItemDNN + HSTU Sequential Encoder with InfoNCE & LogQ Debias.
Optimized for single NVIDIA H800 / Hopper GPUs (with multi-GPU DDP support).
Includes full TensorBoard logging, retrieval evaluation (Recall@K, MRR, NDCG),
and fault-tolerant atomic checkpointing & resumption.

Expected Inputs:
  - items parquet (canonical_items.parquet):
      item_id: i64
      100, 101, 102, 112, 114, 115, 116, 117, 118, 119, 120, 121, 122: i64 (0 = missing/pad)
      mm_emb: array[f32, 1024]
      has_mm_emb: bool
  - seq parquet (seq.parquet):
      user_id: i64
      seq: list of dicts [{"item_id": int, "action_type": int (0 or 1), "timestamp": int}]
"""

import argparse
import json
import logging
import math
import os
import random
import signal
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# Graceful TensorBoard import
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("hstu_stage1")

DEFAULT_CAT_COLS = [
    "100", "101", "102", "112", "114", "115", "116", "117", "118", "119", "120", "121", "122"
]


def init_distributed():
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP is configured but CUDA is unavailable.")
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return True, dist.get_rank(), dist.get_world_size(), local_rank
    return False, 0, 1, 0


# =========================================================================
# Feature Store & Fast Dataset
# =========================================================================

class ItemFeatureStore(nn.Module):
    """
    Tensors indexed by contiguous item internal indices (1..N).
    Index 0 is strictly reserved for PAD/UNK/MISSING.
    Buffers are registered with persistent=False to avoid bloating model checkpoints.
    """
    def __init__(self, sem: torch.Tensor, has_mm: torch.Tensor, cats: torch.Tensor):
        super().__init__()
        self.register_buffer("sem", sem, persistent=False)         # [N+1, embedding_dim]
        self.register_buffer("has_mm", has_mm, persistent=False)   # [N+1, 1]
        self.register_buffer("cats", cats, persistent=False)       # [N+1, num_cats]

    def get(self, indices: torch.Tensor):
        return self.sem[indices], self.has_mm[indices], self.cats[indices]


def build_feature_store(
    df: pd.DataFrame,
    item_id_col: str = "item_id",
    embedding_col: str = "mm_emb",
    has_emb_col: str = "has_mm_emb",
    embedding_dim: int = 1024,
    cat_col_candidates: Optional[List[str]] = None,
    dtype: torch.dtype = torch.float32,
):
    if item_id_col not in df.columns:
        raise ValueError(f"Missing item ID column: {item_id_col}")
    if df[item_id_col].isna().any():
        raise ValueError(f"{item_id_col} contains null IDs.")
    if df[item_id_col].duplicated().any():
        duplicates = int(df[item_id_col].duplicated().sum())
        raise ValueError(f"Expected one row per item; found {duplicates:,} duplicate item IDs.")

    ids = df[item_id_col].to_numpy(dtype=np.int64)
    item_to_idx = {int(iid): i + 1 for i, iid in enumerate(ids)}
    idx_to_item = np.zeros(len(ids) + 1, dtype=np.int64)
    idx_to_item[1:] = ids
    n_plus_pad = len(ids) + 1

    # 1. Parse Multimodal Vectors & Presence Indicator
    sem = torch.zeros((n_plus_pad, embedding_dim), dtype=dtype)
    has_mm = torch.zeros((n_plus_pad, 1), dtype=dtype)

    has_mm_bool = np.zeros(len(df), dtype=bool)
    if has_emb_col in df.columns:
        has_mm_bool = df[has_emb_col].fillna(False).to_numpy(dtype=bool)

    if embedding_col in df.columns:
        raw_embs = df[embedding_col].values
        # Fast vectorized path if column is already a 2D numpy array
        if isinstance(raw_embs, np.ndarray) and raw_embs.ndim == 2 and raw_embs.shape[1] == embedding_dim:
            sem[1:] = torch.from_numpy(raw_embs.astype(np.float32)).to(dtype)
            has_mm[1:, 0] = torch.from_numpy(has_mm_bool.astype(np.float32)).to(dtype)
        else:
            for i, val in enumerate(raw_embs):
                if not has_mm_bool[i] or val is None:
                    continue
                try:
                    arr = np.asarray(val, dtype=np.float32).reshape(-1)
                    if arr.size == embedding_dim and np.isfinite(arr).all():
                        sem[i + 1] = torch.from_numpy(arr).to(dtype)
                        has_mm[i + 1, 0] = 1.0
                except (TypeError, ValueError):
                    continue
    else:
        logger.warning("Embedding column '%s' not found; using zero vectors.", embedding_col)

    # 2. Parse Categorical Columns
    if cat_col_candidates is None:
        cat_col_candidates = DEFAULT_CAT_COLS

    matched_cat_cols = []
    for c in cat_col_candidates:
        if c in df.columns:
            matched_cat_cols.append(c)
        elif str(c) in df.columns:
            matched_cat_cols.append(str(c))
        else:
            try:
                c_int = int(c)
                if c_int in df.columns:
                    matched_cat_cols.append(c_int)
            except ValueError:
                pass

    logger.info("Matched %d categorical feature columns: %s", len(matched_cat_cols), matched_cat_cols)

    cat_arrays = []
    cardinalities = {}
    for col in matched_cat_cols:
        col_name = str(col)
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(dtype=np.int64)

        # 0 is strictly reserved for PAD / Missing
        non_zero_unique = np.sort(np.unique(vals[vals != 0]))
        cardinalities[col_name] = len(non_zero_unique) + 1

        col_arr = np.zeros(n_plus_pad, dtype=np.int64)
        if len(non_zero_unique) > 0:
            nonzero_mask = (vals != 0)
            positions = np.searchsorted(non_zero_unique, vals[nonzero_mask])
            matched = (non_zero_unique[positions] == vals[nonzero_mask])
            mapped_codes = np.where(matched, positions + 1, 0)
            nonzero_indices = np.where(nonzero_mask)[0] + 1
            col_arr[nonzero_indices] = mapped_codes

        cat_arrays.append(torch.from_numpy(col_arr).unsqueeze(1))

    cats = torch.cat(cat_arrays, dim=1) if cat_arrays else torch.zeros((n_plus_pad, 1), dtype=torch.long)
    if not cat_arrays:
        cardinalities["__dummy__"] = 1

    valid_mm_pct = 100.0 * float(has_mm[1:].mean()) if len(ids) else 0.0
    logger.info("Catalog Items=%s | with valid mm_emb=%.2f%% | Categoricals=%d",
                f"{len(ids):,}", valid_mm_pct, len(matched_cat_cols))

    return (
        ItemFeatureStore(sem, has_mm, cats),
        item_to_idx,
        idx_to_item,
        cardinalities,
        [str(c) for c in matched_cat_cols],
    )


class FastNextItemDataset(Dataset):
    """
    Parses sequences containing dicts:
      [{"item_id": 2905777, "action_type": 0, "timestamp": 1745919240}, ...]
    Supports both leave-one-out and user-split validation modes.
    """
    def __init__(
        self,
        seq_df: pd.DataFrame,
        item_to_idx: Dict[int, int],
        max_seq_len: int = 100,
        seq_col: str = "seq",
        min_history: int = 1,
        max_examples_per_sequence: int = 0,
        split: str = "train",
        split_mode: str = "leave_one_out",
    ):
        if split not in {"train", "valid"}:
            raise ValueError("split must be 'train' or 'valid'")

        self.samples = []
        dropped_short = 0
        raw_sequences = seq_df[seq_col].tolist()

        for raw_seq in raw_sequences:
            if raw_seq is None or not isinstance(raw_seq, (list, np.ndarray)) or len(raw_seq) == 0:
                continue

            item_seq, act_seq = [], []
            for interaction in raw_seq:
                if isinstance(interaction, dict):
                    iid = interaction.get("item_id")
                    act = interaction.get("action_type")
                else:
                    iid = interaction
                    act = 0

                if iid is None:
                    continue
                try:
                    mapped = item_to_idx.get(int(iid))
                except (TypeError, ValueError):
                    mapped = None

                if mapped is not None:
                    item_seq.append(mapped)
                    # 0 = PAD, 1 = Exposed, 2 = Click, 3 = UNK/Missing
                    if act is None or pd.isna(act):
                        act_token = 3  # UNK
                    else:
                        try:
                            val = int(act)
                            if val == 1:
                                act_token = 2  # Click
                            elif val == 0:
                                act_token = 1  # Exposed (no click)
                            else:
                                act_token = 3  # Corrupted / unexpected value -> UNK
                        except (TypeError, ValueError):
                            act_token = 3      # Fallback -> UNK

                    act_seq.append(act_token)

            if len(item_seq) < min_history + 1:
                dropped_short += 1
                continue

            if split_mode == "leave_one_out":
                if split == "valid":
                    # Predict final interaction using all preceding interactions
                    end_positions = [len(item_seq) - 1]
                else:
                    # Train on interactions prior to the held-out final interaction
                    end_positions = list(range(min_history, len(item_seq) - 1))
            else:
                # user_split mode: user pool is already partitioned
                if split == "valid":
                    end_positions = [len(item_seq) - 1]
                else:
                    end_positions = list(range(min_history, len(item_seq)))

            if max_examples_per_sequence > 0:
                end_positions = end_positions[-max_examples_per_sequence:]

            for end in end_positions:
                start = max(0, end - max_seq_len)
                hist_items = item_seq[start:end]
                hist_actions = act_seq[start:end]
                target_item = item_seq[end]
                self.samples.append((hist_items, hist_actions, target_item))

        if not self.samples:
            raise ValueError(f"No valid next-item samples generated for {split}. Check sequence lengths and IDs.")

        logger.info("[%s] Total Samples: %s | Skipped short: %s",
                    split.upper(), f"{len(self.samples):,}", f"{dropped_short:,}")
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hist_items, hist_actions, target = self.samples[idx]
        n = len(hist_items)

        padded_items = np.zeros(self.max_seq_len, dtype=np.int64)
        padded_actions = np.zeros(self.max_seq_len, dtype=np.int64)
        valid_mask = np.zeros(self.max_seq_len, dtype=np.bool_)

        padded_items[:n] = hist_items
        padded_actions[:n] = hist_actions
        valid_mask[:n] = True

        return {
            "history_items": torch.from_numpy(padded_items),
            "history_actions": torch.from_numpy(padded_actions),
            "mask": torch.from_numpy(valid_mask),
            "target": torch.tensor(target, dtype=torch.long),
        }


# =========================================================================
# Model Architecture: ItemDNN & HSTU Sequence Encoder
# =========================================================================

class ItemDNN(nn.Module):
    """
    Fuses multimodal embedding, presence flag, and categorical features.
    L2-normalizes the final representation in float32.
    """
    def __init__(
        self,
        sem_dim: int,
        cat_cardinalities: Dict[str, int],
        cat_cols: List[str],
        cat_emb_dim: int = 32,
        output_dim: int = 256,
        hidden_dims: Tuple[int, ...] = (512, 256),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cat_cols = list(cat_cols)
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cat_cardinalities[c], cat_emb_dim, padding_idx=0)
            for c in self.cat_cols
        ])

        sem_hidden = hidden_dims[0]
        self.sem_proj = nn.Sequential(
            nn.Linear(sem_dim + 1, sem_hidden),
            nn.LayerNorm(sem_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        cat_dim = len(self.cat_cols) * cat_emb_dim
        fusion_dim = sem_hidden + cat_dim
        layers = []
        in_dim = fusion_dim
        for h in hidden_dims[1:]:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers += [nn.Linear(in_dim, output_dim)]
        self.mlp = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, sem: torch.Tensor, has_mm: torch.Tensor, cats: torch.Tensor):
        # sem: [..., sem_dim], has_mm: [..., 1], cats: [..., num_cats]
        h_sem = self.sem_proj(torch.cat([sem.float(), has_mm.float()], dim=-1))

        parts = [h_sem]
        if self.cat_cols:
            cat_embs = [emb(cats[..., i].long()) for i, emb in enumerate(self.cat_embeddings)]
            parts.append(torch.cat(cat_embs, dim=-1))

        out = self.mlp(torch.cat(parts, dim=-1))
        return F.normalize(out.float(), p=2, dim=-1)


class HSTUBlock(nn.Module):
    """
    Hierarchical Sequential Transduction Unit (HSTU) layer (ICML 2024).
    Replaces softmax attention with pointwise aggregated gated attention.
    Uses fused linear projections to maximize Hopper GPU kernel occupancy.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        max_seq_len: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_seq_len = max_seq_len

        # Fused linear projection for U, Q, K, V
        self.in_proj = nn.Linear(dim, 4 * dim)
        self.act = nn.SiLU()

        # Learnable relative positional bias
        self.rel_pos_bias = nn.Embedding(max_seq_len, num_heads)

        self.attn_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor, causal_mask: torch.Tensor):
        # x: [B, L, D], valid_mask: [B, L], causal_mask: [L, L]
        B, L, D = x.shape
        residual = x
        x_norm = self.pre_norm(x)

        # Fused projection
        proj = self.in_proj(x_norm)
        u_lin, q_lin, k_lin, v_lin = proj.chunk(4, dim=-1)

        u = self.act(u_lin)
        q = self.act(q_lin).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.act(k_lin).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_lin.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Relative position bias
        pos = torch.arange(L, device=x.device)
        rel_dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp(min=0, max=self.max_seq_len - 1)
        rel_bias = self.rel_pos_bias(rel_dist).permute(2, 0, 1).unsqueeze(0)
        scores = scores + rel_bias

        # Gated attention mask: causal & both key and query validity
        valid_keys = valid_mask.unsqueeze(1).unsqueeze(2)      # [B, 1, 1, L]
        valid_queries = valid_mask.unsqueeze(1).unsqueeze(3)   # [B, 1, L, 1]
        active_mask = causal_mask.unsqueeze(0).unsqueeze(1) & valid_keys & valid_queries

        # Pointwise activation (SiLU)
        attn = self.act(scores) * active_mask.float()

        # Aggregate and gate
        attn_out = torch.matmul(attn, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)

        gated = self.attn_norm(attn_out) * u
        y = self.dropout(self.out_proj(gated))

        return residual + y


class HSTUSequenceEncoder(nn.Module):
    """
    Multi-layer HSTU sequential transducer for user representations.
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        max_seq_len: int = 100,
        num_blocks: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.blocks = nn.ModuleList([
            HSTUBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(pos)

        # Zero out padding slots before transformer processing
        x = x * valid_mask.unsqueeze(-1).to(x.dtype)

        causal_mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))

        for block in self.blocks:
            x = block(x, valid_mask=valid_mask, causal_mask=causal_mask)

        x = self.final_norm(x)

        # Gather last valid position
        lengths = valid_mask.long().sum(dim=1).clamp_min(1)
        last_repr = x[torch.arange(B, device=x.device), lengths - 1]
        return F.normalize(last_repr.float(), p=2, dim=-1)


class SequentialRecModel(nn.Module):
    """
    Joint ItemDNN + HSTU model with InfoNCE loss, LogQ correction,
    and single-GPU / multi-GPU negative contrastive learning.
    """
    def __init__(
        self,
        feature_store: ItemFeatureStore,
        item_dnn: ItemDNN,
        seq_encoder: HSTUSequenceEncoder,
        temperature: float = 0.05,
        learnable_temperature: bool = False,
        log_q: Optional[torch.Tensor] = None,
        use_log_q: bool = True,
        log_q_weight: float = 1.0,
        cross_gpu_negatives: bool = False,
    ):
        super().__init__()
        self.feature_store = feature_store
        self.item_dnn = item_dnn
        self.seq_encoder = seq_encoder
        self.use_log_q = use_log_q
        self.log_q_weight = log_q_weight
        self.cross_gpu_negatives = cross_gpu_negatives

        self.input_layer_norm = nn.LayerNorm(item_dnn.output_dim)

        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))
        else:
            self.register_buffer("temperature", torch.tensor(temperature, dtype=torch.float32))

        # Increase embedding size to 4: 0=PAD, 1=Exposed, 2=Click, 3=UNK
        self.action_emb = nn.Embedding(4, item_dnn.output_dim, padding_idx=0)

        if log_q is None:
            log_q = torch.zeros(feature_store.sem.shape[0], dtype=torch.float32)
        self.register_buffer("log_q", log_q.float(), persistent=False)

    def get_item_embeddings(self, indices: torch.Tensor) -> torch.Tensor:
        sem, has_mm, cats = self.feature_store.get(indices)
        emb = self.item_dnn(sem, has_mm, cats)
        # Ensure PAD token embedding is zeroed out
        pad_mask = (indices == 0).unsqueeze(-1)
        return emb.masked_fill(pad_mask, 0.0)

    def encode_user(self, history_items: torch.Tensor, history_actions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        item_reps = self.get_item_embeddings(history_items)
        act_reps = self.action_emb(history_actions)
        hist_input = self.input_layer_norm(item_reps + act_reps)
        return self.seq_encoder(hist_input, mask)

    def compute_loss_and_logits(
        self,
        history_items: torch.Tensor,
        history_actions: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        user = self.encode_user(history_items, history_actions, mask)
        positive = self.get_item_embeddings(target)
        candidate_log_q = self.log_q[target]

        t = self.temperature.clamp(min=1e-3, max=1.0)

        if self.cross_gpu_negatives and dist.is_initialized() and dist.get_world_size() > 1:
            all_positives = torch.cat(dist_nn.all_gather(positive), dim=0)
            all_log_q = torch.cat(dist_nn.all_gather(candidate_log_q), dim=0)

            sim_matrix = user @ all_positives.T
            logits = sim_matrix / t
            if self.use_log_q:
                logits = logits - self.log_q_weight * all_log_q.unsqueeze(0)
            labels = torch.arange(user.size(0), device=user.device) + dist.get_rank() * user.size(0)
        else:
            sim_matrix = user @ positive.T
            logits = sim_matrix / t
            if self.use_log_q:
                logits = logits - self.log_q_weight * candidate_log_q.unsqueeze(0)
            labels = torch.arange(user.size(0), device=user.device)

        loss = F.cross_entropy(logits, labels)
        return loss, logits, sim_matrix, labels, user

    def forward(
        self,
        history_items: torch.Tensor,
        history_actions: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        loss, _, _, _, _ = self.compute_loss_and_logits(history_items, history_actions, mask, target)
        return loss


# =========================================================================
# Metrics, Evaluation & Checkpointing
# =========================================================================

@torch.no_grad()
def compute_ranking_metrics(sim_matrix: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Computes exact in-batch retrieval metrics based on uncorrupted cosine similarity.
    """
    B = sim_matrix.size(0)
    target_scores = sim_matrix.gather(1, labels.unsqueeze(1))
    ranks = (sim_matrix > target_scores).sum(dim=1) + 1  # 1-indexed

    r1 = (ranks <= 1).float().mean().item()
    r5 = (ranks <= 5).float().mean().item()
    r10 = (ranks <= 10).float().mean().item()
    r50 = (ranks <= 50).float().mean().item()
    mrr = (1.0 / ranks.float()).mean().item()
    ndcg10 = torch.where(
        ranks <= 10,
        1.0 / torch.log2(ranks.float() + 1.0),
        torch.zeros_like(ranks, dtype=torch.float),
    ).mean().item()

    # Representation separation metrics
    diag_mask = torch.eye(B, device=sim_matrix.device, dtype=torch.bool)
    pos_cos = sim_matrix.diagonal().mean().item()
    neg_cos = sim_matrix[~diag_mask].mean().item() if B > 1 else 0.0

    return {
        "recall@1": r1,
        "recall@5": r5,
        "recall@10": r10,
        "recall@50": r50,
        "mrr": mrr,
        "ndcg@10": ndcg10,
        "pos_cosine": pos_cos,
        "neg_cosine": neg_cos,
        "cosine_gap": pos_cos - neg_cos,
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    epoch: int,
    amp_dtype: torch.dtype,
    grad_clip_norm: float = 1.0,
    tb_writer: Optional[SummaryWriter] = None,
    log_freq: int = 50,
    global_step: int = 0,
    max_steps: int = 0,
) -> Tuple[float, int]:
    model.train()
    total_loss, n_batches = 0.0, 0
    start_time = time.time()
    batch_start = time.time()

    for step, batch in enumerate(loader):
        if max_steps > 0 and step >= max_steps:
            logger.info("Reached max_steps_per_epoch (%d); finishing epoch %d early.", max_steps, epoch)
            break
        history_items = batch["history_items"].to(device, non_blocking=True)
        history_actions = batch["history_actions"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=(device.type == "cuda"),
        ):
            loss = model(history_items, history_actions, mask, target)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_val = float(loss.detach())
        total_loss += loss_val
        n_batches += 1
        global_step += 1

        if step > 0 and step % log_freq == 0:
            elapsed = max(time.time() - batch_start, 1e-6)
            steps_per_sec = log_freq / elapsed
            batch_size = history_items.size(0)
            samples_per_sec = steps_per_sec * batch_size
            current_lr = optimizer.param_groups[0]["lr"]

            logger.info(
                "Epoch %d | Step %d/%d (Global %d) | Loss: %.5f | GradNorm: %.2f | LR: %.2e | %.1f samples/s",
                epoch, step, len(loader), global_step, loss_val, float(grad_norm), current_lr, samples_per_sec
            )

            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", loss_val, global_step)
                tb_writer.add_scalar("train/learning_rate", current_lr, global_step)
                tb_writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                tb_writer.add_scalar("train/throughput_samples_per_sec", samples_per_sec, global_step)

            batch_start = time.time()

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_val_batches: int = 0,  # 0 = evaluate all, >0 = cap batches
) -> Dict[str, float]:
    model.eval()
    total_loss, n_batches = 0.0, 0
    metric_sums = {
        "recall@1": 0.0,
        "recall@5": 0.0,
        "recall@10": 0.0,
        "recall@50": 0.0,
        "mrr": 0.0,
        "ndcg@10": 0.0,
        "pos_cosine": 0.0,
        "neg_cosine": 0.0,
        "cosine_gap": 0.0,
    }

    eval_model = model.module if isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)) else model

    # Use enumerate(loader) so we can track the step index and break early
    for step, batch in enumerate(loader):
        if max_val_batches > 0 and step >= max_val_batches:
            logger.info("Reached max_val_batches (%d); finishing validation early.", max_val_batches)
            break

        history_items = batch["history_items"].to(device, non_blocking=True)
        history_actions = batch["history_actions"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=(device.type == "cuda"),
        ):
            loss, logits, sim_matrix, labels, _ = eval_model.compute_loss_and_logits(
                history_items, history_actions, mask, target
            )

        total_loss += float(loss.detach())
        batch_metrics = compute_ranking_metrics(sim_matrix, labels)
        for k, v in batch_metrics.items():
            metric_sums[k] += v
        n_batches += 1

    if n_batches == 0:
        raise ValueError("Validation loader produced no batches.")

    results = {k: v / n_batches for k, v in metric_sums.items()}
    results["loss"] = total_loss / n_batches
    return results


def save_checkpoint(
    path: str,
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.amp.GradScaler],
    best_metric: float,
    args: argparse.Namespace,
    cardinalities: Dict[str, int],
    cat_cols: List[str],
):
    """
    Saves a fault-tolerant atomic checkpoint. Writing to a temp file first avoids corrupting
    existing checkpoints upon abrupt termination.
    """
    raw_model = model.module if isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)) else model
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "item_dnn_state_dict": raw_model.item_dnn.state_dict(),
        "seq_encoder_state_dict": raw_model.seq_encoder.state_dict(),
        "action_emb_state_dict": raw_model.action_emb.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "args": vars(args),
        "cardinalities": cardinalities,
        "cat_cols": cat_cols,
        "rng_states": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)
    logger.info("Saved atomic checkpoint to %s", path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, int, float]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    logger.info("Loading checkpoint from %s ...", path)
    checkpoint = torch.load(path, map_location=device)

    raw_model = model.module if isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)) else model
    raw_model.item_dnn.load_state_dict(checkpoint["item_dnn_state_dict"])
    raw_model.seq_encoder.load_state_dict(checkpoint["seq_encoder_state_dict"])
    if "action_emb_state_dict" in checkpoint:
        raw_model.action_emb.load_state_dict(checkpoint["action_emb_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    rng = checkpoint.get("rng_states", {})
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("python") is not None:
        random.setstate(rng["python"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    global_step = checkpoint.get("global_step", 0)
    best_metric = checkpoint.get("best_metric", -float("inf"))

    logger.info("Resumed from Epoch %d | Global Step %d | Best Metric: %.4f",
                checkpoint.get("epoch", 0), global_step, best_metric)
    return start_epoch, global_step, best_metric


@torch.no_grad()
def export_catalog_embeddings(
    model: nn.Module,
    idx_to_item: np.ndarray,
    output_dir: str,
    device: torch.device,
    batch_size: int = 8192,
    amp_dtype: torch.dtype = torch.bfloat16,
):
    """
    Exports final L2-normalized item catalog embeddings for downstream SID tokenization / FAISS.
    """
    raw_model = model.module if isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)) else model
    raw_model.eval()
    all_embs = []
    total = len(idx_to_item) - 1

    logger.info("Exporting %s catalog embeddings in batches of %d...", f"{total:,}", batch_size)
    for start in range(1, total + 1, batch_size):
        end = min(start + batch_size, total + 1)
        indices = torch.arange(start, end, device=device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=(device.type == "cuda"),
        ):
            emb = raw_model.get_item_embeddings(indices)
        all_embs.append(emb.float().cpu().numpy())

    embeddings = np.concatenate(all_embs, axis=0).astype(np.float32)
    item_ids = idx_to_item[1:]

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "item_embeddings.npy"), embeddings)
    np.save(os.path.join(output_dir, "item_ids.npy"), item_ids)

    # Fast Parquet export via PyArrow (avoids massive Python list overhead)
    parquet_path = os.path.join(output_dir, "item_embeddings.parquet")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        emb_list_array = pa.FixedSizeListArray.from_arrays(
            pa.array(embeddings.ravel(), type=pa.float32()),
            embeddings.shape[1],
        )
        table = pa.Table.from_arrays([pa.array(item_ids, type=pa.int64()), emb_list_array], names=["item_id", "embedding"])
        pq.write_table(table, parquet_path)
    except Exception as e:
        logger.warning("PyArrow fast write failed (%s); falling back to pandas parquet export.", e)
        pd.DataFrame({"item_id": item_ids, "embedding": list(embeddings)}).to_parquet(parquet_path, index=False)

    try:
        import faiss
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, os.path.join(output_dir, "item_index.faiss"))
        logger.info("Saved FAISS IndexFlatIP to %s", output_dir)
    except ImportError:
        logger.info("faiss not installed; skipping FAISS index export.")

    logger.info("Successfully exported %s embeddings (dim=%d) to %s", f"{len(item_ids):,}", embeddings.shape[1], output_dir)


def estimate_log_q(seq_df: pd.DataFrame, item_to_idx: Dict[int, int], n_items: int, smoothing: float = 1.0):
    counts = np.zeros(n_items + 1, dtype=np.float64)
    for raw_seq in seq_df["seq"].tolist():
        if raw_seq is None or not isinstance(raw_seq, (list, np.ndarray)) or len(raw_seq) == 0:
            continue
        for interaction in raw_seq:
            iid = interaction.get("item_id") if isinstance(interaction, dict) else interaction
            if iid is not None:
                try:
                    idx = item_to_idx.get(int(iid))
                except (TypeError, ValueError):
                    idx = None
                    
                if idx is not None:
                    counts[idx] += 1.0

    counts[1:] += smoothing
    counts[0] = 1.0  # PAD token
    probs = counts / counts.sum()
    return torch.from_numpy(np.log(probs).astype(np.float32))


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.001):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =========================================================================
# Main Execution Pipeline
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 1: HSTU + ItemDNN Sequential Generative Retrieval")
    # Data paths & columns
    parser.add_argument("--items_path", default="canonical_items.parquet", help="Path to items parquet")
    parser.add_argument("--seq_path", default="seq.parquet", help="Path to user sequence parquet")
    parser.add_argument("--output_dir", default="./output_stage1_embeddings")
    parser.add_argument("--item_id_col", default="item_id")
    parser.add_argument("--embedding_col", default="mm_emb")
    parser.add_argument("--has_emb_col", default="has_mm_emb")
    parser.add_argument("--seq_col", default="seq")

    # Sequence & split options
    parser.add_argument("--max_seq_len", type=int, default=100)
    parser.add_argument("--min_history", type=int, default=1)
    parser.add_argument("--max_examples_per_sequence", type=int, default=10,
                        help="0 = all prefixes; >0 caps training samples per user sequence for high throughput.")
    parser.add_argument("--split_mode", default="leave_one_out", choices=["leave_one_out", "user_split"])
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio for user_split mode")

    # Model Architecture
    parser.add_argument("--raw_emb_dim", type=int, default=1024)
    parser.add_argument("--output_dim", type=int, default=256)
    parser.add_argument("--cat_emb_dim", type=int, default=32)
    parser.add_argument("--item_dnn_hidden_dims", default="512,256")
    parser.add_argument("--num_blocks", type=int, default=3, help="Number of HSTU layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--learnable_temperature", action="store_true")

    # InfoNCE & LogQ
    parser.add_argument("--no_log_q", action="store_true", help="Disable LogQ debiasing")
    parser.add_argument("--log_q_smoothing", type=float, default=1.0)
    parser.add_argument("--log_q_weight", type=float, default=1.0)
    parser.add_argument("--cross_gpu_negatives", action="store_true", help="Enable cross-GPU DDP negative gather")

    # Training & Optimization
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size per GPU (H800 handles 1024-2048)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr_ratio", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true", help="Use standard FP16 instead of BF16")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for kernel fusion")

    # Checkpointing & Resumption
    parser.add_argument("--resume", type=str, default=None, help="Explicit path to checkpoint file to resume from")
    parser.add_argument("--auto_resume", action="store_true", help="Resume automatically from checkpoint_last.pt if found")
    parser.add_argument("--save_freq", type=int, default=1, help="Epoch frequency for periodic checkpoints")
    parser.add_argument("--metric_for_best", default="recall@10", choices=["recall@10", "recall@5", "loss", "mrr"])

    # DataLoader & Hardware
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--tb_log_dir", type=str, default=None)
    parser.add_argument("--export_batch_size", type=int, default=8192)

    parser.add_argument("--max_steps_per_epoch", type=int, default=0, help="0 = full epoch; >0 caps steps per epoch for fast pilot testing on real data.")
    parser.add_argument("--max_val_batches", type=int, default=0, help="0 = full validation; >0 caps validation batches for fast pilot testing.")
    parser.add_argument("--max_seq_rows", type=int, default=0,help="0 = load all sequences; >0 loads only the first N rows for fast pilot testing.")

    args = parser.parse_args()

    # Hopper / H800 Hardware Accelerations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    distributed, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if rank != 0:
        logger.setLevel(logging.WARNING)

    os.makedirs(args.output_dir, exist_ok=True)

    # TensorBoard setup
    tb_writer = None
    if rank == 0:
        if HAS_TENSORBOARD:
            tb_dir = args.tb_log_dir or os.path.join(args.output_dir, "tb_logs")
            tb_writer = SummaryWriter(log_dir=tb_dir)
            logger.info("TensorBoard logging initialized at: %s", tb_dir)
            tb_writer.add_text("args", json.dumps(vars(args), indent=2))
        else:
            logger.warning("torch.utils.tensorboard not found; proceeding with console logging.")

    # 1. Load Data
    if rank == 0:
        logger.info("Loading Parquet data: items='%s', seq='%s'...", args.items_path, args.seq_path)
    items_df = pd.read_parquet(args.items_path)

    if args.max_seq_rows > 0:
        logger.info("Pilot mode: reading only first %d sequence rows...", args.max_seq_rows)
        # Fast row-group read via pyarrow or slice
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(args.seq_path)
            # Read just the first row group (~10k-50k rows in 1 second)
            seq_df = pf.read_row_group(0).to_pandas()
            if len(seq_df) > args.max_seq_rows:
                seq_df = seq_df.iloc[:args.max_seq_rows]
        except Exception:
            seq_df = pd.read_parquet(args.seq_path).iloc[:args.max_seq_rows]
    else:
        seq_df = pd.read_parquet(args.seq_path)

    # 2. Build Feature Store
    feature_store_dtype = torch.bfloat16 if (device.type == "cuda" and not args.fp16) else torch.float32
    store, item_to_idx, idx_to_item, cardinalities, cat_cols = build_feature_store(
        items_df,
        item_id_col=args.item_id_col,
        embedding_col=args.embedding_col,
        has_emb_col=args.has_emb_col,
        embedding_dim=args.raw_emb_dim,
        cat_col_candidates=DEFAULT_CAT_COLS,
        dtype=feature_store_dtype,
    )

    # 3. Create Datasets
    if args.split_mode == "user_split":
        unique_users = seq_df.index.to_numpy()
        np.random.shuffle(unique_users)
        split_idx = int(len(unique_users) * (1.0 - args.val_ratio))
        train_df = seq_df.iloc[unique_users[:split_idx]].reset_index(drop=True)
        valid_df = seq_df.iloc[unique_users[split_idx:]].reset_index(drop=True)
    else:
        train_df, valid_df = seq_df, seq_df

    train_dataset = FastNextItemDataset(
        train_df,
        item_to_idx=item_to_idx,
        max_seq_len=args.max_seq_len,
        seq_col=args.seq_col,
        min_history=args.min_history,
        max_examples_per_sequence=args.max_examples_per_sequence,
        split="train",
        split_mode=args.split_mode,
    )

    valid_dataset = FastNextItemDataset(
        valid_df,
        item_to_idx=item_to_idx,
        max_seq_len=args.max_seq_len,
        seq_col=args.seq_col,
        min_history=args.min_history,
        max_examples_per_sequence=0,
        split="valid",
        split_mode=args.split_mode,
    )

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    # 4. Build Model
    log_q = None
    if not args.no_log_q:
        log_q = estimate_log_q(seq_df, item_to_idx, len(idx_to_item) - 1, smoothing=args.log_q_smoothing).to(device)

    item_hidden_dims = tuple(int(x.strip()) for x in args.item_dnn_hidden_dims.split(",") if x.strip())
    item_dnn = ItemDNN(
        sem_dim=args.raw_emb_dim,
        cat_cardinalities=cardinalities,
        cat_cols=cat_cols,
        cat_emb_dim=args.cat_emb_dim,
        output_dim=args.output_dim,
        hidden_dims=item_hidden_dims,
        dropout=args.dropout,
    )

    seq_encoder = HSTUSequenceEncoder(
        hidden_dim=args.output_dim,
        max_seq_len=args.max_seq_len,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )

    model = SequentialRecModel(
        feature_store=store,
        item_dnn=item_dnn,
        seq_encoder=seq_encoder,
        temperature=args.temperature,
        learnable_temperature=args.learnable_temperature,
        log_q=log_q,
        use_log_q=(not args.no_log_q),
        log_q_weight=args.log_q_weight,
        cross_gpu_negatives=args.cross_gpu_negatives,
    ).to(device)

    if args.compile and hasattr(torch, "compile"):
        logger.info("Applying torch.compile() for Hopper execution...")
        model = torch.compile(model)

    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )

    # 5. Optimizer & LR Scheduler
    use_fused_adam = (device.type == "cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=use_fused_adam,
    )

    total_training_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_training_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    # Precision & Scaler
    amp_dtype = torch.float16 if args.fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler(device.type) if (args.fp16 and device.type == "cuda") else None

    # Resumption handling
    start_epoch = 1
    global_step = 0
    best_metric = -float("inf") if args.metric_for_best != "loss" else float("inf")

    last_checkpoint_path = os.path.join(args.output_dir, "checkpoint_last.pt")
    resume_target = args.resume
    if resume_target is None and args.auto_resume and os.path.isfile(last_checkpoint_path):
        resume_target = last_checkpoint_path

    if resume_target:
        start_epoch, global_step, best_metric = load_checkpoint(
            resume_target, model, optimizer, scheduler, scaler, device=device
        )

    # Graceful shutdown handler
    def handle_interrupt(signum, frame):
        logger.warning("Execution interrupted (signal %d). Saving emergency checkpoint...", signum)
        if rank == 0:
            emergency_path = os.path.join(args.output_dir, "checkpoint_interrupted.pt")
            save_checkpoint(
                emergency_path, epoch, global_step, model, optimizer, scheduler, scaler,
                best_metric, args, cardinalities, cat_cols
            )
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    # 6. Training & Validation Loop
    logger.info("Starting training: Epochs %d..%d | Steps/epoch: %d | Warmup steps: %d",
                start_epoch, args.epochs, len(train_loader), warmup_steps)

    for epoch in range(start_epoch, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_start = time.time()
        train_loss, global_step = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            amp_dtype=amp_dtype,
            grad_clip_norm=args.grad_clip_norm,
            tb_writer=tb_writer,
            log_freq=args.log_freq,
            global_step=global_step,
            max_steps=args.max_steps_per_epoch
        )

        val_metrics = evaluate(model, valid_loader, device, amp_dtype, args.max_val_batches)
        epoch_sec = time.time() - epoch_start

        if rank == 0:
            logger.info(
                "--- Epoch %d/%d (%d sec) | Train Loss: %.5f | Val Loss: %.5f | "
                "Val R@1: %.4f | Val R@5: %.4f | Val R@10: %.4f | Val R@50: %.4f | Val MRR: %.4f | Cos Gap: %.4f ---",
                epoch, args.epochs, int(epoch_sec), train_loss, val_metrics["loss"],
                val_metrics["recall@1"], val_metrics["recall@5"], val_metrics["recall@10"],
                val_metrics["recall@50"], val_metrics["mrr"], val_metrics["cosine_gap"],
            )

            if tb_writer is not None:
                tb_writer.add_scalar("epoch/train_loss", train_loss, epoch)
                tb_writer.add_scalar("eval/loss", val_metrics["loss"], epoch)
                tb_writer.add_scalar("eval/recall@1", val_metrics["recall@1"], epoch)
                tb_writer.add_scalar("eval/recall@5", val_metrics["recall@5"], epoch)
                tb_writer.add_scalar("eval/recall@10", val_metrics["recall@10"], epoch)
                tb_writer.add_scalar("eval/recall@50", val_metrics["recall@50"], epoch)
                tb_writer.add_scalar("eval/mrr", val_metrics["mrr"], epoch)
                tb_writer.add_scalar("eval/ndcg@10", val_metrics["ndcg@10"], epoch)
                tb_writer.add_scalar("eval/pos_cosine_mean", val_metrics["pos_cosine"], epoch)
                tb_writer.add_scalar("eval/neg_cosine_mean", val_metrics["neg_cosine"], epoch)
                tb_writer.add_scalar("eval/cosine_gap", val_metrics["cosine_gap"], epoch)

            # Determine best metric
            curr_metric = val_metrics["loss"] if args.metric_for_best == "loss" else val_metrics[args.metric_for_best]
            is_best = (curr_metric < best_metric) if args.metric_for_best == "loss" else (curr_metric > best_metric)

            if is_best:
                best_metric = curr_metric
                save_checkpoint(
                    os.path.join(args.output_dir, "checkpoint_best.pt"),
                    epoch, global_step, model, optimizer, scheduler, scaler,
                    best_metric, args, cardinalities, cat_cols
                )
                logger.info("New best %s: %.5f! Saved checkpoint_best.pt", args.metric_for_best, best_metric)

            # Save latest checkpoint every epoch
            save_checkpoint(
                last_checkpoint_path,
                epoch, global_step, model, optimizer, scheduler, scaler,
                best_metric, args, cardinalities, cat_cols
            )

            # Periodic saving
            if args.save_freq > 0 and (epoch % args.save_freq == 0):
                save_checkpoint(
                    os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt"),
                    epoch, global_step, model, optimizer, scheduler, scaler,
                    best_metric, args, cardinalities, cat_cols
                )

    # 7. Final Model Export
    if rank == 0:
        logger.info("Training completed. Exporting final weights & catalog embeddings...")
        raw_model = model.module if distributed else model

        torch.save({
            "item_dnn_state_dict": raw_model.item_dnn.state_dict(),
            "seq_encoder_state_dict": raw_model.seq_encoder.state_dict(),
            "action_emb_state_dict": raw_model.action_emb.state_dict(),
            "cardinalities": cardinalities,
            "cat_cols": cat_cols,
            "args": vars(args),
        }, os.path.join(args.output_dir, "stage1_rec_model.pt"))

        with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
            json.dump({
                "raw_emb_dim": args.raw_emb_dim,
                "cat_cols": cat_cols,
                "output_dim": args.output_dim,
                "cat_cardinalities": cardinalities,
            }, f, indent=2)

        export_catalog_embeddings(
            model=raw_model,
            idx_to_item=idx_to_item,
            output_dir=args.output_dir,
            device=device,
            batch_size=args.export_batch_size,
            amp_dtype=amp_dtype,
        )

        if tb_writer is not None:
            tb_writer.close()

        logger.info("Stage 1 Pipeline Finished. Artifacts saved in: %s", args.output_dir)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
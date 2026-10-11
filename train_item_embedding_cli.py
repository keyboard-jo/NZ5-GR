#!/usr/bin/env python3
"""
Stage 1: ItemDNN + HSTU Sequential Encoder with InfoNCE & LogQ Debias.
Optimized for NVIDIA H800 / Hopper GPUs with native BF16, TF32, and DDP cross-GPU negatives.

Expected Inputs:
  - items parquet (canonical_items.parquet):
      item_id: i64
      100, 101, 102, 112, 114, 115, 116, 117, 118, 119, 120, 121, 122: i64 (0 = missing)
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
    """
    def __init__(self, sem: torch.Tensor, has_mm: torch.Tensor, cats: torch.Tensor):
        super().__init__()
        self.register_buffer("sem", sem)         # [N+1, 1024]
        self.register_buffer("has_mm", has_mm)   # [N+1, 1]
        self.register_buffer("cats", cats)       # [N+1, 13]

    def get(self, indices: torch.Tensor):
        return self.sem[indices], self.has_mm[indices], self.cats[indices]


def build_feature_store(
    df: pd.DataFrame,
    item_id_col: str = "item_id",
    embedding_col: str = "mm_emb",
    has_emb_col: str = "has_mm_emb",
    embedding_dim: int = 1024,
    cat_col_candidates: Optional[List[str]] = None,
):
    if item_id_col not in df:
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
    sem = torch.zeros((n_plus_pad, embedding_dim), dtype=torch.float32)
    has_mm = torch.zeros((n_plus_pad, 1), dtype=torch.float32)

    has_mm_bool = np.zeros(len(df), dtype=bool)
    if has_emb_col in df.columns:
        has_mm_bool = df[has_emb_col].fillna(False).to_numpy(dtype=bool)

    if embedding_col in df.columns:
        raw_embs = df[embedding_col].tolist()
        for i, val in enumerate(raw_embs):
            if not has_mm_bool[i] or val is None:
                continue
            try:
                arr = np.asarray(val, dtype=np.float32).reshape(-1)
                if arr.size == embedding_dim and np.isfinite(arr).all():
                    sem[i + 1] = torch.from_numpy(arr)
                    has_mm[i + 1, 0] = 1.0
            except (TypeError, ValueError):
                continue
    else:
        logger.warning("Embedding column '%s' not found; using zero vectors.", embedding_col)

    # 2. Parse Categorical Columns (Handle string or int column names)
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
        
        # 0 is reserved for PAD / Missing. Non-zero values are mapped to 1..K
        non_zero_unique = np.sort(np.unique(vals[vals != 0]))
        cardinalities[col_name] = len(non_zero_unique) + 1  # +1 for index 0

        tensor = torch.zeros((n_plus_pad, 1), dtype=torch.long)
        if len(non_zero_unique) > 0:
            nonzero_mask = (vals != 0)
            # Fast binary search mapping
            positions = np.searchsorted(non_zero_unique, vals[nonzero_mask])
            matched = (non_zero_unique[positions] == vals[nonzero_mask])
            mapped_codes = np.where(matched, positions + 1, 0)
            tensor[1:, 0][nonzero_mask] = torch.from_numpy(mapped_codes).long()

        cat_arrays.append(tensor)

    cats = torch.cat(cat_arrays, dim=1) if cat_arrays else torch.zeros((n_plus_pad, 1), dtype=torch.long)
    if not cat_arrays:
        cardinalities["__dummy__"] = 1

    valid_mm_pct = 100.0 * float(has_mm[1:].mean()) if len(ids) else 0.0
    logger.info("Catalog Items=%s | with valid mm_emb=%.2f%% | Categoricals=%d",
                f"{len(ids):,}", valid_mm_pct, len(matched_cat_cols))

    return ItemFeatureStore(sem, has_mm, cats), item_to_idx, idx_to_item, cardinalities, [str(c) for c in matched_cat_cols]


class FastNextItemDataset(Dataset):
    """
    Parses sequences containing dicts:
      [{"item_id": 2905777, "action_type": 0, "timestamp": 1745919240}, ...]
    Leverages item_id, action_type, and chronological ordering.
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
    ):
        if split not in {"train", "valid"}:
            raise ValueError("split must be 'train' or 'valid'")

        self.samples = []
        dropped_short = 0
        raw_sequences = seq_df[seq_col].tolist()

        for raw_seq in raw_sequences:
            if not raw_seq or not isinstance(raw_seq, (list, np.ndarray)):
                continue

            item_seq, act_seq = [], []
            for interaction in raw_seq:
                if isinstance(interaction, dict):
                    iid = interaction.get("item_id")
                    act = interaction.get("action_type", 0)
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
                    # action_type: map 0 -> 1, 1 -> 2 (0 is reserved for PAD)
                    act_seq.append(1 if act == 0 else 2)

            if len(item_seq) < min_history + 1:
                dropped_short += 1
                continue
            
            if split == "valid":
                # Predict the final interaction from all preceding interactions.
                end_positions = [len(item_seq) - 1]
            else:
                # Never train on the final interaction reserved for validation.
                end_positions = list(range(min_history, len(item_seq) - 1))

                if max_examples_per_sequence > 0:
                    end_positions = end_positions[-max_examples_per_sequence:]

            if max_examples_per_sequence > 0:
                end_positions = end_positions[-max_examples_per_sequence:]

            for end in end_positions:
                start = max(0, end - max_seq_len)
                hist_items = item_seq[start:end]
                hist_actions = act_seq[start:end]
                target_item = item_seq[end]
                self.samples.append((hist_items, hist_actions, target_item))

        if not self.samples:
            raise ValueError("No valid next-item samples generated. Check ID overlap and sequence length.")

        logger.info("Total Next-Item Samples: %s (Skipped too short: %s)",
                    f"{len(self.samples):,}", f"{dropped_short:,}")
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
    Fuses multimodal embedding (1024-d), its presence flag, and 13 categorical features.
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
        # Fuses 1024-dim embedding + 1-dim presence flag
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
                nn.Dropout(dropout)
            ]
            in_dim = h
        layers += [nn.Linear(in_dim, output_dim)]
        self.mlp = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, sem: torch.Tensor, has_mm: torch.Tensor, cats: torch.Tensor):
        # sem: [B, 1024], has_mm: [B, 1], cats: [B, 13]
        h_sem = self.sem_proj(torch.cat([sem.float(), has_mm.float()], dim=-1))
        
        parts = [h_sem]
        if self.cat_cols:
            cat_embs = [emb(cats[..., i].long()) for i, emb in enumerate(self.cat_embeddings)]
            parts.append(torch.cat(cat_embs, dim=-1))

        out = self.mlp(torch.cat(parts, dim=-1))
        return F.normalize(out.float(), p=2, dim=-1)


class HSTUBlock(nn.Module):
    """
    Hierarchical Sequential Transduction Unit (HSTU) layer.
    Reference: Meta's 'Actions Speak Louder than Words' (ICML 2024).
    Replaces softmax attention and heavy MLPs with Pointwise Aggregated Gated Attention.
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

        # Linear projections for U, Q, K, V
        self.u_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.act = nn.SiLU()

        # Learnable relative positional bias rab_p
        self.rel_pos_bias = nn.Embedding(max_seq_len, num_heads)

        # Output projection and normalization
        self.attn_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor, causal_mask: torch.Tensor):
        # x: [B, L, D], valid_mask: [B, L] bool, causal_mask: [L, L] bool
        B, L, D = x.shape
        residual = x
        x_norm = self.pre_norm(x)

        # 1. Pointwise Projections (Eq. 1)
        u = self.act(self.u_proj(x_norm))                                            # [B, L, D]
        q = self.act(self.q_proj(x_norm)).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d_h]
        k = self.act(self.k_proj(x_norm)).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d_h]
        v = self.v_proj(x_norm).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)            # [B, H, L, d_h]

        # 2. Pointwise Aggregated Attention (Eq. 2)
        # S = (Q K^T) / sqrt(d_h) + rab
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)     # [B, H, L, L]

        # Relative distance bias
        pos = torch.arange(L, device=x.device)
        rel_dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp(min=0, max=L - 1)    # [L, L]
        rel_bias = self.rel_pos_bias(rel_dist).permute(2, 0, 1).unsqueeze(0)         # [1, H, L, L]
        scores = scores + rel_bias

        # Pointwise Attention with causal & key validity gating
        # Mask future positions (j > i) and padded keys (valid_mask == False)
        valid_keys = valid_mask.unsqueeze(1).unsqueeze(2)                            # [B, 1, 1, L]
        active_attn_mask = causal_mask.unsqueeze(0).unsqueeze(1) & valid_keys        # [B, 1, L, L]

        # Pointwise non-linear activation (SiLU)
        attn = self.act(scores) * active_attn_mask.float()                           # [B, H, L, L]

        # Attention aggregation
        attn_out = torch.matmul(attn, v)                                             # [B, H, L, d_h]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)               # [B, L, D]

        # 3. Pointwise Transformation (Eq. 3): Y = f2( Norm(A V) \odot U )
        gated = self.attn_norm(attn_out) * u
        y = self.out_proj(gated)
        y = self.dropout(y)

        return residual + y


class HSTUSequenceEncoder(nn.Module):
    """
    Multi-layer HSTU sequential transducer for user representation.
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        max_seq_len: int = 100,
        num_blocks: int = 2,
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
        # x: [B, L, D], valid_mask: [B, L]
        B, L, D = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(pos)

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
    and cross-GPU DDP negative sampling.
    """
    def __init__(
        self,
        feature_store: ItemFeatureStore,
        item_dnn: ItemDNN,
        seq_encoder: HSTUSequenceEncoder,
        temperature: float = 0.05,
        log_q: Optional[torch.Tensor] = None,
        cross_gpu_negatives: bool = True,
    ):
        super().__init__()
        self.feature_store = feature_store
        self.item_dnn = item_dnn
        self.seq_encoder = seq_encoder
        self.temperature = temperature
        self.cross_gpu_negatives = cross_gpu_negatives

        # Action embedding: 0 = PAD, 1 = action 0, 2 = action 1
        self.action_emb = nn.Embedding(3, item_dnn.output_dim, padding_idx=0)

        if log_q is None:
            log_q = torch.zeros(feature_store.sem.shape[0], dtype=torch.float32)
        self.register_buffer("log_q", log_q.float())

    def get_item_embeddings(self, indices: torch.Tensor) -> torch.Tensor:
        sem, has_mm, cats = self.feature_store.get(indices)
        return self.item_dnn(sem, has_mm, cats)

    def forward(
        self,
        history_items: torch.Tensor,
        history_actions: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Encode user sequence
        item_reps = self.get_item_embeddings(history_items)        # [B, L, D]
        act_reps = self.action_emb(history_actions)                # [B, L, D]
        hist_input = item_reps + act_reps                          # Fuse action semantics
        user = self.seq_encoder(hist_input, mask)                  # [B, D]

        # 2. Encode target items
        positive = self.get_item_embeddings(target)                # [B, D]
        candidate_log_q = self.log_q[target]                       # [B]

        # 3. Contrastive InfoNCE with optional DDP Cross-GPU Negatives
        if self.cross_gpu_negatives and dist.is_initialized() and dist.get_world_size() > 1:
            # Differentiable all_gather across H800 cluster
            all_positives = torch.cat(dist_nn.all_gather(positive), dim=0)        # [B * World, D]
            all_log_q = torch.cat(dist_nn.all_gather(candidate_log_q), dim=0)    # [B * World]

            logits = (user @ all_positives.T) / self.temperature - all_log_q.unsqueeze(0)
            labels = torch.arange(user.size(0), device=user.device) + dist.get_rank() * user.size(0)
        else:
            logits = (user @ positive.T) / self.temperature - candidate_log_q.unsqueeze(0)
            labels = torch.arange(user.size(0), device=user.device)

        loss = F.cross_entropy(logits, labels)
        return loss


# =========================================================================
# Training & Export Utilities
# =========================================================================

def train_epoch(model, loader, optimizer, device, epoch, amp_dtype):
    model.train()
    total_loss, n_batches = 0.0, 0
    start = time.time()

    for step, batch in enumerate(loader):
        history_items = batch["history_items"].to(device, non_blocking=True)
        history_actions = batch["history_actions"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Native Hopper BF16 Autocast (no GradScaler needed)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=(device.type == "cuda"),
        ):
            loss = model(history_items, history_actions, mask, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.detach())
        n_batches += 1

        if step > 0 and step % 100 == 0:
            elapsed = max(time.time() - start, 1e-6)
            logger.info("Epoch %d | Step %d/%d | Loss: %.5f | Speed: %.1f steps/s",
                        epoch, step, len(loader), float(loss.detach()), (step + 1) / elapsed)

    return total_loss / max(n_batches, 1)

@torch.no_grad()
def evaluate_epoch(model, loader, device, amp_dtype):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        history_items = batch["history_items"].to(device, non_blocking=True)
        history_actions = batch["history_actions"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=(device.type == "cuda"),
        ):
            loss = model(history_items, history_actions, mask, target)

        total_loss += loss.item()
        n_batches += 1

    if n_batches == 0:
        raise ValueError("Validation loader produced no batches.")

    return total_loss / n_batches

@torch.no_grad()
def export_catalog_embeddings(model, idx_to_item, output_dir, device, batch_size=8192, amp_dtype=torch.bfloat16):
    """
    Exports final L2-normalized item catalog embeddings for downstream SID tokenization.
    """
    model.eval()
    all_embs = []
    total = len(idx_to_item) - 1

    for start in range(1, total + 1, batch_size):
        end = min(start + batch_size, total + 1)
        indices = torch.arange(start, end, device=device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=(device.type == "cuda"),
        ):
            emb = model.get_item_embeddings(indices)
        all_embs.append(emb.float().cpu().numpy())

    if not all_embs:
        raise ValueError("No catalog items found to export.")

    embeddings = np.concatenate(all_embs, axis=0).astype(np.float32)
    item_ids = idx_to_item[1:]

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "item_embeddings.npy"), embeddings)
    np.save(os.path.join(output_dir, "item_ids.npy"), item_ids)

    pd.DataFrame({"item_id": item_ids, "embedding": list(embeddings)}).to_parquet(
        os.path.join(output_dir, "item_embeddings.parquet"), index=False
    )

    try:
        import faiss
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, os.path.join(output_dir, "item_index.faiss"))
        logger.info("Saved FAISS IndexFlatIP to %s", output_dir)
    except ImportError:
        logger.info("faiss not installed; skipping FAISS export.")

    logger.info("Successfully exported %s item embeddings (dim=%d) to %s",
                f"{len(item_ids):,}", embeddings.shape[1], output_dir)


def estimate_log_q(seq_df: pd.DataFrame, item_to_idx: Dict[int, int], n_items: int, smoothing: float = 1.0):
    counts = np.zeros(n_items + 1, dtype=np.float64)
    for raw_seq in seq_df["seq"].tolist():
        if not raw_seq:
            continue
        for interaction in raw_seq:
            iid = interaction.get("item_id") if isinstance(interaction, dict) else interaction
            if iid is not None:
                idx = item_to_idx.get(int(iid))
                if idx is not None:
                    counts[idx] += 1.0
    counts[1:] += smoothing
    counts[0] = 1.0
    probs = counts / counts.sum()
    return torch.from_numpy(np.log(probs).astype(np.float32))


# =========================================================================
# Main Execution
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 1: HSTU + ItemDNN Sequential Generative Retrieval")
    parser.add_argument("--items_path", default="canonical_items.parquet", help="Path to canonical items parquet")
    parser.add_argument("--seq_path", default="seq.parquet", help="Path to user sequence parquet")
    parser.add_argument("--output_dir", default="./output_stage1_embeddings")
    parser.add_argument("--item_id_col", default="item_id")
    parser.add_argument("--embedding_col", default="mm_emb")
    parser.add_argument("--has_emb_col", default="has_mm_emb")
    parser.add_argument("--raw_emb_dim", type=int, default=1024)
    parser.add_argument("--max_seq_len", type=int, default=100)
    parser.add_argument("--min_history", type=int, default=1)
    parser.add_argument("--max_examples_per_sequence", type=int, default=10,
                        help="0 = all prefixes; >0 caps training samples per sequence for fast throughput.")
    parser.add_argument("--output_dim", type=int, default=256)
    parser.add_argument("--cat_emb_dim", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=3, help="Number of HSTU layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads in HSTU")
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size per GPU (H800 can handle 1024-2048)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cross_gpu_negatives", action="store_true")
    args = parser.parse_args()

    # Hopper / H800 Hardware Accelerations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
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

    # 1. Load Parquet Datasets
    if rank == 0:
        logger.info("Loading Parquet datasets...")
    items_df = pd.read_parquet(args.items_path)
    seq_df = pd.read_parquet(args.seq_path)

    # 2. Build Feature Store
    store, item_to_idx, idx_to_item, cardinalities, cat_cols = build_feature_store(
        items_df,
        item_id_col=args.item_id_col,
        embedding_col=args.embedding_col,
        has_emb_col=args.has_emb_col,
        embedding_dim=args.raw_emb_dim,
        cat_col_candidates=DEFAULT_CAT_COLS,
    )

    # 3. Build Next-Item Dataset
    train_dataset = FastNextItemDataset(
        seq_df,
        item_to_idx=item_to_idx,
        max_seq_len=args.max_seq_len,
        min_history=args.min_history,
        max_examples_per_sequence=args.max_examples_per_sequence,
        split="train"
    )

    valid_dataset = FastNextItemDataset(
        seq_df,
        item_to_idx=item_to_idx,
        max_seq_len=args.max_seq_len,
        min_history=args.min_history,
        max_examples_per_sequence=0,
        split="valid",
    )

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
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

    # 4. Initialize Models
    log_q = estimate_log_q(seq_df, item_to_idx, len(idx_to_item) - 1).to(device)

    item_dnn = ItemDNN(
        sem_dim=args.raw_emb_dim,
        cat_cardinalities=cardinalities,
        cat_cols=cat_cols,
        cat_emb_dim=args.cat_emb_dim,
        output_dim=args.output_dim,
        hidden_dims=(512, 256),
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
        log_q=log_q,
        cross_gpu_negatives=(not args.no_cross_gpu_negatives),
    ).to(device)

    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )

    # H800 Fused AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=(device.type == "cuda")
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    # Hopper Native BF16 Precision
    amp_dtype = torch.bfloat16

    # 5. Training Loop
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, amp_dtype)
        valid_loss = evaluate_epoch(
            model, valid_loader, device, amp_dtype
        )
        scheduler.step()
        if rank == 0:
            logger.info(
                "Epoch %d/%d | Train Loss: %.6f | Valid Loss: %.6f",
                epoch, args.epochs, train_loss, valid_loss
            )

    # 6. Export Checkpoints & Final Catalog Embeddings
    if rank == 0:
        raw_model = model.module if distributed else model
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save({
            "item_dnn_state_dict": raw_model.item_dnn.state_dict(),
            "seq_encoder_state_dict": raw_model.seq_encoder.state_dict(),
            "cat_cardinalities": cardinalities,
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
            raw_model, idx_to_item, args.output_dir, device, batch_size=8192, amp_dtype=amp_dtype
        )
        logger.info("Stage 1 Pipeline Finished. Ready for Stage 2 SID Tokenization.")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
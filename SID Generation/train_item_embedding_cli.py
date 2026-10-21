#!/usr/bin/env python3
"""
train_item_embedding_cli.py
=============================================================================
Stage 1: ItemDNN + HSTU Sequential Generative Retrieval with InfoNCE & LogQ.
Optimized for NVIDIA H800 / Hopper GPUs (with multi-GPU DDP support).
=============================================================================
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
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

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


# =============================================================================
# Feature Store & Datasets
# =============================================================================

class ItemFeatureStore(nn.Module):
    def __init__(self, sem: torch.Tensor, has_mm: torch.Tensor, cats: torch.Tensor):
        super().__init__()
        self.register_buffer("sem", sem, persistent=False)
        self.register_buffer("has_mm", has_mm, persistent=False)
        self.register_buffer("cats", cats, persistent=False)

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
        raise ValueError("Expected one row per item; found duplicate item IDs.")

    ids = df[item_id_col].to_numpy(dtype=np.int64)
    item_to_idx = {int(iid): i + 1 for i, iid in enumerate(ids)}
    idx_to_item = np.zeros(len(ids) + 1, dtype=np.int64)
    idx_to_item[1:] = ids
    n_plus_pad = len(ids) + 1

    sem = torch.zeros((n_plus_pad, embedding_dim), dtype=dtype)
    has_mm = torch.zeros((n_plus_pad, 1), dtype=dtype)

    has_mm_bool = np.zeros(len(df), dtype=bool)
    if has_emb_col in df.columns:
        has_mm_bool = df[has_emb_col].fillna(False).to_numpy(dtype=bool)

    if embedding_col in df.columns:
        raw_embs = df[embedding_col].values
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

    if cat_col_candidates is None:
        cat_col_candidates = DEFAULT_CAT_COLS

    matched_cat_cols = [c for c in cat_col_candidates if c in df.columns or str(c) in df.columns]
    cat_arrays = []
    cardinalities = {}

    for col in matched_cat_cols:
        col_name = str(col)
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        non_zero_unique = np.sort(np.unique(vals[vals != 0]))
        cardinalities[col_name] = len(non_zero_unique) + 1

        col_arr = np.zeros(n_plus_pad, dtype=np.int64)
        if len(non_zero_unique) > 0:
            nonzero_mask = (vals != 0)
            positions = np.searchsorted(non_zero_unique, vals[nonzero_mask])
            matched = (non_zero_unique[positions] == vals[nonzero_mask])
            col_arr[np.where(nonzero_mask)[0] + 1] = np.where(matched, positions + 1, 0)
        cat_arrays.append(torch.from_numpy(col_arr).unsqueeze(1))

    cats = torch.cat(cat_arrays, dim=1) if cat_arrays else torch.zeros((n_plus_pad, 1), dtype=torch.long)
    if not cat_arrays:
        cardinalities["__dummy__"] = 1

    return (
        ItemFeatureStore(sem, has_mm, cats),
        item_to_idx,
        idx_to_item,
        cardinalities,
        [str(c) for c in matched_cat_cols],
    )


class PreprocessedParquetDataset(Dataset):
    """Loads flat, pre-materialized Parquet tables with zero CPU parsing overhead."""
    def __init__(self, parquet_path: str, max_seq_len: int = 100):
        if not os.path.isfile(parquet_path):
            raise FileNotFoundError(f"Parquet table not found: {parquet_path}")

        logger.info("Loading preprocessed dataset: %s...", parquet_path)
        t0 = time.time()
        if HAS_PYARROW:
            table = pq.read_table(parquet_path)
            self.hist_items = table["history_items"].to_pylist()
            self.hist_actions = table["history_actions"].to_pylist()
            self.targets = table["target"].to_numpy(zero_copy_only=False)
        else:
            df = pd.read_parquet(parquet_path)
            self.hist_items = df["history_items"].tolist()
            self.hist_actions = df["history_actions"].tolist()
            self.targets = df["target"].to_numpy(dtype=np.int64)

        self.max_seq_len = max_seq_len
        logger.info("Loaded %s ready-to-train samples in %.2f s.", f"{len(self.targets):,}", time.time() - t0)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        h_items = self.hist_items[idx]
        h_acts = self.hist_actions[idx]
        n = min(len(h_items), self.max_seq_len)

        padded_items = np.zeros(self.max_seq_len, dtype=np.int64)
        padded_acts = np.zeros(self.max_seq_len, dtype=np.int64)
        valid_mask = np.zeros(self.max_seq_len, dtype=np.bool_)

        padded_items[:n] = h_items[:n]
        padded_acts[:n] = h_acts[:n]
        valid_mask[:n] = True

        return {
            "history_items": torch.from_numpy(padded_items),
            "history_actions": torch.from_numpy(padded_acts),
            "mask": torch.from_numpy(valid_mask),
            "target": torch.tensor(int(self.targets[idx]), dtype=torch.long),
        }


class FastNextItemDataset(Dataset):
    def __init__(self, parquet_path: str, max_seq_len: int = 100):
        df = pd.read_parquet(parquet_path)
        self.hist_items = df["history_items"].to_list()
        self.hist_actions = df["history_actions"].to_list()
        self.targets = df["target"].to_numpy(dtype=np.int64)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        h_items = self.hist_items[idx]
        h_acts = self.hist_actions[idx]
        n = min(len(h_items), self.max_seq_len)

        padded_items = np.zeros(self.max_seq_len, dtype=np.int64)
        padded_acts = np.zeros(self.max_seq_len, dtype=np.int64)
        mask = np.zeros(self.max_seq_len, dtype=np.bool_)

        padded_items[:n] = h_items[:n]
        padded_acts[:n] = h_acts[:n]
        mask[:n] = True

        return {
            "history_items": torch.from_numpy(padded_items),
            "history_actions": torch.from_numpy(padded_acts),
            "mask": torch.from_numpy(mask),
            "target": torch.tensor(self.targets[idx], dtype=torch.long),
        }

# =============================================================================
# Architecture: ItemDNN & HSTU Sequence Encoder
# =============================================================================

class ItemDNN(nn.Module):
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

        in_dim = sem_hidden + (len(self.cat_cols) * cat_emb_dim)
        layers = []
        for h in hidden_dims[1:]:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, output_dim)]
        self.mlp = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, sem: torch.Tensor, has_mm: torch.Tensor, cats: torch.Tensor):
        h_sem = self.sem_proj(torch.cat([sem.float(), has_mm.float()], dim=-1))
        parts = [h_sem]
        if self.cat_cols:
            cat_embs = [emb(cats[..., i].long()) for i, emb in enumerate(self.cat_embeddings)]
            parts.append(torch.cat(cat_embs, dim=-1))
        out = self.mlp(torch.cat(parts, dim=-1))
        return F.normalize(out.float(), p=2, dim=-1)


class HSTUBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, max_seq_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_seq_len = max_seq_len

        self.in_proj = nn.Linear(dim, 4 * dim)
        self.act = nn.SiLU()
        self.rel_pos_bias = nn.Embedding(max_seq_len, num_heads)
        self.attn_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor, causal_mask: torch.Tensor):
        B, L, D = x.shape
        residual = x
        proj = self.in_proj(self.pre_norm(x))
        u_lin, q_lin, k_lin, v_lin = proj.chunk(4, dim=-1)

        u = self.act(u_lin)
        q = self.act(q_lin).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.act(k_lin).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_lin.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        pos = torch.arange(L, device=x.device)
        rel_dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp(min=0, max=self.max_seq_len - 1)
        scores = scores + self.rel_pos_bias(rel_dist).permute(2, 0, 1).unsqueeze(0)

        active_mask = (causal_mask.unsqueeze(0).unsqueeze(1) &
                       valid_mask.unsqueeze(1).unsqueeze(2) &
                       valid_mask.unsqueeze(1).unsqueeze(3))
        attn = self.act(scores) * active_mask.float()
        attn_out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)

        gated = self.attn_norm(attn_out) * u
        return residual + self.dropout(self.out_proj(gated))


class HSTUSequenceEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, max_seq_len: int = 100, num_blocks: int = 3, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.blocks = nn.ModuleList([
            HSTUBlock(dim=hidden_dim, num_heads=num_heads, max_seq_len=max_seq_len, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        x = (x + self.pos_emb(pos)) * valid_mask.unsqueeze(-1).to(x.dtype)

        causal_mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
        for block in self.blocks:
            x = block(x, valid_mask=valid_mask, causal_mask=causal_mask)

        x = self.final_norm(x)
        lengths = valid_mask.long().sum(dim=1).clamp_min(1)
        last_repr = x[torch.arange(B, device=x.device), lengths - 1]
        return F.normalize(last_repr.float(), p=2, dim=-1)


class SequentialRecModel(nn.Module):
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

        # LayerNorm Scale Alignment (Prevents action embedding from swamping item representations)
        self.input_layer_norm = nn.LayerNorm(item_dnn.output_dim)

        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))
        else:
            self.register_buffer("temperature", torch.tensor(temperature, dtype=torch.float32))

        # 4-Token Action Vocabulary: 0=PAD, 1=Exposure, 2=Click, 3=UNK
        self.action_emb = nn.Embedding(4, item_dnn.output_dim, padding_idx=0)

        if log_q is None:
            log_q = torch.zeros(feature_store.sem.shape[0], dtype=torch.float32)
        self.register_buffer("log_q", log_q.float(), persistent=False)

    def get_item_embeddings(self, indices: torch.Tensor) -> torch.Tensor:
        sem, has_mm, cats = self.feature_store.get(indices)
        emb = self.item_dnn(sem, has_mm, cats)
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
            logits = (sim_matrix / t) - (self.log_q_weight * all_log_q.unsqueeze(0) if self.use_log_q else 0)
            labels = torch.arange(user.size(0), device=user.device) + dist.get_rank() * user.size(0)
        else:
            sim_matrix = user @ positive.T
            logits = (sim_matrix / t) - (self.log_q_weight * candidate_log_q.unsqueeze(0) if self.use_log_q else 0)
            labels = torch.arange(user.size(0), device=user.device)

        loss = F.cross_entropy(logits, labels)
        return loss, logits, sim_matrix, labels, user

    def forward(self, history_items: torch.Tensor, history_actions: torch.Tensor, mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss, _, _, _, _ = self.compute_loss_and_logits(history_items, history_actions, mask, target)
        return loss


# =============================================================================
# Metrics, Training & Checkpoints
# =============================================================================

@torch.no_grad()
def compute_ranking_metrics(sim_matrix: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    B = sim_matrix.size(0)
    target_scores = sim_matrix.gather(1, labels.unsqueeze(1))
    ranks = (sim_matrix > target_scores).sum(dim=1) + 1

    r1 = (ranks <= 1).float().mean().item()
    r5 = (ranks <= 5).float().mean().item()
    r10 = (ranks <= 10).float().mean().item()
    r50 = (ranks <= 50).float().mean().item()
    mrr = (1.0 / ranks.float()).mean().item()
    ndcg10 = torch.where(ranks <= 10, 1.0 / torch.log2(ranks.float() + 1.0), torch.zeros_like(ranks, dtype=torch.float)).mean().item()

    diag_mask = torch.eye(B, device=sim_matrix.device, dtype=torch.bool)
    pos_cos = sim_matrix.diagonal().mean().item()
    neg_cos = sim_matrix[~diag_mask].mean().item() if B > 1 else 0.0

    return {
        "recall@1": r1, "recall@5": r5, "recall@10": r10, "recall@50": r50,
        "mrr": mrr, "ndcg@10": ndcg10, "pos_cosine": pos_cos, "neg_cosine": neg_cos,
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
    tb_writer: Optional[Any] = None,
    log_freq: int = 50,
    global_step: int = 0,
    max_steps: int = 0,
) -> Tuple[float, int]:
    model.train()
    total_loss, n_batches = 0.0, 0
    batch_start = time.time()

    for step, batch in enumerate(loader):
        if max_steps > 0 and step >= max_steps:
            logger.info("Cap max_steps_per_epoch (%d) reached; finishing epoch.", max_steps)
            break
        history_items = batch["history_items"].to(device, non_blocking=True)
        history_actions = batch["history_actions"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
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
            samples_per_sec = (log_freq / elapsed) * history_items.size(0)
            current_lr = optimizer.param_groups[0]["lr"]

            logger.info("Epoch %d | Step %d/%d (Global %d) | Loss: %.5f | GradNorm: %.2f | LR: %.2e | %.1f samples/s",
                        epoch, step, len(loader), global_step, loss_val, float(grad_norm), current_lr, samples_per_sec)

            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", loss_val, global_step)
                tb_writer.add_scalar("train/learning_rate", current_lr, global_step)
                tb_writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                tb_writer.add_scalar("train/throughput", samples_per_sec, global_step)
            batch_start = time.time()

    return total_loss / max(n_batches, 1), global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_val_batches: int = 0,
    tb_writer: Optional[Any] = None,
    global_step: int = 0,
) -> Dict[str, float]:
    model.eval()
    total_loss, n_batches = 0.0, 0
    metric_sums = {k: 0.0 for k in ["recall@1", "recall@5", "recall@10", "recall@50", "mrr", "ndcg@10", "pos_cosine", "neg_cosine", "cosine_gap"]}
    eval_model = model.module if isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)) else model

    for step, batch in enumerate(loader):
        if max_val_batches > 0 and step >= max_val_batches:
            break
        history_items = batch["history_items"].to(device, non_blocking=True)
        history_actions = batch["history_actions"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            loss, logits, sim_matrix, labels, _ = eval_model.compute_loss_and_logits(history_items, history_actions, mask, target)

        total_loss += float(loss.detach())
        for k, v in compute_ranking_metrics(sim_matrix, labels).items():
            metric_sums[k] += v
        n_batches += 1

    if n_batches == 0:
        raise ValueError("Validation loader produced zero batches.")

    results = {k: v / n_batches for k, v in metric_sums.items()}
    results["loss"] = total_loss / n_batches

    # Log all validation metrics and loss to TensorBoard
    if tb_writer is not None:
        for k, v in results.items():
            tb_writer.add_scalar(f"val/{k}", v, global_step)

    return results

def save_checkpoint(path: str, epoch: int, global_step: int, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, scaler, best_metric: float, args, cardinalities, cat_cols):
    raw = model.module if isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)) else model
    state = {
        "epoch": epoch, "global_step": global_step,
        "item_dnn_state_dict": raw.item_dnn.state_dict(),
        "seq_encoder_state_dict": raw.seq_encoder.state_dict(),
        "action_emb_state_dict": raw.action_emb.state_dict(),
        "input_layer_norm_state_dict": raw.input_layer_norm.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric, "args": vars(args),
        "cardinalities": cardinalities, "cat_cols": cat_cols,
    }
    if scheduler: state["scheduler_state_dict"] = scheduler.state_dict()
    if scaler: state["scaler_state_dict"] = scaler.state_dict()

    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
    logger.info("Saved atomic checkpoint to %s", path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None, scaler=None, device=torch.device("cpu")):
    checkpoint = torch.load(path, map_location=device)
    raw = model.module if isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)) else model
    raw.item_dnn.load_state_dict(checkpoint["item_dnn_state_dict"])
    raw.seq_encoder.load_state_dict(checkpoint["seq_encoder_state_dict"])
    if "action_emb_state_dict" in checkpoint: raw.action_emb.load_state_dict(checkpoint["action_emb_state_dict"])
    if "input_layer_norm_state_dict" in checkpoint: raw.input_layer_norm.load_state_dict(checkpoint["input_layer_norm_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint: optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint: scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler and "scaler_state_dict" in checkpoint: scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint.get("epoch", 0) + 1, checkpoint.get("global_step", 0), checkpoint.get("best_metric", -float("inf"))


@torch.no_grad()
def export_catalog_embeddings(model: nn.Module, idx_to_item: np.ndarray, output_dir: str, device: torch.device, batch_size: int = 8192, amp_dtype: torch.dtype = torch.bfloat16):
    raw = model.module if isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)) else model
    raw.eval()
    all_embs = []
    total = len(idx_to_item) - 1
    logger.info("Exporting %s catalog embeddings in batches of %d...", f"{total:,}", batch_size)

    for start in range(1, total + 1, batch_size):
        end = min(start + batch_size, total + 1)
        indices = torch.arange(start, end, device=device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            emb = raw.get_item_embeddings(indices)
        all_embs.append(emb.float().cpu().numpy())

    embeddings = np.concatenate(all_embs, axis=0).astype(np.float32)
    item_ids = idx_to_item[1:]
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "item_embeddings.npy"), embeddings)
    np.save(os.path.join(output_dir, "item_ids.npy"), item_ids)

    parquet_path = os.path.join(output_dir, "item_embeddings.parquet")
    if HAS_PYARROW:
        try:
            emb_list_array = pa.FixedSizeListArray.from_arrays(pa.array(embeddings.ravel(), type=pa.float32()), embeddings.shape[1])
            table = pa.Table.from_arrays([pa.array(item_ids, type=pa.int64()), emb_list_array], names=["item_id", "embedding"])
            pq.write_table(table, parquet_path)
        except Exception:
            pd.DataFrame({"item_id": item_ids, "embedding": list(embeddings)}).to_parquet(parquet_path, index=False)
    else:
        pd.DataFrame({"item_id": item_ids, "embedding": list(embeddings)}).to_parquet(parquet_path, index=False)

    if HAS_FAISS:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, os.path.join(output_dir, "item_index.faiss"))

    logger.info("Exported %s catalog embeddings (dim=%d) to %s", f"{len(item_ids):,}", embeddings.shape[1], output_dir)


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.001):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 1: HSTU + ItemDNN Sequential Generative Retrieval")

    # File Paths (Fully Parameterized)
    parser.add_argument("--items_path", default="canonical_items.parquet", help="Path to canonical items parquet")
    parser.add_argument("--train_samples_path", default=None, help="Path to preprocessed train_samples.parquet")
    parser.add_argument("--val_samples_path", default=None, help="Path to preprocessed val_samples.parquet")
    parser.add_argument("--seq_path", default="seq.parquet", help="Path to raw user sequence parquet")
    parser.add_argument("--output_dir", default="./output_stage1_embeddings")

    # Columns (Parameterized)
    parser.add_argument("--item_id_col", default="item_id")
    parser.add_argument("--embedding_col", default="mm_emb")
    parser.add_argument("--has_emb_col", default="has_mm_emb")
    parser.add_argument("--seq_col", default="seq")
    parser.add_argument("--cat_cols", default=None, help="Comma-separated categorical columns")

    # Hyperparameters (Parameterized)
    parser.add_argument("--max_seq_len", type=int, default=100)
    parser.add_argument("--min_history", type=int, default=1)
    parser.add_argument("--max_examples_per_sequence", type=int, default=10)
    parser.add_argument("--raw_emb_dim", type=int, default=1024)
    parser.add_argument("--output_dim", type=int, default=256)
    parser.add_argument("--cat_emb_dim", type=int, default=32)
    parser.add_argument("--item_dnn_hidden_dims", default="512,256")
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--learnable_temperature", action="store_true")

    # Training Options (Parameterized)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr_ratio", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no_log_q", action="store_true")
    parser.add_argument("--log_q_weight", type=float, default=1.0)
    parser.add_argument("--cross_gpu_negatives", action="store_true")

    # Hardware & Debug Flags
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--metric_for_best", default="recall@10", choices=["recall@10", "recall@5", "loss", "mrr"])
    parser.add_argument("--tb_log_dir", type=str, default=None)
    parser.add_argument("--export_batch_size", type=int, default=8192)
    parser.add_argument("--max_steps_per_epoch", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--max_seq_rows", type=int, default=0)

    args = parser.parse_args()

    # GPU optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    distributed, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if rank != 0:
        logger.setLevel(logging.WARNING)

    os.makedirs(args.output_dir, exist_ok=True)
    tb_writer = None
    if rank == 0 and HAS_TENSORBOARD:
        tb_writer = SummaryWriter(log_dir=args.tb_log_dir or os.path.join(args.output_dir, "tb_logs"))

    # 1. Feature Store
    items_df = pd.read_parquet(args.items_path)
    cat_cols_list = [c.strip() for c in args.cat_cols.split(",")] if args.cat_cols else DEFAULT_CAT_COLS
    store_dtype = torch.bfloat16 if (device.type == "cuda" and not args.fp16) else torch.float32

    store, item_to_idx, idx_to_item, cardinalities, cat_cols = build_feature_store(
        items_df, item_id_col=args.item_id_col, embedding_col=args.embedding_col,
        has_emb_col=args.has_emb_col, embedding_dim=args.raw_emb_dim,
        cat_col_candidates=cat_cols_list, dtype=store_dtype,
    )

    # 2. Datasets
    if args.train_samples_path and args.val_samples_path:
        train_dataset = PreprocessedParquetDataset(args.train_samples_path, max_seq_len=args.max_seq_len)
        valid_dataset = PreprocessedParquetDataset(args.val_samples_path, max_seq_len=args.max_seq_len)
    else:
        seq_df = pd.read_parquet(args.seq_path)
        if args.max_seq_rows > 0:
            seq_df = seq_df.iloc[:args.max_seq_rows]
        train_dataset = FastNextItemDataset(seq_df, item_to_idx, max_seq_len=args.max_seq_len, split="train")
        valid_dataset = FastNextItemDataset(seq_df, item_to_idx, max_seq_len=args.max_seq_len, split="valid")

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(sampler is None),
        sampler=sampler, num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True, persistent_workers=(args.num_workers > 0), prefetch_factor=4 if args.num_workers > 0 else None,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=False, persistent_workers=(args.num_workers > 0), prefetch_factor=4 if args.num_workers > 0 else None,
    )

    # 3. Model
    item_dnn = ItemDNN(
        sem_dim=args.raw_emb_dim, cat_cardinalities=cardinalities, cat_cols=cat_cols,
        cat_emb_dim=args.cat_emb_dim, output_dim=args.output_dim,
        hidden_dims=tuple(int(x.strip()) for x in args.item_dnn_hidden_dims.split(",")),
        dropout=args.dropout,
    )
    seq_encoder = HSTUSequenceEncoder(
        hidden_dim=args.output_dim, max_seq_len=args.max_seq_len,
        num_blocks=args.num_blocks, num_heads=args.num_heads, dropout=args.dropout,
    )
    model = SequentialRecModel(
        feature_store=store, item_dnn=item_dnn, seq_encoder=seq_encoder,
        temperature=args.temperature, learnable_temperature=args.learnable_temperature,
        use_log_q=(not args.no_log_q), log_q_weight=args.log_q_weight,
        cross_gpu_negatives=args.cross_gpu_negatives,
    ).to(device)

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    if distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # 4. Optimizer & Schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=(device.type == "cuda"))
    total_training_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=int(total_training_steps * args.warmup_ratio), total_steps=total_training_steps, min_lr_ratio=args.min_lr_ratio)
    amp_dtype = torch.float16 if args.fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler(device.type) if (args.fp16 and device.type == "cuda") else None

    # Resumption
    start_epoch, global_step = 1, 0
    best_metric = -float("inf") if args.metric_for_best != "loss" else float("inf")
    resume_target = args.resume or (os.path.join(args.output_dir, "checkpoint_last.pt") if args.auto_resume else None)
    if resume_target and os.path.isfile(resume_target):
        start_epoch, global_step, best_metric = load_checkpoint(resume_target, model, optimizer, scheduler, scaler, device=device)

    # 5. Training Loop
    logger.info("Beginning training: Epochs %d..%d | Steps/epoch: %d", start_epoch, args.epochs, len(train_loader))
    for epoch in range(start_epoch, args.epochs + 1):
        if sampler: sampler.set_epoch(epoch)
        t0 = time.time()
        train_loss, global_step = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch, amp_dtype, args.grad_clip_norm, tb_writer, args.log_freq, global_step, args.max_steps_per_epoch)
        val_metrics = evaluate(model, valid_loader, device, amp_dtype, args.max_val_batches, tb_writer, global_step)

        if rank == 0:
            epoch_duration = int(time.time() - t0)

            logger.info("--- Epoch %d/%d (%ds) | Train Loss: %.5f | Val Loss: %.5f | Val R@10: %.4f | MRR: %.4f | Gap: %.4f ---",
                        epoch, args.epochs, epoch_duration, train_loss, val_metrics["loss"], val_metrics["recall@10"], val_metrics["mrr"], val_metrics["cosine_gap"])
            curr_metric = val_metrics["loss"] if args.metric_for_best == "loss" else val_metrics[args.metric_for_best]
            is_best = (curr_metric < best_metric) if args.metric_for_best == "loss" else (curr_metric > best_metric)
            if is_best:
                best_metric = curr_metric
                save_checkpoint(os.path.join(args.output_dir, "checkpoint_best.pt"), epoch, global_step, model, optimizer, scheduler, scaler, best_metric, args, cardinalities, cat_cols)
            save_checkpoint(os.path.join(args.output_dir, "checkpoint_last.pt"), epoch, global_step, model, optimizer, scheduler, scaler, best_metric, args, cardinalities, cat_cols)

            if tb_writer is not None:
                # Log epoch-level metrics using either `epoch` or `global_step` as the x-axis
                tb_writer.add_scalar("train/epoch_loss", train_loss, epoch)
                tb_writer.add_scalar("train/epoch_duration_sec", epoch_duration, epoch)
                
                # If you want explicit epoch-indexed tracking for key metrics:
                tb_writer.add_scalar("val_epoch/recall_at_10", val_metrics["recall@10"], epoch)
                tb_writer.add_scalar("val_epoch/mrr", val_metrics["mrr"], epoch)
                tb_writer.add_scalar("val_epoch/cosine_gap", val_metrics["cosine_gap"], epoch)

    # 6. Final Export
    if rank == 0:
        logger.info("Exporting final Stage 1 artifacts...")
        raw = model.module if distributed else model
        torch.save({
            "item_dnn_state_dict": raw.item_dnn.state_dict(),
            "seq_encoder_state_dict": raw.seq_encoder.state_dict(),
            "action_emb_state_dict": raw.action_emb.state_dict(),
            "input_layer_norm_state_dict": raw.input_layer_norm.state_dict(),
            "cardinalities": cardinalities, "cat_cols": cat_cols, "args": vars(args),
        }, os.path.join(args.output_dir, "stage1_rec_model.pt"))

        export_catalog_embeddings(model=raw, idx_to_item=idx_to_item, output_dir=args.output_dir, device=device, batch_size=args.export_batch_size, amp_dtype=amp_dtype)
        if tb_writer: tb_writer.close()
        logger.info("Stage 1 Pipeline Finished. Artifacts saved in %s", args.output_dir)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
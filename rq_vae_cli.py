import os
import sys
import json
import math
import argparse
import time
import logging
from datetime import datetime
from collections import defaultdict

import h5py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import KMeans

try:
    import faiss  # faiss-gpu supplies the GPU K-Means backend
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

# Enable TensorFloat-32 for faster matrix multiplications on modern GPUs (Ampere/Hopper/Blackwell)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- GLOBAL CONSTANTS & HYPERPARAMETERS ---
RANDOM_SEED = 42
DEFAULT_INPUT_PATH = 'tencent_gr_parquet/train/mm_emb_82.h5'
DEFAULT_OUTPUT_PATH = 'tencent_gr_parquet/train/index_rqvae.json'
DEFAULT_MODEL_PATH = 'rqvae_model.pth'
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 4096
DEFAULT_INFERENCE_BATCH_SIZE = 16384
DEFAULT_MAX_SAMPLES = None
DEFAULT_INPUT_DIM = 1024
DEFAULT_HIDDEN_DIM = 64
DEFAULT_CODEBOOK_SIZE = 256
DEFAULT_NUM_CODEBOOKS = 4
DEFAULT_BETA = 0.25
DEFAULT_LR = 5e-4
DEFAULT_WARMUP_STEPS = 2_000
DEFAULT_LOG_INTERVAL = 200
DEFAULT_DEAD_CODE_FIX_INTERVAL = 500
DEFAULT_SAVE_INTERVAL = 5
DEFAULT_JITTER = 0.01
DEFAULT_ENTROPY_LOSS_WEIGHT = 0.05
DEFAULT_STAGE_WEIGHTS = [1.5, 1.2, 1.0, 1.0]
DEFAULT_USAGE_DECAY = 0.99
DEFAULT_DEAD_CODE_THRESHOLD = 0.1
DEFAULT_RUN_BASELINE = False

# Validation Defaults
DEFAULT_VAL_RATIO = 0.1
DEFAULT_VAL_INTERVAL = 1

# Model initialization constants
TRUNC_NORMAL_STD = 0.02
TRUNC_NORMAL_BOUND = 0.04
EPSILON = 1e-8

# Dataset & Loader constants
INIT_SAMPLE_SIZE = 500_000
INIT_BATCH_SIZE = 4096
BUFFER_CAPACITY_PER_STAGE = 20_000
DEFAULT_ITEM_ID_COL = "item_id"
DEFAULT_EMBEDDING_COL = "embedding"

# K-Means constants
KMEANS_N_INIT = 3          # FAISS `nredo` == sklearn `n_init` restarts
KMEANS_RANDOM_STATE = 42
KMEANS_N_ITER = 50         # FAISS Lloyd iterations per restart (GPU, so cheap)
KMEANS_ASSIGN_CHUNK = 200_000  # rows per FAISS search batch during label assignment

# Training & Optimization constants
GRAD_CLIP_MAX_NORM = 1.0
WEIGHT_DECAY = 1e-4
MIN_LR_RATIO = 0.01

# UI/Logging constants
BANNER_WIDTH = 85


# --- LOGGING & FORMATTING UTILITIES ---

def setup_logger(log_dir: str, log_filename: str = "training.log") -> logging.Logger:
    """Configures multi-sink logger to both stdout and a persistent file."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("RQVAE")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Stream Handler (stdout)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    log_path = os.path.join(log_dir, log_filename)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def log_banner(logger: logging.Logger, title: str, width: int = BANNER_WIDTH):
    logger.info("=" * width)
    logger.info(title.center(width))
    logger.info("=" * width)


def log_section(logger: logging.Logger, section_name: str, width: int = BANNER_WIDTH):
    logger.info(f"┌── [ {section_name} ] " + "─" * max(0, width - len(section_name) - 7))


def log_card(logger: logging.Logger, title: str, items: dict, width: int = BANNER_WIDTH):
    logger.info("┌" + "─" * (width - 2) + "┐")
    logger.info(f"│ {title.upper():<{width - 4}} │")
    logger.info("├" + "─" * (width - 2) + "┤")
    for key, val in items.items():
        k_str = f"  • {key}:"
        v_str = str(val)
        padding = width - len(k_str) - len(v_str) - 5
        logger.info(f"│{k_str} {' ' * max(0, padding)}{v_str}  │")
    logger.info("└" + "─" * (width - 2) + "┘")


# Single source of truth for the stdio metrics table: (header, column width, value format).
# Header, rows and footer are all derived from this list, so the header can never drift out of
# sync with the row actually printed (it previously did: the header declared 7 columns while the
# training loop emitted 11, which broke the whole table layout).
LOG_TABLE_COLUMNS = [
    ("Epoch", 5, "d"),
    ("Step", 6, "d"),
    ("Total Loss", 10, ".4f"),
    ("Recon Loss", 10, ".4f"),
    ("CB Loss", 9, ".4f"),
    ("CM Loss", 9, ".4f"),
    ("Perp #0", 9, ".1f"),
    ("Perp #N", 9, ".1f"),
    ("Res R0", 10, ".4f"),
    ("Res RN", 10, ".4f"),
    ("LR", 12, ".5e"),
]


def _table_rule(left: str, joint: str, right: str) -> str:
    return left + joint.join("─" * (w + 2) for _, w, _ in LOG_TABLE_COLUMNS) + right


def log_table_header(logger: logging.Logger):
    logger.info(_table_rule("┌", "┬", "┐"))
    logger.info("│" + "│".join(f" {title:<{w}} " for title, w, _ in LOG_TABLE_COLUMNS) + "│")
    logger.info(_table_rule("├", "┼", "┤"))


def log_table_row(logger: logging.Logger, epoch: int, step: int, loss: float, recon_loss: float,
                  cb_loss: float, cm_loss: float, perp_first: float, perp_last: float,
                  res_first: float, res_last: float, lr: float):
    """Emits one metrics row formatted with the exact column spec used by the header."""
    values = (epoch, step, loss, recon_loss, cb_loss, cm_loss,
              perp_first, perp_last, res_first, res_last, lr)
    cells = [f" {v:<{w}{spec}} " for (_, w, spec), v in zip(LOG_TABLE_COLUMNS, values)]
    logger.info("│" + "│".join(cells) + "│")


def log_table_footer(logger: logging.Logger):
    logger.info(_table_rule("└", "┴", "┘"))


# --- DATASET LOADER ---
class EmbeddingDataset(Dataset):
    """
    Dataset class for loading embeddings from an HDF5 or Parquet file.
    """
    def __init__(self, input_file_path: str, max_samples: int = None, logger: logging.Logger = None):
        super().__init__()
        self.input_file_path = input_file_path
        self.max_samples = max_samples

        if logger:
            logger.info(f"  ► Loading dataset into memory from: {input_file_path}")

        if self.input_file_path.endswith('.parquet') or self.input_file_path.endswith('.pq'):
            import pyarrow.parquet as pq

            table = pq.read_table(self.input_file_path, columns=[DEFAULT_ITEM_ID_COL, DEFAULT_EMBEDDING_COL])
            if self.max_samples is not None:
                table = table.slice(0, self.max_samples)

            oids = table[DEFAULT_ITEM_ID_COL].to_pandas().values
            embs = np.stack(table[DEFAULT_EMBEDDING_COL].to_numpy())
            self.item_ids = [str(oid) for oid in oids]
            self.embeddings = torch.from_numpy(embs).float()
        else:
            with h5py.File(self.input_file_path, 'r') as f:
                oid_ds = f[DEFAULT_ITEM_ID_COL]
                emb_ds = f[DEFAULT_EMBEDDING_COL]

                total_len = len(oid_ds)
                if self.max_samples is not None:
                    total_len = min(total_len, self.max_samples)

                oids = oid_ds[:total_len]
                embs = emb_ds[:total_len]

            self.item_ids = [oid.decode('utf-8') if isinstance(oid, bytes) else str(oid) for oid in oids]
            self.embeddings = torch.from_numpy(embs).float()

        if logger:
            logger.info(f"  ✔ Successfully loaded {len(self.item_ids):,} samples into memory (Shape: {self.embeddings.shape}).")

    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, idx):
        return self.embeddings[idx]


# --- MODEL ARCHITECTURE ---
class RQVAE(nn.Module):
    """
    Residual Quantized Variational AutoEncoder (RQ-VAE) model.
    """
    def __init__(
        self,
        input_dim=256,
        hidden_dim=64,
        beta=0.25,
        codebook_sizes=[256, 256, 256, 256],
        temperature=1.0,
        jitter=0.01,
        entropy_loss_weight=0.05,
        stage_weights=None,
        is_run_baseline: bool = False,
    ):
        super().__init__()
        self.is_run_baseline = is_run_baseline
        self.beta = beta
        self.codebook_sizes = codebook_sizes
        self.hidden_dim = hidden_dim
        self.num_stages = len(codebook_sizes)

        # Baseline disables the additional mechanisms.
        self.temperature = temperature
        self.jitter = 0.0 if is_run_baseline else jitter
        self.entropy_loss_weight = 0.0 if is_run_baseline else entropy_loss_weight
        self.stage_weights = (
            [1.0] * self.num_stages
            if is_run_baseline
            else (stage_weights if stage_weights is not None else [1.0] * self.num_stages)
        )

        # Encoder: 256 -> 256 -> 128 -> 128 -> 64
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )

        # Bottleneck Normalization
        self.bottleneck_norm = (
            None
            if is_run_baseline
            else nn.LayerNorm(hidden_dim)
        )

        # Decoder: 64 -> 128 -> 128 -> 256 -> 256 (Symmetrical)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

        # Codebooks: Unconstrained in magnitude to allow multi-scale residual decay
        self.codebooks = nn.ParameterList()
        for codebook_size in codebook_sizes:
            cb = nn.Parameter(torch.FloatTensor(codebook_size, hidden_dim))
            with torch.no_grad():
                nn.init.trunc_normal_(cb, std=TRUNC_NORMAL_STD, a=-TRUNC_NORMAL_BOUND, b=TRUNC_NORMAL_BOUND)
            self.codebooks.append(cb)


    def forward(self, batch_embs, compute_diagnostics: bool = False):
        # 1. Continuous Latent Space
        z = self.encoder(batch_embs)
        if self.bottleneck_norm is not None:
            z = self.bottleneck_norm(z)

        remainder = z + (torch.randn_like(z) * self.jitter) if (self.training and self.jitter > 0) else z

        quantized_sum = torch.zeros_like(z)

        total_codebook_loss = z.new_zeros(())
        total_commitment_loss = z.new_zeros(())
        weighted_entropy_loss = z.new_zeros(())

        clusters = []
        residuals = [remainder.detach()] if compute_diagnostics else None

        if not self.is_run_baseline:
            max_entropy = math.log(self.codebook_sizes[0])

        # 2. Sequential Residual Vector Quantization Cascade
        for d, (codebook, weight) in enumerate(zip(self.codebooks, self.stage_weights)):
            # Euclidean affinity: S = 2 * (r @ e.T) - ||e||^2
            e_sq = torch.sum(codebook ** 2, dim=-1, keepdim=True).t()
            dot = torch.matmul(remainder, codebook.t())
            logits = 2.0 * dot - e_sq

            # Discrete nearest-codebook assignment
            codebook_indices = logits.argmax(dim=-1)
            clusters.append(codebook_indices)
            quantized = codebook[codebook_indices]

            # Commitment + Codebook Loss
            cb_loss_d = F.mse_loss(quantized, remainder.detach())
            cm_loss_d = F.mse_loss(remainder, quantized.detach())
            total_codebook_loss = total_codebook_loss + cb_loss_d
            total_commitment_loss = total_commitment_loss + cm_loss_d

            if not self.is_run_baseline:
                probs = F.softmax(logits / self.temperature, dim=-1)
                avg_probs = probs.mean(dim=0)
                stage_entropy = -torch.sum(avg_probs * torch.log(avg_probs + EPSILON))
                max_entropy = math.log(self.codebook_sizes[d])
                stage_entropy_loss = (max_entropy - stage_entropy) / max_entropy
                weighted_entropy_loss = weighted_entropy_loss + weight * stage_entropy_loss

            quantized_sum = quantized_sum + quantized
            remainder = remainder - quantized.detach()

            if compute_diagnostics and residuals is not None:
                residuals.append(remainder.detach())


        rqvae_loss = total_codebook_loss + self.beta * total_commitment_loss

        # 3. Straight-Through Estimator (STE)
        latent_restored = z + (quantized_sum - z).detach()
        embeddings_restored = self.decoder(latent_restored)
        recon_loss = F.mse_loss(embeddings_restored, batch_embs)

        loss = recon_loss + rqvae_loss
        if not self.is_run_baseline:
            entropy_loss = weighted_entropy_loss / sum(self.stage_weights)
            loss = loss + self.entropy_loss_weight * entropy_loss
        else:
            entropy_loss = z.new_zeros(())

        clusters_tensor = torch.stack(clusters, dim=-1)

        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'rqvae_loss': rqvae_loss,
            'codebook_loss': total_codebook_loss,
            'commitment_loss': total_commitment_loss,
            'entropy_loss': entropy_loss,
            'clusters': clusters_tensor,
            'residuals_per_stage': residuals,
        }

    @torch.inference_mode()
    def encode_indices(self, batch_embs):
        """High-throughput inference matching the exact Euclidean Voronoi regions."""
        z = self.encoder(batch_embs)
        if self.bottleneck_norm is not None:
            z = self.bottleneck_norm(z)

        remainder = z
        all_indices = []
        for codebook in self.codebooks:
            e_sq = torch.sum(codebook ** 2, dim=-1, keepdim=True).t()
            dot = torch.matmul(remainder, codebook.t())
            logits = 2.0 * dot - e_sq
            idx = logits.argmax(dim=-1)
            all_indices.append(idx)
            remainder = remainder - codebook[idx]

        return torch.stack(all_indices, dim=-1)


# --- MODEL Diagnostic ---
class RQVAEDiagnostics:
    """Computes hard assignment metrics, residual statistics, SID uniqueness, and stability."""

    @staticmethod
    def compute_stage_diagnostics(clusters: torch.Tensor, codebook_sizes: list):
        """
        Calculates:
        - Hard code usage, dead-code %, max code frequency
        - Hard assignment entropy H = -sum(p * log p) and Perplexity = exp(H)
        """
        N, num_stages = clusters.shape
        metrics = []

        for d in range(num_stages):
            cb_size = codebook_sizes[d]
            idx = clusters[:, d]
            counts = torch.bincount(idx, minlength=cb_size).float()

            used_codes = (counts > 0).sum().item()
            dead_code_pct = (1.0 - (used_codes / cb_size)) * 100.0
            max_freq = (counts.max() / N).item()

            # Empirical distribution p_k for k where count > 0
            p = counts[counts > 0] / N
            entropy = -(p * torch.log(p + 1e-12)).sum().item()
            perplexity = math.exp(entropy)

            metrics.append({
                "stage": d,
                "used_codes": used_codes,
                "total_codes": cb_size,
                "dead_code_pct": dead_code_pct,
                "max_freq": max_freq,
                "entropy": entropy,
                "perplexity": perplexity,
            })
        return metrics

    @staticmethod
    def compute_residual_norms(residuals: list):
        """
        Computes E[||r_m||^2] as well as mean and median of ||r_m||_2 for each stage m in [0..M].
        """
        res_stats = []
        for d, r in enumerate(residuals):
            # r has shape (N, hidden_dim)
            sq_norm = torch.sum(r ** 2, dim=-1)            # ||r||^2
            l2_norm = torch.sqrt(sq_norm + 1e-12)          # ||r||_2

            e_sq = sq_norm.mean().item()
            mean_l2 = l2_norm.mean().item()
            median_l2 = l2_norm.median().item()

            res_stats.append({
                "stage": d,
                "expected_sq_norm": e_sq,
                "mean_l2": mean_l2,
                "median_l2": median_l2,
            })
        return res_stats

    @staticmethod
    def compute_sid_uniqueness(clusters: torch.Tensor):
        """
        Evaluates prefix uniqueness at each depth:
        prefix_depth_d = unique_prefixes / total_items.
        """
        N, depth = clusters.shape
        uniqueness_ratios = []

        for d in range(1, depth + 1):
            sub_codes = clusters[:, :d]
            # Fast GPU row-uniqueness
            unique_rows = torch.unique(sub_codes, dim=0).shape[0]
            uniqueness_ratios.append(unique_rows / N)

        return uniqueness_ratios

    @staticmethod
    def compute_assignment_stability(model, sample_batch):
        """Confirms repeated inference returns identical codes."""
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        raw_model.eval()
        with torch.inference_mode():
            codes1 = raw_model.encode_indices(sample_batch)
            codes2 = raw_model.encode_indices(sample_batch)
        stability = (codes1 == codes2).all().float().item()
        return stability


# --- K-MEANS (GPU / FAISS) ---
def _faiss_gpu_available() -> bool:
    """True when FAISS is importable and exposes at least one usable CUDA device."""
    if not FAISS_AVAILABLE:
        return False
    try:
        return faiss.get_num_gpus() > 0
    except Exception:
        return False


def _faiss_assign(x: np.ndarray, centroids: np.ndarray, use_gpu: bool) -> np.ndarray:
    """Nearest-centroid labels via a FAISS flat-L2 index (chunked to bound host memory)."""
    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(np.ascontiguousarray(centroids, dtype=np.float32))
    if use_gpu:
        index = faiss.index_cpu_to_all_gpus(index)

    labels = np.empty(x.shape[0], dtype=np.int64)
    for start in range(0, x.shape[0], KMEANS_ASSIGN_CHUNK):
        chunk = np.ascontiguousarray(x[start:start + KMEANS_ASSIGN_CHUNK], dtype=np.float32)
        _, idx = index.search(chunk, 1)
        labels[start:start + chunk.shape[0]] = idx.reshape(-1)
    return labels


def fit_kmeans(x: np.ndarray, n_clusters: int, logger: logging.Logger):
    """
    Fits K-Means on the GPU through FAISS and returns (centroids, labels).

    FAISS mirrors the previous sklearn setup (nredo == n_init restarts, k-means++ seeding) but runs
    the Lloyd iterations on the GPU. sklearn is only kept as a CPU fallback for environments without
    a FAISS GPU build, so the pipeline still runs unchanged there.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    n_clusters = int(min(n_clusters, x.shape[0]))

    if _faiss_gpu_available():
        try:
            kmeans = faiss.Kmeans(
                x.shape[1],
                n_clusters,
                niter=KMEANS_N_ITER,
                nredo=KMEANS_N_INIT,
                seed=KMEANS_RANDOM_STATE,
                verbose=False,
                gpu=True,
                min_points_per_centroid=1,
            )
            kmeans.train(x)
            centroids = np.ascontiguousarray(kmeans.centroids, dtype=np.float32)
            return centroids, _faiss_assign(x, centroids, use_gpu=True)
        except Exception as exc:
            logger.warning(f"    ⚠ FAISS GPU K-Means failed ({exc}); falling back to sklearn CPU K-Means.")

    kmeans = KMeans(n_clusters=n_clusters, n_init=KMEANS_N_INIT, random_state=KMEANS_RANDOM_STATE)
    kmeans.fit(x)
    return kmeans.cluster_centers_.astype(np.float32), kmeans.labels_.astype(np.int64)


# --- CODEBOOK INITIALIZATION ---
def init_codebooks_batched(model, dataset_embeddings, codebook_sizes, device, logger, sample_size=INIT_SAMPLE_SIZE):
    """
    Initialize codebooks using (GPU/FAISS) K-Means on a sampled subset of training embeddings.
    Centroids preserve true magnitude (un-normalized).
    """
    total_len = len(dataset_embeddings)
    effective_sample_size = min(sample_size, total_len)
    sample_indices = np.random.choice(total_len, size=effective_sample_size, replace=False)
    sampled_data = dataset_embeddings[sample_indices]

    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw_model.eval()
    encoded_list = []
    batch_sz = INIT_BATCH_SIZE

    with torch.no_grad():
        for i in range(0, len(sampled_data), batch_sz):
            batch_data = sampled_data[i:i + batch_sz].to(device, non_blocking=True)
            z_e = raw_model.encoder(batch_data)
            
            # Conditionally apply LayerNorm if it exists (i.e., not running baseline)
            if raw_model.bottleneck_norm is not None:
                z_rep = raw_model.bottleneck_norm(z_e)
            else:
                z_rep = z_e
                
            encoded_list.append(z_rep.cpu().float().numpy())

    current_residuals = np.concatenate(encoded_list, axis=0)
    codebooks = []

    for cb_idx, cb_size in enumerate(codebook_sizes):
        t0 = time.time()
        centroids, labels = fit_kmeans(current_residuals, cb_size, logger)
        codebooks.append(torch.from_numpy(centroids))

        # Residual update for next stage
        current_residuals = current_residuals - centroids[labels]

        logger.info(f"    ✔ Codebook #{cb_idx} (size {cb_size}) K-Means fitted in {time.time() - t0:.2f}s.")

    return codebooks


# --- DEAD CODE BUFFER & REVIVAL ---
class GPURadialResidualFixer:
    """
    GPU circular buffer with per-step EMA usage tracking and optimizer-state-resetting revival.
    """
    def __init__(self, num_codebooks: int, hidden_dim: int, buffer_size: int = BUFFER_CAPACITY_PER_STAGE, device='cuda'):
        self.num_codebooks = num_codebooks
        self.buffer_size = buffer_size
        self.device = device
        self.buffers = [torch.zeros(buffer_size, hidden_dim, dtype=torch.float32, device=device) for _ in range(num_codebooks)]
        self.ptrs = [0] * num_codebooks
        self.has_wrapped = [False] * num_codebooks
        self.usage = [None] * num_codebooks

    def update_buffers(self, residuals_per_stage: list):
        with torch.no_grad():
            for d, res in enumerate(residuals_per_stage[:self.num_codebooks]):
                n = min(res.shape[0], self.buffer_size)
                ptr = self.ptrs[d]
                res_slice = res[:n].to(dtype=self.buffers[d].dtype, device=self.buffers[d].device)
                if ptr + n <= self.buffer_size:
                    self.buffers[d][ptr:ptr + n] = res_slice
                    self.ptrs[d] = (ptr + n) % self.buffer_size
                    if ptr + n == self.buffer_size:
                        self.has_wrapped[d] = True
                else:
                    first_part = self.buffer_size - ptr
                    self.buffers[d][ptr:] = res_slice[:first_part]
                    self.buffers[d][:n - first_part] = res_slice[first_part:]
                    self.ptrs[d] = n - first_part
                    self.has_wrapped[d] = True

    def update_usage(self, clusters_counts: list, decay: float = DEFAULT_USAGE_DECAY):
        with torch.no_grad():
            for d, counts in enumerate(clusters_counts):
                if self.usage[d] is None:
                    self.usage[d] = counts.float().clone()
                else:
                    self.usage[d].mul_(decay).add_(counts, alpha=1.0 - decay)

    def fix_dead_codes_ema(self, model, optimizer=None, threshold: float = DEFAULT_DEAD_CODE_THRESHOLD):
        total_revived = 0
        with torch.no_grad():
            for d, (cb_param, cb_usage) in enumerate(zip(model.codebooks, self.usage)):
                if cb_usage is None:
                    continue

                valid_capacity = self.buffer_size if self.has_wrapped[d] else self.ptrs[d]
                if valid_capacity == 0:
                    continue

                dead_indices = (cb_usage < threshold).nonzero(as_tuple=True)[0]
                num_dead = len(dead_indices)

                if num_dead > 0:
                    rand_idx = torch.randint(0, valid_capacity, (num_dead,), device=self.device)
                    # Assign buffered residual matching parameter dtype and device
                    cb_param.data[dead_indices] = self.buffers[d][rand_idx].to(dtype=cb_param.dtype, device=cb_param.device)

                    expected_mean_usage = self.usage[d].sum().item() / cb_param.shape[0]
                    self.usage[d][dead_indices] = max(1.0, expected_mean_usage)

                    # Reset Adam optimizer momentum and variance states
                    if optimizer is not None and cb_param in optimizer.state:
                        opt_state = optimizer.state[cb_param]
                        if 'exp_avg' in opt_state:
                            opt_state['exp_avg'][dead_indices] = 0.0
                        if 'exp_avg_sq' in opt_state:
                            opt_state['exp_avg_sq'][dead_indices] = 0.0

                    total_revived += num_dead

        return total_revived


# --- VALIDATION EVALUATION ---
@torch.inference_mode()
def evaluate_rqvae(model, dataloader, device, amp_dtype, use_amp):
    """Evaluates validation loss, codebook/commitment loss, SID prefix uniqueness, and stability."""
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_cb = 0.0
    total_cm = 0.0
    total_samples = 0
    all_codes = []

    first_batch = None

    for (batch_embs,) in dataloader:
        batch_embs = batch_embs.to(device, non_blocking=True)
        bs = batch_embs.size(0)

        if first_batch is None:
            first_batch = batch_embs

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = model(batch_embs, compute_diagnostics=False)

        total_loss += output['loss'].item() * bs
        total_recon += output['recon_loss'].item() * bs
        total_cb += output['codebook_loss'].item() * bs
        total_cm += output['commitment_loss'].item() * bs
        total_samples += bs

        # Collect SIDs
        all_codes.append(output['clusters'])

    all_codes = torch.cat(all_codes, dim=0)

    # 1. Prefix uniqueness across the entire validation set
    sid_uniqueness = RQVAEDiagnostics.compute_sid_uniqueness(all_codes)

    # 2. Assignment stability on a sample batch
    stability = RQVAEDiagnostics.compute_assignment_stability(model, first_batch)

    return {
        'total_loss': total_loss / total_samples,
        'recon_loss': total_recon / total_samples,
        'codebook_loss': total_cb / total_samples,
        'commitment_loss': total_cm / total_samples,
        'sid_uniqueness': sid_uniqueness,
        'stability': stability,
    }


# --- CHECKPOINT MANAGEMENT ---
def save_checkpoint(model, optimizer, scaler, scheduler, epoch, step, filepath, best_val_loss=None):
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'best_val_loss': best_val_loss,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
    }
    torch.save(checkpoint, filepath)


# --- MAIN PIPELINE ---
def main():
    pipeline_start_time = time.time()
    parser = argparse.ArgumentParser(description="Production-grade RQ-VAE Semantic ID Quantization CLI Pipeline")
    parser.add_argument('--input_path', type=str, default=DEFAULT_INPUT_PATH, help='Path to input file (HDF5/Parquet)')
    parser.add_argument('--output_path', type=str, default=DEFAULT_OUTPUT_PATH, help='Path to output JSON mapping')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path to save final PyTorch model')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file (.pth) to resume from')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device target')
    parser.add_argument('--dtype', type=str, default='bf16', choices=['fp32', 'fp16', 'bf16'], help='Precision dtype')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of epochs')
    parser.add_argument('--warmup_ae_epochs', type=int, default=0, help='Epochs to train autoencoder before K-Means init')
    parser.add_argument('--max_steps', type=int, default=None, help='Target maximum training steps')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Training batch size')
    parser.add_argument('--inference_batch_size', type=int, default=DEFAULT_INFERENCE_BATCH_SIZE, help='Inference batch size')
    parser.add_argument('--val_ratio', type=float, default=DEFAULT_VAL_RATIO, help='Ratio of dataset reserved for validation (0.0 to disable)')
    parser.add_argument('--val_interval', type=int, default=DEFAULT_VAL_INTERVAL, help='Validation frequency in epochs')
    parser.add_argument('--save_best', action=argparse.BooleanOptionalAction, default=True, help='Track and save checkpoint with lowest validation loss')
    parser.add_argument('--run_baseline', action=argparse.BooleanOptionalAction, default=DEFAULT_RUN_BASELINE, help='Run the baseline model without jitter, layernorm, and entropy loss')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--max_samples', type=lambda x: None if x.lower() == 'none' else int(x), default=DEFAULT_MAX_SAMPLES, help='Limit samples')
    parser.add_argument('--input_dim', type=int, default=DEFAULT_INPUT_DIM, help='Input embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=DEFAULT_HIDDEN_DIM, choices=[32, 64], help='Hidden codebook dimension')
    parser.add_argument('--codebook_size', type=int, default=DEFAULT_CODEBOOK_SIZE, help='Size of each codebook')
    parser.add_argument('--num_codebooks', type=int, default=DEFAULT_NUM_CODEBOOKS, help='Number of residual codebooks')
    parser.add_argument('--beta', type=float, default=DEFAULT_BETA, help='Commitment loss beta coefficient')
    parser.add_argument('--temperature', type=float, default=1.0, help='Softmax temperature for Euclidean entropy loss')
    parser.add_argument('--jitter', type=float, default=DEFAULT_JITTER, help='Scale of Gaussian jitter added during training')
    parser.add_argument('--entropy_loss_weight', type=float, default=DEFAULT_ENTROPY_LOSS_WEIGHT, help='Weight of entropy regularization in total loss')
    parser.add_argument('--stage_weights', type=float, nargs='+', default=DEFAULT_STAGE_WEIGHTS, help='Entropy-loss weights for each residual stage')
    parser.add_argument('--usage_decay', type=float, default=DEFAULT_USAGE_DECAY, help='EMA decay for tracking codebook usage')
    parser.add_argument('--dead_code_threshold', type=float, default=DEFAULT_DEAD_CODE_THRESHOLD, help='Threshold for reviving dead codes')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'adagrad'], help='Optimizer')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR, help='Learning rate')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'none'], help='Scheduler type')
    parser.add_argument('--warmup_steps', type=int, default=DEFAULT_WARMUP_STEPS, help='Linear warmup steps')
    parser.add_argument('--compile', action='store_true', help='Enable torch.compile for graph fusion')
    parser.add_argument('--log_dir', type=str, default=None, help='TensorBoard & Log output directory')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory to save checkpoints')
    parser.add_argument('--add_unique_id', action='store_true', help='Append unique collision-solver suffix ID')
    parser.add_argument('--unique_id_bound', type=int, default=None, help='Upper bound for unique collision solver IDs')
    parser.add_argument('--clip-suffix', action=argparse.BooleanOptionalAction, default=True, help='Clip suffix IDs to unique_id_bound')
    parser.add_argument('--log_interval', type=int, default=DEFAULT_LOG_INTERVAL, help='Frequency to log metrics (steps)')
    parser.add_argument('--dead_code_fix_interval', type=int, default=DEFAULT_DEAD_CODE_FIX_INTERVAL, help='Frequency to revive dead codes (steps)')
    parser.add_argument('--save_interval', type=int, default=DEFAULT_SAVE_INTERVAL, help='Epoch frequency to save periodic checkpoints')

    args = parser.parse_args()

    # Reproducibility Seeds
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    DEVICE = torch.device(args.device)

    dtype_map = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
    AMP_DTYPE = dtype_map[args.dtype]
    USE_AMP = (DEVICE.type == 'cuda') and (args.dtype != 'fp32')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = args.log_dir if args.log_dir else os.path.join('runs', f'rqvae_{timestamp}')
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else os.path.join("checkpoints", f"rqvae_{timestamp}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize Logger & TensorBoard Writer
    logger = setup_logger(log_dir=log_dir)
    writer = SummaryWriter(log_dir=log_dir)


    try:
        log_banner(logger, "RQ-VAE SEMANTIC ID QUANTIZATION PIPELINE")

        param_summary = {
            "Device": str(DEVICE),
            "Precision / AMP": f"{args.dtype} (AMP Enabled: {USE_AMP})",
            "Input Dim → Hidden Dim": f"{args.input_dim} → {args.hidden_dim}",
            "Codebooks Architecture": f"{args.num_codebooks} x {args.codebook_size} entries",
            "Commitment Beta": args.beta,
            "Baseline Run (Unregularized)": "ENABLED" if args.run_baseline else "Disabled",
            "Entropy Weight": 0.0 if args.run_baseline else args.entropy_loss_weight,
            "Stage Weights": "Uniform 1.0" if args.run_baseline else ", ".join(map(str, args.stage_weights)),
            "Jitter Scale": 0.0 if args.run_baseline else args.jitter,
            "Validation Ratio": f"{args.val_ratio * 100:.1f}% (Eval every {args.val_interval} epoch(s))" if args.val_ratio > 0 else "Disabled",
            "Unique ID Collision Suffix": f"Bound: {args.unique_id_bound} | Clip: {args.clip_suffix}" if args.add_unique_id else "Disabled",
            "Optimizer & LR": f"{args.optimizer.upper()} (lr={args.lr}, warmup={args.warmup_steps})",
            "Epochs / Max Steps": f"{args.epochs} epochs / {args.max_steps if args.max_steps else 'N/A'} max steps",
            "AE Warmup Epochs": args.warmup_ae_epochs,
            "Resume Checkpoint": args.resume if args.resume else "None (Fresh Start)",
            "Log Intervals": f"Log every {args.log_interval} steps",
            "Dead Code Intervals": f"Fix every {args.dead_code_fix_interval} steps (threshold={args.dead_code_threshold})",
            "Batch Size (Train/Infer)": f"{args.batch_size} / {args.inference_batch_size}",
            "Compiled Model": args.compile,
            "Dataset Source": args.input_path,
            "Output Index Path": args.output_path,
            "Log Directory": log_dir,
        }
        log_card(logger, "Runtime Configuration", param_summary)

        # 1. Initialize Model
        model = RQVAE(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            beta=args.beta,
            codebook_sizes=[args.codebook_size] * args.num_codebooks,
            temperature=args.temperature,
            jitter=args.jitter,
            entropy_loss_weight=args.entropy_loss_weight,
            stage_weights=args.stage_weights,
            is_run_baseline=args.run_baseline
        ).to(DEVICE)

        # 2. Exclude Codebooks from Weight Decay
        codebook_params = list(model.codebooks.parameters())
        other_params = [p for n, p in model.named_parameters() if not n.startswith('codebooks')]

        if args.optimizer == 'adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
        elif args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW([
                {'params': other_params, 'weight_decay': WEIGHT_DECAY},
                {'params': codebook_params, 'weight_decay': 0.0}
            ], lr=args.lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # 3. Load & Partition Dataset
        t_data_start = time.time()
        dataset = EmbeddingDataset(args.input_path, max_samples=args.max_samples, logger=logger)
        total_samples = len(dataset)

        if args.val_ratio > 0.0:
            val_size = int(total_samples * args.val_ratio)
            train_size = total_samples - val_size

            # Deterministic permutation for split reproducibility
            rng = np.random.default_rng(RANDOM_SEED)
            permuted_indices = rng.permutation(total_samples)
            train_idx = permuted_indices[:train_size]
            val_idx = permuted_indices[train_size:]

            train_dataset = TensorDataset(dataset.embeddings[train_idx])
            val_dataset = TensorDataset(dataset.embeddings[val_idx])
            logger.info(f"  ✔ Dataset split: {train_size:,} train samples ({100*(1 - args.val_ratio):.1f}%), {val_size:,} validation samples ({100*args.val_ratio:.1f}%) [Partitioned in {time.time() - t_data_start:.2f}s]")
        else:
            train_dataset = TensorDataset(dataset.embeddings)
            val_dataset = None
            train_size = total_samples
            val_size = 0
            logger.info(f"  ✔ Dataset ready with {train_size:,} train samples (Validation disabled).")

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers if DEVICE.type == 'cuda' else 0,
            pin_memory=(DEVICE.type == 'cuda'),
            persistent_workers=(args.num_workers > 0 and DEVICE.type == 'cuda'),
            drop_last=True
        )

        val_dataloader = None
        if val_dataset is not None:
            val_dataloader = DataLoader(
                dataset=val_dataset,
                batch_size=args.inference_batch_size,
                shuffle=False,
                num_workers=args.num_workers if DEVICE.type == 'cuda' else 0,
                pin_memory=(DEVICE.type == 'cuda'),
                persistent_workers=(args.num_workers > 0 and DEVICE.type == 'cuda'),
                drop_last=False
            )

        # 4. Optional Autoencoder Warmup
        if args.warmup_ae_epochs > 0 and not args.resume:
            log_section(logger, "STAGE 0: AUTOENCODER RECONSTRUCTION WARMUP")
            t_ae_start = time.time()
            ae_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
            if model.bottleneck_norm is not None:
                ae_params += list(model.bottleneck_norm.parameters())

            ae_optimizer = torch.optim.Adam(ae_params, lr=args.lr)
            for ae_ep in range(args.warmup_ae_epochs):
                total_ae_loss = 0.0
                model.train()
                for (batch_embs,) in train_dataloader:
                    batch_embs = batch_embs.to(DEVICE, non_blocking=True)
                    ae_optimizer.zero_grad(set_to_none=True)
                    z = model.encoder(batch_embs)
                    if model.bottleneck_norm is not None:
                        z = model.bottleneck_norm(z)
                    rec = model.decoder(z)
                    loss_ae = F.mse_loss(rec, batch_embs)
                    loss_ae.backward()
                    ae_optimizer.step()
                    total_ae_loss += loss_ae.item()
                avg_ae_loss = total_ae_loss / len(train_dataloader)
                logger.info(f"  ► AE Warmup Epoch {ae_ep + 1}/{args.warmup_ae_epochs} - Reconstruction Loss: {avg_ae_loss:.6f}")
            logger.info(f"  ✔ Autoencoder latent space pre-warmed successfully in {time.time() - t_ae_start:.2f}s.")

        # 5. Learning Rate Scheduler & Scaler
        steps_per_epoch = len(train_dataloader)
        total_training_steps = args.max_steps if args.max_steps is not None else (args.epochs * steps_per_epoch)

        if args.scheduler == 'cosine':
            def lr_lambda(current_step: int):
                if current_step < args.warmup_steps:
                    return float(current_step) / float(max(1, args.warmup_steps))
                total_decay_steps = max(1, total_training_steps - args.warmup_steps)
                progress = float(current_step - args.warmup_steps) / float(total_decay_steps)
                progress = min(1.0, max(0.0, progress))
                return max(MIN_LR_RATIO, 0.5 * (1.0 + math.cos(math.pi * progress)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = None

        scaler_enabled = (DEVICE.type == 'cuda' and args.dtype == 'fp16')
        try:
            scaler = torch.amp.GradScaler(device=DEVICE.type, enabled=scaler_enabled)
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

        gpu_fixer = GPURadialResidualFixer(
            num_codebooks=args.num_codebooks,
            hidden_dim=args.hidden_dim,
            buffer_size=BUFFER_CAPACITY_PER_STAGE,
            device=str(DEVICE)
        )

        start_epoch = 0
        current_step = 0
        best_val_loss = float('inf')
        best_ckpt_path = os.path.join(checkpoint_dir, "best_rqvae_model.pth")

        # 6. Resume or Initialize Codebooks via K-Means
        if args.resume and os.path.isfile(args.resume):
            log_section(logger, "STAGE 1: RESUMING FROM CHECKPOINT")
            logger.info(f"  ► Loading checkpoint state from: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=DEVICE)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if scaler is not None and checkpoint.get('scaler_state_dict') is not None:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])

                start_epoch = checkpoint.get('epoch', 0) + 1
                current_step = checkpoint.get('step', 0)
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))

                if scheduler is not None:
                    if checkpoint.get('scheduler_state_dict') is not None:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    else:
                        scheduler.last_epoch = current_step
            else:
                model.load_state_dict(checkpoint)

            logger.info(f"  ✔ Successfully resumed from Epoch {start_epoch}, Step {current_step} (Best Val Loss: {best_val_loss:.6f}).")
        else:
            log_section(logger, "STAGE 1: CODEBOOK INITIALIZATION (TRAIN SPLIT ONLY)")
            t_km_start = time.time()
            sample_init_size = min(INIT_SAMPLE_SIZE, len(train_dataset))
            logger.info(f"  ► Sampling max {sample_init_size:,} training embeddings for K-Means initialization...")

            centroids_list = init_codebooks_batched(
                model=model,
                dataset_embeddings=train_dataset.tensors[0],
                codebook_sizes=[args.codebook_size] * args.num_codebooks,
                device=DEVICE,
                logger=logger,
                sample_size=sample_init_size
            )
            with torch.no_grad():
                for i, centroids in enumerate(centroids_list):
                    model.codebooks[i].copy_(centroids.to(DEVICE))
            logger.info(f"  ✔ Initialized {args.num_codebooks} codebooks successfully via batched K-Means in {time.time() - t_km_start:.2f}s.")

        # 7. Optional Graph Compilation
        train_model = model
        if args.compile and hasattr(torch, 'compile') and DEVICE.type == 'cuda':
            logger.info("  ► Compiling model with torch.compile...")
            train_model = torch.compile(model)

        # 8. Model Training Loop
        log_section(logger, "STAGE 2: MODEL TRAINING LOOP")

        log_table_header(logger)

        reached_max_steps = False

        for epoch in range(start_epoch, args.epochs):
            if reached_max_steps:
                break

            epoch_start_time = time.time()

            for (batch_embs,) in train_dataloader:
                train_model.train()
                batch_embs = batch_embs.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                    model_output = train_model(batch_embs, compute_diagnostics=True)
                    loss = model_output['loss']

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                nn.utils.clip_grad_norm_(raw_model.encoder.parameters(), GRAD_CLIP_MAX_NORM)
                if raw_model.bottleneck_norm is not None:
                    nn.utils.clip_grad_norm_(raw_model.bottleneck_norm.parameters(), GRAD_CLIP_MAX_NORM)
                nn.utils.clip_grad_norm_(raw_model.codebooks.parameters(), GRAD_CLIP_MAX_NORM)
                nn.utils.clip_grad_norm_(raw_model.decoder.parameters(), GRAD_CLIP_MAX_NORM)

                scaler.step(optimizer)
                scaler.update()

                if scheduler is not None:
                    scheduler.step()

                current_step += 1

                if gpu_fixer is not None:
                    gpu_fixer.update_buffers(model_output['residuals_per_stage'])
                    counts = [
                        torch.bincount(model_output['clusters'][:, d], minlength=args.codebook_size)
                        for d in range(args.num_codebooks)
                    ]
                    gpu_fixer.update_usage(counts, decay=args.usage_decay)

                    if current_step % args.dead_code_fix_interval == 0 and current_step < 0.6 * total_training_steps:
                        total_revived = gpu_fixer.fix_dead_codes_ema(raw_model, optimizer=optimizer, threshold=args.dead_code_threshold)
                        if total_revived > 0:
                            writer.add_scalar("Training/Dead_Codes_Revived", total_revived, current_step)
                            logger.info(f"  [Step {current_step}] Revived {total_revived} dead centroid(s).")

                # --- LOGGING STEP ---
                # current_step was already incremented above, so it now equals the number of
                # completed steps. The old check ran *before* the increment plus a `current_step == 1`
                # special case, which printed two back-to-back rows (steps 0 and 1) per interval and
                # never logged on clean multiples of log_interval.
                should_log = (
                    (args.log_interval > 0 and current_step % args.log_interval == 0)
                    or (args.max_steps is not None and current_step == args.max_steps)
                )
                if should_log:
                    # Scalar loss transfers (grouped together)
                    step_loss = loss.item()
                    recon_loss = model_output['recon_loss'].item()
                    cb_loss = model_output['codebook_loss'].item()
                    cm_loss = model_output['commitment_loss'].item()
                    rq_loss = model_output['rqvae_loss'].item()
                    entropy_loss = model_output['entropy_loss'].item()
                    curr_lr = optimizer.param_groups[0]['lr']

                    # Tensorboard Scalar Losses
                    writer.add_scalar("Loss/Train/Total", step_loss, current_step)
                    writer.add_scalar("Loss/Train/Reconstruction", recon_loss, current_step)
                    writer.add_scalar("Loss/Train/Codebook", cb_loss, current_step)
                    writer.add_scalar("Loss/Train/Commitment", cm_loss, current_step)
                    writer.add_scalar("Loss/Train/RQVAE", rq_loss, current_step)
                    writer.add_scalar("Loss/Train/Entropy", entropy_loss, current_step)
                    writer.add_scalar("Training/Learning_Rate", curr_lr, current_step)

                    # Hard Code Usage & Entropy Diagnostics
                    stage_diags = RQVAEDiagnostics.compute_stage_diagnostics(model_output['clusters'], raw_model.codebook_sizes)
                    for diag in stage_diags:
                        d = diag['stage']
                        writer.add_scalar(f"Stage_{d}/Used_Codes", diag['used_codes'], current_step)
                        writer.add_scalar(f"Stage_{d}/Dead_Code_Pct", diag['dead_code_pct'], current_step)
                        writer.add_scalar(f"Stage_{d}/Max_Code_Freq", diag['max_freq'], current_step)
                        writer.add_scalar(f"Stage_{d}/Entropy", diag['entropy'], current_step)
                        writer.add_scalar(f"Stage_{d}/Perplexity", diag['perplexity'], current_step)

                    # Residual Norms (E[||r_m||^2], mean & median)
                    res_diags = RQVAEDiagnostics.compute_residual_norms(model_output['residuals_per_stage'])
                    for r_stat in res_diags:
                        m = r_stat['stage']
                        writer.add_scalar(f"Residual/R_{m}_Expected_Sq_Norm", r_stat['expected_sq_norm'], current_step)
                        writer.add_scalar(f"Residual/R_{m}_Mean_L2", r_stat['mean_l2'], current_step)
                        writer.add_scalar(f"Residual/R_{m}_Median_L2", r_stat['median_l2'], current_step)

                    # Lean Stdio Table Row (widths come from LOG_TABLE_COLUMNS, matching the header)
                    log_table_row(
                        logger,
                        epoch=epoch,
                        step=current_step,
                        loss=step_loss,
                        recon_loss=recon_loss,
                        cb_loss=cb_loss,
                        cm_loss=cm_loss,
                        perp_first=stage_diags[0]['perplexity'],
                        perp_last=stage_diags[-1]['perplexity'],
                        res_first=res_diags[0]['expected_sq_norm'],
                        res_last=res_diags[-1]['expected_sq_norm'],
                        lr=curr_lr,
                    )

                if args.max_steps is not None and current_step >= args.max_steps:
                    log_table_footer(logger)
                    logger.info(f"  ► Target max_steps reached ({args.max_steps}). Terminating training loop.")
                    reached_max_steps = True
                    break

            epoch_time = time.time() - epoch_start_time

            if val_dataloader is not None and ((epoch + 1) % args.val_interval == 0 or (epoch + 1) == args.epochs):
                t_val_start = time.time()
                val_metrics = evaluate_rqvae(
                    model=train_model,
                    dataloader=val_dataloader,
                    device=DEVICE,
                    amp_dtype=AMP_DTYPE,
                    use_amp=USE_AMP
                )
                val_elapsed = time.time() - t_val_start

                # Log Validation Metrics to TensorBoard
                writer.add_scalar("Loss/Val/Total", val_metrics['total_loss'], current_step)
                writer.add_scalar("Loss/Val/Reconstruction", val_metrics['recon_loss'], current_step)
                writer.add_scalar("Loss/Val/Codebook", val_metrics['codebook_loss'], current_step)
                writer.add_scalar("Loss/Val/Commitment", val_metrics['commitment_loss'], current_step)
                writer.add_scalar("Diagnostics/Val_Assignment_Stability", val_metrics['stability'], current_step)

                for d, uniq_ratio in enumerate(val_metrics['sid_uniqueness']):
                    writer.add_scalar(f"SID_Uniqueness/Val_Depth_{d+1}", uniq_ratio, current_step)

                # Format SID uniqueness progression string (e.g., D1: 22% -> D2: 74% -> D4: 99%)
                uniq_str = " -> ".join([f"D{i+1}:{r*100:.1f}%" for i, r in enumerate(val_metrics['sid_uniqueness'])])

                logger.info(
                    f"  ▶ [Validation @ Epoch {epoch + 1} | Step {current_step}] "
                    f"Total: {val_metrics['total_loss']:.4f} | Recon: {val_metrics['recon_loss']:.4f} | "
                    f"CB: {val_metrics['codebook_loss']:.4f} | CM: {val_metrics['commitment_loss']:.4f} | "
                    f"Stability: {val_metrics['stability']*100:.1f}%\n"
                    f"    └─ Prefix Uniqueness: [{uniq_str}] (Eval Time: {val_elapsed:.2f}s)"
                )

                # Save Best Checkpoint
                if args.save_best and val_metrics['total_loss'] < best_val_loss:
                    best_val_loss = val_metrics['total_loss']
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        scheduler=scheduler,
                        epoch=epoch,
                        step=current_step,
                        filepath=best_ckpt_path,
                        best_val_loss=best_val_loss
                    )
                    logger.info(f"  ★ New best validation loss: {best_val_loss:.6f}! Saved checkpoint to: {best_ckpt_path}")

            # Periodic Regular Checkpoint
            if not reached_max_steps and (((epoch + 1) % args.save_interval == 0) or ((epoch + 1) == args.epochs)):
                ckpt_p = os.path.join(checkpoint_dir, f"rqvae_epoch_{epoch + 1}.pth")
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    scheduler=scheduler,
                    epoch=epoch,
                    step=current_step,
                    filepath=ckpt_p,
                    best_val_loss=best_val_loss
                )
                logger.info(f"Saved periodic checkpoint for epoch {epoch + 1} to {ckpt_p} (Epoch elapsed: {epoch_time:.2f}s)")

        if not reached_max_steps:
            log_table_footer(logger)

        # Save Final State Dict
        os.makedirs(os.path.dirname(os.path.abspath(args.model_path)) or '.', exist_ok=True)
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        torch.save(raw_model.state_dict(), args.model_path)
        logger.info(f"  ✔ Saved final model weights to: {args.model_path}")

        # 9. Corpus Inference (Encode Full Dataset)
        log_section(logger, "STAGE 3: INFERENCE & SEMANTIC ID GENERATION")
        if args.save_best and os.path.isfile(best_ckpt_path):
            logger.info(f"  ► Loading best validation model ({best_ckpt_path}) for Stage 3 inference...")
            best_checkpoint = torch.load(best_ckpt_path, map_location=DEVICE)
            raw_model.load_state_dict(best_checkpoint['model_state_dict'])

        logger.info(f"  ► Encoding full dataset through encoder (batch size = {args.inference_batch_size})...")

        t_infer_start = time.time()
        raw_model.eval()
        all_clusters = []

        with torch.inference_mode():
            for i in range(0, len(dataset.embeddings), args.inference_batch_size):
                batch_slice = dataset.embeddings[i:i + args.inference_batch_size].to(DEVICE, non_blocking=True)
                with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                    indices = raw_model.encode_indices(batch_slice)
                all_clusters.append(indices.cpu())

        all_clusters_tensor = torch.cat(all_clusters, dim=0)
        infer_duration = time.time() - t_infer_start
        throughput = len(dataset) / max(1e-5, infer_duration)

        logger.info(f"  ✔ Encoded {len(dataset):,} items in {infer_duration:.2f}s (Throughput: {throughput:,.1f} items/sec).")

        # Full corpus prefix uniqueness diagnostics
        corpus_sid_uniqueness = RQVAEDiagnostics.compute_sid_uniqueness(all_clusters_tensor)
        for d, r in enumerate(corpus_sid_uniqueness):
            writer.add_scalar(f"Corpus_SID_Uniqueness/Depth_{d+1}", r, current_step)

        raw_semantic_mapping = {}
        sem_2_ids = defaultdict(list)
        all_clusters_np = all_clusters_tensor.numpy()
        for item_id, codes in zip(dataset.item_ids, all_clusters_np):
            codes_list = codes.tolist()
            raw_semantic_mapping[item_id] = codes_list
            if args.add_unique_id:
                sem_2_ids[tuple(codes_list)].append(item_id)

        # 10. Codebook Utilization & Perplexity Report
        log_section(logger, "STAGE 4: CODEBOOK UTILIZATION & PERPLEXITY REPORT")
        cb_metrics = {}
        per_cb_utilization = []
        per_cb_perplexity = []

        for cb_idx in range(args.num_codebooks):
            cb_assignments = all_clusters_np[:, cb_idx]

            num_unique = len(np.unique(cb_assignments))
            utilization_pct = (num_unique / args.codebook_size) * 100.0
            per_cb_utilization.append(utilization_pct)

            counts = np.bincount(cb_assignments, minlength=args.codebook_size)
            probs = counts / np.sum(counts)
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs))
            perplexity = float(np.exp(entropy))
            per_cb_perplexity.append(perplexity)

            cb_metrics[f"Codebook #{cb_idx}"] = (
                f"Active: {num_unique}/{args.codebook_size} ({utilization_pct:.1f}%), "
                f"Perplexity: {perplexity:.1f}"
            )
            writer.add_scalar(f"Corpus_Utilization/Codebook_{cb_idx}", utilization_pct, current_step)
            writer.add_scalar(f"Corpus_Perplexity/Codebook_{cb_idx}", perplexity, current_step)

        mean_utilization = float(np.mean(per_cb_utilization))
        mean_perplexity = float(np.mean(per_cb_perplexity))

        cb_metrics["Average Utilization"] = f"{mean_utilization:.2f}%"
        cb_metrics["Average Perplexity"] = f"{mean_perplexity:.2f}"
        cb_metrics["Final Corpus Full SID Uniqueness"] = f"{corpus_sid_uniqueness[-1]*100:.2f}%"

        writer.add_scalar("Corpus_Utilization/Average", mean_utilization, current_step)
        writer.add_scalar("Corpus_Perplexity/Average", mean_perplexity, current_step)
        writer.add_scalar("Corpus_SID_Uniqueness/Full_Depth", corpus_sid_uniqueness[-1], current_step)

        log_card(logger, "Corpus Codebook Utilization & Perplexity", cb_metrics)

        # 11. Collision Resolution & Index Saving
        log_section(logger, "STAGE 5: COLLISION RESOLUTION & INDEX SAVING")

        semantic_mapping = {}
        if args.add_unique_id:
            unique_bound = args.unique_id_bound if args.unique_id_bound is not None else args.codebook_size
            logger.info(f"  ► Resolving collisions with deterministic ranking suffix (bound = {unique_bound})...")

            overflow_count = 0
            max_collision_len = 0
            total_collisions = 0

            for semantics, item_ids in sem_2_ids.items():
                num_items = len(item_ids)
                if num_items > 1:
                    total_collisions += 1
                    if num_items > max_collision_len:
                        max_collision_len = num_items

                if num_items > unique_bound:
                    overflow_count += 1

                for rank_idx, item_id in enumerate(item_ids):
                    suffix_id = (unique_bound - 1) if (args.clip_suffix and rank_idx >= unique_bound) else rank_idx
                    semantic_mapping[item_id] = list(semantics) + [int(suffix_id)]

            collision_summary = {
                "Total Unique Semantic Tuples": f"{len(sem_2_ids):,}",
                "Tuples with Collisions (>1 item)": f"{total_collisions:,}",
                "Max Items Sharing One Tuple": max_collision_len,
                "Groups Exceeding Bound": overflow_count,
            }
            log_card(logger, "Collision Resolution Summary", collision_summary)

            writer.add_scalar("Collisions/Total_Collisions", total_collisions, current_step)
            writer.add_scalar("Collisions/Max_Collision_Length", max_collision_len, current_step)
            writer.add_scalar("Collisions/Overflow_Groups", overflow_count, current_step)
        else:
            semantic_mapping = raw_semantic_mapping
            logger.info("  ► Collision resolution skipped (--add_unique_id not set).")

        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or '.', exist_ok=True)
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(semantic_mapping, f)

        logger.info(f"  ✔ Serialized semantic ID mapping for {len(semantic_mapping):,} items to: {args.output_path}")

        total_elapsed = time.time() - pipeline_start_time
        final_summary = {
            "Total Pipeline Runtime": f"{total_elapsed:.2f}s",
            "Best Validation Loss": f"{best_val_loss:.6f}" if best_val_loss != float('inf') else "N/A",
            "Full SID Uniqueness": f"{corpus_sid_uniqueness[-1]*100:.2f}%",
            "Total Items Indexed": f"{len(semantic_mapping):,}",
            "Index Output File": args.output_path,
            "TensorBoard Log Directory": log_dir,
            "Status": "COMPLETED SUCCESSFULLY"
        }
        log_card(logger, "Pipeline Execution Summary", final_summary)
        log_banner(logger, "PIPELINE FINISHED")

    except Exception as e:
        logger.exception(f"Fatal error encountered during pipeline execution: {e}")
        raise e
    finally:
        writer.flush()
        writer.close()


if __name__ == '__main__':
    main()
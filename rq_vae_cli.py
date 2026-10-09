import os
import json
import math
import argparse
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import KMeans
from collections import defaultdict

# Enable TensorFloat-32 for faster matrix multiplications on modern GPUs (Ampere/Hopper/Blackwell)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- GLOBAL CONSTANTS & HYPERPARAMETERS ---
RANDOM_SEED = 42
DEFAULT_H5_PATH = 'tencent_gr_parquet/train/mm_emb_82.h5'
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
DEFAULT_ENTROPY_WEIGHT = 0.05
DEFAULT_LR = 5e-4
DEFAULT_WARMUP_STEPS = 2000
DEFAULT_LOG_INTERVAL = 200
DEFAULT_DEAD_CODE_FIX_INTERVAL = 500
DEFAULT_SAVE_INTERVAL = 5

# Model initialization constants
TRUNC_NORMAL_STD = 0.02

# Dataset & Loader constants
INIT_SAMPLE_SIZE = 100000
INIT_BATCH_SIZE = 4096
BUFFER_CAPACITY_PER_STAGE = 20000

# K-Means constants
KMEANS_N_INIT = 3
KMEANS_RANDOM_STATE = 42

# Training & Optimization constants
GRAD_CLIP_MAX_NORM = 1.0
WEIGHT_DECAY = 1e-4
MIN_LR_RATIO = 0.01

# UI/Logging constants
BANNER_WIDTH = 72


class HDF5EmbeddingDataset(Dataset):
    def __init__(self, h5_path: str, max_samples: int = None):
        super().__init__()
        self.h5_path = h5_path
        self.max_samples = max_samples

        print(f"  ► Loading dataset into memory from {h5_path}...")
        with h5py.File(self.h5_path, 'r') as f:
            oid_ds = f['anonymous_cid']
            emb_ds = f['embedding']
            
            total_len = len(oid_ds)
            if self.max_samples is not None:
                total_len = min(total_len, self.max_samples)
            
            oids = oid_ds[:total_len]
            embs = emb_ds[:total_len]

        self.item_ids = [oid.decode('utf-8') if isinstance(oid, bytes) else str(oid) for oid in oids]
        self.embeddings = torch.from_numpy(embs).float()
        print(f"  ✔ Successfully loaded {len(self.item_ids):,} samples into memory.")

    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, idx):
        return self.embeddings[idx]


class RQVAE(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=64, beta=0.25, codebook_sizes=[256, 256, 256, 256], temperature=1.0):
        super().__init__()
        self.beta = beta
        self.codebook_sizes = codebook_sizes
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # Encoder: input_dim -> 512 -> 256 -> 128 -> hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )

        # Bottleneck Normalization: Stabilizes latent space representation
        self.bottleneck_norm = nn.LayerNorm(hidden_dim)

        # Decoder: hidden_dim -> 128 -> 256 -> 512 -> input_dim
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

        # Codebooks: Unconstrained in magnitude to allow multi-scale residual decay
        self.codebooks = nn.ParameterList()
        for codebook_size in codebook_sizes:
            cb = nn.Parameter(torch.FloatTensor(codebook_size, hidden_dim))
            with torch.no_grad():
                nn.init.trunc_normal_(cb, std=TRUNC_NORMAL_STD, a=-0.04, b=0.04)
            self.codebooks.append(cb)

    def forward(self, batch_embs):
        # 1. Continuous Latent Space
        latent_vector = self.encoder(batch_embs)
        z_norm = self.bottleneck_norm(latent_vector)

        # Add slight Gaussian jitter during training to spread dense clusters
        if self.training:
            jitter = torch.randn_like(z_norm) * 0.05
            remainder = z_norm + jitter
        else:
            remainder = z_norm

        quantized_sum = torch.zeros_like(z_norm)
        rqvae_loss = 0.0
        entropy_loss = 0.0
        clusters = []
        residuals_per_stage = []

        max_entropy = math.log(self.codebook_sizes[0])

        # 2. Sequential Residual Vector Quantization Cascade
        for codebook in self.codebooks:
            residuals_per_stage.append(remainder.detach())

            # Tensor Core GEMM formulation: S = 2 * (r . e^T) - ||e||^2
            # Mathematically equivalent to -||r - e||^2 up to constant ||r||^2
            e_sq = torch.sum(codebook ** 2, dim=-1, keepdim=True).t()    # (1, K)
            dot = torch.matmul(remainder, codebook.t())                    # (B, K)
            logits = 2.0 * dot - e_sq                                      # (B, K)

            # Discrete Voronoi assignment (argmax affinity == argmin Euclidean distance)
            codebook_indices = logits.argmax(dim=-1)
            clusters.append(codebook_indices)
            quantized = codebook[codebook_indices]                         # (B, D)

            # Commitment + Codebook Loss (unscaled MSE preserving true energy hierarchy)
            loss_d = F.mse_loss(quantized, remainder.detach()) + \
                     self.beta * F.mse_loss(remainder, quantized.detach())
            rqvae_loss = rqvae_loss + loss_d

            # Differentiable Euclidean Softmax Entropy Loss
            probs = F.softmax(logits / self.temperature, dim=-1)
            avg_probs = probs.mean(dim=0)                                  # (K,)
            stage_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))
            stage_entropy_loss = (max_entropy - stage_entropy) / max_entropy
            entropy_loss = entropy_loss + stage_entropy_loss

            # Update residual: subtracts full-scale learned quantized vectors
            quantized_sum = quantized_sum + quantized
            remainder = remainder - quantized.detach()

        entropy_loss = entropy_loss / len(self.codebooks)

        # 3. Straight-Through Estimator across entire cascade
        # Note: z_norm (clean) is used for the skip connection to protect base reconstruction
        latent_restored = z_norm + (quantized_sum - z_norm).detach()

        # 4. Decode & Reconstruct
        embeddings_restored = self.decoder(latent_restored)
        recon_loss = F.mse_loss(embeddings_restored, batch_embs)
        loss = recon_loss + rqvae_loss

        clusters_counts = [
            torch.bincount(cluster, minlength=cb_size)
            for cb_size, cluster in zip(self.codebook_sizes, clusters)
        ]

        return {
            'loss': loss,
            'entropy_loss': entropy_loss,
            'clusters_counts': clusters_counts,
            'clusters': torch.stack(clusters, dim=-1),
            'residuals_per_stage': residuals_per_stage,
            'metrics': dict(
                loss=loss.detach().item(),
                recon_loss=recon_loss.detach().item(),
                rqvae_loss=rqvae_loss.detach().item(),
                entropy_loss=entropy_loss.detach().item()
            ),
        }

    @torch.inference_mode()
    def encode_indices(self, batch_embs):
        """High-throughput inference matching the exact Euclidean Voronoi regions."""
        z = self.bottleneck_norm(self.encoder(batch_embs))
        remainder = z
        all_indices = []

        for codebook in self.codebooks:
            e_sq = torch.sum(codebook ** 2, dim=-1, keepdim=True).t()
            dot = torch.matmul(remainder, codebook.t())
            logits = 2.0 * dot - e_sq
            idx = logits.argmax(dim=-1)
            all_indices.append(idx)

            quantized = codebook[idx]
            remainder = remainder - quantized

        return torch.stack(all_indices, dim=-1)


def init_codebooks_batched(model, dataset_embeddings, codebook_sizes, device, sample_size=INIT_SAMPLE_SIZE):
    total_len = len(dataset_embeddings)
    sample_indices = np.random.choice(total_len, size=min(sample_size, total_len), replace=False)
    sampled_data = dataset_embeddings[sample_indices]

    model.eval()
    encoded_list = []
    batch_sz = INIT_BATCH_SIZE
    with torch.no_grad():
        for i in range(0, len(sampled_data), batch_sz):
            batch_data = sampled_data[i:i+batch_sz].to(device)
            z_e = model.encoder(batch_data)
            z_norm = model.bottleneck_norm(z_e)
            encoded_list.append(z_norm.cpu().numpy())
    
    current_residuals = np.concatenate(encoded_list, axis=0)
    
    codebooks = []
    for cb_size in codebook_sizes:
        kmeans = KMeans(n_clusters=cb_size, n_init=KMEANS_N_INIT, random_state=KMEANS_RANDOM_STATE)
        kmeans.fit(current_residuals)
        
        # PRESERVE TRUE SCALE: Do NOT normalize centroids
        centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        codebooks.append(centroids)
        
        # True residual subtraction for next stage's K-Means
        labels = kmeans.labels_
        quantized = kmeans.cluster_centers_[labels]
        current_residuals = current_residuals - quantized

    return codebooks


def compute_entropy_loss(clusters_counts, codebook_size, eps=1e-8):
    entropy_loss = 0.0
    for counts in clusters_counts:
        probs = counts.float() / (counts.sum() + eps)
        probs = probs[probs > 0]
        # Maximize entropy H(p) <=> minimize -H(p) = sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + eps))
        max_entropy = math.log(codebook_size)
        # Normalized deficit from uniform distribution
        entropy_loss += (max_entropy - entropy) / max_entropy
    return entropy_loss / len(clusters_counts)


class GPURadialResidualFixer:
    """Zero-overhead GPU circular buffer with per-step EMA usage tracking and optimizer-state-resetting revival."""
    def __init__(self, num_codebooks: int, hidden_dim: int, buffer_size: int = BUFFER_CAPACITY_PER_STAGE, device='cuda'):
        self.num_codebooks = num_codebooks
        self.buffer_size = buffer_size
        self.device = device
        self.buffers = [torch.zeros(buffer_size, hidden_dim, device=device) for _ in range(num_codebooks)]
        self.ptrs = [0] * num_codebooks
        self.has_wrapped = [False] * num_codebooks
        self.usage = [None] * num_codebooks

    def update_buffers(self, residuals_per_stage: list):
        with torch.no_grad():
            for d, res in enumerate(residuals_per_stage):
                n = min(res.shape[0], self.buffer_size)
                ptr = self.ptrs[d]
                if ptr + n <= self.buffer_size:
                    self.buffers[d][ptr:ptr+n] = res[:n]
                    self.ptrs[d] = (ptr + n) % self.buffer_size
                    if ptr + n == self.buffer_size:
                        self.has_wrapped[d] = True
                else:
                    first_part = self.buffer_size - ptr
                    self.buffers[d][ptr:] = res[:first_part]
                    self.buffers[d][:n-first_part] = res[first_part:n]
                    self.ptrs[d] = n - first_part
                    self.has_wrapped[d] = True

    def update_usage(self, clusters_counts: list, decay: float = 0.99):
        """Invoked on every training step to reliably track code usage over time."""
        with torch.no_grad():
            for d, counts in enumerate(clusters_counts):
                if self.usage[d] is None:
                    self.usage[d] = counts.float().clone()
                else:
                    self.usage[d].mul_(decay).add_(counts, alpha=1.0 - decay)

    def fix_dead_codes_ema(self, model, optimizer=None, threshold: float = 0.1):
        """Revives dead codes using buffered residuals, using a lower threshold and dynamic scale reset."""
        total_revived = 0
        with torch.no_grad():
            for d, (cb_param, cb_usage) in enumerate(zip(model.codebooks, self.usage)):
                if cb_usage is None:
                    continue

                valid_capacity = self.buffer_size if self.has_wrapped[d] else self.ptrs[d]
                if valid_capacity == 0:
                    continue

                # Lowered threshold to protect long-tail/rarely-used codes
                dead_indices = (cb_usage < threshold).nonzero(as_tuple=True)[0]
                num_dead = len(dead_indices)

                if num_dead > 0:
                    # Randomly sample from only the populated slice of the buffer
                    rand_idx = torch.randint(0, valid_capacity, (num_dead,), device=self.device)
                    cb_param.data[dead_indices] = self.buffers[d][rand_idx]
                    
                    # Dynamically compute expected batch mean usage to scale the warm reset correctly 
                    # (matches bincount EMA scale instead of using buffer capacity)
                    expected_mean_usage = self.usage[d].sum().item() / cb_param.shape[0]
                    self.usage[d][dead_indices] = max(1.0, expected_mean_usage)

                    # Reset Adam / AdamW momentum and second moments to prevent centroid kick-out
                    if optimizer is not None and cb_param in optimizer.state:
                        opt_state = optimizer.state[cb_param]
                        if 'exp_avg' in opt_state:
                            opt_state['exp_avg'][dead_indices] = 0.0
                        if 'exp_avg_sq' in opt_state:
                            opt_state['exp_avg_sq'][dead_indices] = 0.0

                    total_revived += num_dead

        return total_revived


def save_checkpoint(model, optimizer, scaler, scheduler, epoch, step, filepath):
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
    }
    torch.save(checkpoint, filepath)


# --- FORMATTING & LOGGING HELPERS ---

def print_banner(title: str):
    width = BANNER_WIDTH
    print("\n" + "=" * width)
    print(f"{title.center(width)}")
    print("=" * width)

def print_section(section_name: str):
    width = BANNER_WIDTH
    print(f"\n┌── [ {section_name} ] " + "─" * (width - len(section_name) - 7))

def print_card(title: str, items: dict, width: int = BANNER_WIDTH):
    print("┌" + "─" * (width - 2) + "┐")
    print(f"│ {title.upper():<{width - 4}} │")
    print("├" + "─" * (width - 2) + "┤")
    for key, val in items.items():
        k_str = f"  • {key}:"
        v_str = str(val)
        padding = width - len(k_str) - len(v_str) - 5
        print(f"│{k_str} {' ' * max(0, padding)}{v_str}  │")
    print("└" + "─" * (width - 2) + "┘")

def print_table_header():
    print(f"┌──────────┬──────────┬─────────────┬─────────────┬─────────────┬───────────────┐")
    print(f"│ Epoch    │ Step     │ Total Loss  │ Recon Loss  │ RQ-VAE Loss │ Learning Rate │")
    print(f"├──────────┼──────────┼─────────────┼─────────────┼─────────────┼───────────────┤")

def print_table_row(epoch: int, step: int, loss: float, recon_loss: float, rq_loss: float, lr: float):
    print(f"│ {epoch:<8d} │ {step:<8d} │ {loss:<11.4f} │ {recon_loss:<11.4f} │ {rq_loss:<11.4f} │ {lr:<13.6e} │")

def print_table_footer():
    print(f"└──────────┴──────────┴─────────────┴─────────────┴─────────────┴───────────────┘")


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Revised RQ-VAE Semantic ID Quantization CLI Pipeline")
    parser.add_argument('--h5_path', type=str, default=DEFAULT_H5_PATH, help='Path to input HDF5 file')
    parser.add_argument('--output_path', type=str, default=DEFAULT_OUTPUT_PATH, help='Path to output JSON mapping')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path to save the trained PyTorch model')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file (.pth) to resume training from')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cpu, cuda:0, etc.)')
    parser.add_argument('--dtype', type=str, default='bf16', choices=['fp32', 'fp16', 'bf16'], help='Precision dtype for GPU training')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of epochs')
    parser.add_argument('--warmup_ae_epochs', type=int, default=0, help='Optional epochs to train autoencoder before K-Means codebook init')
    parser.add_argument('--max_steps', type=int, default=None, help='Target maximum training steps')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Training batch size')
    parser.add_argument('--inference_batch_size', type=int, default=DEFAULT_INFERENCE_BATCH_SIZE, help='Inference batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--max_samples', type=lambda x: None if x.lower() == 'none' else int(x), default=DEFAULT_MAX_SAMPLES, help='Limit samples')
    parser.add_argument('--input_dim', type=int, default=DEFAULT_INPUT_DIM, help='Input embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=DEFAULT_HIDDEN_DIM, choices=[32, 64], help='Hidden codebook dimension')
    parser.add_argument('--codebook_size', type=int, default=DEFAULT_CODEBOOK_SIZE, help='Size of each codebook')
    parser.add_argument('--num_codebooks', type=int, default=DEFAULT_NUM_CODEBOOKS, help='Number of residual codebooks')
    parser.add_argument('--beta', type=float, default=DEFAULT_BETA, help='Commitment loss beta coefficient')
    parser.add_argument('--entropy_weight', type=float, default=DEFAULT_ENTROPY_WEIGHT, help='Weight for codebook entropy / diversity loss') # <-- Add this line
    parser.add_argument('--temperature', type=float, default=1.0, help='Softmax temperature for Euclidean entropy loss')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'adagrad'], help='Optimizer')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR, help='Learning rate')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'none'], help='Scheduler type')
    parser.add_argument('--warmup_steps', type=int, default=DEFAULT_WARMUP_STEPS, help='Linear warmup steps')
    parser.add_argument('--compile', action='store_true', help='Enable torch.compile for graph fusion')
    parser.add_argument('--log_dir', type=str, default=None, help='TensorBoard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory to save checkpoints')
    parser.add_argument('--add_unique_id', action='store_true', help='Append unique collision-solver suffix ID')
    parser.add_argument('--unique_id_bound', type=int, default=None, help='Upper bound for unique collision solver IDs')
    parser.add_argument('--clip-suffix',action=argparse.BooleanOptionalAction, default=True, help='Clip suffix IDs to unique_id_bound (default: True). ''Use --no-clip-suffix to allow suffix IDs to grow.')
    parser.add_argument('--log_interval', type=int, default=DEFAULT_LOG_INTERVAL, help='Frequency to log metrics')
    parser.add_argument('--dead_code_fix_interval', type=int, default=DEFAULT_DEAD_CODE_FIX_INTERVAL, help='Frequency to revive dead codes')
    parser.add_argument('--save_interval', type=int, default=DEFAULT_SAVE_INTERVAL, help='Epoch frequency to save checkpoints')

    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    DEVICE = torch.device(args.device)

    dtype_map = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
    AMP_DTYPE = dtype_map[args.dtype]
    USE_AMP = (DEVICE.type == 'cuda') and (args.dtype != 'fp32')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = args.log_dir if args.log_dir else os.path.join('runs', f'rqvae_{timestamp}')
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else os.path.join("checkpoints", f"rqvae_{timestamp}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    print_banner("RQ-VAE SEMANTIC ID QUANTIZATION PIPELINE")

    param_summary = {
        "Device": str(DEVICE),
        "Precision / AMP": f"{args.dtype} (AMP: {USE_AMP})",
        "Input Dim → Hidden Dim": f"{args.input_dim} → {args.hidden_dim}",
        "Codebooks Architecture": f"{args.num_codebooks} x {args.codebook_size} entries",
        "Commitment Beta": args.beta,
        "Entropy Weight": args.entropy_weight,
        "Unique ID Collision Suffix": f"Bound: {args.unique_id_bound} | Clip: {args.clip_suffix}" if args.add_unique_id else "Disabled",
        "Optimizer & LR": f"{args.optimizer.upper()} (lr={args.lr}, warmup={args.warmup_steps})",
        "Epochs / Max Steps": f"{args.epochs} epochs / {args.max_steps if args.max_steps else 'N/A'} max steps",
        "AE Warmup Epochs": args.warmup_ae_epochs,
        "Resume Checkpoint": args.resume if args.resume else "None (Fresh Start)",
        "Log Intervals": f"Log every {args.log_interval} steps",
        "Dead Code Intervals": f"Dead Code Fix every {args.dead_code_fix_interval} steps",
        "Batch Size (Train/Infer)": f"{args.batch_size} / {args.inference_batch_size}",
        "Compiled Model": args.compile,
        "Dataset Source": args.h5_path,
        "Output Index Path": args.output_path,
    }
    print_card("Runtime Parameters & System Config", param_summary)

    # 1. Initialize Core Model
    model = RQVAE(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        beta=args.beta,
        codebook_sizes=[args.codebook_size] * args.num_codebooks,
        temperature=args.temperature
    ).to(DEVICE)

    # 2. Exclude Codebooks from Weight Decay to Prevent Centroid Shrinkage
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

    # 3. Load Dataset into Memory
    dataset = HDF5EmbeddingDataset(args.h5_path, max_samples=args.max_samples)
    
    train_dataset = TensorDataset(dataset.embeddings)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers if DEVICE.type == 'cuda' else 0,
        pin_memory=(DEVICE.type == 'cuda'),
        persistent_workers=(args.num_workers > 0 and DEVICE.type == 'cuda'),
        drop_last=True
    )

    # 4. Optional Warmup Autoencoder without Quantization (Stabilizes K-Means Centroids)
    if args.warmup_ae_epochs > 0 and not args.resume:
        print_section("STAGE 0: AUTOENCODER RECONSTRUCTION WARMUP")
        ae_optimizer = torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.bottleneck_norm.parameters()) + list(model.decoder.parameters()),
            lr=args.lr
        )
        for ae_ep in range(args.warmup_ae_epochs):
            total_ae_loss = 0.0
            model.train()
            for (batch_embs,) in train_dataloader:
                batch_embs = batch_embs.to(DEVICE, non_blocking=True)
                ae_optimizer.zero_grad(set_to_none=True)
                z = model.bottleneck_norm(model.encoder(batch_embs))
                rec = model.decoder(z)
                loss_ae = F.mse_loss(rec, batch_embs)
                loss_ae.backward()
                ae_optimizer.step()
                total_ae_loss += loss_ae.item()
            avg_ae_loss = total_ae_loss / len(train_dataloader)
            print(f"  ► AE Warmup Epoch {ae_ep + 1}/{args.warmup_ae_epochs} - Reconstruction Loss: {avg_ae_loss:.6f}")
        print("  ✔ Autoencoder latent space pre-warmed successfully.")

    # 5. Learning Rate Scheduler
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

    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda' and args.dtype == 'fp16'))
    
    # Residual Buffer with per-step tracking and Adam-safe code revival
    gpu_fixer = GPURadialResidualFixer(
        num_codebooks=args.num_codebooks, 
        hidden_dim=args.hidden_dim, 
        buffer_size=BUFFER_CAPACITY_PER_STAGE, 
        device=str(DEVICE)
    )

    start_epoch = 0
    current_step = 0

    # 6. Resume or Initialize Codebooks
    if args.resume and os.path.isfile(args.resume):
        print_section("STAGE 1: RESUMING FROM CHECKPOINT")
        print(f"  ► Loading checkpoint state from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=DEVICE)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scaler is not None and checkpoint.get('scaler_state_dict') is not None:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])

            start_epoch = checkpoint.get('epoch', 0) + 1
            current_step = checkpoint.get('step', 0)

            if scheduler is not None:
                if checkpoint.get('scheduler_state_dict') is not None:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                else:
                    scheduler.last_epoch = current_step
        else:
            model.load_state_dict(checkpoint)

        print(f"  ✔ Successfully loaded model state. Resuming from Epoch {start_epoch}, Step {current_step}.")
    else:
        print_section("STAGE 1: CODEBOOK INITIALIZATION")
        sample_init_size = min(INIT_SAMPLE_SIZE, len(dataset))
        print(f"  ► Sampling max {sample_init_size:,} embeddings for K-Means initialization from memory...")
        
        centroids_list = init_codebooks_batched(
            model=model, 
            dataset_embeddings=dataset.embeddings, 
            codebook_sizes=[args.codebook_size] * args.num_codebooks, 
            device=DEVICE, 
            sample_size=sample_init_size
        )
        with torch.no_grad():
            for i, centroids in enumerate(centroids_list):
                model.codebooks[i].copy_(centroids.to(DEVICE))
        print(f"  ✔ Initialized {args.num_codebooks} codebooks successfully via batched K-Means.")

    # 7. Optional Graph Compilation
    train_model = model
    if args.compile and hasattr(torch, 'compile') and DEVICE.type == 'cuda':
        print("  ► Compiling model with torch.compile...")
        train_model = torch.compile(model)

    # 8. Model Training Loop
    print_section("STAGE 2: MODEL TRAINING LOOP")
    print_table_header()

    reached_max_steps = False

    for epoch in range(start_epoch, args.epochs):
        if reached_max_steps:
            break
            
        for (batch_embs,) in train_dataloader:
            train_model.train()
            batch_embs = batch_embs.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                model_output = train_model(batch_embs)

                entropy_loss = model_output['entropy_loss']
                loss = model_output['loss'] + args.entropy_weight * entropy_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            nn.utils.clip_grad_norm_(raw_model.encoder.parameters(), GRAD_CLIP_MAX_NORM)
            nn.utils.clip_grad_norm_(raw_model.bottleneck_norm.parameters(), GRAD_CLIP_MAX_NORM)
            nn.utils.clip_grad_norm_(raw_model.codebooks, GRAD_CLIP_MAX_NORM)
            nn.utils.clip_grad_norm_(raw_model.decoder.parameters(), GRAD_CLIP_MAX_NORM)

            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()
            
            current_step += 1

            # Update GPU buffers and update EMA tracking EVERY step
            gpu_fixer.update_buffers(model_output['residuals_per_stage'])
            gpu_fixer.update_usage(model_output['clusters_counts'])

            # Periodically revive dead codes using zeroed optimizer states
            if current_step % args.dead_code_fix_interval == 0:
                total_revived = gpu_fixer.fix_dead_codes_ema(raw_model, optimizer=optimizer)
                if total_revived > 0:
                    writer.add_scalar("Training/Dead_Codes_Revived", total_revived, current_step)

            if current_step % args.log_interval == 0 or current_step == 1:
                for i, counts in enumerate(model_output['clusters_counts']):
                    active_codes = (counts > 0).sum().item()
                    writer.add_scalar(f"Codebook_{i}/Batch_Active_Codes", active_codes, current_step)

                writer.add_scalar("Loss/Total", loss.item(), current_step)
                writer.add_scalar("Loss/Reconstruction", model_output['metrics']['recon_loss'], current_step)
                writer.add_scalar("Loss/RQ-VAE", model_output['metrics']['rqvae_loss'], current_step)
                writer.add_scalar("Loss/Entropy", entropy_loss.item(), current_step)
                writer.add_scalar("Training/Learning_Rate", optimizer.param_groups[0]['lr'], current_step)

                print_table_row(
                    epoch=epoch,
                    step=current_step,
                    loss=loss.item(),
                    recon_loss=model_output['metrics']['recon_loss'],
                    rq_loss=model_output['metrics']['rqvae_loss'],
                    lr=optimizer.param_groups[0]['lr']
                )

            if args.max_steps is not None and current_step >= args.max_steps:
                print_table_footer()
                print(f"  ► Target max_steps reached ({args.max_steps}). Terminating training loop.")
                reached_max_steps = True
                break

        if not reached_max_steps:
            if ((epoch + 1) % args.save_interval == 0) or ((epoch + 1) == args.epochs):
                checkpoint_path = os.path.join(checkpoint_dir, f"rqvae_epoch_{epoch}.pth")
                save_checkpoint(model, optimizer, scaler, scheduler, epoch, current_step, checkpoint_path)

    if not reached_max_steps:
        print_table_footer()

    os.makedirs(os.path.dirname(os.path.abspath(args.model_path)) or '.', exist_ok=True)
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save(raw_model.state_dict(), args.model_path)
    print(f"  ✔ Saved final model weights to {args.model_path}")

    # 9. High-Throughput Semantic ID Generation (Encoder-Only Inference)
    print_section("STAGE 3: INFERENCE & SEMANTIC ID GENERATION")
    print(f"  ► Running dataset through encoder only (batch size = {args.inference_batch_size})...")

    raw_model.eval()
    all_clusters = []
    
    with torch.inference_mode():
        for i in range(0, len(dataset.embeddings), args.inference_batch_size):
            batch_slice = dataset.embeddings[i:i + args.inference_batch_size].to(DEVICE, non_blocking=True)
            with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                indices = raw_model.encode_indices(batch_slice)
            all_clusters.append(indices.cpu())

    all_clusters_tensor = torch.cat(all_clusters, dim=0).numpy()

    raw_semantic_mapping = {}
    sem_2_ids = defaultdict(list)
    for item_id, codes in zip(dataset.item_ids, all_clusters_tensor):
        codes_list = codes.tolist()
        raw_semantic_mapping[item_id] = codes_list
        if args.add_unique_id:
            sem_2_ids[tuple(codes_list)].append(item_id)

    # 10. Codebook Utilization Report
    print_section("STAGE 4: CODEBOOK UTILIZATION REPORT")
    cb_metrics = {}
    per_cb_utilization = []
    for cb_idx in range(args.num_codebooks):
        cb_assignments = all_clusters_tensor[:, cb_idx]
        num_unique = len(np.unique(cb_assignments))
        utilization_pct = (num_unique / args.codebook_size) * 100.0
        per_cb_utilization.append(utilization_pct)
        cb_metrics[f"Codebook #{cb_idx}"] = f"{num_unique} / {args.codebook_size} active ({utilization_pct:.2f}%)"
        writer.add_scalar(f"Corpus_Utilization/Codebook_{cb_idx}", utilization_pct, current_step)

    mean_utilization = np.mean(per_cb_utilization)
    cb_metrics["Average Utilization"] = f"{mean_utilization:.2f}%"
    writer.add_scalar("Corpus_Utilization/Average", mean_utilization, current_step)
    
    print_card("Corpus Codebook Utilization", cb_metrics)

    # 11. Deterministic Collision Resolution & Index Saving
    print_section("STAGE 5: COLLISION RESOLUTION & INDEX SAVING")

    semantic_mapping = {}
    if args.add_unique_id:
        unique_bound = args.unique_id_bound if args.unique_id_bound is not None else args.codebook_size
        print(f"  ► Resolving collisions with deterministic ranking suffix (bound = {unique_bound})...")
        
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
                if args.clip_suffix and rank_idx >= unique_bound:
                    suffix_id = unique_bound - 1
                else:
                    suffix_id = rank_idx

                semantic_mapping[item_id] = list(semantics) + [int(suffix_id)]

        collision_summary = {
            "Total Unique Semantic Tuples": f"{len(sem_2_ids):,}",
            "Tuples with Collisions (>1 item)": f"{total_collisions:,}",
            "Max Items Sharing One Tuple": max_collision_len,
            "Groups Exceeding Bound": overflow_count,
        }
        print_card("Collision Resolution Summary", collision_summary)
    else:
        semantic_mapping = raw_semantic_mapping
        print("  ► Collision resolution skipped (--add_unique_id not set).")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or '.', exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(semantic_mapping, f)

    writer.close()

    elapsed = time.time() - start_time
    final_summary = {
        "Total Processing Time": f"{elapsed:.2f}s",
        "Total Items Indexed": f"{len(semantic_mapping):,}",
        "Index Output File": args.output_path,
        "Status": "COMPLETED SUCCESSFULLY"
    }
    print_card("Pipeline Execution Summary", final_summary)
    print_banner("PIPELINE FINISHED")


if __name__ == '__main__':
    main()
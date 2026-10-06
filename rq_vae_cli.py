import os
import json
import math
import argparse
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import KMeans
from collections import deque, defaultdict

# Enable TensorFloat-32 for faster matrix multiplications on modern GPUs (Ampere/Hopper)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class HDF5EmbeddingStreamDataset(IterableDataset):
    def __init__(self, h5_path: str, chunk_size: int = 4096, shuffle_buffer: int = 50000, max_samples: int = None):
        super().__init__()
        self.h5_path = h5_path
        self.chunk_size = chunk_size
        self.shuffle_buffer = shuffle_buffer
        self.max_samples = max_samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        with h5py.File(self.h5_path, 'r', swmr=True) as f:
            oid_ds = f['anonymous_cid']
            emb_ds = f['embedding']
            
            total_len = len(oid_ds)
            if self.max_samples is not None:
                total_len = min(total_len, self.max_samples)
            
            if worker_info is not None:
                per_worker = int(np.ceil(total_len / worker_info.num_workers))
                start_idx = worker_info.id * per_worker
                end_idx = min(start_idx + per_worker, total_len)
            else:
                start_idx, end_idx = 0, total_len

            buffer = []
            for idx in range(start_idx, end_idx, self.chunk_size):
                try:
                    chunk_end = min(idx + self.chunk_size, end_idx)
                    batch_oids = oid_ds[idx:chunk_end]
                    batch_embs = emb_ds[idx:chunk_end]
                    
                    for oid, emb in zip(batch_oids, batch_embs):
                        if isinstance(oid, bytes):
                            oid = oid.decode('utf-8')
                        item_data = {
                            'item_id': str(oid),
                            'embedding': torch.from_numpy(emb).float()
                        }
                        
                        if self.shuffle_buffer <= 0:
                            yield item_data
                            continue

                        if len(buffer) < self.shuffle_buffer:
                            buffer.append(item_data)
                        else:
                            idx_to_yield = np.random.randint(0, len(buffer))
                            yield buffer[idx_to_yield]
                            buffer[idx_to_yield] = item_data
                except Exception as e:
                    print(f"  [!] Warning: Failed to read chunk at index {idx}. Skipping. Error: {e}")
                    continue
                        
            if self.shuffle_buffer > 0 and len(buffer) > 0:
                np.random.shuffle(buffer)
                for item in buffer:
                    yield item

def batch_processor(batch):
    item_ids = [item['item_id'] for item in batch]
    embeddings = torch.stack([item['embedding'] for item in batch])
    return {'item_id': item_ids, 'embedding': embeddings}

class RQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, beta, codebook_sizes):
        super().__init__()
        self.register_buffer('beta', torch.tensor(beta))

        self.encoder = self.make_encoding_tower(input_dim, hidden_dim)
        self.decoder = self.make_encoding_tower(hidden_dim, input_dim)

        self.codebook_sizes = codebook_sizes
        self.codebooks = torch.nn.ParameterList()
        for codebook_size in codebook_sizes:
            cb = torch.FloatTensor(codebook_size, hidden_dim)
            with torch.no_grad():
                torch.nn.init.trunc_normal_(cb, std=0.02, a=-2 * 0.02, b=2 * 0.02)
            self.codebooks.append(cb)

    @staticmethod
    def make_encoding_tower(d1, d2, bias=False):
        return torch.nn.Sequential(
            torch.nn.Linear(d1, d1),
            torch.nn.ReLU(),
            torch.nn.Linear(d1, d2),
            torch.nn.ReLU(),
            torch.nn.Linear(d2, d2, bias=bias)
        )

    @staticmethod
    def get_codebook_indices(remainder, codebook):
        dist = torch.cdist(remainder, codebook)
        return dist.argmin(dim=-1)

    def forward(self, inputs):
        latent_vector = self.encoder(inputs['embedding'])

        latent_restored = 0
        rqvae_loss = 0
        clusters = []
        remainder = latent_vector
        for codebook in self.codebooks:
            codebook_indices = self.get_codebook_indices(remainder, codebook)
            clusters.append(codebook_indices)
            quantized = codebook[codebook_indices]
            codebook_vectors = remainder + (quantized - remainder).detach()
            
            rqvae_loss += self.beta * torch.nn.functional.mse_loss(remainder, quantized.detach())
            rqvae_loss += torch.nn.functional.mse_loss(quantized, remainder.detach())

            latent_restored += codebook_vectors
            remainder = remainder - codebook_vectors

        embeddings_restored = self.decoder(latent_restored)
        recon_loss = torch.nn.functional.mse_loss(embeddings_restored, inputs['embedding'])
        loss = (recon_loss + rqvae_loss).mean()

        clusters_counts = []
        for codebook_size, cluster in zip(self.codebook_sizes, clusters):
            clusters_counts.append(torch.bincount(cluster, minlength=codebook_size))

        return {
            'loss': loss,
            'clusters_counts': clusters_counts,
            'clusters': torch.stack(clusters).T,
            'embedding_hat': embeddings_restored,
            'metrics': dict(
                loss=loss.detach(),
                recon_loss=recon_loss.mean().item(),
                rqvae_loss=rqvae_loss.mean().item()
            ),
        }

def init_codebooks_batched(model, h5_path, codebook_sizes, device, sample_size=100000):
    with h5py.File(h5_path, 'r') as f:
        emb_ds = f['embedding']
        total_len = len(emb_ds)
        
        sample_indices = np.random.choice(total_len, size=min(sample_size, total_len), replace=False)
        sample_indices.sort()
        sampled_data = emb_ds[sample_indices]

    model.eval()
    encoded_list = []
    batch_sz = 4096
    with torch.no_grad():
        for i in range(0, len(sampled_data), batch_sz):
            batch_data = torch.tensor(sampled_data[i:i+batch_sz], dtype=torch.float32).to(device)
            encoded_list.append(model.encoder(batch_data).cpu().numpy())
    
    current_residuals = np.concatenate(encoded_list, axis=0)
    
    codebooks = []
    for cb_size in codebook_sizes:
        kmeans = KMeans(n_clusters=cb_size, n_init=3, random_state=42)
        kmeans.fit(current_residuals)
        centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        codebooks.append(centroids)
        
        labels = kmeans.labels_
        quantized = centroids[labels].numpy()
        current_residuals = current_residuals - quantized

    return codebooks

class ReservoirDeadCodeFixer:
    def __init__(self, buffer_size=50000, hidden_dim=32):
        self.buffer_size = buffer_size
        self.hidden_dim = hidden_dim
        self.buffer = deque(maxlen=buffer_size)

    def update_buffer(self, latent_vectors: torch.Tensor):
        with torch.no_grad():
            for vec in latent_vectors.detach().cpu():
                self.buffer.append(vec)

    def fix_dead_codes(self, codebook: torch.nn.Parameter, cluster_counts: torch.Tensor):
        if len(self.buffer) < 100:
            return 0

        dead_indices = (cluster_counts == 0).nonzero(as_tuple=True)[0]
        if len(dead_indices) == 0:
            return 0

        buffer_list = list(self.buffer)
        
        with torch.no_grad():
            for idx in dead_indices:
                random_lat_idx = np.random.randint(0, len(buffer_list))
                replacement_vector = buffer_list[random_lat_idx].to(codebook.device)
                codebook.data[idx] = replacement_vector
        
        return len(dead_indices)

def save_checkpoint(model, optimizer, scaler, epoch, step, filepath):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None
    }
    torch.save(checkpoint, filepath)

# --- VISUAL FORMATTING LOGGING HELPERS ---

def print_banner(title: str):
    width = 72
    print("\n" + "=" * width)
    print(f"{title.center(width)}")
    print("=" * width)

def print_section(section_name: str):
    width = 72
    print(f"\n┌── [ {section_name} ] " + "─" * (width - len(section_name) - 7))

def print_card(title: str, items: dict, width: int = 72):
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
    parser = argparse.ArgumentParser(description="RQ-VAE Training and Inference CLI Pipeline")
    parser.add_argument('--h5_path', type=str, default='tencent_gr_parquet/train/mm_emb_82.h5', help='Path to input HDF5 file')
    parser.add_argument('--output_path', type=str, default='tencent_gr_parquet/train/index_rqvae.json', help='Path to output JSON mapping')
    parser.add_argument('--model_path', type=str, default='rqvae_model.pth', help='Path to save the trained PyTorch model')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu, cuda:0, etc.)')
    parser.add_argument('--dtype', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'], help='Precision dtype for GPU training (defaults to fp32)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs (overridden if max_steps reached)')
    parser.add_argument('--max_steps', type=int, default=None, help='Target maximum training steps (overrides epoch count when reached)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--max_samples', type=int, default=10000, help='Limit number of samples for local CPU testing (None for all)')
    parser.add_argument('--input_dim', type=int, default=1024, help='Input embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden codebook dimension')
    parser.add_argument('--codebook_size', type=int, default=256, help='Size of each codebook')
    parser.add_argument('--num_codebooks', type=int, default=3, help='Number of residual codebooks')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss beta coefficient')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'adagrad'], help='Optimizer selection')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'none'], help='Learning rate scheduler type')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Linear warmup steps for scheduler')
    parser.add_argument('--log_dir', type=str, default=None, help='TensorBoard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory to save checkpoints')
    parser.add_argument('--add_unique_id', action='store_true', help='Append a unique collision-solver suffix ID to semantic IDs')
    parser.add_argument('--unique_id_bound', type=int, default=None, help='Upper bound for unique collision solver IDs (defaults to codebook_size if None)')

    args = parser.parse_args()

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    DEVICE = torch.device(args.device)

    # Resolve Precision / Dtype Configuration
    dtype_map = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }
    AMP_DTYPE = dtype_map[args.dtype]
    USE_AMP = (DEVICE.type == 'cuda') and (args.dtype != 'fp32')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.log_dir is None:
        log_dir = os.path.join('runs', f'rqvae_{timestamp}')
    else:
        log_dir = args.log_dir

    if args.checkpoint_dir is None:
        checkpoint_dir = os.path.join("checkpoints", f"rqvae_{timestamp}")
    else:
        checkpoint_dir = args.checkpoint_dir

    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    print_banner("RQ-VAE SEMANTIC ID QUANTIZATION PIPELINE")

    # 1. PARAMETER LISTING BLOCK
    param_summary = {
        "Device": str(DEVICE),
        "Precision / AMP": f"{args.dtype} (AMP: {USE_AMP})",
        "Input Dim → Hidden Dim": f"{args.input_dim} → {args.hidden_dim}",
        "Codebooks Architecture": f"{args.num_codebooks} x {args.codebook_size} entries",
        "Commitment Beta": args.beta,
        "Optimizer & LR": f"{args.optimizer.upper()} (lr={args.lr}, warmup={args.warmup_steps})",
        "Epochs / Max Steps": f"{args.epochs} epochs / {args.max_steps if args.max_steps else 'N/A'} max steps",
        "Batch Size": args.batch_size,
        "Dataset Source": args.h5_path,
        "Output Index Path": args.output_path,
        "TensorBoard Logs": log_dir,
        "Checkpoint Dir": checkpoint_dir,
    }
    print_card("Runtime Parameters & System Config", param_summary)

    # 2. STAGE: CODEBOOK INITIALIZATION
    print_section("STAGE 1: CODEBOOK INITIALIZATION")
    sample_init_size = min(100000, args.max_samples if args.max_samples else 100000)
    print(f"  ► Sampling max {sample_init_size:,} embeddings for K-Means initialization...")
    
    dataset = HDF5EmbeddingStreamDataset(
        args.h5_path, 
        chunk_size=4096, 
        shuffle_buffer=min(20000, args.max_samples if args.max_samples else 20000), 
        max_samples=args.max_samples
    )
    train_dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, collate_fn=batch_processor)

    model = RQVAE(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        beta=args.beta,
        codebook_sizes=[args.codebook_size] * args.num_codebooks
    ).to(DEVICE)

    centroids_list = init_codebooks_batched(model, args.h5_path, [args.codebook_size] * args.num_codebooks, device=DEVICE, sample_size=sample_init_size)
    with torch.no_grad():
        for i, centroids in enumerate(centroids_list):
            model.codebooks[i].copy_(centroids.to(DEVICE))
    print(f"  ✔ Initialized {args.num_codebooks} codebooks successfully via batched K-Means.")

    # Optimizer setup
    if args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.scheduler == 'cosine':
        def lr_lambda(current_step: int):
            if current_step < args.warmup_steps:
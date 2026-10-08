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
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import KMeans
from collections import deque, defaultdict

# Enable TensorFloat-32 for faster matrix multiplications on modern GPUs (Ampere/Hopper)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- GLOBAL CONSTANTS & HYPERPARAMETERS ---
RANDOM_SEED = 42
DEFAULT_H5_PATH = 'tencent_gr_parquet/train/mm_emb_82.h5'
DEFAULT_OUTPUT_PATH = 'tencent_gr_parquet/train/index_rqvae.json'
DEFAULT_MODEL_PATH = 'rqvae_model.pth'
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_SAMPLES = None
DEFAULT_INPUT_DIM = 1024
DEFAULT_HIDDEN_DIM = 32
DEFAULT_CODEBOOK_SIZE = 256
DEFAULT_NUM_CODEBOOKS = 3
DEFAULT_BETA = 0.25
DEFAULT_LR = 5e-4
DEFAULT_WARMUP_STEPS = 1000
DEFAULT_LOG_INTERVAL = 100
DEFAULT_DEAD_CODE_FIX_INTERVAL = 100
DEFAULT_SAVE_INTERVAL = 1

# Model initialization constants
TRUNC_NORMAL_STD = 0.02

# Dataset & Loader constants
INIT_SAMPLE_SIZE = 100000
INIT_BATCH_SIZE = 4096
INFERENCE_BATCH_SIZE = 512

# K-Means constants
KMEANS_N_INIT = 3
KMEANS_RANDOM_STATE = 42

# Training & Optimization constants
GRAD_CLIP_MAX_NORM = 1.0
WEIGHT_DECAY = 1e-4
MIN_LR_RATIO = 0.01
DEAD_CODE_FIXER_MIN_BUFFER = 100
SHUFFLE_BUFFER_SIZE = 50000 # Kept only for reservoir fixer buffer size

# UI/Logging constants
BANNER_WIDTH = 72

class HDF5EmbeddingDataset(Dataset):
    def __init__(self, h5_path: str, max_samples: int = None):
        super().__init__()
        self.h5_path = h5_path
        self.max_samples = max_samples

        print(f"  ► Loading entire dataset into memory from {h5_path}...")
        with h5py.File(self.h5_path, 'r') as f:
            oid_ds = f['anonymous_cid']
            emb_ds = f['embedding']
            
            total_len = len(oid_ds)
            if self.max_samples is not None:
                total_len = min(total_len, self.max_samples)
            
            # Load arrays fully into memory
            oids = oid_ds[:total_len]
            embs = emb_ds[:total_len]

        self.item_ids = [oid.decode('utf-8') if isinstance(oid, bytes) else str(oid) for oid in oids]
        self.embeddings = torch.from_numpy(embs).float()
        print(f"  ✔ Successfully loaded {len(self.item_ids):,} samples into memory.")

    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, idx):
        return {
            'item_id': self.item_ids[idx],
            'embedding': self.embeddings[idx]
        }

def batch_processor(batch):
    item_ids = [item['item_id'] for item in batch]
    embeddings = torch.stack([item['embedding'] for item in batch])
    return {'item_id': item_ids, 'embedding': embeddings}

class RQVAE(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=32, beta=0.25, codebook_sizes=[256, 256, 256]):
        super().__init__()
        self.register_buffer('beta', torch.tensor(beta))

        # Encoder: 1024 -> 512 -> 256 -> 128 -> hidden_dim (32, 64)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )

        # Decoder: hidden_dim (32, 64) -> 128 -> 256 -> 512 -> 1024
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

        self.codebook_sizes = codebook_sizes
        self.codebooks = torch.nn.ParameterList()
        for codebook_size in codebook_sizes:
            cb = torch.FloatTensor(codebook_size, hidden_dim)
            with torch.no_grad():
                torch.nn.init.trunc_normal_(cb, std=0.02, a=-0.04, b=0.04)
            self.codebooks.append(cb)

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
            encoded_list.append(model.encoder(batch_data).cpu().numpy())
    
    current_residuals = np.concatenate(encoded_list, axis=0)
    
    codebooks = []
    for cb_size in codebook_sizes:
        kmeans = KMeans(n_clusters=cb_size, n_init=KMEANS_N_INIT, random_state=KMEANS_RANDOM_STATE)
        kmeans.fit(current_residuals)
        centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        codebooks.append(centroids)
        
        labels = kmeans.labels_
        quantized = centroids[labels].numpy()
        current_residuals = current_residuals - quantized

    return codebooks

class ReservoirDeadCodeFixer:
    def __init__(self, buffer_size=SHUFFLE_BUFFER_SIZE, hidden_dim=DEFAULT_HIDDEN_DIM):
        self.buffer_size = buffer_size
        self.hidden_dim = hidden_dim
        self.buffer = deque(maxlen=buffer_size)

    def update_buffer(self, latent_vectors: torch.Tensor):
        with torch.no_grad():
            for vec in latent_vectors.detach().cpu():
                self.buffer.append(vec)

    def fix_dead_codes(self, codebook: torch.nn.Parameter, cluster_counts: torch.Tensor):
        if len(self.buffer) < DEAD_CODE_FIXER_MIN_BUFFER:
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

def save_checkpoint(model, optimizer, scaler, scheduler, epoch, step, filepath):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
    }
    torch.save(checkpoint, filepath)

# --- VISUAL FORMATTING LOGGING HELPERS ---

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
    parser = argparse.ArgumentParser(description="RQ-VAE Training and Inference CLI Pipeline")
    parser.add_argument('--h5_path', type=str, default=DEFAULT_H5_PATH, help='Path to input HDF5 file')
    parser.add_argument('--output_path', type=str, default=DEFAULT_OUTPUT_PATH, help='Path to output JSON mapping')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path to save the trained PyTorch model')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file (.pth) to resume training from')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu, cuda:0, etc.)')
    parser.add_argument('--dtype', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'], help='Precision dtype for GPU training (defaults to fp32)')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of epochs (overridden if max_steps reached)')
    parser.add_argument('--max_steps', type=int, default=None, help='Target maximum training steps (overrides epoch count when reached)')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size')
    parser.add_argument('--max_samples', type=lambda x: None if x.lower() == 'none' else int(x), default=DEFAULT_MAX_SAMPLES, help='Limit number of samples (None for all)')
    parser.add_argument('--input_dim', type=int, default=DEFAULT_INPUT_DIM, help='Input embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=DEFAULT_HIDDEN_DIM, choices=[32, 64], help='Hidden codebook dimension (must be 32 or 64)')
    parser.add_argument('--codebook_size', type=int, default=DEFAULT_CODEBOOK_SIZE, help='Size of each codebook')
    parser.add_argument('--num_codebooks', type=int, default=DEFAULT_NUM_CODEBOOKS, help='Number of residual codebooks')
    parser.add_argument('--beta', type=float, default=DEFAULT_BETA, help='Commitment loss beta coefficient')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'adagrad'], help='Optimizer selection')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR, help='Learning rate')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'none'], help='Learning rate scheduler type')
    parser.add_argument('--warmup_steps', type=int, default=DEFAULT_WARMUP_STEPS, help='Linear warmup steps for scheduler')
    parser.add_argument('--log_dir', type=str, default=None, help='TensorBoard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory to save checkpoints')
    parser.add_argument('--add_unique_id', action='store_true', help='Append a unique collision-solver suffix ID to semantic IDs')
    parser.add_argument('--unique_id_bound', type=int, default=None, help='Upper bound for unique collision solver IDs (defaults to codebook_size if None)')
    parser.add_argument('--log_interval', type=int, default=DEFAULT_LOG_INTERVAL, help='Frequency (in steps) to log metrics to console and TensorBoard')
    parser.add_argument('--dead_code_fix_interval', type=int, default=DEFAULT_DEAD_CODE_FIX_INTERVAL, help='Frequency (in steps) to check for and revive dead codebook entries')
    parser.add_argument('--save_interval', type=int, default=DEFAULT_SAVE_INTERVAL, help='Frequency (in epochs) to save model checkpoints')

    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
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
        "Resume Checkpoint": args.resume if args.resume else "None (Fresh Start)",
        "Log Intervals": f"Log every {args.log_interval} steps",
        "Dead Code Intervals": f"Dead Code Fix every {args.dead_code_fix_interval} steps",
        "Skip Intervals": f"Save every {args.save_interval} epochs",
        "Batch Size": args.batch_size,
        "Dataset Source": args.h5_path,
        "Output Index Path": args.output_path,
        "TensorBoard Logs": log_dir,
        "Checkpoint Dir": checkpoint_dir,
    }
    print_card("Runtime Parameters & System Config", param_summary)

    # Initialize Core Model
    model = RQVAE(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        beta=args.beta,
        codebook_sizes=[args.codebook_size] * args.num_codebooks
    ).to(DEVICE)

    # Initialize Optimizer
    if args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Initialize Scheduler
    if args.scheduler == 'cosine':
        def lr_lambda(current_step: int):
            if current_step < args.warmup_steps:
                return float(current_step) / float(max(1, args.warmup_steps))
            
            target_max_steps = args.max_steps if args.max_steps is not None else INIT_SAMPLE_SIZE
            total_decay_steps = max(1, target_max_steps - args.warmup_steps)
            progress = float(current_step - args.warmup_steps) / float(total_decay_steps)
            progress = min(1.0, max(0.0, progress))
            return max(MIN_LR_RATIO, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda' and args.dtype == 'fp16'))
    reservoir_fixer = ReservoirDeadCodeFixer(buffer_size=SHUFFLE_BUFFER_SIZE, hidden_dim=args.hidden_dim)

    # Dataset loader setup - Fully loaded into memory
    dataset = HDF5EmbeddingDataset(args.h5_path, max_samples=args.max_samples)
    train_dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, collate_fn=batch_processor)

    start_epoch = 0
    current_step = 0

    # 2. STAGE: RESUME OR INITIALIZE CODEBOOKS
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
                    print(f"  [!] Legacy checkpoint detected. Fast-forwarded LR scheduler to Step {current_step}.")
        else:
            model.load_state_dict(checkpoint)
            print("  [!] Loaded raw model weights. Optimizer and scheduler states reset.")

        print(f"  ✔ Successfully loaded model state. Resuming from Epoch {start_epoch}, Step {current_step}.")
    else:
        print_section("STAGE 1: CODEBOOK INITIALIZATION")
        sample_init_size = min(INIT_SAMPLE_SIZE, args.max_samples if args.max_samples else INIT_SAMPLE_SIZE)
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

    # 3. STAGE: TRAINING LOOP
    print_section("STAGE 2: MODEL TRAINING LOOP")
    print_table_header()

    reached_max_steps = False

    for epoch in range(start_epoch, args.epochs):
        if reached_max_steps:
            break
            
        for batch in train_dataloader:
            model.train()
            batch_embs = batch['embedding'].to(DEVICE)
            
            with torch.no_grad():
                latent_vectors = model.encoder(batch_embs)
                reservoir_fixer.update_buffer(latent_vectors)

            optimizer.zero_grad()
            
            with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                model_output = model({'embedding': batch_embs})
                loss = model_output['loss']

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.encoder.parameters(), GRAD_CLIP_MAX_NORM)
            nn.utils.clip_grad_norm_(model.codebooks, GRAD_CLIP_MAX_NORM)
            nn.utils.clip_grad_norm_(model.decoder.parameters(), GRAD_CLIP_MAX_NORM)

            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()
            
            current_step += 1

            if current_step % args.dead_code_fix_interval == 0:
                total_revived = 0
                for cb_param, counts in zip(model.codebooks, model_output['clusters_counts']):
                    revived = reservoir_fixer.fix_dead_codes(cb_param, counts)
                    total_revived += revived
                
                if total_revived > 0:
                    writer.add_scalar("Training/Dead_Codes_Revived", total_revived, current_step)

            if current_step % args.log_interval == 0 or current_step == 1:
                # Log active codes count per codebook
                for i, counts in enumerate(model_output['clusters_counts']):
                    active_codes = (counts > 0).sum().item()
                    writer.add_scalar(f"Codebook_{i}/Batch_Active_Codes", active_codes, current_step)

                # Log loss terms & learning rate
                writer.add_scalar("Loss/Total", loss.item(), current_step)
                writer.add_scalar("Loss/Reconstruction", model_output['metrics']['recon_loss'], current_step)
                writer.add_scalar("Loss/RQ-VAE", model_output['metrics']['rqvae_loss'], current_step)
                writer.add_scalar("Training/Learning_Rate", optimizer.param_groups[0]['lr'], current_step)

                # Format terminal output
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
    torch.save(model.state_dict(), args.model_path)
    print(f"  ✔ Saved final model checkpoint to {args.model_path}")

    # 4. STAGE: SEMANTIC ID GENERATION
    print_section("STAGE 3: INFERENCE & SEMANTIC ID GENERATION")
    print("  ► Running loaded dataset through RQ-VAE encoder...")

    inference_loader = DataLoader(dataset=dataset, batch_size=INFERENCE_BATCH_SIZE, shuffle=False, collate_fn=batch_processor)

    raw_semantic_mapping = {}
    sem_2_ids = defaultdict(list)
    model.eval()

    with torch.inference_mode():
        for batch in inference_loader:
            batch['embedding'] = batch['embedding'].to(DEVICE)
            
            with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                model_output = model(batch)
                
            for item_id, semantic_ids in zip(batch['item_id'], model_output['clusters'].cpu().tolist()):
                raw_semantic_mapping[item_id] = semantic_ids
                if args.add_unique_id:
                    sem_2_ids[tuple(semantic_ids)].append(item_id)

    # 5. STAGE: CODEBOOK UTILIZATION REPORT
    print_section("STAGE 4: CODEBOOK UTILIZATION REPORT")
    all_assigned_codes = np.array(list(raw_semantic_mapping.values()))
    
    cb_metrics = {}
    per_cb_utilization = []
    for cb_idx in range(args.num_codebooks):
        cb_assignments = all_assigned_codes[:, cb_idx]
        num_unique = len(np.unique(cb_assignments))
        utilization_pct = (num_unique / args.codebook_size) * 100.0
        per_cb_utilization.append(utilization_pct)
        cb_metrics[f"Codebook #{cb_idx}"] = f"{num_unique} / {args.codebook_size} active ({utilization_pct:.2f}%)"
        writer.add_scalar(f"Corpus_Utilization/Codebook_{cb_idx}", utilization_pct, current_step)

    mean_utilization = np.mean(per_cb_utilization)
    cb_metrics["Average Utilization"] = f"{mean_utilization:.2f}%"
    writer.add_scalar("Corpus_Utilization/Average", mean_utilization, current_step)
    
    print_card("Corpus Codebook Utilization", cb_metrics)

    # 6. STAGE: COLLISION RESOLUTION & SUMMARY
    print_section("STAGE 5: COLLISION RESOLUTION & INDEX SAVING")

    semantic_mapping = {}

    if args.add_unique_id:
        unique_bound = args.unique_id_bound if args.unique_id_bound is not None else args.codebook_size
        print(f"  ► Resolving collisions with unique ID suffix (bound = {unique_bound})...")
        
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
                collision_solvers = np.random.choice(unique_bound, size=num_items, replace=True).tolist()
            else:
                collision_solvers = np.random.permutation(unique_bound)[:num_items].tolist()

            for item_id, solver in zip(item_ids, collision_solvers):
                semantic_mapping[item_id] = list(semantics) + [int(solver)]

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

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
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
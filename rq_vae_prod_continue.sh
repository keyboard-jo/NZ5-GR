#!/bin/bash
#SBATCH --job-name=rqvae-gpu-resume
#SBATCH --output=logs/gpu-resume-%j.out
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --gpus-per-node=1

module purge

module load Anaconda3/2023.09-0
module load cuda12.2/toolkit/12.2.2

# Safely initialize and activate local Conda environment
eval "$(conda shell.bash hook)"

conda activate "~/.conda/envs/rqvae_env"

# Create output directories
mkdir -p logs checkpoints test_output

CHECKPOINT_DIR="checkpoints/gpu_prod"

# Auto-detect latest epoch checkpoint, or allow explicit override via RESUME_PATH environment variable
if [ -n "$RESUME_PATH" ]; then
    LATEST_CKPT="$RESUME_PATH"
else
    LATEST_CKPT=$(ls -v ${CHECKPOINT_DIR}/rqvae_epoch_*.pth 2>/dev/null | tail -n 1)
fi

if [ -z "$LATEST_CKPT" ] || [ ! -f "$LATEST_CKPT" ]; then
    echo "[!] Error: No valid checkpoint found in '${CHECKPOINT_DIR}' to resume from."
    exit 1
fi

echo "► Resuming training from checkpoint: ${LATEST_CKPT}"

# Execute GPU Resume Task
python rq_vae_cli.py \
  --resume "${LATEST_CKPT}" \
  --h5_path <dir_to_data> \
  --output_path <dir_to_output> \
  --model_path <dir_to_model> \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --device "cuda" \
  --dtype "bf16" \
  --batch_size 4096 \
  --input_dim 1024 \
  --hidden_dim 64 \
  --num_codebooks 4 \
  --codebook_size 256 \
  --beta 0.25 \
  --optimizer "adamw" \
  --scheduler "cosine" \
  --lr 5e-4 \
  --epochs 50 \
  --warmup_steps 2000 \
  --log_interval 200 \
  --dead_code_fix_interval 100 \
  --add_unique_id
#!/bin/bash
#SBATCH --job-name=rqvae-gpu-test
#SBATCH --output=logs/gpu-test-%j.out
#SBATCH --time=00:15:00
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

# Execute GPU Pilot Test
python rq_vae_cli_2.py \
  --h5_path <dir_to_data> \
  --output_path <dir_to_output> \
  --model_path <dir_to_model> \
  --checkpoint_dir dir_to_checkpoint \
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
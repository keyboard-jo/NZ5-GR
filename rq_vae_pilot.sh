#!/bin/bash
#SBATCH --job-name=rqvae-gpu-test
#SBATCH --output=logs/gpu-test-%j.out
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=<your-account>
#SBATCH --partition=<your-gpu-partition>
#SBATCH --gpus-per-node=1

# Safely initialize and activate local Conda environment
eval "$(conda shell.bash hook)"
conda activate ./.conda

# Create output directories
mkdir -p logs checkpoints test_output

# Execute GPU Pilot Test
python rq_vae_cli.py \
  --h5_path data/mm_emb_82.h5 \
  --device cuda \
  --dtype fp32 \
  --max_samples 100000 \
  --max_steps 200 \
  --batch_size 1024 \
  --codebook_size 256 \
  --num_codebooks 3 \
  --hidden_dim 32 \
  --epochs 5 \
  --optimizer adamw \
  --lr 5e-4 \
  --scheduler cosine \
  --warmup_steps 50 \
  --add_unique_id \
  --model_path test_output/rqvae_model_test.pth \
  --output_path test_output/index_rqvae_test.json \
  --checkpoint_dir checkpoints/gpu_test
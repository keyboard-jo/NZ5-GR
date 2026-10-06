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

# Ensure local Conda environment is active
source activate ./.conda

# Create output dirs if not existing
mkdir -p logs checkpoints test_output

python rq_vae_cli.py \
  --h5_path data/mm_emb_82.h5 \
  --device cuda \
  --max_samples 100000 \
  --batch_size 1024 \
  --codebook_size 256 \
  --num_codebooks 3 \
  --epochs 5 \
  --optimizer adamw \
  --lr 5e-4 \
  --scheduler cosine \
  --warmup_steps 100 \
  --add_unique_id \
  --model_path test_output/rqvae_model_test.pth \
  --output_path test_output/index_rqvae_test.json \
  --checkpoint_dir checkpoints/gpu_test
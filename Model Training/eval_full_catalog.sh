#!/bin/bash
#SBATCH --job-name=digr_fullcat_eval
#SBATCH --output=logs/digr_fullcat_%j.out
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=mscbdtspod2026
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1

module purge
module load Anaconda3/2023.09-0
module load cuda12.2/toolkit/12.2.2

eval "$(conda shell.bash hook)"
conda activate "${HOME}/.conda/envs/rqvae_env"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

export HF_HUB_CACHE="${HOME}/NZ5/train_model/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p logs

DATA_DIR=/scratch/mscbdtspod2026/jksulaiman/data/digr_data
RUN_DIR=/scratch/mscbdtspod2026/jksulaiman/data/digr_runs/digr_resume_20261025_113432

# Scores the clicked-subset checkpoint over ALL 4,783,154 catalog items, the
# same space the popularity baseline was measured in.  No training, one pass.
python eval_full_catalog.py \
  --checkpoint "${RUN_DIR}/best" \
  --data_dir "${DATA_DIR}" \
  --split val \
  --batch_size 128 \
  --local_files_only \
  --out_json "${RUN_DIR}/full_catalog_metrics.json"

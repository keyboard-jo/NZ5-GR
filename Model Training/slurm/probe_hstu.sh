#!/bin/bash
#SBATCH --job-name=probe_hstu
#SBATCH --output=logs/probe_hstu_%j.out
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=mscbdtspod2026
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1

# Run from the parent dir so `logs/` and `python *.py` keep resolving the way
# they did when this script lived next to the code. `sbatch slurm/<name>.sh`.
cd "$(dirname "$0")/.." || exit 1

module purge
module load Anaconda3/2023.09-0
module load cuda12.2/toolkit/12.2.2

eval "$(conda shell.bash hook)"
conda activate "${HOME}/.conda/envs/rqvae_env"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p logs

# Stage-1 artifacts currently live under ~/NZ5/make_data. The probe wants a GPU
# (4.78M x 256 scoring plus the ~10 GB bf16 feature store), so move the two paths
# below to /scratch first and update them. The checkpoint dir must also hold
# item_embeddings.npy and item_ids.npy (they do: exported next to the .pt).
STAGE1_DIR="${HOME}/NZ5/make_data/archive/2026-10-13-ei-1/test_output/stage1_pilot"
ITEMS="${HOME}/NZ5/make_data/data/canonical_items_cleaned.parquet"

DATA_DIR=/scratch/mscbdtspod2026/jksulaiman/data/digr_data
OUT_DIR=/scratch/mscbdtspod2026/jksulaiman/data/digr_runs/probe_hstu
mkdir -p "${OUT_DIR}"

# Scores the trained HSTU tower (encode_user -> cosine vs all 4,783,154 item
# vectors) on the exact DiGR val split. Actions are approximated as all-click
# because the DiGR tokens drop them; `--actions exact` (seq.parquet) is the
# follow-up if this lower bound is inconclusive.
python probe_hstu.py \
  --checkpoint "${STAGE1_DIR}/stage1_rec_model.pt" \
  --items_path "${ITEMS}" \
  --data_dir "${DATA_DIR}" \
  --split val \
  --history_window 20 \
  --actions approx \
  --ks 1,10,100,1000 \
  --topk 1000 \
  --batch_size 512 \
  --device auto \
  --dtype bfloat16 \
  --out_json "${OUT_DIR}/probe_hstu_metrics.json"

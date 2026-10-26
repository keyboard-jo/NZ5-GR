#!/bin/bash
#SBATCH --job-name=digr_emb_nextclick
#SBATCH --output=logs/digr_emb_nextclick_%j.out
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=mscbdtspod2026
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1

# Run from the parent dir so `logs/` and `python *.py` keep resolving the way
# they did when the slurm wrappers lived next to the code.
cd "$(dirname "$0")/.." || exit 1

module purge
module load Anaconda3/2023.09-0
module load cuda12.2/toolkit/12.2.2

eval "$(conda shell.bash hook)"
conda activate "${HOME}/.conda/envs/rqvae_env"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p logs

DATA_DIR="${DATA_DIR:-/scratch/mscbdtspod2026/jksulaiman/data/digr_data}"
# The parquet the RQ-VAE runs consumed (see local/2026-16-10-3-rq/rq_vae_prod.sh).
# --embedding_ids is not needed for a parquet: rows are aligned by item_id.
EMBEDDINGS="${EMBEDDINGS:-/scratch/mscbdtspod2026/jksulaiman/data/item_embeddings.parquet}"
OUT_JSON="${OUT_JSON:-${HOME}/NZ5/digr_baselines/embedding_knn_nextclick.json}"

# Does the Stage-1 geometry predict the next click, or only its own nearest
# neighbours? Scores the last clicked item's embedding against val/targets.npy
# over all 4,783,154 catalog items, plus the depth-1/2 SID buckets ranked by
# cosine. One pass, no training, no checkpoint.
#
# The 4.78M x 256 matrix is 4.9 GB in float32 and the parquet read makes a second
# copy, hence --mem=64G. Add `--max_rows 5000` for a smoke test (about a minute).
python embedding_knn_nextclick.py \
  --data_dir "${DATA_DIR}" \
  --embeddings "${EMBEDDINGS}" \
  --split val \
  --ks 1,10,100,1000 \
  --windows 1,3,5 \
  --sid_depths 1,2 \
  --topk 1000 \
  --out_json "${OUT_JSON}"

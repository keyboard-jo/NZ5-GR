#!/bin/bash
#SBATCH --job-name=digr_baselines
#SBATCH --output=logs/digr_baselines_%j.out
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --account=mscbdtspod2026
#SBATCH --partition=normal

module purge
module load Anaconda3/2023.09-0

eval "$(conda shell.bash hook)"
conda activate "${HOME}/.conda/envs/rqvae_env"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p logs

DATA_DIR=/scratch/mscbdtspod2026/jksulaiman/data/digr_data
OUT_DIR=/scratch/mscbdtspod2026/jksulaiman/data/digr_runs/baselines

# Three ranks over all 4,783,154 catalog items. No model, no GPU, no checkpoint:
# popularity is the frequency floor, B3 counts SID-level transitions, B4 counts
# item co-clicks. All are graded on the same val targets as the DiGR run.
python baselines_digr.py \
  --data_dir "${DATA_DIR}" \
  --eval_split val \
  --baselines popularity,sid_transition,coclick \
  --history_window 1 \
  --smoothing 0.5 \
  --weighting uniform \
  --workers 8 \
  --out_json "${OUT_DIR}/baselines_h1.json"

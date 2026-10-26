#!/bin/bash
#SBATCH --job-name=digr_baselines
#SBATCH --output=logs/digr_baselines_%j.out
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --account=mscbdtspod2026
#SBATCH --partition=cpu

module purge
module load Anaconda3/2023.09-0

eval "$(conda shell.bash hook)"
conda activate "${HOME}/.conda/envs/rqvae_env"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p logs

# The cpu partition does not mount /scratch, so the data has to come from $HOME.
DATA_DIR="${DATA_DIR:-${HOME}/NZ5/train_model/archive/2026-10-23-preprocess-01/digr_data}"
OUT_DIR="${OUT_DIR:-${HOME}/NZ5/digr_baselines}"
mkdir -p "${OUT_DIR}"

# Two baselines over all 4,783,154 catalog items. No model, no GPU, no
# checkpoint. B3 (sid_transition) is left out: levels 0-3 are 256x256 tables
# holding 34M transitions, i.e. saturated, so it scored 7.5x worse than
# popularity. It is still selectable with --baselines sid_transition.
#
# The history window is swept because the published co-click run used window 1,
# the weakest form, and already tied DiGR at R@10 and R@100. 10000 is included
# so the recall curve can be read instead of extrapolated.
for W in 1 3 5; do
  python baselines_digr.py \
    --data_dir "${DATA_DIR}" \
    --eval_split val \
    --baselines popularity,coclick \
    --history_window "${W}" \
    --smoothing 0.5 \
    --weighting uniform \
    --ks 1,10,100,1000,10000 \
    --workers 8 \
    --out_json "${OUT_DIR}/baselines_h${W}.json"
done

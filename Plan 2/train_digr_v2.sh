#!/bin/bash
#SBATCH --job-name=digr_v2_train
#SBATCH --output=train_v2_%j.out
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=mscbdtspod2026
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1

# S4: DiGR on the S3 stream, 2 x 8,192 SID, action token, 128 gap bins.
#
# Same config as the frozen run digr_0p6b_e3_full_sampled300k, which is the row
# the table compares against. Anything this file changes is a second axis, so it
# changes one thing and that thing is the temperature, explained below.
#
#   sbatch "Plan 2/train_digr_v2.sh"                       # 3 epochs, about 3.2 h
#   sbatch "Plan 2/train_digr_v2.sh" --max_steps 30 --run_name smoke_v2
#                                                          # shape check, 5 min
#
# Temperature. The score is z = (1/L) * sum_l e_l(s_l) . h, divided by the
# temperature, so the model sees 1/(L * T). With L=5 and T=0.07 that is 2.857,
# and with L=2 it would be 7.143, 2.5x sharper for reasons that have nothing to
# do with the data. L=2 at T=0.175 holds 1/(L*T) at the value the frozen run was
# tuned at, so the score scale is unchanged and the SID and the event stream are
# the only differences.
#
# 153,461,256 train tokens against 221,500,536, so 3 epochs should take about
# 3h10m against the frozen run's 4h34m. --save_steps 300 keeps mid-epoch
# checkpoints, so a wall-clock kill is resumable with --resume.
cd "$(dirname "$0")" || exit 1

DATA_DIR="${HOME}/NZ5/plan2/digr_data"
OUT_DIR="${HOME}/NZ5/plan2/runs"

for f in "${DATA_DIR}/vocab.json" "${DATA_DIR}/preprocess_meta.json" \
         "${DATA_DIR}/catalog_item_ids.npy" "${DATA_DIR}/train/tokens.npy"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

module purge
module load Anaconda3/2023.09-0
module load cuda12.2/toolkit/12.2.2
eval "$(conda shell.bash hook)"
conda activate rqvae_env

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 ARROW_NUM_THREADS=8
export HF_HUB_CACHE="${HOME}/NZ5/train_model/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Everything from Plan 2/ is copied into ~/NZ5/plan2/, so train_digr.py and
# preprocess_digr_v2.py sit in the same directory as this script and the cd above
# already puts us there. Locally they are in different directories; train_digr.py
# handles both, so nothing here depends on it.
python train_digr.py \
    --mode train \
    --model_name Qwen/Qwen3-0.6B \
    --local_files_only \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUT_DIR}" \
    --run_name digr_v2_sid2x8192_e3_sampled300k \
    --epochs 3 \
    --batch_size 512 \
    --sampled_softmax \
    --num_negatives 300000 \
    --label_subset full \
    --temperature 0.175 \
    --save_steps 300 \
    --log_interval 20 \
    "$@"

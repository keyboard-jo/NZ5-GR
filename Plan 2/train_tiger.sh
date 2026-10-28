#!/bin/bash
#SBATCH --job-name=tiger_train
#SBATCH --output=tiger_%j.out
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=mscbdtspod2026
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1

# TIGER, run against either SID release. The script reads everything it needs from
# digr_data/, so the only difference between the two is which directory is
# --data_dir and what the level widths are.
#
#   sbatch "Plan 2/train_tiger.sh"                    # v2, 2 x 8,192, PREFIX=1, about 3 h
#   SID_VERSION=1 sbatch "Plan 2/train_tiger.sh"      # v1, 4 x 256 + 731, about 4.5 h
#   PREFIX=0 sbatch "Plan 2/train_tiger.sh"           # additive ablation, not the paper
#   EPOCHS=1 sbatch "Plan 2/train_tiger.sh"           # one epoch
#
# Scoring. Two arms, and the flag picks the objective, not the level count.
#
# PREFIX=1 (default, the paper's TIGER). The codes go into the input and level l is
# predicted from the separator plus codes 1..l-1, which is the factorisation in the
# TIGER paper. The score of an item now depends on the path taken to it, so all
# items can no longer be scored independently: candidates have to be generated,
# which means a beam, which means a trie, because a beam will walk into prefixes
# that no item owns. That last point is what makes the tree mandatory whenever
# PREFIX=1, at two levels as much as at five. The cost is inference: a beam of
# width b over L levels is O(b * L) forwards per query instead of one, and the
# cut-offs it can report are capped at b, so R@1000 and R@10000 are not computable
# in this arm. Validation during training therefore reports loss and `cover@k`
# only; `--mode eval` refuses this checkpoint on purpose and the ranking numbers
# come from the beam pass.
#
# PREFIX=0 (the ablation, already measured). Codes are not in the input; every level
# reads the separator state and the score of an item is the sum of its own code's
# log-probability at each level. That score is computable for all 4,783,154 items in
# one forward pass, so the evaluator reports the same five cut-offs as the DiGR row.
# No trie and no beam are needed, because nothing is generated. Row length equals
# DiGR's, so an epoch costs the same. (It is 90 tokens against DiGR's 89, one
# separator, so about 1% more.) At two levels this score is additive over the two
# code rows, which is the same function class as DiGR, so it measures a loss change
# rather than the factorisation.
#
# Level widths are read from vocab.json, never guessed, so v1 (5 levels) and v2
# (2 levels) both work with no flag.
# Training is full fine-tuning with no LoRA, to match the frozen DiGR run,
# which was also full fine-tuning.
cd "$(dirname "$0")" || exit 1

SID_VERSION="${SID_VERSION:-2}"
EPOCHS="${EPOCHS:-3}"
PREFIX="${PREFIX:-1}"
SUFFIX=""
[ "${PREFIX}" = "1" ] && SUFFIX="_prefix"

if [ "${SID_VERSION}" = "1" ]; then
    DATA_DIR="${DATA_DIR:-${HOME}/NZ5/train_model/archive/2026-10-23-preprocess-01/digr_data}"
    RUN_NAME="${RUN_NAME:-tiger_v1_sid4x256_e${EPOCHS}${SUFFIX}}"
else
    DATA_DIR="${DATA_DIR:-${HOME}/NZ5/plan2/digr_data}"
    RUN_NAME="${RUN_NAME:-tiger_v2_sid2x8192_e${EPOCHS}${SUFFIX}}"
fi
OUT_DIR="${OUT_DIR:-${HOME}/NZ5/plan2/runs}"

PREFIX_FLAG="--no-prefix_conditioning"
[ "${PREFIX}" = "1" ] && PREFIX_FLAG="--prefix_conditioning"

for f in "${DATA_DIR}/vocab.json" "${DATA_DIR}/preprocess_meta.json" \
         "${DATA_DIR}/catalog_item_ids.npy" "${DATA_DIR}/train/tokens.npy"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

echo "SID version ${SID_VERSION}: ${DATA_DIR}"
echo "run name ${RUN_NAME}, prefix conditioning ${PREFIX}"

module purge
module load Anaconda3/2023.09-0
module load cuda12.2/toolkit/12.2.2
eval "$(conda shell.bash hook)"
conda activate rqvae_env

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 ARROW_NUM_THREADS=8
export HF_HUB_CACHE="${HOME}/NZ5/train_model/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python train_tiger.py \
    --mode train \
    --model_name Qwen/Qwen3-0.6B \
    --local_files_only \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --epochs "${EPOCHS}" \
    --batch_size 512 \
    --lr 1e-4 \
    --gradient_checkpointing \
    --reinit_embeddings \
    "${PREFIX_FLAG}" \
    --save_steps 300 \
    --log_interval 20 \
    "$@"

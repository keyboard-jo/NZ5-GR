#!/bin/bash
#SBATCH --job-name=abl_train
#SBATCH --output=abl_train_%j.out
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=mscbdtspod2026
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1

# Task 1 of docs/tasks.md, training half. One 2-epoch DiGR run per mixed cell.
#
#   ARM=a sbatch train_arm.sh     # 2x8192 SID, v1 features
#   ARM=b sbatch train_arm.sh     # 4x256+731 SID, v2 features
#
# Resuming after a wall-clock kill or a node failure:
#
#   ARM=a RESUME=auto sbatch train_arm.sh        # this arm's newest final/
#   ARM=a RESUME=/path/to/run/final sbatch train_arm.sh
#
# `final/` is rewritten atomically every --save_steps micro-batches (300), so it
# is the newest state including a mid-epoch save, and --resume skips the
# micro-batches of the current epoch that were already consumed. --epochs stays
# the total target (2), not the number left.
#
# Everything else is the frozen v2 configuration: Qwen3-0.6B, batch 512, sampled
# softmax with 300k uniform negatives, full label subset, 2 epochs. Epoch 2 is
# where the v2 sweep bottomed out, and 3 epochs overfits, so 2 is the comparison
# and it is also the cheaper one.
#
# The temperature is derived, not typed. The score the model sees is
# 1/(L * T), and the frozen run was tuned at 2.857 with L=5 and T=0.07. Changing
# L for reasons that have nothing to do with the data would silently rescale the
# logits, so T is set to 0.35 / L, which holds 1/(L*T) at 2.857 for any L. The
# script reads L out of the arm's own vocab.json and refuses an arm whose level
# count is not the one that arm is supposed to have, because a wrong --data_dir
# is otherwise invisible until the numbers look odd.
set -u

ARM="${ARM:-a}"
BASE="${BASE:-${HOME}/NZ5/ablation}"
CODE_DIR="${CODE_DIR:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ -z "${CODE_DIR}" ]; then
    for d in "${HOME}/NZ5/plan2" "${HOME}/NZ5/ablation" "${HERE}"; do
        if [ -f "${d}/train_digr.py" ]; then CODE_DIR="${d}"; break; fi
    done
fi
[ -n "${CODE_DIR}" ] && [ -f "${CODE_DIR}/train_digr.py" ] || {
    echo "train_digr.py not found. Set CODE_DIR to the directory holding" >&2
    echo "train_digr.py, preprocess_digr_v2.py and eval_full_catalog.py." >&2
    exit 1
}

case "${ARM}" in
    a) EXPECT_LEVELS=2; EXPECT_SIZES="[8192, 8192]"; RUN_NAME=digr_abl_sidv2_featv1_e2 ;;
    b) EXPECT_LEVELS=5; EXPECT_SIZES="[256, 256, 256, 256, 731]"; RUN_NAME=digr_abl_sidv1_featv2_e2 ;;
    *) echo "ARM must be a or b, got '${ARM}'" >&2; exit 1 ;;
esac

DATA_DIR="${BASE}/digr_data_${ARM}"
OUT_DIR="${BASE}/runs"

for f in "${DATA_DIR}/vocab.json" "${DATA_DIR}/preprocess_meta.json" \
         "${DATA_DIR}/catalog_item_ids.npy" "${DATA_DIR}/train/tokens.npy"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; echo "run: ARM=${ARM} sbatch prep_arm.sh" >&2; exit 1; }
done

module purge
module load Anaconda3/2023.09-0
module load cuda12.2/toolkit/12.2.2

eval "$(conda shell.bash hook)"
conda activate "${HOME}/.conda/envs/rqvae_env" 2>/dev/null \
    || conda activate rqvae_env 2>/dev/null \
    || true

python -c "import torch, transformers" >/dev/null 2>&1 || {
    echo "no python with torch and transformers after activating the env" >&2
    exit 1; }

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 ARROW_NUM_THREADS=8
export HF_HUB_CACHE="${HOME}/NZ5/train_model/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

LEVELS=$(python -c "import json;print(json.load(open('${DATA_DIR}/vocab.json'))['num_levels'])")
if [ "${LEVELS}" != "${EXPECT_LEVELS}" ]; then
    echo "arm ${ARM} expects ${EXPECT_LEVELS} SID levels, ${DATA_DIR} has ${LEVELS}" >&2
    exit 1
fi
# The level count alone does not identify the codebook: 4x256+731 is five levels
# and so is any other five-level SID. The sizes do, so check them here too, before
# the head is built and the temperature is derived from the wrong L.
SIZES=$(python -c "import json;print(json.load(open('${DATA_DIR}/vocab.json'))['sid_sizes'])")
if [ "${SIZES}" != "${EXPECT_SIZES}" ]; then
    echo "arm ${ARM} expects sid_sizes ${EXPECT_SIZES}, ${DATA_DIR} has ${SIZES}" >&2
    echo "that is a different SID, not a different amount of training" >&2
    exit 1
fi
TEMP=$(python -c "print(f'{0.35/${LEVELS}:.4f}')")

# --- resume -----------------------------------------------------------------
# The checkpoint is the authority on the recipe: on resume the trainer takes the
# vocabulary, the level count, the temperature and the window from the checkpoint
# and refuses a data dir whose manifest disagrees, so a resume cannot silently
# become a different arm. Passing --temperature here is inert in that case.
RESUME_FLAG=()
RESUME_DIR=""
if [ -n "${RESUME:-}" ]; then
    if [ "${RESUME}" = "auto" ]; then
        RESUME_DIR="${OUT_DIR}/${RUN_NAME}/final"
        [ -d "${RESUME_DIR}" ] || {
            echo "RESUME=auto but ${RESUME_DIR} does not exist; nothing to resume." >&2
            echo "submit without RESUME for a fresh run, or name the checkpoint:" >&2
            echo "  ls -d ${OUT_DIR}/${RUN_NAME}/*/" >&2
            exit 1
        }
    else
        RESUME_DIR="${RESUME}"
    fi
    [ -d "${RESUME_DIR}" ] || {
        echo "RESUME must be a checkpoint directory, got: ${RESUME_DIR}" >&2; exit 1; }
    [ -f "${RESUME_DIR}/head_config.json" ] || {
        echo "${RESUME_DIR} has no head_config.json, so it is not a checkpoint" >&2; exit 1; }
    RESUME_FLAG=(--resume "${RESUME_DIR}")
else
    [ -d "${OUT_DIR}/${RUN_NAME}/final" ] && \
        echo "note: ${OUT_DIR}/${RUN_NAME}/final exists; set RESUME=auto to continue it"
fi

echo "=============================================================="
echo "arm         : ${ARM}"
echo "data         : ${DATA_DIR}"
echo "run name     : ${RUN_NAME}"
echo "sid levels   : ${LEVELS}   temperature ${TEMP}   (1/(L*T) = 2.857)"
echo "output       : ${OUT_DIR}/${RUN_NAME}"
echo "code         : ${CODE_DIR}"
echo "resume       : ${RESUME_DIR:-no (fresh run)}"
echo "=============================================================="

python "${CODE_DIR}/train_digr.py" \
    --mode train \
    --model_name Qwen/Qwen3-0.6B \
    --local_files_only \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --epochs 2 \
    --batch_size 512 \
    --sampled_softmax \
    --num_negatives 300000 \
    --label_subset full \
    --temperature "${TEMP}" \
    --save_steps 300 \
    --log_interval 20 \
    "${RESUME_FLAG[@]}" \
    "$@"

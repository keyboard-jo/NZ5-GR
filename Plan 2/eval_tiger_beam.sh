#!/bin/bash
#SBATCH --job-name=tiger_beam
#SBATCH --output=tiger_beam_%j.out
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=mscbdtspod2026
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1

# The ranking pass for a prefix-conditioned TIGER checkpoint. `--mode eval` refuses
# this checkpoint on purpose (its score depends on the path taken, so it cannot be
# computed for all 4.78M items in one forward), so the beam is the only way to turn
# `cover@k` in the training log into an actual R@k for the table in docs/results.md.
#
#   sbatch eval_tiger_beam.sh                      # beam 20 over all 98,304 val rows
#   BEAM_ROWS=2048 sbatch eval_tiger_beam.sh       # smoke test, minutes not hours
#   BEAM_WIDTH=100 sbatch eval_tiger_beam.sh       # adds R@100 and nDCG@100
#   CKPT=runs/<run>/final sbatch eval_tiger_beam.sh
#   PREFIX=0 sbatch eval_tiger_beam.sh             # ablation leg, see below
#   DUMP_ROWS=dumps/beam_v2_prefix.npz sbatch eval_tiger_beam.sh
#
# What the dump is for. `budget_match.py` needs the tuples the beam kept, because
# 20 tuples carry a variable number of catalogue items (mean ~30 here, up to 315 in
# one tuple) and only that count makes a matched-budget comparison with DiGR honest.
# The dump adds seconds to a 47-minute run and cannot be reconstructed afterwards,
# so it is on whenever DUMP_ROWS is set. Use it with BEAM_ROWS=0: a partial dump is
# refused by the reader, since a budget from one row set cannot be paired with a
# rank from another.
#
# What comes back. R@k, nDCG@k and MRR@10 for every k up to the beam width, all
# counted end to end: a row whose target never entered the beam is a miss, so these
# are directly comparable with DiGR's and co-click's rows. Cut-offs above the beam
# width are not reported and cannot be, which is the one asymmetry in that table and
# it runs against this arm, not for it. R@1000 and R@10000 stay empty.
#
# Read the decoded line first. `beam@20: N/98304 targets decoded` must come out at
# or below the training log's `cover@20` (43.30% at epoch 2). Higher means the beam
# is scoring from a state the head never saw, and every rank below it is wrong.
#
# The ablation leg. A prefix-free checkpoint scored through the beam is not the same
# run as `--mode eval`, because the prefix tokens go into the row and a prefix-free
# head never learned on that state. It prints a warning and the numbers are
# indicative only. The authoritative prefix-free row is the exhaustive one already
# in the table (R@1 1.44%, R@10 2.45%, R@100 6.59%, R@1000 20.08%).

cd "$(dirname "$0")" || exit 1

CKPT="${CKPT:-runs/tiger_v2_sid2x8192_e2_prefix/best}"
DATA_DIR="${DATA_DIR:-${HOME}/NZ5/plan2/digr_data}"
PREFIX="${PREFIX:-1}"
BEAM_WIDTH="${BEAM_WIDTH:-20}"
BEAM_ROWS="${BEAM_ROWS:-0}"
EVAL_BATCH="${EVAL_BATCH:-128}"
DUMP_ROWS="${DUMP_ROWS:-}"

DUMP_FLAG=()
if [ -n "${DUMP_ROWS}" ]; then
    mkdir -p "$(dirname "${DUMP_ROWS}")"
    DUMP_FLAG=(--dump_rows "${DUMP_ROWS}")
    [ "${BEAM_ROWS}" = "0" ] || \
        echo "  ! DUMP_ROWS set with BEAM_ROWS=${BEAM_ROWS}; budget_match.py refuses a
     partial dump, so set BEAM_ROWS=0 for a run whose dump will be used" >&2
fi

# Empty unless the ablation leg is asked for, so it splits to no argument.
PREFIX_FLAG="--prefix_conditioning"
[ "${PREFIX}" = "1" ] || PREFIX_FLAG=""

[ -d "$CKPT" ] || { echo "missing checkpoint dir: $CKPT" >&2; exit 1; }
for f in "${DATA_DIR}/vocab.json" "${DATA_DIR}/preprocess_meta.json" \
         "${DATA_DIR}/catalog_item_ids.npy"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

echo "checkpoint  ${CKPT}"
echo "data        ${DATA_DIR}"
echo "beam width  ${BEAM_WIDTH}, rows $([ "${BEAM_ROWS}" = "0" ] && echo all || echo "${BEAM_ROWS}"), prefix conditioning ${PREFIX}"
echo "dump        ${DUMP_ROWS:-none}"

module purge
module load Anaconda3/2023.09-0
module load cuda12.2/toolkit/12.2.2
eval "$(conda shell.bash hook)"
conda activate rqvae_env

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 ARROW_NUM_THREADS=8
export HF_HUB_CACHE="${HOME}/NZ5/train_model/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python train_tiger.py \
    --mode beam \
    --model_name Qwen/Qwen3-0.6B \
    --local_files_only \
    --data_dir "${DATA_DIR}" \
    --checkpoint "${CKPT}" \
    ${PREFIX_FLAG} \
    --beam_width "${BEAM_WIDTH}" \
    --beam_rows "${BEAM_ROWS}" \
    --eval_batch_size "${EVAL_BATCH}" \
    "${DUMP_FLAG[@]}" \
    "$@"

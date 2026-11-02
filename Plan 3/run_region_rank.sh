#!/bin/bash
#SBATCH --job-name=digr_region_rank
#SBATCH --output=region_rank_%j.out
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=mscbdtspod2026
#SBATCH --partition=normal
#
# Is DiGR's early-k loss on crowded rows a COUNTING effect or a SCORING effect?
#
#   CKPT=~/NZ5/ablation/runs/digr_abl_sidv2_featv1_e2/best \
#   DATA_DIR=~/NZ5/ablation/digr_data_a \
#   DUMP_PATH=~/NZ5/ablation/region_rank_a.npz \
#   sbatch "Plan 3/run_region_rank.sh"
#
# The dump is the point: if the answer is "counting" the analysis is finished and
# a finer SID is justified, and if it is "scoring" the same dump recomputes the
# table with `--report_only` for free. Nothing here trains anything.
set -euo pipefail

CKPT="${CKPT:-${HOME}/NZ5/ablation/runs/digr_abl_sidv2_featv1_e2/best}"
DATA_DIR="${DATA_DIR:-${HOME}/NZ5/ablation/digr_data_a}"
SPLIT="${SPLIT:-val}"
KS="${KS:-1,10,100,1000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
REGION_CHUNK="${REGION_CHUNK:-131072}"
OUT_JSON="${OUT_JSON:-${DUMP_PATH:-}}"
DUMP_PATH="${DUMP_PATH:-}"
CHECK_DUMP="${CHECK_DUMP:-}"
MAX_ROWS="${MAX_ROWS:-}"

cd "$(dirname "$0")"

module purge
module load Anaconda3/2023.09-0
module load cuda12.2/toolkit/12.2.2
eval "$(conda shell.bash hook)"
conda activate "${HOME}/.conda/envs/rqvae_env" || conda activate rqvae_env || true

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python - <<'PY' || { echo "no python with numpy + torch after activating the env" >&2; exit 1; }
import numpy, torch
print(f"numpy {numpy.__version__}, torch {torch.__version__}, cuda {torch.cuda.is_available()}")
PY

if [ ! -d "${CKPT}" ]; then
    echo "missing checkpoint dir: ${CKPT}" >&2
    echo "  that comes from a DiGR training run, e.g. \$BASE/runs/digr_abl_sidv2_featv1_e2/best" >&2
    exit 1
fi
if [ ! -f "${DATA_DIR}/vocab.json" ]; then
    echo "missing ${DATA_DIR}/vocab.json" >&2
    echo "  pass the preprocessed dir the checkpoint was TRAINED on: a v2 checkpoint on" >&2
    echo "  a v1 directory reads plausible tokens that mean something else" >&2
    exit 1
fi

ARGS=(--checkpoint "${CKPT}" --data_dir "${DATA_DIR}" --split "${SPLIT}"
      --ks "${KS}" --batch_size "${BATCH_SIZE}"
      --region_chunk "${REGION_CHUNK}")
[ -n "${OUT_JSON}"  ] && ARGS+=(--out_json "${OUT_JSON}")
[ -n "${DUMP_PATH}" ] && ARGS+=(--dump_path "${DUMP_PATH}")
[ -n "${CHECK_DUMP}" ] && ARGS+=(--check_dump "${CHECK_DUMP}")
if [ -n "${MAX_ROWS}" ]; then
    ARGS+=(--max_rows "${MAX_ROWS}")
    echo "WARNING: MAX_ROWS=${MAX_ROWS} subsamples the val rows. The numbers will not"
    echo "         be comparable to any other run; this is a smoke test only." >&2
fi

echo "=================================================================="
echo "  region multiplicity: counting versus scoring"
echo "  ckpt        : ${CKPT}"
echo "  data_dir    : ${DATA_DIR}"
echo "  split       : ${SPLIT}   ks: ${KS}"
echo "  chunk/batch : ${REGION_CHUNK} / ${BATCH_SIZE}"
echo "  dump        : ${DUMP_PATH:-<none>}"
echo "  check_dump  : ${CHECK_DUMP:-<none>}"
echo "=================================================================="

python region_rank.py "${ARGS[@]}" "$@"

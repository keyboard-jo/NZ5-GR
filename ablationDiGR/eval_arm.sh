#!/bin/bash
#SBATCH --job-name=abl_eval
#SBATCH --output=abl_eval_%j.out
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=mscbdtspod2026
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1

# Task 1 of docs/tasks.md, scoring half. Full-catalogue eval plus the per-row rank
# dump, so the arm can be placed on the matched-budget table beside DiGR v2 and
# co-click instead of only on fixed cut-offs.
#
#   ARM=a sbatch eval_arm.sh
#   ARM=b sbatch eval_arm.sh
#
# The dump is not optional here even though the script does not need it. A rank
# that was not written at eval time cannot be recovered without re-running the
# whole evaluation, and the mixed cells are exactly the arms nobody has dumped
# yet. 1.8 minutes of GPU against 2 GPU-hours of training to avoid a stupid rerun.
set -u

ARM="${ARM:-a}"
BASE="${BASE:-${HOME}/NZ5/ablation}"
CODE_DIR="${CODE_DIR:-}"
CKPT="${CKPT:-final}"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ -z "${CODE_DIR}" ]; then
    for d in "${HOME}/NZ5/plan2" "${HOME}/NZ5/ablation" "${HERE}"; do
        if [ -f "${d}/eval_full_catalog.py" ]; then CODE_DIR="${d}"; break; fi
    done
fi
[ -n "${CODE_DIR}" ] && [ -f "${CODE_DIR}/eval_full_catalog.py" ] || {
    echo "eval_full_catalog.py not found. Set CODE_DIR to the directory holding it." >&2
    exit 1
}

case "${ARM}" in
    a) RUN_NAME=digr_abl_sidv2_featv1_e2 ;;
    b) RUN_NAME=digr_abl_sidv1_featv2_e2 ;;
    *) echo "ARM must be a or b, got '${ARM}'" >&2; exit 1 ;;
esac

DATA_DIR="${BASE}/digr_data_${ARM}"
RUN_DIR="${BASE}/runs/${RUN_NAME}"
OUT_JSON="${RUN_DIR}/full_catalog_metrics_${CKPT}.json"
DUMP="${BASE}/dumps/${RUN_NAME}.npz"

[ -d "${RUN_DIR}/${CKPT}" ] || {
    echo "missing checkpoint dir: ${RUN_DIR}/${CKPT}" >&2
    echo "run: ARM=${ARM} sbatch train_arm.sh" >&2
    exit 1
}

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

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HUB_CACHE="${HOME}/NZ5/train_model/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

mkdir -p "${BASE}/dumps"

echo "=============================================================="
echo "arm         : ${ARM}"
echo "checkpoint  : ${RUN_DIR}/${CKPT}"
echo "data         : ${DATA_DIR}"
echo "ranks dump   : ${DUMP}"
echo "metrics      : ${OUT_JSON}"
echo "=============================================================="

# k=5 is carried so the short cut-offs sit beside the archive rows that recorded
# it. k=10000 is what the v1 cell was measured at and is the only cut-off wide
# enough to compare against DiGR v1's published 2.5940%.
python "${CODE_DIR}/eval_full_catalog.py" \
    --checkpoint "${RUN_DIR}/${CKPT}" \
    --data_dir "${DATA_DIR}" \
    --split val \
    --ks "1,5,10,100,1000,10000" \
    --batch_size 128 \
    --local_files_only \
    --out_json "${OUT_JSON}" \
    --dump_ranks "${DUMP}" \
    "$@"

status=$?
[ ${status} -eq 0 ] || exit ${status}

python - "${OUT_JSON}" "${DUMP}" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print("metrics:")
for k in ("recall@1", "recall@5", "recall@10", "recall@100", "recall@1000",
          "recall@10000", "ndcg@10", "mrr"):
    if k in m:
        print(f"  {k:12s} {m[k]:.4%}" if "ndcg" not in k and "mrr" not in k
              else f"  {k:12s} {m[k]:.5f}")
import numpy as np
d = np.load(sys.argv[2])
r = d["rank"]
print(f"dump: {r.size:,} rows, max rank {r.max():,}, "
      f"misses (>=10001) {(r > 10000).sum():,}")
print("expected: val 98,304 rows and recall@10 near 3.66% if the arm is behaving")
PY

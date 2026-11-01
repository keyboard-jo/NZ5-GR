#!/bin/bash
#SBATCH --job-name=abl_submit
#SBATCH --output=abl_submit_%j.out
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --account=mscbdtspod2026
#SBATCH --partition=cpu

# Submit the whole task-1 ablation as a dependency chain.
#
#   sbatch ablationDiGR/run_all.sh
#   bash   ablationDiGR/run_all.sh          # also fine, same thing
#
# Wave 1  prep a and prep b, in parallel                   ~15-25 min each
# Wave 2  train a then train b, each after its own prep     ~2.1 h and ~3.1 h
# Wave 3  eval a and eval b, each after its own training     ~1.8 min each
#
# This job submits six others and exits, so it wants no resources of its own: one
# cpu and ten minutes is plenty. Read its .out file for the six job ids.
#
# Nothing here is idempotent in the slurm sense: the prep scripts pass --overwrite,
# so re-running wave 1 replaces the data dirs and invalidates the checkpoints. If a
# prep needs a second attempt, delete the dependent run or resubmit from prep on.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

PREP="${HERE}/prep_arm.sh"
TRAIN="${HERE}/train_arm.sh"
EVAL="${HERE}/eval_arm.sh"

for f in "${PREP}" "${TRAIN}" "${EVAL}"; do
    [ -f "${f}" ] || { echo "missing ${f}" >&2; exit 1; }
done

# Refuse to submit against inputs that are missing or of the wrong vintage. A v2
# directory where v1 belongs passes every existence test and trains the wrong arm.
if [ -f "${HERE}/check_inputs.sh" ]; then
    if ! bash "${HERE}/check_inputs.sh"; then
        echo >&2
        echo "inputs are not ready, nothing submitted" >&2
        exit 1
    fi
    echo
fi

STAMP="$(date +%Y-%m-%dT%H:%M)"
echo "submitting task 1 ablation, ${STAMP}"

PA=$(sbatch --parsable --export=ALL,ARM=a "${PREP}")
PB=$(sbatch --parsable --export=ALL,ARM=b "${PREP}")
echo "  prep  a ${PA}   b ${PB}"

TA=$(sbatch --parsable --dependency=afterok:"${PA}" --export=ALL,ARM=a "${TRAIN}")
TB=$(sbatch --parsable --dependency=afterok:"${PB}" --export=ALL,ARM=b "${TRAIN}")
echo "  train a ${TA}   b ${TB}"

EA=$(sbatch --parsable --dependency=afterok:"${TA}" --export=ALL,ARM=a "${EVAL}")
EB=$(sbatch --parsable --dependency=afterok:"${TB}" --export=ALL,ARM=b "${EVAL}")
echo "  eval  a ${EA}   b ${EB}"

cat <<EOF

Watch:
  squeue -u $(whoami)
  tail -f abl_prep_${PA}.out   abl_train_${TA}.out   abl_eval_${EA}.out

Both arms have to land on the same rows as every other arm, so check the prep
banner prints 1,719,430 train rows and 98,304 val rows for each. A different row
count means an axis other than the intended one moved, and the arm is not part of
the 2x2 anymore.

When both evals finish, the 2x2 reads:

  SID          features   DiGR R@10
  4x256+731    v1         2.5940%   (measured, archive)
  2x8192       v2         3.6570%   (measured, plan2)
  2x8192       v1         arm a     -> closes the codebook question
  4x256+731    v2         arm b     -> closes the features question

Put the two new numbers in docs/results.md and replace the two "to run" rows of
the 2x2 in docs/tasks.md section 1. The mixed cells also feed the matched-budget
table, with one join per arm because the SID differs:

  python "Model Training/budget_match.py" \\
      --beam <beam dump> --arm DiGR=\${BASE}/dumps/digr_abl_sidv2_featv1_e2.npz \\
      --data_dir \${BASE}/digr_data_a --out_dir \${BASE}/dumps/join_a

Each arm has its own SID, so the join is one invocation per arm, with that arm's
--data_dir and that arm's dumps. The sid_sizes guard refuses a cross-arm join
rather than quietly counting buckets from the wrong catalogue.
EOF

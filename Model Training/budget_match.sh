#!/bin/bash
#SBATCH --job-name=digr_budget_match
#SBATCH --output=budget_match_%j.out
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --account=mscbdtspod2026
#SBATCH --partition=cpu

# The matched-budget table, built from whatever dumps are sitting in `./dump`.
#
# Every arm is scored at the number of items the TIGER beam named on that row, so the
# comparison stops being `beam 20 vs top 30` and becomes `both arms were allowed this
# many items`. No model, no GPU, no torch: this only reads npz files and counts, which
# is why it wants the cpu partition. Seconds of work, so the time limit is there to
# stop a bad path from holding a node.
#
#   sbatch budget_match.sh
#   WINDOW=h10 sbatch budget_match.sh             # co-click at a different window
#   DUMP_DIR=../dumps sbatch budget_match.sh      # dumps live elsewhere
#   STRICT=1 sbatch budget_match.sh               # a drift warning becomes a failure
#   sbatch budget_match.sh --seen_items clicked.npy --split test
#   bash budget_match.sh                          # also fine on a login node
#
# Anything after the script name goes straight to budget_match.py.
#
# What it expects in `dump/` (names are for the reader only, the script finds files by
# content):
#   *.npz with a `tuples` array                      the beam dump, required
#   *.npz with `kind` or no `tuples`                 DiGR's per-row ranks
#   baselines/<window>/<baseline>/*.npz              co-click and popularity chunks
#
# Exit code 1 means nothing was written. That is deliberate: every guard in
# budget_match.py refuses rather than warns, because a table built on misaligned rows
# looks exactly like a table built on real ones.

# Run from the script's own dir so `dump/` and `budget_match.py` resolve the way they
# do when it is run by hand. `sbatch budget_match.sh` and `bash budget_match.sh` both
# end up here.
cd "$(dirname "$0")" || exit 1

module purge
module load Anaconda3/2023.09-0

eval "$(conda shell.bash hook)"
conda activate "${HOME}/.conda/envs/rqvae_env" 2>/dev/null \
    || conda activate rqvae_env 2>/dev/null \
    || true

python -c "import numpy" >/dev/null 2>&1 || {
    echo "no python with numpy after activating the env" >&2; exit 1; }

DUMP_DIR="${DUMP_DIR:-dump}"
DATA_DIR="${DATA_DIR:-${HOME}/NZ5/plan2/digr_data}"
OUT_DIR="${OUT_DIR:-${DUMP_DIR}}"
WINDOW="${WINDOW:-h20}"
TAG="${TAG:-}"

[ -d "$DUMP_DIR" ] || { echo "missing dump dir: $DUMP_DIR" >&2; exit 1; }

# Print the first npz in a directory whose keys include (or exclude) a name. Done in
# python because the beam dump and the rank dumps are told apart by their contents,
# not by a filename, and a wrong pick here costs 47 minutes of GPU to redo.
pick_npz() {  # <dir> <key> [absent]
    python - "$1" "$2" "${3:-}" <<'PY'
import glob, os, sys
import numpy as np
folder, key, absent = sys.argv[1], sys.argv[2], sys.argv[3]
for path in sorted(glob.glob(os.path.join(folder, "*.npz"))):
    try:
        with np.load(path) as z:
            has = key in z
    except Exception:
        continue
    if has != bool(absent):
        print(path)
        break
PY
}

BEAM="${BEAM:-$(pick_npz "$DUMP_DIR" tuples)}"
if [ -z "$BEAM" ]; then
    echo "no beam dump in ${DUMP_DIR}/ (nothing with a 'tuples' array). Contents:" >&2
    ls -1 "$DUMP_DIR" | sed 's/^/  /' >&2
    echo "That file comes from: DUMP_ROWS=... sbatch eval_tiger_beam.sh" >&2
    exit 1
fi

ARMS=()
if [ -n "${DIGR:-}" ]; then
    ARMS+=("DiGR=${DIGR}")
else
    found="$(pick_npz "$DUMP_DIR" tuples absent)"
    [ -n "$found" ] && ARMS+=("DiGR=${found}")
fi
for kind in coclick popularity; do
    dir="${DUMP_DIR}/baselines/${WINDOW}/${kind}"
    [ -d "$dir" ] && ARMS+=("${kind}_${WINDOW}=${dir}")
done
[ ${#ARMS[@]} -gt 0 ] || echo "  ! no rank dumps found, so only TIGER's own reach is reported" >&2

echo "dump dir    ${DUMP_DIR}"
echo "beam        ${BEAM}"
echo "data dir    ${DATA_DIR}   (SID the beam's tuples are counted on)"
echo "arms        ${ARMS[*]:-none}"
echo "window      ${WINDOW}"
echo

args=(--beam "$BEAM" --data_dir "$DATA_DIR" --out_dir "$OUT_DIR")
[ -n "$TAG" ] && args+=(--tag "$TAG")
[ "${STRICT:-0}" = "1" ] && args+=(--strict)
for arm in "${ARMS[@]}"; do
    args+=(--arm "$arm")
done

python budget_match.py "${args[@]}" "$@"

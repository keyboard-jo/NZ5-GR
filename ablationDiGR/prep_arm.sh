#!/bin/bash
#SBATCH --job-name=abl_prep
#SBATCH --output=abl_prep_%j.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=mscbdtspod2026
#SBATCH --partition=cpu

# Task 1 of docs/tasks.md, preprocessing half. Builds the two mixed cells of the
# 2x2 whose corners are already measured:
#
#   ARM=a   the v2 SID (2 x 8,192) with the v1 event layout
#           no action token, 64 gap bins            -> $BASE/digr_data_a
#   ARM=b   the v1 SID (4x256+731) with the v2 event layout
#           action token, 128 gap bins              -> $BASE/digr_data_b
#
#   ARM=a sbatch prep_arm.sh
#   ARM=b sbatch prep_arm.sh
#
# Everything that must not move is fixed here, because a second axis silently
# appearing is exactly what this task exists to avoid: the same sequence file,
# the same user feature file, --anchors_per_record 2, --val_ratio 0.1, and
# --expect_val_split against the frozen v1 archive, which makes the script verify
# the same val users and the same val targets before it writes anything. A row
# count that differs from 1,719,430 train / 98,304 val means something else moved
# and the arm is not comparable. No --clicks_only, no --add_clock, no --max_rows.
#
# The only intended differences between the two arms are the catalogue and the
# two feature flags. That is what the shell prints in its banner, so read it.
set -u

ARM="${ARM:-a}"
BASE="${BASE:-${HOME}/NZ5/ablation}"
CODE_DIR="${CODE_DIR:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# preprocess_digr_v2.py lives wherever the Plan 2 copies were staged. Check the
# usual spots rather than assuming one, and say what to copy when it is absent.
if [ -z "${CODE_DIR}" ]; then
    for d in "${HOME}/NZ5/plan2" "${HOME}/NZ5/ablation" "${HERE}"; do
        if [ -f "${d}/preprocess_digr_v2.py" ]; then CODE_DIR="${d}"; break; fi
    done
fi
[ -n "${CODE_DIR}" ] && [ -f "${CODE_DIR}/preprocess_digr_v2.py" ] || {
    echo "preprocess_digr_v2.py not found. Set CODE_DIR to the directory holding" >&2
    echo "preprocess_digr_v2.py, train_digr.py and eval_full_catalog.py." >&2
    exit 1
}

SEQ="${SEQ:-${HOME}/NZ5/make_data/data/seq.parquet}"
FEAT="${FEAT:-${HOME}/NZ5/make_data/data/user_feat.parquet}"
V2_SID_JSON="${V2_SID_JSON:-${HOME}/NZ5/train_rq_vae/archive/2026-10-27-02/test_output/index_rqvae_3x1024.json}"
# The v1 SID comes from 2026-10-16-03, which is the directory the archived v1
# digr_data was actually generated from: its log prints
#   SID=[256, 256, 256, 256, 731] @ [2, 258, 514, 770, 1026]
# There is a second index_rqvae_prod.json under 2026-10-14-2 with the same name and
# a wider last level (code 913), i.e. an earlier run. Same filename, different
# codebook, which is why check_inputs.sh compares the json against the archive's
# own catalog_sid_codes.npy instead of trusting either name.
V1_SID_JSON="${V1_SID_JSON:-${HOME}/NZ5/train_rq_vae/archive/2026-10-16-03/test_output/index_rqvae_prod.json}"
V1_ARCHIVE="${V1_ARCHIVE:-${HOME}/NZ5/train_model/archive/2026-10-23-preprocess-01/digr_data}"

case "${ARM}" in
    a)
        CATALOG="${V2_SID_JSON}"
        NUM_LEVELS=2
        EXPECT_SID="[8192, 8192]"
        FEAT_FLAGS=(--no-add_action --n_temporal_bins 64)
        OUT="${BASE}/digr_data_a"
        NEED=("${V2_SID_JSON}")
        ;;
    b)
        CATALOG="${V1_SID_JSON}"
        NUM_LEVELS=5
        EXPECT_SID="[256, 256, 256, 256, 731]"
        FEAT_FLAGS=(--add_action --n_temporal_bins 128)
        OUT="${BASE}/digr_data_b"
        NEED=("${V1_SID_JSON}")
        ;;
    *)  echo "ARM must be a or b, got '${ARM}'" >&2; exit 1 ;;
esac

# The v1 archive is also used to define the val split this arm has to reproduce,
# so both arms are scored on the same 98,304 rows.
for f in "${SEQ}" "${FEAT}" "${NEED[@]}" \
         "${V1_ARCHIVE}/val/users.npy" "${V1_ARCHIVE}/val/targets.npy"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

mkdir -p "${BASE}"

module purge
module load Anaconda3/2023.09-0

eval "$(conda shell.bash hook)"
conda activate "${HOME}/.conda/envs/rqvae_env" 2>/dev/null \
    || conda activate rqvae_env 2>/dev/null \
    || true

# The preprocessor needs numpy and pyarrow only: no torch, no GPU. Say which one is
# missing rather than letting the traceback surface forty lines into the run.
python -c "import numpy, pyarrow" >/dev/null 2>&1 || {
    echo "no python with numpy and pyarrow after activating the env" >&2
    echo "preprocess_digr_v2.py needs exactly those two and nothing else" >&2
    exit 1; }

export OMP_NUM_THREADS=16 ARROW_NUM_THREADS=16

echo "=============================================================="
echo "arm             : ${ARM}"
echo "catalogue       : ${CATALOG}"
echo "sid levels kept : ${NUM_LEVELS}"
echo "expected sizes  : ${EXPECT_SID}"
echo "feature flags   : ${FEAT_FLAGS[*]}"
echo "output          : ${OUT}"
echo "val split check : ${V1_ARCHIVE}"
echo "code            : ${CODE_DIR}"
echo "=============================================================="

python "${CODE_DIR}/preprocess_digr_v2.py" \
    --sequence_path "${SEQ}" \
    --catalog_path "${CATALOG}" \
    --user_feat_path "${FEAT}" \
    --num_sid_levels "${NUM_LEVELS}" \
    "${FEAT_FLAGS[@]}" \
    --anchors_per_record 2 \
    --val_ratio 0.1 \
    --expect_val_split "${V1_ARCHIVE}" \
    --output_dir "${OUT}" \
    --overwrite \
    "$@"

status=$?
if [ ${status} -ne 0 ]; then
    echo "preprocessing failed with ${status}" >&2
    exit ${status}
fi

# The numbers that say the arm is comparable, and one that says it is the right
# arm at all. The vocab is derived as max(code)+1 per level, so a vintage whose
# top code happens never to appear would come back a size short and shift every
# offset above it. That would train fine and be the wrong experiment, so it is a
# hard failure here rather than a surprising number later.
python - "${OUT}" "${EXPECT_SID}" <<'PY'
import json, sys
d, expect = sys.argv[1], [int(t) for t in sys.argv[2].strip("[]").split(",")]
meta = json.load(open(f"{d}/preprocess_meta.json"))
vocab = json.load(open(f"{d}/vocab.json"))
for split in ("train", "val"):
    s = meta["split_stats"][split]
    print(f"{split:5s} rows {s['rows_kept']:>9,}  tokens {s['tokens']:>12,}  "
          f"anchors/record {s['anchors_per_record']:.3f}")
print(f"sid_sizes       {vocab['sid_sizes']}")
print(f"n_temporal      {vocab['n_temporal']}   n_action {vocab['n_action']}   "
      f"event_width {vocab['event_width']}   vocab {vocab['total_size']:,}")
print(f"collisions      {meta.get('catalog_collisions')}")
fail = []
if list(vocab["sid_sizes"]) != expect:
    fail.append(f"sid_sizes {vocab['sid_sizes']} != {expect}")
for split, want in (("train", 1719430), ("val", 98304)):
    got = meta["split_stats"][split]["rows_kept"]
    if got != want:
        fail.append(f"{split} rows {got:,} != {want:,}")
if fail:
    print()
    for f in fail:
        print(f"MISMATCH {f}", file=sys.stderr)
    print("this arm is not the arm the ablation claims; do not train on it",
          file=sys.stderr)
    sys.exit(1)
print("matches        sid_sizes and both row counts, so only the intended axes moved")
PY

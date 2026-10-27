#!/bin/bash
#SBATCH --job-name=digr_prep_v2
#SBATCH --output=logs/digr_prep_v2_%j.out
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=mscbdtspod2026
#SBATCH --partition=cpu

# SID v2 preprocessing: 2 x 8,192 codes, an action token per event, 128 gap bins.
# Writes a new digr_data directory; the frozen one is never touched.
#
#   sbatch "Plan 2/slurm/preprocess_digr_v2.sh"
#
# The cpu partition does not mount /scratch, so everything comes from $HOME.
# The 2 x 8,192 index has to be reachable from here: copy
# index_rqvae_2x8192.json next to the rest of the RQ-VAE run, or point
# SID_JSON at it.
cd "$(dirname "$0")/../.." || exit 1

module purge
module load Anaconda3/2023.09-0

eval "$(conda shell.bash hook)"
conda activate "${HOME}/.conda/envs/rqvae_env"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export ARROW_NUM_THREADS=16

mkdir -p logs

SEQ="${SEQ:-${HOME}/NZ5/make_data/data/seq.parquet}"
USERS="${USERS:-${HOME}/NZ5/make_data/data/user_feat.parquet}"
SID_JSON="${SID_JSON:-${HOME}/NZ5/train_rq_vae/archive/2026-10-27-02/test_output/index_rqvae_2x8192.json}"
OUT_DIR="${OUT_DIR:-${HOME}/NZ5/train_model/archive/sid_v2_2x8192/digr_data}"

mkdir -p "${OUT_DIR}"
echo "seq     ${SEQ}"
echo "users   ${USERS}"
echo "sid     ${SID_JSON}"
echo "out     ${OUT_DIR}"

# --max_rows is deliberately not set: the val split has to come out at the same
# 98,304 rows as the frozen build, or DiGR's 2.5940% R@10 is not comparable to
# whatever this run produces. Check that number in preprocess_meta.json before
# spending GPU hours on S4.
#
# --anchors_per_record 2 is what the cluster copy of the frozen build used
# (`docs/data.md` section 4); the default of 3 is the local snapshot's recipe and
# would inflate the row count by about 37%. A v2 stream that reads 221.5M tokens
# against today's 221.5M is the point of S3, so the anchors have to match.
python "Plan 2/preprocess_digr_v2.py" \
    --sequence_path "${SEQ}" \
    --catalog_path "${SID_JSON}" \
    --user_feat_path "${USERS}" \
    --user_feat_cols 103 104 105 106 107 108 109 110 \
    --num_sid_levels 2 \
    --add_action \
    --anchors_per_record 2 \
    --n_temporal_bins 128 \
    --val_ratio 0.02 \
    --output_dir "${OUT_DIR}" \
    --overwrite

# The clock token is a separate arm, not part of the default build, because it
# costs a fifth token in every event (187.5M train tokens against 153.5M). Run
# it only when the default build has an answer:
#     CLOCK=1 sbatch "Plan 2/slurm/preprocess_digr_v2.sh"
if [ -n "${CLOCK}" ]; then
  python "Plan 2/preprocess_digr_v2.py" \
      --sequence_path "${SEQ}" \
      --catalog_path "${SID_JSON}" \
      --user_feat_path "${USERS}" \
      --user_feat_cols 103 104 105 106 107 108 109 110 \
      --num_sid_levels 2 \
      --add_action \
      --add_clock \
      --anchors_per_record 2 \
      --n_temporal_bins 128 \
      --val_ratio 0.02 \
      --output_dir "${OUT_DIR}_clock" \
      --overwrite
fi

echo "done, now check ${OUT_DIR}/preprocess_meta.json for splits.val == 98304"

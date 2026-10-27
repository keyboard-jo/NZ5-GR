#!/bin/bash
#SBATCH --job-name=digr_prep_v2
#SBATCH --output=prep_v2_%j.out
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=mscbdtspod2026
#SBATCH --partition=cpu

# S3: rebuild the event stream on the 2 x 8,192 SID.
#
# This file is one command. Everything else is a flag on the python script, so
# extra arms need no edit here:
#
#   sbatch "Plan 2/preprocess_digr_v2.sh"             # default build
#   sbatch "Plan 2/preprocess_digr_v2.sh" --add_clock # clock arm
#
# --expect_val_split is mandatory: the split is a per-user hash, so a wrong
# --val_ratio gives a valid but different val set and nothing would notice. The
# python refuses to start without it or --allow_new_split.
#
# --anchors_per_record 2 matches the cluster copy of the frozen build, and
# --val_ratio 0.1 matches the archived split; the python defaults are the local
# snapshot's recipe and would give a different row count and a 19.6k-row val set.
# The 2 x 8,192 index is called index_rqvae_3x1024.json because that run reused
# the previous script's --output_path. The name is wrong, the content is two
# codes of 8,192.
cd "$(dirname "$0")" || exit 1

# Fail in seconds, not after conda and the catalogue load.
for f in ~/NZ5/make_data/data/seq.parquet \
         ~/NZ5/make_data/data/user_feat.parquet \
         ~/NZ5/train_rq_vae/archive/2026-10-27-02/test_output/index_rqvae_3x1024.json \
         ~/NZ5/train_model/archive/2026-10-23-preprocess-01/digr_data/val/users.npy; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

module purge
module load Anaconda3/2023.09-0
eval "$(conda shell.bash hook)"
conda activate rqvae_env
export OMP_NUM_THREADS=16 ARROW_NUM_THREADS=16

python preprocess_digr_v2.py \
    --sequence_path ~/NZ5/make_data/data/seq.parquet \
    --catalog_path ~/NZ5/train_rq_vae/archive/2026-10-27-02/test_output/index_rqvae_3x1024.json \
    --user_feat_path ~/NZ5/make_data/data/user_feat.parquet \
    --num_sid_levels 2 \
    --anchors_per_record 2 \
    --val_ratio 0.1 \
    --expect_val_split ~/NZ5/train_model/archive/2026-10-23-preprocess-01/digr_data \
    --output_dir ~/NZ5/train_model/archive/sid_v2_2x8192/digr_data \
    --overwrite \
    "$@"

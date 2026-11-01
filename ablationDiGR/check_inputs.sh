#!/bin/bash
#SBATCH --job-name=abl_check_inputs
#SBATCH --output=abl_check_inputs_%j.out
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --account=mscbdtspod2026
#SBATCH --partition=cpu

# Check every input the ablation needs, before spending a slot on it.
#
#   sbatch ablationDiGR/check_inputs.sh
#   bash   ablationDiGR/check_inputs.sh          # also fine on a login node
#
# Prints each path, whether it exists, its size, and a fingerprint against the
# value the run expects. Read-only, a few seconds of work, which is why it gets a
# cpu node and a short limit. Exit code 1 means at least one input is missing or
# is the wrong vintage, and run_all.sh will not submit anything.
#
# Every path below is an env var, so a moved file needs no edit:
#
#   SEQ=/new/seq.parquet sbatch ablationDiGR/check_inputs.sh
#   V1_SID_JSON=/new/index.json sbatch ablationDiGR/check_inputs.sh
set -u

# The fingerprints need numpy and pyarrow. A batch node does not inherit the login
# shell, so activate explicitly. Both blocks are guarded so the script also runs on
# a workstation that has no `module` command at all.
if command -v module >/dev/null 2>&1; then
    module purge
    module load Anaconda3/2023.09-0 2>/dev/null || true
fi

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${HOME}/.conda/envs/rqvae_env" 2>/dev/null \
        || conda activate rqvae_env 2>/dev/null \
        || true
fi

BASE="${BASE:-${HOME}/NZ5/ablation}"
SEQ="${SEQ:-${HOME}/NZ5/make_data/data/seq.parquet}"
FEAT="${FEAT:-${HOME}/NZ5/make_data/data/user_feat.parquet}"
V2_SID_JSON="${V2_SID_JSON:-${HOME}/NZ5/train_rq_vae/archive/2026-10-27-02/test_output/index_rqvae_3x1024.json}"
V1_SID_JSON="${V1_SID_JSON:-${HOME}/NZ5/train_rq_vae/archive/2026-10-16-03/test_output/index_rqvae_prod.json}"
V1_ARCHIVE="${V1_ARCHIVE:-${HOME}/NZ5/train_model/archive/2026-10-23-preprocess-01/digr_data}"

fail=0
say() { printf '%-14s %-8s %s\n' "$1" "$2" "$3"; }

echo "BASE for outputs: ${BASE}"
echo

echo "== 1. the event stream, both arms =="
if [ -f "${SEQ}" ]; then
    say seq "$(du -h "${SEQ}" | cut -f1)" "${SEQ}"
else
    say seq MISSING "${SEQ}"; fail=1
fi
echo

echo "== 2. the static user features, both arms =="
if [ -f "${FEAT}" ]; then
    say user_feat "$(du -h "${FEAT}" | cut -f1)" "${FEAT}"
else
    say user_feat MISSING "${FEAT}"; fail=1
fi
echo

echo "== 3. the two SID catalogues, one per arm =="
if [ -f "${V2_SID_JSON}" ]; then
    say sid_v2_arm_a "$(du -h "${V2_SID_JSON}" | cut -f1)" "${V2_SID_JSON}"
else
    say sid_v2_arm_a MISSING "${V2_SID_JSON}"; fail=1
fi
if [ -f "${V1_SID_JSON}" ]; then
    say sid_v1_arm_b "$(du -h "${V1_SID_JSON}" | cut -f1)" "${V1_SID_JSON}"
else
    say sid_v1_arm_b MISSING "${V1_SID_JSON}"; fail=1
fi
echo

echo "== 4. the v1 archive: it defines the val split both arms must reproduce =="
for f in val/users.npy val/targets.npy; do
    if [ -f "${V1_ARCHIVE}/${f}" ]; then
        say v1 "$f" "$(du -h "${V1_ARCHIVE}/${f}" | cut -f1)"
    else
        say v1 MISSING "${V1_ARCHIVE}/${f}"; fail=1
    fi
done
echo

if command -v python >/dev/null 2>&1; then
    echo "== fingerprints, beside what the run expects =="
    python - "${SEQ}" "${FEAT}" "${V2_SID_JSON}" "${V1_SID_JSON}" "${V1_ARCHIVE}" <<'PY' || fail=1
import json, os, sys

seq, feat, v2, v1sid, v1arch = sys.argv[1:6]
bad = []

CATALOG_SIZE = 4783154
WANT = {"v2 (arm a)": [8192, 8192], "v1 (arm b)": [256, 256, 256, 256, 731]}


def ok(label, detail):
    print(f"   OK       {label}: {detail}")


def mismatch(label, detail):
    print(f"   MISMATCH {label}: {detail}")
    bad.append(label)


def broken(label, err):
    print(f"   BROKEN   {label}: {err}")
    bad.append(label)


try:
    import numpy as np
except Exception as e:                                    # noqa: BLE001
    print(f"numpy not available ({e}); cannot fingerprint the arrays")
    sys.exit(1)

try:
    import pyarrow.parquet as pq
    have_pa = True
except Exception as e:                                    # noqa: BLE001
    have_pa = False
    print(f"pyarrow not available ({e}); skipping the two parquet files")
    bad.append("pyarrow")


def parquet(label, path):
    if not os.path.exists(path):
        print(f"   MISSING  {label}: {path}")
        bad.append(label)
        return
    if not have_pa:
        return
    try:
        pf = pq.ParquetFile(path)
    except Exception as e:                                # noqa: BLE001
        broken(label, f"not a readable parquet ({e})")
        return
    ok(label, f"{pf.metadata.num_rows:,} rows, {pf.metadata.num_row_groups} row "
              f"groups, columns {pf.schema_arrow.names}")


def sid_json(label, path, want_sizes):
    """The sizes matter more than the name: a 2-level file where a 5-level one
    belongs produces a valid, differently tokenised, wrong experiment."""
    if not os.path.exists(path):
        print(f"   MISSING  {label}: {path}")
        bad.append(label)
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        sizes = [max(int(v[l]) for v in d.values()) + 1
                 for l in range(max(len(v) for v in d.values()))]
    except Exception as e:                                # noqa: BLE001
        broken(label, e)
        return None
    detail = f"{len(d):,} items, {len(sizes)} codes each, max code + 1 per level {sizes}"
    if len(d) != CATALOG_SIZE or sizes != want_sizes:
        mismatch(label, detail + f" (expected {CATALOG_SIZE:,} items and {want_sizes})")
    else:
        ok(label, detail)
    return d


parquet("seq.parquet", seq)
parquet("user_feat.parquet", feat)

d2 = sid_json("v2 SID (arm a)", v2, WANT["v2 (arm a)"])
d1 = sid_json("v1 SID (arm b)", v1sid, WANT["v1 (arm b)"])

# The val split reference, and a cross-check that arm b's json really is the SID
# the archived v1 checkpoints were trained against. The .npy pair is optional:
# only if it is there can the two be compared.
# Object dtype here is normal: the archived user ids can be python strings, and
# numpy refuses to memory-map a python-object array. Load it, do not mmap it.
try:
    u = np.load(os.path.join(v1arch, "val", "users.npy"), allow_pickle=True)
    # One entry per val ROW, aligned with val/targets.npy, not one per user: the v1
    # val split is 98,304 records with ~1 anchor each, so the two arrays line up.
    if u.shape != (98304,):
        mismatch("v1 val/users.npy",
                 f"{u.shape} {u.dtype} (expected (98304,), one per val row)")
    else:
        ok("v1 val/users.npy",
           f"{u.shape} {u.dtype}, one user id per val row, "
           f"{np.unique(u).size:,} distinct users")
except Exception as e:                                    # noqa: BLE001
    broken("v1 val/users.npy", e)

try:
    t = np.load(os.path.join(v1arch, "val", "targets.npy"), mmap_mode="r")
    if t.shape != (98304,):
        mismatch("v1 val/targets.npy", f"{t.shape} (expected (98304,))")
    else:
        ok("v1 val/targets.npy", f"{t.shape}, catalogue row per val row")
except Exception as e:                                    # noqa: BLE001
    broken("v1 val/targets.npy", e)

# The archived v1 arrays, i.e. what the archived checkpoint actually trained on.
# The json above has to agree with these, which is the check that catches a same-named
# prod json from a different directory.
codes_p = os.path.join(v1arch, "catalog_sid_codes.npy")
ids_p = os.path.join(v1arch, "catalog_item_ids.npy")
archive_codes = None
try:
    archive_codes = np.load(codes_p, mmap_mode="r")
    sizes = [int(archive_codes[:, l].max()) + 1
             for l in range(archive_codes.shape[1])]
    detail = (f"{archive_codes.shape} {archive_codes.dtype}, max code + 1 per "
              f"level {sizes}")
    want = WANT["v1 (arm b)"]
    if tuple(archive_codes.shape) != (CATALOG_SIZE, len(want)) or sizes != want:
        mismatch("v1 archive catalog_sid_codes.npy",
                 detail + f" (expected ({CATALOG_SIZE}, {len(want)}) and {want})")
    else:
        ok("v1 archive catalog_sid_codes.npy", detail)
except Exception as e:                                    # noqa: BLE001
    broken("v1 archive catalog_sid_codes.npy", e)

try:
    av = json.load(open(os.path.join(v1arch, "vocab.json")))
    detail = f"sid_sizes {av.get('sid_sizes')} total_size {av.get('total_size')}"
    if av.get("sid_sizes") != WANT["v1 (arm b)"] or av.get("total_size") != 1966:
        mismatch("v1 archive vocab.json",
                 detail + " (expected [256, 256, 256, 256, 731] and 1,966)")
    else:
        ok("v1 archive vocab.json", detail + ", the v1 vocabulary in writing")
except Exception as e:                                    # noqa: BLE001
    broken("v1 archive vocab.json", e)

try:
    ai = np.load(ids_p)
    if ai.shape != (CATALOG_SIZE,):
        mismatch("v1 archive catalog_item_ids.npy",
                 f"{ai.shape} {ai.dtype} (expected ({CATALOG_SIZE},))")
    else:
        ok("v1 archive catalog_item_ids.npy", f"{ai.shape} {ai.dtype}")
except Exception as e:                                    # noqa: BLE001
    broken("v1 archive catalog_item_ids.npy", e)

# Informational only: the prod RQ-VAE json in that archive tree is NOT the SID the
# archived checkpoint trained on. It is reported so nobody points arm b at it, and
# the comparison joins on ITEM ID because a dict has no row order worth trusting.
if os.path.exists(v1sid) and archive_codes is not None:
    try:
        with open(v1sid) as f:
            jd = json.load(f)
        j_sizes = [max(int(v[l]) for v in jd.values()) + 1
                   for l in range(max(len(v) for v in jd.values()))]
        want = WANT["v1 (arm b)"]
        print(f"   note     {os.path.basename(v1sid)} covers levels {j_sizes}, "
              f"the archive's is {want}")
        ai = np.load(ids_p, mmap_mode="r")
        rng = np.random.default_rng(0)
        idx = rng.choice(len(ai), size=min(50000, len(ai)), replace=False)
        same = absent = 0
        for i in idx:
            row = jd.get(str(int(ai[i])))
            if row is None:
                absent += 1
            elif [int(x) for x in row] == list(archive_codes[i]):
                same += 1
        differ = len(idx) - same - absent
        if absent:
            print(f"   note     {absent:,} of {len(idx):,} sampled ids are not in "
                  "that json, so it may be a partial export")
        elif differ:
            mismatch("v1 SID vs archive arrays",
                     f"{differ:,} of {len(idx):,} sampled items carry different "
                     "codes: this json is not the SID the archived v1 data was "
                     "built from")
            print("            there is more than one index_rqvae_prod.json; the "
                  "v1 one is under archive/2026-10-16-03")
        elif j_sizes != want:
            print(f"   note     the same code per item on all {len(idx):,} sampled "
                  "items, but the level sizes differ")
            print(f"            json {j_sizes} vs archive {want}: the json keeps "
                  "codebook slots the archive does not")
        else:
            ok("v1 SID vs archive arrays",
               f"{len(idx):,} sampled items identical and the sizes match, so this "
               "json is the artefact the archived v1 data came from")
    except Exception as e:                                # noqa: BLE001
        print(f"   note     could not compare {os.path.basename(v1sid)} ({e})")

# Optional, only if a second digr_data is named: the val rows must be the same rows,
# not merely the same count, or two arms are not scored on the same questions.
other = os.environ.get("V2_DATA_DIR", "")
if other and os.path.exists(os.path.join(other, "val", "users.npy")):
    try:
        a = np.load(os.path.join(v1arch, "val", "users.npy"), allow_pickle=True)
        b = np.load(os.path.join(other, "val", "users.npy"), allow_pickle=True)
        bt = np.load(os.path.join(other, "val", "targets.npy"), allow_pickle=True)
        at = np.load(os.path.join(v1arch, "val", "targets.npy"), allow_pickle=True)
        same_users = a.shape == b.shape and bool((a == b).all())
        same_targets = at.shape == bt.shape and bool((at == bt).all())
        detail = (f"{other}: users {'identical' if same_users else 'DIFFER'}, "
                  f"targets {'identical' if same_targets else 'DIFFER'}")
        if same_users and same_targets:
            ok("val rows vs another data dir", detail)
        else:
            mismatch("val rows vs another data dir",
                     detail + " (the two are not scored on the same rows)")
    except Exception as e:                                # noqa: BLE001
        broken("val rows vs another data dir", e)

print()
if bad:
    print(f"fingerprints: {len(bad)} problem(s) in {', '.join(bad)}")
    sys.exit(1)
print("fingerprints: every file matches what the run expects")
PY
else
    echo "no python on PATH, skipping fingerprints"
fi

echo
echo "== how to find a missing file =="
cat <<'EOF'
  find ~/NZ5 -maxdepth 6 -name 'seq.parquet' -o -name 'user_feat.parquet' 2>/dev/null
  ls ~/NZ5/train_rq_vae/archive/*/test_output/index_rqvae*.json
  find ~/NZ5 -maxdepth 8 -type d -name 'digr_data' 2>/dev/null

Which archive holds which SID, by codebook size:

  index_rqvae_prod.json    4x256+731  (v1)   archive/2026-10-16-03
  index_rqvae_prod.json    a wider codebook    archive/2026-10-14-2  (earlier run)
  index_rqvae_3x1024.json  2x8192     (v2)   archive/2026-10-27-02

Two files share the name index_rqvae_prod.json. The v1 digr_data was generated from
the 2026-10-16-03 one, whose own log prints SID=[256,256,256,256,731] and vocab
total=1,966. The 2026-10-14-2 one has a last level reaching code 913 and is not v1.

Override any of them per submission, without editing the scripts:

  SEQ=/new/seq.parquet          ARM=a sbatch ablationDiGR/prep_arm.sh
  V1_SID_JSON=/new/index.json   ARM=b sbatch ablationDiGR/prep_arm.sh
  V1_ARCHIVE=/new/digr_data     ARM=b sbatch ablationDiGR/prep_arm.sh
EOF

if [ ${fail} -ne 0 ]; then
    echo
    echo "RESULT: at least one input is missing or wrong, fix before submitting."
    exit 1
fi
echo
echo "RESULT: all inputs present and matching."

# Plan 3: is DiGR's early-k loss on crowded rows counting or scoring?

One question, one script, one number.

## The question

DiGR scores `s_i = (1/tau) h . V_i` with `V_i = (1/L) sum_l E_l[c_il]`, so every item
sharing a SID tuple has **exactly** the same score. 53.51% of the catalogue shares a 2-code
tuple.

Measured on `local/2026-29-10-1-full-eval/dumps/digr_val.npz`:

| bucket size | rows | share | R@1 | R@10 |
| --- | --- | --- | --- | --- |
| 1 | 36,017 | 36.64% | **2.427%** | **5.703%** |
| 2 | 17,275 | 17.57% | 0.590% | 3.757% |
| 3-4 | 16,451 | 16.73% | 0.310% | 3.210% |
| 5-8 | 12,441 | 12.66% | 0.040% | 1.953% |
| 9-16 | 7,639 | 7.77% | 0.039% | 1.217% |
| 17-315 | 8,481 | 8.63% | 0.024% | 0.330% |

9.3x at `R@1`, monotone in bucket size, and it survives popularity controls. The decisive
control is co-click, which has no codebook at all: 0.855% vs 0.777% on the same rows, a 1.1x
gap. So the gap belongs to DiGR's item representation, not to the items or the queries.

Two opposite fixes fit that fact:

- **counting.** `rank = 1 + #{i : s_i > s_t}` counts items. A tuple ranked above the target
  donates its whole size to the count, so a crowded code space inflates the rank
  arithmetically. Fix: finer tuples (deeper or wider SID). Item scoring is irrelevant.
- **scoring.** The model fails to put the target's region on top when that region is
  crowded. Fix: better item scoring (per-level weights, a non-linear composition). A finer
  codebook cannot help.

## Why it is measurable in one pass

The region rank is computable on its own. With `s_r` the score of tuple `r`,
`region_rank = 1 + #{r : s_r > s_rt}` is exactly the item rank an **injective** codebook
would produce from the same scores. So the counterfactual "what if every tuple held one
item" costs nothing extra: same scores, sizes set to 1. Both counts come out of one GEMM
pass, because they differ only in the weight.

## Run it

```bash
CKPT=~/NZ5/ablation/runs/digr_abl_sidv2_featv1_e2/best \
DATA_DIR=~/NZ5/ablation/digr_data_a \
DUMP_PATH=~/NZ5/ablation/region_rank_a.npz \
sbatch "Plan 3/run_region_rank.sh"
```

`--check_dump <a DiGR --dump_ranks npz>` adds an end-to-end check: the item rank computed here
must reproduce the dumped ranks, which proves the region sizes are aligned with the catalogue.
Add `CHECK_DUMP=~/NZ5/ablation/dumps/digr_a.npz` to the submission above if that dump exists.

Re-print the table with no GPU:

```bash
python "Plan 3/region_rank.py" --report_only ~/NZ5/ablation/region_rank_a.npz
```

## Read the output

```
    R@1     measured   1.0549%   injective ceiling   2.4270%   (2.30x, +1.37 pp)
```

`measured` is what the arm does. `injective ceiling` is the same tuple scores ranked as if
every tuple held one item, so the gap between them is **all** the counting effect and
nothing else.

- ceiling near measured (gain under ~1.15x) means **scoring**. A finer SID is wasted effort;
  the item vector is what to change.
- ceiling far above measured (gain over ~1.45x) means **counting**. A finer codebook is the
  fix and better item scoring will not find it.
- in between means both moves are worth something.

The script prints the verdict in words too, from `region_lib.verdict`.

## Cost

One pass over the distinct tuples, not the catalogue: 3.0M rows for 2x8192 against 4.78M.
A few minutes on one GPU, against 1.8 min for a full-catalogue eval. `--region_dtype bf16`
halves the tuple matrix (6.1 GB instead of 12.3 GB) but rounds near-ties between distinct
tuples, so `fp32` is the default and matches the `item_logits` path that produced the dumps.

## Files

| file | role |
| --- | --- |
| `region_lib.py` | numpy only: packing, region counts, the by-size table, the verdict |
| `region_rank.py` | torch: loads the checkpoint, builds region vectors, counts, reports |
| `test_region_lib.py` | 16 tests, brute-force cross-checks, no torch |
| `test_region_rank.py` | 8 tests on the counting and vector arithmetic |
| `run_region_rank.sh` | sbatch, one GPU |

`bash -n "Plan 3/run_region_rank.sh"` and `python -m pytest "Plan 3" -q` are the two checks
worth running before a submission.

## What this does not settle

Whether a finer SID is worth its cost. If the answer is "counting", the next question is
whether a third level earns back the extra token per event, and section 2.4 of `docs/README.md`
already puts a measured value on one: knowing the true `c1` removes 38.7% of the level-2
entropy. This script only says which lever moves.

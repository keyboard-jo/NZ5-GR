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
item" costs nothing extra: same scores, sizes set to 1. Both counts come out of one pass,
because they differ only in the weight.

**Which tie convention is used, and why it matters.** Two ranks are reported for each row and
they are not interchangeable:

- `region_rank` and `item_rank` both use the evaluator's own convention, the target's
  **position in a `topk` list**, including the `kmax + 1` miss rule. `item_rank` therefore
  reproduces a `--dump_ranks` file exactly, which is what makes `--check_dump` a hard test
  rather than an approximate one.
- `fav_item_rank` uses the generous convention, `1 + #{items scoring strictly higher}`. It is
  a diagnostic only: every member of a tuple has a bit-identical score, so the generous
  convention hands the target the first slot of its whole tie group while the evaluator
  places it by catalogue index. The gap between the two is the price of collisions in the
  metric itself, separate from the price in the loss.

Report `item_rank` beside published numbers and `fav_item_rank` only as the `1/g` bound.
Mixing them across arms is the mistake `docs/evaluation.md` section 6 documents.

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
| `tie_break.py` | numpy only, CPU: what popularity inside a block buys, and the true ceiling |
| `test_tie_break.py` | 12 tests on a 9-item catalogue checked by hand |

`bash -n "Plan 3/run_region_rank.sh"` and `python -m pytest "Plan 3" -q` are the two checks
worth running before a submission.

## The tie break, and why its ceiling is lower than it looks

`region_rank.py` prints an **injective** column: the same scores with each tuple holding one
item. That is not the ceiling of a tie break. It puts a whole tuple in a single position, so it
credits the fix for skipping every other item in a tuple ranked above the target, which no
within-block reordering can do.

`tie_break.py` writes the rank of a row as `rank = base + offset + 1`, where `base` is how many
items sit in tuples that scored strictly higher and `offset` is the target's place inside its
own block. `base` is fixed by the scores. A tie break can only drive `offset` to 0, so the real
ceiling is `base + 1`. On the v2 dump that is `R@1` 1.0549% to 1.2909% and `R@10` 3.6570% to
4.0070%, against the injective 1.3414% and 4.3742%.

```bash
python "Plan 3/tie_break.py"   --arm DiGR_v2=local/2026-29-10-1-full-eval/dumps/digr_val.npz   --data_dir local/2026-27-10-1-pre/digr_data
```

`base` comes from a dump carrying `fav_item_rank` (written by `region_rank.py`). Without it the
script assumes the tie order is ascending catalogue row, recovers `base` from the measured rank,
and prints which of the two it did. The guard is `base >= 0` on every row: an arm whose scores
are not one-value-per-tuple fails it, because reordering such an arm moves every row instead of
resolving a tie. `Plan 3/tie_break.py` refuses co-click and popularity for exactly this reason.

Numbers quoted above come from the first v2 run. The re-run on `digr_v2_sid2x8192_e3_sampled300k`
gives measured `R@1` 1.0549% and `R@10` 3.6570%, with the injective column at 1.2889% and
5.0283%. Re-run `tie_break.py` before quoting a tie-break ceiling from that dump, since the two
numbers move together.

Measured on the v2 dump, popularity inside the block takes `R@1` from 1.0549% to 1.2065% and
`R@10` from 3.6570% to 3.9286%, which is about two thirds of the ceiling at `R@1` and three
quarters at `R@10`. It also costs something: the target is the most clicked item in its block on
80.31% of rows, so 19.69% of rows trade a lucky break for a worse one.

## What this does not settle

Whether a finer SID is worth its cost. If the answer is "counting", the next question is
whether a third level earns back the extra token per event, and section 2.4 of `docs/README.md`
already puts a measured value on one: knowing the true `c1` removes 38.7% of the level-2
entropy. This script only says which lever moves.

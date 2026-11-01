# ablationDiGR

Task 1 of `docs/tasks.md`, the 2x2 that separates the new SID from the new
features. Two arms that have never been run, four jobs each way.

| SID | features | data dir | run | R@10 |
| --- | --- | --- | --- | --- |
| `4x256+731` | v1: no action token, 64 gap bins | archive v1 | measured | 2.5940% |
| `2x8192` | v2: action token, 128 gap bins | plan2 v2 | measured | 3.6570% |
| `2x8192` | v1 | **arm a** | **to run** | ? |
| `4x256+731` | v2 | **arm b** | **to run** | ? |

S4 moved three axes at once, so the 1.41x difference between the two measured
corners has no attribution. Splitting it needs the two mixed cells, and nothing
else about the recipe may change.

## Files

| file | what it does |
| --- | --- |
| `run_all.sh` | submits all six jobs with `afterok` dependencies |
| `prep_arm.sh` | `ARM=a|b` sbatch, tokenises one arm into `$BASE/digr_data_<arm>` |
| `train_arm.sh` | `ARM=a|b` sbatch, 2 epochs into `$BASE/runs/<run_name>` |
| `eval_arm.sh` | `ARM=a|b` sbatch, full-catalogue eval plus a per-row rank dump |
| `make_v1_catalog.py` | archive `.npy` pair to parquet, for cross-checking a SID by hand |
| `check_inputs.sh` | prints each input path with a size and a fingerprint against the expected value |

## Running it

Every script here is a slurm script with its own `#SBATCH` header, so every one of
them goes through `sbatch`. Nothing needs a login-node `bash` invocation.

```bash
sbatch ablationDiGR/run_all.sh
```

That submits six jobs and exits; its `.out` file lists the six job ids. Or submit
the waves yourself:

```bash
sbatch ablationDiGR/check_inputs.sh                                   # gate, ~10 s
ARM=a sbatch ablationDiGR/prep_arm.sh
ARM=b sbatch ablationDiGR/prep_arm.sh
ARM=a sbatch ablationDiGR/train_arm.sh     # after its prep succeeds
ARM=b sbatch ablationDiGR/train_arm.sh
ARM=a sbatch ablationDiGR/eval_arm.sh      # after its training succeeds
ARM=b sbatch ablationDiGR/eval_arm.sh
```

The resources each script asks for:

| script | partition | time | cpus | mem | gpus |
| --- | --- | --- | --- | --- | --- |
| `run_all.sh` | cpu | 00:10:00 | 1 | 2G | - |
| `check_inputs.sh` | cpu | 00:30:00 | 1 | 8G | - |
| `prep_arm.sh` | cpu | 12:00:00 | 16 | 64G | - |
| `train_arm.sh` | normal | 06:00:00 | 8 | default | 1 |
| `eval_arm.sh` | normal | 01:00:00 | 8 | default | 1 |

`BASE` defaults to `$HOME/NZ5/ablation` and everything is written under it
(`digr_data_a`, `digr_data_b`, `runs/`, `dumps/`). Override it per submission
with `BASE=... sbatch ...` if the scratch space is elsewhere.

`CODE_DIR` is where `preprocess_digr_v2.py`, `train_digr.py` and
`eval_full_catalog.py` live. Default search order is `$HOME/NZ5/plan2`,
`$HOME/NZ5/ablation`, then this directory, so copying these scripts next to the
Plan 2 copies is enough. Set `CODE_DIR` explicitly to skip the search.

## What is deliberately held fixed

The sequence file, the user feature file, `--anchors_per_record 2`,
`--val_ratio 0.1`, 2 epochs, batch 512, 300k uniform negatives, full label
subset, Qwen3-0.6B, and `--expect_val_split` against the frozen v1 archive.

That last one is the guard worth reading twice. The split is a per-user hash with
a ratio, so a wrong `--val_ratio` produces a valid but *different* val set and
nothing downstream notices. `--expect_val_split` makes the preprocessor check the
val users before tokenising and the val targets after, so the arm is scored on the
same 98,304 rows as every other row in the tables. Both prep runs must print
`train 1,719,430` and `val 98,304`. A different count means a second axis moved.

## Two things that are derived rather than typed

**The temperature.** The score the model sees is `1/(L*T)`, tuned at 2.857 with
`L=5` and `T=0.07`. Arm a has `L=2`, and leaving `T` at 0.07 would make the logits
2.5x sharper for reasons that have nothing to do with the data. `train_arm.sh`
reads `L` from the arm's own `vocab.json` and sets `T = 0.35/L`, which is 0.175
for arm a and 0.07 for arm b. It also refuses an arm whose `sid_sizes` are not
`[8192, 8192]` for a or `[256, 256, 256, 256, 731]` for b, because a level count
alone does not identify a codebook and pointing an arm at the wrong data dir is
otherwise invisible until the numbers look strange.

**The inputs.**

| arm | catalogue | SID |
| --- | --- | --- |
| a | `train_rq_vae/archive/2026-10-27-02/test_output/index_rqvae_3x1024.json` | `2x8192` |
| b | `train_rq_vae/archive/2026-10-16-03/test_output/index_rqvae_prod.json` | `4x256+731` |

**Two different files are called `index_rqvae_prod.json`.** The v1 digr_data was
generated from the `2026-10-16-03` one, and its own preprocessing log prints the SID
it built: `SID=[256, 256, 256, 256, 731] @ [2, 258, 514, 770, 1026]`, vocab
`total=1,966`. The `2026-10-14-2` copy has an earlier, wider codebook whose last
level reaches code 913. Pointing arm b at that one produces a valid five-level SID
with the wrong N, so `check_inputs.sh` compares whichever json you give it against
the archive's own `catalog_sid_codes.npy`, joined on `item_id`, and fails on any
disagreement.

`make_v1_catalog.py` is not in the flow. It converts the archive's `.npy` pair to a
parquet, which is useful for the same cross-check by hand or if a json ever goes
missing, but the json is the artefact and it is what arm b reads.

The v1 archive also defines the val split both arms must reproduce, so it is
required either way. A v1 directory where v2 belongs passes `test -f` happily.

`val/users.npy` is one user id per val ROW, 98,304 long, aligned with
`val/targets.npy`. It is not one entry per user: v1 gives about one anchor per val
record, so 98,304 rows come from 98,304 distinct users. A data dir built with more
anchors per record would hold fewer distinct users at the same row count, which is
why the row count alone does not prove two runs answered the same questions. To
check that directly:

```bash
V2_DATA_DIR=$HOME/NZ5/plan2/digr_data sbatch ablationDiGR/check_inputs.sh
```

It compares both arrays elementwise and says `users identical, targets identical`,
or names the difference.

**The codebook size.** The vocab is derived as `max(code) + 1` per level, so a
vintage whose top code happens to be absent would come back one size short and
shift every offset above it. That trains fine and is the wrong experiment, so
`prep_arm.sh` compares the produced `sid_sizes` against `[8192, 8192]` for a and
`[256, 256, 256, 256, 731]` for b, and exits nonzero on any difference, alongside
the two row counts.

## Cost

| wave | arm a | arm b |
| --- | --- | --- |
| prep | ~15 min, 16 cpu | ~25 min, 16 cpu (five levels, more tokens) |
| train | ~2.1 GPU-h (153M tokens) | ~3.1 GPU-h (221M tokens) |
| eval | 1.8 min GPU | 1.8 min GPU |

Roughly 5.5 GPU-h in total for both arms, plus two IO passes.

## Reading the result

The two mixed cells split the 1.41x.

- Arm a near 3.66% means the codebook is the whole story and the features are
  noise.
- Arm a near 2.59% means the codebook did nothing and the features did the work.
- In between means both, and the four cells give the split without needing an
  interaction term.

Record both numbers in `docs/results.md`, fill the two "to run" cells of the 2x2
in `docs/tasks.md` section 1, and then note that neither arm needs a second
evaluation, because the rank dump written by `eval_arm.sh` already covers every
cut-off from 1 to 10,000 and feeds the matched-budget join.

## Known ways this bites

- **A wall-clock kill mid-epoch** is resumable with `--resume` because
  `--save_steps 300` keeps mid-epoch checkpoints. Pass it through as
  `ARM=a sbatch ablationDiGR/train_arm.sh --resume`.
- **Re-running prep invalidates training.** `--overwrite` replaces the data dir
  and the rows will not match the checkpoint's expectations.
- **Each arm has its own SID**, so a matched-budget join needs one invocation per
  arm with `--data_dir` pointing at that arm's data dir. Joining arm a's dump
  against arm b's catalogue is refused rather than silently wrong, because the
  `sid_sizes` guard catches it.

# Tasks

Four things, in the order that changes what we can claim. The first is the one the
co-supervisor's review asks for and the only one that is blocking.

## 1. Ablation: the new SID versus the new features (do this first)

**Why.** S4 moved three axes at once: the codebook went from `4x256+731` to `2x8192`, the
action token was added, and the gap bins went from 64 to 128. DiGR v2 is 1.41x v1 on R@10
and we cannot say which of the three bought it. That is the gap the supervisor flagged, and
it is a real gap: the current write-up implies the codebook did it, and that is unproven.

**The 2x2.** Two cells are already measured, two need a run.

| SID | features | data dir | run | R@10 |
| --- | --- | --- | --- | --- |
| `4x256+731` | v1: no action token, 64 gap bins | archive v1 | measured | 2.5940% |
| `2x8192` | v2: action token, 128 gap bins | plan2 v2 | measured | 3.6570% |
| `2x8192` | v1 | new | **to run** | ? |
| `4x256+731` | v2 | new | **to run** | ? |

**How.** `Plan 2/preprocess_digr_v2.py` owns both switches, so each mixed arm is one
preprocessing pass plus one training run. The SID is an input (`--catalog_path` plus
`--catalog_sid_cols`), the features are flags (`--no-add_action --n_temporal_bins 64` gives
the v1 layout).

**Cost.** One preprocessing pass per new arm, then one 2-epoch run at about 2.1 GPU-h each.
Two arms, so roughly 4-5 GPU-h plus two IO passes. Everything else held fixed: 2 epochs,
same negatives, same 0.6B backbone.

**How to read the result.** The two mixed cells split the 1.41x. If `2x8192` with v1 features
is already near 3.66%, the codebook is the whole story and the features are noise. If it sits
near 2.59%, the codebook did nothing and the features did the work. A result in between means
both, and the 2x2 gives the split without needing an interaction term.

**Launch both arms with `--dump_ranks`.** These two cells are trained with the v2 SID and v1
features, which is a combination no arm has been dumped for, and `budget_match.py` only reads
ranks that were written at evaluation time. The flag costs one file per arm and a rerun
otherwise. Same rule for any further arm: dump at the first run or do the eval twice.

## 2. Diagnose the new SID, and measure whether the tie groups actually cost anything

**Why.** Already done once in `archive/sid_results.md` and `archive/sid_diagnostics.md`; redo
it against v2 and add the collision check the supervisor asked for.

**Known already, so do not re-measure from scratch.** 53.51% of items share a 2-code tuple,
the largest group is 315 items, and the tie ceiling is R@1 53.57% / R@10 93.20%. Collisions
are structural at two levels, not a defect, so the useful question is not whether they exist
but how much of the achieved R@k the ceiling already explains.

**To do.** Full-tuple and per-level collision sizes; the tie ceiling for every arm at its own
cut-off; items per level-0 code against the 39-195 the DiGR paper's own catalogues sit at;
and the unrun cells in `archive/sid_results.md` (`2x16384`, `3x16384`) if they are still
wanted.

**The tie measurement, which is the part that is still conjecture.** Currently the claim that
collisions cost nothing rests on a `1/g` model (a tie group of size `g` caps `R@k` at
`min(1, k/g)`) that has never been checked against what the model actually does. Three
checks, all CPU, all on the dumps already in `local/2026-29-10-1-full-eval/dumps` plus
`catalog_sid_codes.npy`.

1. **Do exact ties even happen.** A colliding item's score is `h . (E[c1] + E[c2]) / 2`, and
the order `topk` returns inside a group is catalogue index order only if those floats collide
exactly. If they differ in the last bits, the group is a near-tie group and the `1/g` model is
the wrong model. This check decides the shape of the other two.
2. **Rank within group.** For colliding targets, histogram `rank - group_start`. Under the
lottery model this is uniform. Skewed toward the start means ties break in the arm's favour,
skewed toward the end means it is being penalised.
3. **`R@k` split by group size, and `R@k` restricted to unique-tuple targets.** The second is
the collision-free skill number. `README.md` 2.3(d) predicts `R@1` about 2.06% from a 0.513
dilution factor; if the unique-target subset does not land near the same figure, that
arithmetic is wrong and the tie break it justifies is not worth building.

**Why this is a task and not a footnote.** `README.md` 2.3(d) proposes a content-side tie
break worth up to 2x on `R@1`, and its entire value depends on checks 1 and 2. Under a
near-tie reading, the fix is not a tie break at all but a re-ranker over roughly 11 items per
row, which is different work at a different cost.

**Cost.** Seconds to minutes, CPU. Deferred only because item 1 is what changes the claims.

## 3. TIGER on SID v1, so the method comparison stops crossing two changes at once

**Why.** DiGR has a v1 cell and a v2 cell. TIGER has only a v2 cell. Every "TIGER beats
DiGR" sentence therefore crosses the SID change and the feature change as well as the
factorisation, and the SID/feature step is measured at 1.41x on DiGR, which is larger than
the 1.11x effect being attributed to the factorisation (13.9994% against 12.5804% at the
matched budget). Until this run exists, the factorisation claim is not supported by its own
evidence.

| SID | features | DiGR | TIGER |
| --- | --- | --- | --- |
| `4x256+731` | v1: no action token, 64 gap bins | 2.5940% `R@10` | **to run** |
| `2x8192` | v2: action token, 128 gap bins | 3.6570% `R@10` | beam 192.2 items/row |

**What.** Two TIGER runs on the archived v1 data,
`$HOME/NZ5/train_model/archive/2026-10-23-preprocess-01/digr_data`, so no preprocessing and no
new SID.

(a) **Prefix-free, 5 levels, 2 epochs.** The direct analogue of the existing prefix-free arm,
comparable to DiGR v1 on the same rows and the same cut-offs.

(b) **Prefix-conditioned plus a beam pass.** The paper's setting at full depth, which has
never been run at more than two levels.

**Why (b) is the informative one.** Beam width is 20 in both vintages but the budget it buys
is set entirely by the SID. At `2x8192` the beam named 192.2 items per row on average, because
a two-code tuple holds 1.593 items and the beam concentrates on the fat cells. At
`4x256+731` the 731-wide suffix level makes a full tuple near-unique, 27.67% share a 4-code
prefix but the suffix separates them, so 20 tuples name about 20 items. Same width, roughly
ten times the list. This run is the second point on the budget curve, and it is what says
whether TIGER's advantage is a property of the factorisation or an artefact of being allowed
to name 192 items.

**Cost.** The v1 stream is 221,500,536 tokens against v2's 153,461,256, so about 4.6 GPU-h
per arm at the 3.2 GPU-h the v2 arm took, roughly 9 GPU-h for both. The head is
`1024 x 1755 = 1,797,120` parameters rather than 16,777,216, so this also compares the two
methods at a tenth of TIGER's head capacity, which is the setting DiGR's own paper uses.

**Dumps, or the comparison cannot be made at all.** `Model Training/budget_match.py` reads
per-row ranks and never re-scores anything, so a row that was not dumped cannot be added to
the table afterwards, and re-deriving one means re-running the arm. Every v1 arm therefore
has to be launched with its dump flag the first time.

1. **DiGR v1** needs `--dump_ranks`, which the archived run does not have. The archive JSON
holds the cut-offs and not the ranks, and the per-row rank is exactly what the join needs. One
`eval_full_catalog.py` pass over the v1 catalog, about 1.8 minutes of GPU, given the v2 pass
took 1.8 minutes.
2. **TIGER v1** needs `--dump_rows` on the beam pass, which is free inside the 47-minute
inference. The prefixes, ranks, decoded targets and the 20 chosen tuples all come out of the
same loop.
3. **Co-click and popularity are already done.** Their existing dumps were produced against
the v1 archive data dir, so the `h5`/`h10`/`h20` windows and the popularity order are already
native to the v1 SID and are reused as they stand. Only DiGR and TIGER need new dumps.

Run the join into its own output directory, `dumps_v1/`, so the two vintages do not overwrite
each other's `budget_match.md`. The `--data_dir` for the v1 join must be the v1 data dir, and
`budget_match.py` enforces that itself by refusing a beam dump whose `sid_sizes` disagree with
that directory's `vocab.json`, which reads `[256, 256, 256, 256, 731]` against the v2
`[8192, 8192]`.

**Why the v1 table is the cleaner one.** At `4x256+731` a full tuple is near-unique, so
`N_r` sits at about 20 for every row and the matched budget degenerates toward a fixed
`k = 20`, which is the ordinary cut-off comparison. At `2x8192` it ranged from 25 to 1,439
items per row, which is why that table needed a per-row budget in the first place. So the v1
table is the one where the two methods are compared on genuinely equal terms, and the v2 table
is the one that measures what a tuned two-level SID buys.

**Caveat to state out loud.** Five levels through `prefix_tree_index` and `beam_search` is a
code path that has never run. The two-level beam alone needed four fixes before it produced a
trustworthy number, so budget a debugging cycle and do not treat the first output as a
result.

**How to read it.** If prefix-free TIGER v1 lands near DiGR v1's 2.5940%, the factorisation
gap is small and stable across SIDs and the SID/feature step lifts both arms alike. If TIGER
v1 already beats DiGR v1 by a wide margin, the factorisation is doing real work and the v2
comparison understated it. If TIGER v1 loses to DiGR v1, the v2 win came from the SID change
and `README.md` section 2.4 needs rewriting.

## 4. Bigger backbone (deferred, last)

**Why last.** DiGR's ceiling looks like signal rather than capacity: P(top10|top1000) sits at
13-15% for every arm, including popularity at 9.12% with no query at all. TIGER is the more
plausible beneficiary because it fits conditional distributions over 8,192 outcomes per
prefix, and its conditional level-2 loss of 4.7051 against the unconditional 7.6731 says
there is structure left to fit.

**What.** 1.7B against the current 0.6B, same SID and same features, 2 epochs, both arms.
One row per arm beside the 0.6B rows, same cut-offs, same data.

**Caveat to state out loud.** This project has never run a backbone other than 0.6B, so any
claim about scaling right now is a prediction. It becomes a measurement or it does not go in
the write-up.

## Not in scope

Reranking, behavioural item representations, and ANN retrieval. Reasons are in
`archive/problems.md` section 1 and none of them changed.

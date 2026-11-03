# Tasks

Five things, in the order that changes what we can claim. Four are closed: the ablation is in
[`ablation.md`](ablation.md); the SID diagnosis and the tie measurement are in [`sid.md`](sid.md),
`README.md` §2.3 and `Plan 3/`; the TIGER-on-v1 comparison is in `README.md` §1.10; and the
region multiplicity gate is §4 below. What remains is one missing dump and the bigger
backbone.

## 1. Ablation: the new SID versus the new features — **done**

**Why.** S4 moved three axes at once: the codebook went from `4x256+731` to `2x8192`, the
action token was added, and the gap bins went from 64 to 128. DiGR v2 is 1.41x v1 on R@10
and we cannot say which of the three bought it. That is the gap the supervisor flagged, and
it is a real gap: the current write-up implies the codebook did it, and that is unproven.

**The 2x2.** All four cells are measured.

| SID | features | data dir | run | R@10 |
| --- | --- | --- | --- | --- |
| `4x256+731` | v1: no action token, 64 gap bins | archive v1 | measured | 2.5940% |
| `2x8192` | v2: action token, 128 gap bins | plan2 v2 | measured | 3.6570% |
| `2x8192` | v1 | `digr_data_a` | measured | 3.2725% |
| `4x256+731` | v2 | `digr_data_b` | measured | 2.3275% |

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

**Result, 2026-11-01.** Both mixed cells are measured and none of the three expectations above
holds. `2x8192` with v1 features lands at 3.2725%, in between, so neither "the codebook is the
whole story" nor "the features did the work" is right. The in-between conclusion is also
wrong: the 2x2 does not give a split, because the two axes interact and the feature step
reverses sign with the codebook.

| cell | SID | features | R@1 | R@10 | R@100 | R@1000 | R@10000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `v1sid_v1feat` | `4x256+731` | v1 | 0.7924% | 2.5940% | 7.5094% | 21.6003% | 49.7691% |
| `v1sid_v2feat` | `4x256+731` | v2 | 0.6958% | 2.3275% | 7.3059% | 21.5291% | 50.2177% |
| `v2sid_v1feat` | `2x8192` | v1 | 0.9705% | 3.2725% | 9.8674% | 25.5239% | 53.1972% |
| `v2sid_v2feat` | `2x8192` | v2 | 1.0549% | 3.6570% | 10.9182% | 27.6622% | 55.7058% |

Codebook 1.2616x at v1 features and 1.5712x at v2 features. Features 0.8973x at the v1
codebook and 1.1175x at the v2 codebook. All six paired tests are `p < 1e-12`, so the epoch
asymmetry is settled, and the arm that trained most is not the winner. The codebook gain is
concentrated in popular targets: 1.35x on the hottest popularity quintile and 0.79x on
targets that never appear in train.

Full breakdown, including the interaction, the popularity strata and what not to claim, is in
[`ablation.md`](ablation.md). Do not quote a single multiplier per axis anywhere.

**Launch both arms with `--dump_ranks`.** These two cells are trained with the v2 SID and v1
features, which is a combination no arm has been dumped for, and `budget_match.py` only reads
ranks that were written at evaluation time. The flag costs one file per arm and a rerun
otherwise. Same rule for any further arm: dump at the first run or do the eval twice.

## 2. Diagnose the new SID, and measure whether the tie groups actually cost anything — **done**

**Why.** Already done once in `archive/sid_results.md` and `archive/sid_diagnostics.md`; redo
it against v2 and add the collision check the supervisor asked for.

**Known already, so do not re-measure from scratch.** 53.51% of items share a 2-code tuple,
the largest group is 315 items, and the tie ceiling is R@1 62.78% / R@10 95.96% (re-derived
from `SID Diagnostics/index_rqvae_2x8192.json`; the archived 53.57% / 93.20% came from an
earlier quantiser run with 2.56M distinct tuples rather than 3.00M). Collisions
are structural at two levels, not a defect, so the useful question is not whether they exist
but how much of the achieved R@k the ceiling already explains.

**Done, 2026-11-01 to 2026-11-02.** Where each item landed:

1. **Full-tuple and per-level collision sizes**: [`sid.md`](sid.md) §1.9, *Every level,
   measured*, has all five v1 levels and both v2 levels with buckets, items per bucket, entropy
   and share of the level's own `log2 size`. v1's level 4 is 18.5% and is a collision rank.
2. **The tie ceiling at every cut-off**: same section, plus `README.md` §2.3(d). 62.78% at `R@1`
   and 95.96% at `R@10` for v2. It is a codebook property, not an arm property, so there is no
   per-arm version of it to compute.
3. **Items per level-0 code**: 583.88 for v2, 18,684.20 for v1, against the paper's 39-195. This
   is the single figure that says how far the codebook is from the published setting.
4. **The unrun cells, `2x16384` and `3x16384`: deliberately not run.** They existed to answer
   the counting-versus-scoring question by widening, and section 4 answered it directly by
   ranking the same tuple scores injectively (1.34x at `R@10`). A wider codebook would only
   re-measure a question that already has a number.

What the diagnosis cannot say is in [`sid.md`](sid.md) as well: lift, occupancy and bits
quantify what a codebook can express and never what an item means. Sections 4 and 5 are where
that changes into a behavioural claim.

**Cost actually spent.** Minutes of CPU for the diagnosis, 1.7 minutes of GPU for the
tie measurement.

**The tie measurement: answered by reading the code, so nothing to run.**

1. **Exact ties do happen, by construction.** `V_i` is the same sum of the same level rows for
every member of a tuple, so their `s_i` is bit-identical rather than nearly equal. There is no
near-tie mixture to disentangle.
2. **The break inside a group is arbitrary, not favourable.** `evaluate()` ranks by the
target's position in `topk`, so a tied target is placed by catalogue index, and item id
carries no behavioural signal. The consequences are asymmetric across arms: DiGR is charged
for its ties while TIGER's prefix-free arm is not. `evaluation.md` section 6 records the three
conventions and the direction of the bias.
3. **The `R@k` split by group size was measured** and is in `README.md` 2.3(d): unique-tuple
targets reach `R@1` 2.427% against 1.055% overall, and the gap survives inside every
popularity quintile while co-click, which has no codebook, shows 1.1x on the same rows.

Consequence for the fix that 2.3(d) proposed: a content-side *index* cannot help. A break by
catalogue index is arbitrary, so a third code carrying that index is unlearnable, and the
favourable-tie reading that made it look worth 2x does not hold. What remains is the
counting-versus-scoring split, which moved to section 4 and to `Plan 3/`.

**Cost.** Seconds to minutes of CPU, and it was spent: the diagnosis on 2026-11-01, the tie
measurement on 2026-11-02. This section is closed.

## 3. TIGER on SID v1, so the method comparison stops crossing two changes at once — done

**Why.** DiGR has a v1 cell and a v2 cell. TIGER has only a v2 cell. Every "TIGER beats
DiGR" sentence therefore crosses the SID change and the feature change as well as the
factorisation, and the SID/feature step is measured at 1.41x on DiGR, which is larger than
the 1.11x effect being attributed to the factorisation (13.9994% against 12.5804% at the
matched budget). Until this run exists, the factorisation claim is not supported by its own
evidence.

| SID | features | DiGR | TIGER |
| --- | --- | --- | --- |
| `4x256+731` | v1: no action token, 64 gap bins | 2.5940% `R@10` | beam 27.6 items/row, 10.0677% reach |
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
`4x256+731` the 731-wide suffix level makes a full tuple near-unique: 3,459,854 distinct 5-tuples
over 4,783,154 items, a mean of 1.3825 items per tuple, so 20 tuples name about 27.6 items.
(The other two figures are easy to confuse. 27.67% of items are *excess*, beyond one per
distinct tuple. 39.28% of items sit in a tuple that holds more than one, which is what
1,878,898 means in the v1 log. Neither is the list length.) Same width, roughly
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

**Result, 2026-11-03.** Only arm **(b)** was run, and it is the informative one. Data is the
archived v1 dir (`4x256+731`, v1 features, 3 epochs, `tiger_v1_sid4x256_e3_prefix`); one
`--mode beam` pass over all 98,304 rows at width 20, 89.9323% of rows missing.

| arm | items named | reach |
| --- | --- | --- |
| TIGER v1 prefix, beam 20 | **27.6** | **10.0677%** `R@20 tuples` |
| coclick h20 | 27.6 | 6.9625% (interpolated) |
| DiGR v1 | 27.6 | 4.7920% (interpolated, archive JSON) |
| DiGR arm b | 27.6 | 4.5536% (interpolated) |

So the **second branch** fired, and not the one this task anticipated: at equal budget TIGER
is 2.10x DiGR v1, where the two-level arm in the v2 table was 1.11x behind DiGR. `R@1
1.7558%` is the best in the project and, because a v1 tuple holds 1.3825 items, its item-level
value lies in a narrow interval, 1.27% to 1.76%, so unlike the v2 row it can be compared at
rank 1. `nDCG@10 0.04217`, `MRR@10 0.03223`, effective rank 4.69.

The arm also loses somewhere, and badly: cold-target recall is one row in 15,715, against
0.1400% for co-click and 0.1973% for DiGR, a 4,276x coldest-to-hottest ratio against 93x and
46x. Twenty named cells out of 4,783,154 make a target unreachable when its cell is never
visited, and the cells visited are the fat ones.

**What this run does not establish.** Three things, each of which changes how the sentence
above may be written. It does not isolate depth, because v1 differs from v2 in depth, width
and injectivity at once, and its head is 9.3x smaller. It does not say TIGER's *lists* are
popularity lists, because the strata measure whether cold targets are found, not what the 27.6
named items are, and the lists are in fact highly personalised (67,007 distinct cells over
98,304 rows, mean pairwise Jaccard 0.004). And arm **(a)**, prefix-free at five levels, was
never run, so the five-level SID as read by the plain oracle is still unmeasured.

**Still missing for the matched-budget join:** DiGR v1's `--dump_ranks` pass, about 1.8 min of
GPU. Its cells above come from the archive JSON with an interpolation in between, so the v1
table in `README.md` §1.10 is one dump short of being fully measured.

## 4. Region multiplicity: counting or scoring (gate for any item-side move) — done
**Result, 2026-11-02.** Run on arm a in 1.7 min, and the item column reproduces the
`--dump_ranks` file on 100.00000% of 98,304 rows, so the left side is the leaderboard measure
itself. Same tuple scores ranked injectively: `R@1` 0.9705% to 1.1414% (**1.18x**), `R@10`
3.2725% to 4.3742% (**1.34x**), `R@100` 9.8674% to 13.0168% (1.32x), `R@1000` 25.5239% to
31.5450% (1.24x). Full table in `sid.md` 4 and `README.md` 2.3(d).

So it is **mixed and mostly scoring**: a quarter of what these scores can already reach at
`R@10` is lost to counting, three quarters is the model's own ranking. Consequences: move (d) in
`README.md` 2.3 is capped at 1.18x at `R@1`, so it drops in priority; a finer codebook keeps its
place, since that quarter is real and a finer code is the only way to collect it, but it is a
remainder not a direction; and item-side capacity stays priced at the 0.15 slope, so
concat-plus-MLP stays off the list. The earlier decision rule was right about the shape and
wrong about the size: neither outcome near 2.43% nor near 1.06% occurred.

**Why (kept as the record).** Measured 2026-11-02 on the v2 dump: `R@1` is 2.427% when the
target is alone in its SID tuple and 0.262% when it shares one, a 9.3x gap that is monotone in
group size and that survives popularity controls. Co-click, which has no codebook, shows 1.1x on
the same rows, so the gap is DiGR's item representation rather than the items. `README.md`
2.3(d) has the table. Note the 9.3x mixes "no tie" with "easier rows"; the controlled figure is
the 1.34x above.

**What.** ~~`Plan 3/region_rank.py`, one GPU pass, no training.~~ It scores the 3.0M *distinct
tuples* instead of the 4.78M items and reports two ranks per row:

- `item_rank = 1 + sum over tuples above of size`, which is what `item_logits` computes;
- `region_rank = 1 + #{tuples above}`, which is the item rank an injective codebook would
give from the *same* scores.

The gap between them is the whole counting effect. Everything else is scoring.

**The decision.** ~~Injective `R@1` near 2.43% means the loss is arithmetic and a finer SID is
justified. Near 1.06% means the scores are wrong on crowded rows and the lever is the item
vector (per-level weights, non-linear composition), not the codebook.~~ Superseded by the
result above: 1.18x at `R@1`, between the two bounds, and the proportion is a quarter counting
to three quarters scoring at `R@10`.

**Cost.** A few minutes of one GPU. `DUMP_PATH` writes per-row ranks, so `--report_only`
re-derives the table with no GPU afterwards.

## 5. Bigger backbone (deferred, last)

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

# NZ5-GR — current record

This file is the live record for the project. Everything written before 2026-10-29 is in
`docs/archive/` and stays there as history: it contains the measurements, derivations and
retired plans that led here, including conclusions this file corrects. When the two
disagree, this file wins.

The archived files refer to each other as `docs/x.md`. Those paths are now
`docs/archive/x.md`; nothing else inside them was edited, so their text still reads as it
did when it was written.

How every number below is produced, including the miss convention, the tie handling, the
matched-budget construction and the guards that refuse a misaligned dump, is documented
separately in [`evaluation.md`](evaluation.md). Read that before comparing anything here
with a number from another paper.

The SID itself has its own file, [`sid.md`](sid.md). Both vintages were diagnosed with the
same instrument, and it records what that instrument can and cannot establish: bucket lift,
items per code, the collision rate, the tie ceiling, and the fact that none of it is a
claim about behaviour.

[`terminology.md`](terminology.md) fixes the vocabulary. Several words in this project have
more than one candidate and picking the wrong one changes the claim, most of all tuple, which
the files otherwise call a cell, a bucket or a region, and recall, which does not mean the same
thing on the item axis as on the tuple axis but is called recall on both.

Two things are recorded below. The measured result of the matched-budget pass, which is
the first comparison in this project where every arm was allowed the same number of items
per row. And the direction, which survived that result and is now sharper for it.

A third is now measured and sits at the end of the first part: what the two SIDs can and
cannot express, read straight from the quantiser output (§1.9).

And the attribution question is answered. [`ablation.md`](ablation.md) has the 2x2 that
separates the new SID from the new features, and it corrects two things this file used to
imply: the two effects are not additive, and the 1.41x is a popular-item gain rather than a
cold-start one.

The depth question now has an answer too. TIGER at full depth on the v1 SID, at the 20 items
its beam names, reaches 10.07% against DiGR's 3.56% on the same data, so it wins its own
comparison where the two-level arm lost its own (§1.10). Read that with its caveats: the
two vintages differ in more than depth, and the strata table says the win is heavily
popularity-weighted.

What to do next is in [`tasks.md`](tasks.md): four of the five items are closed and every cell
in §1.10 is a count rather than a bracket, so the only run left is the bigger backbone, which
is deliberately last.

---

# 1. The result: matched budget, 2026-10-29

## 1.1 Why a new pass was needed

Every previous comparison in `archive/results.md` compared arms at the same *cut-off*
(`R@10`, `R@1000`), which silently gives each arm a different sized shortlist. TIGER's
beam names a fixed number of SID tuples; DiGR ranks the whole 4,783,154 item catalogue.
`R@10` for TIGER means "target's tuple is among the first 10 tuples"; for DiGR it means
"target is among the first 10 items". Those are different questions and the table they
produced was unreadable in both directions.

The pass was meant to fix that, and it started from a wrong assumption: that a beam of 20
tuples names about 32 items, because the catalogue averages 1.593 items per tuple.

## 1.2 The correction: the beam names 192 items, not 32

| quantity | value |
| --- | --- |
| catalogue | 4,783,154 items, 2 levels of 8,192 |
| distinct level-0+1 tuples | 3,002,708 |
| catalogue mean items per tuple | 1.593 |
| 20 tuples at the catalogue rate | 31.9 items |
| **items actually named per row** | **mean 192.17, median 171, min 25, max 1439** |
| implied items per *selected* tuple | 9.6 (6.0x the catalogue rate) |
| the same, restricted to target-set items | mean 91.41, median 83, min 9, max 410 |

A beam is a *scored* search, not a random sample of tuples. It walks into the level-0
codes carrying the most next-item mass, and the categories holding popular items are
exactly the fat ones. So it selects dense cells by construction, every row, with no short
beams (min 20 tuples kept on all 98,304 rows).

This is a real property of the method, not a bug. It also means a beam's width *understates*
its coverage by about 6x, which nobody had accounted for.

**The shape of $N_r$, over all 98,304 rows.**

| statistic | items named per row |
| --- | --- |
| mean | 192.17 |
| standard deviation | 105.76 (coefficient of variation 0.55) |
| median | 171 |
| interquartile range | 118 to 240 (122 wide) |
| p1 / p99 | 47 / 548 |
| p99.9 | 799 |
| min / max | 25 / 1,439 |
| skew | +1.65 |
| Gini | 0.288 |
| share of all named items held by the fattest 10% of rows | 22.0% |
| rows whose 20 tuples are all collision free | 0 |

Full percentile row: p5 67, p10 82, p25 118, p50 171, p75 240, p90 324, p95 393, p99 548.

So a `k = 192` cell is honest about the average and wrong about any given row. The middle half
of rows name between 118 and 240 items, a factor of 2, and the mean sits above the median
because of the right tail. Only 1.4% of rows come in under 50 items and 87.2% under 300, while
no row names fewer than 25. Every row carries at least one tuple with a collision in it.

**The budget is not independent of difficulty.** Decode rate rises with $N_r$:

| $N_r$ quartile | range | rows | mean $N_r$ | decoded |
| --- | --- | --- | --- | --- |
| 1 | 25 to 118 | 24,949 | 86.6 | 11.39% |
| 2 | 118 to 171 | 24,241 | 144.9 | 11.95% |
| 3 | 171 to 240 | 24,640 | 203.6 | 12.45% |
| 4 | 240 to 1,439 | 24,474 | 335.1 | 14.55% |

Decoded rows average 204.1 named items against 190.5 for the rows it missed. Fat cells are the
popular ones, so a longer list correlates with an easier row. The per-row table stays fair,
because both arms get the same budget on the same row, but the number is implicitly
popularity-weighted relative to a fixed-$k$ table and that is worth knowing when it is quoted.

## 1.3 The table

Same 98,304 validation rows, same 4,783,154 item catalogue, every arm scored by whether
the single true target is inside the items that arm names. `matched @ N_r` allows each arm
the exact number of items the beam named on that row.

| arm | items named / row | matched @ N_r | nDCG@N_r | MRR@10 | @ k=171 | @ k=192 | @ k=1000 | @ k=10000 | cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **DiGR v2** | full catalogue | **13.9994%** | **0.04013** | **0.01742** | 13.8224% | 14.4989% | 27.6622% | 55.7058% | 1.8 min GPU |
| coclick h20 | ~192 | 13.7034% | 0.03832 | 0.01539 | 13.5824% | 14.1825% | 25.3021% | 46.8882% | 43 s CPU |
| coclick h10 | ~192 | 13.5518% | 0.03797 | 0.01529 | 13.4369% | 14.0371% | 24.8311% | — | 43 s CPU |
| coclick h5 | ~192 | 12.9985% | 0.03673 | 0.01499 | 12.8621% | 13.4145% | 23.6176% | — | 43 s CPU |
| **TIGER prefix, beam 20** | **192.2** | **12.5804%** | n/a † | n/a † | 12.5804% | 12.5804% | n/a | n/a | 46.6 min GPU |
| popularity | ~192 | 6.3070% | 0.01673 | 0.00648 | 6.2083% | 6.6060% | 14.7288% | 36.8215% | < 1 s CPU |

`matched @ N_r`, `nDCG@N_r` and `MRR@10` are the three like-for-like columns. The `R@k`
columns are comparable between the full-catalogue arms only.

**† Why TIGER has no `nDCG@N_r` or `MRR@10`.** Both metrics weight a hit by its position,
and TIGER's positions are **tuple** positions, not item positions: its position 3 holds
about 29 items while DiGR's holds one. Measuring it on that axis is not a larger value of
the same quantity, it is a different quantity, inflated by the ~9.6 items per position. Its
own tuple-axis figures are `nDCG 0.0592` and `MRR@10 0.03803`, against DiGR's item-axis
`0.04013` and `0.01742`, and printing those two side by side is exactly the comparison the
blank cells refuse to make. `12.5804%` stays in `matched @ N_r` because its own reach needs
no axis conversion: a decoded row is a hit, a missing row is not.

Every cell above is measured from the four dumps. Earlier revisions carried `~est` cells for
the co-click and popularity rows, derived by applying DiGR's uniform-$k$ to per-row-budget
shift. Measured against those estimates: h20 13.68% -> 13.7034%, h10 13.5% -> 13.5518%,
h5 13.4% -> 12.9985%. The first two were within 0.05 pp, the h5 estimate was 0.40 pp out,
and all three estimates are retired.

**Reading.** At the budget the beam actually uses, DiGR is 1.11x TIGER, and co-click sits
between them. TIGER is **10.1% behind DiGR** and 8.2% behind plain co-click h20. The earlier
claim that TIGER was 1.77x ahead was an artefact of assuming a 30-item budget.

The two rank-aware columns sharpen one ordering and leave the rest alone. DiGR's edge over
co-click h20 widens from +2.2% on matched recall to +4.7% on `nDCG@N_r`, so DiGR's hits sit
shallower inside the same budget even where the raw hit counts are close. Against popularity
DiGR is 2.4x on `nDCG@N_r` and 2.7x on `MRR@10`, so the count baseline is further behind
than its recall alone already showed.


### The short cut-offs, for reference only

People ask for these, so here they are in a separate table. They are **not** comparable, for
the reasons under the table.

| arm | R@1 | R@5 | R@10 | MRR@10 | source |
| --- | --- | --- | --- | --- | --- |
| **TIGER prefix, beam 20** † | **2.0406%** | **6.0964%** | **8.8969%** | **0.03803** | dumps |
| **TIGER prefix v1, beam 20** | **1.7558%** | **5.0741%** | **7.4849%** | **0.03223** | dumps |
| DiGR v2 | 1.0549% | 2.6245% | 3.6570% | 0.01742 | dumps |
| coclick h20 | 0.8057% | 2.4882% | 3.7069% | 0.01539 | dumps |
| coclick h10 | 0.7985% | 2.4780% | 3.6794% | 0.01529 | dumps |
| coclick h5 | 0.7853% | 2.4241% | 3.6011% | 0.01499 | dumps |
| popularity | 0.4395% | 0.9074% | 1.3438% | 0.00648 | dumps |
| TIGER prefix-free v2 | 1.4400% | — | 2.4500% | 0.01700 | archive |
| DiGR v1 (4x256 + 731) | 0.7924% | 1.8514% | 2.5940% | 0.01258 | dumps |
| item-embedding kNN, last 1 | 0.0173% | — | 0.5636% | 0.00151 | archive |
| HSTU tower probe | 0.0458% | — | 0.4354% | 0.00131 | archive |
| B3 SID transition | 0.0244% | — | 0.1801% | 0.00060 | archive |

Two things are wrong with a short cut-off across these rows, and the table is here to be read
with both in mind.

**TIGER's columns are on the wrong axis on the v2 SID** (†, same marker as above). Its `R@10`
is the share of rows where the target sat in the first ten **tuples**, and ten v2 tuples hold
about 96 items, so the v2 row's apparent 2.4x over DiGR at `R@10` is a 96-item list against a
10-item list, which is exactly the artefact the main table refuses to print. Measured honestly
the v2 beam has no `R@10` at all, only an `R@~96`.

The v1 beam row carries no marker, because there a five-code tuple is unique and holds exactly
one item, so its columns are ordinary item-level cut-offs and its `R@10` really is a top ten out
of 4.78M. Its list is only 20 items long, though, which is the second caveat below.

**A beam cannot return fewer than 192 items on v2, or 20 on v1.** Its short cut-offs are not
the same decision problem as a top-10 over 4.78M, because the list it draws its top 10 from was
already 192 long, or 20. Reading the column as each arm's own top-10 is the only correct reading. It is not a
ranking of the arms, and the archive rows make the point by themselves: prefix-free TIGER has
the best `R@1` of any full-catalogue arm (1.4400% against DiGR's 1.0549%) and the worst
`R@10` of the three main candidates (2.4500%).

The `—` cells were never recorded, not withheld. The archive tables kept `R@1`, `R@10` and
`MRR` only; `R@5` existed for the TIGER rows because `rank_metrics` returns it, and DiGR v2's
`R@5` was in the training log to two significant figures (2.6000%) until it was recomputed
exactly from the dumps above (2.6245%).

## 1.4 The crossover

Reach per item named is not a constant, so the ranking inverts with the budget:

| named items | who leads |
| --- | --- |
| 10 | co-click h20 (3.7069%) over DiGR (3.6570%) |
| ~100 | co-click h20 (10.9873%) over DiGR (10.9182%), a tie |
| **138** | **DiGR passes TIGER's flat 12.5804% (12.5946%)** |
| 192 | DiGR (14.4989%) over co-click (14.1825%) over TIGER (12.5804%) |
| 1,000 | DiGR (27.6622%) over co-click (25.3021%), +9% |
| 10,000 | DiGR (55.7058%) over co-click (46.8882%), +19% |

There is no single winner. Below 138 items TIGER's reach per item is the best of the three;
above it DiGR is. The beam lands at 192, on the losing side of its own crossover.

Depth DiGR needs to match a competitor's k=192 reach: k=160 against coclick h5, k=178
against h10, k=182 against h20, k=34 against popularity.

## 1.5 Where TIGER loses: the second code

| gate | value |
| --- | --- |
| level-0 `cover@20` | 42,578 / 98,304 rows = 43.31% |
| conditional level-1 success | 12,367 / 42,578 = **29.0%** |
| decoded rows landing at rank <= 10 of the beam | 8,749 / 12,367 = 70.75% |

So in 71% of the rows where TIGER finds the right first code, the right second code is
outside the top 20 of 8,192 candidates. That single gate is the whole loss. The ordering
inside the shortlist works; not enough rows enter it.

Beam log, 46.6 min, `logs/tiger_beam_728414.out`: `R@1 2.04%`, `R@5 6.10%`, `R@10 8.90%`,
`nDCG@1 0.02041`, `nDCG@5 0.04090`, `nDCG@10 0.04993`, `MRR@10 0.03803`.

## 1.6 The two factorisations fail on different rows

| | rows | share |
| --- | --- | --- |
| rows TIGER decodes | 12,367 | 12.5804% |
| ... of which DiGR also has the target inside 192 | 8,032 | 64.95% of TIGER's rows |
| DiGR gets at 192, TIGER does not | 6,221 | 6.33% |
| TIGER gets, DiGR does not | 4,335 | 4.41% |
| **union** | **18,588** | **18.9087%** |

The union is +31% over DiGR alone and +50% over TIGER alone. Explicit prefix conditioning
and implicit vector superposition are complementary, not redundant. Note this is evidence,
not a proposed arm: `archive/problems.md` section 1 excludes re-ranking, and a two-arm
fusion would be a new method rather than a comparison.

## 1.7 What this invalidates

- **"Beam 20 is about R@30"** — off by 6x, 192 items not 32.
- **"TIGER is 1.77x DiGR at matched budget"** — dead. At the real budget DiGR leads.
- **Every interpolated cell in `archive/results.md` section 14** that assumed ~30 named
  items, including DiGR 7.12%, co-click 7.18%, popularity 2.99%, prefix-free TIGER 4.43%.
- **"The next run is beam 60"** — beam 60 pushes further past the crossover, into the
  region where DiGR already leads. The informative runs are narrower, see section 2.5.

Never compare `cover@k` with `R@k` either. `cover@1000` 91.55% is reach over roughly
584,000 items (one level-0 code averages 584 items); DiGR's `R@1000` 27.66% is over 1,000
items. The gap is the same mistake in a different column.

## 1.8 How it was produced

Four dumps, each carrying the split row index so they are joined positionally and any gap
or duplicate is a hard failure.

| dump | flag | cost |
| --- | --- | --- |
| beam rows, ranks, chosen tuples | `train_tiger.py --dump_rows` | 46.6 min GPU |
| DiGR per-row target rank | `eval_full_catalog.py --dump_ranks` | 1.8 min GPU |
| co-click and popularity per-row rank | `baselines_digr.py --dump_ranks` | 4 s + 43 s CPU |
| join and count | `Model Training/budget_match.{py,sh}` | seconds |

Guards that ran: `eval_full_catalog.py` recomputes every `R@k` from the dumped ranks before
writing and exits on any disagreement; the join refuses a beam dump built on a different
SID, tuple shapes that do not match the width, a dump that misses or repeats a row, tuples
the catalogue does not own, an arm whose `targets` array is misaligned, any `rank == -1`,
and a named tuple that carries no items. TIGER's own reach is computed as
`mean(rank <= beam_width)`, never `rank <= N_r`, because a miss is stored as
`beam_width + 1` and the naive test would score every miss as a hit and print 100%.

Reproduced exactly, independently: DiGR `R@1/R@10/R@100` = 1.0549 / 3.6570 / 10.9182 and
loss 11.0133; TIGER decoded 12,367 of 98,304.

## 1.9 The two SIDs, measured from the quantiser output

Both vintages were re-diagnosed straight from the json the quantiser wrote, with
`SID Diagnostics/diagnose_sid.py`. No preprocessing, no GPU, 35 s for both files. This is the
only measurement in this file that is not about a model: it is about what the codes can and
cannot express, so it bounds every arm equally.

| SID | depth | buckets | items/bkt | largest group | `R@1` ceiling | `R@10` ceiling | lift | lift/1k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2x8192 | 1 | 8,192 | 583.88 | 3,838 | 0.17% | 1.71% | 463.3x | 7.589 |
| 2x8192 | 2 | 3,002,708 | 1.59 | 315 | 62.78% | 95.96% | 7,880.6x | 42.486 || 4x256+731 | 1 | 256 | 18,684.20 | 39,883 | 0.01% | 0.05% | 46.9x | 0.949 |
| 4x256+731 | 2 | 57,122 | 83.74 | 5,086 | 1.19% | 9.93% | 1,437.6x | 7.276 |
| 4x256+731 | 3 | 1,506,041 | 3.18 | 1,405 | 31.49% | 76.14% | 8,257.6x | 19.946 |
| 4x256+731 | 4 | 3,459,854 | 1.38 | 731 | 72.33% | 96.42% | 4,505.0x | 29.389 |
| 4x256+731 | 5 | 4,783,154 | 1.00 | 1 | 100.00% | 100.00% | 827.2x | 17.293 |

`lift` is `recall / uniform hit rate` on the same 98,304 val rows, so it rises mechanically as
buckets shrink and is not comparable across widths. `lift/1k` normalises by candidates,
`recall / (query bucket / 1000)` in percent, which is comparable but only meaningful at the
depth whose bucket is not far from 1,000 items.

Coverage: `2x8192` uses all 8,192 codes in both levels at 12.8110 and 12.7668 of 13 bits
(98.5% and 98.2%), spending 26.00 bits against the 22.19 an identity needs. `4x256+731` uses
all 256 in each of the first four levels (98.4% to 99.8% of 8 bits) but the last level is 731
dense values holding only 1.7644 of 9.5137 bits, a ratio of 18.5%.

Four things follow.

**The last level of v1 is a collision rank, and it owns a fifth of every item vector.** 731
dense values at 18.5% of their own `log2(size)` is the `--add_unique_id` signature. `V_i`
for v1 is the mean of five rows, so 20% of it is a rank inside a collision group, which
carries no semantics. Level 5 is a different waste: it makes the catalogue injective
(`R@1` ceiling 100%), so it is near-identity and also owns a fifth. `2x8192` has no such
level; both its levels are 98%+ dense codes.

**v1's codes peak at depth 3 and then fall 90%.** Lift runs 46.9x, 1,437.6x, 8,257.6x, then
4,505.0x and 827.2x. Levels 4 and 5 split density, not behaviour. So the usable depth of
`4x256+731` is three, and the depth at which DiGR v1 was trained is two levels past it.

**`2x8192` is still rising at its only usable depth**, 7,880.6x at depth 2, and it is never
injective, so on this catalogue it is a partition whose last level is doing real work. That
is the case for width here: not more bits, but a second level that has not run out.

**The tie ceiling does not bind, and that is not the same as collisions being free.** The
ceilings are 53.57% at `R@1` and 93.20% at `R@10` when averaged over the val targets, which is
the population `R@k` averages over, against measured 1.05% and 3.66%. Read on its own that says
the codes are fine,
which does not follow: a hash is injective with a 100% ceiling and carries no behaviour, and
`4x256+731` at depth 5 is exactly that case, injective at 827.2x lift. The rows that settle it
are the 36.64% whose target sits alone in its tuple, where no tie can be charged and `R@1` is
still 2.43% against a ceiling of 100%. Collisions are not the binding constraint at 3.66%
`R@10`; the ranking is. The size of the remainder is measured in [`sid.md`](sid.md) 4: holding
the scores fixed and ranking the same tuples injectively gains 1.34x at `R@10`. `sid.md` 4
states the joint test with lift. The table above reports 62.78% and 95.96% for the same codebook
because it averages over the **catalogue** instead, where every item counts once; that is a
different and less useful base here, since a metric over rows cares about the groups the rows
land in. Both are correct, and the row base is the one that bounds `R@k`.

---

## 1.10 TIGER on the v1 SID: the depth arm wins its own comparison

The arm in §1.3 is the one that lost, and it lost at two levels. This is the same model at
full depth, on the archived v1 data, so both sides of this comparison are v1 SID with v1
features and it is a matched pair with DiGR rather than with the v2 row.

**The budget is 20 items, not 192.** A v1 five-code tuple is unique: 4,783,154 distinct
tuples over 4,783,154 items, because level 4 is a collision rank, which is also why its entropy
is only 18.5% of its own width. The `1.3825 items per tuple` quoted elsewhere is items per
**four**-level prefix (3,459,854 distinct), not per full tuple. So twenty tuples name exactly
twenty items, and this whole comparison is a plain top-20. The v2 beam named 192.2 from the same
width only because there a tuple holds 1.593 items and the beam concentrates on the fat cells.
Same beam, 9.6 times the list.

**Measured, all four arms, same 98,304 rows.** `R@20` is the beam's whole list; the `R@100`
column and beyond are what the item arms can reach once they are allowed to keep going.

| arm | cap | misses | eff. rank | R@1 | R@5 | R@10 | R@20 | R@100 | R@1000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **TIGER prefix, beam 20** | 20 | 89.9323% | **4.69** | **1.7558%** | **5.0741%** | **7.4849%** | **10.0677%** | n/a | n/a |
| coclick h20 | 10,000 | 53.1118% | 481.21 | 0.8057% | 2.4882% | 3.7069% | 5.2684% | 10.9873% | 25.3021% |
| DiGR v1 | 10,000 | 50.2309% | 823.44 | 0.7924% | 1.8514% | 2.5940% | 3.5553% | 7.5094% | 21.6003% |
| popularity | 10,000 | 63.1785% | 990.23 | 0.4395% | 0.9074% | 1.3438% | 1.9796% | 4.7862% | 14.7288% |

**At equal budget, counted row by row.** `budget_match.py` sums the bucket sizes of the beam's
own twenty tuples on each row and asks each arm whether its rank fits. Because a v1 tuple holds
one item, `N_r` is 20 on every row and this is a straight top-20, which is the one case where the
matched calculation and a fixed cut-off agree exactly.

| arm | matched | nDCG@N_r | MRR@10 |
| --- | --- | --- | --- |
| **TIGER prefix, beam 20** | **10.0677%** | **0.04217** | **0.03223** |
| coclick h20 | 5.2684% | 0.02437 | 0.01539 |
| DiGR v1 | 3.5553% | 0.01811 | 0.01258 |
| popularity | 1.9796% | 0.00968 | 0.00648 |

So TIGER is **2.83x DiGR v1**, 1.91x co-click h20 and 5.09x popularity, where the same arm at
two levels was 1.11x **behind** DiGR at its own 192-item budget. Its `nDCG@N_r` and `MRR@10` are
reportable here, unlike the v2 row, for the same reason the budget is simple: tuple position and
item position are the same position when a tuple holds one item.

**Where the hits sit.** `nDCG@10 / R@10` is the mean weight of a hit, so it says how deep
inside the top ten the hits land: TIGER **0.5633** (deepest), DiGR v1 0.6053 (shallowest),
co-click 0.5513, popularity 0.6019. The beam finds the most targets and puts them furthest
down its own list, while DiGR v1 finds the fewest among the three real arms but places them
highest. Effective rank says the same thing in rank units: 4.69 against 823.44 and 481.21.

**`R@1` is exactly item-level here, unlike in §1.3.** A v1 tuple holds one item, so no interval
is needed: `1.7558%` is a plain item-level rank-1 number. It beats every other arm in this file:
DiGR v2 1.0549%, prefix-free TIGER v2 1.4400%, co-click 0.8057%, DiGR v1 0.7924%. For the v2
beam the corresponding interval was 0.21% to 2.04%, which is why that row's short cut-offs are
refused and this one's are not.

**Overlap.** At `k = 10` the beam contains DiGR almost entirely: DiGR finds 547 rows the beam
misses and the beam finds 5,355 that DiGR misses, for a union of 8.0414% against the beam's own
7.4849%. Co-click is the arm with something the beam lacks, 1.9297% of rows on its own and a
union of 9.4147%. Against popularity the union is only 7.8695%, because popularity is a fixed
order and the beam already covers it.

| pair | only the first | only the second | union |
| --- | --- | --- | --- |
| beam vs DiGR v1 | 5.4474% | 0.5564% | **8.0414%** |
| beam vs coclick h20 | 5.7078% | 1.9297% | **9.4147%** |
| beam vs popularity | 6.5257% | 0.3845% | 7.8695% |

All six pairs are significant under McNemar at `k = 10`, including beam against DiGR
(`b = 5,355`, `c = 547`, chi-squared 3,915.2). The printed `p = 0` is `erfc` underflow, not a
zero probability.

**Where it loses, it loses badly.** `R@10` by target popularity quintile, coldest to hottest:

| arm | tail | lower | middle | upper | head | head / tail |
| --- | --- | --- | --- | --- | --- | --- |
| TIGER v1 beam | 0.0064% | 0.0802% | 2.1296% | 7.2769% | 27.3696% | **4,276x** |
| coclick h20 | 0.1400% | 0.4719% | 1.4073% | 3.2165% | 12.9951% | 93x |
| DiGR v1 | 0.1782% | 0.3964% | 0.6477% | 1.6083% | 9.9788% | 56x |
| popularity | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 6.6744% | n/a |

On the coldest fifth the beam finds one target in 15,715 rows. Co-click, with the same number
of slots, finds 22x more, so this is not arithmetic. Naming twenty cells out of 4,783,154 means
a target whose cell the beam never visits is unreachable, and the cells it visits carry the
next-item mass. DiGR scores every item, so it cannot have a coverage hole, and its 56x spread
is the flattest of the four arms that can rank at all.

**Two caveats that belong beside the number.**

1. **This is an association, not an attribution to depth.** v1 differs from v2 in three ways at
   once: five levels instead of two, narrower levels (256 against 8,192), and injectivity at
   full depth. The head is also 9.3x smaller, so capacity is not the explanation. Depth is the
   most plausible single lever, and this is the run that makes that worth testing, but the clean
   test is a third vintage: three levels at 8,192, or v1 truncated to its first three.
2. **This measures reach, not what the arm recommends.** The strata say the beam misses cold
   targets. They do not say the 20 items it names are popular items. Those are different
   claims and only the first is measured. Its lists are highly personalised: 67,007 distinct
   cells across 98,304 rows, and mean pairwise Jaccard 0.004 between two rows.


# 2. The direction

Unchanged by the result above, and now with a sharper first step.

## 2.1 Where the headroom is, in one identity

```
R@10 = R@1000  x  P(target in top 10 | target in top 1000)
     = 27.6622% x 13.22%
```

The second factor is the part a better ranker could still win. It is 13-15% for *every*
arm, including co-click with exact per-item rows (14.65%) and popularity with no query at
all (9.12%). No arm in this project is close to good at ordering the head; they differ in
how deep they can reach.

## 2.2 Capacity on the item side is spent

Going from 4x256+731 (1,755 vectors, 1.80M parameters) to 2x8192 (16,386 vectors, 16.78M
parameters) is 9.3x the parameters for 1.41x the `R@10`:

```
slope = log(1.41) / log(9.3) = 0.15
```

A 0.15 exponent means another 10x on the item side buys about 1.4x, and another 100x buys
about 2x. Two of the three axes that moved in that step were query-side anyway (the action
token and the 128 gap bins). The 53.51% of items sharing a 2-code tuple and the 315-item
tie group are real. The tie ceiling is not the binding constraint at 3.66%, but that is an
argument about tie resolution and not about whether the representation is adequate: the
36.64% of targets sitting alone in a tuple, where no tie operates, still reach only 2.43%
`R@1`.

Consequence: a bigger codebook is not the big move. S9 at 16,384 is off the table.

## 2.3 The four moves worth making

**(a) A co-click code prior inside the score.** Legal under the content-only rule because it
is a query-side prior, and a cold item gets zero.

```
s_i = h . V_i  +  sum_l  lambda_l * log p_hat_l[c_il]
```

The table in section 2.1 says the discriminative work to be done is inside the top 1,000,
and the count table says behaviour knows something content does not: co-click reaches
14.65% conditional where DiGR reaches 13.22%. This puts a measured quantity into the score
instead of hoping the vector absorbs it.

**(b) A mixture-of-intents head.** One vector per user cannot hold several reasons to click.

```
log sum_j pi_j exp(h_j . V_i)
```

Each component is a separator; the mixture is still one summation over the item vectors at
serving time.

**(c) More query information, not more item capacity.** Co-click's `R@10` climbs 2.5238 →
3.3783 → 3.6011 → 3.6794 → 3.7069 at windows 1 / 3 / 5 / 10 / 20, and the sweep has
asymptoted. Going to a 60-event window with 3 anchors is more information per row, not more
compute per row.

**(d) A content-side tie break.** The SID cannot order items that share a code tuple, and on
this vocabulary that is 53.51% of the catalogue, mean group 6.39, largest 315. `V_i` is the
mean of two shared rows, so two colliding items get bit-identical logits and `topk` orders
them by catalogue index. The cost is not uniform across cut-offs, which is the reason this is
worth a move rather than a caveat. With 46.49% of targets unique, the colliding ones average
`(6.39 - 0.4649) / 0.5351 = 11.07` items per group, so

```
R@1  dilution = 0.4649 + 0.5351/11.07    = 0.513     49% of the skill is invisible
R@10 dilution = 0.4649 + 0.5351*10/11.07 = 0.948     only 5%
```

The reported `R@1` of 1.0549% is therefore about 2.06% of tuple-level accuracy. That figure
assumes the favourable break, where the target wins its block, and it is now withdrawn: see
the measured ceiling below.

Legal under the content-only rule, applied to exact ties only:

```
s_i = h . V_i  +  delta * h . u_i        when h . V_i ties
```

`u_i` is a per-item content vector, so cold items work, no codes change, the SID is not
retrained, and the cost is one extra dot product on the tied items only. It is the smallest
possible use of the clause section 2.6 leaves open.

Ceiling, measured rather than derived: [`Plan 3/region_rank.py`](../Plan%203/region_rank.py)
ranks the same scores over distinct tuples, and on arm a that is `R@1` 0.9705% to 1.1414%, a
**1.18x** cap. A tie break that always put the target first inside its block would still be
working inside that ceiling, so the 2.06% above is too high and is withdrawn. The one clean
stratum comparison replaces it: on unique-tuple rows only, v2 reaches 2.427% `R@1` against arm
a's 2.254%, which is the same 1.08x the arms show overall, so collisions have been spreading
the gap out rather than hiding it. Nothing at `R@10` or beyond either way: this is a precision
move and should not be sold as a recall move.

**The check that decides whether the arithmetic holds.** The tie order is not a lottery and it
is not chosen in the arm's favour: `evaluate()` ranks by the target's position in `topk`, which
for items that share a tuple means ascending catalogue index, and item id carries no
behavioural signal ([`evaluation.md`](evaluation.md) section 6). Split DiGR's measured `R@1` by
whether the target shares its tuple and compare against `1/11.07`. Minutes on
`local/2026-29-10-1-full-eval/dumps/digr_val.npz` plus `catalog_sid_codes.npy`; no GPU, no
retraining.

**Measured, 2026-11-02.** That split, on the v2 dump:

| tuple size | rows | share | `R@1` | `R@10` |
| --- | --- | --- | --- | --- |
| 1 | 36,017 | 36.64% | **2.427%** | **5.703%** |
| 2 | 17,275 | 17.57% | 0.590% | 3.757% |
| 3-4 | 16,451 | 16.73% | 0.310% | 3.210% |
| 5-8 | 12,441 | 12.66% | 0.040% | 1.953% |
| 9-16 | 7,639 | 7.77% | 0.039% | 1.217% |
| 17-315 | 8,481 | 8.63% | 0.024% | 0.330% |

9.3x at `R@1` between unique and shared targets, monotone in group size, and it is not
popularity: the gap survives inside every popularity quintile, at 1.6x, 3.4x, 8.7x, 8.9x and
3.0x. The control that settles it is co-click, which has no codebook at all and shows **1.1x**
on the same rows (0.855% on unique tuple targets against 0.777% on shared ones). So the gap
belongs to DiGR's item representation, not to the items or the queries.

That 9.3x is a description of the population, not a forecast of the fix: unique-tuple rows are
also the easier rows, so it mixes "no tie" with "easier". The controlled figure, holding the
scores fixed and ranking the same tuples injectively, is **1.18x at `R@1` and 1.34x at
`R@10`** (section 2.3(d) below and [`sid.md`](sid.md) 4).

Against the `1/11.07` prediction, shared targets imply `0.262% x 11.26 = 2.95%` tuple-level
accuracy against `2.427%` measured directly on unique targets. That is **below** the
prediction, which is what an arbitrary break gives rather than what a favourable one would:
the evaluator places a tied target at its `topk` position, i.e. by catalogue index, so the
model gets neither the best nor the average position inside a group as a matter of rule. So
the measured `2.427%` is the figure to quote and the derived `2.06%` above is too high, with the
`1.18x` injective cap as the limit on what any tie break can recover from it.

**What this still does not say, and why it matters.** The loss is either *counting* -- the
rank is the target's position in the sorted list, so every item inside every tuple ranked
above the target is counted, and a crowded code space inflates the rank arithmetically -- or
*scoring* -- the model does not put a crowded region on top often enough. The fixes are
opposite: a finer SID versus a better item vector. `Plan 3/region_rank.py` separates them in
one GPU pass, because the region rank is exactly the item rank an *injective* codebook would
produce from the same scores, measured under the same tie convention.

The answer is **mixed, and mostly scoring.** On arm a, with the scores held fixed, ranking the
same tuples injectively takes `R@10` from 3.2725% to 4.3742% and `R@1` from 0.9705% to 1.1414%,
so **a quarter of what these scores can already reach at `R@10` and 15% at `R@1` is lost to
counting** and the rest is the model's own ranking. Counting is therefore real and worth
collecting through a finer codebook, and it is not the lever that turns 3.66% into 8-11%. The
row-by-row version is in [`sid.md`](sid.md) section 4, and the run command in
`Plan 3/README.md`.

One point that simplifies it: the ties are **exact**, not near-ties. Every member of a tuple
gets the same $V_i$ row, so their scores are bit-identical and there is no last-bit spread to
disentangle. The only thing deciding a target's position inside its group is the sort order,
which is catalogue index.

The linear-summation-to-concat-plus-MLP change is not on this list on its own: it is
item-side capacity, and section 2.2 prices that at 0.15.

## 2.4 TIGER's axis is depth, not width

The factorisation the paper describes needs a real prefix tree. At two levels with
`PREFIX=0` the arm sits in one function class with DiGR: both are additive over the same two
shared code rows and both are linear in the separator state, so that run compared losses,
not factorisations.

The measured prize per extra level, from the prefix run: knowing the true `c1` removes 2.968
of 7.6731 nats of level-2 entropy, **38.7%**. That is what a third level is worth, and it is
larger than anything width has bought.

**Measured, 2026-11-03.** The five-level arm settles the direction of this section: at its own
20-item budget it reaches 10.0677% against DiGR v1's 3.5553%, 2.83x, where the two-level arm
was 1.11x behind DiGR at its own 192-item budget (§1.10). So the factorisation does work once
the tree is deep enough to localise, and the v2 shortfall was a property of a two-level code,
not of the architecture. Two limits on that reading: v1 changes depth, width and injectivity
together, so depth is implicated rather than isolated; and the arm is the most popularity-
weighted in the file (4,276x coldest to hottest quintile), so the win is concentrated where
the data is dense. A three-level SID at 8,192 is the run that would separate them, and it is
the same run §2.5 item 4 asks for.

## 2.5 The next five runs, in order

1. **Re-run the join** (`sbatch Model Training/budget_match.sh`, seconds, cpu partition).
   The picker bug is fixed: `run_baselines.sh` writes both kinds flat into
   `dumps/baselines/h20/`, the picker wanted a per-kind subdirectory and skipped all three
   windows. It has since been re-run and the `est` cells in section 1.3 are measurements.
   What is still missing for the v1 vintage is DiGR v1's own dump, 1.8 min of GPU, which is
   the last gap in §1.10.
2. **The tie-order check: run 2026-11-02** (minutes, CPU). Result in section 2.3(d): 9.3x at
   `R@1` between unique-tuple and shared-tuple targets, and 1.1x for co-click on the same
   rows, which puts the gap on DiGR's representation rather than on the data. It opened
   `Plan 3/region_rank.py`, which has now also run (1.7 min, 100.00000% of the dumped ranks
   reproduced): the same tuple scores ranked injectively reach **1.34x at `R@10`, 1.32x at
   `R@100`, 1.24x at `R@1000`, 1.18x at `R@1`**. So a quarter of what these scores can already
   reach at `R@10` is lost to counting and three quarters is the model's own ranking. Three
   consequences: move (d), a content-side tie break, is capped at 1.18x at `R@1` and drops down
   the list; a finer codebook keeps its place, because that quarter is real and a finer code is
   the only way to collect it; and none of this changes the 0.15 slope for item-side capacity,
   which is why concat-plus-MLP stays off the list.
3. **Beam 10 and beam 12** (~25 min each, about 96 and 115 items named), on the **v2** arm.
   This was the frontier test on the v2 budget: TIGER had to beat DiGR at 96 items (10.6%) or
   115, not at 192. Beam 60 would push the wrong way. §1.10 has since answered the depth
   question on the v1 arm, where TIGER is 2.83x DiGR at its own budget, so these two runs are
   now about whether the *two-level* arm can be rescued rather than about whether depth works.
   Worth doing only if a 2-level SID stays in the plan.
4. **Three levels at 8,192 codes.** This is now the run that settles the caveat in §1.10,
   because it changes depth alone and holds width and injectivity fixed. It also attacks the
   29% level-1 gate, since a third level makes each decision 8x cheaper to hit inside a
   20-wide beam. Needs a quantiser run of about 1 GPU-h.
5. **The SASRec bound that was never run** (`archive/tasks.md` 3.5, about 15 min): a tiny
   sequence model over raw item ids bounds what any SID method can hope for on this data.
   Without it there is no way to say whether 14% is the data or the factorisation.

## 2.6 What is out of scope, and stays out

Re-ranking. The deliverable is a comparison of SID retrieval methods; a second stage measures
the second stage and could be bolted onto any arm. ANN and continuous-vector retrieval are
deferred, not cancelled. Behavioural SID was retired: content is the only input a cold item
has, so behaviour inside the representation breaks cold start twice. A content encoder that
ranks better while needing nothing but content at serving time still violates nothing.

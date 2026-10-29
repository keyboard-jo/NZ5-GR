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

Two things are recorded below. The measured result of the matched-budget pass, which is
the first comparison in this project where every arm was allowed the same number of items
per row. And the direction, which survived that result and is now sharper for it.

What to do next is in [`tasks.md`](tasks.md): an ablation that separates the new SID from the
new features, a redone SID diagnosis, and a bigger-backbone run that is deliberately last.

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
| DiGR v2 | 1.0549% | 2.6245% | 3.6570% | 0.01742 | dumps |
| coclick h20 | 0.8057% | 2.4882% | 3.7069% | 0.01539 | dumps |
| coclick h10 | 0.7985% | 2.4780% | 3.6794% | 0.01529 | dumps |
| coclick h5 | 0.7853% | 2.4241% | 3.6011% | 0.01499 | dumps |
| popularity | 0.4395% | 0.9074% | 1.3438% | 0.00648 | dumps |
| TIGER prefix-free v2 | 1.4400% | — | 2.4500% | 0.01700 | archive |
| DiGR v1 (4x256 + 731) | 0.7924% | — | 2.5940% | 0.01258 | archive |
| item-embedding kNN, last 1 | 0.0173% | — | 0.5636% | 0.00151 | archive |
| HSTU tower probe | 0.0458% | — | 0.4354% | 0.00131 | archive |
| B3 SID transition | 0.0244% | — | 0.1801% | 0.00060 | archive |

Two things are wrong with a short cut-off across these rows, and the table is here to be read
with both in mind.

**TIGER's columns are on the wrong axis** (†, same marker as above). Its `R@10` is the share
of rows where the target sat in the first ten **tuples**, and ten tuples hold about 96 items.
So its apparent 2.4x over DiGR at `R@10` is a 96-item list against a 10-item list, which is
exactly the artefact the main table refuses to print. Measured honestly it has no `R@10` at
all, only an `R@~96`.

**A beam cannot return fewer than 192 items.** Its short cut-offs are not the same decision
problem as a top-10 over 4.78M, because the list it draws its top 10 from was already 192
long. Reading the column as each arm's own top-10 is the only correct reading. It is not a
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

---

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
tie group are real, but the tie ceiling (`R@1` 53.57%, `R@10` 93.20%) is not the binding
constraint at 3.66%.

Consequence: a bigger codebook is not the big move. S9 at 16,384 is off the table.

## 2.3 The three moves worth making

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

## 2.5 The next four runs, in order

1. **Re-run the join** (`sbatch Model Training/budget_match.sh`, seconds, cpu partition).
   The picker bug is fixed: `run_baselines.sh` writes both kinds flat into
   `dumps/baselines/h20/`, the picker wanted a per-kind subdirectory and skipped all three
   windows. Turns the two `est` cells in section 1.3 into measurements. Nothing else needs
   re-running; all four dumps are complete and were verified.
2. **Beam 10 and beam 12** (~25 min each, about 96 and 115 items named). This is the real
   test of depth, because it sits on the frontier: TIGER has to beat DiGR at 96 items
   (10.6%) or 115, not at 192. Beam 60 would push the wrong way.
3. **Three levels at 1,024 codes.** The loss is the 29% level-1 gate with 8,192 candidates.
   Making each decision 8x cheaper to hit inside a 20-wide beam attacks it directly. Needs
   a quantiser run of about 1 GPU-h.
4. **The SASRec bound that was never run** (`archive/tasks.md` 3.5, about 15 min): a tiny
   sequence model over raw item ids bounds what any SID method can hope for on this data.
   Without it there is no way to say whether 14% is the data or the factorisation.

## 2.6 What is out of scope, and stays out

Re-ranking. The deliverable is a comparison of SID retrieval methods; a second stage measures
the second stage and could be bolted onto any arm. ANN and continuous-vector retrieval are
deferred, not cancelled. Behavioural SID was retired: content is the only input a cold item
has, so behaviour inside the representation breaks cold start twice. A content encoder that
ranks better while needing nothing but content at serving time still violates nothing.

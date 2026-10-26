# DiGR vs baselines: consolidated results

All numbers are on the **same 98,304 val rows**, the **same val targets**, and the
**same full catalog of 4,783,154 items**. One target per row, so hit rate and
recall coincide. `MRR` is truncated at rank 10 everywhere.

| method | cost | R@1 | R@10 | R@100 | R@1000 | nDCG@10 | MRR |
|---|---|---|---|---|---|---|---|
| popularity (train target frequency) | < 1 s CPU | 0.4395% | 1.3438% | 4.7862% | 14.7288% | 0.00809 | 0.00648 |
| B3 SID transition (first order, smoothing 0.5) | 3,805 s CPU, 8 workers | 0.0244% | 0.1801% | 0.9410% | 4.2236% | 0.00087 | 0.00060 |
| B4 co-click kNN (window 1, decay 0.5) | 6 s CPU | 0.5676% | 2.5248% | 7.4605% | 18.5862% | 0.01405 | 0.01065 |
| DiGR 0.6B, 3 epochs, 300k negatives | 4.6 GPU-h + 2.9 min eval | 0.7924% | 2.5940% | 7.5094% | 21.6003% | 0.01570 | 0.01258 |

DiGR over co-click: R@1 **1.40x**, R@10 **1.03x**, R@100 **1.01x**, R@1000
**1.16x**, nDCG@10 1.12x, MRR 1.18x.
DiGR over popularity: R@1 1.80x, R@10 1.93x.
DiGR over B3: R@1 32.4x, R@10 14.4x.

Compute ratio DiGR : co-click = 4.6 GPU-h : 6 s = about **2,760x**.

## 1. The old popularity R@10 was wrong

The previously recorded popularity baseline was R@1 0.4395% / R@10 0.9644%.
Re-measured by `baselines_digr.py` on the same rows it gives R@1 0.4395% /
R@10 **1.3438%**. R@1 reproduces to eight digits, ranks 2-10 do not, so the
earlier figure came from a different frequency vector. The old number is retired;
every "lift over popularity" claim quoted before this is 39% too generous at
R@10: DiGR beats popularity 1.93x at R@10, not 2.69x.

## 2. A 6-second count table matches the 4.6-GPU-hour model at R@10

B4 co-click kNN counts `(previous item -> next item)` on the train split, weights
by decay over the history window, and fills cold queries from popularity. That is
the entire method, it takes 6 seconds, and it scores R@10 2.5248% against DiGR's
2.5940% - a 1.03x difference, well inside noise on 98,304 rows. At R@100 it is a
dead heat: 7.4605% against 7.5094%, 49 rows out of 98,304.

DiGR's real edges are at R@1 (`0.7924%` vs `0.5676%`, 779 vs 558 hits, about 6
standard errors, so it is a real effect), at MRR (1.18x), and at R@1000
(`21.6003%` vs `18.5862%`, ~2,970 extra hits, ~15 standard errors). Read plainly:
**the LLM does not rank better, it sharpens the top-1 and extends the tail.**
Between rank 10 and rank 100 the two systems are indistinguishable.

DiGR also uses the full 19.87-event history plus static user features; B4 uses a
single item. The incremental value of all that extra input over one co-click is
1.40x at R@1, 1.03x at R@10 and 1.01x at R@100.

## 3. B3 (SID transition) is a negative result, and why

R@10 0.1801% is 7.5x **worse** than popularity, for 63 minutes of compute. The
build log says why:

```
level 0: 34,019,640 transitions over  65,536 cells (98.6% dense)
level 1: 34,019,640 transitions over  65,536 cells (100.0% dense)
level 2: 34,019,640 transitions over  65,536 cells (100.0% dense)
level 3: 34,019,640 transitions over  65,536 cells (100.0% dense)
level 4: 34,019,640 transitions over 534,361 cells ( 3.0% dense)
```

Levels 0-3 each have only 256 x 256 = 65,536 `(previous code, next code)` cells
and 34M transitions to fill them: 519 hits per cell on average, essentially
saturated. A saturated conditional is near-uniform, so `P(code | prev_code)`
carries almost no signal, and summing five such log-probabilities ranks
near-ties, resolved by the popularity tie-break. The SID was designed to compress
4,783,154 items into 5 x 256 codes, not to serve as a first-order transition
table. TIGER-style results come from conditioning on the whole history, not from
one-step code counts.

Kept in the code as an opt-in baseline, dropped from the default.

## 4. The recall curve is log-linear, and that is the deployment story

The four measured points for DiGR are remarkably straight in `ln R` against
`log10 k`:

| k | 1 | 10 | 100 | 1000 |
|---|---|---|---|---|
| R@k | 0.7924% | 2.5940% | 7.5094% | 21.6003% |
| slope per decade | - | 1.19 | 1.06 | 1.06 |

A constant slope of 1.06 nats per decade means **every 10x more candidates buys
2.89x more recall**. Extrapolating: R@10,000 is about 62% (and co-click's flatter
curve lands near 46%). If that holds, a two-stage pipeline keeping 10,000
candidates - 0.2% of the catalog - would recover roughly 60% of targets, which is
a workable design.

Earlier reading of R@1000 alone (21.6%, "far past 1,000 needed") was too
pessimistic; the curve says the useful region is around k = 10,000, not beyond
the horizon. It is one more eval run to measure instead of extrapolate:
`eval_full_catalog.py --ks 1,10,100,1000,10000`.

## 5. Open items

- Re-run `eval_full_catalog.py` after copying the fixed file: the run above printed
the retired popularity reference (0.9644%, lift 2.69x) and no co-click line. The
measured DiGR rows themselves are valid.
- Measure `--ks 1,10,100,1000,10000` for DiGR and co-click to replace the 62%
extrapolation with a number.
- B4 at `--history_window 3` and `5`. The current run used window 1; a wider
  window costs seconds and is the strongest fair version of the baseline. If it
  beats DiGR at R@10 outright, the conclusion is settled.
- Decide which metric the project claims. R@1 / MRR / R@1000 is where DiGR wins;
  R@10 and R@100 are ties.
- Not yet spent: more exposure (3 anchors/record = 2.48M rows instead of 1.72M)
  or a larger backbone. Do not spend it before the cheap items above.

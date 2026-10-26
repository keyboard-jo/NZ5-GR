# DiGR vs baselines: consolidated results

All numbers are on the **same 98,304 val rows**, the **same val targets**, and the
**same full catalog of 4,783,154 items**. One target per row, so hit rate and
recall coincide. `MRR` is truncated at rank 10 everywhere.

| method | cost | R@1 | R@10 | R@100 | R@1000 | R@10000 | nDCG@10 | MRR |
|---|---|---|---|---|---|---|---|---|
| popularity (train target frequency) | < 1 s CPU | 0.4395% | 1.3438% | 4.7862% | 14.7288% | 36.8215% | 0.00809 | 0.00648 |
| B3 SID transition (first order, smoothing 0.5) | 3,805 s CPU, 8 workers | 0.0244% | 0.1801% | 0.9410% | 4.2236% | - | 0.00087 | 0.00060 |
| B4 co-click kNN, window 1, decay 0.5 | 4 s + 46 s CPU | 0.5686% | 2.5238% | 7.4646% | 18.5822% | 39.4318% | 0.01405 | 0.01065 |
| B4 co-click kNN, window 3, decay 0.5 | 4 s + 46 s CPU | 0.7355% | 3.3783% | 9.6130% | 21.9757% | 42.3777% | 0.01865 | 0.01405 |
| **B4 co-click kNN, window 5, decay 0.5** | **4 s + 46 s CPU** | **0.7853%** | **3.6011%** | **10.4492%** | **23.6176%** | **44.1905%** | **0.01989** | **0.01499** |
| DiGR 0.6B, 3 epochs, 300k sampled negatives | 4.6 GPU-h + 3.2 min eval | 0.7924% | 2.5940% | 7.5094% | 21.6003% | 49.7691% | 0.01570 | 0.01258 |

co-click (window 5) over DiGR: R@1 **0.99x**, R@10 **1.39x**, R@100 **1.39x**,
R@1000 **1.09x**, R@10000 0.89x, nDCG@10 1.27x, MRR 1.19x.
DiGR over co-click (window 1): R@1 1.39x, R@10 1.03x, R@100 1.01x, R@1000 1.16x,
R@10000 1.26x.
DiGR over popularity: R@1 1.80x, R@10 1.93x, R@10000 1.35x.
DiGR over B3: R@1 32.4x, R@10 14.4x, R@100 8.0x.

Compute ratio DiGR : co-click = 4.6 GPU-h : 4 s = about **4,100x**.

## 1. The old popularity R@10 was wrong

The previously recorded popularity baseline was R@1 0.4395% / R@10 0.9644%.
Re-measured on the same rows it gives R@10 **1.3438%**. R@1 reproduces to eight
digits, ranks 2-10 do not, so the earlier figure came from a different frequency
vector. The old number is retired; every "lift over popularity" claim quoted
before this is 39% too generous at R@10: DiGR beats popularity 1.93x at R@10,
not 2.69x.

## 2. A co-click count table beats the model wherever the model was ahead

The published co-click run used window 1, the weakest form. Sweeping the window
changes the picture completely:

| window | R@1 | R@10 | R@100 | R@1000 | R@10000 |
|---|---|---|---|---|---|
| 1 | 0.5686% | 2.5238% | 7.4646% | 18.5822% | 39.4318% |
| 3 | 0.7355% | 3.3783% | 9.6130% | 21.9757% | 42.3777% |
| 5 | 0.7853% | 3.6011% | 10.4492% | 23.6176% | 44.1905% |
| DiGR | 0.7924% | 2.5940% | 7.5094% | 21.6003% | 49.7691% |

At window 5 co-click ties DiGR at R@1 (0.7853% vs 0.7924%, 7 rows out of 98,304)
and beats it everywhere else up to rank 1,000: 1.39x at R@10, 1.39x at R@100,
1.09x at R@1000, 1.27x at nDCG@10, 1.19x at MRR. The earlier "DiGR sharpens the
top-1" reading was an artifact of comparing against the weakest baseline; that
edge is gone.

DiGR's only remaining win is R@10000: 49.7691% against 44.1905%, a 1.13x edge in
the far tail. That is the one band where a model that scores all 4,783,154 items
beats a table that can only propose a few dozen.

The window sweep has not plateaued (window 3 to 5 still adds 6.8% at R@1 and
6.6% at R@10), and val rows average 19.87 events, so window 10 and 20 are worth
46 seconds each.

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
table.

Kept in the code as an opt-in baseline, dropped from the default.

## 4. The recall curve flattens, and the earlier extrapolation was too optimistic

DiGR measured, with the slope of `ln R` per decade of k:

| k | 1 | 10 | 100 | 1000 | 10000 |
|---|---|---|---|---|---|
| R@k | 0.7924% | 2.5940% | 7.5094% | 21.6003% | 49.7691% |
| slope per decade | - | 1.19 | 1.06 | 1.06 | 0.83 |

Fitting the straight part (k = 10 to 1,000) and extrapolating predicted about 62%
at k = 10,000. The measured value is 49.77%, so the tail slope drops from 1.06 to
0.83 nats per decade, as it must before saturating at 100%. The extrapolation was
12 points too optimistic; the measured number is the one to quote.

Practical reading: a two-stage pipeline keeping 10,000 candidates (0.21% of the
catalog) recovers half the targets, and co-click with the same budget recovers
44%. Neither is a clean win for the model.

## 5. The asymmetry that still matters for deployment

A co-click list is short by construction: 5.37 distinct successors per item on
average (max 566), so a window-5 query proposes roughly 5.37 x 5 = 27 real
candidates and then pads with the popularity order. Most of its R@1000 and
R@10000 therefore *is* the popularity baseline (36.8215% at R@10000), not
co-click signal. DiGR, by contrast, produces a genuinely model-ranked list at
every depth.

So the two methods are not the same product. If the downstream reranker can only
afford tens of candidates, co-click is the better and 3,000x cheaper choice. If it
needs a catalog-wide ranking with a long tail, only DiGR provides one. That was
the one claim left to DiGR, and it is a deployment claim, not a ranking claim.

## 6. TIGER: decisions carried forward

Confirmed by the project owner: TIGER uses the **direct SID as input** (not item
title text), and must include **the user feature block and the temporal token**.

Both already exist in the token row built by `preprocess_digr.py`, so the input
tensor is unchanged and the substitution is one of objective:

```
row = [sep][feature tokens][sep][temporal][sid x 5] [temporal][sid x 5] ...
```

- input: 137 tokens of the existing row, same as DiGR
- target: `catalog_sid_codes[target]`, 5 code tokens, teacher forced
- objective: `P(item) = P(c1) P(c2|c1) P(c3|c1c2) P(c4|c1c2c3) P(c5|...)`,
  cross-entropy on the 5 target positions only
- decode: trie-constrained beam over valid SID prefixes, since only 3,459,854 of
  256^4 = 4.29e9 four-code prefixes exist (0.08% valid)
- backbone: Qwen3-0.6B, unchanged, so DiGR vs TIGER is a single-variable ablation
- the level-5 suffix code (731 values) is a collision breaker with no content
  signal; 1,878,898 items (39%) share a four-code prefix. Keep it in the target
  and also report a 4-code diagnostic.

The bar TIGER has to clear is no longer DiGR: it is co-click at window 5, i.e.
**R@10 3.6011% and R@100 10.4492%**. Beam size caps the reachable k, so R@100 at
beam 100 is the headline metric.

## 7. Open items

- co-click at `--history_window 10` and `20`. 46 s each, and the sweep has not
  plateaued.
- B3 at the wide cut-offs, only if the negative result is going into the report.
- Check whether train and val share users (`users.npy` in both splits). If they
  do, both methods benefit but co-click benefits more directly from exact pair
  memorisation, which weakens the headline. Cheap to check, worth doing before
  anything is claimed publicly.
- `REF_POPULARITY_R10` is off by one row (1.3428% recorded vs 1.3438% measured);
  correct the constant.
- Not yet spent: more exposure (3 anchors/record = 2.48M rows instead of 1.72M),
  a larger backbone, or TIGER. TIGER first, because it tests the hypothesis that
  the additive score is the problem.

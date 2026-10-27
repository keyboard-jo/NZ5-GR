# DiGR vs baselines: consolidated results

Companion files: `docs/math.md` (the formulas and why each baseline behaves as it
does), `docs/data.md` (what the numbers are), `docs/problems.md` (what this table
means for the direction of the project).

All numbers are on the **same 98,304 val rows**, the **same val targets**, and the
**same full catalog of 4,783,154 items**. One target per row, so hit rate and
recall coincide. `MRR` is truncated at rank 10 everywhere.

| method | cost | R@1 | R@10 | R@100 | R@1000 | R@10000 | nDCG@10 | MRR |
|---|---|---|---|---|---|---|---|---|
| popularity (train target frequency) | < 1 s CPU | 0.4395% | 1.3438% | 4.7862% | 14.7288% | 36.8215% | 0.00809 | 0.00648 |
| B3 SID transition (first order, smoothing 0.5) | 3,805 s CPU, 8 workers | 0.0244% | 0.1801% | 0.9410% | 4.2236% | - | 0.00087 | 0.00060 |
| B4 co-click kNN, window 1, decay 0.5 | 4 s + 46 s CPU | 0.5686% | 2.5238% | 7.4646% | 18.5822% | 39.4318% | 0.01405 | 0.01065 |
| B4 co-click kNN, window 3, decay 0.5 | 4 s + 46 s CPU | 0.7355% | 3.3783% | 9.6130% | 21.9757% | 42.3777% | 0.01865 | 0.01405 |
| B4 co-click kNN, window 5, decay 0.5 | 4 s + 44 s CPU | 0.7853% | 3.6011% | 10.4492% | 23.6176% | 44.1905% | 0.01989 | 0.01499 |
| B4 co-click kNN, window 10, decay 0.5 | 4 s + 44 s CPU | 0.7985% | 3.6794% | 10.8531% | 24.8311% | 46.2148% | 0.02030 | 0.01529 |
| **B4 co-click kNN, window 20, decay 0.5** | **4 s + 43 s CPU** | **0.8057%** | **3.7069%** | **10.9873%** | **25.3021%** | **46.8882%** | **0.02044** | **0.01539** |
| DiGR 0.6B, 3 epochs, 300k sampled negatives | 4.6 GPU-h + 3.2 min eval | 0.7924% | 2.5940% | 7.5094% | 21.6003% | 49.7691% | 0.01570 | 0.01258 |
| HSTU tower (window 20, approximate actions) | 20 s GPU + 41 min I/O | 0.0458% | 0.4354% | 2.9063% | 13.5122% | - | 0.00201 | 0.00131 |
| item-embedding kNN, mean of last 1 clicked | 9 min GPU | 0.0173% | 0.5636% | 2.4638% | 8.1807% | - | - | 0.00151 |
| item-embedding kNN, mean of last 3 clicked | 9 min GPU | 0.0376% | 0.3560% | 2.2156% | 9.7646% | - | - | 0.00112 |
| item-embedding kNN, mean of last 5 clicked | 9 min GPU | 0.0539% | 0.3174% | 2.0742% | 9.6965% | - | - | 0.00110 |

DiGR is frozen at `digr_0p6b_e3_full_sampled300k/final`. Epoch 2 wrote a lower
val loss (11.2342 against 11.2431) but epoch 3 ranks better at every cut-off, so
`final` is the checkpoint behind the row above and no further DiGR runs are
planned.

co-click (window 20) over DiGR: R@1 **1.02x**, R@10 **1.43x**, R@100 **1.46x**,
R@1000 **1.17x**, R@10000 0.94x, nDCG@10 1.30x, MRR 1.22x.
DiGR over co-click (window 20): R@1 0.98x, R@10 0.70x, R@100 0.68x, R@1000 0.85x,
R@10000 1.06x.
co-click (window 20) over popularity: R@1 1.83x, R@10 2.76x, R@100 2.30x,
R@1000 1.72x, R@10000 1.27x.
DiGR over popularity: R@1 1.80x, R@10 1.93x, R@100 1.57x, R@1000 1.47x,
R@10000 1.35x.
DiGR over B3: R@1 32.4x, R@10 14.4x, R@100 8.0x.

Compute ratio DiGR : co-click = 4.6 GPU-h : 4 s = about **4,100x**.

## 1. The old popularity R@10 was wrong

The previously recorded popularity baseline was R@1 0.4395% / R@10 0.9644%.
Re-measured on the same rows it gives R@10 **1.3438%**. R@1 reproduces to eight
digits, ranks 2-10 do not, so the earlier figure came from a different frequency
vector. The old number is retired; every "lift over popularity" claim quoted
before this is 39% too generous at R@10: DiGR beats popularity 1.93x at R@10,
not 2.69x.

## 2. A co-click count table beats the model at every cut-off but the far tail

The published co-click run used window 1, the weakest form. Sweeping the window
changes the picture completely, and window 20 is where it closes:

| window | R@1 | R@10 | R@100 | R@1000 | R@10000 |
|---|---|---|---|---|---|
| 1 | 0.5686% | 2.5238% | 7.4646% | 18.5822% | 39.4318% |
| 3 | 0.7355% | 3.3783% | 9.6130% | 21.9757% | 42.3777% |
| 5 | 0.7853% | 3.6011% | 10.4492% | 23.6176% | 44.1905% |
| 10 | 0.7985% | 3.6794% | 10.8531% | 24.8311% | 46.2148% |
| 20 | 0.8057% | 3.7069% | 10.9873% | 25.3021% | 46.8882% |
| DiGR | 0.7924% | 2.5940% | 7.5094% | 21.6003% | 49.7691% |

Three things changed as the window widened.

**The R@1 tie is gone.** At window 5 co-click tied DiGR at rank 1 (0.7853%
against 0.7924%, 7 rows out of 98,304). At window 20 it wins there as well,
0.8057% against 0.7924%, so DiGR reproduces at exactly one cut-off out of the five.

**The margins widened.** Window 20 beats DiGR by 1.43x at R@10, 1.46x at R@100,
1.17x at R@1000, 1.30x at nDCG@10 and 1.22x at MRR. The earlier "DiGR sharpens
the top" reading was an artifact of comparing against the weakest baseline.

**The sweep has closed.** Window 5 to 10 added 2.2% at R@10, window 10 to 20 added
0.7%. The steps shrink by roughly 0.35x per doubling, which puts the asymptote
only about 0.015 points above window 20 (R@10 near 3.72%, R@1 near 0.814%).
Widening further costs another job for a movement smaller than one row in two
thousand.

DiGR's only remaining win is R@10000: 49.7691% against 46.8882%, a 1.06x edge in
the far tail, down from 1.13x against window 5. That is still the one band where a
model that scores all 4,783,154 items beats a table that can only propose a few
dozen candidates, but it is now the only one.

Why the far end of the window still matters despite `decay 0.5`: a successor seen
once at distance 10 contributes 0.5^10 = 9.8e-4, while one seen 566 times at
distance 19 contributes 566 x 0.5^19 = 1.1e-3. The tail keeps adding signal for a
handful of very common transitions, which is why window 20 still gained over 10
and why that gain is shrinking.

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
47%. Neither is a clean win for the model.

## 5. The asymmetry that still matters for deployment

A co-click list is short by construction: 5.37 distinct successors per item on
average (max 566), so a window-20 query proposes roughly 5.37 x 20 = 107 real
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

The bar TIGER has to clear is no longer DiGR: it is co-click at window 20, i.e.
**R@10 3.7069% and R@100 10.9873%**. Beam size caps the reachable k, so R@100 at
beam 100 is the headline metric.

## 7. Open items

- B3 at the wide cut-offs, only if the negative result is going into the report.
- Check whether train and val share users (`users.npy` in both splits). If they
  do, both methods benefit but co-click benefits more directly from exact pair
  memorisation, which weakens the headline. Cheap to check, worth doing before
  anything is claimed publicly.
- Not yet spent: more exposure (3 anchors/record = 2.48M rows instead of 1.72M),
  a larger backbone, or TIGER. TIGER first, because it tests the hypothesis that
  the additive score is the problem.
- The one embedding probe not yet run: the Stage 1 HSTU **query** vector instead
  of a pooled history. Section 8 tests the item tower's content geometry, which is
  not what the served retriever queries with. If the HSTU query also fails, the
  two-tower model is the thing to fix; if it succeeds, only the probe was wrong and
  the catalog vectors stay usable.

---

## 8. The embedding does not predict the next click

`Model Training/embedding_knn_nextclick.py`, 9 min on one H800, exact cosine over
the full 4,783,154 x 256 matrix, scored against the same val targets as everything
else. The query is the mean of the last k clicked item vectors, re-normalised.

| cut-off | popularity | embedding (last 1) | ratio |
|---|---|---|---|
| R@1 | 0.4395% | 0.0173% | 25x worse |
| R@10 | 1.3438% | 0.5636% | 2.4x worse |
| R@100 | 4.7862% | 2.4638% | 1.9x worse |
| R@1000 | 14.7288% | 8.1807% | 1.8x worse |

The item embedding loses to a frequency count at every cut-off. It is not noise,
it is the wrong quantity. 37.43% of the w1 top-10 slots are items that appear as a
train target somewhere, against a 10.22% base rate, so the embedding does retrieve
clickable items; it retrieves someone's clickable item, not this row's. R@1 of
0.0173% is 17 rows out of 98,304, because the nearest neighbour of the last clicked
item is usually a near-duplicate of that item.

Longer history helps deep cut-offs and hurts the top. w1 to w5 moves R@1000 from
8.18% to 9.70% and R@10 from 0.5636% to 0.3174%: mean pooling over more items blurs
the query into a genre centroid, which is the popularity smoother the R@1 column
says it is.

Two facts that make this a measurement rather than a bug: the target is already in
the row's own history for only 0.05% of rows, and the last clicked item is missing
from the catalog for 0.0000% of rows.

The SID is not the culprit, which is the point of `docs/sid_results.md`. Against the
same real targets, a depth-1 bucket holds 19,308 items and contains the target
18.3156% of the time (45x lift over 0.40% of the catalog); a depth-2 bucket holds
345.9 items and contains it 2.5167% of the time (348x lift over 0.0072%). The
depth-1 bucket at 19,308 candidates sits inside the range the embedding's own kNN
curve predicts for that budget (about 16% to 20%, log-linear extrapolation of
0.5636% at 10, 2.4638% at 100, 8.1807% at 1,000), so the code faithfully compresses
the geometry and loses nothing. The geometry itself is weak.

The deployment reading: the behaviour is not in the item vectors. A 25.7M-pair
co-occurrence count built in 4 seconds reaches 23.6176% at 1,000 candidates; the
codes need 19,308 candidates to reach 18.3156%, and 8.1807% is all the raw vectors
manage at 1,000. That gap is why DiGR, which reads the codes, cannot outrun a count
table.

---

## 9. The Stage 1 sequence tower is below popularity too

`Model Training/probe_hstu.py` asks the other half of the same question. Stage 1
exports item vectors from `ItemDNN` alone, so section 8 measures the item side. The
HSTU sequence encoder is where the behaviour was actually trained, and the probe
scores its user vector against the same full catalog on the same val rows.

Window 20 events, approximate actions, `bfloat16`, 98,304 rows, 12,716,408 tokens.
Cosine against a 4,783,154 x 256 item matrix.

| method | R@1 | R@10 | R@100 | R@1000 | nDCG@10 |
|---|---|---|---|---|---|
| HSTU tower | 0.0458% | 0.4354% | 2.9063% | 13.5122% | 0.2008% |
| popularity | 0.4395% | 1.3438% | 4.7862% | 14.7288% | 0.8090% |
| DiGR | 0.7924% | 2.5940% | 7.5094% | 21.6003% | 1.5700% |
| co-click, window 20 | 0.8057% | 3.7069% | 10.9873% | 25.3021% | 2.0440% |

The sequence tower loses to a frequency count at every cut-off. R@1 is 9.6x below
popularity and even R@1000 is 8% short, so this is not a badly calibrated good
model, it is a model with very little user-specific signal.

The shape inside the head says the same thing. HSTU keeps 3.2% of its top 1,000
inside its top 10, against 9.1% for popularity and 12.0% for DiGR. A flat head on
top of a collapsed R@1 is the signature of a broad popularity direction plus
noise, not of a sharp per-user ranking.

One bug was found and fixed before this number was recorded.
`HSTUSequenceEncoder` reads `x[:, lengths - 1]` as the query, training left-aligns
the sequence, and the first version of the probe right-aligned it, so padded rows
read a padding position. The fix (`to_chronological`) was uploaded and the run
reported here is the patched one, confirmed byte-identical to
`Model Training/probe_hstu.py`:

| | right-aligned | chronological | change |
|---|---|---|---|
| R@1 | 0.0458% | 0.0458% | none |
| R@10 | 0.4283% | 0.4354% | +1.66% relative |
| R@100 | 2.8798% | 2.9063% | +0.92% relative |
| R@1000 | 13.3931% | 13.5122% | +0.89% relative |

The bug cost about 1.7% relative at R@10 and nothing at R@1, because the pooled
state at a padding position still aggregates the valid prefix through attention,
and only rows with fewer than 20 events were affected. The value of the rerun is
not the delta. It is that the weakness can no longer be attributed to the probe.

Two independent probes now agree. Item vector as the query gives 0.5636% R@10.
User vector as the query gives 0.4354%. Both are below the uninformative baseline
of 1.3438%, on the same geometry that `docs/sid_results.md` shows the quantiser
reproduces faithfully (depth-1 bucket recall 18.3156% matches the raw kNN curve).
So the weakness is upstream of the SID, and it is in both towers.

Diagnostics from the same run: 0.0000% unmapped history events, so the history
decoding is clean, and 0.1455% of rows already contain the target inside their
20-event window, which is the trivially recoverable repeat case.

---

## 10. The official candidate pool, and why it is not the evaluation space

The TencentGR dataset paper describes the competition protocol. Evaluation uses a
single global candidate set of 660,000 de-duplicated ads, shared by every user.
Every ground-truth target is guaranteed to be in it, and non-target items are
sampled so that about 40% of the pool appears as a target for at least one user.
The metrics are HitRate@10 and NDCG@10, and the leaderboard score is
`0.31 * HitRate@10 + 0.69 * NDCG@10`. K=100 was tried and dropped for giving less
separation. The 10M track uses a 3,637,720-ad pool and weighted metrics with a
conversion weight of 2.5.

`data/candidate.parquet` is that pool: 660,000 rows, 660,000 unique `item_id`,
660,000 unique `retrieval_id`, and `retrieval_id` runs 0 to 659,999 with no gaps,
so it is a dense enumeration of the pool and nothing more. It carries no labels.

Measured against our own artefacts:

| quantity | value |
|---|---|
| pool items that map into `data/exported_indexer/i.json` | 511,029 (77.43%) |
| pool items with no entry in `i.json` or `item_feat.parquet` | 148,971 (22.57%) |
| of those, present in `data/mm_emb_82_1024.parquet` | 148,947 (99.98%) |
| val targets inside the pool | 79,064 of 98,304 (80.43%) |
| train targets inside the pool | 1,298,907 of 1,719,430 (75.54%) |

The unmapped 149k are not unembeddable. `mm_emb_82_1024.parquet` has 4,896,667
rows, a superset of the 4,783,154 in the index. They have no SID because `i.json`
and `item_feat.parquet` were both built from the items that appear in at least one
sequence in `seq.parquet`, and these are ads that were never shown to these 1M
users.

We do not evaluate on the pool, for three reasons.

1. The ceiling is 80.43%. Only 79,064 of our 98,304 val targets are inside it, and
   all of those are mappable, since val targets are catalog items by construction.
   A perfect method would score at most 80.43% HitRate@10, which turns a ranking
   comparison into a statement about pool coverage.
2. It measures coverage as well as ranking. 22.57% of the pool cannot be produced
   by any method built on our index universe, so the three arms would be scored
   partly on how much of the pool they can represent.
3. It would not be comparable anyway. We do not have their test users or their
   labels, so a pool score of ours would not sit next to the published numbers.
   Our val split is a split of the provided training users.

What we do take from the protocol is the reporting convention, since it costs
nothing and it is the one thing that is comparable in definition. The combined
score on our full-catalog space:

| method | HR@10 | NDCG@10 | 0.31*HR + 0.69*NDCG |
|---|---|---|---|
| co-click, window 20 | 3.7069% | 2.0440% | 0.0256 |
| co-click, window 5 | 3.6011% | 1.9890% | 0.0249 |
| DiGR | 2.5940% | 1.5700% | 0.0189 |
| popularity | 1.3438% | 0.8090% | 0.0097 |
| HSTU tower | 0.4354% | 0.2008% | 0.0027 |

K=10 stays the headline, which is what we already lead with. The cut-offs at 100,
1,000 and 10,000 stay as diagnostics and are labelled as non-standard.

If a leaderboard-comparable number is ever wanted, the route is to assign codes to
the 148,971 missing pool items with the frozen Stage 1 tower and the frozen
RQ-VAE. Existing SIDs would not change, because code assignment is per item and
the codebooks are already fitted, and DiGR scores an item through its codes alone,
so the frozen checkpoint would apply with no retraining. That is recorded as
deferred, not planned.

---

## 11. A wider codebook makes buckets sharper, and that was not expected

The SID release plan (`docs/plan_v2.md`) was built on a prediction: a wider codebook
would raise the head's parameter count while leaving bucket quality flat, because the
embedding geometry already sits at its packing bound. The prediction was testable and
`Plan 2/sid_diagnose.py` tested it. It was wrong.

Same 98,304 val rows, same 4,783,154-item catalogue, same instrument. Only the
quantiser changes, from 4 x 256 plus the collision suffix to 3 x 1,024, both fitted
on the same frozen 256-d Stage 1 embedding.

| depth | 4 x 256 + 731 | 3 x 1,024 |
|---|---|---|
| bucket size | 18,684.20 | 4,671.05 |
| recall | 18.3156% | 12.3840% |
| lift over its own base rate | 46.9x | 126.8x |
| depth-2 bucket size | 345.9 | 67.22 |
| depth-2 recall | 2.5167% | 1.1190% |
| depth-2 lift | 1,437.6x | 5,968.8x |
| depth-3 recall | 0.5483% | 0.19836% |
| depth-3 lift | 8,257.6x | 6,732.1x |

Raw recall falls because the bucket shrank 3.44x. Lift is size-normalised, so it is
the comparable column, and it rises 2.7x at depth 1 and 4.2x at depth 2. The finer
partition puts the true next item in a much smaller and much more label-pure cell.

Both quantisers are healthy, so this is not a codebook-collapse artefact. The frozen
4 x 256 SID runs at 7.87 to 7.98 bits of 8 with all four codebooks 100% utilised. The
3 x 1,024 SID runs at 9.8544, 9.8285 and 9.9085 bits of 10, also all 100% utilised.
Its occupancy is 1,024 tuples of 4,671.05 items at depth 1, 533,417 of 8.97 at depth
2, and 3,393,796 of 1.41 at depth 3, where 29.05% of items share a tuple but the
largest tuple holds 372 items, so a fourth 1,024-wide level cannot collide at all.

The reasoning error is worth recording because it shaped the whole plan. The packing
bound limits how well cells separate points in the embedding's own metric. It says
nothing about how well they separate the click. A k-means partition with four times
as many cells tracks the density modes more closely, so cells get more label-pure
even when the points inside them are no better separated. The plan reasoned about the
wrong quantity and then presented the conclusion as arithmetic.

The token budget is a side effect worth noting. The largest depth-3 tuple holds 372
items, well under 1,024, so a fourth 1,024-wide level cannot collide and the five
level SID that exists only because 256-wide codes collide at depth 4 is unnecessary.
At three codes an item is 3 tokens instead of 5 and an event is 4 tokens instead of
6, so 221,500,536 train tokens fall to about 153.5M at the same event count, which
leaves room for the per-event action and gap tokens.

What this points at is items per code, which is the number that has to match the
DiGR paper for its accuracy to be reachable. The paper's own catalogues sit at 39 to
195 items per level-0 code. Ours started at 18,684, and 3 x 1,024 only takes that to
4,671. Two codebooks of 16,384 take it to 292, which is the regime the paper was
evaluated in and the configuration OnePiece ran on this data, so that is the next SID
rather than another 1,024-wide variant.


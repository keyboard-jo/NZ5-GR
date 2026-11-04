# The SID: what the diagnosis says

Two vintages exist and both are still in use. This file distils the diagnosis of each
from `archive/sid_diagnostics.md`, `archive/sid_results.md` and `archive/plan_v2.md`,
keeps the numbers, drops the derivation, and records what the instrument cannot
establish. The instrument itself is `Plan 2/sid_diagnose.py`, which is unchanged
across both runs, so the two columns below are directly comparable.

Everything here is measured on the same 98,304 val rows.

---

## 1. The two versions

| | v1 | v2 |
|---|---|---|
| geometry | 4 x 256 + 731 collision suffix, 5 levels | 2 x 8,192, 2 levels |
| rows / parameters | 1,755 / 1,797,120 | 16,384 / 16,777,216 |
| parameters per catalogue item | 0.376 | 3.51 |
| items per level-0 code | 18,684 | 584 |
| depth-1 lift | 46.9x | 463.3x |
| peak lift | 8,257.6x at depth 3 | 7,880.6x at depth 2 |
| depth-1 candidate bucket | 19,308 | 745 |
| collision share | **none**: 1.000 items per full 5-code tuple, injective. The 1.38 figure is per 4-code **prefix**, of which 39.28% of items share one (27.67% excess), largest 731 | 1.59 items per 2-code tuple; 53.51% share one (37.22% excess), largest 315 |
| tie ceiling, R@1 / R@10 | none needed, injective at depth 5 (100.00% / 100.00%) | 53.57% / 93.20% over val targets |
| event stream | 9.11 feature tokens per row, no action token, 64 gap bins | 4 tokens per event, action token, 128 gap bins |
| what it produced | DiGR v1 R@10 2.5940% | DiGR v2 R@10 3.6570% |

The catalogue is 4,783,154 items in both cases. The last row is the only one that mixes
two changes, and Section 5 treats it as such.

### Why a collision happens at all

The quantiser assigns codes by clustering. Level 0 partitions all 4,783,154 items into 8,192
cells, and the next level partitions each of those cells again. An item's tuple is the list of
cells it falls into, so two items collide when every level puts them in the same cell.

Nothing is broken when that happens. Two levels give `8,192^2 = 67,108,864` possible tuples and
the catalogue only occupies 3,002,708 of them, but the assignment is not spread evenly. Cells go
where the items are, and the objective is to place each cell's own vector close to its members,
not to keep members apart. Dense regions of the embedding get fat cells and sparse regions get
thin ones. If those 3,002,708 cells had equal sizes, 7.13% of items would share one; the measured
figure is 53.51%, 7.5 times higher, and the gap is that bias.

The count is also forced before any question of how the cells were chosen. Fitting 4,783,154
items into 3,002,708 occupied cells requires at least 1,780,446 of them to share a cell with
something else. Only more cells removes that, which is why the fix is a third level rather than
a better tie-break. What a tie-break does is recover the positions the collision costs inside
the codebook that exists.

### Every level, measured

The instrument reports one row per depth, and this is the full record for both vintages. It is
read straight from the json the quantiser wrote, with `SID Diagnostics/diagnose_sid.py`, so it
involves no preprocessing and no model, and it bounds every arm equally.

| SID | depth | level size | codes used | entropy (bits) | entropy / log2 size | buckets | items per bucket | largest bucket | `R@1` ceiling | `R@10` ceiling | lift | lift/1k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `2x8192` | 1 | 8,192 | 8,192 | 12.8110 | 0.9855 | 8,192 | 583.88 | 3,838 | 0.17% | 1.71% | 463.3x | 7.589 |
| `2x8192` | 2 | 8,192 | 8,192 | 12.7668 | 0.9821 | 3,002,708 | 1.59 | 315 | 62.78% | 95.96% | 7,880.6x | 42.486 |
| `4x256+731` | 1 | 256 | 256 | 7.9243 | 0.9905 | 256 | 18,684.20 | 39,883 | 0.01% | 0.05% | 46.9x | 0.949 |
| `4x256+731` | 2 | 256 | 256 | 7.8739 | 0.9842 | 57,122 | 83.74 | 5,086 | 1.19% | 9.93% | 1,437.6x | 7.276 |
| `4x256+731` | 3 | 256 | 256 | 7.9337 | 0.9917 | 1,506,041 | 3.18 | 1,405 | 31.49% | 76.14% | 8,257.6x | 19.946 |
| `4x256+731` | 4 | 256 | 256 | 7.9804 | 0.9976 | 3,459,854 | 1.38 | 731 | 72.33% | 96.42% | 4,505.0x | 29.389 |
| `4x256+731` | 5 | 731 | 731 | 1.7644 | **0.1855** | 4,783,154 | 1.00 | 1 | 100.00% | 100.00% | 827.2x | 17.293 |

Both ceilings in this table average over the **catalogue**. Averaged over the val targets, which
is the population a row metric averages over, the `2x8192` depth-2 pair is `53.57%` / `93.20%`,
with mean group 6.39 rather than 4.71. See section 4.

Bits: `2x8192` spends 26.00 against the 22.19 an identity needs, so 3.81 redundant;
`4x256+731` spends 41.51, so 19.32 redundant, and it is injective at depth 5. Both spend
their budget densely at every level except the last of `4x256+731`, where 731 dense values
hold 1.7644 of 9.5137 bits, the `--add_unique_id` collision-rank signature at 18.5%.

Two notes on reading it. `lift` is bucket recall divided by the bucket's share of the
catalogue, so it rises mechanically as buckets shrink and is **not** comparable across
widths; comparing vintages needs `lift/1k`, `recall / (query bucket / 1000)` in percent,
which is only meaningful at a depth whose bucket is not far from 1,000 items. And a ceiling
is a hard cap, not a measurement: two items sharing a tuple receive the same score from any
scorer that reads only the SID, so a group of size `g` cannot be told apart and caps `R@k`
at `min(1, k/g)` averaged over the split.

The four conclusions drawn from this table are in `README.md` 1.9. If the two ever disagree,
derive from the json again rather than reconciling the text.

---

## 2. What the diagnosis asks

Four questions, in the order the instrument asks them. Only the second and fourth ever
failed here.

1. **Coverage.** Are any codes dead? Are bits being spent on nothing? Both vintages
   passed. v2 used 8,192 of 8,192 codes at both levels, at 12.8110 and 12.7668 bits of
   13, with perplexity 7,186.2 and 6,969.3.
2. **Is a code informative about behaviour?** Measured as lift, never as raw recall.
   Lift is bucket recall divided by the bucket's share of the catalogue, so shrinking a
   bucket does not earn credit for free.
3. **Does a level earn its slot?** The marginal lift from adding it.
4. **Is the partition fine enough for the head?** Items per code, held against the DiGR
   paper's own catalogues. Nothing else in this file is a fair measure of resolution,
   because the head pools codes and carries no item-specific information by design, so
   the only thing separating two items is the difference between sums of shared rows.

---

## 3. Diagnosis of v1

**Bucket structure.** 256 groups at depth 1 holding 18,684 items each, 57,122 at depth
2 (83.7 items), 1,506,041 at depth 3 (3.18), 3,459,854 at depth 4 (1.38). Resolution
explodes at level 3, from 57 thousand groups to 1.5 million.

**The codes track the geometry, level by level.** Cosine similarity between two items
sharing a prefix: 0.6705 at depth 1, 0.8617 at depth 2, 0.9287 at depth 3, 0.9501 at
depth 4, against a random baseline of 0.084 and a mean nearest-neighbour cosine of
0.9290. A shared depth-3 prefix is as similar as a nearest neighbour, so level 4 is
already resolving below the neighbourhood scale.

**The crowding is the collapse, not a defect on top of it.** For 4,783,154 points spread
evenly on a sphere, the expected nearest-neighbour cosine at the measured participation
ratio of 16.8 is 0.9286. The measured value is 0.9290. The space is as packed as its
dimensionality forces, so there is no spare structure for a quantiser to recover.

**The quantiser lost nothing.** A depth-1 bucket contains 71.158% of the embedding's own
top 10, and 18.3156% of the real next click at 19,308 candidates (lift 46.9x). Both are
far above a random bucket, 45x for the second. So the codes were a faithful compression
and the embedding was the weak half, which is the conclusion `embedding_weakness.md`
reaches from the other direction.

---

## 4. Diagnosis of v2

**Resolutions.** 584 items per level-0 code, depth-1 lift 463.3x, a 745-item query
bucket. At a matched candidate budget the ordering holds and widens: recall per 1,000
candidates is 0.949 for 4 x 256, 2.204 for 3 x 1,024, and 7.589 for 2 x 8,192, which is
8.0x the original at equal budget.

**The quantiser stayed healthy.** 8,192 of 8,192 codes active at both levels, best val
loss 0.001523, 516 s for the whole pipeline.

**Collisions did not improve, and the prediction of how many was wrong.** 46.49% of items
hold a two-code tuple of their own (2,223,538 of them), so 53.51% of the catalogue sits in a
group of two or more, largest group 315. The uniform birthday bound for 67.1M cells predicts
7.13%, so the measured rate is 7.5x the prediction. k-means cells cluster where the items
are instead of spreading evenly.

**The reading that collisions cost nothing is wrong, and it was mine.** Two items with the
same codes get the same score from any scorer that reads the SID, so a tie group of size `g`
caps recall at `1/g` at rank 1 and `min(1, k/g)` at rank `k`. Measured for 2 x 8,192 the mean
tie group over val targets is 6.39 (median 2, p90 14, largest 239), so the ceilings are 53.57%
and 93.20%. DiGR sits at 1.05% and 3.66%, and the old conclusion from that gap was that ties
cost nothing yet.

The averaging base matters more than it looks. The table above averages over the **catalogue**,
every item counting once, which gives mean group 4.71, p90 8, largest 315 and ceilings 62.78% /
95.96%. Measured instead over the 98,304 **val targets**, the same codebook gives 6.39 / 2 / 14 /
239 and 53.57% / 93.20%, because a row is more likely to land on a packed tuple than a random
item is. The row base is the one that bounds a metric averaged over rows, so the row figures are
the ones to quote, and the archived `53.57% / 93.20%` was never a different quantiser run: it
was this file on the other base.

Two things are wrong with it. A high ceiling is not health: a hash of the item id is
injective and its ceiling is 100%, and `4x256+731` is already that case here, because depth 5
is injective by construction, carries a 100.00% ceiling, and drops lift to 827.2x.
Distinguishability bought with levels that carry no information is how the narrow codebook
became the worse representation.

And the real reason to park collisions is not the distance to the ceiling. It is the rows
where the tie cannot operate at all: 36.64% of val targets sit alone in their tuple, so no
tie-break applies, and their `R@1` is 2.43% against a ceiling of 100%. Where the model has
every advantage the ceiling can offer it still misses 97.6% of the time. That puts the loss in
the ranking, which is the item representation, and not in the tie mechanism.

So the ceiling is half of a joint test and on its own it says nothing about whether a
representation is good enough. Read it beside lift, as a rough guide rather than a threshold:

| measured `R@k` against its ceiling | lift | reading | what to change |
| --- | --- | --- | --- |
| far below, under roughly 20% | low | coarse and uninformative | the codes carry almost nothing; change the objective, or widen |
| far below | high | informative, and the head cannot exploit it | the codes are not the constraint, the scorer is |
| close, over roughly 80% | high | at the resolution limit | a finer SID buys the remainder |
| close | low | injective and useless, which is a hash | fewer or re-aimed codes, not finer ones |

`2x8192` is the second row today: lift 7,880.6x at depth 2, with `R@10` at 3.81% of its
ceiling. The codes carry information and the ranking is not using it, which is an argument
about the head and the query side. It is not an argument that the representation is adequate.

One consequence for `Plan 3/`: its injective-counterfactual column answers "what would a free,
perfect tie-break buy", and treats the extra resolution as carrying the same information as
the levels already there. That makes it an upper bound rather than a forecast, and the narrow
codebook is the measured test of it.

The 3 x 1,024 figures in the archived table (mean group 5.55, R@1 64.10%, R@10 94.85%)
predate this measurement and that json is not in the repo, so they are quoted as recorded
rather than re-derived.

**Where the collisions cost you, measured.** Wording aside, the size of the effect is now a
measurement rather than an implication. `Plan 3/region_rank.py` scores one checkpoint twice,
once over the 4.78M items and once over the 3.00M distinct tuples, both through the model's
own `item_logits`, so the two columns differ only in how many positions a fat tuple occupies.
On **DiGR v2** (`digr_v2_sid2x8192_e3_sampled300k/best`, the arm whose 3.6570% is the
leaderboard number) the item column reproduces the `--dump_ranks` file on 100.00000% of the
98,304 rows, so the left side of the table is the leaderboard measure itself and not an
approximation of it.

| target's tuple size | rows | share | `g` | item `R@1` | item `R@10` | item `R@100` | injective `R@1` | injective `R@10` | injective `R@100` | inflation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 36,017 | 36.64% | 1.00 | 2.427% | 5.703% | 12.533% | 2.427% | 6.713% | 15.573% | 644.2 |
| 2 | 17,275 | 17.57% | 2.00 | 0.590% | 3.757% | 10.790% | 1.065% | 4.758% | 13.893% | 689.0 |
| 3-4 | 16,451 | 16.73% | 3.38 | 0.310% | 3.210% | 10.176% | 0.754% | 4.437% | 13.519% | 705.0 |
| 5-8 | 12,441 | 12.66% | 6.17 | 0.040% | 1.953% | 10.345% | 0.338% | 3.657% | 14.267% | 767.2 |
| 9-16 | 7,639 | 7.77% | 11.68 | 0.039% | 1.217% | 9.085% | 0.380% | 3.207% | 13.261% | 790.4 |
| 17 and over | 8,481 | 8.63% | 39.66 | 0.024% | 0.330% | 8.254% | 0.165% | 3.219% | 13.595% | 848.6 |
| all rows | 98,304 | 100% | 6.39 | 1.0549% | 3.6570% | 10.9182% | 1.2889% | 5.0283% | 14.4185% | |

| metric | measured | same scores, injective | gain |
| --- | --- | --- | --- |
| `R@1` | 1.0549% | 1.2889% | 1.22x |
| `R@10` | 3.6570% | 5.0283% | 1.37x |
| `R@100` | 10.9182% | 14.4185% | 1.32x |
| `R@1000` | 27.6622% | 33.9783% | 1.23x |

`inflation` is the mean of `item rank − region rank` within the band, with both capped at
`kmax + 1`, so the cap censors it heavily and it should be read as an ordering hint rather
than a count. Ten rows (0.010%) report a tuple rank larger than their item rank, which cannot
happen if both count the tuples above; the cause is two distinct tuples scoring
bit-identically, which the two `topk` calls order differently.

Three things follow. The effect is real and it is **not** the 9.3x that the singleton
subpopulation suggests, because that comparison confounds "no ties" with "those rows are
easier"; the controlled question, holding the scores fixed, is 1.37x at `R@10`. It is also
uneven: the 36.64% of rows with a unique tuple reach 2.427% `R@1` on merit, while the 8.63%
in tuples of 17 or more reach 0.024%, and the 60.98% of rows with at least one tuple ranked
above the target is where the whole loss lives. And it decays with depth, 1.37x at `R@10`
down to 1.23x at `R@1000`, which is the signature of a positional penalty rather than a
missing signal.

So "counting" is worth about 1.3x at the head, conditional on the extra resolution carrying
information. That is smaller than the 1.57x the codebook already bought, and much smaller than
the 2.3x to 3x the direction needs. It is a remainder to collect, not the big move, and it
belongs to a third code from a finer quantiser rather than to a collision index, since an
index in catalogue order is unlearnable and v1 carries that level already.

**What it costs in training.** The measurement above is about positions in a list. There is a
second cost, in the loss, and it does not appear as a ratio on any metric.

If `g` items share a tuple they share a score, so the model cannot put probability on one of
them. The best it can do is give the group some mass and let the target take `1/g` of it, which
means its loss cannot fall below `log g` however good the ranking is. Averaged over the 98,304
val targets that floor is **1.0450 nats**, so **9.9683 is the best these codes allow** against
a measured 11.0133, and 23.9% of the 4.3677-nat margin is unreachable by construction.

Two things follow. Losses are not comparable across SIDs: v2 at 11.0133 against v1 at 11.2431
reads as 0.23 nats of improvement, and on equal footing it is 1.2748, because v1 pays no tax.
And the gradient can only move whole tuples. Raising the target's score raises its tie-mates
with it, so the model learns to rank blocks and cannot learn to rank inside one. On 17.8% of
training rows a tie-mate appears among the 300,000 sampled negatives, and there the positive and
that negative push the same code rows in opposite directions.

The injective column is an upper bound on what any tie mechanism inside the existing codebook
could recover, not an estimate of it. Injective ranking counts one position per tuple above
the target, while a perfect intra-block tie break would still count every member of every
tuple above it, so the true tie-break ceiling sits below these numbers. On arm a the same
measurement gave 1.18x at `R@1` and 1.34x at `R@10`, so the shape does not depend on which
two-level checkpoint is used.

**The flat-lift prediction failed twice, and the failure is instructive.** The original
argument was geometric: 4.78M points of effective dimension about 17 have a bounded
number of well-separated cells, so 4x the cells buys about `4^(1/17)`, an 8.6%
improvement, and the measured 1-NN cosine already sat at the packing bound so there was
nothing left to take. Measured instead: 46.9x to 126.8x to 463.3x, a curve that
steepened rather than flattened. The error was moving from geometry to labels. The
packing bound constrains how well cells separate points in the embedding's own metric
and says nothing about how well they separate the click. A k-means partition with four
times the cells tracks the density modes more closely, so cells become more label-pure
even though the points inside them are no better separated.

So width is the main lever on how much a SID says about an item, and it was the first
thing in this project to move bucket quality at all.

---

## 5. What the diagnosis cannot tell you

Three limits, all of which matter when these numbers get cited.

**It judges a SID against the geometry, not the click.** The 71.158% in Section 3 and
the 18.3156% are the same code measured against two different ground truths: the
embedding's own top 10, and what the user actually clicked next. Only the second is a
behavioural claim, and it is 3.9x smaller. Neither says the SID is predictive; they say
it is a faithful compression of whatever the embedding contains.

**It does not establish that the embedding contains behaviour.** The behavioural probe
put the embedding at 0.5636% R@10 against the real next click, below popularity's
1.3438%. A faithful compression of a content code is still a content code.

**It has no attribution between width and the event stream.** v2 moved the codebooks, the
action token and the gap bins from 64 to 128 at the same time, so DiGR's 2.5940% to
3.6570% is the sum of those three and not the width. The SID-only number is the lift
table, and lift is measured with no model in it at all.

Two further caveats. 2 x 16,384 is an untested prediction, not a result: it was ranked
above 2 x 8,192 in the plan, and the axis was closed by decision rather than by observed
saturation. And the pre-committed gate that required depth-2 collision share under 1%
was retired, because it cannot be passed at two levels at any width; the tie ceiling
replaced it.

---

## 6. What to carry forward

- **Items per code is the number to quote, not bits.** Ours is 584 at v2. The DiGR paper's
  catalogues sit at 39 to 195 items per level-0 code, and that gap, not the vocabulary
  bits, is the scale difference between its numbers and ours.

  **Provenance of that band.** It is derived, not quoted. The paper (`arXiv:2602.07847`)
  says the SIDs come from RQ-VAE and never states the codebook width or the number of
  levels: a full-text search for "256", "codebook" and "levels" returns nothing, and its
  Table 1 is statistics only. The 39 to 195 figures come from dividing its three catalogue
  sizes by 256, the width TIGER's RQ-VAE uses. The arithmetic fits exactly (Musical
  Instruments about 10k items gives 39, Arts and Crafts about 28k gives 109, Video Games
  about 50k gives 195), so 256 is almost certainly right, but anybody checking the paper
  will not find it there. Quote the band as "derived from their catalogue sizes assuming
  256-wide codebooks", and lean on the catalogue size for the scale argument, which needs
  no assumption: their catalogues are 10k to 50k items against our 4,783,154.
- **The tie ceiling is half a check, and lift is the other half.** A ceiling cannot separate a
good representation from a hash, because a hash is injective. Read the two together: high
lift with a distant ceiling means the scorer is the constraint, and high lift with a close
ceiling means the codes are. The rows where no tie can operate are the cleanest single test,
because they remove the tie mechanism instead of arguing about it.
- **The width axis is closed by decision.** The next SID question is not wider codes but
  behaviour-aware codes, and that work is bounded by the rules in `README.md` 2.7: the
  representation stays a function of content at serving time, so a co-occurrence vector
  cannot be the code space while a content encoder that ranks better can.

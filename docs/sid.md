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
| collision share | 27.67% share a 4-code prefix | 53.51% share a 2-code tuple, largest 315 |
| tie ceiling, R@1 / R@10 | not computed | 53.57% / 93.20% |
| event stream | 9.11 feature tokens per row, no action token, 64 gap bins | 4 tokens per event, action token, 128 gap bins |
| what it produced | DiGR v1 R@10 2.5940% | DiGR v2 R@10 3.6570% |

The catalogue is 4,783,154 items in both cases. The last row is the only one that mixes
two changes, and Section 5 treats it as such.

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

**Collisions did not improve, and the prediction of how many was wrong.** Only 62.78% of
items hold a two-code tuple of their own, so 53.51% of the catalogue sits in a group of
two or more, largest group 315. The uniform birthday bound for 67.1M cells predicts
7.13%, so the measured rate is 7.5x the prediction. k-means cells cluster where the items
are instead of spreading evenly.

**That turned out not to matter, and the tie ceiling is why.** Two items with the same
codes get the same score from any scorer that reads the SID, so a tie group of size `g`
caps recall at `1/g` at rank 1 and `min(1, k/g)` at rank `k`. Measured mean tie group is
5.55 for 3 x 1,024 and 6.39 for 2 x 8,192, capping R@1 at 64.10% and 53.57% and R@10 at
94.85% and 93.20%. DiGR sat at 0.79% and 2.59% on the v1 SID, so the ceiling is not
remotely binding and the remaining collisions cost nothing yet.

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
- **The tie ceiling is the check to run on any future SID.** One run of the instrument
  prints it, and it is the only bound that does not require training a model to find.
- **The width axis is closed by decision.** The next SID question is not wider codes but
  behaviour-aware codes, and that work is bounded by the rules in `README.md` 2.6: the
  representation stays a function of content at serving time, so a co-occurrence vector
  cannot be the code space while a content encoder that ranks better can.

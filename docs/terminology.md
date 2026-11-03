# Terminology

The vocabulary this project uses, so that two documents cannot quietly mean different things
by the same word. Definitions only; the numbers live in [`README.md`](README.md), the metric
construction in [`evaluation.md`](evaluation.md), the codes in [`sid.md`](sid.md).

Each entry gives the term we use, what it means, and the words we do **not** use for it,
because they already mean something else here.

## 1. The prediction task

**Row**:
One example: a user's history cut at an anchor, plus the single item they acted on next.
_Avoid_: sample, instance, record (a record holds several rows)

**Anchor**:
The event in a user's history where a row is cut; everything before it is the history and the
next held-out click is the target.
_Avoid_: cut point, split point

**Target**:
The single item a row asks about. Every metric here is computed on one target per row.
_Avoid_: label (labels are the TIGER code sequence), positive (there is exactly one)

**Catalogue**:
The set of items a method may return, 4,783,154 in the two production SIDs.
_Avoid_: corpus, inventory

**Cold item / tail item**:
An item that is never a training target. Cold describes the item's standing in the training
data, not anything about its content.
_Avoid_: rare item, long-tail item (long tail is the shape of a distribution)

**Candidate pool**:
The 660,000-item shortlist the competition provided. Retired here, because scoring against a
shortlist flatters every method.
_Avoid_: candidate set, retrieval pool

**Val row**:
One of the 98,304 rows every arm is scored on. The same rows, in the same order, for every
arm.
_Avoid_: test set (there is no separate test split)

## 2. The codes

**SID**:
Semantic ID: a short sequence of codebook indices that stands in for an item. An item is
identified only by its codes; nothing else about it is visible to a method at serving time.
_Avoid_: token (a token is a position in the model's vocabulary; a code becomes a token after
an offset is added), hash, cluster id

**Codebook / level**:
One codebook is one quantiser layer with a fixed number of entries; a level is that layer's
position in the sequence, level 0 first.
_Avoid_: layer (reserved for the model), stage (reserved for the two-stage pipeline)

**SID v1**:
The 5-level codebook `4 x 256 + 731`. Level 4 is a collision rank, so a full tuple is unique
and identifies exactly one item. Its four-level prefixes are what it shares: 3,459,854 of
them, a mean of 1.3825 items per four-prefix.
_Avoid_: the old SID, the fine SID

**SID v2**:
The 2-level codebook `2 x 8192`. A tuple holds 1.593 items on average with a 315-item worst
case, so it identifies a neighbourhood rather than an item.
_Avoid_: the new SID, the wide SID, the 3x1024 SID (a different, never-deployed candidate)

**Tuple**:
The exact sequence of codes an item carries, and therefore the finest thing a code-based
method can distinguish. Two items in one tuple are indistinguishable to DiGR and to TIGER.
_Avoid_: cell, bucket, region, cluster (these four words all mean tuple here, and using them
interchangeably hides which level of the tree is being discussed)

**Tuple multiplicity**:
How many items share a tuple. The catalogue average is 1.593 for v2 and **1.000 for v1**,
because v1's last level is a collision rank. The comparable v1 figure is items per *four*-level
prefix, 1.3825.
_Avoid_: group size (a tie group is a multiplicity class, not a tuple)

**Collision / tie group**:
A tuple holding more than one item, and the set of items inside it. 53.51% of v2 items sit in
a collision.
_Avoid_: clash, duplicate, merge

**Injective**:
A codebook whose tuples are all singletons, so codes name items one-to-one. v1 is injective at
depth 5; v2 never is.
_Avoid_: unique, perfect (unique describes the tuple, not the codebook)

**Tie ceiling**:
The recall a method would reach if it ranked items inside a collision at random. It is the
value a metric would show for a perfect tuple ranker with an arbitrary item ranker.
_Avoid_: group bound, maximum recall

**Suffix level**:
A final level added only to separate otherwise identical items, so it carries no meaning. v1's
731-wide level 4 is one.
_Avoid_: unique-id level, disambiguator

**Coverage / utilisation**:
The share of a codebook's entries that any item uses, and how evenly. A level that is 98% used
but at 98% of its own entropy is spending bits to say little.
_Avoid_: fill rate, occupancy (occupancy is the per-bucket item count)

**Bits spent / bits needed**:
`sum(log2 size)` over levels, against `log2(catalogue)`. The gap is redundancy: v2 spends 26.00
bits to name 22.19 bits of item.
_Avoid_: information content, entropy (entropy is one level's, not the sum)

## 3. The two methods

**DiGR**:
A method that gives each item one vector, the mean of its code rows, and scores it with a
single dot product against the query state. It scores the whole catalogue and returns a total
order over all 4.78M items.
_Avoid_: the linear model, the embedding model

**TIGER**:
A method that scores a code sequence one level at a time, each level conditioned on the ones
above it, and returns only what its beam names.
_Avoid_: the generative model, the autoregressive baseline

**Factorisation**:
The property being compared: whether a method can express item-item structure that a single
dot product cannot, by combining codes rather than averaging them.
_Avoid_: interaction (reserved for the ablation's non-additive term), decomposition

**Prefix-free / prefix-conditioned**:
Two oracle settings for TIGER. Prefix-free predicts all codes from the history alone, so the
levels are independent. Prefix-conditioned feeds the true earlier codes back, so each level
predicts what the previous level did not already say.
_Avoid_: teacher forcing (that is the training mechanism, not the arm), autoregressive vs not

**Beam / beam width**:
The beam keeps the W highest-scoring prefixes at every level and expands them, so W is the
number of candidates carried between levels. Beam 20 returns 20 tuples, which is between 28
and 192 items depending on the SID.
_Avoid_: top-k (that is an item cut-off), search width

**Prefix tree**:
The structure that limits a beam to code sequences the catalogue actually contains, so no
named tuple is empty.
_Avoid_: trie (true but opaque), index

**Baseline B3 / SID transition**:
A count-based ranker over code transitions: it scores an item by the smoothed log-frequency of
each of its codes following the history's codes. It uses the same additive form as DiGR and
none of its performance.
_Avoid_: the SID baseline, the Markov baseline

**Baseline B4 / co-click**:
A count-based ranker over item pairs: it scores an item by how often it followed the history's
items, decaying with distance, then fills any remaining slots in popularity order. It never
looks at content.
_Avoid_: the kNN baseline, the co-visitation baseline

**Popularity**:
One fixed ranking by training frequency, identical for every row. It exists to show what
query-free ordering already achieves.
_Avoid_: most-popular, global prior

**Arm**:
One method-plus-input configuration that gets evaluated: an ablation cell, a baseline, or a
TIGER setting.
_Avoid_: variant, run (a run is one execution of an arm)

## 4. Reading a ranking

**Rank**:
The position of the target in the arm's ordering, 1 for first. Most of this project's
disagreements are about what rank means when the ordering has ties; see
[`evaluation.md`](evaluation.md) §6.
_Avoid_: score, position (position is rank for baselines only)

**R@k / recall@k**:
The share of rows whose target sits at rank k or better. Recall is the general term and is used
freely, as the literature does; what it always has to carry is its unit, item recall for a
method that orders items and tuple recall for a beam, since the two are not interchangeable.
_Avoid_: hit rate, accuracy, HR@k

**Miss**:
A row whose target is beyond the arm's cap. Stored as `cap + 1` so that it sorts after every
hit without pretending to be a rank.
_Avoid_: failure, not-found

**Cap / kmax**:
The widest cut-off an arm can support: its beam width for a beam, the cut-off it was
evaluated at otherwise. A cut-off wider than the cap is not a measurement.
_Avoid_: limit, max-k

**cover@k**:
The share of rows whose target's first code is among the k first codes the model ranked
highest. It is the ceiling on a beam of that width, and it is not recall.
_Avoid_: top-k coverage, candidate recall

**nDCG@k**:
Discounted gain at k with binary relevance and one target per row, so a hit at rank r counts
`1/log2(r+1)`. It says where inside the top k the hits land, which recall cannot.
_Avoid_: gain, DCG (un-normalised)

**MRR@k**:
The mean of `1/rank` over rows that hit, zero otherwise. It weights the very top of the list
more heavily than nDCG does.
_Avoid_: mean rank, inverse rank

**Mean hit weight / `nDCG@k / R@k`**:
The average discount among hits, so 1.0 means every hit is at rank 1 and 0.30 means the average
hit sits at rank 10. It converts a pair of metrics into a statement about list shape.
_Avoid_: quality factor, discount rate

**Effective rank**:
`2` raised to the mean of `log2(rank)` over hits, i.e. the geometric mean position of a hit. It
reports list shape on the same scale as rank, and it ignores misses, so it is always read next
to the miss share.
_Avoid_: mean rank (arithmetic, dominated by outliers), median rank

**Tuple recall / reach**:
The share of rows whose target is anywhere in the tuples an arm named, which is the finest thing
a beam can report because a beam cannot order items inside a tuple. TIGER's 12.5804% on v2 is
its tuple recall at width 20; the same rows scores 0.21% to 2.04% as item recall.
_Avoid_: using it as if it were item recall in a table that also holds item arms

## 5. Comparing arms

**Matched budget / `matched @ N_r`**:
The like-for-like column: every arm is allowed exactly the number of items TIGER's beam named
on that row, `N_r`, and a hit means the target's rank is at most `N_r`.
_Avoid_: equal-k, fair recall, normalised recall

**`N_r`**:
The number of items the beam named on row r: the sum of the multiplicities of the tuples it
kept. It ranges from 25 to 1,439 on v2, so it is a budget per row, not a single k.
_Avoid_: budget, list length (both are what it measures)

**Tuple axis / item axis**:
TIGER's positions are tuple positions and DiGR's are item positions. A position on the tuple
axis holds about 9.6 items on v2 and 1.38 on v1, so rank-aware metrics are only comparable when
the axis is the same.
_Avoid_: rank level, granularity

**Popularity stratum / quintile**:
Rows binned by how often their target appears in the training targets, coldest fifth first. It
separates "works on everything" from "works on what is popular".
_Avoid_: popularity band, frequency bucket

**Strata ratio**:
The hottest quintile's recall divided by the coldest's. It is the compact statement of how
popularity-weighted an arm is.
_Avoid_: skew, bias factor

**Union recall**:
The share of rows where at least one of two arms is right, which is the ceiling on what
combining them could buy.
_Avoid_: oracle recall, ensemble recall

**List overlap / Jaccard**:
The mean Jaccard similarity between the item sets two rows were given, which says whether an
arm personalises or hands out the same list.
_Avoid_: diversity (diversity is about item content, which this project excludes)

**McNemar test**:
The paired test for two arms on the same rows: it counts the rows only one arm gets right. It
is the only significance test that respects the pairing.
_Avoid_: chi-squared test (McNemar is a special case, but the pairing is the point), t-test

## 6. Diagnosing a codebook and its costs

**Lift**:
An item's probability of being retrieved through its code, divided by the probability of
picking it at random. Depth 1 is a 463x lift for v2, so it is large and it is mostly not
comparable across widths.
_Avoid_: gain, advantage

**Lift per 1,000 candidates / `lift/1k`**:
Lift rescaled to a comparison at 1,000 candidates named: recall divided by the share of the
catalogue those candidates cover. This is the only lift figure comparable across codebook
widths and depths, and `lift/1k` is the accepted short form.
_Avoid_: normalised lift, lift/recall (there is no such ratio; lift and recall are different
measurements)

**Items per level-0 code**:
The size of the first level's neighbourhood, 584 for v2 and 18,684 for v1. The published
baselines on this task sit between 39 and 195, which is the scale this project is far from.
_Avoid_: bucket size (too vague), category size

**Collision tax**:
The cross-entropy a code-based softmax can never collect because a collision's members share
one logit. It is `log(g)` nats plus the logit-spread term, 1.045 nats on average for v2.
_Avoid_: tie penalty, alias cost

**Reachable cross-entropy**:
Measured loss minus collision tax: what an arm could reach with a perfect item-level ranking
over the same tuple scores. It is the honest way to compare loss between an injective and a
non-injective codebook.
_Avoid_: best loss, floor

**Counterfactual gain / inflation**:
Plan 3's two numbers. Counterfactual gain is the recall the same scores would reach if each
tuple were split into singletons; inflation is the rank positions a row loses to other rows'
collisions.
_Avoid_: injective uplift, tie loss

**Rank dump**:
A per-row record of `row`, `target` and `rank` that every evaluator writes so that tables can
be recomputed without a model. A row that was never dumped cannot be added to a table
afterwards.
_Avoid_: results file, log

## 7. Names we use for the analyses

**Ablation / the 2x2**:
The four-cell DiGR grid crossing the two SIDs with the two feature sets, which separates the
codebook effect from the feature effect.
_Avoid_: sweep, grid search

**Diagnosis**:
Reading a codebook's own output for coverage, occupancy, lift and tie structure. It says what
a codebook can express and never what an item means.
_Avoid_: analysis (too broad), audit

**Tie measurement**:
The check of whether collisions actually cost recall, done by splitting rows by tuple
multiplicity and controlling for popularity and for a method that has no SID.
_Avoid_: tie test, collision analysis

**Depth**:
The number of levels a method decodes through. It is the axis this project tests for TIGER, as
opposed to width, which changes how many candidates each level offers.
_Avoid_: height, length

**Direction**:
The chosen next moves after the measured results, kept separate from the task list because it
is a plan and not a claim.
_Avoid_: roadmap, backlog

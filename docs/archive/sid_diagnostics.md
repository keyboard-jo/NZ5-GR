# Diagnosing a generated SID

A playbook for judging a set of semantic ids before anyone trains a model on them.
Written for our use of a SID, which is not the usual one.

## What we ask a SID to do

The literature mostly uses a SID as a generation target: the model emits the codes
autoregressively and a trie keeps it inside the valid set. We do not do that. Our
model is DiGR, which takes the codes as an item *representation*:

```
v_i = (1/L) * sum_l e_l(s_il)          # a mean of L shared rows
z(x, i) = v_i . h_x / tau              # the only thing that ranks
```

Two consequences follow, and every check below is aimed at one of them.

The first is that items with identical codes are indistinguishable. Not "close",
identical. The score is a dot product against a vector built from the codes, so if
two items share all `L` codes they get the same logit for every query, forever. The
code is the item as far as the model is concerned, so any property of the codes is a
property of the model.

The second is that the head's capacity is set by the code space. The row count
is `sum_l |C_l|`, so per-item parameters are `sum_l |C_l| * d_llm / |I|`. Widening
the codebooks is the only way to give the item side more to work with, short of
abandoning the shared-row design.

So a SID for this use has to be judged on whether its partition is fine enough and
whether it is aligned with behaviour, not on whether it can be decoded.

## The four questions

1. Does the code space cover the catalogue, with no wasted codes and no wasted bits?
2. Is a shared code informative about what users do?
3. Does each additional level add information, or only split?
4. Is the partition fine enough for the head, given the catalogue size?

Question 1 is free. Question 2 and 3 need labels. Question 4 is arithmetic.

## The instrument

One script answers questions 1 to 3, in about 20 seconds for 4.78M items:

```bash
.venv/Scripts/python.exe "Plan 2/sid_diagnose.py" \
    --data_dir "local/2026-20-10-2-pre/digr_data" \
    --codes "local/2026-27-10-2-rq/codes_2x8192_aligned.npy" \
    --vocab "local/2026-27-10-2-rq/vocab_2x8192.json" \
    --query_codes "local/2026-20-10-2-pre/digr_data/catalog_sid_codes.npy" \
    --query_vocab "local/2026-20-10-2-pre/digr_data/vocab.json" \
    --json "local/2026-27-10-2-rq/diagnose_2x8192.json"
```

`--codes` and `--vocab` describe the SID under test. The `--query_*` pair describes
the SID the val split was built with, so the query item can be translated between two
code spaces. Without that, a new SID can only be scored against itself, which is not
what you want when the honest question is "is this cut better than the one I have".

`--no_lift` skips the split and reports only the code side, which is what to run
first because it needs no data.

The script prints, for each depth: bucket count, items per bucket, the share of items
in a bucket of two or more, the largest bucket, the target recall, the lift over
uniform, the size biased query bucket, and at the end a tie ceiling. Every one of
those is used below.

## Question 1: coverage

### Dead codes and unused bits

Reported per level as codes used out of codebook size, and entropy in bits against
`log2(size)`.

| SID | level 0 | level 1 | level 2 | level 3 | level 4 |
|---|---|---|---|---|---|
| 4 x 256 + 731 | 256/256, 7.9243 of 8 | 256/256, 7.8739 | 256/256, 7.9337 | 256/256, 7.9804 | 731/731, 1.7644 of 9.514 |
| 3 x 1,024 | 1024/1024, 9.8544 of 10 | 1024/1024, 9.8285 | 1024/1024, 9.9085 | | |
| 2 x 8,192 | 8192/8192, 12.8110 of 13 | 8192/8192, 12.7668 | | | |

All three pass. This is the check everybody runs and it is the least informative one.
Full utilisation says the quantiser was not starved; it says nothing about whether
the codes mean anything. Our worst SID was 100% utilised at every semantic level and
still had a 39,883 item bucket, and its fifth level was 731 out of 731 used while
carrying no semantics at all (see the traps).

Watch the dead-code counter during training rather than only at the end. The 2 x 8,192
run revived 1,055 centroids in a single interval at step 500, so a snapshot taken
before convergence would have shown 13% of the codebook unused and looked like
collapse.

### Bits spent on what

A useful reframing: total bits is `sum_l log2 |C_l|`, and the catalogue needs
`log2 4,783,154 = 22.2` bits to be identifiable. Everything above that is redundancy,
everything below is collisions.

| SID | bits | identity | worst depth-2 bucket |
|---|---|---|---|
| 4 x 256 + 731 | 32 + 9.5 | injective at depth 5 | 731 |
| 3 x 1,024 | 30 | 29.05% of items share a triple, largest 372 | 372 |
| 2 x 8,192 | 26 | 53.51% of items share a pair, largest 315 | 315 |
| 2 x 4,096 | 24 | predicted 26% collision | n/a |

Spending fewer bits than 22.2 is a design error. Spending more is only an error if
the extra level carries no semantics, because then it is a lookup table of arbitrary
labels, which is what our fifth level was.

## Question 2: is a code informative about behaviour

Occupancy tells you the partition is fine. It does not tell you the partition is
useful. The quantity that does is lift:

```
bucket_share      = items in the target's bucket / |I|
recall@k_bucket   = P(target is in the query's bucket at depth k)
lift              = recall / bucket_share
```

Read as "how many times more likely than chance is the target to sit under the same
code as the query item". Lift is scale free, so it can be compared across
codebook widths and catalogues, which raw bucket recall cannot.

Our three cuts, on the same 98,304 val rows:

| depth | 4 x 256 + 731 | 3 x 1,024 | 2 x 8,192 |
|---|---|---|---|
| 1 | 18.3156%, 46.9x | 12.3840%, 126.8x | 5.6559%, 463.3x |
| 2 | 2.5167%, 1437.6x | 1.1190%, 5968.8x | 0.2625%, 7880.6x |
| 3 | 0.5483%, **8257.6x** | 0.1984%, 6732.1x | |
| 4 | 0.1302%, 4505.0x | | |
| 5 | 0.0173%, 827.2x | | |

Read the columns top to bottom, not left to right. Recall always falls with depth
because the bucket shrinks; lift is what should rise.

That table is the whole diagnosis of the old SID. Lift peaks at depth 3 and then
falls by 45% at depth 4 and by another 82% at depth 5. Levels 4 and 5 split buckets
without making the target any more likely to share a prefix, so three of the five
codes the model reads are buying nothing but identity. Yet the model reads all five
with equal weight, so 40% of every item vector is noise.

### Comparing widths at a matched budget

Do not compare lift across widths directly, because lift rises mechanically when the
bucket shrinks. Normalise by candidates instead:

```
lift per thousand candidates = recall / (size_biased_bucket / 1000)
```

| SID | items per level-0 code | lift per 1,000 candidates |
|---|---|---|
| 4 x 256 + 731 | 18,684 | 0.949 |
| 3 x 1,024 | 4,671 | 2.204 |
| 2 x 8,192 | 584 | 7.589 |

Or fit the curve. Distinct depth-2 buckets went 533,417 (1.05M cells) to 3,002,708
(67.1M cells), 64x the cells for 5.6x the buckets, a log-log slope of 0.41. That
slope is a prediction: another 4x in cells should give about 5.34M buckets. When a
slope is available, the next width is a test rather than a guess.

### Size bias, and why the query bucket is printed

`items_per_bucket` from occupancy is a catalogue average. The bucket an actual query
lands in is bigger, because popular items live in dense regions. For 4 x 256 + 731
those numbers are 18,684.20 and 19,308.02, so the average query is dealt 3.3% worse
odds than the catalogue average suggests. For 2 x 8,192 it is 583.88 against 745.28,
27% worse. Both get reported; quote the size biased one when talking about what a
query experiences.

## Question 3: does a level earn its slot

Two tests, both cheap, and they disagree often enough that both are worth running.

The first is the marginal lift above: a level earns its slot if lift rises. Falling
lift means the level is splitting density, not behaviour.

The second is the tie ceiling. Identical codes score identically, so for a tie group
of size `g` a perfect ranker can place the target no better than uniformly inside the
group. Averaged over rows:

```
ceiling@k = mean over rows of min(1, k / group_size)
```

That is the recall the best possible head could reach on this SID, with the query
encoder held fixed. Measured on our cuts:

| SID | mean group | median | p90 | largest | R@1 ceiling | R@10 ceiling |
|---|---|---|---|---|---|---|
| 3 x 1,024 | 5.55 | 1 | 11 | 372 | 64.10% | 94.85% |
| 2 x 8,192 | 6.39 | 2 | 14 | 239 | 53.57% | 93.20% |

DiGR scores R@1 0.79% and R@10 2.59%, so both are far from their ceilings and the
tie rate costs nothing today. That is the point of printing it: it tells you whether
collisions are a real constraint or a number that looks alarming in a table. A SID
whose R@1 ceiling is 3% while the model achieves 2.8% has a resolution problem, and
no amount of head capacity will fix it.

## Question 4: is the partition fine enough for the head

This is the question the standard checklist omits, and on our data it was the answer.

The head reads each item through `L` shared rows, so an item's whole parameter
budget is `sum_l |C_l| * d_llm / |I|`:

| SID | rows | params | per item | items per level-0 code |
|---|---|---|---|---|
| 4 x 256 + 731 | 1,755 | 1,797,120 | 0.376 | 18,684 |
| 3 x 1,024 | 3,072 | 3,145,728 | 0.658 | 4,671 |
| 2 x 8,192 | 16,384 | 16,777,216 | 3.51 | 584 |
| 2 x 16,384 | 32,768 | 33,554,432 | 7.02 | 292 |
| 2 x 32,768 | 65,536 | 67,108,864 | 14.03 | 146 |

The relevant comparison is the source literature's catalogue scale. DiGR's Amazon
subsets run 39 to 195 items per level-0 code (Musical Instruments about 10k items,
Arts and Crafts about 28k, Video Games about 50k). Our starting point was 18,684,
which is 96 to 479 times coarser, and in that regime a code says almost nothing about
an item, so the mean vector cannot either.

Sequence cost runs the other way. An event is `1 + L` tokens plus whatever per-event
features you add, so a shorter SID buys the richer stream:

| SID | tokens per event | train tokens at 19.79 events per row |
|---|---|---|
| 4 x 256 + 731 | 6 | 221.5M |
| 2 x 8,192 | 4 | 153.5M |
| 2 x 8,192 plus clock | 5 | 187.5M |

## The checks that need geometry

The codes come from a vector space, and properties of that space bound what any
quantiser cut from it can do. Worth reporting once per embedding:

- Effective rank of the 256-d item vectors: 24.8, participation ratio 16.8, 23
  dimensions for 90% of variance. The space is far lower dimensional than its width.
- All norms exactly 1.0, so cosine is the only available geometry and the vectors
  are already on the sphere.
- 1-NN cosine against all 4.78M items: 0.9290. Cosine between items sharing a
  depth-3 prefix: 0.9287. Packing bound for 4.78M points in the effective dimension:
  0.9286.

That last triple is the striking one. The partition reproduces the true nearest
neighbour to four decimal places, and the true nearest neighbour is already at the
packing bound. Two things follow. The quantiser is faithful, so bucket quality
cannot be blamed on the cut. And there is no geometric room left, so "better codes"
cannot mean "codes that separate points better" in the embedding's own metric. It
has to mean "codes that align with something the embedding does not measure", which
is behaviour. This is the check that correctly predicted width would buy resolution
and correctly killed the plan to hunt for a better geometry.

## Behaviour purity, the check to run when lift looks good

Lift can be inflated by popularity: the top of the popularity distribution is dense,
so dense buckets look predictive. The cross-check is precision rather than recall.
Take the bucket a query item lands in and ask what fraction of the items in it were
ever clicked:

```
precision = items in the query's bucket that were ever clicked / bucket size
base rate = 1,580,961 / 4,783,154 = 33.0%
```

If precision sits near the base rate the lift is density and the codes know nothing.
If it is well above, the bucket is behaviourally coherent. Our depth-1 bucket holds
0.39% of the catalogue and captures 18.3156% of next clicks, so its precision is far
above the base rate, but the test is what proves it rather than assuming it.

Split by warm and cold items as well. 1,580,961 items (33.0%) were ever clicked and
3,202,193 (67.0%) were not, and the code space cannot give the same vector to items
with wildly different behaviour without blurring both. A SID whose lift comes
entirely from warm items is fine for retrieval as long as you know it.

## Failure signatures

| symptom | what it means | what to do |
|---|---|---|
| all codes used, lift flat | the cut is dense, not semantic | judge the embedding, not the cut |
| lift peaks then falls | later levels encode identity, not behaviour | drop the levels, they are noise in the mean |
| one level with entropy ratio near 0.2 | that level is a lookup key, not a code | treat it as a suffix and never give it equal weight |
| collision share 7.5x the uniform bound | k-means concentrates where items are | expected, not a bug, but measure the tie ceiling |
| largest bucket a large share of the catalogue | resolution too low for the catalogue size | widen the codebooks |
| tie ceiling close to measured recall | the codes, not the head, are the constraint | widen before spending GPU hours on the head |
| high bucket lift, low model recall | the head cannot use a good SID | look at parameters per item, loss, negatives |
| low bucket lift, low model recall | the representation is the problem | fix codes or retrain the tower, not the head |
| share of items in a bucket of one is high, injectivity reached early | you have more codes than you need | stop widening, spend the tokens elsewhere |

## Order of work

1. `--no_lift` for coverage, occupancy and tie ceiling. Seconds, no data needed.
2. Lift with the matching `--query_*` pair. Under a minute.
3. Geometry, once per embedding, when a width change is being considered.
4. Behaviour purity, when lift surprises you.
5. Only then train. A model run on an unmeasured SID cannot distinguish a bad SID
   from a bad model, which is how we spent four GPU-hours learning that a 39,883 item
   bucket and a 2.59% R@10 were the same fact.

## What a SID diagnosis cannot tell you

It cannot separate "good codes, weak head" from "weak codes, good head" without
scoring the model, because lift is a property of the codes alone. The gap between
lift and model recall is where the head lives, and that gap is a separate
measurement.

It cannot judge behaviour if the label is content derived. Lift against a target that
was chosen by the same embedding is circular.

It says nothing about generation quality. Trie validity, beam size and prefix
survival matter when you generate tokens, and none of that applies to us. Do not
carry generation-side conclusions into an item-representation SID.

## Traps

Never grade a SID on prefix recall alone. Shrinking a bucket lowers recall
mechanically, so a wider SID always looks worse on recall and better on lift. Report
both, and lift per thousand candidates to compare across widths.

Never quote the catalogue average bucket for a query. The size biased number is
3% to 27% worse on our data.

Never leave an unset unique-id bound. Our fifth level was `--add_unique_id
--no-clip-suffix` with no bound, so it grew to 731 codes whose meaning is "rank of
this collision group", reused across 1,878,898 unrelated groups. It was fully
utilised, its entropy ratio was 0.1855, and giving it one fifth of every item vector
was pure noise. If a level's distribution is flat over a small alphabet and its
values are a collision rank, it is not a semantic code.

Mixed-radix packing overflows int32. At four 256-wide levels the key reaches
`255 * 256^3 = 4,278,190,080`, past int32 and past a Windows C long. Symptoms are
silently wrong recall and a lower query hit rate, then `OverflowError: Python int too
large to convert to C long` one level later. Cast to int64 and refuse radix products
over `2**62`.

Check the row alignment and the file hash. Our catalogue row is `rid - 1` because
the preprocessor sorts and de-duplicates ascending; a different sort order silently
pairs every item with another item's codes and every metric becomes noise. Hash the
codes file and record it next to any result.

Do not subsample and compare. `--max_rows` changes which queries are counted and
moves lift by several percent. Same rows or no comparison.

Do not trust one split. A single 98,304 row split is enough to detect a 2x
difference and not enough to defend a 5% one.

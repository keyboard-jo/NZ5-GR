# Plan: wider Semantic IDs and a richer event stream

Written after `docs/results.md` and `docs/sid_results.md`, before any of it is
built. Scope is unchanged: three retrieval methods that use RQ-VAE Semantic IDs,
one backbone, one dataset, one evaluator. This document adds a second SID release
and a second data release, and says what each is expected to buy and how the claim
gets falsified.

## What is proposed

Three changes, and they are independent enough to fail separately.

1. Re-quantise the frozen content embedding with wider codebooks, so the vocabulary
   of code tokens grows.
2. Use that new SID everywhere: baselines, DiGR, TIGER, one table.
3. Teach the model more per event. Today an event is `[SID x 5, gap bin]`. Add
   action type, session structure, and finer time.

Point 2 is a consequence of 1 and 3, not a separate bet. Point 1 and point 3 both
raise the ceiling on a fixed spend, and they multiply: a wider codebook reading a
sequence that still cannot distinguish an exposure from a click is a wasted refit.

TIGER keeps the frozen SID. It is a method comparison, not a representation
comparison, and rebuilding it on every SID release is a tax with no information in
it. The three-arm table on the current SID stands as the reference row.

## What the current SID actually is

Measured from `local/2026-20-10-2-pre/digr_data/catalog_sid_codes.npy`,
4,783,154 items, 5 levels, codebooks 256 x 4 plus 731.

| depth | distinct tuples | items per tuple | share of catalog |
| ----- | --------------- | --------------- | ---------------- |
| 1 | 256 | 18,684.20 | 0.39063% |
| 2 | 57,122 | 83.74 | 0.0017507% |
| 3 | 1,506,041 | 3.18 | 6.7e-5% |
| 4 | 3,459,854 | 1.38 | 2.9e-5% |
| 5 | 4,783,154 | 1.00 | 2.1e-5% |

Every level uses every code: 256 of 256 for levels 0 to 3, and 731 for level 4,
which matches `sid_sizes[4]` in `vocab.json`. All four codebooks sit at 7.87 to
7.98 bits of entropy against a ceiling of 8, so no codebook is collapsed in the
usual sense.

Two things follow, and both matter for the plan.

**The partition is extremely top-heavy.** Half the catalogue's structure is one
8-bit branching factor. Depth 3 already has 3.18 items in a bucket and depth 4 is
1.38, so after the third level the code is not carrying clusters, it is carrying
identity. That is why level 4 exists at all: 1,323,300 items share a depth-4 tuple
with at least one other item, and 731 codes of arbitrary rank suffix break those
ties.

**The geometry has already run out.** `docs/sid_results.md` records effective rank
24.8 of 256, participation ratio 16.8, a measured 1-NN cosine of 0.9290, and a
packing bound of 0.9286 for that participation ratio. A bucket of 3 items has a
mean intra-bucket cosine of 0.9287. So a bucket of three is already 99.97% of the
way to the tightest possible bucket, which is a bucket of one.

## Correction 1: score buckets as lift, not as raw recall

Shrinking a bucket mechanically lowers the chance that the target is inside it. If
the depth-1 bucket goes from 18,684 items to 4,671, a bucket-based recall number
falls by roughly 4x for free. The comparable quantity is recall divided by the
bucket's share of the catalogue.

| SID | depth-1 items | base rate | recall needed for the same lift |
| --- | ------------- | --------- | ------------------------------- |
| 256-wide, current | 18,684 | 0.39063% | 18.3156% (measured, lift 46.9x) |
| 1,024-wide | 4,671 | 0.09766% | 4.579% |
| 2 x 16,384, level 1 | 292 | 0.006106% | 0.286% |

Use lift from here on. And note the older 19,308 figure in `docs/tasks.md` and
`docs/results.md`: that is a size-biased mean, about 3.3% above the uniform
18,684.20. The lift changes by the same 3.3%, so it is cosmetic, but the uniform
number is the one to quote.

## Correction 2, revised after measurement: width buys resolution, and the first
## draft of this section was wrong

The first draft predicted that a wider codebook would raise the head's capacity while
leaving bucket lift flat. The argument was geometric: 4.78M points of effective
dimension about 17 have a bounded number of well-separated cells, so 4x more cells
buys about `4^(1/17)`, an 8.6% improvement, and the measured 1-NN cosine already sits
at the packing bound, so there was no 8.6% left to take.

That prediction was testable and it has been tested. It failed.

Same 98,304 val rows, same instrument (`Plan 2/sid_diagnose.py`):

| SID | depth-1 recall | depth-1 lift | mean query bucket |
|---|---|---|---|
| 4 x 256 + 731 collision suffix | 18.3156% | 46.9x | 19,308 |
| 3 x 1,024 | 12.3840% | 126.8x | 5,618.40 |

Lift is size-normalised, so the two rows are comparable, and the finer partition is
2.7x more label-pure. Depth 2 moves the same way, 1,437.6x to 5,968.8x, and depth 3
goes 8,257.6x to 6,732.1x.

The error was in the step from geometry to labels. The packing bound limits how well
cells separate points in the embedding's own metric and says nothing about how well
they separate the click. A k-means partition with four times as many cells tracks the
density modes more closely, so the cells get more label-pure even though the points
inside them are no better separated. The first draft reasoned about the wrong
quantity and then treated the conclusion as arithmetic.

So width is not a capacity argument with a resolution footnote. It is the main lever
on how much a SID says about an item, and this 2.7x is the first time anything in
this project moved bucket quality at all.

## Correction 2b: items per code is the number to hold against the paper

The DiGR paper is explicit that the shared, mean-pooled code embeddings carry "no
item-specific information", which is a designed property and not a defect: it is what
lets an item be scored the moment it has a SID. The consequence is that the only
thing separating two items is the difference between two sums of shared vectors, so
the resolution of the fingerprint is items per code.

| catalogue | items per level-0 code at 256-wide levels |
|---|---|
| paper, Musical Instruments, about 10k items | 39 |
| paper, Arts and Crafts Sewing, about 28k | 109 |
| paper, Video Games, about 50k | 195 |
| ours, 4 codebooks of 256 | 18,684 |
| ours, 3 codebooks of 1,024 | 4,671 |
| ours, 2 or 3 codebooks of 16,384 | 292 |

The paper's catalogues are 96 to 479 times finer than our starting point, and that is
the entire scale difference. It also explains why its accuracy figures do not
transfer: at 195 items per code, four shared codes nearly identify an item.

Landing in the paper's regime needs about 24,500 codes per level, so roughly 16,384
wide. The configuration this analysis pointed at was 2 x 16,384: 32,768 rows,
33,554,432 parameters, 7.02 per item, 67 MB in bf16, 292 items under each level-0
code. It was not run. What was cut was **2 x 8,192**, at 584 items per level-0 code,
and that closed the axis (see S1 in the phases below), because the measured lift was
already 8.0x the original at a matched candidate budget and the tie ceiling showed
injectivity was not the binding constraint. The 16,384 row stays in the table as an
untested prediction, not as the plan.

The cost is exact identity, and it is small. 16,384 squared is 268,435,456 tuples for
4,783,154 items, a mean occupancy of 0.018. If the partition were uniform the
birthday bound puts about 42,600 pairs of items on a shared depth-2 tuple, which is at
most 1.78% of the catalogue and only affects targets inside those pairs. A learned
partition is lumpier than uniform, so the measured number will be worse and has to be
measured rather than assumed. It was measured once, at 8,192 wide: the bound predicted
7.13% and 53.51% of items turned out to share a tuple, 7.5x the prediction.

| config | rows | params | per item | items/level-0 | codes per item | depth-2 collisions |
|---|---|---|---|---|---|---|
| 4 x 256 + 731, the SID DiGR was trained on | 1,755 | 1,797,120 | 0.376 | 18,684 | 5 | 72.33% share a depth-4 prefix |
| 3 x 1,024, measured | 3,072 | 3,145,728 | 0.658 | 4,671 | 3 | none, largest tuple 372 |
| 2 x 8,192, measured, endorsed | 16,384 | 16,777,216 | 3.51 | 584 | 2 | 53.51% share a tuple, largest 315 |
| 2 x 16,384, not run | 32,768 | 33,554,432 | 7.02 | 292 | 2 | at most 1.78%, uniform bound |
| 2 x 32,768, not run | 65,536 | 67,108,864 | 14.03 | 146 | 2 | at most 0.45%, uniform bound |
| 3 x 16,384, not run | 49,152 | 50,331,648 | 10.52 | 292 | 3 | none, 4.4e12 tuples |

What ends up being endorsed is 2 x 8,192, one step short of this table's ranking. The
reason is that 36 bits was never the target, 463.3x bucket lift was, and the axis was
closed once the ceiling stopped binding. The rows the 16,384 configurations would add
are real (3.51 to 7.02 or 14.03 parameters per item) but they buy resolution we have
no use for while R@1 sits under one percent.

Three codes per item was the earlier recommendation and it was arrived at from the
4 x 256 collision rate, where 72.33% of items share a depth-4 prefix and the fifth
level is mandatory. At 16,384 wide that tail is a couple of percent, so the reason for
the third code evaporates. Running 3 x 16,384 as the first point would have paid a
token per item to solve a problem two levels already solve.

One thing width cannot buy: with mean pooling, level `l` contributes `e_l(c_il) / L`
to every item that shares code `c_il`, so the four levels are forced to equal weight
by construction. DiGR's Eq. 1 allows any `Psi_phi`, and it instantiates a
parameter-free mean. Replacing it with a concat plus learnable per-position weights
is inside the paper's family, keeps item-atomic scoring, and stays beam-free. That is
a separate arm, after width.

## Correction 3: the width change can pay for itself in sequence length

With 256-wide codebooks the fifth level is mandatory: 27.7% of items collide at
depth 4, so level 5 becomes an arbitrary 731-way collision rank. At 3 x 1,024 the
largest depth-3 bucket is 372 items, measured, so a fourth level cannot collide at
all and the SID drops to three codes. That takes an item from 5 tokens to 3, and an
event from 6 tokens to 4 once the action token is added, a 33% shorter event. Train
tokens go from 221,500,536 to about 153.5M at the same event count.

That saving is what pays for the gap bins in the next section. With one gap token per
event it is 5 tokens per event and about 186M tokens, still below today's number, so
the richer stream is free in sequence length. Without the width change it would cost
6 tokens plus a gap, a 17% increase.

The catch: any item still colliding at depth 3 becomes indistinguishable from its
twin. Record the collision count in `preprocess_meta.json` and refuse to proceed if
it exceeds about 0.1% of the catalogue. Collisions are the one thing a SID cannot
absorb, because two items with one code can never be ranked apart.

## The event stream: what is there, what is missing

Measured on `data/seq.parquet`, row group 0, 4,944,464 events, 54,812 users.

| what | today | where it goes | verdict |
| ---- | ----- | ------------- | ------- |
| item identity | 5 SID tokens | per event | keep, but 3 tokens on the 3 x 1,024 SID |
| relative gap | one log-spaced bin of 64, clamp 30 days | per event | keep, raise to 128 bins, free |
| action type | **dropped** | nowhere | add, 1 token, highest value |
| absolute time | **absent** | nowhere | add, 1 token, second highest |
| session | **absent** | nowhere | derive from gaps, overlaps the gap token, lowest value |
| user features | 9.11 tokens per row, 145-token vocabulary | per row | untouched for now |

The action measurement, which is the case for adding it:

```
action 0, exposure   89.4459%
action 1, click       9.6848%
action NaN, unknown   0.8694%
```

Nine of every ten events in a history are exposures. The model currently cannot
distinguish an exposure from a click, so it reads "you looked at this" without
knowing whether you bought it. That is not a modelling subtlety, it is information
being thrown away after it was already read off the parquet file, since
`preprocess_digr.py` uses `action_type` to choose anchors and then emits the same
six tokens either way. Four new vocabulary entries fix it.

The gap measurement, which is the case against a session token:

```
p10 210s | p25 732s | p50 6,462s (1.8h) | p75 42,096s | p90 86,280s | p99 368,472s
share of events with gap <= 30 min: 35.93%
share of events with gap <= 24h:    90.04%
```

At a 30-minute session boundary there are 3,132,961 sessions over 4,944,464 events,
which is 1.58 events per session. A token that says "position 0 or 1 in a session"
is close to a second copy of the gap token. At a 24-hour boundary it is 10.2 events
per session and it starts to carry something, but so does the gap token it already
sits next to. Rank it below action and below absolute time, and expect it to
disappoint.

Absolute time is the interesting gap. There is no periodicity anywhere in the
current representation: no hour of day, no day of week, and the relative gap is
clamped at 30 days so anything older looks identical. For a click-prediction target
that is a real omission, one token, and a 168-way vocabulary if hour and weekday are
combined.

## Token arithmetic

Train tokens are 221,500,536 over 34,019,640 event slots and 1,719,430 rows, which
is 128.83 per row: 10.11 row-level tokens and 118.72 from 19.79 events at 6 tokens
each. Each new per-event token therefore adds 34.0M tokens, or 15.4%.

| configuration | tokens per event | train tokens | vs today |
| ------------- | ---------------- | ------------ | -------- |
| today | 6 = 5 SID + gap | 221.5M | 1.00x |
| 2-code SID, gap only | 3 = 2 SID + gap | 119.4M | 0.54x |
| plus action | 4 | 153.5M | 0.69x |
| plus hour and weekday | 5 | 187.5M | 0.85x |
| plus session | 6 | 221.5M | 1.00x |

This table was written for a four-level SID, where the fifth level had to be traded
for the action token just to break even. At 2 x 16,384 the item is two codes, so the
fifth level, the fourth level and the collision suffix are all gone and three tokens
of room appear per event. Action, gap, hour and weekday come to five tokens and
187.5M train tokens, which is 15% less compute than today for three pieces of new
information. Even adding a session token lands exactly on today's 221,500,536.

Action is the row that matters, since 89.4% of events are exposures. Hour and weekday
stay behind a flag so they can be ablated without a code change. Session is the one
to skip, for the reason in the gap table above.

## Phases

Ordered, cheapest gate first. GPU hours are from the frozen DiGR run, 4h34m for
three epochs at batch 512.

- **S0, instrument (CPU, half a day).** Put the bucket-lift table and the effective
  rank numbers into the SID QA so every future SID is scored the same way. Add a
  `--sid_dir` check that prints depth occupancy, collisions, and lift against the
  same 98,304 val rows. Without this the width experiment has no denominator.
- **S1, width sweep (GPU 1-2h, CPU half a day).** Quantise the frozen 256-d
  embedding at two settings: 2 x 16,384 as the target, which puts level 0 at 292
  items per code and is OnePiece's configuration, and 2 x 32,768 if the measured
  depth-2 collision share comes back above 2%, which puts level 0 at 146 and halves
  the collision bound four times over. 4 x 256 and 3 x 1,024 are already measured as
  the controls. Nothing else changes. Report depth occupancy, collision count, and
  bucket lift at depths 1 to 3.

  **Measured so far.** Three points on the curve, same 98,304 val rows, same
  instrument (`Plan 2/sid_diagnose.py`):

  | SID | rows | items per level-0 code | depth-1 lift | depth-1 query bucket | peak lift |
  |---|---|---|---|---|---|
  | 4 x 256 + 731 | 1,755 | 18,684 | 46.9x | 19,308 | 8,257.6x at depth 3 |
  | 3 x 1,024 | 3,072 | 4,671 | 126.8x | 5,618 | 6,732.1x at depth 3 |
  | 2 x 8,192 | 16,384 | 584 | 463.3x | 745 | 7,880.6x at depth 2 |

  Depth-1 lift rose by 2.70x and then 3.65x, so the curve is not flattening, it is
  steepening. At a matched candidate budget the ordering is the same and larger:
  recall per 1,000 candidates is 0.949 for 4 x 256, 2.204 for 3 x 1,024 and 7.589
  for 2 x 8,192, which is 8.0x the original at equal budget. The 2 x 8,192 run
  itself was healthy: both codebooks 8,192 of 8,192 active at 12.8110 and 12.7668
  bits of 13, perplexity 7,186.2 and 6,969.3, best val loss 0.001523, 516 s for the
  whole pipeline.

  What 2 x 8,192 did not fix is collisions. Only 62.78% of items hold a two-code
  tuple of their own, so 2,559,616 items (53.51% of the catalogue) sit in a group of
  two or more, the largest group holds 315, and the uniform birthday bound for 67.1M
  cells predicted 7.13%, so the measured rate is 7.5x the prediction because k-means
  cells cluster where the items are rather than spreading evenly. So the
  pre-committed
  gate in `docs/tasks.md` S1, measured depth-2 collision share under 1%, cannot be
  passed at two levels at any width, and it should be replaced. The quantity that
  actually matters is the tie ceiling: two items with the same codes get the same
  score from any scorer that reads the SID, so a tie group of size `g` caps recall
  at `1/g` at rank 1 and `min(1, k/g)` at rank `k`. Measured on the val targets,
  mean tie group is 5.55 for 3 x 1,024 and 6.39 for 2 x 8,192, which caps R@1 at
  64.10% and 53.57% and R@10 at 94.85% and 93.20%. DiGR sits at 0.79% and 2.59%, so
  the ceiling is far away and not binding; the instrument now prints it, so any
  future SID can be checked against it in one run.

  The axis was closed here. There was a reason to expect 2 x 16,384 to close the
  collision gap: distinct depth-2 buckets went from 533,417 for 3 x 1,024 (1.05M
  cells) to 3,002,708 for 2 x 8,192 (67.1M cells), so 64x the cells bought 5.6x the
  distinct buckets, a log-log slope of 0.41, and another 4x in cells (268M) predicts
  about 5.34M distinct buckets against 4,783,154 items, i.e. effective injectivity.
  That prediction is recorded and untested, because the decision was to stop running
  SIDs: 2 x 8,192 already gives 463.3x lift at 584 items per code and the tie ceiling
  says the remaining collisions cost nothing until R@1 is near 50%. The next GPU
  hours go to S4 instead, which is the only step that can move the three-arm table.
- **S2, verdict.** The flat-lift hypothesis is dead: 256 to 1,024 moved depth-1 lift
  from 46.9x to 126.8x, and 1,024 to 8,192 moved it to 463.3x, so width buys
  resolution and the first draft of Correction 2 was wrong twice over. The width axis
  closes at 2 x 8,192 by decision rather than by observed saturation, so the claim in
  `docs/embedding_weakness.md` section 9 becomes items per code, measured at three
  points, with the 16,384 point named as untested.
- **S3, event stream (CPU, one preprocess pass).** Action token, gap bins 64 to 128,
  two code levels, hour and weekday as an opt-in flag so they can be ablated without
  a code change. The default build is 4 tokens per event and 153.5M train tokens;
  with the flag on it is 5 and 187.5M. **Built**: `Plan 2/preprocess_digr_v2.py`,
  877.6 s, 1,719,430 train rows and 98,304 val rows, 153,461,256 and 8,808,890
  tokens, vocabulary 16,662, event width 4, and the split check confirmed the same
  98,304 val users and the same targets as the frozen build.
- **S4, DiGR on the new data (GPU 4.5-5.5h).** Same frozen config, same 300k
  negatives, same three epochs, same evaluator. Two axes moved together here on
  purpose, because S1 and S3 both gate on the same run being worth doing.
  **Run**, 3h11m at 63.3 min per epoch. Epoch 2 is the best checkpoint: val loss
  11.0133 against 11.2431 frozen, R@1 1.0400% against 0.7924%, R@10 3.6600% against
  2.5940%, nDCG@10 0.02170 against 0.01570, MRR@10 0.01710 against 0.01258. Against
  co-click window 20 that is ahead on R@1, nDCG, MRR and the combined score, and
  level on R@10. Epoch 3 overfits, so two epochs is the setting. Cut-offs beyond 10
  are still unmeasured: `eval_full_catalog.py` on the checkpoint is the open item.
  Full numbers and caveats in `docs/results.md` section 12.
- **S5, baselines on the new data (CPU, minutes).** Popularity and co-click need no
  SID at all, so they are control rows and must reproduce exactly. `sid_transition`
  needs the new codes and is the one baseline that can move.
- **S6, TIGER.** On the SID that S1 and S4 endorse, not on both. The three-arm table
  is one table, one SID, three methods.

## What this does not do

It does not touch the item embedding, which is still the thing with a 17-dimensional
effective rank and a 0.376-parameter-per-item read-out. Widening a codebook over a
weak geometry is a capacity fix, and capacity is not obviously the binding
constraint: the same head structure with 2.3x more rows may simply memorise more
popularity. Phase 3b, and its 3b.0 gate on whether behaviour is predictable from
content, is still the only path that changes what the vectors can express. S1 to S6
are worth doing because they are cheap and because they remove two confounds from
the eventual three-arm table, not because they are expected to close the 5x gap.

It also does not re-open the candidate-set question. Pool-restricted scoring stays
retired until there is a query-only candidate generator to justify it.

## Traps

- **Comparing bucket recall across SID releases without dividing by bucket share.**
  The single easiest way to produce a fake 4x regression or a fake improvement.
- **Assuming a wider codebook cannot collapse.** 1,024 codes against 4.78M items is
  fine, 16,384 against 4.78M starts to need greedy re-assignment and dead-code
  restarts, exactly as OnePiece does. Check entropy per codebook, not just that
  every code is used once.
- **Dropping the fifth level without checking collisions.** Silent, and it caps
  recall in a way no amount of training fixes.
- **Adding tokens and compute at the same time as changing the SID,** then
  attributing the result to either. S1 and S3 are separated so that S4's outcome can
  be read off two independently measured halves.
- **Paying for the preprocess twice.** The pass over 1.8 GB of parquet with 221M
  output tokens is the slowest CPU step in the project. Decide the token layout
  once, in S3, with the absolute-time flag already in it.

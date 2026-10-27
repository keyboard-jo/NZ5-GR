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

## Correction 2: width is a capacity win, not a resolution win

The expectation that a wider codebook makes buckets sharper is probably wrong on
this embedding, and the reason is arithmetic rather than opinion. With 4.78M points
whose effective dimension is about 17, the number of well-separated cells is bounded
by the geometry, not by how many rows the codebook has. Going from 256 to 1,024
rows is a 4x increase in cell count, but in a 17-dimensional space 4x more cells
buys about `4^(1/17)`, an 8.6% improvement in separation. The measured 1-NN cosine
already equals the packing bound, so there is no 8.6% left to take.

What the wider codebook does buy is real and separate: the item side of every SID
model is a set of shared code token embeddings, and today there are 1,755 of them
for 4,783,154 items, which is 0.376 parameters per item. At 4 x 1,024 the SID
vocabulary is 4,096 rows, 2.3x more, so 0.88 parameters per item. That is a change
in the head's ceiling, and the head is where the failure was named.

So the prediction to test is split, and either half is useful.

- Depth-1 and depth-2 bucket recall **lift** barely moves. Within a few percent.
- The three-arm table improves, because the item side has 2.3x more parameters and
  the depth-4 code is closer to injective.

The test costs about an hour of GPU and no preprocessing: quantise the same 256-d
vectors, then score buckets with the existing CPU scripts. Do that before any DiGR
rerun. If the lift moves a lot, the diagnosis in `docs/embedding_weakness.md`
section 9 is wrong and needs rewriting, which is worth knowing.

## Correction 3: the width change can pay for itself in sequence length

With 256-wide codebooks the fifth level is mandatory: 27.7% of items collide at
depth 4. With 1,024-wide codebooks the depth-4 partition has 1.1e12 possible tuples
for 4.78M items, so it should be close to injective and the fifth level should have
almost nothing to do. Dropping it takes an item from 5 tokens to 4, which is a 20%
shorter event and a 15.4% cut in train tokens. That saving is what pays for the new
per-event tokens in the next section.

The catch: any item still colliding at depth 4 becomes indistinguishable from its
twin. Record the collision count in `preprocess_meta.json` and refuse to proceed if
it exceeds about 0.1% of the catalogue. Collisions are the one thing a SID cannot
absorb, because two items with one code can never be ranked apart.

## The event stream: what is there, what is missing

Measured on `data/seq.parquet`, row group 0, 4,944,464 events, 54,812 users.

| what | today | where it goes | verdict |
| ---- | ----- | ------------- | ------- |
| item identity | 5 SID tokens | per event | keep, but 4 tokens after the width change |
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
| 4-wide SID, drop level 5, add action | 6 = 4 SID + gap + action | 221.5M | 1.00x |
| plus absolute time | 7 | 255.5M | 1.15x |
| plus session | 8 | 289.6M | 1.31x |

The first row of that table is the recommendation: same compute, one piece of
information swapped for another. The current fifth level is an arbitrary 731-way
rank suffix that separates items with identical depth-4 codes, and an action token
is a fact about what the user did. Trading one for the other is close to free and
strictly more informative.

## Phases

Ordered, cheapest gate first. GPU hours are from the frozen DiGR run, 4h34m for
three epochs at batch 512.

- **S0, instrument (CPU, half a day).** Put the bucket-lift table and the effective
  rank numbers into the SID QA so every future SID is scored the same way. Add a
  `--sid_dir` check that prints depth occupancy, collisions, and lift against the
  same 98,304 val rows. Without this the width experiment has no denominator.
- **S1, width sweep (GPU 1-2h, CPU half a day).** Quantise the frozen 256-d
  embedding at three settings: 4 x 256 as the control, 4 x 1,024, and 2 x 16,384 to
  match OnePiece's resolution. Nothing else changes. Report depth occupancy,
  collision count, and bucket lift at depths 1 and 2.
- **S2, verdict (no compute).** If S1's lift is flat, say so in `docs/results.md`
  and keep the capacity argument as the reason to adopt a wider SID. If the lift
  moves, rewrite `docs/embedding_weakness.md` section 9.
- **S3, event stream (CPU, one preprocess pass).** Action token, gap bins 64 to 128,
  drop the level-5 token only if S1 reports under 0.1% collisions, absolute time as
  an opt-in flag so it can be ablated without a code change. Rebuild `digr_data`
  and record the token count and the manifest counts.
- **S4, DiGR on the new data (GPU 4.5-5.5h).** Same frozen config, same 300k
  negatives, same three epochs, same evaluator. Two axes moved together here on
  purpose, because S1 and S3 both gate on the same run being worth doing.
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

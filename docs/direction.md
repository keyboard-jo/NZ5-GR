# The direction: where the next 5x can and cannot come from

**Read this before choosing any experiment.** Written 2026-10-29, after the
three-arm table closed, and it is the file that decides what the next GPU hours
are for. It contradicts the implicit plan in `docs/tasks.md` (S7, S9, and "one
embedding experiment first") and says why.

The question it answers: the two candidate next steps were (a) rebuild the item
embedding and (b) upgrade DiGR's composition from linear summation to concat +
MLP. Both are **item-side capacity** moves. The argument below is that the
item-side capacity axis has a measured slope of 0.15 and is therefore spent, and
that the remaining headroom is **query-side information** plus **dose**.

---

## 1. The decomposition that decides everything

`R@10 = R@1000 × P(top 10 | target is in the top 1,000)`.

For DiGR v2 that is exactly `27.6622% × 13.22% = 3.657%`. Two knobs, nothing else:
how often the target is anywhere in the list, and how well the list is ordered.

| method | R@1 | R@10 | R@100 | R@1000 | P(top10 \| top1000) |
|---|---|---|---|---|---|
| co-click w20 | 0.8057% | 3.7069% | 10.9873% | 25.3021% | **14.65%** |
| DiGR v2 | 1.0549% | 3.6570% | 10.9182% | 27.6622% | 13.22% |
| TIGER v2 | 1.4400% | 2.4500% | 6.5900% | 20.0800% | 12.20% |
| DiGR v1 | 0.7924% | 2.5940% | 7.5094% | 21.6003% | 12.01% |
| popularity | 0.4395% | 1.3438% | 4.7862% | 14.7288% | 9.12% |

Three facts fall out.

**1.1 Per-item resolution is worth 11%.** The co-click table has exact per-item
rows, 5.37 successors for every item, 25,669,281 stored pairs, built in 47 seconds.
A 0.6B transformer reading 16,384 shared code rows scores 13.22% on the same
conditional. The best per-item resolution available in this repo buys **14.65%
against 13.22%**, +11% relative. That is the entire measured value of "giving the
item side per-item degrees of freedom", and it bounds what a new embedding, a
learned residual, an MLP composition or the deferred ANN arm can be worth on the
ordering half of the metric.

**1.2 The query is worth 1.45x.** Popularity has no query at all and reaches 9.12%
on the same conditional. A 20-event history plus 145 user-feature tokens plus an
action token reaches 13.22%. So everything the model knows about *this user* buys
1.45x over knowing nothing, on the half of the metric that measures ordering. The
query is not fine-tuned; it is barely present.

**1.3 The absolute headroom at R@10 is 7.6x, not unlimited.** `R@10 ≤ R@1000`, so
27.66% is a hard cap on any reordering of DiGR's own list. R@10 18.3% (5x) requires
`R@1000 ≥ 18.3%` **and** `P(top10 | top1000) ≈ 20%+`. Since no method measured here
exceeds 14.65% on the conditional, the 5x is not a ranking problem. It is an
information problem.

---

## 2. The item-side slope is 0.15, so item-side work is finished

The one clean item-side experiment in the repo, everything else held fixed:

| SID | rows | params | params/item | R@10 |
|---|---|---|---|---|
| 4 × 256 + 731 | 1,755 | 1.80M | 0.376 | 2.5940% |
| 2 × 8,192 | 16,386 | 16.78M | 3.51 (**9.3x**) | 3.6570% (**1.41x**) |

`log(1.41) / log(9.3) = 0.15`. Every **10x** in item-side degrees of freedom has
returned **1.4x**. Reaching 5x on that curve needs roughly six more orders of
magnitude of rows, i.e. 32,768-wide codebooks at 4+ levels and 10.6 parameters per
item, for a predicted ~1.25x. The axis is not steep and it is not going to get
steeper:

| move | predicted R@10 gain | why |
|---|---|---|
| S9, 2 × 16,384 (292 items/code) | ~1.15x | same 0.15 slope, one doubling of rows |
| 2 × 32,768 (146 items/code) | ~1.25x | same slope, 4x rows, 268 MB bf16 |
| concat + MLP composition (DiGR Eq. 1 with a learned `Psi_phi`) | ~1.2-1.4x | breaks the forced `1/L` weighting and gives pair-injectivity, but 53.51% of items share a 2-code tuple and a tie cannot be broken by any function of the tuple |
| per-item residual table, warm items only | ~1.1-1.3x | bounded by 1.1 above: co-click's exact rows are the measured ceiling |
| Stage 1 rebuild in isolation | unknown, plausibly ≤1.2x | both towers already score below a frequency count (0.4354% and 0.5636% R@10), so the same objective at 30x scale has no evidence behind it |

Also note what the 1.41x actually contained. Three axes moved: codebook width,
the action token, and 64 → 128 gap bins. **Two of the three are query-side**, and
the action token gave the model one bit — exposure vs click — on the 89.4% of
events it had been reading as clicks. The width evidence is independent and real
(bucket lift 46.9x → 463.3x), but the ranking gain is unattributed, and S7 exists
to split it. Its result matters more than as a footnote: if the action token
carried a large share of the 1.41x, then the project has already demonstrated on a
**one-bit** change that query-side information pays, which is the thesis of this
file.

---

## 3. The direction

Three pieces. All three are legal under both standing exclusions in
`docs/problems.md` §1: **no reranking** (none of these reorders another method's
list), and **the item representation stays a function of content** (all three add
information to the query or to the score, never to the item vector, so cold items
are untouched and the code space stays one space).

### 3.1 Graph-expanded query: the co-click prior in the score

The count table beats the model at ranks 1 through 1,000 for 47 seconds of CPU
against 3.2 GPU-hours, and the model **cannot see it**. The sequence gives the
model codes; it never gives it the item-item graph. 25,669,281 pairs are already
counted in `Model Training/baselines_digr.py`, and multi-hop propagation over that
graph is strictly more information than the first-order table is currently using.

The banned version is a behaviour SID (co-occurrence vector → code), rejected
because it is undefined for a cold item and splits one code space into two. The
legal version puts the same signal on the query side:

```
s_i = h . V_i  +  sum_l  lambda_l * log p_hat_l[ c_{i,l} ]
```

where `p_hat_l` is the empirical distribution of level-`l` codes among the co-click
successors of the last `k` history items, and `lambda_l` is learned (2-4 scalars).
Cold items have no successors, get `0`, and are scored exactly as today. Nothing
is reranked, no token is added, no preprocess pass is needed, and the frozen DiGR v2
checkpoint applies with a one-epoch warm start.

This is the winners' "GNN over the user-item graph" row from
`docs/problems.md` §6, implemented as a prior instead of as a per-item vector,
which is what makes it survive the content-only rule.

### 3.2 Longer history and more anchors: this is information, not compute

Today: `max_seq_len` 20, gaps clamped at 30 days, 2 anchors per user record,
153,461,256 train tokens, 63.3 min per epoch, 3.2 GPU-hours for the whole run.

The gap distribution says the window is about one session: p50 6,462 s (1.8h),
p90 86,280 s, and 90.04% of events are within 24h. A 20-event window on a user who
has 90 events throws away most of the history that exists. `docs/data.md` §5 already
records the local 3-anchor build at 2,480,142 rows and 317,812,410 tokens.

| window | anchors | train tokens | epochs at 41.5k tok/s |
|---|---|---|---|
| 20 | 2 | 153M (today) | 63 min |
| 20 | 3 | ~230M | 92 min |
| 60 | 3 | ~690M | 4.6 h |
| 60 | 4 | ~920M | 6.2 h |

Two epochs at window 60 / 3 anchors is ~9-12 GPU-hours. This project has spent
roughly 15 GPU-hours in total on a question the competition winners answered with
two orders of magnitude more. **Dose is the highest-ceiling bet on this list and
the only one that is unambiguously bounded below by the current numbers.** It also
attacks the right knob: length is query information, which is the half of the
decomposition that is stuck at 13%.

### 3.3 Mixture-of-intents head

`P(top10 | top1000) = 13.22%` is what it costs to compress a user's intent into a
single 1,024-d `h` and one dot product. A mixture keeps the generative form and
the single pass:

```
s_i = log sum_j  pi_j * exp( h_j . V_i )
```

with `J` intent vectors `h_j` (a small learned read-out over the same sequence, or
`J` clustered history states) and `pi_j` gated by the sequence. Cost is `J`
gathers over the same `V`, so `J = 4` is still cheaper than a reranker and is
exhaustive over all 4,783,154 items. This is the generative part of DiGR taken
further rather than abandoned, and it is the only item-independent way to raise
intra-list precision.

### 3.4 TIGER: depth is its axis, and it was given the opposite

TIGER was run at 2 × 8,192, prefix-free. That is the one configuration in which
TIGER **is** DiGR: `h . (e(c1) + e(c2)) / (L tau)` against
`log p(c1|h) + log p(c2|h)` are the same additive family over the same two shared
rows. The conclusion "the objective is not the ceiling" is therefore correct and
also nearly uninformative — the paper's factorisation was never tested.

The paper's claim is that `h` is re-read from a context containing `c1..c_{l-1}`,
so the number of distinct decision contexts is the product of the prefix counts
rather than one additive read. That needs **three or more levels** to be
expressible. DiGR's axis is width (few levels, many codes); TIGER's axis is depth
(many levels, conditional). The SID release gave DiGR exactly what it wanted and
TIGER exactly what it could not use.

The fair arm: 3-4 levels × 1,024-4,096, `PREFIX=1`, beam, reporting `cover@k` as
the stated ceiling. Cost ~2h10m plus a beam pass. It is not the 5x, but as it
stands the table's TIGER row is a loss change and not a method comparison, and any
reviewer who knows the paper will say so.

**Measured 2026-10-28** (`docs/results.md` §14, run
`tiger_v2_sid2x8192_e2_prefix`, 2h08m for two epochs). The `PREFIX=1` arm has now
run at 2 × 8,192 and gives the number this section was waiting for: conditional
level-2 cross-entropy **4.7051 nats** against the prefix-free arm's unconditional
**7.6731**. Knowing the true level-0 code therefore removes **2.968 nats** of
level-1's 7.673 nats of uncertainty — **38.7%** — for one level of depth. Level-0
coverage saturates at 91.55% at beam 1,000 and actually *falls* 0.28 pp between
epochs while beam-1 rises 0.82 pp, so the arm buys head precision with tail
coverage.

Two consequences, and they change the priority in §4:

- **The level-0 ceiling is not the constraint.** Any `R@k(beam 1000)` up to 91.55%
is reachable, and the prefix-free arm's realized 20.08% is nowhere near it. The
middle-of-list weakness is ordering, not coverage.
- **The beam pass ran on 2026-10-29, and depth pays at 1.77x.** Beam 20 reaches the
  target for **12.58%** of val rows, and cover@20 reproduces the training log's
  43.30% row for row, so the path is verified. But beam 20 names only ~30 items, so
  the comparable statement is 12.58% reach against DiGR's **7.12%** in its top 30 of
  4,783,154, and against co-click's 7.18%. Its `R@10 8.90%` is a rank inside those 20
  tuples and must not be placed beside DiGR's 3.6570%; that comparison overstates the
  arm by about 1.4x. Two thirds of the covered rows (12,367/42,578 = 29.0%) are lost
  at the level-1 joint, which makes beam width the next cheap lever and a four-level
  SID the actual test of the paper.

Depth at 4 levels remains the real test of the paper, and 2.97 nats per conditional
step is now the measured size of the prize for adding one.

---

## 4. Measure these before spending a GPU hour (all CPU)

**4.1 What can a single-stage generator do on this data at all?** `docs/tasks.md`
3.5: a tiny SASRec or GRU4Rec over raw item ids, ~15 minutes, still unrun. This is
the most valuable unspent measurement in the repo, because the bar may be wrong.
OnePiece's 18.85% is a **stage-2** number: SID beam → InfoNCE re-rank → filters,
over a 660k pool. Comparing a single-stage full-catalog generator to a reranked
pool-restricted pipeline is a category error, and it may have inflated the target
by 2-3x. If SASRec lands near 5%, then 18.85% is out of reach for any single-stage
item-id model here and the project's target should be re-set, not chased. If it
lands near 12%, SID methods are losing 3x to a trivial baseline and *that* is the
headline finding.

**4.2 Oracle union of the three arms.** For each val row, is the target inside the
top `k` of DiGR, co-click, popularity, or any of them? One hour of CPU, no GPU.
- `union@1000 ≈ 45%` → the two signals are complementary, §3.1 is worth building.
- `union@1000 ≈ 30%` → they are redundant, skip §3.1 and go straight to §3.2.

This sizes the prize before any run, which is the pattern `docs/plan_v2.md` used
successfully for the width sweep.

**4.3 Re-score DiGR on a target-bearing universe.** 488,945 items appear as train
targets (10.22% of the catalog); 3,202,193 are never clicked at all. Recounting
DiGR's ranks with non-target items removed, and separately on the 660k pool shape,
takes minutes and says how much of the gap to the published number is denominator.
It does not replace the full-catalog table (the reasons in `docs/results.md` §10
stand). It is a second column that recalibrates what "5x" means.

---

## 5. Predicted outcome, stated in advance so it can be falsified

If §3.1, §3.2 and §3.3 all land at the modest end of their ranges:

| step | R@10 | multiple |
|---|---|---|
| DiGR v2 today | 3.657% | 1.00x |
| + co-click code prior, learned `lambda_l` | 5.0-5.5% | 1.4-1.5x |
| + window 60 / 3 anchors | 7.0-8.5% | 1.9-2.3x |
| + mixture-of-intents head | 8.5-11% | 2.3-3.0x |

**2.5-3x is the realistic target. 5x requires the top of every range plus dose
beyond what is listed here.** Anyone promising 5x from an embedding rebuild or a
better composition is on the 0.15 curve and will miss by a factor of three.

Kill criteria, so the plan cannot drift:
- `union@1000 < 35%` → §3.1 is dead, the two signals are the same signal.
- window 60 / 3 anchors below 1.2x → dose is not the answer at this scale either,
  and the conclusion is that the task is close to noise in this restricted setting.
- mixture-of-intents below 1.1x → intra-list precision is not intent ambiguity, and
  §1.1's 11% is the real ceiling; re-baseline the bar and write the negative result.

---

## 6. Kill list, with the reason

- **S9, 2 × 16,384.** Predicted ~1.15x by the 0.15 slope. The tie ceiling for
  2 × 8,192 is R@1 53.57% / R@10 93.20% against a measured 1.05% / 3.66%, so
  collisions cost nothing and injectivity has no mechanism to pay.
- **concat + MLP as a standalone arm.** Right instinct, wrong axis, and it is
  already the deferred 4th arm in the paper's own family; worth one run only after
  §3.3, because the MLP can express the intent interaction more cheaply than a
  mixture and the two should not both be built blind.
- **A Stage 1 rebuild on its own.** Both towers are below a frequency count. The
  only Stage-1 change with published evidence is D2's 100k-380k catalog negatives
  plus a live `log_q`; that is a training-config fix, not an architecture, and it
  belongs with a re-quantise or not at all.
- **Reranking, behaviour SIDs, and a larger backbone.** Excluded for the reasons in
  `docs/problems.md` §1, and none of them improves the measurement this project
  exists to make.

---

## 7. Why this file is separate

`docs/problems.md` records the diagnosis and `docs/tasks.md` the ordered list. Both
were written when the item side looked like the ceiling, and the tasks list still
puts S7 and S9 next. Sections 1 and 2 above are the arithmetic that changed the
answer: the conditional `P(top10 | top1000)` is 13-15% for *every* method ever
measured here, including one with exact per-item rows, so item-side refinements
cannot move it. The item side is now a solved axis with a known slope; the query
side has had one bit of information added to it in the whole project.

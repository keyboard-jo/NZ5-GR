# Tasks

Working list, ordered. Scope is fixed by `docs/problems.md` section 1: compare
three retrieval methods that use RQ-VAE Semantic IDs, all on the same backbone,
same data, same evaluator.

- **DiGR**: additive read of 1,755 shared code rows. Built, trained, frozen.
- **TIGER**: per-level cross-entropy over the same codes, additive score when prefix-free. Built, trained, measured.
- **heuristic**: count tables. Built, and currently winning.

Continuous-vector retrieval and ANN are deferred, not cancelled. See the last
section for why they are out of this table.

State as of 2026-10-26.

Three rules govern the order below.

- **One axis moves at a time.** The SID set is frozen before the three-arm table
  is measured, so a method comparison is never confounded by an embedding change.
- **A SID release is a release, not an edit.** New codes invalidate every number in
  the table at once, so a new SID ships as a complete table or it does not ship.
  `docs/plan_v2.md` is the current release plan.
- **The embedding is not assumed to be the blocker.** DiGR already scores 4.6x the
  R@10 of the embedding it was built from, so the item side is a suspect, not a
  proven cause. See `docs/embedding_weakness.md` section 9.

---

## Phase 0: freeze what exists (done)

- [x] Freeze DiGR at `digr_0p6b_e3_full_sampled300k/final`. Not `best`, because
      epoch 2 has the lower val loss and epoch 3 ranks better.
- [x] `REF_POPULARITY_R10` corrected to exactly `1321 / 98304`
      (`0.013437906901041666`) in both `Model Training/baselines_digr.py` and
      `Model Training/eval_full_catalog.py`, with a test that pins the two
      modules to the same value.
- [x] `docs/README.md` index, `docs/results.md` holding every measured table.
- [x] `Model Training/` reorganised: notebooks into `notebooks/`, slurm wrappers
      into `slurm/`, an index README at the root of that directory.
- [x] `Model Training/embedding_knn_nextclick.py` and its wrapper, so the item
      embedding can be scored against real next clicks. Result: it loses to
      popularity at every cut-off. Recorded, not a blocker.
- [x] SID prefixes scored against real targets instead of self-graded.

---

## Phase 1: what the data actually says (done)

- [x] 1.1 How often is the target an item the user already clicked? **0.05%** of
      val rows. The harness essentially forbids repeats, so a reranker over
      history is not the answer.
- [x] 1.2 Depth-1 SID bucket of the last clicked item holds the target
      **18.3156%** of the time, over 19,308 candidates on average. Depth 2 holds
      it **2.5167%** of the time over 345.9. Against real targets, not
      self-graded.
- [x] 1.3 The quantization is faithful: 18.3156% at depth 1 matches the 16% to
      20% that the raw vectors' own kNN curve extrapolates to. The RQ-VAE is not
      destroying signal.
- [x] 1.4 Target coverage. 488,945 of 4,783,154 items (10.22%) appear as a
      training target. 37.43% of the top-10 slots filled by the embedding are
      training targets against that 10.22% base rate, so it is still a popularity
      smoother, just a weak one.
- [x] 1.5 User overlap between train and val: still unmeasured. Low cost, low
      priority now that the failure is localised to the score.

---

## Phase 2: fix the bar before building TIGER (done)

- [x] 2.1 Co-click at windows 10 and 20, same 98,304 val rows, same
      `--ks 1,10,100,1000,10000`. Window 20 won.
- [x] 2.2 Recorded in `docs/results.md`, and `REF_COCLICK` in both
      `Model Training/baselines_digr.py` and `Model Training/eval_full_catalog.py`
      now carries the window 20 numbers, with windows 1, 5 and 10 kept beside it
      so the sweep does not have to be rerun.
- [x] 2.3 TIGER acceptance bar restated: **beat R@10 3.7069% and R@100 10.9873%**.
- [x] 2.4 The sweep is closed, not deferred. R@10 went 3.6011 to 3.6794 to 3.7069,
      steps shrinking 0.35x per doubling, so the asymptote is about 0.015 points
      higher and further widening is not worth a job.

---

## Phase 3: protocol, then the trie (closed)

**Closed.** It was written for a five-level SID, where a beam over 256^5 tuples
needs a trie. The endorsed SID has two levels of 8,192, and the exact score of every
item is a two-column gather over the catalogue's own codes, so a trie would be a
slower route to the same numbers. The protocol decision in 3.1 stands and the rest
is kept only as the record of why it was dropped.

These were written for the five-level SID and a beam search. Three of them no longer
apply, which is recorded here so nobody rebuilds them.

- [x] 3.1 **Fifth-level protocol: decided.** The headline table scores **exact
      item identity** for all four arms. 39.3% of items share their first four
      codes and level 5 is an arbitrary rank inside the group, so a prefix hit
      would inflate recall by up to 39% relative and only for the SID arms. A
      prefix-hit variant is a footnote, never the headline. Confirmed by the user
      (m03114, paraphrased). The question is moot now: the 2 x 8,192 SID has no
      suffix level at all.
- [x] 3.2 **Build the trie. Dropped.** Only 3,459,854 of 4,294,967,296 four-code
      prefixes exist, so unconstrained decoding over five levels would emit invalid
      codes almost always. That is a five-level problem. At two levels the score of
      every item is `p1[c1_i] + p2[c2_i]`, a gather over codes that all 4,783,154
      items already own, so no prefix index is needed and nothing can be emitted
      that is not a real item.
- [x] 3.3 **Measure decode cost for beam widths. Dropped for the same reason.**
      There is no beam width to report. R@10,000 is reached by sorting 4,783,154
      exact scores.
- [x] 3.4 **Sliced evaluator. Dropped.** It existed because beam 1,000 would score
      TIGER over 1,000 candidates while DiGR was scored over 4,783,154, so the two
      rows would not be comparable. With a two-level SID both arms are scored over
      the full catalogue by the same code, which is a stronger comparison than the
      sliced one would have been.
- [ ] 3.5 Optional sanity run. A tiny SASRec or GRU4Rec over item ids, about 15
      minutes on GPU, tells us whether an item-id sequence model clears co-click
      at all on this slice. It is not part of the three-arm table but it bounds
      what any SID method can hope for.

---

## Phase 3b: rebuild the item representation (closed, out of scope)

**Closed by decision (m05965).** The item representation stays a function of item
content alone, because content is the only input a cold item has. Everything below
is kept as the record of the reasoning, not as planned work. The current Stage 1
embedding is not pure, since ItemDNN and HSTU run in tandem, but the measured
effect is small enough that a structural-only vector is what the pipeline
effectively produces already, so the rule closes the strand rather than resetting
it.

The invariant that decides every choice here: **one code space, defined for every
item, and comparable across items.** A code for a never-clicked item has to sit in
the same geometry as a code for a popular one, or the level-0 prefix stops meaning
one thing and the trie and the per-level head inherit a split vocabulary.

1,580,961 items (33.0%) have behaviour and 3,202,193 (67.0%) have features but no
interaction. This is therefore not a cold-start problem about missing data. Every
item has a content vector. What two thirds of the catalogue lacks is clicks.

That leaves exactly one distinction that matters.

- Behaviour as an **input**. The code becomes a function of interactions, so items
  without interactions are handled by a different rule. This is where the cold
  start objection bites.
- Behaviour as a **teacher**. The item vector stays a function of content for every
  item, and behaviour shapes the mapping only during training. Cold items are
  covered by construction, warm items inherit behaviour-shaped geometry, and there
  is one code space.

Prefer the teacher. It preserves the property the current SID has: identical content
gives identical codes. Asymmetry in how *accurate* the behaviour estimate is for an
item is fine. Asymmetry in the *geometry* is not.

The whole question then reduces to a number that costs no GPU: how much of a
behavioural vector is predictable from a content vector.

Two directions remain, and they are multiplicative rather than additive. A wider
head trained on content codes cannot show its capacity, and better codes read
through 1,755 rows stay under-used, so each run in isolation returns a false
negative. They also share the one expensive step, the RQ-VAE refit.

- **A, post-hoc fusion.** Concatenate the content vector with a co-click SVD vector
  and re-quantise. No training.
- **B, retrain Stage 1 so behaviour infuses correctly.** D2 sampled softmax with a
  real `log_q`, D1 per-item residual, behaviour as a teacher or an input (B2), then
  re-export and re-quantise.

- [ ] 3b.0 **Gate: is behaviour predictable from content?** The deciding
      measurement. Take the 1,580,961 items that have both a content vector and a
      behavioural vector, hold out 10%, fit ridge regression and a two layer MLP
      from `c_i` to `b_i`, and report held-out cosine. Then quantise the predicted
      vectors and measure depth-1 bucket recall against real val targets. High
      cosine means distillation works and the cold start objection disappears. Low
      cosine means a content-only code cannot carry behaviour, the SID stays
      content, and behaviour belongs in the head and the reranker instead. Half a
      day of CPU, and either answer redirects the phase.
- [ ] 3b.1 **Build the behavioural vector.** Co-click counts at window 20 give
      25,669,281 distinct pairs, about 16 successors for each of the 1,580,961
      clicked items. Form the pair matrix, normalise so degree does not dominate
      (PPMI, or the symmetric `D^-1/2 A D^-1/2` form, both standard for
      co-occurrence embeddings), truncated SVD at rank 256, rows L2-normalised.
      Shrink low counts so an item with one successor does not get a confident
      direction. Undefined for never-clicked items, which is the point.
- [ ] 3b.2 **Gate the vector on its own, before any quantiser.** Score it against
      real val targets with `embedding_knn_nextclick.py`. The content vector scores
      0.5636% R@10 and raw co-click scores 3.7069%. If the SVD vector cannot beat
      0.5636% it is a popularity artefact and the whole branch dies for half a day
      of CPU. Also inspect the correlation between vector norm and click count, and
      the top-1 neighbour of sampled popular, mid-tail and cold items.
- [ ] 3b.3 **Choose the fusion mode, in this order of preference.**
      (i) Distil: `e_i = normalize(MLP(c_i) + W * pred_b(c_i))`, where `pred_b` is
      the 3b.0 map. Every item is defined, cold items included, and the code stays
      a pure function of content. Warm items may blend actual behaviour with the
      prediction, since both are estimates of the same quantity.
      (ii) Learn the fusion: `e_i = normalize(MLP([c_i ; b_i]))` with `b_i = 0` for
      cold items, trained against the retrieval loss so the zero slice is
      in-distribution rather than incidental.
      (iii) Post-hoc concat with zero fill. Cheapest and weakest, because after unit
      normalisation cold items sit on a face of the sphere where half the
      coordinates are zero, so the quantiser spends codes on the fact that two
      populations exist, and two items with identical content but different
      interaction status get different codes.
- [ ] 3b.4 **Re-quantise.** Same experiment as S1 in Phase 6, so run it there and
      read the result here. `--input_dim 512` must be passed explicitly, the default
      is 1024. The item side is 1,797,120 parameters for 4,783,154 items today; 4 x
      1,024 takes it to about 4.5M and depth-1 buckets from 18,684 items to about
      4,671.
- [ ] 3b.5 **Gate the codes, CPU only.** Depth-1 and depth-2 bucket recall against
      real val targets, the same method that produced 18.3156% at 18,684
      candidates. The bar is the same **lift**, 46.9x, which at 4,671 candidates per
      bucket is 4.579% recall. Comparing raw recall across releases is the trap in
      `docs/plan_v2.md`. Check
      injectivity (all 4,783,154 five-code tuples still distinct) and concentration
      (largest 4-prefix bucket against level 5 capacity, and where never-clicked
      items land, since they now carry a predicted behaviour rather than a zero).
- [ ] 3b.6 **Gate D2 in parallel.** Sampled softmax over 100k to 380k catalogue
      negatives plus a real `log_q` distribution, nothing else changed. The
      deviation that took DiGR from popularity-level to 2.5940%, and the only Stage
      1 fix with published evidence behind it. Gate with
      `embedding_knn_nextclick.py` as in 3b.2: if the retrained item tower does not
      beat popularity's 1.3438% R@10 against real next clicks, do not quantise it.
- [ ] 3b.7 **Combine (B2), only if 3b.5 and 3b.6 both pass.** Retrain Stage 1 with
      D2, D1 (`r_i` per-item residual, zero for cold items) and behaviour as the
      teacher from 3b.3(i). Re-export, re-quantise at the 3b.4 config, then one
      DiGR run. Compare against the frozen 2.5940% R@10 and co-click 3.7069%.

Widening is not a separate experiment: bigger codebooks and new vectors both live
inside the same re-quantisation, so they travel together and 3b.5 judges the
combined result. Centroid distillation (D6) stays dropped, since the vectors it
would average score 0.4354% R@10, below a frequency count.

---

## Phase 3c: validate the HSTU tower

**Built, run, and answered in the negative.** The item side is behaviourally weak
(effective rank 24.8, pooled kNN loses to popularity).
`docs/embedding_weakness.md` section 3 predicted the behaviour was hiding in the
HSTU user encoder, and the probe tested that directly. It does not hold. The tower
scores 0.4354% R@10, also below a frequency count, so the behaviour is not in
either tower.

- [x] 3c.1 `Model Training/probe_hstu.py` plus `slurm/probe_hstu.sh`. Loads
      `stage1_rec_model.pt`, rebuilds ItemDNN, HSTU, action embedding and input
      norm from the state dicts, and rebuilds the feature store from
      `canonical_items.parquet` with the checkpoint's saved args, because the
      store is a non-persistent buffer and is not in the checkpoint.
- [x] 3c.2 Query `encode_user(history_items, history_actions, mask)` against the
      full 4,783,154 item vectors, cosine, R@1/10/100/1000 and MRR@10 on the exact
      val rows.
- [x] 3c.3 Actions are approximated to a constant click action, because DiGR
      tokens carry codes and a temporal token but not action types. That biases
      the query DOWN, so it cannot manufacture a good result.
- [x] 3c.4 **Bug found on the first run and fixed.** `HSTUSequenceEncoder` reads
      `x[:, lengths - 1]`, so the last position is the query. Training
      left-aligns the sequence and the probe right-aligned it, so the query read
      padding on padded rows. Fixed with `to_chronological`.
- [x] 3c.5 **Rerun and record, done.** Patched run on job 716604, confirmed to be
      the fixed file (identical md5 to `Model Training/probe_hstu.py`). R@1
      0.0458%, R@10 0.4354%, R@100 2.9063%, R@1000 13.5122%, MRR@10 0.1311%,
      nDCG@10 0.2008%. The alignment bug was worth +1.66% relative at R@10 and
      nothing at R@1, because a padding position still aggregates the valid prefix
      through attention.
- [ ] 3c.6 **Unblock on I/O, now optional.** The probe spends its life re-reading
      `seq.parquet` at about 16 MB/s of `/scratch` bandwidth, so a pass costs 41
      minutes of I/O for 20 seconds of GPU work. Convert the needed columns to
      `.npy` before any further probe runs. The number is already in hand, so this
      is no longer on the critical path.
- [x] 3c.7 **Answer: the bottom band.** 0.4354% R@10 against a 0.5% floor, so
      Stage 1 learned no usable behaviour in either tower. See
      `docs/embedding_weakness.md` section 7.

---

## Phase 4: build TIGER (built, measured)

**Promoted to next by decision** (user, 2026-10-28). The attribution arm that would
have separated codebook width from the action token is deferred behind it: the
deliverable is the three-arm table and TIGER is the only row still missing, so a
run that only explains an existing row can wait.

The three-arm table is now complete. TIGER's prefix-free arm on the same 2 x 8,192
codes reaches R@1 1.4400%, R@10 2.4500%, R@100 6.5900%, R@1000 20.0800%,
R@10000 45.7600%, nDCG@10 0.01870, MRR@10 0.01700 in 3.2 GPU-hours. It takes rank 1
from both DiGR and the count table and loses from rank 10 outwards, and
`docs/results.md` section 13 has the reading: at two levels the two arms are one
additive function class, so that run compares two losses, not two factorisations.

The SID it runs on is the 2 x 8,192 release that S1 endorsed and S4 trained on. That
simplifies the design in a way worth stating before any code is written: **with two
levels there is no trie and no beam.** The exact full-catalog score of every item is
`log p1[c1_i] + log p2[c2_i]`, two gathers over `catalog_sid_codes.npy`, so TIGER is
scored over all 4,783,154 items in the same space, with the same evaluator, as DiGR.
The five-level trie work in 3.2 to 3.4 is therefore closed rather than deferred.

The design is still direct SID input, user feature block carried, temporal tokens
carried, same backbone, same content SID, same 1,719,430 train rows and 98,304 val
rows. `tokens.npy`, `offsets.npy`, `targets.npy`, `LengthBucketBatchSampler`,
mid-epoch checkpointing and `catalog_sid_codes.npy` are reused unchanged.

- [x] 4.1 Head. One per-level cross-entropy over that level's own vocabulary,
      replacing DiGR's single 4,783,154-way softmax. `logits_l = h W_l` with `W_l`
      of shape `[1024, sid_sizes[l]]`, so the count is
      `1024 * sum(sid_sizes)`: 16.8M at 2 x 8,192 and 1.8M at 4 x 256 plus the
      731-wide fifth level. `TigerHead` at `Model Training/train_tiger.py:381`,
      `level_cross_entropies` at `:466`.
- [x] 4.2 Inputs. Append `SEP, c1, .., cL` after the last event and read level `l`
      from the position of the code before it, so one teacher-forced forward pass
      gives every term. Implemented behind `--prefix_conditioning`, default OFF:
      with it off the codes are not in the input at all and every level reads the
      separator state, which is what makes the exact full-catalogue score a single
      forward pass. `TigerCollate` at `:318`.
- [x] 4.3 Inference. `evaluate_tiger` at `:528` scores all 4,783,154 items by
      gathering each level's log-probability column at the item's own code and
      summing, then ranks every target. That is exact, so all five cut-offs up to
      10,000 are reportable and the number is comparable with the DiGR row. Beam
      search at `:245` and the prefix tree at `:185` are tied to
      `--prefix_conditioning`: with codes in the input an item's score depends on
      the path taken to it, so items cannot be scored independently, candidates
      have to be generated, and a beam will propose prefixes no item owns. The tree
      is mandatory in that arm at two levels as much as at five, and the cut-offs
      it can report are capped at the beam width, so `R@1000` and `R@10000` are not
      computable there. Used by `--mode beam`, and `--mode eval` refuses a
      prefix-conditioned checkpoint instead of scoring it with the wrong rule.
      The ceiling is measurable without running a beam: `evaluate_tiger` reports
      `cover@b` for b in 1, 10, 20, 100, 1000, the share of rows whose target's
      first code is inside the model's own top b at level 0. Every beam arm's
      recall is bounded by it, so the faithful arm's ceiling costs one extra topk
      per batch on top of a run we do anyway.
- [x] 4.4 Reuse, do not rebuild: `_load_static_meta`, `_load_splits`,
      `LengthBucketBatchSampler`, the atomic checkpoint swap, `--resume`,
      `--save_steps`, `--log_interval` and the CSV log are imported from
      `train_digr.py`. `Model Training/train_tiger.py` is 1,298 lines, the estimate
      was 400 to 600, and the overrun is the second-SID-release support and the beam
      arm.
- [x] 4.5 Budget two epochs, and expect no training-time saving over DiGR. Row
      length at 2 x 8,192 is 90 tokens without prefix conditioning against DiGR's 89,
      and 92 with it. Measured: 3,358 steps per epoch, 41.5k tok/s, epoch 1 61.6 min,
      epoch 2 64.2 min, epoch 3 65.7 min, plus 2.5 min of validation each. The TIGER
      saving is not in training at all: it is in inference, where a beam of 20
      replaces scoring 4,783,154 items per query. At this catalogue size the
      exhaustive score is a 1.7 minute GEMM over the whole val set, so the beam is
      the more expensive of the two here. That is worth reporting rather than
      hiding: the speed argument for generative retrieval assumes a query-side beam
      that is cheap relative to a catalogue-side score, and with 4.78M items and two
      levels it is not.
- [x] 4.6 Run all five cut-offs and write `R@1/10/100/1000/10000`, nDCG@10 and
      MRR@10 into `docs/results.md` beside the DiGR row. Done, and the answer is
      split: rank 1 1.4400% against DiGR's 1.0549% and co-click's 0.8057%, then
      2.4500% and 6.5900% at R@10 and R@100 against the bar's 3.7069% and 10.9873%.
      Predicting the codes step by step does not order the middle of the list
      better; it sharpens the marginals that decide the first guess.
- [x] 4.7 Close the gaps this run exposed, in order. (a) **Done.** `format_metrics`
      now prints `R@5` and `cover@1/10/20/100/1000`, and every epoch writes
      `metrics.jsonl` plus `metrics.json` beside the checkpoints, so nothing the
      evaluator computes is dropped. `evaluate_tiger` gained `rank_items`, and
      training with `--prefix_conditioning` ranks nothing (the exhaustive path is
      the prefix-free score) while still reporting loss and `cover@k`.
      `head_config.json`, which `_read_head_config` had always read and nothing had
      ever written, is now written by `_save_tiger_checkpoint`, so a checkpoint states
      which arm it is; `--mode eval` refuses on that record rather than on the flag.
      (b) Reproduce the R@1 1.4400% on a second code path, because this table has
      already had one silent ranking bug in `evaluate_tiger`. (c) Run `PREFIX=1` for
      two epochs, about 2h10m, then `--mode beam` for the faithful arm, reporting
      `cover@b` and the cut-offs up to the beam width only. `Plan 2/train_tiger.sh`
      now defaults to `PREFIX=1` and names the run `..._prefix` so it cannot collide
      with the prefix-free run.

**Both SID releases are supported from one script.** Level widths are read from
`vocab.json`, so 4 x 256 plus the 731-wide collision level and 2 x 8,192 both run
with no flag beyond `--data_dir`. `Plan 2/train_tiger.sh` selects between them with
`SID_VERSION=1` or `SID_VERSION=2`, and `PREFIX=1` turns on the paper's
factorisation. The four cut-offs, the evaluator, the sampler and the checkpointing
are the same code for both, so the two rows differ only in the SID. Tests:
`Model Training/test_train_tiger.py`, 34 of them, CPU only, stub backbone.

---

## Phase 5: the table and the write-up

- [x] 5.1 One table, four rows (popularity, co-click, DiGR, TIGER), five cut-offs,
      one column for cost, one column for the candidate space. Done in
      `docs/results.md`.
- [x] 5.2 The conditional-ranking column: "given the target is in the top 1,000,
      how often does it reach the top 10". This is the column that makes the
      failure legible, and it now reads co-click 14.65%, DiGR v2 13.22%, TIGER
      12.20%, DiGR v1 12.01%, popularity 9.12%.
- [ ] 5.3 State the degrees-of-freedom arithmetic explicitly: DiGR's item side is
      1,755 rows of 1,024 dimensions, 1,797,120 parameters for 4,783,154 items,
      **0.376 per item**.
- [ ] 5.4 State what the co-click table has that the model does not: 25,669,281
      distinct pairs, 5.37 successors per item. This is the honest answer to "why
      does a count table win".
- [ ] 5.5 Regenerate every number with `eval_full_catalog.py` and
      `baselines_digr.py`. Do not quote a number from a notebook, and do not quote
      the retired ones in `docs/problems.md` section 10.

---

## Phase 6: SID v2 and a richer event stream

Reasoning, arithmetic and the falsifiable predictions are in `docs/plan_v2.md`.
Three claims to hold it to. A wider codebook buys resolution as well as capacity, and
both are measured: depth-1 bucket lift goes 46.9x to 126.8x between 256-wide and
1,024-wide, and items per level-0 code go 18,684 to 4,671, then to 292 at 16,384 wide,
which is the DiGR paper's own regime. The earlier "capacity, not resolution" argument
is dropped along with the packing-bound reasoning behind it. And the event stream is
discarding the exposure-versus-click distinction on 89.4% of all events.

The measured occupancy it builds on, 4,783,154 items. On the frozen SID, depth 1 is
256 tuples of 18,684.20 items, depth 2 is 57,122 of 83.74, depth 3 is 1,506,041 of
3.18, depth 4 is 3,459,854 of 1.38, depth 5 is injective, and every codebook sits at
7.87 to 7.98 bits of 8, so nothing is collapsed.

On the 3 x 1,024 SID already built, depth 1 is 1,024 tuples of 4,671.05, depth 2 is
533,417 of 8.97, depth 3 is 3,393,796 of 1.41 with the largest bucket at 372 items, so
a fourth 1,024-wide level cannot collide at all. Its codebooks sit at 9.8544, 9.8285
and 9.9085 bits of 10. Depth-1 bucket lift goes from 46.9x to 126.8x between the two
SIDs, so S1 is now about how far that curve runs rather than whether it moves.

- [x] S0 **Instrument.** Depth occupancy, collision count and bucket lift at depths
      1 and 2, for any SID directory, against the same 98,304 val rows. Without
      this there is no denominator for S1, and the release cannot be compared.
      `Plan 2/sid_diagnose.py`, 33 tests, and it is also what measured the 2 x 8,192
      tie ceiling.
- [x] S1 **Width sweep, three points done, one left.** 2 x 8,192 was cut as run
      `local/2026-27-10-2-rq`: 16,384 rows, 3.51 parameters per item, 584 items per
      level-0 code, both codebooks 8,192 of 8,192 active at 12.8110 and 12.7668 bits
      of 13, perplexity 7,186.2 and 6,969.3, best val loss 0.001523, 516 s.
      Measured on the same 98,304 val rows: depth-1 lift 463.3x at a query bucket of
      745.28, depth-2 lift 7,880.6x at a query bucket of 6.18. Depth-1 lift went
      46.9x, 126.8x, 463.3x across the three SIDs, and recall per 1,000 candidates
      went 0.949, 2.204, 7.589. So width buys resolution, the curve is steepening,
      and 2 x 8,192 is 8.0x the original at a matched candidate budget.
- [x] S1b **Width axis closed at 2 x 8,192.** No further SID runs. The collision gate
      written here before the measurement is withdrawn: only 62.78% of items hold a
      two-code tuple of their own, so 2,559,616 items (53.51%) sit in a group of two
      or more, largest 315, against a uniform bound of 7.13%. A 1% gate cannot be
      passed at two levels at any width. The gate is now the tie ceiling that
      `Plan 2/sid_diagnose.py` prints: identical codes score identically, so a tie
      group of size `g` caps recall at `1/g` at rank 1. Measured ceilings are R@1
      64.10% and R@10 94.85% for 3 x 1,024, and R@1 53.57% and R@10 93.20% for
      2 x 8,192, against DiGR at 0.79% and 2.59%, so the ceiling is far away and the
      2 x 8,192 tie rate is accepted. The endorsed SID is therefore 2 x 8,192, 584
      items per level-0 code, 3.51 parameters per item, depth-1 lift 463.3x, and 8.0x
      the original recall at a matched candidate budget. 2 x 16,384 stays unrun, and
      the prediction that it would reach injectivity is recorded as untested.
- [x] S2 **Verdict.** Width buys resolution: 256 to 1,024 took depth-1 lift from
      46.9x to 126.8x and 1,024 to 8,192 took it to 463.3x, so the flat-lift hypothesis
      is dead twice over and the first draft of Correction 2 was wrong. The width axis
      closes at 2 x 8,192 by decision rather than by saturation, so
      `docs/embedding_weakness.md` section 9 is rewritten around items per code.
- [x] S3 **Event stream, one preprocess pass.** Action token (exposure, click,
      unknown), gap bins 64 to 128, two code levels, hour and weekday behind a flag
      so they can be ablated. Default build is 4 tokens per event and 153.5M train
      tokens, against 221.5M today; with the flag on it is 5 and 187.5M. No session
      token, the gap already encodes the boundary. The script exists and is tested:
      `Plan 2/preprocess_digr_v2.py` with one shell wrapper beside it,
      `Plan 2/preprocess_digr_v2.sh`, which is a single python call plus `"$@"` so
      every arm is a flag rather than an edit. `--num_sid_levels 2 --add_action`
      builds the default stream, `--add_clock` adds the clock arm, `--no-add_action
      --n_temporal_bins 64` reproduces the v1 stream token for token, and
      `--expect_val_split DIR` holds the run to an archived split: it checks the val
      users in seconds before the tokeniser starts, then the val users, the
      catalogue and the val targets after the pass. That decision is mandatory, so
      a run must either reproduce an archived split or pass `--allow_new_split`. 14 tests in `Plan 2/test_preprocess_digr_v2.py`.
      **Built** in 877.6 s: 1,719,430 train rows, 98,304 val rows, 153,461,256 and
      8,808,890 tokens, vocabulary 16,662, event width 4, and the split check
      reproduced the frozen val users and targets exactly.
- [x] S4 **DiGR on the new data.** `Plan 2/train_digr_v2.sh`, frozen config, 300k
      negatives, three epochs, same evaluator, about 3h10m because the stream is
      153,461,256 tokens against 221,500,536. Two axes move together on purpose.
      One flag is not the frozen value: `--temperature 0.175` instead of 0.07,
      because the score carries `1/(L * temperature)` and L drops from 5 to 2, so
      0.07 would make the logits 2.5x sharper for a reason that has nothing to do
      with the data. Holding `1/(L*T)` at 2.857 keeps the score scale where the
      frozen run tuned it, leaving the SID and the stream as the only changes.
      Smoke it first with `--max_steps 30 --run_name smoke_v2`.
      **Run and finished** (`runs/digr_v2_sid2x8192_e3_sampled300k`, 3h11m, 63.3 min
      per epoch at 41.5k tok/s). Epoch 2 is the best checkpoint: val loss 11.0133
      against the frozen 11.2431, R@1 1.0400% against 0.7924%, R@10 3.6600% against
      2.5940%, nDCG@10 0.02170 against 0.01570, MRR@10 0.01710 against 0.01258,
      combined 0.31*HR + 0.69*NDCG 0.0263 against 0.0189. Against co-click window 20
      it is ahead on R@1 (1.29x), nDCG (1.06x), MRR (1.11x) and the combined score
      (1.03x), and level on R@10 (0.99x). Epoch 3 overfits and adds nothing, so two
      epochs is the setting to reuse. Numbers and caveats: `docs/results.md`
      section 12.
- [x] S4b **Full-catalog evaluation of the S4 checkpoint** with
      `eval_full_catalog.py` on `best/` over the same 98,304 rows and the same
      4,783,154 items, 1.7 min of GPU. R@1 1.0549%, R@10 3.6570%, R@100 10.9182%,
      R@1000 27.6622%, R@10000 55.7058%, nDCG@10 0.02190, MRR@10 0.01742, loss
      11.0133. Ahead of the frozen row at every cut-off (1.33x, 1.41x, 1.45x, 1.28x,
      1.12x). Ahead of co-click window 20 at ranks 1, 1,000 and 10,000 by 7.9, 16.5
      and 55 standard errors, level at ranks 10 and 100 where the gaps are 49 and 68
      rows. `P(top 10 | top 1000)` 13.22% against co-click's 14.65%. Table and
      reading: `docs/results.md` section 12.
      The first attempt at this used the frozen v1 `digr_data` against the v2
      checkpoint and produced R@10 0.0437% with loss 26.3504, which is now refused
      by `check_data_dir_matches_checkpoint()` in `eval_full_catalog.py` before any
      scoring. Eight tests in `Model Training/test_eval_full_catalog.py`.
- [ ] S5 **Baselines on the new data.** Popularity and co-click need no SID, so they
      are control rows and must reproduce to the digit, and the S3 build checked the
      val users, the catalogue and the val targets against the frozen split, so the
      published rows already apply. `sid_transition` is the one baseline that can
      move: it now reads 2-code tuples, which are far coarser than the 5-code ones it
      was built on. Minutes of CPU, and it belongs with the table rather than with
      the experiments.
- [x] S6 **TIGER on the SID that S1 and S4 endorse.** Promoted out of this list into
      Phase 4, which now carries the design. Deferred no longer: it is the missing
      row in the three-arm table and the only head that tests a different objective.
- [ ] S7 **Attribution, deferred by decision.** The arm that separates the three
      changes S4 made: same 2 x 8,192 codes, `--no-add_action --n_temporal_bins 64`,
      two epochs, about 2h10m. If the ranking holds near 1.41x then the codebook did
      all of it and the action token is optional; if it falls back toward the frozen
      numbers then the stream carried part of the gain. Deferred behind TIGER
      (m05706) because it explains an existing row rather than adding one, but it is
      the first question anyone will ask about section 12, and it is cheap.
- [x] S8 **Reranker over DiGR's top 1,000, dropped as out of scope** (m05965). A
      reranker scores another method's list, so it cannot answer which retrieval
      method is best, and its numbers are conditional on that list. The
      measurement stays on record because it describes the generator: the target
      sits in DiGR's top 1,000 on 27.66% of val rows, which a reranker could not
      push past, and reaches the top 10 on 13.22% of those rows.
- [ ] S9 **16,384 wide, deferred.** 292 items per level-0 code, inside the DiGR
      paper's 39 to 195 band, 4.7x the item-side parameters of 8,192 and 98 MB bf16.
      An hour for the quantiser and three for the model. Worth one run only after S7
      says whether the last widening's slope is still steep; the step from 256 to
      8,192 bought 41% at R@10, and the question is what the next 2x buys.

---

## Deferred: continuous-vector retrieval and ANN

The design that won both competitions, and the only one with per-item resolution.
Not part of the three-arm table, by decision, but it is the first thing after it.

- [ ] D.1 A query tower and one vector per item, trained with InfoNCE over a large
      negative bank, served by ANN. This is the arm with 1,024 parameters per item
      instead of 0.376.
- [ ] D.2 Action-type conditioning, per the published winners. Currently an
      explicit non-goal in the DiGR design, and the highest-value change with
      published evidence behind it.
- [ ] D.3 Richer time features: absolute timestamps, relative gaps, session
      structure, periodicity. We currently have one coarse temporal token.
- [ ] D.4 A data-scale check. 1.72M sequences and 34M events is small next to a
      competition dataset, and some of the 5x gap may be dose rather than design.

Do not start these before the three-arm table exists. Without a measured bar, any
of them can be declared a success.

---

## Next step

**`docs/direction.md` (2026-10-29) now governs the order below.** In short: run the
three CPU measurements in its section 4 first (SASRec bound, oracle union of the
three arms, target-universe rescore), then the query-side work in its section 3
(co-click code prior, window 60 / 3 anchors, mixture-of-intents head), each with a
stated prediction and a kill criterion. S9 is demoted by the 0.15 item-side slope,
and the fair TIGER arm is depth at `PREFIX=1`, not another release at two levels.
1. **Phase 5, the write-up.** The table is complete: popularity, co-click, DiGR v1,
   DiGR v2 and TIGER on one val split, one evaluator and one candidate space. What
   is left is 5.3 to 5.5 and the prose around them.
2. **S5, the control rows.** Minutes of CPU. `sid_transition` is the only row that
   can move, because it now reads two-code tuples instead of five.
3. **4.7, the three gaps the TIGER run exposed.** (a) print `cover@k` and `recall@5`
   and write a `metrics.json`; (b) reproduce TIGER's R@1 on a second code path;
   (c) run `PREFIX=1` for two epochs and a beam pass, which is the only SID arm that
   is not additive over the code rows.

Deferred behind those, in the order they are worth doing. S7, attribution, 2h10m,
because it explains the 1.41x rather than adding a row and it is the first thing a
reader asks. S9, the width run at 16,384, one GPU hour for the quantiser and three
for the model, held back until S7 says whether the last widening's slope is still
steep.

Phase 3b is out of scope (m05965). The item representation stays a function of
content alone, so the frozen content SID remains the reference for every table
already published, and nothing in Phase 6 or Phase 4 moves it. Reranking is out of
scope for the same reason the table exists: the question is which retrieval method
is best, and a reranker only reorders someone else's list.

# Tasks

Working list, ordered. Scope is fixed by `docs/problems.md` section 1: compare
three retrieval methods that use RQ-VAE Semantic IDs, all on the same backbone,
same data, same evaluator.

- **DiGR**: additive read of 1,755 shared code rows. Built, trained, frozen.
- **TIGER**: autoregressive factorisation over the five codes. Not built.
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

## Phase 3b: rebuild the item representation

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

## Phase 4: build TIGER (next)

**Promoted to next by decision** (user, 2026-10-28). The attribution arm that would
have separated codebook width from the action token is deferred behind it: the
deliverable is the three-arm table and TIGER is the only row still missing, so a
run that only explains an existing row can wait.

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

- [ ] 4.1 Head. Two per-level cross-entropies over 8,192 classes each, replacing
      DiGR's single 4,783,154-way softmax. `logits_l = h W_l` with `W_l` of shape
      `[1024, 8192]`. The parameter count lands at 16.8M, which is exactly DiGR's
      code-embedding table for this SID (16,384 rows of 1,024), so the two heads
      cost the same and differ only in how the columns are used: DiGR adds them into
      one item score, TIGER normalises each level separately. The loss for row `j`
      is `CE(p1, c1[target_j]) + CE(p2, c2[target_j])`.
- [ ] 4.2 Inputs. Append `SEP, c1, c2` after the last event, so the hidden state at
      `SEP` predicts `c1` and the hidden state at `c1` predicts `c2`. Teacher-forced
      during training, one forward pass gives both terms. Row length goes 89 to 92.
- [ ] 4.3 Inference. No beam, no trie, no constrained decoding: gather the two
      log-probability columns at each item's own codes and sort. That is exact over
      the full catalogue, so all five cut-offs up to 10,000 are reportable and the
      number is directly comparable with the DiGR row.
- [ ] 4.4 Reuse, do not rebuild: the training loop, the sampler, atomic
      checkpointing, `--resume`, `--save_steps`, `--log_interval`, the CSV log and
      the val-loss early stop all exist in `Model Training/train_digr.py`. Estimate
      400 to 600 lines in a new `Model Training/train_tiger.py`, less than the 600
      to 800 the five-level plan needed because the beam and the trie are gone.
- [ ] 4.5 Budget two epochs. 92 tokens against 89, and the head is cheaper than a
      4.78M-row gather, so expect about a DiGR epoch, 63 minutes, not the 1.3x to
      1.6x the per-level-token plan predicted.
- [ ] 4.6 Run all five cut-offs and write `R@1/10/100/1000/10000`, nDCG@10 and
      MRR@10 into `docs/results.md` beside the DiGR row. The bar it has to clear is
      co-click window 20: R@10 3.7069% and R@100 10.9873%. DiGR's own row is
      R@10 3.6570% and R@100 10.9182%, so the interesting question is whether
      predicting the codes step by step orders the middle of the list better than
      one additive score over the whole catalogue.

---

## Phase 5: the table and the write-up

- [ ] 5.1 One table, four rows (popularity, co-click, DiGR, TIGER), five cut-offs,
      one column for cost, one column for the candidate space.
- [ ] 5.2 The conditional-ranking column: "given the target is in the top 1,000,
      how often does it reach the top 10". This is the column that makes the
      failure legible, and it currently reads co-click 14.65%, DiGR 12.01%,
      popularity 9.12%.
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
- [ ] S8 **Reranker over DiGR's top 1,000, deferred.** CPU only. The ceiling is now
      measured rather than assumed: the target sits in DiGR's top 1,000 on 27.66% of
      val rows, so a perfect reranker reaches R@100 27.66% (2.5x the current 10.92%)
      and R@10 27.66% (7.6x the current 3.66%). Features: co-click successors,
      content cosine, popularity, recency, the SID prefix. It is the only remaining
      idea that attacks the 13.22% conversion without touching the towers.
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

1. **Build TIGER on the 2 x 8,192 SID** (Phase 4). It is the missing row, and the
   only experiment left that tests a different objective rather than a different
   input. Four hundred lines, two epochs, about two hours, and no trie or beam
   because two levels make the exact score a two-column gather.
2. **S5, the control rows.** Minutes of CPU, so the table's baselines sit on the
   same data as the model rows.
3. **Phase 5, the table and the write-up.**

Deferred behind those, in the order they are worth doing. S7, attribution, 2h10m,
because it explains the 1.41x rather than adding a row and it is the first thing a
reader asks. S8, the reranker over the top 1,000, CPU, with a measured ceiling of
27.66% at R@100 and 7.6x headroom at R@10. S9, the width run at 16,384, one GPU hour
for the quantiser and three for the model, held back until S7 says whether the last
widening's slope is still steep.

Phase 3b stays where it is. Nothing in Phase 6 or Phase 4 changes the item
representation, so the behaviour-SID question is untouched by any of the above and
the frozen content SID remains the reference for the tables already published.

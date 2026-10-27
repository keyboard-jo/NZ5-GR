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

## Phase 3: protocol, then the trie

These are small, cheap, and they decide whether TIGER's numbers mean anything.

- [x] 3.1 **Fifth-level protocol: decided.** The headline table scores **exact
      item identity** for all four arms. 39.3% of items share their first four
      codes and level 5 is an arbitrary rank inside the group, so a prefix hit
      would inflate recall by up to 39% relative and only for the SID arms. A
      prefix-hit variant is a footnote, never the headline. Confirmed by the user
      (m03114, paraphrased).
- [ ] 3.2 **Build the trie over the 4,783,154 code tuples.** Only 3,459,854 of the
      4,294,967,296 possible four-code prefixes exist, so unconstrained decoding
      would emit invalid codes almost always. A prefix index already exists in
      `Model Training/baselines_digr.py` (`build_prefix_index`), so this is a
      reuse, not new code.
- [ ] 3.3 **Measure decode cost for beam widths 10, 50, 100, 500.** R@k above the
      beam width is structurally unreachable, so the beam has to be at least 1,000
      for the R@1000 and R@10000 rows to be comparable. If 1,000 is too slow, the
      table gets a beam-width column and the comparison is stated per cut-off.
- [ ] 3.4 **Sliced evaluator.** At beam 1,000 the candidate set is 1,000 per query,
      not 4,783,154, so TIGER is scored by candidate-set ranking while DiGR and
      co-click are scored by full-catalog ranking. Same metric names, different
      candidate space. Write the one page that says so before writing the code.
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

## Phase 4: build TIGER (back burner)

**Deferred by decision.** TIGER is engineering rather than exploration: the design
is settled and the outcomes are predictable, a per-level head at or above DiGR's
2.5940%. It is worth one run eventually, but it is not where the information is, and
measuring a head on codes we already know are content-only understates what the head
can do. Phase 3b comes first. The frozen content SID numbers stay as the historical
reference for the table.

Design is settled: **direct SID input**, user feature block carried, temporal
tokens carried, same backbone, and the existing content SID. The input format
does not change, so
`tokens.npy`, `offsets.npy`, `targets.npy`, `LengthBucketBatchSampler`,
mid-epoch checkpointing and `catalog_sid_codes.npy` are all reused as they are.

- [ ] 4.1 Head. Replace the single full-catalog softmax with five per-level terms.
      The target for row `j` is `catalog_sid_codes[targets[j]]`, five codes, and
      the loss is the sum of five cross-entropies over 256, 256, 256, 256, 731.
      Level 5 has 731 classes, so the last level stays cheap.
- [ ] 4.2 Inputs. Append the five code tokens to the row after the last event, in
      the same layout the events already use: temporal token, then five code
      tokens. Teacher-forced during training, so one forward pass gives all five
      losses.
- [ ] 4.3 Inference. Trie-constrained beam, level by level, no teacher forcing.
      Report the beam width with every number.
- [ ] 4.4 Reuse, do not rebuild: the training loop, the sampler, atomic
      checkpointing, `--resume`, `--save_steps`, `--log_interval`, the CSV log and
      the early-stop-on-val-loss logic all exist in `Model Training/train_digr.py`.
      Estimate 600 to 800 lines in a new `Model Training/train_tiger.py`.
- [ ] 4.5 Budget one epoch first. The target sequence is six tokens instead of
      one, so expect 1.3x to 1.6x a DiGR epoch, roughly 2 to 2.5 h. Extend with
      `--resume` only if the curve is still moving.
- [ ] 4.6 Run all five cut-offs, write `R@1/10/100/1000/10000`, nDCG@10, MRR@10,
      and the beam width into `docs/results.md`.

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
Two claims to hold it to. A wider codebook is a capacity win for the head, 0.376
to 0.88 parameters per item, and not a resolution win, because the geometry already
sits at its packing bound. And the event stream is discarding the
exposure-versus-click distinction on 89.4% of all events.

The measured occupancy it builds on, 4,783,154 items: depth 1 is 256 tuples of
18,684.20 items, depth 2 is 57,122 of 83.74, depth 3 is 1,506,041 of 3.18, depth 4
is 3,459,854 of 1.38, depth 5 is injective. All codebooks are at 7.87 to 7.98 bits
of 8, so nothing is collapsed.

- [ ] S0 **Instrument.** Depth occupancy, collision count and bucket lift at depths
      1 and 2, for any SID directory, against the same 98,304 val rows. Without
      this there is no denominator for S1, and the release cannot be compared.
- [ ] S1 **Width sweep, about one GPU hour.** 4 x 256 as the control, 4 x 1,024, and
      2 x 16,384 to match OnePiece's resolution. Nothing else changes. Report
      occupancy, collisions and lift.
- [ ] S2 **Verdict, no compute.** Flat lift means the capacity argument is the
      reason to adopt a wider SID and `docs/embedding_weakness.md` section 9 stands.
      A moving lift means section 9 is wrong and gets rewritten.
- [ ] S3 **Event stream, one preprocess pass.** Action token (exposure, click,
      unknown), gap bins 64 to 128, drop the fifth level only if S1 reports under
      0.1% collisions, absolute time behind a flag so it can be ablated. Target is
      6 tokens per event and the same 221.5M train tokens as today.
- [ ] S4 **DiGR on the new data.** Frozen config, 300k negatives, three epochs, same
      evaluator. Two axes move together here on purpose.
- [ ] S5 **Baselines on the new data.** Popularity and co-click need no SID, so they
      are control rows and must reproduce to the digit. `sid_transition` is the one
      baseline that can move.
- [ ] S6 **TIGER on the SID that S1 and S4 endorse**, once, then the three-arm table.

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

Phase 6, in order, and the first three steps are cheap.

1. S0, the instrument. Occupancy plus bucket lift in one script, so any SID can be
   scored in minutes instead of an afternoon.
2. S1, the width sweep. One GPU hour and half a day of CPU. It answers whether the
   geometry is the binding constraint or the head is, and both answers are useful.
3. S3, the event stream. The action token is the highest-value single change
   available: 89.4% of events in a history are exposures and the model cannot tell
   them from clicks.

Phase 3b waits behind S1, because if the width sweep confirms the geometry is
saturated then 3b.0 is the only change that can move it. The frozen content SID
stays untouched while Phase 6 is in flight, so the DiGR and baseline rows remain
valid as the reference. `docs/plan_v2.md` states both predictions before the runs.

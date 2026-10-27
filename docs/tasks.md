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

Two rules govern the order below.

- **One axis moves at a time.** The SID set is frozen before the three-arm table
  is measured, so a method comparison is never confounded by an embedding change.
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

## Phase 3b: inject behaviour into the item side

Read `docs/embedding_weakness.md` sections 8 to 11 first. Content stays the base
because it covers every item. Behaviour is added on top, never instead, and the
67% never-clicked items keep the pure content prior. Every item still gets a full
five-code SID, so the hybrid lives inside the embedding and not in the
tokenisation.

- [x] 3b.1 **Centroid distillation (D6). Dropped, not deferred.** The plan was to
      average the user vectors of each item's clickers into that item's vector, as
      an hour-long gate on whether behaviour in the item space helps at all. 3c.5
      answered the gate without running it: the vectors that would be averaged
      score 0.4354% R@10, below the 1.3438% of a frequency count, so averaging them
      cannot produce a strong item vector. See `docs/embedding_weakness.md`
      section 7.
- [ ] 3b.2 **Behavioural SID arm (D3). The real embedding experiment.** Truncated
      SVD of the train co-click matrix, dim 256, rows L2-normalised, zero-filled
      for the 67% never-clicked items, then
      `e_i = normalize(MLP(content_i) + W * b_i)` and quantise the combined space
      with a fresh RQ-VAE. Same DiGR head, same backbone, same evaluator, one axis
      changed: the codes. The matrix it starts from already scores 3.7069% R@10 on
      its own, so the question is what survives quantisation. Run it after TIGER,
      so the head is measured on a SID that stayed frozen for both.
- [ ] 3b.3 **Watch injectivity.** A behaviour-clustered space concentrates popular
      items into few 4-code prefixes, and level 5 has only 731 codes to
      disambiguate inside a prefix. Check the largest 4-prefix bucket before
      training. If it exceeds 731 the SID is no longer injective and needs more
      tail capacity or a collision resolver, which is the mechanism the
      competition's Technical Innovation winner built for its second level.
- [ ] 3b.4 **Gate it.** Same 98,304 val rows, same cut-offs, same catalog. If the
      behaviour SID does not beat 2.5940% R@10 under the DiGR head, content codes
      are not the binding constraint and the head is. Stop there.
- [ ] 3b.5 Stage 1 retrain, the high-ceiling arm (D2, D1, D5). Sampled softmax over
      100k to 380k catalog negatives instead of in-batch 1024, a real `log_q`
      sampling distribution, a per-item behavioural residual `r_i` initialised to
      zero so cold items keep the content prior, and gated (FiLM) action fusion
      instead of addition. Re-export, re-quantise, then re-run DiGR and TIGER.
      Only open this after TIGER and 3b.2 have both landed. It is the most
      expensive instrument and it bundles several unknowns, so it goes last.

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

## Phase 4: build TIGER

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

TIGER is unchanged and stays first in the queue. It runs on the existing content
SID, and its bar is R@10 3.7069% and R@100 10.9873% (co-click window 20). Nothing
in the embedding work moves that bar.

Two things run in parallel, both cheap, neither of them a retrain.

- 3.2 to 3.4, the trie, the decode-cost measurement and the sliced-evaluator note,
  which TIGER needs anyway.
- The write-up of what the probes closed: the item side and the user side are both
  below popularity, D6 is dropped, and the behavioural SID (3b.2) is the next
  embedding experiment rather than a Stage 1 retrain.

After TIGER, run 3b.2 on a SID derived from the co-click matrix. That is the
cheapest test of whether codes can carry behaviour, and it reuses the head TIGER
leaves behind.

Do not open 3b.5 (the Stage 1 retrain) until TIGER and 3b.2 have both landed, and do
not change the SID set between the DiGR and TIGER numbers.

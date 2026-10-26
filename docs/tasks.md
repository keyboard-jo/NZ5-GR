# Tasks

State as of 2026-10-26. DiGR is trained, measured and **frozen** at
`digr_0p6b_e3_full_sampled300k/final`, the baselines are in, and the docs are
written. TIGER is the next build because it is in the proposal. This file is the
order of work, with the reason each item exists.

The core problem the ordering solves: TIGER changes the **objective** but keeps the
**Semantic ID**. Nothing measured so far separates three very different ceilings.

- the LLM encoder is weak
- the SID representation is the ceiling
- the next-click task is mostly noise at this data size

If the second or third is true, TIGER lands where DiGR landed and the proposal's
head-to-head reads badly. The items in Phase 1 and 2 are what make TIGER's number
interpretable instead of just another point. They are all CPU-minutes or one short
GPU job, and they have to happen before the code gets written.

---

## Phase 0. Freeze the record

- [x] **Decide DiGR is frozen at `digr_0p6b_e3_full_sampled300k/final`.**
      Four more GPU-hours on DiGR delays TIGER and cannot beat co-click.
      Done: `results.md` names that one checkpoint and stops citing the others.
- [x] **Fix `REF_POPULARITY_R10`** to exactly `1321 / 98304` (1.3438%). The old
      constant was `1320 / 98304` = 1.3428%, one row short, and the copy in
      `eval_full_catalog.py` had drifted in the eighth digit as well.
      Done: both modules now carry the fraction, and a test asserts they agree.
- [x] **Add a `docs/README.md` index** listing `math.md`, `data.md`,
      `sid_math.md`, `problems.md`, `results.md`, `tasks.md`, one line each,
      so a reader knows where to start.

---

## Phase 1. The cheap measurements that decide what TIGER's result means

All of these are CPU, minutes, and each one can change the conclusion.

- [ ] **1.1 co-click at window 10 and 20.** 46 s each.
      Why: the window sweep has not plateaued (window 3 to 5 still adds 6.8% at
      R@1 and 6.6% at R@10), so the bar TIGER has to clear is still moving. Right
      now it is R@10 3.6011% / R@100 10.4492% at window 5. Building TIGER against
      a bar that turns out to be 4.5% would waste the run.
      Done when `baselines_h10.json` and `baselines_h20.json` exist and the bar is
      written down as a single number.
- [ ] **1.2 Repeat-consumption rate.** On the 98,304 val rows, compute
      `P(target == last event)`, `P(target in history)`, `P(target in the co-click
      successors of the last 5)`.
      Why: this is the quantity co-click is really exploiting. If 30% of targets
      are literally the previous item, the task is partly memorisation and every
      model number is capped by it. It also measures the headroom TIGER can win by
      simply attending to its own input.
      Done when the three fractions are in `results.md`.
- [ ] **1.3 Train and val user overlap.** `users.npy` exists in both splits.
      Why: if the same users appear in train and val, co-click memorises exact
      user-event pairs through the item graph and its advantage is partly a split
      artefact. Cheap to check, and it is the first question a reviewer asks.
      Done when overlap is reported as a fraction of val rows.
- [ ] **1.4 Does the SID carry co-click signal at all?** Compare
      `P(target's depth-1 code | last item's depth-1 code)` against the marginal
      base rate `1/256 = 0.39%`, and the same at depth 2 against `1/65536`.
      Why: B3 already proved first-order code transitions are useless because
      levels 0 to 3 are 100% dense over 65,536 cells with 34M transitions on them.
      This measures whether the codes are hopeless or merely badly used. If depth
      1 gives no lift, TIGER-over-SID cannot beat co-click-over-item-ids and that
      has to be known before the run, not after.
      Done when both lifts are printed next to the 1/256 base rate.
- [ ] **1.5 Target-side coverage.** How many of the 98,304 val targets appear in
      the train target set, and how many are items with zero training exposure.
      Why: a target that never appeared in train cannot be learned, only
      generalised to. This sets the floor that no model can cross and explains part
      of the 50% miss at R@10000.
      Done when the coverage fraction is in `results.md`.

---

## Phase 2. The one experiment that tests the whole premise

- [ ] **2.1 SASRec or GRU4Rec over raw item ids, same sequences, same split.**
      Roughly 15 to 30 min on the H800, or a few hours on a big CPU.
      Why: this is the only measurement that separates "the representation is the
      ceiling" from "the task is the ceiling". It uses no SIDs, no content, no LLM,
      just 4,783,154 item embeddings and the same 1,719,430 sequences.
      - If it reaches R@10 around 15%, the SID is the bottleneck. TIGER keeps the
        SID, so it inherits the bottleneck, and the honest proposal framing becomes
        "we reproduce the published framework and show the representation is the
        limit", which needs the number to make the claim.
      - If it lands near 7%, the ceiling is the task and the data, and no
        generative-retrieval variant will clear it.
      - If it lands near 3%, then co-click's edge is memorisation only and the
        LLM story is intact.
      This is also the standard baseline any reviewer will ask for, so it is not
      optional work.
      Done when R@1 / R@10 / R@100 are in `results.md` on the same 98,304 rows.
- [ ] **2.2 A matched-capacity item-id baseline if 2.1 is close to DiGR.**
      Matrix factorisation or EASE over item ids, R@100 style.
      Why: DiGR's scoring head is 1,755 code rows times 1,024 = 1,797,120
      parameters. The co-click table holds 25,669,281 distinct pairs, 14.3x more
      capacity. That gap alone could explain the whole result, with no statement
      about the LLM either way.
      Done when one capacity-matched model has a number.

---

## Phase 3. Measurement infrastructure TIGER needs and does not have

- [ ] **3.1 Build the prefix trie.** 3,459,854 valid four-code prefixes out of
      `256^4 = 4,294,967,296`, so 0.08% of combinations are legal. Unconstrained
      generation would emit garbage IDs almost always. Minutes of CPU, one artifact
      next to `digr_data/`.
      Done when the trie loads and enumerates the top-k continuations of a prefix.
- [ ] **3.2 Measure the decode cost before committing to a beam size.** Constrained
      beam decoding is 4 sequential forward passes per step width, so it is far
      more expensive than `evaluate()`'s single item-logit pass. Beam 100 is needed
      to report R@100 because beam size caps reachable k.
      Why: unknown cost, and it decides whether the TIGER eval is 3 minutes or 3
      hours. Measure on 2,000 rows first.
      Done when a seconds-per-row figure exists for beam 10 / 100 / 1000.
- [ ] **3.3 Decide and document the fifth-level protocol.** 1,878,898 items (39.3%)
      share a four-code prefix, and the level-5 code has 731 values whose ordering
      is an arbitrary rank, not content.
      Why: this single choice changes every TIGER number.
      - score five codes: level 5 must be predicted from co-click within the group,
        which is a memory task, not a language task
      - score four codes and call a shared prefix a hit: inflates recall by up to
        39% relative to DiGR and co-click, which are scored on exact item identity
        - score four codes strictly, requiring the full five: keeps comparability
        and loses whatever the level-5 rank could have given
      Pick one, write the reason in `results.md`, and never quote a TIGER number
      without it.
- [ ] **3.4 Build a sliced evaluator.** Every number so far is one average over
      98,304 rows. Add slices: cold user (no history in train), warm user, repeat
      target (already in history) versus new target, and head / body / tail of the
      popularity distribution.
      Why: this is where the proposal's real claim lives. co-click wins the average
      because it memorises pairs; DiGR's only structural advantage is a genuine
      catalog-wide ranking with a long tail, and a short candidate list cannot do
      that. A single average actively hides the difference the work is about.
      Done when DiGR, co-click and popularity all report the same slices.

---

## Phase 4. Build TIGER

Confirmed already: direct SID as input (not title text), plus the user feature
block and the temporal token, on the unchanged Qwen3-0.6B, so DiGR versus TIGER is
a single-variable ablation of the objective.

- [ ] **4.1 Target construction.** `catalog_sid_codes[targets]` plus `sid_offsets`,
      five positions, teacher forced, loss on those positions only.
      Zero preprocessing change: the row already carries SIDs in the same vocab.
- [ ] **4.2 The factorised objective and loss.**
      `P(item) = P(c1) P(c2|c1) P(c3|c1c2) P(c4|c1c2c3) P(c5|...)`, five CE terms
      per row instead of one softmax over the catalog. Reuse the existing dataset,
      collate, sampler, checkpointing and resume machinery unchanged.
- [ ] **4.3 Trie-constrained inference** using 3.1, reporting at whatever the
      protocol in 3.3 requires.
- [ ] **4.4 Run it.** Expect about 90 minutes per epoch, the same as DiGR, because
      the 137-token forward pass dominates and the five extra prediction positions
      are almost free. Three epochs is 4.6 GPU-hours, same budget as DiGR.
- [ ] **4.5 Report against the real bar**, which is co-click at the best window
      from 1.1, not DiGR at R@10 2.5940%. Also report the slices from 3.4, since
      the average may not be where TIGER wins.

---

## Phase 5. Write it up

- [ ] **5.1 Decide the claim before writing the prose.** Current evidence says a
      four-second co-click count table beats a 4.6-GPU-hour model at every cut-off
      below 1,000. The defensible claims that survive are: catalog-wide ranking
      with a long tail, cold-start handling, and user-feature conditioning. The
      indefensible claim is "beats the baselines". Write to the first list.
- [ ] **5.2 `results.md` gets the honest table**, with compute cost next to every
      metric, because the 4,100x cost ratio is itself a finding.
- [ ] **5.3 `problems.md` keeps the negative results**: B3 dead by construction, the
      label-subset experiment, the popularity metric bug, the 62% extrapolation that
      measured 49.77%, the subset-space checkpoint selection.

---

## Compute

4.6 GPU-hours spent on DiGR. Remaining Phase 1 and Phase 3 work is CPU only.
Phase 2.1 is roughly 0.3 to 0.5 GPU-hours. Phase 4 is 4.6 GPU-hours plus whatever
3.2 shows decoding costs. Budget about 6 GPU-hours to finish, excluding sweeps, and
probe any sweep on 1,000 steps (about 26 minutes) rather than a full run.

## The short answer to "what before TIGER"

Three things, in this order.

1. **2.1, the item-id sequential baseline.** One short GPU job that decides whether
   TIGER can possibly win, and therefore how the proposal's results chapter is
   framed. Everything else in Phase 1 is cheap context for it.
2. **1.1, the co-click window sweep to 10 and 20.** TIGER needs a fixed number to
   be measured against, and the current bar is still rising.
3. **3.3 and 3.4, the fifth-level protocol and the sliced evaluator.** Both change
   TIGER's numbers before a single line of training code is written, and both are
   decisions, not work.

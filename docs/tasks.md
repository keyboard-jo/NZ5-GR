# Tasks

Three things, in the order that changes what we can claim. The first is the one the
co-supervisor's review asks for and the only one that is blocking.

## 1. Ablation: the new SID versus the new features (do this first)

**Why.** S4 moved three axes at once: the codebook went from `4x256+731` to `2x8192`, the
action token was added, and the gap bins went from 64 to 128. DiGR v2 is 1.41x v1 on R@10
and we cannot say which of the three bought it. That is the gap the supervisor flagged, and
it is a real gap: the current write-up implies the codebook did it, and that is unproven.

**The 2x2.** Two cells are already measured, two need a run.

| SID | features | data dir | run | R@10 |
| --- | --- | --- | --- | --- |
| `4x256+731` | v1: no action token, 64 gap bins | archive v1 | measured | 2.5940% |
| `2x8192` | v2: action token, 128 gap bins | plan2 v2 | measured | 3.6570% |
| `2x8192` | v1 | new | **to run** | ? |
| `4x256+731` | v2 | new | **to run** | ? |

**How.** `Plan 2/preprocess_digr_v2.py` owns both switches, so each mixed arm is one
preprocessing pass plus one training run. The SID is an input (`--catalog_path` plus
`--catalog_sid_cols`), the features are flags (`--no-add_action --n_temporal_bins 64` gives
the v1 layout).

**Cost.** One preprocessing pass per new arm, then one 2-epoch run at about 2.1 GPU-h each.
Two arms, so roughly 4-5 GPU-h plus two IO passes. Everything else held fixed: 2 epochs,
same negatives, same 0.6B backbone.

**How to read the result.** The two mixed cells split the 1.41x. If `2x8192` with v1 features
is already near 3.66%, the codebook is the whole story and the features are noise. If it sits
near 2.59%, the codebook did nothing and the features did the work. A result in between means
both, and the 2x2 gives the split without needing an interaction term.

## 2. Diagnose the new SID, including full-SID collisions

**Why.** Already done once in `archive/sid_results.md` and `archive/sid_diagnostics.md`; redo
it against v2 and add the collision check the supervisor asked for.

**Known already, so do not re-measure from scratch.** 53.51% of items share a 2-code tuple,
the largest group is 315 items, and the tie ceiling is R@1 53.57% / R@10 93.20%. Collisions
are structural at two levels, not a defect, so the useful question is not whether they exist
but how much of the achieved R@k the ceiling already explains.

**To do.** Full-tuple and per-level collision sizes; the tie ceiling for every arm at its own
cut-off; items per level-0 code against the 39-195 the DiGR paper's own catalogues sit at;
and the unrun cells in `archive/sid_results.md` (`2x16384`, `3x16384`) if they are still
wanted.

**Cost.** Seconds to minutes, CPU. Deferred only because item 1 is what changes the claims.

## 3. Bigger backbone (deferred, last)

**Why last.** DiGR's ceiling looks like signal rather than capacity: P(top10|top1000) sits at
13-15% for every arm, including popularity at 9.12% with no query at all. TIGER is the more
plausible beneficiary because it fits conditional distributions over 8,192 outcomes per
prefix, and its conditional level-2 loss of 4.7051 against the unconditional 7.6731 says
there is structure left to fit.

**What.** 1.7B against the current 0.6B, same SID and same features, 2 epochs, both arms.
One row per arm beside the 0.6B rows, same cut-offs, same data.

**Caveat to state out loud.** This project has never run a backbone other than 0.6B, so any
claim about scaling right now is a prediction. It becomes a measurement or it does not go in
the write-up.

## Not in scope

Reranking, behavioural item representations, and ANN retrieval. Reasons are in
`archive/problems.md` section 1 and none of them changed.

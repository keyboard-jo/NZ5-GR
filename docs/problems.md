# Where this project stands, and why

This file records the direction of the work: what we tried, what it forced, what
is still unresolved, and what evidence would flip the current plan. It is
deliberately about decisions, not about code.

---

## 1. Status in one paragraph

The DiGR implementation works end to end. It trains a 0.6B LLM backbone over
RQ-VAE Semantic IDs, scores the full 4.78M catalog, and beats a popularity
baseline. It also loses to a 46-second co-click count table at every cut-off
from rank 10 to rank 1,000, winning only at rank 1 (7 rows) and rank 10,000. That single comparison is why the plan changed. The current
plan is to stop tuning DiGR, keep it as the frozen reference, and test whether
the *objective* was the ceiling by running TIGER on the identical backbone, data,
SIDs, features and evaluator. Before that, the SID itself needs to be
interrogated, because it is the one input every method shares and nobody has
measured it properly.

---

## 2. How we got here

| step | what it established |
|---|---|
| DiGR implemented, trained 1 epoch | the model beats popularity at R@1 1.1x, R@10 1.9x. Train loss and val loss agreed to four decimals, which is the signature of memorising the marginal only |
| `evaluate()` bug found | miss rank was 0, so recall counted misses as hits and nDCG was infinite. Every metric before this fix is void |
| `--label_subset clicked` tried | 3 epochs on 504,404 labels looked better until it was scored on the full catalog: loss 11.4362 against 11.3160 for 1 epoch of full labels. The subset was solving a problem the sampled softmax had already solved |
| full labels, 3 epochs, 300k negatives | best DiGR: R@10 2.5940%, 4.6 GPU-h. Val loss bottomed at epoch 2 (11.2342) and rose at epoch 3 (11.2431) while train loss kept falling to 10.9344, a 0.31-nat gap. More epochs is dead |
| popularity re-measured | the recorded R@10 of 0.9644% was wrong, the correct value is 1.3438%. Every "lift over popularity" claim was 39% too generous at R@10 |
| B3 SID transition | 18x worse than popularity at R@1. A first-order table over 256-way codebooks has 519 observations per cell, which is not enough for a conditional to beat a marginal. Dropped |
| B4 co-click, window sweep | window 5 wins: R@10 3.6011%, R@100 10.4492%, in 46 s of CPU. It beats DiGR at R@10, R@100, R@1000, nDCG@10 and MRR (DiGR edges R@1 by 7 rows) |
| DiGR at R@10,000 | 49.7691% against co-click's 44.1905%. DiGR's wins are R@1 (7 rows) and the far tail, at 4,100x the compute |
| TIGER chosen as the next run | the one experiment that changes the objective while holding everything else fixed |

---

## 3. The result that changed the direction

Same 98,304 val rows, same targets, same 4,783,154-item catalog.

| method | cost | R@1 | R@10 | R@100 | R@1000 | R@10000 |
|---|---|---|---|---|---|---|
| popularity | under 1 s CPU | 0.4395% | 1.3438% | 4.7862% | 14.7288% | 36.8215% |
| B3 SID transition | 63 min CPU, 8 workers | 0.0244% | 0.1801% | 0.9410% | 4.2236% | - |
| B4 co-click, window 5 | 46 s CPU | 0.7853% | 3.6011% | 10.4492% | 23.6176% | 44.1905% |
| DiGR 0.6B, 3 epochs | 4.6 GPU-h | 0.7924% | 2.5940% | 7.5094% | 21.6003% | 49.7691% |

Read plainly: a table of which items were clicked after which other items, built
in 46 seconds from the same training rows, recovers more relevant items in the
top 1,000 than a 0.6B language model that was trained for four and a half GPU
hours. The model wins only at rank 1 (by 7 rows) and past rank 1,000, and at
rank 10,000 its advantage is 5.6 points, which is roughly what you get from having a smoother score function over
the 3.2M never-clicked items rather than a hard count table.

Two honest caveats, both of which cut against the model and not for it:

- the co-click candidate list is about 27 items deep before popularity padding
  (mean out-degree 5.37 times a window of 5), so its R@1000 and R@10000 are
  mostly the popularity list. Its wins at R@10 and R@100 are the real ones,
  because those ranks are inside the data-backed part of the list.
- co-click has 25,669,281 learned counts against DiGR's 1,797,120 score-head
  parameters, so this is not "the model is badly trained". It is closer to
  "a 25M-entry count table is a strong model for this signal, and an additive
  read of 1,755 shared code vectors is not".

---

## 4. Three candidate ceilings

Everything is currently explained by one of these. They are not exclusive, but
they predict different next steps, so they need to be separated.

**A. The task is close to noise.** Next click from 20 events and 8 static
features may simply not be predictable much beyond popularity plus recent
co-occurrence. Predicts: every method converges to the same band, hyperparameter
changes move R@10 by less than the gap between methods, and more capacity or more
epochs buy nothing.

**B. The objective or architecture is the ceiling.** DiGR's score is
`(1/(L tau)) * h . mean_l E_l[c_il]`, an additive read of 1,755 shared code
vectors. It cannot express "this level-2 code matters only when level 0 is that".
TIGER's autoregressive factorisation can. Predicts: TIGER pulls clearly ahead at
R@10 and R@100 while DiGR stays where it is.

**C. The SID is the ceiling.** The RQ-VAE was fit to item content similarity, not
to next-click behaviour, so two items with the same four codes are not
necessarily items a user would click next. Predicts: TIGER also lands near
co-click, because it is reading the same codes with the same loss of information.

Current best guess is B and C together. B is cheap to test and C is cheap to
measure, so both come before anything expensive.

---

## 5. Why TIGER next, and what it does not fix

TIGER is the right next run because it changes exactly one thing. Same Qwen3-0.6B,
same `digr_data`, same SIDs, same user-feature block, same temporal tokens, same
`eval_full_catalog.py`. Only the head changes, from one full-catalog softmax over
an additive score to five per-level softmaxes with `h` re-read at each level.

That makes it a clean test of B. If TIGER lands at co-click's numbers, B is false
and C moves to the front. If it lands clearly above, B is confirmed and the 1.8M
parameter head was the problem all along.

What it does not fix:

- It does not address A. If the task is near-noise, TIGER will plateau too, one
  week later.
- It does not address C. TIGER reads the same codes.
- It cannot report R@100 with a beam smaller than 100, and beam search is
  approximate, so its numbers are a lower bound. DiGR and co-click score every
  item exactly. This asymmetry has to be stated whenever the table is shown.

There is also a cheaper, less fashionable test in the same direction that has not
been done: a small sequential item-id model (SASRec or GRU4Rec, a few million
parameters, 15 minutes on one GPU) with per-item embeddings. That tests B
directly, because per-item embeddings have no shared-code bottleneck at all. It
is arguably a better use of one GPU hour than TIGER, and it is the strongest
missing baseline in the whole project: we currently have no sequential neural
baseline in item space, only a count table and a code-space model.

---

## 6. The SID question, now at the front

The SID is the one input every method shares. If it is the ceiling, then no head
and no backbone will help, and the RQ-VAE needs retraining or replacing.

What is already known, and what it does and does not prove:

- The code is not broken. All 4,783,154 five-code tuples are unique, so there are
  no collisions. The collision statistics that looked alarming (1,878,898 items
  "with duplicate SID", up to 731 per group) are counts over *four-code* prefixes,
  which is by construction, because level 5 exists to break them.
- 39.3% of items share a four-code prefix with at least one other item, and 60.7%
  are alone in theirs. So the code discriminates most items, and the fifth level
  is mostly a no-op.
- The user's own probe showed a level-0 code agreement of 71% between query and
  target against a 0.39% base rate. Coarse codes carry real signal.
- What is *not* known: whether the fine codes carry next-click signal, or only
  content similarity. Depth-3 prefixes leave 84% of queries with no candidate at
  all, which is exactly what a code that encodes content detail (colour, brand,
  size) rather than intent would look like.

The checks that would settle it, all cheap and none needing a GPU:

1. Compare the lift of "target shares the query's level-0 code" with the lift of
   "target appears in the query's co-click list". If they are similar, the code
   carries about as much as the count table and DiGR has no new signal to add.
   If co-click is much stronger, the code is throwing away behaviour.
2. Repeat for depth 2, 3, 4 and check whether the lift decays smoothly or
   collapses. A collapse means the deeper codes are noise for this task.
3. Measure how much of the val target set is even reachable: coverage of val
   targets in train targets, and user overlap between splits. If most val rows
   predict an item the model has never had a chance to learn, A dominates.

---

## 7. Numbers we had to retire

Kept here so nobody re-quotes them.

| retired | why | current |
|---|---|---|
| all metrics before the `evaluate()` fix | miss rank was 0, recall counted misses, nDCG infinite | `Model Training/train_digr.py:939` |
| popularity R@10 0.9644% | came from a different frequency vector; ranks 2-10 did not reproduce | 1.3438% |
| "DiGR beats co-click" | compared against window 1, the weakest co-click form | co-click window 5 wins at R@10 and R@100 |
| "DiGR will reach about 62% at R@10,000" | log-linear extrapolation of the recall curve | measured 49.7691% |
| "`--label_subset clicked` is the fix" | made no epoch faster, and put 81.5% of probability mass on items it had never trained on | reverted to full labels |
| "5-code SIDs have 1.88M collisions" | those are four-code prefix groups | zero collisions over five codes |
| DiGR val-loss-best checkpoint | epoch 2 has the best val loss, epoch 3 ranks best | use `final`, not `best` |

---

## 8. Open decisions

1. Order of work. Current plan: these docs, then SID checks, then TIGER. The
   alternative is TIGER first, because it is the only thing that tests B and it
   is already specified.
2. Whether to add the item-space sequential baseline (SASRec or GRU4Rec) before
   or after TIGER. It is one GPU hour and it tests B from the other side.
3. Which input TIGER reads. The plan is the SID, not item text, so that the
   comparison isolates the objective. Text input would be a different experiment.
4. Whether the eventual deliverable is a model or a pipeline. If co-click plus a
   reranker already serves the product, the honest output of this project may be
   "the generative retriever is not worth 4,100x the compute on this data", which
   is a legitimate result and should be written down as such rather than buried.
5. What the acceptance bar is for TIGER. Proposed: it must beat co-click at
   R@10 and R@100, not just beat DiGR.

---

## 9. What would change the current plan

- TIGER lands at or below co-click at R@10. Then B is false, C and A lead, and the
  next move is the RQ-VAE, not more models.
- The SID checks show code-sharing lift far below co-click lift. Then C is
  confirmed regardless of what TIGER does, and the SID has to be rebuilt with a
  behavioural objective before any generative retriever is worth training.
- The item-space baseline matches co-click. Then nothing about Semantic IDs is
  load-bearing and the whole framework choice is the wrong question.
- TIGER clearly beats co-click at R@10 and R@100. Then B is confirmed, the head
  was the problem, and the question becomes whether a better head on the same
  codes can be trained in reasonable GPU hours.

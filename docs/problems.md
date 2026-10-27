# Where this project stands, and why

This file records the direction of the work: what we tried, what it forced, what is
still unresolved, and what evidence would flip the current plan. It is about
decisions, not about code.

State as of 2026-10-26.

---

## 1. Scope

The question this project answers is: **among retrieval methods that use RQ-VAE
Semantic IDs, which one is best on this data?**

Three arms, everything else held fixed (same Qwen3-0.6B backbone, same
`digr_data`, same SID vocabulary, same user feature block, same temporal tokens,
same `eval_full_catalog.py`, same 98,304 val rows).

| arm | what scores the item | learning |
|---|---|---|
| **DiGR** | additive read of 1,755 shared code rows | learned, trained, frozen |
| **TIGER** | autoregressive factorisation over the 5 codes | learned, not built yet |
| **heuristic** | count tables over item ids and over codes | none |

Continuous-vector retrieval, meaning a learned query encoder plus one vector per
item plus ANN, is the design all three competition winners used. It is deliberately
**out of scope for now**. It is the strongest comparison and it comes after the
three-arm table exists, not before.

---

## 2. Status in one paragraph

DiGR works end to end. It trains a 0.6B backbone over Semantic IDs, scores the full
4,783,154-item catalog and beats popularity. It loses to a 47-second co-click count
table at every cut-off except rank 10,000. A paper reporting 18.85% HR@10 on the
same competition data says that gap is real and not a metric artifact, which makes
the failure a property of our model rather than of the task. The diagnosis is now
specific: the item side of DiGR's score has 0.376 free parameters per item, so it
cannot store a transition pair, while the count table stores 25.7M of them. TIGER
is the next build because it keeps the SID and changes only the objective, which is
the one variable the diagnosis says is worth changing.

---

## 3. How we got here

| step | what it established |
|---|---|
| DiGR implemented, trained 1 epoch | beats popularity 1.1x at R@1, 1.9x at R@10. Train and val loss agreed to four decimals, the signature of learning the marginal only |
| `evaluate()` bug found | a miss got rank 0, so recall counted misses as hits and nDCG was infinite. Every metric before this fix is void |
| `--label_subset clicked` tried | 3 epochs on 504,404 labels looked better until scored on the full catalog: loss 11.4362 against 11.3160 for 1 epoch of full labels. It also put 81.5% of probability mass on items it had never trained on. Reverted |
| full labels, 3 epochs, 300k negatives | best DiGR: R@10 2.5940%, 4.6 GPU-h. Val loss bottomed at epoch 2 and rose at epoch 3 while train loss kept falling to 10.9344, a 0.31-nat gap. More epochs is dead |
| popularity re-measured | the recorded R@10 of 0.9644% was wrong, the correct value is 1.3438% |
| B3 SID transition | 18x worse than popularity at R@1. A first-order table over 256-way codebooks has 519 observations per cell, not enough for a conditional to beat a marginal. Dropped |
| B4 co-click, window sweep | window 20 wins: R@1 0.8057%, R@10 3.7069%, R@100 10.9873%, in 47 s of CPU, and it beats DiGR at every cut-off but rank 10,000 |
| item-embedding kNN probe | the 256-d item vectors, queried by pooled history, lose to popularity at every cut-off. The content geometry does not encode intent |
| SID prefixes scored against real targets | depth-1 bucket holds the target 18.3156% of the time at 19,308 candidates, depth-2 2.5167% at 345.9. The 71% figure previously quoted was self-graded and is retired |
| OnePiece comparison | 18.85% HR@10 on the same competition data, an 7.3x gap at HR and 6.8x at nDCG. The failure is ours, not the task's |

---

## 4. The result that changed the direction

Same 98,304 val rows, same targets, same 4,783,154-item catalog.

| method | cost | R@1 | R@10 | R@100 | R@1000 | R@10000 |
|---|---|---|---|---|---|---|
| popularity | under 1 s CPU | 0.4395% | 1.3438% | 4.7862% | 14.7288% | 36.8215% |
| B3 SID transition | 63 min CPU, 8 workers | 0.0244% | 0.1801% | 0.9410% | 4.2236% | - |
| item-embedding kNN, last 1 | 9 min GPU | 0.0173% | 0.5636% | 2.4638% | 8.1807% | - |
| **B4 co-click, window 20** | **4 s + 43 s CPU** | **0.8057%** | **3.7069%** | **10.9873%** | **25.3021%** | **46.8882%** |
| DiGR 0.6B, 3 epochs | 4.6 GPU-h | 0.7924% | 2.5940% | 7.5094% | 21.6003% | 49.7691% |

---

## 5. Where the failure is

Measuring at a single cut-off hides it. The gap between cut-offs shows it.

If the target is already inside the top 1,000, how often does it reach the top 10:

| method | R@10 given the target is in the top 1,000 |
|---|---|
| co-click, window 20 | 14.65% |
| DiGR 0.6B | 12.01% |
| popularity | 9.12% |
| SID depth-1 bucket + cosine | 7.45% |
| item-embedding kNN | 6.89% |

Four point six GPU-hours of a 0.6B language model moved that number from 9.12% to
12.01%. That is the whole return, and it is what sent the project looking for a
different cause.

The cause is not the encoder, the sequence length, the number of epochs or the loss.
The score is `z = V h / tau` with `v_i` a mean over the item's five shared code
vectors, and the DiGR paper describes the consequence as a feature: the embeddings
"are shared across items, and mean pooling introduces no additional composition
parameters and no item-specific information". Nothing is misimplemented. The design
buys inductive generalisation by giving up per-item detail, and that trade is cheap
when a catalogue is small.

It is not cheap at 4,783,154 items. DiGR's item side is five levels of 256, 256, 256,
256 and 731 code rows, 1,755 rows of 1,024 dimensions, which is 1,797,120 free
parameters. Levels 0 to 3 group 18,684 items each, and even level 4 groups 6,543,
which is **0.376 degrees of freedom per item**. Two items cannot be scored apart
except through a row that thousands of other items also use. The paper's own
catalogues sit at 39 to 195 items per level-0 code, 96 to 479 times finer, so its
reported accuracy is not reachable from here without changing the resolution.

The count table has the opposite shape. 25,669,281 distinct pairs, 5.37 successors
per item. Its top 10 for a query is roughly those 5.37 known next items plus
popularity padding, and that beats the model. The model sees the same pairs during
training and cannot store one of them.

So the ceiling is set by the resolution of the score, which is a setting rather than a
property, and it has already been measured once. Holding everything else fixed and
quantising the same 256-d vectors at 3 x 1,024 instead of 4 x 256 plus a collision
suffix moved depth-1 bucket lift from 46.9x to 126.8x, so the partition became 2.7x
more label-pure. That also falsified the earlier prediction in `docs/plan_v2.md` that
width could only help capacity. The next point on that curve has a known arithmetic
answer: 16,384 wide puts level 0 at 292 items per code, which is where the paper
sits.

---

## 6. The OnePiece benchmark, and why it rules out the easy excuse

OnePiece, on the same Tencent dataset, reports HR@10 0.1885 on valid and 0.1307 on
test at epochs 5 to 8.

The tempting reading is that 18.85% is a candidate-set metric and our 3.60% is a
full-catalog metric, so the two are not comparable. That reading is wrong, and it
was checked. Taking co-click's measured rank distribution and asking what HR@10 it
would report in a 100-way candidate set with uniform negatives gives **99.0%**,
because the median val target sits at catalog rank 18,855 and therefore beats
almost every randomly drawn negative. There is no candidate-set reading that turns
3.60% into 18.85%. The comparison is fair and we are 5x behind.

The gap decomposes into things the three winning teams did and we did not:

| the leaders | us |
|---|---|
| generate a user vector, ANN over per-item vectors | additive read of 1,755 shared rows |
| per-position action conditioning, FiLM and attention bias | action type deliberately never encoded |
| absolute timestamps, relative gaps, session structure, Fourier periodicity | one coarse temporal token per event |
| up to 380K negatives, InfoNCE over continuous vectors | 300K negatives, sampled softmax over codes |
| GNN over the user-item graph, 2048-d hidden, 8 layers | 0.6B decoder, no graph |
| competition-scale data, many GPU-hours | 1.72M sequences, 34M events, 4.6 GPU-h |

The first row is the one this project can act on, and it is out of scope for now by
decision. The second row is the highest-value change with published evidence
behind it. The last row may explain part of the 5x on its own and should be
confirmed before the model is blamed again.

---

## 7. The SID question, now answered

The earlier plan held that the SID might be the ceiling. It is not.

- All 4,783,154 five-code tuples are unique, so there are no collisions. The
  statistics that looked alarming (1,878,898 items grouped, up to 731 per group)
  are four-code prefix counts, which is by construction because level 5 exists to
  break them.
- Scored against the real next click, the depth-1 bucket of the last item contains
  the target 18.3156% of the time while holding 0.40% of the catalog, a 45x lift.
  Depth 2 gives 2.5167% into 345.9 items, a 348x lift. The codes carry signal at
  every depth, decaying smoothly.
- The depth-1 bucket at 19,308 candidates lands where the raw vectors' own kNN
  curve predicts it should, roughly 16% to 20% by log-linear extrapolation. **The
  quantiser loses nothing relative to the embedding.**

So the codes are a faithful compression of a representation that does not encode
intent. That is a property of Stage 1, not of the RQ-VAE, and it is not what is
stopping us: TIGER and DiGR both read the same codes, and the count table that
beats them reads item ids.

The 71% figure previously quoted for depth 1 was the embedding's own top-10 used as
ground truth. Against real targets it is 18.3156%. Retired.

---

## 8. The three ceilings, updated

**A. The task is close to noise.** Weakened but not dead. The OnePiece number shows
18.85% is reachable on this data by other people, so the task is not the ceiling at
3.6%. A remains possible only for the restricted setting we chose.

**B. The item side of the score is the ceiling.** This is now the leading
explanation and it is measured, not hypothesised: 0.376 degrees of freedom per
item, a 12.01% conditional ranking against a count table's 14.65%, and a model that
cannot reproduce the pairs it was trained on. TIGER tests the objective half of
this. The representation half needs per-item parameters, which is the deferred ANN
arm.

**C. The SID is the ceiling.** Rejected as the primary cause. Section 7 shows the
codes preserve the geometry faithfully. The geometry is weak, but that is Stage 1
and it affects every SID-based arm equally.

---

## 9. Why TIGER is still the next build

TIGER changes one thing. Same backbone, same data, same SIDs, same feature block,
same evaluator. The head changes from one full-catalog softmax over an additive
score to five per-level terms with the hidden state re-read at each level.

That matters because the additive score is the measured problem. DiGR forces every
item through 1,755 shared rows with no interaction between levels, so the score for
an item is a sum of five terms that other items share. TIGER's factorisation gives
the item a distinct conditional probability at each level, and its score can depend
on which level-0 code the user is in when choosing level 2.

What TIGER does not fix:

- It keeps the SID, so it inherits whatever the codes do not carry.
- It still shares the 731 level-5 output vectors across every prefix. What it
  gains over DiGR is the conditioning: the prefix flows into the hidden state, so
  the 731 vectors are re-weighted per bucket and each bucket gets its own ranking
  instead of one global scale. Within a bucket there are at most 731 items, so the
  deep level is close to per-item resolution. The cap that remains is the top four
  codes deciding which bucket an item lands in.
- Beam search is approximate. Its numbers are a lower bound at R@k for k above the
  beam width, and that asymmetry has to be stated whenever the table is shown.

---

## 10. Retired numbers

| retired | why | current |
|---|---|---|
| all metrics before the `evaluate()` fix | miss rank was 0, recall counted misses, nDCG infinite | `Model Training/train_digr.py:939` |
| popularity R@10 0.9644% | came from a different frequency vector, ranks 2-10 did not reproduce | 1.3438% |
| "DiGR beats co-click" | compared against window 1, the weakest co-click form | co-click window 20 wins from R@1 to R@1000 |
| "DiGR's edge is R@1, co-click only wins deeper" | true only against window 5; window 20 takes rank 1 as well | DiGR wins only at R@10000, 1.06x |
| "DiGR will reach about 62% at R@10,000" | log-linear extrapolation of the recall curve | measured 49.7691% |
| "`--label_subset clicked` is the fix" | made no epoch faster and put 81.5% of mass on untrained items | reverted to full labels |
| "5-code SIDs have 1.88M collisions" | those are four-code prefix groups | zero collisions over five codes |
| DiGR val-loss-best checkpoint | epoch 2 has the best val loss, epoch 3 ranks best | use `final`, not `best` |
| "depth-1 SID recall is 71.158%" | the "target" was the embedding's own top-10, so it graded the geometry against itself | 18.3156% against real targets |
| "the task caps out around 4% R@10" | inferred from our own methods converging | OnePiece reaches 18.85% HR@10 on the same data |

---

## 11. Open decisions

1. The bar. **Resolved** (m03136): the window sweep closed at 20, with R@10
   3.6011% to 3.6794% to 3.7069% in shrinking steps. The bar TIGER has to clear is
   R@10 3.7069% and R@100 10.9873%.
2. Fifth-level protocol. **Decided** (m03114): exact item identity for all arms;
   prefix hits are a footnote only. 39.3% of items share a four-code prefix and
   level 5 is an arbitrary rank inside the group.
3. Whether the heuristic arm should be SID-based as well. It currently uses item
   ids, which makes it a stronger bar than a code-based heuristic would be. That is
   deliberate, but it should be stated in the write-up.
4. When the ANN arm enters. It is the design the winners used and the only one
   with per-item resolution, so it is the natural follow-up, but it is not part of
   the three-arm table.
5. The acceptance bar for TIGER. Proposed: beat co-click at R@10 and R@100. Beating
   DiGR alone is not a result, because DiGR already loses to a count table.

---

## 12. What would change the plan

- TIGER lands at or below co-click at R@10. Then the SID-based family is beaten by
  a count table, the answer to the project's question is "none of them", and the
  ANN arm with per-item vectors becomes the main line rather than a follow-up.
- TIGER clearly beats co-click at R@10 and R@100. Then the objective was the
  ceiling and the write-up is "the head was the problem", with the factorisation as
  the contribution.
- TIGER ties DiGR exactly. Then the objective is not the active variable, the
  1,755-row item side is, and no amount of head design on shared codes will help.
- The data scale check shows the competition used 100x our events. Then part of the
  5x is dose, no head change closes it, and the report has to say so explicitly.

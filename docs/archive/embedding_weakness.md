# Why the Stage 1 item embeddings are behaviourally weak

The SID is built from an embedding that was trained with a behavioural objective,
so the reasonable expectation is that it carries behaviour. It mostly does not.
This file explains the mechanism, using `SID Generation/train_item_embedding_cli.py`
as the evidence and the OnePiece paper (arXiv:2512.07424) as the reference that
shows what a working version of the same idea looks like.

---

## 1. What the pipeline actually is

Stage 1 is `train_item_embedding_cli.py`: an ItemDNN plus an HSTU sequence
encoder, trained end to end with InfoNCE and LogQ correction.

Item side. The ItemDNN maps raw item features to a 256-d vector, L2-normalised.
Its inputs are:

- `mm_emb`, a 1024-d multimodal (MLLM) content embedding, plus a `has_mm_emb`
  flag. Projected by `sem_proj` (1025 -> 512 -> LayerNorm -> SiLU).
- categorical columns (up to 13, default `DEFAULT_CAT_COLS`), each embedded at
  dim 32 and concatenated.

The MLP then goes 512 + (13 x 32) = 928 -> 256 -> 256, and the output is
`F.normalize(out, p=2, dim=-1)`. This output is what `export_catalog_embeddings`
writes to `item_embeddings.parquet`, and that file is what the RQ-VAE quantises
into SIDs.

User side. The HSTU (`HSTUSequenceEncoder`) is 3 blocks, hidden 256, 4 heads,
max length 100. Its input is `input_layer_norm(item_reps + action_emb)`, the item
embedding plus a 4-token action embedding, and its output is the last position,
L2-normalised.

Loss. `compute_loss_and_logits` computes the user vector, the target item vector,
and InfoNCE over the batch:

```
logits = (user @ target.T) / temperature - log_q_weight * log_q(target)
loss = cross_entropy(logits, arange(B))
```

Defaults: temperature 0.05, batch 1024, in-batch negatives only (no cross-GPU
negatives unless the flag is set), LogQ weight 1.0, AdamW lr 1e-3, cosine schedule.

So both the item side and the user side receive gradient from the behavioural
objective. The question is why the item side ends up content-dominated anyway.

---

## 2. The measured symptom

From `docs/sid_results.md` and the kNN probe:

- effective rank of the 256-d item embeddings: **24.8**.
- pooled-history kNN over the embeddings (last 1 / 3 / 5 items, mean-pooled,
  re-normalised) scores **below popularity** at every cut-off: R@10 0.5636%
  against popularity's 1.3438%.
- the depth-1 SID bucket (which is just the RQ-VAE's coarsest code) holds the
  real next click 18.3156% of the time, so the codes are a faithful compression
  of the embedding. The embedding itself is what is weak.

Effective rank 24.8 out of 256 is a dimensional-collapse signature. The item
vectors all live in a small subspace, which means they encode a few dominant
directions (content, a couple of popularity-ish axes) and almost nothing else.

---

## 3. The core finding: the behaviour went into the user encoder, not the item

The InfoNCE gradient flows to both towers, but not symmetrically.

The user encoder (HSTU) is re-applied to every sequence and gets a direct,
per-example behavioural gradient. It is where the model invests its capacity to
predict the next item, and it is small enough that this is easy.

The item encoder (ItemDNN) is content-initialised and shared. A given item
appears as a target in a small fraction of batches, so its vector receives a
weak, averaged gradient, and the strongest component of that gradient is
popularity, which LogQ then subtracts away. What remains is a thin residual, and
it is not enough to move the vector far from its content prior.

The result is exactly what we measure: the behaviour is real and is learned, but
it lives in the HSTU user vector. The item vector stays a content code with low
rank.

Two downstream facts follow from this.

The kNN probe (`Model Training/embedding_knn_nextclick.py`) pools history item
embeddings and retrieves by item-vector similarity. It never runs the trained HSTU
user encoder. So it measures only the item side, which is the weak side. It is a
probe of the wrong tower, and its "loses to popularity" verdict does not mean the
trained model has no behaviour. It means the behaviour is not in the item vectors.

The SID is built from the item vectors. DiGR and TIGER then discard the HSTU
entirely and re-learn history-to-code prediction from scratch in a new backbone.
So for the SID-based methods, the item vector is the only conduit from Stage 1,
and it is the weak conduit. That is the real reason the SID is behaviourally
weak: it quantises the item side of a model that stored its behaviour on the user
side.

---

## 4. Why the item side ended up weak, ranked

### 4.1 The Stage 1 model is roughly 30x too small

3 HSTU blocks at hidden 256 with 4 heads is about 2M parameters. OnePiece's
tokeniser model is a 40-layer HSTU+MoE at 57M parameters. The item embedding
inherits its behavioural signal only through the user vector; a 3-block user
encoder cannot produce a good one, so the gradient it sends back to the ItemDNN
is noisy and small, and the item vector barely moves.

OnePiece's own scaling table makes the point: on their data, Hitrate goes 0.2770
at 8 layers to 0.3219 at 40 layers, a 16% relative gain from depth alone. Depth
is not decorative; it is the mechanism that makes the user vector sharp enough to
train a sharp item vector.

### 4.2 Tied dual-tower plus L2 norm plus low temperature invites collapse

The same ItemDNN produces both the history-item vectors and the target vector
(`get_item_embeddings` is called for both), both are L2-normalised, and the
temperature is 0.05. Tied weights plus normalisation plus a low temperature is
the standard recipe for the dimensional collapse that shows up as effective rank
24.8.

OnePiece avoids the tied-tower collapse structurally with a dual-path ItemDNN
(section 3.2 of the paper): a ReLU path for the Transformer input and an identity
path for the FAISS index. The item vector that does retrieval is the identity
path, which stays close to the raw fused features and cannot collapse the way a
free nonlinear projection can.

### 4.3 In-batch negatives are too few

The only negatives are the other targets in the batch, 1024 of them against a
4.78M catalog. The contrastive signal is therefore coarse: the model only has to
lift the target above ~1023 batchmates, most of which are unrelated. Fine-grained
behaviour never has to be learned, so the item vector settles into a coarse
content-plus-residual shape.

### 4.4 LogQ is configured at full weight but is inert

LogQ is meant to subtract popularity from the logits. For a weak model, popularity
is the easiest and largest behavioural signal, so the intent of a full LogQ weight
was to force the model past it. **In this run it does nothing at all**, because
`main()` never populates a real sampling distribution and `log_q` stays the zero
tensor from `__init__` (section 8.5). So the model is trained with no popularity
correction, and the exported item vectors keep whatever popularity component the
content and category frequencies put into them. This is a wiring defect, not a
trade-off.

### 4.5 Content dominates the input

`mm_emb` is 1024 dimensions and the ItemDNN's first layer is a dedicated
1025 -> 512 projection onto it. Content is the largest input by an order of
magnitude, and every mechanism above fails to displace it. The behavioural
gradient would have to re-organise the entire content manifold, and it does not.

---

## 5. What OnePiece does differently

OnePiece (arXiv:2512.07424) is the Tencent Algorithm Competition case study
(10th preliminary, 9th semi-final, 13th final), and it is the closest reference
implementation to this exact pipeline. The differences that matter:

| axis | Stage 1 here | OnePiece |
|---|---|---|
| item encoder | single nonlinear ItemDNN, tied towers | dual-path (ReLU for the transformer, identity for the FAISS index) |
| sequence model | 3 blocks, 256 dim, ~2M params | 40 layers, 128 dim, 8 heads, MoE, 57M params |
| optimizer / lr | AdamW, 1e-3 | Muon, 0.08 |
| temperature | 0.05 | 0.02 |
| negatives | in-batch 1024 | in-batch 512, LogQ, dual-tower scoring |
| SID | 5 codes, 4x256 + 731 | 2 codes, 2x16384, residual K-means + greedy re-assignment |
| inference | (DiGR/TIGER full-catalog) | SID beam (20 -> 384 pairs) then InfoNCE re-rank then filters |
| result | R@10 2.59% (DiGR), 3.71% (co-click) | HR@10 18.85% valid, 13.07% test |

Three of these rows are the actionable ones. The model size (30x) is why their
user vector is sharp enough to train a sharp item vector. The dual-path ItemDNN
is why their item side does not collapse. The two-stage inference (beam then
InfoNCE re-rank) is why their item side never has to be behaviourally complete on
its own: the behavioural precision lives in the user query, and the item side only
needs to be a decent content index.

Note the metric caveat that is now permanent: their HR@10 is over a beam-search
candidate set of a few hundred items, not the full catalog, so 18.85% is not
directly comparable to our full-catalog R@10. The comparison stands as "a working
system on this data", not as "same denominator".

---

## 6. What this means for the SID and for the plan

The reconciliation from `docs/tasks.md` Phase 3b survives and sharpens:

Content is a valid base because it is universal and honest, and OnePiece shows a
content-ish item side can win when the behaviour lives in the user query. The
defect is not "content is wrong", it is "our item side is the only thing the SID
sees, and our item side is weak because the behaviour went to the user encoder".

There are two ways to get behaviour into the SID, and they are complementary.

1. Fix Stage 1 and re-export the item embedding. Bigger HSTU, untie or dual-path
   the ItemDNN, raise the learning rate (Muon-style), more negatives (cross-GPU
   or a memory bank), and consider a gentler LogQ weight. This attacks 4.1 to
   4.4 directly and produces item vectors that carry behaviour because the user
   vector was strong enough to push it there.

2. Enrich the item embedding before RQ-VAE. Concatenate the content vector with a
   behavioural vector (truncated SVD of the co-click matrix, zero-filled for cold
   items) and quantise the 512-d result. This is task 3b.2, and it injects
   behaviour directly rather than waiting for Stage 1 to learn it. It is now the
   first embedding experiment, because the probe showed the Stage 1 half of the
   premise (a strong user vector) does not hold.

There is a third, already-deferred option that this analysis puts back on the
table as the honest conclusion: the behaviour already exists in the trained HSTU
user vector, and the pipeline discards it. Using that vector directly with an ANN
index is the two-tower arm, which is the design OnePiece actually uses for its
final scoring. It is out of scope for the three-arm SID table by decision, but
this file makes the reason visible: the SID arms are being asked to reconstruct,
inside a new backbone, a signal that Stage 1 already computed and threw away.

---

## 7. Status of the probe: answered, bottom band

`Model Training/probe_hstu.py` tests section 3 directly: it rebuilds the ItemDNN
and HSTU from `stage1_rec_model.pt` and scores the user vector against the full
catalog on the exact val rows. It has run, and the answer is the worst of the
three bands in `docs/tasks.md` 3c.7.

| method | R@1 | R@10 | R@100 | R@1000 | nDCG@10 |
|---|---|---|---|---|---|
| HSTU tower, window 20 | 0.0458% | 0.4354% | 2.9063% | 13.5122% | 0.2008% |
| item embedding, last 1 clicked | 0.0173% | 0.5636% | 2.4638% | 8.1807% | - |
| popularity | 0.4395% | 1.3438% | 4.7862% | 14.7288% | 0.8090% |

One bug was found on the first run and fixed. `HSTUSequenceEncoder` reads
`x[:, lengths - 1]` as the query, training left-aligns the sequence and the probe
right-aligned it, so padded rows read a padding position. The fix was
`to_chronological`. The rerun used that fix and the file on the cluster was
confirmed byte-identical to `Model Training/probe_hstu.py`
(`a4d53ca4db811d22387e595647076911`). R@1 was unchanged at 0.0458%, R@10 moved
from 0.4283% to 0.4354% (+1.66% relative), R@100 from 2.8798% to 2.9063%, R@1000
from 13.3931% to 13.5122%. The bug cost about 1.7% relative at R@10 and nothing at
R@1, because a padding position still aggregates the valid prefix through
attention, and only rows with fewer than 20 events were affected.

Actions are approximated to a constant click action, because DiGR tokens carry
codes and a temporal token but not action types. That biases the query down, so it
cannot manufacture a good result.

So section 3 is settled in the negative, not the way it was written. The behaviour
is not in the user encoder either. Both towers of Stage 1 score below a frequency
count, on the same content geometry, and the conclusion is the fourth option: Stage
1 did not learn usable behaviour anywhere. A rebuild of the item side alone would
be rebuilding the wrong half.

---

## 8. Why behaviour never reached the item vector

Six mechanisms. The first is structural, the rest are severity.

**8.1 The export path has no access to behaviour.** `export_catalog_embeddings`
calls `get_item_embeddings(i)`, which is `ItemDNN(sem_i, has_mm_i, cats_i)`. No
history, no user, no action. Everything the HSTU learned lives in `seq_encoder`
and `action_emb`, and neither is touched when the item vectors are written. The
exported vectors are a pure function of content, by construction.

**8.2 There are no per-item parameters anywhere.** `ItemDNN` is content to vector:
one shared MLP, 13 category embedding tables, and a projection of the 1024-d
multimodal vector. No item index ever enters it. Two items with identical content
get identical vectors, so the item side cannot store "item 123 follows item 456".

**8.3 The only behavioural supervision is an averaged gradient.** InfoNCE does
reach `ItemDNN`, but each item is a negative for most users and a positive for a
few. The gradient on item `j` pulls it toward the average of the user vectors of
its clickers. Since user vectors are dominated by their last clicked item's
content, that average is a content cluster centroid. That is exactly the measured
result: the tower's R@10 is 0.43%, sitting on the content geometry.

**8.4 In-batch negatives, B = 1024.** The model was only ever asked to rank 1024
items against each other. Catalog R@10 over 4.78M was never the objective, so
in-batch recall looked healthy while catalog ranking collapsed. First place in the
competition used large negative banks; third place scaled negatives to 380K. This
run used 1024.

**8.5 `log_q` is silently inert.** `SequentialRecModel.__init__` defaults
`log_q = None` to `torch.zeros(...)`. `main()` passes `use_log_q` and
`log_q_weight` but never a real sampling distribution, so the popularity
correction contributes exactly zero. An intended component is switched off and
nothing complains.

**8.6 Action conditioning is additive and weak.**
`hist_input = input_layer_norm(item_reps + act_reps)` mixes an unnormalised action
vector into a unit-norm item vector, over a 4-token vocabulary. All three leading
write-ups use per-position gated fusion, FiLM, or attention biasing. This is the
weakest form of the thing they all treat as central.

Summary: the model was not undertrained. There was nowhere for behaviour to be
stored.

---

## 9. Is this the bottleneck? Not first

The evidence makes the embedding a suspect, not a proven cause.

| method | R@1 | R@10 | what carries the signal |
| --- | --- | --- | --- |
| HSTU tower, window 20 | 0.046% | 0.44% | content geometry |
| content kNN, pooled history | 0.017% | 0.56% | content geometry |
| popularity | 0.44% | 1.34% | frequency |
| DiGR on our content SID | 0.79% | 2.59% | learned history to code |
| co-click, window 20 | 0.81% | 3.71% | counts |

DiGR recovers **4.6x more R@10 than the embedding it was built from**. A
generative retriever does not need a behaviour-rich embedding. It needs an
injective, well-spread vocabulary and enough label signal to fit the transition
distribution. The SID is injective and near maximum entropy, so it qualifies.

What an embedding rebuild also cannot fix on the frozen SID: 27.67% of items share a
4-code prefix, so exact-identity R@1 is limited by how the codes partition the space,
not only by how good the vectors are. Width does fix that. At 3 x 1,024 the depth-3
collision rate is 29.05% but the largest depth-3 bucket is 372 items, so a fourth
1,024-wide level is arithmetically guaranteed to be injective, and at 2 x 16,384
depth 2 carries at most a 1.78% uniform-bound collision share.

But the embedding is not off the hook, and the reason is sharper than "the vectors
are weak". The DiGR head reads an item through shared rows: `E_l[c_il]` is one of
256 rows at each of the first four levels, and 18,684 items read the same level 0
row on average. Every gradient step on that row averages over all of those items.
With content codes, the averaged-over set is content-similar but behaviourally
arbitrary, so the shared row learns a blur. With behaviour-aware codes, the
averaged-over set is behaviourally alike and the same row becomes meaningful. The
head is weak because the codes are content codes. That is the mechanism, and it
says a behavioural code is not a nice-to-have for the SID route, it is one fix for
the head's central weakness.

Width attacks the same mechanism from the other side, and that is measured now rather
than argued. Quantising the same 256-d vectors at 3 x 1,024 instead of 4 x 256 plus a
collision suffix cuts the averaged-over set from 18,684 items to 4,671 and moves
depth-1 bucket lift from 46.9x to 126.8x. So the mechanism above is real and the
number of items sharing a row is the dial. Behavioural codes make each shared set
homogeneous; wider codes make each shared set small. Both cut the blur, and width
needs no new data, no new negatives and no retraining of either tower, which makes it
the cheaper first move. At 16,384 wide the shared set is 292 items, which is where
the DiGR paper's own catalogues sit, so that is the point to aim at rather than
1,024, and two codes of that width are enough.

What it does not say is that Stage 1 is the right instrument. The cheapest way to
test the mechanism is to quantise a behaviour-derived vector, which needs no
retraining of either tower and no new negatives, because the co-click matrix
already exists and its own recall is already measured at 3.7069% R@10. That is D3,
and it belongs before any Stage 1 retrain, but after the width sweep, which costs a
GPU hour instead of a preprocessing pass.

---

## 10. If the embedding is rebuilt, what to inject

Always the same shape: item vector = content prior + behavioural evidence, with
behaviour only where it exists (1.58M ever-clicked items, 67% cold). A pure
per-item id table is not available, because 3.2M rows would never see a positive.

- **D6, centroid distillation. Dropped.** Take the trained HSTU, compute for
  every item the mean of its clickers' user vectors, then blend
  `normalize((1 - a) * e_i + a * centroid_i)`. This was going to be the cheap gate.
  The probe closed it instead: the user vectors being averaged score 0.4354% R@10,
  below a frequency count, so averaging them cannot produce a strong item vector.
  The gate it was meant to test is already answered, and running it would spend an
  hour confirming an arithmetic impossibility.
- **D3, behaviour-derived input. Next, and cheap.** Item2vec or truncated SVD over
  the co-occurrence matrix, then `e_i = normalize(MLP(content_i) + W b_i)`, with
  `b_i = 0` for the 67% cold items, then quantise the combined space. Second place
  used exactly this. It tests the mechanism above directly, and it needs no Stage 1
  retraining, so the frozen content SID stays intact and the three-arm comparison
  is unaffected until that arm is switched on deliberately.
- **D1, per-item behavioural residual.** `e_i = normalize(MLP(content_i) + r_i)`,
with `r_i` a free 256-d row initialised to zero and updated only for interacted
items, so cold items keep the pure content prior. About 2.4 GB in bf16. This is
what gives the item side real resolution.
- **D2, catalogue-scale objective.** Replace the B x B diagonal with sampled
softmax over 100k to 380k catalog negatives, and populate `log_q` with the true
sampling distribution. The same trick as `--sampled_softmax` in DiGR. Two dozen
lines, and the highest expected value per line in this list.
- **D5, real action gating.** Per-position FiLM or gated fusion instead of
addition, plus true action types.
- **D3, behaviour-derived input.** Item2vec or truncated SVD over the
co-occurrence matrix, then `e_i = normalize(MLP(content_i) + W b_i)`, with
`b_i = 0` for cold items. Second place used exactly this.
- **D4, two vectors.** A content vector for coverage and a behavioural vector for
warm items, with the SID taken from the behavioural space and a content fallback.
More faithful to production, twice the pipeline.

Constraints that survive every option: the SID exists for all 4,783,154 items and
stays injective, cold items keep a content prior, the headline metric is exact item
identity, and the vocabulary stays inside the 1,966 LLM tokens.

---

## 11. Order of work

The goal is "which SID retrieval method is best", so at most one thing may move
per comparison. Two rules bound this file (m05965): reranking is out, because a
second stage over a shortlist measures the second stage, and the item
representation stays a function of content, because content is the only input a
cold item has.

1. TIGER on the frozen content SID. It changes the head and nothing else, so it
   isolates the head. Its bar is co-click window 20, R@10 3.7069%.
2. A better content tower stays open, but only in the content-only direction. A
   weak item vector produces a weak SID however it is cut, so the rebuild is worth
   doing on its own terms, as long as the vector remains a function of content at
   serving time.

The behaviour-SID arm (D3) is closed under the second rule. A co-occurrence vector
is undefined for an item that arrived today, so it cannot be the code space. A
Stage 1 retrain that injects behaviour as an input is the same move and is closed
with it. D6 is dropped as before, since the towers it would average score 0.4354%
R@10, below a frequency count.

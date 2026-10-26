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

### 4.4 LogQ at full weight removes the one signal a small model can learn

LogQ subtracts popularity from the logits. For a weak model, popularity is the
easiest and largest behavioural signal. Removing it entirely leaves a residual
that the 2M-parameter model cannot fit, so the item side falls back on content,
which is the only structure left in its inputs.

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
   behavioural vector (truncated SVD of the co-click matrix, zero-filled for
   cold items) and quantise the 512-d result. This is the Phase 3b enriched arm,
   and it injects behaviour directly rather than waiting for Stage 1 to learn it.

There is a third, already-deferred option that this analysis puts back on the
table as the honest conclusion: the behaviour already exists in the trained HSTU
user vector, and the pipeline discards it. Using that vector directly with an ANN
index is the two-tower arm, which is the design OnePiece actually uses for its
final scoring. It is out of scope for the three-arm SID table by decision, but
this file makes the reason visible: the SID arms are being asked to reconstruct,
inside a new backbone, a signal that Stage 1 already computed and threw away.

---

## 7. Concrete next checks, cheapest first

- Confirm the actual Stage 1 hyperparameters. The defaults above (batch 1024,
  lr 1e-3, 3 blocks, temperature 0.05, LogQ 1.0) are what the code ships with; the
  real run's values are saved in `output_dir/stage1_rec_model.pt` under `args`.
- Score the trained HSTU user vector against the item embeddings, not the pooled
  history. This is a ~10-line variant of the kNN probe and it is the one test
  that would confirm section 3 directly: if the HSTU query beats popularity while
  the pooled-history query does not, the behaviour is proven to be in the user
  tower.
- If that confirms it, the cheapest behavioural SID is the Phase 3b enrichment
  (co-click SVD), because it does not require retraining Stage 1.
- Fixing Stage 1 (bigger model, dual-path, more negatives) is the higher-ceiling
  but more expensive option, and it belongs after TIGER is measured, per the
  order in `docs/tasks.md`.

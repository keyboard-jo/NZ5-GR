# Embedding and SID quality

What the two EDA notebooks measured, and what it implies. Sources:
`SID Generation/embeddings_quality_eda.ipynb` (item embeddings) and
`SID Generation/sid_quality_eda.ipynb` (the codes). Embedding numbers are on a
50,000-item sample (1.05% of the catalog) with seed 42; SID structure numbers are
over all 4,783,154 items.

Read this as: the representation is healthy, its capacity is low, and nothing here
has yet been tested against a click.

---

## 1. The embedding is numerically clean

| check | value | reading |
|---|---|---|
| norm min / median / max | 1.00000 / 1.00000 / 1.00000 | `ItemDNN` ends in `F.normalize(p=2)`, so inner product is cosine and there is no magnitude channel carrying popularity |
| non-finite entries | 0 | |
| dead dimensions (std < 1e-3) | 0 / 256 | every dimension is used |
| duplicate item_ids | 0 | |
| cos(item, global mean direction) | mean 0.291, p90 0.538, max 0.847 | moderate cone, not a collapsed blob |

All exports are unit norm, which is why every retrieval measurement below can use
`faiss.IndexFlatIP` and be exactly cosine.

---

## 2. Dimensional collapse is real and large

| measure | value |
|---|---|
| effective rank (exp of entropy of the spectrum) | **24.8** of 256 |
| participation ratio `(sum L)^2 / sum L^2` | **16.8** of 256 |
| dimensions for 90% / 99% of variance | 23 / 51 |
| top-10 explained variance ratios | 0.127, 0.113, 0.082, 0.080, 0.066, 0.046, 0.043, 0.039, 0.034, 0.031 (66% combined) |

A 256-d contrastive space carrying about 17 to 25 dimensions of real variance. That
is the number the model trained at, and it is the ceiling that any downstream code
inherits.

---

## 3. But the space is not degenerate, it is dense

| measure | value | reading |
|---|---|---|
| random-pair cosine | mean +0.0843, std 0.2372, p99.9 +0.823 | mild anisotropy |
| uniformity | -3.1891 | |
| pairs with cosine > 0.5 / 0.8 / 0.95 | 4.772% / 0.151% / 0.003% | bulk not collapsed |
| cosine to 1st neighbour | mean **0.9290**, p10 0.8721, min 0.7089 | tight local clusters |
| cosine to neighbours 1..20 | mean 0.8645 | |
| retrieval margin (1-NN minus random) | **+0.8450** | strong signal |
| items whose 1-NN cosine < 0.5 | 0.00% | no isolated items |

The important consequence: the useful signal for ranking is **thin**. Going from the
first neighbour to the twentieth costs only 0.065 of cosine (0.9290 to 0.8645),
while the offset from a random item is 0.845. Any quantizer has to preserve a 0.065
margin sitting on top of a 0.845 base.

---

## 4. Hubs and near-duplicates are not a problem

| measure | value | reading |
|---|---|---|
| never-retrieved items (10,000 queries x 20 slots) | 6.18% | |
| Gini of neighbour counts | **-0.3737** | negative means *less* concentrated than uniform |
| top 1% of items hold | 3.46% of slots | uniform would be ~1%, so mild |
| mean cosine to the top hub | +0.0370 | below the random-pair mean of 0.0843, so hubs are not universally close, they are just frequently nearest |
| items with a > 0.99 neighbour | 0.59% (118 / 20,000) | |
| duplicate graph | 128 edges, largest component 7 items | no giant collapse |

---

## 5. The codebooks are healthy

Four codebooks of 256 over the full catalog:

| stage | utilization | dead codes | entropy | perplexity | largest code |
|---|---|---|---|---|---|
| 1 | 1.0000 | 0 | 7.924 bits | 242.9 | 39,883 (0.83%) |
| 2 | 1.0000 | 0 | 7.874 bits | 234.6 | 43,127 (0.90%) |
| 3 | 1.0000 | 0 | 7.934 bits | 244.5 | 30,121 (0.63%) |
| 4 | 1.0000 | 0 | 7.980 bits | 252.5 | 25,603 (0.54%) |

Maximum entropy is 8 bits for 256 codes. Every stage is within 1.5% of it, and the
largest level-0 code holds 0.83% of the catalog against 0.39% for uniform. There is
no codebook collapse and no unused capacity.

---

## 6. The code's bucket structure

| depth | groups | mean items per group | unique codes |
|---|---|---|---|
| 1 | 256 | 18,684 | 256 |
| 2 | 57,122 | 83.7 | 57,122 |
| 3 | 1,506,041 | 3.18 | 1,506,041 |
| 4 | 3,459,854 | 1.38 | 3,459,854 |

27.67% of items share their four-code prefix with at least one other item, which is
what the unbounded level-5 suffix exists to break. Level 3 is where resolution
explodes, from 57 thousand groups to 1.5 million.

---

## 7. The code follows the geometry, level by level

Cosine between two items sharing a prefix, 5,000 sampled pairs per depth:

| depth | mean | median | std | random baseline |
|---|---|---|---|---|
| 1 | 0.6705 | 0.6837 | 0.143 | 0.084 |
| 2 | 0.8617 | 0.8796 | 0.081 | 0.084 |
| 3 | **0.9287** | 0.9383 | 0.043 | 0.084 |
| 4 | 0.9501 | 0.9586 | 0.036 | 0.084 |

Compare with the 1-NN mean of 0.9290. **A shared depth-3 prefix is as similar as a
nearest neighbour.** Depth 4 is tighter than the average 1-NN, which means level 4
is already resolving below the neighbourhood scale.

Fraction of a query's 10 nearest neighbours that share its prefix (50,000 queries,
500,000 pairs):

| depth | share of the 10-NN |
|---|---|
| 1 | 71.81% |
| 2 | 17.21% |
| 3 | 2.58% |
| 4 | 0.33% |

And measured against the embedding's own top-10 as ground truth, on the 50,000-item
sample:

| depth | recall@10 | mean candidates in bucket | zero-candidate queries |
|---|---|---|---|
| 1 | 71.158% | 213.8 | 0.0% |
| 2 | 16.828% | 4.13 | 21.7% |
| 3 | 2.626% | 0.31 | 84.0% |
| 4 | 0.346% | 0.04 | 97.2% |

In the full catalog those bucket sizes are about 95.7x larger, so a depth-1 bucket
holds roughly 18,684 items (0.39% of the catalog). Within the sample the bucket is
213.8 items, or 0.43%, and it captures 71.8% of the true nearest neighbours, which
is a **166x lift** over the 0.428% random baseline in the table above.

---

## 8. The arithmetic that frames everything

For N points spread evenly on a sphere of dimension d, the expected nearest-neighbour
cosine is roughly `1 - N^(-2/(d-1))/2`. With N = 4,783,154:

| d | expected 1-NN cosine |
|---|---|
| 16.8 (participation ratio) | 0.9286 |
| 24.8 (effective rank) | 0.8627 |
| measured | **0.9290** |

The measured value matches the participation-ratio prediction to three decimals,
and is *tighter* than a uniform packing at the effective-rank dimensionality would
be. So the crowding is not a pathology on top of the collapse, the crowding *is* the
collapse: the space is roughly as packed as its dimensionality forces. There is no
spare structure to recover by working on the quantizer, because the quantizer can
only place points in the space the item tower produced.

---

## 9. What none of this says

Every number above is judged against the embedding's own geometry. In
`sid_quality_eda.ipynb` cell 24 the ground truth `true_neighbors` comes from
`index.search` over the same embeddings, so "recall@10 = 71.158%" means "a depth-1
bucket contains 71% of the embedding's top-10", not "it finds the item the user
clicks next".

So the one thing the notebooks do not establish is the premise they are usually
cited for: that this behavioural two-tower geometry predicts next-click. The item
tower sees only `mm_emb` plus 13 categorical features, so as an *item vector* it is
content-derived; behaviour enters only through the InfoNCE objective against the
HSTU sequence tower. Whether that leaves pairwise item similarity predictive of the
next click is unmeasured, and it is the question that decides whether the SID is
worth defending.

Concretely missing, as of when this file was written:

- anything about behaviour at all, since notebook sections 7 to 10 (mm_emb probe,
  adjacent-history pairs, item_id leakage, clustering) were skipped

### 9a. Measured since (2026-10-25)

`Model Training/embedding_knn_nextclick.py`, log `logs/digr_emb_nextclick_715249.out`,
98,304 val rows, ground truth is the real next click rather than the embedding's own
neighbours:

| probe | result against the real target |
|---|---|
| embedding kNN, mean of last 1 clicked | R@1 0.0173%  R@10 0.5636%  R@100 2.4638%  R@1000 8.1807% |
| embedding kNN, mean of last 5 clicked | R@1 0.0539%  R@10 0.3174%  R@100 2.0742%  R@1000 9.6965% |
| depth-1 bucket | 18.3156% candidate recall, 19,308 mean candidates |
| depth-2 bucket | 2.5167% candidate recall, 345.9 mean candidates |

So the 71.158% in section 7 becomes **18.3156%** once the ground truth is what the
user clicks next, and 16.828% becomes **2.5167%**. Both stay far above a random
bucket (45x and 348x), so the codes are a faithful compression of the geometry. The
geometry is the problem: the raw embedding loses to a popularity count at every
cut-off, and a depth-1 bucket at 19,308 candidates lands where the embedding's own
kNN curve says it should, so nothing is being lost in the quantiser.

Full comparison table in `docs/results.md` section 8.

---

## 10. Caveats and numbers not to cite

- The sample is 50,000 items out of 4,783,154. Every candidate count and bucket size
  in sections 3, 4 and 7 is about 95.7x smaller than the real catalog.
- `Split-half KS distance: 1.0000` in the anisotropy cell is not a plausible value
  for two halves of one sample and looks like a bug in that cell. Do not cite it.
  Every other statistic in that cell is fine and reproduces.
- The hierarchy example (most frequent prefixes, feature 100) shows the parent
  `(178, 77)` at 89.2% value `6.0` and its child `(178, 77, 157)` at 87.5% value
  `6.0`. The child does not purify the feature, and a sibling branch `(188, 224)` is
  100% value `6.0`. Depth-3 and deeper codes are therefore not clean partitions of
  any single categorical feature, which is consistent with them encoding fine
  geometry rather than category.
- Effective rank and participation ratio disagree by a factor of 1.5 (24.8 against
  16.8). Both are reported above; do not quote one without the other.

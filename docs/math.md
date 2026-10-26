# The math

Every formula here is the one the committed code actually runs, not a plausible
reconstruction. File and line references point at the implementation. Numbers in
`code` are values measured on the current data, kept so a reader can check the
arithmetic without rerunning anything.

---

## 0. Symbols

| symbol | meaning | value on this data |
|---|---|---|
| `K = |I|` | catalog size, the training denominator | 4,783,154 |
| `L` | SID levels per item | 5 |
| `d` | backbone hidden size | 1,024 |
| `V_l` | codebook size at level `l` | 256, 256, 256, 256, 731 |
| `E_l` | embedding table for level `l`, `(V_l, d)` | loaded from `catalog_sid_codes` |
| `c_{i,l}` | code of item `i` at level `l` | 0-based |
| `h` | query vector: last non-pad hidden state | `(d,)` |
| `tau` | temperature | 0.07 |
| `W` | all five code tables stacked | `1755 x d` |

Total code rows: `2*256 + 731 = 1755`. Item vectors live in a `256`-dim space
and are built from 1,755 shared vectors.

---

## 1. What is measured

`evaluate()` (`Model Training/train_digr.py:905`) scores every val row against
the full catalog, then turns each row into one integer `rank` (1-based) of the
unique target.

```
recall@k   = mean( rank <= k )                   # one target per row, so this
                                                 # is also the hit rate
ndcg@k     = mean( rank <= k ? 1/log2(rank+1) : 0 )   # IDCG = 1, single relevant
mrr@10     = mean( rank <= 10 ? 1/rank : 0 )
```

Two conventions matter and both were wrong at some point in this project.

A miss is assigned `rank = kmax + 1`, not 0
(`Model Training/train_digr.py:939`). With `rank = 0` a miss passes every
`rank <= k` test, so recall was counting misses as hits, and
`1/log2(0+1) = 1/0` made nDCG infinite. Any number produced before that fix is
void.

MRR is pinned to 10 by `MRR_K` (`Model Training/train_digr.py:212`) even when
the cut-off list is widened to 1,000 or 10,000. Without the pin, adding
`recall@1000` silently redefines MRR as MRR@1000, which inflates it. The same
pin exists in the baseline scorer (`Model Training/baselines_digr.py`, `MRR_K`,
`EVAL_KS`).

Useful reference points:

```
uniform R@k       = k / 4,783,154          # R@1 = 2.09e-7
uniform CE        = ln(4,783,154) = 15.381
```

---

## 2. Popularity

```python
count = bincount(train_targets, minlength=K)     # once, over 1,719,430 rows
order = argsort(-count, kind="stable")           # ties broken by catalog index
```

There is no history term at all. The score of item `i` is how many times `i`
appears as a target in the *train* split, and the same fixed list is served to
every query.

Mean count per catalog item is `1,719,430 / 4,783,154 = 0.359`. Only 488,945
items have a non-zero count, so a target item is seen `1,719,430 / 488,945 =
3.52` times on average. That is a very thin frequency estimate, and it is the
reason the popularity baseline is beatable at all.

---

## 3. B3: first-order SID transition

### Tables

`build_transition_counts()` (`Model Training/baselines_digr.py:383`) builds one
`V_l x V_l` count table per level:

```
A_l[a, b] = #{ (t, t+1) : c_{x(t),l} = a, c_{x(t+1),l} = b }
          + #{ rows : c_{last,l} = a, c_{target,l} = b }
```

The first sum runs over consecutive events inside a row, the second splices the
held-out target onto the last event, because that is the transition being
predicted. Both levels of the pair are compared *within the same level*; there is
no cross-level interaction.

Check: train has 34,019,640 events in 1,719,430 rows.
`34,019,640 - 1,719,430 = 32,300,210` consecutive pairs, plus `1,719,430`
anchor pairs, is exactly `34,019,640`. Every level table sums to that, so
`A_l.sum() == 34,019,640` for all `l`. Density: level 0 uses 98.6% of its 65,536
cells (about 526 transitions per used cell), level 4 uses 3.0% of its 534,361
cells.

### Score

`aggregate_transition_rows()` (`:439`) collapses the history first:

```
agg_l[b] = sum_t  0.5^t * A_l[ q_{t,l}, b ]        # t = 0 is the last event
```

`score_sid_transition()` (`:459`) then reads one entry per catalog item:

```
score(j) = sum_l  w_l * log( agg_l[ c_{j,l} ] + s ) + 1e-6 * log1p(popularity_j)
              s = 0.5 (smoothing),  w_l = 1 (weighting="uniform")
```

The per-query normaliser `log Z_l[q_l]` is dropped deliberately. It is constant
across `j`, so it cannot change the ranking; keeping it would only cost a log-sum
per level.

### Why it is expected to fail, and does

At level 0 the row `A_0[a, :]` is a distribution over 256 successors. With
~526 hits per used cell spread over a 98.6%-dense row, the row is close to flat,
so `log(agg_0[b])` is nearly the same number for every `b` that shares level 0
with the query. The score therefore degenerates into a smooth function of code
frequency, which is a blurry popularity list with a coarser resolution than the
real popularity count. Measured R@1 0.0244% against popularity's 0.4395%, i.e.
18x worse, which is what a blurred popularity list should do.

The design lesson: a first-order transition table has `V_l^2` free cells and only
34M observations. That is 519 observations per cell at level 0 and 64 per cell at
level 4 (`34,019,640 / 534,361`). There is not enough data per cell for a
conditional distribution to beat a marginal.

---

## 4. B4: co-click kNN

### Table

`build_coclick()` (`Model Training/baselines_digr.py:513`) counts item-to-item
pairs directly, in item space rather than code space:

```
C[a, b] = #{ consecutive events (a -> b) } + #{ (last event a -> target b) }
```

stored as a CSR with `int32` counts. Measured: 34,019,640 raw pairs collapse to
**25,669,281 distinct pairs**, mean out-degree **5.37**, max out-degree **566**.
The lookup from SID tuple to catalog row is injective: all 4,783,154 tuples are
unique, which is the check that the 5-code SID really has no collisions.

### Score

`top_from_coclick()` (`:562`):

```
total[b] = sum_{t=0}^{w-1}  0.5^t * C[ q_t, b ]        restricted to successors
out      = argtop_k( total )                            of the window items
then fill any remaining slots from the popularity order
```

`w = 5` in the reported run. No normalisation by the source item's out-degree,
which is a real weakness (a popular source item floods the candidate list) but
was not the binding constraint.

### The structural point

The candidate set is the union of successors of 5 items. Mean out-degree is 5.37,
so about **27 candidate slots are backed by data** and everything past that is
popularity padding. That single sentence explains every number in the results
table:

- ranks 1-27 are where real signal can appear: R@1 0.785%, R@10 3.60%, R@100
  10.45% (partly padded, so partly popularity)
- R@1000 23.62% and R@10000 44.19% are increasingly just the popularity list.
  Popularity alone gives 14.73% and 36.82%, so co-click adds +8.9 and +7.4
  points over 1,000 and 10,000 slots. It is not doing much in the tail.

Compared with DiGR's 49.77% at R@10,000, co-click loses 5.6 points, and that is
the only cut-off where it loses.

---

## 5. DiGR

### 5.1 The score

`item_logits()` (`Model Training/train_digr.py:411`):

```
s_i = ( 1 / (L * tau) ) * sum_{l=0}^{L-1}  h . E_l[ c_{i,l} ]
```

which is the DiGR score with `L = 5`, `tau = 0.07`, so the constant is
`1/(5*0.07) = 2.857`. `l2_normalize` is off in every reported run (when on, `h`
and every code vector are L2-normalised first).

### 5.2 The exact factorisation

Because the sum is over levels and `h` does not depend on `l`:

```
s_i = (1/(L*tau)) * h . ( sum_l E_l[c_{i,l}] )
    = (1/tau) * h . V_i ,      V_i := (1/L) * sum_l E_l[c_{i,l}]
```

`V` is precomputed once, `(K, d)`, and `precompute_vectors()` (`:487`) does
exactly this. Nothing is approximated: the "generative" scorer is, algebraically,
a single dot product between the query vector and a fixed item vector.

Cost falls from `B * K * L` gathers to `B * K` multiplies. That is why a 4.78M-way
softmax is affordable at all, and it is also why the head is cheap: the entire
item representation is a 4.78M-row matrix that is the average of 5 lookups into
1,755 rows.

### 5.3 Loss

Full-softmax path:

```
Z   = sum_j exp(s_j)                        # j over all 4,783,154 items
CE  = -s_target + log Z
```

Uniform CE is 15.381, so 11.24 means the model narrowed the distribution by
`exp(15.381 - 11.243) = exp(4.138) = 62.7x`. In probability terms the average
target sits at about `e^-11.243 = 1.31e-5`, or rank 76,000 of 4.78M. That is the
whole achievement: 62x narrower than guessing, 76,000 times wider than knowing.

### 5.4 Sampled softmax, and its residual bias

`item_logits_sampled()` (`:448`) scores one positive plus `N` negatives:

```
logits = [ s_pos , s_n1 + log(K/N), ..., s_nN + log(K/N) ]      CE against index 0
```

so the estimator is

```
CE_hat = -s_pos + log( exp(s_pos) + (K/N) * sum_n exp(s_n) )
```

with negatives drawn `iid ~ Uniform(1..K)` **with replacement, and not excluding
the positive**. Bias:

```
E[ exp(s_pos) + (K/N) * sum_n exp(s_n) ]  =  exp(s_pos) + Z
```

The positive is counted twice in expectation, so the estimator targets
`log(Z + exp(s_pos))` instead of `log Z`. The excess is

```
bias = log(1 + p_pos) <= p_pos = exp(-11.243) = 1.31e-5 nats
```

Five orders of magnitude below the loss and far below the sampling noise at
`N = 300,000` (which covers 6.3% of the catalog). Also: `neg_tokens` is drawn
once per step and shared across all 512 rows (`:1292`), so gradients within a
batch are correlated. That costs variance, not bias.

Both facts are worth knowing if the loss is ever compared across `N`. The
`log(K/N)` term is `log(4,783,154 / 300,000) = 2.769` in the reported run.

### 5.5 What the score can and cannot express

Substituting 5.2 into the ranking, and writing `u_l = E_l . h` so that `u_l` is a
length-`V_l` vector of code affinities:

```
s_i = (1/(L*tau)) * sum_l  u_l[ c_{i,l} ]
```

Three consequences, all structural rather than empirical.

It is an additive model with no interaction terms. Two independent pieces of
evidence for item `i` (say "its level-0 code is popular" and "its level-2 code
matched my history") can only be added. There is no `u_l[a] * u_m[b]` term. In
factorisation-machine language this is the linear part only.

Every code vector is shared by many items. All four first levels have 256 codes,
so each level-0 vector is shared by `4,783,154 / 256 = 18,684` items, and each
level-5 vector by `4,783,154 / 731 = 6,544`. Since the score is a plain sum,
items inside a group are separated only by the one level where they differ.

The whole item-side parameter count is `1,755 * 1,024 = 1,797,120` numbers, and
those describe 4,783,154 items. That is 0.38 free parameters per item. A
per-item embedding table would have `4.78M * 1024 = 4.9e9`, and the co-click
table has 25,669,281 non-zero entries, which is 14.3x the head's parameter count.
So the comparison "DiGR versus a count table" is a comparison between 1.8M and
25.7M learned numbers on the same data.

Inside the model there is no cross-level interaction, and
`evaluate()`'s `rank` inherits that limitation exactly.

### 5.6 The label-subset arithmetic

`--label_subset clicked` restricted the softmax denominator to the 504,404
`train ∪ val` targets (`resolve_label_subset`, `:665`). Under a softmax over the
subset, the reported CE is

```
CE_subset = -log( p_full(target) / Z_subset ),   Z_subset = p_full(subset)
CE_full   = CE_subset - log Z_subset
```

Measured on the same model: `CE_subset = 9.7457`, `CE_full = 11.4362`, so
`log Z_subset = -1.6905` and `Z_subset = exp(-1.6905) = 18.45%`. The model put 18.5% of its
probability mass on the 10.5% of items it was trained on, and 81.5% on the 89.5%
it never saw as a positive or a negative. That is not a bug in the arithmetic, it
is the arithmetic: with negatives also sampled from the subset (`:1292`), nothing
in the objective ever pushed the other 4.28M items down, and they took their
frequency-proportional share of mass.

Which is why the subset was reverted. It made the epoch no faster (87.3 to 87.8
min/epoch, against 88.7 min/epoch with the full denominator), because the epoch
cost is the 221.5M-token backbone forward pass, not the softmax width.

### 5.7 Deviations from the paper

Faithful: the score form 5.1, the `1/(L*tau)` scale, one softmax over the whole
catalog, the exponential moving average of the query tower, RQ-VAE Semantic IDs.

Deviations, each deliberate:

| deviation | why |
|---|---|
| Qwen3-0.6B decoder-only instead of T5 encoder; `h` is the last non-pad token | the paper's encoder pooling is not available in a causal LM, and last-token is what the causal attention has actually read |
| static user features prepended as tokens, plus one temporal token per event | replaces the paper's ID token; actions are never tokens |
| 5 SID levels, the 5th being a collision suffix | RQ-VAE here is 4 codebooks x 256 on 4.78M items, so a 5th level disambiguates |
| `--sampled_softmax` (training) | the paper approximates inference only, never training. Opt-in, off by default, and the reported run is not the paper's setting |
| `--label_subset` | tried and reverted, see 5.6 |
| full fine-tune, not LoRA | LoRA was only 12% slower, but full fine-tune removes an adapter from every checkpoint path. Both are supported |
| gradient checkpointing on | at `B=512` the un-checkpointed activations are about 127 GB against an 80 GB card |

---

## 6. TIGER

Not implemented yet. The math is here because it determines what the comparison
will and will not mean.

### 6.1 Objective

```
p(c | h) = prod_{l=1}^{L} p_l( c_l | h, c_1..c_{l-1} )
CE       = -sum_{l=1}^{L} log p_l( c_l | h, c_1..c_{l-1} )
```

One forward pass with teacher forcing scores all five levels: the input is
`h`-tokens followed by `c_1..c_{L-1}`, the labels are `c_1..c_L`. This is the
chain rule applied to the item identity, so the item-level quantity is
`-log p(item)`, exactly as in DiGR 5.3. The difference is the parameterisation:
DiGR's `p(item)` is the log-bilinear additive form of 5.1; TIGER's is a product
of five conditionals, and `h` is re-read at every level, so different prefixes can
be handled by different computations.

### 6.2 The two losses are not on the same scale

Each `p_l` is a softmax over `V_l`, so the product is normalised over the whole
code product space:

```
prod_l V_l = 256*256*256*256*731 = 3.14e12 tuples
valid items                      = 4,783,154   (1.52e-6 of the space)
```

So `sum over valid p < 1`, and writing `Z_valid = sum over valid p`:

```
CE_TIGER = -log q(item) + log(1 / Z_valid),   q = p / Z_valid
```

`q` is what is comparable to DiGR's item distribution. The two loss numbers differ
by the constant `log(1/Z_valid)`, which is measurable by summing the trie
probabilities and must be reported alongside the loss. Do not compare a TIGER
loss of, say, 23 nats with DiGR's 11.24 nats and conclude anything.

Rough scale, from `ln 256 = 5.545` and `ln 731 = 6.594`: the entropy ceiling over
the code space is `4*5.545 + 6.594 = 28.77`, and a plausible converged value is
somewhere in the low twenties.

### 6.3 The suffix level is the soft spot

Level 5 is a rank index, not a semantic code. But it is not pure noise either:
`2,904,256` items (60.7%) are alone in their 4-code prefix, so their level-5 code
is 0. Conditional on the prefix being right, level 5 is mostly free. Its
contribution to ranking is a within-group tie-break inside the 1,878,898 items
that do share a prefix.

### 6.4 Decoding

`R@k` is produced by beam search over a prefix trie. Two facts follow directly
from that and should be stated before the run:

- A beam of width `b` returns at most `b` candidates, so `R@k` for `k > b` is
  meaningless and `R@k` for `k = b` is capped by 1. Reported `R@100` needs
  `b >= 100`. DiGR and co-click have no such cap, because they score all
  candidates.
- Beam search is not exhaustive, so TIGER's `R@k` is a lower bound on what the
  distribution supports. An exhaustive trie walk over the 4.78M valid items is
  possible but costs 4.78M forward passes, i.e. it is not.

`Model Training/baselines_digr.py` and `eval_full_catalog.py` therefore remain
the only exactly-comparable evaluators, and TIGER numbers should be reported as
"beam b, R@k for k <= b".

### 6.5 What TIGER can express that DiGR cannot

`h` is re-encoded at every level, so the effective per-item function is a sum of
five conditionals each of which is computed from a different context. The number
of distinct decision contexts is up to `prod` of the prefix counts rather than a
single additive read, so TIGER can represent "this level-2 code matters only when
level 0 is this" and DiGR cannot. That is the specific hypothesis under test, and
it is falsifiable: if TIGER's `R@10` lands near co-click's 3.60%, the additive
score was not the binding ceiling and the next suspect is the SID itself.

---

## 7. Cost

| path | operations | measured |
|---|---|---|
| DiGR train step, `N=300k` | `B*d*1755` gemm + `B*N*L` gathers | 1.63 s/step at `B=512` |
| DiGR full eval, `B=128` | `B*d*1755` + `B*K*L` gathers (`~49 GB` of memory traffic) | 3.2 min / 98,304 rows |
| DiGR export | one `(K,d)` matrix, fp16 | 9.8 GB |
| B4 co-click eval | `w` CSR row reads + one argpartition | 46 s / 98,304 rows, single core |
| B3 eval | 5 x `(V_l, V_l)` row sums + 5 x `K` gathers | 3,805 s / 98,304 rows, 8 workers |
| popularity | one sort | under 1 s |

---

## 8. Arithmetic that must check out

These are the consistency tests already used to catch errors. All hold.

```
34,019,640 events - 1,719,430 rows          = 32,300,210 consecutive pairs
32,300,210 + 1,719,430 anchors              = 34,019,640  = every A_l.sum()
2 + 4*256 + 731 + 64 + (87+3+8+15+20+5+4+3) = 1966        = vocab total_size
256*256*256*256*731                         = 3.14e12      = TIGER's space
1,719,430*1 + 15,663,266 feat + 34,019,640*6 = 221,500,536 = train tokens
ln(4,783,154)                               = 15.3807      = uniform CE
ln(504,404)                                 = 13.1311      = subset uniform CE
exp(9.7457 - 11.4362)                       = 0.1845       = subset mass share
1,755 * 1,024                               = 1,797,120    = head parameters
25,669,281 / 1,797,120                      = 14.28        = capacity ratio
```

The metrics themselves reproduce: popularity `R@1 = 0.00439453125` is exactly
`432/98,304`, and the recomputed `R@10 = 0.013437906901041666` is exactly
`1321/98,304`. When a cut-off reproduces to the count but a lower one does not,
the two numbers came from different inputs, which is how the retired popularity
`R@10` of 0.9644% was caught.

# Mathematics of item-embedding training and SID generation

This document describes the mathematics of two scripts:

- `train_item_embedding_cli.py` produces the item embeddings written to
  `item_embeddings.parquet`. It is a two-tower retriever: ItemDNN item tower,
  HSTU sequence tower, InfoNCE with LogQ correction.
- `rq_vae_cli.py` consumes those embeddings and produces the item-to-SID mapping
  in `index_rqvae.json`. It is an RQ-VAE baseline plus a modified variant.

The two parts are independent models with independent notation. Where a symbol is
reused with a different meaning it is flagged at the point of reuse.

---

# Part I: Stage 1, ItemDNN and HSTU two-tower retrieval

## 1. Notation and data

Let the catalog be $\mathcal{I} = \{i_1, \dots, i_N\}$. Define the index map

$$
\iota : \mathcal{I} \to \{1, \dots, N\}, \qquad \iota(i_k) = k,
$$

and reserve index $0$ for padding, so the feature store has $M = N + 1$ rows and
row $0$ is the zero row. Define $\bar e_0 = \mathbf{0}$ and $\bar e_k$ for $k \ge 1$ as
below.

Each item $k$ carries three raw fields:

$$
s_k \in \mathbb{R}^{d_s}, \quad d_s = 1024
\qquad \text{(multimodal semantic embedding, \texttt{mm\_emb})},
$$

$$
h_k \in \{0,1\} \qquad \text{(\texttt{has\_mm\_emb} presence flag)},
$$

$$
c_k = (c_{k,1}, \dots, c_{k,C}) \in \mathbb{Z}^{C}, \quad C = 13
\qquad \text{(categorical integer codes)}.
$$

If $h_k = 0$ or the vector is malformed or non-finite, the code stores $s_k = \mathbf{0}$
and keeps $h_k = 0$. The model can therefore distinguish "missing" from
"numerically zero" because $h_k$ is concatenated to $s_k$.

A training sample is a tuple

$$
\big(x_{1:L},\, a_{1:L},\, m_{1:L},\, y\big),
$$

with $L \le L_{\max} = 100$,

$$
x_t \in \{0, \dots, N\} \ \text{(history item indices, newest last)},
\qquad
a_t \in \{0,1,2,3\},
$$

$$
a_t =
\begin{cases}
0 & \text{PAD} \\
1 & \text{exposure} \\
2 & \text{click} \\
3 & \text{unknown}
\end{cases},
\qquad
m_t = \mathbb{1}[x_t \neq 0] \in \{0,1\},
$$

and $y \in \{1, \dots, N\}$ the next-item target.

For each categorical column $j$, the raw integer values are remapped. Let
$V_j = \{v \neq 0 : v \text{ occurs in column } j\}$ and let $\rho_j$ be the rank of
$v$ inside $\operatorname{sort}(V_j)$. Then

$$
c'_{k,j} =
\begin{cases}
\rho_j(v) + 1, & v \neq 0 \\
0, & \text{otherwise}
\end{cases},
\qquad
K_j = |V_j| + 1 .
$$

Hence every code is in $\{0\} \cup \{1, \dots, K_j - 1\}$ with $0$ as padding, and
column cardinality is $K_j$.

## 2. Item tower: ItemDNN

### Parameter shapes

$C = 13$ embedding tables, each with $d_{\text{cat}} = 32$:

$$
E_j \in \mathbb{R}^{K_j \times d_{\text{cat}}},
\qquad E_j(0) = \mathbf{0} \ \text{(padding row frozen)}.
$$

Semantic projection:

$$
W_s \in \mathbb{R}^{512 \times (d_s + 1)}, \qquad b_s \in \mathbb{R}^{512}.
$$

Trunk: $W_1 \in \mathbb{R}^{256 \times (512 + 32C)}$, $b_1$, then
$W_2 \in \mathbb{R}^{256 \times 256}$, $b_2$. Output width $d_e = 256$.

### Forward pass

For item index $k$ define the augmented semantic input and its projection

$$
\phi_k = \operatorname{Dropout}\Big(\operatorname{SiLU}\big(\operatorname{LN}(W_s [s_k; h_k] + b_s)\big)\Big) \in \mathbb{R}^{512}.
$$

Concatenate with the categorical embeddings,

$$
g_k = \big[\phi_k;\, E_1(c'_{k,1});\, \dots;\, E_C(c'_{k,C})\big]
\in \mathbb{R}^{512 + 32C} = \mathbb{R}^{928},
$$

and push through the MLP,

$$
f_k = W_2 \, \operatorname{Dropout}\Big(\operatorname{SiLU}\big(\operatorname{LN}(W_1 g_k + b_1)\big)\Big) + b_2 \in \mathbb{R}^{256}.
$$

Finally L2-normalize onto the unit sphere and zero the padding row:

$$
\hat e_k = \frac{f_k}{\|f_k\|_2},
\qquad
\bar e_k = \hat e_k \odot \mathbb{1}[k \neq 0].
$$

So all item representations live on $S^{255}$, and $\bar e_0 = \mathbf{0}$.

## 3. Sequence tower: HSTU

### Input composition

Let $A \in \mathbb{R}^{4 \times 256}$ be the action embedding table with row $0$
(PAD) frozen at zero. The sequence input is

$$
z_t^{(0)} = \Big(\operatorname{LN}_{\text{in}}\big(\bar e_{x_t} + A_{a_t}\big) + p_t\Big) \odot m_t,
\qquad p_t \in \mathbb{R}^{256},
$$

where $p_t$ is a learned absolute position embedding. The $\operatorname{LN}_{\text{in}}$ is the
"scale alignment": the action embedding and the item embedding are normalized
*after* summation, so a large action vector cannot swamp the item vector. The
multiplication by $m_t$ hard-zeroes padded slots.

### HSTU block

Three identical blocks ($B_{\text{blocks}} = 3$), $H = 4$ heads, head width
$d_h = 256/4 = 64$. For block input $z \in \mathbb{R}^{L \times 256}$:

$$
\begin{align}
\tilde z &= \operatorname{LN}_{\text{pre}}(z), \\
[u_{\text{lin}}, q_{\text{lin}}, k_{\text{lin}}, v_{\text{lin}}]
&= \tilde z \, W_{\text{in}}^{\!\top} + b_{\text{in}},
\qquad W_{\text{in}} \in \mathbb{R}^{1024 \times 256}, \\
u &= \operatorname{SiLU}(u_{\text{lin}}), \quad q = \operatorname{SiLU}(q_{\text{lin}}),
\quad k = \operatorname{SiLU}(k_{\text{lin}}), \quad v = v_{\text{lin}}.
\end{align}
$$

Reshape to heads with $q^{(h)}, k^{(h)}, v^{(h)} \in \mathbb{R}^{L \times 64}$.
Note that **no softmax is used**. The attention weights are the SiLU of the raw
scores,

$$
\begin{align}
S^{(h)}_{ij} &= \frac{\langle q_i^{(h)}, k_j^{(h)} \rangle}{\sqrt{d_h}} + b_h(r_{ij}), \\
r_{ij} &= \min\!\big(\max(i - j, 0),\, L_{\max} - 1\big),
\qquad b_h : \{0, \dots, L_{\max} - 1\} \to \mathbb{R}, \\
M_{ij} &= \mathbb{1}[j \le i] \cdot m_i \cdot m_j, \\
A^{(h)}_{ij} &= \operatorname{SiLU}\big(S^{(h)}_{ij}\big) \cdot M_{ij}, \\
o_i &= \bigoplus_{h=1}^{H} \sum_{j=1}^{L} A^{(h)}_{ij} \, v_j^{(h)} \in \mathbb{R}^{256}.
\end{align}
$$

$b_h$ is a learned relative distance bias, an embedding table of shape
$(L_{\max}, H)$, one scalar per head per integer distance, clamped at
$L_{\max} - 1$. The mask $M$ enforces causality ($j \le i$) and drops any key or
query slot that is padding. Because $\operatorname{SiLU}$ can be negative and is not normalized,
attention is unnormalized and signed; nothing is divided by the row sum. The
reference HSTU formulation divides $A$ by the valid sequence length; this
implementation does not.

The block output is a gated residual update,

$$
z_i \leftarrow z_i + \operatorname{Dropout}\Big(W_{\text{out}}\big(\operatorname{LN}_{\text{attn}}(o_i) \odot u_i\big) + b_{\text{out}}\Big),
$$

so $u$ acts as a multiplicative input-dependent gate and $\operatorname{LN}_{\text{attn}}$
renormalizes the attention aggregate before it is scaled by the gate.

### Pooling to a user vector

After the last block,

$$
z = \operatorname{LN}_{\text{final}}(z),
\qquad
\ell = \max\!\Big(1, \sum_{t=1}^{L} m_t\Big),
\qquad
u = \frac{z_{\ell - 1}}{\|z_{\ell - 1}\|_2} \in S^{255}.
$$

The user representation is the last valid position's hidden state, L2-normalized.
This is the "causal last-token" readout, consistent with next-item prediction.

## 4. Shared embedding space and the scoring function

The item tower and the user tower share the catalog embedding table defined by
ItemDNN. The target item is embedded with the same ItemDNN as the history items,
so

$$
\text{score}(u, y) = \langle u, \bar e_y \rangle = \cos(u, f_y) \in [-1, 1].
$$

Both sides are unit norm, so the score is exactly the cosine similarity. The full
forward output of the model is not a prediction but a loss, and the caller also
receives the similarity matrix

$$
P_{ij} = \langle u_i, \bar e_{y_j} \rangle, \qquad P \in \mathbb{R}^{B \times B}.
$$

## 5. Loss: InfoNCE with LogQ correction

### Temperature

$$
T = \mathrm{clamp}(\tau,\, 10^{-3},\, 1.0), \qquad \tau = 0.05 \ \text{(default)},
$$

where $\tau$ is either a fixed buffer or a learned scalar parameter (the
`learnable_temperature` flag). Writing $T$ as a clamped scalar prevents the
learned version from collapsing the logits' dynamic range.

### Logits

Let $q_j = \log Q(y_j)$ be the log sampling probability of item $y_j$, stored in a
buffer of length $N + 1$ (row $0$ unused). For a local batch of size $B$,

$$
\Lambda_{ij} = \frac{P_{ij}}{T} - \lambda \, q_j,
\qquad \lambda = \texttt{log\_q\_weight} = 1.0.
$$

The candidate set is the batch itself, so position $j$ with $y_j$ is a negative for
every row $i \neq j$ and the positive for row $i = j$. If $Q$ is uniform, $q_j$ is
constant over $j$ and the second term is a constant shift that cancels in the
softmax; when $Q$ is popularity or frequency based, the term subtracts the log
probability of each item, which approximates sampling correction, meaning popular
items get a penalty proportional to how often they were sampled instead of a free
win.

### Cross-entropy objective

With labels $\ell_i = i$,

$$
\mathcal{L}
= -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\Lambda_{ii})}{\sum_{j=1}^{B} \exp(\Lambda_{ij})}
= \frac{1}{B} \sum_{i=1}^{B} \Big[\log \sum_j e^{\Lambda_{ij}} - \Lambda_{ii}\Big].
$$

The gradient with respect to the scores is the usual softmax-minus-one-hot,

$$
\frac{\partial \mathcal{L}}{\partial \Lambda_{ij}} = \frac{1}{B}\big(\pi_{ij} - \delta_{ij}\big),
\qquad
\pi_{ij} = \frac{e^{\Lambda_{ij}}}{\sum_k e^{\Lambda_{ik}}},
$$

and therefore

$$
\frac{\partial \mathcal{L}}{\partial P_{ij}} = \frac{1}{BT}\big(\pi_{ij} - \delta_{ij}\big),
\qquad
\frac{\partial \mathcal{L}}{\partial q_j} = -\frac{\lambda}{B} \sum_i \big(\pi_{ij} - \delta_{ij}\big).
$$

Two consequences matter. First, the effective learning rate on the similarity is
scaled by $1/T$, which makes $T = 0.05$ a $20\times$ gradient amplifier relative to
$T = 1$. Second, gradient flows into the ItemDNN through two paths at once:
through $\bar e_{y_j}$ on the positive and negative sides of the denominator, and
through the user vector $u_i$, which itself contains $\bar e_{x_t}$ for every
history token. The item tower is therefore trained as a shared bi-encoder rather
than a frozen lookup.

## 6. Multi-GPU: cross-GPU negatives

If `cross_gpu_negatives` is set and $\text{world\_size} = W > 1$, all ranks
all-gather the positive embeddings and their LogQ values
(`dist_nn.all_gather` is autograd-aware):

$$
\tilde P = \big[\, \bar e_{y} \text{ from rank } 0 \,|\, \cdots \,|\, \text{from rank } W-1 \,\big]
\in \mathbb{R}^{WB \times 256},
\qquad
\tilde q \in \mathbb{R}^{WB}.
$$

Then

$$
\Lambda_{ij} = \frac{\langle u_i, \tilde P_j \rangle}{T} - \lambda \tilde q_j
\in \mathbb{R}^{B \times WB},
\qquad
\ell_i = i + B \cdot \text{rank},
$$

so the negative set grows to $WB$ while each GPU keeps its own $B$ rows. Otherwise
the loss is local, with candidate set size $B$. DDP synchronizes the parameter
gradients; `DistributedSampler` shards the data with `set_epoch(epoch)` for correct
shuffling.

## 7. Evaluation metrics

Rank of the positive under a descending sort of similarities:

$$
r_i = 1 + \#\{j : P_{ij} > P_{ii}\}.
$$

Then

$$
\begin{align}
\mathrm{Recall@}k &= \frac{1}{B} \sum_{i=1}^{B} \mathbb{1}[r_i \le k],
\qquad k \in \{1, 5, 10, 50\}, \\
\mathrm{MRR} &= \frac{1}{B} \sum_{i=1}^{B} \frac{1}{r_i}, \\
\mathrm{NDCG@}10 &= \frac{1}{B} \sum_{i=1}^{B} \frac{\mathbb{1}[r_i \le 10]}{\log_2(r_i + 1)}, \\
\mathrm{pos} &= \frac{1}{B} \sum_i P_{ii}, \\
\mathrm{neg} &= \frac{1}{B(B-1)} \sum_{i \neq j} P_{ij}, \\
\text{gap} &= \mathrm{pos} - \mathrm{neg}.
\end{align}
$$

The candidate set behind every one of these numbers is the batch, not the catalog.
$P \in \mathbb{R}^{B \times B}$, so $r_i$ is the positive's rank among $B$ items and
$\mathrm{Recall@}k$ only means something relative to the batch size. At $B = 4096$ a
`Recall@50` of $0.4$ says the positive lands in the top $50$ of $4096$, which is
$1.2\%$ of the batch. It is not $40\%$ catalog recall and should never be quoted as
if it were. Only the $\mathrm{pos}$, $\mathrm{neg}$ and $\text{gap}$ columns behave
the same way in a batch and in a catalog. There is also no miss case to handle here:
the positive is always inside the candidate set, so $r_i$ always exists and always
satisfies $r_i \le B$. An evaluator that ranks against the full 4.78M-item catalog does
have to invent a rank for misses, and the DiGR one uses $k_{\max}+1$ so that the miss
fails every $\le k$ test. Finally, ties count as wins, because the rank uses a strict
$>$ comparison.

The gap is the separation statistic: positive cosine minus mean negative cosine,
and it is only meaningful because both sides are unit norm.

## 8. Optimization

AdamW with decoupled weight decay $\eta_{\text{wd}} = 10^{-4}$, base learning rate
$\eta_0 = 10^{-3}$, fused kernel on CUDA. The schedule is linear warmup followed by
cosine decay to a floor:

$$
\eta(t) =
\begin{cases}
\eta_0 \cdot \dfrac{t+1}{W}, & t < W, \\[2mm]
\eta_0 \Big[\rho + \tfrac{1}{2}(1 - \rho)\big(1 + \cos(\pi p)\big)\Big], & t \ge W,
\end{cases}
\qquad
p = \frac{t - W}{T_{\text{tot}} - W},
$$

with warmup fraction $0.05$, $W = \lfloor T_{\text{tot}} \cdot 0.05 \rfloor$,
$T_{\text{tot}} = E \cdot \lfloor n_{\text{train}} / B \rfloor$, and floor
$\rho = \texttt{min\_lr\_ratio} = 0.01$ (the function's default is $0.001$,
overridden here). One scheduler step per optimizer step, so the warmup is in steps,
not epochs.

Gradient clipping uses the global norm

$$
g = \sqrt{\sum_{p \in \Theta} \|\nabla_p\|_2^2},
\qquad
\nabla_p \leftarrow \nabla_p \cdot \min\!\Big(1, \frac{c}{g + \epsilon}\Big),
\qquad c = 1.0,
$$

applied after `unscale_` when the FP16 GradScaler is active. Mixed precision is
bfloat16 by default (no scaler) or float16 with a GradScaler under `--fp16`. TF32
matmuls are enabled, and the checkpoint is written atomically via a `.tmp` rename
containing the item tower, sequence tower, action table, input LayerNorm,
optimizer, scheduler, scaler, epoch, step, best metric, cardinalities, and
columns.

## 9. Inputs and outputs

**Inputs.** `canonical_items.parquet` (item ID, 1024-d embedding, presence flag, 13
categorical code columns), plus `train_samples.parquet` and `val_samples.parquet`
with columns `history_items`, `history_actions`, `target`. Histories shorter than
$L_{\max}$ are right-padded and a boolean validity mask is returned alongside them.

**Per-forward outputs.** The loss scalar, the logits $\Lambda$, the similarity
matrix $P$, the labels, and the user vectors $u$.

**Final artifacts.** `stage1_rec_model.pt` (weights plus cardinalities and columns
for reproducible inference) and, from `export_catalog_embeddings`, the full catalog
matrix

$$
\big[\bar e_1; \dots; \bar e_N\big] \in \mathbb{R}^{N \times 256}
$$

written to `item_embeddings.npy` and `item_embeddings.parquet`, the ID array
`item_ids.npy` with the row-order correspondence $k \mapsto i_k$, and a
`faiss.IndexFlatIP` index over the same matrix. Flat inner product is the correct
index because the vectors are already unit norm, so inner product equals cosine.
Item IDs never enter the network; only their dense representations do.

---

# Part II: Stage 2, RQ-VAE and SID construction

**Notation note.** Part II reuses two symbols from Part I with new meanings. Here
$M$ is the number of codebooks, not the feature-store row count, and $B$ is the
batch size as before.

## 10. Notation and data

Let the catalog embeddings be $x_1, \dots, x_N \in \mathbb{R}^{d_x}$ with
$d_x = \texttt{input\_dim} = 256$, loaded together with their item IDs from an HDF5
or Parquet file. That is the width of the item tower's output,
$[\bar e_1; \dots; \bar e_N] \in \mathbb{R}^{N \times 256}$ from section 9. It has to
be passed explicitly, because `--input_dim` defaults to $1024$.

A deterministic permutation $\pi$ of the index set splits the
corpus into

$$
\mathcal{T} = \pi(1{:}N_{\text{tr}}),
\qquad
\mathcal{V} = \pi(N_{\text{tr}}+1{:}N),
\qquad
N_{\text{tr}} = N(1 - \rho), \quad \rho = \texttt{val\_ratio} = 0.1 .
$$

Nothing in the model sees an item ID. The ID is only reattached after quantization,
as the key of the output dictionary.

The model is a residual quantized autoencoder (RQ-VAE) with a quantized bottleneck.
It has three learnable parts:

$$
\underbrace{f_\theta : \mathbb{R}^{d_x} \to \mathbb{R}^{d_h}}_{\text{encoder}},
\qquad
\underbrace{\{E_1, \dots, E_M\},\ E_d \in \mathbb{R}^{K \times d_h}}_{\text{$M$ codebooks}},
\qquad
\underbrace{g_\phi : \mathbb{R}^{d_h} \to \mathbb{R}^{d_x}}_{\text{decoder}},
$$

with $d_h = \texttt{hidden\_dim} = 64$, $M = \texttt{num\_codebooks} = 4$,
$K = \texttt{codebook\_size} = 256$. The codebook entry $E_{d,k}$ is the $k$-th
centroid of stage $d$. Write $e_{d,k} \in \mathbb{R}^{d_h}$ for that row.

## 11. Encoder, bottleneck normalization, jitter

### Encoder

Four fully connected layers with ReLU between them, and no output activation:

$$
\begin{align}
a^{(1)} &= \mathrm{ReLU}(W_1 x + b_1), && W_1 \in \mathbb{R}^{256 \times d_x}, \\
a^{(2)} &= \mathrm{ReLU}(W_2 a^{(1)} + b_2), && W_2 \in \mathbb{R}^{128 \times 256}, \\
a^{(3)} &= \mathrm{ReLU}(W_3 a^{(2)} + b_3), && W_3 \in \mathbb{R}^{128 \times 128}, \\
z &= W_4 a^{(3)} + b_4, && W_4 \in \mathbb{R}^{64 \times 128}.
\end{align}
$$

### Parameter-free LayerNorm, then unit-sphere rescale

Let $d_h = 64$ and define the reduction

$$
\mu(z) = \frac{1}{d_h} \sum_{j=1}^{d_h} z_j,
\qquad
\sigma^2(z) = \frac{1}{d_h} \sum_{j=1}^{d_h} \bigl(z_j - \mu(z)\bigr)^2,
\qquad
\tilde z = \frac{z - \mu(z)\mathbf{1}}{\sqrt{\sigma^2(z) + \epsilon}},
\quad \epsilon = 10^{-8}.
$$

Because $\operatorname{LN}$ is built with `elementwise_affine=False`, there is no learnable gain
$\gamma$ and no bias $\beta$. The normalization is followed by a fixed scalar
rescale:

$$
\boxed{\ \bar z = \frac{\tilde z}{\sqrt{d_h}}\ }
\qquad \Longrightarrow \qquad
\|\bar z\|_2^2 = \frac{\sigma^2(z)}{\sigma^2(z) + \epsilon} \approx 1 .
$$

So every latent lies on a sphere of radius about $1$ in $\mathbb{R}^{64}$. Two
consequences are load-bearing later: the commitment loss and the residual norms are
all on a fixed scale, and pure Gaussian noise added to $\bar z$ has to be small
(see the jitter subsection below).

The affine-free choice follows from a failure of the affine version. With learnable
$\gamma$, the optimizer shrank $\gamma$ on the most informative axes to lower the
commitment loss, which compressed $\bar z$ into a narrow tube and made the codebooks
see an anisotropic, shrinking target.

In `--run_baseline` mode `bottleneck_norm` is `None`, so `_normalize_latent` returns
$z$ untouched and neither the $\operatorname{LN}$ nor the $1/\sqrt{d_h}$ is applied.

### Jitter

During training only, with probability controlled by the flag, the stage-0 residual
is perturbed:

$$
r^{(0)} = \bar z + \varepsilon,
\qquad \varepsilon \sim \mathcal{N}(\mathbf{0}, J^2 I_{d_h}),
\qquad \mathbb{E}\|\varepsilon\|_2^2 = d_h J^2 .
$$

The default is $J = \texttt{jitter} = 0.0$. The reason is a magnitude comparison.
With $\|\bar z\|^2 \approx 1$ the stage-3 residual has norm squared around
$2.5 \times 10^{-4}$. At $J = 0.01$ the injected noise has
$\mathbb{E}\|\varepsilon\|^2 = 64 \cdot 10^{-4} = 6.4 \times 10^{-3}$, about $25$
times the signal it is added to, so stages 2 and 3 would quantize noise. The
admissible range is $J \in [0, 0.001]$. Baseline sets $J = 0$ regardless of the
flag.

## 12. Residual vector quantization cascade

Let $r^{(0)}$ be the stage-0 input defined above. For stages $d = 1, \dots, M$ the
codebook assignment is the nearest centroid under Euclidean distance, computed
through an affinity that avoids the square root:

$$
s_d(r, k) = 2\langle r, e_{d,k} \rangle - \|e_{d,k}\|_2^2
= \|r\|_2^2 - \|r - e_{d,k}\|_2^2 .
$$

Since the term $\|r\|_2^2$ does not depend on $k$,

$$
k_d = \arg\max_{k \in \{1, \dots, K\}} s_d\bigl(r^{(d-1)}, k\bigr)
= \arg\min_{k} \bigl\|r^{(d-1)} - e_{d,k}\bigr\|_2^2 ,
$$

and the quantized vector is $q_d = e_{d,k_d}$. The cascade advances by

$$
r^{(d)} = r^{(d-1)} - \mathrm{sg}\!\left[q_d\right],
\qquad \mathrm{sg}[\cdot] \ \text{is \texttt{.detach()}} .
$$

The stop-gradient on $q_d$ means the stage-$d$ codebook receives no gradient through
the residual path of later stages. It still receives gradient through its own loss
terms. Because $\|\bar z\| \approx 1$ and each stage removes the component it can
represent, the residual norms decay with depth, and the four indices together form a
semantic ID

$$
\mathrm{SID}(x) = \bigl(k_1(x),\, k_2(x),\, k_3(x),\, k_4(x)\bigr)
\in \{1, \dots, K\}^M,
\qquad K^M = 256^4 \approx 4.3 \times 10^{9}.
$$

## 13. Objective

### Per-stage codebook and commitment terms

For stage $d$ let $\mathcal{B}$ be the batch of size $B$. Two mean-squared errors
are formed with opposite stop-gradients:

$$
\begin{align}
\mathcal{L}^{cb}_d &= \frac{1}{B\,d_h} \sum_{i=1}^{B}
\bigl\| q_{d,i} - \mathrm{sg}[r^{(d-1)}_i] \bigr\|_2^2,
&& \frac{\partial \mathcal{L}^{cb}_d}{\partial e_{d,k}} \neq 0,
\ \ \frac{\partial \mathcal{L}^{cb}_d}{\partial \theta} = 0, \\[1mm]
\mathcal{L}^{cm}_d &= \frac{1}{B\,d_h} \sum_{i=1}^{B}
\bigl\| r^{(d-1)}_i - \mathrm{sg}[q_{d,i}] \bigr\|_2^2,
&& \frac{\partial \mathcal{L}^{cm}_d}{\partial \theta} \neq 0,
\ \ \frac{\partial \mathcal{L}^{cm}_d}{\partial e_{d,k}} = 0 .
\end{align}
$$

The first pulls each selected centroid onto the residual it was chosen for, so it is
the codebook fitting term. The second pulls the encoder output toward its assigned
centroid, so it is the encoder fitting term. Aggregating over depth,

$$
\mathcal{L}_{\text{rqvae}} = \sum_{d=1}^{M} \Bigl( \mathcal{L}^{cb}_d + \beta \, \mathcal{L}^{cm}_d \Bigr),
\qquad \beta = \texttt{beta} = 0.1 .
$$

The codebook rows are stored as unconstrained real parameters (no norm constraint,
no projection), so their magnitudes are free to shrink stage by stage along with the
residuals.

### Straight-through estimator and reconstruction

The decoder must see the quantized vector in the forward pass and the continuous
latent in the backward pass. This is the straight-through identity

$$
\tilde z = \bar z + \mathrm{sg}\!\left[\sum_{d=1}^{M} q_d - \bar z\right]
\quad \Longrightarrow \quad
\tilde z \Big|_{\text{forward}} = \sum_{d=1}^{M} q_d,
\qquad
\frac{\partial \tilde z}{\partial \bar z} \Big|_{\text{backward}} = I .
$$

So the decoder input is the sum of the four chosen centroids, and the gradient from
the decoder reaches the encoder as if no quantization had happened. The
reconstruction term is

$$
\hat x = g_\phi(\tilde z),
\qquad
\mathcal{L}_{\text{rec}} = \frac{1}{B\,d_x} \sum_{i=1}^{B} \bigl\| \hat x_i - x_i \bigr\|_2^2 .
$$

The reconstruction target is the raw input embedding, not a normalized one, so
$\mathcal{L}_{\text{rec}}$ is measured on the original scale. The decoder mirrors the
encoder, $64 \to 128 \to 128 \to 256 \to d_x$, each step $\mathrm{ReLU}(W u + b)$
and a linear output layer.

### Entropy regularizer

A stage can collapse to a handful of centroids. To spread the assignments, the model
adds a differentiable proxy for code usage. For each stage $d$, using a detached copy
of the codebook,

$$
\begin{align}
\delta_{ij} &= \mathrm{clamp}\Bigl(
\|r_i\|_2^2 - 2\langle r_i, \mathrm{sg}[e_{d,j}] \rangle + \|\mathrm{sg}[e_{d,j}]\|_2^2,\ 0
\Bigr)
= \bigl\| r_i - \mathrm{sg}[e_{d,j}] \bigr\|_2^2, \\
c_d &= \frac{1}{BK} \sum_{i=1}^{B} \sum_{j=1}^{K} \delta_{ij}, \\
a_{ij} &= -\frac{\delta_{ij}}{\tau \, (c_d + \epsilon)},
\qquad \tau = \texttt{temperature}, \\
p_{ij} &= \frac{e^{a_{ij}}}{\sum_{j'} e^{a_{ij'}}},
\qquad
\bar p_j = \frac{1}{B} \sum_{i=1}^{B} p_{ij}, \\
H_d &= -\sum_{j=1}^{K} \bar p_j \log(\bar p_j + \epsilon),
\qquad
H^{\max}_d = \log K, \\
\mathcal{L}^{\text{ent}}_d &= \frac{H^{\max}_d - H_d}{H^{\max}_d}
= 1 - \frac{H_d}{\log K} \in [0, 1].
\end{align}
$$

The stage weights $w_d = \texttt{stage\_weights} = [2.0, 1.5, 1.0, 1.0]$ shift emphasis
toward the shallow stages, and

$$
\mathcal{L}_{\text{ent}} = \frac{\sum_{d=1}^{M} w_d \, \mathcal{L}^{\text{ent}}_d}{\sum_{d=1}^{M} w_d}
= \frac{\sum_d w_d \, \mathcal{L}^{\text{ent}}_d}{5.5}.
$$

The total objective is

$$
\boxed{\ \mathcal{L} = \mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{rqvae}}
+ \lambda_{\text{ent}} \, \mathcal{L}_{\text{ent}} \ }
\qquad \lambda_{\text{ent}} = \texttt{entropy\_loss\_weight} = 0.002 .
$$

Three details of this regularizer carry its meaning.

The codebook is detached in the entropy branch. Because $\lambda_{\text{ent}} = 0.002$
is on the same order as the typical $\mathcal{L}^{cb}$ of $10^{-3}$ (about twice it),
live gradients would
let the cheapest way to flatten $\bar p$ be a collapse of all centroids onto one
point, or the migration of unused centroids into occupied Voronoi cells. Either
answers the entropy penalty while destroying the tessellation. Detaching makes the
term a pure encoder regularizer: it only tells the residual to spread across the
cells that already exist.

The batch mean distance $c_d$ is the per-stage temperature scale. The raw affinity
$s_d$ and the true squared distance differ by the per-row constant $\|r_i\|_2^2$, and
a constant per row cancels in the row softmax, so the forward probabilities of the
raw and expanded forms agree row for row. What the division by $c_d$ actually changes
is the scale of the logits: after division, the logits have batch mean $1$ at every
stage regardless of $\|r^{(d-1)}\|_2^2$, so $\tau$ produces the same degree of
sharpness at stage $1$ and stage $4$. Without it, the stage-dependent magnitude of
$\delta$ would make the same $\tau$ hyper-peaked early and flat late. The clamp at
zero only removes small negative values caused by floating-point rounding of the
expansion $\|r\|^2 - 2\langle r, e \rangle + \|e\|^2$.

The regularizer is a soft version of the hard assignment, and it tracks it. The hard
index is $\arg\min_j \delta_{ij}$. The softmax approaches that one-hot as
$\tau \to 0$, and the mean over the batch, $\bar p$, is the empirical usage
distribution the hard counts estimate. Driving $\mathcal{L}^{\text{ent}} \to 0$ drives
$\bar p$ toward uniform.

In `--run_baseline` mode `entropy_loss_weight`, `jitter` and `stage_weights` are
forced to $0$, $0$ and all ones, and the bottleneck normalization is removed.

## 14. Diagnostics

Let $\{k_{d,i}\}_{i=1}^{n}$ be the stage-$d$ hard assignments over a set of $n$ items
and

$$
\mathrm{cnt}_k = \#\{i : k_{d,i} = k\}.
$$

Then

$$
\text{used}_d = \#\{k : \mathrm{cnt}_k > 0\},
\qquad
\text{dead\%}_d = \Bigl(1 - \frac{\text{used}_d}{K}\Bigr) 100,
\qquad
\text{maxfreq}_d = \frac{\max_k \mathrm{cnt}_k}{n},
$$

and the hard-assignment entropy and perplexity

$$
p_k = \frac{\mathrm{cnt}_k}{n} \ \text{over the used } k,
\qquad
H_d = -\sum_{k} p_k \log p_k,
\qquad
\mathrm{PPL}_d = e^{H_d} \in [1, K].
$$

For the residual chain $\{r^{(m)}\}_{m=0}^{M}$ the reported statistics are

$$
\mathbb{E}\|r^{(m)}\|_2^2 = \frac{1}{n} \sum_{i} \|r^{(m)}_i\|_2^2,
\qquad
\overline{\|r^{(m)}\|_2},
\qquad
\mathrm{median}_i \|r^{(m)}_i\|_2 .
$$

Prefix uniqueness measures how much of the ID space is used at each depth:

$$
\mathrm{uniq}_d = \frac{\bigl|\{\,(k_{1,i}, \dots, k_{d,i}) : i = 1, \dots, n \,\}\bigr|}{n},
\qquad d = 1, \dots, M,
$$

so $\mathrm{uniq}_M$ is the fraction of items that have a distinct full SID, and
$1 - \mathrm{uniq}_M$ is the collision rate. Assignment stability reruns the encoder
on the same batch twice and checks

$$
\mathrm{stab} = \mathbb{1}\bigl[\mathrm{encode}(x) = \mathrm{encode}(x)\bigr] \in \{0, 1\},
$$

which is an exact test for nondeterminism in the nearest-centroid argmax. All of
these are averaged with per-batch weighting $\sum \text{value} \cdot B / \sum B$
during validation.

## 15. Codebook initialization by K-Means

Draw a uniform sample without replacement $\mathcal{S} \subset \mathcal{T}$ of size
$\min(500{,}000, |\mathcal{T}|)$, encode it under `no_grad` through the same
normalized map,

$$
\bar z_i = \text{\_normalize\_latent}\bigl(f_\theta(x_i)\bigr), \qquad i \in \mathcal{S} .
$$

Then process the stages in order. At stage $d$, fit

$$
\min_{\{c_k\},\, a} \ \sum_{i \in \mathcal{S}} \bigl\| r^{(d-1)}_i - c_{a(i)} \bigr\|_2^2
$$

by Lloyd's algorithm with $k$-means++ seeding, $n_{\text{init}} = 3$ restarts, and
$50$ iterations per restart (FAISS on GPU, sklearn on CPU as fallback), take the
winning centroids as $E_d$ and the induced labels $a$, then advance

$$
r^{(d)}_i = r^{(d-1)}_i - c_{a(i)} .
$$

Centroid magnitudes are kept as returned by K-Means. Nothing normalizes them, so
stage $d$ centroids are on the scale of the stage-$(d-1)$ residuals and shrink with
depth. Using the same normalized map as the training forward pass makes the K-Means
objective and the training objective describe the same geometry.

## 16. Dead-code detection and revival

Write $S_b = \texttt{BUFFER\_CAPACITY\_PER\_STAGE} = 20{,}000$ for the buffer
capacity, $\gamma = \texttt{usage\_decay} = 0.99$, and
$\theta = \texttt{dead\_code\_threshold} = 0.1$.

**Residual buffers.** Each stage keeps a ring buffer $B_d \in \mathbb{R}^{S_b \times d_h}$
that stores the stage-$(d-1)$ residuals actually presented to it. Entries are
overwritten in circular order, so $B_d$ holds the most recent $S_b$ residuals for that
stage, and a wrap flag records whether the buffer is full.

**EMA usage.** For each stage a usage vector is maintained across steps $t$,

$$
u^{(d)}_k \leftarrow \gamma \, u^{(d)}_k + (1 - \gamma) \, \mathrm{cnt}^{(t)}_k,
\qquad
u^{(d)}_k \big|_{t=0} = \mathrm{cnt}^{(0)}_k .
$$

Because $\mathrm{cnt}$ is a raw per-batch count, the units of $u$ are counted
assignments, exponentially weighted.

**Revival.** Every `dead_code_fix_interval` steps, and only while
$t < 0.6 \, T_{\text{tot}}$, a code $k$ with $u^{(d)}_k < \theta$ is treated as dead.
Let

$$
V_d =
\begin{cases}
S_b, & \text{buffer has wrapped} \\
\text{write pointer}, & \text{otherwise}
\end{cases}
$$

be the number of valid slots. For each dead index a uniform draw
$j \sim \mathrm{Unif}\{0, \dots, V_d - 1\}$ is made and

$$
e_{d,k} \leftarrow B_d[j],
\qquad
u^{(d)}_k \leftarrow \max\!\left(1,\ \frac{1}{K} \sum_{l} u^{(d)}_l\right),
$$

and the Adam moments for that centroid row are zeroed,

$$
m_{d,k} \leftarrow 0, \qquad v_{d,k} \leftarrow 0 .
$$

The centroid is therefore moved onto a real residual sample of the current
distribution, and its optimizer state is cleared so the move is not immediately
undone by stale momentum. The $0.6 \, T_{\text{tot}}$ cutoff freezes the codebook
once the encoder has largely converged, so late revival cannot keep perturbing a
settled assignment.

## 17. Optimization

**Parameter groups.** AdamW with two groups. The encoder and decoder use
$(\eta_0,\ \lambda_{\text{wd}})$ with $\eta_0 = \texttt{lr} = 5 \times 10^{-4}$ and
$\lambda_{\text{wd}} = \texttt{WEIGHT\_DECAY} = 10^{-4}$. The codebooks use
$(\alpha \eta_0,\ 0)$ with $\alpha = \texttt{CODEBOOK\_LR\_MULT} = 2.0$ and no weight
decay. Two reasons to treat them differently. Each centroid sees only its own
cluster's share of the batch gradient, roughly $1/K$ of the rows, while the encoder
receives a dense gradient from every stage at once, so an equal step size lets the
encoder outrun the codebooks. And weight decay on a centroid is decay toward the
origin, which competes with K-Means.

**Schedule.** With $W = \texttt{warmup\_steps} = 2000$, $T$ the total steps
(`max_steps` if set, otherwise $\texttt{epochs} \times \texttt{steps\_per\_epoch}$),
and $\rho = \texttt{MIN\_LR\_RATIO} = 0.01$,

$$
\eta(t) = \eta_0 \cdot
\begin{cases}
\dfrac{t}{W}, & t < W, \\[2.5mm]
\max\!\Bigl(\rho,\ \tfrac12\bigl(1 + \cos(\pi p)\bigr)\Bigr), & t \ge W,
\end{cases}
\qquad
p = \mathrm{clip}\!\left(\frac{t - W}{T - W},\, 0,\, 1\right),
$$

and the codebook group tracks $\alpha \eta(t)$ because the group multiplier is fixed
and the scheduler scales both groups together. The warmup term starts at $0$ when
$t = 0$, unlike the Stage-1 schedule in Part I which used $(t+1)/W$.

**Gradient clipping.** The norm is clipped separately per module rather than
globally,

$$
\nabla \leftarrow \nabla \cdot \min\!\left(1, \frac{1}{\|\nabla\|_2 + \epsilon}\right)
\quad \text{applied to } \theta,\ \{E_d\},\ \phi \ \text{independently},
$$

which bounds each module's gradient at $2$-norm $1$ instead of bounding their sum.

**Autoencoder warmup (stage 0).** If `warmup_ae_epochs` is positive and no
checkpoint is resumed, the encoder and decoder alone are trained on

$$
\mathcal{L}_{\text{ae}} = \frac{1}{B d_x}
\bigl\| g_\phi(\text{\_normalize\_latent}(f_\theta(x))) - x \bigr\|_2^2
$$

for that many epochs, reusing the main optimizer so its Adam moments survive into
the quantized phase. The codebooks compute no gradient in this stage and are
unchanged. This pre-conditions the latent geometry that K-Means is about to fit.

**Mixed precision.** `bf16` by default (no loss scaling needed), `fp16` with a
`GradScaler`, `fp32` on CPU. TF32 is enabled for the matmuls.

## 18. Inference, SID construction, collisions

Encoder inference repeats the cascade with the same affinities but subtracts the true
centroid rather than a stopped-gradient one, which is the same arithmetic because no
gradient is needed:

$$
k_d = \arg\max_k \bigl( 2\langle r^{(d-1)}, e_{d,k} \rangle - \|e_{d,k}\|_2^2 \bigr),
\qquad
r^{(d)} = r^{(d-1)} - e_{d,k_d} .
$$

If `save_best` is set and a best checkpoint exists, the state is swapped to that
checkpoint before this pass, so the exported IDs come from the model that minimized
validation loss rather than the last epoch.

Each item then gets a tuple $k(x) = (k_1, \dots, k_M)$. When `add_unique_id` is set,
items are grouped by tuple and each group is ranked,

$$
\mathcal{G}(\kappa) = \{i : k(x_i) = \kappa\},
\qquad
\text{rank}(i) = \text{position of } i \text{ within } \mathcal{G}(\kappa),
$$

and the appended suffix is

$$
s(i) =
\begin{cases}
\text{rank}(i), & \text{rank}(i) \le \beta_u \ \ \text{or \texttt{clip-suffix} is off}, \\
\beta_u - 1, & \text{otherwise},
\end{cases}
\qquad
\beta_u = \texttt{unique\_id\_bound}\ \text{or}\ K .
$$

The final ID is
$k(x_i) \mathbin{\|} [s(i)] \in \{1, \dots, K\}^M \times \{0, \dots, \beta_u - 1\}$,
which is unique because two items in different groups differ in the prefix and two
items in the same group differ in the suffix. Clipping makes the suffix a bounded
integer at the cost of merging the overflow group into one ID, and the reported
overflow count is the number of groups with $|\mathcal{G}| > \beta_u$.

The production run never set `unique_id_bound`, so the clamp never fires and the
suffix is just the full $0$-based rank. The last level then grows to whatever the
largest group needs. For the 4,783,154-item catalog the biggest four-code group holds
731 items, so the fifth level has exactly 731 values, which is what `vocab.json`
records (`sid_sizes[4] = 731`). Setting a bound instead turns the suffix into a
clipped integer and merges every group larger than $\beta_u$ into a single ID. With
the default $\beta_u = K = 256$ that would have collapsed every group of more than 256
items onto one suffix, which is why leaving it unset is the safe choice when the group
sizes are not known in advance.

## 19. Inputs and outputs

**Input.** An HDF5 or Parquet file with an `item_id` column and an `embedding` column
of width $d_x$. Everything is loaded into host memory as a single float32 tensor,
then split into train and validation by the seeded permutation. The train loader
drops the last partial batch; the validation loader uses the larger inference batch
size and does not shuffle.

**Model output per batch.** A dictionary with the total loss, the reconstruction
loss, the aggregated $\mathcal{L}_{\text{rqvae}}$, its codebook and commitment parts,
the entropy term, the cluster index tensor of shape $(B, M)$, and the residual chain
for diagnostics.

**Final artifacts.** The model weights (`rqvae_model.pth`), a best validation
checkpoint and periodic checkpoints, TensorBoard scalars for every loss, per-stage
usage, perplexity, residual norms, validation SID uniqueness and corpus-level
utilization, and the semantic ID mapping

$$
\texttt{index\_rqvae.json} =
\Bigl\{\ \mathrm{item\_id}_i \mapsto
\bigl[k_1(x_i), \dots, k_M(x_i)\bigr] \ \text{or} \ \bigl[k_1, \dots, k_M, s\bigr]
\ \Bigr\}_{i=1}^{N}.
$$

## 20. Hyperparameters (defaults in the file)

| Symbol | Value | Argument |
|---|---|---|
| $d_x$ | $256$ | `--input_dim` (defaults to $1024$; must match the embedding width) |
| $d_h$ | $64$ | `--hidden_dim` (choices $32$, $64$) |
| $M$ | $4$ | `--num_codebooks` |
| $K$ | $256$ | `--codebook_size` |
| $\beta$ | $0.1$ | `--beta` |
| $\tau$ | $1.0$ | `--temperature` |
| $\lambda_{\text{ent}}$ | $0.002$ | `--entropy_loss_weight` |
| $w_d$ | $[2.0, 1.5, 1.0, 1.0]$ | `--stage_weights` |
| $J$ | $0.0$ | `--jitter` |
| $B$ | $4096$ | `--batch_size` |
| $B_{\text{inf}}$ | $16384$ | `--inference_batch_size` |
| $\eta_0$ | $5 \times 10^{-4}$ | `--lr` |
| $\alpha$ | $2.0$ | `CODEBOOK_LR_MULT` |
| $W$ | $2000$ | `--warmup_steps` |
| $\rho$ | $0.01$ | `MIN_LR_RATIO` |
| $\lambda_{\text{wd}}$ | $10^{-4}$ | `WEIGHT_DECAY` |
| $\gamma$ | $0.99$ | `--usage_decay` |
| $\theta$ | $0.1$ | `--dead_code_threshold` |
| $\rho$ (val) | $0.1$ | `--val_ratio` |
| $E$ | $50$ | `--epochs` |

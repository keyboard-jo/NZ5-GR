# How the evaluation works

This document describes the benchmark, exactly, as the code implements it. It is meant
to be read by someone who has not seen the project: what is measured, how each arm's
ranking is computed, which conventions could inflate a number, and what stops them. Every
formula below is a transcription of a line of code, cited as `file:line`.

The guiding rule is that a number in this project has to be worse than the truth rather
than better. Where a convention is favourable to an arm, it is named as such in
[section 6](#6-ties-and-what-they-cost).

---

## 1. Notation

| symbol | meaning |
| --- | --- |
| $N$ | evaluation rows. $N = 98{,}304$ for the validation split |
| $r \in \{1, \dots, N\}$ | a row: one user history and one true next item |
| $t_r$ | the target item id of row $r$, an index into the catalogue |
| $\mathcal{C}$ | the catalogue, $|\mathcal{C}| = 4{,}783{,}154$ |
| $\rho_r$ | the rank the arm under test gives $t_r$, $1$-based |
| $k_{\max}$ | the widest reported cut-off, $k_{\max} = \max(\texttt{EVAL\_KS}) = 10{,}000$ |
| $\mathcal{B}(t)$ | the set of items sharing $t$'s SID tuple, the *bucket* of $t$ |
| $W$ | beam width, default $20$ (`train_tiger.py:135`) |

`EVAL_KS = (1, 5, 10, 100, 1000, 10000)` (`train_tiger.py:140`) and `MRR_K = 10`
(`train_tiger.py:141`). The DiGR evaluator defaults to `EVAL_KS = (1, 5, 10)`
(`train_digr.py:244`) and is widened per run with `--ks`; `eval_full_catalog.py` assigns
`T.EVAL_KS = ks` before calling it. The baselines derive $k_{\max}$ as `KMAX = max(EVAL_KS)`
(`baselines_digr.py:63`). The published runs used
`--ks 1,10,100,1000,10000`, so $k_{\max} = 10{,}000$ everywhere in the recorded tables.

## 2. The task and the evaluation set

Each row is a window of up to 20 of the user's past events, cut at a chosen anchor
point, terminated by a separator, and the prediction target is the next item the user
interacted with after that point. Every row has **exactly one** relevant item. That single
fact drives the whole metric design in [section 3](#3-the-metrics).

The evaluation set is the `val` split as written by preprocessing:

| quantity | value |
| --- | --- |
| rows | 98,304 |
| catalogue ranked against | 4,783,154 items |
| items appearing as a train or val target | 504,404 (10.55% of the catalogue) |
| items appearing anywhere in a history | 1,580,961 (33.05%) |
| items never clicked, so never a target | 3,202,193 (67.0%) |
| rows whose target already appears in that row's own history | 0.05% |

Every arm ranks the **whole catalogue**, including the 67% that can never be a target.
Those items cannot be hits, but they do occupy rank positions, so they push real hits
deeper. This is the standard full-catalogue convention and it is applied identically to
every arm; an earlier evaluation that ranked a 660,000-item candidate pool was retired
precisely because it flattered every arm by shortening the list.

## 3. The metrics

With one relevant item per row, *recall at $k$* and *hit rate at $k$* are the same
quantity. The code reports both names (`acc@1`, `acc@5`, and `recall@k`) and they agree by
construction.

### 3.1 Recall at $k$

$$
\mathrm{R@}k \;=\; \frac{1}{N} \sum_{r=1}^{N} \mathbb{1}\!\left[\rho_r \le k\right]
$$

`train_digr.py:1107-1108`, `train_tiger.py:500-501`, `baselines_digr.py:665-667`.

### 3.2 nDCG at $k$

Because there is exactly one relevant item, the ideal DCG at every cut-off is the gain of a
single item at rank 1:

$$
\mathrm{IDCG@}k \;=\; \frac{1}{\log_2(1+1)} \;=\; 1
\qquad\Longrightarrow\qquad
\mathrm{nDCG@}k \;=\; \frac{1}{N} \sum_{r=1}^{N} \frac{\mathbb{1}\!\left[\rho_r \le k\right]}{\log_2(\rho_r + 1)}
$$

`train_digr.py:1109`, `train_tiger.py:502-505`, `baselines_digr.py:668`.

Read it as a *weighted* recall: $\mathrm{R@}k$ weights every hit inside the top $k$
identically, while $\mathrm{nDCG@}k$ weights a hit at rank $\rho$ by
$1/\log_2(\rho+1)$. The two therefore differ whenever hits are not spread evenly over
$1..k$: $\mathrm{nDCG@}k \le \mathrm{R@}k$ always, with equality only if every hit is at
rank 1.

The ratio is a useful diagnostic, because it recovers the mean position of the hits:

$$
\frac{\mathrm{nDCG@}k}{\mathrm{R@}k} \;=\; \overline{\,1/\log_2(\rho+1) \;\big|\; \rho \le k\,}
$$

**Check.** $\mathrm{nDCG@1} = \mathrm{R@1}$ must hold exactly, since
$1/\log_2(1+1) = 1$. Every run is expected to satisfy it; a printed pair that does not is a
bug, not a rounding difference.

### 3.3 MRR, truncated at 10

$$
\mathrm{MRR} \;=\; \frac{1}{N} \sum_{r=1}^{N} \frac{\mathbb{1}[\rho_r \le 10]}{\max(\rho_r, 1)}
$$

The cut-off is fixed at `MRR_K = 10` and is **deliberately not tied to `EVAL_KS`**
(`train_digr.py:244-245`). A run that widens `EVAL_KS` to report `R@1000` still reports
MRR over the top 10 only. Otherwise a rank-500 hit would contribute $1/500$ and a wide
recall curve would quietly inflate MRR. This was a real bug; the guard is the comment.

## 4. The miss convention

A target outside the arm's list is assigned

$$
\rho_r \;=\; k_{\max} + 1 \;=\; 10{,}001
$$

never $0$ (`train_digr.py:1076-1080`, `baselines_digr.py:661`). The reason is that
$0$ satisfies every test of the form $\rho_r \le k$, so a miss would be counted as a hit at
every cut-off, and $1/\log_2(\rho+1)$ would divide by zero. An earlier DiGR run reported
infinite nDCG from exactly this. The sentinel values are visible in the rank dumps: the
maximum stored rank is `10001`.

## 5. How each arm's ranking is computed

All four arms rank the same rows in the same order against the same catalogue. They differ
in how $\rho_r$ is produced, and those differences matter for reading the tables.

### 5.1 DiGR: an exact truncated sort

One vector per catalogue item, precomputed once:

$$
V_i \;=\; \frac{1}{L}\sum_{l=1}^{L} E_l\!\left[c_{il}\right]
\qquad
s_i \;=\; \frac{1}{\tau}\, h_r^{\top} V_i
$$

with $L = 2$ levels, $E_l$ the level-$l$ code-embedding matrix, $c_{il}$ the $l$-th SID code
of item $i$, $h_r$ the separator state for row $r$, and $\tau$ the temperature
(`train_digr.py:515-536` for the precompute, `:455` for the full-catalogue score). The model that is *trained* uses a sampled softmax; the
model that is *evaluated* is scored against all of $\mathcal{C}$:

```python
logits = model.item_logits(h)                        # [B, |C|], every item
topk = logits.topk(min(kmax, logits.shape[1]), 1).indices
match = topk == targets.unsqueeze(1)
rank = where(match.any(1), match.int().argmax(1) + 1, kmax + 1)
```

`retrieve()` (`train_digr.py:537`) does the same thing in chunks so peak memory stays
bounded, and takes a global top-$k$ over the concatenated chunks, so chunking changes
nothing about the result.

$\rho_r$ is therefore a **clean prefix of an exact total order over all 4,783,154 items.**
There is no approximation, no candidate pool, no backfill. The evaluation loss is likewise
the exact full-softmax NLL, not the sampled estimate, so losses are comparable across arms.

### 5.2 TIGER prefix-free: rank by counting, not by sorting

The additive score of a prefix-free TIGER head is the sum of per-level log-probabilities:

$$
u_i \;=\; \sum_{l=1}^{L} \log p_l\!\left(c_{il} \mid h_r\right)
\qquad
\rho_r \;=\; 1 + \#\!\left\{\, i \in \mathcal{C} \;:\; u_i > u_{t_r} \,\right\}
$$

`train_tiger.py:512-537`, the sum at `:537`. Counting is used instead of a $[B, |\mathcal{C}|]$ top-$k$ because
it is exact and needs no giant tensor.

Note the strict inequality: **items tied with the target are not counted as higher**, so a
tied target is reported at the best rank it could hold. This is favourable to the arm
(see [section 6](#6-ties-and-what-they-cost)).

### 5.3 TIGER prefix-conditioned: the beam is the only path

When the head is conditioned on the path taken, the score of level $l$ depends on the codes
chosen at levels $1..l-1$, so the model cannot be scored for all items in one forward pass.
`--mode eval` refuses such a checkpoint (`train_tiger.py:1230-1240`) rather than silently
reporting the prefix-free score as if it were the conditioned one. The only ranking path is
`--mode beam`.

The beam keeps $W$ tuples per level:

$$
S(t_{1:l}) \;=\; \sum_{j=1}^{l} \log p_j\!\left(t_j \mid h_r, t_{1:j-1}\right)
\qquad
\text{keep} \;=\; \operatorname{top-}W_t \; S(t_{1:l})
$$

over codes that at least one catalogue item owns (`allowed = _index.children(prefix)`,
`train_tiger.py:1342`), so the beam can only ever emit tuples the catalogue contains
(`train_tiger.py:246-288`).

The finished beam is a sorted list of at most $W$ tuples, so

$$
\rho_r \;=\; \begin{cases}
j & \text{if } t_r = \text{the } j\text{-th tuple, } 1 \le j \le W \\
W + 1 & \text{if } t_r \text{ is not in any of the } W \text{ tuples}
\end{cases}
$$

This rank is a **tuple** rank, not an item rank, and a miss is stored as $W+1$ so the printed
$\mathrm{R@}k$ is already end to end and needs no separate ceiling multiplier.

### 5.4 The count baselines

*popularity* ranks by train-target frequency and ignores the query entirely: one global
order, identical for all $N$ rows, truncated to $k_{\max}$
(`baselines_digr.py:696`). Its $\rho_r$ is the target's position in that single list.

*co-click* is a two-regime list (`baselines_digr.py:575-620`). For a row, take the last 20
history items $h_1, \dots, h_{20}$ with $h_1$ the **most recent**
(`last_event_codes`, `baselines_digr.py:359-360`, docstring "most recent first"), look up
$h_t$'s co-click successors $(i, \mathrm{cnt})$, and accumulate

$$
\sigma_i \;=\; \sum_{t=1}^{20} \mathrm{cnt}(h_t \to i) \cdot \gamma^{\,t}, \qquad \gamma = 0.5
$$

then sort. Because only items reachable as a successor of something in the history get a
score, that scored set is typically a few dozen to a few hundred items, not $k_{\max}$. The
remaining slots are filled **in popularity order**, skipping anything already chosen. So
co-click's list is $[\text{behavioural candidates}] \mathbin{+\!\!+} [\text{popularity}]$, and
its rank beyond the candidate set is a no-query ranking. It has no item representation at
all, only a sparse count table.

*sid_transition* uses smoothed log code frequencies plus a popularity tie-break of
magnitude `TIE_EPS = 1e-6` (`baselines_digr.py:496-497`). It is not in the recorded tables.

## 6. Ties, and what they cost

There are three distinct places where ties enter. Only one of them is generous to an arm,
and it is stated here rather than buried.

**(a) TIGER prefix-free ties are resolved in the arm's favour.** Section 5.2 counts only
strictly higher scores, so if 300 items share the target's tuple, the target is reported at
the best of those 300 positions instead of the average. Any alternative (average rank,
worst rank) would report a worse number. The measured tie structure of the 2x8192 SID:

| quantity | value |
| --- | --- |
| items sharing a 2-code tuple with at least one other item | 2,559,616 (53.51%) |
| largest tie group | 315 items |
| implied $\mathrm{R@1}$ ceiling if ranks were assigned per tuple | 53.57% |
| implied $\mathrm{R@10}$ ceiling | 93.20% |

At the observed 3.66% these ceilings are not binding, but they are the reason a
tuple-level number can never be read as an item-level one.

**(b) The beam's rank is a bucket rank.** All items in a bucket share a position. That is
the method's own resolution limit, not an evaluation choice, and it is why the beam arm is
reported against a budget rather than against $\mathrm{R@10}$.

**(c) The count baselines have a deterministic tie-break.** co-click accumulates into
`np.unique`-sorted candidate ids and then does a stable sort, so equal totals are ordered by
ascending item id; `sid_transition` adds the popularity epsilon explicitly. No arm gets a
free tie.

## 7. `cover@k` is not recall

The training log prints `cover@k`. It is

$$
\mathrm{cover@}k \;=\; \hat{\Pr}\!\left[\, t_{r,1} \in \operatorname{top-}k \text{ first codes} \,\right]
$$

the share of rows whose **first** code the model placed in its own top $k$
(`train_tiger.py:566`, accumulated at `:592-593`, divided at `:612-613`; `BEAM_COVER_KS` at `:544`). Because one first code averages
$4{,}783{,}154 / 8{,}192 \approx 584$ items, `cover@1000 = 91.55%` is reach over roughly
**584,000 items**, not 1,000. It must never be set beside a $\mathrm{R@}k$ from a
full-catalogue arm. It is not a bound on a cut-off wider than the beam, but `cover@W` *is* a
hard ceiling on a beam of width $W$, because the beam can only ever emit items whose first
code the model put in its own top $W$
(`level-0 cover@20: 42578/98304 rows (43.31%), which caps targets decoded`).

## 8. The matched budget

Comparing arms at a fixed $k$ silently gives each arm a differently sized shortlist: DiGR
at $k$ means $k$ items, while a beam of $W$ names the union of the items in $W$ buckets,
which averaged 192 items per row for $W = 20$. The matched-budget pass replaces the fixed
cut-off with a per-row budget.

### 8.1 The budget

$$
N_r \;=\; \sum_{j=1}^{W} \left|\mathcal{B}\!\left(t_{rj}\right)\right| \cdot \mathbb{1}\!\left[t_{rj} \neq \varnothing\right]
$$

where $t_{rj}$ is the $j$-th tuple the beam returned for row $r$ and $\varnothing$ marks the
$-1$ padding used when a beam returns fewer than $W$ tuples
(`budget_match.py:422`, with the padding mask `present = tuples[:, :, 0] >= 0` at `:360`). A **second, tighter** view uses only items that appear as a
train or val target:

$$
N_r^{\text{seen}} \;=\; \sum_{j=1}^{W} \left|\mathcal{B}(t_{rj}) \cap \mathcal{T}\right| \cdot \mathbb{1}\!\left[t_{rj} \neq \varnothing\right]
$$

with $\mathcal{T}$ the target set (10.55% of the catalogue), or the whole clicked set
(33.05%) when `--seen_items` is passed. The two views are reported in separate columns and
labelled in the header, because they are not interchangeable.

**The measured distribution.** $N_r$ is not a fixed budget with noise around it. Over the
98,304 validation rows it has mean 192.17, standard deviation 105.76, median 171, interquartile
range 118 to 240, minimum 25, maximum 1,439, and skew $+1.65$. The coefficient of variation is
$0.55$, so the budget varies by about half its own size, and the mean sits 21 items above the
median because of the right tail. Concentration is mild, a Gini of $0.288$ with the fattest
decile of rows holding $22.0\%$ of all named items, so no small group of rows dominates the
figure. The full percentile table is in `README.md` section 1.2.

Two consequences for how the column should be read.

1. **A single $k$ cannot stand in for it.** A fixed $k = 192$ overstates the budget on the
   half of rows that name fewer than 171 items and understates it on the tail that reaches
   1,439. That is the entire reason the per-row construction exists, rather than a matter of
   tidiness.
2. **$N_r$ is correlated with row difficulty, by construction.** Decode rate climbs with the
   budget, from $11.39\%$ in the thinnest $N_r$ quartile to $14.55\%$ in the fattest, and
   decoded rows average 204.1 named items against 190.5 for the rest. Fat tuples are the
   popular ones, so a longer list goes with an easier row. The comparison stays fair because
   both arms are given the same $N_r$ on the same row, but `match_all` is implicitly
   popularity-weighted relative to a fixed-$k$ table, and quoting it without that caveat would
   flatter whichever arm is stronger on popular rows.

### 8.2 The columns

$$
\texttt{match\_all} \;=\; \frac{1}{N}\sum_{r} \mathbb{1}\!\left[\rho_r \le N_r\right]
\qquad
\texttt{match\_seen} \;=\; \frac{1}{N}\sum_{r} \mathbb{1}\!\left[\rho_r \le N_r^{\text{seen}}\right]
$$

$$
\texttt{match\_kmed} \;=\; \frac{1}{N}\sum_{r} \mathbb{1}\!\left[\rho_r \le \operatorname{median}_r N_r\right]
$$

The rank-aware twins replace the cut-off the same way, or not at all:

$$
\texttt{ndcg@N\_r} \;=\; \frac{1}{N}\sum_{r} \frac{\mathbb{1}[\rho_r \le N_r]}{\log_2(\rho_r+1)}
\qquad
\texttt{mrr10} \;=\; \frac{1}{N}\sum_{r} \frac{\mathbb{1}[\rho_r \le 10]}{\max(\rho_r,1)}
$$

nDCG needs the budget for the same reason recall does: which ranks exist depends on how long
the list is. MRR does not, because its cut-off is the fixed $k = 10$, so it carries the same
value in both columns. Both are **undefined for an arm whose ranks are tuple positions** and
`budget_match.py` writes `n/a` rather than a number, since a tuple position holding about
9.6 items is not a position on the item axis.

`match_all` is the primary column and the **only like-for-like one** in the project: every
arm is asked the same question, "was the target inside the first $N_r$ items you would have
returned". `match_kmed` gives every arm one flat budget, equal to the beam's median, which
is easier to state in prose and slightly harder on the arms whose shortlist is longer than
the median.

TIGER's own reach is reported as

$$
\texttt{tiger\_reach} \;=\; \frac{1}{N}\sum_r \mathbb{1}\!\left[\rho_r \le W\right]
$$

and **never** as $\rho_r \le N_r$. Since a miss is stored as $W+1$ and $N_r \ge 25$ on every
row, the test $\rho_r \le N_r$ would score every miss as a hit and print 100%
(`budget_match.py`, and `test_budget_match.py::test_tiger_reach_never_counts_a_miss_as_a_hit`
asserts the naive version returns 1.0).

## 9. What makes the numbers checkable

Every arm writes its per-row ranks to disk, each with the **split row index**, so the four
dumps are joined positionally and any misalignment is a hard failure rather than a plausible
number.

| dump | produced by | flag |
| --- | --- | --- |
| `rows`, `rank`, `targets`, `covered0`, `tuples` | `train_tiger.py --mode beam` | `--dump_rows` |
| `rows`, `targets`, `rank` | `eval_full_catalog.py` | `--dump_ranks` |
| `rows`, `targets`, `rank` per chunk | `baselines_digr.py` | `--dump_ranks` |
| the table | `budget_match.py` | reads all three |

Guards that refuse rather than warn:

| guard | what it prevents |
| --- | --- |
| `eval_full_catalog.py` recomputes every $\mathrm{R@}k$ from the dumped ranks before writing, and exits on a disagreement $> 10^{-9}$ | a dump that does not reproduce the metrics it came with |
| beam dump `sid_sizes` must equal `vocab.json`'s SID | counting buckets on a different SID |
| tuple shape must be `[N, W, L]`, codes must be catalogue-owned | a truncated or mismatched beam dump |
| every arm's `targets` array must equal the beam's, positionally | two runs that scored different rows in the same order |
| row indices must cover the split exactly once, all `rank != -1` | a partial or reordered dump |
| on every decoded row, the tuple at `rank-1` must equal the target's codes | a dump whose ranks and tuples disagree |
| own-cut-off drift against published values, $> 0.02$ pp warns, `--strict` fails | a silently different run |
| `--mode eval` refuses a prefix-conditioned checkpoint | reporting the prefix-free score as the conditioned one |
| `--beam_rows` defaults to 512 and takes the **first** rows, not a sample; a full run must pass 0 | a table built on 512 rows |

## 10. Traps, and the specific defence

| trap | why it inflates | defence |
| --- | --- | --- |
| rank $0$ for a miss | a miss passes every $\rho \le k$ test; nDCG becomes infinite | $\rho = k_{\max}+1$ (`train_digr.py:1076`) |
| $k_{\max}$ narrower than the widest reported $k$ | a target outside the computed list is scored as found | `kmax = max(EVAL_KS)` (`baselines_digr.py:63`) |
| MRR following a widened `EVAL_KS` | deep hits leak into an average meant for the top 10 | `MRR_K = 10`, independent (`train_digr.py:244`) |
| comparing `cover@k` with `R@k` | 584,000 items against 1,000 | section 7 |
| a shortlist reported as a cut-off | a 192-item shortlist read as `R@30` | the whole of section 8 |
| `rank <= N_r` applied to a beam | every miss counted as a hit, 100% | `rank <= W` |
| sampling the validation rows | a 512-row table read as 98,304 | `--beam_rows 0` on every recorded run |
| a candidate pool | a shorter list for every arm | full-catalogue ranking only |
| prefix-free score reported for a conditioned model | a rule the model was never trained on | refusal at `train_tiger.py:1230` |
| ties resolved at the average rank | a number no arm can beat or compare against | strict inequality, stated in section 6 |

## 11. What this benchmark does not measure

Stated plainly, because the tables are easy to over-read.

1. **One target, one position.** This is next-item retrieval, not a graded relevance
   ranking. Real recommendation has several valid next items; CTR-style users click many
   things the ground truth does not contain. Every arm is penalised for a plausible
   alternative, and the penalty is not uniform across methods.
2. **Validation only.** No test split is held out. The validation rows were used for model
   selection (2 epochs was chosen on them), so every reported number is mildly optimistic
   as an estimate of held-out performance.
3. **No significance test.** With $N = 98{,}304$ and $p \approx 0.13$, the binomial standard
   error is $\sqrt{p(1-p)/N} \approx 0.11$ pp, so a 95% half-width for a single proportion
   is about $0.21$ pp, and for a difference of two independent proportions the standard
   error is $\sqrt{2}$ times that, $\approx 0.16$ pp. No paired test is reported even
   though every arm is scored on the same rows and the arms agree on many of them, which
   makes the unpaired figures pessimistic. **Read differences under about 0.2 pp as
   unresolved.** As a calibration, the DiGR versus co-click h20 gap at $k = 192$ is
   0.32 pp (14.4989% vs 14.1825%), which is roughly two unpaired standard errors: it
   clears the bar without being comfortable, so the ordering is suggestive rather than
   settled until a paired test is run.
4. **Ranks are not comparable across arms at a fixed $k$.** A tuple rank, an item rank and
   a popularity-padded rank are different objects. Only $\texttt{match\_all}$,
   $\texttt{ndcg@N\_r}$ and $\texttt{mrr10}$ compare like with like, and the last two are
   defined only for an arm that ranks items.
5. **The 67% of the catalogue that is never clicked** occupies rank positions for every arm.
   That is the standard convention, but it means these numbers are not comparable with
   papers that rank a candidate pool of a few hundred items.
6. **Cost is not in the table.** DiGR scores 4.78M items per row; co-click reads a sparse
   table; the beam runs 20 tuples. The tables compare quality at a named list length, not
   quality per unit of compute.

## 12. Reproducing it

```bash
# 1. DiGR ranks over the full catalogue
sbatch "Model Training/slurm/eval_full_catalog.sh"          # ~1.8 min GPU on 98,304 rows
#   env: DATA_DIR, RUN_DIR, CKPT, KS=1,10,100,1000,10000
#        DUMP_RANKS=.../dumps/digr_val.npz

# 2. the count baselines, three co-click windows
sbatch "Model Training/slurm/run_baselines.sh"              # ~43 s CPU
#   env: DUMP_RANKS=.../dumps/baselines    -> baselines/h{5,10,20}/<kind>_*.npz

# 3. the beam, for a prefix-conditioned TIGER checkpoint
DUMP_ROWS="$PWD/dumps/beam_prefix_w20.npz" \
  sbatch "Plan 2/eval_tiger_beam.sh"                        # ~47 min GPU, beam 20

# 4. the matched-budget table
sbatch "Model Training/budget_match.sh"                     # seconds, cpu partition
```

Regenerate the per-`k` curves from the dumps without re-running any model:

```python
import glob, numpy as np
ranks = np.concatenate([np.load(f)["rank"] for f in
                        sorted(glob.glob("dumps/baselines/h20/coclick_*.npz"))])
print("R@192 =", (ranks <= 192).mean())
```

The 2026-10-29 run of this pipeline is recorded with its measured values in
[`docs/README.md`](README.md).

# Questions and answers

Short answers to the questions this project actually gets asked. Two or three paragraphs
each, results only, no method. Every number here is measured on the same 98,304 validation
rows over the same 4,783,154-item catalogue, and the method behind each one is in
[`evaluation.md`](evaluation.md).

---

## Q1. Which SID version is better, and why?

**v2, `2 x 8,192`.** On every structural test except one it is the healthier codebook, and the
one exception is a property that only matters because of how v1 is built.

The decisive number is how many items a single code leaves you with. v1 keeps 18,684 items in
its average first bucket, v2 keeps 584, and the DiGR paper's own catalogues leave 39 to 195. So
v1 is 96x too coarse at the first split and v2 is 3x. That is why v1 needs two codes before a
ranker has a list it can afford to read, while v2 is there after one, and it is why v1 spends
6 tokens per event against v2's 4.

One caveat on that band. The paper names RQ-VAE as its SID constructor but never states the
codebook width or the number of levels, so 39 to 195 is derived by dividing its three catalogue
sizes by 256, the width TIGER's RQ-VAE uses. The arithmetic fits all three catalogues exactly,
and the scale claim does not depend on it: their catalogues are 10k to 50k items against our
4,783,154, whatever width they used. See `sid.md` §6.

The second number is waste. v1's address length is 41.51 bits to name a catalogue that needs
22.19, so it carries 656,391 addresses per item, and its fifth level is a serial number that
holds only 18.5% of the entropy its width allows. v2 spends 26.00 bits, 14 addresses per item,
and both of its levels run at 98% of their entropy. v1's lift also peaks at depth 3 and falls
90% by depth 5, meaning its last two levels split density rather than behaviour, so DiGR v1 was
trained two levels past the point where the codes said anything new. v2 was still rising at its
last depth and simply ran out of codes.

What v1 has and v2 does not is injectivity: a five-code tuple names exactly one item, so nothing
ties. v2 puts 53.51% of the catalogue into shared tuples, and that costs a measured 1.22x at
`R@1` and 1.37x at `R@10`. It is a real cost, and it is smaller than the 1.57x the codebook
bought, so the verdict does not change. Full detail in [`sid.md`](sid.md).

---

## Q2. Which model is better on the v1 SID?

**TIGER with a prefix tree and beam search**, and it is not close.

| arm | `R@1` | `R@10` | `R@20` |
| --- | --- | --- | --- |
| TIGER, beam 20 | **1.7558%** | **7.4849%** | **10.0677%** |
| co-click, window 20 | 0.8057% | 3.7069% | 5.2684% |
| DiGR | 0.7924% | 2.5940% | 3.5553% |
| popularity | 0.4395% | 1.3438% | 1.9796% |

These are ordinary item-level numbers, because on this SID each tuple holds exactly one item,
so there is no tie ambiguity anywhere in the table. TIGER is 2.2x DiGR at `R@1` and 2.9x at
`R@10` on the same codes, the same features and the same rows.

The limits are equally clear. A beam of 20 can never name more than 20 items, so `R@20` is its
hard ceiling at 10.0677%, and it has nothing to say at `k = 100` or beyond, where DiGR reaches
7.51% and 49.77% at `k = 10,000`. It is also the most popularity-weighted arm in the file, at
4,276x from the coldest to the hottest popularity quintile, against 93x for co-click and 46x for
DiGR, so most of the win lives where the data is dense.

---

## Q3. Which model is better on the v2 SID?

**DiGR, at every cut-off wide enough to be a real question.** The two arms name very different
numbers of items: 20 tuples on v2 cover about 192 items because the beam concentrates on fat
cells, so the comparison has to be made at an equal budget.

Matched at 192 items, DiGR reaches **13.9994%** and TIGER's beam **12.5804%**, which is 1.11x
for DiGR. At a fixed `k = 192` the gap is the same order, 14.4989% against 12.5804%. Co-click
window 20 sits at 13.7034%, about two unpaired standard errors behind DiGR, so DiGR leads the
field but not by a comfortable margin.

Below that budget the question is not answerable. A beam of 20 asks "is the target in my 20",
while DiGR asks "is the target in my top 20 out of 4.78M", and TIGER's position-1 number on v2 is
a tuple rank whose item-level value lies somewhere in 0.49% to 2.04%, which overlaps DiGR's
1.05% to 1.29% entirely. On v1 that ambiguity disappears, which is why Q2 has a clean answer and
Q3 does not.

---

## Q4. Which model is better overall?

**There is no single winner, and the two arms fail on different rows.** On a matched budget with
the same vintage, TIGER wins on v1 by 2.83x and DiGR wins on v2 by 1.11x, so the honest summary
depends on which codebook you are willing to pay for.

For a full-catalogue ranking, which is what a production retriever needs, DiGR v2 is the best arm
in the project: 3.6570% at `R@10`, 27.6622% at `R@1,000` and 55.7058% at `R@10,000`, with a list
that is genuinely model-ranked at every depth. TIGER's beam is the sharpest thing at the head but
it is capped at 20 items and cannot answer the deep cut-offs at all.

The strongest argument that neither is close to done is the union. On v2, DiGR and co-click
together reach 22.3582% at 192 items, 54% above the better of them alone, and of the rows TIGER's
beam decodes 64.95% are also in DiGR's top 192. The arms are not competing for the same rows.

---

## Q5. Why does TIGER on v1 look so good when v1 is the unhealthy SID?

**Because the tree's advantage is larger than the codebook's disadvantage, and the two multiply.**

`R@10` decomposes exactly on the same rows: TIGER is **2.885x** DiGR on the five-level code with
the model being the only difference, the two-level code is **1.410x** DiGR on v1 with the codebook
being the only difference, and `2.885 / 1.410 = 2.047`, which is the observed 7.4849% against
3.6570%. So the factorisation gain outweighs the codebook gain rather than the two both helping.

The mechanism is conditional probability. DiGR computes `s_i = (1/L) sum_l h·E_l[c_il]`, which is
additive with fixed weights, so it cannot say "given this first code, the second one should count
for more". TIGER computes `p(c_l | h, c_<l)`, which can. That advantage grows with depth, and it
is measurable: at two levels the beam **loses** to DiGR, 0.87x, because there is only one
conditioning step and the level-0 code is already informative. The claim is therefore "the tree
wins where the codebook is deep", not "the tree wins".

Two things stop this from being a fair headline. The beam's `R@10` is capped at its own reach of
10.0677%, and it achieved 74% of that, while DiGR is at 3.66% of a much higher ceiling; and v1
changes depth, width and injectivity in one step, so depth is implicated rather than isolated.
Injectivity in particular is not a performance advantage, it is why the number needs no bracket.

---

## Q6. What is the cost of collisions?

**1.37x at `R@10`, 1.22x at `R@1`, and 1.0450 nats of unreachable loss.**

Measured by scoring one DiGR v2 checkpoint twice over the same rows, once across all 4,783,154
items and once across the 3,002,708 distinct tuples, so the two columns differ only in how many
positions a fat tuple occupies. The item column reproduces the published ranks on 100.00000% of
rows, so the left side is the leaderboard measure itself:

| | measured | same scores, one item per tuple | gain |
| --- | --- | --- | --- |
| `R@1` | 1.0549% | 1.2889% | 1.22x |
| `R@10` | 3.6570% | 5.0283% | 1.37x |
| `R@100` | 10.9182% | 14.4185% | 1.32x |
| `R@1,000` | 27.6622% | 33.9783% | 1.23x |

So about 27% of what these scores can already reach at `R@10` is lost to counting and 73% is the
model's own ranking. The cost is not the 9.3x implied by comparing unique-tuple rows against
shared-tuple rows, because that comparison confounds "no ties" with "those rows are easier"; the
controlled number is 1.37x.

In the loss there is a separate floor. `g` items in one tuple share a score, so the target can
receive at most `1/g` of the group's probability and its loss cannot fall below `log g`. Averaged
over the validation targets that floor is **1.0450 nats** against a 4.3677-nat margin, so 23.9%
of what the model learned is uncollectable by construction. It also means cross-entropy cannot be
compared across SIDs: v2 and v1 look 0.23 nats apart and are 1.27 nats apart once each pays its
own tax.

---

## Q7. What did the DiGR ablation find?

**The codebook and the features are both worth something, but not to the same arm, and the two do
not add.**

The 2x2 crosses codebook (`4x256+731` against `2x8192`) with feature layout (`v1` against `v2`,
which adds an action token and 128 gap bins). At `R@10`:

| | v1 features | v2 features | feature effect |
| --- | --- | --- | --- |
| **v1 codebook** | 2.5940% | 2.3275% | **0.90x** |
| **v2 codebook** | 3.2725% | 3.6570% | **1.12x** |
| **codebook effect** | **1.26x** | **1.57x** | |

The reversal is the finding. The new features are worth 1.12x on the wide codebook and 0.90x on
the narrow one, both at `p < 1e-12` on a paired test, so there is an interaction of 0.2194 nats
and the upgrade as a whole is 1.4098x rather than the product of its parts taken separately. Any
claim of the form "the codebook is worth 1.57x" has to name the feature layout it was measured in.

Compute is not the explanation. v1 consumed 664.5M training tokens against v2's 307.0M, so the
arm that trained 2.17x longer lost by 1.41x, and the codebook gain is a lower bound. Sorting the
four cells by `R@10` gives 3.27, 3.66, 2.33, 2.59, which is no order at all. Full tables,
paired tests and strata breakdowns in [`ablation.md`](ablation.md).

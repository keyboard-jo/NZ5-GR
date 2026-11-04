# The ablation: what the 1.41x was made of

Four DiGR runs on the same 98,304 val rows, the same 4,783,154 item catalogue and the same
0.6B backbone, differing only in which SID the items are quantised into and which features
the rows carry. This file records the result and, more importantly, the two ways the obvious
reading of it is wrong.

Every number here is measured from the rank dumps of all four arms, produced in one pass of
`Dump Analysis/dump_metrics.py`. Nothing is interpolated. The conventions, including what a
miss is and how ties are charged, are in [`evaluation.md`](evaluation.md).

## 1. The 2x2

| cell | SID | features | epochs | R@1 | R@10 | R@100 | R@1000 | R@10000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `v1sid_v1feat` | `4x256+731` | v1: no action token, 64 gap bins | 3 | 0.7924% | 2.5940% | 7.5094% | 21.6003% | 49.7691% |
| `v1sid_v2feat` | `4x256+731` | v2: action token, 128 gap bins | 2 | 0.6958% | 2.3275% | 7.3059% | 21.5291% | **50.2177%** |
| `v2sid_v1feat` | `2x8192` | v1 | 2 | 0.9705% | 3.2725% | 9.8674% | 25.5239% | 53.1972% |
| `v2sid_v2feat` | `2x8192` | v2 | 2 | **1.0549%** | **3.6570%** | **10.9182%** | **27.6622%** | **55.7058%** |

Train tokens per epoch, which is the honest cost axis and not the epoch count:

| cell | tokens/epoch | epochs | total | head params |
| --- | --- | --- | --- | --- |
| `v2sid_v1feat` | 119,441,616 | 2 | 238.9M | 16,777,216 |
| `v2sid_v2feat` | 153,461,256 | 2 | 307.0M | 16,777,216 |
| `v1sid_v1feat` | 221,500,536 | 3 | 664.5M | 1,797,120 |
| `v1sid_v2feat` | 255,520,176 | 2 | 511.0M | 1,797,120 |

The epoch counts are not equal and cannot be made equal without retraining two archived runs.
That does not rescue anything here. The arm that trained **most** (`v1sid_v1feat`, 664.5M
tokens) is not the winner, and every conclusion below rests on a pair that differs in exactly
one axis at exactly 2 epochs.

## 2. Every difference is real

Paired McNemar on the same 98,304 rows at `k = 10`, one degree of freedom, continuity
corrected. `b` counts rows only the first arm reaches, `c` rows only the second.

| pair | axis | b | c | b − c | p |
| --- | --- | --- | --- | --- | --- |
| `v1sid_v1feat` vs `v1sid_v2feat` | features, v1 codebook | 790 | 528 | **−262** | 6.5e-13 |
| `v2sid_v1feat` vs `v2sid_v2feat` | features, v2 codebook | 901 | 1,279 | **+378** | 6.8e-16 |
| `v1sid_v1feat` vs `v2sid_v1feat` | codebook, v1 features | 1,396 | 2,063 | +667 | 1.0e-29 |
| `v1sid_v2feat` vs `v2sid_v2feat` | codebook, v2 features | 1,173 | 2,480 | +1,307 | 1.5e-103 |
| `v1sid_v1feat` vs `v2sid_v2feat` | both | 1,368 | 2,413 | +1,045 | 1.2e-64 |
| `v1sid_v2feat` vs `v2sid_v1feat` | cross | 1,203 | 2,132 | +929 | 4.2e-58 |

Two checks that the table is internally consistent rather than four unrelated arrays. `b − c`
divided by 98,304 reproduces the `R@10` gap to four decimals in every row, for example 667 /
98,304 = 0.6785 pp against 3.2725 − 2.5940 = 0.6785 pp. And the two clean pairs, which hold
the epoch count at 2, are significant on their own at `p = 7e-16` and `p = 2e-103`.

So the earlier objection that the 1.41x might be an artefact of DiGR v1's third epoch is
settled. The smallest of the six effects has `p < 1e-12`.

## 3. The two changes are not additive, and the feature step flips sign

Define the log-ratio effect of each axis, at `k = 10`:

```
a_SID(f)  = log R@10(v2 SID, f)  - log R@10(v1 SID, f)      features settings f
a_feat(s) = log R@10(s, v2 feat) - log R@10(s, v1 feat)     SID settings s
```

| quantity | at the other setting | log-ratio | as a multiplier |
| --- | --- | --- | --- |
| `a_SID` | v1 features | 0.2324 | **1.2616x** |
| `a_SID` | v2 features | 0.4518 | **1.5712x** |
| `a_feat` | v1 codebook | −0.1084 | **0.8973x** |
| `a_feat` | v2 codebook | 0.1111 | **1.1175x** |

The interaction is the difference between the two readings of the same axis,
`0.4518 − 0.2324 = 0.2194` nats, equivalently `log(1.5712 / 1.2616)`. It is not small. As a
multiplier, `1.2616 x 1.1175 = 1.4098`, which is the observed full upgrade
`3.6570 / 2.5940 = 1.4098` exactly, so the 2x2 is multiplicative by construction and the
additive reading is what makes it look like "the codebook is 1.57x and the features are
1.12x".

**The feature step is a 1.12x win on the wide codebook and a 0.90x loss on the narrow one,
both beyond chance.** The action token and the finer gap grid are not a fixed improvement
that happens to be smaller elsewhere. They reverse. A summary that quotes one multiplier per
axis is picking the favourable column, and this file exists because an earlier draft of
`tasks.md` proposed exactly that ("the 2x2 gives the split without needing an interaction
term"). It does not. There is an interaction and it dominates the feature axis.

The current shipped configuration is still correct, since `v2sid_v2feat` wins every cut-off
and beats `v2sid_v1feat` at `p = 7e-16`. It is justified jointly, not by two independent wins.

## 4. Where each effect lands: the codebook buys the head, the features buy the tail

`R@10` by target popularity, quintiles over `log1p(train target count)` across the 98,304 val
targets. "Tail" means the val target never appears as a training target at all, which is not
the same as a new item, since it may still appear in training histories as an exposure.

| arm | tail (freq 0) | lower (2.2) | middle (9.6) | upper (36.5) | head (449) |
| --- | --- | --- | --- | --- | --- |
| rows | 15,715 | 21,192 | 21,459 | 20,146 | 19,792 |
| `v1sid_v1feat` | 0.1782% | 0.3964% | 0.6477% | 1.6083% | 9.9788% |
| `v1sid_v2feat` | 0.1973% | 0.3586% | 0.5732% | 1.2558% | 9.1198% |
| `v2sid_v1feat` | **0.1400%** | 0.2265% | 0.4846% | 1.8912% | 13.4499% |
| `v2sid_v2feat` | 0.1973% | 0.2926% | 0.6710% | 2.6159% | 14.3038% |

Reading the codebook axis at fixed features:

| comparison | tail | lower | middle | upper | head |
| --- | --- | --- | --- | --- | --- |
| `4x256+731` to `2x8192`, v1 features | **0.79x** | 0.57x | 0.75x | 1.18x | 1.35x |
| `4x256+731` to `2x8192`, v2 features | 1.00x | 0.82x | 1.17x | 2.08x | 1.57x |
| full upgrade, `v1sid_v1feat` to `v2sid_v2feat` | 1.11x | 0.74x | 1.04x | 1.63x | 1.43x |

**The wide codebook alone loses 21% on targets that were never a training target and gains
35% on the hottest quintile.** The full upgrade is 1.11x on the tail against 1.43x on the
head. So the headline 1.41x is a popular-item gain, and it must not be presented as a
cold-start result.

The mechanism is the one measured in `sid.md`. `2x8192` gives 1.59 items per tuple against
1.00 for five levels, and the items that share tuple space are the popular ones, so more codes
buys precision exactly where the buckets were crowded. Five levels are injective, so there is
nothing there to win, and the wider codebook's extra registers are spent on items that
already had their own code. That is why the gain is concentrated in the head and why the tail
does not move, or moves backwards.

This matters for the project's framing. Cold reach is the stated reason the representation
stays content-only, and this ablation says the SID change did not buy any.

## 5. What did not move

`nDCG@10 / R@10` is the mean weight of a hit, so it says where inside the top ten the hits
sit. Across the four cells it is 0.6053, 0.6008, 0.6023, 0.5988, a range of 0.0065.

| arm | nDCG@10 / R@10 | effective rank | misses |
| --- | --- | --- | --- |
| `v1sid_v1feat` | 0.6053 | 823.44 | 50.2309% |
| `v1sid_v2feat` | 0.6008 | 856.65 | 49.7823% |
| `v2sid_v1feat` | 0.6023 | 653.94 | 46.8028% |
| `v2sid_v2feat` | 0.5988 | 602.79 | 44.2942% |

Both changes move rows between hit and miss without reordering anything inside the top ten.
Effective rank, `2 ** mean(log2(rank))` over hits only, separates the two axes' magnitudes
from the other side: the codebook compresses hit depth by roughly 200 positions, the feature
step by roughly 30, and on the narrow codebook the feature step moves it the wrong way.

## 6. One exception at the bottom of the list

`v1sid_v2feat` beats `v1sid_v1feat` at `R@10000`, 50.2177% against 49.7691%, and has fewer
misses, 49.78% against 50.23%. So on the narrow codebook the extra features buy a little
broad reach while losing shallow precision. The two curves cross near `k = 1,900`. This is
the same crossover the earlier write-up recorded, not a new effect, and it is the only
cut-off where the "worse" arm wins.

## 7. How redundant the four cells are

Pair union at `k = 10`, the share of rows some arm in the pair reaches.

| pair | A only | B only | union | vs best single |
| --- | --- | --- | --- | --- |
| `v1sid_v1feat` vs `v2sid_v2feat` | 1.3916% | 2.4546% | **5.0486%** | 1.38x |
| `v1sid_v1feat` vs `v2sid_v1feat` | 1.4201% | 2.0986% | 4.6926% | 1.43x over 3.2725% |
| `v1sid_v2feat` vs `v2sid_v2feat` | 1.1932% | 2.5228% | 4.8503% | 1.33x |
| `v2sid_v1feat` vs `v2sid_v2feat` | 0.9165% | 1.3011% | 4.5736% | 1.25x |
| `v1sid_v2feat` vs `v2sid_v1feat` | 1.2238% | 2.1688% | 4.4963% | 1.37x |
| `v1sid_v1feat` vs `v1sid_v2feat` | 0.8036% | 0.5371% | 3.1311% | 1.21x |

Compare DiGR with co-click at a matched 192 items, where the union was 22.36% against DiGR's
14.50%, or 1.54x. Four cells of one architecture with two knobs are far more redundant with
each other than DiGR is with a different family. The codebook axis adds more union than the
feature axis, which is the same ordering as the effect sizes, so nothing here suggests the two
axes fail on qualitatively different rows.

## 8. What this changes in the write-up

1. **Quote a range, not a multiplier, per axis.** Codebook 1.26x to 1.57x depending on the
   feature setting, features 0.90x to 1.12x depending on the codebook, interaction 0.22 nats.
2. **Do not call 1.41x a cold-start result.** It is 1.43x on the hottest popularity quintile
   and 1.11x on targets that never appear in train, and the codebook half of it is 0.79x on
   that tail.
3. **The epoch asymmetry is not a live objection.** Say the token counts, note that the
   most-trained arm lost, and cite the two 2-epoch pairs.
4. **The two changes are jointly, not independently, validated.** The shipped configuration
   is right; the attribution is two-way.

## 9. What it does not establish

- Nothing about TIGER. This is DiGR's factorisation on four token streams, so the 1.41x says
  what a codebook and a feature set are worth to a linear item scorer. TIGER's side has since
  been run: task 3 in [`tasks.md`](tasks.md) puts it 2.83x ahead of DiGR v1 at the same budget
  on the same codes (`README.md` §1.10), which is a depth result and not a re-run of this one.
- Nothing about a wider codebook. The axis tested is `4x256+731` against `2x8192`, not
  `2x8192` against `2x16384`, so the curve's shape beyond 8,192 is unmeasured.
- Nothing about why the feature step reverses. The popularity strata localise it, they do
  not explain it. The untested candidate is that 128 gap bins put half as many events in
  each bin as 64 did, and the narrow codebook leaves less capacity to absorb that.

## 10. Reproduction

```bash
python "Dump Analysis/dump_metrics.py" \
  --arm "v1sid_v1feat=dumps/digr_v1.npz" \
  --arm "v1sid_v2feat=dumps/digr_abl_sidv1_featv2_e2.npz" \
  --arm "v2sid_v1feat=dumps/digr_abl_sidv2_featv1_e2.npz" \
  --arm "v2sid_v2feat=dumps/digr_val.npz" \
  --data_dir ~/NZ5/plan2/digr_data \
  --ks 1,10,100,1000,10000 --out_dir analysis_ablation
```

The two mixed arms come from `ablationDiGR/`, which also records the preprocessing flags, the
temperature derivation and the input fingerprints. The two corner cells are the archived v1
run and DiGR v2.

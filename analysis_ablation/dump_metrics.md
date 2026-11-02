# Dump analysis

98,304 rows, 4 arm(s), same row set and same targets for every arm.

| arm | cap | misses | effective rank |
|---|---|---|---|
| v1sid_v1feat | 10,000 | 50.2309% | 823.44 |
| v1sid_v2feat | 10,000 | 49.7823% | 856.65 |
| v2sid_v1feat | 10,000 | 46.8028% | 653.94 |
| v2sid_v2feat | 10,000 | 44.2942% | 602.79 |

`cap` is the deepest position the dump can express. A miss is stored at `cap + 1`,
so `effective rank` is computed over hits only and is not flattered by sentinels.

## The curve

| arm | R@1 | R@10 | R@100 | R@1000 | R@10000 |
|---|---|---|---|---|---|
| v1sid_v1feat | 0.7924% | 2.5940% | 7.5094% | 21.6003% | 49.7691% |
| v1sid_v2feat | 0.6958% | 2.3275% | 7.3059% | 21.5291% | 50.2177% |
| v2sid_v1feat | 0.9705% | 3.2725% | 9.8674% | 25.5239% | 53.1972% |
| v2sid_v2feat | 1.0549% | 3.6570% | 10.9182% | 27.6622% | 55.7058% |

### nDCG (top of the list only)

| arm | nDCG@1 | nDCG@10 | nDCG@100 | nDCG@1000 | nDCG@10000 |
|---|---|---|---|---|---|
| v1sid_v1feat | 0.00792 | 0.01570 | 0.02516 | 0.04182 | 0.06592 |
| v1sid_v2feat | 0.00696 | 0.01398 | 0.02356 | 0.04035 | 0.06491 |
| v2sid_v1feat | 0.00970 | 0.01971 | 0.03248 | 0.05111 | 0.07486 |
| v2sid_v2feat | 0.01055 | 0.02190 | 0.03599 | 0.05594 | 0.08004 |

### MRR

| arm | MRR@1 | MRR@10 | MRR@100 | MRR@1000 | MRR@10000 |
|---|---|---|---|---|---|
| v1sid_v1feat | 0.00792 | 0.01258 | 0.01411 | 0.01457 | 0.01467 |
| v1sid_v2feat | 0.00696 | 0.01116 | 0.01270 | 0.01316 | 0.01326 |
| v2sid_v1feat | 0.00970 | 0.01574 | 0.01783 | 0.01836 | 0.01846 |
| v2sid_v2feat | 0.01055 | 0.01742 | 0.01973 | 0.02031 | 0.02041 |

`nDCG@k / R@k` is the mean weight of a hit, so it says where inside the top k
the hits land. 1.0 means every hit is at rank 1.

| arm | nDCG@10 / R@10 |
|---|---|
| v1sid_v1feat | 0.6053 |
| v1sid_v2feat | 0.6008 |
| v2sid_v1feat | 0.6023 |
| v2sid_v2feat | 0.5988 |

## Pair union at k = 10

The share of rows some arm reaches, which no single-arm number shows.

| pair | A only | B only | union | disagreement |
|---|---|---|---|---|
| v1sid_v1feat vs v1sid_v2feat | 0.8036% | 0.5371% | **3.1311%** | 1.3407% |
| v1sid_v1feat vs v2sid_v1feat | 1.4201% | 2.0986% | **4.6926%** | 3.5187% |
| v1sid_v1feat vs v2sid_v2feat | 1.3916% | 2.4546% | **5.0486%** | 3.8462% |
| v1sid_v2feat vs v2sid_v1feat | 1.2238% | 2.1688% | **4.4963%** | 3.3925% |
| v1sid_v2feat vs v2sid_v2feat | 1.1932% | 2.5228% | **4.8503%** | 3.7160% |
| v2sid_v1feat vs v2sid_v2feat | 0.9165% | 1.3011% | **4.5736%** | 2.2176% |

## Paired significance at k = 10 (McNemar)

Both arms see the same rows, so this is the right test. `p` is a chi-square
with one degree of freedom and a continuity correction.

| pair | A hit B miss | A miss B hit | chi2 | p |
|---|---|---|---|---|
| v1sid_v1feat vs v1sid_v2feat | 790 | 528 | 51.685 | 6.52e-13 |
| v1sid_v1feat vs v2sid_v1feat | 1,396 | 2,063 | 128.232 | 9.98e-30 |
| v1sid_v1feat vs v2sid_v2feat | 1,368 | 2,413 | 288.267 | 1.19e-64 |
| v1sid_v2feat vs v2sid_v1feat | 1,203 | 2,132 | 258.226 | 4.18e-58 |
| v1sid_v2feat vs v2sid_v2feat | 1,173 | 2,480 | 466.914 | 1.5e-103 |
| v2sid_v1feat vs v2sid_v2feat | 901 | 1,279 | 65.197 | 6.78e-16 |

## Recall@10 by target popularity

Quintiles over `log1p(train target count)`, ordered cold to hot. A content-only
arm should hold up toward the tail; a popularity rider collapses there.

| arm | tail | lower | middle | upper | head |
|---|---|---|---|---|---|
| rows | 15,715 | 21,192 | 21,459 | 20,146 | 19,792 |
| mean freq | 0.0 | 2.2 | 9.6 | 36.5 | 449.0 |
| v1sid_v1feat | 0.1782% | 0.3964% | 0.6477% | 1.6083% | 9.9788% |
| v1sid_v2feat | 0.1973% | 0.3586% | 0.5732% | 1.2558% | 9.1198% |
| v2sid_v1feat | 0.1400% | 0.2265% | 0.4846% | 1.8912% | 13.4499% |
| v2sid_v2feat | 0.1973% | 0.2926% | 0.6710% | 2.6159% | 14.3038% |

## Not reported, and why

- precision@k, MAP, HR@k: with one relevant item per row they are R@k relabelled.
- diversity and serendipity: they need item content, which this project excludes.
- per-row costs: a rank dump does not record how many items were scored.

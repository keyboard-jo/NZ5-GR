# Dump analysis

98,304 rows, 4 arm(s), same row set and same targets for every arm.

| arm | cap | misses | effective rank |
|---|---|---|---|
| tiger_v1_beam | 20 | 89.9323% | 4.69 |
| digr_v1 | 10,000 | 50.2309% | 823.44 |
| coclick_h20 | 10,000 | 53.1118% | 481.21 |
| popularity | 10,000 | 63.1785% | 990.23 |

`cap` is the deepest position the dump can express. A miss is stored at `cap + 1`,
so `effective rank` is computed over hits only and is not flattered by sentinels.

## The curve

| arm | R@1 | R@5 | R@10 | R@20 | R@100 | R@1000 | R@10000 |
|---|---|---|---|---|---|---|---|
| tiger_v1_beam | 1.7558% | 5.0741% | 7.4849% | 10.0677% | 10.0677% | 10.0677% | 10.0677% |
| digr_v1 | 0.7924% | 1.8514% | 2.5940% | 3.5553% | 7.5094% | 21.6003% | 49.7691% |
| coclick_h20 | 0.8057% | 2.4882% | 3.7069% | 5.2684% | 10.9873% | 25.3021% | 46.8882% |
| popularity | 0.4395% | 0.9074% | 1.3438% | 1.9796% | 4.7862% | 14.7288% | 36.8215% |

### nDCG (top of the list only)

| arm | nDCG@1 | nDCG@5 | nDCG@10 | nDCG@20 | nDCG@100 | nDCG@1000 | nDCG@10000 |
|---|---|---|---|---|---|---|---|
| tiger_v1_beam | 0.01756 | 0.03441 | 0.04217 | 0.04872 | 0.04872 | 0.04872 | 0.04872 |
| digr_v1 | 0.00792 | 0.01331 | 0.01570 | 0.01811 | 0.02516 | 0.04182 | 0.06592 |
| coclick_h20 | 0.00806 | 0.01653 | 0.02044 | 0.02437 | 0.03467 | 0.05176 | 0.07037 |
| popularity | 0.00439 | 0.00670 | 0.00809 | 0.00968 | 0.01469 | 0.02638 | 0.04524 |

### MRR

| arm | MRR@1 | MRR@5 | MRR@10 | MRR@20 | MRR@100 | MRR@1000 | MRR@10000 |
|---|---|---|---|---|---|---|---|
| tiger_v1_beam | 0.01756 | 0.02905 | 0.03223 | 0.03404 | 0.03404 | 0.03404 | 0.03404 |
| digr_v1 | 0.00792 | 0.01160 | 0.01258 | 0.01323 | 0.01411 | 0.01457 | 0.01467 |
| coclick_h20 | 0.00806 | 0.01380 | 0.01539 | 0.01646 | 0.01777 | 0.01827 | 0.01835 |
| popularity | 0.00439 | 0.00592 | 0.00648 | 0.00691 | 0.00753 | 0.00785 | 0.00793 |

`nDCG@k / R@k` is the mean weight of a hit, so it says where inside the top k
the hits land. 1.0 means every hit is at rank 1.

| arm | nDCG@10 / R@10 |
|---|---|
| tiger_v1_beam | 0.5633 |
| digr_v1 | 0.6053 |
| coclick_h20 | 0.5513 |
| popularity | 0.6019 |

## Pair union at k = 10

The share of rows some arm reaches, which no single-arm number shows.

| pair | A only | B only | union | disagreement |
|---|---|---|---|---|
| tiger_v1_beam vs digr_v1 | 5.4474% | 0.5564% | **8.0414%** | 6.0038% |
| tiger_v1_beam vs coclick_h20 | 5.7078% | 1.9297% | **9.4147%** | 7.6375% |
| tiger_v1_beam vs popularity | 6.5257% | 0.3845% | **7.8695%** | 6.9102% |
| digr_v1 vs coclick_h20 | 1.7721% | 2.8849% | **5.4789%** | 4.6570% |
| digr_v1 vs popularity | 2.0284% | 0.7782% | **3.3722%** | 2.8066% |
| coclick_h20 vs popularity | 3.1250% | 0.7619% | **4.4688%** | 3.8869% |

## Paired significance at k = 10 (McNemar)

Both arms see the same rows, so this is the right test. `p` is a chi-square
with one degree of freedom and a continuity correction.

| pair | A hit B miss | A miss B hit | chi2 | p |
|---|---|---|---|---|
| tiger_v1_beam vs digr_v1 | 5,355 | 547 | 3915.156 | 0 |
| tiger_v1_beam vs coclick_h20 | 5,611 | 1,897 | 1836.224 | 0 |
| tiger_v1_beam vs popularity | 6,415 | 378 | 5363.359 | 0 |
| digr_v1 vs coclick_h20 | 1,742 | 2,836 | 260.954 | 1.06e-58 |
| digr_v1 vs popularity | 1,994 | 765 | 546.569 | 7.02e-121 |
| coclick_h20 vs popularity | 3,072 | 749 | 1411.066 | 8.27e-309 |

## Recall@10 by target popularity

Quintiles over `log1p(train target count)`, ordered cold to hot. A content-only
arm should hold up toward the tail; a popularity rider collapses there.

| arm | tail | lower | middle | upper | head |
|---|---|---|---|---|---|
| rows | 15,715 | 21,192 | 21,459 | 20,146 | 19,792 |
| mean freq | 0.0 | 2.2 | 9.6 | 36.5 | 449.0 |
| tiger_v1_beam | 0.0064% | 0.0802% | 2.1296% | 7.2769% | 27.3696% |
| digr_v1 | 0.1782% | 0.3964% | 0.6477% | 1.6083% | 9.9788% |
| coclick_h20 | 0.1400% | 0.4719% | 1.4073% | 3.2165% | 12.9951% |
| popularity | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 6.6744% |

## Cells named by tiger_v1_beam

A beam ranks cells of the catalogue, not items. These are cell counts.

| quantity | value |
|---|---|
| beam width | 20 |
| levels | 5 |
| cells named per row | 20.0 (min 20, max 20) |
| distinct cells per row | 20.0 |
| distinct cells overall | 67,007 |
| mean pairwise Jaccard between two rows | 0.004 |

Item-level catalogue coverage is not recoverable from a rank dump. It needs the
bucket sizes, which `budget_match.py` reads from `catalog_sid_codes.npy`.

## Not reported, and why

- precision@k, MAP, HR@k: with one relevant item per row they are R@k relabelled.
- diversity and serendipity: they need item content, which this project excludes.
- per-row costs: a rank dump does not record how many items were scored.

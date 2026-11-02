# Dump analysis

98,304 rows, 3 arm(s), same row set and same targets for every arm.

| arm | cap | misses | effective rank |
|---|---|---|---|
| tiger_v1_beam | 20 | 89.9323% | 4.69 |
| digr_v1sid_v2feat | 10,000 | 49.7823% | 856.65 |
| coclick_h20 | 10,000 | 53.1118% | 481.21 |

`cap` is the deepest position the dump can express. A miss is stored at `cap + 1`,
so `effective rank` is computed over hits only and is not flattered by sentinels.

## The curve

| arm | R@1 | R@5 | R@10 | R@100 | R@1000 | R@10000 |
|---|---|---|---|---|---|---|
| tiger_v1_beam | 1.7558% | 5.0741% | 7.4849% | 10.0677% | 10.0677% | 10.0677% |
| digr_v1sid_v2feat | 0.6958% | 1.6602% | 2.3275% | 7.3059% | 21.5291% | 50.2177% |
| coclick_h20 | 0.8057% | 2.4882% | 3.7069% | 10.9873% | 25.3021% | 46.8882% |

### nDCG (top of the list only)

| arm | nDCG@1 | nDCG@5 | nDCG@10 | nDCG@100 | nDCG@1000 | nDCG@10000 |
|---|---|---|---|---|---|---|
| tiger_v1_beam | 0.01756 | 0.03441 | 0.04217 | 0.04872 | 0.04872 | 0.04872 |
| digr_v1sid_v2feat | 0.00696 | 0.01184 | 0.01398 | 0.02356 | 0.04035 | 0.06491 |
| coclick_h20 | 0.00806 | 0.01653 | 0.02044 | 0.03467 | 0.05176 | 0.07037 |

### MRR

| arm | MRR@1 | MRR@5 | MRR@10 | MRR@100 | MRR@1000 | MRR@10000 |
|---|---|---|---|---|---|---|
| tiger_v1_beam | 0.01756 | 0.02905 | 0.03223 | 0.03404 | 0.03404 | 0.03404 |
| digr_v1sid_v2feat | 0.00696 | 0.01028 | 0.01116 | 0.01270 | 0.01316 | 0.01326 |
| coclick_h20 | 0.00806 | 0.01380 | 0.01539 | 0.01777 | 0.01827 | 0.01835 |

`nDCG@k / R@k` is the mean weight of a hit, so it says where inside the top k
the hits land. 1.0 means every hit is at rank 1.

| arm | nDCG@10 / R@10 |
|---|---|
| tiger_v1_beam | 0.5633 |
| digr_v1sid_v2feat | 0.6008 |
| coclick_h20 | 0.5513 |

## Pair union at k = 10

The share of rows some arm reaches, which no single-arm number shows.

| pair | A only | B only | union | disagreement |
|---|---|---|---|---|
| tiger_v1_beam vs digr_v1sid_v2feat | 5.6366% | 0.4791% | **7.9641%** | 6.1157% |
| tiger_v1_beam vs coclick_h20 | 5.7078% | 1.9297% | **9.4147%** | 7.6375% |
| digr_v1sid_v2feat vs coclick_h20 | 1.5411% | 2.9205% | **5.2480%** | 4.4617% |

## Paired significance at k = 10 (McNemar)

Both arms see the same rows, so this is the right test. `p` is a chi-square
with one degree of freedom and a continuity correction.

| pair | A hit B miss | A miss B hit | chi2 | p |
|---|---|---|---|---|
| tiger_v1_beam vs digr_v1sid_v2feat | 5,541 | 471 | 4273.912 | 0 |
| tiger_v1_beam vs coclick_h20 | 5,611 | 1,897 | 1836.224 | 0 |
| digr_v1sid_v2feat vs coclick_h20 | 1,515 | 2,871 | 418.610 | 4.9e-93 |

## Recall@10 by target popularity

Quintiles over `log1p(train target count)`, ordered cold to hot. A content-only
arm should hold up toward the tail; a popularity rider collapses there.

| arm | tail | lower | middle | upper | head |
|---|---|---|---|---|---|
| rows | 15,715 | 21,192 | 21,459 | 20,146 | 19,792 |
| mean freq | 0.0 | 2.2 | 9.6 | 36.5 | 449.0 |
| tiger_v1_beam | 0.0064% | 0.0802% | 2.1296% | 7.2769% | 27.3696% |
| digr_v1sid_v2feat | 0.1973% | 0.3586% | 0.5732% | 1.2558% | 9.1198% |
| coclick_h20 | 0.1400% | 0.4719% | 1.4073% | 3.2165% | 12.9951% |

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

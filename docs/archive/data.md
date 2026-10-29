# The data

What is in the project, what each number is, and what the model is actually fed.
Anything marked *derived* is arithmetic over the files listed above it, so it can
be rechecked without a GPU.

---

## 1. Where things live

```
cluster scratch   /scratch/mscbdtspod2026/jksulaiman/data/
                    digr_data/                         preprocessed, what training reads
                    digr_runs/                         checkpoints, logs, catalogs
cluster archive   $HOME/NZ5/train_model/archive/
                    2026-10-23-preprocess-01/digr_data  same split, used by the CPU baselines
local             C:/Users/justin/Documents/GitHub/NZ5-GR/
                    Model Training/notebooks/data_snapshot.md   raw-file sample
                    SID Generation/                     RQ-VAE
```

The GPU partition can read `/scratch`; the CPU partition cannot, which is why the
baselines read the archived copy of the same split.

---

## 2. The raw files

### `seq.parquet`

The behaviour log. One row per user, one column `seq`:

```python
{"user_id": 953188,
 "seq": [{"item_id": 2905777, "action_type": 0, "timestamp": 1745919240}, ...]}
```

Raw `item_id` is a business key (for example `20000006363`), not the catalog
index. `user_id` is an integer in the hundreds of thousands. This file is the
only source of history; there are no negative samples and no impression log, so
the task is next-click prediction, not ranking against served candidates.

### `user_feat.parquet`

One row per user, `user_id` plus eight static feature columns, sampled as
`103, 104, 105, 106, 107, 108, 109, 110`. Values are small enumerations, some
list-valued (`[6]`, `[15]`), many `<NA>`.

The eight columns' vocabulary sizes are `[87, 3, 8, 15, 20, 5, 4, 3]`, which is
145 token ids in total. That is the entire user-side input to DiGR.

### `index_rqvae_prod.json`

The SID lookup, `item_id -> [c0, c1, c2, c3, c4]`, 4,783,154 entries:

```python
{"26":      [76, 73, 51, 14, 0],
 "3492133": [76, 73, 51, 14, 1],
 "521243":  [76, 73, 51, 14, 2], ...}
```

Those three consecutive entries are the whole reason the fifth level exists: they
share all four RQ-VAE codes and differ only in the last, which is a collision
suffix, not a learned codebook. Level 5 has exactly 731 values because the
largest 4-code group has 731 members.

### `candidate.parquet`

660,000 rows, `item_id` plus `retrieval_id` plus item feature columns
(`100, 101, 102, 112, 114, 115, 116, 117`, ...), each a
`{"cold_start": 0, "feature_value": "6"}` dict. This is a precomputed candidate
pool with item content features, not the catalog.

- 511,029 of 660,000 rows have a `retrieval_id` (77.43%).
- 80.43% of val targets fall inside this pool.

The pool is a deployment artefact. It is not used for training and not used for
evaluation, because DiGR and both baselines score the full catalog.

### `SID Generation/`

The RQ-VAE that turned item content features into the SIDs. Four codebooks of
256 on 256-dim item vectors (the width of the item tower's output in
`item_embeddings.parquet`, passed as `--input_dim 256`; `--num_codebooks 4`,
`rq_vae_cli.py:881` maps that to `codebook_sizes=[256]*num_codebooks`), plus the
level-5 collision suffix added afterwards. The RQ-VAE's own `--input_dim` default
is 1024, so the flag has to be set explicitly. The production run is
`local/2026-16-10-3-rq/rq_vae_prod.sh`. The suffix is unbounded: `unique_id_bound`
was left unset, so level 5 grew to the largest four-code group (731 items).

---

## 3. The preprocessed artefact, `digr_data/`

```
manifest.json            paths, max_seq_len, preprocessing flags
vocab.json               token ids and offsets
catalog_item_ids.npy     (4,783,154,)  canonical catalog index (1..N) per catalog row
catalog_sid_codes.npy    (4,783,154, 5) int32, codes without the vocab offset
catalog_meta.json        sizes, collision stats
users.npy                user ids over the harvested rows
gap_edges.npy            gap edges for the anchor sampler
train/tokens.npy         int32 flat, the token stream
train/offsets.npy        int64, n_rows + 1
train/targets.npy        int64, catalog row index of the held-out target
train/users.npy
train/stats.json
val/  same five files
```

Every split file is `tokens.npy`, `offsets.npy`, `targets.npy`, `users.npy`,
`stats.json` (`Model Training/preprocess_digr.py`, `load_split` at `:2151`,
which mmaps `tokens.npy` so 846 MB does not enter RAM).

Targets are catalog-row indices, not business keys and not SID tuples.

---

## 4. Row layout and the token budget

One row is one training example:

```
[ feature tokens ][ SEP ][ (temporal token, c0 c1 c2 c3 c4) x m ]
```

- `SEP` is id `1`, `pad` is id `0`, and every code token is `>= 2`, so `SEP` is
  unique inside a row and its position is the boundary
  (`row_feature_lengths`, `Model Training/baselines_digr.py:236`).
- `m` events, `m <= 20` (`max_seq_len`), mean 19.79 on train.
- Row length is therefore `1 + feat_len + 6*m`, which the loader asserts on, so a
  split built with a different window is rejected rather than silently misread.

Vocabulary, `vocab.json`:

| ids | what | size |
|---|---|---|
| 0 | pad | 1 |
| 1 | sep | 1 |
| 2 .. 257 | SID level 0 | 256 |
| 258 .. 513 | SID level 1 | 256 |
| 514 .. 769 | SID level 2 | 256 |
| 770 .. 1025 | SID level 3 | 256 |
| 1026 .. 1756 | SID level 4 (collision suffix) | 731 |
| 1757 .. 1820 | temporal bucket | 64 |
| 1821 .. 1965 | user features | 145 |
| | **total** | **1966** |

Check: `2 + 4*256 + 731 + 64 + (87+3+8+15+20+5+4+3) = 1966`. Matches
`total_size` exactly.

Two design facts that are easy to get wrong later:

- the temporal token is one bucket per event and is *not* part of the SID. It lets
  the model know recency without knowing wall-clock time.
- `action_type` is never emitted as a token. Every event is a click regardless of
  its action code.

---

## 5. Split sizes

| | rows | events | events/row | tokens |
|---|---|---|---|---|
| train | 1,719,430 | 34,019,640 | 19.79 | 221,500,536 |
| val | 98,304 | 1,953,759 | 19.87 | - |

Val is 5.4% of the 1,817,734 harvested rows.

Token arithmetic, which is the check that the layout is understood:

```
rows*1 + feature_tokens + events*6 = tokens
1,719,430 + 15,663,266 + 204,117,840 = 221,500,536
```

so feature tokens total 15,663,266, i.e. **9.11 per row**, inside the 8..16 range
the preprocessor allows. If a future change moves `feat_len` the token count
moves with it and this identity stops holding, which is the alarm.

### Anchors

`--anchors_per_record` controls how many (history, target) pairs are cut from each
user record. The cluster copy uses 2 and yields the 1,719,430 rows above. A local
run with 3 yields 2,480,142 rows / 317,812,410 tokens. Same users, same events,
more cuts, 1.44x the tokens.

Every reported DiGR run used the 2-anchor split. Any comparison against the
baselines must stay on the same split, which is the archived copy.

---

## 6. What each component reads

| component | inputs |
|---|---|
| `train_digr.py` | `train/`, `val/`, `catalog_sid_codes`, `vocab.json`, `catalog_item_ids` |
| `eval_full_catalog.py` | the above plus a checkpoint |
| `baselines_digr.py` popularity | `train/targets.npy` only |
| `baselines_digr.py` B4 co-click | `tokens.npy`, `offsets.npy`, `targets.npy`, `catalog_sid_codes` |
| `baselines_digr.py` B3 | the same as B4 |

The baselines never load the model, never touch the GPU, and never materialise the
catalog. B4 builds a 25,669,281-entry CSR from the 34,019,640 raw pairs in under
5 seconds on one core, then answers a query in about 205 ms.

---

## 7. Coverage and sparsity, which is what makes this hard

| quantity | value |
|---|---|
| catalog items | 4,783,154 |
| items that appear anywhere in the click logs | 1,580,961 (33.0%) |
| items never clicked | 3,202,193 (67.0%) |
| distinct targets in train | 488,945 |
| distinct targets in train ∪ val | 504,404 (10.5% of the catalog) |
| mean target count per catalog item | 0.359 |
| mean target count per *seen* item | 3.52 |
| items alone in their 4-code prefix | 2,904,256 (60.7%) |
| items sharing a 4-code prefix | 1,878,898 (39.3%) |
| largest 4-code group | 731 |

The last three are the only SID-quality measurements in the repo. They say the
code is fine grained: 39% of items are not uniquely identified by four codes,
which is why the fifth level exists, and why "the SID has collisions" is false
while "the SID encodes a lot of arbitrary detail" is true.

A separate probe supplied by the user (5,000 queries against a 50,000-item
sample, so candidate counts are about 1/96 of the true catalog) reported, for
"does the target share the query's prefix at depth d":

| depth | prefix hit rate | lift over base rate | mean candidates | queries with 0 candidates |
|---|---|---|---|---|
| 1 | 71.158% | 166x | 213.8 | 0.0% |
| 2 | 16.828% | 2,104x | 4.13 | 21.7% |
| 3 | 2.626% | 2,626x | 0.31 | 84.0% |
| 4 | 0.346% | - | 0.036 | 97.24% |

**These are self-graded and were later superseded.** The probe's "target" is the
embedding's own nearest neighbour, so the table measures bucket purity of the
geometry, not next-click. Scored against the real val targets the same buckets give
**18.3156%** at depth 1 and **2.5167%** at depth 2. See `docs/results.md` section 8.
Do not quote the numbers in this table as retrieval recall.

Read as a retrieval table, this says the coarsest code carries most of the
signal: depth 1 alone agrees 71% of the time against a 0.39% base rate. Read as
a candidate generator it says depth 3 already leaves 84% of queries with nothing,
which is the opposite of useful for retrieval but irrelevant to DiGR, which scores
every item anyway. It is recorded here as the starting point for the SID checks,
not as a conclusion.

---

## 8. What the data does not contain

- No impressions, no served-candidate log, no negatives. Everything is a click.
- No item text or images kept in the repo. The RQ-VAE input vectors are not in
  `digr_data`; only the resulting codes are.
- No per-event session id, so "consecutive" means consecutive inside a user's
  sequence, which may span unrelated sessions.
- No cold-start split. 67% of items are never clicked and are still ranked by
  every method, which is a large part of why the tail numbers (R@1000, R@10000)
  look the way they do.

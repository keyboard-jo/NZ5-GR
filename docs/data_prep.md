# Data preparation: from SID and user features to `digr_data`

Everything between raw logs and a directory the trainers can read. One script owns
this stage, `Plan 2/preprocess_digr_v2.py`, and the trainers never re-tokenise; they
load the arrays it writes.

This page is the runbook. For why the layout looks the way it does, and what the
model reads back out of it, see `learn/data_prep_explained.pdf`. For what the
numbers mean once a model has been trained on them, see `evaluation.md`.

## 0. In one sentence

Three files go in (an event log, a static user table, an item to SID map) and a
directory comes out holding an integer token stream per row plus the catalogue
arrays that the token ids refer to.

## 1. What you supply

| flag | file | what it is |
| --- | --- | --- |
| `--sequence_path` | `seq.parquet` | one row per user history, in one of two shapes (below) |
| `--user_feat_path` | `user_feat.parquet` | optional, one row per user, every column becomes tokens |
| `--catalog_path` | `index_rqvae_*.json` or a parquet | the item universe and its SID codes |

`--sequence_path`, nested shape, which is what `data/seq.parquet` is:

```
user_id : int64
seq     : list<struct<item_id, action_type, timestamp>>
```

Every event is kept whether it was a click, an exposure or a null action. With
`--add_action` the action becomes its own token, so the three stay distinguishable.
A null `action_type` maps to `ACTION_UNKNOWN` and can never become a target.

`--sequence_path`, flat shape: `history_items` plus optional `history_actions`,
`history_timestamps`, `target`, `split`. Note that
`SID Generation/preprocess_sequence_cli.py` also writes `history_items`/`target` but
with **1-based catalogue indices** and no user id, so remap those before using them
here.

`--catalog_path` accepts the RQ-VAE CLI dump directly, `{"26": [76, 73, 51, 14, 0], ...}`,
or a parquet with `sids` / wide `sid_1 .. sid_L`. `--num_sid_levels N` keeps the
first N levels and `0` keeps all of them.

Timestamps may be epoch seconds, ms or us; the scale is detected from the median
magnitude.

## 2. What you get

| file | contents |
| --- | --- |
| `vocab.json` | the token layout: `n_special`, `sid_sizes`, every offset, `total_size`, `event_width` |
| `catalog_item_ids.npy` | catalogue order, which **is** the label space `y` |
| `catalog_sid_codes.npy` | `[n_items, L]` int32 codes, row `i` matching `catalog_item_ids[i]` |
| `catalog_meta.json` | `{is_int, size, num_levels}` |
| `user_feat_meta.json` | feature columns, cardinalities, block lengths |
| `users.npy` / `users_meta.json` | legacy empty profile vocabulary, read by `train_digr.py`, holds no user |
| `gap_edges.npy` | the temporal bin edges |
| `train/`, `val/`, optional `infer/` | `tokens.npy`, `offsets.npy`, `targets.npy`, `users.npy`, `stats.json` |
| `preprocess_meta.json` | the manifest: sizes, `window`, `anchors_per_record`, `anchor_gap`, sources, counts, `catalog_collisions` |

The per-split arrays are CSR. `offsets` has `rows + 1` entries, row `r` is
`tokens[offsets[r] : offsets[r+1]]`, `targets[r]` is a catalogue row index, and
`users[r]` is the raw user id.

## 3. One row, exactly

```
[F_1 .. F_M, SEP, (T_1, A_1, [C_1], S_1_1 .. S_1_L), ... ]

len  = 1 + M + T * slot
slot = 1 + A + C + L        (also stored as vocab.event_width)
T   <= --window, default 20, newest event last, anchor excluded
```

So a row is the user's static block, a separator, and then up to twenty events.
The SID block sits **last inside every slot**, which is what lets the trainer read
the query item as "the last L tokens of the row" with no extra bookkeeping.

How big the static block is depends on `user_feat.parquet`, not on the SID, and
**it is not a constant**. Every column takes at least one slot, and a list-valued
column takes one slot per value it actually has. In this project's file four of the
eight columns are lists, so

| | value |
| --- | --- |
| columns | 8 |
| list columns | `106` up to 8 values, `108` up to 2, `110` up to 2, `107` is a list of length 1 |
| `block_len_min` | 8 |
| `block_len_mean` | 9.11 |
| `block_len_max` | 16 |

and `train/stats.json` agrees: `feat_tokens` 15,663,266 over 1,719,430 rows is
**9.11 slots per row**. So `M` in the formula is the user's own block, 8 to 16.

Check it against the measured token counts, with `T` averaging 19.79 events:

| build | slot | row length |
| --- | --- | --- |
| v2, `2 x 8192` | `1 + 1 + 2 = 4` | `1 + 9.11 + 19.79 x 4 = 89.25` |
| v1, `4 x 256 + 731` | `1 + 0 + 5 = 6` | `1 + 9.11 + 19.79 x 6 = 128.82` |

Both match the measured 89.3 and 128.8. The eight numbers in the vocabulary
(`87, 3, 8, 15, 20, 5, 4, 3`, summing to 145) are **widths per column, not tokens
per row**: each is that column's `cardinality + 1` reserved for UNK, and each column
owns a contiguous range (`local_base` 0, 87, 90, 98, 113, 133, 138, 142). A user
missing from `user_feat.parquet` gets the UNK token of each column, so their block
is the minimum 8 and no row is short.

Two consequences worth stating.

The anchor is not in its own history. `T` ends one event before the label, so the
newest token of a row is the last code of the event immediately preceding the click.

The richer event is also the shorter one. A 5-level SID with no action token is
`slot = 6`. A 2-level SID with an action token is `slot = 4`. Adding features is
cheaper than adding codes.

## 4. How token ids are assigned

`cursor = n_special = 2` (PAD is 0, SEP is 1), and then each block is laid out in
this order, with **no two blocks sharing a range**:

1. `sid_offsets[l]` for each level, `sid_sizes[l]` wide
2. `offset_temporal`, `n_temporal` wide
3. `offset_action`, `n_action` wide (0 when `--add_action` is off)
4. `offset_clock`, `n_clock` wide (0 by default)
5. `offset_features`, then one contiguous range per feature column
6. `total_size`

The v2 build as it stands:

| block | offset | width |
| --- | --- | --- |
| level 0 | 2 | 8,192 |
| level 1 | 8,194 | 8,192 |
| temporal | 16,386 | 128 |
| action | 16,514 | 3 |
| clock | 16,517 | 0 |
| features `[87, 3, 8, 15, 20, 5, 4, 3]` | 16,517 | 145 |
| **total** | | **16,662** |

The v1 build, for comparison: `[2, 258, 514, 770, 1026]` for the five SID levels,
temporal at 1,757 (64 wide), features at 1,821, `total_size` 1,966.

Disjointness is not cosmetic. A gap bucket numbered 3 and a SID code numbered 3
would otherwise be the same row of the embedding table; the only thing keeping them
apart is that their offsets differ. Any new token type must take a fresh range.

## 5. Time becomes a bucket

The gap `Delta` between consecutive events is bucketed rather than used raw:

```
edges = np.geomspace(1.0, max_gap_seconds, n_bins - 1)   # 127 edges for 128 bins
bin   = np.digitize(Delta, edges)                        # 0 if Delta < 1s
bin   = np.clip(bin, 0, n_bins - 1)
T_i   = offset_temporal + bin
```

A row's first event has `Delta = 0`, so it always lands in bin 0.

The spacing is geometric so the bins are even in log time: fine resolution for
seconds, coarse for weeks. With `--n_temporal_bins 128` and
`--max_gap_seconds 2592000` (30 days) the ratio between neighbouring edges is
`2592000^(1/126) = 1.124`, so each bin is about 12% wider than the last.

Bin 0 merges two different situations, first event and sub-second gap, because
neither says anything about behaviour. That merge is the reason a row's first event
always reads as bin 0.

## 6. Anchors: how many rows one record yields

An **anchor** is an event whose item becomes the label. Only clicks can be anchors.
The history of a row is the up to `--window` events directly before its anchor,
whatever their action type.

Anchors are taken from the right. A click is accepted only when it sits at least
`--anchor_gap` events before the last accepted anchor, so two rows of one record
never share almost the same history. A skipped click does not consume the quota.
At most `--anchors_per_record` rows come from one record, default 3.

With the cluster build's `--anchors_per_record 2`, 1,001,845 user records become
1,719,430 train rows, 1.72 per record.

The val split keeps **one** anchor per record, its last click, so metrics stay cheap
and comparable. The infer split has no label and keeps the whole windowed history.

`stats.json` per split records why rows were lost, one binding limit per record:
`rows_window_truncated`, `rows_short`, `rows_gap_limited`, `rows_capped_by_anchors`,
`rows_without_click`, `rows_missing_target`, `rows_non_monotonic`, plus
`unknown_items`, `unknown_actions` and `feat_tokens`.

## 7. The val split is a hash, so it must be declared

Validation membership is a per-user hash:

```
bucket = zlib.crc32(str(user_id).encode("utf-8")) % 10000
val    = bucket < round(val_ratio * 10000)
```

Rows without a user id fall back to a multiplicative hash of the row index. With no
user column at all, the hash falls back again.

This is why a wrong `--val_ratio` is invisible: it produces a well formed val split
with different users in it, and nothing downstream can tell. So the script refuses
to run unless you say which split you mean:

- `--expect_val_split <archived digr_data>` holds the run to that split. Val users are
  checked *before* the tokeniser runs, and val users, catalogue and targets again
  after the pass.
- `--allow_new_split` declares a deliberately new one.

Passing neither is an error. Two builds are only directly comparable if they were
made with the same split, which is why the ablation lives on the archived split.

## 8. Flags that change the build

| flag | default | effect |
| --- | --- | --- |
| `--num_sid_levels` | 0, all | keep the first N codes. A 2-level SID needs `2` |
| `--codebook_size`, `--code_offset` | from the file | only for parquet catalogues that need remapping |
| `--n_temporal_bins` | 128 | gap resolution. v1 used 64 |
| `--add_action` | on | the action token. `--no-add_action` for a v1-shaped event |
| `--add_clock` | off | adds `1 + weekday*24 + hour`, one token per event |
| `--window` (alias `--max_seq_len`) | 20 | events of history per row |
| `--anchors_per_record` | 3 | rows per record. The cluster build uses 2 |
| `--anchor_gap` | 5 | minimum events between two anchors of one record |
| `--min_history` | 5 | events required before an anchor |
| `--clicks_only` | off | flat input only, drops non-clicks from the history |
| `--val_ratio` | 0.02 | must be declared, see section 7. The cluster build uses 0.1 |
| `--user_feat_cols`, `--max_feat_len` | every other column, longest seen | the static block |
| `--max_rows`, `--max_val_rows` | 0, all | smoke tests only |
| `--overwrite` | off | refuse to write into a non-empty directory otherwise |

## 9. Two worked invocations

The default build behind every number in `README.md` section 1:

```
sbatch "Plan 2/preprocess_digr_v2.sh"
```

Its header carries the settings that must not drift: `--anchors_per_record 2`,
`--val_ratio 0.1`, `--expect_val_split` pointed at the archived v1 `digr_data`, and
the note that the `2 x 8,192` index is named `index_rqvae_3x1024.json` because that
run reused the previous script's output path.

The two ablation arms, from `ablationDiGR/prep_arm.sh`:

```
# arm a: new SID, old features
--num_sid_levels 2  --no-add_action  --n_temporal_bins 64      # -> digr_data_a

# arm b: old SID, new features
--num_sid_levels 5  --add_action     --n_temporal_bins 128     # -> digr_data_b
```

The only difference between the two arms is which catalogue goes in. Anything else
would make the comparison an experiment about two things.

## 10. Invariants to check after a build

A build that finishes is not a build that is correct. Check these, in this order.

### What a correct build contains

On the cluster data, with `--anchors_per_record 2` and `--val_ratio 0.1`, the two
vintages differ **only in how many tokens each event costs**. Same records, same
anchors, same split, same rows.

| quantity | v1, `4 x 256 + 731` | v2, `2 x 8192` | why they differ |
| --- | --- | --- | --- |
| user records read | 1,001,845 | 1,001,845 | the same log |
| train rows | 1,719,430 | 1,719,430 | 2 anchors per record |
| val rows | 98,304 | 98,304 | one anchor per record, same hash split |
| train tokens | 221,500,536 | 153,461,256 | slot 6 against slot 4 |
| train tokens per row | 128.8 | 89.3 | same reason |
| val tokens | 12,716,408 | 8,808,890 | same reason |
| val tokens per row | 129.4 | 89.6 | same reason |
| vocabulary | 1,966 | 16,662 | 5 levels of 256/731 against 2 of 8,192, and 64 gap bins against 128 |
| `event_width` | 6 | 4 | `1 + L` against `1 + 1 + L` |
| static block `M` | 8 to 16, mean 9.11 | same | the same `user_feat.parquet`; four of its eight columns are lists |
| fixed tokens per row | `1 + M`, mean 10.11 | same | the separator plus the static block |

So the v1 stream is **1.44x longer** than v2 per row while its vocabulary is **8.5x
smaller**. That trade is the whole point of the SID comparison: v1 buys a small head
with a long sequence, v2 buys a short sequence with a big head.

The local snapshots are a different recipe and will not match those rows: the local
v2 build used 3 anchors per record and produced 2,480,142 train rows, against the
cluster's 1,719,430. Check the recipe in `preprocess_meta.json` before comparing a
local build with a cluster one.

### The checks

**Row counts.** `train 1,719,430` and `val 98,304` on the cluster data. A different
number means `--anchors_per_record`, `--val_ratio` or `--window` moved.

**`sid_sizes`.** `vocab.json` derives each level's width as `max(code) + 1`, so a
level whose top code never appears comes back one short and shifts every offset
above it. The script warns; do not ignore it. Expected `[8192, 8192]` for v2 and
`[256, 256, 256, 256, 731]` for v1.

**`event_width`.** `1 + A + C + L`. Expected 4 for the v2 build and 6 for v1. A
value you did not intend means `--add_action`, `--add_clock` or `--num_sid_levels`
was wrong, and a wrong `event_width` still trains.

**Tokenizer count.** `153,461,256` tokens for the v2 build, `89` per row. v1:
`221,500,536`, `129` per row.

**Collisions.** `preprocess_meta.json["catalog_collisions"]`:

| build | distinct codes | items sharing one | share | largest |
| --- | --- | --- | --- | --- |
| `2 x 8192` | 3,002,708 | 2,559,616 | 0.5351 | 315 |
| `4 x 256 + 731` | 4,783,154 distinct full tuples | 0 | 0 | 1 |

The v1 row is worth reading twice. Its five levels are injective, and its *four*
semantic levels give the 3,459,854 / 1,878,898 / 731 figures that its own log prints,
because the fifth level is a collision rank rather than a category. Whatever the
number says, it is a ceiling on any metric scored by exact item identity.

Lookup or decode in the **catalogue order** of `catalog_item_ids.npy` copies, never
in the json's order. The json is in quantiser file order and the two do not agree.

## 11. Traps

**Absolute paths rebinding.** Do not add a `cd "$(dirname "$0")"` to a wrapper that
reads env vars carrying relative paths; it silently reinterprets them.

**CRLF.** `sbatch` refuses a script with CRLF endings, and an editor on Windows writes
them. `sed -i 's/\r$//' *.sh` after a copy, or keep `core.autocrlf` off for `.sh`.

**The 39 to 195 band is derived, not quoted.** It comes from dividing the paper's
catalogue sizes by 256, TIGER's width. Do not cite it as a stated setting.

**Two `index_rqvae_prod.json` files exist.** `archive/2026-10-16-03` holds the
`4 x 256 + 731` SID that every v1 number came from, and `archive/2026-10-14-2` holds
a different prod run with a wider last level (`[255, 256, 256, 256, 914]`). Same
basename, different codebook. `ablationDiGR/check_inputs.sh` fingerprints both
before a run for exactly this reason.

**A wrong split does not error.** It builds, it trains, and every comparison
afterwards is between two different samples of users. Section 7.

## 12. Reproduce the check before a long job

```
sbatch ablationDiGR/check_inputs.sh      # fingerprints all four inputs
python "Plan 2/preprocess_digr_v2.py" --help
```

The first prints `OK`, `MISMATCH`, `BROKEN` or `MISSING` per file with the observed
value beside the expected one, and exits non-zero if anything is off. It runs in
seconds and it is the only cheap guard against a codebook that looks fine.

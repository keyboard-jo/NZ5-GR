# Plan 2

Diagnostics for a Semantic ID assignment, plus whatever the next SID release needs.
Nothing here trains anything.

## sid_diagnose.py

Answers four questions about a set of RQ-VAE codes with numpy alone, no model and
no GPU.

1. Are the codebooks healthy? Codes used, entropy in bits, against the ceiling.
2. How top heavy is the partition? Items per bucket at each prefix depth.
3. How many items collide? Items sharing a prefix with another item.
4. How much better than chance is a bucket at holding the real next click?

```bash
# everything, from a preprocessed digr_data directory
python "Plan 2/sid_diagnose.py" --data_dir local/2026-20-10-2-pre/digr_data

# codes only, for a fresh RQ-VAE output that has not been preprocessed yet
python "Plan 2/sid_diagnose.py" --codes new_codes.npy --vocab vocab.json --no_lift

# keep the numbers for a comparison
python "Plan 2/sid_diagnose.py" --data_dir a --json a.json
python "Plan 2/sid_diagnose.py" --data_dir b --json b.json
```

`--split_dir` defaults to `<data_dir>/val` and needs `tokens.npy`, `offsets.npy`
and `targets.npy`. `--max_rows` subsamples, `--no_lift` skips the split entirely.
The catalogue does not have to be the one the split was built from, but
`catalog_sid_codes.npy` and `vocab.json` must describe the same item ordering.

### Read lift, not recall

Recall at prefix depth `d` is the fraction of val rows whose target shares the
query's first `d` codes. That number always *falls* as `d` grows, because the
bucket shrinks, and a smaller bucket is mechanically less likely to contain the
target. So recall cannot compare two SIDs, and it cannot justify a wider one.

`lift` is recall divided by the bucket's share of the catalogue, which is what
recall would be if the codes carried no information. A SID that halves every
bucket and halves recall has bought nothing and shows a flat lift. Only lift
moving upward is evidence that the codes are sharper.

The one trap this cannot fix: lift is bounded by the bucket size, so a very wide
SID shows high lift at depth 1 for arithmetic reasons. Compare lift at matched
bucket size, or at the depth where lift peaks.

### Content SID, measured

`local/2026-20-10-2-pre/digr_data`, 4,783,154 items, 98,304 val rows, 0 unmapped
queries, 0 targets outside the catalogue.

Codebooks: levels 0 to 3 sit at 7.874 to 7.980 bits of 8, every code used. Level 4
holds 731 values at 1.764 bits of 9.514, an entropy ratio of 0.19, because it is the
collision-breaking suffix rather than a real codebook.

| depth | buckets | items/bucket | catalogue share | shared items | recall | lift |
|---|---|---|---|---|---|---|
| 1 | 256 | 18,684.20 | 0.0054% | 4,782,898 (99.99%) | 18.3156% | 46.9x |
| 2 | 57,122 | 83.74 | 1.19% | 4,726,032 (98.81%) | 2.5167% | 1437.6x |
| 3 | 1,506,041 | 3.18 | 31.49% | 3,277,113 (68.51%) | 0.5483% | 8257.6x |
| 4 | 3,459,854 | 1.38 | 72.33% | 1,323,300 (27.67%) | 0.1302% | 4505.0x |
| 5 | 4,783,154 | 1.00 | 100% | 0 | 0.0173% | 827.2x |

Two readings.

Lift peaks at depth 3 and then falls. Past the third level the extra codes do not
make the target more likely to share the query's prefix, they only split the bucket,
so levels 4 and 5 are bookkeeping rather than signal. Level 5 is injective, and its
recall is the rate at which the val target is already the last clicked item.

The depth-1 bucket holds 18,684 items on average and catches 18.3% of targets, 46.9
times chance. That is the ceiling a method reading only the first code has to work
with, and it is the number to beat when a wider SID is compared.

### Width sweep, measured

Three SIDs cut from the same frozen 256-d embedding, scored against the same 98,304
val rows. `local/2026-27-10-1-rq` holds the 3 x 1,024 run, `local/2026-27-10-2-rq`
the 2 x 8,192 run.

| SID | rows | items per level-0 code | depth-1 lift | depth-1 query bucket | peak lift |
|---|---|---|---|---|---|
| 4 x 256 + 731 | 1,755 | 18,684 | 46.9x | 19,308.02 | 8,257.6x at depth 3 |
| 3 x 1,024 | 3,072 | 4,671 | 126.8x | 5,618.40 | 6,732.1x at depth 3 |
| 2 x 8,192 | 16,384 | 584 | 463.3x | 745.28 | 7,880.6x at depth 2 |

Read the whole thing at a matched candidate budget, where the comparison is fair:
recall per 1,000 candidates is 0.949, 2.204 and 7.589, so the widest SID is 8.0x the
original. Raw depth-1 recall falls the other way, 18.3156% to 12.3840% to 5.6559%,
because the bucket shrank by 26x. That is the trap this section opened with.

What width does not fix is collisions. Only 62.78% of items hold a two-code tuple of
their own, so 2,559,616 items (53.51%) sit in a group of two or more, against a
uniform birthday bound of 7.13%, because k-means cells cluster where the items are.
So a collision share is not a gate any two-level SID can pass, and the quantity worth
printing is the tie ceiling below. The width axis was closed here by decision: 2 x
8,192 is the endorsed SID, and 2 x 16,384 stays unrun with its injectivity prediction
recorded as untested.

### The tie ceiling

Two items with identical codes get identical scores from any scorer that reads the
SID, because the item vector is a function of the codes. A tie group of size `g`
therefore caps recall at `1/g` at rank 1 and `min(1, k/g)` at rank `k`, and averaging
that over the split's targets says what a collision rate costs in accuracy rather
than in bookkeeping.

| SID | mean tie group of the target | median | p90 | largest | R@1 ceiling | R@10 ceiling |
|---|---|---|---|---|---|---|
| 4 x 256 + 731 | 1.00 | 1 | 1 | 1 | 100% | 100% |
| 3 x 1,024 | 5.55 | 1 | 11 | 372 | 64.10% | 94.85% |
| 2 x 8,192 | 6.39 | 2 | 14 | 239 | 53.57% | 93.20% |

DiGR scores R@1 0.79% and R@10 2.59% on this split, so the ceiling is far away and
not binding. It only becomes binding near 50% R@1, which is why the plan accepts the
2 x 8,192 tie rate instead of demanding injectivity. The script prints this section
on every run, so any future SID can be checked in one command.

### A trap worth remembering

Codes live in `.npy` as `int32`. Mixed-radix packing at depth 4 reaches
`255 * 256**3 = 4,278,190,080`, past the `int32` ceiling, and on Windows a C long is
also 32 bits. Packing without an explicit `int64` cast either wraps silently or
raises `OverflowError: Python int too large to convert to C long`. The first run of
this script reported depth 4 at 0.0621% recall with only 49.8% of queries finding a
bucket, and depth 5 crashed. Both were wrong. `load_codes` casts to `int64` and
`pack_keys` refuses a radix product at or above `2**62`.

## convert_rqvae_index.py

`rq_vae_cli.py` writes a JSON object mapping item id to a list of codes. Every tool
here wants a dense `int32` array indexed by catalogue row, plus a vocab saying how
wide each level is. This converts one to the other and refuses to guess.

```bash
python "Plan 2/convert_rqvae_index.py" \
    --index local/2026-27-10-2-rq/test_output/index_rqvae_3x1024.json \
    --item_ids local/2026-20-10-2-pre/digr_data/catalog_item_ids.npy \
    --output_dir local/2026-27-10-2-rq --tag 2x8192
```

Writes `codes_<tag>_aligned.npy` and `vocab_<tag>.json`, in the order of
`--item_ids` so row `r` is item `ids[r]`. It raises on a catalogued item missing
from the index, on ragged code lengths and on negative codes, and it derives each
level width from the highest code present rather than from what the run was built
with, because the index carries no such field. The `--tag` matters: `rq_vae_cli.py`
writes file names from whatever the script said, so a 2 x 8,192 run named
`index_rqvae_3x1024.json` is normal and the tag is what keeps the two apart.

A missing item in the index is the failure to look for. The output of `rq_vae_cli.py`
covers every row of the embedding parquet, which is a superset of the catalogue, so a
missing catalogue item means the alignment is wrong, not the quantiser.

Passing `--codes` that differs from the SID in `--data_dir` used to fail with
`KeyError: 'sid_offsets'`, because the token stream has to be decoded with the SID it
was tokenised with before the query can be re-read from the new one. The script now
notices the mismatch and resolves queries through `--data_dir`'s
`catalog_sid_codes.npy` automatically, printing which file it used.

## preprocess_digr_v2.py

A copy of `Model Training/preprocess_digr.py` that tokenises the event itself, not
just its item. One line of layout changes:

```
v1  (T, S1 .. SL)              1 + L tokens per event
v2  (T, A, [C], S1 .. SL)      1 + A + C + L tokens per event
```

`T` is the log-spaced gap to the previous event, `A` the action, `C` an optional
clock, `S` the SID codes. Everything else, including the feature block, the
anchors, the val split and the file names, is unchanged.

```bash
# on the cluster: one shell wrapper sits beside this file and is a single call
sbatch "Plan 2/preprocess_digr_v2.sh"              # default build
sbatch "Plan 2/preprocess_digr_v2.sh" --add_clock  # clock arm

# locally, SID v2: two levels, action token, 128 gap bins, no clock
python "Plan 2/preprocess_digr_v2.py" \
    --sequence_path data/seq.parquet \
    --catalog_path local/2026-27-10-2-rq/test_output/index_rqvae_3x1024.json \
    --num_sid_levels 2 \
    --user_feat_path data/user_feat.parquet \
    --output_dir digr_data_v2

# the v1 stream, byte for byte, out of the same file
python "Plan 2/preprocess_digr_v2.py" --no-add_action --n_temporal_bins 64 ...
```

`--add_action` is on by default and `--add_clock` is off. The action token is
where v1 was throwing information away: it used `action_type` to choose anchors
and then dropped it, so 89.4% of this dataset's events, the exposures, were
indistinguishable from clicks in the sequence. The slot order is published in
`vocab.json` as `action_token_order`, so nothing downstream has to guess it, and
the SID block stays last inside the slot, so `train_digr.py`'s "the last L tokens
of a row are the last event's codes" needs no change.

The split decision is mandatory: `--expect_val_split <previous digr_data>` or `--allow_new_split`, and passing neither is an error, because a build that invents its own split looks exactly like one that reproduced the archived split. `--expect_val_split` is the comparability guard. It compares two different
populations on purpose: `val/users.npy` in the reference holds only the rows that
survived the anchor filters (98,304 of the 100,350 records the hash selected), so the
pre-flight check reads the reference `preprocess_meta.json` for both numbers and
requires the run to select the same 100,350. Strict equality on users, catalogue and
targets follows after the pass. Val membership
is `crc32(user_id) % 10000 < round(val_ratio * 10000)` inside `_val_row_mask`, so it
is a pure function of the user column and `--val_ratio`, and the flag checks it in
seconds, right after the hash, before the tokeniser runs. Targets cannot be
predicted that cheaply because the anchor logic decides which click becomes the val
row, so the flag checks them the other way: `_check_val_targets` compares the saved
val split against the reference as a user to target map once the pass has finished,
after confirming `catalog_item_ids.npy` is the same list, since a target is a row
index. The archived split used `--val_ratio 0.1` while this script defaults to 0.02, and 0.02 raises nothing while
producing a 19.6k-row val set, so the guard is what makes a run comparable rather
than merely self-consistent.

Per-row cost, at the measured 19.79 events per row and 9.11 feature tokens:

| stream | slot | tokens per row | train tokens | against today |
|---|---|---|---|---|
| v1, 5 levels, no action | 6 | 128.85 | 221.5M | - |
| v2, 2 levels, action | 4 | 89.28 | 153.5M | -30.7% |
| v2, 2 levels, action, clock | 5 | 109.07 | 187.5M | -15.4% |

So the richer event is also the cheaper one, twice over: two codes instead of five
buys back two tokens, and the action spends one of them.

Two guards that matter more than they look. `catalog_collision_stats` now counts
duplicates over every level kept and reports the share of the catalogue that shares
a code, because a v2 SID has no `--add_unique_id` suffix and a shared code is a pair
of items that can never be ranked apart; a share above 1% logs a warning. And
`assemble_chunk` asserts that `vocab.event_width` equals 1 gap plus the enabled
tokens plus the number of SID levels, so a vocab built from one catalogue cannot be
run against another.

## test_preprocess_digr_v2.py

`pytest "Plan 2" -q` runs 9 tests for the v2 stream on mock data. They pin the row
length at `1 + M + T * event_width`, the three action token values and their slot,
the codes staying last in the slot, `--no-add_action --n_temporal_bins 64`
reproducing the v1 stream token for token, the clock bucket for a known Monday and
for a missing timestamp, the vocab round trip including a v1 `vocab.json` that has
no action or clock keys, and the collision share on a catalogue with a duplicate
code.

## test_convert_rqvae_index.py

`pytest "Plan 2" -q` runs 7 tests. They pin that alignment follows `item_ids` rather
than the JSON's own order, that a missing item, ragged lengths and negative codes all
raise, that the width comes from the observed maximum rather than a declared
codebook size, and the CLI round trip.

## train_digr_v2.sh

One command for S4, the training half. It points at `digr_data/` from the
preprocessor and at `runs/`, keeps the frozen run's config, and takes any flag on
the command line, so the smoke test needs no edit:

    sbatch train_digr_v2.sh --max_steps 30 --run_name smoke_v2

`--temperature 0.175` is the only value that differs from the frozen run. The
score is `1/(L * temperature)` and `L` falls from 5 to 2, so leaving 0.07 would
multiply every logit by 2.5 for no reason connected to the data. Holding `1/(L*T)`
at 2.857 keeps the score scale identical, which leaves the SID and the event
stream as the only two things that changed. The trainer now derives that default
from the level count, so the flag documents intent rather than carrying the value.

Result, `runs/digr_v2_sid2x8192_e3_sampled300k`, 3h11m at 63.3 min per epoch:

| epoch | val loss | R@1 | R@10 | nDCG@10 | MRR@10 |
|---|---|---|---|---|---|
| 1 | 11.3101 | 0.6700% | 2.8200% | 0.01580 | 0.01210 |
| 2 | 11.0133 | 1.0400% | 3.6600% | 0.02170 | 0.01710 |
| 3 | 11.1558 | 0.9900% | 3.6800% | 0.02150 | 0.01690 |

Epoch 2 is the checkpoint to use: best val loss, and epoch 3 only adds overfitting.
Against the frozen 5-level run that is +31% R@1, +41% R@10, +38% nDCG@10 and +36%
MRR@10 at 30% less compute per epoch. Cut-offs past 10 need `eval_full_catalog.py`
on `best/`. Full table and caveats in `docs/results.md` section 12.

## test_sid_diagnose.py

`pytest "Plan 2" -q` runs 33 tests on mock data. They pin the packing convention
(level 0 least significant, injective on full radix, `int32` input does not
overflow), the prefix index (bucket numbering shared between catalogue and query,
absent prefix flagged), the codebook report (uniform codes sit at their ceiling, a
single code gives zero bits), occupancy (buckets never decrease with depth, the last
level is injective), the last-event extraction out of the token layout, and the two
properties of lift that matter: an uninformative bucket reads 1.0x, and halving
every bucket leaves lift flat while recall falls.

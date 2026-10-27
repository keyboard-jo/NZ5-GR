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

### A trap worth remembering

Codes live in `.npy` as `int32`. Mixed-radix packing at depth 4 reaches
`255 * 256**3 = 4,278,190,080`, past the `int32` ceiling, and on Windows a C long is
also 32 bits. Packing without an explicit `int64` cast either wraps silently or
raises `OverflowError: Python int too large to convert to C long`. The first run of
this script reported depth 4 at 0.0621% recall with only 49.8% of queries finding a
bucket, and depth 5 crashed. Both were wrong. `load_codes` casts to `int64` and
`pack_keys` refuses a radix product at or above `2**62`.

## test_sid_diagnose.py

`pytest "Plan 2" -q` runs 29 tests on mock data. They pin the packing convention
(level 0 least significant, injective on full radix, `int32` input does not
overflow), the prefix index (bucket numbering shared between catalogue and query,
absent prefix flagged), the codebook report (uniform codes sit at their ceiling, a
single code gives zero bits), occupancy (buckets never decrease with depth, the last
level is injective), the last-event extraction out of the token layout, and the two
properties of lift that matter: an uninformative bucket reads 1.0x, and halving
every bucket leaves lift flat while recall falls.

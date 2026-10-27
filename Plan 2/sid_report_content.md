# Content SID: diagnosis and the next configuration

Everything here is measured on the one Semantic ID the project has. It is the SID
the frozen DiGR run, both co-click baselines and the two probes were all built on,
so these are the reference numbers for any future SID release.

## Provenance

| item | value |
|---|---|
| codes | `local/2026-20-10-2-pre/digr_data/catalog_sid_codes.npy` |
| md5 | `28032537e3a901f573932bbdbd579a83` |
| shape | 4,783,154 x 5, `int32`, 95,663,208 bytes |
| vocab | `local/2026-20-10-2-pre/digr_data/vocab.json` |
| codebook sizes | `[256, 256, 256, 256, 731]` |
| vocabulary | 1,966 tokens = 2 special + 4 x 256 + 731 + 64 temporal + 145 feature |
| split scored | `local/2026-20-10-2-pre/digr_data/val`, 98,304 rows |

The three local preprocessed snapshots (`2026-20-10-1-pre`, `2026-20-10-2-pre`,
`2026-23-10-1-pre`) all carry the identical `catalog_sid_codes.npy`, same md5, so
there is only one SID in the project and no ambiguity about which one this describes.

It is the same SID the cluster run used, by three independent checks: catalogue
4,783,154 items, vocabulary 1,966 tokens made of exactly the same parts, and the
depth-1 recall of 18.3156% that was already recorded against the cluster side. The
cluster copy itself was not checksummed because it lives on `/scratch`; one line
settles it if needed:

```bash
md5sum /scratch/mscbdtspod2026/jksulaiman/data/digr_data/catalog_sid_codes.npy
```

The fifth level is not a codebook. It is the collision suffix from
`--add_unique_id`, and it reached 731 values because the run disabled clipping with
`--no-clip-suffix`. The default path is different and worth knowing: with
`--unique_id_bound` unset the bound falls back to `--codebook_size`, so clipping
would have capped this level at 256 values and any group larger than 256 would have
collapsed onto its last suffix. Read the level as bookkeeping either way, not as a
semantic level.

## Method

`Plan 2/sid_diagnose.py` answers four questions with numpy alone, no model and no
GPU. For each prefix depth `d` it buckets the whole catalogue by the first `d` codes
and reports:

- `recall`, the fraction of val rows whose target shares the query's first `d` codes
- `lift`, that recall divided by the bucket's share of the catalogue

Lift is the number to read. Recall at depth `d` always falls as `d` grows, because
the bucket shrinks and a smaller bucket is mechanically less likely to hold the
target. A SID that halves every bucket and halves recall has bought nothing and
shows a flat lift. Only lift moving up is evidence that the codes carry more signal.

## Codebooks

| level | size | used | entropy | maximum | utilisation | entropy ratio |
|---|---|---|---|---|---|---|
| 0 | 256 | 256 | 7.9243 | 8.000 | 100% | 0.9905 |
| 1 | 256 | 256 | 7.8739 | 8.000 | 100% | 0.9842 |
| 2 | 256 | 256 | 7.9337 | 8.000 | 100% | 0.9917 |
| 3 | 256 | 256 | 7.9804 | 8.000 | 100% | 0.9976 |
| 4 | 731 | 731 | 1.7644 | 9.514 | 100% | 0.1855 |

No dead codes anywhere. The four real codebooks sit at 98 to 99.8% of their entropy
ceiling, which is the strongest available evidence that the residual quantiser is
not collapsed and has no spare room at this width.

Level 4 sits at 1.76 bits of 9.51 because it is a suffix over 731 collision groups,
not a partition of the embedding space. Comparing its entropy ratio to the others is
meaningless.

## Occupancy

| depth | buckets | items per bucket | catalogue share | items sharing a prefix | largest bucket |
|---|---|---|---|---|---|
| 1 | 256 | 18,684.20 | 0.0054% | 4,782,898 (99.99%) | 39,883 |
| 2 | 57,122 | 83.74 | 1.19% | 4,726,032 (98.81%) | 5,086 |
| 3 | 1,506,041 | 3.18 | 31.49% | 3,277,113 (68.51%) | 1,405 |
| 4 | 3,459,854 | 1.38 | 72.33% | 1,323,300 (27.67%) | 731 |
| 5 | 4,783,154 | 1.00 | 100% | 0 | 1 |

Two things to notice. Depth 2 uses 57,122 of 65,536 possible cells, so the codes are
clustered rather than uniform and the uniform expectation of 100% usage does not
hold. Depth 5 is injective, so every item is distinguishable at the full five codes,
which is the property the tokeniser was designed for.

## Lift

| depth | items per bucket (query side) | recall | uniform base | lift |
|---|---|---|---|---|
| 1 | 19,308.02 | 18.3156% | 0.0391% | 46.9x |
| 2 | 345.88 | 2.5167% | 0.00175% | 1437.6x |
| 3 | 27.49 | 0.5483% | 0.0000664% | 8257.6x |
| 4 | 4.43 | 0.1302% | 0.0000289% | 4505.0x |
| 5 | 1.00 | 0.0173% | 0.0000209% | 827.2x |

0 unmapped queries and 0 targets outside the catalogue, so nothing is being lost at
the edges.

Three findings.

**Lift peaks at depth 3 and then falls.** Level 4 halves the bucket from 3.18 to
1.38 items and loses 45% of the lift doing it, which means the fourth code does not
make the target more likely to share the query's prefix. It only splits the bucket.
That is the signature of an arbitrary disambiguator rather than a semantic grouping,
and level 5 is the same thing taken further.

**The first code is worth 46.9x chance and holds 18,684 items.** Any method that
reads one code starts from that ceiling. This is the number a wider SID has to
improve on, compared at matched bucket size or at peak lift, never as raw recall.

**Depth 3 is where the partition stops meaning anything new.** 1,506,041 buckets
over 4,783,154 items, 68.51% of items sharing their prefix with someone, and the
highest lift in the family. So three real levels carry almost all the signal and the
fourth is already half arbitrary.

## A trap that cost an hour

Codes are stored as `int32`. Mixed-radix packing at depth 4 reaches
`255 * 256**3 = 4,278,190,080`, past the `int32` ceiling, and on Windows a C long is
32 bits as well. The first run reported depth 4 at 0.0621% recall with only 49.8% of
queries finding a bucket, and depth 5 died with
`OverflowError: Python int too large to convert to C long`. Both were artefacts.
`load_codes` now casts to `int64` and `pack_keys` refuses a radix product at or
above `2**62`, with a test pinning the `int32` case.

## The next configuration

The user's instinct was four levels at 1024. The width is right, the level count is
not: at 1024 wide the third level is already effectively injective, so a fourth
level becomes the arbitrary suffix that levels 4 and 5 are today, one level earlier.

Expected values are uniform expectations over 4,783,154 items; the measured
distribution is clustered, so treat the counts as lower bounds on items per bucket.

| config | bits | SID tokens per event | embedding rows | params per item | depth-1 items | last level items |
|---|---|---|---|---|---|---|
| today, 4 x 256 + 731 | 32 + 9.5 | 5 | 1,755 | 0.376 | 18,684.20 | 1.00 |
| 4 x 1024 | 40 | 4 | 4,096 | 0.877 | 4,671.0 | 1.0022 at depth 3, level 4 redundant |
| 3 x 1024 | 30 | 3 | 3,072 | 0.658 | 4,671.0 | 1.0022 |
| 2 x 4096 | 24 | 2 | 8,192 | 1.754 | 1,167.8 | 1.149 |
| 2 x 16384 | 28 | 2 | 32,768 | 7.014 | 291.9 | 1.017 |

Recommended: **three codebooks of 1024, no unique-id suffix**, and the alternative
worth one run is **two codebooks of 4096**.

Three levels at 1024 is 30 bits of addressing for 4,783,154 items, which needs 22.2,
so there is 7.8 bits of slack and the last level is nearly injective on its own.
Every level stays semantic, the arbitrary suffix disappears, and parameters per item
go from 0.376 to 0.658. It also saves two tokens per event, which is exactly the
budget the richer event stream wants back: with an action token and a gap token the
count is `1 + 3 + 2 = 6` per event, identical to today's `1 + 5`.

Two codebooks of 4096 is the token-efficient option. In two tokens it reaches 1.149
items per bucket, which is finer than today's depth 4 at 1.38, so it does in two
tokens what the current SID needs four for, and it carries 4.7x the item-side
parameters. The cost is that the top-level bucket drops to 1,168 items, so the
coarse grouping the head sees first is much finer and the second level does more of
the work. If it holds 46.9x lift at depth 1 it is the better SID; if lift collapses
the width outran the geometry.

Both are falsifiable in the same way, and the same script decides: run the release,
then compare lift at matched bucket size against 46.9x at depth 1 and 8257.6x at
peak. Flat lift means the width bought resolution and nothing else, which makes it a
capacity change and not a quality change. `docs/plan_v2.md` states both predictions
before the runs.

### Command

```bash
python "SID Generation/rq_vae_cli.py" \
  --input_path <256-d item embeddings> \
  --input_dim 256 \
  --hidden_dim 64 \
  --num_codebooks 3 \
  --codebook_size 1024 \
  --stage_weights 2.0 1.5 1.0 \
  --epochs 50 \
  --batch_size 4096
```

Two details that matter.

`--add_unique_id` is omitted on purpose. At 30 bits the suffix has nothing to
resolve, and leaving it on repeats the 731-value level.

`--stage_weights` must match `--num_codebooks`. The default is four weights,
`[2.0, 1.5, 1.0, 1.0]`, and the loss divides by `sum(stage_weights)` while the
per-stage loop zips codebooks against weights. With three codebooks the used weights
sum to 4.5 but the divisor stays 5.5, so the entropy term runs 18% weaker than
intended, silently. Passing three weights avoids it.

Also do not omit `--input_dim`. Its default is 1024 while this embedding is 256-d.

## Reproduce

```bash
python "Plan 2/sid_diagnose.py" --data_dir local/2026-20-10-2-pre/digr_data \
  --json "Plan 2/report_content_sid.json"
```

About a minute on CPU. After the next SID is preprocessed, point the same command at
the new `digr_data` directory and compare the lift table above against it.

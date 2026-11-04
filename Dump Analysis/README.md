# Dump Analysis

Metrics over the per-row rank dumps, for the questions a single `R@10` column
cannot answer. numpy only, CPU only, no model, no GPU.

Two files:

- `dump_metrics.py` the tool
- `test_dump_metrics.py` 28 tests, hand-checkable fixtures

```bash
python -m pytest "Dump Analysis/test_dump_metrics.py" -q
```

## What it reads

Any npz holding `rows`, `targets`, `rank`, which is what all three producers
write:

| producer | flag | keys |
|---|---|---|
| `Model Training/eval_full_catalog.py` | `--dump_ranks` | rows, targets, rank, n_rows, split, checkpoint, catalog_size, kind |
| `Model Training/baselines_digr.py` | `--dump_ranks DIR` | rows, targets, rank per `<kind>_<lo:06d>.npz` chunk |
| `Model Training/train_tiger.py --mode beam` | `--dump_rows` | the above plus tuples, beam_width, sid_sizes, covered0 |

An arm may be one file, a glob, or a directory of chunks. Chunks are stacked and
reordered by `rows`, so a fork pool that delivers them out of order is fine. A
gap, a duplicate, or two arms disagreeing on the same row's target is a hard
exit rather than a silent pairing.

## What it reports

| section | why it exists |
|---|---|
| curve | `R@k`, `nDCG@k`, `MRR@k` for any cut-offs you pass, from one rank array |
| `list` | how many candidates the arm ranked per row. For a beam it is the summed size of the cells it named, counted from the catalogue; for the rest it is `--candidates` or the cap |
| `nDCG@k / R@k` | the mean weight of a hit, so it says where inside the top k hits land. DiGR 0.5988 against co-click 0.5513 while `R@10` is a tie, which is the whole point of reporting nDCG |
| effective rank | `2 ** mean(log2(rank))`, uses the entire list instead of one cut-off |
| pair union | `|hit_A ∪ hit_B| / n` for every pair, the only view that shows how differently two arms fail |
| McNemar | paired significance on the same rows, which is the right test here and much sharper than comparing two unpaired proportions |
| popularity strata | `R@k` by target frequency quintile, ordered cold to hot. The honest answer to "is this just popularity" |
| groups | optional per-row grouping, from any npy of group ids |
| cells | when the dump carries beam tuples: cells named per row, distinct cells, cross-row Jaccard |

## Conventions, inherited not invented

`rank` is 1-based and an arm that could not reach the target stores `cap + 1`, so
a row is a hit at `k` when `rank <= k` and `k <= cap`.

**Cut-offs wider than `cap` are clamped.** For a TIGER beam the cap is the beam
width, so asking a width-20 beam for `R@100` returns its `R@20`. Without this,
`rank <= 100` would score a stored miss as a hit and print a perfect score. This
is the single most common way to inflate a beam, and there is a test for it.

The cap is `beam_width` when the dump has one, otherwise `rank.max() - 1` when
`rank.max()` looks like a sentinel, otherwise `rank.max()`. Override with
`--cap NAME=N` when you know better. A clamped cut-off prints a note to stderr.

**A beam's cell positions are converted to item ranks.** A beam orders cells and
cannot order the items inside one, so a raw cell position is not an item position
and `R@1`, `nDCG@10` and `MRR@10` mean something different from what they mean for
an arm that ranks items. When `--data_dir` is given, any arm whose dump carries
`tuples` is converted: the members of a cell are taken in **catalogue order**,
which is the tie-break `topk` already applies inside `train_digr.evaluate`, and an
item's rank becomes the summed size of every cell above it plus its position
inside its own cell. No best case, no worst case. The same code reproduces the v1
beam's cut-offs exactly (1.7558% / 5.0741% / 7.4849%), because a v1 cell holds one
item, which is the check that the conversion is implemented correctly. `--item_axis
off` keeps the cell axis instead, and then the cap is the beam width again.

**A miss is charged as a miss.** Whether the arm could have reached the item
given a wider list is a different question, answered by
`Model Training/budget_match.py`. Effective rank is computed over hits only,
because substituting the sentinel for an unknown rank flatters the arm.

## Not reported, and why

- `precision@k`, `MAP`, `HR@k`: with one relevant item per row they are `R@k`
  under another name.
- diversity and serendipity: they need item content, which this project excludes.
- item-level catalogue coverage: a rank dump does not record which items were
  named, only where the target landed. `budget_match.py` recovers budget
  information from `catalog_sid_codes.npy` instead.

## Worked example

The v2 table, on the dumps in `local/2026-29-10-1-full-eval`:

```bash
python "Dump Analysis/dump_metrics.py" \
  --arm "TIGER_beam=dumps/beam_prefix_w20.npz" \
  --arm "DiGR_v2=dumps/digr_val.npz" \
  --arm "coclick_h20=dumps/baselines/h20/coclick_*.npz" \
  --arm "popularity=dumps/baselines/h20/popularity_*.npz" \
  --data_dir ~/NZ5/plan2/digr_data \
  --candidates DiGR_v2=4783154 --candidates coclick_h20=10000 \
  --ks 1,5,10,20,100,1000 --out_dir analysis
```

`--data_dir` does two jobs: it supplies `train/targets.npy` for the popularity
strata, and it supplies `catalog_sid_codes.npy`, which is what converts the beam's
cell positions to item ranks. Without it the beam keeps the cell axis and its cap
stays at the beam width. `--candidates` is only needed for arms whose list length
the dump cannot reveal, which is every arm that ranks items directly.

Two things this prints that the headline table does not:

- **DiGR against co-click is not significant at `R@10`.** 2,535 rows only DiGR
  reaches, 2,584 only co-click, `chi2` 0.45, `p` 0.50. Both beats popularity by
  a wide margin, but neither beats the other at that cut-off.
- **Popularity collapses on cold targets, DiGR does not.** The popularity arm
  scores exactly 0.0000% on targets that never appear in train, because its list
  is one fixed order. DiGR gets 0.1973% there, co-click 0.1400%. That is the
  content-only claim holding up where it has to.

## Exit codes

Every refusal is `SystemExit` with a message naming the arm, the file and the
bad value: `matched no files`, `holds more than one baseline`, `does not cover
rows 0..N exactly once`, `disagrees with A on N targets`, `entries`. A refusal
writes nothing.

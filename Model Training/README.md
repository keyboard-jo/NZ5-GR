# Model Training

Code that turns the preprocessed splits into numbers. Everything runs on the
cluster; nothing here trains locally.

## Pipelines

| file | what it does |
| --- | --- |
| `preprocess_digr.py` | events to token rows, catalog and vocab |
| `preprocess_candidate_digr.py` | candidate list to pool ids |
| `train_digr.py` | DiGR, LLM backbone over five-code SIDs: train, infer, export |
| `eval_full_catalog.py` | score one checkpoint over all 4,783,154 items |
| `baselines_digr.py` | popularity, SID transition, co-click. No model, no GPU |
| `embedding_knn_nextclick.py` | Stage-1 embedding kNN against the real next click |

## Folders

- `slurm/` sbatch wrappers. Submit them from this directory:
  `sbatch slurm/run_baselines.sh`
- `notebooks/` the EDA that sized the data and analysed SID quality

## Tests

```
.venv/Scripts/python.exe -m pytest "Model Training" -q
```

Mock data, fake backbones, no GPU, and never a real training run.

## Where the numbers live

- `../docs/README.md` the live record: the matched-budget result and the direction
- `../docs/archive/` everything written before 2026-10-29, frozen as history
- `../docs/archive/results.md` older result tables (superseded in part, see `docs/README.md`)
- `../docs/archive/math.md` formulas, and what each baseline actually optimises
- `../docs/archive/sid_results.md` codebook and embedding geometry
- `../docs/archive/tasks.md` the pre-archive task list

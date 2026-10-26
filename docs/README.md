# Docs

Index only. Each file stands alone.

| file | what is in it |
|---|---|
| `math.md` | the formulas: popularity, SID transition, co-click kNN, DiGR, and why each behaves the way it does |
| `data.md` | the numbers behind them: catalog, splits, vocab, token layout, what a row of tokens is |
| `sid_math.md` | how the item embeddings and the Semantic IDs are produced |
| `results.md` | the measured table, every method on the same 98,304 rows, with compute cost |
| `problems.md` | what those results mean for the direction of the project, including what went wrong |
| `tasks.md` | what is left to do, in order, with the reason each item exists |

Current state in one line: a four-second co-click count table beats the 4.6
GPU-hour DiGR model at every cut-off from rank 10 to rank 1,000, and DiGR wins only rank 1 (7 rows) and the far tail.
Read `results.md` for the numbers and `problems.md` for what that implies.

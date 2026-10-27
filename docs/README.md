# Docs

Index only. Each file stands alone.

| file | what is in it |
|---|---|
| `math.md` | the formulas: popularity, SID transition, co-click kNN, DiGR, and why each behaves the way it does |
| `data.md` | the numbers behind them: catalog, splits, vocab, token layout, what a row of tokens is |
| `sid_math.md` | how the item embeddings and the Semantic IDs are produced |
| `sid_diagnostics.md` | how to judge a generated SID for our use of it: coverage, bucket lift, per-level value, tie ceiling, geometry, and the failure signatures |
| `results.md` | the measured table, every method on the same 98,304 rows, with compute cost |
| `problems.md` | what those results mean for the direction of the project, including what went wrong |
| `tasks.md` | what is left to do, in order, with the reason each item exists |
| `plan_v2.md` | the next SID release and the richer event stream: what a wider codebook buys, what the event stream is throwing away, and the predictions in advance |
| `embedding_weakness.md` | why the Stage 1 item embeddings are behaviourally weak, and what OnePiece does differently |
| `recap_supervisor.md` | a self-contained status snapshot for the co-supervisor, current as of 2026-10-26 |
| `../Plan 2/README.md` | the SID diagnostic script, how to read bucket lift, and the content SID measured |

Current state in one line: a 47-second co-click count table beats the 4.6
GPU-hour DiGR model at every cut-off except rank 10,000, the DiGR head reads an item
through a code shared by 18,684 items, and widening the codebook from 256 to 8,192
rows took bucket lift from 46.9x to 463.3x on the same embedding, so the endorsed SID
is 2 x 8,192 and the next release pairs it with an event stream that can tell an
exposure from a click, since 89.4% of history events are exposures the model
currently cannot distinguish. Read `results.md` for the numbers, `problems.md` for
what they imply, `plan_v2.md` for what is being changed next and what it is expected
to buy, and `sid_diagnostics.md` before judging any new SID.

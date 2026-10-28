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

Current state in one line: on the endorsed 2 x 8,192 SID with an action token in the
stream, DiGR leads co-click at rank 1 (1.05% against 0.81%), is level with it at
ranks 10 and 100, and leads again at 1,000 and 10,000 (27.66% against 25.30%, 55.71%
against 46.89%), for 3.2 GPU-hours against 47 seconds. Widening the codebook is what
moved it: 18,684 items per level-0 code became 584 and bucket lift went from 46.9x
to 463.3x on the same embedding, while the Stage 1 towers stay below a frequency
count. Read `results.md` for the numbers, `problems.md` for what they imply,
`plan_v2.md` for what changes next and what it is expected to buy, and
`sid_diagnostics.md` before judging any new SID.

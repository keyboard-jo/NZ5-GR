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

Current state in one line: the three-arm table is complete on the endorsed 2 x 8,192
SID, and the answer is split by cut-off. TIGER owns rank 1 (1.44% against DiGR's
1.05% and co-click's 0.81%), DiGR owns ranks 10 to 10,000 (3.66% and 10.92% level
with co-click, then 27.66% and 55.71% against 25.30% and 46.89%), and the count
table stays in front of both from rank 10 to 1,000 for 47 seconds against 3.2 GPU
hours. Widening the codebook is what moved DiGR: 18,684 items per level-0 code
became 584 and bucket lift went from 46.9x to 463.3x on the same embedding, while
the Stage 1 towers stay below a frequency count. At two levels a prefix-free score
is additive over two shared code rows, so DiGR and prefix-free TIGER sit in one
function class and their gap is a loss difference, not a factorisation. Read
`results.md` for the numbers, `problems.md` for what they imply, `plan_v2.md` for
what changes next and what it is expected to buy, and `sid_diagnostics.md` before
judging any new SID.

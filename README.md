# chunkbench

**Which chunking strategy actually retrieves best?** Benchmark fixed-size,
overlapping, recursive, and semantic chunking for RAG on retrieval recall — same corpus,
same ranker, only the chunking changes.

Pure Python. Zero dependencies.

## Why this exists

Every RAG tutorial says "chunk your documents" and moves on. In production,
chunking is a retrieval hyperparameter: split an answer across a chunk
boundary and recall drops even though the text is technically in the index.
chunkbench makes the tradeoff measurable with a labeled corpus where each
question names its gold document and the exact answer phrases a retrieved
chunk must contain.

## Install

```bash
pip install .
# or just run it from the repo:
python -m chunkbench demo
```

## Demo (real output)

```bash
$ python -m chunkbench demo
================================================================
 chunkbench demo: which chunking strategy retrieves best?
================================================================

corpus: 8 production-ML docs, 20 labeled questions
retrieval: identical TF-IDF ranker for every strategy (only chunking varies)

strategy    recall@1  recall@3  recall@5     MRR  chunks avg chars
------------------------------------------------------------------
fixed          0.600     0.850     0.850   0.708      25       464
recursive      0.650     0.850     0.850   0.750      32       334
overlap        0.600     0.750     0.800   0.671      29       443
semantic       0.600     0.750     0.750   0.667      38       282
------------------------------------------------------------------
winner: recursive (recall@5 0.850, MRR 0.750, 32 chunks)

questions no strategy retrieved in top 10:
  [fixed] In LoRA, what does the alpha hyperparameter control?
  [fixed] Why must feature stores enforce point-in-time correctness?
  [fixed] What is the difference between canary and shadow deployments?
  [overlap] In LoRA, what does the alpha hyperparameter control?
  [overlap] Why must feature stores enforce point-in-time correctness?
  [overlap] What is the difference between canary and shadow deployments?
  [overlap] Why do vector databases need metadata filtering?
  [recursive] In LoRA, what does the alpha hyperparameter control?
  [recursive] Why must feature stores enforce point-in-time correctness?
  [semantic] In LoRA, what does the alpha hyperparameter control?
  [semantic] Why must feature stores enforce point-in-time correctness?
  [semantic] What is the difference between canary and shadow deployments?
  [semantic] What should every ML deployment gate define in advance?
  [semantic] What reward structure makes prefix caching effective?
```

The honest takeaway: boundary-aware recursive chunking beats the naive
sliding window, and fine-grained semantic chunks can *hurt* — smaller chunks
dilute term weights in the ranker. Widening the sliding window's overlap to
20% does not rescue it either: it lands between fixed and semantic on this
corpus. Chunking is a tradeoff, not a ladder.

## Usage

```bash
# list strategies
chunkbench strategies

# benchmark everything on the built-in corpus
chunkbench run

# one strategy, custom chunk size
chunkbench run --strategy recursive --size 800

# overlap strategy with a custom overlap (default: 20% of --size)
chunkbench run --strategy overlap --overlap-chars 100

# your own corpus: a directory of .md files + a questions JSON
chunkbench run --docs ./my_docs --questions ./questions.json --json

# sweep a grid of window sizes x overlap fractions, pick the best config
chunkbench sweep --strategy fixed,overlap --sizes 250,500,1000 --overlaps 0,0.1,0.25
chunkbench sweep --sizes 250,500,750,1000 --overlaps 0,0.25 --json
```

Questions JSON format:

```json
[
  {"text": "where does a gift card refund go?",
   "gold_doc": "refunds",
   "must_contain": ["store credit"]}
]
```

A hit requires a top-k chunk from `gold_doc` containing every phrase, so
answers split across chunk boundaries count as misses.

## Strategies

| strategy    | how it works |
|-------------|--------------|
| `fixed`     | sliding character window (default 500 chars, 50 overlap) |
| `overlap`   | sliding character window with a wide overlap (default 500 chars, 100 overlap = 20% of window), tunable via `--overlap-chars` |
| `recursive` | split on the strongest boundary that fits (paragraph, line, sentence), then greedily merge back up to the size limit |
| `semantic`  | sentence grouping that closes a chunk early on topic shift (cosine similarity of word-count vectors below 0.15) |

All chunks carry exact character offsets back into the source document, so
results are auditable down to the span.

## Sweep: find the best config per corpus

`chunkbench sweep` runs the benchmark across a grid of window sizes and
overlap fractions (fractions of the window — `0.25` means 25% overlap, so
the grid stays meaningful as the window grows), ranks every config by
recall@5 then MRR, and names the winner:

```bash
$ chunkbench sweep --strategy fixed,overlap --sizes 250,500,1000 --overlaps 0,0.1,0.25
sweep: 18 configs (2 strategies x 3 sizes x 3 overlaps)

strategy        size  overlap   recall@1   recall@3   recall@5     MRR  chunks
--------------------------------------------------------------------------
 * fixed        1000      100     0.600     0.900     0.900   0.875      16
   overlap      1000      100     0.600     0.900     0.900   0.875      16
   ...
--------------------------------------------------------------------------
best config: fixed size=1000 overlap=100 (recall@5 0.900, MRR 0.875, 16 chunks) — marked * above
```

The winner is a fact about *this* corpus, not a recommendation — larger
windows dilute term weights in the ranker, so sweeping on your own corpus
is the whole point. `semantic` sweeps over sizes only (overlap is n/a for
it); grids over 200 runs are rejected so a typo never turns into an
afternoon of benchmarking. `--json` emits every config's scores plus the
best config as machine-readable data.

## Tests

```bash
python -m pytest tests/ -q
# 37 passed
```

## Layout

```
chunkbench/
  chunkers.py    # fixed / overlap / recursive / semantic strategies
  corpus.py      # built-in 8-doc production-ML corpus + 20 labeled questions
  retrieval.py   # pure-Python TF-IDF ranker (identical for every strategy)
  eval.py        # benchmark harness: recall@k, MRR, per-question ranks
  report.py      # terminal table + JSON reports
  cli.py         # demo / run / strategies commands
examples/        # sample custom corpus (docs/*.md + questions.json)
tests/           # 37 tests
```

# Benchmarks

`bench.py` measures softrag's own overhead: ingestion throughput, query latency
at several corpus sizes, and bytes on disk per chunk.

```bash
python benchmarks/bench.py                          # 1k / 10k / 50k chunks
python benchmarks/bench.py --sizes 100000 --queries 100
python benchmarks/bench.py --dimensions 1536 --json results.json
```

The embedder is `HashEmbedder`, which is local and fast. That is deliberate: with
a real embedding API in the loop, every number would be a measurement of that
API's latency instead of this library's. Add the embedding cost of your own model
on top of everything below.

## Results

Windows 11, Python 3.12, 384-dimension vectors, ~600-character chunks, 30 queries
per configuration, `top_k=5`. Latency is the median (p50) in milliseconds.

| chunks | ingest | chunks/s | db size | bytes/chunk |
| -----: | -----: | -------: | ------: | ----------: |
|  1,842 |  0.77s |    2,392 |    5.1M |       2,753 |
| 18,307 |  6.91s |    2,651 |   45.6M |       2,492 |

| chunks | hybrid | vector | keyword | + filter | + MMR |
| -----: | -----: | -----: | ------: | -------: | -----: |
|  1,842 |  2.3ms |  1.6ms |   0.4ms |    2.5ms |  6.3ms |
| 18,307 | 17.1ms | 13.4ms |   3.1ms |   21.0ms | 24.7ms |

Those are after the fixes below. For scale, the same benchmark against the
first working version of this rewrite reported 26.7 ms hybrid, 9.6 ms keyword,
75.6 ms filtered and 48.5 ms with MMR at a comparable corpus size.

Disk cost is dominated by the vectors: 384 float32 values is 1,536 bytes, so
~2.5 KB per chunk means the text, metadata and both indexes together add roughly
1 KB. A 1,536-dimension model costs about 6 KB per chunk instead.

## How to read these numbers

**Vector search is exhaustive, and therefore linear.** `sqlite-vec` scores every
vector rather than maintaining an approximate index (no HNSW, no IVF), so
latency grows in proportion to the corpus: 10x the chunks is about 10x the
search time. That is a real ceiling, and it is the honest reason softrag is
positioned for small and medium corpora. As a rule of thumb, expect the low tens
of milliseconds around 20k chunks and a few hundred at 500k. Past roughly a
million vectors, a dedicated ANN store is the right tool.

The upside of exhaustive search is that recall is exactly 1.0. There is no
`ef_search` to tune, no index build step, and no accuracy that quietly degrades
as you add documents.

**Filtered search costs about the same as unfiltered.** Getting there took some
care. A filter resolves against the ordinary `documents` table first; if it
matches few enough rows, those rows are then scored exactly, one keyed lookup at
a time. The obvious implementation — collecting the ids and asking for
`doc_id IN (...)` — is a trap: `vec0` answers an `IN` with a full table scan, so
batching ids that way was measured at 75ms where the current code takes 28ms.

**MMR is opt-in for a reason.** `diversity>0` adds several milliseconds: it
loads the candidate vectors back and runs the greedy selection in pure Python.
Worth it when your corpus contains near-duplicate passages; wasted otherwise.

**Keyword search got faster by doing less.** Dropping stopwords and
high-document-frequency terms before building the FTS5 query cut keyword
latency from 9.6 ms to 3.1 ms, because far fewer documents match in the first
place. That the same change also improved retrieval quality is not a
coincidence: the candidates it stopped fetching were the noisy ones.

**Ingestion throughput here is not yours.** ~2,300 chunks/second is the storage
layer alone. With a real embedding model, ingestion is bounded almost entirely by
that model — an API call or a GPU forward pass, both orders of magnitude slower
than the SQLite write.

## Reproducing

Numbers vary with disk, CPU and vector width. Re-run with your own dimensions
before quoting anything:

```bash
python benchmarks/bench.py --dimensions 1536 --sizes 10000 50000
```

## Retrieval quality

`retrieval_quality.py` asks whether hybrid search earns its place. It uses a
small labelled set built so each failure mode is present: queries with a rare
exact token (`ERR_4417`), paraphrases sharing no content words with the
document that answers them, and queries with both.

MRR over all 13 queries, 8 documents:

| mode | HashEmbedder (weak) | MiniLM (strong) |
| ---- | ------------------: | --------------: |
| keyword only | 0.615 | 0.615 |
| vector only  | 0.564 | **1.000** |
| hybrid       | **0.673** | 0.885 |

Read honestly, that says two things.

**Hybrid is the right default, not the best answer to every case.** When the
embedder is weak, hybrid beats either half (+0.058 MRR over the best single
mode). When the embedder is strong and the corpus is small and topically
distinct — every document about a different subject — dense retrieval already
answers everything, and mixing in a weaker ranked list can only cost you
(-0.115). A larger corpus with many documents per topic moves the balance back,
because that is where exact terms start to matter for telling near-duplicates
apart.

The practical advice is to measure on your own corpus rather than trust either
default. That is what `softrag.eval` is for:

```python
from softrag import compare

results = compare(
    rag,
    my_labelled_queries,
    variants={
        "hybrid": {"mode": "hybrid"},
        "vector": {"mode": "vector"},
        "keyword": {"mode": "keyword"},
    },
)
```

**Building the keyword query is where most of the quality lives.** An OR over
every token in *"when can we ship code to customers"* matches whatever contains
*can*, *we* and *to*, and rank fusion then promotes that noise above correct
dense hits — measurably worse than doing no keyword search at all. BM25 cannot
save it, because BM25 only reweights documents that already matched. Dropping
stopwords and terms above a document-frequency cutoff before the MATCH is built
took the `mixed` group from 0.875 to 1.000 nDCG@3 and made fusion weights stop
mattering at all — a good sign that the noise, not the weighting, was the
problem.

```bash
python benchmarks/retrieval_quality.py                    # instant, no downloads
python benchmarks/retrieval_quality.py --embedder local   # real embeddings
```

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
|  2,002 |  0.83s |    2,419 |    5.3M |       2,627 |
| 19,941 |  8.80s |    2,266 |   50.8M |       2,545 |

| chunks | hybrid | vector | keyword | + filter | + MMR |
| -----: | -----: | -----: | ------: | -------: | ----: |
|  2,002 |   2.7ms |  1.7ms |   1.0ms |    3.1ms | 10.0ms |
| 19,941 |  26.3ms | 15.8ms |   7.8ms |   28.8ms | 36.8ms |

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

**MMR is opt-in for a reason.** `diversity>0` adds roughly 10ms, because it
loads the candidate vectors back and does the greedy selection in pure Python.
Worth it when your corpus contains near-duplicate passages; wasted otherwise.

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

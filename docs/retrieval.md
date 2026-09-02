# Retrieval

Everything about what comes back from `rag.search()` and, therefore, what a
generated answer is built from.

- [The three modes](#the-three-modes)
- [Reciprocal Rank Fusion](#reciprocal-rank-fusion)
- [Reading the scores](#reading-the-scores)
- [Tuning](#tuning)
- [Diversity (MMR)](#diversity-mmr)
- [Context expansion](#context-expansion)
- [Metadata filters](#metadata-filters)
- [Reranking](#reranking)
- [Is it retrieval or generation?](#is-it-retrieval-or-generation)

## The three modes

```python
rag.search("query", mode="hybrid")    # default: both, fused
rag.search("query", mode="vector")    # dense only
rag.search("query", mode="keyword")   # BM25 only
```

**Vector** search embeds the query and finds chunks whose embeddings are closest
by cosine distance. It matches *meaning*: "how do I get my money back" retrieves
a paragraph about refunds even with no shared words. It is bad at rare literal
tokens — an error code, a SKU, a surname, a version string — because those carry
little semantic signal and embedding models routinely blur them together.

**Keyword** search is SQLite's FTS5 with BM25 ranking. It matches *tokens*, so
it nails `ERR_CONN_4021` and misses paraphrases entirely. softrag quotes each
query term as a phrase and joins them with `OR`, which makes any user text a
valid FTS5 query (an unescaped `NOT` in a question is otherwise a syntax error)
and means documents matching more of the query rank higher rather than
documents needing every term.

**Hybrid** runs both and fuses the results. It is the default because the two
failure modes are almost disjoint.

Mode falls back automatically: if no query vector is available, hybrid degrades
to keyword rather than returning nothing.

## Reciprocal Rank Fusion

The two retrievers return incomparable numbers. FTS5's BM25 score is negative,
unbounded and corpus-dependent; cosine distance is bounded in `[0, 2]`. Adding,
averaging or min-max normalising them lets whichever scale happens to be larger
dominate the ranking for reasons that have nothing to do with relevance. RRF
sidesteps the problem by discarding magnitudes and keeping only ordering — the
part both retrievers actually agree on:

```
                          weight_i
score(d) =   Σ        ─────────────────
           lists i     k + rank_i(d)
```

where `rank_i(d)` is `d`'s 1-based position in list `i` (absent documents
contribute nothing) and `k = 60`, the constant from Cormack et al. (2009). `k`
damps the top of each list: with `k = 60`, rank 1 contributes `1/61 ≈ 0.0164`
and rank 5 contributes `1/65 ≈ 0.0154` — close enough that a document both
retrievers rank in their top ten beats a document only one retriever loves.

```python
from softrag import reciprocal_rank_fusion

reciprocal_rank_fusion([[1, 2, 3], [3, 1]])
# [(1, 0.03252...), (3, 0.03227...), (2, 0.01613...)]
```

Document `1` wins: it is first in one list and second in the other. Document `3`
is nearly tied — third in one list but second in the other. Document `2`, seen
by only one retriever, trails both.

The weights come from `vector_weight` and `keyword_weight`. A weight of `0`
excludes that list entirely, which is a softer way to spell `mode="vector"`.

## Reading the scores

`hit.score` means something different per mode. This trips people up:

| Mode      | `hit.score`                            | Typical range          |
| --------- | -------------------------------------- | ---------------------- |
| `hybrid`  | fused RRF score                        | ~0.008 – 0.033         |
| `vector`  | `1 - distance/2`, i.e. cosine similarity | 0 – 1                |
| `keyword` | BM25, min-max normalised over the result set | 0 – 1            |

Hybrid scores are *not* similarities and are not comparable between queries.
They are only meaningful as an ordering. If you need a calibrated relevance
number — for a confidence threshold, say — use `mode="vector"` and read
`hit.vector_distance`, or add a [reranker](#reranking) whose scores mean
something.

Every hit also carries the provenance of its ranking:

```python
for hit in rag.search("connection timeout", top_k=5):
    print(round(hit.score, 4), hit.ranks, hit.vector_distance, hit.bm25)
# 0.0328 {'vector': 1, 'keyword': 1}  0.5527  -1.81e-06
# 0.0161 {'vector': 2}                1.0     None
```

`hit.ranks` is the fastest way to see which retriever found something. A hit
with only a `keyword` rank was invisible to your embedding model, and vice
versa.

## Tuning

Every knob has a default in `RagConfig`, an override per engine, and an override
per call.

```python
import softrag
from softrag import RagConfig

# per engine
rag = softrag.connect("kb.db", top_k=8, diversity=0.3)

# or explicitly
rag = softrag.connect("kb.db", config=RagConfig(top_k=8, candidates=100))

# per call — always wins
rag.search("query", top_k=3, mode="vector", candidates=200)
```

| Knob             | Default          | What it does                                                     |
| ---------------- | ---------------- | ---------------------------------------------------------------- |
| `top_k`          | `5`              | Hits returned, and context blocks handed to the model.           |
| `mode`           | `"hybrid"`       | `hybrid` / `vector` / `keyword`.                                 |
| `candidates`     | `max(4*top_k,20)`| Candidates *each* retriever contributes before fusion.           |
| `vector_weight`  | `1.0`            | RRF weight of the dense list.                                    |
| `keyword_weight` | `1.0`            | RRF weight of the BM25 list.                                     |
| `diversity`      | `0.0`            | MMR strength; `0` disables the pass entirely.                    |
| `expand_context` | `0`              | Neighbouring chunks glued onto each hit.                         |

**`top_k`.** The cost of a large `top_k` is context length, not retrieval time.
Three to five chunks of ~1000 characters suits a focused factual question; ten
or more suits "summarise everything we know about X". Too many chunks measurably
hurts answer quality — the relevant sentence gets buried.

**`candidates`.** This is the cheapest quality knob in the library. Fusion can
only rank documents it was given, so a document at BM25 rank 25 is invisible to
RRF when each list is 20 long. Raising `candidates` to 100 or 200 costs
milliseconds on an index of hundreds of thousands of chunks and materially
improves hybrid recall. It never changes how many hits you get back — that is
`top_k`.

**Weights.** Leave them at `1.0` unless you have evidence. When you do, the
usual evidence is a corpus of identifiers (code, SKUs, log lines) where
`keyword_weight=2.0` helps, or a corpus of prose questions phrased nothing like
the source text where `vector_weight=2.0` helps. Measure with `search()` on real
queries before committing.

## Diversity (MMR)

The classic hybrid-search failure is five hits that are five near-identical
copies of the same paragraph — a duplicated FAQ entry, a changelog repeated in
three release notes. Maximal Marginal Relevance fixes it by greedily picking, at
each step, the candidate maximising:

```
(1 - diversity) * sim(query, c)  -  diversity * max sim(c, already_picked)
```

```python
rag.search("deployment steps", top_k=5, diversity=0.4)
```

`0` disables MMR entirely (the default). `0.2`–`0.5` is the useful band: enough
to break up duplicates, not so much that the best hit gets displaced by
something merely different. `1.0` maximises dissimilarity and ignores relevance,
which is almost never what you want.

MMR runs over the fused candidate list and needs vectors, so it is a no-op in
`mode="keyword"`.

## Context expansion

Chunking cuts documents at arbitrary places, and the sentence that answers the
question is sometimes one line past the boundary. `expand_context` widens each
hit with its neighbouring chunks from the same source:

```python
hits = rag.search("what is the escalation path", top_k=3, expand_context=1)
```

A radius of `1` glues on the chunk before and the chunk after; `2` takes two on
each side. Neighbours are joined in document order and deduplicated across hits,
so two adjacent hits do not hand the model the same paragraph twice.

The cost is context length — `expand_context=1` roughly triples the text per
hit — so lower `top_k` when you raise it. Ranking is unaffected: expansion
happens after selection, so the extra text never influences which chunks were
picked.

## Metadata filters

Filters are plain dicts compiled to parameterised SQL over the JSON metadata
column. Values are bound as parameters, never interpolated, so user input in a
filter is safe by construction.

```python
rag.search("budget", where={"team": "finance", "year": {"$gte": 2024}})
rag.query("budget", where={"$or": [{"pinned": True}, {"year": 2025}]})
rag.delete(where={"status": "archived"})
```

### Operators

| Operator    | Example                                | Meaning                                    |
| ----------- | -------------------------------------- | ------------------------------------------ |
| *(bare)*    | `{"team": "legal"}`                    | Equality.                                  |
| `$eq`       | `{"year": {"$eq": 2025}}`              | Equality, explicit.                        |
| `$ne`       | `{"status": {"$ne": "draft"}}`         | Not equal.                                 |
| `$gt` `$gte`| `{"pages": {"$gt": 10}}`               | Greater than / or equal.                   |
| `$lt` `$lte`| `{"score": {"$lte": 0.5}}`             | Less than / or equal.                      |
| `$in`       | `{"ext": {"$in": [".md", ".rst"]}}`    | Membership. Empty list matches nothing.    |
| `$nin`      | `{"ext": {"$nin": [".log"]}}`          | Negated membership.                        |
| `$like`     | `{"title": {"$like": "%invoice%"}}`    | SQL `LIKE`; `%` and `_` are wildcards.     |
| `$contains` | `{"tags": {"$contains": "urgent"}}`    | Element of a JSON array, or substring of a string. |
| `$exists`   | `{"author": {"$exists": True}}`        | Key present (`False` for absent).          |
| `$and`      | `{"$and": [{...}, {...}]}`             | All must match.                            |
| `$or`       | `{"$or": [{...}, {...}]}`              | Any must match.                            |
| `$not`      | `{"$not": {"status": "archived"}}`     | Negation.                                  |

Two things worth knowing:

- Multiple keys in one dict are ANDed: `{"team": "legal", "year": 2025}`.
- Multiple operators on one field are also ANDed, which is how you express a
  range: `{"year": {"$gte": 2024, "$lt": 2026}}`.
- Dotted field names reach into nested metadata: `{"doc.author.name": "Ada"}`.
- Booleans round-trip as JSON `0`/`1`, so `{"pinned": True}` matches what
  `metadata={"pinned": True}` stored.

Ingestion attaches metadata automatically — `kind`, `filename`, `extension`,
`bytes`, `path` for files; `kind`, `url`, `title` for web pages — so
`where={"extension": ".md"}` works with no bookkeeping of your own.

### Filters do not cost recall

Most vector stores implement a filter by over-fetching approximate neighbours
and discarding the ones that do not match, which silently returns fewer (or no)
results when the filter is selective. softrag resolves the filter *first*
against the indexed `documents` table and scores only the surviving rows
exactly. A filter matching 40 chunks out of a million returns the best of those
40, every time. Above ~20,000 matching rows exact rescoring stops paying for
itself and softrag falls back to over-fetched KNN plus filtering — by which
point the filter is not selective enough for the difference to matter.

There is also a cheaper single-source shortcut, which skips JSON extraction
entirely:

```python
rag.search("query", source="handbook.pdf")
```

## Reranking

A bi-encoder embeds the query and each document separately, so it can index
ahead of time but never sees the pair together. A cross-encoder scores
`(query, document)` jointly — markedly more accurate, markedly slower — which
makes it right as a second stage over a few dozen candidates.

```python
import softrag
from softrag.providers.local import CrossEncoderReranker   # pip install 'softrag[rerank]'

rag = softrag.connect("kb.db", reranker=CrossEncoderReranker())

rag.search("query")                      # reranked
rag.search("query", rerank=False)        # this one is not
rag.search("query", rerank=OtherReranker())   # just this once
```

When a reranker is active, softrag automatically retrieves
`max(candidates, top_k)` hits before reranking and trims to `top_k` after — a
reranker only helps if it is given more to choose from.

Any object with `rerank(query, hits, *, top_k) -> list[Hit]` is a valid
reranker; see [providers.md](providers.md#writing-a-reranker).

## Is it retrieval or generation?

When an answer is wrong, exactly one of two things happened: the right text was
never retrieved, or it was retrieved and the model ignored it. `search()`
settles the question in one call, because it returns precisely what `query()`
would have built its prompt from.

```python
for i, hit in enumerate(rag.search("your question", top_k=5), 1):
    print(f"[{i}] {hit.score:.4f} {hit.ranks} {hit.source}#{hit.index}")
    print(hit.text[:300], "\n")
```

**The answer is not in the hits.** It is a retrieval problem. In rough order of
how often it is the cause:

1. *The content was never indexed.* Check `rag.sources()` and `len(rag)`. A
   silent extraction failure — a scanned PDF with no text layer, a
   JavaScript-rendered page — leaves an `IngestResult` with `.error` set that
   nobody looked at.
2. *A filter is excluding it.* Re-run without `where=` and `source=`.
3. *`candidates` is too low.* Try `candidates=200`. If the hit appears, fusion
   was never shown it.
4. *Wrong mode for the query.* Identifiers and rare tokens: try
   `mode="keyword"`. Paraphrases: try `mode="vector"`. If one mode finds it and
   hybrid does not, adjust the weights.
5. *Chunks are too large.* A 4000-character chunk dilutes its own embedding
   until it matches nothing specifically. See
   [ingestion.md](ingestion.md#choosing-a-chunk-size).
6. *`HashEmbedder` is in use.* If you never configured an embedder and no key or
   daemon was found, softrag warned once and fell back to it. `rag.stats()`
   shows `dimensions=256`, its default width.

**The answer is in the hits but the reply is wrong.** It is a generation
problem: try a stronger chat model, a lower temperature, a stricter prompt, or
fewer chunks (the relevant one may be buried).

```python
answer = rag.query("your question")
print(answer.prompt)     # exactly what the model was sent
print(answer.context)    # exactly the context blocks it saw
```

**A quick A/B harness.** Since `search()` is cheap and needs no chat model, you
can compare settings directly:

```python
for name, kwargs in {
    "hybrid":   {},
    "vector":   {"mode": "vector"},
    "keyword":  {"mode": "keyword"},
    "wide":     {"candidates": 200},
    "diverse":  {"diversity": 0.4},
}.items():
    hits = rag.search("your question", top_k=3, **kwargs)
    print(f"{name:>8}: {[h.source for h in hits]}")
```

Run it over a handful of questions whose right answer you know, and pick the
configuration that puts the right source first most often.

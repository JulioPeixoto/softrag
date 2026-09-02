"""Local embeddings with sentence-transformers, plus a cross-encoder reranker.

A useful middle ground: embeddings run on your machine (no per-token cost, no
data leaving it) while generation can come from wherever you like. This example
pairs them with Ollama so the whole pipeline stays local.

The reranker is the interesting part. First-stage retrieval is a bi-encoder --
query and document embedded separately -- which is what makes indexing possible
at all. A cross-encoder scores the pair together, which is markedly more
accurate and far too slow to run over a whole corpus. Running it over the top
few dozen candidates gets most of the accuracy for a fraction of the cost.

Requires:
    pip install 'softrag[local]'
    Ollama running, for the generation half:  ollama pull llama3.2

The first run downloads the models (~90 MB for MiniLM, ~1.1 GB for the
reranker); after that everything is cached and offline.

Run:
    python examples/local_embeddings.py
"""

from __future__ import annotations

from softrag import Rag
from softrag.providers.local import SentenceTransformerEmbedder
from softrag.providers.ollama import OllamaChat, is_available

CORPUS = {
    "sqlite-wal": (
        "Write-ahead logging lets one writer and many readers work at the same "
        "time. Readers see a consistent snapshot while a write is in progress, "
        "and the log is folded back into the database at a checkpoint."
    ),
    "sqlite-vec": (
        "sqlite-vec stores vectors in a virtual table and scores every one of "
        "them on a KNN query. Search is exact rather than approximate, so "
        "recall is perfect and latency grows with the size of the corpus."
    ),
    "fts5": (
        "FTS5 is SQLite's full-text index. It ranks matches with BM25, which "
        "rewards rare terms and penalises long documents. An external-content "
        "table keeps the index in step with the table it indexes."
    ),
    "backups": (
        "Backing up an embedded index is copying a file. Do it after a "
        "checkpoint so the write-ahead log is folded in, or copy the -wal "
        "sidecar alongside it."
    ),
}

QUESTION = "how does the database stay consistent while something is writing?"


def main() -> int:
    embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    print(f"embedder: {embedder} -> {embedder.dimensions} dimensions")

    chat = OllamaChat("llama3.2") if is_available() else None
    rag = Rag(embed_model=embedder, chat_model=chat, db_path=":memory:", auto=False)

    for name, text in CORPUS.items():
        rag.add_text(text, name=name)
    print(f"indexed {len(rag)} chunks\n")

    print("--- without reranking ---")
    for hit in rag.search(QUESTION, top_k=3):
        print(f"  {hit.score:.4f}  {hit.source}")

    try:
        from softrag.providers.local import CrossEncoderReranker

        reranker = CrossEncoderReranker()
    except Exception as exc:  # the model download is large and may be skipped
        print(f"\n(skipping reranker: {exc})")
        reranker = None

    if reranker is not None:
        print("\n--- with a cross-encoder reranker ---")
        # candidates=20 gives the reranker something to work with; reranking the
        # same 3 hits it was already going to return changes nothing.
        for hit in rag.search(QUESTION, top_k=3, candidates=20, rerank=reranker):
            print(f"  {hit.score:.4f}  {hit.source}")

    if chat is not None:
        print("\n--- answer ---")
        answer = rag.query(QUESTION, top_k=3, rerank=reranker)
        print(answer)
        print(f"\nsources: {answer.sources}")
    else:
        print("\n(no Ollama daemon; skipping generation)")

    rag.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

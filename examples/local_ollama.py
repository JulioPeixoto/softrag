"""Fully offline RAG with Ollama.

Nothing leaves the machine: embeddings, generation and the index are all local.
softrag talks to Ollama over plain HTTP, so no extra Python package is needed
beyond softrag itself.

Requires:
    Ollama running (https://ollama.com), plus the two models:
        ollama pull nomic-embed-text
        ollama pull llama3.2

Run:
    python examples/local_ollama.py
"""

from __future__ import annotations

import sys

from softrag import Rag
from softrag.providers.ollama import OllamaChat, OllamaEmbedder, base_url, is_available

NOTES = {
    "postgres-incident": (
        "On 12 March the primary database ran out of connections after a "
        "deploy raised the pool size on every worker. Recovery was to roll the "
        "deploy back and restart pgbouncer. The fix was to cap the total pool "
        "across workers rather than per worker."
    ),
    "cache-incident": (
        "On 3 April a cache stampede took the API down for eleven minutes when "
        "a popular key expired during peak traffic. Recovery was to warm the "
        "key manually. The fix was request coalescing plus a jittered TTL."
    ),
    "deploy-process": (
        "Deploys go out on Tuesdays and Thursdays. A change sits in staging for "
        "one business day before promotion. Rollback re-points the traffic "
        "router at the previous revision and takes under a minute."
    ),
}


def main() -> int:
    if not is_available():
        print(
            f"No Ollama daemon at {base_url()}. Start it with: ollama serve",
            file=sys.stderr,
        )
        return 1

    rag = Rag(
        embed_model=OllamaEmbedder("nomic-embed-text"),
        chat_model=OllamaChat("llama3.2", temperature=0),
        db_path="incidents.db",
    )

    for name, text in NOTES.items():
        result = rag.add_text(text, name=name, metadata={"kind": "postmortem"})
        print(f"  {result}")

    print(f"\nindexed {len(rag)} chunks in {rag.stats().path}")

    print("\n--- retrieval only (no model call) ---")
    for hit in rag.search("what caused the outage in April?", top_k=2):
        print(f"  {hit.score:.4f}  {hit.source}: {hit.text[:70]}...")

    print("\n--- generated answer ---")
    for delta in rag.query(
        "Summarise both incidents and their root causes.", stream=True
    ):
        print(delta, end="", flush=True)
    print()

    rag.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

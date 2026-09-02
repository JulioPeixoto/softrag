"""softrag in 40 lines, with no API key, no network and no model download.

Requires:
    pip install softrag

Uses HashEmbedder (hashes word n-grams into a vector) and EchoChatModel (returns
the prompt instead of generating). Neither is good at its job -- that is the
point: everything you see below came out of the index, so this exercises
storage, chunking and hybrid retrieval end to end without depending on anything.

Run:
    python examples/quickstart.py
"""

from __future__ import annotations

import softrag

DOCUMENTS = {
    "refunds": (
        "Refunds are processed within 5 business days of approval. "
        "Approval requires the original order number and proof of purchase."
    ),
    "shipping": (
        "Shipping is free on orders above 50 EUR. Below that threshold a flat "
        "fee of 4.90 EUR applies. Express delivery costs 12 EUR."
    ),
    "warranty": (
        "Every device carries a 24 month warranty covering manufacturing "
        "defects. Accidental damage is not covered and repairs are billed."
    ),
    "support": (
        "Support is available Monday to Friday, 09:00 to 18:00 CET. Error code "
        "ERR_CONN_4021 means the device could not reach the update server."
    ),
}


def main() -> None:
    # ":memory:" keeps this example self-contained. Pass a path to persist.
    rag = softrag.Rag(
        db_path=":memory:",
        embed_model=softrag.HashEmbedder(dimensions=256),
        chat_model=softrag.EchoChatModel(),
    )

    print("== Ingest ==")
    for name, text in DOCUMENTS.items():
        result = rag.add_text(text, name=name, metadata={"kind": "policy"})
        print(
            f"  {result.source:<9} added={result.chunks_added} chars={result.characters}"
        )

    stats = rag.stats()
    print(
        f"  -> {stats.documents} chunks from {stats.sources} sources, "
        f"dim={stats.dimensions}, schema v{stats.schema_version}"
    )

    print("\n== Re-ingest is free ==")
    again = rag.add_text(DOCUMENTS["refunds"], name="refunds")
    print(
        f"  added={again.chunks_added} skipped={again.chunks_skipped} "
        "(content unchanged, nothing was re-embedded)"
    )

    print("\n== search(): hybrid, and what each retriever contributed ==")
    for hit in rag.search("how long do refunds take", top_k=3):
        print(f"  {hit.score:.4f}  ranks={hit.ranks!s:<32} {hit.source}")
        print(f"          {hit.text[:70]}...")

    print("\n== search(): an exact token BM25 finds and vectors would miss ==")
    for hit in rag.search("ERR_CONN_4021", top_k=2, mode="keyword"):
        print(f"  {hit.score:.4f}  {hit.source}: {hit.text[:60]}...")

    print("\n== query(): EchoChatModel returns the prompt it was given ==")
    answer = rag.query("How long do refunds take?", top_k=2)
    print(f"  sources: {answer.sources}")
    print(f"  context the model saw ({len(answer.context)} chars):")
    for line in answer.context.splitlines():
        print(f"    | {line[:76]}")

    print("\n== Management ==")
    print(f"  sources: {[s.source for s in rag.sources()]}")
    print(f"  deleted {rag.delete(source='warranty')} chunks from 'warranty'")
    print(f"  {len(rag)} chunks remain")

    rag.close()


if __name__ == "__main__":
    main()

"""End-to-end RAG with OpenAI.

Indexes a few documents, answers a question with citations, and streams a second
answer token by token.

Requires:
    pip install 'softrag[openai]'
    export OPENAI_API_KEY=sk-...

Run:
    python examples/openai_rag.py
"""

from __future__ import annotations

import os
import sys

from softrag import Rag
from softrag.providers.openai import OpenAIChat, OpenAIEmbedder

DOCUMENTS = {
    "refunds": (
        "Refunds are processed within five business days of an approved "
        "request. Annual plans are refunded pro rata for the unused portion of "
        "the term. Requests need the original order reference."
    ),
    "shipping": (
        "Standard shipping takes three to five business days. Express shipping "
        "arrives the next business day if ordered before 2pm. We ship to 40 "
        "countries; customs charges are the recipient's responsibility."
    ),
    "warranty": (
        "Every device carries a 24 month warranty covering manufacturing "
        "defects. Accidental damage is not covered. Warranty claims require the "
        "serial number printed on the underside of the device."
    ),
}


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY first.", file=sys.stderr)
        return 1

    # text-embedding-3-small supports shortened vectors. 512 dimensions instead
    # of the default 1536 makes the index three times smaller and search
    # correspondingly faster, at a small cost in accuracy.
    rag = Rag(
        embed_model=OpenAIEmbedder("text-embedding-3-small", dimensions=512),
        chat_model=OpenAIChat("gpt-4.1-mini", temperature=0),
        db_path=":memory:",
    )

    for name, text in DOCUMENTS.items():
        rag.add_text(text, name=name, metadata={"kind": "policy"})
    print(f"indexed {len(rag)} chunks\n")

    answer = rag.query("How long does a refund take, and what do I need?")
    print(answer)
    print(f"\nsources: {answer.sources}")
    for hit in answer.hits:
        print(f"  {hit.score:.4f}  {hit.source}")

    print("\n--- streamed ---")
    stream = rag.query("What is not covered by the warranty?", stream=True)
    for delta in stream:
        print(delta, end="", flush=True)
    print(f"\n\nsources: {stream.sources}")

    rag.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

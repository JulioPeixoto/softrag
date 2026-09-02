"""Every metadata filter operator, demonstrated against a real index.

Requires:
    pip install softrag

No API key and no network: uses HashEmbedder, so retrieval quality is poor but
the filters -- which is what this example is about -- behave exactly as they
would with a real embedder. Filters compile to parameterised SQL over the JSON
metadata column, so they are applied by SQLite, not in Python.

Run:
    python examples/filtering.py
"""

from __future__ import annotations

import softrag

REPORTS = [
    (
        "q1-2024",
        "Q1 2024 revenue grew 12 percent year over year.",
        {
            "year": 2024,
            "quarter": 1,
            "team": "finance",
            "pages": 14,
            "tags": ["revenue", "public"],
            "author": "ada",
        },
    ),
    (
        "q2-2024",
        "Q2 2024 revenue was flat against a strong comparison quarter.",
        {
            "year": 2024,
            "quarter": 2,
            "team": "finance",
            "pages": 9,
            "tags": ["revenue"],
            "author": "ada",
        },
    ),
    (
        "q1-2025",
        "Q1 2025 revenue grew 8 percent, driven by the EU region.",
        {
            "year": 2025,
            "quarter": 1,
            "team": "finance",
            "pages": 22,
            "tags": ["revenue", "public"],
            "status": "draft",
        },
    ),
    (
        "hiring-2025",
        "Engineering hired four people in the first half of 2025.",
        {
            "year": 2025,
            "team": "engineering",
            "pages": 3,
            "tags": ["headcount"],
            "author": "grace",
        },
    ),
    (
        "infra-2025",
        "The platform migration to the new region finished in June.",
        {
            "year": 2025,
            "team": "engineering",
            "pages": 31,
            "tags": ["infrastructure"],
            "status": "final",
        },
    ),
]

QUERY = "revenue growth and headcount"


def build() -> softrag.Rag:
    rag = softrag.Rag(db_path=":memory:", embed_model=softrag.HashEmbedder())
    for name, text, metadata in REPORTS:
        rag.add_text(text, name=name, metadata=metadata)
    return rag


def show(rag: softrag.Rag, label: str, where: dict) -> None:
    hits = rag.search(QUERY, top_k=10, where=where)
    print(f"  {label:<46} -> {[h.source for h in hits]}")


def main() -> None:
    rag = build()
    print(f"Indexed {len(rag)} chunks. Query is always {QUERY!r};")
    print("only the filter changes.\n")

    print("== Comparison ==")
    show(rag, '{"team": "finance"}', {"team": "finance"})
    show(rag, '{"year": {"$eq": 2025}}', {"year": {"$eq": 2025}})
    show(rag, '{"team": {"$ne": "finance"}}', {"team": {"$ne": "finance"}})
    show(rag, '{"pages": {"$gt": 20}}', {"pages": {"$gt": 20}})
    show(rag, '{"year": {"$gte": 2025}}', {"year": {"$gte": 2025}})
    show(rag, '{"pages": {"$lt": 10}}', {"pages": {"$lt": 10}})
    show(rag, '{"quarter": {"$lte": 1}}', {"quarter": {"$lte": 1}})

    print("\n== Membership and text ==")
    show(rag, '{"team": {"$in": ["engineering"]}}', {"team": {"$in": ["engineering"]}})
    show(rag, '{"team": {"$nin": ["engineering"]}}', {"team": {"$nin": ["engineering"]}})
    show(rag, '{"author": {"$like": "a%"}}', {"author": {"$like": "a%"}})
    show(rag, '{"tags": {"$contains": "public"}}', {"tags": {"$contains": "public"}})

    print("\n== Presence ==")
    show(rag, '{"status": {"$exists": True}}', {"status": {"$exists": True}})
    show(rag, '{"status": {"$exists": False}}', {"status": {"$exists": False}})

    print("\n== Composition ==")
    show(
        rag,
        '{"team": "finance", "year": 2024}   (implicit AND)',
        {"team": "finance", "year": 2024},
    )
    show(
        rag,
        '{"year": {"$gte": 2024, "$lt": 2025}}   (a range)',
        {"year": {"$gte": 2024, "$lt": 2025}},
    )
    show(
        rag,
        '{"$and": [{"team": "finance"}, {"pages": {"$gt": 10}}]}',
        {"$and": [{"team": "finance"}, {"pages": {"$gt": 10}}]},
    )
    show(
        rag,
        '{"$or": [{"team": "engineering"}, {"quarter": 1}]}',
        {"$or": [{"team": "engineering"}, {"quarter": 1}]},
    )
    show(rag, '{"$not": {"team": "finance"}}', {"$not": {"team": "finance"}})

    print("\n== Filters apply to delete() too ==")
    removed = rag.delete(where={"status": "draft"})
    print(f"  rag.delete(where={{'status': 'draft'}}) removed {removed} chunk(s)")
    print(f"  remaining sources: {[s.source for s in rag.sources()]}")

    print("\n== An unknown operator fails loudly, not silently ==")
    try:
        rag.search(QUERY, where={"year": {"$between": [2024, 2025]}})
    except softrag.ConfigurationError as exc:
        print(f"  ConfigurationError: {exc}")

    rag.close()


if __name__ == "__main__":
    main()

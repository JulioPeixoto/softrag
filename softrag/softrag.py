"""
softrag.py
----------

Minimal local-first Retrieval-Augmented Generation (RAG) library backed by
SQLite + sqlite-vec. Everything (documents, embeddings, cache) lives
inside a single `.db` file.

"""

from __future__ import annotations

import os
import sqlite3
import json
import hashlib
from pathlib import Path
from typing import Sequence, Dict, Any, List, Callable

import sqlite_vec
import trafilatura
import fitz


SQLITE_PAGE_SIZE = 32_768
EMBED_DIM = 1_536

EmbedFn = Callable[[str], List[float]]
ChatFn = Callable[[str, Sequence[str]], str]

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def pack_vector(vec: Sequence[float]) -> bytes:
    """Pack Python list[float] → bytes accepted by sqlite-vec (vec_f32)."""
    import struct

    return struct.pack(f"{len(vec)}f", *vec)

# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class Rag:
    """Lightweight RAG engine with pluggable LLM back-end via dependency injection."""

    def __init__(
        self, *, embed_model, chat_model, db_path: str | os.PathLike = "softrag.db"
    ):
        """Create a new Softrag engine.

        Parameters
        ----------
        db_path : str | Path
            Where to store the SQLite file.
        embed_fn : Callable[[str], list[float]]
            Function that converts text → embedding. Defaults to OpenAI.
        chat_fn : Callable[[str, Sequence[str]], str]
            Function that receives (question, context_chunks) and returns answer text.
        """
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.db_path = Path(db_path)
        self.db: sqlite3.Connection | None = None
        self._ensure_db()

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def add_file(
        self, path: str | os.PathLike, metadata: Dict[str, Any] | None = None
    ) -> None:
        text = self._extract_file(path)
        self._persist(text, {"source": str(path), **(metadata or {})})

    def add_web(self, url: str, metadata: Dict[str, Any] | None = None) -> None:
        text = self._extract_web(url)
        self._persist(text, {"url": url, **(metadata or {})})

    def query(self, question: str, *, top_k: int = 5) -> str:
        ctx = self._retrieve(question, top_k)
        prompt = f"Context:\\n{'\\n\\n'.join(ctx)}\\n\\nQuestion: {question}"
        return self.chat_model.invoke(prompt)

    # -------------------------------------------------------------------
    # Internal helpers – DB
    # -------------------------------------------------------------------

    def _ensure_db(self) -> None:
        """Initialize SQLite + sqlite-vec, with better error reporting."""
        first_time = not self.db_path.exists()
        self.db = sqlite3.connect(self.db_path)
        self.db.execute("PRAGMA journal_mode=WAL;")
        self.db.execute(f"PRAGMA page_size={SQLITE_PAGE_SIZE};")

        try:
            self.db.enable_load_extension(True)
            sqlite_vec.load(self.db)  
        except Exception as e:
            raise RuntimeError(f"Failed to load sqlite-vec extension: {e}") from e
        finally:
            self.db.enable_load_extension(False)

        # Verify that key functions are present
        funcs = [row[0] for row in
                self.db.execute("SELECT name FROM pragma_function_list").fetchall()]
        missing = {"vec_distance_cosine"} - set(funcs)
        if missing:
            raise RuntimeError(
                "sqlite-vec did not register expected functions; "
                f"available: {funcs[:10]}…"
            )

        if first_time:
            self._create_schema()

    def _create_schema(self) -> None:
        sql = f"""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            metadata JSON
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts
        USING fts5(text, content='documents', content_rowid='id');

        CREATE VIRTUAL TABLE IF NOT EXISTS embeddings
        USING vec0(
            doc_id INTEGER,
            embedding FLOAT[{EMBED_DIM}]
        );
        """
        with self.db:
            self.db.executescript(sql)

    # ---------------------------- Extraction ---------------------------

    def _extract_file(self, path: str | os.PathLike) -> str:
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            return "\n".join(
                page.get_text("text", sort=True) for page in fitz.open(path)
            )
        if ext in {".txt", ".md"}:
            return Path(path).read_text(encoding="utf-8", errors="ignore")
        raise ValueError(f"Unsupported file type: {ext}")

    def _extract_web(self, url: str) -> str:
        html = trafilatura.fetch_url(url)
        if not html:
            raise RuntimeError(f"Unable to fetch {url}")
        return trafilatura.extract(html, include_comments=False) or ""

    # ---------------------------- Persistence --------------------------

    def _persist(self, text: str, metadata: Dict[str, Any]) -> None:
        chunks = self._split(text)
        with self.db:
            for chunk in chunks:
                h = sha256(chunk)
                if self.db.execute(
                    "SELECT 1 FROM documents WHERE json_extract(metadata,'$.hash')=?",
                    (h,),
                ).fetchone():
                    continue
                cur = self.db.execute(
                    "INSERT INTO documents(text, metadata) VALUES (?, ?)",
                    (chunk, json.dumps({**metadata, "hash": h})),
                )
                doc_id = cur.lastrowid
                vec = pack_vector(self.embed_model.embed_query(chunk))
                self.db.execute(
                    "INSERT INTO embeddings(doc_id, embedding) VALUES (?, ?)",
                    (doc_id, vec),
                )
                self.db.execute(
                    "INSERT INTO docs_fts(rowid, text) VALUES (?, ?)", (doc_id, chunk)
                )

    # ---------------------------- Retrieval ---------------------------

    def _retrieve(self, query: str, k: int) -> List[str]:
        q_vec = pack_vector(self.embed_model.embed_query(query))
        sql = """
        WITH kw AS (
            SELECT id, 1.0/(bm25(docs_fts)+1) AS score
              FROM docs_fts
             WHERE docs_fts MATCH ?
             LIMIT 20
        ),
        vec AS (
            SELECT doc_id AS id, 1.0 - vec_distance_cosine(embedding, ?) AS score
              FROM embeddings
             ORDER BY score DESC
             LIMIT 20
        ),
        merged AS (
            SELECT id, score FROM kw
            UNION ALL
            SELECT id, score FROM vec
        )
        SELECT text FROM documents WHERE id IN (
            SELECT id FROM merged ORDER BY score DESC LIMIT ?
        );
        """
        rows = self.db.execute(sql, (query, q_vec, k)).fetchall()
        return [r[0] for r in rows]

    # ---------------------------- Utilities ---------------------------

    @staticmethod
    def _split(text: str) -> List[str]:
        return [p.strip() for p in text.split("\n\n") if p.strip()]


# ---------------------------------------------------------------------------
# Convenience export
# ---------------------------------------------------------------------------

__all__ = ["Rag", "EmbedFn", "ChatFn"]

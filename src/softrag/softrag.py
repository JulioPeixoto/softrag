"""
softrag
-------

Minimal local-first Retrieval-Augmented Generation (RAG) library
using SQLite with sqlite-vec. All data (documents, embeddings, cache)
is stored in a single `.db` file.

This library provides a simple RAG implementation that can be easily
integrated with different language models and embeddings.
"""

from __future__ import annotations

import os
import sqlite3
import json
import hashlib
import struct
from pathlib import Path
from typing import Sequence, Dict, Any, List, Callable

import sqlite_vec
import trafilatura
import fitz


SQLITE_PAGE_SIZE = 32_768
EMBED_DIM = 1_536

EmbedFn = Callable[[str], List[float]]
ChatFn = Callable[[str, Sequence[str]], str]


def sha256(data: str) -> str:
    """Calculate the SHA-256 hash of a string.
    
    Args:
        data: String to be hashed.
        
    Returns:
        Hexadecimal string representing the SHA-256 hash.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def pack_vector(vec: Sequence[float]) -> bytes:
    """Convert list of floats to binary format accepted by sqlite-vec.
    
    Args:
        vec: Sequence of float values (embedding).
        
    Returns:
        Binary data ready for storage in SQLite.
    """
    return struct.pack(f"{len(vec)}f", *vec)


class Rag:
    """Lightweight RAG engine with pluggable LLM backend via dependency injection.
    
    This class implements a Retrieval-Augmented Generation system that
    stores documents and their embeddings in a SQLite database, allowing
    semantic queries and retrieval of relevant documents for use with LLMs.
    
    Attributes:
        embed_model: Model to generate text embeddings.
        chat_model: Model to generate context-based responses.
        db_path: Path to the SQLite database file.
        db: Connection to the SQLite database.
    """

    def __init__(
        self, *, embed_model, chat_model, db_path: str | os.PathLike = "softrag.db"
    ):
        """Initialize a new Softrag engine.
        
        Args:
            embed_model: Model for embedding generation.
            chat_model: Model for response generation.
            db_path: Path to the SQLite database file.
        """
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.db_path = Path(db_path)
        self.db: sqlite3.Connection | None = None
        self._ensure_db()

    def add_file(
        self, path: str | os.PathLike, metadata: Dict[str, Any] | None = None
    ) -> None:
        """Add file content to the database.
        
        Args:
            path: Path to the file to be processed.
            metadata: Additional metadata to be stored with the document.
        
        Raises:
            ValueError: If the file type is not supported.
        """
        text = self._extract_file(path)
        self._persist(text, {"source": str(path), **(metadata or {})})

    def add_web(self, url: str, metadata: Dict[str, Any] | None = None) -> None:
        """Add web page content to the database.
        
        Args:
            url: URL of the web page to be processed.
            metadata: Additional metadata to be stored with the document.
            
        Raises:
            RuntimeError: If the URL cannot be accessed.
        """
        text = self._extract_web(url)
        self._persist(text, {"url": url, **(metadata or {})})

    def query(self, question: str, *, top_k: int = 5, stream: bool = False):
        """Answer a question using relevant documents as context.
        
        Args:
            question: Question to be answered.
            top_k: Number of documents to retrieve as context.
            stream: If True, returns a generator that yields response chunks.
            
        Returns:
            If stream=False: Complete response as a string.
            If stream=True: Generator yielding response chunks.
        """
        ctx = self._retrieve(question, top_k)
        prompt = f"Context:\n{'\n\n'.join(ctx)}\n\nQuestion: {question}"
        
        if not stream:
            return self.chat_model.invoke(prompt)
        else:
            return self._stream_response(prompt)

    def _stream_response(self, prompt: str):
        """Stream the response from the chat model.
        
        Args:
            prompt: The prompt to send to the chat model.
            
        Yields:
            Chunks of the response as they become available.
        """
        if hasattr(self.chat_model, "stream"):
            for chunk in self.chat_model.stream(prompt):
                if hasattr(chunk, "content"):
                    yield chunk.content
                else:
                    yield chunk
        
        elif hasattr(self.chat_model, "completions") and hasattr(self.chat_model.completions, "create"):
            response = self.chat_model.completions.create(
                model=self.chat_model.model,
                prompt=prompt,
                stream=True
            )
            for chunk in response:
                if hasattr(chunk, "choices") and chunk.choices:
                    yield chunk.choices[0].text
        
        elif hasattr(self.chat_model, "generate_stream"):
            for chunk in self.chat_model.generate_stream(prompt):
                yield chunk
            
        else:
            full_response = self.chat_model.invoke(prompt)
            words = full_response.split()
            for i in range(0, len(words), 2): 
                yield " ".join(words[i:i+2])

    def _ensure_db(self) -> None:
        """Initialize SQLite with sqlite-vec and verify functionality.
        
        Raises:
            RuntimeError: If the expected sqlite-vec functions are not available.
        """
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

        funcs = [row[0] for row in
                self.db.execute("SELECT name FROM pragma_function_list").fetchall()]
        missing = {"vec_distance_cosine"} - set(funcs)
        if missing:
            raise RuntimeError(
                "sqlite-vec did not register the expected functions; "
                f"available: {funcs[:10]}…"
            )

        if first_time:
            self._create_schema()

    def _create_schema(self) -> None:
        """Create the required tables in the SQLite database."""
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

    def _extract_file(self, path: str | os.PathLike) -> str:
        """Extract text from a file.
        
        Args:
            path: Path to the file.
            
        Returns:
            Extracted text from the file.
            
        Raises:
            ValueError: If the file type is not supported.
        """
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            return "\n".join(
                page.get_text("text", sort=True) for page in fitz.open(path)
            )
        if ext in {".txt", ".md"}:
            return Path(path).read_text(encoding="utf-8", errors="ignore")
        raise ValueError(f"Unsupported file type: {ext}")

    def _extract_web(self, url: str) -> str:
        """Extract text from a web page.
        
        Args:
            url: URL of the web page.
            
        Returns:
            Extracted text from the web page.
            
        Raises:
            RuntimeError: If the URL cannot be accessed.
        """
        html = trafilatura.fetch_url(url)
        if not html:
            raise RuntimeError(f"Unable to access {url}")
        return trafilatura.extract(html, include_comments=False) or ""

    def _persist(self, text: str, metadata: Dict[str, Any]) -> None:
        """Persist text, splitting into chunks and calculating embeddings.
        
        Args:
            text: Text to be stored.
            metadata: Metadata associated with the text.
        """
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

    def _retrieve(self, query: str, k: int) -> List[str]:
        """Retrieve the most relevant documents for a query.
        
        Combines keyword search (FTS5) and vector similarity.
        
        Args:
            query: Query to be searched.
            k: Number of documents to be returned.
            
        Returns:
            List of relevant document texts.
        """
        # Prepare FTS query by handling special characters
        fts_query = " OR ".join(word for word in query.replace(",", " ").replace("?", " ").split() if len(word) > 2)
        
        # Check if the documents table exists and has records
        if not self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='documents'").fetchone():
            return ["No documents in the database. Add content using add_file() or add_web() first."]
        
        count = self.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        if count == 0:
            return ["The database is empty. Add content using add_file() or add_web() first."]
            
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
        rows = self.db.execute(sql, (fts_query, q_vec, k)).fetchall()
        return [r[0] for r in rows]

    @staticmethod
    def _split(text: str) -> List[str]:
        """Split the text into chunks for processing.
        
        Args:
            text: Text to be split.
            
        Returns:
            List of text chunks.
        """
        return [p.strip() for p in text.split("\n\n") if p.strip()]


__all__ = ["Rag", "EmbedFn", "ChatFn"]

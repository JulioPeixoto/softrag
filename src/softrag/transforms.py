"""Query-time and ingest-time transforms.

Retrieval quality is not only a function of the index. What you embed at ingest
time and what you embed at query time are both choices, and the cheapest wins in
RAG usually come from changing one of them rather than from swapping the
retriever.

* :func:`hyde` and :func:`expand_query` rewrite the *query* before it is
  embedded.
* :func:`contextualize` and :class:`ContextualChunker` rewrite the *chunks*
  before they are indexed.

Every transform here costs LLM calls, and every one of them degrades to the
untransformed input on failure: a slow or unhelpful rewrite must never turn a
working query, or a working ingest, into an exception.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
from collections.abc import Sequence
from typing import Any

from .chunking import Chunker, resolve_chunker
from .providers import adapt_chat_model
from .retrieval import reciprocal_rank_fusion
from .types import ChatModel, Hit

log = logging.getLogger("softrag.transforms")

__all__ = [
    "CONTEXTUAL_PROMPT",
    "ContextualChunker",
    "contextualize",
    "expand_query",
    "hyde",
    "multi_query_search",
]

DEFAULT_HYDE_PROMPT = """\
Write a short passage, three or four sentences, that reads like an excerpt from \
a document which answers the question below.

Write it as confident prose in the vocabulary the real document would use. It \
does not matter whether the details are correct -- the passage is used only as a \
search probe, never shown to anyone.

Question: {question}

Passage:"""

DEFAULT_EXPANSION_PROMPT = """\
Rewrite the search query below in {n} different ways, so that documents phrased \
differently from the original still match.

Vary the vocabulary and the phrasing, keep the meaning, and keep each rewrite on \
its own line. Answer with the {n} rewrites and nothing else.

Query: {question}"""

#: Anthropic's published Contextual Retrieval prompt, reproduced verbatim.
CONTEXTUAL_PROMPT = """\
<document>
{document}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

_LIST_PREFIX = re.compile(r"^\s*(?:[-*•]|\(?\d+[.)])\s*")


# --------------------------------------------------------------------------- #
# Query transforms
# --------------------------------------------------------------------------- #


def hyde(question: str, chat_model: ChatModel | Any, *, prompt: str | None = None) -> str:
    """Hypothetical Document Embeddings: embed a fake answer, not the question.

    A question and the passage that answers it are different kinds of text. "How
    long is the refund window?" is short, interrogative and shares almost no
    vocabulary with "Refunds are issued within 30 days of purchase, provided the
    item is unopened." Embedding models place them further apart than their
    relevance deserves, which is the asymmetry every dense retriever fights.

    HyDE (Gao et al., 2022) sidesteps it: ask a chat model to *write* the passage
    it thinks answers the question, then embed that. The fabricated passage is
    wrong on the details and irrelevant as an answer, but it is the right shape
    and the right vocabulary, so it lands much closer to the real document than
    the question ever would. The retriever finds the real passage; the fake one
    is thrown away.

    The cost is one LLM call per query, on the latency path, and a failure mode
    of its own: for a question about something the model has no idea about, the
    hallucinated passage can drift off-topic. Fusing HyDE results with plain
    query results, or keeping BM25 in the mix, guards against that.

    Args:
        question: The user's question.
        chat_model: Any chat backend, adapted through
            :func:`~softrag.providers.adapt_chat_model`.
        prompt: Template with a ``{question}`` field. Defaults to
            :data:`DEFAULT_HYDE_PROMPT`.

    Returns:
        The hypothetical passage, or ``question`` unchanged if the model failed
        or returned nothing usable.

    Example:
        >>> probe = hyde("How long is the refund window?", chat)  # doctest: +SKIP
        >>> hits = rag.search(probe)                              # doctest: +SKIP
    """
    template = prompt or DEFAULT_HYDE_PROMPT
    try:
        reply = adapt_chat_model(chat_model).complete(template.format(question=question))
    except Exception as exc:
        log.warning("HyDE generation failed (%s); using the raw question", exc)
        return question

    passage = _clean(reply)
    if not passage:
        log.warning("HyDE returned an empty passage; using the raw question")
        return question
    return passage


def expand_query(question: str, chat_model: ChatModel | Any, *, n: int = 3) -> list[str]:
    """Generate paraphrases of a query, to search for all of them.

    One phrasing of a question reaches one neighbourhood of the embedding space.
    Asking the same thing three ways and fusing the results covers more of it,
    which is the cheapest available fix for a query whose vocabulary happens not
    to match the corpus.

    Args:
        question: The user's question.
        chat_model: Any chat backend.
        n: How many paraphrases to ask for.

    Returns:
        ``[question, *paraphrases]``, deduplicated case-insensitively, with the
        original always first. Just ``[question]`` if generation or parsing
        failed -- so the caller can use the result unconditionally.

    Example:
        >>> expand_query("refund window?", chat)      # doctest: +SKIP
        ['refund window?', 'How many days do I have to return an item?', ...]
    """
    if n <= 0:
        return [question]

    try:
        reply = adapt_chat_model(chat_model).complete(
            DEFAULT_EXPANSION_PROMPT.format(question=question, n=n)
        )
    except Exception as exc:
        log.warning("query expansion failed (%s); using the query alone", exc)
        return [question]

    variants = _parse_variants(reply)
    if not variants:
        log.warning("could not parse any rewrites out of the model reply")
        return [question]

    out = [question]
    seen = {question.strip().lower()}
    for variant in variants:
        key = variant.lower()
        if key not in seen:
            seen.add(key)
            out.append(variant)
        if len(out) > n:
            break
    return out


def _parse_variants(reply: Any) -> list[str]:
    """Pull query rewrites out of a reply, JSON array or one-per-line."""
    text = reply if isinstance(reply, str) else str(reply)
    if not text.strip():
        return []

    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
        except (ValueError, TypeError):
            payload = None
        if isinstance(payload, list):
            items = [str(item).strip() for item in payload if isinstance(item, str)]
            items = [item for item in items if item]
            if items:
                return items

    lines = []
    for raw in text.splitlines():
        line = _clean(_LIST_PREFIX.sub("", raw)).strip("\"'")
        # Drop preamble like "Here are three rewrites:" and empty separators.
        if line and not line.endswith(":"):
            lines.append(line)
    return lines


def multi_query_search(
    rag: Any,
    question: str,
    *,
    n: int = 3,
    top_k: int = 5,
    chat_model: ChatModel | Any = None,
    **search_kwargs: Any,
) -> list[Hit]:
    """Search once per paraphrase and fuse the result lists.

    :func:`expand_query` widens the query, each variant is searched
    independently, and the ranked id lists are combined with
    :func:`~softrag.retrieval.reciprocal_rank_fusion`. A document found by
    several phrasings rises; a document only one odd phrasing liked does not.
    Each document appears exactly once in the output, at its fused rank.

    Each variant retrieves ``2 * top_k`` candidates so fusion has room to
    disagree before the list is cut back to ``top_k``.

    Args:
        rag: A :class:`~softrag.engine.Rag`, or anything with ``search()``.
        question: The user's question.
        n: How many paraphrases to generate.
        top_k: How many hits to return after fusion.
        chat_model: Chat backend for the expansion. Defaults to the engine's own
            ``rag.chat_model``.
        **search_kwargs: Forwarded to ``rag.search()``.

    Returns:
        At most ``top_k`` hits, best first, with ``hit.score`` set to the fused
        RRF score.

    Example:
        >>> hits = multi_query_search(rag, "refund window?", n=3)  # doctest: +SKIP
    """
    if top_k <= 0:
        return []

    model = chat_model
    if model is None:
        try:
            # Rag.chat_model is a property that raises when none is configured.
            model = rag.chat_model
        except Exception as exc:
            log.warning(
                "no chat model available for query expansion (%s); "
                "falling back to a single search",
                exc,
            )
            model = None

    queries = [question] if model is None else expand_query(question, model, n=n)
    if len(queries) == 1:
        return list(rag.search(question, top_k=top_k, **search_kwargs))

    per_query_k = max(2 * top_k, top_k)
    ranked_lists: list[list[int]] = []
    hits_by_id: dict[int, Hit] = {}
    for query in queries:
        try:
            hits = rag.search(query, top_k=per_query_k, **search_kwargs)
        except Exception as exc:
            log.warning("search failed for query variant %r (%s); skipping", query, exc)
            continue
        ranked_lists.append([hit.id for hit in hits])
        for hit in hits:
            hits_by_id.setdefault(hit.id, hit)

    if not ranked_lists:
        return []

    out: list[Hit] = []
    for doc_id, score in reciprocal_rank_fusion(ranked_lists)[:top_k]:
        hit = hits_by_id.get(doc_id)
        if hit is None:  # pragma: no cover - defensive
            continue
        hit.score = score
        out.append(hit)
    return out


# --------------------------------------------------------------------------- #
# Ingest transforms
# --------------------------------------------------------------------------- #


def contextualize(
    document: str,
    chunks: Sequence[str],
    chat_model: ChatModel | Any,
    *,
    prompt: str | None = None,
    max_workers: int = 4,
) -> list[str]:
    """Prepend a chunk-specific blurb situating each chunk in its document.

    This is Anthropic's Contextual Retrieval, and :data:`CONTEXTUAL_PROMPT` is
    their published prompt reproduced verbatim. The problem it solves is the one
    every chunking strategy creates: a chunk reading "revenue grew by 3% over the
    previous quarter" has lost the name of the company and the quarter it belongs
    to, so no query mentioning either can match it. The fix is to hand the model
    the whole document plus the chunk and ask for one or two sentences of
    context, then index ``context + chunk``.

    Anthropic report the top-20 retrieval failure rate dropping from 5.7% to
    2.9% when contextual embeddings are combined with contextual BM25 (and to
    1.9% with a reranker on top).

    The caveat is cost: one LLM call per chunk, at ingest time. A 500-chunk
    document is 500 calls. Prompt caching over the document part makes this much
    cheaper on backends that support it, but it is never free -- treat it as
    something you turn on for a corpus you will query many times.

    Calls run concurrently and output order always matches ``chunks``. A chunk
    whose call fails is passed through unchanged with a warning, so one bad
    response cannot fail a whole ingest.

    Args:
        document: The full source text the chunks were cut from.
        chunks: The chunks, in document order.
        chat_model: Any chat backend.
        prompt: Template with ``{document}`` and ``{chunk}`` fields. Defaults to
            :data:`CONTEXTUAL_PROMPT`.
        max_workers: Concurrent LLM calls.

    Returns:
        One string per input chunk, each the context blurb followed by a blank
        line and the original chunk.

    Example:
        >>> contextualize(report, chunks, chat)[0]     # doctest: +SKIP
        'This chunk is from ACME Corp's Q2 2024 report...\\n\\nRevenue grew by 3%...'
    """
    if not chunks:
        return []

    template = prompt or CONTEXTUAL_PROMPT
    model = adapt_chat_model(chat_model)

    def annotate(item: tuple[int, str]) -> str:
        index, chunk = item
        try:
            reply = model.complete(template.format(document=document, chunk=chunk))
        except Exception as exc:
            log.warning(
                "contextualisation failed for chunk %d (%s); indexing it as-is",
                index,
                exc,
            )
            return chunk
        context = _clean(reply)
        if not context:
            log.warning("empty context for chunk %d; indexing it as-is", index)
            return chunk
        return f"{context}\n\n{chunk}"

    workers = max(1, min(max_workers, len(chunks)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        # executor.map preserves input order, which chunk indices depend on.
        return list(pool.map(annotate, enumerate(chunks)))


class ContextualChunker:
    """A chunker that contextualises every chunk it produces.

    Wraps any chunker so :func:`contextualize` runs as part of splitting, which
    makes contextual retrieval a one-argument change:

        rag = Rag(chunker=ContextualChunker(None, chat_model))

    Being a plain callable ``str -> list[str]``, it fits anywhere a chunker is
    accepted, including ``Rag(chunker=...)`` and ``rag.add(..., chunker=...)``.

    Bear the cost in mind -- one LLM call per chunk, on every ingest -- and see
    :func:`contextualize` for the numbers that justify it.

    Args:
        chunker: The underlying chunker: ``None`` for the default recursive
            chunker, a separator string, or any callable.
        chat_model: Any chat backend.
        prompt: Template with ``{document}`` and ``{chunk}`` fields.
        max_workers: Concurrent LLM calls per document.

    Example:
        >>> chunker = ContextualChunker(None, chat_model)   # doctest: +SKIP
        >>> rag = Rag(chunker=chunker)                      # doctest: +SKIP
    """

    __slots__ = ("chat_model", "chunker", "max_workers", "prompt")

    def __init__(
        self,
        chunker: Chunker | str | None,
        chat_model: ChatModel | Any,
        *,
        prompt: str | None = None,
        max_workers: int = 4,
    ) -> None:
        self.chunker = resolve_chunker(chunker)
        self.chat_model = chat_model
        self.prompt = prompt
        self.max_workers = max_workers

    def __call__(self, text: str) -> list[str]:
        """Split ``text`` and return the chunks with their context prepended."""
        chunks = list(self.chunker(text))
        if not chunks:
            return []
        return contextualize(
            text,
            chunks,
            self.chat_model,
            prompt=self.prompt,
            max_workers=self.max_workers,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ContextualChunker(chunker={self.chunker!r})"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _clean(reply: Any) -> str:
    """Normalise a model reply into stripped plain text."""
    text = reply if isinstance(reply, str) else str(reply)
    return text.strip()

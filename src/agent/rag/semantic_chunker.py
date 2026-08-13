"""Semantic chunking via chonkie.

Splits text on semantic boundaries (sentence-level embeddings + similarity
threshold) instead of naive character/recursive splits.  Semantic chunks keep
topically-related sentences together, which materially improves retrieval
quality for e-books and long documents.

The embedding model is the *same* nomic-embed-text instance used by the vector
store (shared process-wide singleton), so semantic chunking does not load a
second copy of the ~137MB ONNX model.

If ``chonkie`` is not installed, :func:`semantic_chunk_text` returns ``None``
so callers fall back to the existing recursive chunker.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# SemanticChunker splits sentences on ". "/"! "/"? "/newline.  Chinese text
# uses "。" without a trailing space, so add CJK-friendly delimiters to avoid
# treating an entire paragraph of Chinese prose as one "sentence".
_DELIM = [". ", "! ", "? ", "\n", "。", "！", "？", "；", "，"]

# chonkie is an optional dependency.  Resolve its BaseEmbeddings base class at
# import time if available; otherwise fall back to ``object`` so the module can
# still be imported (semantic_chunk_text() returns None in that case).
try:
    from chonkie.embeddings.base import BaseEmbeddings as _BaseEmbeddings
except ImportError:  # pragma: no cover - chonkie absent
    _BaseEmbeddings = object


def _is_chonkie_available() -> bool:
    try:
        import chonkie  # noqa: F401
        return True
    except ImportError:
        return False


def semantic_chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 80,
    min_chunk_size: int = 100,
    overlap_percent: float = 0.10,
    embedding_function=None,
) -> Optional[List[str]]:
    """Split ``text`` into semantic chunks using chonkie's SemanticChunker.

    Args:
        text: The text to split.
        chunk_size: Target maximum chunk size in **tokens** (800).
        chunk_overlap: Overlap between consecutive chunks in **tokens**.
            Defaults to 10% of ``chunk_size`` (80).  When ``overlap_percent``
            is set, ``int(chunk_size * overlap_percent)`` overrides this.
        min_chunk_size: Target minimum chunk size in **tokens** (100). Chunks
            smaller than this are greedily merged into their neighbours.
        overlap_percent: Fraction of ``chunk_size`` used as overlap (0.10 =>
            80 tokens), preserving continuity across chunk boundaries.  Takes
            precedence over ``chunk_overlap``.
        embedding_function: A callable ``embed_documents(list[str]) ->
            list[list[float]]``. When omitted, the shared nomic-embed-text
            singleton is used.

    Returns:
        List of chunk strings, or ``None`` if chonkie is unavailable or no
        embedding function could be resolved (callers should fall back).
    """
    if not text or not text.strip():
        return []

    if not _is_chonkie_available():
        logger.debug("semantic chunking skipped: chonkie not installed")
        return None

    try:
        ef = embedding_function or _shared_embedding_function()
        if ef is None:
            logger.debug("semantic chunking skipped: no embedding function available")
            return None

        from chonkie import SemanticChunker

        embeddings = _ChonkieEmbeddingAdapter(ef)

        # When chonkie is present, _BaseEmbeddings is the real base class and
        # the isinstance check in SemanticChunker.__init__ will pass.  When it
        # fell back to ``object``, chonkie can't be imported anyway (handled
        # above by _is_chonkie_available), so this branch is defensive only.
        chunker = SemanticChunker(
            embedding_model=embeddings,
            threshold=0.7,          # lower => larger, more coherent groups
            chunk_size=chunk_size,  # hard ceiling in tokens
            min_sentences_per_chunk=2,
            min_characters_per_sentence=20,
            delim=_DELIM,
        )
        chunks = chunker.chunk(text)
        result = [c.text.strip() for c in chunks if c.text.strip()]
        merged = _merge_to_target_size(result, chunk_size, min_chunk_size)
        return _apply_overlap(merged, chunk_size, chunk_overlap, overlap_percent)
    except Exception as e:
        # Never let a chunking failure abort ingestion — the caller falls back
        # to the recursive chunker.
        logger.warning("semantic chunking failed (%s), falling back to recursive: %s",
                       type(e).__name__, e)
        return None


def _merge_to_target_size(
    chunks: List[str],
    chunk_size: int,
    min_chunk_size: int,
) -> List[str]:
    """Greedily merge adjacent chunks toward the [min, max] token target.

    SemanticChunker splits on *semantic* boundaries, so chunks can be smaller
    than the target when a document has many short, topically-distinct
    sentences.  We merge adjacent chunks (which are already topically
    contiguous, since semantic splitting only cuts at low-similarity points)
    until each reaches ``min_chunk_size`` tokens, without exceeding
    ``chunk_size``.
    """
    from .chunker import estimate_tokens

    if not chunks:
        return chunks

    merged: List[str] = []
    current = chunks[0]

    for nxt in chunks[1:]:
        if estimate_tokens(current) < min_chunk_size and \
                estimate_tokens(current) + estimate_tokens(nxt) <= chunk_size:
            current = current + "\n\n" + nxt
        else:
            merged.append(current)
            current = nxt
    merged.append(current)
    return merged


def _apply_overlap(
    chunks: List[str],
    chunk_size: int,
    chunk_overlap: int,
    overlap_percent: float,
) -> List[str]:
    """Add token-level overlap between consecutive chunks.

    SemanticChunker produces non-overlapping chunks, so a sentence split at a
    chunk boundary loses its surrounding context.  We prepend to each chunk a
    tail slice of the *previous* chunk — chosen at a sentence boundary so we
    never split a sentence mid-way — so that retrieval across a boundary still
    sees the surrounding sentences.

    The overlap size is ``int(chunk_size * overlap_percent)`` when
    ``overlap_percent`` is set, otherwise the explicit ``chunk_overlap``.
    """
    from .chunker import estimate_tokens

    if len(chunks) <= 1:
        return chunks

    overlap = int(chunk_size * overlap_percent) if overlap_percent else chunk_overlap
    if overlap <= 0:
        return chunks

    result: List[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        # Overlap is always derived from the *original* previous chunk (not the
        # overlap-padded one), so overlap never compounds across boundaries.
        tail = _tail_by_tokens(chunks[i - 1], overlap)
        if tail:
            result.append(tail + "\n\n" + chunks[i])
        else:
            result.append(chunks[i])
    return result


def _tail_by_tokens(text: str, max_tokens: int) -> str:
    """Return the trailing portion of ``text`` up to ``max_tokens`` tokens,
    aligned to a sentence boundary (never splitting a sentence)."""
    from .chunker import estimate_tokens

    if not text or max_tokens <= 0:
        return ""

    # Split into sentences keeping delimiters attached, so we can re-join.
    import re
    parts = re.split(r"(?<=[。！？!?\.])\s*|\n", text)
    parts = [p for p in parts if p.strip()]

    tail: List[str] = []
    total = 0
    for p in reversed(parts):
        tk = estimate_tokens(p)
        if total + tk > max_tokens and tail:
            break
        tail.insert(0, p)
        total += tk
    return "".join(tail).strip()


# ---------------------------------------------------------------------------
# embedding singleton (shared with vector store)
# ---------------------------------------------------------------------------

_shared_ef = None
_shared_ef_resolved = False


def _shared_embedding_function():
    """Return the process-wide nomic-embed-text embedding function, if available.

    Imported lazily to avoid a hard dependency on fastembed/vector_store for
    callers that never enable RAG.
    """
    global _shared_ef, _shared_ef_resolved
    if _shared_ef_resolved:
        return _shared_ef
    _shared_ef_resolved = True
    try:
        from .vector_store import get_or_create_embedding_function
        _shared_ef = get_or_create_embedding_function()
    except Exception as e:
        logger.debug("shared embedding function unavailable: %s", e)
        _shared_ef = None
    return _shared_ef


# ---------------------------------------------------------------------------
# chonkie embedding adapter
# ---------------------------------------------------------------------------

class _ChonkieEmbeddingAdapter(_BaseEmbeddings):
    """Adapt a FastEmbed embedding function to chonkie's BaseEmbeddings interface.

    chonkie's SemanticChunker accepts either a model-name string or a
    ``BaseEmbeddings`` instance.  To keep the exact nomic-embed-text ONNX model
    shared with the vector store (instead of loading potion-base-32M or a
    second nomic copy), we subclass BaseEmbeddings and wrap the existing
    embedding function.  Required members:

      * ``embed(text)`` / ``embed_batch(texts)`` — sentence embeddings
      * ``similarity(u, v)`` — cosine similarity
      * ``get_tokenizer()`` — returns our ``estimate_tokens`` callable (chonkie
        wraps it via CallableAutoTokenizer for ``count_tokens``)
      * ``dimension`` — embedding vector dimension
    """

    def __init__(self, ef):
        super().__init__()
        self._ef = ef
        # Resolve dimension lazily (requires one embedding call) to avoid
        # forcing model load at construction time.
        self._dim = None

    # -- embedding ---------------------------------------------------------
    def _embed_many(self, texts: List[str]):
        """Embed a list of texts, tolerating either ``embed_documents`` or the
        callable ``__call__`` protocol (the test fake exposes ``__call__``)."""
        if hasattr(self._ef, "embed_documents"):
            return self._ef.embed_documents(texts)
        return self._ef(texts)

    def embed(self, text: str):
        return self._to_array(self._embed_many([text])[0])

    def embed_batch(self, texts: List[str]):
        return [self._to_array(v) for v in self._embed_many(list(texts))]

    @staticmethod
    def _to_array(vec):
        import numpy as np
        return np.asarray(vec, dtype="float32")

    # -- similarity --------------------------------------------------------
    def similarity(self, u, v) -> float:
        import numpy as np
        u = np.asarray(u, dtype="float32")
        v = np.asarray(v, dtype="float32")
        denom = float(np.linalg.norm(u) * np.linalg.norm(v))
        if denom == 0.0:
            return 0.0
        return float(np.dot(u, v) / denom)

    # -- dimension ---------------------------------------------------------
    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed("probe"))
        return self._dim

    # -- tokenizer ---------------------------------------------------------
    def get_tokenizer(self) -> Any:
        """Return a token counter callable.

        chonkie wraps callables via ``CallableAutoTokenizer`` whose
        ``count_tokens(text)`` calls it directly.  We use the same heuristic
        as the recursive chunker (``estimate_tokens``) so chunk sizes are
        measured consistently across both paths.
        """
        from .chunker import estimate_tokens
        return estimate_tokens

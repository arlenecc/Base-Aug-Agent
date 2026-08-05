"""Vector store — embedding + similarity search using LanceDB, with BGE reranker.

Embeddings are generated locally via FastEmbed (ONNX Runtime) using
nomic-ai/nomic-embed-text-v1.5 — no API server or HuggingFace dependency.
Vectors are stored in LanceDB (Lance columnar format on local disk).
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default embedding model — nomic-embed-text-v1.5 via FastEmbed (ONNX Runtime)
DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"
_TABLE_NAME = "knowledge_base"

# nomic-embed-text-v1.5 requires query/document prefixes for best results
_DOC_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "


# ---------------------------------------------------------------------------
# FastEmbed-based embedding function (local ONNX Runtime, no API needed)
# ---------------------------------------------------------------------------


class FastEmbedEmbeddingFunction:
    """Embedding function using FastEmbed (ONNX Runtime) for local inference.

    Loads nomic-ai/nomic-embed-text-v1.5 via ONNX Runtime — no GPU, no
    HuggingFace network calls, no PyTorch dependency. The model is cached
    after first download (~130MB quantized).
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self._model_name = model_name
        self._model = None  # lazy-init

    def _ensure_model(self):
        if self._model is None:
            from fastembed import TextEmbedding
            logger.info("Loading FastEmbed model: %s ...", self._model_name)
            self._model = TextEmbedding(model_name=self._model_name)
            logger.info("FastEmbed model loaded: %s", self._model_name)

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Embed a list of document texts. Returns a list of embedding vectors."""
        if not input:
            return []
        self._ensure_model()
        # Add document prefix for nomic-embed-text-v1.5
        prefixed = [_DOC_PREFIX + t if not t.startswith(_DOC_PREFIX) else t for t in input]
        embeddings = list(self._model.embed(prefixed))
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text with the query prefix."""
        self._ensure_model()
        prefixed = _QUERY_PREFIX + text if not text.startswith(_QUERY_PREFIX) else text
        embeddings = list(self._model.embed([prefixed]))
        return embeddings[0].tolist()

    def name(self) -> str:
        return f"fastembed:{self._model_name}"


# ---------------------------------------------------------------------------
# Vector store backed by LanceDB
# ---------------------------------------------------------------------------


class VectorStore:
    """Local vector store backed by LanceDB with FastEmbed embeddings.

    Uses nomic-ai/nomic-embed-text-v1.5 via ONNX Runtime for local inference.
    Supports optional BGE reranker for precise post-retrieval scoring.
    """

    def __init__(
        self,
        persist_dir: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        embedding_function: Optional[Any] = None,
    ):
        self._persist_dir = persist_dir
        self._embedding_model = embedding_model
        self._rerank_model = rerank_model
        self._custom_ef = embedding_function  # for testing injection
        self._db = None
        self._table = None
        self._ef = None  # embedding function
        self._initialized = False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        try:
            import lancedb
        except ImportError:
            raise ImportError(
                "lancedb is required for RAG vector storage. pip install lancedb"
            )

        os.makedirs(self._persist_dir, exist_ok=True)

        self._db = lancedb.connect(self._persist_dir)

        # Use injected embedding function (for testing) or create FastEmbed one
        if self._custom_ef is not None:
            self._ef = self._custom_ef
        else:
            self._ef = FastEmbedEmbeddingFunction(
                model_name=self._embedding_model,
            )

        # Open existing table or defer creation until first add
        if _TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(_TABLE_NAME)

        self._initialized = True
        count = self._table.count_rows() if self._table is not None else 0
        logger.info("VectorStore initialized: %d documents in store", count)

    def _ensure_table(self) -> Optional[Any]:
        """Open existing table if it exists, return None if not yet created."""
        if self._table is not None:
            return self._table
        if _TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(_TABLE_NAME)
        return self._table

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """Add document chunks to the vector store. Returns number of chunks added."""
        self._ensure_initialized()
        if not chunks:
            return 0

        # Build records (embed in batches to avoid OOM on large knowledge bases)
        records: List[Dict[str, Any]] = []
        ids_to_delete: List[str] = []
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            texts = [c["text"] for c in batch_chunks]
            vectors = self._ef(texts)
            for j, chunk in enumerate(batch_chunks):
                idx = i + j
                source = chunk.get("source", "unknown")
                chunk_idx = chunk.get("chunk_index", idx)
                chunk_id = f"{_hash_source(source)}_{chunk_idx}"
                ids_to_delete.append(chunk_id)
                records.append({
                    "id": chunk_id,
                    "text": chunk["text"],
                    "vector": _normalize(vectors[j]),
                    "source": source,
                    "chunk_index": chunk_idx,
                })

        table = self._ensure_table()

        if table is None:
            # First add: create table with all records
            self._table = self._db.create_table(_TABLE_NAME, records)
            logger.info("LanceDB table '%s' created with %d records", _TABLE_NAME, len(records))
            return len(records)

        # Upsert: delete existing records with same IDs, then add
        for i in range(0, len(ids_to_delete), 100):
            batch_ids = ids_to_delete[i:i + 100]
            id_list = ", ".join(f"'{x}'" for x in batch_ids)
            try:
                table.delete(f"id IN ({id_list})")
            except Exception as e:
                # IDs may not exist yet — log but don't crash the whole ingest
                logger.debug("VectorStore: delete batch failed (likely new IDs): %s", e)

        # Add records in batches
        total = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            table.add(batch)
            total += len(batch)

        logger.info("VectorStore: added %d chunks", total)
        return total

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search the vector store for relevant chunks.

        Returns list of {text, source, chunk_index, score} dicts.
        """
        self._ensure_initialized()
        if self._table is None or self._table.count_rows() == 0:
            return []

        # Embed query (use embed_query for proper prefix handling)
        if hasattr(self._ef, 'embed_query'):
            query_vec = _normalize(self._ef.embed_query(query))
        else:
            query_vec = _normalize(self._ef([query])[0])

        # Vector search
        search = self._table.search(query_vec).limit(top_k)
        if where:
            search = search.where(where)
        results = search.to_list()

        output: List[Dict[str, Any]] = []
        for r in results:
            output.append({
                "text": r.get("text", ""),
                "source": r.get("source", ""),
                "chunk_index": r.get("chunk_index", 0),
                "score": r.get("_distance", 0.0),
            })

        return output

    # ------------------------------------------------------------------
    # reranking with BGE
    # ------------------------------------------------------------------

    def _get_reranker(self):
        """Lazy-load the BGE reranker model."""
        if not hasattr(self, "_reranker"):
            try:
                from FlagEmbedding import FlagReranker
                self._reranker = FlagReranker(
                    self._rerank_model,
                    use_fp16=False,
                )
                logger.info("BGE reranker loaded: %s", self._rerank_model)
            except ImportError:
                logger.warning(
                    "FlagEmbedding not installed — reranking disabled. "
                    "pip install FlagEmbedding"
                )
                self._reranker = None
            except Exception as e:
                logger.warning("Failed to load reranker: %s", e)
                self._reranker = None
        return self._reranker

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Re-rank candidates using BGE reranker for precise scoring.

        Falls back to distance-based similarity scoring when the reranker
        is unavailable. In both cases scores are normalized to [0, 1]
        (higher is better) and results are returned sorted descending.
        """
        if not candidates:
            return []

        reranker = self._get_reranker()
        if reranker is None:
            # Fallback: LanceDB L2 distance on normalized vectors.
            # For unit vectors, L2 ranges [0, 2] (0 = identical).
            # Map to similarity [0, 1] (higher = better).
            for c in candidates:
                try:
                    d = float(c.get("score", 0.0))
                except (TypeError, ValueError):
                    d = 0.0
                c["score"] = max(0.0, min(1.0, 1.0 - d / 2.0))
            candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return candidates[:top_k]

        # Build pairs for reranker
        pairs = [[query, c["text"]] for c in candidates]
        # normalize=True applies sigmoid so scores land in [0, 1].
        scores = reranker.compute_score(pairs, normalize=True)

        # Normalize to list if single result
        if not isinstance(scores, list):
            scores = [scores]

        # Attach rerank scores and sort (higher is better)
        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        # Return top_k, mapping rerank_score to score for downstream
        results = candidates[:top_k]
        for r in results:
            r["score"] = r.get("rerank_score", r.get("score", 0))
        return results

    def search_with_rerank(
        self,
        query: str,
        top_k: int = 3,
        candidate_multiplier: int = 4,
        where: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search with reranking: retrieve candidates, rerank, return top_k.

        Retrieves top_k * candidate_multiplier results from vector search,
        then uses BGE reranker to pick the most relevant top_k.
        """
        candidates = self.search(query, top_k=top_k * candidate_multiplier, where=where)
        if not candidates:
            return []
        return self.rerank(query, candidates, top_k=top_k)

    def clear(self) -> None:
        """Delete all documents from the store."""
        self._ensure_initialized()
        if self._table is None:
            return
        count = self._table.count_rows()
        if count > 0:
            # Drop and recreate empty table
            self._db.drop_table(_TABLE_NAME)
            self._table = None
            logger.info("VectorStore: cleared %d documents", count)

    def count(self) -> int:
        self._ensure_initialized()
        if self._table is None:
            return 0
        return self._table.count_rows()

    def list_sources(self) -> List[str]:
        """Return unique source filenames in the store."""
        self._ensure_initialized()
        if self._table is None or self._table.count_rows() == 0:
            return []
        # Use LanceDB projection to load only the source column, not all data
        try:
            all_data = self._table.to_arrow(columns=["source"]).to_pylist()
        except Exception:
            # Fallback: load all columns if projection not supported
            all_data = self._table.to_arrow().to_pylist()
        sources = sorted({r.get("source", "") for r in all_data if r and r.get("source")})
        return sources


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_source(source: str) -> str:
    """Create a short stable hash for a source path."""
    import hashlib
    return hashlib.md5(source.encode()).hexdigest()[:12]


def _normalize(vec: List[float]) -> List[float]:
    """Normalize a vector to unit length for cosine similarity via L2 distance."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]

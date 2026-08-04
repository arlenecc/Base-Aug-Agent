"""Vector store — embedding + similarity search using ChromaDB, with BGE reranker."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default embedding model — small, fast, multilingual
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"


class VectorStore:
    """Local vector store backed by ChromaDB with sentence-transformers embeddings.

    Supports optional BGE reranker for precise post-retrieval scoring.
    """

    def __init__(
        self,
        persist_dir: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        rerank_model: str = DEFAULT_RERANK_MODEL,
    ):
        self._persist_dir = persist_dir
        self._embedding_model = embedding_model
        self._rerank_model = rerank_model
        self._client = None
        self._collection = None
        self._ef = None  # embedding function
        self._initialized = False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except ImportError:
            raise ImportError(
                "chromadb is required for RAG vector storage. pip install chromadb"
            )

        os.makedirs(self._persist_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(path=self._persist_dir)

        # Always use local_files_only to avoid multi-minute hangs when
        # HuggingFace is unreachable. The model must be pre-cached.
        try:
            self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self._embedding_model,
                local_files_only=True,
            )
        except Exception:
            raise ImportError(
                f"Failed to load embedding model '{self._embedding_model}'. "
                "Ensure sentence-transformers is installed and the model is cached. "
                "Run: python -c \"from sentence_transformers import SentenceTransformer; "
                f"SentenceTransformer('{self._embedding_model}')\""
            )

        self._collection = self._client.get_or_create_collection(
            name="knowledge_base",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        self._initialized = True
        logger.info(
            "VectorStore initialized: %d documents in collection",
            self._collection.count(),
        )

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

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            source = chunk.get("source", "unknown")
            chunk_idx = chunk.get("chunk_index", i)
            chunk_id = f"{_hash_source(source)}_{chunk_idx}"
            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append({
                "source": source,
                "chunk_index": chunk_idx,
            })

        # Batch insert
        total = 0
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            # Upsert: if the same ID exists, update it
            self._collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
            )
            total += len(batch_ids)

        logger.info("VectorStore: added %d chunks", total)
        return total

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search the vector store for relevant chunks.

        Returns list of {text, source, chunk_index, score} dicts.
        """
        self._ensure_initialized()
        if self._collection.count() == 0:
            return []

        kwargs: Dict[str, Any] = {
            "query_texts": [query],
            "n_results": top_k,
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        output: List[Dict[str, Any]] = []
        if not results.get("ids") or not results["ids"][0]:
            return output

        for i in range(len(results["ids"][0])):
            output.append({
                "text": results["documents"][0][i] if results.get("documents") else "",
                "source": results["metadatas"][0][i].get("source", "") if results.get("metadatas") else "",
                "chunk_index": results["metadatas"][0][i].get("chunk_index", 0) if results.get("metadatas") else 0,
                "score": results["distances"][0][i] if results.get("distances") else 0.0,
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
            # Fallback: ChromaDB cosine distance d in [0, 2] (lower = closer).
            # Map to a similarity score in [0, 1] (higher = better) and sort
            # descending so the fallback stays consistent with the reranker.
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
        where: Optional[Dict[str, Any]] = None,
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
        """Delete all documents from the collection."""
        self._ensure_initialized()
        count = self._collection.count()
        if count > 0:
            all_ids = self._collection.get()["ids"]
            self._collection.delete(ids=all_ids)
            logger.info("VectorStore: cleared %d documents", count)

    def count(self) -> int:
        self._ensure_initialized()
        return self._collection.count()

    def list_sources(self) -> List[str]:
        """Return unique source filenames in the store."""
        self._ensure_initialized()
        if self._collection.count() == 0:
            return []
        all_meta = self._collection.get()["metadatas"]
        sources = sorted(set(m.get("source", "") for m in all_meta if m))
        return sources


def _hash_source(source: str) -> str:
    """Create a short stable hash for a source path."""
    import hashlib
    return hashlib.md5(source.encode()).hexdigest()[:12]
"""RAG engine — orchestrates document ingestion, chunking, embedding, and retrieval."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .parsers import extract_directory, extract_text
from .cleaner import clean_text, normalize_markdown
from .chunker import chunk_documents
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGEngine:
    """Local knowledge base engine.

    Ingests documents from a knowledge base directory, extracts text,
    cleans it, chunks it, embeds it, and stores vectors for retrieval.
    """

    def __init__(self, workspace: str, knowledge_base: str = ""):
        self._workspace = workspace
        self._knowledge_base = knowledge_base
        self._rag_dir = os.path.join(workspace, "rag")
        self._markdown_dir = os.path.join(self._rag_dir, "documents")
        self._vector_dir = os.path.join(self._rag_dir, "vectors")
        self._store: Optional[VectorStore] = None

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def rag_dir(self) -> str:
        return self._rag_dir

    @property
    def markdown_dir(self) -> str:
        return self._markdown_dir

    @property
    def knowledge_base(self) -> str:
        return self._knowledge_base

    @knowledge_base.setter
    def knowledge_base(self, value: str) -> None:
        self._knowledge_base = value

    # ------------------------------------------------------------------
    # ingestion pipeline
    # ------------------------------------------------------------------

    def ingest(self, force: bool = False) -> Dict[str, Any]:
        """Run the full ingestion pipeline on the knowledge base directory.

        Returns a summary dict with statistics.
        """
        if not self._knowledge_base or not os.path.isdir(self._knowledge_base):
            return {"error": f"Knowledge base directory not found: {self._knowledge_base}"}

        os.makedirs(self._rag_dir, exist_ok=True)
        os.makedirs(self._markdown_dir, exist_ok=True)

        stats: Dict[str, Any] = {
            "knowledge_base": self._knowledge_base,
            "files_found": 0,
            "files_extracted": 0,
            "files_skipped": 0,
            "total_chars": 0,
            "chunks": 0,
            "errors": [],
        }

        # Step 1: Extract text from all documents
        logger.info("RAG: extracting documents from %s", self._knowledge_base)
        extracted = extract_directory(self._knowledge_base, recursive=True)
        stats["files_found"] = len(extracted)

        if not extracted:
            logger.info("RAG: no supported documents found")
            return stats

        # Step 2: Clean and save as markdown
        documents: List[Dict[str, Any]] = []
        for filepath, raw_text in extracted:
            rel_path = os.path.relpath(filepath, self._knowledge_base)
            md_name = _safe_filename(rel_path) + ".md"
            md_path = os.path.join(self._markdown_dir, md_name)

            if os.path.exists(md_path) and not force:
                # Already processed — read from cache
                with open(md_path, "r", encoding="utf-8") as f:
                    cleaned = f.read()
                stats["files_skipped"] += 1
            else:
                cleaned = clean_text(raw_text)
                cleaned = normalize_markdown(cleaned)
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(cleaned)
                stats["files_extracted"] += 1

            if cleaned.strip():
                documents.append({"source": filepath, "text": cleaned})
                stats["total_chars"] += len(cleaned)

        # Step 3: Chunk documents
        chunks = chunk_documents(documents)
        logger.info("RAG: %d documents -> %d chunks", len(documents), len(chunks))

        # Step 4: Embed and store
        store = self._get_store()
        if force:
            store.clear()
        try:
            added = store.add(chunks)
        except Exception as e:
            logger.error("RAG: vector store add failed after clear: %s", e)
            stats["chunks"] = 0
            stats["errors"].append(f"Vector store add failed: {e}")
            return stats
        stats["chunks"] = added

        logger.info(
            "RAG ingestion complete: %d files -> %d chunks stored",
            stats["files_extracted"] + stats["files_skipped"], added,
        )
        return stats

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search the knowledge base with reranking for precise results.

        Retrieves top_k * 4 candidates from vector search, then uses BGE
        reranker to pick the most relevant top_k results.
        """
        store = self._get_store()
        if store.count() == 0:
            return []
        return store.search_with_rerank(query, top_k=top_k)

    def search_formatted(self, query: str, top_k: int = 3) -> str:
        """Search and return formatted results as a string."""
        results = self.search(query, top_k)
        if not results:
            return "（知识库中未找到相关内容）"

        lines = [f"从知识库检索到 {len(results)} 条相关内容（已重排序）：\n"]
        for i, r in enumerate(results, 1):
            source = os.path.basename(r["source"])
            lines.append(f"### [{i}] 来源: {source} (相关度: {r['score']:.4f})")
            lines.append(r["text"])
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # management
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return current status of the knowledge base."""
        store = self._get_store()
        return {
            "knowledge_base": self._knowledge_base,
            "rag_dir": self._rag_dir,
            "chunks_stored": store.count(),
            "sources": store.list_sources(),
            "has_knowledge_base": bool(self._knowledge_base and os.path.isdir(self._knowledge_base)),
        }

    def clear(self) -> None:
        """Clear all stored vectors and cached markdown files."""
        store = self._get_store()
        store.clear()
        # Also clear cached markdown
        if os.path.isdir(self._markdown_dir):
            import shutil
            shutil.rmtree(self._markdown_dir, ignore_errors=True)
            os.makedirs(self._markdown_dir, exist_ok=True)
        logger.info("RAG: knowledge base cleared")

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _get_store(self) -> VectorStore:
        if self._store is None:
            self._store = VectorStore(persist_dir=self._vector_dir)
        return self._store


def _safe_filename(path: str) -> str:
    """Convert a relative path to a safe filename."""
    # Replace path separators and special chars
    safe = path.replace(os.sep, "_").replace(" ", "_")
    # Remove any remaining problematic chars
    safe = "".join(c for c in safe if c.isalnum() or c in "._-")
    return safe or "document"
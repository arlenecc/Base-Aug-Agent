"""RAG engine — orchestrates document ingestion, chunking, embedding, and retrieval."""
from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from .parsers import extract_directory, extract_text, SUPPORTED_EXTENSIONS
from .cleaner import clean_text, normalize_markdown
from .chunker import chunk_documents
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

# Default number of worker threads for parallel ingestion. Tuned for typical
# local-disk + CPU-bound parsing workloads (PDF OCR, docx parsing, etc.).
# Embedding/vector-store writes stay serial because LanceDB's table lock is
# not thread-safe and the embedding model instance can't be shared across threads.
DEFAULT_PARALLEL_WORKERS = 4


class RAGEngine:
    """Local knowledge base engine.

    Ingests documents from a knowledge base directory, extracts text,
    cleans it, chunks it, embeds it, and stores vectors for retrieval.

    The per-document pipeline (parse → clean → write markdown → chunk) runs
    in a thread pool to parallelize CPU-bound parsing (esp. PDF OCR). Vector
    store writes remain serial for thread safety.
    """

    def __init__(
        self,
        workspace: str,
        knowledge_base: str = "",
        embedding_model: str = "",
        embedding_function: Any = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        rerank_model: str = "BAAI/bge-reranker-base",
        parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    ):
        self._workspace = workspace
        self._knowledge_base = knowledge_base
        self._rag_dir = os.path.join(workspace, "rag")
        self._markdown_dir = os.path.join(self._rag_dir, "documents")
        self._vector_dir = os.path.join(self._rag_dir, "vectors")
        self._embedding_model = embedding_model or "nomic-ai/nomic-embed-text-v1.5"
        self._custom_ef = embedding_function  # for testing injection
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._rerank_model = rerank_model
        self._parallel_workers = max(1, int(parallel_workers))
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

        Pipeline (per-document stages run in parallel with a thread pool):
            1. Walk directory + extract text   (serial — pure I/O)
            2. For each file: clean → markdown → chunk  (parallel)
            3. Embed + store all chunks        (serial — LanceDB table lock)
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
        # Lock guarding stats mutation across worker threads.
        stats_lock = threading.Lock()

        # Step 1: Walk + extract text from all documents (serial).
        # Filesystem traversal is I/O-bound and extract_directory already
        # handles per-file errors; parallelizing it would add complexity for
        # little gain (the heavy work is parsing, which we parallelize next).
        logger.info("RAG: extracting documents from %s", self._knowledge_base)

        root = Path(self._knowledge_base)
        stats["files_found"] = sum(
            1 for fp in root.rglob("*")
            if fp.is_file()
            and fp.suffix.lower() in SUPPORTED_EXTENSIONS
            and not fp.name.startswith(".")
        )

        extracted = extract_directory(
            self._knowledge_base, recursive=True, errors=stats["errors"],
        )

        if not extracted:
            logger.info("RAG: no supported documents found")
            return stats

        # Step 2: Per-document pipeline (parallel).
        # Each task: read cache or clean → write markdown → chunk.
        # Returns (filepath, cleaned_or_None, chunks_list, flag, char_count).
        logger.info(
            "RAG: processing %d files with %d worker threads",
            len(extracted), self._parallel_workers,
        )

        all_chunks: List[Dict[str, Any]] = []

        def _process_one(item):
            filepath, raw_text = item
            rel_path = os.path.relpath(filepath, self._knowledge_base)
            md_name = _safe_filename(rel_path) + ".md"
            md_path = os.path.join(self._markdown_dir, md_name)

            cache_fresh = (
                not force
                and os.path.exists(md_path)
                and os.path.exists(filepath)
                and os.path.getmtime(filepath) <= os.path.getmtime(md_path)
            )
            if cache_fresh:
                try:
                    with open(md_path, "r", encoding="utf-8") as f:
                        cleaned = f.read()
                except OSError:
                    cleaned = ""
                flag = "skipped"
            else:
                cleaned = clean_text(raw_text)
                cleaned = normalize_markdown(cleaned)
                try:
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(cleaned)
                except OSError as e:
                    # Markdown write failure is non-fatal — keep going with
                    # the in-memory cleaned text so the file still gets
                    # chunked and embedded.
                    logger.warning("RAG: failed to write markdown cache for %s: %s", filepath, e)
                flag = "extracted"

            if not cleaned.strip():
                return (filepath, None, [], flag, 0)

            # Chunk this single document now so chunking is also parallel.
            doc_chunks = chunk_documents(
                [{"source": filepath, "text": cleaned}],
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
            return (filepath, cleaned, doc_chunks, flag, len(cleaned))

        with ThreadPoolExecutor(max_workers=self._parallel_workers) as ex:
            futures = {ex.submit(_process_one, item): item for item in extracted}
            for fut in as_completed(futures):
                filepath = futures[fut][0]
                try:
                    _, cleaned, doc_chunks, flag, char_count = fut.result()
                except Exception as e:
                    msg = f"{os.path.basename(filepath)}: {e}"
                    logger.warning("RAG: processing failed for '%s': %s", filepath, e)
                    with stats_lock:
                        stats["errors"].append(msg)
                    continue

                with stats_lock:
                    if flag == "skipped":
                        stats["files_skipped"] += 1
                    else:
                        stats["files_extracted"] += 1
                    if cleaned:
                        stats["total_chars"] += char_count
                    all_chunks.extend(doc_chunks)

        if not all_chunks:
            logger.info("RAG: no chunks produced from %d files", len(extracted))
            return stats

        logger.info(
            "RAG: %d files -> %d chunks total",
            stats["files_extracted"] + stats["files_skipped"], len(all_chunks),
        )

        # Step 3: Embed and store (serial — LanceDB table writes are not
        # thread-safe and the embedding model instance can't be shared).
        store = self._get_store()
        if force:
            store.clear()
        try:
            added = store.add(all_chunks)
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
            self._store = VectorStore(
                persist_dir=self._vector_dir,
                embedding_model=self._embedding_model,
                embedding_function=self._custom_ef,
            )
        return self._store


def _safe_filename(path: str) -> str:
    """Convert a relative path to a safe, collision-free filename.

    Uses a short hash of the full relative path to avoid collisions between
    e.g. ``dir1/file.txt`` and ``dir1_file.txt`` which would otherwise produce
    the same safe name.
    """
    import hashlib
    h = hashlib.md5(path.encode("utf-8")).hexdigest()[:12]
    # Build a readable prefix from the basename for easier debugging
    base = os.path.basename(path)
    safe_base = "".join(c for c in base if c.isalnum() or c in "._-")[:32]
    prefix = safe_base or "document"
    return f"{prefix}_{h}"
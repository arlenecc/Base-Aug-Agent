"""RAG engine — orchestrates document ingestion, chunking, embedding, and retrieval."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from .parsers import extract_text, SUPPORTED_EXTENSIONS
from .cleaner import clean_text, normalize_markdown
from .chunker import chunk_documents
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

# Default number of worker threads for parallel ingestion. Tuned for typical
# local-disk + CPU-bound parsing workloads (PDF OCR, docx parsing, etc.).
# Embedding/vector-store writes stay serial because LanceDB's table lock is
# not thread-safe and the embedding model instance can't be shared across threads.
DEFAULT_PARALLEL_WORKERS = 4

# worker→主线程队列容量：限制 worker 最多超前产出多少文件的 chunks，
# 超过后 worker 阻塞（背压），避免大知识库时 chunks 在内存堆积。
_QUEUE_CAPACITY_FACTOR = 2  # = parallel_workers * 2

# 哨兵：worker 全部完成后投递到队列，主线程据此退出消费循环
_SENTINEL = object()


# ---------------------------------------------------------------------------
# Manifest — tracks which documents have been processed so incremental syncs
# can skip unchanged files entirely (no chunk, no embed, no upsert).
# ---------------------------------------------------------------------------

def _manifest_path(rag_dir: str) -> str:
    """Path to the incremental-sync manifest inside the RAG directory."""
    return os.path.join(rag_dir, "manifest.json")


def _load_manifest(rag_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load the document manifest.

    Returns a dict mapping source_path -> {mtime, size, chunk_count, content_hash}.
    Returns {} if the manifest is missing or corrupted.
    """
    path = _manifest_path(rag_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("files"), dict):
            return data["files"]
        return {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_manifest(rag_dir: str, files: Dict[str, Dict[str, Any]]) -> None:
    """Persist the manifest atomically (temp + os.replace)."""
    os.makedirs(rag_dir, exist_ok=True)
    path = _manifest_path(rag_dir)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"files": files}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _file_signature(filepath: str) -> Dict[str, Any]:
    """Compute the lightweight signature of a file (no content read).

    Used to decide whether a file needs reprocessing without parsing it.
    """
    st = os.stat(filepath)
    return {"mtime": st.st_mtime, "size": st.st_size}


def _content_hash(text: str) -> str:
    """Stable hash of cleaned text content.

    Used to detect content changes even when mtime/size are unchanged
    (rare but possible). Also stored so manual manifest inspection is
    traceable.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]



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
        self._store_lock = threading.Lock()  # 保护 _store 懒初始化的线程安全
        self._cancelled = threading.Event()  # 取消标志，cancel() 设置后各阶段检查退出

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
    # ingestion pipeline (流式：worker 并行解析+chunk，主线程增量 embed+store)
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """请求取消正在进行的 ingest。各阶段检查 _cancelled 后提前退出。"""
        self._cancelled.set()

    def _reset_cancel(self) -> None:
        self._cancelled.clear()

    def _is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def ingest(
        self,
        force: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """Run the ingestion pipeline on the knowledge base directory.

        增量同步：通过 manifest.json 记录每个已处理文档的签名（mtime+size）。
        - 未修改文件：完全跳过（不 parse、不 chunk、不 embed、不 upsert）
        - 新增/修改文件：完整处理（parse → clean → chunk → embed → upsert）
        - 已删除文件：清理对应向量和 manifest 条目（无论 force 与否）

        流式架构：worker 池并行解析+chunk，主线程通过有界队列消费并增量写入
        vector store。内存占用 = O(parallel_workers * 单文件 chunks) + O(batch_size)，
        不再随知识库总规模线性增长。

        Args:
            force: 强制重新处理所有文件（忽略 manifest，仍走 upsert，不清库）
            progress_callback: 回调 (done, total, current_file) -> None

        Returns:
            统计 dict
        """
        self._reset_cancel()
        if not self._knowledge_base or not os.path.isdir(self._knowledge_base):
            logger.warning("RAG ingest: knowledge base dir not found: %s", self._knowledge_base)
            return {"error": f"Knowledge base directory not found: {self._knowledge_base}"}

        logger.info(
            "RAG ingest start: kb=%s force=%s workers=%d chunk_size=%d overlap=%d",
            self._knowledge_base, force, self._parallel_workers,
            self._chunk_size, self._chunk_overlap,
        )
        os.makedirs(self._rag_dir, exist_ok=True)
        os.makedirs(self._markdown_dir, exist_ok=True)

        stats: Dict[str, Any] = {
            "knowledge_base": self._knowledge_base,
            "files_found": 0,
            "files_extracted": 0,
            "files_skipped": 0,
            "files_deleted": 0,
            "total_chars": 0,
            "chunks": 0,
            "errors": [],
            "cancelled": False,
        }
        stats_lock = threading.Lock()

        # Step 1: 扫描知识库目录，收集所有支持的文件
        logger.info("RAG: scanning %s", self._knowledge_base)
        root = Path(self._knowledge_base)
        all_files: List[str] = []
        for fp in root.rglob("*"):
            if (fp.is_file()
                    and fp.suffix.lower() in SUPPORTED_EXTENSIONS
                    and not fp.name.startswith(".")):
                all_files.append(str(fp))
        total_files = len(all_files)
        stats["files_found"] = total_files
        if total_files == 0:
            # Even with no files, clean up manifest entries for deleted sources
            # and save the (now empty) manifest so the deletion persists.
            new_manifest = _load_manifest(self._rag_dir)
            self._cleanup_deleted_files(all_files, stats, stats_lock,
                                        old_manifest=new_manifest, new_manifest=new_manifest)
            try:
                _save_manifest(self._rag_dir, new_manifest)
            except OSError as e:
                logger.warning("RAG: failed to save manifest: %s", e)
            return stats
        if self._is_cancelled():
            stats["cancelled"] = True
            return stats

        # Step 2: load manifest and partition files into "to-process" vs "skip"
        manifest = _load_manifest(self._rag_dir)
        new_manifest: Dict[str, Dict[str, Any]] = {}
        files_to_process: List[str] = []  # new/modified files needing full pipeline
        for filepath in all_files:
            sig = _file_signature(filepath)
            prev = manifest.get(filepath)
            if (not force
                    and prev is not None
                    and prev.get("mtime") == sig["mtime"]
                    and prev.get("size") == sig["size"]):
                # Unchanged: carry over manifest entry, skip entirely.
                new_manifest[filepath] = prev
                with stats_lock:
                    stats["files_skipped"] += 1
                if progress_callback is not None:
                    progress_callback(stats["files_extracted"] + stats["files_skipped"],
                                      total_files, os.path.basename(filepath))
            else:
                files_to_process.append(filepath)
        logger.info(
            "RAG: %d files found, %d to process, %d unchanged (skipped)",
            total_files, len(files_to_process), stats["files_skipped"],
        )

        store = self._get_store()

        # Step 3: worker 并行解析+chunk，主线程消费并增量写入
        # 只有 files_to_process 才进入 worker 池——跳过的文件不解析不 chunk
        if files_to_process:
            q: "queue.Queue[Any]" = queue.Queue(maxsize=self._parallel_workers * _QUEUE_CAPACITY_FACTOR)

            def _worker(filepaths: List[str]) -> None:
                """单个 worker：处理分配给它的文件列表，把 chunks 推入队列。"""
                try:
                    for filepath in filepaths:
                        if self._is_cancelled():
                            return
                        t0 = time.monotonic()
                        try:
                            # ---- 整个文件处理流程（extract → clean → chunk → put）----
                            # 包在一个 try 块中，任何阶段出错都记录到 errors
                            # 并跳过该文件，而不是让 worker 静默退出。
                            text = extract_text(filepath)
                        except Exception as e:
                            msg = f"{os.path.basename(filepath)}: {e}"
                            logger.warning("RAG: extract failed for '%s' after %.2fs: %s",
                                           filepath, time.monotonic() - t0, e)
                            with stats_lock:
                                stats["errors"].append(msg)
                            continue
                        if not text.strip():
                            msg = f"{os.path.basename(filepath)}: extracted text is empty"
                            logger.warning(msg)
                            with stats_lock:
                                stats["errors"].append(msg)
                            continue
                        logger.debug(
                            "RAG: extracted %s (%d chars in %.2fs)",
                            os.path.basename(filepath), len(text), time.monotonic() - t0,
                        )
                        # clean → chunk → put 包在 try 中，避免 clean_text/
                        # normalize_markdown/chunk_documents 异常导致 worker
                        # 静默退出（异常会丢失，不记录到 stats["errors"]）。
                        try:
                            rel_path = os.path.relpath(filepath, self._knowledge_base)
                            md_name = _safe_filename(rel_path) + ".md"
                            md_path = os.path.join(self._markdown_dir, md_name)
                            if (not force
                                    and os.path.exists(md_path)
                                    and os.path.getmtime(filepath) <= os.path.getmtime(md_path)):
                                try:
                                    with open(md_path, "r", encoding="utf-8") as f:
                                        cleaned = f.read()
                                except OSError:
                                    cleaned = ""
                            else:
                                cleaned = clean_text(text)
                                cleaned = normalize_markdown(cleaned)
                                try:
                                    with open(md_path, "w", encoding="utf-8") as f:
                                        f.write(cleaned)
                                except OSError as e:
                                    logger.warning("RAG: failed to write markdown cache for %s: %s", filepath, e)
                            del text

                            if not cleaned.strip():
                                continue

                            c_hash = _content_hash(cleaned)
                            cleaned_len = len(cleaned)
                            doc_chunks = chunk_documents(
                                [{"source": filepath, "text": cleaned}],
                                chunk_size=self._chunk_size,
                                chunk_overlap=self._chunk_overlap,
                            )
                            logger.debug(
                                "RAG: chunked %s → %d chunks in %.2fs",
                                os.path.basename(filepath), len(doc_chunks),
                                time.monotonic() - t0,
                            )
                            del cleaned
                        except Exception as e:
                            # clean/chunk 阶段异常：记录并跳过该文件，不中断整个同步。
                            msg = f"{os.path.basename(filepath)}: clean/chunk failed: {e}"
                            logger.error("RAG: %s", msg, exc_info=True)
                            with stats_lock:
                                stats["errors"].append(msg)
                            continue

                        with stats_lock:
                            stats["files_extracted"] += 1
                            stats["total_chars"] += cleaned_len

                        if self._is_cancelled():
                            return
                        # 背压：队列满时阻塞，避免内存堆积。
                        # 使用 timeout + 循环检查取消标志，避免主线程因
                        # 取消/异常停止消费后 worker 永久阻塞在 put 上。
                        item = (filepath, doc_chunks, c_hash, len(doc_chunks))
                        while not self._is_cancelled():
                            try:
                                q.put(item, timeout=2.0)
                                break
                            except queue.Full:
                                continue
                        else:
                            return
                        # Release local references so memory is freed before
                        # the next file is loaded (important for large PDFs:
                        # without this, doc_chunks stays alive until next iter).
                        del doc_chunks, item
                except Exception as e:
                    # 兜底：任何未预期的异常都记录，避免 worker 静默死亡。
                    logger.error("RAG worker unexpected error: %s", e, exc_info=True)
                    with stats_lock:
                        stats["errors"].append(f"worker: {e}")
                finally:
                    # 投递哨兵也要带超时，防止队列满且主线程已退出消费时
                    # worker 永久卡在 put。
                    try:
                        q.put(_SENTINEL, timeout=5.0)
                    except queue.Full:
                        pass

            # 把文件列表分片给各 worker（保证负载均衡）
            chunks_per_worker = (len(files_to_process) + self._parallel_workers - 1) // self._parallel_workers
            file_shards = [
                files_to_process[i * chunks_per_worker:(i + 1) * chunks_per_worker]
                for i in range(self._parallel_workers)
            ]

            threads = []
            for shard in file_shards:
                t = threading.Thread(target=_worker, args=(shard,), daemon=True)
                t.start()
                threads.append(t)

            # Step 4: 主线程消费队列，逐文件增量写入 vector store
            def _chunk_iter() -> Iterator[Dict[str, Any]]:
                sentinels_seen = 0
                expected_sentinels = self._parallel_workers
                empty_streak = 0
                while sentinels_seen < expected_sentinels:
                    if self._is_cancelled():
                        return
                    try:
                        item = q.get(timeout=2.0)
                        empty_streak = 0
                    except queue.Empty:
                        empty_streak += 1
                        # Safety: if the queue has been empty for a long time
                        # AND all worker threads have exited, a sentinel was
                        # likely lost (worker couldn't put it due to a full
                        # queue + cancellation). Exit to avoid hanging forever.
                        if empty_streak >= 5 and all(not t.is_alive() for t in threads):
                            logger.warning(
                                "RAG: %d/%d sentinels received, all workers done — "
                                "exiting consumer (sentinel likely lost)",
                                sentinels_seen, expected_sentinels,
                            )
                            return
                        continue
                    if item is _SENTINEL:
                        sentinels_seen += 1
                        continue
                    filepath, doc_chunks, c_hash, n_chunks = item
                    # Update manifest for this processed file.
                    # _file_signature 可能因文件被删除/权限问题抛 OSError，
                    # 隔离异常避免整个 ingest 失败——跳过 manifest 更新即可，
                    # chunks 已写入向量库，下次同步会重新处理（manifest 未更新）。
                    try:
                        new_manifest[filepath] = {
                            **_file_signature(filepath),
                            "content_hash": c_hash,
                            "chunk_count": n_chunks,
                        }
                    except OSError as e:
                        logger.warning("RAG: failed to update manifest for %s: %s", filepath, e)
                    for c in doc_chunks:
                        yield c
                    # 进度反馈：每消费完一个文件回调一次。
                    # 隔离回调异常，避免 UI 端异常中断整个 ingest。
                    if progress_callback is not None:
                        try:
                            with stats_lock:
                                done = stats["files_extracted"] + stats["files_skipped"]
                            progress_callback(done, total_files, os.path.basename(filepath))
                        except Exception as e:
                            logger.warning("RAG: progress_callback error: %s", e)

            def _on_batch(batch_added: int, total_added: int) -> None:
                if progress_callback is not None:
                    try:
                        progress_callback(stats["files_extracted"] + stats["files_skipped"],
                                          total_files, f"已写入 {total_added} chunks")
                    except Exception as e:
                        logger.warning("RAG: on_batch callback error: %s", e)

            try:
                logger.info("RAG: starting add_streaming (batch_size=100)")
                added = store.add_streaming(_chunk_iter(), batch_size=100, on_batch=_on_batch)
                stats["chunks"] = added
                logger.info("RAG: add_streaming complete — %d chunks written", added)
            except Exception as e:
                logger.error("RAG: streaming add failed: %s", e, exc_info=True)
                # Cancel workers so they stop producing chunks into a queue
                # that no one is consuming. Without this, workers keep running
                # (blocked on q.put with timeout loops) until they've processed
                # all their files — wasting CPU and holding memory.
                self.cancel()
                stats["chunks"] = store.count()
                stats["errors"].append(f"Vector store add failed: {e}")

            # 等待所有 worker 线程结束
            for t in threads:
                t.join(timeout=5.0)
        else:
            # All files skipped — no new chunks written. Report current store count.
            stats["chunks"] = store.count()
            logger.info("RAG: all %d files skipped (already synced)", total_files)

        if self._is_cancelled():
            stats["cancelled"] = True
            stats["chunks"] = store.count()
            logger.info("RAG ingest cancelled: %d/%d files extracted, %d chunks stored",
                        stats["files_extracted"], stats["files_found"], stats["chunks"])
            return stats

        # Step 5: 清理已删除文件对应的旧向量 + manifest 条目
        self._cleanup_deleted_files(all_files, stats, stats_lock, manifest, new_manifest)

        # Step 6: 保存 manifest（即使全部 skip 也保存，以更新删除的条目）
        try:
            _save_manifest(self._rag_dir, new_manifest)
            logger.info("RAG: manifest saved (%d entries) to %s",
                        len(new_manifest), _manifest_path(self._rag_dir))
        except OSError as e:
            logger.warning("RAG: failed to save manifest: %s", e)

        logger.info(
            "RAG ingestion complete: %d found, %d extracted, %d skipped, "
            "%d deleted, %d chunks stored (cancelled=%s)",
            stats["files_found"], stats["files_extracted"], stats["files_skipped"],
            stats["files_deleted"], stats["chunks"], stats["cancelled"],
        )
        return stats

    def _cleanup_deleted_files(
        self,
        current_files: List[str],
        stats: Dict[str, Any],
        stats_lock: threading.Lock,
        old_manifest: Optional[Dict[str, Dict[str, Any]]] = None,
        new_manifest: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Delete vectors and manifest entries for files removed from the KB.

        Runs on every ingest (force or not): if a file was in the manifest
        but is no longer on disk, its vectors are stale and must be removed.
        """
        if old_manifest is None:
            old_manifest = _load_manifest(self._rag_dir)
        if new_manifest is None:
            new_manifest = dict(old_manifest)  # caller will update
        current_set = set(current_files)
        removed = [fp for fp in old_manifest if fp not in current_set]
        if not removed:
            return
        store = self._get_store()
        for source in removed:
            store.delete_by_source(source)
            new_manifest.pop(source, None)
            with stats_lock:
                stats["files_deleted"] += 1
            logger.info("RAG: removed deleted file from store: %s", source)

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search the knowledge base with reranking for precise results.

        Retrieves top_k * 4 candidates from vector search, then uses BGE
        reranker to pick the most relevant top_k results.
        """
        t0 = time.monotonic()
        store = self._get_store()
        store_count = store.count()
        if store_count == 0:
            logger.info("RAG search: store empty, returning 0 results (query=%r)", query[:60])
            return []
        logger.info(
            "RAG search: query=%r top_k=%d store_count=%d",
            query[:60], top_k, store_count,
        )
        results = store.search_with_rerank(query, top_k=top_k)
        logger.info(
            "RAG search: returned %d results in %.2fs",
            len(results), time.monotonic() - t0,
        )
        return results

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
        """Clear all stored vectors, cached markdown files, and manifest."""
        store = self._get_store()
        store.clear()
        # Also clear cached markdown
        if os.path.isdir(self._markdown_dir):
            import shutil
            shutil.rmtree(self._markdown_dir, ignore_errors=True)
            os.makedirs(self._markdown_dir, exist_ok=True)
        # Clear the incremental-sync manifest so the next ingest processes
        # everything from scratch (no stale "already processed" entries).
        manifest = _manifest_path(self._rag_dir)
        if os.path.exists(manifest):
            try:
                os.remove(manifest)
            except OSError as e:
                logger.warning("RAG: failed to remove manifest: %s", e)
        logger.info("RAG: knowledge base cleared")

    def close(self) -> None:
        """Release the underlying VectorStore's resources.

        Drops the cached VectorStore (LanceDB connection, FastEmbed ONNX model,
        BGE reranker). After close(), a new VectorStore is lazily created on
        the next operation. Call this when the RAGEngine is no longer needed
        (e.g. after SyncKnowledgeWorker finishes) to avoid leaking ~600MB of
        C++ heap memory per instance.
        """
        if self._store is not None:
            with self._store_lock:
                if self._store is not None:
                    try:
                        self._store.close()
                    except Exception as e:
                        logger.debug("RAG: VectorStore close error: %s", e)
                    self._store = None

    def reload(self) -> None:
        """Drop the cached table handle so the next search sees fresh data.

        Call this after SyncKnowledgeWorker finishes writing — the agent's
        RAGEngine holds a stale table snapshot that won't reflect newly
        written rows until the table is re-opened.
        """
        with self._store_lock:
            if self._store is not None:
                try:
                    self._store.reload()
                    logger.info("RAG: VectorStore reloaded (table handle refreshed)")
                except Exception as e:
                    logger.warning("RAG: VectorStore reload error: %s", e)

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _get_store(self) -> VectorStore:
        # Double-check locking：避免并发 search() 时创建多个 VectorStore 实例
        if self._store is None:
            with self._store_lock:
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
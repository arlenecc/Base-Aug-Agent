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
        logger.debug("RAG: manifest not found at %s (first sync)", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("files"), dict):
            logger.debug("RAG: loaded manifest with %d entries from %s", len(data["files"]), path)
            return data["files"]
        logger.warning("RAG: manifest has unexpected format, ignoring")
        return {}
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("RAG: failed to load manifest: %s", e)
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
        logger.info("RAG: cancel requested")
        self._cancelled.set()

    def _reset_cancel(self) -> None:
        self._cancelled.clear()

    def _is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def ingest(
        self,
        force: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
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
        # 辅助函数：同时写 logger 和 log_callback（如果提供）
        def _log(msg: str, *args) -> None:
            if args:
                msg = msg % args
            logger.info(msg)
            if log_callback is not None:
                log_callback(msg)

        if not self._knowledge_base or not os.path.isdir(self._knowledge_base):
            logger.warning("RAG ingest: knowledge base dir not found: %s", self._knowledge_base)
            return {"error": f"Knowledge base directory not found: {self._knowledge_base}"}

        t_start = time.monotonic()
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

        # Step 1: 扫描知识库目录
        _log("━━━ Step 1/6: 扫描知识库目录 ━━━")
        _log("  正在扫描: %s", self._knowledge_base)
        root = Path(self._knowledge_base)
        all_files: List[str] = []
        ext_counts: Dict[str, int] = {}
        for fp in root.rglob("*"):
            if (fp.is_file()
                    and fp.suffix.lower() in SUPPORTED_EXTENSIONS
                    and not fp.name.startswith(".")):
                all_files.append(str(fp))
                ext = fp.suffix.lower()
                ext_counts[ext] = ext_counts.get(ext, 0) + 1
        total_files = len(all_files)
        stats["files_found"] = total_files
        ext_summary = ", ".join(f"{ext}({cnt})" for ext, cnt in sorted(ext_counts.items()))
        _log("  扫描完成: 共发现 %d 个文件 [%s]", total_files, ext_summary if ext_summary else "无")
        if progress_callback is not None:
            try:
                progress_callback(-1, 0, f"扫描完成: {total_files} 个文件")
            except Exception:
                pass
        if total_files == 0:
            # Even with no files, clean up manifest entries for deleted sources
            # and save the (now empty) manifest so the deletion persists.
            new_manifest = _load_manifest(self._rag_dir)
            self._cleanup_deleted_files(all_files, stats, stats_lock,
                                        old_manifest=dict(new_manifest),
                                        new_manifest=new_manifest)
            try:
                _save_manifest(self._rag_dir, new_manifest)
            except OSError as e:
                logger.warning("RAG: failed to save manifest: %s", e)
            return stats
        if self._is_cancelled():
            stats["cancelled"] = True
            return stats

        # Step 2: load manifest and partition
        _log("━━━ Step 2/6: 增量对比（Manifest） ━━━")
        manifest = _load_manifest(self._rag_dir)
        if manifest:
            _log("  已加载同步记录: %d 个文件有历史记录", len(manifest))
        else:
            _log("  无历史同步记录，将全量处理")
        new_manifest: Dict[str, Dict[str, Any]] = {}
        files_to_process: List[str] = []  # new/modified files needing full pipeline
        skipped_detail: List[str] = []
        new_detail: List[str] = []
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
                skipped_detail.append(f"  ✓ 跳过(未修改): {os.path.basename(filepath)}")
                if progress_callback is not None:
                    progress_callback(stats["files_extracted"] + stats["files_skipped"],
                                      total_files, os.path.basename(filepath))
            else:
                files_to_process.append(filepath)
                reason = "新增" if prev is None else "已修改"
                new_detail.append(f"  → 待处理({reason}): {os.path.basename(filepath)}")
        # 汇总输出
        if skipped_detail:
            for line in skipped_detail:
                _log(line)
        if new_detail:
            for line in new_detail:
                _log(line)
        _log(
            "  对比结果: 总计%d 文件, %d 个待处理, %d 个跳过(未修改)",
            total_files, len(files_to_process), stats["files_skipped"],
        )
        if progress_callback is not None:
            try:
                progress_callback(-1, 0,
                    f"增量对比: {len(files_to_process)}待处理, {stats['files_skipped']}跳过")
            except Exception:
                pass

        # Step 2b: 初始化向量库
        if progress_callback is not None:
            try:
                progress_callback(-1, 0, "初始化向量库...")
            except Exception:
                pass
        store = self._get_store()
        if progress_callback is not None:
            try:
                progress_callback(-1, 0, "向量库就绪")
            except Exception:
                pass

        # Step 3: worker 并行解析+chunk，主线程消费并增量写入
        # 只有 files_to_process 才进入 worker 池——跳过的文件不解析不 chunk
        if files_to_process:
            _log("━━━ Step 3/6: 解析文档 + 清洗 + 切片（%d 个文件，%d 个并行 Worker） ━━━",
                        len(files_to_process), self._parallel_workers)
            q: "queue.Queue[Any]" = queue.Queue(maxsize=self._parallel_workers * _QUEUE_CAPACITY_FACTOR)

            # 全局文件序号计数器（用于日志显示 "文件 3/10"）
            _file_counter = [0]  # 用 list 包装以在闭包中修改
            _file_counter_lock = threading.Lock()

            # 辅助函数：触发 UI 日志刷新。
            # progress_callback(done=-1, ...) 是"纯日志刷新"信号，
            # _on_sync_progress 收到 done<0 时只消费 _log_buffer 不更新进度条。
            def _flush_logs(hint: str = "") -> None:
                if progress_callback is not None:
                    try:
                        progress_callback(-1, 0, hint)
                    except Exception:
                        pass

            def _worker(filepaths: List[str]) -> None:
                """单个 worker：处理分配给它的文件列表，把 chunks 推入队列。"""
                worker_id = threading.get_ident()
                _log("  [Worker-%d] 启动，负责 %d 个文件", worker_id % 1000, len(filepaths))
                try:
                    for filepath in filepaths:
                        if self._is_cancelled():
                            _log("  [Worker-%d] 收到取消信号，退出", worker_id % 1000)
                            return
                        # 获取全局文件序号
                        with _file_counter_lock:
                            _file_counter[0] += 1
                            file_no = _file_counter[0]
                        fname = os.path.basename(filepath)
                        t0 = time.monotonic()
                        _log("  [%d/%d] 开始处理: %s", file_no, len(files_to_process), fname)
                        _flush_logs(fname)
                        try:
                            # ---- 1. 解析文件 ----
                            _log("    ├─ 解析文件: %s", fname)
                            _flush_logs(f"解析: {fname}")
                            text = extract_text(filepath)
                            t_extract = time.monotonic() - t0
                            _log("    ├─ 解析完成: %s (提取 %d 字符, 耗时 %.2fs)", fname, len(text), t_extract)
                            _flush_logs(f"解析完成: {fname}")
                        except Exception as e:
                            msg = f"{os.path.basename(filepath)}: {e}"
                            logger.warning("    └─ ❌ 解析失败: %s (耗时 %.2fs): %s", fname, time.monotonic() - t0, e)
                            with stats_lock:
                                stats["errors"].append(msg)
                            continue
                        if not text.strip():
                            msg = f"{os.path.basename(filepath)}: extracted text is empty"
                            logger.warning("    └─ ⚠ 提取文本为空: %s", fname)
                            with stats_lock:
                                stats["errors"].append(msg)
                            continue
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
                                # ---- 2. 使用缓存的 Markdown ----
                                try:
                                    with open(md_path, "r", encoding="utf-8") as f:
                                        cleaned = f.read()
                                    _log("    ├─ 清洗文本: %s (使用缓存 Markdown, %d 字符)", fname, len(cleaned))
                                except OSError:
                                    cleaned = ""
                            if not (not force and os.path.exists(md_path)
                                    and os.path.getmtime(filepath) <= os.path.getmtime(md_path)):
                                # ---- 2. 清洗文本 ----
                                t_clean = time.monotonic()
                                _log("    ├─ 清洗文本: %s (原始 %d 字符)", fname, len(text))
                                _flush_logs(f"清洗: {fname}")
                                cleaned = clean_text(text)
                                t_clean_end = time.monotonic()
                                _log("    ├─ 清洗完成: %s → %d 字符 (耗时 %.2fs)", fname, len(cleaned), t_clean_end - t_clean)
                                # ---- 3. 转换为 Markdown ----
                                _log("    ├─ 转换 Markdown: %s", fname)
                                _flush_logs(f"Markdown: {fname}")
                                cleaned = normalize_markdown(cleaned)
                                _log("    ├─ Markdown 完成: %s (%d 字符)", fname, len(cleaned))
                                _flush_logs(f"Markdown完成: {fname}")
                                try:
                                    with open(md_path, "w", encoding="utf-8") as f:
                                        f.write(cleaned)
                                    logger.debug("    ├─ Markdown 已缓存: %s", md_name)
                                except OSError as e:
                                    logger.warning("    ├─ Markdown 缓存写入失败: %s: %s", filepath, e)
                            del text

                            if not cleaned.strip():
                                logger.warning("    └─ ⚠ 清洗后文本为空: %s", fname)
                                continue

                            # ---- 4. 文档切片 ----
                            t_chunk = time.monotonic()
                            _flush_logs(f"切片: {fname}")
                            c_hash = _content_hash(cleaned)
                            cleaned_len = len(cleaned)
                            doc_chunks = chunk_documents(
                                [{"source": filepath, "text": cleaned}],
                                chunk_size=self._chunk_size,
                                chunk_overlap=self._chunk_overlap,
                            )
                            t_chunk_end = time.monotonic()
                            _log(
                                "    ├─ 文档切片: %s → %d 个切片 (每片约 %d tokens, 重叠 %d tokens, 耗时 %.2fs)",
                                fname, len(doc_chunks), self._chunk_size, self._chunk_overlap,
                                t_chunk_end - t_chunk,
                            )
                            _flush_logs(f"切片完成: {fname} ({len(doc_chunks)}片)")
                            del cleaned
                        except Exception as e:
                            # clean/chunk 阶段异常：记录并跳过该文件，不中断整个同步。
                            msg = f"{os.path.basename(filepath)}: clean/chunk failed: {e}"
                            logger.error("    └─ ❌ 清洗/切片失败: %s: %s", fname, e, exc_info=True)
                            with stats_lock:
                                stats["errors"].append(msg)
                            continue

                        with stats_lock:
                            stats["files_extracted"] += 1
                            stats["total_chars"] += cleaned_len

                        # 文件处理完成总耗时
                        _log(
                            "    └─ ✅ 文件处理完成: %s (%d 切片, 总耗时 %.2fs)",
                            fname, len(doc_chunks), time.monotonic() - t0,
                        )
                        _flush_logs(f"✅ {fname} ({len(doc_chunks)}切片)")

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
                    logger.error("  [Worker-%d] 异常退出: %s", worker_id % 1000, e, exc_info=True)
                    with stats_lock:
                        stats["errors"].append(f"worker: {e}")
                finally:
                    # 投递哨兵也要带超时，防止队列满且主线程已退出消费时
                    # worker 永久卡在 put。
                    _log("  [Worker-%d] 完成所有文件，发送完成信号", worker_id % 1000)
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

            _flush_logs(f"启动 {len(threads)} 个 Worker")

            # Step 4: 主线程消费队列，逐文件增量写入 vector store
            _log("━━━ Step 4/6: 向量化 + 存入向量库 ━━━")
            def _chunk_iter() -> Iterator[Dict[str, Any]]:
                sentinels_seen = 0
                expected_sentinels = self._parallel_workers
                empty_streak = 0
                # 使用较短的 q.get 超时（0.5s），确保取消后最多 0.5s 内能响应。
                _QGET_TIMEOUT = 0.5
                # 总超时 = empty_streak * _QGET_TIMEOUT，设为 15s（30 次 × 0.5s）
                _MAX_EMPTY_STREAK = 30
                consumer_file_no = 0
                while sentinels_seen < expected_sentinels:
                    if self._is_cancelled():
                        return
                    try:
                        item = q.get(timeout=_QGET_TIMEOUT)
                        empty_streak = 0
                    except queue.Empty:
                        empty_streak += 1
                        if empty_streak >= _MAX_EMPTY_STREAK and all(not t.is_alive() for t in threads):
                            logger.warning(
                                "  向量化消费者: %d/%d 个 Worker 完成信号已收到, 所有 Worker 已退出, 退出消费",
                                sentinels_seen, expected_sentinels,
                            )
                            return
                        continue
                    if item is _SENTINEL:
                        sentinels_seen += 1
                        _log("  收到 Worker 完成信号 (%d/%d)", sentinels_seen, expected_sentinels)
                        continue
                    filepath, doc_chunks, c_hash, n_chunks = item
                    consumer_file_no += 1
                    fname = os.path.basename(filepath)
                    _log(
                        "  [消费 %d] 向量化排队: %s (%d 个切片等待嵌入)",
                        consumer_file_no, fname, n_chunks,
                    )
                    # Update manifest for this processed file.
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
                    if progress_callback is not None:
                        try:
                            with stats_lock:
                                done = stats["files_extracted"] + stats["files_skipped"]
                            progress_callback(done, total_files, os.path.basename(filepath))
                        except Exception as e:
                            logger.warning("RAG: progress_callback error: %s", e)

            def _on_batch(batch_added: int, total_added: int) -> None:
                _log("  ├─ 向量化批次写入: +%d 切片 (累计 %d)", batch_added, total_added)
                if progress_callback is not None:
                    try:
                        progress_callback(stats["files_extracted"] + stats["files_skipped"],
                                          total_files, f"已写入 {total_added} 切片")
                    except Exception as e:
                        logger.warning("RAG: on_batch callback error: %s", e)

            try:
                _log("  开始流式向量化写入 (每批 %d 个切片)", 20)
                added = store.add_streaming(
                    _chunk_iter(),
                    batch_size=20,
                    on_batch=_on_batch,
                    is_cancelled=self._is_cancelled,
                )
                stats["chunks"] = added
                _log("  向量化写入完成: 共 %d 个切片已存入向量库", added)
            except Exception as e:
                logger.error("  向量化写入失败: %s", e, exc_info=True)
                # 注意：这里不能调用 self.cancel()——向量化写入异常（如模型下载
                # 失败、磁盘满、LanceDB 写入异常）不是用户主动取消，不应该设置
                # cancelled 标志。否则 UI 会错误地显示"同步已取消"并丢弃已
                # 解析的数据。改为将异常记录到 errors，让 ingest 正常结束。
                stats["chunks"] = store.count()
                stats["errors"].append(f"Vector store add failed: {e}")

            # 等待所有 worker 线程结束
            _log("  等待所有 Worker 线程结束...")
            for i, t in enumerate(threads):
                t.join(timeout=5.0)
                if t.is_alive():
                    logger.warning("  Worker 线程 %d 未在 5s 内结束", i + 1)
            _log("  所有 Worker 线程已结束")
        else:
            # All files skipped — no new chunks written. Report current store count.
            stats["chunks"] = store.count()
            _log("  所有 %d 个文件均未修改，无需重新处理", total_files)

        cancelled = self._is_cancelled()
        if cancelled:
            stats["cancelled"] = True
            stats["chunks"] = store.count()
            _log("⚠ 同步已被用户取消: %d/%d 文件已提取, %d 切片已存储",
                        stats["files_extracted"], stats["files_found"], stats["chunks"])

        # Step 5: 清理已删除文件
        _log("━━━ Step 5/6: 清理已删除文件 ━━━")
        self._cleanup_deleted_files(all_files, stats, stats_lock, manifest, new_manifest)
        if stats["files_deleted"] > 0:
            _log("  已清理 %d 个已删除文件的旧向量", stats["files_deleted"])
        else:
            _log("  无已删除文件需要清理")

        # Step 6: 保存 manifest
        _log("━━━ Step 6/6: 保存同步记录 ━━━")
        try:
            _save_manifest(self._rag_dir, new_manifest)
            _log("  同步记录已保存: %d 个文件条目 → %s",
                        len(new_manifest), _manifest_path(self._rag_dir))
        except OSError as e:
            logger.warning("  同步记录保存失败: %s", e)

        if cancelled:
            t_total = time.monotonic() - t_start
            _log("══════════════════════════════════════════")
            _log("⏹ 知识库同步已取消 (耗时 %.1fs): 扫描 %d 文件, 提取 %d, 跳过 %d, 切片 %d",
                t_total, stats["files_found"], stats["files_extracted"],
                stats["files_skipped"], stats["chunks"],
            )
            _log("══════════════════════════════════════════")
            return stats

        t_total = time.monotonic() - t_start
        _log("══════════════════════════════════════════")
        _log("✅ 知识库同步完成 (耗时 %.1fs)", t_total)
        _log("   扫描文件: %d | 新增/更新: %d | 跳过(未修改): %d | 清理删除: %d",
            stats["files_found"], stats["files_extracted"],
            stats["files_skipped"], stats["files_deleted"],
        )
        _log("   总字符数: %d | 向量切片: %d | 错误: %d",
            stats["total_chars"], stats["chunks"], len(stats.get("errors", [])),
        )
        if stats.get("errors"):
            _log("   错误详情:")
            for err in stats["errors"][:10]:
                _log("     - %s", err)
            if len(stats["errors"]) > 10:
                _log("     ... (还有 %d 个错误)", len(stats["errors"]) - 10)
        _log("══════════════════════════════════════════")
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
        logger.info("  发现 %d 个已删除文件，清理对应向量...", len(removed))
        store = self._get_store()
        for source in removed:
            store.delete_by_source(source)
            new_manifest.pop(source, None)
            with stats_lock:
                stats["files_deleted"] += 1
            logger.info("  🗑 已清理: %s", os.path.basename(source))

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
            logger.info("🔍 知识库检索: 向量库为空，无结果返回 (查询=%r)", query[:60])
            return []
        logger.info(
            "🔍 知识库检索开始: 查询=%r 目标结果数=%d 向量库总量=%d",
            query[:80], top_k, store_count,
        )
        logger.info("  ├─ 初检: 从 %d 个向量中检索 top %d 候选", store_count, top_k * 4)
        results = store.search_with_rerank(query, top_k=top_k)
        t_total = time.monotonic() - t0
        if results:
            logger.info("  └─ ✅ 检索完成: 返回 %d 条结果 (耗时 %.2fs)", len(results), t_total)
            for i, r in enumerate(results, 1):
                source = os.path.basename(r.get("source", ""))
                score = r.get("score", 0)
                text_preview = r.get("text", "")[:80].replace("\n", " ")
                logger.info("      [%d] %s (相关度: %.4f) | %s...", i, source, score, text_preview)
        else:
            logger.info("  └─ ⚠ 检索完成: 无匹配结果 (耗时 %.2fs)", t_total)
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
        chunk_count = store.count()
        sources = store.list_sources()
        logger.info("📊 知识库状态查询: 切片数=%d 来源文件数=%d", chunk_count, len(sources))
        return {
            "knowledge_base": self._knowledge_base,
            "rag_dir": self._rag_dir,
            "chunks_stored": chunk_count,
            "sources": sources,
            "has_knowledge_base": bool(self._knowledge_base and os.path.isdir(self._knowledge_base)),
        }

    def clear(self) -> None:
        """Clear all stored vectors, cached markdown files, and manifest."""
        logger.info("🗑 正在清空知识库...")
        store = self._get_store()
        store.clear()
        # Also clear cached markdown
        if os.path.isdir(self._markdown_dir):
            import shutil
            shutil.rmtree(self._markdown_dir, ignore_errors=True)
            os.makedirs(self._markdown_dir, exist_ok=True)
            logger.info("  已清除缓存 Markdown 文件")
        # Clear the incremental-sync manifest so the next ingest processes
        # everything from scratch (no stale "already processed" entries).
        manifest = _manifest_path(self._rag_dir)
        if os.path.exists(manifest):
            try:
                os.remove(manifest)
                logger.info("  已清除同步记录文件")
            except OSError as e:
                logger.warning("  同步记录文件删除失败: %s", e)
        logger.info("  知识库已清空")

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
                        rerank_model=self._rerank_model,
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
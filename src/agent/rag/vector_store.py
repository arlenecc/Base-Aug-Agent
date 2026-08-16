"""Vector store — embedding + similarity search using LanceDB, with BGE reranker.

Embeddings are generated locally via FastEmbed (ONNX Runtime) using
nomic-ai/nomic-embed-text-v1.5 — no API server or HuggingFace dependency.
Vectors are stored in LanceDB (Lance columnar format on local disk).
"""
from __future__ import annotations

import logging
import math
import os
import threading
import warnings
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default embedding model — quantized nomic-embed-text-v1.5 via FastEmbed (ONNX Runtime).
# Using the -Q variant: model_quantized.onnx (~137MB) instead of model.onnx (~548MB).
# Same 768-dim embeddings, same quality, 4x smaller download.
DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5-Q"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"
_TABLE_NAME = "knowledge_base"

# 文档元数据表：存储每个文档的缩略版本（目录结构 + 章节标题）与完整 Markdown，
# 用于 Meta-context + Targeted RAG（先返回全文档结构概览，再按需检索细节）。
_DOCUMENTS_TABLE_NAME = "documents"

# nomic-embed-text-v1.5 requires query/document prefixes for best results
_DOC_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "

# BM25 全文检索的预分词列名（jieba 分词后空格连接，供原生 FTS 的 simple
# tokenizer 按空格切分索引，解决中文无空格无法分词的问题）。
_FTS_COLUMN = "text_fts"

# jieba 分词器缓存（进程级单例，避免每次调用重复初始化词典）。
_jieba = None
_jieba_lock = threading.Lock()
_jieba_tried = False


def _get_jieba():
    """Lazily import and return jieba, or None if unavailable."""
    global _jieba, _jieba_tried
    if _jieba_tried:
        return _jieba
    with _jieba_lock:
        if _jieba_tried:
            return _jieba
        _jieba_tried = True
        try:
            import jieba
            _jieba = jieba
            logger.debug("jieba 分词器已加载（用于 BM25 中文全文检索）")
        except ImportError:
            _jieba = None
            logger.info("jieba 未安装，BM25 中文分词退化为空白切分")
    return _jieba


def _tokenize_for_fts(text: str) -> str:
    """Tokenize text for the BM25 full-text index.

    Uses jieba for CJK word segmentation, then joins tokens with spaces so the
    native FTS ``simple`` tokenizer can index each word.  English words are
    kept intact (jieba segments them on whitespace/punctuation).  When jieba
    is unavailable, returns the original text (English still works via
    whitespace; CJK degrades to whole-sentence tokens).
    """
    jieba = _get_jieba()
    if jieba is None or not text:
        return text or ""
    tokens = [t.strip() for t in jieba.cut(text) if t.strip()]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# FastEmbed-based embedding function (local ONNX Runtime, no API needed)
# ---------------------------------------------------------------------------


class FastEmbedEmbeddingFunction:
    """Embedding function using FastEmbed (ONNX Runtime) for local inference.

    Loads nomic-ai/nomic-embed-text-v1.5 via ONNX Runtime — no GPU, no
    HuggingFace network calls, no PyTorch dependency. The model is cached
    after first download (~130MB quantized).

    Thread-safety: ONNX Runtime InferenceSession.run() is thread-safe,
    so a single instance can be shared across RAG, graph_memory, etc.
    Use get_or_create_embedding_function() to get a shared singleton.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self._model_name = model_name
        self._model = None  # lazy-init

    def _ensure_model(self):
        if self._model is None:
            from fastembed import TextEmbedding
            logger.info("  🔧 正在加载嵌入模型: %s (首次加载, 约 130MB)...", self._model_name)
            import time as _time
            _time.sleep(0.01)  # 释放 GIL 让 UI 更新
            try:
                # Use a persistent cache dir instead of the default system
                # temp dir (/var/folders/.../T/) which macOS cleans regularly.
                # This prevents the model from being re-downloaded after
                # every system cleanup.
                cache_dir = os.path.join(
                    os.path.expanduser("~"),
                    ".cache",
                    "fastembed",
                )
                os.makedirs(cache_dir, exist_ok=True)
                # Suppress FastEmbed's "model has been updated on HuggingFace"
                # UserWarning — it fires on every instantiation and is not
                # actionable for the user.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    self._model = TextEmbedding(
                        model_name=self._model_name,
                        cache_dir=cache_dir,
                    )
            except Exception as e:
                # 模型加载失败最常见的原因是首次使用时需要从 HuggingFace
                # 下载 ONNX 模型文件（约 130MB），但网络无法访问 huggingface.co。
                # 给用户提供明确的错误信息和解决方案。
                msg = (
                    f"无法加载嵌入模型 '{self._model_name}'：{e}\n\n"
                    f"可能原因及解决方案：\n"
                    f"1. 首次使用需从 HuggingFace 下载模型（~130MB），网络无法访问\n"
                    f"   → 设置镜像站: export HF_ENDPOINT=https://hf-mirror.com\n"
                    f"2. 新版 huggingface_hub 默认使用 Xet 传输，镜像站可能不支持\n"
                    f"   → 禁用 Xet: export HF_HUB_DISABLE_XET=1\n"
                    f"3. 模型缓存目录权限不足\n"
                    f"   → 检查 ~/.cache/huggingface/ 目录权限\n"
                    f"4. 也可手动下载模型到本地缓存：\n"
                    f"   HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \\\n"
                    f"     python3 -c \"from huggingface_hub import snapshot_download; \\\n"
                    f"     snapshot_download('nomic-ai/nomic-embed-text-v1.5')\"\n\n"
                    f"原始错误: {type(e).__name__}: {e}"
                )
                logger.error(msg)
                raise RuntimeError(msg) from e
            logger.info("  ✅ 嵌入模型加载完成: %s", self._model_name)
            _time.sleep(0.01)  # 释放 GIL 让 UI 更新

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

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts (with document prefix).

        Batched: a single ONNX forward pass for the whole list, much faster
        than calling embed_query() per text.
        """
        return self(texts)

    def name(self) -> str:
        return f"fastembed:{self._model_name}"


# ---------------------------------------------------------------------------
# Process-level embedding function singleton
# ---------------------------------------------------------------------------

_EF_SINGLETONS: Dict[str, "FastEmbedEmbeddingFunction"] = {}
_EF_SINGLETONS_LOCK = threading.Lock()


def get_or_create_embedding_function(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> "FastEmbedEmbeddingFunction":
    """Return a process-wide shared FastEmbedEmbeddingFunction.

    The ONNX model (~137MB) is loaded once and shared across all callers
    (RAG VectorStore, graph_memory GraphMemoryStore, etc.).  ONNX Runtime
    InferenceSession.run() is thread-safe, so concurrent callers can share
    a single model instance without locking on inference.

    This avoids loading multiple copies of the same ~137MB model when both
    RAG and graph_memory are active (which previously cost ~274MB).
    """
    with _EF_SINGLETONS_LOCK:
        ef = _EF_SINGLETONS.get(model_name)
        if ef is None:
            ef = FastEmbedEmbeddingFunction(model_name=model_name)
            _EF_SINGLETONS[model_name] = ef
        return ef


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
        self._documents_table = None  # 文档元数据表（缩略版本 + markdown）
        self._documents_lock = threading.Lock()  # 保护 documents 表懒创建 + 写操作
        # documents 表内容缓存：避免 get_document_digest / list_documents 每次
        # 都全表 to_pylist()（表内含完整 markdown，反复全量扫描既慢又占内存）。
        # 写入（upsert/delete/clear）时置 None 失效，下次读取时重新加载。
        self._documents_cache: Optional[List[Dict[str, Any]]] = None
        self._ef = None  # embedding function
        self._initialized = False
        self._table_checked = False  # 是否已检查过 table_names（避免重复磁盘扫描）
        self._fts_ready = False  # BM25 全文索引是否已就绪
        self._fts_checked = False  # 是否已尝试创建过 FTS 索引

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        logger.info("  正在初始化向量库...")
        try:
            import lancedb
        except ImportError:
            raise ImportError(
                "lancedb is required for RAG vector storage. pip install lancedb"
            )

        os.makedirs(self._persist_dir, exist_ok=True)

        logger.info("  连接 LanceDB: %s", self._persist_dir)
        self._db = lancedb.connect(self._persist_dir)

        # Use injected embedding function (for testing) or create FastEmbed one
        if self._custom_ef is not None:
            self._ef = self._custom_ef
        else:
            # Use the process-wide shared singleton to avoid loading
            # multiple copies of the ~137MB ONNX model (RAG + graph_memory).
            self._ef = get_or_create_embedding_function(self._embedding_model)

        # Open existing table or defer creation until first add
        if _TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(_TABLE_NAME)
            self._check_schema_migration()

        # Open the documents metadata table (if it already exists).
        if _DOCUMENTS_TABLE_NAME in self._db.table_names():
            self._documents_table = self._db.open_table(_DOCUMENTS_TABLE_NAME)

        self._initialized = True
        count = self._table.count_rows() if self._table is not None else 0
        logger.info("  向量库初始化完成: 当前存储 %d 条向量记录", count)

    def _check_schema_migration(self) -> None:
        """Migrate an existing table to include the ``text_fts`` column.

        The BM25 full-text index lives on a pre-tokenized ``text_fts`` column.
        Tables created before this feature lack the column; since LanceDB
        schemas are immutable, we rebuild the table (data survives — it is
        re-read from the old table and re-written with the new column).
        """
        if self._table is None:
            return
        try:
            schema = self._table.schema
            if _FTS_COLUMN in schema.names:
                return
        except Exception:
            return

        logger.info("  ⚙ 检测到旧版向量表（缺少 %s 列），自动迁移...", _FTS_COLUMN)
        try:
            # Read all existing rows via PyArrow (无 pandas 依赖)。
            arrow_table = self._table.to_arrow()
            records = []
            for row in arrow_table.to_pylist():
                rec = {
                    "id": row["id"],
                    "text": row["text"],
                    "vector": row["vector"],
                    "source": row["source"],
                    "chunk_index": row["chunk_index"],
                }
                rec[_FTS_COLUMN] = _tokenize_for_fts(rec["text"])
                records.append(rec)

            # LanceDB OSS 无 rename_table，只能 drop 后重建。数据已在内存
            # 的 records 中，drop 旧表后 create 失败可再次尝试；即使最终
            # 失败，RAG 数据也可从源文件重新同步恢复（幂等）。
            self._db.drop_table(_TABLE_NAME)
            self._table = None
            if records:
                self._table = self._db.create_table(_TABLE_NAME, records)
            self._table_checked = True
            logger.info("  ⚙ 向量表迁移完成（新增 %s 列，%d 条记录）", _FTS_COLUMN, len(records))
        except Exception as e:
            logger.warning("  ⚠ 向量表迁移失败（BM25 索引将不可用）: %s", e)
            self._table = None
            self._table_checked = False

    def _ensure_table(self) -> Optional[Any]:
        """Open existing table if it exists, return None if not yet created.

        Re-checks ``table_names()`` on every call when ``_table`` is None,
        because the table may have been created by another RAGEngine instance
        (e.g. SyncKnowledgeWorker) after this VectorStore was first initialized.
        Caching "table doesn't exist" was a premature optimization that caused
        a real bug: after sync wrote data via a separate engine, this engine's
        search() still returned empty because it never re-checked.

        LanceDB's ``table_names()`` is cheap (directory listing), so the
        per-call cost is acceptable for correctness.
        """
        if self._table is not None:
            return self._table
        if _TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(_TABLE_NAME)
            self._table_checked = True
        return self._table

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """Add document chunks to the vector store. Returns number of chunks added.

        Upsert 语义：先按 source 删除该来源的全部旧记录（处理文档缩减后
        残留旧 chunks 的情况），再插入新记录。embedding 分批进行以避免 OOM。
        """
        self._ensure_initialized()
        if not chunks:
            return 0

        # Build records (embed in batches to avoid OOM on large knowledge bases)
        records: List[Dict[str, Any]] = []
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            texts = [c["text"] for c in batch_chunks]
            vectors = self._ef(texts)
            for j, chunk in enumerate(batch_chunks):
                idx = i + j
                source = chunk.get("source", "unknown")
                chunk_idx = chunk.get("chunk_index", idx)
                chunk_id = f"{_hash_source(source)}_{chunk_idx}"
                records.append({
                    "id": chunk_id,
                    "text": chunk["text"],
                    "vector": _normalize(vectors[j]),
                    "source": source,
                    "chunk_index": chunk_idx,
                    _FTS_COLUMN: _tokenize_for_fts(chunk["text"]),
                })

        table = self._ensure_table()

        if table is None:
            # First add: create table with all records
            self._table = self._db.create_table(_TABLE_NAME, records)
            self._table_checked = True
            logger.info("LanceDB table '%s' created with %d records", _TABLE_NAME, len(records))
            return len(records)

        # Upsert: 按 source 删除该来源的全部旧记录。
        # 用 source 的 md5 hash 作为 LIKE 模式（hash 是 hex 字符串，无 SQL 注入风险），
        # 既能覆盖 chunk_index 0..N 的所有旧记录，也能清理文档缩减后多出的旧 chunks。
        sources_to_replace = sorted({c.get("source", "unknown") for c in chunks})
        for source in sources_to_replace:
            h = _hash_source(source)
            try:
                # id 格式为 "{hash}_{chunk_index}"，LIKE '{hash}_%' 匹配该 source 的所有 chunk
                table.delete(f"id LIKE '{h}_%'")
            except Exception as e:
                logger.debug("VectorStore: delete by source '%s' failed: %s", source, e)

        # Add records in batches（单批失败不中断，最大限度保留数据）
        total = 0
        failed_batches = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            try:
                table.add(batch)
                total += len(batch)
            except Exception as e:
                failed_batches += 1
                logger.error(
                    "VectorStore: add batch %d/%d failed: %s",
                    i // batch_size + 1, (len(records) + batch_size - 1) // batch_size, e,
                )
                if failed_batches >= 3:
                    # 连续失败太多，可能系统性问题，中断避免持续报错
                    raise
        if failed_batches > 0:
            logger.warning("VectorStore: %d batch(es) failed during add", failed_batches)

        # 数据写入后重建 BM25 全文索引，覆盖新增的行。
        self._fts_ready = False
        self._ensure_fts_index(self._table)

        logger.info("VectorStore: added %d chunks", total)
        return total

    def delete_by_source(self, source: str) -> int:
        """删除指定 source 的全部记录。返回删除前的记录数（若可查）。

        用于 force 重新 ingest 时清理已从知识库中删除的文件对应的旧向量。
        """
        self._ensure_initialized()
        table = self._ensure_table()
        if table is None:
            return 0
        h = _hash_source(source)
        try:
            table.delete(f"id LIKE '{h}_%'")
            logger.info("VectorStore: deleted records for source '%s'", source)
            return 1
        except Exception as e:
            logger.warning("VectorStore: delete_by_source '%s' failed: %s", source, e)
            return 0

    def add_streaming(
        self,
        chunk_iter,
        batch_size: int = 100,
        on_batch: Optional[Any] = None,
        is_cancelled: Optional[Any] = None,
    ) -> int:
        """流式增量写入：接受 chunk 迭代器，每批 embed 完立即写入并释放。

        用于大知识库 ingest，避免 H3 隐患（全部 records 含 vector 同时在内存）。
        内存占用 = O(batch_size) 而非 O(总 chunks 数)。

        Args:
            chunk_iter: 产出 chunk dict 的迭代器（必须有 source / chunk_index / text）
            batch_size: 每批 embed + 写入的 chunk 数
            on_batch: 可选回调 (batch_added, total_added) -> None
            is_cancelled: 可选回调 () -> bool，返回 True 时中断写入并返回当前计数

        Returns:
            成功写入的 chunk 总数（取消时返回已写入数）
        """
        self._ensure_initialized()

        # 第一批特殊处理：用于判断是否需要 create_table
        first_batch: List[Dict[str, Any]] = []
        exhausted = False
        for _ in range(batch_size):
            try:
                first_batch.append(next(chunk_iter))
            except StopIteration:
                exhausted = True
                break

        if not first_batch:
            logger.info("  向量化写入: 无切片需要写入（空迭代器）")
            return 0

        # 第一批处理前检查取消标志
        if is_cancelled is not None and is_cancelled():
            return 0

        # Embed 第一批
        logger.info("  ├─ 向量化: 首批 %d 个切片正在嵌入...", len(first_batch))
        first_records = self._embed_chunks(first_batch, 0)
        logger.info("  ├─ 向量化完成: 首批 %d 个切片", len(first_records))

        table = self._ensure_table()
        # 跟踪已删除过旧数据的 source，避免后续批次重复写入时新旧数据共存。
        sources_seen: set = set()
        if table is None:
            # 首次：创建表（用第一批 records），无需删除旧数据。
            self._table = self._db.create_table(_TABLE_NAME, first_records)
            self._table_checked = True
            table = self._table  # 更新局部变量，后续批次需要 table 引用
            logger.info("  ├─ 向量库: 新建 LanceDB 表 '%s', 写入 %d 条记录", _TABLE_NAME, len(first_records))
            total = len(first_records)
            sources_seen = {r["source"] for r in first_records}
            if on_batch:
                on_batch(len(first_records), total)
            if exhausted:
                logger.info("  └─ 向量化写入完成: 共 %d 个切片 (仅一批)", total)
                return total
        else:
            # 已有表：先按 source 删除旧记录
            existing_count = table.count_rows()
            sources_seen = {r["source"] for r in first_records}
            logger.info(
                "  ├─ 向量库: 已有表 (%d 行), 将替换 %d 个来源的旧数据",
                existing_count, len(sources_seen),
            )
            for source in sources_seen:
                h = _hash_source(source)
                try:
                    table.delete(f"id LIKE '{h}_%'")
                except Exception as e:
                    logger.debug("VectorStore: delete by source '%s' failed: %s", source, e)
            # 写入第一批
            try:
                table.add(first_records)
                total = len(first_records)
                logger.info("  ├─ 向量库写入: 首批 %d 条记录", total)
                if on_batch:
                    on_batch(len(first_records), total)
            except Exception as e:
                logger.error("  向量库写入失败(首批): %s", e)
                raise
            if exhausted:
                logger.info("  └─ 向量化写入完成: 共 %d 个切片 (仅一批)", total)
                return total

        # 后续批次
        chunk_idx = len(first_batch)
        batch_buf: List[Dict[str, Any]] = []
        batch_num = 1
        deleted_sources: set = sources_seen.copy() if table is not None else set()
        for chunk in chunk_iter:
            batch_buf.append(chunk)
            if len(batch_buf) >= batch_size:
                if is_cancelled is not None and is_cancelled():
                    return total
                batch_num += 1
                logger.info("  ├─ 向量化: 第 %d 批 (%d 切片) 正在嵌入...", batch_num, len(batch_buf))
                if table is not None:
                    new_sources = {c["source"] for c in batch_buf} - deleted_sources
                    if new_sources:
                        for source in new_sources:
                            h = _hash_source(source)
                            try:
                                table.delete(f"id LIKE '{h}_%'")
                            except Exception:
                                pass
                        deleted_sources.update(new_sources)
                records = self._embed_chunks(batch_buf, chunk_idx)
                try:
                    table.add(records)
                    total += len(records)
                    logger.info("  ├─ 向量库写入: 第 %d 批 %d 条 (累计 %d)", batch_num, len(records), total)
                    if on_batch:
                        on_batch(len(records), total)
                except Exception as e:
                    logger.error("  向量库写入失败(第 %d 批): %s", batch_num, e)
                    raise
                chunk_idx += len(batch_buf)
                batch_buf.clear()
        # 处理剩余
        if batch_buf:
            if is_cancelled is not None and is_cancelled():
                return total
            batch_num += 1
            logger.info("  ├─ 向量化: 最后批次 (%d 切片) 正在嵌入...", len(batch_buf))
            if table is not None:
                new_sources = {c["source"] for c in batch_buf} - deleted_sources
                if new_sources:
                    for source in new_sources:
                        h = _hash_source(source)
                        try:
                            table.delete(f"id LIKE '{h}_%'")
                        except Exception:
                            pass
            records = self._embed_chunks(batch_buf, chunk_idx)
            try:
                table.add(records)
                total += len(records)
                logger.info("  ├─ 向量库写入: 最后批次 %d 条 (累计 %d)", len(records), total)
                if on_batch:
                    on_batch(len(records), total)
            except Exception as e:
                logger.error("  向量库写入失败(最后批次): %s", e)
                raise

        logger.info("  └─ 向量化写入完成: 共 %d 个切片 (%d 批次)", total, batch_num)

        # 数据写入后重建 BM25 全文索引，覆盖新增的行。
        # 注：LanceDB 的 FTS 索引不会在 add() 后自动增量更新，
        # 必须用 replace=True 重建才能检索到新写入的切片。
        self._fts_ready = False
        self._ensure_fts_index(self._table)

        return total

    def _embed_chunks(
        self, chunks: List[Dict[str, Any]], start_idx: int,
    ) -> List[Dict[str, Any]]:
        """Embed 一批 chunks，返回 records（含 vector）。内部辅助方法。

        将每批拆分为更小的子批次（每子批 5 个 chunk）进行 embedding，
        每个子批次完成后主动 time.sleep() 释放 GIL，让主线程有机会
        处理 Qt 事件（日志渲染、进度更新）。ONNX Runtime 的 embed()
        在 C++ 计算期间持有 GIL，整批一次性嵌入会导致 UI 长时间冻结。
        """
        import time as _time
        _SUB_BATCH = 5  # 每子批 5 个 chunk，平衡吞吐与 UI 响应
        texts = [c["text"] for c in chunks]
        all_vectors: List[List[float]] = []
        for sub_start in range(0, len(texts), _SUB_BATCH):
            sub_texts = texts[sub_start:sub_start + _SUB_BATCH]
            sub_vectors = self._ef(sub_texts)
            all_vectors.extend(sub_vectors)
            # 每个子批次完成后释放 GIL，让 Qt 事件循环有机会处理
            # 排队的日志/进度信号。这是关键：ONNX Runtime 在 embed()
            # 期间持有 GIL，只有在 Python 层主动 sleep 时才释放。
            _time.sleep(0.01)
        vectors = all_vectors
        records: List[Dict[str, Any]] = []
        for j, chunk in enumerate(chunks):
            idx = start_idx + j
            source = chunk.get("source", "unknown")
            chunk_idx = chunk.get("chunk_index", idx)
            chunk_id = f"{_hash_source(source)}_{chunk_idx}"
            records.append({
                "id": chunk_id,
                "text": chunk["text"],
                "vector": _normalize(vectors[j]),
                "source": source,
                "chunk_index": chunk_idx,
                _FTS_COLUMN: _tokenize_for_fts(chunk["text"]),
            })
        return records

    # ------------------------------------------------------------------
    # BM25 full-text search (hybrid retrieval)
    # ------------------------------------------------------------------

    def _ensure_fts_index(self, table) -> bool:
        """Ensure a BM25 full-text index exists on the ``text_fts`` column.

        Creates the index on first use and after each ingest (the index must
        be refreshed to cover newly written rows).  The ``text_fts`` column is
        pre-tokenized with jieba (space-joined).  We use LanceDB's *native*
        FTS (Tantivy support was removed in lancedb 0.37); the native ``simple``
        tokenizer splits on whitespace, so each space-separated jieba token is
        indexed as its own term — sidestepping the lack of a compiled CJK
        tokenizer in the native FTS backend.

        Returns True if the index is available for querying.
        """
        if self._fts_ready:
            return True

        # 原生 FTS 配置（两种 lancedb 版本共用同一语义）：
        #  - simple base tokenizer 按空格/标点切分，配合 jieba 预分词列，
        #    每个空格分隔的 jieba token 即为一个独立 term；
        #  - ascii_folding=False 保留 CJK 码点；
        #  - stem=False / remove_stop_words=False 避免破坏中文词形。
        fts_kwargs = dict(
            base_tokenizer="simple",
            lower_case=True,
            stem=False,
            remove_stop_words=False,
            ascii_folding=False,
            with_position=True,
        )

        # lancedb >= 0.37：create_index(config=FTS(...))（Tantivy 已移除）。
        # lancedb 0.25.x：create_fts_index(use_tantivy=False, ...)。
        try:
            from lancedb.index import FTS
            table.create_index(
                _FTS_COLUMN,
                config=FTS(**fts_kwargs),
                replace=True,
            )
        except TypeError:
            # 0.25.x 的 create_index 无 config 参数，回退到 create_fts_index。
            table.create_fts_index(
                _FTS_COLUMN,
                use_tantivy=False,
                replace=True,
                **fts_kwargs,
            )
        except Exception as e:
            logger.warning("  ⚠ BM25 全文索引创建失败 (回退纯向量检索): %s", e)
            self._fts_checked = True
            self._fts_ready = False
            return False

        self._fts_ready = True
        self._fts_checked = True
        logger.info("  ├─ BM25 全文索引已创建 (原生 FTS simple tokenizer + jieba 预分词)")
        return True

    def search_fts(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Keyword (BM25) full-text search over the ``text_fts`` column.

        Returns list of {text, source, chunk_index, score, fts_score} dicts,
        where ``score`` is the BM25 score (higher is better).  Used together
        with vector search for hybrid retrieval.

        ``where`` is an optional SQL filter (e.g. ``source LIKE '%book.md'``)
        applied to the FTS search as well, so targeted retrieval restricts both
        channels consistently.
        """
        self._ensure_initialized()
        table = self._ensure_table()
        if table is None:
            return []

        if not self._ensure_fts_index(table):
            return []

        try:
            # Tokenize the query the same way as the indexed text.
            tokenized_query = _tokenize_for_fts(query)
            q = (
                table.search(tokenized_query, query_type="fts", fts_columns=[_FTS_COLUMN])
            )
            if where:
                q = q.where(where)
            results = q.limit(top_k).to_list()
        except Exception as e:
            logger.warning("  ⚠ BM25 全文检索失败: %s", e)
            return []

        output: List[Dict[str, Any]] = []
        for r in results:
            # Tantivy returns BM25 score in `_score` (higher is better).
            raw = float(r.get("_score", 0.0))
            output.append({
                "text": r.get("text", ""),
                "source": r.get("source", ""),
                "chunk_index": r.get("chunk_index", 0),
                "score": raw,
                "fts_score": raw,
            })
        return output

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
        table = self._ensure_table()
        if table is None:
            logger.info("  向量检索: 向量表不存在 (查询=%r)", query[:60])
            return []
        row_count = table.count_rows()
        if row_count == 0:
            logger.info("  向量检索: 向量表为空 (查询=%r)", query[:60])
            return []

        # Embed query (use embed_query for proper prefix handling)
        logger.info("  ├─ 查询向量化: 将查询文本转为向量...")
        if hasattr(self._ef, 'embed_query'):
            query_vec = _normalize(self._ef.embed_query(query))
        else:
            query_vec = _normalize(self._ef([query])[0])

        # Vector search
        logger.info("  ├─ 向量相似度搜索: 在 %d 条向量中检索 top %d...", row_count, top_k)
        search = table.search(query_vec).limit(top_k)
        if where:
            search = search.where(where)
        results = search.to_list()
        logger.info("  ├─ 向量检索完成: 获得 %d 个候选切片", len(results))

        output: List[Dict[str, Any]] = []
        for r in results:
            # LanceDB 返回 _distance（L2 距离，越小越相似）。
            # 转换为相似度分数 [0,1]（越高越相似），统一下游接口语义。
            if "_distance" in r:
                distance = float(r["_distance"])
                # 对归一化向量，L2 距离 d ∈ [0,2]，余弦相似度 cos = 1 - d²/2
                similarity = max(0.0, 1.0 - (distance * distance) / 2.0)
            else:
                # _distance 字段缺失（罕见：表 schema 不兼容或旧表迁移）。
                # 使用中等分数避免将无效结果排到最前或最后。
                logger.debug("VectorStore.search: _distance field missing for result, using default score 0.5")
                similarity = 0.5
            output.append({
                "text": r.get("text", ""),
                "source": r.get("source", ""),
                "chunk_index": r.get("chunk_index", 0),
                "score": similarity,
            })

        return output

    # ------------------------------------------------------------------
    # reranking with BGE
    # ------------------------------------------------------------------

    def _get_reranker(self):
        """Lazy-load the BGE reranker model with timeout protection.

        Returns the reranker instance, or None if loading fails (network
        timeout, missing dependency, etc.). Callers must handle None by
        falling back to distance-based ranking.

        Uses ``_reranker_state`` to track loading state so concurrent callers
        don't each spawn their own load thread (which would leak ~500MB of
        PyTorch weights per extra thread).  On timeout, the background thread
        is NOT abandoned — when it eventually finishes it stores the result
        (or error) on ``self`` so a later call can pick it up without reloading.
        """
        state = getattr(self, "_reranker_state", "idle")
        if state == "done":
            return self._reranker
        if state == "failed":
            return None
        if state == "loading":
            # Another caller already started the load. Don't spawn a duplicate;
            # return None so this call falls back to distance ranking, and the
            # result will be available on a later call.
            return None

        # torch/numpy 兼容性探测：FlagEmbedding 依赖 torch，而某些旧版 torch
        # （如 macOS Intel 上最高可装的 2.2.2）与 numpy 2.x 不兼容，调用
        # ``tensor.numpy()`` 会抛 ``Numpy is not available``，导致重排序崩溃。
        # 这里不凭 numpy 版本号猜测，而是真正执行一次 tensor→numpy 转换来
        # 验证 torch 与 numpy 是否兼容，避免误伤 numpy 2.x + 新版 torch 的
        # 正常组合（如 arm64 上的 torch 2.13 + numpy 2.5 是完全兼容的）。
        try:
            import torch as _torch
            try:
                _torch.tensor([1.0, 2.0]).numpy()
            except Exception as _e:
                import numpy as _np
                logger.warning(
                    "  ⚠ BGE Reranker 不可用: torch %s 与 numpy %s 不兼容 "
                    "（tensor.numpy() 失败: %s）。重排序回退到向量距离排序。"
                    "如需启用，请升级 torch 或降级 numpy 到 1.26.x。",
                    getattr(_torch, "__version__", "?"),
                    getattr(_np, "__version__", "?"),
                    _e,
                )
                self._reranker = None
                self._reranker_state = "failed"
                return None
        except ImportError:
            pass  # torch 缺失时交由后续 import FlagEmbedding 报错处理

        try:
            from FlagEmbedding import FlagReranker
        except ImportError:
            logger.warning(
                "  ⚠ FlagEmbedding 未安装, 重排序功能不可用. pip install FlagEmbedding"
            )
            self._reranker = None
            self._reranker_state = "failed"
            return None

        logger.info("  🔧 正在加载 BGE Reranker 模型: %s (首次加载, 约 500MB)...", self._rerank_model)
        import time as _time
        _time.sleep(0.01)  # 释放 GIL 让 UI 更新

        # Mark loading before spawning so concurrent callers see it.
        self._reranker_state = "loading"

        def _load():
            try:
                self._reranker = FlagReranker(
                    self._rerank_model,
                    use_fp16=False,
                )
                self._reranker_state = "done"
                logger.info("  ✅ BGE Reranker 加载完成: %s", self._rerank_model)
            except Exception as e:
                logger.warning("  ⚠ BGE Reranker 加载失败: %s", e)
                self._reranker = None
                self._reranker_state = "failed"

        try:
            import threading
            t = threading.Thread(target=_load, daemon=True)
            t.start()
            t.join(timeout=120)  # 2 分钟超时，足够下载 500MB

            if t.is_alive():
                # 加载仍在后台进行（网络慢）。不丢弃结果——后台线程完成时会
                # 写入 self._reranker/_reranker_state，后续调用直接复用。
                logger.warning(
                    "  ⚠ BGE Reranker 加载超时 (2 分钟) — 仍在后台下载中。"
                    " 本次重排序回退到向量距离排序；加载完成后自动启用。"
                )
        except Exception as e:
            logger.warning("  ⚠ BGE Reranker 加载失败: %s", e)
            self._reranker = None
            self._reranker_state = "failed"

        return self._reranker

    def _rerank_fallback(
        self,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Fall back to vector-distance ranking when BGE reranker is unavailable.

        candidates 的 score 已由 search() 转换为相似度 [0,1]（越高越好）。
        回退模式下分数是余弦距离的近似（非 BGE sigmoid 语义分），因此不做
        0.7 阈值过滤——阈值仅对 reranker 的语义分数有意义。
        """
        logger.info("  ├─ 重排序: BGE Reranker 不可用, 使用向量距离排序 (回退模式)")
        for c in candidates:
            try:
                s = float(c.get("score", 0.0))
            except (TypeError, ValueError):
                c["score"] = 0.0
                continue
            if s > 1.0:
                c["score"] = max(0.0, 1.0 - (s * s) / 2.0)
            else:
                c["score"] = s
        candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        result = candidates[:top_k]
        logger.info("  ├─ 重排序完成(回退): %d → %d 条结果", len(candidates), len(result))
        return result

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 3,
        min_similarity: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Re-rank candidates using BGE reranker for precise scoring.

        Falls back to distance-based similarity scoring when the reranker
        is unavailable. In both cases scores are normalized to [0, 1]
        (higher is better) and results are returned sorted descending.

        Results whose final score is below ``min_similarity`` are dropped
        (they are not semantically relevant enough to send to the LLM), so
        the returned list may be shorter than ``top_k``.
        """
        if not candidates:
            return []

        # 使用注入的自定义 embedding（如测试的 FakeEmbeddingFunction）时，
        # 向量空间与 BGE reranker 的语义空间不一致，rerank 打分无意义，
        # 直接回退到向量距离排序。
        if self._custom_ef is not None:
            return self._rerank_fallback(candidates, top_k)

        reranker = self._get_reranker()
        if reranker is None:
            return self._rerank_fallback(candidates, top_k)

        # Build pairs for reranker
        logger.info("  ├─ BGE Reranker 精排: 对 %d 个候选进行交叉编码打分...", len(candidates))
        pairs = [[query, c["text"]] for c in candidates]
        try:
            # normalize=True applies sigmoid so scores land in [0, 1].
            scores = reranker.compute_score(pairs, normalize=True)
        except Exception as e:
            # FlagEmbedding 与 transformers 5.x 存在 API 不兼容（如
            # XLMRobertaTokenizer 缺少 prepare_for_model），打分失败时回退到
            # 向量距离排序，避免整个检索流程崩溃。
            logger.warning(
                "  ⚠ BGE Reranker 打分失败, 回退到向量距离排序: %s", e,
            )
            self._reranker = None  # 标记不可用，后续直接走回退分支
            return self._rerank_fallback(candidates, top_k)

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

        # 输出重排序前后的分数变化
        if results:
            logger.info("  ├─ BGE 重排序完成: %d → %d 条结果", len(candidates), len(results))
            for i, r in enumerate(results[:3]):
                source = os.path.basename(r.get("source", ""))
                logger.info("  │   [%d] %s (重排序分: %.4f)", i + 1, source, r["score"])

        return self._filter_by_similarity(results, min_similarity)

    def _filter_by_similarity(
        self,
        results: List[Dict[str, Any]],
        min_similarity: float,
    ) -> List[Dict[str, Any]]:
        """Drop results whose final score is below ``min_similarity``.

        A rerank score (BGE sigmoid, or cosine-similarity fallback) below the
        threshold means the chunk is not semantically close enough to the
        query to be worth sending to the LLM.
        """
        if not results or min_similarity <= 0:
            return results
        kept = [r for r in results if r.get("score", 0.0) >= min_similarity]
        if len(kept) < len(results):
            dropped = len(results) - len(kept)
            logger.info(
                "  ├─ 相似度过滤: 丢弃 %d 条 < %.2f 的结果, 保留 %d 条",
                dropped, min_similarity, len(kept),
            )
        return kept

    def search_with_rerank(
        self,
        query: str,
        top_k: int = 3,
        candidate_multiplier: int = 4,
        where: Optional[str] = None,
        min_similarity: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Hybrid retrieval: vector similarity + BM25 keyword, then BGE rerank.

        Pipeline:
          1. Vector search: top_k * candidate_multiplier candidates.
          2. BM25 keyword search: top_k * candidate_multiplier candidates.
          3. Merge & dedupe by (source, chunk_index).
          4. BGE rerank the merged candidate set.
          5. Drop any result whose rerank score < min_similarity (0.7).
          6. Return up to top_k results.

        Args:
            query: Search query.
            top_k: Final number of results to return.
            candidate_multiplier: Candidates per retrieval channel = top_k × N.
            where: Optional SQL filter (applied to vector search only).
            min_similarity: Reject rerank results below this score.
        """
        n_candidates = top_k * candidate_multiplier

        # 1. Vector similarity search.
        vec_candidates = self.search(query, top_k=n_candidates, where=where)
        logger.info(
            "VectorStore.search_with_rerank: 向量检索 %d 条候选", len(vec_candidates),
        )

        # 2. BM25 keyword search (同样的 where 过滤，保证定向检索一致)。
        fts_candidates = self.search_fts(query, top_k=n_candidates, where=where)
        logger.info(
            "VectorStore.search_with_rerank: BM25 检索 %d 条候选", len(fts_candidates),
        )

        # 3. Merge & dedupe by (source, chunk_index), keeping vector candidate
        #    first (its score is a cosine similarity in [0,1]).
        merged: List[Dict[str, Any]] = []
        seen: set = set()
        for c in vec_candidates + fts_candidates:
            key = (c.get("source", ""), c.get("chunk_index", 0))
            if key in seen:
                continue
            seen.add(key)
            merged.append(c)

        if not merged:
            logger.info("VectorStore.search_with_rerank: 混合检索无候选")
            return []

        logger.info(
            "VectorStore.search_with_rerank: 混合候选 %d 条 (向量 %d + BM25 %d, 去重后 %d)",
            len(merged), len(vec_candidates), len(fts_candidates), len(merged),
        )

        # 4. BGE rerank + 5. min_similarity filter.
        results = self.rerank(query, merged, top_k=top_k, min_similarity=min_similarity)
        logger.info(
            "VectorStore.search_with_rerank: %d 候选 → %d 条 (rerank + 相似度≥%.2f 过滤)",
            len(merged), len(results), min_similarity,
        )

        # 6. 当 rerank 严格过滤后无结果（例如查询词质量差——拼音、口语化、
        #    噪声词等导致 rerank 语义分普遍低于阈值）时，回退到向量距离排序，
        #    至少返回 top_k 条候选，避免 LLM 完全拿不到上下文。
        if not results and merged:
            logger.info(
                "VectorStore.search_with_rerank: rerank 过滤后无结果, 回退到向量距离排序",
            )
            results = self._rerank_fallback(merged, top_k)
        return results

    def clear(self) -> None:
        """Delete all documents from the store."""
        self._ensure_initialized()
        table = self._ensure_table()
        if table is None:
            return
        count = table.count_rows()
        if count > 0:
            # Drop and recreate empty table
            self._db.drop_table(_TABLE_NAME)
            self._table = None
            self._table_checked = False  # 重置标记，允许下次 add 重新创建表
            logger.info("VectorStore: cleared %d documents", count)
        # 同步清空文档元数据表。
        if _DOCUMENTS_TABLE_NAME in self._db.table_names():
            self._db.drop_table(_DOCUMENTS_TABLE_NAME)
            self._documents_table = None
            self._documents_cache = None
            logger.info("VectorStore: cleared documents metadata table")

    def count(self) -> int:
        self._ensure_initialized()
        table = self._ensure_table()
        if table is None:
            return 0
        return table.count_rows()

    def close(self) -> None:
        """Release heavy resources (LanceDB connection, embedding model, reranker).

        After close(), the VectorStore cannot be used; create a new instance to
        access the same data (LanceDB persists to disk, so data survives).

        Important: FastEmbed's ONNX Runtime InferenceSession and FlagReranker
        hold significant C++ heap memory (~130MB + ~500MB). Python's GC may not
        reclaim these promptly when the Python object is dropped — explicitly
        nulling the references here ensures the C++ destructors run as soon
        as possible, which is critical when SyncKnowledgeWorker creates a fresh
        RAGEngine per sync run.
        """
        # Try to explicitly release the ONNX InferenceSession held by
        # FastEmbed's OnnxEmbeddings. FastEmbed wraps the model in a
        # `model` attribute (ort.InferenceSession). Calling .release() on
        # the session frees the C++ memory immediately instead of waiting
        # for Python's cyclic GC to eventually notice the dead reference.
        # This is best-effort: if the attribute layout changes in a future
        # FastEmbed version, the AttributeError is swallowed and we fall
        # back to simply dropping the Python reference.
        self._table = None
        self._db = None
        self._documents_cache = None
        if self._ef is not None:
            # Only release the ONNX session if this VectorStore owns a
            # private (non-shared) embedding function.  When using the
            # process-wide singleton (get_or_create_embedding_function),
            # the session may still be in use by graph_memory or another
            # VectorStore — releasing it would crash those callers.
            is_shared = self._ef in _EF_SINGLETONS.values()
            if not is_shared:
                try:
                    inner = getattr(self._ef, "model", None) or getattr(self._ef, "_model", None)
                    if inner is not None:
                        session = getattr(inner, "session", None) or getattr(inner, "_session", None)
                        if session is not None and hasattr(session, "release"):
                            try:
                                session.release()
                            except Exception:
                                pass
                except Exception:
                    pass
            self._ef = None
        if hasattr(self, "_reranker"):
            # FlagReranker holds a transformers model with PyTorch tensors.
            # Try to move it to CPU and del before nulling, so CUDA/PyTorch
            # resources are freed promptly.
            reranker = self._reranker
            try:
                model = getattr(reranker, "model", None)
                if model is not None and hasattr(model, "cpu"):
                    model.cpu()
            except Exception:
                pass
            self._reranker = None
        self._reranker_state = "idle"
        self._initialized = False
        self._table_checked = False
        logger.info("VectorStore: resources released")

    def reload(self) -> None:
        """Drop the cached table handle so the next operation re-opens it.

        LanceDB table objects hold a version snapshot at open time. When another
        VectorStore instance (e.g. SyncKnowledgeWorker's) writes new rows to the
        same table, this instance's cached table object won't see them —
        count_rows() and search() still return the old snapshot's data.

        Call this after a sync completes to ensure the agent's VectorStore sees
        the freshly written data on its next search.
        """
        self._table = None
        self._table_checked = False
        logger.debug("VectorStore: table handle dropped for reload")

    def list_sources(self) -> List[str]:
        """Return unique source filenames in the store."""
        self._ensure_initialized()
        table = self._ensure_table()
        if table is None or table.count_rows() == 0:
            return []
        # 用 pyarrow compute 的 unique() 在 Arrow 层去重，
        # 避免 to_pylist() 把全部行转换成 Python 对象（10 万行 = 10 万字符串）
        try:
            import pyarrow.compute as pc
            arrow_tbl = table.to_arrow(columns=["source"])
            unique_sources = pc.unique(arrow_tbl["source"]).to_pylist()
            return sorted([s for s in unique_sources if s])
        except Exception:
            # Fallback: load source column and dedupe in Python
            try:
                all_data = table.to_arrow(columns=["source"]).to_pylist()
            except Exception:
                all_data = table.to_arrow().to_pylist()
            sources = sorted({r.get("source", "") for r in all_data if r and r.get("source")})
            return sources

    # ------------------------------------------------------------------
    # documents metadata table (Meta-context + Targeted RAG)
    # ------------------------------------------------------------------

    def _ensure_documents_table(self):
        """Open or create the documents metadata table (thread-safe).

        用双重检查锁保护：ingest 的多 worker 并行调用 upsert_document 时，
        首次创建表会并发竞争——若不加锁，第二个 worker 会在 create_table 时报
        "Table already exists"。锁内再次检查后，创建或打开由唯一一个线程完成。
        """
        if self._documents_table is not None:
            return self._documents_table
        with self._documents_lock:
            if self._documents_table is not None:
                return self._documents_table
            self._ensure_initialized()
            if _DOCUMENTS_TABLE_NAME in self._db.table_names():
                self._documents_table = self._db.open_table(_DOCUMENTS_TABLE_NAME)
            else:
                # 首次创建：空表（含占位空记录，后续 upsert 时清理）。
                self._documents_table = self._db.create_table(
                    _DOCUMENTS_TABLE_NAME,
                    data=[{
                        "doc_id": "",
                        "doc_name": "",
                        "digest": "",
                        "markdown": "",
                        "chapters": "",
                    }],
                )
            return self._documents_table

    def upsert_document(
        self,
        doc_name: str,
        digest: str,
        markdown: str,
        chapters: str = "",
    ) -> None:
        """Store or update a document's digest (缩略版本) + full markdown.

        ``doc_id`` is a stable md5 of ``doc_name`` so re-ingesting the same
        document overwrites the previous entry (upsert semantics).

        The delete+add pair is serialized under ``_documents_lock`` because
        ingest workers call this concurrently (one per file). LanceDB table
        writes are not thread-safe, and interleaving delete/add from multiple
        threads could corrupt rows or raise "Table already exists"-style races.
        """
        self._ensure_initialized()
        with self._documents_lock:
            table = self._ensure_documents_table()
            doc_id = _hash_source(doc_name)

            # Delete any previous entry (and the placeholder row), then add the new one.
            try:
                table.delete(f"doc_id = '{doc_id}'")
                table.delete("doc_id = ''")
            except Exception:
                pass
            table.add([{
                "doc_id": doc_id,
                "doc_name": doc_name,
                "digest": digest,
                "markdown": markdown,
                "chapters": chapters,
            }])
            logger.info("  ├─ 文档元数据已写入: %s (digest %d 字符)", doc_name, len(digest))
            # 写入后失效缓存，下次读取时重新加载。
            self._documents_cache = None

    def _load_documents(self) -> List[Dict[str, Any]]:
        """Load (and cache) all non-placeholder document rows.

        Returns a list of dicts, filtered to drop the placeholder row
        (``doc_name`` empty).  Cached in ``_documents_cache``; callers should
        hold ``_documents_lock`` when the cache could be invalidated
        concurrently (or accept a benign re-read on a cache miss).
        """
        if self._documents_cache is not None:
            return self._documents_cache
        table = self._ensure_documents_table()
        if table is None:
            return []
        try:
            rows = table.to_arrow().to_pylist()
        except Exception as e:
            logger.warning("读取文档元数据失败: %s", e)
            return []
        rows = [r for r in rows if r.get("doc_name")]
        self._documents_cache = rows
        return rows

    def get_document_digest(self, doc_name: str) -> Optional[Dict[str, Any]]:
        """Return {doc_name, digest, markdown, chapters} for a document by name.

        Matches by exact basename or by substring (so the model can reference
        a book by a partial title).  Returns None if not found.
        """
        self._ensure_initialized()
        rows = self._load_documents()
        if not rows:
            return None

        # 1. Exact match (case-insensitive) on doc_name or basename.
        target = doc_name.strip().lower()
        for r in rows:
            name = (r.get("doc_name") or "").lower()
            if name == target or os.path.basename(name) == target:
                return {
                    "doc_name": r.get("doc_name", ""),
                    "digest": r.get("digest", ""),
                    "markdown": r.get("markdown", ""),
                    "chapters": r.get("chapters", ""),
                }
        # 2. Substring match (partial title).
        for r in rows:
            name = (r.get("doc_name") or "").lower()
            if target and (target in name or name in target):
                return {
                    "doc_name": r.get("doc_name", ""),
                    "digest": r.get("digest", ""),
                    "markdown": r.get("markdown", ""),
                    "chapters": r.get("chapters", ""),
                }
        return None

    def list_documents(self) -> List[str]:
        """Return the list of stored document names (for the outline tool)."""
        self._ensure_initialized()
        rows = self._load_documents()
        return sorted({r.get("doc_name", "") for r in rows if r.get("doc_name")})

    def delete_document(self, doc_name: str) -> None:
        """Remove a document's metadata entry."""
        self._ensure_initialized()
        with self._documents_lock:
            table = self._ensure_documents_table()
            if table is None:
                return
            doc_id = _hash_source(doc_name)
            try:
                table.delete(f"doc_id = '{doc_id}'")
            except Exception as e:
                logger.warning("删除文档元数据失败: %s", e)
            self._documents_cache = None


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

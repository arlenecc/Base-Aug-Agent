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
        self._table_checked = False  # 是否已检查过 table_names（避免重复磁盘扫描）

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
    ) -> int:
        """流式增量写入：接受 chunk 迭代器，每批 embed 完立即写入并释放。

        用于大知识库 ingest，避免 H3 隐患（全部 records 含 vector 同时在内存）。
        内存占用 = O(batch_size) 而非 O(总 chunks 数)。

        Args:
            chunk_iter: 产出 chunk dict 的迭代器（必须有 source / chunk_index / text）
            batch_size: 每批 embed + 写入的 chunk 数
            on_batch: 可选回调 (batch_added, total_added) -> None

        Returns:
            成功写入的 chunk 总数
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
            logger.info("VectorStore.add_streaming: no chunks to write (empty iterator)")
            return 0

        # Embed 第一批
        first_records = self._embed_chunks(first_batch, 0)

        table = self._ensure_table()
        if table is None:
            # 首次：创建表（用第一批 records）
            self._table = self._db.create_table(_TABLE_NAME, first_records)
            self._table_checked = True
            logger.info("LanceDB table '%s' created (streaming) with %d records",
                        _TABLE_NAME, len(first_records))
            total = len(first_records)
            if on_batch:
                on_batch(len(first_records), total)
            if exhausted:
                return total
        else:
            # 已有表：先按 source 删除旧记录
            existing_count = table.count_rows()
            sources_seen = {r["source"] for r in first_records}
            logger.info(
                "VectorStore.add_streaming: opened existing table (%d rows), "
                "replacing %d sources",
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
                if on_batch:
                    on_batch(len(first_records), total)
            except Exception as e:
                logger.error("VectorStore: streaming add first batch failed: %s", e)
                raise
            if exhausted:
                return total

        # 后续批次：每批 embed 完立即写入，records 用完即释放
        chunk_idx = len(first_batch)
        batch_buf: List[Dict[str, Any]] = []
        batch_num = 1
        for chunk in chunk_iter:
            batch_buf.append(chunk)
            if len(batch_buf) >= batch_size:
                records = self._embed_chunks(batch_buf, chunk_idx)
                try:
                    table.add(records)
                    total += len(records)
                    batch_num += 1
                    if batch_num % 5 == 0:  # 每 5 批打一次，避免日志爆炸
                        logger.debug(
                            "VectorStore.add_streaming: batch %d written, %d chunks total",
                            batch_num, total,
                        )
                    if on_batch:
                        on_batch(len(records), total)
                except Exception as e:
                    logger.error("VectorStore: streaming add batch at idx %d failed: %s", chunk_idx, e)
                    raise
                chunk_idx += len(batch_buf)
                batch_buf.clear()
        # 处理剩余
        if batch_buf:
            records = self._embed_chunks(batch_buf, chunk_idx)
            try:
                table.add(records)
                total += len(records)
                batch_num += 1
                if on_batch:
                    on_batch(len(records), total)
            except Exception as e:
                logger.error("VectorStore: streaming add final batch failed: %s", e)
                raise

        logger.info("VectorStore: streaming added %d chunks total (%d batches)", total, batch_num)
        return total

    def _embed_chunks(
        self, chunks: List[Dict[str, Any]], start_idx: int,
    ) -> List[Dict[str, Any]]:
        """Embed 一批 chunks，返回 records（含 vector）。内部辅助方法。"""
        texts = [c["text"] for c in chunks]
        vectors = self._ef(texts)
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
            })
        return records

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
            logger.info("VectorStore.search: table does not exist yet (query=%r)", query[:60])
            return []
        row_count = table.count_rows()
        if row_count == 0:
            logger.info("VectorStore.search: table exists but is empty (query=%r)", query[:60])
            return []

        # Embed query (use embed_query for proper prefix handling)
        if hasattr(self._ef, 'embed_query'):
            query_vec = _normalize(self._ef.embed_query(query))
        else:
            query_vec = _normalize(self._ef([query])[0])

        # Vector search
        search = table.search(query_vec).limit(top_k)
        if where:
            search = search.where(where)
        results = search.to_list()
        logger.debug(
            "VectorStore.search: %d candidates from %d rows (top_k=%d)",
            len(results), row_count, top_k,
        )

        output: List[Dict[str, Any]] = []
        for r in results:
            # LanceDB 返回 _distance（L2 距离，越小越相似）。
            # 转换为相似度分数 [0,1]（越高越相似），统一下游接口语义。
            distance = float(r.get("_distance", 0.0))
            # 对归一化向量，L2 距离 d ∈ [0,2]，余弦相似度 cos = 1 - d²/2
            similarity = max(0.0, 1.0 - (distance * distance) / 2.0)
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
            # Fallback: candidates 的 score 已由 search() 转换为相似度 [0,1]（越高越好）。
            # 如果 score 仍是原始 L2 距离（向后兼容），用正确公式转换：
            # 对归一化向量，L2 距离 d ∈ [0,2]，余弦相似度 cos = 1 - d²/2
            logger.info("VectorStore.rerank: BGE reranker unavailable, using distance-based fallback")
            for c in candidates:
                try:
                    s = float(c.get("score", 0.0))
                except (TypeError, ValueError):
                    # score 不是合法数值（None/字符串等），置为 0 以免
                    # 后续 sort 因类型不一致崩溃
                    c["score"] = 0.0
                    continue
                # 如果 s > 1，说明是原始 L2 距离（相似度不会 > 1），用正确公式转换
                if s > 1.0:
                    c["score"] = max(0.0, 1.0 - (s * s) / 2.0)
                else:
                    c["score"] = s
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
            logger.info("VectorStore.search_with_rerank: no candidates from vector search")
            return []
        results = self.rerank(query, candidates, top_k=top_k)
        logger.info(
            "VectorStore.search_with_rerank: %d candidates → %d after rerank (top_k=%d)",
            len(candidates), len(results), top_k,
        )
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
        if self._ef is not None:
            try:
                # FastEmbed OnnxEmbeddings stores the InferenceSession at
                # .model.session or .model (depending on version).
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

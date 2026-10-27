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
from typing import Any, Callable, Dict, Iterator, List, Optional, Set

from .parsers import extract_text, SUPPORTED_EXTENSIONS
from .cleaner import clean_text, fix_hard_wraps, clean_markdown_document
from .chunker import chunk_documents
from .document_outline import build_digest, build_toc, convert_plain_text_to_markdown
from .pandoc_parser import (
    PANDOC_EBOOK_EXTS,
    PANDOC_SUPPORTED_EXTENSIONS,
    build_chapter_chunks,
    convert_document,
    convert_ebook,
    is_pandoc_available,
)
from .vector_store import DEFAULT_RERANK_MODEL, VectorStore

logger = logging.getLogger(__name__)


# ---- 内存采样（默认关闭；设置 RAG_MEM_LOG 指向文件时开启）----
# 仅用于排查同步期内存峰值：后台线程每 1s 记录当前 RSS 与进程峰值，各相位
# 通过 _mem_mark 打点，便于把内存曲线对应到解析/切片/向量化阶段。正常运行为
# 空操作（_MEM_LOG_PATH 为空时所有函数直接返回），零开销。
_MEM_LOG_PATH = os.environ.get("RAG_MEM_LOG") or ""
_mem_proc = None
_mem_stop: Optional[threading.Event] = None
_mem_peak = 0

# 同步心跳间隔（秒）：同步是事件驱动日志的——每个文件打「开始处理/解析完成/
# 切片完成」，但**单个大文件（扫描书 OCR/版面/表格）可能连续跑几分钟**，期间
# 没有任何输出，用户看到「很久没日志」会以为卡死。心跳线程按间隔补一条
# 「进行中」进度（已耗时 / 已完成 x·y / 各在飞文件已处理时长），让长转换期间
# 每分钟都有反馈。RAG_HEARTBEAT_S=0 关闭，可调间隔。
_HEARTBEAT_INTERVAL_S = float(os.environ.get("RAG_HEARTBEAT_S", "60") or 0)


def _mem_rss() -> int:
    global _mem_proc
    if _mem_proc is None:
        try:
            import psutil
            _mem_proc = psutil.Process(os.getpid())
        except Exception:
            return 0
    try:
        return _mem_proc.memory_info().rss
    except Exception:
        return 0


def _mem_footprint() -> tuple:
    """返回 (rss, footprint_proxy, mx_active, mx_cache)，单位字节。

    **必须用 footprint 口径**（2026-10-26 排障教训）：Activity Monitor 的
    「内存」列是 phys_footprint（含 Metal/MLX 缓冲与换页），而 RSS 不含——
    实测同一进程 RSS 2.2GB vs footprint 12.1GB，差 10GB。只记 RSS 会完全
    看不到嵌入期 MLX cache 与换页的增长，误判「内存没涨」。
    footprint 代理 = RSS + swap + USS 不可得时的近似：macOS psutil 的
    memory_full_info 提供 uss/swap；MLX 侧直接读 mx.get_active/cache_memory。
    """
    global _mem_proc
    if _mem_proc is None:
        try:
            import psutil
            _mem_proc = psutil.Process(os.getpid())
        except Exception:
            return 0, 0, 0, 0
    try:
        mi = _mem_proc.memory_info()
        rss = mi.rss
        swap = 0
        try:
            swap = _mem_proc.memory_swap()
        except Exception:
            pass
        # USS（私有内存）取不到时用 0
        uss = 0
        try:
            uss = _mem_proc.memory_full_info().uss
        except Exception:
            pass
        mx_active = mx_cache = 0
        try:
            import mlx.core as mx
            mx_active = mx.get_active_memory()
            mx_cache = mx.get_cache_memory()
        except Exception:
            pass
        # MLX/Metal 缓冲不在 RSS、也不在 swap 里（实测 cache 11GB 时
        # rss+swap 仅 0.9GB）——必须单独加进 footprint 口径。
        fp = max(rss, uss) + swap + mx_active + mx_cache
        return rss, fp, mx_active, mx_cache
    except Exception:
        return 0, 0, 0, 0


def _mem_mark(msg: str) -> None:
    if not _MEM_LOG_PATH:
        return
    try:
        rss, fp, mx_active, mx_cache = _mem_footprint()
        global _mem_peak
        _mem_peak = max(_mem_peak, fp)
        ts = time.strftime("%H:%M:%S")
        with open(_MEM_LOG_PATH, "a") as f:
            f.write(
                f"{ts} MARK {msg} fp={fp / 1e9:.2f}GB rss={rss / 1e9:.2f}GB "
                f"mx_active={mx_active / 1e9:.2f}GB mx_cache={mx_cache / 1e9:.2f}GB "
                f"peak={_mem_peak / 1e9:.2f}GB\n"
            )
    except Exception:
        pass


def _mem_sampler(interval: float = 1.0) -> None:
    global _mem_peak
    try:
        with open(_MEM_LOG_PATH, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} START pid={os.getpid()}\n")
            while not _mem_stop.is_set():
                rss, fp, mx_active, mx_cache = _mem_footprint()
                _mem_peak = max(_mem_peak, fp)
                f.write(f"{time.strftime('%H:%M:%S')} fp={fp / 1e9:.2f}GB "
                        f"rss={rss / 1e9:.2f}GB mx_active={mx_active / 1e9:.2f}GB "
                        f"mx_cache={mx_cache / 1e9:.2f}GB peak={_mem_peak / 1e9:.2f}GB\n")
                _mem_stop.wait(interval)
            f.write(f"{time.strftime('%H:%M:%S')} STOP peak={_mem_peak / 1e9:.2f}GB\n")
    except Exception:
        pass


def _is_stale_digest(old_digest: str) -> bool:
    """判断 documents 表中已存的 digest 是否为旧格式、需要重刷。

    旧格式特征（任一即判定为 stale）：
      * 含「# 章节摘要」段 —— 历史版本会生成逐章摘要，已废弃；
      * 含「# 目录」但后面只有空白（目录为空）—— 说明标题提取失败。
    """
    if "# 章节摘要" in old_digest:
        return True
    # 目录为空的旧记录：`# 目录\n` 后紧跟空白，无任何标题行。
    if "# 目录" in old_digest:
        body = old_digest.split("# 目录", 1)[1].strip()
        if not body:
            return True
    return False

# Default number of worker threads for parallel ingestion. Tuned for typical
# local-disk + CPU-bound parsing workloads (PDF OCR, docx parsing, etc.).
# Embedding/vector-store writes stay serial because LanceDB's table lock is
# not thread-safe and the embedding model instance can't be shared across threads.
# 默认 2：docling 并发实例数由 rag_pdf_small_concurrency 单独限流（默认 2 路），
# worker 数只要 >= 并发实例数即可让这些实例真正重叠，无需更高。
DEFAULT_PARALLEL_WORKERS = 2

# worker→主线程队列容量：限制 worker 最多超前产出多少文件的 chunks，
# 超过后 worker 阻塞（背压），避免大知识库时 chunks 在内存堆积。
_QUEUE_CAPACITY_FACTOR = 2  # = parallel_workers * 2

# 哨兵：worker 全部完成后投递到队列，主线程据此退出消费循环
_SENTINEL = object()

# 向量库写入批大小。LanceDB 每次 add 都会产生新 fragment，批太小会在大库上
# 留下海量碎片，检索退化为逐段扫描；批太大则单次失败要重做的量变多。
_WRITE_BATCH_SIZE = 200


def _md_meta_path(md_path: str) -> str:
    """Markdown 缓存的 sidecar 路径：保存「缓存文件名 → 原始文档名」映射。"""
    return md_path + ".meta.json"


def _write_md_meta(md_path: str, payload: Dict[str, Any]) -> None:
    """写 sidecar。失败只告警——它只是加速/纠错的辅助数据，不影响主流程。"""
    try:
        with open(_md_meta_path(md_path), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except OSError as e:
        logger.warning("Markdown meta 写入失败: %s: %s", md_path, e)


def _read_md_meta(md_path: str) -> Optional[Dict[str, Any]]:
    """读 sidecar；不存在或损坏返回 None。"""
    try:
        with open(_md_meta_path(md_path), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------------
# Manifest — tracks which documents have been processed so incremental syncs
# can skip unchanged files entirely (no chunk, no embed, no upsert).
# ---------------------------------------------------------------------------

def _manifest_path(rag_dir: str) -> str:
    """Path to the incremental-sync manifest inside the RAG directory."""
    return os.path.join(rag_dir, "manifest.json")


# 进程内的 manifest 串行化（跨进程由 _ManifestLock 的 flock 保证）。
#
# 必须是**不可重入**的 Lock，并且与 flock 的重入计数解耦：flock 绑定在 open
# file description 上，同一线程对同一文件再 open 一次拿到的 fd 与已有的
# LOCK_EX **互斥**（不区分同进程）。若这里用 RLock，嵌套进入就会让线程第二次
# flock 自己等自己 → 永久挂起。用普通 Lock 时嵌套会立刻在线程锁上暴露。
_MANIFEST_THREAD_LOCK = threading.Lock()
# 线程本地的重入深度：同一线程嵌套进入只加计数，不再重复 flock。
_MANIFEST_TLS = threading.local()


def _manifest_lock_path(rag_dir: str) -> str:
    """Path to the manifest lock file (cross-process advisory lock)."""
    return os.path.join(rag_dir, "manifest.lock")


def _load_manifest_unlocked(rag_dir: str) -> Dict[str, Dict[str, Any]]:
    """``_load_manifest`` 的实现（不加锁，供持锁路径内部调用）。"""
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


def _load_manifest(rag_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load the document manifest (加锁读，避免读到写入一半的状态）。"""
    with _ManifestLock(rag_dir):
        return _load_manifest_unlocked(rag_dir)


class _ManifestLock:
    """manifest 的跨进程/跨实例互斥锁（fcntl.flock  advisory lock）。

    同步可能由多个入口并发触发：UI 的「同步知识库」用一个 RAGEngine，agent
    侧的自动 ingest 用另一个，二者各持一份内存中的 manifest 快照。旧实现
    ``_save_manifest`` 是**全量覆盖写**——后写者会用自己那份旧快照抹掉前者
    刚写入的条目，于是「已同步」的文件被判定为「未同步」，下次全量重跑。

    Windows 上没有 fcntl：退化为「只加线程锁 + 直接写」（该平台不支持多进程
    同步场景，且不会比现状更差）。
    """

    _HAVE_FCNTL = True
    try:
        import fcntl  # noqa: F401
    except ImportError:  # pragma: no cover - Windows
        _HAVE_FCNTL = False

    def __init__(self, rag_dir: str):
        self._rag_dir = rag_dir
        self._fh = None
        self._thread_lock = _MANIFEST_THREAD_LOCK
        # 本次 __enter__ 是否真的拿到了 flock（嵌套时不重复加锁/解锁）。
        self._acquired = False

    def __enter__(self):
        depth = getattr(_MANIFEST_TLS, "depth", 0)
        if depth > 0:
            # 同一线程重入：只加深计数，不再 flock（否则会等自己）。
            _MANIFEST_TLS.depth = depth + 1
            return self
        self._thread_lock.acquire()
        _MANIFEST_TLS.depth = 1
        self._acquired = True
        if self._HAVE_FCNTL:
            try:
                import fcntl
                os.makedirs(self._rag_dir, exist_ok=True)
                self._fh = open(_manifest_lock_path(self._rag_dir), "a+")
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
            except Exception as e:
                logger.debug("RAG: manifest 文件锁不可用 (%s)，退化为仅线程锁", e)
                if self._fh is not None:
                    try:
                        self._fh.close()
                    except Exception:
                        pass
                    self._fh = None
        return self

    def __exit__(self, *exc):
        depth = getattr(_MANIFEST_TLS, "depth", 0)
        if depth > 1:
            _MANIFEST_TLS.depth = depth - 1
            return False
        _MANIFEST_TLS.depth = 0
        if self._fh is not None:
            try:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None
        if self._acquired:
            self._acquired = False
            self._thread_lock.release()
        return False


def _save_manifest(
    rag_dir: str,
    files: Dict[str, Dict[str, Any]],
    removed: Optional[set] = None,
) -> None:
    """Persist the manifest atomically (temp + os.replace), **读改写合并**。

    不再全量覆盖：写入前重新读一次磁盘上的最新内容，把本次调用没碰过的条目
    保留下来。否则两个并发的同步（UI 一个、agent 一个）会互相抹掉对方的
    记录。

    ``removed`` 是**本次有意删除**的 source 集合：这些键即使在磁盘版本里
    存在也必须删掉（否则「别的实例刚写入的条目」和「我刚删掉的条目」无法
    区分，删除会被合并逻辑复活）。
    """
    os.makedirs(rag_dir, exist_ok=True)
    path = _manifest_path(rag_dir)
    with _ManifestLock(rag_dir):
        merged = dict(_load_manifest_unlocked(rag_dir))
        for key in removed or ():
            merged.pop(key, None)
        merged.update(files)
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"files": merged}, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


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


# 采样指纹：头/中/尾各取这么多字节。整库同步时对每个「size 疑似冲突」的
# 文件读 3×64KB，判别字节级副本的代价相对一次 PDF 解析（分钟级）可忽略。
_FP_SAMPLE_BYTES = 64 * 1024


def _file_fingerprint(filepath: str) -> str:
    """文件采样指纹：size + 头/中/尾三段采样内容的 sha256 摘要。

    用途：判别「同一本书放在不同目录」的**字节级副本**——在解析前就去重，
    省掉一次完整的 PDF/Office 解析（OCR/版面/表格，分钟级），也避免同一
    内容多份向量进库造成检索噪声。三段采样 + size 对「同内容同字节」的副本
    足够；对「同内容但字节不同」（重新保存、格式转换）无能为力——那类由
    同步后的 content_hash 清洗兜底（见 _cleanup_duplicate_content）。

    小文件（<= 3 段采样总量）直接全量读，指纹即完整内容摘要。
    """
    size = os.path.getsize(filepath)
    h = hashlib.sha256()
    h.update(b"fp-v1\x00")
    h.update(str(size).encode("ascii"))
    with open(filepath, "rb") as f:
        if size <= _FP_SAMPLE_BYTES * 3:
            h.update(f.read())
            return h.hexdigest()
        h.update(f.read(_FP_SAMPLE_BYTES))               # 头段
        f.seek(max(0, size // 2 - _FP_SAMPLE_BYTES // 2))
        h.update(f.read(_FP_SAMPLE_BYTES))               # 中段
        f.seek(max(0, size - _FP_SAMPLE_BYTES))
        h.update(f.read(_FP_SAMPLE_BYTES))               # 尾段
    return h.hexdigest()


def _escape_like(s: str) -> str:
    """Escape a value used inside a SQL ``LIKE`` literal (with ``ESCAPE '\\'``).

    1. SQL 字符串字面量：单引号翻倍，否则文件名含 ``'``（如 ``it's.pdf``）
       会拼出畸形 SQL / 注入。
    2. LIKE 通配符：``%`` ``_`` ``\\`` 前面加反斜杠。LanceDB 的 LIKE 默认
       不带 ESCAPE 子句，必须在使用处显式写 ``ESCAPE '\\'``，否则 ``\\%``
       不会被当作字面量，含 ``%``/``_`` 的文件名会匹配到所有文档。
    """
    s = s.replace("\\", "\\\\")
    for ch in ("%", "_"):
        s = s.replace(ch, "\\" + ch)
    return s.replace("'", "''")


def _lightweight_fallback(filepath: str) -> tuple:
    """PDF / 图像文件的轻量兜底：OvisOCR3 管线（2026-10 起取代 PyMuPDF+RapidOCR）。

    PDF 与图像文件先走 OvisOCR3（PyMuPDF 取图 + VLM 转 markdown，表格
    区域裁剪识别后填回原位）；引擎不可用或解析失败时退回
    ``parsers.extract_text`` 作最终安全网（文本页直取 + 图页跳过告警）。
    非 PDF/图像格式 ovis 直接返回 None，开销可忽略。
    """
    try:
        from .ovis_ocr import extract_markdown as _ovis_extract
        md = _ovis_extract(filepath)
        if md and md.strip():
            return md, True
    except Exception as e:
        logger.warning("OvisOCR3 兜底失败，退回轻量解析: %s", e)
    return extract_text(filepath), False


def _extract_with_pdf_engine(filepath: str, pdf_engine: str = "docling") -> tuple:
    """Extract text, preferring the configured Markdown engine (docling/marker/ovis).

    Returns ``(text, is_markdown)`` where ``is_markdown`` is True when the
    text came from a Markdown engine (so the caller can skip clean/normalize).

    回退链（两层，语义不同，勿合并）：
      1. 配置的主引擎「不可用」（未安装/导入失败）→ 尝试另一个 Markdown
         引擎。配置指向一个没有安装的引擎时，不该直接掉到轻量解析器而
         浪费另一个重型引擎。
      2. 主引擎「可用但单文件转换失败」→ 走轻量兜底（OvisOCR3 管线 →
         parsers.extract_text）。故意不跨引擎兜底：mid-ingest 冷启动另一个
         重型引擎（可能触发模型下载）会把个别坏文件放大成整库同步的
         长时间停顿。
    """
    if pdf_engine == "ovis":
        # 第三方案：PyMuPDF + OvisOCR3 VLM。不可用/失败 → 轻量兜底
        # （其中会再试一次 ovis，失败标志已缓存，开销可忽略），不冷启动
        # docling/marker 重型引擎。
        try:
            from .ovis_ocr import extract_markdown as _ovis_extract
            md = _ovis_extract(filepath)
            if md and md.strip():
                return md, True
            logger.info("ovis 引擎无输出，回退轻量解析: %s",
                        os.path.basename(filepath))
        except Exception as e:
            logger.warning("ovis 引擎异常，回退轻量解析: %s", e)
        return _lightweight_fallback(filepath)

    if pdf_engine == "marker":
        try:
            from .marker_parser import (extract_markdown as _extract_primary,
                                        is_marker_available as _primary_available)
        except Exception:
            _extract_primary, _primary_available = None, None
        try:
            from .docling_parser import extract_markdown as _extract_alt
        except Exception:
            _extract_alt = None
    else:
        try:
            from .docling_parser import (extract_markdown as _extract_primary,
                                         is_docling_available as _primary_available)
        except Exception:
            _extract_primary, _primary_available = None, None
        try:
            from .marker_parser import extract_markdown as _extract_alt
        except Exception:
            _extract_alt = None

    if _extract_primary is not None and _primary_available is not None \
            and _primary_available():
        markdown = _extract_primary(filepath)
        if markdown is not None:
            return markdown, True
        # 主引擎可用但该文件转换失败 → 只走轻量兜底（见 docstring）。
        return _lightweight_fallback(filepath)

    # 主引擎不可用（未安装/导入失败）→ 另一个 Markdown 引擎顶上；它若同样
    # 不可用，extract_markdown 会立即返回 None，开销可忽略。
    if _extract_alt is not None:
        markdown = _extract_alt(filepath)
        if markdown is not None:
            return markdown, True
    return _lightweight_fallback(filepath)


def _shard_files_balanced(files: List[str], num_workers: int) -> List[List[str]]:
    """把文件列表分片给 worker：按大小降序 + 贪心最轻负载（LPT 调度）。

    旧实现按 ``ceil(n/workers)`` 连续切片：n=9、workers=4 时第 4 片为空
    （3×3=9 已分完），第 4 个 worker 领空启动即退；且慢文件（扫描书要
    逐页 OCR）按文件名顺序扎在同一片时，其余 worker 早早干完长时空转，
    同步尾延迟被拉长。

    现按文件大小降序，每次把文件分给当前累计字节数最小、并列时当前
    已分配文件数最少的 worker：最大的 ``num_workers`` 个文件必然分属
    不同的 worker，n ≥ workers 时空片不可能出现，慢书自动摊开。大小
    只是处理时长的近似（扫描书按页数 OCR、文本书直读），但显著优于
    随机顺序的连续切片。
    """
    if num_workers <= 1:
        return [list(files)]
    sized: List[List[int]] = []
    for f in files:
        try:
            sized.append([os.path.getsize(f), 0, 0])
        except OSError:
            # 扫描后文件消失（TOCTOU）：按 0 处理，worker 内再行隔离
            sized.append([0, 0, 0])
    # 大小降序（并列时按原序号稳定），随后贪心分配
    order = sorted(range(len(files)),
                   key=lambda i: (-sized[i][0], i))
    shards: List[List[str]] = [[] for _ in range(num_workers)]
    totals = [0] * num_workers
    counts = [0] * num_workers
    for i in order:
        # 负载键：累计字节 → 已分配文件数 → worker 序号（保证确定性）
        k = min(range(num_workers), key=lambda w: (totals[w], counts[w], w))
        shards[k].append(files[i])
        totals[k] += sized[i][0]
        counts[k] += 1
    return shards



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
        embedding_backend: str = "fastembed",
        embedding_model_path: str = "",
        embedding_threads: int = 6,
        embedding_batch_size: int = 8,
        embedding_max_length: int = 8192,
        chunk_size: int = 800,
        chunk_overlap: int = 80,
        min_chunk_size: int = 100,
        overlap_percent: float = 0.10,
        semantic_chunking: bool = True,
        # 默认用 INT8 量化版 bge-reranker-v2-m3（~570MB，多语种/中文精排）。
        # 旧默认 "BAAI/bge-reranker-base" 是 fp32 ~1.04GB 且非量化——两个真实
        # 调用点（tools/base.py、ui/main_window.py）都不传 rerank_model，
        # 这个默认值就是实际加载的模型（#131）。
        rerank_model: str = DEFAULT_RERANK_MODEL,
        rerank_backend: str = "fastembed",
        rerank_model_path: str = "",
        rerank_mlx_model_path: str = "",
        rerank_jina_model_path: str = "",
        rerank_threads: int = 4,
        rerank_batch_size: int = 16,
        rerank_max_length: int = 8192,
        rerank_enabled: bool = True,
        parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
        pdf_engine: str = "docling",
        ocr_backend: str = "mlx",
        ocr_model: str = "",
        ocr_model_path: str = "",
        ocr_gguf_path: str = "",
        ocr_mmproj_path: str = "",
        ocr_tokenizer_path: str = "",
        ocr_vlm_handler: str = "",
        ocr_chat_format: str = "",
        docling_dynamic_scale: bool = False,
        docling_vlm: bool = False,
        pdf_small_concurrency: int = 2,
        pdf_large_threshold_mb: float = 20.0,
        pdf_batch_threshold_mb: float = 50.0,
        pdf_batch_pages: int = 50,
    ):
        self._workspace = workspace
        self._knowledge_base = knowledge_base
        self._rag_dir = os.path.join(workspace, "rag")
        self._markdown_dir = os.path.join(self._rag_dir, "documents")
        self._vector_dir = os.path.join(self._rag_dir, "vectors")
        self._embedding_model = embedding_model or "nomic-ai/nomic-embed-text-v1.5-Q"
        self._embedding_backend = embedding_backend
        self._embedding_model_path = embedding_model_path
        self._embedding_threads = max(1, int(embedding_threads))
        self._embedding_batch_size = max(1, int(embedding_batch_size))
        self._embedding_max_length = max(8, int(embedding_max_length))
        self._custom_ef = embedding_function  # for testing injection
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_chunk_size = min_chunk_size
        self._overlap_percent = overlap_percent
        # 语义切片开关（config.rag_semantic_chunking）：关 = 直接字符递归切片
        self._semantic_chunking = bool(semantic_chunking)
        self._rerank_model = rerank_model
        self._rerank_backend = rerank_backend
        self._rerank_model_path = rerank_model_path
        self._rerank_mlx_model_path = rerank_mlx_model_path
        self._rerank_jina_model_path = rerank_jina_model_path
        self._rerank_threads = max(1, int(rerank_threads))
        self._rerank_batch_size = max(1, int(rerank_batch_size))
        self._rerank_max_length = max(8, int(rerank_max_length))
        self._rerank_enabled = bool(rerank_enabled)
        self._parallel_workers = max(1, int(parallel_workers))
        # PDF 主引擎（docling|marker|ovis）。未知值一律回退 docling，避免配置
        # 笔误把整库同步打到意外的路径上。ovis = PyMuPDF + OvisOCR3 VLM，
        # 同时是 PDF/图像的轻量兜底（2026-10 起取代 PyMuPDF+RapidOCR）。
        self._pdf_engine = pdf_engine if pdf_engine in ("docling", "marker", "ovis") else "docling"
        # OvisOCR3 配置：ovis 主路与 PDF 兜底共用，惰性加载，同步结束释放。
        try:
            from .ovis_ocr import configure as _ovis_configure
            _ovis_configure(
                ocr_backend, ocr_model, ocr_model_path,
                gguf_path=ocr_gguf_path, mmproj_path=ocr_mmproj_path,
                tokenizer_path=ocr_tokenizer_path,
                vlm_handler=ocr_vlm_handler, chat_format=ocr_chat_format,
            )
        except Exception as e:
            logger.debug("RAG: ovis configure error: %s", e)
        # docling 动态 OCR 倍率（默认关，全篇固定 scale=2.0；表格密集件
        # 实测 +87% 耗时而 char 级质量收益为零，见 config 项注释）。
        try:
            from .docling_parser import configure_dynamic_scale
            configure_dynamic_scale(docling_dynamic_scale)
        except Exception as e:
            logger.debug("RAG: docling dynamic scale configure error: %s", e)
        # docling VLM 模式（默认关；需 transformers + 联网下载 Granite-Docling
        # 权重，离线时自动回退标准 PDF 管线）。
        try:
            from .docling_parser import configure_vlm
            configure_vlm(docling_vlm)
        except Exception as e:
            logger.debug("RAG: docling vlm configure error: %s", e)
        # PDF 转换按文件大小限流（小文件 N 路并发 / 大文件独占）。内存吃紧的
        # 机器把 rag_pdf_small_concurrency 设 1 即退回全串行。
        try:
            threshold_bytes = max(1.0, float(pdf_large_threshold_mb)) * 1024 * 1024
            from .docling_parser import configure_concurrency as _dc_cfg
            _dc_cfg(max(1, int(pdf_small_concurrency)), int(threshold_bytes))
        except Exception as e:
            logger.debug("RAG: docling concurrency configure error: %s", e)
        try:
            from .marker_parser import configure_concurrency as _mk_cfg
            _mk_cfg(max(1, int(pdf_small_concurrency)), int(threshold_bytes))
        except Exception as e:
            logger.debug("RAG: marker concurrency configure error: %s", e)
        # 大文件按页分批（阈值 MB / 每批页数），避免整篇载入撑爆内存。
        try:
            from .docling_parser import configure_batching as _db_cfg
            _db_cfg(pdf_batch_threshold_mb, pdf_batch_pages)
        except Exception as e:
            logger.debug("RAG: docling batching configure error: %s", e)
        self._store: Optional[VectorStore] = None
        self._store_lock = threading.Lock()  # 保护 _store 懒初始化的线程安全
        self._cancelled = threading.Event()  # 取消标志，cancel() 设置后各阶段检查退出
        # ingest() 启动的 worker 线程注册表。close() 必须等它们全部退出——
        # 否则 OCR 引擎 / ONNX session / LanceDB 连接会在 worker 手里被释放，
        # 原生层 use-after-free 直接段错误（整进程崩，无 Python traceback）。
        self._ingest_threads: List[threading.Thread] = []
        self._ingest_threads_lock = threading.Lock()

    def _register_ingest_threads(self, threads: List[threading.Thread]) -> None:
        """登记本轮 ingest 的 worker 线程，供 close() 等待。

        用**追加**而非覆盖：agent 侧的 RAGEngine 是长生命周期单例（tools/base.py
        的 _rag_engine），可被反复 ingest()。若上一轮有 worker 未能在超时内退出
        （见 _wait_ingest_threads 的兜底分支），覆盖语义会丢失那批线程句柄，
        close() 就只等新一批——仍在跑的旧 worker 与 release_ocr_engine() /
        VectorStore.close() / release_converter() 并发，原生层 use-after-free
        → 段错误。这正是 b292e10 想消除的族类，覆盖语义留了口子。
        """
        with self._ingest_threads_lock:
            # 先剔除已结束的，避免列表随同步轮次无界增长。
            self._ingest_threads = [t for t in self._ingest_threads if t.is_alive()]
            self._ingest_threads.extend(threads)

    def _wait_ingest_threads(self, timeout: float = 600.0) -> bool:
        """等待 ingest worker 全部退出。返回 True 表示已全部结束。

        兜底超时后返回 False——调用方（close）此时不能安全释放原生资源，
        必须记录而不是静默继续。
        """
        with self._ingest_threads_lock:
            threads = list(self._ingest_threads)
        if not threads:
            return True
        deadline = time.monotonic() + timeout
        for i, t in enumerate(threads):
            while t.is_alive() and time.monotonic() < deadline:
                t.join(timeout=30.0)
            if t.is_alive():
                logger.error(
                    "RAG: ingest worker %d/%d 未在 %.0fs 内退出，"
                    "原生资源不能安全释放",
                    i + 1, len(threads), timeout,
                )
                return False
        with self._ingest_threads_lock:
            self._ingest_threads = []
        return True

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
        # 辅助函数：写入日志。当提供 log_callback（UI 同步场景）时，只走
        # callback——否则会与临时 logging handler 捕获的 logger.info 重复，
        # 导致同一条消息在 UI 日志面板出现两遍。无 callback 时（命令行/测试
        # 直接调用 ingest）走 logger.info 输出到终端。
        def _log(msg: str, *args) -> None:
            if args:
                msg = msg % args
            if log_callback is not None:
                log_callback(msg)
            else:
                logger.info(msg)

        if not self._knowledge_base or not os.path.isdir(self._knowledge_base):
            logger.warning("RAG ingest: knowledge base dir not found: %s", self._knowledge_base)
            return {"error": f"Knowledge base directory not found: {self._knowledge_base}"}

        t_start = time.monotonic()

        # 内存采样（排查同步期峰值，默认关闭；设置 RAG_MEM_LOG 时开启）。
        mem_thread = None
        if _MEM_LOG_PATH:
            global _mem_stop, _mem_peak
            _mem_stop = threading.Event()
            _mem_peak = 0
            mem_thread = threading.Thread(target=_mem_sampler, daemon=True)
            mem_thread.start()

        os.makedirs(self._rag_dir, exist_ok=True)
        os.makedirs(self._markdown_dir, exist_ok=True)

        stats: Dict[str, Any] = {
            "knowledge_base": self._knowledge_base,
            "files_found": 0,
            "files_extracted": 0,
            "files_skipped": 0,
            "files_deleted": 0,
            "files_deduped": 0,
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
            removed = self._cleanup_deleted_files(all_files, stats, stats_lock,
                                                  old_manifest=dict(new_manifest),
                                                  new_manifest=new_manifest)
            try:
                _save_manifest(self._rag_dir, new_manifest, removed=removed)
            except OSError as e:
                logger.warning("RAG: failed to save manifest: %s", e)
            if mem_thread is not None and _mem_stop is not None:
                _mem_stop.set()
                mem_thread.join(timeout=2.0)
            return stats
        if self._is_cancelled():
            stats["cancelled"] = True
            if mem_thread is not None and _mem_stop is not None:
                _mem_stop.set()
                mem_thread.join(timeout=2.0)
            return stats

        # 同步心跳：在飞文件登记表 + 周期打点线程。worker 每开始处理一个文件
        # 就登记（filepath -> 开始时间），处理完（任何出口）注销；心跳读这张
        # 表报告「谁还在跑、跑了多久」。默认 60s 一条，RAG_HEARTBEAT_S=0 关。
        _inflight: Dict[str, float] = {}
        _inflight_lock = threading.Lock()
        _hb_stop = threading.Event()

        def _file_started(fp: str, t0: float) -> None:
            with _inflight_lock:
                _inflight[fp] = t0

        def _file_finished(fp: str) -> None:
            with _inflight_lock:
                _inflight.pop(fp, None)

        def _heartbeat() -> None:
            while not _hb_stop.wait(_HEARTBEAT_INTERVAL_S):
                try:
                    now = time.monotonic()
                    with _inflight_lock:
                        items = sorted(_inflight.items(), key=lambda kv: kv[1])
                    with stats_lock:
                        done = stats["files_extracted"] + stats["files_skipped"]
                    if items:
                        detail = ", ".join(
                            f"{os.path.basename(fp)}({now - t0:.0f}s)"
                            for fp, t0 in items[:4]
                        )
                        if len(items) > 4:
                            detail += f" 等{len(items)}个"
                        _log("  ⏱ 同步进行中 (已 %.0fs): 已完成 %d/%d 个文件, "
                             "处理中 %d 个 [%s]",
                             now - t_start, done, total_files, len(items), detail)
                    else:
                        _log("  ⏱ 同步进行中 (已 %.0fs): 已完成 %d/%d 个文件"
                             "（当前无文件在处理：向量化/收尾阶段）",
                             now - t_start, done, total_files)
                    _flush_logs("同步进行中...")
                except Exception:
                    pass  # 心跳是纯观测，绝不影响同步主流程

        if _HEARTBEAT_INTERVAL_S > 0:
            threading.Thread(target=_heartbeat, daemon=True).start()

        # Step 2: load manifest and partition
        _log("━━━ Step 2/6: 增量对比（Manifest） ━━━")
        # 向量库兼容性自愈（Step 2 前置）：嵌入模型切换（如 nomic 768 维 →
        # bge-small 512 维）后旧表维度不匹配、或 schema 缺列，写入会在
        # Step 4 中途失败并浪费整个同步。先检查并在必要时清库重建；库都
        # 清空了，同步记录也必须作废（否则历史 manifest 会跳过本应重新
        # 入库的文件，造成库空但记录齐全的假同步）。
        store = self._get_store()
        manifest_invalidated = False
        if store.reset_if_incompatible():
            manifest_invalidated = True
            try:
                os.remove(_manifest_path(self._rag_dir))
            except OSError as e:
                logger.warning("  ⚠ 同步记录删除失败（将强制忽略）: %s", e)
            _log("  ⚠ 向量库与当前嵌入模型不兼容，已清空重建 → 同步记录作废，本次全量重建")
        manifest = _load_manifest(self._rag_dir)
        if manifest_invalidated:
            manifest = {}
        if manifest:
            _log("  已加载同步记录: %d 个文件有历史记录", len(manifest))
        else:
            _log("  无历史同步记录，将全量处理")
        new_manifest: Dict[str, Dict[str, Any]] = {}
        files_to_process: List[str] = []  # new/modified files needing full pipeline
        skipped_detail: List[str] = []
        new_detail: List[str] = []
        # 当前磁盘上的文件集合：副本条目（duplicate_of）的「规范源晋升」要用——
        # 规范源被删/移走时，副本升级为待处理，否则书会从知识库里消失。
        current_file_set = set(all_files)
        for filepath in all_files:
            # TOCTOU：rglob 扫描与 os.stat 之间文件可能被删除/重命名（编辑器
            # 临时文件、用户操作、云盘同步）。_chunk_iter 对同一竞态已做了
            # try/except OSError，这里漏防——单文件消失会让整个同步崩掉，
            # 已启动的 worker 与未保存的 manifest 全部作废。逐文件隔离。
            try:
                sig = _file_signature(filepath)
            except OSError as e:
                logger.warning("  ⚠ 文件签名失败（扫描后已消失？）: %s: %s",
                               os.path.basename(filepath), e)
                with stats_lock:
                    # stats["errors"] 是 list（见上方 stats 初始化），旧实现
                    # 写成 `+= 1`：list += int 会抛 TypeError（'int' object is
                    # not iterable），整个 ingest() 直接崩掉，已启动的 worker
                    # 与未保存的 manifest 全部作废——恰好毁掉这段 try 想保护的
                    # 东西。其余各处均用 .append，此处统一。
                    stats["errors"].append(
                        f"{os.path.basename(filepath)}: 文件签名失败（扫描后已消失）: {e}")
                continue
            prev = manifest.get(filepath)
            # 副本晋升：本文件此前被判定为另一路径的副本（duplicate_of），若
            # 规范源已不在本次扫描结果里（被删/移走），这个副本就是仅存的
            # 一份——必须升级为待处理，否则内容会从知识库中消失。
            promoted = bool(
                prev is not None
                and prev.get("duplicate_of")
                and prev["duplicate_of"] not in current_file_set
            )
            # dup_retry：上一轮「规范源解析失败、本副本被让位」留下的重试标记
            # （见 _cleanup_duplicate_content）。带此标记的条目即使
            # mtime/size 没变也要重新处理——否则规范源一直失败时，副本永远
            # 被让位，书会从知识库里消失。
            retry_dup = bool(prev is not None and prev.get("dup_retry"))
            if (not force
                    and not promoted
                    and not retry_dup
                    and prev is not None
                    and prev.get("mtime") == sig["mtime"]
                    and prev.get("size") == sig["size"]):
                # Unchanged: carry over manifest entry, skip entirely.
                new_manifest[filepath] = prev
                with stats_lock:
                    stats["files_skipped"] += 1
                skipped_detail.append(f"  ✓ 跳过(未修改): {os.path.basename(filepath)}")
                if progress_callback is not None:
                    try:
                        progress_callback(stats["files_extracted"] + stats["files_skipped"],
                                          total_files, os.path.basename(filepath))
                    except Exception as e:
                        logger.warning("RAG: progress_callback error: %s", e)
            else:
                files_to_process.append(filepath)
                if promoted:
                    reason = "副本晋升"
                elif retry_dup:
                    reason = "副本重试"
                else:
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

        # Step 2.5: 跨目录副本去重（解析前，按文件采样指纹）
        # 同一本书放在不同目录 = 字节完全相同的副本。逐个解析既浪费（一次
        # PDF OCR/版面/表格是分钟级），又让同一内容多份向量进库、检索重复
        # 命中（噪声）。这里只解析每组指纹的第一个（规范源），其余记 manifest
        # 为 duplicate_of、不解析不写向量；历史上真入过库的副本的残留向量由
        # Step 5.5 的 content_hash 清洗兜底。
        self._dedupe_by_fingerprint(
            files_to_process, all_files, manifest, new_manifest,
            stats, stats_lock, _log,
        )

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
        # 本次实际写入的切片数（取消/失败分支要用它，而不是全库总数）。
        added = 0
        # 取消/写失败时「已完整入库」的文件数。**必须在 if files_to_process 之外
        # 初始化**：全部文件都未修改（走 over 的 skip 分支）时里面那行赋值不会执行，
        # 而第 4 步末尾的 cancelled 分支仍会引用它 → UnboundLocalError 会让整个
        # ingest 崩在第 5/6 步之前（清理删除文件、保存 manifest 全部跳过）。
        kept_files = 0
        if files_to_process:
            _log("━━━ Step 3/6: 解析文档 + 清洗 + 切片（%d 个文件，%d 个并行 Worker） ━━━",
                        len(files_to_process), self._parallel_workers)
            q: "queue.Queue[Any]" = queue.Queue(maxsize=self._parallel_workers * _QUEUE_CAPACITY_FACTOR)

            # 每文件各阶段耗时：解析/清洗/切片 由 worker 记录（Step 3），向量化+
            # 入库 由消费者在该文件完整提交时补算（Step 4）。用于每文件处理结束
            # 时输出一段「哪一阶段是慢/卡点」的汇总日志。dict 单键赋值在 GIL 下
            # 原子，多 worker 写不同 filepath 键无需加锁。
            file_stage_times: Dict[str, Dict[str, float]] = {}
            file_consume_start: Dict[str, float] = {}

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
                # 心跳注销辅助：当前在飞文件（兜底也靠它——worker 级 catch-all
                # 异常发生时保证登记表不留下幽灵条目）。
                _cur_fp = None

                def _finish_cur() -> None:
                    nonlocal _cur_fp
                    if _cur_fp is not None:
                        _file_finished(_cur_fp)
                        _cur_fp = None

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
                        # 心跳登记：本文件进入处理（含可能几分钟的 PDF 转换），
                        # 任何出口都要 _finish_cur 注销（见下方各 continue/
                        # return 与循环末尾）。
                        _cur_fp = filepath
                        _file_started(filepath, t0)
                        try:
                            # ---- 1. 解析文件 ----
                            # 路由优先级：pandoc（epub 系电子书/docx/html/LaTeX）
                            # → 配置的 PDF 主引擎（docling/marker）→ 轻量解析器。
                            # pandoc 返回 None（不可用/转换失败）时逐级回退，与
                            # marker 模式的兜底语义一致。
                            _log("    ├─ 解析文件: %s", fname)
                            _flush_logs(f"解析: {fname}")
                            suffix = os.path.splitext(filepath)[1].lower()
                            ebook_toc: Optional[list] = None  # 非 None = epub 系章节切块路径
                            ebook_chapters: Optional[list] = None  # 锚点精确切分的章节文本
                            pandoc_md: Optional[str] = None
                            if suffix in PANDOC_SUPPORTED_EXTENSIONS and is_pandoc_available():
                                md: Optional[str] = None
                                try:
                                    if suffix in PANDOC_EBOOK_EXTS:
                                        _log("    ├─ pandoc 电子书转换: %s", fname)
                                        result = convert_ebook(filepath)
                                        if result is not None:
                                            md, ebook_toc, ebook_chapters = result
                                    else:
                                        _log("    ├─ pandoc 文档转换: %s", fname)
                                        md = convert_document(filepath)
                                except Exception as e:
                                    logger.warning("    ├─ ⚠ pandoc 转换异常，回退默认解析: %s: %s", fname, e)
                                    ebook_toc, ebook_chapters = None, None
                                if md and md.strip():
                                    # --wrap=none 下段落已是单行；fix_hard_wraps 仍
                                    # 清理残留 <br>/NBSP，并对老电子书源修断行。
                                    pandoc_md = fix_hard_wraps(md)
                            if pandoc_md is not None:
                                text, is_markdown = pandoc_md, True
                            else:
                                text, is_markdown = _extract_with_pdf_engine(filepath, self._pdf_engine)
                            t_extract = time.monotonic() - t0
                            _log("    ├─ 解析完成: %s (提取 %d 字符, 耗时 %.2fs%s)",
                                 fname, len(text), t_extract,
                                 f" [{self._pdf_engine}]" if is_markdown else "")
                            _flush_logs(f"解析完成: {fname}")
                            _mem_mark(f"解析完成 {fname} chars={len(text)}")
                        except Exception as e:
                            msg = f"{os.path.basename(filepath)}: {e}"
                            logger.warning("    └─ ❌ 解析失败: %s (耗时 %.2fs): %s", fname, time.monotonic() - t0, e)
                            with stats_lock:
                                stats["errors"].append(msg)
                            _finish_cur()
                            continue
                        if not text.strip():
                            msg = f"{os.path.basename(filepath)}: extracted text is empty"
                            logger.warning("    └─ ⚠ 提取文本为空: %s", fname)
                            with stats_lock:
                                stats["errors"].append(msg)
                            _finish_cur()
                            continue
                        # clean → chunk → put 包在 try 中，避免 clean_text/
                        # normalize_markdown/chunk_documents 异常导致 worker
                        # 静默退出（异常会丢失，不记录到 stats["errors"]）。
                        clean_dur = 0.0  # 清洗阶段耗时（三条子路径各自计时）
                        try:
                            rel_path = os.path.relpath(filepath, self._knowledge_base)
                            md_name = _safe_filename(rel_path) + ".md"
                            md_path = os.path.join(self._markdown_dir, md_name)
                            # 缓存是否可用。旧实现把「缓存命中」的判断写了两次：
                            # 读缓存失败的 except 分支只把 cleaned 置空，但第二处
                            # 判断仍然命中 → cleaned 保持 ""，后面既生成一份空
                            # digest 又跳过切片。这里用同一个变量，读失败即视为
                            # 缓存未命中，强制走重新清洗。
                            cache_is_fresh = (not force
                                    and os.path.exists(md_path)
                                    and os.path.getmtime(filepath) <= os.path.getmtime(md_path))
                            if cache_is_fresh:
                                # ---- 2. 使用缓存的 Markdown ----
                                try:
                                    with open(md_path, "r", encoding="utf-8") as f:
                                        raw = f.read()
                                    # 自愈：历史缓存（旧版本只剥了部分噪音）可能仍含
                                    # <!-- image --> 注释 / raw HTML 标签 / Markdown
                                    # 图片语法 / 内部锚点链接，读出时跑一遍完整清理并
                                    # 落盘，让脏缓存被永久修好（clean_markdown_document
                                    # 幂等，无变化的文件不会触发重写）。
                                    _tc = time.monotonic()
                                    cleaned = clean_markdown_document(raw)
                                    clean_dur = time.monotonic() - _tc
                                    if cleaned != raw:
                                        try:
                                            with open(md_path, "w", encoding="utf-8") as f:
                                                f.write(cleaned)
                                        except OSError:
                                            pass
                                    if not cleaned.strip():
                                        raise OSError("cached markdown is empty")
                                    _log("    ├─ 清洗文本: %s (使用缓存 Markdown, %d 字符)", fname, len(cleaned))
                                except OSError:
                                    cleaned = ""
                                    cache_is_fresh = False
                            if not cache_is_fresh:
                                if is_markdown:
                                    # docling/marker/ovis 已输出规范 Markdown，不跑
                                    # clean_text（其 <[^>]+> 会误删标题/表格/上标语法）。
                                    # 但三引擎都会吐非正文噪音：docling 默认把图片/图题
                                    # 导出成 <!-- image --> 注释、pandoc/marker 透传 raw
                                    # HTML 标签与 Markdown 图片/内部锚点链接……统一用
                                    # clean_markdown_document 无损清理（剥噪音、留正文，
                                    # 保护代码块），否则大量进入切片/嵌入。
                                    _tc = time.monotonic()
                                    cleaned = clean_markdown_document(text)
                                    clean_dur = time.monotonic() - _tc
                                    _log("    ├─ docling Markdown 就绪: %s (%d 字符)", fname, len(cleaned))
                                else:
                                    # ---- 2. 清洗文本 ----
                                    t_clean = time.monotonic()
                                    _log("    ├─ 清洗文本: %s (原始 %d 字符)", fname, len(text))
                                    _flush_logs(f"清洗: {fname}")
                                    cleaned = clean_text(text)
                                    t_clean_end = time.monotonic()
                                    _log("    ├─ 清洗完成: %s → %d 字符 (耗时 %.2fs)", fname, len(cleaned), t_clean_end - t_clean)
                                    # ---- 3. 转换为 Markdown ----
                                    # 非结构化纯文本（txt / OCR 兜底提取等）：先用
                                    # 纯文本启发式还原章节目录结构，再据此把标题行
                                    # 注入成真实 Markdown ``#`` 标题，使后续切片能
                                    # 按章节切分（hybrid_chunker 按 ``#`` 边界）。
                                    # 检测不到标题时等价于原 normalize_markdown（仅
                                    # 做空白归一），不会破坏普通文本。
                                    _log("    ├─ 转换 Markdown: %s", fname)
                                    _flush_logs(f"Markdown: {fname}")
                                    cleaned, _ = convert_plain_text_to_markdown(cleaned)
                                    clean_dur = time.monotonic() - t_clean
                                    _log("    ├─ Markdown 完成: %s (%d 字符)", fname, len(cleaned))
                                    _flush_logs(f"Markdown完成: {fname}")
                                try:
                                    with open(md_path, "w", encoding="utf-8") as f:
                                        f.write(cleaned)
                                    logger.debug("    ├─ Markdown 已缓存: %s", md_name)
                                except OSError as e:
                                    logger.warning("    ├─ Markdown 缓存写入失败: %s: %s", filepath, e)
                            # 无论走缓存还是重新清洗，都补写一次 sidecar：
                            # 「缓存文件名 → 原始文档名」的映射无法从文件名反推
                            # （_safe_filename 会删空格、截断），历史缓存文件也
                            # 借此在下次同步时自动获得正确映射。
                            _write_md_meta(md_path, {
                                "doc_name": os.path.basename(filepath),
                                "source": rel_path,
                            })
                            del text

                            if not cleaned.strip():
                                logger.warning("    └─ ⚠ 清洗后文本为空: %s", fname)
                                continue

                            # ---- 构建文档缩略版本（目录结构 + 标题）----
                            # 只存目录结构（标题层级），不做逐章摘要。
                            # 放在空文本检查之后：空内容不该生成一份
                            # "# 目录\n\n文件名" 的空 digest 入库。
                            # epub 系：优先用 ebooklib 读出的目录树作为缩略版
                            # （书自身的 TOC 比 Markdown 标题提取更准）；TOC 为空
                            # （解析失败/无目录）时回退 Markdown 标题提取。
                            try:
                                if ebook_toc:
                                    self.build_and_store_digest(
                                        rel_path,
                                        cleaned,
                                        md_path=md_path,
                                        digest="# 目录\n\n" + build_toc(ebook_toc),
                                        chapters=ebook_toc,
                                    )
                                else:
                                    self.build_and_store_digest(
                                        # 传相对路径而非 basename：documents 表的
                                        # doc_id 以它为输入，basename 会让不同目录
                                        # 下的同名文件互相覆盖/误删。
                                        rel_path,
                                        cleaned,
                                        md_path=md_path,
                                    )
                                _log("    ├─ 文档缩略版本已生成: %s", fname)
                            except Exception as e:
                                logger.warning("    ├─ ⚠ 缩略版本生成失败: %s: %s", fname, e)

                            # ---- 4. 文档切片 ----
                            t_chunk = time.monotonic()
                            _flush_logs(f"切片: {fname}")
                            c_hash = _content_hash(cleaned)
                            cleaned_len = len(cleaned)
                            if ebook_toc is not None:
                                # 电子书章节切块：优先 pandoc 锚点精确切分
                                # （epub 内容文档边界 + TOC 权威标题，无启发式）；
                                # 无法精确切分（TOC 读不到/锚点不足）时回退到
                                # 原有的标题层级启发式；超长章节交给
                                # chunk_documents 兜底（保证不超嵌入模型上下文）。
                                def _split_oversized(section_text: str) -> List[str]:
                                    return [c["text"] for c in chunk_documents(
                                        [{"source": filepath, "text": section_text}],
                                        chunk_size=self._chunk_size,
                                        chunk_overlap=self._chunk_overlap,
                                        min_chunk_size=self._min_chunk_size,
                                        overlap_percent=self._overlap_percent,
                                        use_semantic=self._semantic_chunking,
                                        # 切片用 model2vec，不注入 nomic EF（避免争抢）
                                        embedding_function=None,
                                    )]

                                # 章节直通上限 = 切片字符硬限（2026-10 起切片
                                # 按字符计，500 字直通；旧值 max(2×, 1200) 曾让
                                # 1100 字纯中文章节 ≈1100 token 被截掉一半）。
                                _max_chars = self._chunk_size
                                if ebook_chapters:
                                    chunk_texts: List[str] = []
                                    for chapter_text in ebook_chapters:
                                        if len(chapter_text) > _max_chars:
                                            chunk_texts.extend(_split_oversized(chapter_text))
                                        else:
                                            chunk_texts.append(chapter_text)
                                else:
                                    chunk_texts = build_chapter_chunks(
                                        cleaned,
                                        None,  # 启发式回退不使用 TOC 标题匹配
                                        max_chars=_max_chars,
                                        oversized_split=_split_oversized,
                                    )
                                doc_chunks = [
                                    {"source": filepath, "chunk_index": i, "text": t}
                                    for i, t in enumerate(chunk_texts)
                                ]
                            else:
                                doc_chunks = chunk_documents(
                                    [{"source": filepath, "text": cleaned}],
                                    chunk_size=self._chunk_size,
                                    chunk_overlap=self._chunk_overlap,
                                    min_chunk_size=self._min_chunk_size,
                                    overlap_percent=self._overlap_percent,
                                    use_semantic=self._semantic_chunking,
                                    # 语义切片强制走 model2vec 静态嵌入（不注入
                                    # 引擎的 nomic EF）：切片只需相邻相似度相对
                                    # 大小，不碰入库 embedding 用的 nomic 及其
                                    # 全局锁，避免与「分片后 embedding 入库」争抢。
                                    embedding_function=None,
                                )
                            t_chunk_end = time.monotonic()
                            _log(
                                "    ├─ 文档切片%s: %s → %d 个切片 (每片约 %d tokens, 重叠 %d tokens, 耗时 %.2fs)",
                                "(章节)" if ebook_toc is not None else "",
                                fname, len(doc_chunks), self._chunk_size, self._chunk_overlap,
                                t_chunk_end - t_chunk,
                            )
                            _flush_logs(f"切片完成: {fname} ({len(doc_chunks)}片)")
                            _mem_mark(f"切片完成 {fname} chunks={len(doc_chunks)}")
                            # 记录本文件前三阶段耗时，供消费者在完整入库时汇总输出。
                            file_stage_times[filepath] = {
                                "parse": t_extract,
                                "clean": clean_dur,
                                "chunk": t_chunk_end - t_chunk,
                            }
                            del cleaned
                        except Exception as e:
                            # clean/chunk 阶段异常：记录并跳过该文件，不中断整个同步。
                            msg = f"{os.path.basename(filepath)}: clean/chunk failed: {e}"
                            logger.error("    └─ ❌ 清洗/切片失败: %s: %s", fname, e, exc_info=True)
                            with stats_lock:
                                stats["errors"].append(msg)
                            _finish_cur()
                            continue

                        with stats_lock:
                            stats["files_extracted"] += 1
                            stats["total_chars"] += cleaned_len

                        # 解析/清洗/切片三阶段小计（向量化+入库在 Step 4，未含在内）。
                        _log(
                            "    └─ ✅ 解析/清洗/切片完成: %s (%d 切片, 耗时 %.2fs，等待向量化)",
                            fname, len(doc_chunks), time.monotonic() - t0,
                        )
                        _flush_logs(f"✅ {fname} ({len(doc_chunks)}切片)")

                        if self._is_cancelled():
                            _finish_cur()
                            return
                        # 背压：队列满时阻塞，避免内存堆积。
                        # 使用 timeout + 循环检查取消标志，避免主线程因
                        # 取消/异常停止消费后 worker 永久阻塞在 put 上。
                        item = (filepath, doc_chunks, c_hash, len(doc_chunks))
                        while not self._is_cancelled() and not _abort.is_set():
                            try:
                                q.put(item, timeout=2.0)
                                break
                            except queue.Full:
                                continue
                        else:
                            _finish_cur()
                            return
                        # Release local references so memory is freed before
                        # the next file is loaded (important for large PDFs:
                        # without this, doc_chunks stays alive until next iter).
                        _finish_cur()
                        del doc_chunks, item
                except Exception as e:
                    # 兜底：任何未预期的异常都记录，避免 worker 静默死亡。
                    logger.error("  [Worker-%d] 异常退出: %s", worker_id % 1000, e, exc_info=True)
                    with stats_lock:
                        stats["errors"].append(f"worker: {e}")
                finally:
                    # 心跳兜底注销：异常路径可能跳过各出口的 _finish_cur，
                    # 这里保证登记表不留下幽灵条目（心跳永远只报真实在飞文件）。
                    _finish_cur()
                    # 投递哨兵也要带超时，防止队列满且主线程已退出消费时
                    # worker 永久卡在 put。
                    _log("  [Worker-%d] 完成所有文件，发送完成信号", worker_id % 1000)
                    try:
                        q.put(_SENTINEL, timeout=5.0)
                    except queue.Full:
                        pass

            # 消费端（add_streaming）失败/中途放弃时置位，让所有 worker 立即
            # 退出 put 循环。旧实现只靠 _is_cancelled()——向量化写入异常并非
            # 用户取消，不会置位该标志，worker 会永远阻塞在 q.put 上：线程与
            # 其持有的整文档切片内存一起泄漏到进程退出。
            _abort = threading.Event()

            # 把文件列表分片给各 worker（LPT 负载均衡，见 _shard_files_balanced）
            # 动态调整并发数：文件数少于默认 worker 数时，只启动必要数量的线程
            num_workers = min(self._parallel_workers, len(files_to_process))
            file_shards = _shard_files_balanced(files_to_process, num_workers)

            threads = []
            for shard in file_shards:
                t = threading.Thread(target=_worker, args=(shard,), daemon=True)
                t.start()
                threads.append(t)
            # 注册到引擎：close() 据此等待这些线程退出（见 _wait_ingest_threads）
            self._register_ingest_threads(threads)

            _flush_logs(f"启动 {len(threads)} 个 Worker ({len(files_to_process)} 个文件)")

            # Step 4: 主线程消费队列，逐文件增量写入 vector store
            _log("━━━ Step 4/6: 向量化 + 存入向量库 ━━━")
            # 已消费文件的累计切片边界（按消费顺序）：取消/失败时用它判断哪些
            # 文件**已经完整入库**，只有这些文件才写进同步记录——下次同步跳过
            # 它们，从剩余文件继续，已完成的解析/清洗/切片/向量化不白做。
            written_files: List[tuple] = []   # (filepath, n_chunks, 累计切片数)
            cum_chunks = 0
            # 已真正 table.add 提交的切片数（每批回调刷新）。
            committed_chunks = {"n": 0}

            def _fully_written_files() -> Dict[str, Dict[str, Any]]:
                """返回已完整入库的文件 → manifest 条目。

                判据：该文件的切片全部落在已提交的批次里（累计边界 ≤ 已提交数）。
                只写了一半的文件不返回——下次重新处理（幂等：add_streaming 会先按
                source 删旧记录再插入），绝不会出现「记录已同步但内容缺一半」。
                """
                done: Dict[str, Dict[str, Any]] = {}
                for fp, _n, cum_end in written_files:
                    if cum_end > committed_chunks["n"]:
                        break
                    if fp in pending_manifest:
                        done[fp] = pending_manifest[fp]
                return done

            def _log_file_stage_times(fp: str) -> None:
                """每文件**完整入库**时输出各阶段耗时，便于定位慢/卡点在哪一步。

                解析/清洗/切片 来自 worker 记录的 file_stage_times；向量化+入库
                = 完整提交时刻 − 该文件开始进入向量化的时刻（file_consume_start）。
                """
                st = file_stage_times.get(fp) or {}
                cs = file_consume_start.get(fp)
                vec_store = (time.monotonic() - cs) if cs is not None else None
                segs = []
                if "parse" in st:
                    segs.append(f"解析 {st['parse']:.2f}s")
                if "clean" in st:
                    segs.append(f"清洗 {st['clean']:.2f}s")
                if "chunk" in st:
                    segs.append(f"切片 {st['chunk']:.2f}s")
                if vec_store is not None:
                    segs.append(f"向量化+入库 {vec_store:.2f}s")
                total = sum(st.values()) + (vec_store or 0.0)
                _log("  └─ ⏱ 各阶段耗时: %s [%s] 合计 %.2fs",
                     os.path.basename(fp), " | ".join(segs), total)
                _flush_logs(f"⏱ {os.path.basename(fp)} 各阶段合计 {total:.1f}s")

            def _chunk_iter() -> Iterator[Dict[str, Any]]:
                nonlocal cum_chunks
                sentinels_seen = 0
                # 必须等于实际启动的 worker 数（num_workers），而不是
                # self._parallel_workers：文件数少于默认并发数时，实际启动的
                # 线程更少，若仍按 _parallel_workers 期待哨兵，会永远等不到
                # 足够数量，只能靠 15s 空超时兜底退出（白白空等）。
                expected_sentinels = num_workers
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
                    # 记录本文件开始进入向量化的时刻：完整入库时减去即得「向量化+入库」耗时。
                    file_consume_start[filepath] = time.monotonic()
                    _log(
                        "  [消费 %d] 向量化排队: %s (%d 个切片等待嵌入)",
                        consumer_file_no, fname, n_chunks,
                    )
                    # 先把这份待入库记录记到 pending（含签名/content_hash/切片数）。
                    # 它何时「算完成」并入 new_manifest 并落盘，取决于切片是否真的
                    # 全部提交：既有下方 except/cancel 分支的「整体成功/已完整入库」
                    # 兜底，也有 _on_batch 里的「逐文件增量落盘」——只要该文件累计
                    # 切片数 <= 已提交数就立即写 manifest，进程被直接杀掉也不会让
                    # 已完成的文件下次重做（半截文件不会触发、幂等重做）。
                    try:
                        pending_manifest[filepath] = {
                            **_file_signature(filepath),
                            "content_hash": c_hash,
                            "chunk_count": n_chunks,
                        }
                    except OSError as e:
                        logger.warning("RAG: failed to update manifest for %s: %s", filepath, e)
                    for c in doc_chunks:
                        yield c
                    # 记下这个文件的累计切片边界（是否真的入库由 committed_chunks 决定）
                    cum_chunks += n_chunks
                    written_files.append((filepath, n_chunks, cum_chunks))
                    # 进度反馈：每消费完一个文件回调一次。
                    if progress_callback is not None:
                        try:
                            with stats_lock:
                                done = stats["files_extracted"] + stats["files_skipped"]
                            progress_callback(done, total_files, os.path.basename(filepath))
                        except Exception as e:
                            logger.warning("RAG: progress_callback error: %s", e)

            def _on_batch(batch_added: int, total_added: int) -> None:
                # 记录已提交切片数：取消/失败时据此判断哪些文件已完整入库。
                committed_chunks["n"] = total_added
                # 增量落盘：某个文件的所有切片都已提交（cum <= total_added）即视为
                # 该文件「全流程完成」（解析/清洗/切片/向量化/入库），立刻把它写进
                # manifest。这样即使进程被直接杀掉（非点「停止同步」），已完成的文件
                # 下次同步也会因 mtime/size 未变被跳过，不会从头重做；只写了一半的
                # 文件不会触发此处、下次幂等重做（add_streaming 先按 source 删旧再插）。
                for _fp, _n, _cum in written_files:
                    if _cum <= total_added and _fp not in new_manifest:
                        _entry = pending_manifest.get(_fp)
                        if _entry is not None:
                            new_manifest[_fp] = _entry
                            # 该文件已完整入库：输出各阶段耗时汇总（解析/清洗/切片/
                            # 向量化+入库），定位慢/卡点。只此一次（_fp 已进 new_manifest）。
                            _log_file_stage_times(_fp)
                            try:
                                _save_manifest(self._rag_dir, {_fp: _entry})
                                _log("  ├─ 已完整入库，增量保存同步记录: %s",
                                     os.path.basename(_fp))
                            except OSError as e:
                                logger.warning(
                                    "RAG: 增量保存同步记录失败（不影响本次同步）: %s", e)
                _mem_mark(f"向量化批次 +{batch_added} (累计 {total_added})")
                _log("  ├─ 向量化批次写入: +%d 切片 (累计 %d)", batch_added, total_added)
                if progress_callback is not None:
                    try:
                        # 写入线程与 ingest worker 并发改 stats：读也持锁，
                        # 避免读到撕裂的中间值。
                        with stats_lock:
                            done = stats["files_extracted"] + stats["files_skipped"]
                        progress_callback(done, total_files, f"已写入 {total_added} 切片")
                    except Exception as e:
                        logger.warning("RAG: on_batch callback error: %s", e)

            write_failed = False
            pending_manifest: Dict[str, Dict[str, Any]] = {}
            try:
                # 批量 200（旧值 20）：LanceDB 每次 add 都会产生一个新
                # fragment，小批量在大库上留下海量碎片，检索要逐段扫描。
                _log("  开始流式向量化写入 (每批 %d 个切片)", _WRITE_BATCH_SIZE)
                added = store.add_streaming(
                    _chunk_iter(),
                    batch_size=_WRITE_BATCH_SIZE,
                    on_batch=_on_batch,
                    is_cancelled=self._is_cancelled,
                )
                stats["chunks"] = added
                stats["chunks_total"] = store.count()
                _log("  向量化写入完成: 共 %d 个切片已存入向量库", added)
            except Exception as e:
                logger.error("  向量化写入失败: %s", e, exc_info=True)
                # 注意：这里不能调用 self.cancel()——向量化写入异常（如模型下载
                # 失败、磁盘满、LanceDB 写入异常）不是用户主动取消，不应该设置
                # cancelled 标志。否则 UI 会错误地显示"同步已取消"并丢弃已
                # 解析的数据。改为将异常记录到 errors，让 ingest 正常结束。
                #
                # stats["chunks"] 必须记「本次写入量」而不是全库总数：UI 用它
                # 展示「本次新增多少切片」，写全库数会让用户以为全部重算了。
                stats["chunks"] = added
                stats["errors"].append(f"Vector store add failed: {e}")
                write_failed = True
            finally:
                # 让阻塞在 q.put 上的 worker 立即退出（成功路径是空操作）。
                _abort.set()

            # 取消 / 写入失败时：只把「已完整入库」的文件写进同步记录。
            # 用户点「停止同步」后，本次已经解析、清洗、切片、入库的文件应当保留
            # 记录，下次同步直接从剩余文件开始，而不是全部从头再来（已完成的
            # 重活不白做）。只写了一半的文件不记录，下次重做（幂等：add_streaming
            # 会先按 source 删旧记录再插入），绝不会出现「记录已同步但内容缺一半」。
            kept_files = 0
            if write_failed:
                kept = _fully_written_files()
                kept_files = len(kept)
                pending_manifest.clear()
                pending_manifest.update(kept)
                new_manifest.update(kept)
                _log("  ⚠ 写入失败：保留 %d 个已完整入库文件的同步记录，"
                     "其余 %d 个文件下次同步重新处理",
                     len(kept), len(files_to_process) - len(kept))
            elif self._is_cancelled():
                kept = _fully_written_files()
                kept_files = len(kept)
                pending_manifest.clear()
                new_manifest.update(kept)
                _log("  ⚠ 同步被取消：%d 个文件已完整入库并记录，"
                     "下次同步从剩余 %d 个文件继续",
                     len(kept), len(files_to_process) - len(kept))
            else:
                new_manifest.update(pending_manifest)

            # 等待所有 worker 线程结束。
            #
            # **必须等到真正退出，不能只 join(timeout=5) 就放过。** worker 的
            # 取消检查只在「文件之间」，单文件内部（OCR 页循环、docling 转换）
            # 完全不检查——一个扫描大 PDF 能跑几十秒，远超 5s。旧实现超时后
            # 只打一条 warning 就返回，ingest() 随即结束，
            # SyncKnowledgeWorker 的 finally 会调 engine.close()：释放 OCR
            # 引擎 / ONNX session / LanceDB 连接。此时 worker 仍在用这些对象，
            # 原生层就是 use-after-free → **整进程段错误**（用户看到「同步一会
            # 就崩溃」，且没有任何 Python traceback）。
            #
            # 所以这里循环等到线程全部退出，只用一个兜底上限防止永久挂死：
            # 单个文件的处理时长没有硬上限（巨型扫描件可能数分钟），兜底取
            # 足够宽裕的值；超时后仍旧返回（不能让 UI 永久卡在「同步中」），
            # 但会记 error 让用户知道本次结果可能不完整。
            _log("  等待所有 Worker 线程结束...")
            _WORKER_JOIN_TIMEOUT = 30.0        # 单轮 join 超时（循环重试）
            _WORKER_JOIN_DEADLINE = 600.0      # 总兜底上限 10 分钟
            _join_deadline = time.monotonic() + _WORKER_JOIN_DEADLINE
            for i, t in enumerate(threads):
                while t.is_alive() and time.monotonic() < _join_deadline:
                    t.join(timeout=_WORKER_JOIN_TIMEOUT)
                if t.is_alive():
                    # 兜底超时：不阻塞 UI 退出，但明确记录——close() 将与仍在
                    # 运行的 worker 竞态，本次同步结果不可信。
                    logger.error(
                        "  Worker 线程 %d 未在 %.0fs 兜底上限内结束，将继续关闭"
                        "（该线程仍在运行，本次同步结果可能不完整）",
                        i + 1, _WORKER_JOIN_DEADLINE,
                    )
                    stats["errors"].append(
                        f"worker thread {i + 1} did not stop within "
                        f"{_WORKER_JOIN_DEADLINE:.0f}s; sync results may be incomplete"
                    )
            _log("  所有 Worker 线程已结束")
        else:
            # All files skipped — no new chunks written. Report 0 like the
            # cancelled branch (chunks_total is already tracked separately
            # for the "全库总量" display; don't confuse it with this sync's
            # written count).
            stats["chunks"] = 0
            _log("  所有 %d 个文件均未修改，无需重新处理", total_files)

        cancelled = self._is_cancelled()
        if cancelled:
            stats["cancelled"] = True
            # 记本次写入量而非全库总数：UI 展示的是「这次同步了多少切片」。
            stats["chunks"] = added
            _log("⚠ 同步已被用户取消: %d/%d 文件已提取, %d 切片已存储"
                 "（%d 个文件已完整入库，下次同步从剩余文件继续）",
                 stats["files_extracted"], stats["files_found"], stats["chunks"],
                 kept_files)

        # Step 5: 清理已删除文件
        try:
            _log("━━━ Step 5/6: 清理已删除文件 ━━━")
            removed_sources = self._cleanup_deleted_files(
                all_files, stats, stats_lock, manifest, new_manifest)
            if stats["files_deleted"] > 0:
                _log("  已清理 %d 个已删除文件的旧向量", stats["files_deleted"])
            else:
                _log("  无已删除文件需要清理")

            # Step 5.5: 按 content_hash 清洗副本向量（同步收尾，manifest 保存前）
            # 两类「同一内容多来源」：① 历史遗留——本功能前已入库的副本，manifest
            # 里多条同 content_hash 且都带向量；② 指纹漏网——同内容不同字节（重新
            # 保存的 PDF、docx→pdf 导出），解析时指纹不同都入了库。每组只保留一个
            # 规范源，其余来源的向量/digest/缓存一并清掉，manifest 标记副本；
            # 规范源没能入库的副本标记 dup_retry，下轮改由它优先。
            self._cleanup_duplicate_content(
                new_manifest, manifest, stats, stats_lock, _log)

            # Step 6: 保存 manifest
            _log("━━━ Step 6/6: 保存同步记录 ━━━")
            try:
                # 读改写合并 + 跨进程锁：并存的另一个 RAGEngine（agent 侧自动
                # ingest）刚写入的条目不能被这份旧快照抹掉。
                _save_manifest(self._rag_dir, new_manifest, removed=removed_sources)
                _log("  同步记录已保存: %d 个文件条目 → %s",
                            len(new_manifest), _manifest_path(self._rag_dir))
            except OSError as e:
                logger.warning("  同步记录保存失败: %s", e)

            # Step 7: 补做/重刷文档目录结构（缩略版本）。
            # 这一步不依赖「文件是否修改」——即使所有文件都被增量跳过（未修改），
            # 也要扫描 markdown 缓存，为缺记录或旧格式（含废弃的「章节摘要」段、
            # 空目录）的文档重刷为纯目录结构，确保 rag_outline 始终返回正确结果。
            try:
                refreshed = self.ensure_digests_for_cached_documents(refresh_stale=True)
                if refreshed:
                    _log("  目录结构（缩略版本）补做/重刷: %d 篇", refreshed)
            except Exception as e:
                logger.warning("  目录结构补做/重刷失败: %s", e)
        finally:
            # 后台观测线程必须**无条件**停：Step 5~7（清理删除项 / 副本去重 /
            # 保存 manifest / 补刷目录）任何一步抛异常，旧实现都只从两条正常
            # return 路径回到这里，异常路径会让心跳与内存采样线程带着闭包
            # （_inflight / stats / stats_lock）继续活满进程余生——每 60s 向一个
            # 早已结束的同步写日志，且持续持有整份 manifest 与统计结构。
            if mem_thread is not None and _mem_stop is not None:
                _mem_stop.set()
                try:
                    mem_thread.join(timeout=2.0)
                except Exception:
                    pass
            _hb_stop.set()

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
        _log("   扫描文件: %d | 新增/更新: %d | 跳过(未修改): %d | 清理删除: %d | 副本去重: %d",
            stats["files_found"], stats["files_extracted"],
            stats["files_skipped"], stats["files_deleted"],
            stats.get("files_deduped", 0),
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

    def _dedupe_by_fingerprint(
        self,
        files_to_process: List[str],
        all_files: List[str],
        manifest: Dict[str, Dict[str, Any]],
        new_manifest: Dict[str, Dict[str, Any]],
        stats: Dict[str, Any],
        stats_lock: threading.Lock,
        log,
    ) -> List[str]:
        """跨目录字节级副本去重（解析前）：同指纹只留一个规范源解析入库。

        同一本书放在不同目录 = 字节完全相同的副本。逐个解析既浪费（一次
        PDF OCR/版面/表格是分钟级），又让同一内容多份向量进库、检索时重复
        命中（噪声）。判别用「文件采样指纹」（_file_fingerprint）：size 相同
        是必要条件，所以只对 size 冲突的文件现算指纹——包括与**旧 manifest
        条目** size 冲突的（规范源可能已入库、本次未修改，没必要重解析）。

        规范源选取：优先「已在库里的」（不在本次待处理列表 → 零代价），其次
        路径字典序（确定性，跨同步稳定）。被跳过的副本记 new_manifest 为
        ``duplicate_of`` 条目（mtime/size 仍在，下次同步直接认出），不解析、
        不写向量；它历史上若真入过库（旧条目有 chunk_count），残留向量由
        Step 5.5 的 _cleanup_duplicate_content 补清。

        返回被跳过（未解析）的副本路径列表。
        """
        if not files_to_process:
            return []
        # size → [paths]，覆盖全部当前文件（不止待处理的）：已入库文件可能
        # 就是规范源。分区循环里已 stat 过一次，这里无法复用（作用域在
        # ingest 内），但 os.stat 本身极便宜，可忽略。
        by_size: Dict[int, List[str]] = {}
        sigs: Dict[str, Dict[str, Any]] = {}
        for fp in all_files:
            try:
                st = os.stat(fp)
            except OSError:
                continue  # TOCTOU：扫描后消失，分区循环已记 error
            by_size.setdefault(st.st_size, []).append(fp)
            sigs[fp] = {"mtime": st.st_mtime, "size": st.st_size}
        # 候选 size：当前文件内部冲突的，加上与旧 manifest 条目冲突的。
        known_sizes = {
            e.get("size") for e in manifest.values() if e.get("size") is not None
        }
        candidate_sizes = {s for s, ps in by_size.items() if len(ps) > 1}
        candidate_sizes |= known_sizes & set(by_size)
        if not candidate_sizes:
            return []
        fp_registry: Dict[str, str] = {}
        for s in candidate_sizes:
            for fp in by_size[s]:
                try:
                    fp_registry[fp] = _file_fingerprint(fp)
                except OSError as e:
                    logger.warning(
                        "  ⚠ 副本指纹计算失败（该文件不参与去重）: %s: %s",
                        os.path.basename(fp), e,
                    )
        # 指纹分组 → 每组选规范源
        groups: Dict[str, List[str]] = {}
        for fp, fg in fp_registry.items():
            groups.setdefault(fg, []).append(fp)
        # 上一轮标记过「规范源失败、让我重试」的路径 → 这些规范源本轮避开
        # （否则规范源一直解析失败时，副本永远被让位，书从知识库里消失）。
        failed_canonicals = {
            e["dup_retry"] for e in manifest.values() if e.get("dup_retry")
        }
        to_process = set(files_to_process)
        skipped: List[str] = []
        for paths in groups.values():
            if len(paths) < 2:
                continue
            # 规范源：没失败过的优先，其次已入库的（False < True，零代价），
            # 最后路径字典序（确定性，跨同步稳定）。
            canonical = sorted(
                paths,
                key=lambda p: (p in failed_canonicals, p in to_process, p),
            )[0]
            for p in sorted(paths):
                if p == canonical or p not in to_process:
                    continue
                entry = dict(sigs.get(p) or _file_signature(p))
                entry["duplicate_of"] = canonical
                new_manifest[p] = entry
                to_process.discard(p)
                skipped.append(p)
                try:
                    rel = os.path.relpath(canonical, self._knowledge_base)
                except ValueError:
                    rel = canonical
                log("  ⤷ 跳过(目录副本): %s → 与 %s 字节相同，不重复解析入库",
                    os.path.basename(p), rel)
                with stats_lock:
                    stats["files_deduped"] = stats.get("files_deduped", 0) + 1
                    stats["files_skipped"] += 1
        files_to_process[:] = [p for p in files_to_process if p in to_process]
        if skipped:
            log("  副本去重: %d 个文件与规范源字节相同，已跳过解析（不重复入库）",
                len(skipped))
        return skipped

    def _purge_source_vectors(self, store, source: str) -> bool:
        """彻底清除一个来源的向量 + digest 行 + markdown 缓存文件。

        markdown 缓存必须一起删：Step 7 的 ensure_digests_for_cached_documents
        按缓存文件重建 digest 行，不删缓存等于白清（副本又会回到 rag_outline）。
        单来源失败不中断整体清理（调用方逐来源隔离，下次同步重试）。
        """
        try:
            store.delete_by_source(source)
            rel = os.path.relpath(source, self._knowledge_base)
            store.delete_document(rel)
            store.delete_document(os.path.basename(source))
            store.delete_document(source)
            try:
                cache = os.path.join(self._markdown_dir, _safe_filename(rel) + ".md")
                if os.path.exists(cache):
                    os.unlink(cache)
            except OSError as e:
                logger.debug("RAG: 删除副本 markdown 缓存失败(忽略) %s: %s", source, e)
            return True
        except Exception as e:
            logger.warning("  ⚠ 清理副本来源失败 %s: %s（下次同步将重试）", source, e)
            return False

    def _cleanup_duplicate_content(
        self,
        new_manifest: Dict[str, Dict[str, Any]],
        old_manifest: Dict[str, Dict[str, Any]],
        stats: Dict[str, Any],
        stats_lock: threading.Lock,
        log,
    ) -> Set[str]:
        """按 content_hash 清洗副本向量（同步收尾，manifest 保存前）。

        两类「同一内容多来源」：
          1. **历史遗留**：本功能上线前已入库的副本——manifest 里多条同
             content_hash、且都真的有向量；
          2. **指纹漏网**：同内容不同字节（重新保存的 PDF、docx→pdf 导出）——
             解析时指纹不同都入了库，但清洗后文本的 content_hash 相同。
        每组只保留一个规范源（路径字典序第一个，确定性），其余来源的向量 /
        digest / markdown 缓存一并清掉，manifest 标记 ``duplicate_of``。

        另外补清 Step 2.5 标记的字节级副本：若它历史上真入过库（旧条目有
        chunk_count），残留向量在这里删掉；从没入过库的新副本没有向量可清。
        规范源本次没能入库（解析失败/取消）的副本，条目改成 ``dup_retry``
        重试标记——分区循环见标记即重新处理，Step 2.5 选规范源时避开这个
        失败过的规范源，书不会因规范源失败而从知识库里消失。

        幂等：标记过 duplicate_of 的条目不再参与分组；向量已删时
        delete_by_source 是空操作，不会每轮同步重复劳动。

        返回被标记 dup_retry 的副本路径集合（仅用于日志/测试观测）。
        """
        if not new_manifest:
            return set()
        # 1. content_hash 分组（只收没被标记过副本的条目——已标记的说明
        #    已在之前的同步里处理过，不该被反复拉起）。
        by_hash: Dict[str, List[str]] = {}
        for path, entry in new_manifest.items():
            if entry.get("duplicate_of"):
                continue
            ch = entry.get("content_hash")
            if ch:
                by_hash.setdefault(ch, []).append(path)
        # 待清洗表：path -> 规范源（先收集再动手，避免白初始化向量库）
        purge: Dict[str, str] = {}
        for paths in by_hash.values():
            if len(paths) < 2:
                continue
            canonical = sorted(paths)[0]
            for p in sorted(paths):
                if p != canonical:
                    purge[p] = canonical
        # 2. 字节级副本：补清历史残留向量 + 规范源缺失时晋升
        promoted: Set[str] = set()
        for p, entry in list(new_manifest.items()):
            dup_of = entry.get("duplicate_of")
            if not dup_of:
                continue
            if dup_of not in new_manifest:
                # 规范源本次没能入库（解析失败/取消/从未成功）：把副本条目改
                # 成 dup_retry 重试标记（mtime/size 保留）——分区循环见标记即
                # 重新处理，Step 2.5 选规范源时也会避开这个失败过的规范源。
                # 书不会因规范源失败而从知识库里消失；两个都一直失败时按
                # 「轮流当规范源」交替重试，错误在 stats 里可见。
                entry = dict(new_manifest[p])
                entry.pop("content_hash", None)
                entry.pop("chunk_count", None)
                entry["dup_retry"] = dup_of
                new_manifest[p] = entry
                promoted.add(p)
                log("  ⤷ 副本重试: %s（规范源 %s 未入库，下轮同步改由本副本优先）",
                    os.path.basename(p), os.path.basename(dup_of))
                continue
            if p in purge:
                continue  # 已在内容分组里安排过清洗
            old = old_manifest.get(p) or {}
            if old.get("chunk_count"):
                purge[p] = dup_of  # 历史上真入过库 → 有残留向量要清
        if not purge:
            return promoted
        # 3. 有活才初始化向量库：全部跳过的同步不该白拉嵌入模型。
        store = self._get_store()
        cleaned = 0
        for p, canonical in purge.items():
            if not self._purge_source_vectors(store, p):
                continue  # 清失败：保留原条目，下次同步重试
            entry = dict(new_manifest.get(p) or _file_signature(p))
            entry.pop("content_hash", None)
            entry.pop("chunk_count", None)
            entry["duplicate_of"] = canonical
            new_manifest[p] = entry
            cleaned += 1
            log("  ⤷ 清洗(副本): %s → 与 %s 重复，已删除其向量",
                os.path.basename(p), os.path.basename(canonical))
            with stats_lock:
                stats["files_deduped"] = stats.get("files_deduped", 0) + 1
        if cleaned:
            log("  副本清洗: 共清除 %d 个重复来源的向量（保留规范源）", cleaned)
        return promoted

    def _cleanup_deleted_files(
        self,
        current_files: List[str],
        stats: Dict[str, Any],
        stats_lock: threading.Lock,
        old_manifest: Optional[Dict[str, Dict[str, Any]]] = None,
        new_manifest: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Set[str]:
        """Delete vectors and manifest entries for files removed from the KB.

        Runs on every ingest (force or not): if a file was in the manifest
        but is no longer on disk, its vectors are stale and must be removed.

        Returns 本次有意删除的 source 集合——``_save_manifest`` 需要它来区分
        「我删掉的条目」（必须删）与「别的实例并发写入的条目」（必须保留）。
        """
        if old_manifest is None:
            old_manifest = _load_manifest(self._rag_dir)
        if new_manifest is None:
            new_manifest = dict(old_manifest)  # caller will update
        current_set = set(current_files)
        removed = [fp for fp in old_manifest if fp not in current_set]
        if not removed:
            return set()
        logger.info("  发现 %d 个已删除文件，清理对应向量与目录概要...", len(removed))
        store = self._get_store()
        for source in removed:
            # 单个文件的删除失败不能中断整个清理：此时向量写入已完成、manifest
            # 尚未保存（Step 6 在其后），一次 LanceDB 异常带着未保存的 manifest
            # 崩溃退出会让增量同步状态不一致。逐文件隔离；失败的文件**保留在
            # manifest 里**，下次同步重试——旧实现无条件 pop，「下次重试」的
            # 承诺永远不成立（条目已不在 manifest，_cleanup_deleted_files 只
            # 遍历 manifest 键），失败文件的向量会永久残留且仍可被检索。
            try:
                store.delete_by_source(source)
                # 同步清理 documents 元数据表里的目录概要（digest）记录。
                # 三个名字都删：doc_name 存的是相对路径（子目录文件），其
                # doc_id 是 md 缓存路径的哈希——只按 basename/source 找永远
                # 找不到子目录文件的行（digest 孤儿行永久累积，rag_outline
                # 继续列出已删文档）。rel_path 按 doc_name 精确命中。
                store.delete_document(os.path.relpath(source, self._knowledge_base))
                # 两个旧版兼容：basename 哈希（最早版本）与完整路径哈希。
                store.delete_document(os.path.basename(source))
                store.delete_document(source)
                logger.info("  🗑 已清理: %s", os.path.basename(source))
                new_manifest.pop(source, None)
                with stats_lock:
                    stats["files_deleted"] += 1
            except Exception as e:
                logger.warning("  ⚠ 清理已删除文件失败 %s: %s（下次同步将重试）", source, e)
        return set(removed)

    # ------------------------------------------------------------------
    # 向量去重（UI「向量去重」按钮，独立于 ingest 的维护操作）
    # ------------------------------------------------------------------

    def dedupe_vectors(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """向量库去重：按文本内容排查重复向量，保留入库最早的一份，其余删除。

        UI「向量去重」按钮调它。与 ingest 收尾的 content_hash 清洗互补：那个按
        manifest 的 content_hash 做**文件级**分组；本方法直接扫向量库全表按
        text 精确分组（**chunk 级**，能找出 manifest 缺失/不一致的历史重复），
        并按入库时间戳（ingested_at）保留最早的一份——「同一本书先后从两个
        目录入库」这类场景，先入库的目录保留、后入库的副本向量被清掉，降低
        检索噪声。

        整来源被清空（它的所有切片都是别处的重复）时，连带清理 digest 行与
        markdown 缓存，并把 manifest 条目标记为规范源的副本——下次同步不会
        再把它的向量加回来。返回统计 dict（scanned/groups/removed/
        sources_cleaned/errors）。
        """
        def _log(msg: str, *args) -> None:
            if args:
                msg = msg % args
            if log_callback is not None:
                log_callback(msg)
            else:
                logger.info(msg)

        def _progress(done: int, total: int, hint: str) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(done, total, hint)
            except Exception:
                pass

        stats: Dict[str, Any] = {
            "scanned": 0, "groups": 0, "removed": 0,
            "sources_cleaned": 0, "errors": [],
        }
        store = self._get_store()
        total = store.count()
        stats["scanned"] = total
        if total == 0:
            _log("  向量库为空，无需去重")
            return stats
        _log("━━━ 向量去重：扫描 %d 条向量（按文本内容分组） ━━━", total)
        _progress(-1, 0, "正在扫描向量库…")
        scan = store.scan_duplicate_texts()
        groups = scan.get("groups") or []
        source_counts = scan.get("source_counts") or {}
        stats["groups"] = len(groups)
        if not groups:
            _log("  ✅ 未发现重复向量（%d 条全部唯一）", total)
            _progress(-1, 0, "未发现重复向量")
            return stats

        drop_ids: List[str] = []
        drop_by_source: Dict[str, int] = {}
        dup_target: Dict[str, str] = {}   # 被清来源 -> 保留其内容的规范源
        for g in groups:
            keep_src = g["keep"]["source"]
            for d in g["drop"]:
                drop_ids.append(d["id"])
                src = d["source"]
                drop_by_source[src] = drop_by_source.get(src, 0) + 1
                dup_target.setdefault(src, keep_src)
        _log("  发现 %d 组重复向量，共 %d 条待清理（涉及 %d 个来源）",
             len(groups), len(drop_ids), len(drop_by_source))

        _BATCH = 500
        done = 0
        for i in range(0, len(drop_ids), _BATCH):
            chunk = drop_ids[i:i + _BATCH]
            try:
                store.delete_by_ids(chunk)
            except Exception as e:
                stats["errors"].append(f"delete batch: {e}")
                logger.warning("向量去重删除批次失败: %s", e)
            done += len(chunk)
            _progress(done, len(drop_ids), f"清理重复向量 {done}/{len(drop_ids)}")

        new_total = store.count()
        stats["removed"] = total - new_total
        _log("  ✅ 已清理 %d 条重复向量（%d → %d）", stats["removed"], total, new_total)

        # 整来源被清空的：清 digest/缓存 + manifest 标记副本，防止下次同步
        # 把它的向量加回来（也防止 Step 7 按缓存重建它的 digest）。
        fully = [s for s, n in drop_by_source.items()
                 if n >= source_counts.get(s, 0)]
        if fully:
            _log("  %d 个来源的全部向量均为重复，连带清理目录概要并标记副本…",
                 len(fully))
            manifest = _load_manifest(self._rag_dir)
            changed = False
            for src in fully:
                try:
                    self._purge_source_vectors(store, src)
                    entry = manifest.get(src)
                    if entry is not None and not entry.get("duplicate_of"):
                        entry = dict(entry)
                        entry.pop("content_hash", None)
                        entry.pop("chunk_count", None)
                        entry["duplicate_of"] = dup_target.get(src, "")
                        manifest[src] = entry
                        changed = True
                    stats["sources_cleaned"] += 1
                    _log("    └─ %s（内容保留自 %s）",
                         os.path.basename(src),
                         os.path.basename(dup_target.get(src) or "?"))
                except Exception as e:
                    stats["errors"].append(f"{src}: {e}")
                    logger.warning("向量去重后来源清理失败 %s: %s", src, e)
            if changed:
                try:
                    _save_manifest(self._rag_dir, manifest)
                except OSError as e:
                    logger.warning("向量去重后 manifest 保存失败: %s", e)
        return stats

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 3,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search the knowledge base with hybrid retrieval + reranking.

        Retrieves top_k * 4 candidates each from vector-similarity search and
        BM25 keyword search, merges/dedupes, then uses BGE reranker to pick
        the most relevant results.  Results below the 0.7 similarity threshold
        are dropped before returning.

        ``source`` 为可选文档名/关键词过滤：定向检索某本书/文档内的切片
        （配合 rag_outline 实现 Targeted RAG——先掌握全局结构，再按需检索
        特定文档的细节）。
        """
        t0 = time.monotonic()
        store = self._get_store()
        store_count = store.count()
        if store_count == 0:
            logger.info("🔍 知识库检索: 向量库为空，无结果返回 (查询=%r)", query[:60])
            return []
        # 构造 SQL 过滤条件：source 字段存完整路径，用 LIKE 匹配文档名。
        where = None
        if source:
            where = f"source LIKE '%{_escape_like(source)}%' ESCAPE '\\'"
            logger.info("  ├─ 定向检索: 限定来源包含 %r", source)
        logger.info(
            "🔍 知识库检索开始: 查询=%r 目标结果数=%d 向量库总量=%d",
            query[:80], top_k, store_count,
        )
        logger.info("  ├─ 混合初检: 向量相似度 + BM25 关键词, 各取 top %d 候选", top_k * 4)
        results = store.search_with_rerank(query, top_k=top_k, where=where)
        t_total = time.monotonic() - t0
        if results:
            logger.info("  └─ ✅ 检索完成: 返回 %d 条结果 (耗时 %.2fs)", len(results), t_total)
            for i, r in enumerate(results, 1):
                # 用 src_name 避免遮蔽同名方法参数（filter 语义是完整路径，
                # 这里只是 basename 展示）。
                src_name = os.path.basename(r.get("source", ""))
                score = r.get("score", 0)
                text_preview = r.get("text", "")[:80].replace("\n", " ")
                logger.info("      [%d] %s (相关度: %.4f) | %s...", i, src_name, score, text_preview)
        else:
            logger.info("  └─ ⚠ 检索完成: 无匹配结果 (耗时 %.2fs)", t_total)
        return results

    def format_results(
        self,
        results: List[Dict[str, Any]],
        max_chars_per_result: int = 1500,
        max_total_chars: int = 8000,
    ) -> str:
        """把检索结果格式化为可直接喂给主模型的字符串（辅助模型归纳的退化路径）。

        每条结果文本截断到 ``max_chars_per_result``，整体上限 ``max_total_chars``，
        避免超长分片撑爆主模型上下文（曾导致 ``exceed_context_size_error``）。
        """
        if not results:
            return "（知识库中未找到相关内容）"

        lines = [f"从知识库检索到 {len(results)} 条相关内容（已重排序）：\n"]
        total = 0
        for i, r in enumerate(results, 1):
            source = os.path.basename(r["source"])
            header = f"### [{i}] 来源: {source} (相关度: {r['score']:.4f})"
            text = r.get("text", "") or ""
            if len(text) > max_chars_per_result:
                text = text[:max_chars_per_result] + "…"
            lines.append(header)
            total += len(header) + 1
            lines.append(text)
            total += len(text)
            lines.append("")
            total += 1
            if total >= max_total_chars:
                lines.append("…（结果过长，已截断）")
                break
        return "\n".join(lines)

    def search_formatted(
        self,
        query: str,
        top_k: int = 3,
        source: Optional[str] = None,
        max_chars_per_result: int = 1500,
        max_total_chars: int = 8000,
    ) -> str:
        """Search and return formatted results as a string.

        Each result's text is truncated to ``max_chars_per_result`` and the
        total output is capped at ``max_total_chars`` to avoid blowing up the
        LLM context window with oversized chunks (some documents produce very
        long chunks, which previously caused ``exceed_context_size_error``).

        ``source`` 为可选文档名过滤（定向检索某本书）。
        """
        results = self.search(query, top_k, source=source)
        return self.format_results(results, max_chars_per_result, max_total_chars)

    # ------------------------------------------------------------------
    # Meta-context + Targeted RAG (文档缩略版本)
    # ------------------------------------------------------------------

    def build_and_store_digest(
        self,
        doc_name: str,
        markdown: str,
        md_path: str = "",
        digest: Optional[str] = None,
        chapters: Optional[list] = None,
    ) -> str:
        """Build a document's digest (缩略版本) and store it in the metadata table.

        The digest is the document's 目录结构（标题层级），不含逐章摘要——避免
        对每章调用 LLM 带来的慢速与空结果问题。标题层级已足以作为 Targeted RAG
        的全局结构元信息。

        ``markdown`` 只用于**构建**目录结构，不会写入元数据表（全文已缓存在
        ``rag/documents/*.md``，表内再存一份是第三份冗余且会被全量载入内存）。

        ``digest`` / ``chapters`` 允许调用方传入已计算好的结果（批量重刷场景
        里 build_digest 的标题提取很昂贵，避免算两遍）。
        """
        if chapters is None:
            digest, chapters = build_digest(markdown)
        # 兜底：纯文本解析结果（如 PDF OCR）通常没有 Markdown 标题，提取不到
        # 任何章节。此时用文档名作为最小缩略版本，避免 rag_outline 返回空目录。
        if not chapters:
            digest = f"# 目录\n\n{os.path.basename(doc_name)}"
        # Serialize chapters (title + level) for targeted retrieval.
        chapters_json = json.dumps(
            [{"level": c.level, "title": c.title}
             for c in chapters],
            ensure_ascii=False,
        )
        store = self._get_store()
        # 存完整 doc_name（通常是相对路径）：basename 会让不同目录下的同名
        # 文件在 documents 表里共用 doc_id 互相覆盖。doc_name 列同时承担
        # rag_outline 的展示与按名检索（含 basename/子串匹配），存相对路径
        # 不影响这两种用法。
        store.upsert_document(
            doc_name=doc_name,
            digest=digest,
            chapters=chapters_json,
            md_path=md_path,
        )
        return digest

    def get_document_outline(self, doc_name: str) -> Optional[Dict[str, Any]]:
        """Return a document's stored digest + chapter list by (partial) name.

        The full markdown is no longer persisted in the metadata table; use
        the returned ``md_path`` (or the on-disk ingest cache) to read it.
        """
        store = self._get_store()
        return store.get_document_digest(doc_name)

    def list_documents(self) -> List[str]:
        """Return stored document names that have a digest."""
        store = self._get_store()
        return store.list_documents()

    def ensure_digests_for_cached_documents(self, refresh_stale: bool = False) -> int:
        """为已缓存 Markdown 的文档补做/重刷 digest（目录结构）。

        扫描 ``markdown_dir`` 缓存目录，对每个 .md 缓存文件：
          * documents 表中缺记录的 → 补做 digest（关键场景：功能上线前已同步
            的文档，有 markdown 缓存 + 向量，但无 metadata 记录）；
          * ``refresh_stale=True`` 时，对已存在但 digest 为旧格式（含「章节摘要」
            段）或目录为空的记录 → 用当前目录结构重刷。

        旧格式判定：历史版本曾生成「# 章节摘要」段（逐章 LLM/提取式摘要），
        该功能已废弃，残留数据需重刷为纯目录结构。

        Returns 本次补做/重刷的文档数量。
        """
        if not os.path.isdir(self._markdown_dir):
            return 0

        store = self._get_store()
        # 一次性快照 documents 表：旧实现循环内每文件按名查询，而每次
        # upsert 都会作废 _documents_cache → 每重刷 1 篇就全表重载一次，
        # M 篇重刷 = M 次全表扫描。快照后整个循环只做内存匹配。
        records = store.list_document_records()
        by_md_path = {
            (r.get("md_path") or "").strip(): r
            for r in records if (r.get("md_path") or "").strip()
        }
        by_name = {
            (r.get("doc_name") or "").lower(): r
            for r in records if (r.get("doc_name") or "").strip()
        }

        created = 0
        for fn in sorted(os.listdir(self._markdown_dir)):
            if not fn.endswith(".md"):
                continue
            md_path = os.path.join(self._markdown_dir, fn)
            # doc_name 的解析优先级：
            #   1) sidecar（写缓存时落下的原始 basename，最权威）
            #   2) documents 表按 md_path 反查（表里的 md_path 也是原值）
            #   3) 从缓存文件名反推（**不可靠**：_safe_filename 会删掉空格、
            #      并把名字截断到 32 字符，"Report 2024.pdf" → "Report2024.pdf"）
            # 只用第 3 种会导致每次同步都查不到已有记录 → 全文重读 + 重建
            # digest，并用错名字再写一条 → documents 表出现永久重复条目。
            meta = _read_md_meta(md_path)
            doc_name = (meta or {}).get("doc_name") or ""
            if not doc_name:
                doc_name = (by_md_path.get(md_path) or {}).get("doc_name", "")
            if not doc_name:
                stem = fn[:-3]  # 去掉 .md
                doc_name = stem
                if len(stem) > 13 and stem[-13] == "_":
                    suffix = stem[-12:]
                    if all(c in "0123456789abcdef" for c in suffix):
                        doc_name = stem[:-13]

            try:
                # 廉价预检：快照内匹配记录再决定要不要读全文。绝大多数缓存
                # 文件的 digest 记录已是最新——旧实现无条件全文读入 +
                # build_digest，大知识库下每次同步都白跑一遍全文 IO 和解析。
                existing = by_name.get(doc_name.lower())
                if existing is not None:
                    if not refresh_stale:
                        continue
                    if not _is_stale_digest(existing.get("digest", "")):
                        continue
                with open(md_path, "r", encoding="utf-8") as f:
                    markdown = f.read()
                if not markdown.strip():
                    continue
                digest, chapters = build_digest(markdown)
                # 记录存在但格式合法时只有内容真的变了才需要重写。
                if existing is not None and digest == existing.get("digest", ""):
                    continue
                # 传入已计算的 digest/chapters，避免 build_and_store_digest 内部
                # 再跑一遍昂贵的标题提取。md_path 必须传：upsert 的 doc_id 以它
                # 为输入——漏传会落库 hash(doc_name) 的重复行，与已有记录
                # （hash(md_path)）并存，get_document_digest 精确匹配永远命中
                # 旧行，重刷结果对用户不可见且 by_md_path 反查永久失效。
                self.build_and_store_digest(
                    doc_name, markdown,
                    digest=digest, chapters=chapters,
                    md_path=md_path,
                )
                created += 1
                logger.info("  ├─ 补做/重刷缩略版本: %s (%d 字符)", doc_name, len(markdown))
            except Exception as e:
                logger.warning("  ⚠ 补做缩略版本失败 %s: %s", fn, e)
        return created

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
        # 必须持 manifest 锁：并发的另一个实例可能已进入 _save_manifest 的
        # 读改写（它 merged 里是旧条目），我们删完后它 os.replace 一写，
        # manifest **立刻被复活** → 向量库空、记录满，此后所有文件都被判为
        # 「已同步」而跳过，检索永远返回空且不会自愈。
        manifest = _manifest_path(self._rag_dir)
        with _ManifestLock(self._rag_dir):
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

        **先等 ingest worker 全部退出再释放。** 释放 OCR 引擎 / ONNX session /
        LanceDB 连接时，若还有 worker 在用它们，原生层就是 use-after-free →
        整进程段错误（同步「一会就崩溃」、且没有 Python traceback 的根因）。
        正常情况下 ingest() 返回前已等过一轮，这里只是兜底。
        """
        if not self._wait_ingest_threads():
            logger.error(
                "RAG: close() 时仍有 ingest worker 在运行，原生资源释放可能与"
                "其竞态（段错误风险）"
            )
        if self._store is not None:
            with self._store_lock:
                if self._store is not None:
                    try:
                        self._store.close()
                    except Exception as e:
                        logger.debug("RAG: VectorStore close error: %s", e)
                    self._store = None
        # Release the global OCR singleton (~500MB) as well.  It is only needed
        # during ingestion of image-based PDFs; after a sync finishes the
        # engine can reclaim it.  Lazy re-created on the next OCR-requiring run.
        try:
            from .parsers import release_ocr_engine
            release_ocr_engine()
        except Exception as e:
            logger.debug("RAG: release_ocr_engine error: %s", e)
        # 同样释放 docling 的共享 DocumentConverter（重量级模型常驻）。
        try:
            from .docling_parser import release_converter
            release_converter()
        except Exception as e:
            logger.debug("RAG: release_converter error: %s", e)
        # marker 的 surya 服务端子进程（VLM 布局/识别模型常驻其中）。
        try:
            from .marker_parser import release_converter
            release_converter()
        except Exception as e:
            logger.debug("RAG: release marker converter error: %s", e)
        # OvisOCR3 VLM 单例（~0.9GB 统一内存）：ovis 主路 / PDF 兜底共用。
        try:
            from .ovis_ocr import release_engine
            release_engine()
        except Exception as e:
            logger.debug("RAG: release ovis engine error: %s", e)

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
                        embedding_backend=self._embedding_backend,
                        embedding_model_path=self._embedding_model_path,
                        embedding_threads=self._embedding_threads,
                        embedding_batch_size=self._embedding_batch_size,
                        embedding_max_length=self._embedding_max_length,
                        rerank_backend=self._rerank_backend,
                        rerank_model_path=self._rerank_model_path,
                        rerank_mlx_model_path=self._rerank_mlx_model_path,
                        rerank_jina_model_path=self._rerank_jina_model_path,
                        rerank_threads=self._rerank_threads,
                        rerank_batch_size=self._rerank_batch_size,
                        rerank_max_length=self._rerank_max_length,
                        rerank_enabled=self._rerank_enabled,
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
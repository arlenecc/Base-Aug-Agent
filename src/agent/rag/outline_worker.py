"""Async chapter-summary worker (Meta-context 缩略版本补强).

During ingestion, documents get a local extractive fallback digest (目录 +
首句摘要) so the outline tool works immediately.  This module upgrades those
digests with LLM-generated per-chapter summaries (≤50 chars, key data/entities
preserved) — asynchronously and only when the LLM is otherwise idle, so a long
summarization job never blocks normal chat/tool usage.

Design:
  * A single background daemon thread processes a bounded queue of pending
    documents, one chapter at a time, with a small sleep between LLM calls.
  * Work is driven by the knowledge-sync flow (started in parallel with
    ingestion), never by the conversation / Agent loop — the worker is fully
    independent of chat.
  * Progress is tracked in ``rag_dir/summaries.json`` so a restart (or the next
    "同步知识" click) resumes where it left off (already-summarized documents
    are skipped).
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_STATE_FILE = "summaries.json"

# Prompt for a single-chapter summary (kept tight to save tokens).
_SUMMARY_PROMPT = (
    "请用不超过50字概括以下章节的核心内容，保留关键数据、专有名词和实体，"
    "直接输出摘要，不要任何解释或前缀。\n\n章节标题：{title}\n章节内容（节选）：\n{text}"
)

# Cap chapter text sent to the LLM (enough for an accurate summary).
_MAX_CHAPTER_CHARS = 2000


def _state_path(rag_dir: str) -> str:
    return os.path.join(rag_dir, _STATE_FILE)


def load_state(rag_dir: str) -> Dict[str, Any]:
    """Load {doc_name: summarized_bool} from summaries.json."""
    path = _state_path(rag_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(rag_dir: str, state: Dict[str, Any]) -> None:
    os.makedirs(rag_dir, exist_ok=True)
    path = _state_path(rag_dir)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


class OutlineWorker:
    """Background worker that upgrades document digests with LLM summaries.

    Fully independent of the Agent / chat loop: it owns its own LLM client and
    store, processes documents one chapter at a time, and resumes from where it
    left off (``summaries.json``) across runs.  It is driven by the knowledge
    sync flow (started in parallel with ingestion), never by the conversation.
    """

    def __init__(
        self,
        rag_dir: str,
        llm: Any,
        get_document: Callable[[str], Optional[Dict[str, Any]]],
        upsert_document: Callable[..., None],
        max_queue: int = 100,
        idle_sleep: float = 0.5,
        on_progress: Optional[Callable[[str], None]] = None,
    ):
        self._rag_dir = rag_dir
        self._llm = llm  # object with chat_stream(messages) yielding StreamEvent
        self._get_document = get_document
        self._upsert_document = upsert_document
        self._on_progress = on_progress  # 进度回调（用于 UI 日志），线程安全由调用方保证
        self._queue: "queue.Queue[str]" = queue.Queue(maxsize=max_queue)
        self._state: Dict[str, Any] = load_state(rag_dir)
        self._state_lock = threading.Lock()
        self._idle_sleep = idle_sleep
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()  # 用于请求提前停止（完成当前章节后）

    def _emit(self, msg: str) -> None:
        """上报进度（若设置了回调）。"""
        if self._on_progress is not None:
            try:
                self._on_progress(msg)
            except Exception:
                pass

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the background worker thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._cancel.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="outline-worker")
        self._thread.start()
        logger.info("OutlineWorker started (async chapter summarization)")

    def stop(self) -> None:
        """Request the worker to stop after the current chapter completes."""
        self._cancel.set()
        self._stop.set()

    def is_idle(self) -> bool:
        return self._thread is None or not self._thread.is_alive()

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def wait_until_done(self, timeout: Optional[float] = None) -> bool:
        """Block until all enqueued documents are processed (or timeout).

        Returns True if the queue is fully drained and the worker thread has
        exited; False on timeout.  Used by the UI QThread to know when to stop
        polling progress.
        """
        if self._thread is None:
            return True
        deadline = None if timeout is None else time.time() + timeout
        while self._thread.is_alive():
            if deadline is not None and time.time() >= deadline:
                return False
            # 队列空且线程即将退出 -> 等线程真正结束
            self._thread.join(timeout=0.2)
        return True

    def pending_count(self) -> int:
        return self._queue.qsize()

    # ------------------------------------------------------------------
    def enqueue_pending(self, doc_names: List[str]) -> int:
        """Enqueue documents whose digest has not yet been LLM-summarized.

        Returns the number actually enqueued (skipping already-done ones).
        """
        enqueued = 0
        for doc_name in doc_names:
            with self._state_lock:
                if self._state.get(doc_name):
                    continue
            try:
                self._queue.put_nowait(doc_name)
                enqueued += 1
            except queue.Full:
                logger.debug("OutlineWorker queue full, stop enqueueing")
                break
        return enqueued

    def mark_summarized(self, doc_name: str) -> None:
        with self._state_lock:
            self._state[doc_name] = True

    # ------------------------------------------------------------------
    def _run(self) -> None:
        """Process pending documents one chapter at a time."""
        while not self._stop.is_set():
            if self._cancel.is_set():
                break
            try:
                doc_name = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if self._cancel.is_set():
                break
            self._emit(f"开始生成摘要: {doc_name}")
            try:
                self._summarize_document(doc_name)
                self._emit(f"✓ 摘要完成: {doc_name}")
            except Exception as e:
                logger.warning("OutlineWorker: summarize %s failed: %s", doc_name, e)
                self._emit(f"✗ 摘要失败: {doc_name} ({e})")
            finally:
                self._queue.task_done()

    def _summarize_document(self, doc_name: str) -> None:
        if self._cancel.is_set():
            return
        doc = self._get_document(doc_name)
        if doc is None:
            return
        markdown = doc.get("markdown", "")
        if not markdown:
            return

        from .document_outline import build_digest

        def summarizer(title: str, text: str) -> str:
            if self._cancel.is_set():
                return ""
            self._emit(f"    章节摘要: {title}")
            snippet = text[:_MAX_CHAPTER_CHARS]
            messages = [{
                "role": "user",
                "content": _SUMMARY_PROMPT.format(title=title, text=snippet),
            }]
            try:
                buf = []
                for ev in self._llm.chat_stream(messages, temperature=0.3):
                    if ev.type == "content":
                        buf.append(ev.content)
                return "".join(buf).strip()
            except Exception as e:
                logger.debug("OutlineWorker chapter summary error: %s", e)
                return ""
            finally:
                # 章节之间让出一点时间，避免连续长时间占用 LLM。
                time.sleep(self._idle_sleep)

        digest, chapters = build_digest(
            markdown,
            summarizer=summarizer,
            max_summary_chars=50,
        )
        chapters_json = json.dumps(
            [{"level": c.level, "title": c.title, "summary": c.summary}
             for c in chapters],
            ensure_ascii=False,
        )
        self._upsert_document(
            doc_name=doc_name,
            digest=digest,
            markdown=markdown,
            chapters=chapters_json,
        )
        self.mark_summarized(doc_name)
        save_state(self._rag_dir, self._state)
        logger.info("OutlineWorker: LLM digest upgraded for %s", doc_name)

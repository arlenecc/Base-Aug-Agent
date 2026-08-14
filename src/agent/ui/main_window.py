"""PyQt6 main window: config strip, chat/send/thinking column, and log panel.

The agent runs in a background QThread; a signal Bridge carries streaming
events to the UI thread. Blocking confirm/ask_user dialogs are synchronised
with threading.Event so the worker blocks until the user answers.
"""
from __future__ import annotations

import io
import logging
import os
import sys
import threading
import time
from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QFont, QTextCursor, QKeyEvent
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QLineEdit, QPushButton, QComboBox, QTextEdit,
    QMessageBox, QInputDialog, QPlainTextEdit, QProgressBar,
    QStatusBar, QFileDialog,
)

from ..agent import Agent, AgentCallbacks
from ..config import AgentConfig
from ..llm_client import LLMClient, LLMError
from ..skills import Skill
from ..tools import ToolRegistry
from ..rag.engine import RAGEngine


CONFIG_PATH = os.path.expanduser("~/.base-agent/config.json")

logger = logging.getLogger(__name__)

# RAG / sync 相关模块的 logger 名称前缀，用于 UI 日志面板过滤
# 覆盖 RAG 全链路：同步引擎、向量库、解析器、清洗器、切片器、依赖检查、工具调用
#
# logger 名取自各模块的 __name__，而包名随运行方式变化：
#   - python main.py / pip 安装 / PyInstaller 冻结 → "agent.rag.engine"
#   - 从仓库根目录以 src.agent 导入（如部分测试）   → "src.agent.rag.engine"
# 这里从本模块的 __name__ 动态推导包根，写死 "src.agent.*" 会导致
# 生产环境（包名为 agent.*）下过滤器和临时 handler 永远匹配不到任何
# 日志记录，UI 日志面板静默丢失全部 RAG 日志。
_PKG_ROOT = __name__.rsplit(".ui", 1)[0]
_RAG_LOG_PREFIXES = (
    f"{_PKG_ROOT}.rag",
    f"{_PKG_ROOT}.tools.rag_tool",
    f"{_PKG_ROOT}.tools.base",
    f"{_PKG_ROOT}.ui.main_window",
)
# RAG 子模块 logger 全名（同步 worker 临时 handler / 日志桥接 level 设置共用）
_RAG_SUB_LOGGERS = tuple(
    f"{_PKG_ROOT}.rag.{m}"
    for m in ("engine", "vector_store", "chunker", "cleaner", "parsers", "deps")
)

# QSignalLogHandler 实例引用，保存在模块级别以便 closeEvent 清理。
# _setup_log_bridge() 创建并赋值，closeEvent() 从 root logger 移除。
_LOG_HANDLER: Optional[Any] = None


class QSignalLogHandler(logging.Handler, QObject):
    """将 Python logging 日志桥接到 Qt UI 的 log_view。

    通过 pyqtSignal 将日志消息安全地传递到 UI 线程，避免跨线程
    直接操作 QPlainTextEdit 导致的崩溃。

    必须同时继承 QObject 和 logging.Handler，因为 pyqtSignal
    需要 QObject 作为基类才能正常工作。
    """

    message_received = pyqtSignal(str)

    def __init__(self) -> None:
        # 同时继承 QObject 和 logging.Handler 时，super().__init__() 只调用
        # MRO 中第一个父类的 __init__。logging.Handler.__init__() 不会调用
        # super().__init__()，因此 QObject.__init__() 被跳过，导致 pyqtSignal
        # 初始化失败。必须显式调用两个父类的 __init__。
        QObject.__init__(self)
        logging.Handler.__init__(self)
        self.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
        self._emit_count = 0
        self._emit_errors = 0
        self._sync_active = False  # 同步期间为 True，跳过 RAG 日志避免并发崩溃

    def close(self) -> None:
        """Override close() to also remove this handler from
        logging._handlerList IN-PLACE.

        The base Handler.close() only removes the handler from the _handlers
        dict (by name) and sets _closed = True.  But logging.shutdown() —
        registered via atexit — iterates _handlerList, a global list of weak
        references.  If the underlying Qt C++ object has already been
        deleted by the time atexit fires, getattr(h, 'flushOnClose') raises
        RuntimeError: wrapped C/C++ object has been deleted.

        Note: we must use slice assignment (_handlerList[:] = [...]) to
        mutate the list in-place, because shutdown()'s default argument
        binds to the list object at function definition time.  Replacing
        _handlerList with a new list would leave the default argument
        pointing at the stale list.
        """
        try:
            import logging as _logging
            _logging._handlerList[:] = [
                wr for wr in _logging._handlerList
                if wr() is not self
            ]
        except Exception:
            pass
        try:
            super().close()
        except RuntimeError:
            pass

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # 同步期间跳过 RAG 日志——这些日志由 _log_buffer + QTimer 处理，
            # 避免 QSignalLogHandler 的 queued signal 和 _poll_sync_logs 的
            # appendPlainText 同时操作 log_view 导致的并发崩溃。
            if self._sync_active and any(
                record.name.startswith(p) for p in _RAG_LOG_PREFIXES
            ):
                return
            msg = self.format(record)
            self._emit_count += 1
            self.message_received.emit(msg)
        except RuntimeError:
            # Qt 对象在 atexit/logging.shutdown 时可能已被销毁，
            # 忽略以避免 "wrapped C/C++ object has been deleted" 异常。
            pass
        except Exception:
            self._emit_errors += 1
            self.handleError(record)

    def flush(self) -> None:
        """Override flush to prevent RuntimeError during logging.shutdown."""
        try:
            super().flush()
        except RuntimeError:
            pass


def _sanitize_stream_text(text: str) -> str:
    """Normalize line endings and drop overstrike carriage returns.

    Some reasoning models emit bare ``\\r`` characters (spinner / line-overwrite
    animation). Inserting those into a text widget returns the cursor to the
    start of the current line, so subsequent text overprints what is already
    there -> visual overlap. We turn CRLF into ``\\n`` and any lone ``\\r`` into
    ``\\n`` so each chunk lands on its own line and nothing overprints.
    """
    if not text:
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n")


# ===========================================================================
# Bridge: thread-safe signal channel between worker and UI
# ===========================================================================


class Bridge(QObject):
    content = pyqtSignal(str)
    reasoning = pyqtSignal(str)
    tool_start = pyqtSignal(str, object)
    tool_end = pyqtSignal(str, object)
    log = pyqtSignal(str)
    usage = pyqtSignal(dict)
    speed = pyqtSignal(int, float)
    skill = pyqtSignal(object)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    # blocking requests (worker -> UI)
    confirm_request = pyqtSignal(str)
    ask_request = pyqtSignal(str)


class BridgeCallbacks(AgentCallbacks):
    """AgentCallbacks that forwards everything through the Bridge."""

    def __init__(self, bridge: Bridge):
        self.bridge = bridge
        self._confirm_event = threading.Event()
        self._ask_event = threading.Event()
        self._lock = threading.Lock()
        self._confirm_result: bool = True
        self._ask_result: Optional[str] = None
        self._cancelled = False

    # streaming
    def on_content(self, text: str) -> None:
        self.bridge.content.emit(text)
    def on_reasoning(self, text: str) -> None:
        self.bridge.reasoning.emit(text)
    def on_tool_start(self, name: str, args: Dict[str, Any]) -> None:
        self.bridge.tool_start.emit(name, args)
    def on_tool_end(self, name: str, result) -> None:
        self.bridge.tool_end.emit(name, result)
    def on_log(self, line: str) -> None:
        self.bridge.log.emit(line)
    def on_usage(self, usage: Dict[str, int]) -> None:
        self.bridge.usage.emit(usage)
    def on_token_speed(self, total_tokens: int, speed: float) -> None:
        self.bridge.speed.emit(total_tokens, speed)
    def on_skill_suggested(self, skill) -> None:
        self.bridge.skill.emit(skill)
    def on_error(self, message: str) -> None:
        self.bridge.error.emit(message)
    def on_finished(self) -> None:
        self.bridge.finished.emit()

    # blocking dialogs (called from worker thread)
    def confirm(self, message: str) -> bool:
        if self._cancelled:
            return False
        self._confirm_event.clear()
        self.bridge.confirm_request.emit(message)
        self._confirm_event.wait(timeout=600.0)
        with self._lock:
            return self._confirm_result

    def ask_user(self, prompt: str) -> Optional[str]:
        if self._cancelled:
            return None
        self._ask_event.clear()
        self.bridge.ask_request.emit(prompt)
        self._ask_event.wait(timeout=600.0)
        with self._lock:
            return self._ask_result

    # setters called from UI thread
    def set_confirm_result(self, val: bool) -> None:
        with self._lock:
            self._confirm_result = val
        self._confirm_event.set()

    def set_ask_result(self, val: Optional[str]) -> None:
        with self._lock:
            self._ask_result = val
        self._ask_event.set()

    def cancel(self) -> None:
        self._cancelled = True
        self._confirm_event.set()
        self._ask_event.set()


# ===========================================================================
# Worker: runs the agent in a background thread
# ===========================================================================


class AgentWorker(QThread):
    def __init__(self, agent: Agent, callbacks: BridgeCallbacks):
        super().__init__()
        self._agent = agent
        self._callbacks = callbacks
        self._user_text: Optional[str] = None

    def set_input(self, text: str) -> None:
        self._user_text = text

    def run(self) -> None:  # type: ignore[override]
        try:
            assert self._user_text is not None
            self._agent.run(self._user_text)
        except Exception as e:
            self._callbacks.bridge.error.emit(f"{type(e).__name__}: {e}")
        finally:
            self._callbacks.bridge.finished.emit()


class FetchModelsWorker(QThread):
    fetched = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, base_url: str, api_key: str):
        super().__init__()
        self._base_url = base_url
        self._api_key = api_key

    def run(self) -> None:  # type: ignore[override]
        try:
            client = LLMClient(base_url=self._base_url, api_key=self._api_key, model="",
                               timeout=30.0, max_tokens=1)
            models = client.list_models()
            client.close()
            self.fetched.emit(models)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class SyncKnowledgeWorker(QThread):
    """Background worker for RAG knowledge base ingestion.

    IMPORTANT: Custom signals are prefixed with `sync_` to avoid shadowing
    QThread's built-in `finished` signal. Shadowing `finished` breaks Qt's
    thread lifecycle management — `deleteLater` connected to the custom
    signal fires *inside* run() (before cleanup), while the built-in
    `finished` (emitted after run() returns) is left with no listeners.
    """
    sync_finished = pyqtSignal(dict)   # stats dict
    sync_failed = pyqtSignal(str)       # error message
    sync_progress = pyqtSignal(int, int, str)  # done, total, current_file

    def __init__(self, workspace: str, knowledge_base: str, embedding_model: str = "",
                 force: bool = False):
        super().__init__()
        self._workspace = workspace
        self._knowledge_base = knowledge_base
        self._embedding_model = embedding_model
        self._force = force
        self._engine: Optional[RAGEngine] = None
        # 日志缓冲区：子线程写入，主线程在 _poll_sync_logs 中消费。
        # 用 Lock 保护，避免多 worker 线程 append 与主线程清空之间的竞态。
        self._log_buffer: list = []
        self._log_buffer_lock = threading.Lock()

    def _emit_log(self, msg: str) -> None:
        """将日志追加到缓冲区，由 _poll_sync_logs 消费写入日志面板。"""
        with self._log_buffer_lock:
            self._log_buffer.append(msg)

    def run(self) -> None:  # type: ignore[override]
        self._emit_log("[rag] ────────── 同步引擎启动 ──────────")
        logger.info(
            "SyncKnowledgeWorker start: workspace=%s kb=%s force=%s embedding=%s",
            self._workspace, self._knowledge_base, self._force,
            self._embedding_model or "(default)",
        )
        # 安装临时 handler：捕获 RAG 各模块通过 logging 输出的日志
        # （如 vector_store 的向量化过程日志），写入 _log_buffer。
        # engine.py 的 _log() 则直接通过 log_callback 写 buffer，
        # 不依赖 logging handler。
        #
        # IMPORTANT: 临时 handler 只挂在父 logger `agent.rag` 上，利用 logging
        # 的传播机制捕获所有子模块（engine / vector_store / ...）的日志。
        # 若同时给父 logger 和每个子 logger 都挂 handler，子 logger 的消息
        # 会被自己的 handler 捕获一次，再传播到父 logger 的 handler 捕获一次，
        # 导致每条日志重复输出两遍（叠加 log_callback 即三遍）。
        _tmp_handler = logging.StreamHandler(io.StringIO())
        _tmp_handler.setLevel(logging.INFO)
        _tmp_handler.setFormatter(
            logging.Formatter("%(message)s")
        )
        def _on_rag_log(record):
            msg = _tmp_handler.format(record)
            with self._log_buffer_lock:
                self._log_buffer.append(msg)
        _tmp_handler.emit = _on_rag_log
        # 只挂父 logger；子 logger 保持 NOTSET（默认传播到父 logger）。
        _rag_loggers = [f"{_PKG_ROOT}.rag"]
        for _name in _rag_loggers:
            _l = logging.getLogger(_name)
            _l.setLevel(logging.INFO)
            _l.addHandler(_tmp_handler)
        emit_stats = None
        emit_error = None
        try:
            self._engine = RAGEngine(
                workspace=self._workspace,
                knowledge_base=self._knowledge_base,
                embedding_model=self._embedding_model,
            )
            self._emit_log("[rag] RAG 引擎初始化完成，开始同步...")
            stats = self._engine.ingest(
                force=self._force,
                progress_callback=self._on_progress,
                log_callback=self._emit_log,
            )
            logger.info(
                "SyncKnowledgeWorker done: found=%d extracted=%d skipped=%d deleted=%d chunks=%d errors=%d cancelled=%s",
                stats.get("files_found", 0), stats.get("files_extracted", 0),
                stats.get("files_skipped", 0), stats.get("files_deleted", 0),
                stats.get("chunks", 0), len(stats.get("errors", [])),
                stats.get("cancelled", False),
            )
            self._emit_log(
                f"[rag] 同步完成: 扫描{stats.get('files_found',0)}文件, "
                f"提取{stats.get('files_extracted',0)}, 跳过{stats.get('files_skipped',0)}, "
                f"切片{stats.get('chunks',0)}"
            )
            emit_stats = stats
        except Exception as e:
            self._emit_log(f"[rag] ❌ 同步失败: {e}")
            logger.error("SyncKnowledgeWorker failed: %s", e, exc_info=True)
            emit_error = f"{type(e).__name__}: {e}"
        finally:
            # 移除临时 handler
            for _name in _rag_loggers:
                try:
                    logging.getLogger(_name).removeHandler(_tmp_handler)
                except Exception:
                    pass
            # Release the VectorStore's resources (LanceDB connection, FastEmbed
            # ONNX model ~130MB, BGE reranker ~500MB). Without this, every sync
            # run leaks ~600MB of C++ heap memory because the ONNX Runtime
            # InferenceSession is not promptly GC'd by Python.
            if self._engine is not None:
                try:
                    self._engine.close()
                    logger.info("SyncKnowledgeWorker: RAG engine closed")
                except Exception as e:
                    logger.warning("SyncKnowledgeWorker: engine close failed: %s", e)
            # Emit the signal only after cleanup is complete so the main
            # thread's UI reset / reload_rag() runs against a fully released
            # worker-side engine.
            if emit_error is not None:
                logger.info("SyncKnowledgeWorker: emitting sync_failed")
                self.sync_failed.emit(emit_error)
            elif emit_stats is not None:
                logger.info("SyncKnowledgeWorker: emitting sync_finished with stats keys=%s",
                            list(emit_stats.keys()) if isinstance(emit_stats, dict) else type(emit_stats))
                self.sync_finished.emit(emit_stats)
            else:
                # 防御：如果 emit_error 和 emit_stats 都是 None（极端情况：
                # RAGEngine 构造或 ingest 抛出非 Exception 的 BaseException，
                # 如 KeyboardInterrupt），确保 UI 不会永远卡在"同步中"。
                logger.error(
                    "SyncKnowledgeWorker: BOTH emit_error and emit_stats are None! "
                    "Emitting sync_failed as fallback to unblock UI."
                )
                self.sync_failed.emit("Unknown sync error (no stats or error captured)")

    def cancel(self) -> None:
        """请求取消 ingest（在当前文件/批处理完成后生效）。"""
        if self._engine is not None:
            self._engine.cancel()

    def _on_progress(self, done: int, total: int, current: str) -> None:
        self.sync_progress.emit(done, total, current)


class OutlineSummarizeWorker(QThread):
    """独立的后台 worker：为知识库文档生成 LLM 章节摘要（缩略版本补强）。

    与 Agent / 对话完全隔离：拥有自己的 LLMClient 和 RAGEngine，在「同步知识」
    时并行启动。断点续传——每次触发都扫描尚未完成 LLM 摘要的文档继续补做，
    直到全部完成。不影响现有任何功能。
    """
    # 进度信号：(已完成数, 总数, 当前文档名)
    outline_progress = pyqtSignal(int, int, str)
    outline_finished = pyqtSignal()   # 全部完成或无可做任务
    outline_failed = pyqtSignal(str)  # 致命错误
    log = pyqtSignal(str)             # 关键日志（传到 UI 面板便于诊断）

    def __init__(self, workspace: str, knowledge_base: str,
                 embedding_model: str, llm_config: dict):
        super().__init__()
        self._workspace = workspace
        self._knowledge_base = knowledge_base
        self._embedding_model = embedding_model
        self._llm_config = llm_config  # {base_url, api_key, model, timeout, max_tokens}
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    def run(self) -> None:  # type: ignore[override]
        try:
            self._run_impl()
        except Exception as e:
            logger.error("OutlineSummarizeWorker fatal: %s", e, exc_info=True)
            self.outline_failed.emit(f"{type(e).__name__}: {e}")

    def _run_impl(self) -> None:
        # 1. 独立的 LLM 客户端（不复用 self._llm，避免与对话争用连接）。
        cfg = self._llm_config
        if not cfg.get("base_url") or not cfg.get("api_key"):
            self.log.emit("LLM 未配置，跳过章节摘要")
            logger.info("OutlineSummarizeWorker: LLM 未配置，跳过章节摘要")
            self.outline_finished.emit()
            return
        llm = LLMClient(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            model=cfg.get("model", ""),
            timeout=cfg.get("timeout", 120.0),
            max_tokens=cfg.get("max_tokens", 32768),
        )

        # 2. 独立的 RAGEngine（复用 workspace/kb，加载独立的 store 句柄）。
        engine = RAGEngine(
            workspace=self._workspace,
            knowledge_base=self._knowledge_base,
            embedding_model=self._embedding_model,
        )
        worker = None
        try:
            # 3. 先为已缓存但缺缩略版本的文档补做 digest（关键：功能上线前
            #    已同步的文档在 documents 表中无记录，必须从这里补齐）。
            self.log.emit(f"扫描缓存文档补做缩略版本 (markdown_dir={engine.markdown_dir})")
            try:
                created = engine.ensure_digests_for_cached_documents()
                self.log.emit(f"补做缩略版本 {created} 篇")
            except Exception as e:
                self.log.emit(f"补做缩略版本失败: {e}")
                logger.warning("OutlineSummarizeWorker: ensure_digests failed: %s", e)

            # 4. 扫描待摘要文档。
            store = engine._get_store()
            doc_names = store.list_documents()
            self.log.emit(f"待摘要文档数: {len(doc_names)}")
            if not doc_names:
                logger.info("OutlineSummarizeWorker: 无可摘要文档")
                self.outline_finished.emit()
                return

            # 4. 构建 OutlineWorker 并续做未完成章节摘要。
            from ..rag.outline_worker import OutlineWorker

            worker = OutlineWorker(
                rag_dir=engine.rag_dir,
                llm=llm,
                get_document=store.get_document_digest,
                upsert_document=store.upsert_document,
                on_progress=lambda msg: self.log.emit(msg),
            )
            # 启动后台线程
            worker.start()
            # 提交所有文档（内部会跳过已完成的）
            enqueued = worker.enqueue_pending(doc_names)
            logger.info("OutlineSummarizeWorker: %d 篇文档待摘要（共 %d 篇）",
                        enqueued, len(doc_names))
            if enqueued == 0:
                self.outline_finished.emit()
                return

            # 5. 轮询进度，直到完成或取消。
            total = len(doc_names)
            done_prev = 0
            while not self._cancelled.is_set():
                # 已完成数 = 已标记摘要的文档数
                done = sum(1 for d in doc_names if worker._state.get(d))
                if done != done_prev:
                    self.outline_progress.emit(done, total, "")
                    done_prev = done
                # 队列已空且线程退出 -> 全部完成
                if worker.wait_until_done(timeout=0.5):
                    break
            # 收尾：上报最终进度
            final_done = sum(1 for d in doc_names if worker._state.get(d))
            if final_done != done_prev:
                self.outline_progress.emit(final_done, total, "")
            if self._cancelled.is_set():
                logger.info("OutlineSummarizeWorker: 用户取消章节摘要")
            else:
                logger.info("OutlineSummarizeWorker: 章节摘要全部完成")
            self.outline_finished.emit()
        finally:
            # 先停止并等待 worker 线程真正退出，再关闭 engine/llm，
            # 避免 daemon 线程还在使用已关闭的连接/存储（时序竞态）。
            if worker is not None:
                try:
                    worker.stop()
                    worker.wait_until_done(timeout=5.0)
                except Exception as e:
                    logger.warning("OutlineSummarizeWorker: worker stop failed: %s", e)
            try:
                engine.close()
            except Exception as e:
                logger.warning("OutlineSummarizeWorker: engine close failed: %s", e)
            try:
                llm.close()
            except Exception:
                pass


class InstallDepsWorker(QThread):
    """Background worker for checking and installing missing RAG dependencies."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)  # DependencyReport

    def __init__(self, kb_path: str):
        super().__init__()
        self._kb_path = kb_path
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """Signal the worker to stop (best-effort, pip install can't be interrupted)."""
        self._cancelled.set()

    def run(self) -> None:  # type: ignore[override]
        try:
            if self._cancelled.is_set():
                self.progress.emit("依赖检查已取消")
                self.finished.emit(None)
                return
            from ..rag.deps import ensure_dependencies
            report = ensure_dependencies(
                self._kb_path,
                auto_install=True,
                include_core=True,
                progress_callback=self.progress.emit,
            )
            if self._cancelled.is_set():
                self.progress.emit("依赖检查已取消")
                self.finished.emit(None)
                return
            self.finished.emit(report)
        except Exception as e:
            self.progress.emit(f"依赖检查异常: {e}")
            self.finished.emit(None)


# ===========================================================================
# Keyed plain text editor: Enter sends, Shift+Enter newline
# ===========================================================================


class SendEditor(QPlainTextEdit):
    send_requested = pyqtSignal()

    def keyPressEvent(self, e: QKeyEvent) -> None:  # type: ignore[override]
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (e.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.send_requested.emit()
            return
        super().keyPressEvent(e)


# ===========================================================================
# Main window
# ===========================================================================


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Base Augment Agent")
        self.resize(1280, 820)

        self.config = AgentConfig.load(CONFIG_PATH)
        self.config.ensure_workspace()
        self._bridge = Bridge()
        self._callbacks = BridgeCallbacks(self._bridge)
        self._llm: Optional[LLMClient] = None
        self._agent: Optional[Agent] = None
        self._worker: Optional[AgentWorker] = None
        self._fetch_worker: Optional[FetchModelsWorker] = None
        self._sync_worker: Optional[SyncKnowledgeWorker] = None
        self._outline_worker: Optional[OutlineSummarizeWorker] = None
        self._deps_worker: Optional[InstallDepsWorker] = None
        self._deps_cancelled: bool = False  # True if user cancelled during deps check
        self._streaming_assistant = False  # True while appending content to current assistant bubble
        self._busy = False
        self._total_tokens = 0
        # 日志轮询定时器：每 100ms 检查 sync_worker._log_buffer，
        # 把新日志追加到 log_view。绕过 Qt queued signal 机制，
        # 避免在 GIL 竞争下信号无法被主线程处理的问题。
        self._log_poll_timer = QTimer()
        self._log_poll_timer.setInterval(100)  # 100ms
        self._log_poll_timer.timeout.connect(self._poll_sync_logs)
        # Pending skill suggestion — cached when the agent emits
        # on_skill_suggested during a run, and shown as a dialog AFTER the
        # run finishes. Showing a modal dialog during streaming would block
        # the UI event loop and prevent content/reasoning chunks from
        # rendering, making the app look frozen.
        self._pending_skill: Optional[Skill] = None

        self._build_ui()
        self._wire_signals()
        self._setup_log_bridge()
        self._apply_config_to_widgets()
        self._rebuild_agent()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # Top config strip (~1/6 height) + main area below.
        main_split = QSplitter(Qt.Orientation.Vertical)
        outer.addWidget(main_split)

        main_split.addWidget(self._build_config_panel())
        main_split.addWidget(self._build_body_panel())
        main_split.setStretchFactor(0, 1)
        main_split.setStretchFactor(1, 5)
        main_split.setSizes([140, 700])

        # Status bar with progress
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(220)
        self.progress.setRange(0, 0)  # indeterminate off by default
        self.progress.setRange(0, 1)
        self.status.addPermanentWidget(QLabel(""))
        self.status.addPermanentWidget(self.progress)
        self.status_label = QLabel("就绪")
        self.status.addWidget(self.status_label)

    def _build_config_panel(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        self.base_url_edit = QLineEdit(placeholderText="https://api.openai.com/v1")
        self.api_key_edit = QLineEdit(placeholderText="sk-...", echoMode=QLineEdit.EchoMode.Password)
        self.model_combo = QComboBox(editable=True)
        self.model_combo.setMinimumWidth(180)
        self.fetch_btn = QPushButton("获取模型")
        self.temp_edit = QLineEdit("0.7")
        self.temp_edit.setFixedWidth(60)
        self.max_tokens_edit = QLineEdit("32768")
        self.max_tokens_edit.setFixedWidth(80)
        self.top_p_edit = QLineEdit("0.95")
        self.top_p_edit.setFixedWidth(60)
        self.min_p_edit = QLineEdit("0.05")
        self.min_p_edit.setFixedWidth(60)
        self.top_k_edit = QLineEdit("20")
        self.top_k_edit.setFixedWidth(60)
        self.repeat_penalty_edit = QLineEdit("1.0")
        self.repeat_penalty_edit.setFixedWidth(60)
        self.timeout_edit = QLineEdit("120")
        self.timeout_edit.setFixedWidth(60)
        self.workspace_edit = QLineEdit()
        self.workspace_btn = QPushButton("…")
        self.workspace_btn.setFixedWidth(28)
        self.knowledge_base_edit = QLineEdit(placeholderText="（可选）知识库目录")
        self.knowledge_base_btn = QPushButton("…")
        self.knowledge_base_btn.setFixedWidth(28)
        self.connect_btn = QPushButton("应用配置")
        self.connect_btn.setStyleSheet("font-weight:600;")
        self.clear_btn = QPushButton("清空对话")
        self.stop_btn = QPushButton("终止对话")
        self.stop_btn.setEnabled(False)

        # Row 0: Base URL | API Key | 模型 | 获取模型
        row0 = QHBoxLayout()
        row0.setSpacing(8)
        row0.addWidget(QLabel("Base URL"))
        row0.addWidget(self.base_url_edit, 3)
        row0.addWidget(QLabel("API Key"))
        row0.addWidget(self.api_key_edit, 2)
        row0.addWidget(QLabel("模型"))
        row0.addWidget(self.model_combo, 2)
        row0.addWidget(self.fetch_btn)
        lay.addLayout(row0)

        # Row 1: 上下文长度 | 温度 | top_p | min_p | top_k | 重复惩罚 | 超时(秒) | 应用配置
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        row1.addWidget(QLabel("context"))
        row1.addWidget(self.max_tokens_edit)
        row1.addWidget(QLabel("Temp"))
        row1.addWidget(self.temp_edit)
        row1.addWidget(QLabel("top_p"))
        row1.addWidget(self.top_p_edit)
        row1.addWidget(QLabel("min_p"))
        row1.addWidget(self.min_p_edit)
        row1.addWidget(QLabel("top_k"))
        row1.addWidget(self.top_k_edit)
        row1.addWidget(QLabel("repeat-penalty"))
        row1.addWidget(self.repeat_penalty_edit)
        row1.addWidget(QLabel("超时(秒)"))
        row1.addWidget(self.timeout_edit)
        row1.addStretch(1)
        row1.addWidget(self.connect_btn)
        lay.addLayout(row1)

        # Row 2: 工作目录 | … | 知识库目录 | … | 停止
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        row2.addWidget(QLabel("工作目录"))
        row2.addWidget(self.workspace_edit, 1)
        row2.addWidget(self.workspace_btn)
        row2.addWidget(QLabel("知识库"))
        row2.addWidget(self.knowledge_base_edit, 1)
        row2.addWidget(self.knowledge_base_btn)
        self.sync_knowledge_btn = QPushButton("同步知识")
        self.sync_knowledge_btn.setToolTip(
            "解析知识库文档、建立向量索引，支持 RAG 检索\n"
            "左键：增量同步（只处理新增/修改的文件）\n"
            "右键：强制全量重新处理"
        )
        # Right-click context menu: force full reprocess (ignores manifest).
        self.sync_knowledge_btn.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        self._force_sync_action = self.sync_knowledge_btn.addAction("强制全量重新处理")
        self._force_sync_action.triggered.connect(lambda: self._on_sync_knowledge(force=True))
        row2.addWidget(self.sync_knowledge_btn)
        # Stop-sync button: enabled only while a sync is running. Cancels
        # the ingest after the current file completes (cooperative cancel).
        self.stop_sync_btn = QPushButton("停止同步")
        self.stop_sync_btn.setEnabled(False)
        self.stop_sync_btn.setToolTip("停止知识库同步（在当前文件处理完成后生效）")
        row2.addWidget(self.stop_sync_btn)
        lay.addLayout(row2)

        return box

    def _build_body_panel(self) -> QWidget:
        hsplit = QSplitter(Qt.Orientation.Horizontal)

        # Left column: chat / send / thinking
        left = QWidget()
        llay = QVBoxLayout(left)
        llay.setContentsMargins(0, 0, 0, 0)
        llay.setSpacing(4)

        # chat header with controls
        chat_header = QHBoxLayout()
        chat_header.setSpacing(8)
        chat_header.addWidget(QLabel("对话"))
        chat_header.addStretch(1)
        chat_header.addWidget(self.clear_btn)
        chat_header.addWidget(self.stop_btn)
        llay.addLayout(chat_header)

        # chat view
        self.chat_view = QTextEdit()
        self.chat_view.setReadOnly(True)
        self.chat_view.document().setDefaultStyleSheet(
            ".u{color:#2563eb}.a{color:#059669}.t{color:#7c3aed}.sys{color:#9ca3af}"
        )
        llay.addWidget(self.chat_view, 3)

        # send
        send_row = QHBoxLayout()
        self.send_editor = SendEditor()
        self.send_editor.setFixedHeight(64)
        self.send_editor.setPlaceholderText("输入消息…  (Enter 发送 / Shift+Enter 换行)")
        send_row.addWidget(self.send_editor)
        send_col = QVBoxLayout()
        self.send_btn = QPushButton("发送")
        self.send_btn.setFixedHeight(64)
        self.send_btn.setStyleSheet("font-weight:600;")
        send_col.addWidget(self.send_btn)
        send_row.addLayout(send_col)
        llay.addLayout(send_row)

        # thinking
        think_header = QHBoxLayout()
        think_header.addWidget(QLabel("思考过程"))
        self.speed_label = QLabel("0.0 tok/s")
        self.speed_label.setStyleSheet("color:#6b7280;")
        self.token_label = QLabel("总计 0 tokens")
        self.token_label.setStyleSheet("color:#6b7280;")
        think_header.addStretch(1)
        think_header.addWidget(self.speed_label)
        think_header.addWidget(QLabel("|"))
        think_header.addWidget(self.token_label)
        llay.addLayout(think_header)
        self.think_view = QPlainTextEdit()
        self.think_view.setReadOnly(True)
        font = QFont("Menlo")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.think_view.setFont(font)
        llay.addWidget(self.think_view, 2)

        hsplit.addWidget(left)

        # Right column: log
        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.setSpacing(4)
        log_header = QHBoxLayout()
        log_header.addWidget(QLabel("日志"))
        self.log_clear_btn = QPushButton("清空")
        self.log_clear_btn.setFixedWidth(60)
        log_header.addStretch(1)
        log_header.addWidget(self.log_clear_btn)
        rlay.addLayout(log_header)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(font)
        self.log_view.setMaximumBlockCount(5000)  # 限制日志行数，防止内存无限增长
        rlay.addWidget(self.log_view, 1)
        hsplit.addWidget(right)

        hsplit.setStretchFactor(0, 2)
        hsplit.setStretchFactor(1, 1)
        hsplit.setSizes([820, 420])
        return hsplit

    # ------------------------------------------------------------------
    # wiring
    # ------------------------------------------------------------------
    def _wire_signals(self) -> None:
        self.fetch_btn.clicked.connect(self._on_fetch_models)
        self.connect_btn.clicked.connect(self._on_apply_config)
        self.workspace_btn.clicked.connect(self._on_pick_workspace)
        self.knowledge_base_btn.clicked.connect(self._on_pick_knowledge_base)
        self.sync_knowledge_btn.clicked.connect(self._on_sync_knowledge)
        self.stop_sync_btn.clicked.connect(self._on_stop_sync)
        self.clear_btn.clicked.connect(self._on_clear_chat)
        self.log_clear_btn.clicked.connect(self.log_view.clear)
        self.send_btn.clicked.connect(self._on_send)
        self.send_editor.send_requested.connect(self._on_send)
        self.stop_btn.clicked.connect(self._on_stop)

        b = self._bridge
        b.content.connect(self._on_content)
        b.reasoning.connect(self._on_reasoning)
        b.tool_start.connect(self._on_tool_start)
        b.tool_end.connect(self._on_tool_end)
        b.log.connect(self._on_log)
        b.usage.connect(self._on_usage)
        b.speed.connect(self._on_speed)
        b.skill.connect(self._on_skill_suggested)
        b.finished.connect(self._on_finished)
        b.error.connect(self._on_error)
        b.confirm_request.connect(self._on_confirm_request)
        b.ask_request.connect(self._on_ask_request)

    # ------------------------------------------------------------------
    # config <-> widgets
    # ------------------------------------------------------------------
    def _apply_config_to_widgets(self) -> None:
        self.base_url_edit.setText(self.config.base_url)
        self.api_key_edit.setText(self.config.api_key)
        if self.config.model:
            self.model_combo.setEditText(self.config.model)
        self.temp_edit.setText(str(self.config.temperature))
        self.max_tokens_edit.setText(str(self.config.max_tokens))
        self.top_p_edit.setText(str(self.config.top_p))
        self.min_p_edit.setText(str(self.config.min_p))
        self.top_k_edit.setText(str(self.config.top_k))
        self.repeat_penalty_edit.setText(str(self.config.repetition_penalty))
        self.timeout_edit.setText(str(self.config.request_timeout))
        self.workspace_edit.setText(self.config.workspace)
        self.knowledge_base_edit.setText(self.config.knowledge_base)

    def _gather_config_from_widgets(self) -> None:
        self.config.base_url = self.base_url_edit.text().strip()
        self.config.api_key = self.api_key_edit.text().strip()
        self.config.model = self.model_combo.currentText().strip()
        try:
            self.config.temperature = float(self.temp_edit.text().strip() or "0.7")
        except ValueError:
            self.config.temperature = 0.7
        try:
            self.config.max_tokens = int(self.max_tokens_edit.text().strip() or "32768")
        except ValueError:
            self.config.max_tokens = 32768
        # Keep max_context_tokens in sync with max_tokens: the shrink threshold
        # should track the model's actual context window. If the user has
        # explicitly set a different max_context_tokens via config.json, only
        # override when it was smaller than the new max_tokens (i.e. the old
        # 4096 budget that's now too small).
        if self.config.max_context_tokens < self.config.max_tokens:
            self.config.max_context_tokens = self.config.max_tokens
        try:
            self.config.top_p = float(self.top_p_edit.text().strip() or "0.95")
        except ValueError:
            self.config.top_p = 0.95
        try:
            self.config.min_p = float(self.min_p_edit.text().strip() or "0.05")
        except ValueError:
            self.config.min_p = 0.05
        try:
            self.config.top_k = int(self.top_k_edit.text().strip() or "20")
        except ValueError:
            self.config.top_k = 20
        try:
            self.config.repetition_penalty = float(self.repeat_penalty_edit.text().strip() or "1.0")
        except ValueError:
            self.config.repetition_penalty = 1.0
        try:
            self.config.request_timeout = float(self.timeout_edit.text().strip() or "120")
        except ValueError:
            self.config.request_timeout = 120.0
        ws = self.workspace_edit.text().strip()
        if ws:
            self.config.workspace = ws
        self.config.ensure_workspace()
        kb = self.knowledge_base_edit.text().strip()
        if not kb:
            # Default to <workspace>/knowledge_base so RAG tools work out of
            # the box even when the user hasn't explicitly set a path.
            kb = os.path.join(self.config.workspace, "knowledge_base")
        self.config.knowledge_base = kb

    def _rebuild_agent(self) -> None:
        self._gather_config_from_widgets()
        # Shut down the old agent's resources (MCP subprocesses, httpx client)
        # before replacing it — otherwise each "应用配置" click leaks a set of
        # MCP server subprocesses and an open httpx connection.
        old_agent = getattr(self, "_agent", None)
        if old_agent is not None:
            try:
                old_agent.tools.shutdown()
            except Exception:
                pass
        old_llm = getattr(self, "_llm", None)
        if old_llm is not None:
            try:
                old_llm.close()
            except Exception:
                pass
        try:
            self._llm = LLMClient(
                base_url=self.config.base_url, api_key=self.config.api_key,
                model=self.config.model, timeout=self.config.request_timeout,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                min_p=self.config.min_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
            )
            self._agent = Agent(llm=self._llm, config=self.config, callbacks=self._callbacks)
            self._agent.max_iterations = self.config.max_iterations
        except Exception as e:
            self._on_error(f"无法创建 agent: {e}")

    # ------------------------------------------------------------------
    # config actions
    # ------------------------------------------------------------------
    def _on_apply_config(self) -> None:
        # Guard: if the agent is currently running, rebuilding would shut down
        # the old agent's tools/MCP/httpx mid-execution, causing the running
        # AgentWorker to crash when its next tool call hits destroyed tools.
        # The user must wait for the current turn to finish (or stop it first).
        if self._busy:
            QMessageBox.warning(
                self, "正在运行",
                "Agent 正在运行，请先点击「终止对话」停止当前任务，再应用配置。",
            )
            return
        self.config.save(CONFIG_PATH)
        self._rebuild_agent()
        self._append_log(f"[cfg] applied: {self.config.base_url} model={self.config.model}")
        self.status_label.setText("配置已应用")

    def _on_pick_workspace(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择工作目录", self.config.workspace)
        if d:
            self.workspace_edit.setText(d)

    def _on_pick_knowledge_base(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择知识库目录", self.config.knowledge_base or self.config.workspace)
        if d:
            self.knowledge_base_edit.setText(d)

    def _on_sync_knowledge(self, force: bool = False) -> None:
        if self._sync_worker is not None and self._sync_worker.isRunning():
            logger.warning("Sync already running, ignoring duplicate request")
            return
        if self._deps_worker is not None and self._deps_worker.isRunning():
            logger.warning("Deps check already running, ignoring duplicate request")
            return
        kb = self.knowledge_base_edit.text().strip()
        ws = self.workspace_edit.text().strip() or os.path.expanduser("~/.base-agent/workspace")
        if not kb:
            QMessageBox.warning(self, "缺少知识库", "请先选择知识库目录")
            return
        logger.info("Knowledge sync triggered: kb=%s workspace=%s force=%s", kb, ws, force)
        # Step 1: check dependencies before starting sync
        self.sync_knowledge_btn.setEnabled(False)
        self.sync_knowledge_btn.setText("检查依赖…")
        # Enable stop-sync during the deps-check phase too so the user can
        # abort the whole pipeline before sync even starts.
        self.stop_sync_btn.setEnabled(True)
        self.status_label.setText("正在检查依赖…")
        self.progress.setRange(0, 0)
        self._pending_sync_ws = ws
        self._pending_sync_kb = kb
        self._pending_sync_force = force
        logger.info("UI: starting sync — kb=%s force=%s", kb, force)
        self._append_log("[rag] ────────── 开始知识库同步 ──────────")
        self._deps_worker = InstallDepsWorker(kb)
        self._deps_worker.progress.connect(self._on_deps_progress)
        self._deps_worker.finished.connect(self._on_deps_finished)
        self._deps_worker.finished.connect(self._deps_worker.deleteLater)
        self._deps_worker.start()

    def _on_deps_progress(self, msg: str) -> None:
        self.status_label.setText(msg)
        self._append_log(f"[deps] {msg}")

    def _on_deps_finished(self, report) -> None:
        self._deps_worker = None
        # 检查用户是否在依赖检查期间点击了停止同步
        if self._deps_cancelled:
            self._deps_cancelled = False
            logger.info("Deps check finished but user cancelled — aborting sync")
            self.sync_knowledge_btn.setEnabled(True)
            self.sync_knowledge_btn.setText("同步知识")
            self.stop_sync_btn.setEnabled(False)
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.status_label.setText("已取消同步")
            self._append_log("[rag] 已取消同步")
            return
        if report is not None and report.has_blocking:
            # Required deps still missing after auto-install attempt
            logger.warning("Deps check: blocking dependencies missing, aborting sync")
            self.sync_knowledge_btn.setEnabled(True)
            self.sync_knowledge_btn.setText("同步知识")
            self.stop_sync_btn.setEnabled(False)
            self.progress.setRange(0, 1)
            self.status_label.setText("依赖缺失，无法同步")
            QMessageBox.critical(
                self, "依赖缺失",
                f"以下必要依赖未能自动安装，请手动安装后重试：\n\n{report.summary()}"
            )
            return

        # Step 2: start sync
        ws = getattr(self, "_pending_sync_ws", "")
        kb = getattr(self, "_pending_sync_kb", "")
        force = getattr(self, "_pending_sync_force", False)
        if not ws or not kb:
            self.sync_knowledge_btn.setEnabled(True)
            self.sync_knowledge_btn.setText("同步知识")
            self.stop_sync_btn.setEnabled(False)
            return
        self.sync_knowledge_btn.setText("同步中…")
        self.status_label.setText("正在同步知识库…")
        self.progress.setRange(0, 0)
        # stop_sync_btn stays enabled (it was enabled in _on_sync_knowledge)
        # so the user can cancel mid-sync.
        self._append_log("[rag] 依赖检查完成，启动同步引擎...")
        # 同步期间临时禁用 QSignalLogHandler 的 RAG 日志过滤，
        # 避免 QSignalLogHandler 和 _poll_sync_logs 同时操作 log_view
        # 导致的并发崩溃。RAG 日志在同步期间完全由 _log_buffer + QTimer 处理。
        if hasattr(self, '_log_handler') and self._log_handler is not None:
            self._log_handler._sync_active = True
        self._sync_worker = SyncKnowledgeWorker(
            ws, kb,
            embedding_model=self.config.rag_embedding_model,
            force=force,
        )
        self._sync_worker.sync_finished.connect(self._on_sync_finished)
        self._sync_worker.sync_failed.connect(self._on_sync_failed)
        self._sync_worker.sync_progress.connect(self._on_sync_progress)
        self._sync_worker.finished.connect(self._on_sync_thread_finished)
        self._sync_worker.finished.connect(self._sync_worker.deleteLater)
        self._sync_worker.start()
        # 并行启动章节摘要 worker（独立于同步任务与对话，断点续传）。
        self._start_outline_summarize(ws, kb)
        # 启动日志轮询定时器：每 100ms 消费 _log_buffer，实时显示同步日志
        self._log_poll_timer.start()

    def _start_outline_summarize(self, ws: str, kb: str) -> None:
        """并行启动独立的章节摘要 worker（LLM 补强缩略版本）。

        与知识同步、Agent 对话完全隔离：拥有自己的 LLMClient + RAGEngine，
        断点续传未完成的章节摘要。若已有 worker 在运行则跳过（避免重复）。
        """
        if self._outline_worker is not None and self._outline_worker.isRunning():
            logger.info("Outline summarize already running, skip duplicate start")
            return
        try:
            llm_config = {
                "base_url": self.config.base_url,
                "api_key": self.config.api_key,
                "model": self.config.model,
                "timeout": self.config.request_timeout,
                "max_tokens": self.config.max_tokens,
            }
            self._append_log(f"[outline] 启动章节摘要 worker (ws={ws}, kb={kb}, model={llm_config['model']})")
            self._outline_worker = OutlineSummarizeWorker(
                ws, kb,
                embedding_model=self.config.rag_embedding_model,
                llm_config=llm_config,
            )
            self._outline_worker.outline_progress.connect(self._on_outline_progress)
            self._outline_worker.outline_finished.connect(self._on_outline_finished)
            self._outline_worker.outline_failed.connect(self._on_outline_failed)
            self._outline_worker.log.connect(self._on_outline_log)
            self._outline_worker.finished.connect(self._outline_worker.deleteLater)
            self._outline_worker.start()
            logger.info("UI: outline summarize worker started (parallel to sync)")
        except Exception as e:
            logger.warning("UI: start outline summarize failed: %s", e)
            self._append_log(f"[outline] 启动失败: {e}")

    def _on_outline_log(self, msg: str) -> None:
        self._append_log(f"[outline] {msg}")

    def _on_outline_progress(self, done: int, total: int, current: str) -> None:
        self._append_log(f"[outline] 章节摘要进度: {done}/{total}")

    def _on_outline_finished(self) -> None:
        self._append_log("[outline] 章节摘要任务结束")
        self._outline_worker = None

    def _on_outline_failed(self, error: str) -> None:
        self._append_log(f"[outline] 章节摘要失败: {error}")
        self._outline_worker = None

    def _on_sync_finished(self, stats: dict) -> None:
        logger.info("UI: _on_sync_finished called")
        self._log_poll_timer.stop()
        # 恢复 QSignalLogHandler 的 RAG 日志过滤
        if hasattr(self, '_log_handler') and self._log_handler is not None:
            self._log_handler._sync_active = False
        # 最后消费一次 _log_buffer，确保残余日志被显示
        self._poll_sync_logs()
        try:
            self.sync_knowledge_btn.setEnabled(True)
            self.sync_knowledge_btn.setText("同步知识")
            self.stop_sync_btn.setEnabled(False)
            # Fully reset the progress bar: both range AND value. Without
            # setValue(0), a stale value from the last _on_sync_progress
            # call can persist (clamped to the new max), making the bar
            # look "stuck" at 100%.
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            # 输出诊断信息：确认信号已正确接收
            if hasattr(self, '_log_handler') and self._log_handler is not None:
                logger.info("UI: sync_finished received, emit_count=%d emit_errors=%d",
                            self._log_handler._emit_count, self._log_handler._emit_errors)
            self._sync_worker = None
            # 刷新 agent 的 RAG 引擎表句柄，让下一次搜索看到新写入的数据。
            # LanceDB 表对象在打开时获取版本快照，同步后不重新打开会看到旧数据。
            if self._agent is not None:
                try:
                    self._agent.tools.reload_rag()
                except Exception as e:
                    logger.warning("UI: reload_rag after sync: %s", e)
            # 防御：stats 可能不是 dict（极端情况下为 None）
            if not isinstance(stats, dict):
                stats = {}
            if stats.get("cancelled"):
                self.status_label.setText("同步已取消")
                self._append_log("[rag] 同步已用户取消")
                return
            self.status_label.setText("知识库同步完成")
            files = stats.get("files_found", 0)
            extracted = stats.get("files_extracted", 0)
            skipped = stats.get("files_skipped", 0)
            deleted = stats.get("files_deleted", 0)
            chunks = stats.get("chunks", 0)
            errors = stats.get("errors", [])
            # Show incremental stats so the user sees what was actually processed.
            if extracted == 0 and skipped > 0:
                summary = f"知识库已是最新（跳过 {skipped} 个未修改文件）\n\n"
            else:
                summary = (
                    f"知识库同步完成！\n\n"
                    f"扫描文件：{files} 个\n"
                    f"新增/更新：{extracted} 个   跳过：{skipped} 个\n"
                )
                if deleted > 0:
                    summary += f"已清理删除文件：{deleted} 个\n"
                summary += f"向量切片：{chunks} 个\n"
            msg = summary
            if errors:
                msg += f"\n⚠ {len(errors)} 个文件提取失败：\n"
                for e in errors[:5]:
                    msg += f"  - {e}\n"
                if len(errors) > 5:
                    msg += f"  …（还有 {len(errors) - 5} 个）\n"
            msg += "\n后续对话将自动参考知识库内容进行回答。"
            QMessageBox.information(self, "同步完成", msg)
        except Exception as e:
            logger.error("UI: _on_sync_finished error: %s", e, exc_info=True)
            # 确保按钮状态恢复，即使弹窗失败
            try:
                self.sync_knowledge_btn.setEnabled(True)
                self.sync_knowledge_btn.setText("同步知识")
                self.stop_sync_btn.setEnabled(False)
                self.progress.setRange(0, 1)
                self.progress.setValue(0)
                self._sync_worker = None
                self.status_label.setText("同步完成（显示结果时出错）")
            except Exception:
                pass

    def _on_sync_thread_finished(self) -> None:
        """兜底检查：QThread.finished 信号在 run() 返回后自动发射。

        如果 sync_finished / sync_failed 信号已正常处理，此时按钮应该已经
        恢复为"同步知识"状态。如果按钮仍处于同步相关状态，说明同步结束信号
        未被正确处理，需要强制恢复 UI 状态以避免用户界面永久卡住。

        注意：不检查 self._sync_worker is not None，因为 _on_sync_finished
        可能在异常路径中已将其设为 None，但按钮状态未恢复。
        """
        self._log_poll_timer.stop()
        if hasattr(self, '_log_handler') and self._log_handler is not None:
            self._log_handler._sync_active = False
        self._poll_sync_logs()
        btn_text = self.sync_knowledge_btn.text()
        if btn_text in ("同步中…", "检查依赖…", "正在同步知识库…"):
            logger.warning(
                "UI: sync thread finished but button still in '%s' state. "
                "sync_finished/sync_failed signal may have been lost. "
                "Forcing UI reset.",
                btn_text,
            )
            try:
                self.sync_knowledge_btn.setEnabled(True)
                self.sync_knowledge_btn.setText("同步知识")
                self.stop_sync_btn.setEnabled(False)
                self.progress.setRange(0, 1)
                self.progress.setValue(0)
                self.status_label.setText("知识库同步完成（状态已自动恢复）")
            except Exception:
                pass
        # 清理 worker 引用，防止后续误判
        if self._sync_worker is not None:
            self._sync_worker = None

    def _on_sync_failed(self, error: str) -> None:
        logger.info("UI: _on_sync_failed called: %s", error[:200])
        self._log_poll_timer.stop()
        if hasattr(self, '_log_handler') and self._log_handler is not None:
            self._log_handler._sync_active = False
        self._poll_sync_logs()
        if hasattr(self, '_log_handler') and self._log_handler is not None:
            logger.info("UI: sync_failed received, emit_count=%d emit_errors=%d",
                        self._log_handler._emit_count, self._log_handler._emit_errors)
        try:
            self.sync_knowledge_btn.setEnabled(True)
            self.sync_knowledge_btn.setText("同步知识")
            self.stop_sync_btn.setEnabled(False)
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.status_label.setText("知识库同步失败")
            self._sync_worker = None
            QMessageBox.critical(self, "同步失败", f"知识库同步失败：\n{error}")
        except Exception as e:
            logger.error("UI: _on_sync_failed error: %s", e, exc_info=True)
            try:
                self.sync_knowledge_btn.setEnabled(True)
                self.sync_knowledge_btn.setText("同步知识")
                self.stop_sync_btn.setEnabled(False)
                self.progress.setRange(0, 1)
                self.progress.setValue(0)
                self._sync_worker = None
            except Exception:
                pass

    def _on_fetch_models(self) -> None:
        base_url = self.base_url_edit.text().strip()
        api_key = self.api_key_edit.text().strip()
        if not base_url:
            QMessageBox.warning(self, "缺少配置", "请先填写 Base URL")
            return
        self.fetch_btn.setEnabled(False)
        self.status_label.setText("正在获取模型列表…")
        self.progress.setRange(0, 0)
        self._fetch_worker = FetchModelsWorker(base_url, api_key)
        self._fetch_worker.fetched.connect(self._on_models_fetched)
        self._fetch_worker.failed.connect(self._on_fetch_failed)
        self._fetch_worker.finished.connect(self._fetch_worker.deleteLater)
        self._fetch_worker.start()

    def _on_models_fetched(self, models: list) -> None:
        self.fetch_btn.setEnabled(True)
        self.progress.setRange(0, 1)
        current = self.model_combo.currentText().strip()
        self.model_combo.clear()
        self.model_combo.addItems(models)
        if current in models:
            self.model_combo.setCurrentText(current)
        elif models:
            self.model_combo.setCurrentIndex(0)
        self.status_label.setText(f"获取到 {len(models)} 个模型")

    def _on_fetch_failed(self, msg: str) -> None:
        self.fetch_btn.setEnabled(True)
        self.progress.setRange(0, 1)
        self.status_label.setText("获取模型失败")
        QMessageBox.warning(self, "获取模型失败", msg)

    # ------------------------------------------------------------------
    # chat send / stop
    # ------------------------------------------------------------------
    def _on_send(self) -> None:
        if self._busy:
            return
        text = self.send_editor.toPlainText().strip()
        if not text:
            return
        if self._agent is None:
            self._rebuild_agent()
        if self._agent is None:
            QMessageBox.warning(self, "未就绪", "Agent 未初始化，请先应用配置。")
            return

        self._append_user(text)
        self.send_editor.clear()
        self._busy = True
        self._set_controls_busy(True)
        self.think_view.clear()
        self.speed_label.setText("0.0 tok/s")
        self._callbacks._cancelled = False
        self._worker = AgentWorker(self._agent, self._callbacks)
        self._worker.set_input(text)
        # Let Qt clean up the QThread asynchronously once run() returns, instead
        # of calling QThread.wait() from a slot (which can deadlock a nested
        # event loop). `_on_finished` is invoked via bridge.finished first.
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_stop(self) -> None:
        # 终止当前 agent 对话（不再处理同步取消，同步取消由 _on_stop_sync 负责）
        self._callbacks.cancel()
        self.status_label.setText("已请求终止对话（将在当前工具完成后生效）")

    def _on_stop_sync(self) -> None:
        """停止知识库同步：在当前文件处理完成后协作式取消。

        sync_worker.cancel() 设置 RAGEngine 内部的取消标志，
        ingest 循环在处理完当前文件后检查该标志并退出。
        """
        if self._sync_worker is not None and self._sync_worker.isRunning():
            logger.info("User requested sync cancellation")
            self._sync_worker.cancel()
            self.stop_sync_btn.setEnabled(False)  # 防止重复点击
            self.status_label.setText("已请求停止同步（将在当前文件完成后生效）")
            self._append_log("[rag] 用户请求停止同步")
            return
        # deps 检查阶段：记录取消意图，deps 完成后不启动同步。
        if self._deps_worker is not None and self._deps_worker.isRunning():
            logger.info("User requested stop during deps check — will abort after deps")
            self._deps_worker.cancel()
            self._deps_cancelled = True
            self.stop_sync_btn.setEnabled(False)
            self.status_label.setText("正在取消…")
            self._append_log("[rag] 用户请求停止同步（依赖检查完成后取消）")
            return

    def _on_sync_progress(self, done: int, total: int, current: str) -> None:
        """同步进度反馈。日志刷新由 _poll_sync_logs 定时器独立处理。"""
        try:
            if done < 0:
                return
            if total > 0:
                self.progress.setRange(0, total)
                self.progress.setValue(done)
                pct = done * 100 // total
                self.status_label.setText(f"同步中 {done}/{total} ({pct}%) — {current}")
            else:
                self.progress.setRange(0, 0)
                self.status_label.setText(f"同步中 — {current}")
        except Exception:
            pass

    def _poll_sync_logs(self) -> None:
        """定时轮询 sync_worker._log_buffer，把新日志追加到 log_view。

        绕过 Qt queued signal 机制——ONNX Runtime 在 embedding 期间持有
        GIL，sync_progress 信号虽然是 queued 但主线程可能无法及时处理。
        用 QTimer 直接在主线程事件循环中轮询 buffer，只要 GIL 短暂释放
        （_embed_chunks 中的 sleep(0.01)），定时器回调就能执行。

        注意：本方法只负责日志区域的输出，不更新底部状态栏。
        状态栏由 _on_sync_progress 根据进度回调单独更新，避免重复。
        """
        try:
            worker = self._sync_worker
            if worker is None:
                return
            with worker._log_buffer_lock:
                if not worker._log_buffer:
                    return
                buf = worker._log_buffer
                worker._log_buffer = []
            # 批量插入：用 QTextCursor 一次性插入所有行，比逐行
            # appendPlainText 快得多（后者每次触发一次重绘）。
            cursor = self.log_view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText("\n" + "\n".join(buf))
            self._scroll_to_bottom(self.log_view)
        except Exception:
            pass

    def _set_controls_busy(self, busy: bool) -> None:
        self.send_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)
        self.connect_btn.setEnabled(not busy)
        self.fetch_btn.setEnabled(not busy)
        self.clear_btn.setEnabled(not busy)
        self.workspace_btn.setEnabled(not busy)
        self.knowledge_base_btn.setEnabled(not busy)
        self.sync_knowledge_btn.setEnabled(not busy)
        self.progress.setRange(0, 0 if busy else 1)

    # ------------------------------------------------------------------
    # streaming slots
    # ------------------------------------------------------------------
    def _append_user(self, text: str) -> None:
        self._append_chat(f'<p><b class="u">你</b>: {self._escape(text)}</p>')

    def _on_content(self, text: str) -> None:
        text = _sanitize_stream_text(text)
        if not text:
            return
        if not self._streaming_assistant:
            self._append_chat('<p><b class="a">助手</b>: </p>')
            self._streaming_assistant = True
        # Always append at the document end: grab the widget cursor, move the
        # *copy* to End, insert, then write it back so the widget's real cursor
        # is at the end (otherwise ensureCursorVisible scrolls to the wrong place
        # and a later drift could insert into the middle, overlapping text).
        cursor = self.chat_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        # 用 insertText 而非 insertHtml：HTML 的空白折叠规则会丢掉 chunk 之间的
        # 前导/后随空格（"Hello" + " World" 会被渲染成 "HelloWorld"）。
        # insertText 保留原始空格，并把 \n 当作真实换行处理。
        # 思考过程（_on_reasoning）也用 insertPlainText，所以一直显示正常。
        cursor.insertText(text)
        self.chat_view.setTextCursor(cursor)
        self._scroll_to_bottom(self.chat_view)

    def _on_reasoning(self, text: str) -> None:
        text = _sanitize_stream_text(text)
        if not text:
            return
        # Move the widget cursor to the end before inserting. Without this,
        # insertPlainText writes at whatever position the cursor happens to be
        # (e.g. after the user clicked the view to read earlier output), which
        # interleaves new text with existing lines -> overlap.
        cursor = self.think_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.think_view.setTextCursor(cursor)
        self.think_view.insertPlainText(text)
        self._scroll_to_bottom(self.think_view)

    def _on_tool_start(self, name: str, args: Any) -> None:
        import json as _json
        self._streaming_assistant = False  # close any streaming assistant block
        try:
            pretty = _json.dumps(args, ensure_ascii=False)
        except Exception:
            pretty = str(args)
        self._append_chat(f'<p class="t">🔧 <b>{self._escape(name)}</b>({self._escape(pretty)})…</p>')
        self._append_log(f"[tool-start] {name} {pretty}")

    def _on_tool_end(self, name: str, result: Any) -> None:
        out = result.to_message() if hasattr(result, "to_message") else str(result)
        snippet = out if len(out) < 400 else out[:400] + " …"
        cls = "t" if getattr(result, "success", True) else "sys"
        self._append_chat(f'<p class="{cls}">↳ <b>{self._escape(name)}</b> → {self._escape(snippet)}</p>')
        self._append_log(f"[tool-end] {name} success={getattr(result,'success',True)} :: {snippet}")

    def _on_log(self, line: str) -> None:
        self._append_log(line)

    def _on_usage(self, usage: dict) -> None:
        # Don't overwrite _total_tokens here — agent.run() already maintains
        # a running total that includes estimated reasoning tokens when the API
        # underreports. _on_speed (called immediately after) carries the
        # authoritative total. Overwriting here would cause the displayed total
        # to jump between API total (no estimate) and agent total (with estimate).
        pass

    def _on_speed(self, total_tokens: int, speed: float) -> None:
        self._total_tokens = total_tokens
        self.token_label.setText(f"总计 {total_tokens} tokens")
        self.speed_label.setText(f"{speed:.1f} tok/s")

    def _on_skill_suggested(self, skill) -> None:
        """Cache the skill suggestion for showing AFTER the run finishes.

        Showing a modal dialog here would block the UI event loop during
        streaming, preventing content/reasoning chunks from rendering.
        """
        if not isinstance(skill, Skill):
            return
        if skill.name == "_suggested":
            self._pending_skill = skill

    def _show_pending_skill_dialog(self) -> None:
        """Show the skill-suggestion dialog (called after _on_finished)."""
        skill = self._pending_skill
        self._pending_skill = None
        if skill is None or self._agent is None:
            return
        reply = QMessageBox.question(
            self, "固化为技能",
            f"检测到你多次执行类似任务：\n{skill.prompt[:200]}\n\n是否保存为技能以便下次直接执行？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            name, ok = QInputDialog.getText(self, "技能名称", "技能名称：", text="my-skill")
            if ok and name.strip():
                self._agent.skills.create_skill(name.strip(), skill.keywords, skill.prompt)
                # Reload so the worker thread's SkillManager picks up the new
                # skill immediately — otherwise match() in _system_prompt would
                # still see the old (pre-save) in-memory cache and keep
                # re-suggesting the same intent every turn.
                self._agent.skills.reload()
                self._append_log(f"[skill] saved '{name}'")

    def _on_finished(self) -> None:
        if not self._busy:  # guard against double-call from _on_error + bridge.finished
            return
        self._busy = False
        self._streaming_assistant = False
        self._set_controls_busy(False)
        self.status_label.setText("就绪")
        # Join the worker thread before dropping our reference. bridge.finished
        # is emitted from the worker's `finally`, so run() has already returned
        # and wait() returns near-instantly -- but without it, dropping the last
        # Python reference can GC the QThread while its OS thread is still
        # winding down, which Qt aborts as "destroyed while still running".
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.wait(2000)
        # Now that streaming is done and the worker thread has joined, it's
        # safe to show the skill-suggestion modal dialog without blocking
        # the UI event loop during streaming.
        self._show_pending_skill_dialog()

    def _on_error(self, msg: str) -> None:
        self._append_chat(f'<p class="sys">⚠ {self._escape(msg)}</p>')
        self._append_log(f"[error] {msg}")
        # bridge.finished signal will call _on_finished from the worker's finally block

    # ------------------------------------------------------------------
    # blocking dialogs (run in UI thread, unblock worker)
    # ------------------------------------------------------------------
    def _on_confirm_request(self, message: str) -> None:
        reply = QMessageBox.question(self, "确认操作", message,
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        self._callbacks.set_confirm_result(reply == QMessageBox.StandardButton.Yes)

    def _on_ask_request(self, prompt: str) -> None:
        text, ok = QInputDialog.getText(self, "需要你的输入", prompt)
        self._callbacks.set_ask_result(text if ok else None)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _setup_log_bridge(self) -> None:
        """将 RAG/sync 模块的 logger 输出桥接到 UI 的 log_view。

        通过 QSignalLogHandler 把 Python logging 消息安全地转发到
        UI 线程，让用户可以在日志面板中实时看到知识同步的详细进度。

        实现方式：将 handler 注册到 root logger，通过 Filter 只放行
        匹配 _RAG_LOG_PREFIXES 的日志记录。这样做的好处是：
        1. 不依赖 Python logging 层级传播（某些运行时环境下子 logger
           的 propagate 可能被意外关闭）
        2. 后续动态创建的 logger（如 RAG 模块的延迟导入）也会被覆盖
        3. 线程安全：QSignalLogHandler 通过 pyqtSignal 将日志从工作
           线程安全地转发到 UI 线程
        """
        global _LOG_HANDLER
        # 防止重复添加：先移除旧的 handler。
        if _LOG_HANDLER is not None:
            root = logging.getLogger()
            try:
                root.removeHandler(_LOG_HANDLER)
            except Exception:
                pass
            # Remove from _handlerList in-place (see QSignalLogHandler.close
            # for the slice-assignment rationale) then close the old handler.
            try:
                import logging as _logging
                _logging._handlerList[:] = [
                    wr for wr in _logging._handlerList
                    if wr() is not _LOG_HANDLER
                ]
            except Exception:
                pass
            try:
                _LOG_HANDLER.close()
            except Exception:
                pass
        self._log_handler = QSignalLogHandler()
        self._log_handler.message_received.connect(self._append_log)
        self._log_handler.setLevel(logging.DEBUG)

        # 添加 Filter：只放行 RAG 相关模块的日志
        class _RagLogFilter(logging.Filter):
            def filter(self, record):
                return any(
                    record.name.startswith(p) for p in _RAG_LOG_PREFIXES
                )

        self._log_handler.addFilter(_RagLogFilter())

        # 将 handler 挂到 root logger，配合 Filter 实现精确过滤。
        # 同时将相关父 logger 的 level 设为 DEBUG，确保子 logger 的
        # INFO 级别日志能通过有效级别检查。
        root = logging.getLogger()
        root.addHandler(self._log_handler)
        for prefix in _RAG_LOG_PREFIXES:
            logging.getLogger(prefix).setLevel(logging.DEBUG)
        # 同时设置已知子 logger 的 level，防止子 logger 从 root 继承
        # WARNING(30) 导致 INFO(20) 日志在传播到 handler 前被拦截。
        for _name in _RAG_SUB_LOGGERS:
            logging.getLogger(_name).setLevel(logging.DEBUG)

        # 保存到模块级变量，供 closeEvent 清理使用。
        _LOG_HANDLER = self._log_handler
        # 验证日志桥接：这条消息应同时出现在终端和 UI 日志面板中。
        # 如果终端能看到但 UI 面板没有，说明信号槽或 Filter 有问题。
        # 如果 UI 面板能看到，说明桥接完全正常。
        logger.info("Log bridge: RAG/sync logs forwarded to UI panel")
        # 直接追加一条测试消息到 UI 日志面板（绕过 logger 桥接），
        # 验证 log_view 本身可以正常显示文本。
        # 日志桥接已就绪，不输出提示信息到日志区域

    def _append_chat(self, html: str) -> None:
        self.chat_view.append(html)
        self._scroll_to_bottom(self.chat_view)

    def _append_log(self, line: str) -> None:
        self.log_view.appendPlainText(line)
        self._scroll_to_bottom(self.log_view)

    @staticmethod
    def _scroll_to_bottom(widget) -> None:
        """Pin the widget's viewport to the bottom so the latest content is
        always visible.

        ``ensureCursorVisible()`` alone only scrolls *just enough* to reveal the
        cursor; if the user has scrolled up to read earlier output, that leaves
        the view somewhere in the middle rather than at the very bottom. We move
        the cursor to the document end, ask Qt to reveal it, and then explicitly
        snap the vertical scrollbar to its maximum as a belt-and-suspenders
        guarantee that the newest line is on screen.
        """
        cursor = widget.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        widget.setTextCursor(cursor)
        widget.ensureCursorVisible()
        sb = widget.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_clear_chat(self) -> None:
        self.chat_view.clear()
        self.think_view.clear()
        self._streaming_assistant = False
        if self._agent is not None:
            self._agent.reset()

    @staticmethod
    def _escape(text: str) -> str:
        import html as _html
        return _html.escape(text).replace("\n", "<br>")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Clean up background resources before the window closes.

        All QThread workers must be stopped here. If a worker thread is still
        running when the window's widgets are destroyed, the worker's signals
        fire into deleted C++ objects → ``RuntimeError: wrapped C/C++ object
        has been deleted``. For each worker we:
        1. Disconnect signals (so emit() becomes a no-op even if the thread
           sneaks in one last emit before we wait()).
        2. Request cooperative cancel (if the worker supports it).
        3. wait() with a timeout; if it doesn't finish, terminate() as a last
           resort — unsafe, but safer than the crash.
        """
        # --- 停止日志轮询定时器
        try:
            self._log_poll_timer.stop()
        except Exception:
            pass
        # --- AgentWorker: cancel the agent run and join the thread.
        # Without this, closing the window during an agent run leaves the
        # worker thread alive; its bridge.finished/error/content signals
        # will emit into the already-destroyed Bridge → RuntimeError.
        if self._worker is not None:
            try:
                self._worker.disconnect()
            except (TypeError, RuntimeError):
                pass
            # Cooperative cancel: set the _cancelled flag so the agent loop
            # exits after the current tool/chunk. This is faster than
            # terminate() and avoids corrupting tool state mid-execution.
            try:
                self._callbacks.cancel()
            except Exception:
                pass
            if self._worker.isRunning():
                self._worker.wait(3000)
                if self._worker.isRunning():
                    logger.warning("UI: agent worker did not finish in 3s, terminating")
                    self._worker.terminate()
                    self._worker.wait(2000)
            self._worker = None
        # --- FetchModelsWorker: a quick HTTP call, usually finishes fast.
        if self._fetch_worker is not None:
            try:
                self._fetch_worker.disconnect()
            except (TypeError, RuntimeError):
                pass
            if self._fetch_worker.isRunning():
                self._fetch_worker.wait(2000)
                if self._fetch_worker.isRunning():
                    self._fetch_worker.terminate()
                    self._fetch_worker.wait(1000)
            self._fetch_worker = None
        # --- SyncKnowledgeWorker: cooperative cancel (finishes current file).
        if self._sync_worker is not None:
            try:
                self._sync_worker.disconnect()
            except (TypeError, RuntimeError):
                pass
            if self._sync_worker.isRunning():
                self._sync_worker.cancel()
                self._sync_worker.wait(3000)
                if self._sync_worker.isRunning():
                    logger.warning("UI: sync worker did not finish in 3s, terminating")
                    self._sync_worker.terminate()
                    self._sync_worker.wait(2000)
            self._sync_worker = None
        # --- OutlineSummarizeWorker: cooperative cancel (finishes current chapter).
        if self._outline_worker is not None:
            try:
                self._outline_worker.disconnect()
            except (TypeError, RuntimeError):
                pass
            if self._outline_worker.isRunning():
                self._outline_worker.cancel()
                self._outline_worker.wait(3000)
                if self._outline_worker.isRunning():
                    logger.warning("UI: outline worker did not finish in 3s, terminating")
                    self._outline_worker.terminate()
                    self._outline_worker.wait(2000)
            self._outline_worker = None
        # --- InstallDepsWorker: no cancel mechanism, just wait.
        if self._deps_worker is not None:
            try:
                self._deps_worker.disconnect()
            except (TypeError, RuntimeError):
                pass
            if self._deps_worker.isRunning():
                self._deps_worker.wait(2000)
                if self._deps_worker.isRunning():
                    self._deps_worker.terminate()
                    self._deps_worker.wait(1000)
            self._deps_worker = None
        # --- Release agent resources: MCP subprocesses, httpx client, RAG.
        if self._agent is not None:
            try:
                self._agent.tools.shutdown()
            except Exception as e:
                logger.warning("UI: shutdown error: %s", e)
        if self._llm is not None:
            try:
                self._llm.close()
            except Exception as e:
                logger.warning("UI: llm close error: %s", e)

        # --- Remove the QSignalLogHandler from the root logger.
        # Without this, the handler (which holds a reference to the QPlainTextEdit
        # via its signal slot) keeps a C++ object reference alive after the widget
        # is destroyed. Any subsequent log call will emit message_received into
        # freed memory → RuntimeError.
        if _LOG_HANDLER is not None:
            root = logging.getLogger()
            try:
                root.removeHandler(_LOG_HANDLER)
            except Exception:
                pass
            # Remove the handler from logging._handlerList IN-PLACE.
            #
            # logging.shutdown() is registered via atexit and its signature is
            #   def shutdown(handlerList=_handlerList):
            # The default argument binds to the list object at function
            # definition time.  If we replace _handlerList with a new list
            # (_handlerList = [...]), the default argument still points to the
            # OLD list, so shutdown() would still see the deleted handler.
            # Using slice assignment (_handlerList[:] = [...]) mutates the
            # list IN-PLACE, so the default argument sees the updated list.
            #
            # Without this, atexit shutdown() calls getattr(h, 'flushOnClose')
            # on a QSignalLogHandler whose Qt C++ object is already deleted,
            # raising RuntimeError.
            import logging as _logging
            try:
                _logging._handlerList[:] = [
                    wr for wr in _logging._handlerList
                    if wr() is not _LOG_HANDLER
                ]
            except Exception:
                pass
            try:
                _LOG_HANDLER.close()
            except Exception:
                pass
        super().closeEvent(event)


def run() -> None:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    run()

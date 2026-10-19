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
    QStatusBar, QFileDialog, QCheckBox,
)

from ..agent import Agent, AgentCallbacks
from ..config import AgentConfig
from ..llm_client import LLMClient, LLMError
from ..model_backends import create_llm_backend, discover_local_models
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
    # 上下文被压缩的通知（summary, reason）——不转发的话用户只会觉得「失忆」。
    context_shrunk = pyqtSignal(str, str)


class BridgeCallbacks(AgentCallbacks):
    """AgentCallbacks that forwards everything through the Bridge."""

    def __init__(self, bridge: Bridge):
        self.bridge = bridge
        self._confirm_event = threading.Event()
        self._ask_event = threading.Event()
        self._lock = threading.Lock()
        # 默认「拒绝」而不是「同意」：确认弹窗用于 shell_run / code_run 等
        # 高风险操作，若对话框因任何原因没有回话（超时、事件循环被卡住、
        # 信号丢失），安全的一侧是拒绝执行，而不是默默放行。
        self._confirm_result: bool = False
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
    def on_context_shrunk(self, summary: str, reason: str) -> None:
        self.bridge.context_shrunk.emit(summary, reason)
    def on_finished(self) -> None:
        self.bridge.finished.emit()

    # blocking dialogs (called from worker thread)
    # 等待用户回话的最长时间（秒）。超过则视为未获授权 → 拒绝 / 无回答。
    _DIALOG_TIMEOUT = 600.0

    def confirm(self, message: str) -> bool:
        if self._cancelled:
            return False
        self._confirm_event.clear()
        with self._lock:
            self._confirm_result = False
        self.bridge.confirm_request.emit(message)
        # Event.wait() 返回 False 表示超时——此时绝不能沿用默认/上一次结果，
        # 否则高风险操作会在用户没点确认的情况下被执行。
        if not self._confirm_event.wait(timeout=self._DIALOG_TIMEOUT):
            return False
        with self._lock:
            return self._confirm_result

    def ask_user(self, prompt: str) -> Optional[str]:
        if self._cancelled:
            return None
        self._ask_event.clear()
        with self._lock:
            self._ask_result = None
        self.bridge.ask_request.emit(prompt)
        if not self._ask_event.wait(timeout=self._DIALOG_TIMEOUT):
            return None
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

    def reset(self) -> None:
        """Clear per-run state so the next turn starts from a clean slate.

        Without this, ``_cancelled`` stays True after 「终止对话」 and every
        later confirm/ask would be auto-declined; the stale ``_ask_result``
        would also be returned again for the next question.
        """
        self._cancelled = False
        with self._lock:
            self._confirm_result = False
            self._ask_result = None
        self._confirm_event.clear()
        self._ask_event.clear()

    def is_cancelled(self) -> bool:
        """供 Agent.run() 在流式/迭代/工具边界轮询（协作式取消）。"""
        return self._cancelled

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
        client = None
        try:
            client = LLMClient(base_url=self._base_url, api_key=self._api_key, model="",
                               timeout=30.0, max_tokens=1)
            models = client.list_models()
            # 先取结果再 emit：emit 之后 close 若抛异常，错误会出现在
            # finished 之后无法显示。
            client.close()
            client = None
            self.fetched.emit(models)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
        finally:
            # close 不放进 try 的正常路径：list_models 对 4xx/5xx、网络错误
            # 都会抛异常，旧实现此时跳过了 close，httpx 连接池（socket +
            # 线程池）依赖 GC 兜底——每次「获取模型失败」泄漏一个。
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass


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
                 force: bool = False, rerank_model: str = "", pdf_engine: str = "docling",
                 model_options: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._workspace = workspace
        self._knowledge_base = knowledge_base
        self._embedding_model = embedding_model
        self._rerank_model = rerank_model
        self._pdf_engine = pdf_engine or "docling"
        self._model_options = dict(model_options or {})
        self._force = force
        self._engine: Optional[RAGEngine] = None
        # 日志缓冲区：子线程写入，主线程在 _poll_sync_logs 中消费。
        # 用 Lock 保护，避免多 worker 线程 append 与主线程清空之间的竞态。
        self._log_buffer: list = []
        self._log_buffer_lock = threading.Lock()
        # 缓冲区上限：主线程被模态对话框（确认/技能固化/同步结果）阻塞时
        # 事件循环不转，_poll_sync_logs 无法消费，而同步线程仍在持续 append
        # —— 无上限会在长同步 + 弹窗等待的组合下把内存吃光。超限后丢弃最旧的
        # 一半（而不是丢弃新日志），保证「最新进展」始终可见。
        self._log_buffer_max = 10000

    def _emit_log(self, msg: str) -> None:
        """将日志追加到缓冲区，由 _poll_sync_logs 消费写入日志面板。"""
        with self._log_buffer_lock:
            self._log_buffer.append(msg)
            if len(self._log_buffer) > self._log_buffer_max:
                del self._log_buffer[: self._log_buffer_max // 2]

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
            # 复用 _emit_log 而不是直接 append：缓冲区上限只在 _emit_log 里
            # 生效，绕过它会让这条路径无上限增长。
            self._emit_log(_tmp_handler.format(record))
        _tmp_handler.emit = _on_rag_log
        # 只挂父 logger；子 logger 保持 NOTSET（默认传播到父 logger）。
        _rag_loggers = [f"{_PKG_ROOT}.rag"]
        # 记住原 level：setLevel 是**全局副作用**，同步结束后必须恢复，否则
        # 之后该 logger 的 DEBUG 日志会静默丢失。
        _prev_levels = {}
        # 把 handler 暴露到 self：线程被 terminate() 时 Python 的 finally 不保证
        # 执行，handler 会永久残留（闭包强引用整个 worker），下次同步再挂一个
        # → 日志重复 N 遍。_on_sync_thread_finished 会无条件再摘一次。
        self._sync_tmp_handler = _tmp_handler
        self._sync_tmp_loggers = _rag_loggers
        for _name in _rag_loggers:
            _l = logging.getLogger(_name)
            _prev_levels[_name] = _l.level
            _l.setLevel(logging.INFO)
            _l.addHandler(_tmp_handler)
        emit_stats = None
        emit_error = None
        try:
            # rerank_model 空则不传 key：显式传 None 会覆盖 engine 的默认值
            # （INT8 v2-m3），导致 TextCrossEncoder(None) 崩溃（#131）。
            _engine_kwargs = dict(
                workspace=self._workspace,
                knowledge_base=self._knowledge_base,
                embedding_model=self._embedding_model,
            )
            _engine_kwargs.update(self._model_options)
            if self._rerank_model:
                # 非空 = 显式转发 config.rag_rerank_model，配置可覆盖。
                _engine_kwargs["rerank_model"] = self._rerank_model
            # PDF 主引擎（docling|marker，#134），配置可切换。
            _engine_kwargs["pdf_engine"] = self._pdf_engine
            self._engine = RAGEngine(**_engine_kwargs)
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
            for _name, _lvl in _prev_levels.items():
                try:
                    logging.getLogger(_name).setLevel(_lvl)
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


class InstallDepsWorker(QThread):
    """Background worker for checking and installing missing RAG dependencies.

    IMPORTANT: the completion signal is named ``deps_finished``, NOT
    ``finished``.  ``QThread`` already defines a built-in ``finished`` signal
    that is emitted *after* ``run()`` returns; declaring a custom signal with
    the same name shadows it, so anything connected to it (e.g.
    ``deleteLater``) fires from inside ``run()`` — deleting the QThread while
    it is still running.  See the same note on SyncKnowledgeWorker.
    """
    progress = pyqtSignal(str)
    deps_finished = pyqtSignal(object)  # DependencyReport

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
                self.deps_finished.emit(None)
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
                self.deps_finished.emit(None)
                return
            self.deps_finished.emit(report)
        except Exception as e:
            self.progress.emit(f"依赖检查异常: {e}")
            self.deps_finished.emit(None)


class BenchmarkWorker(QThread):
    """Runs the local-model benchmark off the UI thread.

    测速会加载 Embedding/Reranker 权重（MLX 约 137MB、ONNX 约 300MB）并做多轮
    推理，同步跑在 UI 线程上会冻住事件循环数十秒。信号名同样不覆写 Qt 内置的
    ``finished``（原因见 SyncKnowledgeWorker 注释）。
    """

    bench_progress = pyqtSignal(str)
    bench_finished = pyqtSignal(list)
    bench_failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        kind: str = "all",
        rounds: int = 3,
        threads: int = 4,
        embedding_batch_size: int = 8,
        rerank_batch_size: int = 16,
        embedding_path: str = "",
        embedding_model: str = "",
        rerank_path: str = "",
        rerank_model: str = "",
    ):
        super().__init__()
        self._kind = kind
        self._rounds = rounds
        self._threads = threads
        self._embedding_batch_size = embedding_batch_size
        self._rerank_batch_size = rerank_batch_size
        self._embedding_path = embedding_path
        self._embedding_model = embedding_model
        self._rerank_path = rerank_path
        self._rerank_model = rerank_model

    def run(self) -> None:  # type: ignore[override]
        try:
            from ..model_benchmark import (
                benchmark_embeddings,
                benchmark_rerankers,
                save_result,
            )

            results = []
            if self._kind in ("embedding", "all"):
                self.bench_progress.emit("正在测速 Embedding 后端（本地 MLX vs fastembed CPU）…")
                kwargs: Dict[str, Any] = {
                    "batch_size": self._embedding_batch_size,
                    "rounds": self._rounds,
                    "threads": self._threads,
                }
                if self._embedding_path:
                    kwargs["mlx_path"] = self._embedding_path
                if self._embedding_model:
                    kwargs["fastembed_model"] = self._embedding_model
                results.append(benchmark_embeddings(**kwargs))
            if self._kind in ("reranker", "all"):
                self.bench_progress.emit("正在测速 Reranker 后端（本地 ONNX vs fastembed CPU）…")
                kwargs = {
                    "batch_size": self._rerank_batch_size,
                    "rounds": self._rounds,
                    "threads": self._threads,
                }
                if self._rerank_path:
                    kwargs["onnx_path"] = self._rerank_path
                if self._rerank_model:
                    kwargs["fastembed_model"] = self._rerank_model
                results.append(benchmark_rerankers(**kwargs))
            for item in results:
                try:
                    save_result(item)
                except OSError as e:
                    logger.warning("测速结果保存失败: %s", e)
            self.bench_finished.emit(results)
        except Exception as e:
            self.bench_failed.emit(f"{type(e).__name__}: {e}")


# ===========================================================================
# Keyed plain text editor: Enter sends, Shift+Enter newline
# ===========================================================================


class SendEditor(QPlainTextEdit):
    send_requested = pyqtSignal()

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        # 输入法组合态（中文/日文等）：候选框打开时回车用于「选词」，不应发送。
        self._composing = False

    def inputMethodEvent(self, e) -> None:  # type: ignore[override]
        try:
            self._composing = bool(e.preeditString())
        except Exception:
            self._composing = False
        super().inputMethodEvent(e)

    def keyPressEvent(self, e: QKeyEvent) -> None:  # type: ignore[override]
        if (
            e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            and not (e.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        ):
            # 长按回车会持续触发 auto-repeat，导致同一条消息被连发多次。
            if e.isAutoRepeat() or self._composing:
                return
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
        self._llm: Optional[Any] = None
        self._agent: Optional[Agent] = None
        self._worker: Optional[AgentWorker] = None
        self._fetch_worker: Optional[FetchModelsWorker] = None
        self._sync_worker: Optional[SyncKnowledgeWorker] = None
        self._deps_worker: Optional[InstallDepsWorker] = None
        self._bench_worker: Optional[BenchmarkWorker] = None
        self._deps_cancelled: bool = False  # True if user cancelled during deps check
        self._streaming_assistant = False  # True while appending content to current assistant bubble
        # 流式文本的攒批缓冲（见 _on_content / _flush_stream_buffer）。
        self._stream_buf: list = []
        self._stream_flush_timer: Optional[QTimer] = None
        # 思考流的攒批缓冲（见 _on_reasoning / _flush_reason_buffer）。
        self._reason_buf: list = []
        self._reason_flush_timer: Optional[QTimer] = None
        self._busy = False
        # 「获取模型」/「同步知识」各有独立的忙碌态。旧实现让 _set_controls_busy
        # 统一用 self._busy 决定这两个按钮的 enable：agent 一轮结束就把它们
        # 解锁，即使用户的 fetch/sync 还在跑 —— 再点一次会覆盖 worker 引用，
        # 先结束的旧 worker 会把新 worker 的引用清成 None（取消能力丢失、
        # 结果乱序）。这里用独立标志合成 enable 状态。
        self._fetch_busy = False
        self._sync_busy = False
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
        self._pending_image_path: str = ""

        self._build_ui()
        self._wire_signals()
        self._setup_log_bridge()
        self._apply_config_to_widgets()
        self._rebuild_agent()
        self._restore_geometry()

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
        # 保存引用以便持久化分割比例（见 _save_geometry）。
        self._main_split = main_split
        outer.addWidget(main_split)

        main_split.addWidget(self._build_config_panel())
        main_split.addWidget(self._build_body_panel())
        main_split.setStretchFactor(0, 1)
        main_split.setStretchFactor(1, 5)
        main_split.setSizes([260, 580])

        # Status bar with progress
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(220)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
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
        self.llm_backend_combo = QComboBox()
        self.llm_backend_combo.addItems(["openai", "mlx_lm", "mlx_vlm", "onnx", "llama_cpp"])
        self.llm_model_path_edit = QLineEdit(placeholderText="本地主模型目录或 .onnx/.gguf 文件")
        self.llm_model_path_btn = QPushButton("…")
        self.llm_model_path_btn.setFixedWidth(28)
        self.embedding_backend_combo = QComboBox()
        self.embedding_backend_combo.addItems(["fastembed", "mlx", "onnx", "llama_cpp"])
        self.embedding_model_edit = QLineEdit()
        self.embedding_model_path_edit = QLineEdit(placeholderText="本地 Embedding 模型目录/文件")
        self.embedding_model_path_btn = QPushButton("…")
        self.embedding_model_path_btn.setFixedWidth(28)
        self.reranker_backend_combo = QComboBox()
        self.reranker_backend_combo.addItems(["fastembed", "mlx", "onnx", "disabled"])
        self.reranker_model_edit = QLineEdit()
        self.reranker_model_path_edit = QLineEdit(placeholderText="本地 Reranker .onnx 文件")
        self.reranker_model_path_btn = QPushButton("…")
        self.reranker_model_path_btn.setFixedWidth(28)
        self.reranker_enabled_check = QCheckBox("启用精排")
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

        # Row 0: 主模型后端 | 远程模型名 | 获取/扫描模型
        row0 = QHBoxLayout()
        row0.setSpacing(8)
        row0.addWidget(QLabel("主模型后端"))
        row0.addWidget(self.llm_backend_combo)
        row0.addWidget(QLabel("模型"))
        row0.addWidget(self.model_combo, 2)
        row0.addWidget(self.fetch_btn)
        row0.addWidget(QLabel("本地路径"))
        row0.addWidget(self.llm_model_path_edit, 3)
        row0.addWidget(self.llm_model_path_btn)
        lay.addLayout(row0)

        # Row 0b: OpenAI-compatible 网络参数（本地后端下自动禁用）。
        row0b = QHBoxLayout()
        row0b.setSpacing(8)
        row0b.addWidget(QLabel("Base URL"))
        row0b.addWidget(self.base_url_edit, 3)
        row0b.addWidget(QLabel("API Key"))
        row0b.addWidget(self.api_key_edit, 2)
        lay.addLayout(row0b)

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

        row3 = QHBoxLayout()
        row3.setSpacing(8)
        row3.addWidget(QLabel("Embedding"))
        row3.addWidget(self.embedding_backend_combo)
        row3.addWidget(QLabel("模型/目录名"))
        row3.addWidget(self.embedding_model_edit, 2)
        row3.addWidget(QLabel("本地路径"))
        row3.addWidget(self.embedding_model_path_edit, 3)
        row3.addWidget(self.embedding_model_path_btn)
        lay.addLayout(row3)

        row4 = QHBoxLayout()
        row4.setSpacing(8)
        row4.addWidget(QLabel("Reranker"))
        row4.addWidget(self.reranker_backend_combo)
        row4.addWidget(QLabel("模型/目录名"))
        row4.addWidget(self.reranker_model_edit, 2)
        row4.addWidget(QLabel("本地路径"))
        row4.addWidget(self.reranker_model_path_edit, 3)
        row4.addWidget(self.reranker_model_path_btn)
        row4.addWidget(self.reranker_enabled_check)
        lay.addLayout(row4)

        # Row 5: 本地后端测速对比。结果写入 ~/.base-agent/model_benchmark.json，
        # 作为后续在 fastembed / 本地后端之间切换的依据。
        row5 = QHBoxLayout()
        row5.setSpacing(8)
        self.benchmark_btn = QPushButton("测速对比")
        self.benchmark_btn.setToolTip(
            "对比「本地直载后端」与「fastembed CPU 量化版」的吞吐与一致性\n"
            "Embedding: models/nomic-embed-text-v1.5-mlx vs nomic-ai/nomic-embed-text-v1.5-Q\n"
            "Reranker: models/bge-reranker-v2-m3-ONNX vs BAAI/bge-reranker-v2-m3-int8\n"
            "结果保存到 ~/.base-agent/model_benchmark.json"
        )
        self.benchmark_label = QLabel("")
        self.benchmark_label.setStyleSheet("color:#6b7280;")
        row5.addWidget(QLabel("本地后端"))
        row5.addWidget(self.benchmark_btn)
        row5.addWidget(self.benchmark_label, 1)
        lay.addLayout(row5)

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
        self.attach_image_btn = QPushButton("附加图片…")
        self.attach_image_btn.setToolTip("为下一条消息附加一张本地图片（本地 VLM 后端）")
        self.attached_image_label = QLabel("")
        self.attached_image_label.setStyleSheet("color:#6b7280;")
        chat_header.addWidget(self.attach_image_btn)
        chat_header.addWidget(self.attached_image_label)
        self.attach_clear_btn = QPushButton("×")
        self.attach_clear_btn.setFixedWidth(28)
        self.attach_clear_btn.setToolTip("取消附加图片")
        self.attach_clear_btn.setVisible(False)
        chat_header.addWidget(self.attach_clear_btn)
        chat_header.addStretch(1)
        chat_header.addWidget(self.clear_btn)
        chat_header.addWidget(self.stop_btn)
        llay.addLayout(chat_header)

        # chat view
        self.chat_view = QTextEdit()
        self.chat_view.setReadOnly(True)
        # 关闭撤销栈：流式渲染会做数千次 insertText，每一条都会压进
        # QTextDocument 的 undo 栈（无上限），长会话内存持续增长。
        self.chat_view.setUndoRedoEnabled(False)
        # 限制对话区保留的段落数。长会话（多轮工具调用 + 大段输出）会让
        # QTextDocument 无限增长，最终吃光内存并让滚动/重排越来越卡。
        self.chat_view.document().setMaximumBlockCount(5000)
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
        # 同 chat_view：只读展示区不需要撤销栈（流式写入量极大）。
        self.think_view.setUndoRedoEnabled(False)
        font = QFont("Menlo")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.think_view.setFont(font)
        # 防止单次超长流式推理导致内存无限增长（每轮运行前会 clear()）。
        self.think_view.setMaximumBlockCount(10000)
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
        self.log_view.setUndoRedoEnabled(False)
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
        self.llm_model_path_btn.clicked.connect(self._on_pick_llm_model)
        self.embedding_model_path_btn.clicked.connect(self._on_pick_embedding_model)
        self.reranker_model_path_btn.clicked.connect(self._on_pick_reranker_model)
        self.benchmark_btn.clicked.connect(self._on_run_benchmark)
        self.attach_image_btn.clicked.connect(self._on_pick_chat_image)
        self.attach_clear_btn.clicked.connect(self._on_clear_attached_image)
        self.llm_backend_combo.currentTextChanged.connect(self._update_model_controls)
        self.reranker_backend_combo.currentTextChanged.connect(self._update_model_controls)
        self.reranker_backend_combo.currentTextChanged.connect(self._on_reranker_backend_changed)
        self.model_combo.activated.connect(self._on_model_combo_activated)
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
        b.context_shrunk.connect(self._on_context_shrunk)

    # ------------------------------------------------------------------
    # config <-> widgets
    # ------------------------------------------------------------------
    def _apply_config_to_widgets(self) -> None:
        self.base_url_edit.setText(self.config.base_url)
        self.api_key_edit.setText(self.config.api_key)
        self.llm_backend_combo.setCurrentText(self.config.llm_backend)
        self.llm_model_path_edit.setText(self.config.llm_model_path)
        self.embedding_backend_combo.setCurrentText(self.config.rag_embedding_backend)
        self.embedding_model_edit.setText(self.config.rag_embedding_model)
        self.embedding_model_path_edit.setText(self.config.rag_embedding_model_path)
        self.reranker_backend_combo.setCurrentText(self.config.rag_rerank_backend)
        self.reranker_model_edit.setText(self.config.rag_rerank_model)
        self.reranker_model_path_edit.setText(self._reranker_path_for_backend())
        self.reranker_enabled_check.setChecked(self.config.rag_rerank_enabled)
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
        self._update_model_controls()
        self._refresh_benchmark_label()

    def _gather_config_from_widgets(self) -> None:
        self.config.base_url = self.base_url_edit.text().strip()
        self.config.api_key = self.api_key_edit.text().strip()
        self.config.model = self.model_combo.currentText().strip()
        self.config.llm_backend = self.llm_backend_combo.currentText().strip() or "openai"
        self.config.llm_model_path = (
            self.llm_model_path_edit.text().strip() or self.config.llm_model_path
        )
        self.config.rag_embedding_backend = (
            self.embedding_backend_combo.currentText().strip() or "fastembed"
        )
        self.config.rag_embedding_model = (
            self.embedding_model_edit.text().strip() or self.config.rag_embedding_model
        )
        self.config.rag_embedding_model_path = (
            self.embedding_model_path_edit.text().strip() or self.config.rag_embedding_model_path
        )
        self.config.rag_rerank_backend = (
            self.reranker_backend_combo.currentText().strip() or "fastembed"
        )
        self.config.rag_rerank_model = (
            self.reranker_model_edit.text().strip() or self.config.rag_rerank_model
        )
        # MLX 交叉编码器是目录，ONNX 是文件，两者各存一份：切换后端时不必
        # 重新粘贴路径（见 _reranker_path_for_backend）。
        path_value = self.reranker_model_path_edit.text().strip()
        if self.reranker_backend_combo.currentText().strip() == "mlx":
            self.config.rag_rerank_mlx_model_path = (
                path_value or self.config.rag_rerank_mlx_model_path
            )
        else:
            self.config.rag_rerank_model_path = (
                path_value or self.config.rag_rerank_model_path
            )
        self.config.rag_rerank_enabled = self.reranker_enabled_check.isChecked()
        try:
            # 范围钳制 [0, 2]：负数/超范围的值会被服务端直接拒绝，报错难懂。
            self.config.temperature = max(0.0, min(2.0, float(self.temp_edit.text().strip() or "0.7")))
        except ValueError:
            self.config.temperature = 0.7
        try:
            # 下限 1：负数/0 会让 LLM 请求直接被服务端拒绝，报错信息难懂。
            self.config.max_tokens = max(1, int(self.max_tokens_edit.text().strip() or "32768"))
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
            # top_p ∈ (0, 1]：0 会导致无候选 token 可选，>1 被服务端拒绝。
            self.config.top_p = max(0.0001, min(1.0, float(self.top_p_edit.text().strip() or "0.95")))
        except ValueError:
            self.config.top_p = 0.95
        try:
            # min_p ∈ [0, 1)：负数无意义且会被拒绝。
            self.config.min_p = max(0.0, min(0.999, float(self.min_p_edit.text().strip() or "0.05")))
        except ValueError:
            self.config.min_p = 0.05
        try:
            # top_k >= 0：负数会让采样逻辑崩溃或被服务端拒绝。
            self.config.top_k = max(0, int(self.top_k_edit.text().strip() or "20"))
        except ValueError:
            self.config.top_k = 20
        try:
            # repetition_penalty >= 1：小于 1 会反向惩罚（越重复越鼓励），>2 输出退化。
            self.config.repetition_penalty = max(1.0, min(2.0, float(self.repeat_penalty_edit.text().strip() or "1.0")))
        except ValueError:
            self.config.repetition_penalty = 1.0
        try:
            # 下限 1 秒：0/负数会让 httpx 在发请求前就超时，表现为所有对话
            # 立刻失败且看不出原因。
            self.config.request_timeout = max(1.0, float(self.timeout_edit.text().strip() or "120"))
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

    def _release_old_agent_async(self, old_agent) -> None:
        """在后台线程释放旧 agent 的重资源（见 _rebuild_agent 的注释）。

        线程只做 ``close()`` + ``tools.shutdown()``，不碰任何 Qt 控件；
        daemon=True 保证关窗时不会拖住进程退出。
        """

        def _release():
            try:
                old_agent.close()
            except Exception as e:
                logger.warning("UI: 旧 agent 关闭异常: %s", e)
            try:
                old_agent.tools.shutdown()
            except Exception as e:
                logger.warning("UI: 旧工具资源释放异常: %s", e)

        # 保活引用：否则这个线程对象可能在跑完前被 GC。
        t = threading.Thread(target=_release, daemon=True, name="release-old-agent")
        self._release_threads = getattr(self, "_release_threads", [])
        self._release_threads[:] = [x for x in self._release_threads if x.is_alive()]
        self._release_threads.append(t)
        t.start()

    def _rebuild_agent(self) -> None:
        self._gather_config_from_widgets()
        # Shut down the old agent's resources (MCP subprocesses, httpx client)
        # before replacing it — otherwise each "应用配置" click leaks a set of
        # MCP server subprocesses and an open httpx connection.
        old_agent = getattr(self, "_agent", None)
        if old_agent is not None:
            # 旧资源的释放**不能**在主线程同步做：`tools.shutdown()` 会逐个
            # MCP server join(2s)、停子进程、释放 RAG 引擎的 LanceDB 连接与
            # ONNX/OCR 模型（数百 MB）——配了多个 server 时是十几秒的界面冻结。
            # 交给后台线程处理；新 agent 的创建不依赖旧资源，可以同时进行。
            self._release_old_agent_async(old_agent)
        old_llm = getattr(self, "_llm", None)
        if old_llm is not None:
            try:
                old_llm.close()
            except Exception:
                pass
        # 构造失败时必须清空引用：旧 agent / llm 已在上面被 close()，
        # 继续持有它们会让 _on_send 用一套已关闭的 httpx 连接池和 MCP 客户端
        # 发起新一轮对话，报出与真实原因无关的二次错误。
        self._llm = None
        self._agent = None
        new_llm = None
        try:
            new_llm = create_llm_backend(self.config)
            new_agent = Agent(llm=new_llm, config=self.config, callbacks=self._callbacks)
            new_agent.max_iterations = self.config.max_iterations
        except Exception as e:
            # new_llm 已建好但 Agent 构造失败 → 必须关掉它，否则每次失败
            # 泄漏一个 httpx 连接池。
            if new_llm is not None:
                try:
                    new_llm.close()
                except Exception:
                    pass
            self._on_error(f"无法创建 agent: {e}")
            return
        self._llm = new_llm
        self._agent = new_agent

    # ------------------------------------------------------------------
    # config actions
    # ------------------------------------------------------------------
    def _on_apply_config(self) -> None:
        # Guard: if the agent is currently running, rebuilding would shut down
        # the old agent's tools/MCP/httpx mid-execution, causing the running
        # AgentWorker to crash when its next tool call hits destroyed tools.
        # The user must wait for the current turn to finish (or stop it first).
        if self._busy or self._is_worker_running(self._worker):
            QMessageBox.warning(
                self, "正在运行",
                "Agent 正在运行，请先点击「终止对话」停止当前任务，再应用配置。",
            )
            return
        # 同步/依赖安装在跑时同样不能重建：shutdown() 会释放进程级共享的
        # OCR/docling 单例和 RAG 引擎，而同步线程可能正在用它们；同步结束后
        # 的 reload_rag() 也会作用到新 agent 上，结果不可预期。
        if (self._sync_busy or self._is_worker_running(self._sync_worker)
                or self._is_worker_running(self._deps_worker)):
            QMessageBox.warning(
                self, "正在同步",
                "知识库同步/依赖安装进行中，请先停止或等待其完成，再应用配置。",
            )
            return
        self._gather_config_from_widgets()
        self.config.save(CONFIG_PATH)
        self._rebuild_agent()
        self._append_log(
            f"[cfg] applied: backend={self.config.llm_backend} "
            f"model={self.config.model} path={self.config.llm_model_path}"
        )
        self.status_label.setText("配置已应用")

    def _on_pick_workspace(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择工作目录", self.config.workspace)
        if d:
            self.workspace_edit.setText(d)

    def _on_pick_knowledge_base(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择知识库目录", self.config.knowledge_base or self.config.workspace)
        if d:
            self.knowledge_base_edit.setText(d)

    def _update_model_controls(self, *_args) -> None:
        remote = self.llm_backend_combo.currentText() == "openai"
        self.base_url_edit.setEnabled(remote)
        self.api_key_edit.setEnabled(remote)
        self.llm_model_path_edit.setEnabled(not remote)
        self.llm_model_path_btn.setEnabled(not remote)
        self.fetch_btn.setText("获取模型" if remote else "扫描本地")
        multimodal = self.llm_backend_combo.currentText() in ("mlx_vlm", "onnx", "llama_cpp")
        self.attach_image_btn.setEnabled(multimodal)
        # 切到不支持视觉的后端时必须丢弃待发图片：否则 [image:...] 标记会以
        # 纯文本形式喂给纯文本模型，用户看到的是模型对着路径名胡编。
        if not multimodal and self._pending_image_path:
            self._set_attached_image("")
        rerank_disabled = self.reranker_backend_combo.currentText() == "disabled"
        self.reranker_enabled_check.setEnabled(not rerank_disabled)
        if rerank_disabled:
            self.reranker_enabled_check.setChecked(False)

    @staticmethod
    def _dialog_start(value: str) -> str:
        candidate = os.path.expanduser(value or "")
        if os.path.isfile(candidate):
            return os.path.dirname(candidate)
        if os.path.isdir(candidate):
            return candidate
        return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "..", "models")

    def _on_pick_llm_model(self) -> None:
        backend = self.llm_backend_combo.currentText()
        start = self._dialog_start(self.llm_model_path_edit.text())
        if backend == "llama_cpp":
            path, _ = QFileDialog.getOpenFileName(self, "选择 GGUF 主模型", start, "GGUF (*.gguf);;所有文件 (*)")
        elif backend == "onnx":
            path, _ = QFileDialog.getOpenFileName(self, "选择 ONNX 主模型", start, "ONNX (*.onnx);;所有文件 (*)")
        else:
            path = QFileDialog.getExistingDirectory(self, "选择本地主模型目录", start)
        if path:
            self.llm_model_path_edit.setText(path)

    def _on_pick_embedding_model(self) -> None:
        backend = self.embedding_backend_combo.currentText()
        start = self._dialog_start(self.embedding_model_path_edit.text())
        if backend == "llama_cpp":
            path, _ = QFileDialog.getOpenFileName(self, "选择 GGUF Embedding 模型", start, "GGUF (*.gguf);;所有文件 (*)")
        elif backend == "onnx":
            path, _ = QFileDialog.getOpenFileName(self, "选择 ONNX Embedding 模型", start, "ONNX (*.onnx);;所有文件 (*)")
        else:
            path = QFileDialog.getExistingDirectory(self, "选择 Embedding 模型目录", start)
        if path:
            self.embedding_model_path_edit.setText(path)

    def _reranker_path_for_backend(self) -> str:
        """ONNX 是文件、MLX 是目录，两者在 config 里各存一份。"""
        backend = self.reranker_backend_combo.currentText().strip() or "fastembed"
        if backend == "mlx":
            return getattr(self.config, "rag_rerank_mlx_model_path", "")
        return getattr(self.config, "rag_rerank_model_path", "")

    def _on_reranker_backend_changed(self, *_args) -> None:
        mlx = self.reranker_backend_combo.currentText().strip() == "mlx"
        self.reranker_model_path_edit.setPlaceholderText(
            "本地 MLX Reranker 目录" if mlx else "本地 Reranker .onnx 文件"
        )
        self.reranker_model_path_edit.setText(self._reranker_path_for_backend())

    def _on_pick_reranker_model(self) -> None:
        start = self._dialog_start(self.reranker_model_path_edit.text())
        if self.reranker_backend_combo.currentText().strip() == "mlx":
            path = QFileDialog.getExistingDirectory(self, "选择 MLX Reranker 目录", start)
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择 ONNX Reranker 模型", start, "ONNX (*.onnx);;所有文件 (*)"
            )
        if path:
            self.reranker_model_path_edit.setText(path)

    # ------------------------------------------------------------------
    # attached image (local VLM backends)
    # ------------------------------------------------------------------
    def _on_pick_chat_image(self) -> None:
        """Pick a local image to send with the next message.

        本地 VLM 后端（mlx_vlm / onnx / llama_cpp）没有 OpenAI 那种 multipart
        内容协议，这里用文本标记 ``[image:<path>]`` 把路径带进用户消息，由
        ``MLXVLMBackend._media`` 等解析成真正的视觉输入。
        """
        start = self._dialog_start(self._pending_image_path)
        path, _ = QFileDialog.getOpenFileName(
            self, "选择附加图片", start,
            "图片 (*.png *.jpg *.jpeg *.webp *.gif *.bmp);;所有文件 (*)",
        )
        if not path:
            return
        self._set_attached_image(path)
        self._append_log(f"[chat] 已附加图片: {path}")

    def _set_attached_image(self, path: str) -> None:
        self._pending_image_path = path
        self.attached_image_label.setText(os.path.basename(path) if path else "")
        self.attach_clear_btn.setVisible(bool(path))

    def _on_clear_attached_image(self) -> None:
        self._set_attached_image("")

    # ------------------------------------------------------------------
    # local backend benchmark
    # ------------------------------------------------------------------
    def _refresh_benchmark_label(self) -> None:
        """Show the last stored benchmark verdict next to the button."""
        try:
            from ..model_benchmark import recommend, result_for

            embedding = result_for("embedding")
            reranker = result_for("reranker")
        except Exception as e:
            self.benchmark_label.setText(f"测速记录读取失败: {e}")
            return
        if embedding is None and reranker is None:
            self.benchmark_label.setText("尚无测速记录")
            return
        parts = []
        if embedding is not None:
            parts.append(recommend(embedding))
        if reranker is not None:
            parts.append(recommend(reranker))
        self.benchmark_label.setText(" ｜ ".join(parts))

    def _on_run_benchmark(self) -> None:
        if self._is_worker_running(self._bench_worker):
            return
        # 测速会加载数 GB 级权重并占满 CPU：与 agent 推理或知识库同步并发跑，
        # 两边都会互相拖慢，测出来的数字也没有参考价值。
        if self._busy or self._is_worker_running(self._worker):
            QMessageBox.warning(
                self, "正在运行", "Agent 正在运行，请先点击「终止对话」停止当前任务，再测速。"
            )
            return
        if (self._sync_busy or self._is_worker_running(self._sync_worker)
                or self._is_worker_running(self._deps_worker)):
            QMessageBox.warning(
                self, "正在同步", "知识库同步/依赖安装进行中，请等待其完成，再测速。"
            )
            return
        self.benchmark_btn.setEnabled(False)
        self.status_label.setText("正在测速本地后端…")
        self.progress.setRange(0, 0)
        self._bench_worker = BenchmarkWorker(
            kind="all",
            rounds=3,
            threads=max(1, int(getattr(self.config, "onnx_threads", 4))),
            embedding_batch_size=max(1, int(getattr(self.config, "rag_embedding_batch_size", 8))),
            rerank_batch_size=max(1, int(getattr(self.config, "rag_rerank_batch_size", 16))),
            embedding_path=self.embedding_model_path_edit.text().strip(),
            embedding_model=self.embedding_model_edit.text().strip(),
            rerank_path=self.reranker_model_path_edit.text().strip(),
            rerank_model=self.reranker_model_edit.text().strip(),
        )
        self._bench_worker.bench_progress.connect(self._on_benchmark_progress)
        self._bench_worker.bench_finished.connect(self._on_benchmark_finished)
        self._bench_worker.bench_failed.connect(self._on_benchmark_failed)
        self._bench_worker.finished.connect(self._bench_worker.deleteLater)
        self._bench_worker.start()

    def _on_benchmark_progress(self, msg: str) -> None:
        self._append_log(f"[bench] {msg}")
        self.status_label.setText(msg)

    def _on_benchmark_finished(self, results: list) -> None:
        self._bench_worker = None
        self.benchmark_btn.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        try:
            from ..model_benchmark import recommend, summarize

            lines = []
            for item in results or []:
                lines.append(summarize(item))
                suggestion = recommend(item)
                if suggestion:
                    lines.append(f"  → {suggestion}")
            text = "\n".join(lines) or "（无结果）"
        except Exception as e:
            text = f"结果渲染失败: {e}"
        for line in (text.splitlines() or [text]):
            self._append_log(f"[bench] {line}")
        self._refresh_benchmark_label()
        self.status_label.setText("测速完成（结果已保存）")
        QMessageBox.information(self, "本地后端测速对比", text)

    def _on_benchmark_failed(self, error: str) -> None:
        self._bench_worker = None
        self.benchmark_btn.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status_label.setText("测速失败")
        self._append_log(f"[bench] 测速失败: {error}")
        QMessageBox.critical(
            self, "测速失败",
            f"本地后端测速失败：\n{error}\n\n"
            "请确认已安装对应运行时（mlx / onnxruntime / fastembed）且模型路径存在。",
        )

    def _on_model_combo_activated(self, _index: int) -> None:
        if self.llm_backend_combo.currentText() == "openai":
            return
        value = self.model_combo.currentText().strip()
        if os.path.exists(os.path.expanduser(value)):
            self.llm_model_path_edit.setText(value)

    @staticmethod
    def _is_worker_running(worker) -> bool:
        """``worker.isRunning()`` that tolerates an already-deleted QThread.

        Workers are cleaned up with ``deleteLater``, so a stale reference can
        point at a QThread whose C++ object is gone — calling ``isRunning()``
        on it raises ``RuntimeError: wrapped C/C++ object has been deleted``
        and would take down the click handler.
        """
        if worker is None:
            return False
        try:
            return bool(worker.isRunning())
        except RuntimeError:
            return False

    def _on_sync_knowledge(self, force: bool = False) -> None:
        # Agent 正在跑时不允许同步：同步线程与 agent 的 RAG 检索会并发读写
        # 同一个 LanceDB 目录，且同步结束后 UI 线程还会 reload 表句柄。
        if self._busy:
            QMessageBox.warning(
                self, "正在运行",
                "Agent 正在运行，请先点击「终止对话」停止当前任务，再同步知识库。",
            )
            return
        if self._is_worker_running(self._sync_worker):
            logger.warning("Sync already running, ignoring duplicate request")
            return
        if self._is_worker_running(self._deps_worker):
            logger.warning("Deps check already running, ignoring duplicate request")
            return
        kb = self.knowledge_base_edit.text().strip()
        ws = self.workspace_edit.text().strip() or os.path.expanduser("~/.base-agent/workspace")
        if not kb:
            QMessageBox.warning(self, "缺少知识库", "请先选择知识库目录")
            return
        logger.info("Knowledge sync triggered: kb=%s workspace=%s force=%s", kb, ws, force)
        # Step 1: check dependencies before starting sync
        self._sync_busy = True
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
        self._deps_worker.deps_finished.connect(self._on_deps_finished)
        # deleteLater 必须挂到 QThread 内建的 finished（run() 返回后触发），
        # 挂在自定义信号上会在 run() 执行途中销毁线程对象。
        self._deps_worker.finished.connect(self._deps_worker.deleteLater)
        self._deps_worker.start()

    def _reset_sync_ui(self) -> None:
        """结束同步忙碌态：清标志 + 恢复「同步知识」按钮 + 禁用「停止同步」。

        所有同步结束路径（正常/失败/取消/异常兜底）都必须走这里，否则
        ``_sync_busy`` 会永久停在 True —— 而 _set_controls_busy(False) 又
        依赖它决定是否解锁按钮，结果就是同步按钮再也点不动。
        """
        self._sync_busy = False
        # 不能无条件解锁：agent 正在跑时（`self._busy`）同步按钮必须保持禁用，
        # 否则按钮可见可点，点下去只得到一个「正在运行」弹窗 —— 与
        # _set_controls_busy 合成的忙碌态不一致。
        self.sync_knowledge_btn.setEnabled(not self._busy)
        self.sync_knowledge_btn.setText("同步知识")
        self.stop_sync_btn.setEnabled(False)

    def _on_deps_progress(self, msg: str) -> None:
        self.status_label.setText(msg)
        self._append_log(f"[deps] {msg}")

    def _on_deps_finished(self, report) -> None:
        self._deps_worker = None
        # 检查用户是否在依赖检查期间点击了停止同步
        if self._deps_cancelled:
            self._deps_cancelled = False
            logger.info("Deps check finished but user cancelled — aborting sync")
            self._reset_sync_ui()
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.status_label.setText("已取消同步")
            self._append_log("[rag] 已取消同步")
            return
        if report is not None and report.has_blocking:
            # Required deps still missing after auto-install attempt
            logger.warning("Deps check: blocking dependencies missing, aborting sync")
            self._reset_sync_ui()
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.status_label.setText("依赖缺失，无法同步")
            QMessageBox.critical(
                self, "依赖缺失",
                f"以下必要依赖未能自动安装，请手动安装后重试：\n\n{report.summary()}"
            )
            return
        if report is None:
            # worker 的 None 只有「用户取消」一种正常语义（上面已处理）；
            # ensure_dependencies 抛异常时也 emit None。依赖状态未知/缺失时
            # 绝不能照常启动 RAG 同步——那会在缺 lancedb/fastembed 的环境里
            # 跑引擎，产生难懂的二次异常并吞掉真正的失败原因。
            logger.error("Deps check failed with an exception — aborting sync")
            self._reset_sync_ui()
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.status_label.setText("依赖检查失败")
            QMessageBox.critical(
                self, "依赖检查失败",
                "依赖检查过程中出现异常，无法确认解析依赖是否可用。\n"
                "请查看日志面板中的错误信息，修复后重试。"
            )
            return

        # Step 2: start sync
        ws = getattr(self, "_pending_sync_ws", "")
        kb = getattr(self, "_pending_sync_kb", "")
        force = getattr(self, "_pending_sync_force", False)
        if not ws or not kb:
            self._reset_sync_ui()
            # 防御分支同样要复位不确定态进度条——_on_sync_knowledge 已把它设
            # 为 setRange(0,0)，不恢复的话进度条会永久转圈。
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.status_label.setText("未选择知识库目录")
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
            # 显式转发配置（#131）：config.rag_rerank_model 此前是死代码，
            # 没有任何调用点传给引擎，实际加载的一直是 engine 旧默认值。
            rerank_model=self.config.rag_rerank_model,
            # PDF 主引擎（docling|marker，#134）。
            pdf_engine=getattr(self.config, "rag_pdf_engine", "docling"),
            model_options={
                "embedding_backend": self.config.rag_embedding_backend,
                "embedding_model_path": self.config.rag_embedding_model_path,
                "embedding_threads": self.config.onnx_threads,
                "embedding_batch_size": self.config.rag_embedding_batch_size,
                "embedding_max_length": self.config.rag_embedding_max_length,
                "chunk_size": self.config.rag_chunk_size,
                "chunk_overlap": self.config.rag_chunk_overlap,
                "rerank_backend": self.config.rag_rerank_backend,
                "rerank_model_path": self.config.rag_rerank_model_path,
                "rerank_threads": self.config.onnx_threads,
                "rerank_batch_size": self.config.rag_rerank_batch_size,
                "rerank_max_length": self.config.rag_rerank_max_length,
                "rerank_enabled": self.config.rag_rerank_enabled,
            },
        )
        self._sync_worker.sync_finished.connect(self._on_sync_finished)
        self._sync_worker.sync_failed.connect(self._on_sync_failed)
        self._sync_worker.sync_progress.connect(self._on_sync_progress)
        # finished 时 worker 可能已被 _on_sync_finished 置 None——用默认参数
        # 在 connect 时快照引用，保证兜底槽拿到真正的 worker。
        self._sync_worker.finished.connect(
            lambda w=self._sync_worker: self._on_sync_thread_finished(w))
        self._sync_worker.finished.connect(self._sync_worker.deleteLater)
        self._sync_worker.start()
        # 启动日志轮询定时器：每 100ms 消费 _log_buffer，实时显示同步日志
        self._log_poll_timer.start()

    def _on_sync_finished(self, stats: dict) -> None:
        logger.info("UI: _on_sync_finished called")
        self._log_poll_timer.stop()
        # 恢复 QSignalLogHandler 的 RAG 日志过滤
        if hasattr(self, '_log_handler') and self._log_handler is not None:
            self._log_handler._sync_active = False
        # 最后消费一次 _log_buffer，确保残余日志被显示
        self._poll_sync_logs()
        try:
            self._reset_sync_ui()
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
                self._reset_sync_ui()
                self.progress.setRange(0, 1)
                self.progress.setValue(0)
                self._sync_worker = None
                self.status_label.setText("同步完成（显示结果时出错）")
            except Exception:
                pass

    def _remove_sync_tmp_handler(self, worker=None) -> None:
        """摘除同步线程挂上的临时 logging handler（幂等）。

        ``SyncKnowledgeWorker.run()`` 的 finally 会摘一次，但线程被
        ``terminate()`` 时 Python 的 finally 不保证执行；这里作为兜底再摘
        一次。handler 引用挂在 **worker** 对象上（run() 里
        ``self._sync_tmp_handler = ...``），所以必须把 worker 传进来——
        旧实现作为 MainWindow 方法读 window 上的同名属性，永远是 None，
        兜底从未生效。
        """
        if worker is None:
            return
        handler = getattr(worker, "_sync_tmp_handler", None)
        loggers = getattr(worker, "_sync_tmp_loggers", None)
        if handler is None or not loggers:
            return
        for name in loggers:
            try:
                logging.getLogger(name).removeHandler(handler)
            except Exception:
                pass
        worker._sync_tmp_handler = None
        worker._sync_tmp_loggers = None

    def _on_sync_thread_finished(self, worker=None) -> None:
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
        # 无条件补摘一次同步用的临时 handler：线程被 terminate() 时 run() 里的
        # finally 不保证执行，handler 会永久挂在 agent.rag logger 上（闭包还
        # 强引用整个 worker），下次同步再挂一个 → 日志重复输出。
        self._remove_sync_tmp_handler(worker)
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
                self._reset_sync_ui()
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
            self._reset_sync_ui()
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.status_label.setText("知识库同步失败")
            self._sync_worker = None
            QMessageBox.critical(self, "同步失败", f"知识库同步失败：\n{error}")
        except Exception as e:
            logger.error("UI: _on_sync_failed error: %s", e, exc_info=True)
            try:
                self._reset_sync_ui()
                self.progress.setRange(0, 1)
                self.progress.setValue(0)
                self._sync_worker = None
            except Exception:
                pass

    def _on_fetch_models(self) -> None:
        backend = self.llm_backend_combo.currentText().strip() or "openai"
        if backend != "openai":
            models = discover_local_models("models", backend)
            configured = self.llm_model_path_edit.text().strip()
            if configured and os.path.exists(os.path.expanduser(configured)) and configured not in models:
                models.insert(0, configured)
            self._on_models_fetched(models)
            if models:
                self.status_label.setText(f"扫描到 {len(models)} 个本地模型")
            else:
                self.status_label.setText("未扫描到匹配的本地模型")
            return

        base_url = self.base_url_edit.text().strip()
        api_key = self.api_key_edit.text().strip()
        if not base_url:
            QMessageBox.warning(self, "缺少配置", "请先填写 Base URL")
            return
        self._fetch_busy = True
        self.fetch_btn.setEnabled(False)
        self.status_label.setText("正在获取模型列表…")
        self.progress.setRange(0, 0)
        self._fetch_worker = FetchModelsWorker(base_url, api_key)
        self._fetch_worker.fetched.connect(self._on_models_fetched)
        self._fetch_worker.failed.connect(self._on_fetch_failed)
        self._fetch_worker.finished.connect(self._fetch_worker.deleteLater)
        self._fetch_worker.start()

    def _on_models_fetched(self, models: list) -> None:
        # 必须清空引用：worker 已挂 deleteLater，C++ 对象随后被销毁，继续持有
        # 会让 closeEvent 里的 isRunning() 抛 RuntimeError（PyQt 默认 qFatal
        # → 进程 abort）。典型触发：获取模型成功后关闭窗口。
        self._fetch_worker = None
        self._fetch_busy = False
        # 忙碌态合成规则是 not (busy or _fetch_busy)：agent 正在跑时
        # （_on_send 不拦 _fetch_busy），fetch 完成不能无条件解锁按钮/
        # 复位进度条——否则出现「可点击的获取模型按钮 + 空进度条」假空闲态。
        self.fetch_btn.setEnabled(not self._busy)
        if not self._busy:
            self.progress.setRange(0, 1)
        current = self.model_combo.currentText().strip()
        self.model_combo.clear()
        # 服务端可能返回非字符串元素（数字 id / null），addItems 会抛
        # TypeError → 槽函数异常 → PyQt6 abort。统一转成字符串。
        models = [str(m) for m in (models or []) if m is not None]
        self.model_combo.addItems(models)
        if current in models:
            self.model_combo.setCurrentText(current)
        elif models:
            self.model_combo.setCurrentIndex(0)
        elif current:
            # 服务端返回空列表时保留用户已输入的模型名：editable combo 被
            # clear() 后再 addItems([]) 会完全清空，用户得重新手输一遍。
            self.model_combo.setEditText(current)
        self.status_label.setText(f"获取到 {len(models)} 个模型")

    def _on_fetch_failed(self, msg: str) -> None:
        self._fetch_worker = None
        self._fetch_busy = False
        self.fetch_btn.setEnabled(not self._busy)
        if not self._busy:
            self.progress.setRange(0, 1)
        self.status_label.setText(f"获取模型失败: {msg}")
        # agent 运行中不用模态弹窗——它会阻塞事件循环，流式输出/工具回调
        # 全部停摆直到用户点掉弹窗。失败详情在状态栏与日志面板可见。
        if not self._busy:
            QMessageBox.warning(self, "获取模型失败", msg)
        else:
            self._append_log(f"[cfg] 获取模型失败: {msg}")

    # ------------------------------------------------------------------
    # chat send / stop
    # ------------------------------------------------------------------
    def _on_send(self) -> None:
        if self._busy:
            return
        # 只看 _busy 是不够的：_on_finished 在 worker 没能 2s 内 join 时会保留
        # self._worker 并把 _busy 置 False。此时再发送会**覆盖**那个仍在运行的
        # QThread 的最后一个 Python 引用 —— GC 一旦回收就是 Qt 的
        # "QThread: Destroyed while thread is still running" qFatal，进程直接
        # 退出（无提示）。必须等旧线程真正结束。
        if self._is_worker_running(self._worker):
            self.status_label.setText("上一轮对话仍在收尾，请稍候再发送")
            return
        # 同步知识库时禁止对话：两个线程会并发读写同一个 LanceDB 目录，
        # 且同步结束的 reload_rag() 会把 agent 正在用的表句柄置空重开。
        if self._sync_busy or self._is_worker_running(self._sync_worker):
            QMessageBox.information(
                self, "正在同步", "知识库同步进行中，请等待同步完成或先停止同步。"
            )
            return
        text = self.send_editor.toPlainText().strip()
        # 只有图片没有文字时也允许发送：VLM 的常见问题就是「看图说话」。
        if not text and not self._pending_image_path:
            return
        if self._agent is None:
            self._rebuild_agent()
        if self._agent is None:
            QMessageBox.warning(self, "未就绪", "Agent 未初始化，请先应用配置。")
            return
        if self._pending_image_path:
            marker = f"[image:{self._pending_image_path}]"
            text = f"{text}\n{marker}" if text else marker

        self._append_user(text)
        self._set_attached_image("")
        self.send_editor.clear()
        self._busy = True
        self._set_controls_busy(True)
        self.think_view.clear()
        # 视图清空后缓冲里可能还有未 flush 的 reasoning 残段——不清会把上一
        # 轮的尾巴写进新一轮的思考区。
        self._reason_buf.clear()
        if self._reason_flush_timer is not None:
            self._reason_flush_timer.stop()
        self.speed_label.setText("0.0 tok/s")
        self._callbacks.reset()
        self._worker = AgentWorker(self._agent, self._callbacks)
        self._worker.set_input(text)
        # Let Qt clean up the QThread asynchronously once run() returns, instead
        # of calling QThread.wait() from a slot (which can deadlock a nested
        # event loop). `_on_finished` is invoked via bridge.finished first.
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_stop(self) -> None:
        # 终止当前 agent 对话（不再处理同步取消，同步取消由 _on_stop_sync 负责）
        if not self._busy:
            return
        self._callbacks.cancel()
        # 已发出终止请求后禁用按钮：重复点击只是反复刷状态栏，而取消是协作
        # 式的、本身不会更快生效。
        self.stop_btn.setEnabled(False)
        self.status_label.setText("已请求终止对话（将在当前工具完成后生效）")

    def _on_stop_sync(self) -> None:
        """停止知识库同步：在当前文件处理完成后协作式取消。

        sync_worker.cancel() 设置 RAGEngine 内部的取消标志，
        ingest 循环在处理完当前文件后检查该标志并退出。
        """
        # 用 _is_worker_running() 而非直接 isRunning()：worker 挂了
        # deleteLater，C++ 对象可能已销毁，直接调用会抛 RuntimeError。
        if self._is_worker_running(self._sync_worker):
            logger.info("User requested sync cancellation")
            self._sync_worker.cancel()
            self.stop_sync_btn.setEnabled(False)  # 防止重复点击
            self.status_label.setText("已请求停止同步（将在当前文件完成后生效）")
            self._append_log("[rag] 用户请求停止同步")
            return
        # deps 检查阶段：记录取消意图，deps 完成后不启动同步。
        if self._is_worker_running(self._deps_worker):
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
        self.clear_btn.setEnabled(not busy)
        self.workspace_btn.setEnabled(not busy)
        self.knowledge_base_btn.setEnabled(not busy)
        # 「获取模型」「同步知识」有各自的忙碌态：agent 一轮结束不能把它们
        # 解锁（它们的 worker 可能还在跑，再点一次会覆盖 worker 引用）。
        self.fetch_btn.setEnabled(not (busy or self._fetch_busy))
        self.sync_knowledge_btn.setEnabled(not (busy or self._sync_busy))
        # 进度条同理：同步进行中不能被 agent 的 busy 切换重置掉百分比。
        if self._sync_busy:
            return
        self.progress.setRange(0, 0 if busy else 1)
        if not busy:
            self.progress.setValue(0)

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
        # 攒批写入：每个 token chunk 都做一次「取光标 → 移到末尾 → insertText
        # → 写回 → 滚动到底」会让主线程在长回复上持续做重排。这里先入缓冲，
        # 由 _flush_stream_buffer（100ms 定时）一次性 flush。
        self._stream_buf.append(text)
        if self._stream_flush_timer is None:
            self._stream_flush_timer = QTimer(self)
            self._stream_flush_timer.setSingleShot(True)
            self._stream_flush_timer.timeout.connect(self._flush_stream_buffer)
        if not self._stream_flush_timer.isActive():
            self._stream_flush_timer.start(100)

    def _flush_stream_buffer(self) -> None:
        """把攒下的流式文本一次性写入对话视图（见 _on_content）。"""
        if not self._stream_buf:
            return
        text = "".join(self._stream_buf)
        self._stream_buf.clear()
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
        # 光标此刻已在文末，无需再走 _scroll_to_bottom 的第二次
        # textCursor()+movePosition()+setTextCursor()——每个流式 chunk 都要
        # 做一遍这三步，长回复下是主线程最主要的无谓开销。
        self._pin_to_bottom(self.chat_view)

    def _on_reasoning(self, text: str) -> None:
        text = _sanitize_stream_text(text)
        if not text:
            return
        # 与 _on_content 相同的攒批写入：推理模型的 reasoning 流量不低于
        # content，每个 chunk 五步光标操作（取光标→移末→写回→插入→滚动）
        # 会让思考区越长主线程越卡。先入缓冲，100ms 定时统一 flush。
        self._reason_buf.append(text)
        if self._reason_flush_timer is None:
            self._reason_flush_timer = QTimer(self)
            self._reason_flush_timer.setSingleShot(True)
            self._reason_flush_timer.timeout.connect(self._flush_reason_buffer)
        if not self._reason_flush_timer.isActive():
            self._reason_flush_timer.start(100)

    def _flush_reason_buffer(self) -> None:
        """把攒下的 reasoning 文本一次性写入思考视图（见 _on_reasoning）。"""
        if not self._reason_buf:
            return
        text = "".join(self._reason_buf)
        self._reason_buf.clear()
        # Move the widget cursor to the end before inserting. Without this,
        # insertPlainText writes at whatever position the cursor happens to be
        # (e.g. after the user clicked the view to read earlier output), which
        # interleaves new text with existing lines -> overlap.
        cursor = self.think_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.think_view.setTextCursor(cursor)
        self.think_view.insertPlainText(text)
        self._pin_to_bottom(self.think_view)

    def _on_tool_start(self, name: str, args: Any) -> None:
        import json as _json
        # 工具调用前必须先把攒下的流式文本落地，否则工具输出会插在它前面。
        self._flush_stream_buffer()
        self._flush_reason_buffer()
        self._streaming_assistant = False  # close any streaming assistant block
        try:
            pretty = _json.dumps(args, ensure_ascii=False)
        except Exception:
            pretty = str(args)
        self._append_chat(f'<p class="t">🔧 <b>{self._escape(name)}</b>({self._escape(pretty)})…</p>')
        self._append_log(f"[tool-start] {name} {pretty}")

    def _on_tool_end(self, name: str, result: Any) -> None:
        out = result.to_message() if hasattr(result, "to_message") else str(result)
        # 长输出保留头 + 尾：报错摘要、最终状态几乎总在末尾，只截开头会让
        # 用户（和模型）看不到关键信息。完整内容进日志面板。
        if len(out) > 400:
            snippet = f"{out[:260]}\n  …（省略 {len(out) - 380} 字符）…\n{out[-120:]}"
        else:
            snippet = out
        cls = "t" if getattr(result, "success", True) else "sys"
        self._append_chat(f'<p class="{cls}">↳ <b>{self._escape(name)}</b> → {self._escape(snippet)}</p>')
        self._append_log(f"[tool-end] {name} success={getattr(result,'success',True)} :: {out[:4000]}")

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
        # 幂等判据用 worker 引用而不是 _busy：agent.run() 有多条路径调用
        # on_finished()，worker 的 finally 还会再发一次。若某条路径已经把
        # _busy 置 False，后续调用会直接 return，跳过 worker.wait() 和引用
        # 清理 —— QThread 可能在 OS 线程收尾时被 GC（Qt 直接 abort）。
        # 先把攒下的流式文本落地：否则最后一小段要等到 100ms 定时器才显示，
        # 而本轮已经结束了。
        self._flush_stream_buffer()
        worker = self._worker
        if worker is None and not self._busy:
            return
        self._worker = None
        self._busy = False
        self._streaming_assistant = False
        self._set_controls_busy(False)
        # 同步进行中不要让「就绪」覆盖掉同步进度文案。
        if not self._sync_busy:
            self.status_label.setText("就绪")
        # Join the worker thread before dropping our reference. bridge.finished
        # is emitted from the worker's `finally`, so run() has already returned
        # and wait() returns near-instantly -- but without it, dropping the last
        # Python reference can GC the QThread while its OS thread is still
        # winding down, which Qt aborts as "destroyed while still running".
        if worker is not None and not self._join_worker(worker, msecs=300):
            # 超时不能照样丢弃引用：线程还在跑时 GC 会被 Qt abort。保留引用
            # （_on_send 会据此拒绝新消息），并用定时器异步重试——旧实现在这
            # 里阻塞等 2s，同步跑 ONNX 时主线程被 GIL 饿死，很容易命中。
            self._worker = worker
            self._schedule_worker_join(worker)
        # Now that streaming is done and the worker thread has joined, it's
        # safe to show the skill-suggestion modal dialog without blocking
        # the UI event loop during streaming.
        self._show_pending_skill_dialog()

    def _shutdown_worker(self, worker, label: str, cancel: bool = False,
                         wait_ms: int = 3000, term_wait_ms: int = 2000) -> bool:
        """关窗时回收一个 worker 线程。返回 True 表示已确认结束。

        两个关键点：
        1. **不能用通配 ``disconnect()``**：它会把 ``finished → deleteLater``
           一起断掉（那是 `_on_send` 等处精心连的），此后 QThread 只能靠
           Python 引用计数析构；而 terminate() 之后若仍没 join 成功就丢引用，
           GC 会撞上 Qt 的 "Destroyed while thread is still running" qFatal。
           这里改用 ``blockSignals(True)`` 屏蔽业务信号，并保持引用直到确认
           线程结束。
        2. terminate() 后必须**检查 wait() 的返回值**：失败时不能把引用置
           None，宁可让 Python 侧泄漏一个对象，也不能让进程 abort。
        """
        if worker is None:
            return True
        try:
            worker.blockSignals(True)
        except RuntimeError:
            pass
        if cancel:
            # 协作式取消：让 agent 循环在当前工具/分块边界退出，比 terminate()
            # 干净得多。
            try:
                self._callbacks.cancel()
            except Exception:
                pass
        if not self._is_worker_running(worker):
            return True
        try:
            cancel_fn = getattr(worker, "cancel", None)
            if callable(cancel_fn):
                cancel_fn()
        except Exception:
            pass
        if self._join_worker(worker, wait_ms):
            return True
        logger.warning("UI: %s worker 未在 %dms 内结束，强制 terminate", label, wait_ms)
        try:
            worker.terminate()
        except RuntimeError:
            return True
        if self._join_worker(worker, term_wait_ms):
            return True
        # 仍没结束（例如线程卡在不可中断的 D 状态）：**保住引用**，交给
        # Python GC 在进程退出时处理，绝不能在此丢弃。
        logger.error(
            "UI: %s worker 强制终止后仍未结束，保留引用避免 Qt abort", label
        )
        return False

    def _join_worker(self, worker, msecs: int) -> bool:
        """``worker.wait(msecs)`` 的容错版本。

        QThread 被 ``deleteLater`` 后 C++ 对象可能已销毁，``wait()`` 会抛
        RuntimeError；这里按「已结束」处理（对象都没了，自然不需要等）。
        """
        if worker is None:
            return True
        try:
            return bool(worker.wait(msecs))
        except RuntimeError:
            return True

    def _schedule_worker_join(self, worker, attempt: int = 0) -> None:
        """异步轮询回收 agent worker，避免在主线程阻塞等待。"""
        if worker is None:
            return
        if self._join_worker(worker, msecs=200):
            if self._worker is worker:
                self._worker = None
            return
        if attempt >= 20:  # 约 4s，交给 closeEvent 兜底
            logger.warning("UI: agent worker 未能及时退出，保留引用由 closeEvent 兜底")
            return
        QTimer.singleShot(200, lambda: self._schedule_worker_join(worker, attempt + 1))

    def _on_error(self, msg: str) -> None:
        self._append_chat(f'<p class="sys">⚠ {self._escape(self._friendly_error(msg))}</p>')
        self._append_log(f"[error] {msg}")

    @staticmethod
    def _friendly_error(msg: str) -> str:
        """把裸异常串翻译成可执行的排查建议。

        直接把 ``httpx.ConnectError: ...`` / ``KeyError: 'choices'`` 抛给用户
        等于没有信息量。这里只做最小必要的分类，原始信息仍进日志面板。
        """
        low = (msg or "").lower()
        if "connecterror" in low or "connection refused" in low or "connecttimeout" in low:
            return f"{msg}\n→ 无法连接模型服务：请检查 Base URL 是否正确、服务是否已启动。"
        if "401" in low or "unauthorized" in low or "invalid api key" in low:
            return f"{msg}\n→ 认证失败：请检查 API Key 是否正确。"
        if "403" in low or "forbidden" in low:
            return f"{msg}\n→ 拒绝访问：API Key 可能没有该模型的权限。"
        if "404" in low or "not found" in low:
            return f"{msg}\n→ 接口或模型不存在：请检查 Base URL 路径与模型名称。"
        if "429" in low or "rate limit" in low or "quota" in low:
            return f"{msg}\n→ 被限流或配额耗尽：请稍后重试，或联系服务提供商。"
        if "timeout" in low or "timed out" in low:
            return f"{msg}\n→ 请求超时：可在配置里调大「请求超时」，或稍后重试。"
        if "'choices'" in low or "expecting value" in low or "jsondecodeerror" in low:
            return f"{msg}\n→ 返回内容不是合法的 OpenAI 兼容响应：请确认服务端确为 /v1/chat/completions 接口。"
        if "context_length" in low or "context length" in low or "maximum context" in low:
            return f"{msg}\n→ 上下文超出模型窗口：可减少历史长度（清空对话）或调小 max_tokens。"
        return msg
        # bridge.finished signal will call _on_finished from the worker's finally block

    # ------------------------------------------------------------------
    # blocking dialogs (run in UI thread, unblock worker)
    # ------------------------------------------------------------------
    def _on_confirm_request(self, message: str) -> None:
        # 兜底保护：确认消息过长会把 Yes/No 按钮顶出屏幕，这里截断到合理长度。
        if len(message) > 800:
            message = message[:800] + "…\n\n（内容过长已截断）"
        try:
            reply = QMessageBox.question(
                self, "确认操作", message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            result = reply == QMessageBox.StandardButton.Yes
        except Exception as e:
            logger.warning("UI: 确认对话框异常，按「拒绝」处理: %s", e)
            result = False
        finally:
            # 必须无条件回结果：这两个槽一旦抛异常或提前返回，
            # set_confirm_result 永不调用，worker 线程会卡满 600s 才返回拒绝，
            # 期间 UI 一直停在忙碌态。
            self._callbacks.set_confirm_result(result)

    def _on_ask_request(self, prompt: str) -> None:
        try:
            text, ok = QInputDialog.getText(self, "需要你的输入", prompt)
            result = text if ok else None
        except Exception as e:
            logger.warning("UI: 提问对话框异常，按「无输入」处理: %s", e)
            result = None
        finally:
            self._callbacks.set_ask_result(result)

    def _on_context_shrunk(self, summary: str, reason: str) -> None:
        """历史被压缩时给出可见提示。

        否则用户会以为 agent「失忆」——早期内容其实已经进了摘要并存到
        workspace/memory.md（可用 memory_search 找回）。
        """
        preview = (summary or "").strip().replace("\n", " ")[:160]
        self._append_chat(
            f'<p class="sys">🗜 上下文已压缩（{self._escape(reason or "上下文超限")}）'
            f'：早期对话已替换为摘要，可通过 memory_search 找回。'
            f'{(" 摘要：" + self._escape(preview) + "…") if preview else ""}</p>'
        )
        self._append_log(f"[context] shrunk: {reason} | {preview}")

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
    def _pin_to_bottom(widget) -> None:
        """把视口钉到底部（**前置条件**：widget 的光标已经在文末）。

        用于流式插入这类高频路径——调用方刚把光标移到 End 并写回，这里只补
        上「滚动」这一步，省掉 _scroll_to_bottom 里重复的取光标 + 移动 + 写回。
        """
        widget.ensureCursorVisible()
        sb = widget.verticalScrollBar()
        sb.setValue(sb.maximum())

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
        # 丢弃未落地的流式缓冲，避免清空后定时器再把旧文本写回视图。
        self._stream_buf.clear()
        self.chat_view.clear()
        self.think_view.clear()
        self._reason_buf.clear()
        if self._reason_flush_timer is not None:
            self._reason_flush_timer.stop()
        self._streaming_assistant = False
        # 清空对话后 token/速度标签也要复位，否则仍显示上一轮的累计值。
        self._total_tokens = 0
        self.token_label.setText("总计 0 tokens")
        self.speed_label.setText("0.0 tok/s")
        if self._agent is not None:
            self._agent.reset()

    @staticmethod
    def _escape(text: str) -> str:
        import html as _html
        return _html.escape(text).replace("\n", "<br>")

    # ------------------------------------------------------------------
    # window geometry persistence
    # ------------------------------------------------------------------
    def _restore_geometry(self) -> None:
        """恢复上次的窗口尺寸与分割比例（失败就用默认值，不影响启动）。"""
        try:
            from PyQt6.QtCore import QSettings, QByteArray
            s = QSettings()

            def _as_bytes(v):
                # QSettings.value 对 saveGeometry() 存入的 QByteArray 回读时
                # 返回 QByteArray（不是 bytes 子类），Linux ini 后端则返回
                # str——旧实现 isinstance(geo, bytes) 永远为 False，窗口
                # 几何/分割条持久化全平台静默失效。
                if isinstance(v, QByteArray):
                    return bytes(v)
                if isinstance(v, (bytes, bytearray)):
                    return bytes(v)
                return None

            geo = _as_bytes(s.value("ui/geometry"))
            if geo:
                self.restoreGeometry(geo)
            state = _as_bytes(s.value("ui/splitter"))
            if state and hasattr(self, "_main_split"):
                self._main_split.restoreState(state)
        except Exception as e:
            logger.debug("UI: restore geometry failed: %s", e)

    def _save_geometry(self) -> None:
        try:
            from PyQt6.QtCore import QSettings
            s = QSettings()
            s.setValue("ui/geometry", self.saveGeometry())
            if hasattr(self, "_main_split"):
                s.setValue("ui/splitter", self._main_split.saveState())
        except Exception as e:
            logger.debug("UI: save geometry failed: %s", e)

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
        # 本方法末尾会清空模块级 _LOG_HANDLER（见那里的注释）。
        global _LOG_HANDLER
        # --- 停止日志轮询定时器
        try:
            self._log_poll_timer.stop()
        except Exception:
            pass
        # --- AgentWorker: cancel the agent run and join the thread.
        # Without this, closing the window during an agent run leaves the
        # worker thread alive; its bridge.finished/error/content signals
        # will emit into the already-destroyed Bridge → RuntimeError.
        # 只有确认线程已结束才丢弃引用：terminate() 后仍未结束时丢引用会让
        # GC 撞上 Qt 的 "Destroyed while thread is still running" qFatal。
        if self._shutdown_worker(self._worker, "agent", cancel=True):
            self._worker = None
        # --- FetchModelsWorker: a quick HTTP call, usually finishes fast.
        if self._shutdown_worker(self._fetch_worker, "fetch",
                                 wait_ms=2000, term_wait_ms=1000):
            self._fetch_worker = None
        # --- SyncKnowledgeWorker: cooperative cancel (finishes current file).
        if self._shutdown_worker(self._sync_worker, "sync"):
            # terminate 路径下 finished 信号被 blockSignals 吞掉，
            # _on_sync_thread_finished 兜底不会触发——这里显式再摘一次
            # 临时 handler（闭包强引用整个 worker 与 RAG 引擎，不摘会
            # 阻止数百 MB 资源回收）。
            self._remove_sync_tmp_handler(self._sync_worker)
            self._sync_worker = None
        # --- InstallDepsWorker: no cancel mechanism, just wait.
        if self._shutdown_worker(self._deps_worker, "deps",
                                 wait_ms=2000, term_wait_ms=1000):
            self._deps_worker = None
        # --- BenchmarkWorker: loads model weights, give it a bit more time.
        if self._shutdown_worker(self._bench_worker, "bench",
                                 wait_ms=3000, term_wait_ms=1000):
            self._bench_worker = None
        # --- Release agent resources: MCP subprocesses, httpx client, RAG.
        if self._agent is not None:
            try:
                self._agent.tools.shutdown()
            except Exception as e:
                logger.warning("UI: shutdown error: %s", e)
            # agent.close() 释放 fact-extraction 专用的 LLMClient（独立
            # httpx 连接池）。_rebuild_agent() 会做，但窗口直接关闭时不会走
            # 那条路径，连接池会随进程残留。
            try:
                self._agent.close()
            except Exception as e:
                logger.warning("UI: agent close error: %s", e)
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
                _LOG_HANDLER.message_received.disconnect()
            except (TypeError, RuntimeError):
                pass
            try:
                _LOG_HANDLER.close()
            except Exception:
                pass
            try:
                _LOG_HANDLER.deleteLater()
            except RuntimeError:
                pass
            # 置空模块级引用：Handler.close() 不会断开信号，而被
            # message_received 连接的 self._append_log 绑定方法持有整个
            # MainWindow（含 agent / LLM 连接池 / 全部控件）。不清掉这个引用，
            # 关窗后整个窗口在进程生命周期内都无法回收。
            _LOG_HANDLER = None
        self._log_handler = None
        # 屏蔽 Bridge：worker 是**直接**调 bridge.xxx.emit() 的（不是通过
        # worker 的信号），所以 worker.disconnect() 挡不住它们。terminate()
        # 之前已 post 到主线程事件队列的 content / sync_progress 等 queued
        # 事件，会在控件开始销毁后被派发 —— 打到已释放的 C++ 对象上。
        # 用 blockSignals 而不是 disconnect()：后者在这种场景下会让 Qt 打印
        # "wildcard call disconnects from destroyed signal" 警告。
        try:
            self._bridge.blockSignals(True)
        except RuntimeError:
            pass
        self._save_geometry()
        super().closeEvent(event)


def run() -> None:
    """应用入口。

    安装全局 ``sys.excepthook``：PyQt6 中槽函数里未捕获的异常会走
    excepthook 然后直接 abort —— 进程静默退出、没有 traceback，用户只能
    看到「应用闪退」。这里至少把栈打出来，便于定位。
    """
    app = QApplication(sys.argv)
    app.setApplicationName("Base-Aug-Agent")
    app.setOrganizationName("base-agent")

    def _excepthook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.critical(
            "Unhandled exception in Qt slot: %s: %s",
            exc_type.__name__, exc_value, exc_info=(exc_type, exc_value, exc_tb),
        )

    sys.excepthook = _excepthook
    try:
        w = MainWindow()
    except Exception as e:
        logger.critical("MainWindow 初始化失败: %s", e, exc_info=True)
        QMessageBox.critical(
            None, "启动失败",
            f"主窗口初始化失败：\n{type(e).__name__}: {e}\n\n"
            "若为依赖缺失，请安装后重试（RAG 相关：poetry install -E rag）。",
        )
        return
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    run()

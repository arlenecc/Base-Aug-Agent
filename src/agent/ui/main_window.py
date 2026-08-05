"""PyQt6 main window: config strip, chat/send/thinking column, and log panel.

The agent runs in a background QThread; a signal Bridge carries streaming
events to the UI thread. Blocking confirm/ask_user dialogs are synchronised
with threading.Event so the worker blocks until the user answers.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
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
_RAG_LOG_PREFIXES = (
    "src.agent.rag",
    "src.agent.tools.rag_tool",
    "src.agent.tools.base",
    "src.agent.ui.main_window",
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

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.message_received.emit(msg)
        except Exception:
            self.handleError(record)


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

    def run(self) -> None:  # type: ignore[override]
        logger.info(
            "SyncKnowledgeWorker start: workspace=%s kb=%s force=%s embedding=%s",
            self._workspace, self._knowledge_base, self._force,
            self._embedding_model or "(default)",
        )
        # Use a flag + deferred emit so the sync_finished signal is emitted
        # AFTER the finally block completes cleanup (engine.close). Emitting
        # before close() causes the main thread to reset the UI / call
        # reload_rag() while the worker thread is still holding LanceDB /
        # ONNX resources — a potential race.
        emit_stats: Optional[dict] = None
        emit_error: Optional[str] = None
        try:
            self._engine = RAGEngine(
                workspace=self._workspace,
                knowledge_base=self._knowledge_base,
                embedding_model=self._embedding_model,
            )
            stats = self._engine.ingest(force=self._force, progress_callback=self._on_progress)
            logger.info(
                "SyncKnowledgeWorker done: found=%d extracted=%d skipped=%d deleted=%d chunks=%d errors=%d cancelled=%s",
                stats.get("files_found", 0), stats.get("files_extracted", 0),
                stats.get("files_skipped", 0), stats.get("files_deleted", 0),
                stats.get("chunks", 0), len(stats.get("errors", [])),
                stats.get("cancelled", False),
            )
            emit_stats = stats
        except Exception as e:
            logger.error("SyncKnowledgeWorker failed: %s", e, exc_info=True)
            emit_error = f"{type(e).__name__}: {e}"
        finally:
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
    """Background worker for checking and installing missing RAG dependencies."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)  # DependencyReport

    def __init__(self, kb_path: str):
        super().__init__()
        self._kb_path = kb_path

    def run(self) -> None:  # type: ignore[override]
        try:
            from ..rag.deps import ensure_dependencies
            report = ensure_dependencies(
                self._kb_path,
                auto_install=True,
                include_core=True,
                progress_callback=self.progress.emit,
            )
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
        self._deps_worker: Optional[InstallDepsWorker] = None
        self._streaming_assistant = False  # True while appending content to current assistant bubble
        self._busy = False
        self._total_tokens = 0
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
        self._sync_worker = SyncKnowledgeWorker(
            ws, kb,
            embedding_model=self.config.rag_embedding_model,
            force=force,
        )
        self._sync_worker.sync_finished.connect(self._on_sync_finished)
        self._sync_worker.sync_failed.connect(self._on_sync_failed)
        self._sync_worker.sync_progress.connect(self._on_sync_progress)
        # 兜底：如果 sync_finished / sync_failed 信号因任何原因未被正确处理
        # （例如信号槽连接断开、emit 在 Qt 事件队列中被丢弃），QThread.finished
        # 信号会在 run() 返回后由 Qt 自动发射。此时检查按钮状态，如果仍处于
        # "同步中"状态则强制恢复。
        self._sync_worker.finished.connect(self._on_sync_thread_finished)
        self._sync_worker.finished.connect(self._sync_worker.deleteLater)
        self._sync_worker.start()

    def _on_sync_finished(self, stats: dict) -> None:
        logger.info("UI: _on_sync_finished called")
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
        恢复为"同步知识"状态。如果按钮仍处于"同步中"状态，说明同步结束信号
        未被正确处理，需要强制恢复 UI 状态以避免用户界面永久卡住。
        """
        if (self._sync_worker is not None
                and self.sync_knowledge_btn.text() == "同步中…"):
            logger.warning(
                "UI: sync thread finished but button still in 'syncing' state. "
                "sync_finished/sync_failed signal may have been lost. "
                "Forcing UI reset."
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
            self._sync_worker = None

    def _on_sync_failed(self, error: str) -> None:
        logger.info("UI: _on_sync_failed called: %s", error[:200])
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
        # deps 检查阶段没有 cancel 机制，只能等它结束（通常很快）
        if self._deps_worker is not None and self._deps_worker.isRunning():
            logger.info("User requested stop during deps check (no cancel mechanism)")
            self.status_label.setText("依赖检查中，请稍候…")
            return

    def _on_sync_progress(self, done: int, total: int, current: str) -> None:
        """同步进度反馈：更新进度条和状态栏。"""
        try:
            # Guard against late progress signals arriving after sync_finished
            # has already reset the UI. While signals are queued in order, a
            # modal dialog (QMessageBox.information in _on_sync_finished) runs
            # a nested event loop that CAN process stale queued signals. This
            # guard prevents such a late signal from overwriting the
            # "知识库同步完成" status back to "同步中…".
            if self._sync_worker is None:
                return
            if total > 0:
                self.progress.setRange(0, total)
                self.progress.setValue(done)
                pct = done * 100 // total
                self.status_label.setText(f"同步中 {done}/{total} ({pct}%) — {current}")
            else:
                self.progress.setRange(0, 0)
                self.status_label.setText(f"同步中 — {current}")
        except RuntimeError as e:
            # 控件可能已被销毁（窗口关闭后信号仍可能到达）
            logger.debug("UI: _on_sync_progress ignored (control deleted): %s", e)
        except Exception as e:
            logger.warning("UI: _on_sync_progress error: %s", e)

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

        # 保存到模块级变量，供 closeEvent 清理使用。
        _LOG_HANDLER = self._log_handler
        logger.info("Log bridge: RAG/sync logs forwarded to UI panel")

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
        super().closeEvent(event)


def run() -> None:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    run()

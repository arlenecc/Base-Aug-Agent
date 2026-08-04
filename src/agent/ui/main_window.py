"""PyQt6 main window: config strip, chat/send/thinking column, and log panel.

The agent runs in a background QThread; a signal Bridge carries streaming
events to the UI thread. Blocking confirm/ask_user dialogs are synchronised
with threading.Event so the worker blocks until the user answers.
"""
from __future__ import annotations

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
    """Background worker for RAG knowledge base ingestion."""
    finished = pyqtSignal(dict)   # stats dict
    failed = pyqtSignal(str)      # error message

    def __init__(self, workspace: str, knowledge_base: str):
        super().__init__()
        self._workspace = workspace
        self._knowledge_base = knowledge_base

    def run(self) -> None:  # type: ignore[override]
        try:
            engine = RAGEngine(
                workspace=self._workspace,
                knowledge_base=self._knowledge_base,
            )
            stats = engine.ingest(force=True)
            self.finished.emit(stats)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


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
        self.setWindowTitle("Base Agent")
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
        self._streaming_assistant = False  # True while appending content to current assistant bubble
        self._busy = False
        self._total_tokens = 0

        self._build_ui()
        self._wire_signals()
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
        self.stop_btn = QPushButton("停止")
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
        self.sync_knowledge_btn.setToolTip("解析知识库文档、建立向量索引，支持 RAG 检索")
        row2.addWidget(self.sync_knowledge_btn)
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

    def _on_sync_knowledge(self) -> None:
        if self._sync_worker is not None and self._sync_worker.isRunning():
            return
        kb = self.knowledge_base_edit.text().strip()
        ws = self.workspace_edit.text().strip() or os.path.expanduser("~/.base-agent/workspace")
        if not kb:
            QMessageBox.warning(self, "缺少知识库", "请先选择知识库目录")
            return
        self.sync_knowledge_btn.setEnabled(False)
        self.sync_knowledge_btn.setText("同步中…")
        self.status_label.setText("正在同步知识库…")
        self.progress.setRange(0, 0)
        self._sync_worker = SyncKnowledgeWorker(ws, kb)
        self._sync_worker.finished.connect(self._on_sync_finished)
        self._sync_worker.failed.connect(self._on_sync_failed)
        self._sync_worker.finished.connect(self._sync_worker.deleteLater)
        self._sync_worker.start()

    def _on_sync_finished(self, stats: dict) -> None:
        self.sync_knowledge_btn.setEnabled(True)
        self.sync_knowledge_btn.setText("同步知识")
        self.progress.setRange(0, 1)
        self.status_label.setText("知识库同步完成")
        self._sync_worker = None
        files = stats.get("files_found", 0)
        chunks = stats.get("chunks", 0)
        QMessageBox.information(
            self, "同步完成",
            f"知识库同步完成！\n\n"
            f"扫描文件：{files} 个\n"
            f"向量切片：{chunks} 个\n\n"
            f"后续对话将自动参考知识库内容进行回答。"
        )

    def _on_sync_failed(self, error: str) -> None:
        self.sync_knowledge_btn.setEnabled(True)
        self.sync_knowledge_btn.setText("同步知识")
        self.progress.setRange(0, 1)
        self.status_label.setText("知识库同步失败")
        self._sync_worker = None
        QMessageBox.critical(self, "同步失败", f"知识库同步失败：\n{error}")

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
        self._callbacks.cancel()
        self.status_label.setText("已请求停止（将在当前工具完成后生效）")

    def _set_controls_busy(self, busy: bool) -> None:
        self.send_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)
        self.connect_btn.setEnabled(not busy)
        self.fetch_btn.setEnabled(not busy)
        self.progress.setRange(0, 0 if busy else 1)

    # ------------------------------------------------------------------
    # streaming slots
    # ------------------------------------------------------------------
    def _append_user(self, text: str) -> None:
        self._append_chat(f'<p><b class="u">你</b>: {self._escape(text)}</p>')

    def _on_content(self, text: str) -> None:
        import html as _html
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
        piece = _html.escape(text).replace("\n", "<br>")
        cursor.insertHtml(piece)
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
        self._total_tokens = usage.get("total_tokens", self._total_tokens)
        self.token_label.setText(f"总计 {self._total_tokens} tokens")

    def _on_speed(self, total_tokens: int, speed: float) -> None:
        self._total_tokens = total_tokens
        self.token_label.setText(f"总计 {total_tokens} tokens")
        self.speed_label.setText(f"{speed:.1f} tok/s")

    def _on_skill_suggested(self, skill) -> None:
        if not isinstance(skill, Skill):
            return
        if skill.name == "_suggested":
            reply = QMessageBox.question(
                self, "固化为技能",
                f"检测到你多次执行类似任务：\n{skill.prompt[:200]}\n\n是否保存为技能以便下次直接执行？",
            )
            if reply == QMessageBox.StandardButton.Yes:
                name, ok = QInputDialog.getText(self, "技能名称", "技能名称：", text="my-skill")
                if ok and name.strip():
                    self._agent.skills.create_skill(name.strip(), skill.keywords, skill.prompt)
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


def run() -> None:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    run()

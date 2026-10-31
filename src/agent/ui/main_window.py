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
    QStatusBar, QFileDialog, QCheckBox, QScrollArea, QGroupBox,
    QRadioButton, QFrame, QSpinBox,
)

from ..agent import Agent, AgentCallbacks
from ..config import AgentConfig
from ..llm_client import LLMClient, LLMError
from ..model_backends import (
    create_aux_llm_backend,
    create_llm_backend,
    discover_local_models,
)
from ..skills import Skill
from ..tools import ToolRegistry
from ..rag.engine import RAGEngine


CONFIG_PATH = os.path.expanduser("~/.base-agent/config.json")

logger = logging.getLogger(__name__)

# 本地模型类型的 (显示名, 配置值)（#140）。配置值是 model_backends 认识的名字，
# 显示名面向用户：llama_cpp 在 UI 上叫 GGUF，MLX 按 LM/VLM 分开。
LOCAL_LLM_TYPES = (
    ("MLX-LM", "mlx_lm"),
    ("MLX-VLM", "mlx_vlm"),
    ("ONNX", "onnx"),
    ("GGUF", "llama_cpp"),
)
# RAG 后端显示名（#140）：llama_cpp → GGUF，其余保持原名（与配置文件一致）。
RAG_BACKENDS = (
    ("fastembed", "fastembed"),
    ("mlx", "mlx"),
    ("onnx", "onnx"),
    ("GGUF", "llama_cpp"),
)
# 默认模型路径（#140 用户定参）：本地模式未指定时启动即加载这两个。
DEFAULT_MAIN_MODEL_PATH = "models/qwen3.5-4b-agentic-coder-v4-4bit-mlx"
DEFAULT_AUX_MODEL_PATH = "models/Qwen3.5-2B-MLX-4bit"
# PDF 解析引擎三选一（docling / marker / OvisOCR3 第三方案）。
PDF_ENGINES = (
    ("Docling", "docling"),
    ("Marker", "marker"),
    ("OvisOCR3", "ovis"),
)
# 各引擎单选钮的悬停说明（横排常显，tooltip 补充细节）。
_PDF_ENGINE_TIPS = {
    "docling": "docling：版面理解 + ocrmac OCR（默认）",
    "marker": "marker：marker-pdf 2.x（surya OCR，服务端进程架构）",
    "ovis": "OvisOCR3：PyMuPDF + OvisOCR3 VLM，表格区域裁剪识别后填回原位，"
            "同时是 PDF/图像的兜底方案",
}
# OvisOCR3 后端下拉（当前仅 MLX-VLM 有可用权重，ONNX/GGUF 预留）。
OCR_BACKENDS = (
    ("MLX-VLM", "mlx"),
    ("ONNX", "onnx"),
    ("GGUF", "gguf"),
)
# OvisOCR3 默认模型（配置窗口默认值；改 rag/ovis_ocr.py 的 DEFAULT_* 同步生效）。
DEFAULT_OCR_MODEL = "OvisOCR3-MLX-4bit-mixed46"
DEFAULT_OCR_MODEL_PATH = "models/OvisOCR3-MLX-4bit-mixed46"
# 启动时自动把默认模型加载备用。测试里置 1 跳过：加载数 GB 权重会让套件变成
# 分钟级，且 CI 上多半没有对应运行时（见 tests/conftest.py）。
AUTOLOAD_SKIP_ENV = "BASE_AGENT_SKIP_MODEL_AUTOLOAD"


def _select_pdf_engine_radio(window, value: str) -> None:
    """按引擎配置值选中 PDF 引擎下拉项；未知值回落 docling（默认）。

    2026-10-27 由横排单选钮改为下拉（节约该行宽度）；函数名保留。
    """
    combo = getattr(window, "pdf_engine_combo", None)
    if combo is None:
        return
    _select_combo_backend(combo, value if value else "docling")


def _checked_pdf_engine(window) -> str:
    """读 PDF 引擎下拉当前选中项的配置值；空值回落 docling。"""
    combo = getattr(window, "pdf_engine_combo", None)
    if combo is None:
        return "docling"
    return _combo_backend_value(combo) or "docling"


# 切片参数合法范围（与 UI QSpinBox 的 range 一致；QSpinBox 只防误触，
# 这里是 config/直改数值路径的最终防线）。
_CHUNK_SIZE_MIN = 100
_CHUNK_SIZE_MAX = 8192


def _normalize_chunk_params(size: int, overlap: int) -> tuple:
    """夹紧切片参数到合法组合，返回 (chunk_size, chunk_overlap)。

    - size ∈ [100, 8192]：下限防止切片退化成碎片；上限防止单片超长把
      嵌入/精排阶段的内存与注意力开销顶爆。
    - 0 ≤ overlap ≤ size − 1：重叠 ≥ 切片大小会让滑窗切分不前进
      （每个窗口回到同一位置），必须硬性排除。
    """
    size = max(_CHUNK_SIZE_MIN, min(int(size or 0), _CHUNK_SIZE_MAX))
    overlap = max(0, min(int(overlap or 0), size - 1))
    return size, overlap


def _fill_backend_combo(combo: QComboBox, items) -> None:
    """用 (显示名, 配置值) 填充下拉框，配置值存在 itemData 里。"""
    combo.clear()
    for label, value in items:
        combo.addItem(label, value)


def _combo_backend_value(combo: QComboBox) -> str:
    """取下拉框当前选中项的配置值（不是显示名）。"""
    data = combo.currentData()
    return str(data) if data is not None else combo.currentText().strip()


def _select_combo_backend(combo: QComboBox, value: str) -> None:
    """按配置值选中下拉框；找不到就留在第一项。"""
    for i in range(combo.count()):
        if str(combo.itemData(i)) == (value or "").strip():
            combo.setCurrentIndex(i)
            return

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
    """跑对话的**常驻**工作线程（``run()`` 循环，不每条消息退出）。

    为什么必须常驻（MLX 的硬约束）：KV 前缀缓存与其 stream 是**线程局部**的。
    换一个 OS 线程复用上一轮的缓存，MLX 会抛
        RuntimeError: There is no Stream(gpu, N) in current thread
    （已实测确认）。若每条消息让 ``run()`` 跑完返回、下次再 ``start()``，Qt 会
    在新 OS 线程上执行 ``run()``——缓存每轮作废，于是每轮都全量 prefill
    （3500 token ~25s，表现为「每句话都卡几十秒」）。

    因此：首次 ``_ensure_agent_worker`` 里 ``start()`` 一次；之后每条消息只
    ``set_input(text)`` 唤醒同一线程。``cancel()`` 让循环退出（关窗时用）。
    """

    warmup_done = pyqtSignal()

    def __init__(self, agent: Agent, callbacks: BridgeCallbacks):
        super().__init__()
        self._agent = agent
        self._callbacks = callbacks
        self._cond = threading.Condition()
        self._pending_text: Optional[str] = None
        self._warmup_pending = False
        self._stopping = False

    def set_input(self, text: str) -> None:
        with self._cond:
            self._pending_text = text
            self._cond.notify()

    def request_warmup(self) -> None:
        """请求在本线程预热（只 prefill system+tools 前缀，不生成、不写历史）。"""
        with self._cond:
            self._warmup_pending = True
            self._cond.notify()

    def set_agent(self, agent: Agent) -> None:
        """更新本轮要跑的 agent。

        常驻复用后 agent 可能在两次发送之间被 _rebuild_agent 换掉（改配置、
        换模型），必须换上新的，否则跑的还是旧 agent。
        """
        self._agent = agent

    def cancel(self) -> None:
        """让常驻循环退出（关窗时用；取消当前生成用 callbacks.cancel()）。"""
        with self._cond:
            self._stopping = True
            self._cond.notify()

    def run(self) -> None:  # type: ignore[override]
        while True:
            with self._cond:
                while (self._pending_text is None
                       and not self._warmup_pending
                       and not self._stopping):
                    self._cond.wait()
                if self._stopping:
                    return
                text = self._pending_text
                self._pending_text = None
                # 有真实输入时优先处理输入；预热留到空闲时再做。
                do_warmup = text is None and self._warmup_pending
                if do_warmup:
                    self._warmup_pending = False
            if do_warmup:
                try:
                    self._agent.warmup()
                except Exception:
                    pass
                finally:
                    self.warmup_done.emit()
                continue
            try:
                self._agent.run(text)
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
                 model_options: Optional[Dict[str, Any]] = None,
                 ocr_backend: str = "mlx", ocr_model: str = "",
                 ocr_model_path: str = "",
                 ocr_gguf_path: str = "", ocr_mmproj_path: str = "",
                 ocr_tokenizer_path: str = "",
                 ocr_vlm_handler: str = "mtmd", ocr_chat_format: str = "",
                 docling_dynamic_scale: bool = False,
                 docling_vlm: bool = False,
                 pdf_small_concurrency: int = 2,
                 pdf_large_threshold_mb: float = 20.0,
                 pdf_batch_threshold_mb: float = 50.0,
                 pdf_batch_pages: int = 50):
        super().__init__()
        self._workspace = workspace
        self._knowledge_base = knowledge_base
        self._embedding_model = embedding_model
        self._rerank_model = rerank_model
        self._pdf_engine = pdf_engine or "docling"
        self._ocr_backend = ocr_backend or "mlx"
        self._ocr_model = ocr_model
        self._ocr_model_path = ocr_model_path
        self._ocr_gguf_path = ocr_gguf_path
        self._ocr_mmproj_path = ocr_mmproj_path
        self._ocr_tokenizer_path = ocr_tokenizer_path
        self._ocr_vlm_handler = ocr_vlm_handler or "mtmd"
        self._ocr_chat_format = ocr_chat_format
        self._docling_dynamic_scale = bool(docling_dynamic_scale)
        self._docling_vlm = bool(docling_vlm)
        # PDF 大小限流并发：小文件并发路数 / 大文件阈值 MB（engine 转给两个
        # 转换器的并发闸门）。内存吃紧时配 1 路即全串行。
        self._pdf_small_concurrency = max(1, int(pdf_small_concurrency))
        self._pdf_large_threshold_mb = float(pdf_large_threshold_mb)
        # 大文件按页分批（阈值 MB / 每批页数）。
        self._pdf_batch_threshold_mb = float(pdf_batch_threshold_mb)
        self._pdf_batch_pages = max(1, int(pdf_batch_pages))
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
            # PDF 主引擎（docling|marker|ovis，#134），配置可切换。
            _engine_kwargs["pdf_engine"] = self._pdf_engine
            # OvisOCR3 模型（ovis 引擎 / PDF 兜底共用），配置可覆盖。
            _engine_kwargs["ocr_backend"] = self._ocr_backend
            if self._ocr_model:
                _engine_kwargs["ocr_model"] = self._ocr_model
            if self._ocr_model_path:
                _engine_kwargs["ocr_model_path"] = self._ocr_model_path
            # GGUF/mmproj/handler/format 此前只存未转发（引擎永远拿默认值）——
            # 补上；tokenizer 路径为 ONNX 后端新增。
            if self._ocr_gguf_path:
                _engine_kwargs["ocr_gguf_path"] = self._ocr_gguf_path
            if self._ocr_mmproj_path:
                _engine_kwargs["ocr_mmproj_path"] = self._ocr_mmproj_path
            if self._ocr_tokenizer_path:
                _engine_kwargs["ocr_tokenizer_path"] = self._ocr_tokenizer_path
            if self._ocr_vlm_handler:
                _engine_kwargs["ocr_vlm_handler"] = self._ocr_vlm_handler
            if self._ocr_chat_format:
                _engine_kwargs["ocr_chat_format"] = self._ocr_chat_format
            # docling 动态 OCR 倍率（默认关，全篇固定 2.0）。
            _engine_kwargs["docling_dynamic_scale"] = self._docling_dynamic_scale
            _engine_kwargs["docling_vlm"] = self._docling_vlm
            # PDF 大小限流并发（小文件并发 / 大文件独占阈值）。
            _engine_kwargs["pdf_small_concurrency"] = self._pdf_small_concurrency
            _engine_kwargs["pdf_large_threshold_mb"] = self._pdf_large_threshold_mb
            _engine_kwargs["pdf_batch_threshold_mb"] = self._pdf_batch_threshold_mb
            _engine_kwargs["pdf_batch_pages"] = self._pdf_batch_pages
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
                # 置 None 与 VectorDedupWorker 保持一致：否则关窗路径
                # `getattr(self._sync_worker, "_engine", None)` 会拿到一个**已经
                # close 过**的引擎再关一次（并发/重复 close 会在原生层与仍在用它的
                # 线程竞态 → use-after-free 段错误）。
                self._engine = None
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


class VectorDedupWorker(QThread):
    """Background worker for vector-store deduplication（UI「向量去重」按钮）.

    只做向量库的扫描 + 删除（engine.dedupe_vectors），不解析文档、不嵌入，
    因此不需要 embedding/rerank 等模型配置——只要 workspace + knowledge_base。

    信号前缀 ``dedup_``：避开 QThread 内建 ``finished``（同名自定义信号会
    遮蔽它，让挂在 finished 上的 deleteLater 在 run() 执行途中就销毁线程对象，
    见 SyncKnowledgeWorker 的注释）。日志缓冲/临时 handler 模式与
    SyncKnowledgeWorker 相同：engine 的 _log 走 log_callback，agent.rag 各
    子模块的 logging 由临时 handler 收进 _log_buffer。
    """
    dedup_finished = pyqtSignal(dict)
    dedup_failed = pyqtSignal(str)
    dedup_progress = pyqtSignal(int, int, str)

    def __init__(self, workspace: str, knowledge_base: str, parent=None):
        super().__init__(parent)
        self._workspace = workspace
        self._knowledge_base = knowledge_base
        self._engine: Optional[RAGEngine] = None
        self._log_buffer: list = []
        self._log_buffer_lock = threading.Lock()
        self._log_buffer_max = 10000

    def _emit_log(self, msg: str) -> None:
        with self._log_buffer_lock:
            self._log_buffer.append(msg)
            if len(self._log_buffer) > self._log_buffer_max:
                del self._log_buffer[: self._log_buffer_max // 2]

    def run(self) -> None:  # type: ignore[override]
        self._emit_log("[rag] ────────── 向量去重启动 ──────────")
        logger.info("VectorDedupWorker start: workspace=%s kb=%s",
                    self._workspace, self._knowledge_base)
        _tmp_handler = logging.StreamHandler(io.StringIO())
        _tmp_handler.setLevel(logging.INFO)
        _tmp_handler.setFormatter(logging.Formatter("%(message)s"))

        def _on_rag_log(record):
            self._emit_log(_tmp_handler.format(record))

        _tmp_handler.emit = _on_rag_log
        _rag_loggers = [f"{_PKG_ROOT}.rag"]
        _prev_levels = {}
        self._dedup_tmp_handler = _tmp_handler
        self._dedup_tmp_loggers = _rag_loggers
        for _name in _rag_loggers:
            _l = logging.getLogger(_name)
            _prev_levels[_name] = _l.level
            _l.setLevel(logging.INFO)
            _l.addHandler(_tmp_handler)
        emit_stats = None
        emit_error = None
        try:
            self._engine = RAGEngine(
                workspace=self._workspace,
                knowledge_base=self._knowledge_base,
            )
            self._emit_log("[rag] 向量去重：开始扫描向量库…")
            stats = self._engine.dedupe_vectors(
                progress_callback=self._on_progress,
                log_callback=self._emit_log,
            )
            self._emit_log(
                f"[rag] 向量去重完成: 清理 {stats.get('removed', 0)} 条重复向量"
                f"（{stats.get('groups', 0)} 组，{stats.get('sources_cleaned', 0)} 个来源整体去重）"
            )
            emit_stats = stats
        except Exception as e:
            self._emit_log(f"[rag] ❌ 向量去重失败: {e}")
            logger.error("VectorDedupWorker failed: %s", e, exc_info=True)
            emit_error = f"{type(e).__name__}: {e}"
        finally:
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
            # 释放 LanceDB 连接等资源（与 SyncKnowledgeWorker 同纪律：worker
            # 侧的引擎必须在 finally 里关掉，否则每次去重泄漏一份连接）。
            if self._engine is not None:
                try:
                    self._engine.close()
                except Exception as e:
                    logger.warning("VectorDedupWorker: engine close failed: %s", e)
                self._engine = None
            # 先清资源再发信号：主线程的 UI 复位 / reload_rag 跑在彻底释放
            # 之后的 worker 侧引擎之上。
            if emit_error is not None:
                self.dedup_failed.emit(emit_error)
            elif emit_stats is not None:
                self.dedup_finished.emit(emit_stats)
            else:
                self.dedup_failed.emit("Unknown dedup error (no stats or error captured)")

    def _on_progress(self, done: int, total: int, current: str) -> None:
        self.dedup_progress.emit(done, total, current)


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


# ===========================================================================
# Worker: loads a local model's weights in the background (#140)
# ===========================================================================


class LoadModelWorker(QThread):
    """在后台线程把本地模型权重读进内存。

    为什么必须放后台：MLX/GGUF/ONNX 的权重加载是数秒到数十秒量级，放进 UI
    线程会让配置窗口在整个加载期间无响应（连「取消」都点不了）。

    信号命名同样避开 QThread 内置的 ``finished``（见 SyncKnowledgeWorker 的
    注释）：这里用 ``load_done`` / ``load_failed``。
    """

    # payload: dict(role, signature, llm, detail)
    load_done = pyqtSignal(object)
    load_failed = pyqtSignal(str, str)  # role, detail

    def __init__(self, role: str, signature: tuple, config: Any):
        super().__init__()
        self._role = role
        self._signature = signature
        self._config = config

    def run(self) -> None:  # type: ignore[override]
        role = self._role
        try:
            if role == "aux":
                llm = create_aux_llm_backend(self._config)
            else:
                llm = create_llm_backend(self._config)
            # 建好对象只是懒加载壳子，权重还没读。这里主动 _ensure_loaded()
            # 把「加载」这件事做完，按钮的语义才是真的「加载模型」。
            ensure = getattr(llm, "_ensure_loaded", None)
            if callable(ensure):
                ensure()
            detail = getattr(llm, "model_path", "") or getattr(llm, "model", "")
            self.load_done.emit(
                {"role": role, "signature": self._signature,
                 "llm": llm, "detail": str(detail)}
            )
        except Exception as e:
            self.load_failed.emit(role, f"{type(e).__name__}: {e}")


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
# Secondary windows (config / log)
# ===========================================================================


class ConfigWindow(QWidget):
    """配置窗口：承载原本挤在主窗口顶部的配置区。

    控件仍然由 MainWindow 创建并持有（见 ``MainWindow._build_config_widgets``），
    这里只提供容器，因此所有既有的读写/联动逻辑无需改动。
    """

    def __init__(self, main: "MainWindow"):
        super().__init__()
        self.setWindowTitle("配置")
        # #140 之后配置窗口是两个分区、约 8 行：420px 高度会立刻出滚动条，
        # 调到 640 让默认可见范围内能看完 Agent 模型配置区。
        self.resize(1180, 640)
        self._main = main
        container = QWidget(self)
        lay = QVBoxLayout(container)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        main._build_config_widgets(lay)
        # 窗口变矮时出滚动条，行布局不会被压坏。
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        # 关闭只是隐藏：配置控件是 MainWindow 的属性，销毁窗口会连带销毁
        # 那些控件（QWidget 析构会带走子控件），之后主窗口再访问就崩了。
        self.hide()
        event.ignore()


class LogWindow(QWidget):
    """日志窗口：承载原本在主窗口右侧的日志区（同样由 MainWindow 持有控件）。"""

    def __init__(self, main: "MainWindow"):
        super().__init__()
        self.setWindowTitle("日志")
        self.resize(980, 640)
        self._main = main
        container = QWidget(self)
        lay = QVBoxLayout(container)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(4)
        main._build_log_widgets(lay)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(container)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.hide()
        event.ignore()


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
        # _on_finished 的幂等标志（常驻 worker 不再靠「_worker 置空」判重）。
        self._finish_handled = False
        self._fetch_worker: Optional[FetchModelsWorker] = None
        self._sync_worker: Optional[SyncKnowledgeWorker] = None
        # 同步起点卸载的主/辅助 LLM 角色中「确实已加载」的集合；供同步结束后只
        # 重载这些角色（用户事先手动卸载的角色不擅自重新加载）。见 _on_sync_knowledge
        # 与 _reload_models_after_sync。
        self._sync_preloaded_roles: set = set()
        self._deps_worker: Optional[InstallDepsWorker] = None
        self._dedup_worker: Optional["VectorDedupWorker"] = None
        # --- QThread 保活登记表（消除「Destroyed while thread is still running」）
        #
        # 背景：项目历史里踩过多次 Qt 的 `QThread: Destroyed while thread '' is
        # still running` qFatal → 进程直接 abort（没有 Python traceback）。根因是
        # 「自定义信号（xxx_finished）的槽里把最后一个强引用置 None」——那个信号
        # 是在 run() **内部** emit 的，emit 与线程真正结束之间还有一段窗口；窗口内
        # 引用归零 → Py 端包装器析构 → C++ QThread 被删 → Qt 看到 isRunning() →
        # qFatal。少数路径再叠加 `finished → deleteLater` 与模态框（嵌套事件循环
        # 会立刻派发 DeferredDelete）就会稳定命中。
        #
        # 对策：每个 worker 创建后立刻登记进这里，**只有确认线程真的结束后**才
        # 解除登记。业务槽继续照常置 None（那是 UI 状态语义），但不再承担「保持
        # 对象存活」的职责 —— 两者解耦后上述窗口彻底消失。
        self._worker_refs: List[Any] = []
        self._worker_refs_lock = threading.Lock()
        # 本地模型注册表（#140）：_llm 是 agent 正在用的主模型后端，_aux_llm
        # 是辅助模型后端。两者都由 UI 持有并负责 close()。签名记录「这个对象
        # 是用哪份配置建出来的」，配置变了才重建——否则每次应用配置都要重新
        # 读数 GB 权重。
        self._llm_signature: Optional[tuple] = None
        self._aux_llm: Optional[Any] = None
        self._aux_llm_signature: Optional[tuple] = None
        self._model_workers: Dict[str, LoadModelWorker] = {}
        # 每个角色独立的忙碌标志：主模型和辅助模型可以同时加载（各自一个
        # worker），共用一个标志会让第二个加载请求被直接跳过。
        self._model_busy: Dict[str, bool] = {"main": False, "aux": False}
        # 启动自动加载期间：加载失败只记日志不弹窗（开机就弹模态框很吵，
        # 而且用户还没看到界面，点了也不知道在回应什么）。
        self._model_autoloading: bool = False
        # 预热（prefill system+tools 前缀）状态：状态栏要能区分「模型已加载」
        # 与「真正可对话」。预热期间状态栏显示「正在预热模型…」，结束后显示
        # 「就绪」——否则用户无从判断什么时候才轮到自己输入。
        self._warmup_in_progress: bool = False
        self._warmup_started: Optional[float] = None
        # 切换主模型模式（本地/OpenAI）时若 agent 正在对话，无法安全重建，
        # 先记下、等本轮结束（_on_finished）再应用，避免把正在跑的 worker 的
        # 后端换掉导致崩溃。
        self._pending_mode_sync: bool = False
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
        # 任务计划轮询：计划由 agent 线程写入 <workspace>/.agent/task_plan.json，
        # 轮询读取比往工具链里插回调更稳（不依赖工具执行路径，计划被外部
        # 改动时也能跟上）。只在渲染文本变化时才重绘。
        self._plan_poll_timer = QTimer()
        self._plan_poll_timer.setInterval(600)
        self._plan_poll_timer.timeout.connect(self._poll_task_plan)
        self._plan_rendered = ""
        self._plan_poll_timer.start()
        # 闲时章节摘要调度器：无论辅助引擎是否启用都运行，空闲时增量补全每本书的
        # 章节目录摘要（详见 src/agent/rag/book_summary.py）。
        self._summary_scheduler = None
        self._start_book_summary_scheduler()
        # Pending skill suggestion — cached when the agent emits
        # on_skill_suggested during a run, and shown as a dialog AFTER the
        # run finishes. Showing a modal dialog during streaming would block
        # the UI event loop and prevent content/reasoning chunks from
        # rendering, making the app look frozen.
        self._pending_skill: Optional[Skill] = None
        self._pending_image_path: str = ""

        self._build_ui()
        # 配置/日志搬进独立窗口：必须在 _wire_signals / _apply_config_to_widgets
        # 之前构建，因为它们要连接和填充这些窗口里的控件。
        self.config_window = ConfigWindow(self)
        self.log_window = LogWindow(self)
        self._wire_signals()
        self._setup_log_bridge()
        self._apply_config_to_widgets()
        self._rebuild_agent()
        # 启动即加载默认模型（#140 用户要求）。放在 _rebuild_agent 之后：
        # 先把懒加载后端交给 agent，再由后台线程把权重读进同一个对象，
        # agent 无需重建即可用上热模型。
        self._autoload_default_models()
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

        # 顶栏：对话相关操作靠左，窗口级操作（配置 / 日志）靠右。
        outer.addLayout(self._build_top_bar())

        # 主体：左 = 对话（顶到底），右 = 任务计划 / 思考过程。
        body_split = QSplitter(Qt.Orientation.Horizontal)
        # 保存引用以便持久化分割比例（见 _save_geometry）。
        self._main_split = body_split
        outer.addWidget(body_split, 1)
        body_split.addWidget(self._build_chat_panel())
        body_split.addWidget(self._build_side_panel())
        body_split.setStretchFactor(0, 3)
        body_split.setStretchFactor(1, 1)
        body_split.setSizes([900, 380])

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

    def _build_top_bar(self) -> QHBoxLayout:
        """窗口顶栏：左为对话操作，右为「日志」「配置」两个窗口入口。"""
        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.clear_btn = QPushButton("清空对话")
        self.stop_btn = QPushButton("终止对话")
        self.stop_btn.setEnabled(False)
        self.attach_image_btn = QPushButton("附加图片…")
        self.attach_image_btn.setToolTip("为下一条消息附加一张本地图片（本地 VLM 后端）")
        self.attached_image_label = QLabel("")
        self.attached_image_label.setStyleSheet("color:#6b7280;")
        self.attach_clear_btn = QPushButton("×")
        self.attach_clear_btn.setFixedWidth(28)
        self.attach_clear_btn.setToolTip("取消附加图片")
        self.attach_clear_btn.setVisible(False)
        self.log_window_btn = QPushButton("日志")
        self.log_window_btn.setToolTip("打开日志窗口（RAG 同步、工具、检索日志）")
        self.config_window_btn = QPushButton("配置")
        self.config_window_btn.setToolTip("打开配置窗口（模型后端、路径、知识库、测速）")

        bar.addWidget(QLabel("对话"))
        bar.addWidget(self.attach_image_btn)
        bar.addWidget(self.attached_image_label)
        bar.addWidget(self.attach_clear_btn)
        bar.addStretch(1)
        bar.addWidget(self.clear_btn)
        bar.addWidget(self.stop_btn)
        bar.addSpacing(12)
        bar.addWidget(self.log_window_btn)
        bar.addWidget(self.config_window_btn)
        return bar

    def _build_chat_panel(self) -> QWidget:
        """左栏：对话区（顶到底）→ tokens 计数（居中）→ 发送区（最底部）。"""
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

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
        lay.addWidget(self.chat_view, 1)

        # tokens 计数：放在发送框/按钮上方，居中显示。
        self.speed_label = QLabel("0.0 tok/s")
        self.speed_label.setStyleSheet("color:#6b7280;")
        self.token_label = QLabel("总计 0 tokens")
        self.token_label.setStyleSheet("color:#6b7280;")
        tokens_row = QHBoxLayout()
        tokens_row.addStretch(1)
        tokens_row.addWidget(self.speed_label)
        tokens_row.addWidget(QLabel("|"))
        tokens_row.addWidget(self.token_label)
        tokens_row.addStretch(1)
        lay.addLayout(tokens_row)

        send_row = QHBoxLayout()
        self.send_editor = SendEditor()
        self.send_editor.setFixedHeight(64)
        self.send_editor.setPlaceholderText("输入消息…  (Enter 发送 / Shift+Enter 换行)")
        send_row.addWidget(self.send_editor)
        self.send_btn = QPushButton("发送")
        self.send_btn.setFixedHeight(64)
        self.send_btn.setStyleSheet("font-weight:600;")
        send_row.addWidget(self.send_btn)
        lay.addLayout(send_row)
        return panel

    def _build_side_panel(self) -> QWidget:
        """右栏：上 = 任务计划清单，下 = 思考过程（原右侧日志位）。"""
        split = QSplitter(Qt.Orientation.Vertical)
        self._side_split = split

        plan = QWidget()
        play = QVBoxLayout(plan)
        play.setContentsMargins(0, 0, 0, 0)
        play.setSpacing(4)
        plan_header = QHBoxLayout()
        plan_header.addWidget(QLabel("任务计划"))
        self.plan_status_label = QLabel("无计划")
        self.plan_status_label.setStyleSheet("color:#6b7280;")
        plan_header.addStretch(1)
        plan_header.addWidget(self.plan_status_label)
        play.addLayout(plan_header)
        self.plan_view = QTextEdit()
        self.plan_view.setReadOnly(True)
        self.plan_view.setUndoRedoEnabled(False)
        # setMaximumBlockCount 在 QTextDocument 上，不在 QTextEdit 上。
        self.plan_view.document().setMaximumBlockCount(500)
        self.plan_view.setPlaceholderText("任务制定计划后，这里显示目标与每一步的执行情况")
        play.addWidget(self.plan_view, 1)
        split.addWidget(plan)

        think = QWidget()
        tlay = QVBoxLayout(think)
        tlay.setContentsMargins(0, 0, 0, 0)
        tlay.setSpacing(4)
        think_header = QHBoxLayout()
        think_header.addWidget(QLabel("思考过程"))
        think_header.addStretch(1)
        tlay.addLayout(think_header)
        font = self._mono_font()
        self.think_view = QPlainTextEdit()
        self.think_view.setReadOnly(True)
        # 同 chat_view：只读展示区不需要撤销栈（流式写入量极大）。
        self.think_view.setUndoRedoEnabled(False)
        self.think_view.setFont(font)
        # 防止单次超长流式推理导致内存无限增长（每轮运行前会 clear()）。
        self.think_view.setMaximumBlockCount(10000)
        tlay.addWidget(self.think_view, 1)
        split.addWidget(think)

        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 2)
        split.setSizes([240, 360])

        host = QWidget()
        hlay = QVBoxLayout(host)
        hlay.setContentsMargins(0, 0, 0, 0)
        hlay.addWidget(split)
        return host

    @staticmethod
    def _mono_font() -> QFont:
        font = QFont("Menlo")
        font.setStyleHint(QFont.StyleHint.Monospace)
        return font

    def _build_log_widgets(self, lay: QVBoxLayout) -> None:
        """构建日志区（现在住在独立日志窗口里）。"""
        log_header = QHBoxLayout()
        log_header.addWidget(QLabel("日志"))
        self.log_clear_btn = QPushButton("清空")
        self.log_clear_btn.setFixedWidth(60)
        log_header.addStretch(1)
        log_header.addWidget(self.log_clear_btn)
        lay.addLayout(log_header)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setUndoRedoEnabled(False)
        self.log_view.setFont(self._mono_font())
        self.log_view.setMaximumBlockCount(5000)  # 限制日志行数，防止内存无限增长
        lay.addWidget(self.log_view, 1)

    # 采样参数行的规格：(标签, 基础属性名, 宽度, 兜底值)。前缀 "" = 主模型，
    # "aux_" = 辅助模型——两套控件同名不同前缀，处理逻辑可完全复用。
    # 「超时(秒)」已从 UI 移除（2026-10-27 用户要求）：config 的
    # request_timeout / aux_request_timeout 字段保留（后端运行时仍读取，
    # 已保存的配置值不受影响），只是不再从面板编辑。
    _SAMPLING_SPECS = (
        ("context", "max_tokens_edit", 80, "32768"),
        ("Temp", "temp_edit", 60, "0.7"),
        ("top_p", "top_p_edit", 60, "0.95"),
        ("min_p", "min_p_edit", 60, "0.05"),
        ("top_k", "top_k_edit", 60, "20"),
        ("repeat-penalty", "repeat_penalty_edit", 60, "1.0"),
    )
    # 思考模式开关的属性基础名（实际属性 = prefix + 本名，主模型 "" / 辅助
    # "aux_"）。它插在采样行「超时(秒)」之前，不是 _SAMPLING_SPECS 的成员
    # ——后者全是 QLineEdit，开关是 QCheckBox，由 _add_sampling_row 单独插入。
    _THINKING_CHECK_BASE = "enable_thinking_check"
    # KV cache 开关的属性基础名（prefix + 本名）：复用 = 跨轮前缀复用
    # （llama.cpp LlamaCache / MLX prompt cache），量化 = KV cache 量化到
    # int8（llama.cpp type_k/type_v=Q8_0；MLX kv_bits=8）。
    _KV_REUSE_CHECK_BASE = "kv_cache_check"
    _KV_Q8_CHECK_BASE = "kv_q8_check"

    def _build_config_widgets(self, lay: QVBoxLayout) -> None:
        """构建配置区控件（现在住在独立配置窗口里）。

        控件依旧挂在 MainWindow 上（``self.xxx``），这样
        ``_apply_config_to_widgets`` / ``_gather_config_from_widgets`` /
        ``_update_model_controls`` 等处理函数一行都不用改。

        布局分两个区（#140）：
          ① Agent 模型配置 —— 主模型（本地 / OpenAI 兼容二选一）+ 辅助模型；
             辅助模型在模式面板**之外**常驻可见，不随主模型模式切换而隐藏；
          ② 知识同步配置 —— 目录、Embedding、Reranker。
        """
        # --- OpenAI 兼容接口专用 ---
        self.base_url_edit = QLineEdit(placeholderText="https://api.openai.com/v1")
        self.api_key_edit = QLineEdit(placeholderText="sk-...", echoMode=QLineEdit.EchoMode.Password)
        self.model_combo = QComboBox(editable=True)
        self.model_combo.setMinimumWidth(180)
        self.fetch_btn = QPushButton("获取模型")
        # --- RAG ---
        self.embedding_backend_combo = QComboBox()
        _fill_backend_combo(self.embedding_backend_combo, RAG_BACKENDS)
        # 模型名：仅 fastembed 后端用（HuggingFace id 下载），本地后端无意义；
        # 本地后端用「本地路径」。两个字段按后端显隐（见 _update_rag_model_controls）。
        self.embedding_model_edit = QLineEdit(
            placeholderText="HuggingFace 模型名（如 nomic-ai/nomic-embed-text-v1.5-Q）")
        self.embedding_model_path_edit = QLineEdit(placeholderText="本地 Embedding 模型目录/文件")
        self.embedding_model_path_btn = QPushButton("选择")
        # Matryoshka（MRL）向量裁剪（2026-10-28）：对 nomic / jina 等支持 MRL
        # 的嵌入模型生效。启用后入库/查询对称执行「截 N 维 → L2 重归一化 →
        # int8 量化」；关闭则全维 f16 入库（原生 ANN 检索）。
        self.matryoshka_check = QCheckBox("向量裁剪")
        self.matryoshka_check.setToolTip(
            "Matryoshka 向量裁剪（MRL），对 nomic / jina 等支持的模型生效\n"
            "开：入库和查询都截前 N 维 → 重新归一化 → int8 量化（体积 4×↓）\n"
            "关：保留模型全维向量（f16 存储，检索更快）\n"
            "修改后需重新同步知识库（向量空间变化会自动清库重建）"
        )
        self.matryoshka_dim_combo = QComboBox()
        for _d, _tip in (
            (128, "最小体积，速度最快；召回质量最低"),
            (256, "均衡；jina 模型此档质量与全维持平"),
            (512, "官方推荐质量档（nomic 256 维中文实测偏弱）"),
        ):
            self.matryoshka_dim_combo.addItem(f"{_d} 维", _d)
        self.matryoshka_dim_combo.setToolTip(
            "裁剪保留的向量维度（Matryoshka 前缀维度）\n"
            "裁剪后自动完成 L2 重归一化，再按「存储精度」收尾（入库/查询对称）" + "\n"
            + " / ".join(f"{d}维：{t}" for d, t in ((128, "最小体积"), (256, "均衡"), (512, "质量最优")))
        )
        # 存储精度（2026-10-29 新增）：启用裁剪后向量以什么精度入库。
        # int8=体积最小但只能暴力扫描；f16=多 1 倍体积换回 lancedb 原生检索
        # （可建 ANN 索引，大库检索亚线性）。
        self.vector_dtype_combo = QComboBox()
        self.vector_dtype_combo.addItem("int8（省空间）", "int8")
        self.vector_dtype_combo.addItem("f16（原生检索）", "f16")
        self.vector_dtype_combo.setToolTip(
            "向量存储精度（需先开启「向量裁剪」）\n"
            "int8：256B/条（256 维），体积最小；lancedb 原生检索不认 int8 列，"
            "走自管 numpy 暴力扫描（小中库无感）\n"
            "f16：512B/条，可用 lancedb 原生检索并可建 ANN 索引（大库推荐）\n"
            "关闭裁剪时始终为全维 f16（原生检索）"
        )
        # --- PDF 解析引擎（下拉，2026-10-27 由横排单选改）+ 自定义 OCR 模型 ---
        # 改下拉的原因：该行还挤着「自定义 OCR」整段（后端/路径/保存/测试），
        # 三个单选钮 + 标签占掉 ~350px，下拉省出大半。引擎差异说明放进
        # 下拉项 tooltip（_PDF_ENGINE_TIPS 逐项 setItemData 处理）。
        self.pdf_engine_combo = QComboBox()
        _fill_backend_combo(self.pdf_engine_combo, PDF_ENGINES)
        for _value in PDF_ENGINES:
            _idx = self.pdf_engine_combo.findData(_value[1])
            if _idx >= 0:
                self.pdf_engine_combo.setItemData(
                    _idx, _PDF_ENGINE_TIPS.get(_value[1], ""),
                    Qt.ItemDataRole.ToolTipRole)
        # 并发实例（2026-10-28 用户要求）：解析引擎同时持有的转换实例路数
        # （docling/marker 共用 rag_pdf_small_concurrency），1-6 可选、默认 1。
        # 大文件（>阈值）仍独占单实例；每实例 ≈1.9GB 模型权重常驻。
        self.pdf_concurrency_combo = QComboBox()
        for _n in range(1, 7):
            self.pdf_concurrency_combo.addItem(str(_n), _n)
        self.pdf_concurrency_combo.setToolTip(
            "PDF 解析引擎并发实例数（1-6）\n"
            "同时转换的小文件路数：调大提速，代价是每实例 ~1.9GB 常驻内存\n"
            "大文件（>20MB）始终独占单实例串行转换；默认 1 = 最省内存"
        )
        self.ocr_backend_combo = QComboBox()
        _fill_backend_combo(self.ocr_backend_combo, OCR_BACKENDS)
        self.ocr_backend_combo.setToolTip(
            "OvisOCR3 推理后端。mlx=MLX-VLM 权重（默认）；onnx=ONNX 多模态导出"
            "（本地路径指向 .onnx 或目录，需在下方补充行选 tokenizer 目录）；"
            "gguf=llama.cpp GGUF 多模态（本地路径指向 .gguf 文件，必须补选 mmproj "
            "投影文件才能读图）。缺权重时加载阶段给出明确错误。"
        )
        self.ocr_model_path_edit = QLineEdit(
            placeholderText="MLX/ONNX/GGUF 模型目录或 .onnx/.gguf 文件")
        self.ocr_model_path_btn = QPushButton("选择")
        self.ocr_model_path_btn.clicked.connect(self._on_select_ocr_model_path)
        # 次级路径（按后端显隐，与主/辅助模型同一套交互）：GGUF → mmproj 投影
        # 文件（必填才能读图）；ONNX → tokenizer 目录。MLX 不显示。
        self.ocr_mmproj_path_edit = QLineEdit(placeholderText="mmproj / clip 投影文件（必填）")
        self.ocr_mmproj_path_btn = QPushButton("选择")
        self.ocr_mmproj_path_btn.clicked.connect(self._on_select_ocr_mmproj_path)
        self.ocr_tokenizer_path_edit = QLineEdit(
            placeholderText="ONNX tokenizer 目录（含 tokenizer.json，可选）")
        self.ocr_tokenizer_path_btn = QPushButton("选择")
        self.ocr_tokenizer_path_btn.clicked.connect(
            lambda: self._on_pick_secondary_path(self.ocr_tokenizer_path_edit, "tokenizer"))
        # 精排始终启用，没有 disabled 选项；要临时关掉靠配置文件的
        # rag_rerank_enabled=false（会退化为向量距离排序）。
        self.reranker_backend_combo = QComboBox()
        # fastembed/mlx/onnx 三个 UI 选项 + GGUF（llama.cpp RANK 池化，
        # 2026-10-27 新增）。"jina" 是内部后端值（本质就是 ONNXReranker），
        # 已由 config._migrate 并入 "onnx"，UI 不提供。
        _fill_backend_combo(
            self.reranker_backend_combo,
            RAG_BACKENDS[:3] + (("GGUF", "gguf"),),
        )
        self.reranker_model_edit = QLineEdit(
            placeholderText="HuggingFace 模型名（如 jinaai/jina-reranker-v2-base-multilingual）")
        self.reranker_model_path_edit = QLineEdit(placeholderText="本地 Reranker .onnx 文件")
        self.reranker_model_path_btn = QPushButton("选择")
        # --- TopN：reranker 精排后选取的结果数（rag_search 的 top_k 上限/默认）---
        self.rerank_topn_spin = QSpinBox()
        self.rerank_topn_spin.setRange(1, 20)
        self.rerank_topn_spin.setValue(int(getattr(self.config, "rag_rerank_top_n", 3)))
        self.rerank_topn_spin.setToolTip(
            "TopN：reranker 精排之后选取的结果数量\n"
            "rag_search 的 top_k 以此为上限，模型不传时也用它做默认值。\n"
            "候选池 = TopN × 4 × 2 路全量送精排，调大会线性增加检索耗时\n"
            "（jina 精排 ~49 对/s：TopN=3 ≈ 0.5s，TopN=8 ≈ 1.3s）。"
        )
        # --- 切片参数（Row 4「参数设置」）：config.rag_chunk_size/_overlap ---
        # 全链路唯一来源：config → SyncKnowledgeWorker/tools.base → RAGEngine →
        # chunker（含 epub 章节直通上限 engine._max_chars）。字段已在
        # config._INT_FIELDS（字符串/浮点 json 值自动纠正），save() 全量落盘。
        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(100, 8192)
        self.chunk_size_spin.setSingleStep(50)
        self.chunk_size_spin.setSuffix(" 字符")
        self.chunk_size_spin.setToolTip(
            "每个文本切片的最大字符数（硬上限，所有切片环节统一使用：普通文档、\n"
            "epub 章节直通判定、超长章节重切均以此为准，不存在某个环节另切更短）。\n"
            "注意：切片超过当前 Embedding 模型的位置上限时会在模型层被截断\n"
            "（如 bge-small-zh-v1.5 = 512 字符），建议不超过该值，或换长上下文模型。"
        )
        self.chunk_overlap_spin = QSpinBox()
        self.chunk_overlap_spin.setRange(0, 4096)
        self.chunk_overlap_spin.setSingleStep(10)
        self.chunk_overlap_spin.setSuffix(" 字符")
        self.chunk_overlap_spin.setToolTip(
            "相邻切片的重叠字符数（默认取切片大小的 10%）。\n"
            "会被自动夹紧到 < 切片大小（重叠 ≥ 切片会导致切分不前进）。"
        )
        # 语义切片开关（默认开）：关 = 跳过语义切片，直接句边界字符递归切片。
        self.semantic_chunk_check = QCheckBox("语义切片")
        self.semantic_chunk_check.setChecked(
            bool(getattr(self.config, "rag_semantic_chunking", True)))
        self.semantic_chunk_check.setToolTip(
            "开：结构 + 语义混合切片（标题分段 + 相似度找语义边界）\n"
            "关：不做语义切片，直接按句边界贪心打包到切片大小的字符递归切片\n"
            "（不加载 model2vec 嵌入，更省内存，切片更满但边界不按语义）"
        )
        # --- 目录 / 动作 ---
        self.workspace_edit = QLineEdit()
        self.workspace_btn = QPushButton("选择")
        self.knowledge_base_edit = QLineEdit(placeholderText="（可选）知识库目录")
        self.knowledge_base_btn = QPushButton("选择")
        # 注意：「清空对话」「终止对话」属于对话操作，已移到主窗口顶栏，
        # 不随配置区进入配置窗口。

        # ==================================================================
        # 分区一：Agent 模型配置
        # ==================================================================
        agent_group = QGroupBox("Agent 模型配置")
        agent_lay = QVBoxLayout(agent_group)
        agent_lay.setSpacing(6)

        # 主模型模式：本地 / OpenAI 兼容接口，二选一。
        mode_row = QHBoxLayout()
        mode_row.setSpacing(12)
        mode_row.addWidget(QLabel("主模型模式"))
        self.llm_mode_local_radio = QRadioButton("本地模式")
        self.llm_mode_openai_radio = QRadioButton("OpenAI 兼容接口")
        self.llm_mode_local_radio.setToolTip(
            "进程内直载本地权重（MLX-LM / MLX-VLM / ONNX / GGUF），无需网络"
        )
        self.llm_mode_openai_radio.setToolTip("连接 OpenAI 兼容的 /chat/completions 接口")
        mode_row.addWidget(self.llm_mode_local_radio)
        mode_row.addWidget(self.llm_mode_openai_radio)
        mode_row.addStretch(1)
        agent_lay.addLayout(mode_row)

        # --- 本地模式面板：只放主模型 ---
        self.local_mode_panel = QWidget()
        local_lay = QVBoxLayout(self.local_mode_panel)
        local_lay.setContentsMargins(0, 0, 0, 0)
        local_lay.setSpacing(6)

        self._add_local_model_block(
            local_lay, "主模型",
            prefix="llm",
            combo_attr="llm_backend_combo",
            path_attr="llm_model_path_edit",
            btn_attr="llm_model_path_btn",
            picker=self._on_pick_llm_model,
            placeholder="本地主模型目录或 .onnx/.gguf 文件",
            path_value=self.config.llm_model_path or DEFAULT_MAIN_MODEL_PATH,
        )
        # （旧「GGUF 跨轮 KV 复用」点选框已移除：由参数配置行里通用的
        # 「KV复用」「KV量化」点选框取代——新开关同时覆盖 GGUF 与 MLX，
        # 且主/辅助模型各有一份，不再限本地 GGUF 模式。）
        self._add_sampling_row(
            local_lay, "", self._main_sampling_defaults(),
            buttons=(
                ("加载模型", "llm_load_btn", "load:main",
                 "把当前配置的主模型加载到内存备用"),
                ("卸载", "llm_unload_btn", "unload:main",
                 "卸载已加载的主模型，释放内存"),
            ),
        )
        agent_lay.addWidget(self.local_mode_panel)

        # --- OpenAI 兼容接口面板：只放主模型的远程连接 ---
        self.openai_mode_panel = QWidget()
        openai_lay = QVBoxLayout(self.openai_mode_panel)
        openai_lay.setContentsMargins(0, 0, 0, 0)
        openai_lay.setSpacing(6)

        remote_row = QHBoxLayout()
        remote_row.setSpacing(8)
        remote_row.addWidget(QLabel("Base URL"))
        remote_row.addWidget(self.base_url_edit, 3)
        remote_row.addWidget(QLabel("API Key"))
        remote_row.addWidget(self.api_key_edit, 2)
        remote_row.addWidget(QLabel("模型"))
        remote_row.addWidget(self.model_combo, 2)
        remote_row.addWidget(self.fetch_btn)
        openai_lay.addLayout(remote_row)

        self._add_sampling_row(
            openai_lay, "", self._main_sampling_defaults(),
            buttons=(("应用配置", "connect_btn", "apply", None),),
        )
        # connect_btn 由 _add_sampling_row 创建，这里补上强调样式。
        self.connect_btn.setStyleSheet("font-weight:600;")
        agent_lay.addWidget(self.openai_mode_panel)

        # --- 辅助模型：独立于主模型模式，常驻可见 ---
        # 辅助模型只支持本地直载，与「主模型选本地还是接口」无关：主模型走
        # OpenAI 接口时，摘要有个小模型在本地跑反而更划算。所以它放在两个
        # 模式面板**之外**，是 agent_lay 的常驻子节点，不随模式切换隐藏。
        # 分隔线把「主模型」与「辅助模型」在视觉上也切开，避免误以为辅助
        # 模型隶属于本地模式。
        aux_sep = QFrame()
        aux_sep.setFrameShape(QFrame.Shape.HLine)
        aux_sep.setFrameShadow(QFrame.Shadow.Sunken)
        agent_lay.addWidget(aux_sep)

        self._add_local_model_block(
            agent_lay, "辅助模型",
            prefix="aux_llm",
            combo_attr="aux_llm_backend_combo",
            path_attr="aux_llm_model_path_edit",
            btn_attr="aux_llm_model_path_btn",
            picker=self._on_pick_aux_llm_model,
            placeholder="本地辅助模型目录或 .onnx/.gguf 文件",
            path_value=getattr(self.config, "aux_llm_model_path", "") or DEFAULT_AUX_MODEL_PATH,
            # 「启用」选择框紧跟标题（默认不勾选）：控制启动自动加载与知识
            # 同步前后的自动加载；手动「加载模型」按钮不受限（显式操作）。
            enable_attr="aux_llm_enable_check",
            enable_checked=getattr(self.config, "aux_llm_enabled", False),
            enable_tooltip=(
                "启用辅助模型（上下文摘要 / RAG 检索归纳的小模型）。\n"
                "不勾选：启动不加载、知识同步后也不自动重载，相关功能自动"
                "退化（摘要用主模型、检索返回原始分片）。"
            ),
        )
        self._add_sampling_row(
            agent_lay, "aux_", self._aux_sampling_defaults(),
            buttons=(
                ("加载模型", "aux_llm_load_btn", "load:aux",
                 "把当前配置的辅助模型加载到内存备用"),
                ("卸载", "aux_llm_unload_btn", "unload:aux",
                 "卸载已加载的辅助模型，释放内存"),
            ),
        )

        lay.addWidget(agent_group)

        # ==================================================================
        # 分区二：知识同步配置
        # ==================================================================
        kb_group = QGroupBox("知识同步配置")
        kb_lay = QVBoxLayout(kb_group)
        kb_lay.setSpacing(6)

        # Row 1: 工作目录 | 知识库目录 | 应用配置
        # 「应用配置」在规格里只出现在接口模式行末，但本地模式用户同样要能
        # 应用工作目录 / RAG 设置（同步知识按钮只覆盖 RAG，不覆盖 workspace）。
        # 这里放一个同功能的副本，避免本地模式下没有应用入口。
        dir_row = QHBoxLayout()
        dir_row.setSpacing(8)
        dir_row.addWidget(QLabel("工作目录"))
        dir_row.addWidget(self.workspace_edit, 1)
        dir_row.addWidget(self.workspace_btn)
        dir_row.addWidget(QLabel("知识库"))
        dir_row.addWidget(self.knowledge_base_edit, 1)
        dir_row.addWidget(self.knowledge_base_btn)
        self.kb_apply_btn = QPushButton("应用配置")
        self.kb_apply_btn.setStyleSheet("font-weight:600;")
        self.kb_apply_btn.setToolTip("应用工作目录 / 知识库 / Embedding / Reranker 设置")
        self.kb_apply_btn.clicked.connect(self._on_apply_config)
        dir_row.addWidget(self.kb_apply_btn)
        # 知识库三个操作按钮（2026-10-28 用户要求）：依次排在「应用配置」后面，
        # 从 Embedding / Reranker 行挪出——三个行内控件本就挤，操作入口集中在
        # 首行更好找。顺序：开始同步 → 停止同步 → 向量去重。
        self.sync_knowledge_btn = QPushButton("开始同步")
        self.sync_knowledge_btn.setToolTip(
            "解析知识库文档、建立向量索引，支持 RAG 检索\n"
            "左键：增量同步（只处理新增/修改的文件）\n"
            "右键：强制全量重新处理"
        )
        # Right-click context menu: force full reprocess (ignores manifest).
        self.sync_knowledge_btn.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        self._force_sync_action = self.sync_knowledge_btn.addAction("强制全量重新处理")
        self._force_sync_action.triggered.connect(lambda: self._on_sync_knowledge(force=True))
        dir_row.addWidget(self.sync_knowledge_btn)
        # Stop-sync button: enabled only while a sync is running. Cancels
        # the ingest after the current file completes (cooperative cancel).
        self.stop_sync_btn = QPushButton("停止同步")
        self.stop_sync_btn.setEnabled(False)
        self.stop_sync_btn.setToolTip("停止知识库同步（在当前文件处理完成后生效）")
        dir_row.addWidget(self.stop_sync_btn)
        # 向量去重：同步前的库维护操作——扫描向量库按文本内容清理重复向量
        # （保留入库最早的一份），降低检索噪声。点击后需确认（删除不可逆）。
        self.vector_dedup_btn = QPushButton("向量去重")
        self.vector_dedup_btn.setToolTip(
            "扫描向量库，按文本内容找出完全相同的重复向量\n"
            "保留入库最早的一份，其余删除（不可恢复）\n"
            "用于清理同一文档多次入库造成的检索噪声"
        )
        dir_row.addWidget(self.vector_dedup_btn)
        kb_lay.addLayout(dir_row)

        # Row 1.5: PDF 解析引擎（三选一单选钮横列）+ 自定义 OCR 配置（同行）
        # 「自定义 OCR」并入同行（2026-10 用户要求）：选中 ovis 单选时用的
        # 就是这套配置，相邻摆放更直观；路径框省去「本地路径」标签——占位符
        # 已说明用途，且该行变长后标签只会加剧拥挤。
        engine_row = QHBoxLayout()
        engine_row.setSpacing(8)
        engine_row.addWidget(QLabel("PDF 解析引擎"))
        engine_row.addWidget(self.pdf_engine_combo)
        engine_row.addWidget(QLabel("并发实例"))
        engine_row.addWidget(self.pdf_concurrency_combo)
        engine_row.addStretch(1)
        engine_row.addWidget(QLabel("自定义 OCR"))
        engine_row.addWidget(self.ocr_backend_combo)
        engine_row.addWidget(self.ocr_model_path_edit, 3)
        engine_row.addWidget(self.ocr_model_path_btn)
        # 自定义 OCR 的两个操作按钮（#用户需求）：
        #  - 保存设置：把本行 OCR 配置写入并落盘 config，供知识库同步使用；
        #  - 识别测试：弹出独立窗口，用各 OCR 模型对任意图片/PDF 做目测对比。
        self.ocr_save_btn = QPushButton("保存设置")
        self.ocr_save_btn.setToolTip(
            "保存并应用「自定义 OCR」这一行的配置\n"
            "下次知识库同步会使用此处设置的 Ovis OCR 模型"
        )
        self.ocr_save_btn.clicked.connect(self._on_ocr_save_settings)
        engine_row.addWidget(self.ocr_save_btn)
        self.ocr_test_btn = QPushButton("识别测试")
        self.ocr_test_btn.setToolTip(
            "弹出 OCR 识别测试窗口\n"
            "可选 Apple OCR / OvisOCR3 / 混合 / 自定义 OCR，对指定图片或 PDF 做识别并左右对照"
        )
        self.ocr_test_btn.clicked.connect(self._on_ocr_test)
        engine_row.addWidget(self.ocr_test_btn)
        self.convert_tool_btn = QPushButton("转换工具")
        self.convert_tool_btn.setToolTip(
            "弹出文件转 Markdown 工具窗口\n"
            "选文件或目录批量转换成 .md 写入指定目录（不入知识库），"
            "采用本面板的解析引擎与 OCR 设置")
        self.convert_tool_btn.clicked.connect(self._on_convert_tool)
        engine_row.addWidget(self.convert_tool_btn)
        kb_lay.addLayout(engine_row)

        # Row 1.7: 次级路径行（与主/辅助模型同一套交互）——仅 ONNX / GGUF 后端
        # 可见（_update_ocr_secondary_controls）：GGUF → mmproj 投影文件；
        # ONNX → tokenizer 目录。MLX 时整行隐藏。
        ocr_sec_row = QHBoxLayout()
        ocr_sec_row.setSpacing(8)
        self.ocr_mmproj_path_label = QLabel("mmproj 路径")
        ocr_sec_row.addWidget(self.ocr_mmproj_path_label)
        ocr_sec_row.addWidget(self.ocr_mmproj_path_edit, 3)
        ocr_sec_row.addWidget(self.ocr_mmproj_path_btn)
        self.ocr_tokenizer_path_label = QLabel("tokenizer 路径")
        ocr_sec_row.addWidget(self.ocr_tokenizer_path_label)
        ocr_sec_row.addWidget(self.ocr_tokenizer_path_edit, 3)
        ocr_sec_row.addWidget(self.ocr_tokenizer_path_btn)
        kb_lay.addLayout(ocr_sec_row)
        self.ocr_backend_combo.currentIndexChanged.connect(
            self._update_ocr_secondary_controls)

        # Row 2: Embedding 后端 | 模型名(仅 fastembed) / 本地路径(本地后端) | 切片/裁剪参数
        emb_row = QHBoxLayout()
        emb_row.setSpacing(8)
        emb_row.addWidget(QLabel("Embedding"))
        emb_row.addWidget(self.embedding_backend_combo)
        self.embedding_name_label = QLabel("模型名")
        emb_row.addWidget(self.embedding_name_label)
        emb_row.addWidget(self.embedding_model_edit, 2)
        self.embedding_path_label = QLabel("本地路径")
        emb_row.addWidget(self.embedding_path_label)
        emb_row.addWidget(self.embedding_model_path_edit, 3)
        emb_row.addWidget(self.embedding_model_path_btn)
        # 切片参数（2026-10-27 用户要求并入 Embedding 行）：语义切片开关 +
        # 切片大小/重叠——切片产物的嵌入就在本行配置的模型上，参数放一起
        # 便于一起调。收集/回填/同步入口 gather 与「参数设置」独立行时一致。
        emb_row.addWidget(self.semantic_chunk_check)
        emb_row.addWidget(QLabel("切片大小"))
        emb_row.addWidget(self.chunk_size_spin)
        emb_row.addWidget(QLabel("重叠"))
        emb_row.addWidget(self.chunk_overlap_spin)
        # 向量裁剪（Matryoshka）：与切片参数同属「嵌入产物怎么入库」的配置。
        emb_row.addWidget(self.matryoshka_check)
        emb_row.addWidget(self.matryoshka_dim_combo)
        emb_row.addWidget(self.vector_dtype_combo)
        kb_lay.addLayout(emb_row)

        # Row 3: Reranker 后端 | 模型名(仅 fastembed) / 本地路径(本地后端) | TopN
        rerank_row = QHBoxLayout()
        rerank_row.setSpacing(8)
        rerank_row.addWidget(QLabel("Reranker"))
        rerank_row.addWidget(self.reranker_backend_combo)
        self.reranker_name_label = QLabel("模型名")
        rerank_row.addWidget(self.reranker_name_label)
        rerank_row.addWidget(self.reranker_model_edit, 2)
        self.reranker_path_label = QLabel("本地路径")
        rerank_row.addWidget(self.reranker_path_label)
        rerank_row.addWidget(self.reranker_model_path_edit, 3)
        rerank_row.addWidget(self.reranker_model_path_btn)
        rerank_row.addWidget(QLabel("TopN"))
        rerank_row.addWidget(self.rerank_topn_spin)
        kb_lay.addLayout(rerank_row)

        lay.addWidget(kb_group)

    # ------------------------------------------------------------------
    # config widget builders
    # ------------------------------------------------------------------
    def _main_sampling_defaults(self) -> Dict[str, Any]:
        return {
            "max_tokens_edit": self.config.max_tokens,
            "temp_edit": self.config.temperature,
            "top_p_edit": self.config.top_p,
            "min_p_edit": self.config.min_p,
            "top_k_edit": self.config.top_k,
            "repeat_penalty_edit": self.config.repetition_penalty,
            # 思考模式：主模型默认开启（思考过程可见，利于复杂任务）。
            "enable_thinking_check": getattr(self.config, "llm_enable_thinking", True),
            # KV cache 开关：主模型字段（GGUF/MLX 共读）。
            "kv_cache_check": getattr(self.config, "llama_cpp_kv_cache", True),
            "kv_q8_check": getattr(self.config, "llama_cpp_kv_q8", True),
        }

    def _aux_sampling_defaults(self) -> Dict[str, Any]:
        cfg = self.config
        return {
            "max_tokens_edit": getattr(cfg, "aux_max_tokens", 2048),
            "temp_edit": getattr(cfg, "aux_temperature", 0.5),
            "top_p_edit": getattr(cfg, "aux_top_p", 0.95),
            "min_p_edit": getattr(cfg, "aux_min_p", 0.05),
            "top_k_edit": getattr(cfg, "aux_top_k", 20),
            "repeat_penalty_edit": getattr(cfg, "aux_repetition_penalty", 1.0),
            # 思考模式：辅助模型默认**关闭**——它只跑摘要/归纳等轻量子任务，
            # 要的是快与稳，开思考只会多出一大段 reasoning 拖慢速度。
            "enable_thinking_check": getattr(cfg, "aux_llm_enable_thinking", False),
            # KV cache 开关：辅助模型独立字段（经 _AuxConfigView 映射给后端）。
            "kv_cache_check": getattr(cfg, "aux_llama_cpp_kv_cache", True),
            "kv_q8_check": getattr(cfg, "aux_llama_cpp_kv_q8", True),
        }

    def _add_local_model_block(
        self, lay: QVBoxLayout, title: str, *,
        prefix: str, combo_attr: str, path_attr: str, btn_attr: str,
        picker, placeholder: str, path_value: str,
        enable_attr: str = "", enable_checked: bool = False,
        enable_tooltip: str = "",
    ) -> None:
        """一行「模型类型 + 本地路径 + 选择」，上方带一个小标题。

        ``enable_attr`` 非空时在标题右侧放一个「启用」选择框（如辅助模型的
        启用开关，默认不勾选）；主模型不传该参数，行为不变。

        另加一行**次级路径**（按后端显隐，见 _update_secondary_path_controls）：
        GGUF → mmproj 路径（多模态投影，可选）；ONNX → tokenizer 路径（可选）。
        MLX / OpenAI 不显示——它们要么不需要、要么走别的字段。
        """
        label = QLabel(title)
        label.setStyleSheet("font-weight:600;")
        if enable_attr:
            # 标题与「启用」选择框同一行：选择框紧跟标签之后（需求）。
            title_row = QHBoxLayout()
            title_row.setSpacing(8)
            title_row.addWidget(label)
            enable_check = QCheckBox("启用")
            if enable_tooltip:
                enable_check.setToolTip(enable_tooltip)
            enable_check.setChecked(bool(enable_checked))
            setattr(self, enable_attr, enable_check)
            title_row.addWidget(enable_check)
            title_row.addStretch(1)
            lay.addLayout(title_row)
        else:
            lay.addWidget(label)

        row = QHBoxLayout()
        row.setSpacing(8)
        combo = QComboBox()
        _fill_backend_combo(combo, LOCAL_LLM_TYPES)
        combo.setToolTip(
            "MLX-LM：Apple Silicon 文本模型；MLX-VLM：多模态；"
            "ONNX：ONNX Runtime；GGUF：llama.cpp"
        )
        setattr(self, combo_attr, combo)
        edit = QLineEdit(path_value, placeholderText=placeholder)
        setattr(self, path_attr, edit)
        btn = QPushButton("选择")
        btn.clicked.connect(picker)
        setattr(self, btn_attr, btn)

        row.addWidget(QLabel("模型类型"))
        row.addWidget(combo)
        row.addWidget(QLabel("本地路径"))
        row.addWidget(edit, 3)
        row.addWidget(btn)
        lay.addLayout(row)

        # 次级路径行：mmproj（GGUF）/ tokenizer（ONNX），按后端互斥显隐。
        mm_label = QLabel("mmproj 路径")
        mm_edit = QLineEdit(placeholderText="GGUF 多模态投影文件（mmproj，可选）")
        mm_btn = QPushButton("选择")
        mm_btn.clicked.connect(
            lambda: self._on_pick_secondary_path(mm_edit, "mmproj"))
        tok_label = QLabel("tokenizer 路径")
        tok_edit = QLineEdit(placeholderText="ONNX tokenizer 目录（含 tokenizer.json，可选）")
        tok_btn = QPushButton("选择")
        tok_btn.clicked.connect(
            lambda: self._on_pick_secondary_path(tok_edit, "tokenizer"))
        setattr(self, f"{prefix}_mmproj_path_label", mm_label)
        setattr(self, f"{prefix}_mmproj_path_edit", mm_edit)
        setattr(self, f"{prefix}_mmproj_path_btn", mm_btn)
        setattr(self, f"{prefix}_onnx_tokenizer_path_label", tok_label)
        setattr(self, f"{prefix}_onnx_tokenizer_path_edit", tok_edit)
        setattr(self, f"{prefix}_onnx_tokenizer_path_btn", tok_btn)

        sec_row = QHBoxLayout()
        sec_row.setSpacing(8)
        sec_row.addWidget(mm_label)
        sec_row.addWidget(mm_edit, 3)
        sec_row.addWidget(mm_btn)
        sec_row.addWidget(tok_label)
        sec_row.addWidget(tok_edit, 3)
        sec_row.addWidget(tok_btn)
        lay.addLayout(sec_row)

    def _add_sampling_row(self, lay: QVBoxLayout, prefix: str,
                          defaults: Dict[str, Any], buttons=()) -> None:
        """一行采样参数 + 尾部按钮。

        ``prefix`` 为 ``""``（主模型）或 ``"aux_"``（辅助模型）：属性名 =
        前缀 + 基础名，于是 ``temp_edit`` 与 ``aux_temp_edit`` 共用一套处理
        代码，不会出现两套逻辑各自漂移。

        ``buttons`` 每项 ``(文本, 属性名, 动作, 提示)``，动作用
        ``"load:main"`` / ``"unload:aux"`` / ``"apply"`` 表示。
        """
        row = QHBoxLayout()
        row.setSpacing(8)
        for label, base, width, fallback in self._SAMPLING_SPECS:
            # 思考模式 + KV 开关插在采样参数之后（原「超时(秒)」位置）。
            if base == "repeat_penalty_edit":
                self._add_thinking_check(row, prefix, defaults)
                self._add_kv_checks(row, prefix, defaults)
            edit = QLineEdit(str(defaults.get(base, fallback)))
            edit.setFixedWidth(width)
            setattr(self, prefix + base, edit)
            row.addWidget(QLabel(label))
            row.addWidget(edit)
        row.addStretch(1)
        for text, attr, action, tip in buttons:
            btn = QPushButton(text)
            if tip:
                btn.setToolTip(tip)
            btn.clicked.connect(
                lambda _=False, a=action: self._on_model_action(a)
            )
            setattr(self, attr, btn)
            row.addWidget(btn)
        lay.addLayout(row)

    def _add_thinking_check(self, row: QHBoxLayout, prefix: str,
                            defaults: Dict[str, Any]) -> None:
        """在采样行里插入「思考模式」勾选框。

        属性名 = ``prefix + "enable_thinking_check"``，于是主模型
        （``enable_thinking_check``）与辅助模型（``aux_enable_thinking_check``）
        共用同一套读写代码，不会出现两套逻辑各自漂移。

        勾选 → 模板收到 ``enable_thinking=True``，generation prompt 末尾预置
        ``<think>``，模型先思考再作答；不勾 → 传 False，直接作答。
        """
        check = QCheckBox("思考模式")
        check.setToolTip(
            "开启：模型先输出思考过程（显示在「思考过程」面板）再作答，适合复杂任务，\n"
            "但多出一段 reasoning 文本，生成更慢。\n"
            "关闭：直接作答，更快。\n"
            "需要模型的 chat template 支持 enable_thinking；不支持时该参数会被自动忽略。"
        )
        check.setChecked(bool(defaults.get(self._THINKING_CHECK_BASE, True)))
        setattr(self, prefix + self._THINKING_CHECK_BASE, check)
        row.addWidget(check)

    def _add_kv_checks(self, row: QHBoxLayout, prefix: str,
                       defaults: Dict[str, Any]) -> None:
        """在采样行「思考模式」后插入「KV复用」「KV量化」两个勾选框。

        属性名 = ``prefix + 基础名``，主/辅助模型共用同一套读写代码。

        * KV复用（``kv_cache_check``）：跨轮 KV/前缀复用——llama.cpp 挂
          ``LlamaCache``，MLX 用 prompt cache 匹配公共前缀。多轮对话同前缀
          时省掉重复 prefill。
        * KV量化（``kv_q8_check``）：KV cache 量化到 int8——llama.cpp
          ``type_k/type_v = GGML_TYPE_Q8_0``，MLX ``kv_bits=8``。KV 显存减半，
          质量损失可忽略；不支持的架构自动忽略。
        """
        reuse = QCheckBox("KV复用")
        reuse.setToolTip(
            "开启：跨轮 KV/前缀复用——多轮对话的公共前缀不重复 prefill，\n"
            "后续轮次明显变快（llama.cpp LlamaCache / MLX prompt cache）。\n"
            "关闭：每轮完整重新计算，无跨轮状态。"
        )
        reuse.setChecked(bool(defaults.get(self._KV_REUSE_CHECK_BASE, True)))
        setattr(self, prefix + self._KV_REUSE_CHECK_BASE, reuse)
        row.addWidget(reuse)
        q8 = QCheckBox("KV量化")
        q8.setToolTip(
            "开启：KV cache 量化到 int8（llama.cpp type_k/type_v=Q8_0；\n"
            "MLX kv_bits=8），KV 显存减半，质量损失可忽略。\n"
            "关闭：KV 保持 f16。不支持的架构自动忽略该参数。"
        )
        q8.setChecked(bool(defaults.get(self._KV_Q8_CHECK_BASE, True)))
        setattr(self, prefix + self._KV_Q8_CHECK_BASE, q8)
        row.addWidget(q8)

    # ------------------------------------------------------------------
    # wiring
    # ------------------------------------------------------------------
    def _wire_signals(self) -> None:
        self.fetch_btn.clicked.connect(self._on_fetch_models)
        self.connect_btn.clicked.connect(self._on_apply_config)
        self.workspace_btn.clicked.connect(self._on_pick_workspace)
        self.knowledge_base_btn.clicked.connect(self._on_pick_knowledge_base)
        self.llm_model_path_btn.clicked.connect(self._on_pick_llm_model)
        self.aux_llm_model_path_btn.clicked.connect(self._on_pick_aux_llm_model)
        self.embedding_model_path_btn.clicked.connect(self._on_pick_embedding_model)
        self.reranker_model_path_btn.clicked.connect(self._on_pick_reranker_model)
        self.attach_image_btn.clicked.connect(self._on_pick_chat_image)
        self.attach_clear_btn.clicked.connect(self._on_clear_attached_image)
        self.config_window_btn.clicked.connect(self._on_open_config_window)
        self.log_window_btn.clicked.connect(self._on_open_log_window)
        # 模式二选一 + 本地模型类型变化 → 刷新面板可见性与可用性。
        # 切换主模型模式（本地 / OpenAI 兼容）：不仅要切面板可见性，更要让
        # agent 实际使用的后端跟着切——否则「设了本地模式」只是换了显示，
        # 对话仍走旧的 LLMClient（#140 用户反馈的真 bug）。
        self.llm_mode_local_radio.toggled.connect(self._on_mode_changed)
        self.llm_mode_openai_radio.toggled.connect(self._on_mode_changed)
        self.llm_backend_combo.currentTextChanged.connect(self._update_model_controls)
        self.aux_llm_backend_combo.currentTextChanged.connect(self._update_model_controls)
        self.embedding_backend_combo.currentTextChanged.connect(self._update_rag_model_controls)
        self.reranker_backend_combo.currentTextChanged.connect(self._update_model_controls)
        self.reranker_backend_combo.currentTextChanged.connect(self._on_reranker_backend_changed)
        self.reranker_backend_combo.currentTextChanged.connect(self._update_rag_model_controls)
        self.model_combo.activated.connect(self._on_model_combo_activated)
        self.sync_knowledge_btn.clicked.connect(self._on_sync_knowledge)
        self.vector_dedup_btn.clicked.connect(self._on_vector_dedup)
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
        # 模式二选一：本地 / OpenAI 兼容接口。llm_backend 为 openai 时一律
        # 视为接口模式（旧配置迁移后两者应已自洽，见 AgentConfig._migrate）。
        local_mode = getattr(self.config, "llm_mode", "local") != "openai"
        # setChecked 会同步触发 _on_mode_changed；初始化阶段不应触发「重建
        # agent / 自动加载模型」（会让 autoload 跳过失效、并在 _rebuild_agent
        # 之前就重复建后端）。setChecked 是同步发射，包住这两行即可。
        self._suppress_mode_change = True
        self.llm_mode_local_radio.setChecked(local_mode)
        self.llm_mode_openai_radio.setChecked(not local_mode)
        self._suppress_mode_change = False
        # 本地模式下按配置值选中「模型类型」；接口模式下该项被禁用，
        # 仍显示一个合理值（mlx_lm）以免下拉框空白。
        backend_value = self.config.llm_backend
        if backend_value == "openai":
            backend_value = "mlx_lm"
        _select_combo_backend(self.llm_backend_combo, backend_value)
        _select_combo_backend(
            self.aux_llm_backend_combo, getattr(self.config, "aux_llm_backend", "mlx_lm")
        )
        self.llm_model_path_edit.setText(
            self.config.llm_model_path or DEFAULT_MAIN_MODEL_PATH
        )
        self.aux_llm_model_path_edit.setText(
            getattr(self.config, "aux_llm_model_path", "") or DEFAULT_AUX_MODEL_PATH
        )
        # 辅助模型启用开关（默认不勾选）：控制启动自动加载与同步后自动重载。
        self.aux_llm_enable_check.setChecked(
            bool(getattr(self.config, "aux_llm_enabled", False))
        )
        # 次级路径：GGUF→mmproj（复用 llm_vision_projection_path），ONNX→tokenizer。
        self.llm_mmproj_path_edit.setText(
            getattr(self.config, "llm_vision_projection_path", "") or "")
        self.llm_onnx_tokenizer_path_edit.setText(
            getattr(self.config, "llm_onnx_tokenizer_path", "") or "")
        self.aux_llm_mmproj_path_edit.setText(
            getattr(self.config, "aux_llm_vision_projection_path", "") or "")
        self.aux_llm_onnx_tokenizer_path_edit.setText(
            getattr(self.config, "aux_llm_onnx_tokenizer_path", "") or "")
        _select_combo_backend(
            self.embedding_backend_combo, self.config.rag_embedding_backend
        )
        self.embedding_model_edit.setText(self.config.rag_embedding_model)
        self.embedding_model_path_edit.setText(self.config.rag_embedding_model_path)
        self.matryoshka_check.setChecked(
            bool(getattr(self.config, "rag_matryoshka_enabled", True)))
        _mrl_dim = int(getattr(self.config, "rag_matryoshka_dim", 512))
        _mrl_idx = self.matryoshka_dim_combo.findData(_mrl_dim)
        self.matryoshka_dim_combo.setCurrentIndex(
            _mrl_idx if _mrl_idx >= 0 else self.matryoshka_dim_combo.findData(512))
        _dtype = str(getattr(self.config, "rag_vector_dtype", "int8") or "int8").lower()
        self.vector_dtype_combo.setCurrentIndex(
            max(0, self.vector_dtype_combo.findData(_dtype)))
        _select_pdf_engine_radio(self, getattr(self.config, "rag_pdf_engine", "docling"))
        _pc = max(1, min(6, int(getattr(self.config, "rag_pdf_small_concurrency", 1))))
        _pc_idx = self.pdf_concurrency_combo.findData(_pc)
        self.pdf_concurrency_combo.setCurrentIndex(_pc_idx if _pc_idx >= 0 else 0)
        _select_combo_backend(self.ocr_backend_combo,
                              getattr(self.config, "rag_ocr_backend", "mlx"))
        self.ocr_model_path_edit.setText(
            getattr(self.config, "rag_ocr_model_path", "") or DEFAULT_OCR_MODEL_PATH
        )
        self.ocr_mmproj_path_edit.setText(
            getattr(self.config, "rag_ocr_mmproj_path", "") or ""
        )
        self.ocr_tokenizer_path_edit.setText(
            getattr(self.config, "rag_ocr_tokenizer_path", "") or ""
        )
        self._update_ocr_secondary_controls()
        _select_combo_backend(self.reranker_backend_combo, self.config.rag_rerank_backend)
        self.reranker_model_edit.setText(self.config.rag_rerank_model)
        self.reranker_model_path_edit.setText(self._reranker_path_for_backend())
        self.chunk_size_spin.setValue(getattr(self.config, "rag_chunk_size", 500))
        self.chunk_overlap_spin.setValue(getattr(self.config, "rag_chunk_overlap", 50))
        self.semantic_chunk_check.setChecked(
            bool(getattr(self.config, "rag_semantic_chunking", True)))
        self.rerank_topn_spin.setValue(
            int(getattr(self.config, "rag_rerank_top_n", 3)))
        if self.config.model:
            self.model_combo.setEditText(self.config.model)
        self._apply_sampling_to_widgets("")
        self._apply_sampling_to_widgets("aux_")
        self.workspace_edit.setText(self.config.workspace)
        self.knowledge_base_edit.setText(self.config.knowledge_base)
        self._update_model_controls()

    def _apply_sampling_to_widgets(self, prefix: str) -> None:
        """把 config 里的采样参数写回控件。prefix "" = 主模型，"aux_" = 辅助。"""
        cfg = self.config
        values = {
            "max_tokens_edit": getattr(cfg, prefix + "max_tokens"),
            "temp_edit": getattr(cfg, prefix + "temperature"),
            "top_p_edit": getattr(cfg, prefix + "top_p"),
            "min_p_edit": getattr(cfg, prefix + "min_p"),
            "top_k_edit": getattr(cfg, prefix + "top_k"),
            "repeat_penalty_edit": getattr(cfg, prefix + "repetition_penalty"),
            # 超时已从 UI 移除（2026-10-27）：config 字段保留，仅不再回填编辑。
        }
        for base, value in values.items():
            getattr(self, prefix + base).setText(str(value))
        # 三个 QCheckBox 单独写回勾选状态（缺省与 dataclass 保持一致：
        # 思考模式主模型 True / 辅助 False；KV 两项主辅助都默认 True）。
        _think_check = getattr(self, prefix + self._THINKING_CHECK_BASE, None)
        if _think_check is not None:
            _think_check.setChecked(
                bool(getattr(cfg, prefix + "llm_enable_thinking",
                             False if prefix else True))
            )
        _kv_reuse = getattr(self, prefix + self._KV_REUSE_CHECK_BASE, None)
        if _kv_reuse is not None:
            _kv_reuse.setChecked(
                bool(getattr(cfg, prefix + "llama_cpp_kv_cache", True)))
        _kv_q8 = getattr(self, prefix + self._KV_Q8_CHECK_BASE, None)
        if _kv_q8 is not None:
            _kv_q8.setChecked(
                bool(getattr(cfg, prefix + "llama_cpp_kv_q8", True)))

    def _gather_config_from_widgets(self) -> None:
        self.config.base_url = self.base_url_edit.text().strip()
        self.config.api_key = self.api_key_edit.text().strip()
        self.config.model = self.model_combo.currentText().strip()
        # 模式与后端必须一起落盘：接口模式下 llm_backend 恒为 openai（否则
        # 重启后本地路径字段被忽略，用户看到的模式与实际行为相反）。
        local_mode = self.llm_mode_local_radio.isChecked()
        self.config.llm_mode = "local" if local_mode else "openai"
        self.config.llm_backend = (
            _combo_backend_value(self.llm_backend_combo) if local_mode else "openai"
        )
        self.config.llm_model_path = (
            self.llm_model_path_edit.text().strip() or DEFAULT_MAIN_MODEL_PATH
        )
        self.config.aux_llm_backend = _combo_backend_value(self.aux_llm_backend_combo)
        self.config.aux_llm_model_path = (
            self.aux_llm_model_path_edit.text().strip() or DEFAULT_AUX_MODEL_PATH
        )
        # 辅助模型启用开关 → 配置（默认 False：不启用就不自动加载）。
        self.config.aux_llm_enabled = self.aux_llm_enable_check.isChecked()
        # 次级路径：GGUF→mmproj（写回 llm_vision_projection_path），ONNX→tokenizer。
        self.config.llm_vision_projection_path = (
            self.llm_mmproj_path_edit.text().strip()
            or self.config.llm_vision_projection_path
        )
        self.config.llm_onnx_tokenizer_path = (
            self.llm_onnx_tokenizer_path_edit.text().strip()
            or self.config.llm_onnx_tokenizer_path
        )
        self.config.aux_llm_vision_projection_path = (
            self.aux_llm_mmproj_path_edit.text().strip()
            or self.config.aux_llm_vision_projection_path
        )
        self.config.aux_llm_onnx_tokenizer_path = (
            self.aux_llm_onnx_tokenizer_path_edit.text().strip()
            or self.config.aux_llm_onnx_tokenizer_path
        )
        self.config.rag_embedding_backend = (
            _combo_backend_value(self.embedding_backend_combo) or "fastembed"
        )
        self.config.rag_embedding_model = (
            self.embedding_model_edit.text().strip() or self.config.rag_embedding_model
        )
        self.config.rag_matryoshka_enabled = self.matryoshka_check.isChecked()
        _mrl_dim = self.matryoshka_dim_combo.currentData()
        if _mrl_dim:
            self.config.rag_matryoshka_dim = int(_mrl_dim)
        _vdtype = self.vector_dtype_combo.currentData()
        if _vdtype:
            self.config.rag_vector_dtype = str(_vdtype)
        self.config.rag_embedding_model_path = (
            self.embedding_model_path_edit.text().strip() or self.config.rag_embedding_model_path
        )
        self.config.rag_pdf_engine = _checked_pdf_engine(self) or "docling"
        self.config.rag_pdf_small_concurrency = int(
            self.pdf_concurrency_combo.currentData() or 1)
        self._gather_ocr_config_from_widgets()
        self.config.rag_rerank_backend = (
            _combo_backend_value(self.reranker_backend_combo) or "fastembed"
        )
        self.config.rag_rerank_model = (
            self.reranker_model_edit.text().strip() or self.config.rag_rerank_model
        )
        # 切片参数：夹紧非法组合（重叠 ≥ 切片会导致切分不前进）后落 config。
        self.config.rag_chunk_size, self.config.rag_chunk_overlap = (
            _normalize_chunk_params(
                self.chunk_size_spin.value(), self.chunk_overlap_spin.value())
        )
        self.config.rag_semantic_chunking = self.semantic_chunk_check.isChecked()
        self.config.rag_rerank_top_n = self.rerank_topn_spin.value()
        # MLX 交叉编码器是目录，ONNX 是文件，两者各存一份：切换后端时不必
        # 重新粘贴路径（见 _reranker_path_for_backend）。
        path_value = self.reranker_model_path_edit.text().strip()
        if self.config.rag_rerank_backend == "mlx":
            self.config.rag_rerank_mlx_model_path = (
                path_value or self.config.rag_rerank_mlx_model_path
            )
        else:
            self.config.rag_rerank_model_path = (
                path_value or self.config.rag_rerank_model_path
            )
        self._gather_sampling_from_widgets("")
        self._gather_sampling_from_widgets("aux_")
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

    def _gather_ocr_config_from_widgets(self) -> None:
        """从「自定义 OCR 模型」行控件收集 Ovis OCR 配置写入 config。

        单独抽出，供「应用配置」「保存设置（仅 OCR）」两处复用，避免重复逻辑。
        """
        self.config.rag_ocr_backend = (
            _combo_backend_value(self.ocr_backend_combo) or "mlx"
        )
        self.config.rag_ocr_model_path = (
            self.ocr_model_path_edit.text().strip()
            or getattr(self.config, "rag_ocr_model_path", "") or DEFAULT_OCR_MODEL_PATH
        )
        # 模型名自动由「本地路径」最后一段派生（去掉 models/ 前缀），无需用户填写。
        self.config.rag_ocr_model = (
            os.path.basename(self.config.rag_ocr_model_path) or DEFAULT_OCR_MODEL
        )
        # GGUF 模型文件现在统一填在上面的「本地路径」里：选中 GGUF 后端时把
        # 它同步到 rag_ocr_gguf_path（引擎 GGUF 加载器优先读它），其它后端清空。
        self.config.rag_ocr_gguf_path = (
            self.config.rag_ocr_model_path
            if self.config.rag_ocr_backend == "gguf" else ""
        )
        self.config.rag_ocr_mmproj_path = self.ocr_mmproj_path_edit.text().strip()
        self.config.rag_ocr_tokenizer_path = (
            self.ocr_tokenizer_path_edit.text().strip()
        )
        # rag_ocr_vlm_handler / rag_ocr_chat_format 的输入框已随 GGUF 专用行
        # 移除：保留 config 既有值（默认 mtmd / 空），引擎侧继续消费。

    def _gather_sampling_from_widgets(self, prefix: str) -> None:
        """从控件读采样参数写进 config。prefix "" = 主模型，"aux_" = 辅助。

        两套控件同名不同前缀，校验规则完全一致——重复利用同一段代码，
        避免主/辅助两套阈值各自漂移（比如辅助模型的 top_p 忘了夹 (0,1]）。
        """
        def _edit(base: str) -> str:
            return getattr(self, prefix + base).text().strip()

        try:
            # 范围钳制 [0, 2]：负数/超范围的值会被服务端直接拒绝，报错难懂。
            temperature = max(0.0, min(2.0, float(_edit("temp_edit") or "0.7")))
        except ValueError:
            temperature = 0.7
        try:
            # 下限 1：负数/0 会让 LLM 请求直接被服务端拒绝，报错信息难懂。
            max_tokens = max(1, int(_edit("max_tokens_edit") or "32768"))
        except ValueError:
            max_tokens = 32768
        try:
            # top_p ∈ (0, 1]：0 会导致无候选 token 可选，>1 被服务端拒绝。
            top_p = max(0.0001, min(1.0, float(_edit("top_p_edit") or "0.95")))
        except ValueError:
            top_p = 0.95
        try:
            # min_p ∈ [0, 1)：负数无意义且会被拒绝。
            min_p = max(0.0, min(0.999, float(_edit("min_p_edit") or "0.05")))
        except ValueError:
            min_p = 0.05
        try:
            # top_k >= 0：负数会让采样逻辑崩溃或被服务端拒绝。
            top_k = max(0, int(_edit("top_k_edit") or "20"))
        except ValueError:
            top_k = 20
        try:
            # repetition_penalty >= 1：小于 1 会反向惩罚（越重复越鼓励），>2 输出退化。
            repeat = max(1.0, min(2.0, float(_edit("repeat_penalty_edit") or "1.0")))
        except ValueError:
            repeat = 1.0
        # 超时已从 UI 移除（2026-10-27）：config.request_timeout / aux_request_timeout
        # 字段保留原值（后端运行时仍读取），不再从面板收集。

        setattr(self.config, prefix + "temperature", temperature)
        setattr(self.config, prefix + "max_tokens", max_tokens)
        setattr(self.config, prefix + "top_p", top_p)
        setattr(self.config, prefix + "min_p", min_p)
        setattr(self.config, prefix + "top_k", top_k)
        setattr(self.config, prefix + "repetition_penalty", repeat)
        # 思考模式：勾选框 → 布尔。prefix 拼出来的属性名正好对上配置字段
        # （"" → llm_enable_thinking，"aux_" → aux_llm_enable_thinking）。
        _think_check = getattr(self, prefix + self._THINKING_CHECK_BASE, None)
        if _think_check is not None:
            setattr(self.config, prefix + "llm_enable_thinking",
                    _think_check.isChecked())
        # KV cache 开关（"" → llama_cpp_kv_cache / llama_cpp_kv_q8，
        # "aux_" → aux_llama_cpp_kv_cache / aux_llama_cpp_kv_q8）。
        _kv_reuse = getattr(self, prefix + self._KV_REUSE_CHECK_BASE, None)
        if _kv_reuse is not None:
            setattr(self.config, prefix + "llama_cpp_kv_cache",
                    _kv_reuse.isChecked())
        _kv_q8 = getattr(self, prefix + self._KV_Q8_CHECK_BASE, None)
        if _kv_q8 is not None:
            setattr(self.config, prefix + "llama_cpp_kv_q8",
                    _kv_q8.isChecked())

        if not prefix:
            # Keep max_context_tokens in sync with max_tokens: the shrink
            # threshold should track the model's actual context window. If the
            # user has explicitly set a different max_context_tokens via
            # config.json, only override when it was smaller than the new
            # max_tokens (i.e. the old 4096 budget that's now too small).
            if self.config.max_context_tokens < max_tokens:
                self.config.max_context_tokens = max_tokens

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
        # 注意这里**不再无条件 close() 旧 llm**：_ensure_main_llm() 会按签名
        # 判断要不要重建——配置没变就复用手里那个已加载权重的后端。旧实现每次
        # 重建都 close + create，用户点一次「应用配置」就要重新读数 GB 权重。
        # 构造失败时必须清空引用：旧 agent 已在上面被异步关闭，继续持有它会让
        # _on_send 用一套已关闭的 MCP 客户端发起新一轮对话，报出与真实原因无关
        # 的二次错误。
        # 重建前先捕获旧 agent 的后端：重建后若换成不同对象且已加载，必须 close()
        # 掉，否则旧后端的 MLX 权重永远留在 GPU 统一内存（未调 mx.clear_cache()，
        # GC 也不会释放显存）。对话加载路径会留 self._llm=None，此时旧 agent.llm
        # 仍是已加载后端，最容易被孤儿化——这里兜底关闭。
        _old_llm = getattr(self._agent, "llm", None) if self._agent is not None else None
        _old_aux = getattr(self._agent, "aux_llm", None) if self._agent is not None else None
        self._agent = None
        new_llm = None
        new_aux = None
        try:
            new_llm = self._ensure_main_llm()
            # 辅助模型是可选增强：路径不存在 / 运行时缺失时不能把整个 agent 拖垮
            # （摘要退回主模型即可），所以这里单独兜异常。
            try:
                new_aux = self._ensure_aux_llm()
            except Exception as e:
                new_aux = None
                logger.warning("UI: 辅助模型不可用，摘要将退回主模型: %s", e)
            new_agent = Agent(
                llm=new_llm, config=self.config, callbacks=self._callbacks,
                aux_llm=new_aux,
            )
            new_agent.max_iterations = self.config.max_iterations
        except Exception as e:
            # new_llm 已建好但 Agent 构造失败 → 必须关掉它，否则每次失败
            # 泄漏一个 httpx 连接池。
            if new_llm is not None and new_llm is not getattr(self, "_llm", None):
                try:
                    new_llm.close()
                except Exception:
                    pass
            if new_aux is not None and new_aux is not getattr(self, "_aux_llm", None):
                try:
                    new_aux.close()
                except Exception:
                    pass
            self._on_error(f"无法创建 agent: {e}")
            return
        self._llm = new_llm
        self._agent = new_agent
        # 兜底关闭被换下的旧后端（已加载且与新后端不是同一对象才关，避免双关）。
        # 注意：判断依据应是「对象有 close() 且未关闭」，不能依赖后端特有的
        # _model_obj —— LLMClient 没有 _model_obj，旧判断会漏关它、泄漏 httpx
        # 连接池（每次「应用配置」换模型即泄漏一个）。
        for _old in (_old_llm, _old_aux):
            if _old is not None and _old is not new_llm and _old is not new_aux \
                    and getattr(_old, "close", None) is not None \
                    and not getattr(_old, "_closed", False):
                try:
                    _old.close()
                except Exception:
                    pass

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
        # 同步/依赖安装在跑时同样不能重建：shutdown() 会释放进程级共享的
        # OCR/docling 单例和 RAG 引擎，而同步线程可能正在用它们；同步结束后
        # 的 reload_rag() 也会作用到新 agent 上，结果不可预期。
        if (self._sync_busy or self._is_worker_running(self._sync_worker)
                or self._is_worker_running(self._deps_worker)
                or self._is_worker_running(self._dedup_worker)):
            QMessageBox.warning(
                self, "正在同步",
                "知识库同步/依赖安装/向量去重进行中，请先停止或等待其完成，再应用配置。",
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
        local_mode = self.llm_mode_local_radio.isChecked()
        # 面板级切换：本地模式显示「模型类型+本地路径」两套（主/辅），
        # 接口模式显示 Base URL / API Key / 模型下拉。
        self.local_mode_panel.setVisible(local_mode)
        self.openai_mode_panel.setVisible(not local_mode)
        self.fetch_btn.setEnabled(not local_mode)
        self.fetch_btn.setText("获取模型" if not local_mode else "扫描本地")

        backend = _combo_backend_value(self.llm_backend_combo)
        aux_backend = _combo_backend_value(self.aux_llm_backend_combo)
        # GGUF 是文件、其余是目录：文件对话框的过滤器与「需要文件」的校验
        # 都依赖这个区分（见 _on_pick_llm_model / resolve_model_path）。
        self.llm_model_path_edit.setPlaceholderText(
            "本地主模型 .gguf 文件" if backend == "llama_cpp"
            else "本地主模型目录或 .onnx/.gguf 文件"
        )
        self.aux_llm_model_path_edit.setPlaceholderText(
            "本地辅助模型 .gguf 文件" if aux_backend == "llama_cpp"
            else "本地辅助模型目录或 .onnx/.gguf 文件"
        )
        # 多模态后端才能附加图片；mlx_lm 是纯文本。
        multimodal = local_mode and backend in ("mlx_vlm", "onnx", "llama_cpp")
        self.attach_image_btn.setEnabled(multimodal)
        # 切到不支持视觉的后端时必须丢弃待发图片：否则 [image:...] 标记会以
        # 纯文本形式喂给纯文本模型，用户看到的是模型对着路径名胡编。
        if not multimodal and self._pending_image_path:
            self._set_attached_image("")
        self._update_secondary_path_controls("llm")
        self._update_secondary_path_controls("aux_llm")
        self._update_rag_model_controls()
        self._refresh_model_buttons()

    def _update_secondary_path_controls(self, prefix: str) -> None:
        """按后端显隐主/辅助模型的次级路径框：GGUF→mmproj，ONNX→tokenizer。

        MLX / OpenAI 不显示——它们要么不需要（MLX 多模态走 GGUF 体系外的另一套
        投影字段），要么没有这一项。两个次级框互斥：同一时刻只有一个可见。
        """
        combo = getattr(self, f"{prefix}_backend_combo")
        backend = _combo_backend_value(combo)
        is_gguf = backend == "llama_cpp"
        is_onnx = backend == "onnx"
        for lbl_attr, edit_attr, btn_attr, show in (
            (f"{prefix}_mmproj_path_label", f"{prefix}_mmproj_path_edit",
             f"{prefix}_mmproj_path_btn", is_gguf),
            (f"{prefix}_onnx_tokenizer_path_label", f"{prefix}_onnx_tokenizer_path_edit",
             f"{prefix}_onnx_tokenizer_path_btn", is_onnx),
        ):
            getattr(self, lbl_attr).setVisible(show)
            getattr(self, edit_attr).setVisible(show)
            getattr(self, btn_attr).setVisible(show)

    def _update_ocr_secondary_controls(self, *_args) -> None:
        """按 OCR 后端显隐自定义 OCR 的次级路径框（与主/辅助模型同一套交互）。

        注意 OCR 的下拉值与 LLM 不同：GGUF 后端的值是 ``"gguf"``（LLM 是
        ``"llama_cpp"``）。GGUF → mmproj（必填才能读图）；ONNX → tokenizer
        目录；MLX 两个都不显示。同一时刻只有一个可见。
        """
        backend = _combo_backend_value(self.ocr_backend_combo)
        is_gguf = backend == "gguf"
        is_onnx = backend == "onnx"
        for lbl, edit, btn, show in (
            (self.ocr_mmproj_path_label, self.ocr_mmproj_path_edit,
             self.ocr_mmproj_path_btn, is_gguf),
            (self.ocr_tokenizer_path_label, self.ocr_tokenizer_path_edit,
             self.ocr_tokenizer_path_btn, is_onnx),
        ):
            lbl.setVisible(show)
            edit.setVisible(show)
            btn.setVisible(show)

    def _update_rag_model_controls(self, *_args) -> None:
        """按后端显隐 Embedding/Reranker 的「模型名」与「本地路径」字段。

        语义：fastembed 后端用 HuggingFace 模型名下载，无本地路径；mlx/onnx/
        llama_cpp 本地后端用本地路径，模型名只是兜底标签、对用户无意义。两套
        字段互斥显隐，用户看到的恰好是「选后端 + 填那一个标识符」，不再出现
        一个填了也不生效的死字段（用户反馈：名字到底能不能改、改了干嘛）。
        """
        emb_fastembed = _combo_backend_value(self.embedding_backend_combo) == "fastembed"
        rk_fastembed = _combo_backend_value(self.reranker_backend_combo) == "fastembed"
        for name_lbl, name_edit, path_lbl, path_edit, path_btn, is_fastembed in (
            (self.embedding_name_label, self.embedding_model_edit,
             self.embedding_path_label, self.embedding_model_path_edit,
             self.embedding_model_path_btn, emb_fastembed),
            (self.reranker_name_label, self.reranker_model_edit,
             self.reranker_path_label, self.reranker_model_path_edit,
             self.reranker_model_path_btn, rk_fastembed),
        ):
            name_lbl.setVisible(is_fastembed)
            name_edit.setVisible(is_fastembed)
            path_lbl.setVisible(not is_fastembed)
            path_edit.setVisible(not is_fastembed)
            path_btn.setVisible(not is_fastembed)

    def _on_mode_changed(self, checked: bool) -> None:
        # 初始化阶段 _apply_config_to_widgets 的 setChecked 也会触发本函数，
        # 此时不应重建 agent / 自动加载模型（见 _apply_config_to_widgets）。
        if getattr(self, "_suppress_mode_change", False):
            return
        if not checked:
            return
        # 主模型模式被切换（本地 ↔ OpenAI 兼容接口）。
        # 关键修复（#140 用户反馈）：切换模式必须让 agent 实际使用的后端跟着
        # 变，否则「设了本地模式」只是换了面板显示，对话仍走旧的 LLMClient。
        # 两个 radio 都会发 toggled，但只有「被选中」的那次 checked=True 需要
        # 处理，另一个 radio 发的是 False，已在上面 return 忽略，避免一次点击
        # 触发两次重建。
        # 先把模式写进 config 并持久化：否则重启又回到旧模式（迁移逻辑会按
        # llm_backend 反推，OpenAI 用户永远不会「切到本地」生效）。
        self._gather_config_from_widgets()
        try:
            self.config.save(CONFIG_PATH)
        except Exception as e:  # pragma: no cover - 防御
            logger.warning("UI: 保存模式失败: %s", e)
        self._update_model_controls()
        if self._busy:
            # 对话进行中无法安全重建 agent：标记延迟到本轮结束（_on_finished）
            # 再应用，避免把正在跑的 worker 的后端换掉导致崩溃。
            self._pending_mode_sync = True
            self.status_label.setText(
                "对话结束后将切换到" + ("本地模型" if self.llm_mode_local_radio.isChecked() else "OpenAI 接口")
            )
            return
        self._apply_mode_sync()

    def _apply_mode_sync(self) -> None:
        """按当前选中的模式，让 agent 的后端真正生效。"""
        if self.llm_mode_local_radio.isChecked():
            # 本地模式：agent 改用本地后端（懒加载，下次对话即落地本地模型）。
            # 若本地模型尚未读进内存，按「默认加载」的用户预期自动开始加载——
            # 与启动自动加载一致，也避免「设了本地模式却还在走旧后端」的困惑。
            # 测试环境（BASE_AGENT_SKIP_MODEL_AUTOLOAD=1）只切后端不真正读数
            # GB 权重，否则套件会在每个切换模式的用例上卡死。
            self._rebuild_agent()
            if not self._is_model_loaded("main") and not os.environ.get(AUTOLOAD_SKIP_ENV):
                self._on_load_model("main")
        else:
            # OpenAI 兼容接口：轻量切回 LLMClient（无本地权重可载）。
            self._rebuild_agent()

    def _refresh_model_buttons(self) -> None:
        """按「是否已加载 / 是否可操作」刷新加载/卸载按钮。

        辅助模型的按钮**不受主模型模式影响**：辅助模型只支持本地直载，主模型
        选 OpenAI 接口时它照样要能加载/卸载（见 _build_config_widgets 里它放在
        模式面板之外的注释）。

        模型加载成功后，加载按钮文案变「重新加载」、背景变绿，直观表示「模型
        已就绪可用」（见用户反馈）。
        """
        # 模型就绪用的绿色高亮样式。
        GREEN = "background-color:#22c55e;color:#ffffff;"
        local_mode = self.llm_mode_local_radio.isChecked()
        for role, load_btn, unload_btn in (
            ("main", self.llm_load_btn, self.llm_unload_btn),
            ("aux", self.aux_llm_load_btn, self.aux_llm_unload_btn),
        ):
            loaded = self._is_model_loaded(role)
            if not local_mode and role == "main":
                # 接口模式下主模型是网络端点，没有「本地权重」可加载/卸载：
                # 按钮无意义，禁用并改文案，避免误导用户以为本地模型已就绪。
                load_btn.setEnabled(False)
                unload_btn.setEnabled(False)
                load_btn.setText("接口模式")
                unload_btn.setText("—")
                load_btn.setStyleSheet("")
                unload_btn.setStyleSheet("")
                continue
            # 主模型在接口模式下没有本地模型可载；辅助模型永远可以操作。
            # 同步期间禁用「加载」：同步不需要 LLM，且要避免模型在同步中途被
            # 加载进 GPU 把内存推高（与「同步前自动卸载」配合，保证同步期模型
            # 彻底不在 GPU）。卸载按钮保留可用——中途卸载正是我们想要的。
            usable = (local_mode or role == "aux") and not self._model_busy.get(
                role, False) and not self._sync_busy
            load_btn.setEnabled(usable)
            unload_btn.setEnabled(usable)
            load_btn.setText("重新加载" if loaded else "加载模型")
            # 已加载：背景变绿，代表模型可用；未加载：恢复默认样式。
            load_btn.setStyleSheet(GREEN if loaded else "")
            load_btn.setToolTip(
                f"把当前配置的{'主' if role == 'main' else '辅助'}模型加载到内存备用"
                if not loaded
                else "模型已加载，点击可按当前配置重新加载"
            )

    @staticmethod
    def _dialog_start(value: str) -> str:
        candidate = os.path.expanduser(value or "")
        if os.path.isfile(candidate):
            return os.path.dirname(candidate)
        if os.path.isdir(candidate):
            return candidate
        return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "..", "models")

    def _on_pick_llm_model(self) -> None:
        backend = _combo_backend_value(self.llm_backend_combo)
        start = self._dialog_start(self.llm_model_path_edit.text())
        if backend == "llama_cpp":
            path, _ = QFileDialog.getOpenFileName(self, "选择 GGUF 主模型", start, "GGUF (*.gguf);;所有文件 (*)")
        elif backend == "onnx":
            path, _ = QFileDialog.getOpenFileName(self, "选择 ONNX 主模型", start, "ONNX (*.onnx);;所有文件 (*)")
        else:
            path = QFileDialog.getExistingDirectory(self, "选择本地主模型目录", start)
        if path:
            self.llm_model_path_edit.setText(path)

    def _on_pick_aux_llm_model(self) -> None:
        backend = _combo_backend_value(self.aux_llm_backend_combo)
        start = self._dialog_start(self.aux_llm_model_path_edit.text())
        if backend == "llama_cpp":
            path, _ = QFileDialog.getOpenFileName(self, "选择 GGUF 辅助模型", start, "GGUF (*.gguf);;所有文件 (*)")
        elif backend == "onnx":
            path, _ = QFileDialog.getOpenFileName(self, "选择 ONNX 辅助模型", start, "ONNX (*.onnx);;所有文件 (*)")
        else:
            path = QFileDialog.getExistingDirectory(self, "选择本地辅助模型目录", start)
        if path:
            self.aux_llm_model_path_edit.setText(path)

    def _on_pick_embedding_model(self) -> None:
        backend = _combo_backend_value(self.embedding_backend_combo)
        start = self._dialog_start(self.embedding_model_path_edit.text())
        if backend == "llama_cpp":
            path, _ = QFileDialog.getOpenFileName(self, "选择 GGUF Embedding 模型", start, "GGUF (*.gguf);;所有文件 (*)")
        elif backend == "onnx":
            path, _ = QFileDialog.getOpenFileName(self, "选择 ONNX Embedding 模型", start, "ONNX (*.onnx);;所有文件 (*)")
        else:
            path = QFileDialog.getExistingDirectory(self, "选择 Embedding 模型目录", start)
        if path:
            self.embedding_model_path_edit.setText(path)

    def _on_ocr_save_settings(self) -> None:
        """「保存设置」：仅把「自定义 OCR 模型」行的配置写入并落盘 config。

        不重建整个 agent（OCR 配置在知识库同步时由 worker 实时读取），落盘后
        下次同步即可生效。比起「应用配置」更轻量，也不会因其它未保存字段而改动。
        """
        if self._busy:
            QMessageBox.warning(
                self, "正在运行",
                "Agent 正在运行，请先点击「终止对话」停止当前任务，再保存配置。",
            )
            return
        if self._sync_busy or self._is_worker_running(self._sync_worker):
            QMessageBox.warning(
                self, "正在同步",
                "知识库同步进行中，请先等待其完成，再保存 OCR 配置。",
            )
            return
        self._gather_ocr_config_from_widgets()
        try:
            self.config.save(CONFIG_PATH)
        except Exception as e:  # pragma: no cover - 防御
            QMessageBox.critical(self, "保存失败", f"写入配置文件失败：{e}")
            return
        self.status_label.setText("自定义 OCR 模型设置已保存并应用")
        QMessageBox.information(
            self, "已保存",
            "「自定义 OCR 模型」设置已保存。\n"
            "下次知识库同步将使用此处配置的 Ovis OCR 模型。",
        )

    def _on_ocr_test(self) -> None:
        """「识别测试」：弹出独立 OCR 测试窗口（依赖 RAG 解析组件）。"""
        try:
            from .ocr_test_window import OcrTestWindow
        except Exception as e:  # noqa: BLE001 - 缺 RAG 依赖时优雅提示
            logger.warning("OCR 测试窗口导入失败: %s", e)
            QMessageBox.critical(
                self, "不可用",
                "OCR 测试窗口需要 RAG 解析依赖（PyMuPDF / ocrmac / OvisOCR3 等）。\n"
                "请确认已安装对应依赖后再试。",
            )
            return
        existing = getattr(self, "_ocr_test_window", None)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return
        try:
            # 不传 parent：做成真正独立的顶层窗口，可自由拖动/缩放，
            # 避免 macOS 上作为主窗口子面板贴附、无法移动且界面叠乱。
            self._ocr_test_window = OcrTestWindow(self.config)
            self._ocr_test_window.show()
            # macOS 上新建窗口仅靠 show() 不一定提到最前/可能落到别的桌面，
            # 显式 raise + activate 保证可见。
            self._ocr_test_window.raise_()
            self._ocr_test_window.activateWindow()
        except Exception as e:  # noqa: BLE001 - 构造/显示失败给出可见提示而非静默吞掉
            logger.exception("OCR 测试窗口创建失败: %s", e)
            self._ocr_test_window = None
            QMessageBox.critical(
                self, "窗口创建失败",
                f"OCR 识别测试窗口启动失败：\n{e}",
            )

    def _on_convert_tool(self) -> None:
        """「转换工具」：弹出独立的文件转 Markdown 工具窗口。"""
        try:
            from .convert_tool_window import ConvertToolWindow
        except Exception as e:  # noqa: BLE001 - 缺 RAG 依赖时优雅提示
            logger.warning("转换工具窗口导入失败: %s", e)
            QMessageBox.critical(
                self, "不可用",
                "转换工具需要 RAG 解析依赖（PyMuPDF / docling / pandoc 等）。\n"
                "请确认已安装对应依赖后再试。",
            )
            return
        existing = getattr(self, "_convert_tool_window", None)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return
        self._convert_tool_window = ConvertToolWindow(self)
        self._convert_tool_window.show()

    def _on_select_ocr_model_path(self) -> None:
        """选择 OCR 模型：ONNX 选 .onnx 文件、GGUF 选 .gguf 文件、MLX 选目录。"""
        backend = _combo_backend_value(self.ocr_backend_combo)
        start = self._dialog_start(self.ocr_model_path_edit.text())
        if backend == "onnx":
            path, _ = QFileDialog.getOpenFileName(
                self, "选择 ONNX 模型", start, "ONNX (*.onnx);;所有文件 (*)")
        elif backend == "gguf":
            path, _ = QFileDialog.getOpenFileName(
                self, "选择 GGUF OCR 模型", start, "GGUF (*.gguf);;所有文件 (*)")
        else:
            path = QFileDialog.getExistingDirectory(self, "选择 OvisOCR3 模型目录", start)
        if path:
            self.ocr_model_path_edit.setText(path)

    def _on_select_ocr_mmproj_path(self) -> None:
        """选择 GGUF 多模态投影文件（mmproj/clip）。"""
        start = self._dialog_start(self.ocr_mmproj_path_edit.text())
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 mmproj 投影文件", start,
            "mmproj (*.mmproj);;GGUF (*.gguf);;所有文件 (*)")
        if path:
            self.ocr_mmproj_path_edit.setText(path)

    def _reranker_path_for_backend(self) -> str:
        """ONNX 是文件、MLX 是目录，两者在 config 里各存一份。"""
        backend = _combo_backend_value(self.reranker_backend_combo) or "fastembed"
        if backend == "mlx":
            return getattr(self.config, "rag_rerank_mlx_model_path", "")
        return getattr(self.config, "rag_rerank_model_path", "")

    def _on_reranker_backend_changed(self, *_args) -> None:
        backend = _combo_backend_value(self.reranker_backend_combo)
        placeholder = {
            "mlx": "本地 MLX Reranker 目录",
            "gguf": "本地 GGUF Reranker 文件（含 rerank 头的交叉编码器）",
        }.get(backend, "本地 Reranker .onnx 文件")
        self.reranker_model_path_edit.setPlaceholderText(placeholder)
        self.reranker_model_path_edit.setText(self._reranker_path_for_backend())

    def _on_pick_reranker_model(self) -> None:
        start = self._dialog_start(self.reranker_model_path_edit.text())
        backend = _combo_backend_value(self.reranker_backend_combo)
        if backend == "mlx":
            path = QFileDialog.getExistingDirectory(self, "选择 MLX Reranker 目录", start)
        elif backend == "gguf":
            path, _ = QFileDialog.getOpenFileName(
                self, "选择 GGUF Reranker 模型", start, "GGUF (*.gguf);;所有文件 (*)"
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择 ONNX Reranker 模型", start, "ONNX (*.onnx);;所有文件 (*)"
            )
        if path:
            self.reranker_model_path_edit.setText(path)

    def _on_pick_secondary_path(self, edit: "QLineEdit", kind: str) -> None:
        """次级路径选择器：mmproj 选文件，tokenizer 选目录。"""
        start = self._dialog_start(edit.text())
        if kind == "mmproj":
            path, _ = QFileDialog.getOpenFileName(
                self, "选择 mmproj 投影文件", start, "mmproj (*.mmproj);;所有文件 (*)")
        else:
            path = QFileDialog.getExistingDirectory(self, "选择 ONNX tokenizer 目录", start)
        if path:
            edit.setText(path)

    # ------------------------------------------------------------------
    # secondary windows
    # ------------------------------------------------------------------
    def _on_open_config_window(self) -> None:
        window = getattr(self, "config_window", None)
        if window is None:
            return
        window.show()
        window.raise_()
        window.activateWindow()

    def _on_open_log_window(self) -> None:
        window = getattr(self, "log_window", None)
        if window is None:
            return
        window.show()
        window.raise_()
        window.activateWindow()

    # ------------------------------------------------------------------
    # task plan panel
    # ------------------------------------------------------------------
    _PLAN_ICONS = {
        "pending": "○",
        "in_progress": "▶",
        "done": "✓",
        "blocked": "✕",
    }
    _PLAN_COLORS = {
        "pending": "#6b7280",
        "in_progress": "#2563eb",
        "done": "#059669",
        "blocked": "#dc2626",
    }

    def _poll_task_plan(self) -> None:
        """读取 agent 写入的 task_plan.json 并刷新右上面板（仅在变化时重绘）。"""
        if getattr(self, "_closed", False):
            return
        try:
            from ..planner import load_plan

            plan = load_plan(self.config.workspace)
        except Exception:
            return
        html = self._render_plan_html(plan)
        if html == self._plan_rendered:
            return
        self._plan_rendered = html
        self.plan_view.setHtml(html)
        if plan is None:
            self.plan_status_label.setText("无计划")
        else:
            done = sum(1 for s in plan.steps if s.status == "done")
            self.plan_status_label.setText(f"{done}/{len(plan.steps)} 完成")

    def _render_plan_html(self, plan) -> str:
        """把计划渲染成带状态标记与目标/进度标题的 HTML。"""
        if plan is None:
            return '<div style="color:#9ca3af">（当前任务未制定计划）</div>'
        goal = self._escape(getattr(plan, "goal", "") or "")
        steps = list(getattr(plan, "steps", []) or [])
        done = sum(1 for s in steps if s.status == "done")
        lines = [
            f'<div style="font-weight:600;margin-bottom:4px">目标：{goal}</div>',
            f'<div style="color:#6b7280;margin-bottom:6px">'
            f'进度 {done}/{len(steps)}</div>',
        ]
        current_marked = False
        for i, step in enumerate(steps, 1):
            status = step.status if step.status in self._PLAN_ICONS else "pending"
            text = self._escape(step.text or "")
            note = self._escape(step.note or "")
            icon = self._PLAN_ICONS[status]
            color = self._PLAN_COLORS[status]
            # 正在进行的步骤：加粗 + 底色标记（只标第一个，避免多个
            # in_progress 时整屏高亮）。注意 QTextDocument 的 HTML 导入只认
            # background-color，简写 background 会被丢掉。
            highlight = ""
            if status == "in_progress" and not current_marked:
                current_marked = True
                highlight = "font-weight:600;background-color:#eff6ff;"
            suffix = f'<span style="color:#6b7280"> — {note}</span>' if note else ""
            lines.append(
                f'<div style="{highlight}color:{color};padding:1px 2px">'
                f'{icon} {i}. {text}{suffix}</div>'
            )
        if steps and done == len(steps):
            lines.append('<div style="color:#059669;margin-top:6px">全部步骤已完成</div>')
        return "".join(lines)

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

    def _on_model_combo_activated(self, _index: int) -> None:
        # 本地模式下「模型」下拉是扫描本地目录的结果，选中即回填本地路径。
        if self.llm_mode_openai_radio.isChecked():
            return
        value = self.model_combo.currentText().strip()
        if os.path.exists(os.path.expanduser(value)):
            self.llm_model_path_edit.setText(value)

    # ------------------------------------------------------------------
    # 本地模型加载 / 卸载（#140）
    # ------------------------------------------------------------------
    def _model_signature(self, role: str) -> tuple:
        """当前配置对应的模型签名。

        签名变化 = 用户换了模型类型或路径 → 手里那个后端对象已经过期，必须
        重建。用它做「要不要重建」的判据，比每次都比较一堆字段可靠。
        """
        if role == "aux":
            return ("aux", _combo_backend_value(self.aux_llm_backend_combo),
                    self.aux_llm_model_path_edit.text().strip())
        if self.llm_mode_openai_radio.isChecked():
            return ("openai", self.base_url_edit.text().strip(),
                    self.model_combo.currentText().strip())
        return ("local", _combo_backend_value(self.llm_backend_combo),
                self.llm_model_path_edit.text().strip())

    def _loaded_model(self, role: str):
        return self._llm if role == "main" else self._aux_llm

    def _is_model_loaded(self, role: str) -> bool:
        """已把权重读进内存（不只是建了懒加载对象）。"""
        llm = self._loaded_model(role)
        if llm is None or getattr(llm, "_closed", False):
            return False
        # 本地后端的 _model_obj/_llama 在权重读入后才有值；LLMClient 没有
        # 权重可载，视为始终就绪。
        return not hasattr(llm, "_ensure_loaded") or any(
            getattr(llm, attr, None) is not None
            for attr in ("_model_obj", "_llama", "_session")
        )

    # ------------------------------------------------------------------
    # 闲时章节摘要调度器（见 src/agent/rag/book_summary.py）
    # ------------------------------------------------------------------
    def _is_idle_for_summary(self) -> bool:
        """调度器用的「闲时」判定：未退出、无对话在进行、无知识同步在进行。"""
        if getattr(self, "_closing", False):
            return False
        if getattr(self, "_busy", False):
            return False
        if getattr(self, "_sync_active", False):
            return False
        return True

    def _start_book_summary_scheduler(self) -> None:
        """启动（或复用）闲时章节摘要后台线程。"""
        sched = getattr(self, "_summary_scheduler", None)
        if sched is not None and sched.is_alive():
            return
        try:
            from ..rag.book_summary import BookSummaryScheduler

            self._summary_scheduler = BookSummaryScheduler(
                engine_getter=lambda: getattr(self, "_engine", None),
                config=self.config,
                aux_getter=lambda: getattr(self, "_aux_llm", None),
                is_idle=self._is_idle_for_summary,
            )
            self._summary_scheduler.start()
            logger.info("UI: 闲时章节摘要调度器已启动")
        except Exception as e:  # pragma: no cover - 启动失败不致命
            logger.warning("UI: 闲时章节摘要调度器启动失败（忽略）: %s", e)

    def _stop_book_summary_scheduler(self) -> None:
        sched = getattr(self, "_summary_scheduler", None)
        if sched is not None and sched.is_alive():
            sched.stop()

    def _on_model_action(self, action: str) -> None:
        """加载模型 / 卸载 / 应用配置 的统一入口。"""
        if action == "apply":
            self._on_apply_config()
            return
        verb, _, role = action.partition(":")
        if verb == "load":
            self._on_load_model(role)
        elif verb == "unload":
            self._on_unload_model(role)

    def _on_load_model(self, role: str) -> None:
        """把当前配置的模型加载到内存备用（后台线程，避免 UI 冻结）。"""
        if self._model_busy.get(role, False):
            QMessageBox.information(
                self, "请稍候", "正在处理另一个模型操作。"
            )
            return
        if self._busy:
            QMessageBox.warning(
                self, "正在运行",
                "Agent 正在运行，请先点击「终止对话」停止当前任务，再加载模型。",
            )
            return
        if (self._sync_busy or self._is_worker_running(self._sync_worker)
                or self._is_worker_running(self._dedup_worker)):
            QMessageBox.warning(
                self, "正在同步", "知识库同步/向量去重进行中，请先停止或等待完成。")
            return
        # 先把控件里的值收进 config：加载必须用「用户刚填的」参数，而不是上
        # 一次应用配置留下的旧值——否则改了路径/采样参数却加载了旧模型。
        self._gather_config_from_widgets()
        if role == "main" and self.config.llm_mode == "openai":
            QMessageBox.information(
                self, "接口模式", "OpenAI 兼容接口模式无需加载本地模型，请直接「应用配置」。"
            )
            return
        signature = self._model_signature(role)
        label = "主模型" if role == "main" else "辅助模型"
        self._model_busy[role] = True
        self._refresh_model_buttons()
        self.status_label.setText(f"正在加载{label}…")
        self._append_log(f"[model] 开始加载{label}: {signature[2] or '(未填路径)'}")
        worker = LoadModelWorker(
            role=role, signature=signature, config=self.config
        )
        self._model_workers[role] = worker
        worker.load_done.connect(self._on_model_loaded)
        worker.load_failed.connect(self._on_model_load_failed)
        self._retain_worker(worker)
        worker.start()

    def _unload_role(self, role: str) -> bool:
        """真正卸载某角色的 LLM 后端：close() 释放权重 + 换回惰性后端。

        返回 True 表示确有模型被卸载；未加载则返回 False。被「卸载」按钮与
        「同步前自动卸载」共用，避免卸载逻辑被重复粘贴。

        **必须查 agent 侧后端，不能只查 UI 缓存 self._llm**：用户若仅通过「对话」
        触发模型懒加载（chat_stream → _ensure_loaded），权重装在 self._agent.llm
        上，而 self._llm 仍是 None（只有「加载」按钮回调 _on_model_loaded 会写
        self._llm）。若只查 self._llm，同步起点卸载会误判「未加载」而漏掉真正
        装着 ~11GB 权重的 agent.llm，导致同步期 GPU 仍被模型占满。
        """
        label = "主模型" if role == "main" else "辅助模型"
        # 优先取 agent 上真正持有权重的后端；UI 缓存仅作兜底。
        llm = None
        if self._agent is not None:
            llm = getattr(self._agent, "llm" if role == "main" else "aux_llm", None)
        if llm is None or getattr(llm, "_closed", False):
            llm = self._loaded_model(role)
        if llm is None or getattr(llm, "_closed", False):
            return False
        try:
            llm.close()
        except Exception as e:  # pragma: no cover - 防御
            logger.warning("UI: %s 卸载异常: %s", label, e)
        if role == "main":
            self._llm = None
            self._llm_signature = None
            if self._agent is not None:
                self._agent.llm = self._ensure_main_llm()
        else:
            self._aux_llm = None
            self._aux_llm_signature = None
            if self._agent is not None:
                self._agent.aux_llm = self._ensure_aux_llm()
        self._refresh_model_buttons()
        # 卸载后草稿模型引用必须同步摘掉，否则主模型会拿着已关闭的后端去投机解码。
        self._wire_draft_model()
        self._append_log(f"[model] {label} 已卸载")
        logger.info("UI: %s 已卸载", label)
        return True

    def _on_unload_model(self, role: str) -> None:
        """「卸载」按钮：卸载已加载的模型，释放内存。"""
        if self._model_busy.get(role, False):
            QMessageBox.information(self, "请稍候", "正在处理另一个模型操作。")
            return
        if self._busy:
            QMessageBox.warning(
                self, "正在运行",
                "Agent 正在运行，请先点击「终止对话」停止当前任务，再卸载模型。",
            )
            return
        label = "主模型" if role == "main" else "辅助模型"
        if self._unload_role(role):
            self.status_label.setText(f"{label}已卸载")
        else:
            QMessageBox.information(self, "未加载", f"{label}尚未加载。")

    def _on_model_loaded(self, payload: dict) -> None:
        """后台加载完成：换上热后端，并让正在用的 agent 指向它。"""
        role = payload.get("role", "main")
        self._model_busy[role] = False
        if not any(self._model_busy.values()):
            self._model_autoloading = False
        llm = payload.get("llm")
        signature = payload.get("signature")
        label = "主模型" if role == "main" else "辅助模型"
        try:
            if llm is None:
                raise LLMError("加载返回了空后端")
            # 加载期间用户可能又改了指令：签名变了就别装上去，免得装错模型。
            if signature != self._model_signature(role):
                try:
                    llm.close()
                except Exception:
                    pass
                self.status_label.setText(f"{label}配置已变更，未采用本次加载结果")
                self._append_log(f"[model] {label} 加载期间配置已变更，结果丢弃")
                return
            previous = self._loaded_model(role)
            if previous is not None and previous is not llm:
                try:
                    previous.close()
                except Exception:
                    pass
            if role == "main":
                self._llm = llm
                self._llm_signature = signature
                if self._agent is not None:
                    self._agent.llm = llm
            else:
                self._aux_llm = llm
                self._aux_llm_signature = signature
                if self._agent is not None:
                    self._agent.aux_llm = llm
            self._append_log(f"[model] {label} 已加载: {payload.get('detail', '')}")
            self.status_label.setText(f"{label}已加载")
            # 主/辅助任一加载完成都要重新接线草稿模型（投机解码）。
            self._wire_draft_model()
            if role == "main":
                # 后台预热 system+tools 前缀：让用户第一次输入直接命中 KV 缓存
                # （否则首条要全量 prefill system + 工具 schema，约 20s）。
                # 放在状态栏写入之后：预热会把状态改成「正在预热模型…」，
                # 若反过来会被「主模型已加载」盖掉，用户就看不出还没就绪。
                self._request_warmup()
        except Exception as e:
            QMessageBox.critical(self, f"{label}加载失败", f"{type(e).__name__}: {e}")
            self.status_label.setText(f"{label}加载失败")
        finally:
            self._refresh_model_buttons()

    def _wire_draft_model(self) -> None:
        """把已加载的辅助模型挂成主模型的草稿模型（投机解码）。

        不额外加载任何东西：辅助模型本来就在内存里（RAG 浓缩 / 事实抽取也用它）。
        两者必须同词表（现在默认主/辅助都是 qwen3.5，vocab 248320）；不匹配时后端
        会在生成时自行禁用投机解码。
        """
        main_llm = getattr(self, "_llm", None)
        aux_llm = getattr(self, "_aux_llm", None)
        setter = getattr(main_llm, "set_draft_model", None)
        if not callable(setter):
            return
        try:
            if aux_llm is not None and not getattr(aux_llm, "_closed", False):
                setter(aux_llm)
            else:
                setter(None)
        except Exception as e:  # pragma: no cover - 接线失败不影响对话
            logger.debug("UI: 挂载草稿模型失败（忽略）: %s", e)

    def _on_model_load_failed(self, role: str, detail: str) -> None:
        self._model_busy[role] = False
        if not any(self._model_busy.values()):
            self._model_autoloading = False
        label = "主模型" if role == "main" else "辅助模型"
        self.status_label.setText(f"{label}加载失败")
        self._append_log(f"[model] {label} 加载失败: {detail}")
        # 启动自动加载失败不弹窗：开机就弹模态框很吵，用户也还没看到界面。
        if not self._model_autoloading:
            QMessageBox.critical(self, f"{label}加载失败", detail)
        self._refresh_model_buttons()

    def _ensure_main_llm(self):
        """取主模型后端：配置没变就复用手里那个（权重保持热），变了才重建。

        复用是关键：MLX/GGUF 的权重加载要数秒到数十秒，如果每次
        ``_rebuild_agent`` 都新建一个，用户点一次「应用配置」就得重新读一遍
        数 GB 权重。
        """
        signature = self._model_signature("main")
        current = getattr(self, "_llm", None)
        if current is not None and not getattr(current, "_closed", False):
            if getattr(self, "_llm_signature", None) == signature:
                return current
            try:
                current.close()
            except Exception:
                pass
        self._llm = create_llm_backend(self.config)
        self._llm_signature = signature
        return self._llm

    def _ensure_aux_llm(self):
        """取辅助模型后端（语义同 ``_ensure_main_llm``）。"""
        signature = self._model_signature("aux")
        current = getattr(self, "_aux_llm", None)
        if current is not None and not getattr(current, "_closed", False):
            if getattr(self, "_aux_llm_signature", None) == signature:
                return current
            try:
                current.close()
            except Exception:
                pass
        self._aux_llm = create_aux_llm_backend(self.config)
        self._aux_llm_signature = signature
        return self._aux_llm

    def _autoload_default_models(self) -> None:
        """启动时把默认模型加载到内存（#140 用户要求：软件启动即默认加载）。

        在后台线程做：权重加载要数秒到数十秒，放进 ``__init__`` 会让窗口半天
        弹不出来。测试环境通过 ``BASE_AGENT_SKIP_MODEL_AUTOLOAD=1`` 跳过
        （见 tests/conftest.py）——否则套件会真的去读数 GB 权重。
        """
        if os.environ.get(AUTOLOAD_SKIP_ENV):
            return
        self._model_autoloading = True
        try:
            if not self._is_model_loaded("main"):
                self._on_load_model("main")
            # 辅助模型默认不启用（aux_llm_enabled=False）：启动**不加载**，
            # 省下 ~1GB 统一内存与启动时间。勾选「启用」后才会在这里加载。
            if (
                getattr(self.config, "aux_llm_enabled", False)
                and not self._is_model_loaded("aux")
            ):
                self._on_load_model("aux")
        finally:
            # 同步启动的 worker 会在事件循环里跑完，这里先复位标志会让失败
            # 弹窗——但只要 worker 还没结束就把标志留着，由 _on_model_loaded
            # 在最后一个 worker 完成时统一复位。
            if not any(self._model_busy.values()):
                self._model_autoloading = False

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

    # ------------------------------------------------------------------
    # 向量去重（「向量去重」按钮）
    # ------------------------------------------------------------------

    def _on_vector_dedup(self) -> None:
        """扫描向量库清理重复向量：按文本内容分组，保留入库最早的一份。

        与同步互斥（两者都开同一个 LanceDB 目录）；agent 在跑时也禁止
        （检索会并发读库）。删除不可逆，先弹确认框。
        """
        if self._busy:
            QMessageBox.warning(
                self, "正在运行",
                "Agent 正在运行，请先点击「终止对话」停止当前任务，再做向量去重。",
            )
            return
        if self._is_worker_running(self._sync_worker):
            QMessageBox.warning(
                self, "正在同步", "知识库同步正在进行，请先等待或停止同步。")
            return
        if self._is_worker_running(self._dedup_worker):
            logger.warning("Vector dedup already running, ignoring duplicate request")
            return
        kb = self.knowledge_base_edit.text().strip()
        ws = self.workspace_edit.text().strip() or os.path.expanduser("~/.base-agent/workspace")
        if not kb:
            QMessageBox.warning(self, "缺少知识库", "请先选择知识库目录")
            return
        ret = QMessageBox.question(
            self, "向量去重",
            "将扫描向量库中的全部向量，按文本内容找出完全相同的重复项，\n"
            "保留入库最早的一份，其余删除（不可恢复）。\n\n"
            "重复向量不影响正确性，但会让同一内容在检索中重复命中（噪声）。\n"
            "确定继续吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return
        logger.info("Vector dedup triggered: kb=%s workspace=%s", kb, ws)
        # 忙碌态：去重期间禁用同步/去重按钮与模型加载按钮（避免中途把 LLM
        # 装进 GPU 与去重抢内存）。
        self.vector_dedup_btn.setEnabled(False)
        self.vector_dedup_btn.setText("去重中…")
        self.sync_knowledge_btn.setEnabled(False)
        self._refresh_model_buttons()
        self.status_label.setText("正在扫描向量库…")
        self.progress.setRange(0, 0)
        self._append_log("[rag] ────────── 开始向量去重 ──────────")
        # 与同步同纪律：临时禁用 QSignalLogHandler 的 RAG 日志过滤，避免并发
        # 写 log_view；RAG 日志在去重期间由 _log_buffer + QTimer 处理。
        if hasattr(self, '_log_handler') and self._log_handler is not None:
            self._log_handler._sync_active = True
        self._dedup_worker = VectorDedupWorker(ws, kb)
        self._dedup_worker.dedup_progress.connect(self._on_dedup_progress)
        self._dedup_worker.dedup_finished.connect(self._on_dedup_finished)
        self._dedup_worker.dedup_failed.connect(self._on_dedup_failed)
        # deleteLater 必须挂内建 finished（run() 返回后触发），挂自定义信号会在
        # run() 执行途中销毁线程对象（同 SyncKnowledgeWorker 的教训）。
        self._retain_worker(self._dedup_worker)
        # 日志轮询定时器：100ms 消费 _log_buffer，实时显示去重日志。
        self._log_poll_timer.start()
        self._dedup_worker.start()

    def _reset_dedup_ui(self) -> None:
        """结束去重忙碌态：恢复按钮文案/可用性。"""
        self.vector_dedup_btn.setText("向量去重")
        # 不能无条件解锁：agent 在跑时（self._busy）必须保持禁用。
        self.vector_dedup_btn.setEnabled(not self._busy)
        self.sync_knowledge_btn.setEnabled(not self._busy)
        self._refresh_model_buttons()

    def _on_dedup_progress(self, done: int, total: int, current: str) -> None:
        if done < 0:
            # 纯日志/状态刷新信号（不更新进度条）
            self.status_label.setText(current)
            return
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)
        self.status_label.setText(current)

    def _on_dedup_finished(self, stats: dict) -> None:
        logger.info("UI: _on_dedup_finished called")
        self._log_poll_timer.stop()
        if hasattr(self, '_log_handler') and self._log_handler is not None:
            self._log_handler._sync_active = False
        self._poll_dedup_logs()
        self._reset_dedup_ui()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self._dedup_worker = None
        # 刷新 agent 的 RAG 引擎表句柄：LanceDB 表对象打开时取版本快照，
        # 去重删了行之后不重新打开，检索还会看到旧数据。
        if self._agent is not None:
            try:
                self._agent.tools.reload_rag()
            except Exception as e:
                logger.warning("UI: reload_rag after dedup: %s", e)
        if not isinstance(stats, dict):
            stats = {}
        removed = stats.get("removed", 0)
        groups = stats.get("groups", 0)
        cleaned = stats.get("sources_cleaned", 0)
        errors = stats.get("errors", [])
        if removed > 0:
            summary = (
                f"向量去重完成：清理 {removed} 条重复向量\n"
                f"（{groups} 组重复，{cleaned} 个来源的内容已整体保留自其它来源）"
            )
        else:
            summary = "向量库扫描完成：未发现重复向量，无需清理。"
        if errors:
            summary += f"\n\n⚠ {len(errors)} 个批次失败（详见日志）"
        self.status_label.setText("向量去重完成")
        self._append_log(f"[rag] {summary.splitlines()[0]}")
        QMessageBox.information(self, "向量去重", summary)

    def _on_dedup_failed(self, error: str) -> None:
        logger.error("UI: _on_dedup_failed: %s", error)
        self._log_poll_timer.stop()
        if hasattr(self, '_log_handler') and self._log_handler is not None:
            self._log_handler._sync_active = False
        self._poll_dedup_logs()
        self._reset_dedup_ui()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self._dedup_worker = None
        self.status_label.setText("向量去重失败")
        self._append_log(f"[rag] ❌ 向量去重失败: {error}")
        QMessageBox.warning(self, "向量去重失败", f"向量去重失败：\n{error}")

    def _poll_dedup_logs(self) -> None:
        """去重结束时最后消费一次 _log_buffer，确保残余日志显示出来。"""
        worker = self._dedup_worker
        if worker is None:
            return
        try:
            with worker._log_buffer_lock:
                if not worker._log_buffer:
                    return
                buf = worker._log_buffer
                worker._log_buffer = []
            cursor = self.log_view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText("\n" + "\n".join(buf))
            self._scroll_to_bottom(self.log_view)
        except Exception:
            pass

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
        # 先把控件值收进 config（含「参数设置」行的切片大小/重叠）：同步构造
        # SyncKnowledgeWorker 时直接读 self.config.rag_chunk_size/_overlap，
        # 不 gather 就会用到上次「应用配置」的旧值——用户改了切片参数却不点
        # 应用配置直接同步，新参数会静默失效。
        self._gather_config_from_widgets()
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
        # 同步期间禁用模型「加载」按钮，避免主/辅助 LLM 在同步中途被装入 GPU。
        self._refresh_model_buttons()
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
        self._retain_worker(self._deps_worker)
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
        self.sync_knowledge_btn.setText("开始同步")
        self.stop_sync_btn.setEnabled(False)
        # 同步结束：恢复模型「加载」按钮（同步期间被禁用，避免中途加载 LLM）。
        self._refresh_model_buttons()

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
        # 同步期间无法对话，先把主/辅助 LLM 彻底卸载：二者都是 MLX 模型，
        # 权重常驻 GPU 统一内存（约 11GB 的 IOAccelerator），与同步进程叠加
        # 会把内存推到 13GB+。Embedding 模型同步必需（vectorize 要用）则保留。
        # 卸载后 agent 换回惰性后端，下次对话时自动重载，无需在此恢复。
        _unloaded = []
        _unloaded_roles = []
        _aux_enabled = bool(getattr(self.config, "aux_llm_enabled", False))
        for _role in ("main", "aux"):
            if self._unload_role(_role):
                _unloaded.append("主" if _role == "main" else "辅助")
                # 辅助模型未启用时照常卸载释放内存，但**不记入重载名单**：
                # 同步结束后不得把它重新拉起（需求：同步链路检查启用开关）。
                if _role == "aux" and not _aux_enabled:
                    continue
                _unloaded_roles.append(_role)
        # 记录「同步前确实已加载」的角色：同步结束后只重载这些，避免把用户事先
        # 手动卸载的模型在同步后擅自重新加载回来（见 _reload_models_after_sync）。
        self._sync_preloaded_roles = set(_unloaded_roles)
        if _unloaded:
            self._append_log(f"[rag] 同步前已卸载{'/'.join(_unloaded)}对话模型，释放 GPU 显存")
            self.status_label.setText("已卸载对话模型，正在同步知识库…")
        # 稳健兜底：gc 扫描所有「已加载」的 LLM 后端实例并强制 close()。
        # 定向卸载（self._agent.llm / self._llm）查不到的孤儿后端——典型是
        # _rebuild_agent 把旧后端换下、或某个仍存活的 AgentWorker 线程持有——会
        # 一直占着 GPU 统一内存（同步期 +11GB）。只能靠 gc 扫描命中并关掉。
        # 只扫 LLM 后端类，绝不碰同步必需的 embedding 模型。
        try:
            import gc as _gc
            from agent.model_backends import (
                MLXLanguageBackend, MLXVLMBackend, ONNXLanguageBackend, LlamaCppBackend)
            _closed_extra = 0
            # 模型加载 worker 在途时跳过扫描：worker 手里正拿着一个刚建好、
            # 正在读权重的后端，此刻 close() 会把在途加载的资源收掉（表现为
            # 「加载模型」随机失败）。此时新建的后端被 worker 引用着，不是孤儿，
            # 也不会被扫描误杀的对象逻辑覆盖。
            # 守卫不能只看 _model_busy 标志：它在 _on_model_loaded（自定义信号
            # load_done 的槽）里就被复位，而那时 LoadModelWorker 的线程**还没跑完**
            # —— 标志已 False、线程仍在读权重，扫描就会把 worker 手里刚建好的后端
            # 当成「孤儿」close() 掉，表现为「加载模型随机失败」，重则在原生层
            # 与正在 mmap 读权重的线程竞态。故必须同时确认线程真的空闲。
            _gc_scan_safe = (
                not any(self._model_busy.values())
                and not any(
                    _w is not None and _w.isRunning()
                    for _w in self._model_workers.values()
                )
            )
            for _o in _gc.get_objects() if _gc_scan_safe else ():
                if isinstance(_o, (MLXLanguageBackend, MLXVLMBackend,
                                   ONNXLanguageBackend, LlamaCppBackend)):
                    if getattr(_o, "_model_obj", None) is not None and not getattr(_o, "_closed", False):
                        try:
                            _o.close()
                            _closed_extra += 1
                        except Exception as _e:
                            logger.debug("同步前 gc 清理 LLM 后端异常: %s", _e)
            # 重置 UI/agent 侧 LLM 引用为全新惰性后端（下次对话自动重载）
            self._llm = None
            self._aux_llm = None
            self._llm_signature = None
            self._aux_llm_signature = None
            if self._agent is not None:
                self._agent.llm = self._ensure_main_llm()
                self._agent.aux_llm = self._ensure_aux_llm()
            self._refresh_model_buttons()
            if _closed_extra:
                logger.info("UI: 同步前 gc 清理 %d 个游离 LLM 后端，释放 GPU 显存", _closed_extra)
        except Exception as _e:
            logger.debug("同步前 gc 清理 LLM 后端失败(忽略): %s", _e)
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
            ocr_backend=getattr(self.config, "rag_ocr_backend", "mlx"),
            ocr_model=getattr(self.config, "rag_ocr_model", ""),
            ocr_model_path=getattr(self.config, "rag_ocr_model_path", ""),
            ocr_gguf_path=getattr(self.config, "rag_ocr_gguf_path", ""),
            ocr_mmproj_path=getattr(self.config, "rag_ocr_mmproj_path", ""),
            ocr_tokenizer_path=getattr(self.config, "rag_ocr_tokenizer_path", ""),
            ocr_vlm_handler=getattr(self.config, "rag_ocr_vlm_handler", ""),
            ocr_chat_format=getattr(self.config, "rag_ocr_chat_format", ""),
            docling_dynamic_scale=getattr(
                self.config, "rag_docling_dynamic_scale", False),
            docling_vlm=getattr(self.config, "rag_docling_vlm", False),
            # PDF 大小限流并发：小文件并发路数 / 大文件阈值 MB。
            pdf_small_concurrency=getattr(
                self.config, "rag_pdf_small_concurrency", 2),
            pdf_large_threshold_mb=getattr(
                self.config, "rag_pdf_large_threshold_mb", 20.0),
            pdf_batch_threshold_mb=getattr(
                self.config, "rag_pdf_batch_threshold_mb", 50.0),
            pdf_batch_pages=getattr(
                self.config, "rag_pdf_batch_pages", 50),
            model_options={
                "embedding_backend": self.config.rag_embedding_backend,
                "embedding_model_path": self.config.rag_embedding_model_path,
                "embedding_threads": getattr(self.config, "rag_embedding_threads", 6),
                "embedding_batch_size": self.config.rag_embedding_batch_size,
                "embedding_max_length": self.config.rag_embedding_max_length,
                "chunk_size": self.config.rag_chunk_size,
                "chunk_overlap": self.config.rag_chunk_overlap,
                "semantic_chunking": getattr(self.config, "rag_semantic_chunking", True),
                "rerank_backend": self.config.rag_rerank_backend,
                "rerank_model_path": self.config.rag_rerank_model_path,
                # MLX 交叉编码器是目录，与 ONNX 文件路径分开存；漏转发会让
                # RAGEngine 用空 mlx 路径 → create_reranker_backend 抛错 →
                # 静默回退 FastEmbedReranker（加载错误的 ~570MB 模型，
                # MLX 精排路线完全失效，2026-10 审查修复）。
                "rerank_mlx_model_path": getattr(
                    self.config, "rag_rerank_mlx_model_path", ""),
                "rerank_threads": self.config.onnx_threads,
                "rerank_batch_size": self.config.rag_rerank_batch_size,
                "rerank_max_length": self.config.rag_rerank_max_length,
                "rerank_enabled": self.config.rag_rerank_enabled,
                "min_similarity": getattr(
                    self.config, "rag_rerank_min_similarity", 0.55),
                "binary_coarse_enabled": getattr(
                    self.config, "rag_binary_coarse_enabled", False),
                # Matryoshka 向量裁剪：enabled 开 + dim∈{128,256,512} →
                # 截维+重归一化+int8 入库；关 → 全维 f16。
                "matryoshka_enabled": bool(
                    getattr(self.config, "rag_matryoshka_enabled", True)),
                "matryoshka_dim": getattr(self.config, "rag_matryoshka_dim", 512),
                "vector_dtype": getattr(self.config, "rag_vector_dtype", "int8"),
            },
        )
        self._sync_worker.sync_finished.connect(self._on_sync_finished)
        self._sync_worker.sync_failed.connect(self._on_sync_failed)
        self._sync_worker.sync_progress.connect(self._on_sync_progress)
        # finished 时 worker 可能已被 _on_sync_finished 置 None——用默认参数
        # 在 connect 时快照引用，保证兜底槽拿到真正的 worker。
        self._sync_worker.finished.connect(
            lambda w=self._sync_worker: self._on_sync_thread_finished(w))
        self._retain_worker(self._sync_worker)
        self._sync_worker.start()
        # 启动日志轮询定时器：每 100ms 消费 _log_buffer，实时显示同步日志
        self._log_poll_timer.start()

    def _reload_models_after_sync(self) -> None:
        """同步结束后在后台重新加载对话模型，并预热主模型。

        同步起点（_on_sync_knowledge）已把主/辅助 LLM 卸载以释放统一内存；同步
        资源（docling / reranker / RapidOCR / OCR VLM 等）在 SyncKnowledgeWorker
        的 finally 里已彻底释放，此刻重载 LLM 不会与任何重资源叠加（避免资源冲突）。

        只重载「同步前确实已加载」的角色（self._sync_preloaded_roles）：若用户事先
        手动卸载了某角色，不应在同步后擅自把它重新加载回来。
        辅助模型未启用（aux_llm_enabled=False）时一律跳过——即使因竞态出现在
        名单里也不得拉起（双保险，名单在 _on_sync_knowledge 已过滤）。
        主模型加载完成后由 _on_model_loaded 自动预热 system+tools 前缀，使同步后
        第一次对话直接命中 KV 缓存、无需再付数十秒加载 + prefill。

        走后台线程（LoadModelWorker），不阻塞 UI；已在加载 / 已加载的角色跳过。
        （停止同步走的也是本函数：_on_stop_sync 只置取消标志，收尾统一在
        _on_sync_finished/_on_sync_failed → _reload_models_after_sync。）
        """
        roles = getattr(self, "_sync_preloaded_roles", set())
        if not roles:
            return
        for role in roles:
            if role == "aux" and not bool(
                getattr(self.config, "aux_llm_enabled", False)
            ):
                continue  # 未启用的辅助模型不自动重载
            if self._model_busy.get(role, False):
                continue  # 用户正在手动加载/卸载该角色，别抢
            # 同步起点卸载后该角色应是惰性后端；若已加载（异常）则不重复加载。
            existing = self._loaded_model(role)
            if existing is not None and not getattr(existing, "_closed", False):
                continue
            signature = self._model_signature(role)
            # OpenAI 接口模式没有「本地权重」可加载（仅主模型）。
            if role == "main" and self.config.llm_mode == "openai":
                continue
            label = "主模型" if role == "main" else "辅助模型"
            self._model_busy[role] = True
            self._refresh_model_buttons()
            self.status_label.setText(f"正在重新加载{label}…")
            self._append_log(
                f"[model] 同步结束，重新加载{label}: {signature[2] or '(未填路径)'}")
            worker = LoadModelWorker(
                role=role, signature=signature, config=self.config)
            self._model_workers[role] = worker
            worker.load_done.connect(self._on_model_loaded)
            worker.load_failed.connect(self._on_model_load_failed)
            self._retain_worker(worker)
            worker.start()

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
            # 同步资源已释放：在后台重新加载对话模型并预热主模型（释放的统一内存
            # 此刻可回收给 LLM，不会与任何重资源叠加）。只重载「同步前已加载」的
            # 角色；用户中途手动卸载的角色不会被动重新加载。
            self._reload_models_after_sync()
            # 防御：stats 可能不是 dict（极端情况下为 None）
            if not isinstance(stats, dict):
                stats = {}
            if stats.get("cancelled"):
                self.status_label.setText("同步已取消（已完成部分已保留，下次从剩余文件继续）")
                self._append_log("[rag] 同步已用户取消：已完整入库的文件已记录，"
                                "下次同步将从剩余文件继续")
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
            # 向量去重 worker 用另一组属性名（_dedup_tmp_*）
            handler = getattr(worker, "_dedup_tmp_handler", None)
            loggers = getattr(worker, "_dedup_tmp_loggers", None)
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
        恢复为"开始同步"状态。如果按钮仍处于同步相关状态，说明同步结束信号
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
                # 信号丢失说明正常结束回调没跑：这里补一次对话模型重载 + 预热。
                self._reload_models_after_sync()
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
            # 同步起点卸载的对话模型，失败也要恢复（只重载同步前已加载的角色）。
            self._reload_models_after_sync()
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
        backend = (
            "openai" if self.llm_mode_openai_radio.isChecked()
            else _combo_backend_value(self.llm_backend_combo)
        )
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
        self._retain_worker(self._fetch_worker)
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
    def _ensure_agent_worker(self) -> AgentWorker:
        """取常驻的对话线程：首次创建，之后**复用同一条**。

        为什么必须复用（MLX 的硬约束）：prompt cache（KV 前缀复用）与 stream
        是**线程局部**的。旧实现每条消息 `AgentWorker(...)` 新建一个 QThread，
        第二条消息就在新线程里复用上一轮的缓存，MLX 在当前线程的 stream 表里
        找不到建缓存那个线程的 stream，直接抛
            RuntimeError: There is no Stream(gpu, N) in current thread
        表现为「第一条消息正常、第二条必崩」。

        后端已有跨线程作废兜底（model_backends 的 _kv_thread_ident），但那等于
        每轮都把缓存丢掉重建——多轮对话的提速收益就没了。让所有对话跑在同一条
        **常驻**线程上（AgentWorker.run 循环，不每条消息退出），缓存跨轮才真正
        命中。注意：QThread 重新 start() 会跑在**新的 OS 线程**上，所以绝不能
        每条消息 start 一次——只在首次创建时 start，之后仅 set_input 唤醒。
        """
        worker = self._worker
        if worker is None:
            # 只在首次（或 closeEvent 关掉之后）创建并启动一次。
            worker = AgentWorker(self._agent, self._callbacks)
            self._worker = worker
            # 换新线程 → 上一轮线程建的 KV 缓存必须作废：OS 线程终止后
            # threading.get_ident() 会被回收复用，新线程可能拿到**同一个
            # ident** 而被后端误判成同线程，复用已失效的缓存照样崩。这里
            # 显式失效，与后端 _kv_thread_ident 检查构成双保险。
            self._invalidate_llm_kv_caches()
            worker.warmup_done.connect(self._on_warmup_done)
            # 常驻线程同样登记保活：它跑满应用生命周期，是最可能在退出时仍在运行
            # 的那个线程（见 _worker_refs 的注释）。
            self._retain_worker(worker)
            worker.start()
            return worker
        # agent 可能已被 _rebuild_agent 换掉（改配置 / 换模型）：必须换上新的，
        # 否则新一轮跑的还是旧 agent（旧模型、旧工具）。
        worker.set_agent(self._agent)
        return worker

    def _request_warmup(self, allow_busy: bool = False) -> None:
        """让常驻对话线程预填 system+tools 的 KV 前缀（不生成、不写历史）。

        预热跑在常驻线程上，保证缓存与后续生成同线程；用户此时若发消息，输入
        优先，预热留到空闲再做。

        ``allow_busy=True`` 用于「缓存已被作废但当前正在跑一轮」的场景
        （上下文压缩 / 清空对话会把历史改写、缓存整份失效）。这类事件发生在
        一轮对话**内部**，所以不能按空闲过滤掉请求——否则用户下一条消息又要
        全量 prefill。常驻 worker 会在当前轮跑完后才执行预热，不会打断正在
        进行的对话。
        """
        try:
            if self._agent is None:
                return
            # 主模型权重还没读进来就别预热：prime_prefix_cache 会先 _ensure_loaded()，
            # 等于在后台偷偷触发一次完整模型加载（几 GB、几十秒）。「清空对话」
            # 这类入口在模型加载完成前被点到时尤其不能这么做。
            if not self._is_model_loaded("main"):
                return
            if self._busy and not allow_busy:
                return
            worker = self._ensure_agent_worker()
            worker.request_warmup()
            self._warmup_in_progress = True
            if not self._busy:
                # 忙的时候不记时：预热要等本轮结束才跑，现在起算的耗时没有
                # 意义（会把自己等待对话的那段时间算进去）。
                self._warmup_started = time.monotonic()
                self.status_label.setText("正在预热模型…（结束后显示「就绪」）")
            self._append_log("[warmup] 后台预热 system+tools 前缀…")
        except Exception as e:  # pragma: no cover - 预热失败不影响对话
            logger.debug("UI: 预热请求失败（忽略）: %s", e)

    def _on_warmup_done(self) -> None:
        elapsed: Optional[float] = None
        if self._warmup_started is not None:
            elapsed = max(0.0, time.monotonic() - self._warmup_started)
            self._warmup_started = None
        self._warmup_in_progress = False
        suffix = f"（{elapsed:.1f}s）" if elapsed is not None else ""
        self._append_log(f"[warmup] 预热完成，首条输入可直接命中缓存{suffix}")
        # 只在「模型仍在且空闲」时改成就绪，避免盖掉「正在生成…」「已卸载」等
        # 更贴近当下的事实。
        if not self._busy and not self._sync_busy and self._llm is not None:
            self.status_label.setText(f"就绪{suffix}")
        # 预热 prefill 会一次性申请大量临时缓冲，完了立刻归还系统。
        self._release_mlx_scratch_cache()

    def _invalidate_llm_kv_caches(self) -> None:
        """失效主/辅助模型的跨轮 KV 缓存（换线程、切模型等场景）。

        后端不一定实现该方法（HTTP 后端没有 KV 缓存），用 getattr 防御。
        """
        for backend in (getattr(self, "_llm", None), getattr(self, "_aux_llm", None)):
            fn = getattr(backend, "invalidate_kv_cache", None)
            if callable(fn):
                try:
                    fn()
                except Exception as e:  # 绝不能让失效动作影响发送
                    logger.debug("UI: invalidate_kv_cache 失败（忽略）: %s", e)

    def _on_send(self) -> None:
        if self._busy:
            return
        # 常驻 worker 线程始终 isRunning(True)（等待下一条消息），所以这里
        # **不能**再用 `_is_worker_running(self._worker)` 判「上一轮未结束」——
        # 那会永远拒绝发送。忙碌状态统一由 self._busy 表示（_on_send 置位、
        # _on_finished 复位），且 _on_finished 只在本轮 agent.run 返回后触发，
        # 届时 worker 已回到等待态，发送是安全的。
        # 同步知识库时禁止对话：两个线程会并发读写同一个 LanceDB 目录，
        # 且同步结束的 reload_rag() 会把 agent 正在用的表句柄置空重开。
        if (self._sync_busy or self._is_worker_running(self._sync_worker)
                or self._is_worker_running(self._dedup_worker)):
            QMessageBox.information(
                self, "正在同步", "知识库同步/向量去重进行中，请等待完成或先停止。"
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
        # 常驻线程：不再每条消息新建（见 _ensure_agent_worker）。
        # 注意这里**不能**再连 finished → deleteLater：线程要留着复用。
        self._finish_handled = False
        worker = self._ensure_agent_worker()
        # 常驻线程：首次已在 _ensure_agent_worker 里 start()，之后只唤醒。
        worker.set_input(text)

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
        """定时轮询 sync_worker / dedup_worker 的 _log_buffer，追加到 log_view。

        绕过 Qt queued signal 机制——ONNX Runtime 在 embedding 期间持有
        GIL，sync_progress 信号虽然是 queued 但主线程可能无法及时处理。
        用 QTimer 直接在主线程事件循环中轮询 buffer，只要 GIL 短暂释放
        （_embed_chunks 中的 sleep(0.01)），定时器回调就能执行。

        注意：本方法只负责日志区域的输出，不更新底部状态栏。
        状态栏由 _on_sync_progress 根据进度回调单独更新，避免重复。
        """
        try:
            # 同步与向量去重共用轮询：谁在跑就消费谁的 _log_buffer。
            worker = self._sync_worker or self._dedup_worker
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
        self.kb_apply_btn.setEnabled(not busy)
        self.clear_btn.setEnabled(not busy)
        self.workspace_btn.setEnabled(not busy)
        self.knowledge_base_btn.setEnabled(not busy)
        # 模型加载/卸载按钮（#140）：与「应用配置」同纪律——agent 在跑时禁用，
        # 否则重建/换后端会让正在执行的 AgentWorker 撞上已关闭的后端。
        self._refresh_model_buttons()
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

    # 结果冗长、对当前对话没有增量价值的工具：内容只进日志，聊天区最多留一行
    # 「已完成」摘要。典型是 rag_status——它会把知识库目录、切片数、整份已索引
    # 文件清单一次性打出来（实测 80 个文件 ~3000 字符），对话里既挤满屏幕也白白
    # 占据上下文；用户真正关心的内容由随后的 rag_search 结果给出。
    _LOG_ONLY_TOOLS = frozenset({"rag_status"})

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
        self._append_log(f"[tool-start] {name} {pretty}")
        if name in self._LOG_ONLY_TOOLS:
            # 聊天区连调用行也不写：整个调用只留结尾一行摘要，避免「有头无尾」。
            return
        self._append_chat(f'<p class="t">🔧 <b>{self._escape(name)}</b>({self._escape(pretty)})…</p>')

    def _on_tool_end(self, name: str, result: Any) -> None:
        out = result.to_message() if hasattr(result, "to_message") else str(result)
        ok = getattr(result, "success", True)
        self._append_log(f"[tool-end] {name} success={ok} :: {out[:4000]}")
        if name in self._LOG_ONLY_TOOLS:
            # 失败必须让用户直接看见；成功只留一行「完成」——状态内容本身
            # 对当前对话没有增量价值（后面的 rag_search 结果才是要看的）。
            if ok:
                self._append_chat(
                    f'<p class="t">↳ <b>{self._escape(name)}</b> → 完成</p>'
                )
            else:
                self._append_chat(
                    f'<p class="sys">↳ <b>{self._escape(name)}</b> → {self._escape(out[:200])}</p>'
                )
            return
        # 长输出保留头 + 尾：报错摘要、最终状态几乎总在末尾，只截开头会让
        # 用户（和模型）看不到关键信息。完整内容进日志面板。
        if len(out) > 400:
            snippet = f"{out[:260]}\n  …（省略 {len(out) - 380} 字符）…\n{out[-120:]}"
        else:
            snippet = out
        cls = "t" if ok else "sys"
        self._append_chat(f'<p class="{cls}">↳ <b>{self._escape(name)}</b> → {self._escape(snippet)}</p>')
        # task_plan 变更后立即刷新右上面板：不依赖 600ms 轮询，避免「新计划写盘了
        # 但面板还停在旧计划 / 多次更新只看到最后一帧」的问题（#计划面板）。轮询
        # 仍作为兜底。
        if name == "task_plan":
            self._poll_task_plan()

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
        # 先把攒下的流式文本落地：否则最后一小段要等到 100ms 定时器才显示，
        # 而本轮已经结束了。
        self._flush_stream_buffer()
        worker = self._worker
        if worker is None and not self._busy:
            return
        # 幂等：agent.run() 有多条路径调用 on_finished()，worker 的 finally 还会
        # 再发一次。旧实现靠「self._worker 已被置 None 且 _busy 已 False」判重；
        # 改成常驻 worker 后 _worker 不再置空（要留给下一条消息复用），改用显式
        # 标志，由 _on_send 在启动时复位。
        if self._finish_handled:
            return
        self._finish_handled = True
        # 这里**不能**置空 self._worker：它是常驻线程，下一条消息继续复用
        # （见 _ensure_agent_worker）。
        self._busy = False
        self._streaming_assistant = False
        self._set_controls_busy(False)
        # 同步进行中不要让「就绪」覆盖掉同步进度文案。
        if not self._sync_busy:
            self.status_label.setText("就绪")
        # 对话进行中切换了主模型模式：现在 agent 空闲，应用那次延迟的模式切换。
        if self._pending_mode_sync and not self._sync_busy:
            self._pending_mode_sync = False
            self._apply_mode_sync()
        # 常驻 worker 每条消息后**不会**结束（run() 回到等待态），所以这里
        # 不能 join（会等一个永不返回的线程），也不能 schedule_worker_join；
        # 引用一直保留在 self._worker，下一条消息只 set_input 唤醒。
        # 现在流式已结束，可以安全弹出技能建议对话框。
        self._show_pending_skill_dialog()
        # 一轮结束（agent 空闲）→ 把 MLX 的可复用临时缓冲归还系统。否则
        # prefill/生成期间攒下的 buffer 会一直占着不释放，一路顶到统一内存
        # 上限（用户实测 13G → 系统换页卡顿）。只清不再引用的缓存数组，模型
        # 权重和 KV 前缀缓存不受影响。
        self._release_mlx_scratch_cache()

    def _release_mlx_scratch_cache(self) -> None:
        """空闲时释放 MLX 的临时缓冲（mx.clear_cache()）。

        同步/检索进行中不清理：那边正在按形状复用 buffer，清掉等于白扔一次
        预热，而且它自己有按子批清理的节奏。
        """
        if self._sync_busy:
            return
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception as e:  # pragma: no cover - 清理失败不影响对话
            logger.debug("UI: mx.clear_cache() 失败（忽略）: %s", e)

    def _retain_worker(self, worker) -> None:
        """登记 worker 保活，并在其线程**确认结束**后自动解除登记。

        这是消除 `QThread: Destroyed while thread '' is still running` qFatal 的
        统一入口（原理见 ``self._worker_refs`` 的注释）。用法：每个 worker
        ``start()`` 之前调一次，并把原先的 ``finished.connect(worker.deleteLater)``
        换成由本方法连接的兜底槽。
        """
        if worker is None:
            return
        with self._worker_refs_lock:
            if worker not in self._worker_refs:
                self._worker_refs.append(worker)

        def _on_finished():
            # finished 已发 == run() 已返回；但 OS 线程要再走几步才真正终止，
            # 这里再等一小会儿，确保 isRunning() 归 False 后才解除保活。
            try:
                if not worker.wait(2000) and worker.isRunning():
                    logger.warning(
                        "UI: worker %s 在 finished 后仍未终止，继续保活引用（"
                        "宁可泄漏也不能让 Qt 销毁运行中的 QThread）",
                        type(worker).__name__)
                    return
            except RuntimeError:
                pass
            with self._worker_refs_lock:
                try:
                    self._worker_refs.remove(worker)
                except ValueError:
                    pass

        worker.finished.connect(_on_finished)

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
            # 不再置空 self._worker：agent worker 是常驻线程，置空会让下一条
            # 消息新建一条线程，跨轮 KV 前缀复用随之失效（见
            # _ensure_agent_worker 的注释）。引用一直保留到 closeEvent 为止。
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
        # 压缩改写了历史 → agent 已作废旧 KV 缓存，下一条消息本要全量 prefill。
        # 这里排队一次预热（排在常驻线程「本轮跑完」之后），把这段等待挪到
        # 用户还没输入的时候。
        self._request_warmup(allow_busy=True)

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
        # 新对话 = 新任务：清掉上一轮残留的任务计划，避免面板一直停在「多次以前
        # 的计划」（模型并非每次都会主动 task_plan clear）。文件删了，下一轮若仍需
        # 计划，模型会重新 create；右上面板立即复位成「无计划」。
        try:
            from ..planner import clear_plan
            clear_plan(self.config.workspace)
            self._plan_rendered = ""  # 强制下一次轮询/刷新重绘为「无计划」
            self._poll_task_plan()
        except Exception:
            pass
        # reset() 作废旧 KV 缓存；新会话的第一条消息同样不该等 prefill，
        # 这里立刻排队一次预热（此时一般不在对话中）。
        self._request_warmup(allow_busy=True)

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
            # 旧版存的是「配置区/主体」的垂直分割状态，现在 _main_split 是
            # 水平分割（对话 / 右栏），换 key 避免拿旧状态去 restore 新布局。
            state = _as_bytes(s.value("ui/body_splitter"))
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
                s.setValue("ui/body_splitter", self._main_split.saveState())
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
        self._closed = True
        # --- 第一时间屏蔽 bridge 信号：AgentWorker.run() 是**直接**调
        # self._callbacks.bridge.*.emit()，sender 是 Bridge 不是 worker —— 后面
        # _shutdown_worker() 里的 worker.blockSignals(True) 挡不住它们。若到释放
        # agent/llm 时才屏蔽（旧实现放在方法末尾），跨线程把正在被 worker 使用的
        # 对象 close 掉会引发 RuntimeError/段错误。
        try:
            self._bridge.blockSignals(True)
        except Exception:
            pass
        # --- 停止所有定时器。旧实现只停了下面两个，_stream_flush_timer /
        # _reason_flush_timer（流式刷 UI，singleShot 100ms）漏停：关窗后它们仍会
        # fire 并打到已释放的 chat_view / think_view → RuntimeError → abort。
        for timer_name in ("_log_poll_timer", "_plan_poll_timer",
                           "_stream_flush_timer", "_reason_flush_timer"):
            try:
                getattr(self, timer_name).stop()
            except Exception:
                pass
        # --- 收掉配置 / 日志窗口：它们对关闭是 hide，这里显式清理，避免应用
        # 退出后 Qt 才析构时去碰已释放的控件。
        for window_name in ("config_window", "log_window"):
            window = getattr(self, window_name, None)
            if window is None:
                continue
            try:
                window.deleteLater()
            except Exception:
                pass
            setattr(self, window_name, None)
        # --- LoadModelWorker（模型权重加载线程，读数 GB 权重要几十秒）：
        # 旧实现完全没纳管它——self._model_workers 只写不清理，关窗时既不 wait
        # 也不 terminate。于是「启动后几十秒内点关闭」会让一个仍在跑的 QThread
        # 一直活到解释器 teardown（那一刻 sip 析构它 → 可能撞上 Qt 的
        # "Destroyed while thread is still running" qFatal）；且 load_done 是
        # queued 信号，若它晚于本方法末尾的 self._llm.close() 才被处理，
        # _on_model_loaded 会把一个刚 close 的后端重新装回去。
        for _role in list(self._model_workers.keys()):
            _mw = self._model_workers.get(_role)
            if _mw is None:
                continue
            if self._shutdown_worker(_mw, f"model-{_role}",
                                     wait_ms=2000, term_wait_ms=1000):
                self._model_workers.pop(_role, None)
            # 停不下来就保住引用（_shutdown_worker 的既定纪律），绝不能丢。
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
        _sync_engine = getattr(self._sync_worker, "_engine", None)
        _sync_stopped = self._shutdown_worker(self._sync_worker, "sync")
        if _sync_stopped:
            # terminate 路径下 finished 信号被 blockSignals 吞掉，
            # _on_sync_thread_finished 兜底不会触发——这里显式再摘一次
            # 临时 handler（闭包强引用整个 worker 与 RAG 引擎，不摘会
            # 阻止数百 MB 资源回收）。
            self._remove_sync_tmp_handler(self._sync_worker)
            self._sync_worker = None
        # terminate() 路径 run() 的 finally 不执行 → worker 侧 RAG 引擎
        # （LanceDB 连接 / FastEmbed / reranker / OCR，共 ~600MB+）残留。
        # 这里兜底关闭；close() 幂等（等价于重复 drop 缓存），正常结束
        # 路径多调一次是无害的 no-op。
        #
        # **但只在线程已确认结束时才关**：RAGEngine.close() 会先 join 内部那批
        # ingest worker 线程（默认 deadline 600s），而 QThread.terminate() 杀不
        # 掉它们——它们卡在 docling 的原生线程池里没有取消点。旧实现无条件调用，
        # 于是「同步中点关闭窗口」会在这行**同步阻塞最多 10 分钟**：窗口不消失、
        # 事件循环不转，表现为彻底卡死。线程没停就干脆不关，宁可让引擎随进程
        # 退出回收，也不能在主线程 join 第三方原生线程。
        if _sync_engine is not None and _sync_stopped:
            try:
                _sync_engine.close()
            except Exception as e:
                logger.warning("UI: sync worker engine 兜底关闭失败: %s", e)
        elif _sync_engine is not None:
            logger.warning(
                "UI: 同步线程未能停止，跳过 RAGEngine.close()"
                "（避免在主线程 join 原生转换线程导致关窗卡死）")
        # --- InstallDepsWorker: no cancel mechanism, just wait.
        if self._shutdown_worker(self._deps_worker, "deps",
                                 wait_ms=2000, term_wait_ms=1000):
            self._deps_worker = None
        # --- VectorDedupWorker: 去重无取消机制，等待/终止后兜底摘 handler +
        # 关引擎（与 sync worker 同纪律）。
        _dedup_engine = getattr(self._dedup_worker, "_engine", None)
        _dedup_stopped = self._shutdown_worker(self._dedup_worker, "dedup",
                                               wait_ms=2000, term_wait_ms=1000)
        if _dedup_stopped:
            self._remove_sync_tmp_handler(self._dedup_worker)
            self._dedup_worker = None
        # 与 sync worker 同纪律：线程未确认停止时跳过 close()（否则会阻塞在
        # RAGEngine.close() 内部的 ingest worker join 上）。
        if _dedup_engine is not None and _dedup_stopped:
            try:
                _dedup_engine.close()
            except Exception as e:
                logger.warning("UI: dedup worker engine 兜底关闭失败: %s", e)
        elif _dedup_engine is not None:
            logger.warning("UI: 去重线程未能停止，跳过 RAGEngine.close()")
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
        # 辅助模型同样由 UI 持有并负责 close()（见 __init__ 的注释）：
        # 关窗时若不关，MLX 权重（数 GB 统一内存）会残留到进程退出。
        if self._aux_llm is not None:
            try:
                self._aux_llm.close()
            except Exception as e:
                logger.warning("UI: aux llm close error: %s", e)

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

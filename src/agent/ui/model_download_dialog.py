"""首次启动的「下载模型」对话框（PyQt6）。

在 ``main_window.run()`` 启动主窗口之前以模态形式弹出：提醒用户需要联网下载
默认模型（含总大小估算），展示总进度 / 当前文件进度 / 速度 / ETA，并支持取消
（取消即退出应用，因未下载完模型无法使用）。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..hf_downloader import download_models, human_size
from ..model_registry import ROLE_LABELS


class DownloadWorker(QThread):
    """后台下载线程：包装 ``download_models`` 并通过信号回传进度。"""

    progress = pyqtSignal(dict)
    finished = pyqtSignal(bool, str)  # (success, error_msg)

    def __init__(
        self,
        specs: List[Tuple[str, str, str]],
        mirror: str,
    ) -> None:
        super().__init__()
        self._specs = specs
        self._mirror = mirror
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: D401
        try:
            download_models(
                self._specs,
                self._mirror,
                progress_cb=lambda p: self.progress.emit(p),
                cancel_cb=lambda: self._cancel,
            )
            self.finished.emit(True, "")
        except Exception as e:  # 网络/IO 等任何失败
            self.finished.emit(False, f"{type(e).__name__}: {e}")


class ModelDownloadChoiceDialog(QDialog):
    """首次启动、模型缺失时，先让用户确认处理方式。

    - 选「自动下载补全」→ 后续弹出下载进度框（模态，未完成不能进应用）；
    - 选「自行下载并配置」→ 跳过自动下载，直接进应用（用户自行放置/配置模型）；
    - 选「退出应用」或关闭对话框 → 退出应用。
    """

    def __init__(self, missing: List[Tuple[str, str, str]], parent=None) -> None:
        super().__init__(parent)
        self.choice: str = "exit"  # 默认视为退出，避免误进应用
        self.setWindowTitle("首次启动：模型缺失")
        self.setModal(True)
        self.resize(540, 320)

        layout = QVBoxLayout(self)

        list_text = "\n".join(
            f"  • {ROLE_LABELS.get(r, r)}：{name}" for r, name, _ in missing
        )
        self.notice = QLabel(
            "首次启动检测到以下默认模型尚未安装：\n"
            f"{list_text}\n\n请选择如何处理："
        )
        self.notice.setWordWrap(True)
        layout.addWidget(self.notice)

        opt_auto = QLabel(
            "① 自动下载补全（推荐）\n"
            "联网从 HuggingFace 镜像（hf-mirror.com）下载上述默认模型，"
            "下载完成并校验后才能进入应用（支持断点续传）。"
        )
        opt_auto.setWordWrap(True)
        layout.addWidget(opt_auto)

        opt_manual = QLabel(
            "② 自行下载并配置\n"
            "跳过自动下载，由你自行放置 / 配置各阶段（主模型、嵌入、重排等）"
            "要使用的模型。相应功能在模型就绪前不可用。"
        )
        opt_manual.setWordWrap(True)
        layout.addWidget(opt_manual)

        btn_row = QHBoxLayout()
        self.auto_btn = QPushButton("自动下载补全")
        self.manual_btn = QPushButton("自行下载并配置")
        self.exit_btn = QPushButton("退出应用")
        self.auto_btn.clicked.connect(self._on_auto)
        self.manual_btn.clicked.connect(self._on_manual)
        self.exit_btn.clicked.connect(self._on_exit)
        btn_row.addWidget(self.auto_btn)
        btn_row.addWidget(self.manual_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.exit_btn)
        layout.addLayout(btn_row)

    def _on_auto(self) -> None:
        self.choice = "auto"
        self.accept()

    def _on_manual(self) -> None:
        self.choice = "manual"
        self.accept()

    def _on_exit(self) -> None:
        self.choice = "exit"
        self.reject()


def _eta(seconds: float) -> str:
    if seconds <= 0 or seconds == float("inf"):
        return "—"
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}时{m}分"
    if m:
        return f"{m}分{s}秒"
    return f"{s}秒"


class ModelDownloadDialog(QDialog):
    def __init__(
        self,
        specs: List[Tuple[str, str, str]],
        mirror: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.specs = specs
        self.mirror = mirror
        self.succeeded = False
        self._aborted = False

        self.setWindowTitle("首次使用：下载模型")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.resize(560, 380)

        layout = QVBoxLayout(self)

        # 提示文案
        self.notice = QLabel()
        self.notice.setWordWrap(True)
        self._refresh_notice()
        layout.addWidget(self.notice)

        # 模型清单
        self.list_label = QLabel(
            "需要下载的模型：\n"
            + "\n".join(
                f"  • {ROLE_LABELS.get(r, r)}：{Path(target).name}  ({repo})"
                for r, target, repo in self.specs
            )
        )
        self.list_label.setWordWrap(True)
        layout.addWidget(self.list_label)

        # 总进度
        layout.addWidget(QLabel("总进度"))
        self.overall_bar = QProgressBar()
        self.overall_bar.setRange(0, 100)
        self.overall_bar.setValue(0)
        layout.addWidget(self.overall_bar)
        self.overall_info = QLabel("等待开始…")
        layout.addWidget(self.overall_info)

        # 当前文件进度
        layout.addWidget(QLabel("当前文件"))
        self.file_bar = QProgressBar()
        self.file_bar.setRange(0, 100)
        self.file_bar.setValue(0)
        layout.addWidget(self.file_bar)
        self.file_info = QLabel("")
        layout.addWidget(self.file_info)

        # 日志
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(90)
        layout.addWidget(self.log)

        # 按钮
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("开始下载")
        self.retry_btn = QPushButton("重试")
        self.retry_btn.setVisible(False)
        self.cancel_btn = QPushButton("取消并退出")
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.retry_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        self.start_btn.clicked.connect(self.start)
        self.retry_btn.clicked.connect(self.start)
        self.cancel_btn.clicked.connect(self.on_cancel)

        # 工作线程
        self.worker: DownloadWorker | None = None

    # ------------------------------------------------------------------ UI 辅助
    def _refresh_notice(self) -> None:
        self.notice.setText(
            "检测到默认模型尚未安装。\n\n"
            "首次使用需要联网，从 HuggingFace 镜像（hf-mirror.com）下载所需的默认模型。"
            "下载期间请保持网络连接；下载完成后才能正常使用本软件。\n\n"
            "下载支持断点续传：若网络中断，可重新点击「开始下载 / 重试」从已下载处继续。"
        )

    def _log(self, text: str) -> None:
        self.log.append(text)

    def _on_progress(self, p: dict) -> None:
        total = p.get("overall_total") or 0
        done = p.get("overall_done") or 0
        pct = int(done / total * 100) if total > 0 else 0
        self.overall_bar.setValue(max(0, min(100, pct)))
        speed = p.get("speed") or 0.0
        remaining = max(0, total - done)
        eta = _eta(remaining / speed) if speed > 0 else "—"
        self.overall_info.setText(
            f"{human_size(done)} / {human_size(total)}  ·  {pct}%  ·  "
            f"{human_size(speed)}/s  ·  剩余 {eta}"
        )
        # 当前文件
        ft = p.get("file_total") or 0
        fd = p.get("file_done") or 0
        fpct = int(fd / ft * 100) if ft > 0 else 0
        self.file_bar.setValue(max(0, min(100, fpct)))
        self.file_info.setText(
            f"[{p.get('model_index', 0) + 1}/{p.get('model_total', 0)}] "
            f"{p.get('model_local_name', '')}  →  {p.get('filename', '')}  "
            f"({human_size(fd)} / {human_size(ft)})"
        )

    def _on_finished(self, ok: bool, msg: str) -> None:
        if ok:
            self.succeeded = True
            self._log("✅ 全部模型下载完成。")
            QMessageBox.information(self, "完成", "模型下载完成，即将进入应用。")
            self.accept()
            return
        if self._aborted:
            # 用户主动取消 → 直接退出应用。
            self.reject()
            return
        # 真实错误 → 允许重试。
        self._log(f"❌ 下载失败：{msg}")
        self.overall_info.setText("下载失败")
        self.start_btn.setVisible(False)
        self.retry_btn.setVisible(True)
        self.cancel_btn.setText("退出")
        QMessageBox.warning(
            self, "下载失败",
            f"模型下载失败：\n{msg}\n\n可点击「重试」继续（已下载部分会断点续传）。",
        )

    # ------------------------------------------------------------------ 动作
    def start(self) -> None:
        self.start_btn.setVisible(False)
        self.retry_btn.setVisible(False)
        self.cancel_btn.setText("取消并退出")
        self._aborted = False
        self._log("开始下载（先获取文件列表，请稍候）…")
        self.worker = DownloadWorker(self.specs, self.mirror)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def on_cancel(self) -> None:
        self._aborted = True
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self._log("已请求取消，正在停止…")
            # 工作线程会在下一次取消检查时退出并触发 _on_finished(False)
            # （_aborted=True → reject）。若迟迟不退出，这里直接 reject。
            self.worker.wait(3000)
            if self.worker.isRunning():
                self.reject()
        else:
            # 尚未开始下载：直接退出。
            self.reject()

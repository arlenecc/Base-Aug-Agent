"""文件转 Markdown 转换工具窗口（独立于知识库同步流程）。

提供一个独立面板，让用户把任意「待转换文件 / 目录」批量提取转换成
Markdown 文件写入指定目录——**不入知识库、不切片、不向量化**，纯粹是
「文件 → .md」的转换产物，供用户在其它工具里使用。

转换链路完全沿用知识库同步的处理逻辑（engine.ingest 的解析段）：
  - 路由：pandoc（epub 系电子书 / docx / html / LaTeX…）→ 配置的 PDF 主引擎
    （docling / marker / ovis，含大 PDF 按页分批）→ 轻量兜底（OvisOCR3 VLM →
    parsers.extract_text）。图片归一化 / 图片识别由各引擎内部完成
    （docling 内置 OCR、ovis 路径的 PyMuPDF 取图 + VLM）。
  - 清洗：Markdown 引擎输出走 clean_markdown_document（无损清理，剥
    <!-- image --> 注释 / raw HTML / 内部锚点等噪音）；纯文本走
    clean_text + convert_plain_text_to_markdown（纯文本启发式还原章节标题）。
  - 「开始转换」会先把配置面板「PDF 解析引擎 / 自定义 OCR」相关控件值
    收集落盘到 config（等价 _on_ocr_save_settings 的轻量路径，不重建
    agent），再按这些值对解析器模块做 configure 接线——与同步用同一套
    设置、同一套代码路径。
"""

from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QProgressBar, QTextBrowser, QMessageBox,
)

from ..rag.parsers import SUPPORTED_EXTENSIONS
from ..rag.cleaner import (
    clean_text, fix_hard_wraps, clean_markdown_document,
)
from ..rag.document_outline import convert_plain_text_to_markdown
from ..rag.pandoc_parser import (
    PANDOC_EBOOK_EXTS,
    PANDOC_SUPPORTED_EXTENSIONS,
    convert_document,
    convert_ebook,
    is_pandoc_available,
)
from ..rag.engine import _extract_with_pdf_engine

logger = logging.getLogger(__name__)

# 转换工具支持的扩展名 = 同步白名单 ∪ 旧 Office 格式（docling 声明支持）
# ∪ 常见图像（docling 不认图像，经回退链落到 OvisOCR3 VLM 识别）。
CONVERT_EXTENSIONS = SUPPORTED_EXTENSIONS | {
    ".doc", ".xls", ".ppt",
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
}

_FILE_DIALOG_FILTER = (
    "可转换文件 (*.pdf *.epub *.mobi *.azw3 *.azw *.docx *.doc *.xlsx *.xls "
    "*.pptx *.ppt *.html *.htm *.tex *.latex *.txt *.md *.markdown *.rst "
    "*.csv *.tsv *.jpg *.jpeg *.png *.bmp *.tif *.tiff);;"
    "所有文件 (*)"
)


# ---------------------------------------------------------------------------
# 纯逻辑（不依赖 Qt，便于单测）：收集 / 路由 / 清洗 / 输出命名
# ---------------------------------------------------------------------------

def collect_files(source: str) -> List[str]:
    """收集待转换文件：单文件直接返回；目录递归扫描白名单内文件。

    与 engine.ingest 同一语义：跳过隐藏文件（. 开头）。目录内命中白名单
    的文件按路径排序，保证进度可预期。源不存在抛 ValueError。
    """
    source = (source or "").strip()
    if not source:
        raise ValueError("请先选择待转换文件或目录")
    if os.path.isfile(source):
        return [os.path.abspath(source)]
    if not os.path.isdir(source):
        raise ValueError(f"路径不存在: {source}")
    files: List[str] = []
    for root, _dirs, names in os.walk(source):
        for name in sorted(names):
            if name.startswith("."):
                continue
            fp = os.path.join(root, name)
            if os.path.splitext(name)[1].lower() in CONVERT_EXTENSIONS:
                files.append(os.path.abspath(fp))
    return files


def convert_file_to_markdown(path: str, pdf_engine: str = "docling") -> str:
    """单文件 → 清洗后的 Markdown 字符串。

    路由与清洗语义与 engine.ingest 的解析段逐行对应（pandoc → 配置主引擎
    → 轻量兜底；is_markdown 走 clean_markdown_document，纯文本走
    clean_text + convert_plain_text_to_markdown）。失败抛异常，由调用方
    计为该文件转换失败。
    """
    suffix = os.path.splitext(path)[1].lower()

    # ---- 1. 解析（与 engine.py:1136-1157 同构）----
    text: Optional[str] = None
    is_markdown = False
    pandoc_md: Optional[str] = None
    if suffix in PANDOC_SUPPORTED_EXTENSIONS and is_pandoc_available():
        md: Optional[str] = None
        try:
            if suffix in PANDOC_EBOOK_EXTS:
                result = convert_ebook(path)
                md = result[0] if result is not None else None
            else:
                md = convert_document(path)
        except Exception as e:  # pandoc 失败回退默认解析，与同步一致
            logger.warning("pandoc 转换异常，回退默认解析 %s: %s", path, e)
        if md and md.strip():
            # --wrap=none 下段落已是单行；fix_hard_wraps 清理残留断行噪音。
            pandoc_md = fix_hard_wraps(md)
    if pandoc_md is not None:
        text, is_markdown = pandoc_md, True
    else:
        text, is_markdown = _extract_with_pdf_engine(path, pdf_engine)

    if not text or not text.strip():
        raise ValueError("提取文本为空")

    # ---- 2. 清洗 / 转 Markdown（与 engine.py:1220-1249 同构）----
    if is_markdown:
        # docling/marker/ovis/pandoc 已输出规范 Markdown：无损清理剥噪音，
        # 绝不跑 clean_text（其 <[^>]+> 会误删标题/表格语法）。
        return clean_markdown_document(text)
    cleaned = clean_text(text)
    # 非结构化纯文本：纯文本启发式还原章节标题为 Markdown ``#`` 标题。
    cleaned, _ = convert_plain_text_to_markdown(cleaned)
    return cleaned


def unique_output_path(out_dir: str, source_path: str) -> str:
    """为目标 md 文件生成不冲突的输出路径：stem.md / stem (2).md / …"""
    stem = os.path.splitext(os.path.basename(source_path))[0] or "untitled"
    candidate = os.path.join(out_dir, f"{stem}.md")
    n = 2
    while os.path.exists(candidate):
        candidate = os.path.join(out_dir, f"{stem} ({n}).md")
        n += 1
    return candidate


def apply_parser_configure(cfg) -> str:
    """把 config 的解析相关字段落到各解析器模块的全局配置（configure）。

    与 RAGEngine.__init__ 的 configure 段等价（engine.py:641-684），但不
    构造引擎/向量库——转换工具与知识库同步共用同一套模块级设置。注意：
    换 ovis backend 需要引擎重载才彻底生效（ovis_ocr._config 只热更新
    参数），与同步路径行为一致。
    返回归一化后的 pdf_engine（docling|marker|ovis）。
    """
    pdf_engine = getattr(cfg, "rag_pdf_engine", "docling")
    if pdf_engine not in ("docling", "marker", "ovis"):
        pdf_engine = "docling"
    try:
        from ..rag.ovis_ocr import configure as _ovis_configure
        _ovis_configure(
            getattr(cfg, "rag_ocr_backend", "mlx"),
            getattr(cfg, "rag_ocr_model", ""),
            getattr(cfg, "rag_ocr_model_path", ""),
            gguf_path=getattr(cfg, "rag_ocr_gguf_path", ""),
            mmproj_path=getattr(cfg, "rag_ocr_mmproj_path", ""),
            tokenizer_path=getattr(cfg, "rag_ocr_tokenizer_path", ""),
            vlm_handler=getattr(cfg, "rag_ocr_vlm_handler", "mtmd"),
            chat_format=getattr(cfg, "rag_ocr_chat_format", ""),
        )
    except Exception as e:
        logger.debug("转换工具: ovis configure error: %s", e)
    try:
        from ..rag.docling_parser import configure_dynamic_scale
        configure_dynamic_scale(bool(getattr(cfg, "rag_docling_dynamic_scale", False)))
    except Exception as e:
        logger.debug("转换工具: docling dynamic scale configure error: %s", e)
    try:
        from ..rag.docling_parser import configure_vlm
        configure_vlm(bool(getattr(cfg, "rag_docling_vlm", False)))
    except Exception as e:
        logger.debug("转换工具: docling vlm configure error: %s", e)
    try:
        concurrency = max(1, int(getattr(cfg, "rag_pdf_small_concurrency", 2)))
        threshold_bytes = max(1.0, float(getattr(cfg, "rag_pdf_large_threshold_mb", 20.0))) * 1024 * 1024
        from ..rag.docling_parser import configure_concurrency as _dc_cfg
        _dc_cfg(concurrency, int(threshold_bytes))
        from ..rag.marker_parser import configure_concurrency as _mk_cfg
        _mk_cfg(concurrency, int(threshold_bytes))
    except Exception as e:
        logger.debug("转换工具: concurrency configure error: %s", e)
    try:
        from ..rag.docling_parser import configure_batching as _db_cfg
        _db_cfg(getattr(cfg, "rag_pdf_batch_threshold_mb", 50.0),
                getattr(cfg, "rag_pdf_batch_pages", 50))
    except Exception as e:
        logger.debug("转换工具: batching configure error: %s", e)
    return pdf_engine


# ---------------------------------------------------------------------------
# 后台转换线程
# ---------------------------------------------------------------------------

class ConvertWorker(QThread):
    """批量转换线程：逐文件解析 → 清洗 → 写出，信号上报进度与结果。

    取消语义：_cancel 事件只在文件之间检查（单文件解析可能耗时数分钟，
    无法在引擎内部打断）——点「取消」后当前文件会跑完再停。
    """

    range_ready = pyqtSignal(int)                 # 总文件数
    progress = pyqtSignal(int, int, str, str)     # done, total, 当前文件, 环节
    file_done = pyqtSignal(str, str, bool, str)   # 源路径, 输出路径, 是否成功, 备注
    all_done = pyqtSignal(int, int, bool)         # 成功数, 失败数, 是否被取消
    error = pyqtSignal(str)

    def __init__(self, files: List[str], out_dir: str, pdf_engine: str,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._files = list(files)
        self._out_dir = out_dir
        self._pdf_engine = pdf_engine
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        total = len(self._files)
        self.range_ready.emit(total)
        ok = fail = 0
        cancelled = False
        for fp in self._files:
            if self._cancel.is_set():
                cancelled = True
                break
            fname = os.path.basename(fp)
            self.progress.emit(0, total, fname, "解析中")
            try:
                md = convert_file_to_markdown(fp, self._pdf_engine)
            except Exception as e:
                fail += 1
                logger.warning("转换失败 %s: %s", fname, e)
                self.file_done.emit(fp, "", False, str(e))
                self.progress.emit(1, total, fname, "失败")
                continue
            self.progress.emit(0, total, fname, "写出中")
            try:
                out_path = unique_output_path(self._out_dir, fp)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(md)
            except OSError as e:
                fail += 1
                self.file_done.emit(fp, "", False, f"写入失败: {e}")
                self.progress.emit(1, total, fname, "失败")
                continue
            ok += 1
            self.file_done.emit(fp, out_path, True, f"{len(md)} 字符")
            self.progress.emit(1, total, fname, "完成")
        self.all_done.emit(ok, fail, cancelled)


# ---------------------------------------------------------------------------
# 窗口
# ---------------------------------------------------------------------------

class ConvertToolWindow(QWidget):
    """文件转 Markdown 转换工具（非模态独立窗口）。

    构造时传入主窗口引用：开始转换时先从主窗口控件收集「PDF 解析引擎 /
    自定义 OCR」设置落盘 config，再按同一套设置驱动解析器。
    """

    def __init__(self, main_window) -> None:
        super().__init__()
        self._main = main_window
        self._worker: Optional[ConvertWorker] = None
        self._build_ui()

    # -- UI ----------------------------------------------------------------

    def _build_ui(self) -> None:
        self.setWindowTitle("文件转 Markdown 转换工具")
        self.setMinimumWidth(760)
        lay = QVBoxLayout(self)
        lay.setSpacing(6)

        # Row 1: 待转换文件 [路径框] [选择文件] [选择目录]
        src_row = QHBoxLayout()
        src_row.setSpacing(8)
        src_row.addWidget(QLabel("待转换文件"))
        self.src_edit = QLineEdit()
        self.src_edit.setPlaceholderText(
            "单个文件，或一个目录（目录会递归扫描其中所有可转换文件）")
        src_row.addWidget(self.src_edit, 1)
        pick_file_btn = QPushButton("选择文件")
        pick_file_btn.clicked.connect(self._on_pick_source_file)
        src_row.addWidget(pick_file_btn)
        pick_dir_btn = QPushButton("选择目录")
        pick_dir_btn.clicked.connect(self._on_pick_source_dir)
        src_row.addWidget(pick_dir_btn)
        lay.addLayout(src_row)

        # Row 2: 保存目录 [路径框] [选择目录]
        out_row = QHBoxLayout()
        out_row.setSpacing(8)
        out_row.addWidget(QLabel("保存目录"))
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("转换结果（.md）保存目录，留空则保存到源文件所在目录")
        out_row.addWidget(self.out_edit, 1)
        out_btn = QPushButton("选择目录")
        out_btn.clicked.connect(self._on_pick_out_dir)
        out_row.addWidget(out_btn)
        lay.addLayout(out_row)

        # Row 3: [开始转换] [取消]
        act_row = QHBoxLayout()
        act_row.setSpacing(8)
        self.start_btn = QPushButton("开始转换")
        self.start_btn.setStyleSheet("font-weight:600;")
        self.start_btn.setToolTip(
            "采用配置面板「PDF 解析引擎 / 自定义 OCR」的当前设置，\n"
            "把所选文件/目录批量转换成 Markdown 写入保存目录")
        self.start_btn.clicked.connect(self._on_start)
        act_row.addWidget(self.start_btn)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setToolTip("停止转换：当前文件会跑完，之后不再处理后续文件")
        self.cancel_btn.clicked.connect(self._on_cancel)
        act_row.addWidget(self.cancel_btn)
        act_row.addStretch(1)
        lay.addLayout(act_row)

        # 进度区：当前文件/环节标签 + 总体进度条
        self.stage_label = QLabel("就绪")
        lay.addWidget(self.stage_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("待开始 (%v / %m)")
        lay.addWidget(self.progress_bar)

        # 转换日志：逐文件结果（成功/失败 + 输出路径）
        self.log_view = QTextBrowser()
        self.log_view.setPlaceholderText("转换结果将逐文件显示在这里")
        lay.addWidget(self.log_view, 1)

    # -- 选择器 ------------------------------------------------------------

    def _on_pick_source_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择待转换文件", self.src_edit.text().strip(), _FILE_DIALOG_FILTER)
        if path:
            self.src_edit.setText(path)

    def _on_pick_source_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "选择待转换目录", self.src_edit.text().strip())
        if path:
            self.src_edit.setText(path)

    def _on_pick_out_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "选择保存目录", self.out_edit.text().strip())
        if path:
            self.out_edit.setText(path)

    # -- 启动 / 取消 --------------------------------------------------------

    def _on_start(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "转换中", "已有转换任务在进行中")
            return
        try:
            files = collect_files(self.src_edit.text())
        except ValueError as e:
            QMessageBox.warning(self, "路径无效", str(e))
            return
        out_dir = self.out_edit.text().strip() or os.path.dirname(files[0])
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as e:
            QMessageBox.warning(self, "保存目录不可用", f"无法创建保存目录: {e}")
            return
        # 采用配置面板当前设置：控件值 → config 落盘 → 解析器 configure。
        pdf_engine = self._adopt_panel_settings()
        self.log_view.append(
            f"—— 开始转换 {len(files)} 个文件 → {out_dir}（引擎: {pdf_engine}）——")
        self._set_running(True)
        self.progress_bar.setRange(0, max(1, len(files)))
        self.progress_bar.setValue(0)
        self._worker = ConvertWorker(files, out_dir, pdf_engine)
        # all_done 在 run() 内部发出：不能在槽里直接置 _worker = None——
        # 那会丢掉最后一个 Python 引用，C++ 对象在线程收尾前被销毁。
        # 改为 finished（run 真正返回后）再清理。
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.range_ready.connect(self._on_range)
        self._worker.progress.connect(self._on_progress)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _adopt_panel_settings(self) -> str:
        """从主窗口配置面板收集引擎/OCR 设置：落盘 config 并接线解析器。

        等价「保存设置」按钮的轻量路径（只 gather + save，不重建 agent），
        使「开始转换」永远采用面板上可见的当前值，而非上次保存的旧值。
        """
        try:
            from .main_window import CONFIG_PATH
            self._main._gather_ocr_config_from_widgets()
            from .main_window import _checked_pdf_engine
            self._main.config.rag_pdf_engine = _checked_pdf_engine(self._main)
            self._main.config.save(CONFIG_PATH)
        except Exception as e:
            logger.warning("转换工具: 配置落盘失败（继续用已有配置）: %s", e)
        return apply_parser_configure(self._main.config)

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.stage_label.setText("取消中：等待当前文件完成…")
            self.cancel_btn.setEnabled(False)

    def _set_running(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)

    # -- worker 信号槽 -------------------------------------------------------

    def _on_range(self, total: int) -> None:
        self.progress_bar.setRange(0, max(1, total))

    def _on_progress(self, done_inc: int, total: int, fname: str, stage: str) -> None:
        if done_inc:
            self.progress_bar.setValue(self.progress_bar.value() + done_inc)
        self.stage_label.setText(f"当前文件: {fname} · {stage}（共 {total} 个）")

    def _on_file_done(self, src: str, out: str, ok: bool, note: str) -> None:
        fname = os.path.basename(src)
        if ok:
            self.log_view.append(f"✔ {fname} → {out}（{note}）")
        else:
            self.log_view.append(f"✘ {fname} 转换失败: {note}")

    def _on_worker_finished(self) -> None:
        self._worker = None

    def _on_all_done(self, ok: int, fail: int, cancelled: bool) -> None:
        self._set_running(False)
        tail = "（已取消，未处理的文件被跳过）" if cancelled else ""
        self.stage_label.setText(f"转换结束：成功 {ok}，失败 {fail}{tail}")
        self.log_view.append(f"—— 转换结束：成功 {ok}，失败 {fail}{tail} ——")

    def _on_error(self, msg: str) -> None:
        self._set_running(False)
        QMessageBox.critical(self, "转换异常", msg)

    # -- 资源释放 ----------------------------------------------------------

    def closeEvent(self, event) -> None:
        # 与 ocr_test_window 同策略：跑着的 worker 设取消后短暂等待。
        # 单文件解析可能长达数分钟，不能无限阻塞关闭——等 2s 后放行，
        # worker 仍在跑时窗口隐藏、任务在后台继续到当前文件结束。
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(2000)
        super().closeEvent(event)

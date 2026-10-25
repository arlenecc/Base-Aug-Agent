"""OCR 识别测试窗口（独立于知识库同步流程）。

提供一个交互式面板，让用户对任意「图片 / PDF」文件用不同 OCR 模型做识别，
并左右对照原文件与识别结果（Markdown/HTML，含表格）。用于：
  - 在正式把 OCR 模型配进知识库同步前，先目测各模型效果；
  - 对单个指定文件做一次性的 OCR 识别与结果导出。

模型选项：
  - Apple OCR        ：ocrmac(Apple Vision) 优先，空/异常时回退 RapidOCR（parsers）。
  - OvisOCR3         ：OvisOCR3 VLM（用配置里默认的 Ovis 模型）。
  - 混合(Vision+Ovis表格，仅PDF)：文本型 PDF 取文本层、扫描型 PDF 用 Vision 识别，
                       并对扫描页额外用 Ovis 取一份、保留更完整者（务实简化版）。
  - 自定义 OCR       ：用设置面板里「自定义 OCR 模型」保存的那套 Ovis 配置（backend/
                       模型/路径/gguf/mmproj 等）。

图片统一做 `_ocr_normalize`（自动反色 + 对比度拉伸）后再喂给各引擎；PDF 逐页
判别文本型 / 扫描型：有文本层直取文本，否则整页渲染成灰度图送 OCR。
"""

from __future__ import annotations

import io
import logging
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import fitz  # PyMuPDF
from PIL import Image

try:
    import pymupdf4llm
    _PYMUPDF4LLM_AVAILABLE = True
except Exception:  # noqa: BLE001 - 可选依赖，缺失时文本型页回退原生 get_text
    pymupdf4llm = None
    _PYMUPDF4LLM_AVAILABLE = False

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QTextBrowser, QFileDialog, QMessageBox, QSplitter, QFrame,
    QSizePolicy,
)

from ..rag import parsers, ovis_ocr
from ..config import AgentConfig

logger = logging.getLogger(__name__)

# 识别模型下拉框：(显示名, 标识)
OCR_MODELS = (
    ("Apple OCR", "apple"),
    ("OvisOCR3", "ovis"),
    ("混合（Vision+Ovis表格，仅PDF）", "hybrid"),
    ("自定义OCR", "custom"),
)

# 图片扩展名
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
_PDF_EXTS = (".pdf",)


def _arr_to_png_bytes(arr: np.ndarray) -> bytes:
    im = Image.fromarray(arr).convert("L")
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def _pix_to_gray(pix) -> np.ndarray:
    """fitz.Pixmap -> 灰度 2D ndarray。"""
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    if pix.n >= 3:
        gray = arr[..., :3].mean(axis=2).astype(np.uint8)
    else:
        gray = arr[..., 0].copy()
    return gray


# --------------------------------------------------------------------------- #
# OCR 调度
# --------------------------------------------------------------------------- #

def _ensure_ovis_engine(cfg: Dict[str, str]) -> None:
    """按给定 ocr 配置加载（或复用）Ovis 引擎单例。失败抛 OvisUnavailable。"""
    ovis_ocr.configure(
        backend=cfg.get("backend", "") or "mlx",
        model=cfg.get("model", "") or "",
        path=cfg.get("path", "") or "",
        gguf_path=cfg.get("gguf_path", "") or "",
        mmproj_path=cfg.get("mmproj_path", "") or "",
        vlm_handler=cfg.get("vlm_handler", "") or "mtmd",
        chat_format=cfg.get("chat_format", "") or "",
    )
    # _get_engine 是惰性加载总入口（私有但同进程可调用）；显式触发以便捕获失败。
    loader = getattr(ovis_ocr, "_get_engine", None)
    if loader is not None:
        loader()
    else:  # 退路：首次识别时由 ocr_image_to_markdown 内部惰性加载
        pass


def _ocr_image_apple(arr: np.ndarray) -> str:
    """图片走 Apple Vision（ocrmac 优先 + RapidOCR 回退），直接返回文本。"""
    return parsers.ocr_recognize_array(arr)


def _ocr_image_ovis(arr: np.ndarray, cfg: Dict[str, str]) -> str:
    """图片走 OvisOCR3 VLM，返回 markdown。"""
    _ensure_ovis_engine(cfg)
    return ovis_ocr.ocr_image_to_markdown(arr)


def _ov_cfg_from_config(config: AgentConfig) -> Dict[str, str]:
    """从 AgentConfig 取出「自定义 OCR」那套 Ovis 配置。"""
    return {
        "backend": getattr(config, "rag_ocr_backend", "") or "mlx",
        "model": getattr(config, "rag_ocr_model", "") or "",
        "path": getattr(config, "rag_ocr_model_path", "") or "",
        "gguf_path": getattr(config, "rag_ocr_gguf_path", "") or "",
        "mmproj_path": getattr(config, "rag_ocr_mmproj_path", "") or "",
        "vlm_handler": getattr(config, "rag_ocr_vlm_handler", "") or "mtmd",
        "chat_format": getattr(config, "rag_ocr_chat_format", "") or "",
    }


def _ocr_image(arr: np.ndarray, model: str, config: AgentConfig) -> str:
    """对单张归一化灰度图按模型识别。"""
    if model == "apple":
        return _ocr_image_apple(arr)
    if model == "ovis":
        return _ocr_image_ovis(arr, _ov_cfg_from_config(config))
    if model == "custom":
        return _ocr_image_ovis(arr, _ov_cfg_from_config(config))
    # hybrid 对图片：退化为 Ovis（图片主要比拼表格/版面，Ovis 更擅长）
    return _ocr_image_ovis(arr, _ov_cfg_from_config(config))


def _run_image(filepath: str, model: str, config: AgentConfig) -> List[Dict[str, Any]]:
    img = Image.open(filepath).convert("L")
    arr = np.asarray(img)
    arr = parsers._ocr_normalize(arr)
    text = _ocr_image(arr, model, config)
    return [{"page_no": 0, "text": text, "img_png": _arr_to_png_bytes(arr)}]


def _run_pdf(filepath: str, model: str, config: AgentConfig) -> List[Dict[str, Any]]:
    """逐页识别 PDF：文本型页取文本层，扫描型页渲染灰度送 OCR。"""
    doc = fitz.open(filepath)
    pages: List[Dict[str, Any]] = []
    try:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # ~150 DPI
            img_png = pix.tobytes("png")
            text_layer = page.get_text("text").strip()
            has_images = bool(page.get_images(full=True))

            if text_layer:
                # 文本型页：用 PyMuPDF4LLM 转成保留版面/表格的 Markdown
                # （不可用或异常时回退原生 get_text）。
                text = _pdf_text_to_markdown(filepath, i)
            elif has_images:
                # 扫描型页：整页渲染成灰度送 OCR。
                gray = _pix_to_gray(pix)
                gray = parsers._ocr_normalize(gray)
                if model == "apple":
                    text = _ocr_image_apple(gray)
                elif model == "ovis" or model == "custom":
                    text = _ocr_image_ovis(gray, _ov_cfg_from_config(config))
                else:  # hybrid：Vision 与 Ovis 各跑一份，保留更完整者
                    a = _ocr_image_apple(gray)
                    o = _ocr_image_ovis(gray, _ov_cfg_from_config(config))
                    text = o if len(o) >= len(a) else a
            else:
                # 空白页
                text = ""
            pages.append({"page_no": i, "text": text, "img_png": img_png})
    finally:
        doc.close()
    return pages


def _normalize_pymupdf4llm(md) -> str:
    """把 PyMuPDF4LLM 的返回（str / list[dict] / list[str]）规整成 Markdown 文本。"""
    if isinstance(md, str):
        return md
    if isinstance(md, (list, tuple)):
        chunks = []
        for item in md:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                chunks.append(
                    item.get("text") or item.get("markdown") or item.get("body") or ""
                )
        return "\n\n".join(c for c in chunks if c)
    return str(md)


def _pdf_text_to_markdown(filepath: str, page_index: int) -> str:
    """文本型 PDF 页转 Markdown：优先 PyMuPDF4LLM（保留标题/列表/表格）。

    PyMuPDF4LLM 缺失或转换异常时，回退 PyMuPDF 原生 ``get_text("text")``。
    """
    if _PYMUPDF4LLM_AVAILABLE and pymupdf4llm is not None:
        try:
            md = pymupdf4llm.to_markdown(filepath, pages=[page_index])
            text = _normalize_pymupdf4llm(md)
            if text.strip():
                return text
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "pymupdf4llm 转换第 %d 页失败，回退 get_text: %s", page_index, e
            )
    # 回退：原生文本层
    try:
        doc = fitz.open(filepath)
        try:
            return doc[page_index].get_text("text").strip()
        finally:
            doc.close()
    except Exception:  # noqa: BLE001
        return ""


# --------------------------------------------------------------------------- #
# Markdown -> HTML（轻量 GFM，足以正确渲染管道表格 / 标题 / 列表 / 代码）
# --------------------------------------------------------------------------- #

def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inline(s: str) -> str:
    s = _esc(s)
    # 行内代码
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    # 加粗 / 斜体
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", s)
    # 链接
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    return s


def _is_table_sep(line: str) -> bool:
    t = line.strip().strip("|").strip()
    if not t:
        return False
    return all(c in ("-", ":", " ") for c in t) and "-" in t


def _split_row(line: str):
    parts = line.strip().strip("|").split("|")
    return [p.strip() for p in parts]


def _render_table(header, rows) -> str:
    th = "".join(f"<th>{_inline(h)}</th>" for h in header)
    body = ""
    for r in rows:
        # 补齐/截断到 header 列数
        cells = (r + [""] * len(header))[: len(header)]
        body += "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells) + "</tr>"
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def markdown_to_html(md: str) -> str:
    if not md or not md.strip():
        return "<p style='color:gray'>(无识别结果)</p>"
    lines = md.splitlines()
    out: List[str] = []
    i = 0
    in_code = False
    code_buf: List[str] = []
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            if in_code:
                out.append("<pre><code>" + _esc("\n".join(code_buf)) + "</code></pre>")
                code_buf = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue
        # 表格
        if (line.lstrip().startswith("|") and i + 1 < len(lines)
                and _is_table_sep(lines[i + 1])):
            header = _split_row(line)
            i += 2
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(_split_row(lines[i]))
                i += 1
            out.append(_render_table(header, rows))
            continue
        # 标题
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>")
            i += 1
            continue
        # 列表（无序 / 有序）
        if re.match(r"^(\s*)([-*+])\s+", line) or re.match(r"^(\s*)\d+\.\s+", line):
            buf = []
            while i < len(lines) and (
                re.match(r"^(\s*)([-*+])\s+", lines[i])
                or re.match(r"^(\s*)\d+\.\s+", lines[i])
            ):
                mm = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", lines[i])
                buf.append(_inline(mm.group(3)))
                i += 1
            out.append("<ul>" + "".join(f"<li>{x}</li>" for x in buf) + "</ul>")
            continue
        # 空行
        if not line.strip():
            i += 1
            continue
        # 段落（合并连续正文行）
        para = [line]
        i += 1
        while (
            i < len(lines)
            and lines[i].strip()
            and not lines[i].lstrip().startswith("|")
            and not lines[i].strip().startswith("```")
            and not re.match(r"^#{1,6}\s", lines[i])
            and not re.match(r"^(\s*)([-*+])\s+", lines[i])
            and not re.match(r"^(\s*)\d+\.\s+", lines[i])
        ):
            para.append(lines[i])
            i += 1
        out.append("<p>" + _inline(" ".join(s.strip() for s in para)) + "</p>")

    style = (
        "body{font-family:system-ui,'Microsoft YaHei',sans-serif;line-height:1.6;"
        "margin:12px}table{border-collapse:collapse;margin:8px 0}"
        "td,th{border:1px solid #ccc;padding:4px 8px}th{background:#f0f0f0}"
        "pre{background:#f5f5f5;padding:8px;overflow:auto;white-space:pre-wrap}"
        "code{font-family:monospace}img{max-width:100%}"
    )
    return (
        "<html><head><meta charset='utf-8'><style>"
        + style
        + "</style></head><body>"
        + "".join(out)
        + "</body></html>"
    )


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #

class OcrWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(list)   # list[dict{page_no,text,img_png}]
    error = pyqtSignal(str)

    def __init__(self, filepath: str, model: str, config: AgentConfig) -> None:
        super().__init__()
        self._filepath = filepath
        self._model = model
        self._config = config

    def run(self) -> None:  # noqa: D401
        try:
            ext = self._filepath.lower()
            self.progress.emit(f"开始识别：{os.path.basename(self._filepath)}（模型={self._model}）")
            if ext.endswith(_PDF_EXTS):
                pages = _run_pdf(self._filepath, self._model, self._config)
            else:
                pages = _run_image(self._filepath, self._model, self._config)
            self.progress.emit(f"识别完成，共 {len(pages)} 页/张")
            self.finished.emit(pages)
        except Exception as e:  # noqa: BLE001
            logger.exception("OCR test failed")
            self.error.emit(f"识别失败：{e}")


# --------------------------------------------------------------------------- #
# 主窗口
# --------------------------------------------------------------------------- #

class OcrTestWindow(QWidget):
    def __init__(self, config: AgentConfig) -> None:
        super().__init__()
        self._config = config
        self._pages: List[Dict[str, Any]] = []
        self._current = 0
        self._worker: Optional[OcrWorker] = None
        self.setWindowTitle("OCR 识别测试")
        self.resize(1100, 760)
        self._build_ui()
        self._refresh_nav_buttons()

    # -- UI 构建 ----------------------------------------------------------- #
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # 顶部工具栏
        bar = QHBoxLayout()
        bar.setSpacing(8)
        bar.addWidget(QLabel("待识别文件"))
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("图片（png/jpg/...）或 PDF")
        bar.addWidget(self.file_edit, 3)
        pick_btn = QPushButton("选择")
        pick_btn.clicked.connect(self._on_pick_file)
        bar.addWidget(pick_btn)
        bar.addWidget(QLabel("识别模型"))
        self.model_combo = QComboBox()
        # 「自定义 OCR」选项标签自动带上设置面板里配置的模型名（路径最后一段），
        # 让用户在下拉里就能区分当前用的是哪个 Ovis 模型。
        _custom_ocr_path = getattr(self.config, "rag_ocr_model_path", "") or ""
        _custom_ocr_name = os.path.basename(_custom_ocr_path) if _custom_ocr_path else ""
        for label, value in OCR_MODELS:
            if value == "custom" and _custom_ocr_name:
                label = f"自定义OCR ({_custom_ocr_name})"
            self.model_combo.addItem(label, value)
        bar.addWidget(self.model_combo, 2)
        start_btn = QPushButton("开始识别")
        start_btn.setStyleSheet("font-weight:600;")
        start_btn.clicked.connect(self._on_start)
        bar.addWidget(start_btn)
        save_btn = QPushButton("保存结果")
        save_btn.clicked.connect(self._on_save)
        bar.addWidget(save_btn)
        root.addLayout(bar)

        # 左侧导航条（仅多页 PDF 时显示）
        nav = QHBoxLayout()
        nav.setSpacing(6)
        self.prev_btn = QPushButton("上一页")
        self.prev_btn.clicked.connect(self._on_prev)
        self.next_btn = QPushButton("下一页")
        self.next_btn.clicked.connect(self._on_next)
        self.page_label = QLabel("— / —")
        self.page_label.setMinimumWidth(70)
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        nav.addWidget(self.page_label)
        nav.addStretch(1)
        root.addLayout(nav)

        # 左右分栏
        split = QSplitter(Qt.Orientation.Horizontal)

        self.left_view = QLabel()
        self.left_view.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.left_view.setStyleSheet("background:#1e1e1e;border:1px solid #555;")
        self.left_view.setMinimumSize(360, 480)
        self.left_view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        split.addWidget(self.left_view)

        self.right_view = QTextBrowser()
        self.right_view.setOpenExternalLinks(True)
        self.right_view.setHtml("<p style='color:gray'>识别结果将显示在这里。</p>")
        split.addWidget(self.right_view)

        split.setSizes([480, 620])
        root.addWidget(split, 1)

        # 底部状态
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color:#666;")
        root.addWidget(self.status_label)

    # -- 交互 -------------------------------------------------------------- #
    def _on_pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择待识别文件", "",
            "图片/PDF (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.pdf);;所有文件 (*)",
        )
        if path:
            self.file_edit.setText(path)

    def _on_start(self) -> None:
        path = self.file_edit.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "未选择文件", "请先选择要识别的图片或 PDF 文件。")
            return
        model = self.model_combo.currentData()
        if model == "hybrid" and not path.lower().endswith(_PDF_EXTS):
            QMessageBox.information(
                self, "混合模式",
                "「混合（Vision+Ovis表格）」仅对 PDF 有意义，图片将改用 OvisOCR3 识别。",
            )
            model = "ovis"
            # 同步下拉显示
            idx = self.model_combo.findData("ovis")
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "进行中", "上一次识别尚未结束，请等待完成。")
            return
        self._pages = []
        self._current = 0
        self.right_view.setHtml("<p style='color:gray'>识别中…</p>")
        self.left_view.clear()
        self._refresh_nav_buttons()
        self._worker = OcrWorker(path, model, self._config)
        self._worker.progress.connect(self.status_label.setText)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        # 线程自然结束后自动释放 C++ 资源（即便窗口已关闭，后台跑完也会清理）。
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_finished(self, pages: List[Dict[str, Any]]) -> None:
        self._pages = pages
        self._current = 0
        self._show_page(0)
        self._refresh_nav_buttons()

    def _on_error(self, msg: str) -> None:
        self.status_label.setText(msg)
        QMessageBox.critical(self, "识别失败", msg)

    def _on_prev(self) -> None:
        if self._pages and self._current > 0:
            self._current -= 1
            self._show_page(self._current)

    def _on_next(self) -> None:
        if self._pages and self._current < len(self._pages) - 1:
            self._current += 1
            self._show_page(self._current)

    def _show_page(self, idx: int) -> None:
        if not self._pages or idx < 0 or idx >= len(self._pages):
            return
        page = self._pages[idx]
        # 左侧原文件
        img = QImage.fromData(page["img_png"])
        if img.isNull():
            self.left_view.setText("(无法显示原图)")
        else:
            pm = QPixmap.fromImage(img)
            scaled = pm.scaled(
                self.left_view.width() or 460, self.left_view.height() or 640,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.left_view.setPixmap(scaled)
        # 右侧识别结果
        self.right_view.setHtml(markdown_to_html(page["text"]))
        # 页码
        self.page_label.setText(f"{idx + 1} / {len(self._pages)}")

    def _refresh_nav_buttons(self) -> None:
        multi = len(self._pages) > 1
        self.prev_btn.setEnabled(multi)
        self.next_btn.setEnabled(multi)
        if not multi:
            self.page_label.setText("单页" if self._pages else "— / —")

    def _on_save(self) -> None:
        if not self._pages:
            QMessageBox.information(self, "无结果", "请先完成一次识别再保存结果。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存识别结果", "",
            "Markdown (*.md);;HTML (*.html);;纯文本 (*.txt)",
        )
        if not path:
            return
        lower = path.lower()
        if len(self._pages) > 1:
            msg = QMessageBox(self)
            msg.setWindowTitle("保存范围")
            msg.setText("当前为多页文档，保存范围？")
            b_all = msg.addButton("保存全部", QMessageBox.ButtonRole.AcceptRole)
            b_cur = msg.addButton("仅当前页", QMessageBox.ButtonRole.ActionRole)
            msg.addButton("取消", QMessageBox.ButtonRole.RejectRole)
            msg.exec()
            clicked = msg.clickedButton()
            if clicked is b_all:
                save_all = True
            elif clicked is b_cur:
                save_all = False
            else:
                return
        else:
            save_all = False

        if save_all:
            chosen = self._pages
        else:
            chosen = [self._pages[self._current]]

        if lower.endswith(".html"):
            content = self._to_html_doc(chosen)
        elif lower.endswith(".txt"):
            content = "\n\n".join(p["text"] for p in chosen)
        else:
            # .md：保留 markdown 原文，多页用分隔线区分
            parts = []
            for p in chosen:
                if len(chosen) > 1:
                    parts.append(f"# 第 {p['page_no'] + 1} 页\n\n{p['text']}")
                else:
                    parts.append(p["text"])
            content = "\n\n---\n\n".join(parts)

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            self.status_label.setText(f"已保存：{path}")
            QMessageBox.information(self, "已保存", f"识别结果已保存到：\n{path}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "保存失败", f"写入文件失败：{e}")

    def _to_html_doc(self, pages: List[Dict[str, Any]]) -> str:
        chunks = []
        for p in pages:
            if len(pages) > 1:
                chunks.append(f"<h2>第 {p['page_no'] + 1} 页</h2>")
            chunks.append(markdown_to_html(p["text"]))
        joined = "\n".join(chunks)
        return (
            "<html><head><meta charset='utf-8'></head><body>" + joined + "</body></html>"
        )

    # -- 资源释放 ---------------------------------------------------------- #
    def closeEvent(self, event) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(2000)
        # 释放可能被本窗口加载的 OCR 资源，避免长期占用统一内存。
        try:
            ovis_ocr.release_engine()
        except Exception:  # noqa: BLE001
            pass
        try:
            parsers.release_ocr_engine()
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(event)

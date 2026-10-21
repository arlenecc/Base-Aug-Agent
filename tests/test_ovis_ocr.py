# OvisOCR2 引擎单测（全部 mock VLM 推理，不加载模型）
"""覆盖：HTML 表格归一、表格区域双路检测、文本页表格填回、路由分发、
引擎可用性语义。VLM 调用（ocr_image_to_markdown）一律 monkeypatch。"""
import fitz
import numpy as np
import pytest

from agent.rag import ovis_ocr
from agent.rag.ovis_ocr import OvisUnavailable, _html_tables_to_markdown


# ---------- HTML 表格 → markdown ----------

def test_html_table_to_markdown():
    html = ('<table border=1><tr><td>时间</td><td>事件</td></tr>'
            '<tr><td>2025年11月</td><td>ClawdBot诞生</td></tr></table>')
    md = _html_tables_to_markdown(html)
    assert "| 时间 | 事件 |" in md
    assert "|---|---|" in md
    assert "| 2025年11月 | ClawdBot诞生 |" in md


def test_html_table_pipe_escaped():
    """单元格内的 | 必须转义，否则会撑破 markdown 表格结构。"""
    html = "<table><tr><td>a | b</td><td>c</td></tr></table>"
    md = _html_tables_to_markdown(html)
    assert r"a \| b" in md


def test_html_table_ragged_rows_padded():
    """各行单元格数不齐时按最宽行补齐。"""
    html = ("<table><tr><td>a</td><td>b</td><td>c</td></tr>"
            "<tr><td>1</td></tr></table>")
    md = _html_tables_to_markdown(html)
    lines = [l for l in md.splitlines() if l.startswith("|")]
    # 3 列 → 每行 4 个管道符
    assert all(l.count("|") == 4 for l in lines), md


def test_html_table_with_nested_tags_and_entities():
    html = ("<table><tr><th>指标</th><th>数据</th></tr>"
            "<tr><td><b>Stars</b></td><td>1 &amp; 2<br/>3</td></tr></table>")
    md = _html_tables_to_markdown(html)
    assert "| 指标 | 数据 |" in md
    assert "1 & 2 3" in md


def test_plain_text_untouched():
    text = "没有表格的普通文本，< 不是标签。"
    assert _html_tables_to_markdown(text) == text


def test_empty_table_dropped():
    assert _html_tables_to_markdown("<table></table>") == ""


# ---------- 表格区域检测 ----------

def _vector_table_pdf(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    for r in range(5):
        y = 100 + r * 30
        page.draw_line(fitz.Point(50, y), fitz.Point(545, y))
        page.draw_line(fitz.Point(50, y), fitz.Point(50, y + 30))
        page.draw_line(fitz.Point(545, y), fitz.Point(545, y + 30))
    page.insert_text((60, 120), "cell")
    pdf = tmp_path / "vector_table.pdf"
    doc.save(str(pdf))
    doc.close()
    return pdf


def test_vector_table_region_detected(tmp_path):
    pdf = _vector_table_pdf(tmp_path)
    doc = fitz.open(str(pdf))
    regions = ovis_ocr._table_regions(doc[0])
    assert len(regions) >= 1
    x0, y0, x1, y1 = regions[0]
    assert x1 - x0 > 300 and y1 - y0 > 100  # 整表，不是表头行
    doc.close()


def test_ruled_table_with_header_band(tmp_path):
    """横线型表格 + 填充表头带：区域必须从表头带顶延伸到末条框线。"""
    from PIL import Image

    img = np.full((1170, 827), 255, dtype=np.uint8)
    for r in range(8):  # 8 条等距横框（300 DPI 像素坐标）
        y = 250 + r * 66
        img[y:y + 4, 60:770] = 246
    img[166:226, 60:770] = 60  # 表头填充带（厚 60px）
    png = tmp_path / "t.png"
    Image.fromarray(img).convert("L").save(str(png))
    doc = fitz.open()
    page = doc.new_page(width=827 * 72 / 300, height=1170 * 72 / 300)
    page.insert_image(page.rect, filename=str(png))
    pdf = tmp_path / "ruled.pdf"
    doc.save(str(pdf))
    doc.close()

    doc = fitz.open(str(pdf))
    regions = ovis_ocr._table_regions(doc[0])
    assert len(regions) == 1, f"应只检出 1 个整表区域: {regions}"
    x0, y0, x1, y1 = regions[0]
    # 表头带顶（166px@300dpi ≈ 40pt）应被包含；末框（712px ≈ 171pt）也是
    assert y0 <= 45, f"区域顶部未延伸到表头带: y0={y0:.0f}pt"
    assert y1 >= 165, f"区域底部未到末框: y1={y1:.0f}pt"
    doc.close()


def test_text_page_no_table(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    for r in range(5):
        y = 100 + r * 60
        page.insert_text((72, y), "这是一段普通的中文正文内容，没有任何表格结构。")
    pdf = tmp_path / "plain.pdf"
    doc.save(str(pdf))
    doc.close()
    doc = fitz.open(str(pdf))
    assert ovis_ocr._table_regions(doc[0]) == []
    doc.close()


def test_longest_run_span():
    row = np.array([True, True, False, True, True, True, False])
    assert ovis_ocr._longest_run_span(row) == (3, 6)
    assert ovis_ocr._longest_run_span(np.zeros(5, dtype=bool)) == (0, 0)


# ---------- 文本页表格填回 ----------

def _two_block_page_pdf(tmp_path):
    """页首一段引言 + 一个矢量表格 + 页尾一段结语（ASCII：base14 字体无 CJK）。"""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 60), "INTRO_TOP text.")
    for r in range(4):
        y = 200 + r * 30
        page.draw_line(fitz.Point(50, y), fitz.Point(545, y))
        page.draw_line(fitz.Point(50, y), fitz.Point(50, y + 30))
        page.draw_line(fitz.Point(545, y), fitz.Point(545, y + 30))
    page.insert_text((60, 215), "cell text")
    page.insert_text((72, 380), "CONCLUSION_BOTTOM text.")
    pdf = tmp_path / "mixed.pdf"
    doc.save(str(pdf))
    doc.close()
    return pdf


def test_text_page_table_filled_back_in_place(monkeypatch, tmp_path):
    """表格 markdown 必须填在原表格的上下文位置，而非堆到页尾。"""
    pdf = _two_block_page_pdf(tmp_path)
    calls = []

    def _fake_ocr(img):
        calls.append(np.asarray(img).shape)
        return "| c1 | c2 |\n|---|---|\n| a | b |"

    monkeypatch.setattr(ovis_ocr, "ocr_image_to_markdown", _fake_ocr)
    doc = fitz.open(str(pdf))
    md = ovis_ocr._text_page_markdown(doc[0])
    doc.close()

    assert len(calls) == 1, "应恰好对表格区域调用一次 VLM"
    assert "| c1 | c2 |" in md
    i_head = md.find("INTRO_TOP")
    i_table = md.find("| c1 | c2 |")
    i_tail = md.find("CONCLUSION_BOTTOM")
    assert 0 <= i_head < i_table < i_tail, f"填回位置错误: {md!r}"
    assert "cell text" not in md, "表格原始文本应被 VLM 结果取代"


def test_text_page_vlm_failure_keeps_original_text(monkeypatch, tmp_path):
    """VLM 失败时退回表格区域原始文本，不丢内容（文本页的有意降级）。"""
    pdf = _two_block_page_pdf(tmp_path)

    def _boom(img):
        raise RuntimeError("vlm exploded")

    monkeypatch.setattr(ovis_ocr, "ocr_image_to_markdown", _boom)
    doc = fitz.open(str(pdf))
    md = ovis_ocr._text_page_markdown(doc[0])
    doc.close()
    assert "cell text" in md
    assert "INTRO_TOP" in md


def test_text_page_without_table_no_vlm_call(monkeypatch, tmp_path):
    """无表格页不调用 VLM（纯文本直取）。"""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "Plain text page without any table structure." * 3)
    pdf = tmp_path / "plain2.pdf"
    doc.save(str(pdf))
    doc.close()

    def _boom(img):
        raise AssertionError("不应调用 VLM")

    monkeypatch.setattr(ovis_ocr, "ocr_image_to_markdown", _boom)
    doc = fitz.open(str(pdf))
    md = ovis_ocr._text_page_markdown(doc[0])
    doc.close()
    assert "Plain text page" in md


# ---------- extract_markdown 路由与引擎语义 ----------

def test_extract_markdown_unsupported_suffix(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    assert ovis_ocr.extract_markdown(str(f)) is None


def test_extract_markdown_pdf_engine_unavailable(monkeypatch, tmp_path):
    """图片页（无文本层可退）遇引擎不可用 → OvisUnavailable 上抛给路由层。

    注意语义差异：文本页的表格 VLM 失败会被吞掉降级为原文本（有用），
    只有图片页「没有引擎就什么都没有」时才上抛。
    """
    from PIL import Image

    png = tmp_path / "scan.png"
    arr = np.full((400, 300), 255, dtype=np.uint8)
    arr[100:200, 50:250] = 0  # 非空白：有内容
    Image.fromarray(arr).save(str(png))
    doc = fitz.open()
    page = doc.new_page(width=300 * 72 / 150, height=400 * 72 / 150)
    page.insert_image(page.rect, filename=str(png))
    pdf = tmp_path / "scan.pdf"
    doc.save(str(pdf))
    doc.close()

    def _raise():
        raise OvisUnavailable("no model")

    monkeypatch.setattr(ovis_ocr, "_get_engine", _raise)
    with pytest.raises(OvisUnavailable):
        ovis_ocr.extract_markdown(str(pdf))


def test_extract_markdown_image_file(monkeypatch, tmp_path):
    from PIL import Image

    png = tmp_path / "page.png"
    Image.new("L", (400, 200), 255).save(str(png))
    monkeypatch.setattr(ovis_ocr, "ocr_image_to_markdown",
                        lambda img: "# 标题\n\n内容")
    md = ovis_ocr.extract_markdown(str(png))
    assert md == "# 标题\n\n内容"


def test_extract_markdown_pdf_image_page(monkeypatch, tmp_path):
    """无文本层且纯白的页：std<1 跳过 → 无输出 None。"""
    doc = fitz.open()
    doc.new_page(width=300, height=400)
    pdf = tmp_path / "img.pdf"
    doc.save(str(pdf))
    doc.close()

    assert ovis_ocr.extract_markdown(str(pdf)) is None


# ---------- 配置与可用性 ----------

def test_configure_and_resolve(monkeypatch, tmp_path):
    monkeypatch.setattr(ovis_ocr, "_config",
                        {"backend": "mlx", "model": "m", "path": ""})
    ovis_ocr.configure(backend="onnx", model="m2")
    assert ovis_ocr._config["backend"] == "onnx"
    assert ovis_ocr._config["model"] == "m2"
    # 空参数不覆盖
    ovis_ocr.configure(backend="", model="")
    assert ovis_ocr._config["backend"] == "onnx"


def test_is_ovis_available_backend_gate(monkeypatch):
    monkeypatch.setattr(ovis_ocr, "_config",
                        {"backend": "onnx", "model": "m", "path": "x"})
    assert ovis_ocr.is_ovis_available() is False


def test_is_ovis_available_model_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(ovis_ocr, "_config",
                        {"backend": "mlx", "model": "m", "path": str(tmp_path)})
    assert ovis_ocr.is_ovis_available() is True
    monkeypatch.setattr(ovis_ocr, "_config",
                        {"backend": "mlx", "model": "m", "path": str(tmp_path / "nope")})
    assert ovis_ocr.is_ovis_available() is False


def test_engine_backend_not_implemented(monkeypatch):
    """ONNX/GGUF 后端：明确错误而非静默失败。"""
    monkeypatch.setattr(ovis_ocr, "_config",
                        {"backend": "gguf", "model": "m", "path": "x"})
    monkeypatch.setattr(ovis_ocr, "_engine_failed", False)
    with pytest.raises(OvisUnavailable, match="gguf"):
        ovis_ocr._get_engine()


def test_engine_model_dir_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(ovis_ocr, "_config",
                        {"backend": "mlx", "model": "m",
                         "path": str(tmp_path / "missing")})
    monkeypatch.setattr(ovis_ocr, "_engine_failed", False)
    with pytest.raises(OvisUnavailable, match="不存在"):
        ovis_ocr._get_engine()
    # 失败标记：后续调用直接抛，不重复尝试
    with pytest.raises(OvisUnavailable):
        ovis_ocr._get_engine()

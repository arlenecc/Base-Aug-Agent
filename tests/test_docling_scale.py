# 动态 OCR scale：PDF 逐页分析分类 + 分段切分
"""docling 的 OcrAutoOptions.scale 是转换器级参数，动态调整的做法是：
先用 PyMuPDF 逐页分析分类（_analyze_pdf_pages），再按档位切连续段
（_extract_pdf_by_scale 的 runs 构造）。本测试覆盖分类规则与切分逻辑，
全部用合成 PDF，不依赖外部文件。
"""
import fitz
import numpy as np
import pytest

from agent.rag import docling_parser as dp


def _make_raster_pdf(path, pages_pixmaps, dpi):
    """把 (灰度数组) 列表栅格化成纯图片 PDF（模拟扫描件）。"""
    from PIL import Image

    out = fitz.open()
    png = path.with_suffix(".png")
    for arr in pages_pixmaps:
        Image.fromarray(arr).convert("L").save(str(png))
        page = out.new_page(width=arr.shape[1] * 72 / dpi,
                            height=arr.shape[0] * 72 / dpi)
        page.insert_image(page.rect, filename=str(png))
    out.save(str(path))
    out.close()


def _text_page_image(w=827, h=1170):
    """纯文字页：真实字号（10pt@300dpi ≈ 42px 行高）的断续笔画。

    两个不许违反的真实性约束：
      1. 行高必须 >=30px@300dpi（→150 DPI 分类渲染后 >=15px，超过细带
         阈值 5px）；
      2. 笔画必须断续——但注意降采样会把过密的笔画平均成连续暗带（旧版
         用 12px 字 + 4px 间隙、100 DPI 3:1 降采样就中招），所以字距要
         足够开（>=12px@300dpi，150 DPI 2:1 降采样后仍可辨）。
    """
    img = np.full((h, w), 255, dtype=np.uint8)
    for r in range(5):
        y = 200 + r * 120          # 行高 42px（10pt@300dpi），间距充足
        x = 100
        while x < 700:
            img[y:y + 36, x:x + 30] = 0   # 「字符」30px 宽
            x += 42                       # 字间 12px，降采样后仍可辨
    return img


def _table_page_image(w=827, h=1170, gray=246, thick=4):
    """横线型表格页：>=5 条等距通线（模拟印刷框线）。

    ``gray`` 控制框线深浅：浅灰 246 复刻「被降采样成浅灰的细线」真实场景，
    0（黑）覆盖深色框线——检测器不能按墨核深浅判线（黑线本身就是墨核）。
    ``thick`` 为 300 DPI 下的线宽（像素）：4px ≈ 1pt 印刷框线，降到 100 DPI
    仍有 ~1.3px，能被非白(250)阈值抓住。
    """
    img = np.full((h, w), 255, dtype=np.uint8)
    for r in range(8):  # 8 条横框，等距
        y = 200 + r * 66
        img[y:y + thick, 60:770] = gray
    # 行间填一些文字（墨核）证明内容行不会被误判为框线
    for r in range(7):
        y = 210 + r * 66
        img[y + 10:y + 18, 70:400] = 0
    return img


def _dotted_leader_page(w=827, h=1170):
    """目录点线页：通页宽但断续的点线 + 文字（旧实现的误报源）。"""
    img = np.full((h, w), 255, dtype=np.uint8)
    for r in range(6):
        y = 200 + r * 66
        # 断续点线：每 10 像素 3 像素墨
        x = 60
        while x < 770:
            img[y:y + 2, x:min(x + 3, 770)] = 246
            x += 10
        img[y + 10:y + 18, 70:300] = 0  # 文字行（带墨核）
    return img


def _analyze(tmp_path, images, dpi, name="doc.pdf"):
    pdf = tmp_path / name
    _make_raster_pdf(pdf, images, dpi)
    return dp._analyze_pdf_pages(pdf)


# ---------- 分类规则 ----------

def test_text_layer_pdf_uses_default_scale(tmp_path):
    """规则 1：有文本层的 PDF（纯文本文档）→ 默认档。"""
    doc = fitz.open()
    for _ in range(3):
        page = doc.new_page(width=595, height=842)
        # 超过 _TEXT_LAYER_MIN_CHARS(100) 才算文本层页
        page.insert_text((72, 100), "这是一段足够长的中文文本内容。" * 10)
    pdf = tmp_path / "text.pdf"
    doc.save(str(pdf))
    doc.close()

    profiles = dp._analyze_pdf_pages(pdf)
    assert profiles is not None and len(profiles) == 3
    assert all(p.has_text for p in profiles)
    assert all(p.scale == dp._OCR_SCALE_DEFAULT for p in profiles)


def test_low_res_scan_uses_default_scale(tmp_path):
    """规则 2：扫描件原始分辨率 < 150 DPI → 默认档（升采样无意义）。"""
    profiles = _analyze(tmp_path, [_table_page_image()], dpi=96, name="lo.pdf")
    assert profiles is not None
    assert profiles[0].source_dpi is not None
    assert profiles[0].source_dpi < dp._SOURCE_DPI_CEILING
    assert profiles[0].scale == dp._OCR_SCALE_DEFAULT


def test_high_res_table_page_uses_table_scale(tmp_path):
    """规则 3：高分辨率扫描 + 表格页 → 高保档（浅灰/深黑框线都要认）。"""
    for gray in (246, 0):  # 浅灰细线（降采样后 ~246）与黑色印刷框线
        profiles = _analyze(tmp_path, [_table_page_image(gray=gray)], dpi=300,
                            name=f"hi{gray}.pdf")
        assert profiles is not None
        assert profiles[0].has_table, f"gray={gray} 未识别为表格"
        assert profiles[0].scale == dp._OCR_SCALE_TABLE


def test_high_res_text_page_uses_default_scale(tmp_path):
    """规则 4：高分辨率扫描 + 无表格页 → 默认档。"""
    profiles = _analyze(tmp_path, [_text_page_image()], dpi=300, name="txt.pdf")
    assert profiles is not None
    assert not profiles[0].has_table
    assert profiles[0].scale == dp._OCR_SCALE_DEFAULT


def test_dotted_leader_not_table(tmp_path):
    """目录点线页（通页宽但断续）不得判为表格。"""
    profiles = _analyze(tmp_path, [_dotted_leader_page()], dpi=300,
                        name="dots.pdf")
    assert profiles is not None
    assert not profiles[0].has_table
    assert profiles[0].scale == dp._OCR_SCALE_DEFAULT


def test_vector_table_detected(tmp_path):
    """原生数字表格（矢量长直线）→ 高保档。"""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    for r in range(5):
        y = 100 + r * 30
        page.draw_line(fitz.Point(50, y), fitz.Point(545, y))
        page.draw_line(fitz.Point(50, y), fitz.Point(50, y + 30))
        page.draw_line(fitz.Point(545, y), fitz.Point(545, y + 30))
    pdf = tmp_path / "vector.pdf"
    doc.save(str(pdf))
    doc.close()

    profiles = dp._analyze_pdf_pages(pdf)
    assert profiles is not None
    assert profiles[0].has_table
    assert profiles[0].scale == dp._OCR_SCALE_TABLE


def test_mixed_pdf_classification(tmp_path):
    """混合文档：低分辨率表格页 / 高分辨率文字页 / 高分辨率表格页。"""
    profiles = _analyze(
        tmp_path,
        [_table_page_image(), _text_page_image(), _table_page_image()],
        dpi=300, name="mixed.pdf")
    assert profiles is not None and len(profiles) == 3
    assert profiles[0].scale == dp._OCR_SCALE_TABLE   # 表格页
    assert profiles[1].scale == dp._OCR_SCALE_DEFAULT  # 文字页
    assert profiles[2].scale == dp._OCR_SCALE_TABLE   # 表格页


# ---------- 分段切分 ----------

def test_contiguous_runs_split_and_order():
    """连续段构造 + 按段首页码排序 = 原页序（拼接正确性依赖此）。"""
    from agent.rag.docling_parser import _PageProfile

    profiles = [
        _PageProfile(0, False, 300.0, True, 2.5),
        _PageProfile(1, False, 300.0, True, 2.5),
        _PageProfile(2, False, 300.0, False, 2.0),
        _PageProfile(3, False, 300.0, True, 2.5),
    ]
    runs = []
    for p in profiles:
        if runs and runs[-1][0] == p.scale:
            runs[-1][1].append(p.index)
        else:
            runs.append((p.scale, [p.index]))
    assert runs == [(2.5, [0, 1]), (2.0, [2]), (2.5, [3])]

    # 模拟 parts 按段首页码重排（_extract_pdf_by_scale 的拼接逻辑）
    parts = {3: "C", 2: "B", 0: "A"}
    joined = "\n\n".join(parts[k] for k in sorted(parts))
    assert joined == "A\n\nB\n\nC"


# ---------- 分析缓存与失败回退 ----------

# ---------- 动态倍率开关（默认关，config.rag_docling_dynamic_scale） ----------

def _fake_converter(markdown="# fake"):
    class _Doc:
        def export_to_markdown(self):
            return markdown

    class _Result:
        document = _Doc()

    class _FakeConverter:
        def convert(self, path):
            return _Result()

    return _FakeConverter()


def _one_page_pdf(tmp_path, name="one.pdf"):
    """最小单页 PDF（有文本层，内容随意——门控测试只关心路由）。"""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "Any page content for routing tests." * 3)
    pdf = tmp_path / name
    doc.save(str(pdf))
    doc.close()
    return pdf


def test_dynamic_scale_default_off(tmp_path, monkeypatch):
    """默认关闭：extract_markdown 不走逐页分析，整篇固定 2.0（用户要求的默认行为）。"""
    pdf = _one_page_pdf(tmp_path)
    analyzed = []
    monkeypatch.setattr(dp, "is_docling_available", lambda: True)
    monkeypatch.setattr(dp, "_analyze_pdf_pages",
                        lambda p: analyzed.append(p) or None)
    scales_seen = []

    def _fake_get(scale=dp._OCR_SCALE_DEFAULT):
        scales_seen.append(scale)
        return _fake_converter()

    monkeypatch.setattr(dp, "_get_converter", _fake_get)
    md = dp.extract_markdown(str(pdf))
    assert md == "# fake"
    assert not analyzed, "默认关闭时不应触发逐页分析"
    assert scales_seen == [dp._OCR_SCALE_DEFAULT], \
        f"默认应只用 2.0 档转换器，实际: {scales_seen}"


def test_dynamic_scale_opt_in(tmp_path, monkeypatch):
    """显式开启后：走逐页分析 + 分段转换路径。"""
    pdf = _one_page_pdf(tmp_path, name="one2.pdf")
    seg_calls = []
    dp.configure_dynamic_scale(True)
    try:
        monkeypatch.setattr(dp, "is_docling_available", lambda: True)
        monkeypatch.setattr(dp, "_analyze_pdf_pages", lambda p: "FAKE_PROFILES")
        monkeypatch.setattr(dp, "_extract_pdf_by_scale",
                            lambda p, profiles: seg_calls.append(profiles) or "# seg")
        md = dp.extract_markdown(str(pdf))
    finally:
        dp.configure_dynamic_scale(False)  # 恢复默认，避免污染其它测试
    assert md == "# seg"
    assert seg_calls == ["FAKE_PROFILES"]


def test_configure_dynamic_scale_semantics():
    """开关语义：配置即状态，默认关闭。"""
    assert dp._dynamic_scale_enabled is False
    dp.configure_dynamic_scale(True)
    assert dp._dynamic_scale_enabled is True
    dp.configure_dynamic_scale(False)
    assert dp._dynamic_scale_enabled is False


def test_config_default_off():
    """config.py 默认值：动态倍率关闭（2026-10 实测代价/收益不成比例）。"""
    from agent.config import AgentConfig

    assert AgentConfig().rag_docling_dynamic_scale is False


def test_analysis_cache_hit(tmp_path):
    """同文件二次分析命中缓存（同一对象）。"""
    profiles = _analyze(tmp_path, [_text_page_image()], dpi=300, name="c.pdf")
    assert dp._analyze_pdf_pages(tmp_path / "c.pdf") is profiles


def test_analysis_failure_returns_none(tmp_path):
    """损坏/非 PDF 文件分析失败返回 None（调用方回退整篇转换）。"""
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf at all")
    assert dp._analyze_pdf_pages(bad) is None

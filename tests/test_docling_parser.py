"""Tests for the docling unified parser (graceful fallback behaviour).

2026-10（#133）：撤掉「扫描件/图片重 → 轻量路径」拦截后，docling 全量
接管 PDF；旧测试引用的 ``_is_scan_pdf`` / ``_pdf_is_image_heavy`` 已删除，
相关用例随之移除。
"""
from __future__ import annotations

import pytest


def test_extract_markdown_returns_none_when_docling_missing(monkeypatch):
    """Without docling, extract_markdown returns None (caller falls back)."""
    from agent.rag import docling_parser as dp

    monkeypatch.setattr(dp, "is_docling_available", lambda: False)
    assert dp.extract_markdown("anything.pdf") is None


def test_extract_markdown_skips_unsupported_extension(monkeypatch):
    """Non-docling formats (txt/md/csv) return None."""
    from agent.rag import docling_parser as dp

    monkeypatch.setattr(dp, "is_docling_available", lambda: True)
    assert dp.extract_markdown("notes.txt") is None
    assert dp.extract_markdown("data.csv") is None


def test_is_supported_extensions():
    from agent.rag import docling_parser as dp

    assert dp.is_supported(".pdf")
    assert dp.is_supported(".docx")
    assert dp.is_supported(".xlsx")
    assert dp.is_supported(".pptx")
    # .epub 已改由 ebook 路径（pandoc convert_ebook → ebook_parser）处理，
    # 不再进 docling（加载布局模型 + TableFormer 对 epub 纯属浪费）。
    assert not dp.is_supported(".epub")
    assert not dp.is_supported(".txt")
    assert not dp.is_supported(".csv")


def test_extract_markdown_converts_via_docling(monkeypatch, tmp_path):
    """When docling is available, return Markdown (all PDFs go to docling)."""
    from agent.rag import docling_parser as dp

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    class _FakeDoc:
        def export_to_markdown(self):
            return "# Title\n\nSome content."

    class _FakeResult:
        document = _FakeDoc()

    class _FakeConverter:
        def convert(self, path):
            return _FakeResult()

    monkeypatch.setattr(dp, "is_docling_available", lambda: True)
    monkeypatch.setattr(dp, "_get_converter", lambda scale=dp._OCR_SCALE_DEFAULT: _FakeConverter())

    md = dp.extract_markdown(str(pdf))
    assert md == "# Title\n\nSome content."


def test_extract_markdown_converter_init_failure(monkeypatch, tmp_path):
    """If converter init fails, return None (fallback)."""
    from agent.rag import docling_parser as dp

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    monkeypatch.setattr(dp, "is_docling_available", lambda: True)
    monkeypatch.setattr(dp, "_get_converter", lambda scale=dp._OCR_SCALE_DEFAULT: None)

    assert dp.extract_markdown(str(pdf)) is None


def test_extract_markdown_empty_output_returns_none(monkeypatch, tmp_path):
    """Empty markdown must return None so the caller falls back."""
    from agent.rag import docling_parser as dp

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    class _FakeDoc:
        def export_to_markdown(self):
            return "   \n  "

    class _FakeResult:
        document = _FakeDoc()

    class _FakeConverter:
        def convert(self, path):
            return _FakeResult()

    monkeypatch.setattr(dp, "is_docling_available", lambda: True)
    monkeypatch.setattr(dp, "_get_converter", lambda scale=dp._OCR_SCALE_DEFAULT: _FakeConverter())

    assert dp.extract_markdown(str(pdf)) is None


def test_extract_markdown_conversion_error_returns_none(monkeypatch, tmp_path):
    """A per-file conversion failure must not raise (caller falls back)."""
    from agent.rag import docling_parser as dp

    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    class _BrokenConverter:
        def convert(self, path):
            raise RuntimeError("corrupt page 3")

    monkeypatch.setattr(dp, "is_docling_available", lambda: True)
    monkeypatch.setattr(dp, "_get_converter", lambda scale=dp._OCR_SCALE_DEFAULT: _BrokenConverter())

    assert dp.extract_markdown(str(pdf)) is None


def test_build_converter_uses_vlm_pipeline_when_enabled(monkeypatch):
    """rag_docling_vlm 开启时，_build_converter 走 ExtractionVlmPipeline + granite_docling。"""
    pytest.importorskip("docling.pipeline.vlm_pipeline")
    from agent.rag import docling_parser as dp
    from docling.document_converter import PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.pipeline.vlm_pipeline import VlmPipeline
    from docling.datamodel.pipeline_options import VlmPipelineOptions

    captured = {}
    class _FakeConverter:
        def __init__(self, **kwargs):
            captured.update(kwargs)
    monkeypatch.setattr("docling.document_converter.DocumentConverter", _FakeConverter)
    monkeypatch.setattr(dp, "_docling_vlm_enabled", True)
    dp._build_converter(2.0)
    fo = captured["format_options"][InputFormat.PDF]
    assert fo.pipeline_cls is VlmPipeline
    assert isinstance(fo.pipeline_options, VlmPipelineOptions)
    # 默认即 granite_docling 预设（端到端版面/表格/OCR 合一）
    assert fo.pipeline_options.vlm_options is not None


def test_build_converter_standard_pipeline_by_default(monkeypatch):
    """默认（VLM 关）仍走标准 PDF 管线（StandardPdfPipeline + PdfPipelineOptions+OCR）。"""
    from agent.rag import docling_parser as dp
    from docling.document_converter import PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    captured = {}
    class _FakeConverter:
        def __init__(self, **kwargs):
            captured.update(kwargs)
    monkeypatch.setattr("docling.document_converter.DocumentConverter", _FakeConverter)
    monkeypatch.setattr(dp, "_docling_vlm_enabled", False)
    dp._build_converter(2.0)
    fo = captured["format_options"][InputFormat.PDF]
    # 默认 pipeline_cls = StandardPdfPipeline（PdfFormatOption 类默认值）
    assert fo.pipeline_cls.__name__ == "StandardPdfPipeline"
    assert isinstance(fo.pipeline_options, PdfPipelineOptions)
    assert fo.pipeline_options.ocr_options is not None

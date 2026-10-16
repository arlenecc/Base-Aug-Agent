"""Tests for the docling unified parser (graceful fallback behaviour)."""
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
    assert dp.is_supported(".epub")
    assert not dp.is_supported(".txt")
    assert not dp.is_supported(".csv")


def test_extract_markdown_scan_pdf_delegated(monkeypatch, tmp_path):
    """A pure-scan PDF is delegated to the RapidOCR path (returns None)."""
    from agent.rag import docling_parser as dp

    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    monkeypatch.setattr(dp, "is_docling_available", lambda: True)
    monkeypatch.setattr(dp, "_is_scan_pdf", lambda path: True)
    assert dp.extract_markdown(str(pdf)) is None


def test_extract_markdown_converts_via_docling(monkeypatch, tmp_path):
    """When docling is available and the file is text-based, return Markdown."""
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
    monkeypatch.setattr(dp, "_is_scan_pdf", lambda path: False)
    monkeypatch.setattr(dp, "_get_converter", lambda: _FakeConverter())

    md = dp.extract_markdown(str(pdf))
    assert md == "# Title\n\nSome content."


def test_extract_markdown_converter_init_failure(monkeypatch, tmp_path):
    """If converter init fails, return None (fallback)."""
    from agent.rag import docling_parser as dp

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    monkeypatch.setattr(dp, "is_docling_available", lambda: True)
    monkeypatch.setattr(dp, "_is_scan_pdf", lambda path: False)
    monkeypatch.setattr(dp, "_get_converter", lambda: None)

    assert dp.extract_markdown(str(pdf)) is None


def test_extract_markdown_image_heavy_pdf_delegated(monkeypatch, tmp_path):
    """A PDF full of per-page background images is delegated (returns None).

    #116：每页「文本层 + 满页背景图」的中文电子书会让 docling 逐页跑
    位图 OCR（单本 30-70 分钟、内存 30GB+），必须路由到 RapidOCR 路径。
    """
    from agent.rag import docling_parser as dp

    pdf = tmp_path / "decorated.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    monkeypatch.setattr(dp, "is_docling_available", lambda: True)
    monkeypatch.setattr(dp, "_is_scan_pdf", lambda path: False)
    monkeypatch.setattr(dp, "_pdf_is_image_heavy", lambda path: True)
    assert dp.extract_markdown(str(pdf)) is None


class _FakeFitzPage:
    """Fake fitz page: text + one image placed at the given area fraction."""

    def __init__(self, text, img_area_fraction=0.0, page_w=100.0, page_h=100.0):
        self._text = text
        self._frac = img_area_fraction
        self._page_area = page_w * page_h

        class _Rect:
            def __init__(self, area):
                self._area = area

            def __abs__(self):
                return self._area

        self.rect = _Rect(self._page_area)
        self._rect_cls = _Rect

    def get_text(self, _mode):
        return self._text

    def get_images(self, full=True):
        # 每页 1 张图（xref=7）；无图页返回空表。
        return [(7, 0, 0, 0, 0, 0, 0, 0, 0, None)] if self._frac > 0 else []

    def get_image_rects(self, xref):
        return [self._rect_cls(self._frac * self._page_area)]


def _install_fake_fitz(monkeypatch, pages):
    import sys
    import types

    class _FakeDoc:
        def __init__(self, pages):
            self._pages = pages
            self.page_count = len(pages)

        def __getitem__(self, i):
            return self._pages[i]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    fake = types.ModuleType("fitz")
    fake.open = lambda path: _FakeDoc(pages)
    monkeypatch.setitem(sys.modules, "fitz", fake)


def test_pdf_is_image_heavy_full_page_backgrounds(monkeypatch, tmp_path):
    """Every page has text + a full-page background image → heavy."""
    from agent.rag import docling_parser as dp

    pages = [_FakeFitzPage(f"page {i} text", img_area_fraction=1.0) for i in range(20)]
    _install_fake_fitz(monkeypatch, pages)
    assert dp._pdf_is_image_heavy(tmp_path / "book.pdf") is True


def test_pdf_is_image_heavy_small_logos_not_counted(monkeypatch, tmp_path):
    """Text pages with only a tiny corner logo are NOT image-heavy."""
    from agent.rag import docling_parser as dp

    # 5% 面积的小图低于 15% 阈值 → 全部为普通文本页
    pages = [_FakeFitzPage(f"page {i} text", img_area_fraction=0.05) for i in range(20)]
    _install_fake_fitz(monkeypatch, pages)
    assert dp._pdf_is_image_heavy(tmp_path / "book.pdf") is False


def test_pdf_is_image_heavy_image_only_pages(monkeypatch, tmp_path):
    """Pure image pages (no text layer) count towards the ratio."""
    from agent.rag import docling_parser as dp

    pages = [_FakeFitzPage("", img_area_fraction=1.0) for _ in range(20)]
    _install_fake_fitz(monkeypatch, pages)
    assert dp._pdf_is_image_heavy(tmp_path / "scan.pdf") is True


def test_pdf_is_image_heavy_unopenable_returns_false(monkeypatch, tmp_path):
    """Broken/unreadable files must not raise (caller falls through to docling)."""
    from agent.rag import docling_parser as dp

    import sys
    import types

    fake = types.ModuleType("fitz")

    def _boom(_path):
        raise RuntimeError("corrupt pdf")

    fake.open = _boom
    monkeypatch.setitem(sys.modules, "fitz", fake)
    assert dp._pdf_is_image_heavy(tmp_path / "broken.pdf") is False

"""Tests for the marker-pdf parser (parallel path to docling, #134)."""
from __future__ import annotations


def test_extract_markdown_returns_none_when_marker_missing(monkeypatch):
    """Without marker, extract_markdown returns None (caller falls back)."""
    from agent.rag import marker_parser as mp

    monkeypatch.setattr(mp, "is_marker_available", lambda: False)
    assert mp.extract_markdown("anything.pdf") is None


def test_extract_markdown_skips_non_pdf(monkeypatch):
    """marker is PDF-only: every other extension returns None untouched."""
    from agent.rag import marker_parser as mp

    monkeypatch.setattr(mp, "is_marker_available", lambda: True)
    assert mp.extract_markdown("notes.docx") is None
    assert mp.extract_markdown("notes.epub") is None
    assert mp.extract_markdown("notes.txt") is None


def test_is_supported_extensions():
    from agent.rag import marker_parser as mp

    assert mp.is_supported(".pdf")
    assert mp.is_supported(".PDF")  # 大小写不敏感
    assert not mp.is_supported(".docx")
    assert not mp.is_supported(".epub")
    assert not mp.is_supported(".txt")


def test_extract_markdown_converts_via_marker(monkeypatch, tmp_path):
    """When marker is available, return rendered.markdown."""
    from agent.rag import marker_parser as mp

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    class _FakeRendered:
        markdown = "# Heading\n\nBody text."

    class _FakeConverter:
        def __call__(self, path):
            return _FakeRendered()

    monkeypatch.setattr(mp, "is_marker_available", lambda: True)
    monkeypatch.setattr(mp, "_get_converter", lambda: _FakeConverter())

    assert mp.extract_markdown(str(pdf)) == "# Heading\n\nBody text."


def test_extract_markdown_empty_output_returns_none(monkeypatch, tmp_path):
    """Empty markdown must return None so the caller falls back."""
    from agent.rag import marker_parser as mp

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    class _FakeRendered:
        markdown = "  "

    class _FakeConverter:
        def __call__(self, path):
            return _FakeRendered()

    monkeypatch.setattr(mp, "is_marker_available", lambda: True)
    monkeypatch.setattr(mp, "_get_converter", lambda: _FakeConverter())

    assert mp.extract_markdown(str(pdf)) is None


def test_extract_markdown_conversion_error_returns_none(monkeypatch, tmp_path):
    """A per-file conversion failure must not raise (caller falls back)."""
    from agent.rag import marker_parser as mp

    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    class _BrokenConverter:
        def __call__(self, path):
            raise RuntimeError("surya server died")

    monkeypatch.setattr(mp, "is_marker_available", lambda: True)
    monkeypatch.setattr(mp, "_get_converter", lambda: _BrokenConverter())

    assert mp.extract_markdown(str(pdf)) is None


def test_release_converter_shuts_down_models(monkeypatch):
    """release_converter must call shutdown_models with the artifact dict."""
    from agent.rag import marker_parser as mp

    calls = {}

    def _fake_shutdown(artifact_dict):
        calls["artifact"] = artifact_dict

    import types, sys

    fake_marker_models = types.ModuleType("marker.models")
    fake_marker_models.shutdown_models = _fake_shutdown
    monkeypatch.setitem(sys.modules, "marker.models", fake_marker_models)

    converter = types.SimpleNamespace(artifact_dict={"layout": "x"})
    monkeypatch.setattr(mp, "_converter", converter, raising=False)
    mp.release_converter()
    assert calls["artifact"] == {"layout": "x"}
    assert mp._converter is None


def test_release_converter_noop_when_singleton_absent():
    """release_converter with no singleton must not raise."""
    from agent.rag import marker_parser as mp

    saved = mp._converter
    try:
        mp._converter = None
        mp.release_converter()  # should not raise
    finally:
        mp._converter = saved

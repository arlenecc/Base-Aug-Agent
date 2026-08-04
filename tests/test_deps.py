"""Tests for the RAG dependency check and auto-install module."""
import os
import tempfile
from pathlib import Path

import pytest

from src.agent.rag.deps import (
    DepSpec,
    DependencyReport,
    EXTENSION_DEPS,
    RAG_CORE_DEPS,
    VERSION_PINS,
    check_dependencies,
    ensure_dependencies,
    install_packages,
    scan_extensions,
    _is_importable,
)


class TestScanExtensions:
    def test_scan_finds_all_extensions(self, tmp_path):
        (tmp_path / "doc.pdf").write_text("x")
        (tmp_path / "notes.txt").write_text("x")
        (tmp_path / "data.xlsx").write_text("x")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "img.png").write_text("x")
        exts = scan_extensions(str(tmp_path))
        assert ".pdf" in exts
        assert ".txt" in exts
        assert ".xlsx" in exts
        assert ".png" in exts

    def test_scan_empty_dir(self, tmp_path):
        exts = scan_extensions(str(tmp_path))
        assert exts == set()

    def test_scan_nonexistent_dir(self):
        exts = scan_extensions("/nonexistent/path/xyz")
        assert exts == set()

    def test_scan_ignores_hidden_files(self, tmp_path):
        (tmp_path / ".hidden").write_text("x")
        (tmp_path / "visible.txt").write_text("x")
        exts = scan_extensions(str(tmp_path))
        assert ".txt" in exts
        # .hidden has no extension suffix, so it won't be in exts
        assert "" not in exts


class TestCheckDependencies:
    def test_check_returns_report(self, tmp_path):
        (tmp_path / "doc.pdf").write_text("x")
        (tmp_path / "notes.txt").write_text("x")
        report = check_dependencies(str(tmp_path), include_core=False)
        assert isinstance(report, DependencyReport)
        assert ".pdf" in report.scanned_extensions
        assert ".txt" in report.scanned_extensions

    def test_check_pdf_finds_fitz_dep(self, tmp_path):
        (tmp_path / "doc.pdf").write_text("x")
        report = check_dependencies(str(tmp_path), include_core=False)
        # fitz (PyMuPDF) should be in missing or already installed
        fitz_deps = [d for d in report.all_missing if d.import_name == "fitz"]
        if not _is_importable("fitz"):
            assert len(fitz_deps) == 1
            assert fitz_deps[0].required
        else:
            assert len(fitz_deps) == 0

    def test_check_txt_has_no_deps(self, tmp_path):
        (tmp_path / "notes.txt").write_text("x")
        report = check_dependencies(str(tmp_path), include_core=False)
        # .txt requires no external deps
        assert len(report.missing_required) == 0
        assert len(report.missing_optional) == 0

    def test_check_epub_finds_ebooklib(self, tmp_path):
        (tmp_path / "book.epub").write_text("x")
        report = check_dependencies(str(tmp_path), include_core=False)
        epub_deps = [d for d in report.all_missing if d.import_name == "ebooklib"]
        if not _is_importable("ebooklib"):
            assert len(epub_deps) == 1
        else:
            assert len(epub_deps) == 0

    def test_check_core_deps_included(self, tmp_path):
        (tmp_path / "notes.txt").write_text("x")
        report = check_dependencies(str(tmp_path), include_core=True)
        # chromadb and sentence_transformers should be checked
        all_imports = {d.import_name for d in report.all_missing}
        # If they're installed, they won't be in missing; if not, they will
        for dep in RAG_CORE_DEPS:
            if not _is_importable(dep.import_name):
                assert dep.import_name in all_imports

    def test_check_core_deps_excluded(self, tmp_path):
        (tmp_path / "notes.txt").write_text("x")
        report = check_dependencies(str(tmp_path), include_core=False)
        # No core deps should be checked
        for dep in RAG_CORE_DEPS:
            assert dep.import_name not in {d.import_name for d in report.all_missing}

    def test_check_ocr_is_optional(self, tmp_path):
        (tmp_path / "doc.pdf").write_text("x")
        report = check_dependencies(str(tmp_path), include_core=False)
        ocr_deps = [d for d in report.missing_optional if d.import_name == "rapidocr_onnxruntime"]
        if not _is_importable("rapidocr_onnxruntime"):
            assert len(ocr_deps) == 1
            assert not ocr_deps[0].required
        else:
            assert len(ocr_deps) == 0

    def test_check_multiple_formats(self, tmp_path):
        (tmp_path / "doc.pdf").write_text("x")
        (tmp_path / "sheet.xlsx").write_text("x")
        (tmp_path / "slides.pptx").write_text("x")
        (tmp_path / "book.epub").write_text("x")
        (tmp_path / "page.html").write_text("x")
        (tmp_path / "data.csv").write_text("x")
        report = check_dependencies(str(tmp_path), include_core=False)
        assert ".pdf" in report.scanned_extensions
        assert ".xlsx" in report.scanned_extensions
        assert ".pptx" in report.scanned_extensions
        assert ".epub" in report.scanned_extensions
        assert ".html" in report.scanned_extensions
        assert ".csv" in report.scanned_extensions


class TestDependencyReport:
    def test_empty_report(self):
        report = DependencyReport()
        assert not report.has_missing
        assert not report.has_blocking
        assert report.summary() == "所有依赖已就绪。"

    def test_required_missing(self):
        report = DependencyReport()
        report.missing_required.append(DepSpec("foo", "foo", "Test Dep"))
        assert report.has_missing
        assert report.has_blocking
        assert "缺少必要依赖" in report.summary()
        assert "Test Dep" in report.summary()

    def test_optional_missing(self):
        report = DependencyReport()
        report.missing_optional.append(DepSpec("bar", "bar", "Optional Dep"))
        assert report.has_missing
        assert not report.has_blocking
        assert "缺少可选依赖" in report.summary()

    def test_both_missing(self):
        report = DependencyReport()
        report.missing_required.append(DepSpec("foo", "foo", "Required"))
        report.missing_optional.append(DepSpec("bar", "bar", "Optional"))
        assert report.has_missing
        assert report.has_blocking
        assert len(report.all_missing) == 2


class TestIsImportable:
    def test_stdlib_module(self):
        assert _is_importable("os")
        assert _is_importable("json")
        assert _is_importable("pathlib")

    def test_nonexistent_module(self):
        assert not _is_importable("nonexistent_module_xyz_123")


class TestInstallPackages:
    def test_install_empty_list(self):
        success, msg = install_packages([])
        assert success
        assert "无需安装" in msg

    def test_install_already_installed(self):
        # Install a stdlib-only package that's always available
        success, msg = install_packages(["pip"])
        # Should succeed (pip is always installed)
        assert success


class TestEnsureDependencies:
    def test_ensure_no_deps_needed(self, tmp_path):
        (tmp_path / "notes.txt").write_text("x")
        report = ensure_dependencies(
            str(tmp_path), auto_install=True, include_core=False
        )
        assert not report.has_blocking

    def test_ensure_with_nonexistent_kb(self):
        report = ensure_dependencies(
            "/nonexistent/path", auto_install=True, include_core=False
        )
        # No extensions found, no deps to check
        assert not report.has_blocking


class TestExtensionDeps:
    def test_all_supported_extensions_have_entries(self):
        from src.agent.rag.parsers import SUPPORTED_EXTENSIONS
        for ext in SUPPORTED_EXTENSIONS:
            assert ext in EXTENSION_DEPS, f"Extension {ext} missing from EXTENSION_DEPS"

    def test_pdf_has_ocr_as_optional(self):
        pdf_deps = EXTENSION_DEPS[".pdf"]
        ocr_deps = [d for d in pdf_deps if d.import_name == "rapidocr_onnxruntime"]
        assert len(ocr_deps) == 1
        assert not ocr_deps[0].required

    def test_pdf_has_fitz_as_required(self):
        pdf_deps = EXTENSION_DEPS[".pdf"]
        fitz_deps = [d for d in pdf_deps if d.import_name == "fitz"]
        assert len(fitz_deps) == 1
        assert fitz_deps[0].required


class TestVersionPins:
    def test_numpy_pinned(self):
        assert "numpy" in VERSION_PINS
        assert "<2.0" in VERSION_PINS["numpy"]

"""Tests for document parsers (OCR singleton lifecycle)."""
from __future__ import annotations

import sys
import types

from agent.rag import parsers


def _reset_ocr_state():
    """Reset the module-level OCR singleton so tests start clean."""
    parsers._OCR_ENGINE = None


def _fake_rapidocr_module():
    """Build a fake rapidocr_onnxruntime module exposing RapidOCR()."""
    mod = types.ModuleType("rapidocr_onnxruntime")

    class _FakeRapidOCR:
        instances = 0

        def __init__(self):
            _FakeRapidOCR.instances += 1

        def __call__(self, *args, **kwargs):
            return (None, None)

    mod.RapidOCR = _FakeRapidOCR
    return mod


def test_ocr_engine_is_singleton(monkeypatch):
    """Repeated _get_ocr_engine() returns the same instance."""
    _reset_ocr_state()
    mod = _fake_rapidocr_module()
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", mod)

    e1 = parsers._get_ocr_engine()
    e2 = parsers._get_ocr_engine()
    assert e1 is not None
    assert e1 is e2
    # Only one RapidOCR instance was constructed.
    assert mod.RapidOCR.instances == 1


def test_release_ocr_engine_reclaims_and_recreates(monkeypatch):
    """release_ocr_engine() drops the singleton; next get recreates it."""
    _reset_ocr_state()
    mod = _fake_rapidocr_module()
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", mod)

    e1 = parsers._get_ocr_engine()
    assert e1 is not None

    parsers.release_ocr_engine()
    assert parsers._OCR_ENGINE is None

    e2 = parsers._get_ocr_engine()
    assert e2 is not None
    assert e2 is not e1
    # A fresh instance was constructed after release.
    assert mod.RapidOCR.instances == 2


def test_release_ocr_engine_when_not_loaded(monkeypatch):
    """Releasing when the engine was never loaded is a safe no-op."""
    _reset_ocr_state()
    parsers.release_ocr_engine()
    assert parsers._OCR_ENGINE is None


def test_get_ocr_engine_returns_none_when_missing(monkeypatch):
    """Missing rapidocr dependency returns None (caller skips OCR)."""
    _reset_ocr_state()
    # Simulate ImportError by removing the module.
    monkeypatch.delitem(sys.modules, "rapidocr_onnxruntime", raising=False)
    # Force a fresh import attempt by patching the import to raise.
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "rapidocr_onnxruntime":
            raise ImportError("no rapidocr")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert parsers._get_ocr_engine() is None
    assert parsers._OCR_ENGINE is None

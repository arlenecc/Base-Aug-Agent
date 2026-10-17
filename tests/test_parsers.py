"""Tests for document parsers (OCR singleton lifecycle)."""
from __future__ import annotations

import sys
import types

import pytest

from agent.rag import parsers


def _reset_ocr_state():
    """Reset the module-level OCR singleton so tests start clean."""
    parsers._OCR_ENGINE = None
    parsers._OCR_RETRY_AFTER = 0.0


def _fake_rapidocr_module():
    """Build a fake rapidocr_onnxruntime (1.x) module exposing RapidOCR()."""
    mod = types.ModuleType("rapidocr_onnxruntime")

    class _FakeRapidOCR:
        instances = 0

        # RapidOCR accepts config kwargs (create_ocr_engine passes
        # det_limit_side_len / det_limit_type).
        def __init__(self, **kwargs):
            _FakeRapidOCR.instances += 1

        def __call__(self, *args, **kwargs):
            return (None, None)

    mod.RapidOCR = _FakeRapidOCR
    return mod


def _fake_rapidocr3_module():
    """Build a fake rapidocr (3.x) module exposing RapidOCR(params=...)."""
    mod = types.ModuleType("rapidocr")

    class _FakeRapidOCR:
        instances = 0

        def __init__(self, params=None, config_path=None):
            _FakeRapidOCR.instances += 1

        def __call__(self, *args, **kwargs):
            return (None, None)

    mod.RapidOCR = _FakeRapidOCR
    return mod


def _install_fake_ocr(monkeypatch):
    """同时伪造 3.x 与 1.x 两个 rapidocr 模块。

    生产环境装有 rapidocr 3.x 时，只伪造 1.x 模块会让 _build_ocr_engine
    走到真实的 3.x 引擎（加载模型、跑预热）——测试因此既慢又不确定。
    这里把两个入口都换成 fake，无论环境装的是哪一代都能走同一路径。
    """
    mod3 = _fake_rapidocr3_module()
    mod1 = _fake_rapidocr_module()
    monkeypatch.setitem(sys.modules, "rapidocr", mod3)
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", mod1)
    return mod3


def test_ocr_engine_is_singleton(monkeypatch):
    """Repeated _get_ocr_engine() returns the same instance."""
    _reset_ocr_state()
    mod = _install_fake_ocr(monkeypatch)

    e1 = parsers._get_ocr_engine()
    e2 = parsers._get_ocr_engine()
    assert e1 is not None
    assert e1 is e2
    # Only one RapidOCR instance was constructed.
    assert mod.RapidOCR.instances == 1


def test_release_ocr_engine_reclaims_and_recreates(monkeypatch):
    """release_ocr_engine() drops the singleton; next get recreates it."""
    _reset_ocr_state()
    mod = _install_fake_ocr(monkeypatch)

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
    # Simulate ImportError by removing both module entries.
    monkeypatch.delitem(sys.modules, "rapidocr", raising=False)
    monkeypatch.delitem(sys.modules, "rapidocr_onnxruntime", raising=False)
    # Force a fresh import attempt by patching the import to raise.
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name in ("rapidocr", "rapidocr_onnxruntime"):
            raise ImportError("no rapidocr")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert parsers._get_ocr_engine() is None
    assert parsers._OCR_ENGINE is None


def test_build_ocr_engine_prefers_rapidocr3(monkeypatch):
    """3.x 可用时优先构造它（params 传 use_coreml），不碰 1.x 兼容层。"""
    mod3 = _fake_rapidocr3_module()
    mod1 = _fake_rapidocr_module()
    monkeypatch.setitem(sys.modules, "rapidocr", mod3)
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", mod1)
    monkeypatch.setattr(parsers, "_is_apple_silicon", lambda: True)

    eng = parsers._build_ocr_engine()
    assert mod3.RapidOCR.instances == 1
    assert mod1.RapidOCR.instances == 0
    assert eng is not None


def test_build_ocr_engine_falls_back_to_legacy(monkeypatch):
    """3.x 缺失时回退 1.x（ocr_engine.create_ocr_engine 注入 CoreML）。"""
    monkeypatch.delitem(sys.modules, "rapidocr", raising=False)
    mod1 = _fake_rapidocr_module()
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", mod1)

    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "rapidocr":
            raise ImportError("no rapidocr 3.x")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    eng = parsers._build_ocr_engine()
    assert eng is not None
    assert mod1.RapidOCR.instances == 1


def test_ocr_image_cached_supports_rapidocr3_output(monkeypatch):
    """3.x 的 RapidOCROutput（.txts）与 1.x 的 (result, elapse) 都能解析。"""

    class _Out3x:
        txts = ["第一行", "", "第三行"]

    class _Eng3x:
        def __call__(self, img):
            return _Out3x()

    class _Eng1x:
        def __call__(self, img):
            return ([["box", "旧版行", 0.9]], 0.1)

    assert parsers._ocr_image_cached(_Eng3x(), b"img", 1, "f.pdf") == "第一行\n第三行"
    assert parsers._ocr_image_cached(_Eng1x(), b"img", 1, "f.pdf") == "旧版行"


def test_pixmap_is_blank():
    """近纯色位图判 blank；含内容的位图不判。"""

    class _Pix:
        def __init__(self, samples):
            self.samples = samples

    assert parsers._pixmap_is_blank(_Pix(b"\xff" * 10000)) is True
    assert parsers._pixmap_is_blank(_Pix(b"")) is True
    mixed = bytearray(b"\xff" * 10000)
    mixed[5000] = 0
    assert parsers._pixmap_is_blank(_Pix(bytes(mixed))) is False


def test_ocr_recycle_pages_constant():
    """回收周期是主线 #130 实测定标值（改动需有实测依据）。"""
    assert parsers._OCR_RECYCLE_PAGES == 50

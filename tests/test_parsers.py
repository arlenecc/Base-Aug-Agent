"""Tests for document parsers (OCR singleton lifecycle)."""
from __future__ import annotations

import sys
import types

from agent.rag import parsers


# _get_ocr_engine() 的导入候选（顺序即优先级）：先 rapidocr 3.x，ImportError
# 时回退 rapidocr-onnxruntime 1.x。两个名字都必须打桩——只打桩其中一个，另一个
# 会在装了真实包的机器上解析到真包，测试里就真去加载 ONNX 模型了（既慢又让
# instances 计数归零，正是这 3 个用例失败的原因）。
_RAPIDOCR_IMPORT_NAMES = ("rapidocr", "rapidocr_onnxruntime")


class _FakeRapidOCR:
    """Fake RapidOCR used for both candidate import names.

    ``__init__`` 接受可选的 ``params``：Apple Silicon 上 _get_ocr_engine() 会
    走 ``RapidOCR(params={"EngineConfig.onnxruntime.use_coreml": True})`` 分支。
    用**同一个类**注册到两个模块名，实例计数才与实际构造次数一致。
    """
    instances = 0

    def __init__(self, params=None):
        _FakeRapidOCR.instances += 1
        self.params = params

    def __call__(self, *args, **kwargs):
        return (None, None)


def _reset_ocr_state():
    """Reset the module-level OCR singleton so tests start clean."""
    parsers._OCR_ENGINE = None
    _FakeRapidOCR.instances = 0


def _install_fake_rapidocr(monkeypatch):
    """Register the fake under both import names; returns the fake class."""
    for name in _RAPIDOCR_IMPORT_NAMES:
        mod = types.ModuleType(name)
        mod.RapidOCR = _FakeRapidOCR
        monkeypatch.setitem(sys.modules, name, mod)
    return _FakeRapidOCR


def test_ocr_engine_is_singleton(monkeypatch):
    """Repeated _get_ocr_engine() returns the same instance."""
    _reset_ocr_state()
    fake = _install_fake_rapidocr(monkeypatch)

    e1 = parsers._get_ocr_engine()
    e2 = parsers._get_ocr_engine()
    assert e1 is not None
    assert e1 is e2
    # Only one RapidOCR instance was constructed.
    assert fake.instances == 1


def test_release_ocr_engine_reclaims_and_recreates(monkeypatch):
    """release_ocr_engine() drops the singleton; next get recreates it."""
    _reset_ocr_state()
    fake = _install_fake_rapidocr(monkeypatch)

    e1 = parsers._get_ocr_engine()
    assert e1 is not None

    parsers.release_ocr_engine()
    assert parsers._OCR_ENGINE is None

    e2 = parsers._get_ocr_engine()
    assert e2 is not None
    assert e2 is not e1
    # A fresh instance was constructed after release.
    assert fake.instances == 2


def test_release_ocr_engine_when_not_loaded(monkeypatch):
    """Releasing when the engine was never loaded is a safe no-op."""
    _reset_ocr_state()
    parsers.release_ocr_engine()
    assert parsers._OCR_ENGINE is None


def test_get_ocr_engine_returns_none_when_missing(monkeypatch):
    """Missing rapidocr dependency returns None (caller skips OCR)."""
    _reset_ocr_state()
    # Simulate ImportError: drop **both** names from sys.modules (import 会先查
    # sys.modules，不删掉的话 __import__ 补丁根本不会被调用)，再让两个名字的
    # import 都抛错。
    for name in _RAPIDOCR_IMPORT_NAMES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name in _RAPIDOCR_IMPORT_NAMES:
            raise ImportError("no rapidocr")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert parsers._get_ocr_engine() is None
    assert parsers._OCR_ENGINE is None


def _blank_page():
    import numpy as np
    return np.zeros((8, 8), dtype=np.uint8)


def test_ocr_recognize_prefers_ocrmac_and_skips_rapidocr(monkeypatch):
    """ocrmac(Apple Vision) 命中时**不加载** RapidOCR 单例（省 ~500MB）。"""
    _reset_ocr_state()
    fake = _install_fake_rapidocr(monkeypatch)
    monkeypatch.setattr(parsers, "_ocrmac_recognize", lambda img: "视觉识别结果")

    out = parsers.ocr_recognize_array(_blank_page(), 1, "x.pdf")
    assert out == "视觉识别结果"
    assert fake.instances == 0  # RapidOCR 压根没被构造


def test_ocr_recognize_falls_back_to_rapidocr(monkeypatch):
    """ocrmac 返回空（未安装/没识别到）时回退 RapidOCR。"""
    _reset_ocr_state()
    fake = _install_fake_rapidocr(monkeypatch)
    monkeypatch.setattr(parsers, "_ocrmac_recognize", lambda img: "")

    parsers.ocr_recognize_array(_blank_page(), 1, "x.pdf")
    assert fake.instances == 1  # 确实回退了


def test_ocr_backend_available_prefers_ocrmac(monkeypatch):
    """ocrmac 可用即视为有后端，不必加载 RapidOCR。"""
    _reset_ocr_state()
    fake = _install_fake_rapidocr(monkeypatch)
    mod = types.ModuleType("ocrmac")
    monkeypatch.setitem(sys.modules, "ocrmac", mod)

    assert parsers._ocr_backend_available() is True
    assert fake.instances == 0


def test_ocr_backend_available_falls_back_to_rapidocr(monkeypatch):
    """ocrmac 缺失时以 RapidOCR 是否可用来判定。"""
    monkeypatch.delitem(sys.modules, "ocrmac", raising=False)
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "ocrmac":
            raise ImportError("no ocrmac")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    _reset_ocr_state()
    fake = _install_fake_rapidocr(monkeypatch)

    assert parsers._ocr_backend_available() is True
    assert fake.instances == 1

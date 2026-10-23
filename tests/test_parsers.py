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

    ``__init__`` 接受 ``params`` / ``providers`` 等关键字（签名兼容）：生产侧
    rapidocr 3.x 走 ``params={...}``、rapidocr_onnxruntime 1.x 走
    ``providers=[...]``，fake 都要能接住才能计数。
    用**同一个类**注册到两个模块名，实例计数才与实际构造次数一致。
    """
    instances = 0

    def __init__(self, params=None, **kwargs):
        _FakeRapidOCR.instances += 1
        self.params = params

    def __call__(self, *args, **kwargs):
        return (None, None)


def _reset_ocr_state():
    """Reset the module-level OCR state so tests start clean.

    注意：除了旧的全局单例 ``_OCR_ENGINE``，还要清空**引擎池**
    ``_OCR_ENGINES`` / ``_OCR_ENGINE_BUSY``——否则上一个测试留下的池实例
    （或它装过的 fake）会被 ``_acquire_ocr_slot`` 复用，污染本测试的 RapidOCR
    行为（顺序相关、难以排查）。
    """
    parsers._OCR_ENGINE = None
    parsers._OCR_ENGINES = [None] * parsers._OCR_ENGINE_POOL_SIZE
    parsers._OCR_ENGINE_BUSY = [False] * parsers._OCR_ENGINE_POOL_SIZE
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
    # 文本需 >= _OCRMAC_MIN_USEFUL_CHARS 才算「命中」：低于它说明 Vision 只
    # 抓到页眉/碎片，会再补一次 RapidOCR（见该常量与 test_..._thin 用例）。
    _hit = "视觉识别结果" * 12
    monkeypatch.setattr(parsers, "_ocrmac_recognize",
                        lambda img, err_out=None: _hit)

    out = parsers.ocr_recognize_array(_blank_page(), 1, "x.pdf")
    assert out == _hit
    assert fake.instances == 0  # RapidOCR 压根没被构造


def test_ocr_recognize_falls_back_to_rapidocr(monkeypatch):
    """ocrmac 本页识别不到（返回空/异常）时按页回退 RapidOCR。"""
    _reset_ocr_state()
    fake = _install_fake_rapidocr(monkeypatch)
    monkeypatch.setattr(parsers, "_ocrmac_recognize", lambda img, err_out=None: "")

    parsers.ocr_recognize_array(_blank_page(), 1, "x.pdf")
    assert fake.instances == 1  # 仅这一页回退，拉起 RapidOCR


def test_ocr_recognize_fallback_is_per_page_not_global(monkeypatch):
    """按页回退：ocrmac 命中时不加载 rapidocr，下一页空了再独立回退。"""
    _reset_ocr_state()
    fake = _install_fake_rapidocr(monkeypatch)
    seq = {"n": 0}

    def _mock(img, err_out=None):
        seq["n"] += 1
        # 第 1、3 页 ocrmac 命中，第 2 页识别不到
        return "" if seq["n"] == 2 else f"第{seq['n']}页文字" * 12

    monkeypatch.setattr(parsers, "_ocrmac_recognize", _mock)

    r1 = parsers.ocr_recognize_array(_blank_page(), 1, "x.pdf")
    r2 = parsers.ocr_recognize_array(_blank_page(), 2, "x.pdf")
    r3 = parsers.ocr_recognize_array(_blank_page(), 3, "x.pdf")

    assert r1 == "第1页文字" * 12 and r2 == "" and r3 == "第3页文字" * 12
    assert fake.instances == 1  # 只有第 2 页触发回退，引擎只构造一次


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


def test_is_scanned_epub_real_scan(monkeypatch):
    """图多且文字层极少（真扫描版）判定为扫描版。"""
    from agent.rag import pandoc_parser
    monkeypatch.setattr(pandoc_parser, "_epub_scan_profile",
                        lambda p: (250 * 1024 * 1024, 5474, []))
    assert pandoc_parser._is_scanned_epub("x.epub") is True


def test_is_scanned_epub_illustrated_book_not_scanned(monkeypatch):
    """图多但文字层充足（漫画/绘本等图文书）不算扫描版，走 pandoc/docling。"""
    from agent.rag import pandoc_parser
    monkeypatch.setattr(pandoc_parser, "_epub_scan_profile",
                        lambda p: (415 * 1024 * 1024, 483_230, []))
    assert pandoc_parser._is_scanned_epub("x.epub") is False


def test_ocr_recognize_supplements_thin_ocrmac_result(monkeypatch):
    """ocrmac 只抓到碎片（< _OCRMAC_MIN_USEFUL_CHARS）时补 RapidOCR 取更完整的那份。

    实测依据（KB 里的漫画类扫描页，同一批 8 张）：ocrmac 只出页眉几十字符，
    RapidOCR 出正文上千字符 —— 旧逻辑「ocrmac 有结果就直接采用」等于把整页
    正文丢掉，且日志上毫无痕迹。修后 8 张总字符 40 → 1314。
    """
    _reset_ocr_state()

    class _Rich:
        txts = ["这是被 RapidOCR 取回的正文，长度远超 ocrmac 抓到的页眉碎片"]

    class _RichRapidOCR(_FakeRapidOCR):
        def __call__(self, *args, **kwargs):
            return _Rich()

    for name in _RAPIDOCR_IMPORT_NAMES:
        mod = types.ModuleType(name)
        mod.RapidOCR = _RichRapidOCR
        monkeypatch.setitem(sys.modules, name, mod)

    # 只有 2 个字符 —— 典型的「只抓到页眉」
    monkeypatch.setattr(parsers, "_ocrmac_recognize",
                        lambda img, err_out=None: "页眉")

    stats = {}
    out = parsers.ocr_recognize_array(_blank_page(), 1, "x.pdf", stats=stats)
    assert out == "这是被 RapidOCR 取回的正文，长度远超 ocrmac 抓到的页眉碎片"
    assert stats.get("fallback") == 1
    assert stats.get("reasons", {}).get("thin") == 1
    assert stats.get("rapid_won") == 1

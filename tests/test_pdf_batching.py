"""docling 大文件按页分批并发转换路径验证（含 force_small 读者通道与页序拼接）。

覆盖 2026-10 对 ``_extract_pdf_by_batches`` 的重写：大 PDF（> 分批阈值且页数
超过单批页数）被切成 ≤ ``_PDF_BATCH_PAGES`` 页的临时子 PDF，各批通过
``force_small=True`` 走「读者」通道并发转换（最多 ``small_capacity`` 路），再按
原页序拼接返回。真实 docling 转换被 mock，避免加载 torch/布局模型；本文件只
验证分批数学、页序、``force_small`` 路由与并发上限逻辑。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import fitz
import pytest

from agent.rag import docling_parser as dp

MB = 1024 * 1024


class _FakeResult:
    class _Doc:
        def export_to_markdown(self):
            return "# md"

    document = _Doc()


class _FakeConverter:
    def convert(self, path):
        return _FakeResult()


def _make_pdf(path: Path, pages: int) -> None:
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page()
    doc.save(str(path))
    doc.close()


def _sparse_pdf(path: Path, size: int) -> None:
    with open(path, "wb") as f:
        f.truncate(size)


@pytest.fixture
def _reset_batching():
    saved = dp._PDF_BATCH_THRESHOLD_BYTES
    try:
        yield
    finally:
        dp._PDF_BATCH_THRESHOLD_BYTES = saved


def test_extract_pdf_by_batches_splits_joins_and_forces_small(
    monkeypatch, tmp_path, _reset_batching
):
    # 把分批阈值压到极低，让一个只有 120 页、体积很小的 PDF 也能走分批分支
    dp.configure_batching(threshold_mb=0.0001)

    state = {"force_small": [], "active": 0, "max": 0, "lock": threading.Lock()}
    monkeypatch.setattr(dp, "_get_converter", lambda scale=dp._OCR_SCALE_DEFAULT: _FakeConverter())

    def fake_convert(path, scale, force_small=False):
        with state["lock"]:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
        state["force_small"].append(force_small)
        time.sleep(0.02)  # 暴露并发窗口
        d = fitz.open(path)
        n = len(d)
        d.close()
        with state["lock"]:
            state["active"] -= 1
        return f"[{n}p]"

    monkeypatch.setattr(dp, "_convert_with_scale", fake_convert)

    pdf = tmp_path / "big.pdf"
    _make_pdf(pdf, 120)  # 120 页 → 3 批：50 / 50 / 20

    out = dp._extract_pdf_by_batches(pdf)
    assert out is not None
    # 子批按原页序（start 键）排序拼接 → 50/50/20
    assert out == "[50p]\n\n[50p]\n\n[20p]", out
    # 每批都走了 force_small 读者通道
    assert state["force_small"] == [True, True, True]
    # 并发不超过 small_capacity（线程池上限）
    assert state["max"] <= dp._conv_gate.small_capacity
    assert state["max"] >= 1


def test_extract_pdf_by_batches_skips_when_pages_under_batch(monkeypatch, tmp_path, _reset_batching):
    # 页数 <= 单批页数：直接返回 None，交给上层整篇转换（不分批）
    dp.configure_batching(threshold_mb=0.0001)

    called = []
    monkeypatch.setattr(
        dp, "_convert_with_scale",
        lambda p, s, force_small=False: (_ for _ in called.append(1)).throw(
            AssertionError("页数不足不应分批调用转换")))

    pdf = tmp_path / "small.pdf"
    _make_pdf(pdf, 30)  # 30 页 <= 单批 50
    assert dp._extract_pdf_by_batches(pdf) is None
    assert called == []


def test_force_small_routes_large_file_to_reader_channel(monkeypatch, tmp_path):
    # force_small=True 时，无论真实文件多大，传入闸门的大小都是 0（读者通道），
    # 从而大文件批次也能与别的批次并发，而非独占。
    captured = {}
    orig_acquire = dp._conv_gate.acquire

    def spy(size):
        captured.setdefault("sizes", []).append(size)
        return orig_acquire(size)

    monkeypatch.setattr(dp._conv_gate, "acquire", spy)
    monkeypatch.setattr(dp, "_get_converter", lambda scale=dp._OCR_SCALE_DEFAULT: _FakeConverter())

    big = tmp_path / "big.pdf"
    _sparse_pdf(big, 25 * MB)  # 真实 25MB，本应判为大文件独占
    dp._convert_with_scale(str(big), 2.0, force_small=True)
    assert captured["sizes"] == [0]  # size 被强制为 0 → 读者通道

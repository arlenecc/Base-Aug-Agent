"""PDF 转换按文件大小限流的并发测试（docling / marker / 闸门原语）。

策略：
  * 小文件（< 20MB）：最多 2 路并发转换，加快小文件吞吐；
  * 大文件（>= 20MB）：独占（1 路），防止大文件并发把内存撑爆。

覆盖三层：``SizeAwareConversionGate`` 原语语义（含大文件优先、防饿死）、
docling/marker 转换路径的真实并发上限、实例池的复用/容量上限/释放。
"""
from __future__ import annotations

import threading
import time

import pytest

MB = 1024 * 1024
LARGE = 20 * MB


@pytest.fixture(autouse=True)
def _clean_converter_pools():
    """实例池/借出登记是模块级全局，逐个用例前后清空，避免跨用例污染。"""
    from agent.rag import docling_parser as dp
    from agent.rag import marker_parser as mp

    dp._converters.clear()
    dp._checked_out.clear()
    mp._converters.clear()
    yield
    dp._converters.clear()
    dp._checked_out.clear()
    mp._converters.clear()


class _Concurrency:
    """记录并发进入临界区的线程数峰值，以及「大文件在场时是否有人搭车」。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._active = 0
        self.max = 0
        self.large_overlapped = False

    def enter(self, is_large: bool) -> None:
        with self._lock:
            self._active += 1
            self.max = max(self.max, self._active)
            if is_large and self._active > 1:
                self.large_overlapped = True

    def exit(self) -> None:
        with self._lock:
            self._active -= 1


def _run_parallel(fns, join_timeout: float = 10.0) -> None:
    threads = [threading.Thread(target=f) for f in fns]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=join_timeout)
    assert all(not t.is_alive() for t in threads), "线程未在超时内结束（疑似闸门死锁）"


# ---------------------------------------------------------------------------
# SizeAwareConversionGate 原语
# ---------------------------------------------------------------------------

def test_gate_small_files_capped_at_two():
    """小文件最多 2 路并发，第 3 路等前面的释放。"""
    from agent.rag.concurrency_gate import SizeAwareConversionGate

    gate = SizeAwareConversionGate(LARGE, small_capacity=2)
    rec = _Concurrency()

    def work():
        gate.acquire(1 * MB)
        rec.enter(False)
        time.sleep(0.05)
        rec.exit()
        gate.release(1 * MB)

    _run_parallel([work] * 4)
    assert rec.max == 2


def test_gate_large_file_is_exclusive():
    """大文件独占：转换期间不允许任何小文件并发进入。"""
    from agent.rag.concurrency_gate import SizeAwareConversionGate

    gate = SizeAwareConversionGate(LARGE, small_capacity=2)
    rec = _Concurrency()

    def large():
        gate.acquire(LARGE)
        rec.enter(True)
        time.sleep(0.1)
        rec.exit()
        gate.release(LARGE)

    def small():
        gate.acquire(1 * MB)
        rec.enter(False)
        time.sleep(0.05)
        rec.exit()
        gate.release(1 * MB)

    # 大文件先拿到闸门，小文件随后到达必须等它跑完
    large_started = threading.Event()
    large_done = threading.Event()

    def large_seq():
        large_started.set()
        large()
        large_done.set()

    t = threading.Thread(target=large_seq)
    t.start()
    large_started.wait(timeout=5)
    _run_parallel([small] * 3)
    t.join(timeout=10)
    assert large_done.is_set()
    assert not rec.large_overlapped


def test_gate_boundary_is_large():
    """恰好 20MB 视为大文件（>= 阈值即独占）。"""
    from agent.rag.concurrency_gate import SizeAwareConversionGate

    gate = SizeAwareConversionGate(LARGE, small_capacity=2)
    rec = _Concurrency()

    def work():
        gate.acquire(LARGE)  # 恰好 20MB
        rec.enter(True)
        time.sleep(0.05)
        rec.exit()
        gate.release(LARGE)

    _run_parallel([work] * 3)
    assert rec.max == 1
    assert not rec.large_overlapped


def test_gate_large_file_not_starved_by_small():
    """大文件排队时阻止新小文件入场，避免被源源不断的小文件饿死。"""
    from agent.rag.concurrency_gate import SizeAwareConversionGate

    gate = SizeAwareConversionGate(LARGE, small_capacity=2)
    order = []
    order_lock = threading.Lock()

    def _mark(name):
        with order_lock:
            order.append(name)

    small1_in = threading.Event()
    small2_in = threading.Event()

    def small(idx, delay=0.0):
        if delay:
            time.sleep(delay)
        gate.acquire(1 * MB)
        _mark(f"small{idx}-in")
        time.sleep(0.15)
        _mark(f"small{idx}-out")
        gate.release(1 * MB)

    def large():
        time.sleep(0.03)  # 等两个小文件先入场
        _mark("large-wait")
        gate.acquire(LARGE)
        _mark("large-in")
        gate.release(LARGE)
        _mark("large-out")

    def small_late():  # 大文件排队后才到的小文件：必须被挡住
        time.sleep(0.06)
        small(3)

    t_large = threading.Thread(target=large)
    t_s1 = threading.Thread(target=small, args=(1,))
    t_s2 = threading.Thread(target=small, args=(2,))
    t_s3 = threading.Thread(target=small_late)
    for t in (t_large, t_s1, t_s2, t_s3):
        t.start()
    for t in (t_large, t_s1, t_s2, t_s3):
        t.join(timeout=10)
        assert not t.is_alive()

    # 大文件必须插到迟到的小文件之前，否则小文件源源不断会把它饿死
    assert order.index("large-in") < order.index("small3-in")
    assert order.index("large-in") > order.index("small1-in")
    assert order.index("large-in") > order.index("small2-in")


# ---------------------------------------------------------------------------
# docling 转换路径
# ---------------------------------------------------------------------------

def _docling_fake(rec: _Concurrency, is_large: bool, sleep: float = 0.05):
    class _Doc:
        def export_to_markdown(self):
            return "# md"

    class _Result:
        document = _Doc()

    class _Conv:
        def convert(self, path):
            rec.enter(is_large)
            time.sleep(sleep)
            rec.exit()
            return _Result()

    return _Conv()


def _sparse_pdf(path, size: int):
    with open(path, "wb") as f:
        f.truncate(size)  # 稀疏文件，秒建不占实际磁盘块


def test_docling_small_files_run_two_concurrently(monkeypatch, tmp_path):
    from agent.rag import docling_parser as dp

    rec = _Concurrency()
    monkeypatch.setattr(
        dp, "_get_converter",
        lambda scale=dp._OCR_SCALE_DEFAULT: _docling_fake(rec, False))

    files = []
    for i in range(4):
        p = tmp_path / f"s{i}.pdf"
        _sparse_pdf(p, 3 * MB)
        files.append(str(p))

    _run_parallel([lambda f=f: dp._convert_with_scale(f, 2.0) for f in files])
    assert rec.max == 2


def test_docling_large_files_run_exclusively(monkeypatch, tmp_path):
    from agent.rag import docling_parser as dp

    rec = _Concurrency()
    monkeypatch.setattr(
        dp, "_get_converter",
        lambda scale=dp._OCR_SCALE_DEFAULT: _docling_fake(rec, True))

    files = []
    for i in range(3):
        p = tmp_path / f"big{i}.pdf"
        _sparse_pdf(p, 25 * MB)
        files.append(str(p))

    _run_parallel([lambda f=f: dp._convert_with_scale(f, 2.0) for f in files])
    assert rec.max == 1
    assert not rec.large_overlapped


def test_docling_large_file_excludes_concurrent_smalls(monkeypatch, tmp_path):
    """大文件在场时，同时到达的小文件不得与它并发。"""
    from agent.rag import docling_parser as dp

    rec = _Concurrency()
    monkeypatch.setattr(
        dp, "_get_converter",
        lambda scale=dp._OCR_SCALE_DEFAULT: _docling_fake(rec, False))

    big = tmp_path / "big.pdf"
    _sparse_pdf(big, 25 * MB)
    smalls = []
    for i in range(3):
        p = tmp_path / f"s{i}.pdf"
        _sparse_pdf(p, 1 * MB)
        smalls.append(str(p))

    # 假转换器对大小一视同仁，但闸门按真实文件大小分流——大文件必须独占。
    _run_parallel(
        [lambda: dp._convert_with_scale(str(big), 2.0)]
        + [lambda f=f: dp._convert_with_scale(f, 2.0) for f in smalls]
    )
    assert rec.max == 2          # 大文件跑完后小文件仍可 2 路并发
    assert not rec.large_overlapped


# ---------------------------------------------------------------------------
# marker 转换路径
# ---------------------------------------------------------------------------

def _marker_fake(rec: _Concurrency, sleep: float = 0.05):
    class _Rendered:
        markdown = "# md"

    class _Conv:
        def __call__(self, path):
            rec.enter(False)
            time.sleep(sleep)
            rec.exit()
            return _Rendered()

    return _Conv()


def test_marker_small_files_run_two_concurrently(monkeypatch, tmp_path):
    from agent.rag import marker_parser as mp

    rec = _Concurrency()
    monkeypatch.setattr(mp, "is_marker_available", lambda: True)
    monkeypatch.setattr(mp, "_get_converter", lambda: _marker_fake(rec))

    files = []
    for i in range(4):
        p = tmp_path / f"s{i}.pdf"
        _sparse_pdf(p, 3 * MB)
        files.append(str(p))

    _run_parallel([lambda f=f: mp.extract_markdown(f) for f in files])
    assert rec.max == 2


def test_marker_large_files_run_exclusively(monkeypatch, tmp_path):
    from agent.rag import marker_parser as mp

    rec = _Concurrency()
    monkeypatch.setattr(mp, "is_marker_available", lambda: True)
    monkeypatch.setattr(mp, "_get_converter", lambda: _marker_fake(rec))

    files = []
    for i in range(3):
        p = tmp_path / f"big{i}.pdf"
        _sparse_pdf(p, 25 * MB)
        files.append(str(p))

    _run_parallel([lambda f=f: mp.extract_markdown(f) for f in files])
    assert rec.max == 1


# ---------------------------------------------------------------------------
# 实例池：复用 / 容量上限 / 释放
# ---------------------------------------------------------------------------

def test_docling_pool_reuses_instances(monkeypatch):
    from agent.rag import docling_parser as dp

    built = []

    def _fake_build(scale):
        built.append(scale)
        return f"conv-{len(built)}"

    monkeypatch.setattr(dp, "_build_converter", _fake_build)
    monkeypatch.setattr(dp, "_converter_unavailable", False)

    c1 = dp._get_converter(2.0)
    dp._release_pooled_converter(2.0, c1)
    c2 = dp._get_converter(2.0)
    assert c2 == c1            # 复用同一实例，不重复构建
    assert len(built) == 1


def test_docling_pool_caps_at_two_and_release_clears(monkeypatch):
    from agent.rag import docling_parser as dp

    built = []

    def _fake_build(scale):
        built.append(scale)
        return f"conv-{len(built)}"

    monkeypatch.setattr(dp, "_build_converter", _fake_build)
    monkeypatch.setattr(dp, "_converter_unavailable", False)

    c1 = dp._get_converter(2.0)
    c2 = dp._get_converter(2.0)
    c3 = dp._get_converter(2.0)
    assert len(built) == 3     # 三次都在用，池空，各建一个

    dp._release_pooled_converter(2.0, c1)
    dp._release_pooled_converter(2.0, c2)
    dp._release_pooled_converter(2.0, c3)
    assert len(dp._converters[2.0]) == 2   # 空闲池封顶 2 个

    dp.release_converter()
    assert dp._converters == {}
    c4 = dp._get_converter(2.0)
    assert c4 == "conv-4"     # 释放后惰性重建


def test_marker_pool_reuses_and_release_clears(monkeypatch):
    from agent.rag import marker_parser as mp

    built = []

    def _fake_build():
        built.append(1)
        return f"conv-{len(built)}"

    monkeypatch.setattr(mp, "_build_converter", _fake_build)
    monkeypatch.setattr(mp, "_converter_unavailable", False)

    c1 = mp._get_converter()
    mp._release_pooled_converter(c1)
    c2 = mp._get_converter()
    assert c2 == c1
    assert len(built) == 1

    mp.release_converter()    # marker 未安装，shutdown_models 导入失败被吞
    assert mp._converters == []
    c3 = mp._get_converter()
    assert c3 == "conv-2"


def test_parsers_get_converter_returns_none_when_unavailable(monkeypatch):
    """_converter_unavailable 置位后，_get_converter 不再尝试构建。"""
    from agent.rag import docling_parser as dp
    from agent.rag import marker_parser as mp

    def _boom(*a, **k):
        raise AssertionError("unavailable 后不应再构建")

    monkeypatch.setattr(dp, "_build_converter", _boom)
    monkeypatch.setattr(dp, "_converter_unavailable", True)
    assert dp._get_converter(2.0) is None

    monkeypatch.setattr(mp, "_build_converter", _boom)
    monkeypatch.setattr(mp, "_converter_unavailable", True)
    assert mp._get_converter() is None


# ---------------------------------------------------------------------------
# 按实例退役：池化不得掏空「每 N 篇重建一代」的缓存护栏
# ---------------------------------------------------------------------------

def test_docling_instance_retires_after_recycle_docs(monkeypatch):
    """单个实例服满 _DOCLING_RECYCLE_DOCS 篇即退役，池里不留下「老代」实例。

    这是 2026-10 修复的回归：只按全局 50 篇清空闲池时，转换中的实例躲过清理、
    回池后继续服役，OCR/ORT 原生缓存跨转换持续累积 → 内存越涨越高直到 swap。
    """
    from agent.rag import docling_parser as dp

    built = []

    def _fake_build(scale):
        built.append(scale)
        return f"conv-{len(built)}"

    monkeypatch.setattr(dp, "_build_converter", _fake_build)
    monkeypatch.setattr(dp, "_converter_unavailable", False)

    c1 = dp._get_converter(2.0)
    for _ in range(dp._DOCLING_RECYCLE_DOCS - 1):
        # 未满服役期前：归还后必被复用（uses 累加、实例不变）
        dp._release_pooled_converter(2.0, c1)
        assert dp._get_converter(2.0) == c1
    dp._release_pooled_converter(2.0, c1)     # 第 N 次归还 → 退役
    assert dp._converters.get(2.0, []) == []  # 不还池
    assert dp._checked_out == []

    c2 = dp._get_converter(2.0)               # 惰性重建新一代
    assert c2 == "conv-2"
    assert len(built) == 2


def test_docling_two_instances_retire_independently(monkeypatch):
    """两个并发实例各自计数退役，互不影响（2 路并发下的护栏）。"""
    from agent.rag import docling_parser as dp

    built = []

    def _fake_build(scale):
        built.append(scale)
        return f"conv-{len(built)}"

    monkeypatch.setattr(dp, "_build_converter", _fake_build)
    monkeypatch.setattr(dp, "_converter_unavailable", False)

    c1 = dp._get_converter(2.0)
    c2 = dp._get_converter(2.0)
    assert c1 != c2
    # c1 服满退役，c2 只服役 1 次
    for _ in range(dp._DOCLING_RECYCLE_DOCS):
        dp._release_pooled_converter(2.0, c1)
        c1 = dp._get_converter(2.0)
    dp._release_pooled_converter(2.0, c2)
    pool = dp._converters.get(2.0, [])
    assert [e[0] for e in pool] == [c2]      # 只有 c2 回池
    assert [e[1] for e in pool] == [1]       # uses=1


def test_docling_release_unknown_instance_falls_back(monkeypatch):
    """未经池借出的实例（测试 stub _get_converter 的场景）按新实例计，不炸。"""
    from agent.rag import docling_parser as dp

    monkeypatch.setattr(dp, "_converter_unavailable", False)
    dp._release_pooled_converter(2.0, "stub-converter")
    pool = dp._converters.get(2.0, [])
    assert [e[0] for e in pool] == ["stub-converter"]
    assert [e[1] for e in pool] == [1]


# ---------------------------------------------------------------------------
# 闸门热更新 / 在飞计数
# ---------------------------------------------------------------------------

def test_gate_configure_updates_in_place():
    """configure 只改字段不换对象——在飞持有者仍挂在同一把闸门上。"""
    from agent.rag.concurrency_gate import SizeAwareConversionGate

    gate = SizeAwareConversionGate(LARGE, small_capacity=2)
    before = id(gate)
    gate.configure(threshold=10 * MB, small_capacity=3)
    assert id(gate) == before
    assert gate.threshold == 10 * MB
    assert gate.small_capacity == 3
    # 15MB 在新阈值（10MB）下是大文件 → 独占，尽管并发上限被调到了 3
    rec = _Concurrency()

    def work():
        gate.acquire(15 * MB)
        rec.enter(True)
        time.sleep(0.05)
        rec.exit()
        gate.release(15 * MB)

    _run_parallel([work] * 3)
    assert rec.max == 1


def test_gate_active_counts_inflight():
    from agent.rag.concurrency_gate import SizeAwareConversionGate

    gate = SizeAwareConversionGate(LARGE, small_capacity=2)
    assert gate.active() == 0
    gate.acquire(1 * MB)
    assert gate.active() == 1
    gate.acquire(1 * MB)
    assert gate.active() == 2
    gate.release(1 * MB)
    assert gate.active() == 1
    gate.release(1 * MB)
    assert gate.active() == 0


def test_docling_configure_concurrency_updates_gate():
    """parser 的 configure_concurrency 把参数热更新进闸门（不换对象）。"""
    from agent.rag import docling_parser as dp
    from agent.rag import marker_parser as mp

    gate_obj = id(dp._conv_gate)
    dp.configure_concurrency(small_capacity=1, large_threshold_bytes=10 * MB)
    assert id(dp._conv_gate) == gate_obj
    assert dp._conv_gate.small_capacity == 1
    assert dp._conv_gate.threshold == 10 * MB

    mp_gate_obj = id(mp._conv_gate)
    mp.configure_concurrency(small_capacity=4, large_threshold_bytes=30 * MB)
    assert id(mp._conv_gate) == mp_gate_obj
    assert mp._conv_gate.small_capacity == 4
    assert mp._conv_gate.threshold == 30 * MB


def test_engine_init_wires_concurrency_config(tmp_path):
    """RAGEngine.__init__ 把并发参数转发到两个 parser 的闸门。"""
    from agent.rag import docling_parser as dp
    from agent.rag import marker_parser as mp
    from agent.rag.engine import RAGEngine

    kb = tmp_path / "kb"
    kb.mkdir()
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    RAGEngine(
        workspace=str(ws), knowledge_base=str(kb),
        pdf_small_concurrency=1, pdf_large_threshold_mb=10.0,
    )
    assert dp._conv_gate.small_capacity == 1
    assert dp._conv_gate.threshold == 10 * 1024 * 1024
    assert mp._conv_gate.small_capacity == 1
    assert mp._conv_gate.threshold == 10 * 1024 * 1024

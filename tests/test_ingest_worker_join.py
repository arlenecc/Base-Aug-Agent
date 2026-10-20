"""回归测试：ingest() 必须等 worker 真正退出，close() 必须等 ingest worker 退出。

背景（同步「一会就崩溃」根因）：
ingest() 的 worker 取消检查只在「文件之间」，单文件内部（OCR 页循环、
docling 转换）完全不检查——一个扫描大 PDF 能跑几十秒。旧实现
``for t in threads: t.join(timeout=5.0)`` 超时后只打 warning 就返回，
SyncKnowledgeWorker 的 finally 随即调 engine.close() 释放 OCR/ONNX/LanceDB，
而此时 worker 仍在用这些对象 → 原生 use-after-free → 整进程段错误
（用户看不到任何 Python traceback）。

本测试用一个「慢 worker + 会记录 use-after-close 的假资源」复现该竞态，
并验证新实现消除它。
"""
import threading
import time

import pytest

from agent.rag.engine import RAGEngine


class _TrackingResource:
    """记录 close 之后仍被 use 的次数（模拟原生层 use-after-free）。"""

    def __init__(self):
        self.closed = False
        self.use_after_close = 0

    def use(self):
        if self.closed:
            self.use_after_close += 1

    def close(self):
        self.closed = True


def _drain_with_old_semantics(threads, shared, join_timeout=5.0):
    """旧实现的等法：每线程 join(timeout) 一次，超时即放过。"""
    for t in threads:
        t.join(timeout=join_timeout)
    shared.close()
    time.sleep(0.5)
    return sum(1 for t in threads if t.is_alive())


def _drain_with_new_semantics(threads, shared, deadline_cap=30.0):
    """新实现的等法：循环 join 直到线程退出（受兜底上限约束）。"""
    deadline = time.monotonic() + deadline_cap
    for t in threads:
        while t.is_alive() and time.monotonic() < deadline:
            t.join(timeout=1.0)
    shared.close()
    time.sleep(0.5)
    return sum(1 for t in threads if t.is_alive())


def _make_slow_workers(shared, n=2, pages=60, page_sec=0.1, stop=None):
    """模拟 ingest worker：单文件内部不检查取消，文件之间才检查。"""

    def worker():
        for _f in range(3):
            for _p in range(pages):
                time.sleep(page_sec)
                shared.use()
            if stop is not None and stop.is_set():
                return

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(n)]
    for t in threads:
        t.start()
    return threads


def test_old_join_timeout_leaves_workers_running_during_close():
    """复现根因：join(timeout=5) 超时后 close，worker 仍在用已释放资源。"""
    shared = _TrackingResource()
    stop = threading.Event()
    threads = _make_slow_workers(shared, n=2, pages=60, page_sec=0.1, stop=stop)
    time.sleep(0.3)
    stop.set()

    alive = _drain_with_old_semantics(threads, shared, join_timeout=0.5)

    assert alive > 0, "预期旧语义下 close 时仍有 worker 存活"
    assert shared.use_after_close > 0, "预期旧语义产生 use-after-close 竞态"


def test_new_semantics_waits_for_workers_before_close():
    """修复后：等到 worker 真正退出才 close，零 use-after-close。"""
    shared = _TrackingResource()
    stop = threading.Event()
    threads = _make_slow_workers(shared, n=2, pages=20, page_sec=0.05, stop=stop)
    time.sleep(0.2)
    stop.set()

    alive = _drain_with_new_semantics(threads, shared, deadline_cap=30.0)

    assert alive == 0, "新语义必须等到 worker 全部退出"
    assert shared.use_after_close == 0, "新语义下不得出现 use-after-close"


def test_engine_close_waits_for_registered_ingest_threads(tmp_path):
    """RAGEngine.close() 必须等 _register_ingest_threads 登记的线程退出。"""
    engine = RAGEngine(
        workspace=str(tmp_path / "ws"),
        knowledge_base=str(tmp_path / "kb"),
    )
    released = threading.Event()
    running = threading.Event()

    def slow():
        running.set()
        time.sleep(1.5)
        released.set()

    t = threading.Thread(target=slow, daemon=True)
    t.start()
    engine._register_ingest_threads([t])
    assert running.wait(2.0)

    engine.close()          # 必须阻塞到 slow() 结束
    assert released.is_set(), "close() 返回前 worker 必须已退出"
    assert not t.is_alive()


def test_engine_close_reports_when_worker_never_stops(tmp_path, caplog):
    """兜底超时后 close() 不能永久挂死，且必须留下错误日志。"""
    engine = RAGEngine(
        workspace=str(tmp_path / "ws"),
        knowledge_base=str(tmp_path / "kb"),
    )

    def never_stops():
        while True:
            time.sleep(0.05)

    t = threading.Thread(target=never_stops, daemon=True)
    t.start()
    engine._register_ingest_threads([t])

    # 把兜底上限压到极小，避免测试真的等 10 分钟
    orig_wait = engine._wait_ingest_threads

    def fast_wait(timeout=600.0):
        return orig_wait(timeout=0.3)

    engine._wait_ingest_threads = fast_wait

    with caplog.at_level("ERROR"):
        engine.close()

    assert any("不能安全释放" in r.message or "未在" in r.message
               for r in caplog.records), "超时必须记录错误日志"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

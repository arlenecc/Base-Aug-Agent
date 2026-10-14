"""全量复查（第十轮）的回归测试。

覆盖本轮发现的 bug：
  1. engine.ingest：文件在扫描后不可读时，``stats["errors"]`` 必须保持为
     **列表**（旧实现写成 ``stats.get("errors", 0) + 1`` 把列表覆盖成 int
     且不持锁 → 同步收尾 ``len(stats["errors"])`` 抛 TypeError，整轮已完成
     的同步以异常告终）。
  2. ``_work_memory``：OrderedDict 的 ``move_to_end`` 与锁内 ``popitem``
     同为变更操作，必须同一把锁（旧实现锁外 move_to_end，与淘汰路径并发
     修改 → RuntimeError / 条目错乱）。
  3. code_run 写护栏：``open()`` 传入非 str/PathLike 时给出明确 TypeError，
     而不是被 ``realpath("")``（= cwd = workspace）绕过检查后炸出难懂的
     底层 TypeError。
"""
from __future__ import annotations

import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# 1. RAGEngine.ingest：errors 列表不被覆盖成 int
# ---------------------------------------------------------------------------

class _FakeEF:
    """Deterministic fake embedding function (4-dim vectors)."""

    def __call__(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3, 0.4]

    def embed_documents(self, texts):
        return self(texts)


def test_ingest_errors_stays_list_when_file_vanishes(tmp_path, monkeypatch):
    """文件在扫描后不可读 → 走 errors.append 分支，ingest 正常返回。

    旧实现把 stats["errors"] 覆盖成 int：
      - len(stats["errors"]) → TypeError（ingest 收尾崩溃）
      - UI 的 ``for e in errors[:5]`` 同样炸
    """
    from agent.rag import engine as eng

    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "a.txt").write_text("hello world content", encoding="utf-8")

    calls = {"n": 0}

    def _fake_sig(filepath):
        # 第 1 次（Step 2 增量对比）：返回有效签名 → 文件进入待处理；
        # 之后（消费端 _chunk_iter）：返回 None → 触发「文件已不可读」分支。
        calls["n"] += 1
        if calls["n"] == 1:
            return {"mtime": 1.0, "size": 20}
        return None

    monkeypatch.setattr(eng, "_file_signature", _fake_sig)

    e = eng.RAGEngine(
        workspace=str(tmp_path),
        knowledge_base=str(kb),
        embedding_function=_FakeEF(),
    )
    try:
        stats = e.ingest(force=True)
    finally:
        e.close()

    assert "error" not in stats, stats
    assert isinstance(stats["errors"], list), (
        "stats['errors'] 必须保持为列表（旧实现被覆盖成 int）"
    )
    assert any("不可读" in msg for msg in stats["errors"])


# ---------------------------------------------------------------------------
# 2. _work_memory：LRU 缓存并发安全
# ---------------------------------------------------------------------------

def test_work_memory_lru_concurrent_access(tmp_path):
    """多线程并发命中/淘汰缓存不抛异常（move_to_end 与 popitem 同锁）。"""
    from agent.config import AgentConfig
    from agent.tools.memory import (
        _WORK_MEMORY_CACHE_MAX,
        _work_memory,
        _work_memory_by_path,
        _work_memory_lock,
    )

    errors: list = []

    def _worker(i: int) -> None:
        try:
            cfg = AgentConfig(workspace=str(tmp_path / f"ws{i % 16}"))
            wm = _work_memory(cfg)
            wm.set(f"k{i}", f"v{i}")
        except Exception as e:  # pragma: no cover - 竞态才触发
            errors.append(e)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(48)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    with _work_memory_lock:
        assert len(_work_memory_by_path) <= _WORK_MEMORY_CACHE_MAX


# ---------------------------------------------------------------------------
# 3. code_run：护栏对非路径参数给出明确错误
# ---------------------------------------------------------------------------

class _StubCallbacks:
    def is_cancelled(self):
        return False


class _StubRegistry:
    callbacks = _StubCallbacks()
    session = None


def test_code_run_guard_rejects_non_path_open(tmp_path):
    """open(1.5, 'w') → 明确的 TypeError 提示，而不是被 realpath("") 绕过后炸底层。"""
    from agent.config import AgentConfig
    from agent.tools.code_run import CodeRunTool

    tool = CodeRunTool()
    tool.bind(AgentConfig(workspace=str(tmp_path)), _StubRegistry())
    res = tool.run(code="f = open(1.5, 'w')", timeout=10)
    assert not res.success
    assert "expects a path" in (res.error or "")


# ---------------------------------------------------------------------------
# 4. code_run：超时后泄漏线程必须自行退出
# ---------------------------------------------------------------------------

def test_code_run_leaked_thread_exits_after_timeout(config):
    """超时后泄漏线程必须在宽限窗口内撞上已关闭缓冲自行退出。

    旧实现先恢复全局 stdout 再 close 缓冲——泄漏线程的 print() 动态查找
    sys.stdout，拿到的是已恢复的**真实**流，永远碰不到已关闭的缓冲，
    线程永不退出；解释器退出时它持有 stdout 缓冲锁 →
    ``Fatal Python error: _enter_buffered_busy``（pytest 进程 abort，exit=134）。
    """
    import time as _t

    from agent.agent import AgentCallbacks
    from agent.tools import ToolRegistry, code_run as code_run_mod

    reg = ToolRegistry(config=config, callbacks=AgentCallbacks())
    try:
        res = reg.execute("code_run", {
            "code": "while True:\n    print('noise')\n",
            "timeout": 1,
        })
        assert not res.success and "timed out" in res.error.lower()

        # 宽限窗口（50ms）+ 余量：线程必须死亡，否则解释器退出时 abort。
        deadline = _t.monotonic() + 3.0
        alive = True
        while _t.monotonic() < deadline:
            with code_run_mod._leaked_threads_lock:
                alive = any(lt.is_alive() for lt in code_run_mod._leaked_threads)
            if not alive:
                break
            _t.sleep(0.05)
        assert not alive, "泄漏线程未退出：全局 stdout 被过早恢复，永远碰不到已关闭缓冲"

        # 宽限窗口之后全局流必须已恢复（不再是已关闭的 code_run 缓冲）。
        _t.sleep(0.2)
        from agent.tools.code_run import _BoundedIO
        assert not (isinstance(sys.stdout, _BoundedIO) and sys.stdout.closed), \
            "全局 stdout 仍指向已关闭的 code_run 缓冲（延迟恢复失败）"
    finally:
        reg.shutdown()

"""Regression: AgentWorker 必须**常驻同一 OS 线程**（#160）。

若每条消息让 ``run()`` 返回、下次再 ``start()``，Qt 会在**新的 OS 线程**上
执行 ``run()``；而 MLX 的 KV 前缀缓存是线程局部的，跨线程复用会抛
``There is no Stream(gpu, N) in current thread``（实测确认），后端因此每轮
作废缓存 → 每轮全量 prefill ~3500 token（~25s/轮，表现为「每句话都卡」）。

本测试确保：两次 ``set_input`` 都在同一 OS 线程执行，且一轮结束后线程仍常驻。
"""
from __future__ import annotations

import threading

from PyQt6.QtCore import QObject, pyqtSignal

from agent.ui.main_window import AgentWorker


class _Bridge(QObject):
    error = pyqtSignal(str)
    finished = pyqtSignal()


class _Callbacks:
    def __init__(self) -> None:
        self.bridge = _Bridge()


class _RecordingAgent:
    def __init__(self) -> None:
        self.thread_ids: list[int] = []

    def run(self, text: str) -> str:
        self.thread_ids.append(threading.get_ident())
        return text


def test_agent_worker_is_persistent_single_os_thread(qtbot):
    agent = _RecordingAgent()
    cb = _Callbacks()
    worker = AgentWorker(agent, cb)
    worker.start()
    try:
        with qtbot.waitSignal(cb.bridge.finished, timeout=3000):
            worker.set_input("first")
        # 一轮结束后线程不退出（常驻），否则下一条会跑在新线程上。
        assert worker.isRunning()

        with qtbot.waitSignal(cb.bridge.finished, timeout=3000):
            worker.set_input("second")
        assert len(agent.thread_ids) == 2
        # 两次都跑在同一 OS 线程 —— KV 前缀缓存跨轮复用的前提。
        assert agent.thread_ids[0] == agent.thread_ids[1]
        assert worker.isRunning()
    finally:
        worker.cancel()
        worker.wait(3000)


def test_agent_worker_warmup_runs_without_finished(qtbot):
    """预热任务应在常驻线程上执行，且**不**触发 finished（否则 UI 会误判
    一轮对话结束、错乱 busy 状态）。"""
    class _WarmAgent:
        def __init__(self):
            self.warmed = threading.Event()

        def run(self, text):
            pass

        def warmup(self):
            self.warmed.set()

    agent = _WarmAgent()
    cb = _Callbacks()
    worker = AgentWorker(agent, cb)
    finished = []
    warm = []
    cb.bridge.finished.connect(lambda: finished.append(1))
    worker.warmup_done.connect(lambda: warm.append(1))
    worker.start()
    try:
        worker.request_warmup()
        assert agent.warmed.wait(3.0), "预热未被执行"
        qtbot.wait(200)  # 留出时间让可能误发的 finished / warmup_done 被处理
        assert warm == [1]        # 预热完成信号已发
        assert finished == []     # 但绝不触发 finished
        assert worker.isRunning()  # 预热不结束常驻线程
    finally:
        worker.cancel()
        worker.wait(3000)

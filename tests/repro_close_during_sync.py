"""复现「同步进行中关窗 / 反复同步」导致的崩溃。

场景 A：同步跑着，用户直接关窗（closeEvent → _shutdown_worker → terminate）
场景 B：连续多次同步，观察 RSS 增长与 handler 泄漏
"""
import gc
import json
import logging
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from PyQt6.QtWidgets import QApplication

app = QApplication(sys.argv)

tmp = tempfile.mkdtemp()
ws = os.path.join(tmp, "ws")
kb = os.path.join(tmp, "kb")
os.makedirs(ws)
os.makedirs(kb)
for i in range(30):
    with open(os.path.join(kb, f"doc{i}.md"), "w", encoding="utf-8") as f:
        f.write(f"# 文档 {i}\n\n" + ("知识库同步测试内容。" * 40))

import agent.ui.main_window as mw

mw.CONFIG_PATH = os.path.join(tmp, "cfg.json")
with open(mw.CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump({"workspace": ws, "knowledge_base": kb,
               "rag_rerank_enabled": False}, f)

try:
    import psutil
    _p = psutil.Process()
    HAVE = True
except ImportError:
    HAVE = False


def rss():
    return _p.memory_info().rss / 1024 ** 2 if HAVE else -1


def make_worker():
    return mw.SyncKnowledgeWorker(
        ws, kb,
        embedding_model=None,
        model_options={
            "embedding_backend": "fastembed",
            "embedding_threads": 4,
            "embedding_batch_size": 8,
            "embedding_max_length": 8192,
            "chunk_size": 500,
            "chunk_overlap": 50,
            "rerank_enabled": False,
        },
    )


print("=== 场景 B：连续 4 轮同步，观察 RSS 与 handler ===")
rag_logger = logging.getLogger("agent.rag")
for rnd in range(4):
    w = make_worker()
    got = {}
    w.sync_finished.connect(lambda s: got.update(done=1))
    w.sync_failed.connect(lambda e: got.update(failed=str(e)[:80]))
    w.start()
    d = time.time() + 120
    while w.isRunning() and time.time() < d:
        app.processEvents()
        time.sleep(0.02)
    # 关键：等 queued signal 真正被处理
    for _ in range(30):
        app.processEvents()
        time.sleep(0.01)
    gc.collect()
    print("轮%d done=%s failed=%s RSS=%.0fMB handlers=%s" % (
        rnd + 1, got.get("done"), got.get("failed"), rss(),
        [type(h).__name__ for h in rag_logger.handlers]))
    w.deleteLater()

print("\n=== 场景 A：同步中途关窗 ===")
w2 = mw.MainWindow()
w2.show()
w2._sync_worker = make_worker()
w2._sync_busy = True
w2._sync_worker.sync_finished.connect(w2._on_sync_finished)
w2._sync_worker.sync_failed.connect(w2._on_sync_failed)
w2._sync_worker.sync_progress.connect(w2._on_sync_progress)
w2._sync_worker.finished.connect(
    lambda wk=w2._sync_worker: w2._on_sync_thread_finished(wk))
w2._sync_worker.finished.connect(w2._sync_worker.deleteLater)
w2._log_poll_timer.start()
w2._sync_worker.start()
print("同步已启动, RSS=%.0fMB" % rss())
time.sleep(1.0)
for _ in range(20):
    app.processEvents()
    time.sleep(0.02)
print("1 秒后 isRunning:", w2._is_worker_running(w2._sync_worker))

# 模拟关窗
from PyQt6.QtGui import QCloseEvent
ev = QCloseEvent()
w2.closeEvent(ev)
print("closeEvent 后 isRunning:", w2._is_worker_running(w2._sync_worker))
print("closeEvent 后 RSS=%.0fMB" % rss())
gc.collect()
print("gc 后 RSS=%.0fMB" % rss())

# 退出前再处理一会事件
for _ in range(50):
    app.processEvents()
    time.sleep(0.02)

print("\n=== atexit 前状态 ===")
print("模块级 _LOG_HANDLER:", mw._LOG_HANDLER)
print("root handlers:", [type(h).__name__ for h in logging.getLogger().handlers])
print("完成")

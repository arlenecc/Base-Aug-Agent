"""Regression: ``list_sources`` 必须做列投影下推，不得整表读取（#160）。

背景：``list_sources()`` 原用 ``table.to_arrow(columns=["source"])`` 投影，
但本地 LanceDB 版本不接受 ``columns`` 参数 → TypeError → 兜底
``table.to_arrow().to_pylist()`` 把整表（57 万行 × 768 维向量 + 全文）读进
Python，单次 200+ 秒，且 ``status()`` 每轮都调用它 → 「每句消息都卡几分钟」。
本测试确保走查询构造器的 ``select([...])`` 投影，且绝不触发整表 ``to_arrow()``。
"""
from __future__ import annotations

import pyarrow as pa

from agent.rag.vector_store import VectorStore


class _FakeQuery:
    def __init__(self, table: "_FakeTable") -> None:
        self._table = table

    def select(self, cols):
        self._table.selected = list(cols)
        return self

    def limit(self, n):
        return self

    def to_arrow(self):
        # 只返回被选中的列（source），模拟投影下推。
        return self._table._projected


class _FakeTable:
    def __init__(self, sources) -> None:
        self._projected = pa.table({"source": sources})
        self.selected = None
        self.full_scan_called = False

    def count_rows(self) -> int:
        return self._projected.num_rows

    def search(self) -> _FakeQuery:
        return _FakeQuery(self)

    def to_arrow(self):
        # 不带投影的全表读——一旦被调用，说明又退化成 200 秒的慢路径。
        self.full_scan_called = True
        raise AssertionError("list_sources 不得整表 to_arrow() 读取")


def test_list_sources_projects_source_column_only(tmp_path, monkeypatch):
    store = VectorStore(persist_dir=str(tmp_path))
    fake = _FakeTable(["b", "a", "a", ""])
    monkeypatch.setattr(store, "_ensure_initialized", lambda: None)
    monkeypatch.setattr(store, "_ensure_table", lambda: fake)

    assert store.list_sources() == ["a", "b"]
    # 投影下推确实只请求了 source 列
    assert fake.selected == ["source"]
    # 没有退化成整表读取
    assert fake.full_scan_called is False

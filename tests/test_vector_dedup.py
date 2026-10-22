"""向量去重测试（UI「向量去重」按钮 → engine.dedupe_vectors → VectorStore）。

两层：
  1. store 层：scan_duplicate_texts 按 text 精确分组、保留 ingested_at 最早
     的一份（缺失视为最老）；delete_by_ids 分批 IN 删除；
  2. engine 层：汇总分组 → 删重复 → 整来源被清空的连带清 digest/缓存 +
     manifest 标记副本；部分重叠的来源只删重复片、保留独有片。
"""
from __future__ import annotations

import os

import pyarrow as pa
import pytest

MB = 1024 * 1024


# ---------------------------------------------------------------------------
# store 层：假 LanceDB 表
# ---------------------------------------------------------------------------

class _FakeLanceTable:
    """最小 LanceDB 表假件：to_arrow(columns=...) 投影 + delete(where)。"""

    def __init__(self, rows, support_ts: bool = True):
        # rows: [{id, text, source, ingested_at}]
        self._rows = rows
        self._support_ts = support_ts
        self.delete_predicates: list = []

    def to_arrow(self, columns=None):
        cols = columns or ["id", "text", "source"]
        if not self._support_ts and "ingested_at" in cols:
            raise ValueError("column ingested_at not found")
        data = {}
        for c in cols:
            if c == "ingested_at" and not self._support_ts:
                continue
            data[c] = [r.get(c) for r in self._rows]
        return pa.table(data)

    def count_rows(self) -> int:
        return len(self._rows)

    def delete(self, where: str):
        self.delete_predicates.append(where)
        if " IN (" not in where:
            return
        inner = where.split(" IN (", 1)[1].rstrip(")")
        ids = {x.strip().strip("'") for x in inner.split(",")}
        self._rows = [r for r in self._rows if r["id"] not in ids]


def _make_store(tmp_path, rows, **kw):
    from agent.rag.vector_store import VectorStore

    store = VectorStore(persist_dir=str(tmp_path))
    table = _FakeLanceTable(rows, **kw)
    store._ensure_initialized = lambda: None       # type: ignore[method-assign]
    store._ensure_table = lambda: table            # type: ignore[method-assign]
    return store, table


def test_scan_groups_duplicates_and_keeps_earliest(tmp_path):
    """同 text 多来源：保留 ingested_at 最早的一份，其余进 drop。"""
    rows = [
        {"id": "a_0", "text": "同一段内容", "source": "/kb/a.txt", "ingested_at": 200.0},
        {"id": "b_0", "text": "同一段内容", "source": "/kb/b.txt", "ingested_at": 100.0},
        {"id": "c_0", "text": "同一段内容", "source": "/kb/c.txt", "ingested_at": 300.0},
        {"id": "a_1", "text": "独有内容", "source": "/kb/a.txt", "ingested_at": 200.0},
    ]
    store, _ = _make_store(tmp_path, rows)
    scan = store.scan_duplicate_texts()

    assert scan["total"] == 4
    assert len(scan["groups"]) == 1
    g = scan["groups"][0]
    assert g["keep"]["id"] == "b_0"               # 最早入库
    assert g["keep"]["source"] == "/kb/b.txt"
    assert sorted(d["id"] for d in g["drop"]) == ["a_0", "c_0"]
    assert scan["source_counts"] == {"/kb/a.txt": 2, "/kb/b.txt": 1, "/kb/c.txt": 1}


def test_scan_no_duplicates(tmp_path):
    rows = [
        {"id": "a_0", "text": "内容一", "source": "/kb/a.txt", "ingested_at": 1.0},
        {"id": "b_0", "text": "内容二", "source": "/kb/b.txt", "ingested_at": 2.0},
    ]
    store, _ = _make_store(tmp_path, rows)
    scan = store.scan_duplicate_texts()
    assert scan["groups"] == []
    assert scan["total"] == 2


def test_scan_missing_ingested_at_treated_as_oldest(tmp_path):
    """缺 ingested_at（旧表未迁移）视为 0.0 = 最老，历史数据占先。"""
    rows = [
        {"id": "a_0", "text": "同文", "source": "/kb/a.txt", "ingested_at": None},
        {"id": "b_0", "text": "同文", "source": "/kb/b.txt", "ingested_at": 500.0},
    ]
    store, _ = _make_store(tmp_path, rows)
    scan = store.scan_duplicate_texts()
    g = scan["groups"][0]
    assert g["keep"]["id"] == "a_0"               # None → 0.0，最早
    assert [d["id"] for d in g["drop"]] == ["b_0"]


def test_scan_tie_broken_by_source(tmp_path):
    """ingested_at 并列时取 source 字典序小者（确定性）。"""
    rows = [
        {"id": "z_0", "text": "同文", "source": "/kb/z.txt", "ingested_at": 100.0},
        {"id": "a_0", "text": "同文", "source": "/kb/a.txt", "ingested_at": 100.0},
    ]
    store, _ = _make_store(tmp_path, rows)
    scan = store.scan_duplicate_texts()
    assert scan["groups"][0]["keep"]["id"] == "a_0"


def test_scan_falls_back_without_ts_column(tmp_path):
    """表没有 ingested_at 列时退化分组（不按时间，按 source/id 字典序）。"""
    rows = [
        {"id": "b_0", "text": "同文", "source": "/kb/b.txt"},
        {"id": "a_0", "text": "同文", "source": "/kb/a.txt"},
    ]
    store, _ = _make_store(tmp_path, rows, support_ts=False)
    scan = store.scan_duplicate_texts()
    g = scan["groups"][0]
    assert g["keep"]["id"] == "a_0"
    assert [d["id"] for d in g["drop"]] == ["b_0"]


def test_delete_by_ids_batches(tmp_path):
    rows = [{"id": f"a_{i}", "text": "t", "source": "/kb/a.txt",
             "ingested_at": 1.0} for i in range(5)]
    store, table = _make_store(tmp_path, rows)
    deleted = store.delete_by_ids(["a_0", "a_1", "a_2", "a_3", "a_4"], batch=2)
    assert deleted == 5
    assert len(table.delete_predicates) == 3       # 5 个 id / batch=2 → 3 批
    assert table.count_rows() == 0


def test_delete_by_ids_escapes_quotes(tmp_path):
    rows = [{"id": "a'b_0", "text": "t", "source": "/kb/a.txt", "ingested_at": 1.0}]
    store, table = _make_store(tmp_path, rows)
    store.delete_by_ids(["a'b_0"])
    assert "a''b_0" in table.delete_predicates[0]


# ---------------------------------------------------------------------------
# engine 层：假 store
# ---------------------------------------------------------------------------

class _FakeDedupStore:
    """实现 dedupe_vectors 用到的最小 store 契约。"""

    def __init__(self, rows):
        self.rows = list(rows)
        self.deleted_sources: list = []
        self.deleted_documents: list = []
        self.delete_ids_calls: list = []

    def count(self):
        return len(self.rows)

    def delete_by_source(self, source):
        self.deleted_sources.append(source)
        self.rows = [r for r in self.rows if r["source"] != source]

    def delete_document(self, doc_name):
        self.deleted_documents.append(doc_name)

    def delete_by_ids(self, ids, batch=400):
        self.delete_ids_calls.append(list(ids))
        id_set = set(ids)
        self.rows = [r for r in self.rows if r["id"] not in id_set]
        return len(ids)

    def scan_duplicate_texts(self):
        by_text = {}
        counts = {}
        for r in self.rows:
            counts[r["source"]] = counts.get(r["source"], 0) + 1
            by_text.setdefault(r["text"], []).append(r)
        groups = []
        for text, items in by_text.items():
            if len(items) < 2:
                continue
            items = sorted(items, key=lambda x: (x.get("ingested_at") or 0.0,
                                                 x["source"], x["id"]))
            groups.append({"text": text, "keep": items[0], "drop": items[1:]})
        groups.sort(key=lambda g: -len(g["drop"]))
        return {"total": len(self.rows), "groups": groups, "source_counts": counts}


def _make_engine(tmp_path, store):
    from agent.rag.engine import RAGEngine

    kb = tmp_path / "kb"
    kb.mkdir(exist_ok=True)
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    engine = RAGEngine(workspace=str(ws), knowledge_base=str(kb))
    engine._store = store
    return engine, kb


def _row(rid, text, source, ts):
    return {"id": rid, "text": text, "source": source, "ingested_at": ts}


def test_dedupe_vectors_removes_duplicate_and_marks_manifest(tmp_path):
    """整来源都是重复：删向量 + 清 digest/缓存 + manifest 标记副本。"""
    from agent.rag.engine import _load_manifest, _save_manifest

    store = _FakeDedupStore([
        _row("a_0", "同一本书的第一章", "/kb/a.txt", 100.0),
        _row("b_0", "同一本书的第一章", "/kb/b.txt", 200.0),   # 后入库的副本
        _row("b_1", "同一本书的第一章", "/kb/b.txt", 200.0),
    ])
    engine, kb = _make_engine(tmp_path, store)
    _save_manifest(engine._rag_dir, {
        "/kb/a.txt": {"mtime": 1, "size": 1, "content_hash": "h1", "chunk_count": 1},
        "/kb/b.txt": {"mtime": 2, "size": 2, "content_hash": "h1", "chunk_count": 2},
    })
    # b 的历史 markdown 缓存（应被清掉，否则 Step 7 会重建 digest）
    from agent.rag.engine import _safe_filename
    cache_b = os.path.join(engine.markdown_dir,
                           _safe_filename(os.path.relpath("/kb/b.txt", str(kb))) + ".md")
    os.makedirs(engine.markdown_dir, exist_ok=True)
    with open(cache_b, "w", encoding="utf-8") as f:
        f.write("cached")

    logs: list = []
    stats = engine.dedupe_vectors(log_callback=logs.append)

    assert stats["scanned"] == 3
    assert stats["groups"] == 1
    assert stats["removed"] == 2                   # b 的两条都是重复
    assert stats["sources_cleaned"] == 1
    assert [r["source"] for r in store.rows] == ["/kb/a.txt"]
    # 整来源清理：digest 三名 + 缓存文件
    assert "/kb/b.txt" in store.deleted_sources
    assert os.path.relpath("/kb/b.txt", str(kb)) in store.deleted_documents
    assert not os.path.exists(cache_b)
    # manifest：b 标记为 a 的副本
    manifest = _load_manifest(engine._rag_dir)
    assert manifest["/kb/b.txt"]["duplicate_of"] == "/kb/a.txt"
    assert "content_hash" not in manifest["/kb/b.txt"]
    assert manifest["/kb/a.txt"]["content_hash"] == "h1"
    assert any("清理" in l or "重复" in l for l in logs)


def test_dedupe_vectors_keeps_unique_chunks_of_partial_overlap(tmp_path):
    """部分重叠：只删重复片，来源的独有片保留，不触发整来源清理。"""
    store = _FakeDedupStore([
        _row("a_0", "共有章节", "/kb/a.txt", 100.0),
        _row("a_1", "A 独有章", "/kb/a.txt", 100.0),
        _row("b_0", "共有章节", "/kb/b.txt", 200.0),
        _row("b_1", "B 独有章", "/kb/b.txt", 200.0),
    ])
    engine, kb = _make_engine(tmp_path, store)

    stats = engine.dedupe_vectors(log_callback=lambda m: None)

    assert stats["removed"] == 1                   # 只删 b_0（共有章节的副本）
    assert stats["sources_cleaned"] == 0           # b 还有独有片，不整源清理
    assert sorted(r["id"] for r in store.rows) == ["a_0", "a_1", "b_1"]
    assert store.deleted_sources == []             # 没有按 source 删
    assert store.delete_ids_calls == [["b_0"]]


def test_dedupe_vectors_no_duplicates(tmp_path):
    store = _FakeDedupStore([
        _row("a_0", "内容一", "/kb/a.txt", 100.0),
        _row("b_0", "内容二", "/kb/b.txt", 200.0),
    ])
    engine, _ = _make_engine(tmp_path, store)

    stats = engine.dedupe_vectors(log_callback=lambda m: None)
    assert stats["removed"] == 0
    assert stats["groups"] == 0
    assert store.delete_ids_calls == []


def test_dedupe_vectors_empty_store(tmp_path):
    store = _FakeDedupStore([])
    engine, _ = _make_engine(tmp_path, store)

    stats = engine.dedupe_vectors(log_callback=lambda m: None)
    assert stats["scanned"] == 0
    assert stats["removed"] == 0
    assert store.delete_ids_calls == []


def test_dedupe_vectors_progress_callback(tmp_path):
    """删除阶段有进度回调（done/total 递增）。"""
    rows = []
    for i in range(3):
        rows.append(_row(f"a_{i}", f"共有内容{i}", "/kb/a.txt", 100.0))
        rows.append(_row(f"b_{i}", f"共有内容{i}", "/kb/b.txt", 200.0))
    store = _FakeDedupStore(rows)
    engine, _ = _make_engine(tmp_path, store)

    seen: list = []
    engine.dedupe_vectors(progress_callback=lambda d, t, c: seen.append((d, t)))
    assert seen, "应有进度回调"
    assert seen[-1] == (3, 3)                     # 3 条重复全清完


def test_records_carry_ingested_at(tmp_path):
    """新写入的 record 带 ingested_at（去重按它保留最早的一份）。"""
    from agent.rag.vector_store import _INGESTED_AT, VectorStore

    store = VectorStore(persist_dir=str(tmp_path))
    store._ef = lambda texts: [[0.0, 0.0, 0.0] for _ in texts]
    records = store._embed_chunks(
        [{"source": "/kb/a.txt", "text": "内容", "chunk_index": 0}], 0)
    assert len(records) == 1
    assert _INGESTED_AT in records[0]
    assert records[0][_INGESTED_AT] > 0

"""Unit tests for the incremental-sync manifest logic in RAGEngine.

These tests don't require the embedding model — they verify the manifest
load/save, file-signature comparison, and the partitioning logic that
decides which files to process vs skip. The embedding/vector-store layer
is exercised by test_rag_e2e.py and test_rag_full_pipeline.py.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from agent.rag.engine import (
    _content_hash,
    _file_signature,
    _load_manifest,
    _manifest_path,
    _save_manifest,
)


# ---------------------------------------------------------------------------
# manifest helpers
# ---------------------------------------------------------------------------

def test_manifest_path_is_inside_rag_dir(tmp_path):
    p = _manifest_path(str(tmp_path / "rag"))
    assert p.endswith("manifest.json")
    assert os.path.dirname(p).endswith("rag")


def test_load_manifest_returns_empty_when_missing(tmp_path):
    assert _load_manifest(str(tmp_path / "rag")) == {}


def test_load_manifest_returns_empty_when_corrupted(tmp_path):
    rag_dir = tmp_path / "rag"
    rag_dir.mkdir()
    (rag_dir / "manifest.json").write_text("not valid json{{{")
    assert _load_manifest(str(rag_dir)) == {}


def test_load_manifest_returns_empty_when_wrong_shape(tmp_path):
    """A manifest without the 'files' key must return {} not the raw dict."""
    rag_dir = tmp_path / "rag"
    rag_dir.mkdir()
    (rag_dir / "manifest.json").write_text(json.dumps({"wrong_key": {}}))
    assert _load_manifest(str(rag_dir)) == {}


def test_save_and_load_manifest_roundtrip(tmp_path):
    rag_dir = str(tmp_path / "rag")
    files = {
        "/kb/a.txt": {"mtime": 1000.0, "size": 100, "content_hash": "abc", "chunk_count": 3},
        "/kb/b.md": {"mtime": 2000.0, "size": 200, "content_hash": "def", "chunk_count": 5},
    }
    _save_manifest(rag_dir, files)
    loaded = _load_manifest(rag_dir)
    assert loaded == files


def test_save_manifest_is_atomic_no_tmp_leftover(tmp_path):
    """After a successful save, no .tmp file should remain."""
    rag_dir = str(tmp_path / "rag")
    _save_manifest(rag_dir, {})
    assert not os.path.exists(_manifest_path(rag_dir) + ".tmp")


# ---------------------------------------------------------------------------
# file signature & content hash
# ---------------------------------------------------------------------------

def test_file_signature_captures_mtime_and_size(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("hello world")
    sig = _file_signature(str(f))
    assert "mtime" in sig
    assert "size" in sig
    assert sig["size"] == 11


def test_file_signature_changes_when_content_modified(tmp_path):
    """Modifying a file must produce a different signature (mtime changes)."""
    f = tmp_path / "doc.txt"
    f.write_text("original content here")
    sig1 = _file_signature(str(f))
    # Ensure mtime changes even on fast filesystems with coarse mtime resolution.
    time.sleep(0.05)
    f.write_text("modified content here!!")
    os.utime(str(f), None)  # force mtime update
    sig2 = _file_signature(str(f))
    assert sig1 != sig2


def test_content_hash_is_stable():
    """Same input must produce the same hash."""
    h1 = _content_hash("hello world")
    h2 = _content_hash("hello world")
    assert h1 == h2


def test_content_hash_differs_for_different_text():
    assert _content_hash("foo") != _content_hash("bar")


def test_content_hash_is_hex_and_truncated():
    h = _content_hash("test content")
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# incremental ingest partitioning (using a fake VectorStore to avoid model load)
# ---------------------------------------------------------------------------

class _FakeStore:
    """Minimal VectorStore stub for testing ingest logic without loading the
    embedding model. Records all add_streaming calls and delete_by_source calls."""

    def __init__(self):
        self.records = []
        self.deleted_sources = []
        self.deleted_documents = []
        self._count = 0

    def add_streaming(self, chunk_iter, batch_size=100, on_batch=None, is_cancelled=None):
        added = 0
        for chunk in chunk_iter:
            self.records.append(chunk)
            added += 1
        self._count += added
        if on_batch:
            on_batch(added, added)
        return added

    def delete_by_source(self, source):
        self.deleted_sources.append(source)
        return 1

    def delete_document(self, doc_name):
        self.deleted_documents.append(doc_name)

    def count(self):
        return self._count

    def reset_if_incompatible(self):
        """engine.py 同步入口会调用（cache-DB 自愈路径）；桩恒返回 False。"""
        return False

    def list_sources(self):
        return sorted({r.get("source", "") for r in self.records if r.get("source")})

    def clear(self):
        self.records = []
        self._count = 0


def _make_engine(tmp_path, fake_store=None):
    """Build a RAGEngine with a fake embedding function so no model loads."""
    from agent.rag.engine import RAGEngine

    kb = tmp_path / "kb"
    kb.mkdir()
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    engine = RAGEngine(workspace=str(ws), knowledge_base=str(kb))
    if fake_store is not None:
        engine._store = fake_store
    # Create a .txt file in the KB
    (kb / "a.txt").write_text("first document content with enough text to chunk")
    return engine, kb


def test_first_ingest_processes_all_files(tmp_path):
    """First ingest (no manifest) must process all files — none skipped."""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    stats = engine.ingest(force=False)

    assert stats["files_found"] == 1
    assert stats["files_extracted"] == 1
    assert stats["files_skipped"] == 0
    # Manifest must be saved after ingest
    manifest = _load_manifest(engine._rag_dir)
    assert len(manifest) == 1
    entry = list(manifest.values())[0]
    assert "mtime" in entry and "size" in entry and "content_hash" in entry


def test_ingest_logs_per_file_stage_timings(tmp_path):
    """每文件完整入库后，日志应含「解析/清洗/切片/向量化+入库」四阶段耗时汇总，
    便于定位慢/卡点在哪一步。"""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    (kb / "b.md").write_text("# 标题\n\n" + ("第二篇内容，用于切片与计时。" * 60))

    logs: list = []
    engine.ingest(force=False, log_callback=logs.append)

    timing = [ln for ln in logs if "各阶段耗时" in ln]
    # a.txt + b.md 各一行
    assert len(timing) == 2, timing
    for ln in timing:
        for stage in ("解析", "清洗", "切片", "向量化+入库", "合计"):
            assert stage in ln, f"缺少阶段 {stage}: {ln}"
    # worker 侧的前三阶段小计行也应出现（向量化前）
    assert any("解析/清洗/切片完成" in ln for ln in logs)


def test_second_ingest_skips_unchanged_files(tmp_path):
    """Second ingest with force=False must skip all unchanged files — no
    re-parse, no re-chunk, no re-embed, no upsert."""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)

    # First ingest processes everything
    engine.ingest(force=False)
    assert store.count() > 0  # chunks written
    count_after_first = store.count()

    # Second ingest — all files unchanged
    stats = engine.ingest(force=False)
    assert stats["files_found"] == 1
    assert stats["files_skipped"] == 1
    assert stats["files_extracted"] == 0
    # No new chunks added (no re-embed)
    assert store.count() == count_after_first


def test_force_reingest_processes_all_files(tmp_path):
    """force=True must ignore the manifest and reprocess all files."""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)

    engine.ingest(force=False)  # first, populates manifest
    # Add a second file so there are 2 total
    (kb / "b.txt").write_text("second document content for chunking purposes here")

    stats = engine.ingest(force=True)
    assert stats["files_found"] == 2
    assert stats["files_extracted"] == 2
    assert stats["files_skipped"] == 0


def test_modified_file_is_reprocessed(tmp_path):
    """A file whose content changed (mtime/size differ) must be reprocessed."""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    engine.ingest(force=False)

    # Modify the file
    f = kb / "a.txt"
    time.sleep(0.05)
    f.write_text("completely different content that should trigger reprocessing")
    os.utime(str(f), None)

    stats = engine.ingest(force=False)
    assert stats["files_extracted"] == 1, "modified file must be reprocessed"
    assert stats["files_skipped"] == 0


def test_new_file_is_processed_on_incremental_sync(tmp_path):
    """Adding a new file between syncs must process only the new file."""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    engine.ingest(force=False)

    # Add a new file
    (kb / "b.txt").write_text("new document added after first sync")

    stats = engine.ingest(force=False)
    assert stats["files_found"] == 2
    assert stats["files_extracted"] == 1  # only the new file
    assert stats["files_skipped"] == 1    # the old file is unchanged


def test_deleted_file_is_cleaned_up(tmp_path):
    """A file removed from the KB must have its vectors deleted and manifest
    entry removed on the next incremental sync."""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    engine.ingest(force=False)

    # Record the source path
    deleted_source = str(kb / "a.txt")

    # Remove the file
    os.remove(deleted_source)

    stats = engine.ingest(force=False)
    assert stats["files_deleted"] == 1, "deleted file must be cleaned up"
    assert deleted_source in store.deleted_sources, "vectors must be deleted"
    # 目录概要（documents 元数据表）也必须同步清理（用 basename 作为 doc_id）。
    assert "a.txt" in store.deleted_documents, "document digest must be deleted"

    # Manifest must no longer contain the deleted file
    manifest = _load_manifest(engine._rag_dir)
    assert deleted_source not in manifest


def test_manifest_survives_corruption(tmp_path):
    """If the manifest file is corrupted, ingest must treat it as empty and
    process all files (rather than crashing)."""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    # Write a corrupted manifest before ingest
    rag_dir = engine._rag_dir
    os.makedirs(rag_dir, exist_ok=True)
    (open(os.path.join(rag_dir, "manifest.json"), "w")).write("corrupted{{{invalid json")

    stats = engine.ingest(force=False)
    assert stats["files_extracted"] == 1  # processed because manifest treated as empty
    assert stats["files_skipped"] == 0
    # After successful ingest, manifest must be valid
    manifest = _load_manifest(engine._rag_dir)
    assert len(manifest) == 1


def test_clear_removes_manifest(tmp_path):
    """clear() must delete the manifest so the next ingest starts fresh."""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    engine.ingest(force=False)
    assert os.path.exists(_manifest_path(engine._rag_dir))

    engine.clear()
    assert not os.path.exists(_manifest_path(engine._rag_dir))


def test_empty_kb_cleans_up_deleted_files(tmp_path):
    """If all files are removed from the KB, ingest must clean up all old
    vectors and manifest entries."""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    engine.ingest(force=False)
    assert store.count() > 0

    # Remove all files
    for f in kb.iterdir():
        if f.is_file():
            os.remove(str(f))

    stats = engine.ingest(force=False)
    assert stats["files_found"] == 0
    assert stats["files_deleted"] == 1
    # Manifest must be empty
    manifest = _load_manifest(engine._rag_dir)
    assert len(manifest) == 0


# ---------------------------------------------------------------------------
# Regression: agent's VectorStore must see table created by sync worker
# ---------------------------------------------------------------------------

def test_vector_store_rechecks_table_after_external_creation(tmp_path, vector_store_factory):
    """Regression: VectorStore must re-check table existence on every call.

    Previously _ensure_table() cached "table doesn't exist" via _table_checked
    flag. When SyncKnowledgeWorker (separate RAGEngine) later created the
    table, the agent's VectorStore still returned None forever → empty search
    results despite data being on disk.
    """
    persist = str(tmp_path / "vectors")

    # Agent's VectorStore, created at app startup BEFORE any sync.
    agent_store = vector_store_factory(persist)
    agent_store._ensure_initialized()
    # Table doesn't exist yet → _ensure_table returns None
    assert agent_store._ensure_table() is None

    # SyncKnowledgeWorker creates a SEPARATE VectorStore pointing to the same
    # dir and writes data (creates the table on disk).
    sync_store = vector_store_factory(persist)
    sync_store._ensure_initialized()
    sync_store.add([{
        "source": "doc1.md",
        "chunk_index": 0,
        "text": "hello world",
    }])
    assert sync_store.count() == 1
    sync_store.close()

    # Agent's VectorStore must now see the table (re-check, not cached None).
    table = agent_store._ensure_table()
    assert table is not None, "agent_store should see table created by sync_store"
    assert agent_store.count() == 1
    # Search must return results, not empty.
    results = agent_store.search("hello", top_k=5)
    assert len(results) >= 1
    assert results[0]["text"] == "hello world"
    agent_store.close()


def test_rag_engine_search_sees_data_written_by_another_engine(tmp_path, rag_engine_factory):
    """End-to-end regression: two RAGEngine instances pointing to the same
    workspace. The first (agent's, created at startup) must find data written
    by the second (SyncKnowledgeWorker) without needing a restart.
    """
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "note.txt").write_text("LanceDB is a columnar vector database.")

    # Agent's engine, created at startup (table doesn't exist yet).
    agent_engine = rag_engine_factory(str(tmp_path), str(kb))
    # Trigger lazy init by calling search (returns empty, table not yet created).
    results_before = agent_engine.search("database", top_k=3)
    assert results_before == []

    # SyncKnowledgeWorker's engine writes data.
    sync_engine = rag_engine_factory(str(tmp_path), str(kb))
    sync_engine.ingest(force=False)
    sync_engine.close()

    # Agent's engine must now find the data WITHOUT restart.
    results_after = agent_engine.search("database", top_k=3)
    assert len(results_after) >= 1
    assert "LanceDB" in results_after[0]["text"]
    agent_engine.close()


def test_incremental_sync_visible_to_existing_engine(tmp_path, rag_engine_factory):
    """增量同步后，已存在的 agent 引擎应立即看到新数据，无需重启。

    场景：agent 引擎先同步 doc1 并搜索（打开表），然后 sync 引擎追加 doc2，
    agent 引擎立即搜索 doc2 内容应能命中。验证 LanceDB 的表对象不缓存数据，
    agent 持有的旧表对象能看到新写入的行。
    """
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "doc1.txt").write_text("First document about Python basics.")

    # Agent 的引擎：首次同步 + 触发一次搜索（让 VectorStore 打开表）
    agent_engine = rag_engine_factory(str(tmp_path), str(kb))
    agent_engine.ingest(force=False)
    r1 = agent_engine.search("Python", top_k=3)
    assert len(r1) >= 1, "首次同步后应能搜到 doc1"

    # 添加新文件
    (kb / "doc2.txt").write_text("Second document about JavaScript and web development.")

    # SyncKnowledgeWorker 的引擎：增量同步
    sync_engine = rag_engine_factory(str(tmp_path), str(kb))
    stats = sync_engine.ingest(force=False)
    assert stats["files_extracted"] == 1, "应增量处理 doc2"
    assert stats["files_skipped"] == 1, "应跳过 doc1"
    sync_engine.close()

    # 模拟 UI _on_sync_finished 中的 reload_rag() 调用：
    # 刷新 agent 引擎的表句柄，让下一次搜索看到新写入的数据。
    # LanceDB 表对象在打开时获取版本快照，不刷新会看到旧数据。
    agent_engine.reload()

    # Agent 的引擎立即搜索新内容（不重启、不重建）
    r2 = agent_engine.search("JavaScript", top_k=3)
    assert len(r2) >= 1, "增量同步后应立即搜到 doc2"
    assert "JavaScript" in r2[0]["text"]

    # 旧内容仍可搜到
    r3 = agent_engine.search("Python", top_k=3)
    assert len(r3) >= 1, "旧内容仍应可搜到"
    agent_engine.close()


# ---------------------------------------------------------------------------
# 取消后续传：已完整入库的文件保留记录，下次从剩余文件继续
# ---------------------------------------------------------------------------

class _CancelOnSecondFileStore(_FakeStore):
    """消费到第二个文件的切片时模拟用户点「停止同步」。

    与具体哪个文件先被消费无关（worker 分片/调度会变顺序）：第一个文件必然
    已完整写完，第二个刚开始就停。
    """

    def __init__(self, engine):
        super().__init__()
        self._engine = engine

    def add_streaming(self, chunk_iter, batch_size=100, on_batch=None, is_cancelled=None):
        added = 0
        first_source = None
        for chunk in chunk_iter:
            src = chunk.get("source", "")
            if first_source is None:
                first_source = src
            elif src != first_source:
                self._engine.cancel()      # 模拟「停止同步」按钮
                break
            self.records.append(chunk)
            added += 1
        self._count += added
        if on_batch:
            on_batch(added, added)
        return added


def _clear_manifest(engine) -> None:
    """直接删 manifest 文件（_save_manifest 是读改写合并，传 {} 清不掉）。"""
    mf = Path(_manifest_path(engine._rag_dir))
    if mf.exists():
        mf.unlink()


def _make_two_file_kb(engine):
    """在 KB 里放第二个文件（第一个由 _make_engine 创建）。"""
    kb = Path(engine._knowledge_base)
    (kb / "b.txt").write_text("second document with quite a lot of body text " * 40)
    return kb


def test_cancelled_sync_keeps_completed_files_and_resumes(tmp_path):
    """停止同步：已完整入库的文件写进 manifest，下次同步只处理剩余文件。"""
    engine, kb = _make_engine(tmp_path, _FakeStore())
    _make_two_file_kb(engine)

    # 先跑一次完整同步，确认两个文件都能处理
    full_store = _FakeStore()
    engine._store = full_store
    stats = engine.ingest(force=False)
    assert stats["files_extracted"] == 2
    assert len(_load_manifest(engine._rag_dir)) == 2

    # 重新从「首次同步」开始：写完第一个文件就取消
    _clear_manifest(engine)
    stop_store = _CancelOnSecondFileStore(engine)
    engine._store = stop_store
    stats = engine.ingest(force=False)
    assert stats["cancelled"] is True

    manifest = _load_manifest(engine._rag_dir)
    assert len(manifest) == 1                       # 只有第一个文件被记录
    kept_name = os.path.basename(next(iter(manifest)))
    assert kept_name in ("a.txt", "b.txt")
    # 只入库了那一个文件的切片，且全部来自它
    assert stop_store.records
    assert all(r.get("source", "").endswith(kept_name) for r in stop_store.records)

    # 重新同步：已完成的那个被跳过，只处理剩下的
    resume_store = _FakeStore()
    engine._store = resume_store
    stats2 = engine.ingest(force=False)
    assert stats2["files_skipped"] == 1
    assert stats2["files_extracted"] == 1
    assert all(r.get("source", "").endswith(kept_name) is False
               for r in resume_store.records)
    assert len(_load_manifest(engine._rag_dir)) == 2


def test_completed_file_checkpointed_to_manifest_per_file(tmp_path, monkeypatch):
    """每个文件全流程（解析/切片/向量化/入库）完成后立即写入 manifest，而不必等
    收尾的整批保存；这样进程被直接杀掉（非点「停止同步」）时，已完成的文件也已
    落盘，下次同步会因 mtime/size 未变被跳过，不会从头重做。

    用「逐文件调用 on_batch（累计数）」的假 store 模拟真实写入节奏，并 spy 每次
    _save_manifest 的入参，断言存在「只保存单个文件」的增量落盘调用。
    """

    class _FakeStorePerFile(_FakeStore):
        def add_streaming(self, chunk_iter, batch_size=100, on_batch=None, is_cancelled=None):
            added = 0
            cur_src = None
            buf = []
            for chunk in chunk_iter:
                src = chunk.get("source", "")
                if src != cur_src:
                    if buf:
                        added += len(buf)
                        self.records.extend(buf)
                        buf = []
                        if on_batch:
                            on_batch(len(self.records[-added:]), added)
                    cur_src = src
                buf.append(chunk)
            if buf:
                added += len(buf)
                self.records.extend(buf)
                if on_batch:
                    on_batch(len(buf), added)
            self._count += added
            return added

    import agent.rag.engine as engine_mod

    saved_calls: list = []
    orig_save = engine_mod._save_manifest

    def _spy_save(rag_dir, files, removed=None):
        saved_calls.append(set(files.keys()))
        return orig_save(rag_dir, files, removed)

    monkeypatch.setattr(engine_mod, "_save_manifest", _spy_save)

    store = _FakeStorePerFile()
    engine, kb = _make_engine(tmp_path, store)
    _make_two_file_kb(engine)

    stats = engine.ingest(force=False)
    assert stats["files_extracted"] == 2

    # 至少存在「只保存单个文件」的增量落盘调用（证明逐文件 checkpoint）
    single_file_saves = [s for s in saved_calls if len(s) == 1]
    assert single_file_saves, "应存在逐文件增量保存 manifest 的调用"
    saved_names = {os.path.basename(n) for s in single_file_saves for n in s}
    assert saved_names == {"a.txt", "b.txt"}, saved_names

    # 最终 manifest 仍含两个文件
    assert len(_load_manifest(engine._rag_dir)) == 2


def test_cancelled_mid_file_keeps_nothing(tmp_path):
    """取消发生在第一个文件内部（只写了一半）时，不写任何记录。"""
    engine, kb = _make_engine(tmp_path, _FakeStore())
    _make_two_file_kb(engine)
    _clear_manifest(engine)

    class _StopImmediately(_FakeStore):
        def add_streaming(self, chunk_iter, batch_size=100, on_batch=None, is_cancelled=None):
            self._engine_ref = engine
            engine.cancel()          # 一片都没写就取消
            return 0

    store = _StopImmediately()
    engine._store = store
    stats = engine.ingest(force=False)
    assert stats["cancelled"] is True
    assert len(_load_manifest(engine._rag_dir)) == 0   # 没有完整入库的文件
    assert len(store.records) == 0

    # 下次同步从头处理两个文件
    store2 = _FakeStore()
    engine._store = store2
    stats2 = engine.ingest(force=False)
    assert stats2["files_skipped"] == 0
    assert stats2["files_extracted"] == 2


# ---------------------------------------------------------------------------
# docling/marker 规范 Markdown 路径的 HTML 注释剥离（<!-- image --> 等）
# ---------------------------------------------------------------------------



def _make_engine_only_kb(tmp_path, fake_store=None, extra_files=None):
    """Build a RAGEngine WITHOUT the default a.txt (so file counts are exact)."""
    from agent.rag.engine import RAGEngine

    kb = tmp_path / "kb"
    kb.mkdir()
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    engine = RAGEngine(workspace=str(ws), knowledge_base=str(kb))
    if fake_store is not None:
        engine._store = fake_store
    if extra_files:
        for name, content in extra_files.items():
            (kb / name).write_bytes(content)
    return engine, kb


def test_docling_markdown_strips_image_comments_on_ingest(monkeypatch, tmp_path):
    """docling 输出的规范 Markdown 含 <!-- image --> → 入库前被剥掉。

    旧行为：is_markdown 分支直接 cleaned = text（跳过 clean_text），导致
    docling 默认导出的 `<!-- image -->` 占位注释原样进入缓存与切片。
    """
    store = _FakeStore()
    engine, kb = _make_engine_only_kb(
        tmp_path, store, extra_files={"scan.pdf": b"%PDF-1.4 fake"})
    docling_md = (
        "# 报告标题\n\n第一段正文内容。\n\n<!-- image -->\n\n"
        "第二段，紧跟在图片占位之后。\n"
    )
    monkeypatch.setattr(
        "agent.rag.engine._extract_with_pdf_engine",
        lambda fp, pe="docling": (docling_md, True),
    )
    stats = engine.ingest(force=True)
    assert stats["files_extracted"] == 1

    md_files = [f for f in os.listdir(engine.markdown_dir) if f.endswith(".md")]
    assert md_files
    cached = open(os.path.join(engine.markdown_dir, md_files[0]), encoding="utf-8").read()
    assert "<!--" not in cached                 # 缓存文件已剥注释
    assert "# 报告标题" in cached
    # 切片里也不应出现注释标记
    assert all("<!--" not in r.get("text", "") for r in store.records)


def test_cached_markdown_with_comments_is_self_healed(monkeypatch, tmp_path):
    """历史上缓存的含 <!-- image --> 的 markdown，命中缓存时自愈落盘。

    parse 阶段必须不抛异常（用 monkeypatch 保证），随后 cache_is_fresh 命中
    已存的脏缓存，读取时剥掉注释并回写。
    """
    store = _FakeStore()
    engine, kb = _make_engine_only_kb(
        tmp_path, store, extra_files={"scan.pdf": b"%PDF-1.4 fake"})

    from agent.rag.engine import _safe_filename
    rel = os.path.relpath(str(kb / "scan.pdf"), kb)
    name = _safe_filename(rel) + ".md"
    md_path = os.path.join(engine.markdown_dir, name)
    os.makedirs(engine.markdown_dir, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 标题\n\n<!-- image -->\n\n正文。\n")
    assert "<!--" in open(md_path, encoding="utf-8").read()

    # 解析阶段不抛异常（返回什么都行，cache 命中后会用缓存覆盖 parsed text）
    monkeypatch.setattr(
        "agent.rag.engine._extract_with_pdf_engine",
        lambda fp, pe="docling": ("# parsed-placeholder", True),
    )
    engine.ingest(force=False)     # 文件未改 → 命中缓存 → 自愈
    healed = open(md_path, encoding="utf-8").read()
    assert "<!--" not in healed


def test_clean_markdown_document_strips_all_noise_and_keeps_semantics():
    """clean_markdown_document 统一清理 docling/marker/pandoc 的噪音标签，
    但不破坏 Markdown 语义语法，且保护代码块。"""
    from agent.rag.cleaner import clean_markdown_document

    md = (
        "# 标题\n\n"
        "<!-- image -->\n"                                  # HTML 注释
        "<div class='x'>正文段落</div>\n"                   # raw HTML（留正文）
        "脚注<sup>1</sup>示例\n"                            # <sup>（留数字）
        "![图1](figures/a.png)\n"                          # Markdown 图片（整段删）
        "参见[目录](#toc-index)与[索引](#idx)\n"            # 内部锚点链接（留文字）
        "\n```\nvector<int> x = f<a, b>();\n```\n"          # 代码块内 < > 不动
        "\n| a | b |\n| - | - |\n| 1 | 2 |\n"              # 表格语法保留
    )
    out = clean_markdown_document(md)
    assert "<!--" not in out
    assert "<div" not in out and "</div>" not in out
    assert "<sup>" not in out
    assert "![图1]" not in out and "figures/a.png" not in out
    assert "](#toc-index)" not in out and "](#idx)" not in out
    # 链接文字 / 正文 / 脚注数字保留
    assert "正文段落" in out
    assert "脚注" in out and "1" in out
    assert "目录" in out and "索引" in out
    # 语义语法保留
    assert "# 标题" in out
    assert "| a | b |" in out
    # 代码块内 < > 受保护
    assert "vector<int> x = f<a, b>();" in out


def test_clean_markdown_document_is_idempotent():
    from agent.rag.cleaner import clean_markdown_document

    md = "# 标题\n\n<!-- image -->\n<div>x</div>\n![a](b.png)\n[目录](#t)\n"
    once = clean_markdown_document(md)
    twice = clean_markdown_document(once)
    assert once == twice


def test_docling_markdown_full_cleanup_on_ingest(monkeypatch, tmp_path):
    """docling 输出含注释/raw HTML/图片/内部链接 → 入库前一并清掉。"""
    store = _FakeStore()
    engine, kb = _make_engine_only_kb(
        tmp_path, store, extra_files={"scan.pdf": b"%PDF-1.4 fake"})
    docling_md = (
        "# 报告标题\n\n第一段。\n\n"
        "<!-- image -->\n"
        "<div class='cap'>图注</div>\n"
        "![图](fig.png)\n"
        "见[附录](#app)。\n"
    )
    monkeypatch.setattr(
        "agent.rag.engine._extract_with_pdf_engine",
        lambda fp, pe="docling": (docling_md, True),
    )
    engine.ingest(force=True)
    md_files = [f for f in os.listdir(engine.markdown_dir) if f.endswith(".md")]
    cached = open(os.path.join(engine.markdown_dir, md_files[0]), encoding="utf-8").read()
    assert "<!--" not in cached
    assert "<div" not in cached
    assert "![图]" not in cached and "fig.png" not in cached
    assert "](#app)" not in cached
    assert "# 报告标题" in cached and "图注" in cached and "附录" in cached
    assert all(
        "<!--" not in r.get("text", "")
        and "<div" not in r.get("text", "")
        and "](#" not in r.get("text", "")
        for r in store.records
    )


def test_clean_markdown_document_preserves_html_table_blocks():
    """复杂表格以 HTML <table> 块形式出现时，整段保留而不是被拆散。"""
    from agent.rag.cleaner import clean_markdown_document

    table = (
        "<table>\n"
        "  <tr><th>姓名</th><th>年龄</th></tr>\n"
        "  <tr><td>张三</td><td>30</td></tr>\n"
        "</table>"
    )
    md = f"# 标题\n\n正文。\n\n{table}\n\n结尾。\n"
    out = clean_markdown_document(md)
    assert "<table>" in out and "</table>" in out
    assert "<tr>" in out and "<td>" in out and "<th>" in out
    assert "张三" in out and "30" in out
    # 同时其它噪音仍被清掉
    assert "<!--" not in out


def test_clean_text_preserves_html_table_blocks():
    """clean_text（非 markdown 路径）也保住 <table> 块，不被 <[^>]+> 剥掉。"""
    from agent.rag.cleaner import clean_text

    table = (
        "<table>\n<tr><td>cell &amp; more</td></tr>\n</table>"
    )
    out = clean_text(f"前面文字。\n\n{table}\n\n后面文字。")
    assert "<table>" in out and "</table>" in out
    assert "<tr>" in out and "<td>" in out
    assert "cell" in out and "more" in out


def test_docling_markdown_keeps_table_block_on_ingest(monkeypatch, tmp_path):
    """docling 输出若含 HTML <table> 块（复杂表格），入库时整段保留。"""
    store = _FakeStore()
    engine, kb = _make_engine_only_kb(
        tmp_path, store, extra_files={"scan.pdf": b"%PDF-1.4 fake"})
    docling_md = (
        "# 报告\n\n正文。\n\n"
        "<table>\n<tr><th>项目</th><th>值</th></tr>\n"
        "<tr><td>精度</td><td>95%</td></tr>\n</table>\n\n"
        "<!-- image -->\n"
    )
    monkeypatch.setattr(
        "agent.rag.engine._extract_with_pdf_engine",
        lambda fp, pe="docling": (docling_md, True),
    )
    engine.ingest(force=True)
    md_files = [f for f in os.listdir(engine.markdown_dir) if f.endswith(".md")]
    cached = open(os.path.join(engine.markdown_dir, md_files[0]), encoding="utf-8").read()
    assert "<table>" in cached and "</table>" in cached
    assert "<tr>" in cached and "<td>" in cached and "精度" in cached and "95%" in cached
    assert "<!--" not in cached
    assert all("<table>" in r.get("text", "") for r in store.records if "精度" in r.get("text", ""))


# ---------------------------------------------------------------------------
# 空白压缩：markdown / 纯文本里不必要的空行与行尾空格收敛（保留格式、提密度）
# ---------------------------------------------------------------------------

def test_compress_markdown_whitespace_collapses_blanks_and_trailing():
    """连续空行收敛为 1 个、行尾空格剥掉、首尾空行去掉。"""
    from agent.rag.cleaner import compress_markdown_whitespace

    md = (
        "\n\n"
        "# 标题   \n"
        "\n\n\n\n"
        "第一段。\n"
        "   \n"
        "\n"
        "第二段。   \n"
        "\n\n\n"
    )
    out = compress_markdown_whitespace(md)
    assert out == "# 标题\n\n第一段。\n\n第二段。"
    assert "\n\n\n" not in out
    assert not out.startswith("\n") and not out.endswith("\n")
    assert "标题   " not in out and "段。   " not in out


def test_compress_markdown_whitespace_protects_code_blocks():
    """围栏代码块逐字不动：内部空行、行尾空白、缩进全部保留。"""
    from agent.rag.cleaner import compress_markdown_whitespace

    md = (
        "前置段。\n"
        "\n\n\n"
        "```python\n"
        "def f():   \n"
        "\n"
        "\n"
        "    return 1   \n"
        "```\n"
        "\n\n\n"
        "后继段。\n"
    )
    out = compress_markdown_whitespace(md)
    assert "def f():   " in out            # 代码内行尾空白不动
    assert "    return 1   " in out        # 代码内缩进 + 行尾空白不动
    assert "```python\n" in out
    # 代码块内部两个连续空行保留
    block = out.split("```python\n", 1)[1].split("\n```", 1)[0]
    assert block == "def f():   \n\n\n    return 1   "
    # 围栏外的连续空行仍被收敛
    assert "前置段。\n\n```python" in out
    assert "```\n\n后继段。" in out


def test_compress_markdown_whitespace_protects_indented_code():
    """4 空格缩进的代码行逐字保留（含行尾空白）。"""
    from agent.rag.cleaner import compress_markdown_whitespace

    md = "段一。\n\n    code_line();   \n    another();\n\n段二。\n"
    out = compress_markdown_whitespace(md)
    assert "    code_line();   " in out
    assert "    another();" in out
    assert out == "段一。\n\n    code_line();   \n    another();\n\n段二。"


def test_compress_markdown_whitespace_protects_table_blocks():
    """HTML <table> 块整体不动（内部空行/换行/缩进逐字保留）。"""
    from agent.rag.cleaner import compress_markdown_whitespace

    md = (
        "前段。\n\n\n"
        "<table>\n"
        "  <tr>\n\n"
        "    <td>x</td>\n"
        "  </tr>\n"
        "</table>\n\n\n"
        "后段。\n"
    )
    out = compress_markdown_whitespace(md)
    assert "<table>\n  <tr>\n\n    <td>x</td>\n  </tr>\n</table>" in out
    assert "前段。\n\n<table>" in out
    assert "</table>\n\n后段。" in out
    assert "\n\n\n" not in out


def test_compress_markdown_whitespace_keeps_structure_and_indent():
    """标题/列表/引用/表格/行首缩进等格式保留，只剥行尾空白。"""
    from agent.rag.cleaner import compress_markdown_whitespace

    md = (
        "## 小节   \n"
        "\n"
        "- 列表项一   \n"
        "  - 嵌套项   \n"
        "\n"
        "> 引用行   \n"
        "\n"
        "| a | b |\n"
        "| - | - |\n"
        "| 1 | 2 |\n"
        "\n"
        "    缩进段落保留行首\n"
    )
    out = compress_markdown_whitespace(md)
    assert "## 小节\n" in out
    assert "- 列表项一\n" in out
    assert "  - 嵌套项\n" in out          # 嵌套缩进保留
    assert "> 引用行\n" in out
    assert "| a | b |\n| - | - |\n| 1 | 2 |" in out
    assert "    缩进段落保留行首" in out   # 4 空格缩进行当代码行保护
    assert "\n\n" in out                  # 单空行段落分隔保留
    assert "\n\n\n" not in out


def test_compress_markdown_whitespace_idempotent():
    from agent.rag.cleaner import compress_markdown_whitespace

    md = "# 标题\n\n\n第一段。   \n\n\n\n第二段。\n"
    once = compress_markdown_whitespace(md)
    twice = compress_markdown_whitespace(once)
    assert once == twice


def test_clean_markdown_document_compresses_whitespace():
    """公开入口 clean_markdown_document 也做空白压缩（剥注释后残留的空行）。"""
    from agent.rag.cleaner import clean_markdown_document

    md = (
        "# 报告\n\n\n"
        "<!-- image -->\n"
        "\n\n"
        "<div>正文段</div>   \n"
        "\n\n\n"
        "![图](a.png)\n"
        "\n\n"
        "结尾。\n"
    )
    out = clean_markdown_document(md)
    assert out == "# 报告\n\n正文段\n\n结尾。"
    assert "<!--" not in out and "<div" not in out and "![图]" not in out
    assert "\n\n\n" not in out


def test_clean_text_collapses_blank_runs_and_spaces():
    """纯文本路径（clean_text）保证空行收敛、行尾/连续空格清除。"""
    from agent.rag.cleaner import clean_text

    out = clean_text("第一段。   \n\n\n\n第二段，含   多个   空格。\n\n\n")
    assert "\n\n\n" not in out
    assert out == "第一段。\n\n第二段，含 多个 空格。"


def test_docling_markdown_whitespace_compressed_on_ingest(monkeypatch, tmp_path):
    """docling 输出带大量空行/行尾空格 → 落库的缓存 markdown 是压缩后的。"""
    store = _FakeStore()
    engine, kb = _make_engine_only_kb(
        tmp_path, store, extra_files={"scan.pdf": b"%PDF-1.4 fake"})
    docling_md = (
        "# 报告标题   \n"
        "\n\n\n"
        "<!-- image -->\n"
        "\n\n"
        "第一段正文。   \n"
        "\n\n\n\n"
        "第二段正文。\n"
        "\n\n\n"
    )
    monkeypatch.setattr(
        "agent.rag.engine._extract_with_pdf_engine",
        lambda fp, pe="docling": (docling_md, True),
    )
    stats = engine.ingest(force=True)
    assert stats["files_extracted"] == 1

    md_files = [f for f in os.listdir(engine.markdown_dir) if f.endswith(".md")]
    cached = open(os.path.join(engine.markdown_dir, md_files[0]), encoding="utf-8").read()
    assert "\n\n\n" not in cached
    assert "标题   " not in cached
    assert "<!--" not in cached
    assert cached == "# 报告标题\n\n第一段正文。\n\n第二段正文。"
    assert all(
        "\n\n\n" not in r.get("text", "")
        for r in store.records
    )


def test_sync_heartbeat_logs_progress_during_long_file(monkeypatch, tmp_path):
    """长转换期间有心跳进度日志；同步结束后心跳停止（默认 60s，测试调小）。"""
    import time as _time

    import agent.rag.engine as engine_mod

    monkeypatch.setattr(engine_mod, "_HEARTBEAT_INTERVAL_S", 0.3)

    store = _FakeStore()
    engine, kb = _make_engine_only_kb(tmp_path, store)
    (kb / "slow.pdf").write_bytes(b"%PDF-1.4 fake")

    def _slow_extract(fp, pe="docling"):
        _time.sleep(1.5)                       # 模拟分钟级 PDF 转换
        return "# 慢文件\n\n" + "正文内容" * 100, True

    monkeypatch.setattr(engine_mod, "_extract_with_pdf_engine", _slow_extract)

    logs: list = []
    stats = engine.ingest(force=True, log_callback=logs.append)
    assert stats["files_extracted"] == 1

    hb = [l for l in logs if "同步进行中" in l]
    assert hb, "长转换期间应出现心跳进度日志"
    assert any("slow.pdf" in l and "处理中 1 个" in l for l in hb), hb

    # 同步结束后心跳停止：等在飞心跳落地后不应再有新行
    _time.sleep(0.8)
    n = len(logs)
    _time.sleep(0.8)
    assert len(logs) == n


def test_sync_heartbeat_disabled_with_zero_interval(monkeypatch, tmp_path):
    """RAG_HEARTBEAT_S=0 时完全不打心跳（不污染日志）。"""
    import time as _time

    import agent.rag.engine as engine_mod

    monkeypatch.setattr(engine_mod, "_HEARTBEAT_INTERVAL_S", 0)

    store = _FakeStore()
    engine, kb = _make_engine_only_kb(tmp_path, store)
    (kb / "a.txt").write_text("普通小文件，内容足够长以便切片。" * 5, encoding="utf-8")

    logs: list = []
    engine.ingest(force=True, log_callback=logs.append)
    assert not any("同步进行中" in l for l in logs)

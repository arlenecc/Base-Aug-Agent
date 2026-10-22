"""跨目录副本去重 + content_hash 副本清洗测试。

两层防护：
  1. **解析前**：文件采样指纹（_file_fingerprint）识别字节级副本——同一本书
     放在不同目录只解析规范源一次，其余记 manifest 为 duplicate_of，不解析
     不写向量；
  2. **同步后**：按 content_hash 分组清洗——历史遗留的同 content_hash 多来源、
     以及指纹漏网的「同内容不同字节」（重新保存/格式转换），只保留规范源的
     向量，其余的向量/digest/markdown 缓存一并清掉。

另覆盖：规范源被删后副本晋升、规范源解析失败后副本晋升、同 size 不同内容
不误判。
"""
from __future__ import annotations

import os

import pytest

KB_TEXT = (
    "同一本书的正文内容，足够长以便切成多个切片。"
    "第二段继续讨论同一个主题，确保内容哈希稳定。"
    "第三段收尾。"
)


# ---------------------------------------------------------------------------
# 本地最小 fake（自包含，不跨测试模块 import）
# ---------------------------------------------------------------------------

class _FakeStore:
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
        self.records = [r for r in self.records if r.get("source") != source]
        return 1

    def delete_document(self, doc_name):
        self.deleted_documents.append(doc_name)

    def count(self):
        return self._count

    def list_sources(self):
        return sorted({r.get("source", "") for r in self.records if r.get("source")})


def _make_engine(tmp_path, store):
    from agent.rag.engine import RAGEngine

    kb = tmp_path / "kb"
    kb.mkdir(exist_ok=True)
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    engine = RAGEngine(workspace=str(ws), knowledge_base=str(kb))
    engine._store = store
    return engine, kb


def _write(pdir, name, text):
    p = pdir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return str(p)


def _load_manifest(engine):
    from agent.rag.engine import _load_manifest
    return _load_manifest(engine._rag_dir)


# ---------------------------------------------------------------------------
# _file_fingerprint 单元测试
# ---------------------------------------------------------------------------

def test_fingerprint_identical_for_byte_copies(tmp_path):
    from agent.rag.engine import _file_fingerprint

    a = _write(tmp_path, "a/book.txt", KB_TEXT)
    b = _write(tmp_path, "b/book.txt", KB_TEXT)
    assert _file_fingerprint(a) == _file_fingerprint(b)


def test_fingerprint_differs_for_different_content(tmp_path):
    from agent.rag.engine import _file_fingerprint

    a = _write(tmp_path, "a/book.txt", KB_TEXT)
    b = _write(tmp_path, "b/other.txt", KB_TEXT + "完全不同的结尾。")
    assert _file_fingerprint(a) != _file_fingerprint(b)


def test_fingerprint_sampling_catches_middle_and_tail_edits(tmp_path):
    """大文件（>3×64KB 采样）只改中段/尾段也能被识别为不同文件。"""
    from agent.rag.engine import _file_fingerprint

    base = ("x" * 100 + "\n") * 4000          # ~404KB，走采样路径
    big = tmp_path / "big.txt"
    big.write_text(base, encoding="utf-8")

    copy_same = tmp_path / "copy.txt"
    copy_same.write_text(base, encoding="utf-8")
    assert _file_fingerprint(str(big)) == _file_fingerprint(str(copy_same))

    # 改中段
    mid = tmp_path / "mid.txt"
    mid_text = base[: len(base) // 2] + "中段被篡改" + base[len(base) // 2 + 5:]
    mid.write_text(mid_text, encoding="utf-8")
    assert _file_fingerprint(str(big)) != _file_fingerprint(str(mid))

    # 改尾段
    tail = tmp_path / "tail.txt"
    tail.write_text(base[:-10] + "尾部不同!!", encoding="utf-8")
    assert _file_fingerprint(str(big)) != _file_fingerprint(str(tail))

    # 截断（size 变）
    trunc = tmp_path / "trunc.txt"
    trunc.write_text(base[:1000], encoding="utf-8")
    assert _file_fingerprint(str(big)) != _file_fingerprint(str(trunc))


def test_fingerprint_no_false_positive_same_size(tmp_path):
    """同 size 不同内容（尾部差异）不得误判为副本。"""
    from agent.rag.engine import _file_fingerprint

    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("A" * 5000, encoding="utf-8")
    b.write_text("A" * 4999 + "B", encoding="utf-8")
    assert a.stat().st_size == b.stat().st_size
    assert _file_fingerprint(str(a)) != _file_fingerprint(str(b))


# ---------------------------------------------------------------------------
# 解析前指纹去重
# ---------------------------------------------------------------------------

def test_cross_directory_copies_parsed_once(tmp_path):
    """不同目录下的字节级副本：只解析规范源一次，另一个不解析不写向量。"""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    a = _write(kb, "books/a.txt", KB_TEXT)
    b = _write(kb, "archive/backup/a.txt", KB_TEXT)
    canonical, dup = sorted([a, b])  # 规范源 = 路径字典序第一个（确定性）

    stats = engine.ingest(force=False)

    assert stats["files_found"] == 2
    assert stats["files_extracted"] == 1
    assert stats["files_deduped"] == 1
    # 只有一个来源的向量（规范源）
    assert store.list_sources() == [canonical]
    # 副本记进 manifest，指向规范源
    manifest = _load_manifest(engine)
    assert manifest[dup]["duplicate_of"] == canonical
    assert manifest[canonical].get("content_hash")
    assert "duplicate_of" not in manifest[canonical]


def test_copy_of_already_ingested_file_skipped(tmp_path):
    """先入库 A，再添加副本 B：第二次同步 B 不解析、不新增向量。"""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    a = _write(kb, "a.txt", KB_TEXT)

    engine.ingest(force=False)
    records_after_first = len(store.records)
    assert records_after_first > 0

    b = _write(kb, "sub/dir/b.txt", KB_TEXT)
    stats = engine.ingest(force=False)

    assert stats["files_extracted"] == 0     # 没有新解析
    assert stats["files_deduped"] == 1
    assert len(store.records) == records_after_first  # 没有新向量
    assert store.list_sources() == [a]
    manifest = _load_manifest(engine)
    assert manifest[b]["duplicate_of"] == a


def test_duplicate_promoted_when_canonical_deleted(tmp_path):
    """规范源被删后，仅存的副本晋升为待处理并入库（书不会消失）。"""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    a = _write(kb, "a.txt", KB_TEXT)
    b = _write(kb, "sub/b.txt", KB_TEXT)

    engine.ingest(force=False)
    assert _load_manifest(engine)[b]["duplicate_of"] == a

    os.unlink(a)  # 删掉规范源
    stats = engine.ingest(force=False)

    assert stats["files_extracted"] == 1     # 副本被重新处理
    manifest = _load_manifest(engine)
    assert "duplicate_of" not in manifest[b]
    assert manifest[b].get("content_hash")
    assert store.list_sources() == [b]


def test_duplicate_retry_when_canonical_fails_parsing(tmp_path, monkeypatch):
    """规范源解析失败（空输出）时，副本下轮优先——书不因规范源失败消失。"""
    from agent.rag import engine as engine_mod

    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    a = _write(kb, "a.txt", KB_TEXT)
    b = _write(kb, "sub/b.txt", KB_TEXT)

    real_extract = engine_mod._extract_with_pdf_engine

    def _fake_extract(fp, pe="docling"):
        # 规范源 a.txt 返回空 → worker 记「提取文本为空」，不入库
        if fp == a:
            return "", False
        return real_extract(fp, pe)

    monkeypatch.setattr(engine_mod, "_extract_with_pdf_engine", _fake_extract)

    stats = engine.ingest(force=False)
    assert stats["files_extracted"] == 0     # 规范源失败，什么都没入
    manifest = _load_manifest(engine)
    # 副本留下 dup_retry 重试标记（不是被删掉），规范源失败记录在案
    assert manifest[b]["dup_retry"] == a
    assert stats["errors"]

    # 第二次同步：副本优先当规范源并成功入库；失败的 a 反成副本
    stats2 = engine.ingest(force=False)
    assert stats2["files_extracted"] == 1
    assert store.list_sources() == [b]
    manifest2 = _load_manifest(engine)
    assert manifest2[b].get("content_hash")
    assert "dup_retry" not in manifest2[b]
    assert manifest2[a]["duplicate_of"] == b


def test_force_sync_restill_dedupes_and_cleans(tmp_path):
    """force 同步同样走收尾清洗：内容副本的向量不会在 force 后回潮。"""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    a = _write(kb, "a.txt", KB_TEXT)
    b = _write(kb, "sub/b.txt", KB_TEXT + " ")   # 多个空格：字节不同、清洗后同文

    # 第一次同步：两个文件指纹不同都解析了，收尾按 content_hash 清掉副本
    stats = engine.ingest(force=False)
    assert stats["files_extracted"] == 2
    assert stats["files_deduped"] == 1
    assert store.list_sources() == [a]
    manifest = _load_manifest(engine)
    assert manifest[b]["duplicate_of"] == a

    # force 再来一次：两个都重新解析（force 语义），收尾清洗依旧只留规范源
    stats2 = engine.ingest(force=True)
    assert stats2["files_extracted"] == 2
    assert stats2["files_deduped"] == 1
    assert store.list_sources() == [a]
    manifest2 = _load_manifest(engine)
    assert manifest2[b]["duplicate_of"] == a


# ---------------------------------------------------------------------------
# 同步后 content_hash 清洗
# ---------------------------------------------------------------------------

def test_content_hash_cleanup_removes_duplicate_vectors(tmp_path):
    """同内容不同字节（指纹漏网）：入库后按 content_hash 删副本向量。"""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    a = _write(kb, "books/a.txt", KB_TEXT)
    b = _write(kb, "copies/b.txt", KB_TEXT + "   \n\n")  # 尾部空白：字节不同

    stats = engine.ingest(force=False)

    # 两个文件都解析了（指纹不同），但只有一个来源的向量留下
    assert stats["files_extracted"] == 2
    assert stats["files_deduped"] == 1
    assert store.list_sources() == [a]
    assert b in store.deleted_sources
    manifest = _load_manifest(engine)
    assert manifest[b]["duplicate_of"] == a
    assert manifest[a].get("content_hash") == manifest[a]["content_hash"]


def test_content_hash_cleanup_purges_digest_and_cache(tmp_path):
    """副本清洗连带删 digest 行与 markdown 缓存（否则 Step 7 会重建 digest）。"""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    a = _write(kb, "a.txt", KB_TEXT)
    b = _write(kb, "sub/b.txt", KB_TEXT + "  ")

    engine.ingest(force=False)
    assert b in store.deleted_sources
    # digest 三个名字都删（rel_path / basename / 全路径）
    rel_b = os.path.relpath(b, str(kb))
    assert rel_b in store.deleted_documents
    assert os.path.basename(b) in store.deleted_documents
    assert b in store.deleted_documents
    # markdown 缓存文件被删
    from agent.rag.engine import _safe_filename
    cache_b = os.path.join(engine.markdown_dir, _safe_filename(rel_b) + ".md")
    assert not os.path.exists(cache_b)


def test_content_hash_cleanup_is_idempotent(tmp_path):
    """清洗过的副本不再被反复拉起（manifest 标记后退出分组）。"""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    a = _write(kb, "a.txt", KB_TEXT)
    b = _write(kb, "sub/b.txt", KB_TEXT + " ")

    engine.ingest(force=False)
    deleted_after_first = list(store.deleted_sources)

    stats = engine.ingest(force=False)
    assert stats["files_deduped"] == 0        # 没有新的清洗动作
    assert store.deleted_sources == deleted_after_first
    assert store.list_sources() == [a]


def test_three_copies_keep_one(tmp_path):
    """三个内容相同的来源只保留一个规范源。"""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    a = _write(kb, "a.txt", KB_TEXT)
    b = _write(kb, "sub/b.txt", KB_TEXT + " ")
    c = _write(kb, "deep/c.txt", KB_TEXT + "  ")

    engine.ingest(force=False)
    assert store.list_sources() == [a]
    manifest = _load_manifest(engine)
    assert manifest[b]["duplicate_of"] == a
    assert manifest[c]["duplicate_of"] == a


def test_distinct_books_not_deduped(tmp_path):
    """内容不同的文件互不影响（不误伤）。"""
    store = _FakeStore()
    engine, kb = _make_engine(tmp_path, store)
    a = _write(kb, "a.txt", KB_TEXT)
    b = _write(kb, "sub/b.txt", "完全不同的另一本书的内容，长度也要足够切片。")

    stats = engine.ingest(force=False)
    assert stats["files_extracted"] == 2
    assert stats["files_deduped"] == 0
    assert len(store.list_sources()) == 2
    manifest = _load_manifest(engine)
    assert "duplicate_of" not in manifest[a]
    assert "duplicate_of" not in manifest[b]

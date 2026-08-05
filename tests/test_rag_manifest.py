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
        self._count = 0

    def add_streaming(self, chunk_iter, batch_size=100, on_batch=None):
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

    def count(self):
        return self._count

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

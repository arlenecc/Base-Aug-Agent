"""In-process memory stores: work memory (scratchpad) and long-term memory."""
from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
import weakref
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class _JsonStore:
    """A tiny thread-safe JSON key/value file store.

    Uses a delayed-write (debounce) strategy: set() / delete() only mark
    _dirty = True.  The next _flush() call (or the next set() after 0.5s)
    actually writes to disk.  This avoids O(N²) I/O when the agent calls
    work_memory.set() in a tight loop.
    """

    # Minimum interval between two _save() calls (seconds).
    _FLUSH_INTERVAL = 0.5

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._data: dict = {}
        self._dirty = False
        self._last_save = 0.0
        self._load()
        # Register atexit to flush dirty data on process exit so debounced
        # writes don't get lost. Use weakref so the store can still be GC'd
        # (atexit holds a strong ref to the callback, weakref breaks the cycle).
        _store_ref = weakref.ref(self)

        def _atexit_flush():
            store = _store_ref()
            if store is not None:
                store.flush(force=True)

        atexit.register(_atexit_flush)

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (OSError, json.JSONDecodeError):
                self._data = {}

    def reload(self) -> None:
        """Re-read the file from disk, discarding the in-memory cache.

        Used after an external write (e.g. another thread / process saved the
        same store file) so this instance picks up the latest data.
        """
        with self._lock:
            self._load()

    def _save(self) -> None:
        """Atomic write: tmp → rename. Must be called under _lock."""
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
            self._dirty = False
            self._last_save = time.monotonic()
        except OSError as e:
            logger.warning("memory: failed to save %s: %s", self.path, e)

    def _flush(self) -> None:
        """Flush pending changes to disk if dirty and interval elapsed."""
        if not self._dirty:
            return
        if time.monotonic() - self._last_save >= self._FLUSH_INTERVAL:
            self._save()

    def flush(self, force: bool = False) -> None:
        """Force-write pending changes to disk.

        When ``force=True`` the debounce interval is bypassed — used by the
        atexit hook and by callers that need the data on disk immediately
        (e.g. before spawning a subprocess that reads the same store file).
        """
        with self._lock:
            if self._dirty and (force or time.monotonic() - self._last_save >= self._FLUSH_INTERVAL):
                self._save()

    def set(self, key: str, value) -> None:
        with self._lock:
            self._data[key] = value
            self._dirty = True
            self._flush()

    def get(self, key: str, default=None):
        with self._lock:
            return self._data.get(key, default)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)
            self._dirty = True
            self._flush()

    def all(self) -> dict:
        with self._lock:
            return dict(self._data)

    def replace(self, data: dict) -> None:
        with self._lock:
            self._data = dict(data)
            self._dirty = True
            # replace is always a big change — save immediately.
            self._save()


class WorkMemory:
    """A working notepad the agent reads/writes during a task."""

    def __init__(self, path: str):
        self._store = _JsonStore(path)

    def set(self, key: str, value: str) -> None:
        self._store.set(key, value)

    def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    def list(self) -> Dict[str, str]:
        return {k: v for k, v in self._store.all().items()}

    def clear(self) -> None:
        self._store.replace({})


class LongTermMemory:
    """Knowledge-graph long-term memory backed by GraphMemoryStore.

    Replaces the old flat fact-list with a structured graph of entities,
    observations, and relations.  Observations are embedded via FastEmbed
    (nomic-embed-text-v1.5-Q) and indexed in LanceDB for semantic recall.

    The public API stays compatible with the old one (add / add_many /
    search / all / clear) so callers that treat memory as a string list
    keep working — each fact is stored as a single observation on a
    generic "General" entity.  The graph CRUD methods are exposed via
    the ``graph`` attribute (a GraphMemoryStore).
    """

    def __init__(self, path: str, embedding_model: str = "nomic-ai/nomic-embed-text-v1.5-Q"):
        from .graph_memory import GraphMemoryStore
        # path is like .../long_memory.json; the graph file is the same
        # path but with .json → .graph.json so old flat-memory files are
        # not clobbered.
        graph_path = path.replace(".json", ".graph.json") if path.endswith(".json") else path + ".graph"
        self._graph = GraphMemoryStore(path=graph_path, embedding_model=embedding_model)
        self._legacy_store = _JsonStore(path)
        # Migrate old flat facts into the graph on first use.
        self._migrate_legacy_facts()

    @property
    def graph(self) -> "GraphMemoryStore":
        """Direct access to the underlying knowledge-graph store."""
        return self._graph

    def _migrate_legacy_facts(self) -> None:
        """One-time migration: old long_memory.json had {"facts": [...]}."""
        facts = self._legacy_store.get("facts")
        if not isinstance(facts, list) or not facts:
            return
        # Only migrate facts not already in the graph.
        existing = {o for e in self._graph.list_entities() for o in e.get("observations", [])}
        new = [f for f in facts if f and f not in existing]
        if new:
            self._graph.add_observations("General", new)
            logger.info("long_memory: migrated %d legacy facts into graph", len(new))
        # Clear the old file so we don't re-migrate.
        self._legacy_store.set("facts", [])

    # -- compatible API (thin wrappers over the graph) --

    def add(self, fact: str) -> None:
        """Add a single fact as an observation on the General entity."""
        self._graph.add_observations("General", [fact])

    def add_many(self, facts: List[str]) -> None:
        """Add many facts as observations on the General entity."""
        self._graph.add_observations("General", facts)

    def all(self) -> List[str]:
        """Return all observations across all entities (flat list, legacy)."""
        out = []
        for e in self._graph.list_entities():
            out.extend(e.get("observations", []))
        return out

    def search(self, query: str, top_k: int = 8) -> List[str]:
        """Semantic search over observations. Returns observation texts."""
        results = self._graph.search(query, top_k=top_k)
        return [r["text"] for r in results]

    def clear(self) -> None:
        self._graph.clear()

    def close(self) -> None:
        self._graph.close()

    # -- snapshot for prompt injection --

    def snapshot(self, max_items: int = 10) -> str:
        """Compact text digest for prompt injection."""
        return self._graph.snapshot(max_items=max_items)

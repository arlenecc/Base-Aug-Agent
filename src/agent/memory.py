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
    """A list of extracted facts, searchable by keyword."""

    def __init__(self, path: str):
        self._store = _JsonStore(path)
        # Defensive: a corrupted/partially-written store may have "facts": null.
        # isinstance check resets to [] so callers never see None.
        # Directly modify _data to avoid triggering a debounced _save()
        # which would set _last_save and delay subsequent writes.
        with self._store._lock:
            if not isinstance(self._store._data.get("facts"), list):
                self._store._data["facts"] = []

    def add(self, fact: str) -> None:
        facts = list(self._store.get("facts") or [])
        fact = fact.strip()
        if fact and fact not in facts:
            facts.append(fact)
            self._store.set("facts", facts)

    def add_many(self, facts: List[str]) -> None:
        # Batch: read once, deduplicate, append all, write once.
        # Avoids O(N²) read-write cycles when adding many facts.
        existing = list(self._store.get("facts") or [])
        seen = set(existing)
        added = 0
        for f in facts:
            f = f.strip()
            if f and f not in seen:
                existing.append(f)
                seen.add(f)
                added += 1
        if added:
            self._store.set("facts", existing)

    def all(self) -> List[str]:
        return list(self._store.get("facts") or [])

    def search(self, query: str) -> List[str]:
        q = query.lower()
        if not q:
            return list(self.all())
        return [f for f in self.all() if q in f.lower()]

    def clear(self) -> None:
        self._store.set("facts", [])

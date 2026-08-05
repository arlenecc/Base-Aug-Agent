"""In-process memory stores: work memory (scratchpad) and long-term memory."""
from __future__ import annotations

import json
import os
import threading
from typing import Dict, List, Optional


class _JsonStore:
    """A tiny thread-safe JSON key/value file store."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._data: dict = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (OSError, json.JSONDecodeError):
                self._data = {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def set(self, key: str, value) -> None:
        with self._lock:
            self._data[key] = value
            self._save()

    def get(self, key: str, default=None):
        with self._lock:
            return self._data.get(key, default)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)
            self._save()

    def all(self) -> dict:
        with self._lock:
            return dict(self._data)

    def replace(self, data: dict) -> None:
        with self._lock:
            self._data = dict(data)
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
        if "facts" not in self._store.all():
            self._store.set("facts", [])

    def add(self, fact: str) -> None:
        facts = list(self._store.get("facts", []))
        fact = fact.strip()
        if fact and fact not in facts:
            facts.append(fact)
            self._store.set("facts", facts)

    def add_many(self, facts: List[str]) -> None:
        # Batch: read once, deduplicate, append all, write once.
        # Avoids O(N²) read-write cycles when adding many facts.
        existing = list(self._store.get("facts", []))
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
        return list(self._store.get("facts", []))

    def search(self, query: str) -> List[str]:
        q = query.lower()
        if not q:
            return list(self.all())
        return [f for f in self.all() if q in f.lower()]

    def clear(self) -> None:
        self._store.set("facts", [])

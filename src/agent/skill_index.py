"""Skill index: scan the .agent/skills/ directory and maintain a JSON
index table for fast skill-search lookup.

Each skill lives in its own subdirectory under .agent/skills/:

    .agent/skills/
    ├── _index.json          ← auto-generated index table
    ├── pdf-extraction/
    │   ├── skill.json        ← metadata (name, description, keywords, tags)
    │   ├── prompt.md         ← the skill's prompt template
    │   └── ... (other support files)
    └── web-scraping/
        ├── skill.json
        └── prompt.md

The index is rebuilt on a timer and on-demand.  It's a flat list of
{name, description, keywords, tags, path, entry} records, kept in a
single JSON file for O(1) loading.

This module is the *index* layer — it reads skill directories and
produces the index.  The SkillSearchTool (in tools/) uses this index
to serve search queries from the LLM.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INDEX_FILENAME = "_index.json"
_SKILL_META_FILENAME = "skill.json"
_DEFAULT_ENTRY = "prompt.md"
_SCAN_INTERVAL = 300  # seconds between automatic scans (5 min)

# Fields we read from skill.json
_REQUIRED_FIELDS = {"name"}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# Skill record (plain dict for easy JSON round-trip)
# ---------------------------------------------------------------------------
#
#   {
#     "name":        "PDF Extraction",
#     "description": "Extract structured data from PDF files...",
#     "keywords":    ["pdf", "extract", "ocr"],
#     "tags":        ["document", "data"],
#     "path":        "pdf-extraction",          # relative to skills/ dir
#     "entry":       "prompt.md",                # file to read for the prompt
#   }
#
# ---------------------------------------------------------------------------


class SkillIndex:
    """Scans the skills/ directory and maintains _index.json.

    Thread-safe.  The scan is debounced — if called again within
    _SCAN_INTERVAL seconds and no files have changed, it's a no-op.
    """

    def __init__(self, skills_dir: str):
        self._dir = skills_dir
        self._index_path = os.path.join(skills_dir, _INDEX_FILENAME)
        self._lock = threading.Lock()
        self._index: List[Dict[str, Any]] = []
        self._last_scan = 0.0
        self._last_dir_mtime = 0.0
        self._load()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the index from disk (if it exists and is fresh)."""
        if not os.path.exists(self._index_path):
            self._index = []
            return
        try:
            with open(self._index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._index = data.get("skills", [])
            self._last_scan = data.get("updated_at_ts", 0)
            # Sync _last_dir_mtime so _should_scan() doesn't immediately
            # trigger a rescan right after loading a fresh index.
            self._last_dir_mtime = self._dir_mtime()
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("skill_index: failed to load %s: %s", self._index_path, e)
            self._index = []

    def _save(self) -> None:
        """Atomic write of the index."""
        try:
            os.makedirs(self._dir, exist_ok=True)
            tmp = self._index_path + ".tmp"
            payload = {
                "version": 1,
                "updated_at": _now_iso(),
                "updated_at_ts": time.time(),
                "skills": self._index,
            }
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._index_path)
        except OSError as e:
            logger.warning("skill_index: failed to save %s: %s", self._index_path, e)

    # ------------------------------------------------------------------
    # scanning
    # ------------------------------------------------------------------

    def _should_scan(self) -> bool:
        """Check if a rescan is needed (directory mtime changed or timeout)."""
        # Always scan if never scanned
        if self._last_scan == 0:
            return True
        # Scan if interval elapsed
        if time.time() - self._last_scan > _SCAN_INTERVAL:
            return True
        # Scan if directory mtime changed
        try:
            current_mtime = self._dir_mtime()
            if current_mtime != self._last_dir_mtime:
                return True
        except OSError:
            pass
        return False

    def _dir_mtime(self) -> float:
        """Get the most recent mtime among all skill subdirectories."""
        max_mtime = 0.0
        if not os.path.isdir(self._dir):
            return 0.0
        for entry in os.scandir(self._dir):
            if entry.is_dir() and not entry.name.startswith("_"):
                try:
                    mtime = entry.stat(follow_symlinks=False).st_mtime
                    if mtime > max_mtime:
                        max_mtime = mtime
                except OSError:
                    pass
        return max_mtime

    def scan(self, force: bool = False) -> int:
        """Scan skill subdirectories and rebuild the index.

        Returns the number of skills indexed.
        """
        with self._lock:
            if not force and not self._should_scan():
                return len(self._index)

            if not os.path.isdir(self._dir):
                self._index = []
                self._save()
                return 0

            new_index: List[Dict[str, Any]] = []
            for entry in os.scandir(self._dir):
                if not entry.is_dir():
                    continue
                if entry.name.startswith("_") or entry.name.startswith("."):
                    continue
                skill_dir = entry.path
                meta_path = os.path.join(skill_dir, _SKILL_META_FILENAME)
                if not os.path.isfile(meta_path):
                    # Not a skill directory (no skill.json)
                    continue
                record = self._read_skill_meta(entry.name, skill_dir, meta_path)
                if record:
                    new_index.append(record)

            self._index = new_index
            self._last_scan = time.time()
            self._last_dir_mtime = self._dir_mtime()
            self._save()
            logger.info("skill_index: indexed %d skills from %s", len(new_index), self._dir)
            return len(new_index)

    def _read_skill_meta(
        self, dirname: str, skill_dir: str, meta_path: str
    ) -> Optional[Dict[str, Any]]:
        """Read skill.json from a skill directory."""
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if not isinstance(meta, dict):
                return None
            name = meta.get("name", "").strip()
            if not name:
                return None
            entry = meta.get("entry", _DEFAULT_ENTRY)
            # Verify the entry file exists
            entry_path = os.path.join(skill_dir, entry)
            if not os.path.isfile(entry_path):
                # Try fallback
                entry = _DEFAULT_ENTRY
                if not os.path.isfile(os.path.join(skill_dir, entry)):
                    logger.warning(
                        "skill_index: skill '%s' has no entry file (%s)",
                        dirname, entry,
                    )
                    return None
            return {
                "name": name,
                "description": meta.get("description", ""),
                "keywords": [k.lower() for k in meta.get("keywords", [])],
                "tags": meta.get("tags", []),
                "path": dirname,
                "entry": entry,
            }
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("skill_index: failed to read %s: %s", meta_path, e)
            return None

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Keyword-based search over the skill index.

        Returns a list of skill records (including path and entry),
        sorted by relevance score.
        """
        # Ensure index is fresh
        self.scan()

        if not self._index or not query.strip():
            return []

        query_tokens = set(_tokenize(query.lower()))
        if not query_tokens:
            return []

        scored = []
        for skill in self._index:
            score = 0
            # Match against name
            name_tokens = set(_tokenize(skill.get("name", "").lower()))
            score += len(query_tokens & name_tokens) * 3
            # Match against keywords
            kw_tokens = set(skill.get("keywords", []))
            score += len(query_tokens & kw_tokens) * 2
            # Match against tags
            tag_tokens = set(t.lower() for t in skill.get("tags", []))
            score += len(query_tokens & tag_tokens) * 2
            # Match against description (substring)
            desc = skill.get("description", "").lower()
            for token in query_tokens:
                if token in desc:
                    score += 1
            if score > 0:
                scored.append((score, skill))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]

    def _resolve_within(self, *parts: str) -> Optional[str]:
        """Resolve parts under the skills dir, rejecting path traversal.

        Returns the absolute path if it stays inside self._dir, else None.
        The LLM supplies `path`/`entry` strings, so we must defend against
        e.g. path="../../../../etc" or entry="../../../secret".
        """
        base = os.path.realpath(self._dir)
        resolved = os.path.realpath(os.path.join(base, *parts))
        if resolved == base or not resolved.startswith(base + os.sep):
            return None
        return resolved

    def get_skill_path(self, skill_path: str) -> Optional[str]:
        """Return the absolute path to a skill directory by its relative path."""
        abs_path = self._resolve_within(skill_path)
        if abs_path and os.path.isdir(abs_path):
            return abs_path
        return None

    def read_skill_content(self, skill_path: str, entry: str = "prompt.md") -> Optional[str]:
        """Read the content of a skill's entry file."""
        abs_path = self._resolve_within(skill_path, entry)
        if abs_path is None:
            logger.warning("skill_index: rejected unsafe path: %s/%s", skill_path, entry)
            return None
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                return f.read()
        except OSError as e:
            logger.warning("skill_index: failed to read %s: %s", abs_path, e)
            return None

    def list_all(self) -> List[Dict[str, Any]]:
        """Return the full index (triggers a scan if needed)."""
        self.scan()
        with self._lock:
            return list(self._index)

    def reload(self) -> None:
        """Force reload from disk."""
        with self._lock:
            self._load()


# ---------------------------------------------------------------------------
# Tokenizer (shared with skills.py)
# ---------------------------------------------------------------------------

import re

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)

"""Skill manager: 固化 frequently-run tasks into reusable skills.

A skill records a name, trigger keywords, and a prompt template. When the same
intent is requested 2+ times, the manager suggests saving it as a skill.

Also supports directory-based skills under .agent/skills/. Each skill
directory has a skill.json metadata file and a prompt.md entry file.
The SkillIndex (skill_index.py) scans these directories and maintains
an _index.json table for fast lookup via the skill_search tool.
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from .memory import _JsonStore

_TOKEN_RE = re.compile(r"\w+")


def _keywords_of(text: str) -> List[str]:
    return [w.lower() for w in _TOKEN_RE.findall(text)]


@dataclass
class Skill:
    name: str
    keywords: List[str]
    prompt: str
    count: int = 0

    def score(self, text: str) -> int:
        toks = set(_keywords_of(text))
        return sum(1 for k in self.keywords if k.lower() in toks)


class SkillManager:
    SUGGEST_THRESHOLD = 2

    def __init__(self, path: str, skills_dir: str = ""):
        self._store = _JsonStore(path)
        self._skills_dir = skills_dir  # .agent/skills/ directory
        # Defensive: a corrupted store may have null values. Directly modify
        # _data to avoid triggering a debounced _save() which would set
        # _last_save and delay subsequent real writes.
        with self._store._lock:
            if not isinstance(self._store._data.get("skills"), list):
                self._store._data["skills"] = []
            if not isinstance(self._store._data.get("requests"), dict):
                self._store._data["requests"] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def reload(self) -> None:
        """Reload the store from disk.

        Called after an external write (e.g. the UI thread saved a new skill
        via create_skill while the worker thread's SkillManager instance still
        holds the old in-memory cache). Without this, match() in the worker
        would keep returning None for the just-saved skill and re-trigger
        suggestions every turn.
        """
        with self._lock:
            self._store.reload()
            if not isinstance(self._store.get("skills"), list):
                self._store.set("skills", [])
            if not isinstance(self._store.get("requests"), dict):
                self._store.set("requests", {})

    def list(self) -> List[Skill]:
        with self._lock:
            return [Skill(**s) for s in (self._store.get("skills") or [])]

    def create_skill(self, name: str, keywords: List[str], prompt: str) -> Skill:
        skill = Skill(name=name, keywords=[k.lower() for k in keywords], prompt=prompt)
        with self._lock:
            skills = list(self._store.get("skills") or [])
            skills = [s for s in skills if s["name"] != name]
            skills.append(asdict(skill))
            self._store.set("skills", skills)
        return skill

    def delete(self, name: str) -> None:
        with self._lock:
            skills = [s for s in (self._store.get("skills") or []) if s["name"] != name]
            self._store.set("skills", skills)

    def create_dir_skill(self, name: str, keywords: List[str], prompt: str,
                         description: str = "", tags: List[str] = None) -> str:
        """Create a directory-based skill under .agent/skills/.

        This is the new unified format: each skill gets its own directory
        with skill.json + prompt.md.  Returns the skill directory path.

        The old flat-skill create_skill() still works for backwards compat,
        but new skills should use this method.
        """
        if not self._skills_dir:
            raise ValueError("skills_dir not configured")

        # Sanitize name → directory name
        dir_name = re.sub(r"[^\w\-.]", "-", name.lower().strip()).strip("-")
        if not dir_name:
            dir_name = "unnamed-skill"
        # Ensure uniqueness
        skill_dir = os.path.join(self._skills_dir, dir_name)
        counter = 1
        while os.path.exists(skill_dir):
            skill_dir = os.path.join(self._skills_dir, f"{dir_name}-{counter}")
            counter += 1

        os.makedirs(skill_dir, exist_ok=True)

        # Write skill.json
        meta = {
            "name": name,
            "description": description or f"Skill: {name}",
            "keywords": [k.lower() for k in keywords],
            "tags": tags or [],
            "version": "1.0",
            "entry": "prompt.md",
        }
        with open(os.path.join(skill_dir, "skill.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # Write prompt.md
        with open(os.path.join(skill_dir, "prompt.md"), "w", encoding="utf-8") as f:
            f.write(prompt)

        # Also register a flat record so _match_skill() can hit this skill.
        # The flat record's prompt tells the agent to load the full
        # instructions via skill_load instead of embedding them in the
        # system prompt (keeps prompts lean).
        dir_name = os.path.basename(skill_dir)
        with self._lock:
            skills = list(self._store.get("skills") or [])
            if not any(s["name"] == name for s in skills):
                skills.append(asdict(Skill(
                    name=name,
                    keywords=list(keywords),
                    prompt=(
                        f"A saved skill '{name}' matches this task. "
                        f"Call skill_load with path=\"{dir_name}\" to read its "
                        f"instructions, then follow them."
                    ),
                )))
                self._store.set("skills", skills)

        return skill_dir

    def match(self, text: str) -> Optional[Skill]:
        best: Optional[Skill] = None
        best_score = 0
        for s in self.list():
            sc = s.score(text)
            if sc > best_score:
                best_score = sc
                best = s
        return best if best_score > 0 else None

    # ------------------------------------------------------------------
    # request tracking -> suggest固化 after threshold
    # ------------------------------------------------------------------
    @staticmethod
    def _request_key(text: str) -> str:
        """Normalize a request to a stable key for counting similar requests.

        Uses the sorted set of unique keywords (minus stopwords) so paraphrased
        requests like "deploy to staging" and "deploy staging" map to the same
        key. The old implementation joined ALL keywords in original order, so
        dropping or adding a single word produced a different key and the
        counter never reached the suggestion threshold.
        """
        STOPWORDS = {
            "a", "an", "the", "to", "of", "in", "on", "at", "for",
            "and", "or", "is", "are", "was", "were", "be", "been",
            "do", "does", "did", "with", "from", "by", "as", "it",
            "this", "that", "these", "those", "i", "we", "you",
            "please", "me", "my", "our",
        }
        kws = sorted({w for w in _keywords_of(text) if w not in STOPWORDS})
        return " ".join(kws)[:80]

    def record_request(self, text: str) -> Optional[Skill]:
        """Track a user request. Returns a *suggested* (unsaved) Skill once the
        same intent has been seen SUGGEST_THRESHOLD times. The caller decides
        whether to persist it via create_skill().
        """
        key = self._request_key(text)
        if not key:
            return None
        with self._lock:
            requests = dict(self._store.get("requests") or {})
            requests[key] = requests.get(key, 0) + 1
            self._store.set("requests", requests)
            count = requests[key]

        if count >= self.SUGGEST_THRESHOLD:
            kws = sorted(set(_keywords_of(text)))[:8]
            # Don't auto-persist; just suggest. Avoid suggesting repeatedly:
            # if a real skill with overlapping keywords exists, return None.
            existing = self.match(text)
            if existing is not None:
                return None
            return Skill(name="_suggested", keywords=kws, prompt=text, count=count)
        return None

"""Skill manager: 固化 frequently-run tasks into reusable skills.

A skill records a name, trigger keywords, and a prompt template. When the same
intent is requested 2+ times, the manager suggests saving it as a skill.
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

    def __init__(self, path: str):
        self._store = _JsonStore(path)
        if "skills" not in self._store.all():
            self._store.set("skills", [])
        if "requests" not in self._store.all():
            self._store.set("requests", {})
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def list(self) -> List[Skill]:
        return [Skill(**s) for s in self._store.get("skills", [])]

    def create_skill(self, name: str, keywords: List[str], prompt: str) -> Skill:
        skill = Skill(name=name, keywords=[k.lower() for k in keywords], prompt=prompt)
        with self._lock:
            skills = list(self._store.get("skills", []))
            skills = [s for s in skills if s["name"] != name]
            skills.append(asdict(skill))
            self._store.set("skills", skills)
        return skill

    def delete(self, name: str) -> None:
        with self._lock:
            skills = [s for s in self._store.get("skills", []) if s["name"] != name]
            self._store.set("skills", skills)

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
    def record_request(self, text: str) -> Optional[Skill]:
        """Track a user request. Returns a *suggested* (unsaved) Skill once the
        same intent has been seen SUGGEST_THRESHOLD times. The caller decides
        whether to persist it via create_skill().
        """
        key = " ".join(_keywords_of(text))[:80]
        if not key:
            return None
        with self._lock:
            requests = dict(self._store.get("requests", {}))
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

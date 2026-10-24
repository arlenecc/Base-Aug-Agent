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
from typing import Any, Dict, List, Optional

from .memory import _JsonStore

_TOKEN_RE = re.compile(r"\w+")
# 中文没有空格，\w+ 会把整段连续中文当成一个 token（"提取pdf表格" →
# {"提取pdf表格"}），导致 "pdf"/"提取" 这类关键词永远匹配不上，中文技能
# 检索彻底失效。因此额外切出 CJK 子串及其二元组（bigram）用于匹配。
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
# 按「字script」切出极大运行段：CJK 串、拉丁词、数字串各自独立。\w+ 会把
# 「帮我提取pdf里的表格」整个当成一个 token（CJK 与拉丁都是 \w），于是关键词
# "pdf"/"docx"/"api" 永远匹配不上——而真实意图几乎全是中文夹英文术语。
_SCRIPT_RUN_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z]+|[0-9]+")

# 停用词表提到模块级：_request_key() 每轮对话都会调用，旧实现在函数体内重建
# 这个集合，白白重复分配。
_STOPWORDS = frozenset({
    "a", "an", "the", "to", "of", "in", "on", "at", "for",
    "and", "or", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "with", "from", "by", "as", "it",
    "this", "that", "these", "those", "i", "we", "you",
    "please", "me", "my", "our",
})

# 「请求计数」表的条目上限（见 SkillManager.record_request）。
_MAX_TRACKED_REQUESTS = 500


def _keywords_of(text: str) -> List[str]:
    """Tokenize for keyword matching, with CJK + Latin support.

    Text is split into maximal runs by script (CJK / Latin / digits) so a mixed
    phrase like ``帮我提取pdf里的表格`` yields ``pdf`` as its own token — without
    this, ``\\w+`` treats the whole thing as one token and English keywords
    (``pdf`` / ``docx`` / ``api``) can never match, silently disabling skill
    matching for the most common real-world phrasing.  CJK runs additionally
    emit their character bigrams so a keyword like ``提取`` matches inside
    ``提取pdf表格``.
    """
    toks: List[str] = []
    for run in _SCRIPT_RUN_RE.findall(text):
        low = run.lower()
        toks.append(low)
        if _CJK_RUN_RE.fullmatch(run):
            for i in range(len(run) - 1):
                toks.append(run[i:i + 2])
    return toks


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

    def _load_persisted(self) -> List[Skill]:
        """Read persisted flat skill records (skills.json) only.

        A single malformed record (hand-edited skills.json, an aborted write,
        a record from an older schema) used to raise TypeError/KeyError inside
        ``Skill(**s)`` — which propagated out of ``match()`` and broke every
        turn's skill lookup, not just the bad record.  Skip bad entries instead.
        """
        raw = self._store.get("skills") or []
        if not isinstance(raw, list):
            raw = []
        out: List[Skill] = []
        for s in raw:
            if not isinstance(s, dict):
                continue
            try:
                out.append(Skill(
                    name=str(s.get("name", "")),
                    keywords=list(s.get("keywords") or []),
                    prompt=str(s.get("prompt", "")),
                    count=int(s.get("count", 0) or 0),
                ))
            except (TypeError, ValueError):
                continue
        return [s for s in out if s.name]

    def _load_skills(self) -> List[Skill]:
        """合并持久化记录与磁盘目录技能。

        目录技能（``.agent/skills/*`` 与 ``<workspace>/skills/*``）也一并并入：
        它们是用户直接放在磁盘上的，只读 skills.json 会让这些技能永远拿不到
        自动注入（``# Active skill``）——检索工具能命中、自动应用却失效。

        同名去重时**持久化记录优先**：``create_dir_skill`` 会在 skills.json 里
        注册一条 flat 记录，其 prompt 引导模型走 ``skill_load`` 读完整指令；若
        目录技能反客为主，自动注入时就会把原始 prompt.md 直接塞进系统提示，
        违背「保持 prompt 精简」的设计。
        """
        # 先持久化记录，再目录技能：去重保留前者。
        out: List[Skill] = []
        out.extend(self._load_persisted())
        out.extend(self._load_dir_skills())
        seen: set = set()
        unique: List[Skill] = []
        for s in out:
            if not s.name or s.name in seen:
                continue
            seen.add(s.name)
            unique.append(s)
        return unique

    # ------------------------------------------------------------------
    # 目录技能（磁盘上的 skill.json / SKILL.md）
    # ------------------------------------------------------------------

    def _skill_roots(self) -> List[str]:
        """All directories that may hold on-disk skills."""
        roots: List[str] = []
        if self._skills_dir:
            roots.append(self._skills_dir)
        # skills.json 位于 <ws>/.agent/skills.json → 同级 skills/ 是新体系根。
        try:
            ws = os.path.dirname(os.path.dirname(os.path.abspath(self._store.path)))
        except Exception:
            ws = ""
        if ws:
            roots.append(os.path.join(ws, "skills"))
        return [r for r in roots if r and os.path.isdir(r)]

    def _load_dir_skills(self) -> List[Skill]:
        """Scan on-disk skill directories into flat Skill records.

        纯关键词打分用（自动注入每轮都跑，不能碰 embedding），所以只取
        name / keywords / tags / description 做关键词源，prompt 取正文前
        一段（完整内容仍由 skill_load 读）。
        """
        out: List[Skill] = []
        for root in self._skill_roots():
            try:
                entries = sorted(os.listdir(root))
            except OSError:
                continue
            for name in entries:
                if name.startswith((".", "_")) or name in ("tests", "test"):
                    continue
                d = os.path.join(root, name)
                if not os.path.isdir(d):
                    continue
                rec = self._read_dir_skill(d)
                if rec is not None:
                    out.append(rec)
        return out

    def _read_dir_skill(self, skill_dir: str) -> Optional[Skill]:
        """Read one skill directory (skill.json+prompt.md or SKILL.md)."""
        meta: Dict[str, Any] = {}
        prompt = ""
        sj = os.path.join(skill_dir, "skill.json")
        if os.path.isfile(sj):
            try:
                with open(sj, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    meta = loaded
            except (OSError, ValueError):
                meta = {}
            for entry in (meta.get("entry") or "prompt.md", "prompt.md"):
                p = os.path.join(skill_dir, entry)
                if os.path.isfile(p):
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            prompt = f.read(4000)
                    except (OSError, UnicodeDecodeError):
                        prompt = ""
                    break
        else:
            sk = os.path.join(skill_dir, "SKILL.md")
            if not os.path.isfile(sk):
                return None
            try:
                with open(sk, "r", encoding="utf-8") as f:
                    text = f.read()
            except (OSError, UnicodeDecodeError):
                return None
            meta, prompt = self._split_frontmatter(text)
        name = str(meta.get("name") or "").strip()
        if not name:
            return None
        keywords = meta.get("keywords") or meta.get("tags") or []
        if isinstance(keywords, str):
            keywords = [keywords]
        desc = str(meta.get("description") or "").strip()
        kw = [str(k).lower() for k in keywords]
        if desc:
            kw.append(desc.lower())
        return Skill(name=name, keywords=kw, prompt=prompt[:1500])

    @staticmethod
    def _split_frontmatter(text: str):
        """Minimal YAML-frontmatter split for SKILL.md (no yaml dependency)."""
        meta: Dict[str, Any] = {}
        body = text
        m = re.match(r"\A---\s*\n(.*?)\n---\s*\n?", text, re.DOTALL)
        if m:
            body = text[m.end():]
            for line in m.group(1).splitlines():
                if ":" not in line:
                    continue
                key, _, value = line.partition(":")
                key = key.strip().lower()
                value = value.strip().strip('"').strip("'")
                if key in ("name", "description"):
                    meta[key] = value
                elif key in ("tags", "keywords"):
                    inner = value.strip().lstrip("[").rstrip("]")
                    meta["tags"] = [t.strip().strip('"').strip("'")
                                    for t in inner.split(",") if t.strip()]
        return meta, body

    def list(self) -> List[Skill]:
        with self._lock:
            return self._load_skills()

    def create_skill(self, name: str, keywords: List[str], prompt: str) -> Skill:
        # 元素类型不设防的话，UI/调用方传入 [123] 这类 keywords 会在 k.lower()
        # 处抛 AttributeError；Skill.score/match 全生命周期都假设 str。
        skill = Skill(
            name=name,
            keywords=[str(k).lower() for k in (keywords or [])],
            prompt=prompt,
        )
        with self._lock:
            skills = [s for s in self._load_skills() if s.name != name]
            skills.append(skill)
            self._store.set("skills", [asdict(s) for s in skills])
        return skill

    def delete(self, name: str) -> None:
        with self._lock:
            skills = [s for s in self._load_skills() if s.name != name]
            self._store.set("skills", [asdict(s) for s in skills])

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
            "keywords": [str(k).lower() for k in (keywords or [])],
            "tags": [str(t) for t in (tags or [])],
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
            # _load_persisted（不是 _load_skills，后者会把磁盘目录技能也读进来，
            # 写回 skills.json 后会让「目录技能」在去重时优先于本条 flat 记录）。
            skills = self._load_persisted()
            if not any(s.name == name for s in skills):
                skills.append(Skill(
                    name=name,
                    keywords=list(keywords),
                    prompt=(
                        f"A saved skill '{name}' matches this task. "
                        f"Call skill_load with path=\"{dir_name}\" to read its "
                        f"instructions, then follow them."
                    ),
                ))
                self._store.set("skills", [asdict(s) for s in skills])

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
        kws = sorted({w for w in _keywords_of(text) if w not in _STOPWORDS})
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
            raw = self._store.get("requests")
            requests = dict(raw) if isinstance(raw, dict) else {}
            requests[key] = requests.get(key, 0) + 1
            # 请求计数表只增不删：长期使用的 skills.json 会无界膨胀（既占内存
            # 也让每次 set() 的 JSON 序列化越来越大）。超过上限时淘汰最旧的
            # 一半条目（dict 保持插入顺序），但必须跳过当前 key——否则刚 +1
            # 的计数会被立刻删掉，返回值与持久化状态不一致（下次从 1 重新计）。
            if len(requests) > _MAX_TRACKED_REQUESTS:
                evict = _MAX_TRACKED_REQUESTS // 2
                for k in list(requests):
                    if evict <= 0:
                        break
                    if k == key:
                        continue
                    requests.pop(k, None)
                    evict -= 1
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

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
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INDEX_FILENAME = "_index.json"
_SKILL_META_FILENAME = "skill.json"
_SKILL_MD_FILENAME = "SKILL.md"
_DEFAULT_ENTRY = "prompt.md"
_SCAN_INTERVAL = 300  # seconds between automatic scans (5 min)
# entry 文件读取上限（字符）：内容会原样进入模型上下文。
_MAX_ENTRY_CHARS = 200_000

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

    def __init__(self, skills_dir: str, extra_dirs=()):
        self._dir = skills_dir
        # 额外技能根（如 <ws>/skills）。两套目录布局用同一个关键词索引覆盖，
        # 否则新体系的技能只能被语义检索命中、关键词检索和自动注入都拿不到。
        self._extra_dirs = [
            d for d in extra_dirs
            if d and os.path.realpath(d) != os.path.realpath(skills_dir)
        ]
        self._roots = [skills_dir] + self._extra_dirs
        # workspace 根：用于把额外根里的技能映射成跨根唯一的 path。
        self._ws = os.path.dirname(os.path.dirname(os.path.abspath(skills_dir)))
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
            if not isinstance(data, dict):
                raise ValueError("index root is not an object")
            skills = data.get("skills", [])
            self._index = skills if isinstance(skills, list) else []
            ts = data.get("updated_at_ts", 0)
            self._last_scan = ts if isinstance(ts, (int, float)) else 0
            # Sync _last_dir_mtime so _should_scan() doesn't immediately
            # trigger a rescan right after loading a fresh index.
            self._last_dir_mtime = self._dir_mtime()
        except (OSError, json.JSONDecodeError, ValueError, TypeError, AttributeError) as e:
            # 手工写坏 / 版本不兼容的 _index.json 不能让技能检索整体失效：
            # 记日志后按空索引处理，下次 scan() 会重建。
            logger.warning("skill_index: failed to load %s: %s", self._index_path, e)
            self._index = []
            # 关键：同时把 _last_scan 前移。否则它停留在 0，_should_scan() 恒
            # 为真——每次 search()/list_all() 都全量扫描目录并重写一个下次
            # 仍会加载失败的索引文件（磁盘写放大 + 查询延迟）。
            self._last_scan = time.time()

    def _save(self) -> None:
        """Atomic write of the index.

        临时文件用唯一名（而非常量的 ``.tmp``）：并发写入会互相覆盖同一个
        临时文件，产出截断/混杂的 JSON。异常路径还要清理临时文件，否则每次
        失败就在技能目录里留一个垃圾文件。
        """
        tmp = None
        try:
            os.makedirs(self._dir, exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                prefix=".index_", suffix=".tmp", dir=self._dir
            )
            payload = {
                "version": 1,
                "updated_at": _now_iso(),
                "updated_at_ts": time.time(),
                "skills": self._index,
            }
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._index_path)
            tmp = None
        except (OSError, TypeError, ValueError) as e:
            logger.warning("skill_index: failed to save %s: %s", self._index_path, e)
        finally:
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

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
        """Most recent mtime of any skill entry file across all roots."""
        entry_names = {_SKILL_META_FILENAME, _SKILL_MD_FILENAME, "prompt.md"}
        max_mtime = 0.0
        for root in self._roots:
            if not os.path.isdir(root):
                continue
            try:
                for dirpath, dirnames, filenames in os.walk(root):
                    dirnames[:] = [d for d in dirnames
                                   if not d.startswith(("_", "."))]
                    try:
                        mtime = os.stat(dirpath).st_mtime
                        if mtime > max_mtime:
                            max_mtime = mtime
                    except OSError:
                        pass
                    for name in entry_names & set(filenames):
                        try:
                            mtime = os.stat(os.path.join(dirpath, name)).st_mtime
                            if mtime > max_mtime:
                                max_mtime = mtime
                        except OSError:
                            pass
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

            new_index: List[Dict[str, Any]] = []
            seen_paths: set = set()
            for root in self._roots:
                if not os.path.isdir(root):
                    continue
                try:
                    entries = list(os.scandir(root))
                except OSError:
                    continue
                for entry in entries:
                    if not entry.is_dir():
                        continue
                    if entry.name.startswith(("_", ".")):
                        continue
                    record = self._read_dir_meta(root, entry.name, entry.path)
                    if record and record["path"] not in seen_paths:
                        seen_paths.add(record["path"])
                        new_index.append(record)

            self._index = new_index
            self._last_scan = time.time()
            self._last_dir_mtime = self._dir_mtime()
            self._save()
            logger.info("skill_index: indexed %d skills from %s",
                        len(new_index), self._roots)
            return len(new_index)

    def _read_dir_meta(self, root: str, dirname: str,
                       skill_dir: str) -> Optional[Dict[str, Any]]:
        """Read one skill directory, supporting both on-disk layouts.

        * legacy — ``skill.json`` (+ ``prompt.md``);
        * new — ``SKILL.md`` with YAML frontmatter.

        ``path`` is the bare directory name inside the primary root (backwards
        compatible) and workspace-relative inside extra roots (``skills/x``),
        so ids stay unique across roots.
        """
        meta_path = os.path.join(skill_dir, _SKILL_META_FILENAME)
        if os.path.isfile(meta_path):
            record = self._read_skill_meta(dirname, skill_dir, meta_path)
            if record is None:
                return None
            if os.path.realpath(root) != os.path.realpath(self._dir):
                record["path"] = self._ws_rel(skill_dir)
            return record
        md_path = os.path.join(skill_dir, _SKILL_MD_FILENAME)
        if os.path.isfile(md_path):
            return self._read_skill_md(skill_dir, md_path)
        return None

    def _ws_rel(self, skill_dir: str) -> str:
        """Workspace-relative id for a skill dir (falls back to abs path)."""
        try:
            rel = os.path.relpath(skill_dir, self._ws)
        except ValueError:
            return skill_dir
        return rel if not rel.startswith("..") else skill_dir

    def _read_skill_md(self, skill_dir: str, md_path: str) -> Optional[Dict[str, Any]]:
        """Build an index record from a SKILL.md's YAML frontmatter."""
        try:
            with open(md_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError as e:
            logger.warning("skill_index: failed to read %s: %s", md_path, e)
            return None
        m = re.match(r"\A---\s*\n(.*?)\n---\s*\n?", text, re.DOTALL)
        meta: Dict[str, Any] = {}
        if m:
            for line in m.group(1).splitlines():
                if ":" not in line:
                    continue
                key, _, value = line.partition(":")
                key = key.strip().lower()
                value = value.strip().strip('"').strip("'")
                if key == "name":
                    meta["name"] = value
                elif key == "description":
                    meta["description"] = value
                elif key in ("tags", "keywords"):
                    inner = value.strip().lstrip("[").rstrip("]")
                    meta["tags"] = [t.strip().strip('"').strip("'")
                                    for t in inner.split(",") if t.strip()]
        name = str(meta.get("name") or "").strip()
        if not name:
            return None
        keywords = [str(k).lower() for k in (meta.get("keywords") or [])]
        tags = [str(t).lower() for t in (meta.get("tags") or [])]
        desc = str(meta.get("description") or "").strip()
        return {
            "name": name,
            "description": desc,
            "keywords": keywords or tags,
            "tags": tags or keywords,
            "path": self._ws_rel(skill_dir),
            "entry": _SKILL_MD_FILENAME,
            "updated_at": _now_iso(),
        }

    def _read_skill_meta(
        self, dirname: str, skill_dir: str, meta_path: str
    ) -> Optional[Dict[str, Any]]:
        """Read skill.json from a skill directory."""
        try:
            # errors="replace"：GBK 等非 UTF-8 编码的 skill.json 抛
            # UnicodeDecodeError（ValueError 分支，JSONDecodeError 捕不到），
            # 会从 scan() 冒泡到 search() 整体失败——_index 来不及更新，
            # 之后每次 search 都重新扫描、重新失败，全部技能永久不可用。
            # 损坏字节替换后走正常 JSON 解析，坏名/坏类型由下面的归一兜底。
            with open(meta_path, "r", encoding="utf-8", errors="replace") as f:
                meta = json.load(f)
            if not isinstance(meta, dict):
                return None
            # 类型归一：手工编辑的 skill.json 里 "name": 123 / "keywords": 123 /
            # tags 里混非字符串元素，都会让下游在 strip()/lower()/迭代处抛
            # AttributeError/TypeError——而这里只捕获 OSError/JSONDecodeError，
            # 异常会冒泡出 scan() → search() 整个失败，且 _index 来不及更新，
            # 之后每次 search 都重新扫描、重新失败（全部技能永久不可用）。
            raw_name = meta.get("name", "")
            name = raw_name.strip() if isinstance(raw_name, str) else ""
            if not name:
                return None
            entry = meta.get("entry", _DEFAULT_ENTRY)
            if not isinstance(entry, str) or not entry:
                entry = _DEFAULT_ENTRY
            elif os.path.isabs(entry) or ".." in entry.split("/"):
                # 绝对路径 / 越界路径会让 os.path.join 丢弃 skill_dir 指向
                # skills 目录外的文件。
                logger.warning("skill_index: skill '%s' has unsafe entry path: %s", dirname, entry)
                return None
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

            def _str_list(key: str) -> List[str]:
                raw = meta.get(key, [])
                if not isinstance(raw, list):
                    return []
                return [str(x) for x in raw if isinstance(x, (str, int, float))]

            description = meta.get("description", "")
            return {
                "name": name,
                "description": description if isinstance(description, str) else str(description),
                "keywords": [k.lower() for k in _str_list("keywords")],
                "tags": _str_list("tags"),
                "path": dirname,
                "entry": entry,
            }
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
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

        # 在锁内取快照：scan() 结束后另一个线程可能立刻再次扫描并替换
        # self._index，直接遍历会在迭代过程中被改写。
        with self._lock:
            index = list(self._index)

        if not index or not query.strip():
            return []

        query_tokens = set(_tokenize(query.lower()))
        if not query_tokens:
            return []

        scored = []
        for skill in index:
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
        """Resolve parts under a skill root, rejecting path traversal.

        Accepts the several ``path`` spellings the index and its callers use:
        the bare directory name inside the primary root (``docx-convert``), a
        workspace-relative id (``skills/pdf-x`` / ``.agent/skills/docx-c``), or
        an absolute path.  Returns the absolute path only when it lands inside
        one of the registered roots — the LLM supplies these strings, so
        ``path="../../../../etc"`` and ``entry="../../../secret"`` must not
        resolve.
        """
        cleaned = [str(p) for p in parts if p]
        if not cleaned:
            return None
        head = cleaned[0].replace("\\", "/")
        # 只去掉开头的 "./" 段：lstrip("./") 会把 ".agent/..." 的点一起吞掉，
        # 使 workspace 相对 id 解析成错误的路径。
        rel = re.sub(r"^(?:\./)+", "", head)
        bases: List[str] = []
        if os.path.isabs(rel):
            bases.append(os.path.realpath(rel))
        else:
            for root in self._roots:
                bases.append(os.path.realpath(os.path.join(root, rel)))
            if self._ws:
                bases.append(os.path.realpath(os.path.join(self._ws, rel)))
        contained: List[str] = []
        for base in bases:
            resolved = os.path.realpath(os.path.join(base, *cleaned[1:]))
            # 必须严格位于某个技能根之内（等于根本身视为越界）。
            for root in self._roots:
                root_real = os.path.realpath(root)
                if resolved.startswith(root_real + os.sep):
                    contained.append(resolved)
                    break
        if not contained:
            return None
        # 同一 id 在多个根下都"合法"时，优先返回真实存在的那个（例如
        # "skills/pdf-x" 在 <ws>/.agent/skills 下也有个合法的拼接口径，但那里
        # 并没有这个目录）。都不存在时返回第一个，让调用方的 open() 走它
        # 原有的"读取失败"分支，而不是被误判成路径穿越。
        for resolved in contained:
            if os.path.exists(resolved):
                return resolved
        return contained[0]

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
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                # 上限读取：内容会原样进模型上下文，一个误放进 skills 目录的
                # 大文件会被整个读进内存并撑爆上下文。errors="replace"：GBK
                # 的 prompt.md 旧实现直接抛 UnicodeDecodeError（except OSError
                # 捕不到），skill_load 对该技能永久失败。
                text = f.read(_MAX_ENTRY_CHARS + 1)
            if len(text) > _MAX_ENTRY_CHARS:
                text = text[:_MAX_ENTRY_CHARS] + (
                    f"\n… [文件过大，已截断到 {_MAX_ENTRY_CHARS} 字符]"
                )
            return text
        except (OSError, UnicodeDecodeError) as e:
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

_TO_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z]+|[0-9]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


def _tokenize(text: str) -> List[str]:
    """按字script切分：CJK 串、拉丁词、数字串各自独立成 token。

    ``\\w+`` 会把「帮我提取pdf里的表格」整个当成一个 token（CJK 与拉丁都是
    ``\\w``），于是技能关键词里的 ``pdf`` / ``docx`` / ``api`` 永远匹配不上——
    而真实意图几乎全是中文夹英文术语，关键词检索会静默失效。CJK 串额外产出
    整段与二元组，让「提取」能命中「提取pdf表格」。
    """
    toks: List[str] = []
    for run in _TO_TOKEN_RE.findall(text):
        toks.append(run.lower())
        if _CJK_RUN_RE.fullmatch(run):
            for i in range(len(run) - 1):
                toks.append(run[i:i + 2])
    return toks

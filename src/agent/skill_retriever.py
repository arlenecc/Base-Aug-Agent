"""Skill retriever: hybrid (vector + BM25) retrieval over SKILL.md files.

Scans the ``skills`` directory (recursively) for ``SKILL.md`` files, parses
their YAML frontmatter (``name`` / ``description`` / ``tags``) and body
sections (``When to Use This Skill``, ``When NOT to Use``, examples), then
builds a dedicated LanceDB table for retrieval.

Retrieval combines:
  * vector search over the ``description`` embedding and the ``examples``
    embedding, fused 0.6 : 0.4;
  * BM25 full-text search over a combined ``name + tags + description +
    examples`` text column.

The union of vector candidates (3) and BM25 candidates (3) is de-duplicated,
filtered by a minimum similarity of 0.5, then the remaining candidates'
``name + tags + description`` are handed to the LLM for a final pick.

This module is independent of the RAG ``VectorStore`` (which stores document
chunks); it uses its own LanceDB table and the shared FastEmbed embedding
function (nomic-embed-text-v1.5).
"""
from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SKILL_FILENAME = "SKILL.md"
_SKILL_TABLE_NAME = "skills"

# Similarity fusion weights for description vs examples embeddings.
_DESC_WEIGHT = 0.6
_EXAMPLES_WEIGHT = 0.4
# Minimum similarity for a candidate to survive filtering.
_MIN_SIMILARITY = 0.5
# Number of candidates from each channel (vector / BM25).
_VECTOR_TOP_K = 3
_BM25_TOP_K = 3
# Max chars of description/examples kept for embedding and combined text.
# SKILL.md body sections (especially "Examples") can be tens of KB; embedding
# them whole is wasteful (the model truncates at ~2k tokens anyway) and makes
# jieba tokenization + LanceDB storage balloon. Cap to keep memory O(1) per skill.
_MAX_DESC_CHARS = 2000
_MAX_EXAMPLES_CHARS = 3000
_MAX_COMBINED_CHARS = 6000

# Directories to skip while scanning (tests, caches, etc.).
_SKIP_DIR_NAMES = {"tests", "test", "__pycache__", ".git", ".DS_Store", "node_modules"}

# Body section headings that describe "when to use / when NOT to use".
_WHEN_TO_USE_HEADINGS = (
    "when to use this skill",
    "when to use",
    "use this skill when",
    "when you should use",
)
_WHEN_NOT_TO_USE_HEADINGS = (
    "when not to use",
    "when not to use this skill",
    "do not use",
    "don't use",
)
_EXAMPLES_HEADINGS = (
    "examples",
    "example",
    "usage examples",
)


# ---------------------------------------------------------------------------
# SKILL.md parsing
# ---------------------------------------------------------------------------

def _parse_frontmatter(text: str) -> Dict[str, Any]:
    """Parse the YAML frontmatter (``---`` delimited block) at the top.

    The frontmatter is simple key/value YAML; we parse top-level scalar fields
    (``name``, ``description``, ``license``, ``compatibility``,
    ``allowed-tools``) and nested ``metadata`` (version, author), plus a
    ``tags`` list if present.  Returns a flat dict.
    """
    result: Dict[str, Any] = {}
    if not text.startswith("---"):
        return result

    end = text.find("\n---", 3)
    if end == -1:
        return result
    fm_text = text[3:end]

    tags: List[str] = []
    for raw in fm_text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        # Nested metadata block — keep only version/author.
        m = re.match(r"^\s*(\w[\w-]*)\s*:\s*(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if key == "metadata":
            # nested block: skip; we don't need version/author for retrieval
            continue
        # strip surrounding quotes
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key == "tags":
            tags.extend(_parse_list_field(val))
        elif key in ("name", "description", "license", "compatibility", "allowed-tools"):
            result[key] = val
    if tags:
        result["tags"] = tags
    return result


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to ``max_chars`` (no-op if shorter)."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _normalize_vec(vec: List[float]) -> List[float]:
    """Normalize a vector to unit length (for cosine similarity via L2 distance)."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return [float(x) for x in vec]
    return [float(x) / norm for x in vec]


def _parse_list_field(val: str) -> List[str]:
    """Parse a YAML list field value into a list of strings.

    Handles both flow style (``[a, b, c]``) and bare comma-separated strings
    (``a, b``), stripping brackets, quotes, and empty entries.
    """
    s = val.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    items = re.split(r"[,，]", s)
    out = []
    for it in items:
        it = it.strip().strip("\"'")
        if it:
            out.append(it)
    return out


def _extract_section(text: str, headings: Tuple[str, ...]) -> str:
    """Extract the body text under the first matching ``##`` heading.

    Returns the section text (until the next ``##`` heading), or "" if the
    heading is absent.
    """
    lines = text.splitlines()
    capture = False
    out: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            title = stripped[3:].strip().lower()
            if capture:
                break  # reached the next section
            if title in headings:
                capture = True
                continue
        if capture:
            out.append(line)
    return "\n".join(out).strip()


class SkillMetadata:
    """Parsed metadata for a single SKILL.md."""

    __slots__ = ("name", "description", "tags", "dir", "when_to_use",
                 "when_not_to_use", "examples", "combined_text")

    def __init__(self, name: str, description: str, tags: List[str], dir: str,
                 when_to_use: str, when_not_to_use: str, examples: str):
        self.name = name
        self.description = description
        self.tags = tags
        self.dir = dir
        self.when_to_use = when_to_use
        self.when_not_to_use = when_not_to_use
        self.examples = examples
        self.combined_text = self._build_combined()

    def _build_combined(self) -> str:
        parts = [self.name]
        if self.tags:
            parts.append(" ".join(self.tags))
        if self.description:
            parts.append(_truncate(self.description, _MAX_DESC_CHARS))
        if self.when_to_use:
            parts.append(_truncate(self.when_to_use, _MAX_EXAMPLES_CHARS))
        if self.examples:
            parts.append(_truncate(self.examples, _MAX_EXAMPLES_CHARS))
        return _truncate("\n".join(parts), _MAX_COMBINED_CHARS)

    def summary_text(self) -> str:
        """Short text (name + tags + description) handed to the LLM for final pick."""
        tag_str = ", ".join(self.tags) if self.tags else ""
        return f"name: {self.name}\ntags: {tag_str}\ndescription: {self.description}".strip()


def parse_skill_md(text: str, dir: str) -> Optional[SkillMetadata]:
    """Parse a SKILL.md file's text into a :class:`SkillMetadata`.

    Returns None if the file has no ``name`` (not a valid skill).
    """
    fm = _parse_frontmatter(text)
    name = (fm.get("name") or "").strip()
    if not name:
        return None
    description = (fm.get("description") or "").strip()
    tags = fm.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in re.split(r"[,，]", tags) if t.strip()]

    # description may already contain "when to use / when not to use" clauses;
    # keep it whole. Body sections augment it.
    when_to_use = _extract_section(text, _WHEN_TO_USE_HEADINGS)
    when_not_to_use = _extract_section(text, _WHEN_NOT_TO_USE_HEADINGS)
    examples = _extract_section(text, _EXAMPLES_HEADINGS)

    return SkillMetadata(
        name=name,
        description=description,
        tags=tags,
        dir=dir,
        when_to_use=when_to_use,
        when_not_to_use=when_not_to_use,
        examples=examples,
    )


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class SkillScanner:
    """Recursively scan a skills directory for SKILL.md files.

    Thread-safe; debounced like SkillIndex — a rescan is triggered when the
    directory mtime changes or the interval elapses.
    """

    def __init__(self, skills_dir: str, scan_interval: float = 300.0):
        self._dir = skills_dir
        self._scan_interval = scan_interval
        self._lock = threading.Lock()
        self._skills: List[SkillMetadata] = []
        self._last_scan = 0.0
        self._last_mtime = 0.0

    def _dir_mtime(self) -> float:
        max_mtime = 0.0
        if not os.path.isdir(self._dir):
            return 0.0
        try:
            for root, dirs, files in os.walk(self._dir):
                dirs[:] = [d for d in dirs if d not in _SKIP_DIR_NAMES and not d.startswith(".")]
                for fn in files:
                    try:
                        mtime = os.stat(os.path.join(root, fn)).st_mtime
                        if mtime > max_mtime:
                            max_mtime = mtime
                    except OSError:
                        pass
        except OSError:
            pass
        return max_mtime

    def _should_scan(self) -> bool:
        if self._last_scan == 0.0:
            return True
        if time.time() - self._last_scan > self._scan_interval:
            return True
        try:
            if self._dir_mtime() != self._last_mtime:
                return True
        except OSError:
            pass
        return False

    def scan(self, force: bool = False) -> List[SkillMetadata]:
        """Scan the skills directory and return the current skill list."""
        with self._lock:
            if not force and not self._should_scan():
                return list(self._skills)
            skills: List[SkillMetadata] = []
            if os.path.isdir(self._dir):
                for root, dirs, files in os.walk(self._dir):
                    dirs[:] = [d for d in dirs if d not in _SKIP_DIR_NAMES and not d.startswith(".")]
                    if _SKILL_FILENAME not in files:
                        continue
                    path = os.path.join(root, _SKILL_FILENAME)
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            content = f.read()
                    except (OSError, UnicodeDecodeError) as e:
                        logger.warning("skill_retriever: failed to read %s: %s", path, e)
                        continue
                    # dir is relative to the skills root, or absolute if not under it.
                    rel = os.path.relpath(root, self._dir) if root.startswith(self._dir) else root
                    meta = parse_skill_md(content, rel)
                    if meta:
                        skills.append(meta)
            self._skills = skills
            self._last_scan = time.time()
            self._last_mtime = self._dir_mtime()
            logger.info("skill_retriever: scanned %d skills from %s", len(skills), self._dir)
            return list(self._skills)

    def list_skills(self) -> List[SkillMetadata]:
        return self.scan()


# ---------------------------------------------------------------------------
# Vector store (LanceDB) for skills
# ---------------------------------------------------------------------------

class SkillVectorStore:
    """LanceDB-backed store for skill embeddings + BM25 text.

    Columns:
      * ``dir`` (str)            — SKILL.md's directory (relative)
      * ``name`` (str)           — skill name
      * ``tags`` (str)           — comma-joined tags
      * ``description`` (str)    — frontmatter description
      * ``desc_vec`` (vector)    — description embedding
      * ``examples_vec`` (vector)— examples embedding
      * ``combined`` (str)       — name+tags+description+examples combined text
      * ``combined_fts`` (str)   — jieba-tokenized combined (for BM25)
    """

    def __init__(self, db_path: str, embedding_function=None):
        self._db_path = db_path
        self._ef = embedding_function
        self._db = None
        self._table = None
        self._lock = threading.Lock()

    def _get_ef(self):
        if self._ef is None:
            from .rag.vector_store import get_or_create_embedding_function
            self._ef = get_or_create_embedding_function()
        return self._ef

    def _get_db(self):
        if self._db is None:
            import lancedb
            os.makedirs(self._db_path, exist_ok=True)
            self._db = lancedb.connect(self._db_path)
        return self._db

    def _get_table(self):
        db = self._get_db()
        if self._table is None:
            if _SKILL_TABLE_NAME in db.table_names():
                self._table = db.open_table(_SKILL_TABLE_NAME)
            else:
                self._table = None
        return self._table

    def _embed(self, texts: List[str]) -> List[List[float]]:
        ef = self._get_ef()
        return ef.embed_documents(texts)

    def build(self, skills: List[SkillMetadata], batch_size: int = 100) -> int:
        """Build/rebuild the skills table from the given skill list.

        Drops and recreates the table, then processes skills in **batches**:
        embed ``batch_size`` texts per forward pass and write each batch to
        LanceDB immediately.  This keeps peak memory at O(batch_size) instead
        of O(total) — critical for thousands of skills (the previous
        whole-list approach ballooned memory to tens of GB and froze the OS).
        """
        from .rag.vector_store import _tokenize_for_fts

        if not skills:
            return 0

        db = self._get_db()
        with self._lock:
            # Drop old table if present.
            if _SKILL_TABLE_NAME in db.table_names():
                db.drop_table(_SKILL_TABLE_NAME)
            self._table = None

            # Probe the embedding dimension with a single text first.
            probe = self._embed([skills[0].description or skills[0].name])
            dim = len(probe[0]) if probe else 0
            if dim <= 0:
                return 0

            import pyarrow as pa
            vec_type = pa.list_(pa.float32(), dim)
            schema = pa.schema([
                pa.field("dir", pa.string()),
                pa.field("name", pa.string()),
                pa.field("tags", pa.string()),
                pa.field("description", pa.string()),
                pa.field("desc_vec", vec_type),
                pa.field("examples_vec", vec_type),
                pa.field("combined", pa.string()),
                pa.field("combined_fts", pa.string()),
            ])

            table = db.create_table(_SKILL_TABLE_NAME, [], schema=schema)
            total = 0
            for i in range(0, len(skills), batch_size):
                batch = skills[i:i + batch_size]
                # 截断后再 embedding，避免超长 SKILL.md 导致内存/耗时爆炸。
                descriptions = [_truncate(s.description or s.name, _MAX_DESC_CHARS) for s in batch]
                examples = [
                    _truncate(s.examples or s.when_to_use or s.description or s.name, _MAX_EXAMPLES_CHARS)
                    for s in batch
                ]
                desc_vecs = self._embed(descriptions)
                examples_vecs = self._embed(examples)

                rows = []
                for s, dv, ev in zip(batch, desc_vecs, examples_vecs):
                    rows.append({
                        "dir": s.dir,
                        "name": s.name,
                        "tags": ", ".join(s.tags),
                        "description": s.description,
                        "desc_vec": _normalize_vec(dv),
                        "examples_vec": _normalize_vec(ev),
                        "combined": s.combined_text,
                        "combined_fts": _tokenize_for_fts(s.combined_text),
                    })
                table.add(rows)
                total += len(rows)
                logger.info(
                    "skill_retriever: built %d/%d skills",
                    total, len(skills),
                )

            self._table = table
            self._ensure_fts_index(self._table)
            logger.info("skill_retriever: built skills table with %d rows", total)
            return total

    def _ensure_fts_index(self, table) -> None:
        """Create a BM25 full-text index on the ``combined_fts`` column."""
        try:
            from lancedb.index import FTS
            table.create_index("combined_fts", config=FTS())
        except Exception:
            # lancedb 0.25.x fallback
            try:
                table.create_fts_index("combined_fts", use_tantivy=False)
            except Exception as e:
                logger.warning("skill_retriever: BM25 index failed (fallback to vector only): %s", e)

    def search_vector(self, query: str, top_k: int = _VECTOR_TOP_K) -> List[Dict[str, Any]]:
        """Vector search. Returns [{dir, name, tags, description, score}]."""
        table = self._get_table()
        if table is None:
            return []
        ef = self._get_ef()
        qv = ef.embed_query(query)
        # 归一化查询向量，与建表时的归一化向量保持一致（L2 距离转余弦相似度）。
        norm = math.sqrt(sum(x * x for x in qv))
        if norm > 0:
            qv = [x / norm for x in qv]
        try:
            desc_res = table.search(qv, vector_column_name="desc_vec").limit(top_k).to_list()
        except Exception as e:
            logger.warning("skill_retriever: desc vector search failed: %s", e)
            desc_res = []
        try:
            ex_res = table.search(qv, vector_column_name="examples_vec").limit(top_k).to_list()
        except Exception as e:
            logger.warning("skill_retriever: examples vector search failed: %s", e)
            ex_res = []

        # Fuse the two vector results by skill dir, weighting 0.6:0.4.
        fused: Dict[str, Dict[str, Any]] = {}
        for r in desc_res:
            d = r.get("dir", "")
            if not d:
                continue
            fused[d] = self._row_to_dict(r)
        for r in ex_res:
            d = r.get("dir", "")
            if not d:
                continue
            ex_score = self._distance_to_similarity(r.get("_distance"))
            if d in fused:
                # Both channels hit the same skill: weighted fusion.
                desc_score = fused[d]["score"]
                fused[d]["score"] = _DESC_WEIGHT * desc_score + _EXAMPLES_WEIGHT * ex_score
            else:
                entry = self._row_to_dict(r)
                entry["score"] = ex_score
                fused[d] = entry

        results = list(fused.values())
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    @staticmethod
    def _distance_to_similarity(distance) -> float:
        """Convert a normalized L2 distance to a cosine similarity in [0,1].

        归一化后 ||a-b||² = 2 - 2·cos(θ)，故 cos(θ) = 1 - d²/2。
        """
        d = float(distance if distance is not None else 0.0)
        return max(0.0, 1.0 - (d * d) / 2.0)

    def _row_to_dict(self, r: Dict[str, Any]) -> Dict[str, Any]:
        score = self._distance_to_similarity(r.get("_distance"))
        return {
            "dir": r.get("dir", ""),
            "name": r.get("name", ""),
            "tags": r.get("tags", ""),
            "description": r.get("description", ""),
            "score": score,
        }

    def search_bm25(self, query: str, top_k: int = _BM25_TOP_K) -> List[Dict[str, Any]]:
        """BM25 full-text search over the combined text. Returns dict rows."""
        table = self._get_table()
        if table is None:
            return []
        from .rag.vector_store import _tokenize_for_fts
        tokenized = _tokenize_for_fts(query)
        try:
            results = (
                table.search(tokenized, query_type="fts", fts_columns=["combined_fts"])
                .limit(top_k)
                .to_list()
            )
        except Exception as e:
            logger.warning("skill_retriever: BM25 search failed: %s", e)
            return []
        out = []
        for r in results:
            out.append({
                "dir": r.get("dir", ""),
                "name": r.get("name", ""),
                "tags": r.get("tags", ""),
                "description": r.get("description", ""),
                "bm25_score": float(r.get("_score", 0.0)),
            })
        return out

    def close(self) -> None:
        if self._table is not None:
            try:
                self._table = None
            except Exception:
                pass
        if self._db is not None:
            try:
                self._db = None
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Retriever (hybrid search + LLM final pick)
# ---------------------------------------------------------------------------

class SkillRetriever:
    """High-level hybrid skill retriever.

    Orchestrates scanning, indexing, hybrid retrieval, threshold filtering,
    and (optionally) an LLM final pick among candidates.
    """

    def __init__(self, skills_dir: str, db_path: str, embedding_function=None):
        self._scanner = SkillScanner(skills_dir)
        self._store = SkillVectorStore(db_path, embedding_function=embedding_function)
        self._skills_dir = skills_dir

    def ensure_indexed(self) -> None:
        """(Re)build the vector table if the skills dir changed or table is empty."""
        skills = self._scanner.scan()
        table = self._store._get_table()
        if table is None:
            self._store.build(skills)

    def retrieve(self, query: str, llm=None) -> Dict[str, Any]:
        """Hybrid retrieve for a query.

        Returns:
            {"candidates": [...], "picked": <dir|None>, "reason": str}
        ``picked`` is set when ``llm`` is provided and it selects a skill;
        otherwise ``picked`` is None and callers use ``candidates`` directly.

        Fusion semantics:
          * Vector results carry a cosine-similarity ``score`` in [0,1] (the
            description/examples channels already fused 0.6:0.4).
          * BM25 results carry a raw ``bm25_score``; for a unified, comparable
            ranking we also attach a cosine similarity as ``score`` (reused from
            the vector channel when the same skill matched there, otherwise
            looked up on demand).
          * De-duplicated by ``dir``; candidates with ``score < 0.5`` dropped.
        """
        self.ensure_indexed()

        vec_results = self._store.search_vector(query, top_k=_VECTOR_TOP_K)
        bm25_results = self._store.search_bm25(query, top_k=_BM25_TOP_K)

        # De-duplicate by dir. Vector channel is the authoritative score.
        merged: Dict[str, Dict[str, Any]] = {}
        for r in vec_results:
            d = r.get("dir", "")
            if not d:
                continue
            merged[d] = dict(r)

        # BM25 results: keep raw bm25 score; attach a vector similarity as the
        # unified ``score``. If the skill already matched in the vector channel,
        # reuse that score; otherwise look it up via a vector search.
        bm25_by_dir = {r.get("dir", ""): r for r in bm25_results if r.get("dir")}
        for d, bm in bm25_by_dir.items():
            if d in merged:
                merged[d]["bm25_score"] = bm["bm25_score"]
            else:
                sim = self._lookup_similarity(query, d)
                merged[d] = dict(bm)
                merged[d]["bm25_score"] = bm["bm25_score"]
                merged[d]["score"] = sim

        candidates = list(merged.values())
        # Filter by minimum similarity (vector similarity in [0,1]).
        candidates = [c for c in candidates if c["score"] >= _MIN_SIMILARITY]
        candidates.sort(key=lambda x: x["score"], reverse=True)

        if not candidates:
            return {"candidates": [], "picked": None, "reason": "no candidates"}

        picked = None
        reason = "candidates returned"
        if llm is not None:
            picked = self._llm_pick(llm, query, candidates)
            reason = "llm selected" if picked else "llm found no match"

        return {"candidates": candidates, "picked": picked, "reason": reason}

    def _lookup_similarity(self, query: str, dir: str) -> float:
        """Return the cosine similarity of ``query`` against skill ``dir``.

        Used to give BM25-only matches a comparable vector score.  Falls back
        to 0.0 if the skill can't be located or embedding fails.
        """
        table = self._store._get_table()
        if table is None:
            return 0.0
        ef = self._store._get_ef()
        try:
            qv = ef.embed_query(query)
            norm = math.sqrt(sum(x * x for x in qv))
            if norm > 0:
                qv = [x / norm for x in qv]
            res = (
                table.search(qv, vector_column_name="desc_vec")
                .where(f"dir = '{dir}'")
                .limit(1)
                .to_list()
            )
            if res:
                return self._store._distance_to_similarity(res[0].get("_distance"))
        except Exception as e:
            logger.debug("skill_retriever: lookup_similarity failed for %s: %s", dir, e)
        return 0.0

    def _llm_pick(self, llm, query: str, candidates: List[Dict[str, Any]]) -> Optional[str]:
        """Ask the LLM to pick the best skill from candidates.

        Returns the chosen skill's ``dir`` (or None if none match).
        """
        options = "\n\n".join(
            f"[{i}] dir={c['dir']}\n{c['description'][:300]}"
            for i, c in enumerate(candidates)
        )
        prompt = (
            "Given the task and candidate skills, pick the single most relevant "
            "skill, or reply 'none' if none fit.\n\n"
            f"Task: {query}\n\nCandidates:\n{options}\n\n"
            "Reply with only the candidate index number (e.g. '0'), or 'none'."
        )
        try:
            events = list(llm.chat_stream([{"role": "user", "content": prompt}]))
            answer = ""
            for ev in events:
                if ev.get("type") == "content":
                    answer += ev.get("content", "")
            answer = answer.strip()
        except Exception as e:
            logger.warning("skill_retriever: LLM pick failed: %s", e)
            return None

        m = re.search(r"\d+", answer)
        if m and "none" not in answer.lower():
            idx = int(m.group())
            if 0 <= idx < len(candidates):
                return candidates[idx]["dir"]
        return None

    def read_skill_dir(self, dir: str) -> Optional[str]:
        """Return the absolute path to a skill's directory (for reading support files)."""
        base = os.path.realpath(self._skills_dir)
        resolved = os.path.realpath(os.path.join(base, dir))
        if resolved == base or not resolved.startswith(base + os.sep):
            return None
        if os.path.isdir(resolved):
            return resolved
        return None

    def close(self) -> None:
        self._store.close()

"""Knowledge-graph long-term memory: entities, relations, observations.

Replaces the flat fact-list LongTermMemory with a structured graph that
mirrors the Memory MCP server model:

- Entity: a named node (person, project, concept, ...) with a type and
  a list of observations (free-text facts attached to the entity).
- Relation: a directed, labeled edge between two entities
  (source --[label]--> target).

All data is persisted in a single JSON file.  Observations are also
embedded via FastEmbed (nomic-embed-text-v1.5-Q) and indexed in LanceDB
so the agent can do semantic recall instead of keyword matching.

The graph is intentionally simple — no graph database, no query language.
Just dicts, lists, and a vector index.  This keeps the runtime dependency
surface flat while still giving the agent structured, searchable memory.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model (plain dicts; no dataclasses to keep JSON round-trip trivial)
# ---------------------------------------------------------------------------
#
#   entity  = {name, type, observations: [str], created_at}
#   relation = {source: name, target: name, label: str}
#
# The graph is keyed by entity *name* (case-insensitive).  Names must be
# unique — creating an entity whose name already exists merges observations.
# ---------------------------------------------------------------------------


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _norm_name(name: str) -> str:
    """Normalise an entity name for case-insensitive dedup."""
    return name.strip()


def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Graph store
# ---------------------------------------------------------------------------


class GraphMemoryStore:
    """Thread-safe knowledge-graph store with LanceDB semantic search.

    Persistence:
        <path>              — the graph JSON (entities + relations)
        <path>.vectors/     — LanceDB table for observation embeddings

    The vector index is *optional*: if LanceDB or FastEmbed is unavailable
    (e.g. dependencies not installed, model download failed), the store
    still works — ``search`` just falls back to keyword matching.
    """

    def __init__(self, path: str, embedding_model: str = "nomic-ai/nomic-embed-text-v1.5-Q"):
        self._path = path
        self._vector_dir = path + ".vectors"
        self._embedding_model_name = embedding_model
        self._lock = threading.Lock()
        self._vec_lock = threading.Lock()  # protects vector-index lazy init
        self._entities: Dict[str, dict] = {}  # name → entity
        self._relations: List[dict] = []
        self._db = None
        self._table = None
        self._ef = None
        self._vec_ready = False
        self._load()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._entities = {n.lower(): e for n, e in data.get("entities", {}).items()}
            # Normalise: ensure every entity has observations list
            for e in self._entities.values():
                if not isinstance(e.get("observations"), list):
                    e["observations"] = []
            self._relations = data.get("relations", [])
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("graph_memory: failed to load %s: %s", self._path, e)
            self._entities = {}
            self._relations = []

    def _save(self) -> None:
        """Atomic write. Must be called under _lock."""
        try:
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
            tmp = self._path + ".tmp"
            payload = {
                "entities": self._entities,
                "relations": self._relations,
            }
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._path)
        except OSError as e:
            logger.warning("graph_memory: failed to save %s: %s", self._path, e)

    # ------------------------------------------------------------------
    # vector index (lazy init)
    # ------------------------------------------------------------------

    def _ensure_vectors(self) -> None:
        """Lazy-init LanceDB + FastEmbed. Non-fatal on failure.

        Uses a separate _vec_lock with double-checked locking so that
        concurrent callers (e.g. agent thread + tool thread) don't
        race into loading the ONNX model twice.
        """
        if self._vec_ready:
            return
        with self._vec_lock:
            if self._vec_ready:  # double-check
                return
            try:
                import lancedb
                from .rag.vector_store import get_or_create_embedding_function
                os.makedirs(self._vector_dir, exist_ok=True)
                self._db = lancedb.connect(self._vector_dir)
                # Share the process-wide ONNX model with RAG's VectorStore
                # to avoid loading a second ~137MB copy.
                self._ef = get_or_create_embedding_function(self._embedding_model_name)
                if "observations" in self._db.table_names():
                    self._table = self._db.open_table("observations")
                self._vec_ready = True
            except Exception as e:
                # Non-fatal: semantic search just falls back to keyword.
                logger.debug("graph_memory: vector index unavailable: %s", e)
                self._vec_ready = False

    def _ensure_table(self):
        if self._table is not None:
            return self._table
        if self._db is not None and "observations" in self._db.table_names():
            self._table = self._db.open_table("observations")
        return self._table

    def _index_observation(self, entity_name: str, observation: str) -> None:
        """Embed and upsert a single observation into LanceDB.

        Called OUTSIDE self._lock — the embedding (ONNX C++ compute) and
        LanceDB write can take 50-500ms each.  Holding the graph lock during
        that time would block all JSON reads (snapshot, search fallback, etc.)
        and freeze the UI if this runs on the agent thread.
        """
        if not self._vec_ready:
            return
        try:
            table = self._ensure_table()
            oid = _hash(entity_name + "|" + observation)
            # Observations are documents (retrieval targets), so embed with
            # the document prefix — matching _semantic_search which embeds
            # the query with the query prefix.
            vec = self._ef.embed_documents([observation])[0]
            record = {
                "id": oid,
                "entity": entity_name,
                "text": observation,
                "vector": vec,
            }
            if table is None:
                with self._vec_lock:
                    # Re-check under lock to avoid race with concurrent create
                    table = self._table
                    if table is None:
                        self._table = self._db.create_table("observations", [record])
                        return
                    # Delete old record with same id if exists (upsert)
                    try:
                        table.delete(f"id = '{oid}'")
                    except Exception:
                        pass
                    table.add([record])
            else:
                # Delete old record with same id if exists (upsert)
                try:
                    table.delete(f"id = '{oid}'")
                except Exception:
                    pass
                table.add([record])
        except Exception as e:
            logger.debug("graph_memory: index_observation failed: %s", e)

    def _deindex_observations(self, entity_name: str) -> None:
        """Remove all vectors for an entity (e.g. after entity deletion)."""
        if not self._vec_ready:
            return
        try:
            table = self._ensure_table()
            if table is None:
                return
            # Escape single quotes (SQL literal) — entity names come from
            # the LLM and may contain them; otherwise the filter expression
            # would be malformed or injectable.
            safe = entity_name.replace("'", "''")
            table.delete(f"entity = '{safe}'")
        except Exception:
            pass

    def _index_observations_batch(self, entity_name: str, observations: List[str]) -> None:
        """Embed and upsert multiple observations in ONE batch.

        Called OUTSIDE self._lock.  Batching matters: each individual
        embed_query() costs ~10-50ms of ONNX compute, so adding N facts
        from a fact-extraction pass would take N× that.  embed_documents()
        processes the whole list in a single batched ONNX forward pass.
        """
        if not self._vec_ready or not observations:
            return
        try:
            vecs = self._ef.embed_documents(observations)
            records = [
                {
                    "id": _hash(entity_name + "|" + obs),
                    "entity": entity_name,
                    "text": obs,
                    "vector": vec,
                }
                for obs, vec in zip(observations, vecs)
            ]
            with self._vec_lock:
                table = self._ensure_table()
                if table is None:
                    self._table = self._db.create_table("observations", records)
                    return
                for rec in records:
                    try:
                        table.delete(f"id = '{rec['id']}'")
                    except Exception:
                        pass
                table.add(records)
        except Exception as e:
            logger.debug("graph_memory: index_observations_batch failed: %s", e)

    # ------------------------------------------------------------------
    # CRUD: entities
    # ------------------------------------------------------------------

    def create_entity(self, name: str, entity_type: str = "", observations: Optional[List[str]] = None) -> dict:
        """Create an entity, or merge observations if it already exists."""
        name = _norm_name(name)
        to_index: List[str] = []  # collected under lock, indexed outside
        with self._lock:
            key = name.lower()
            if key in self._entities:
                e = self._entities[key]
                if entity_type and not e.get("type"):
                    e["type"] = entity_type
                if observations:
                    for obs in observations:
                        obs = obs.strip()
                        if obs and obs not in e["observations"]:
                            e["observations"].append(obs)
                            to_index.append(obs)
            else:
                e = {
                    "name": name,
                    "type": entity_type,
                    "observations": [o.strip() for o in (observations or []) if o.strip()],
                    "created_at": _now(),
                }
                self._entities[key] = e
                to_index = list(e["observations"])
            self._save()
            result = dict(e)
        # Index outside lock (batched: one ONNX pass for all observations)
        if to_index:
            self._index_observations_batch(name, to_index)
        return result

    def delete_entity(self, name: str) -> bool:
        """Delete an entity and all its relations."""
        name = _norm_name(name)
        with self._lock:
            key = name.lower()
            if key not in self._entities:
                return False
            del self._entities[key]
            self._relations = [
                r for r in self._relations
                if r["source"].lower() != key and r["target"].lower() != key
            ]
            self._deindex_observations(name)
            self._save()
            return True

    def get_entity(self, name: str) -> Optional[dict]:
        key = _norm_name(name).lower()
        with self._lock:
            e = self._entities.get(key)
            return dict(e) if e else None

    def list_entities(self) -> List[dict]:
        with self._lock:
            return [dict(e) for e in self._entities.values()]

    # ------------------------------------------------------------------
    # CRUD: observations
    # ------------------------------------------------------------------

    def add_observations(self, name: str, observations: List[str]) -> int:
        """Add observations to an entity. Creates the entity if missing."""
        name = _norm_name(name)
        cleaned = [o.strip() for o in observations if o and o.strip()]
        if not cleaned:
            return 0
        to_index: List[str] = []
        with self._lock:
            key = name.lower()
            if key not in self._entities:
                # create entity lazily so callers don't need a separate step
                self._entities[key] = {
                    "name": name,
                    "type": "",
                    "observations": [],
                    "created_at": _now(),
                }
            e = self._entities[key]
            for obs in cleaned:
                if obs not in e["observations"]:
                    e["observations"].append(obs)
                    to_index.append(obs)
            if to_index:
                self._save()
        # Index outside lock (batched: one ONNX pass for all observations)
        if to_index:
            self._index_observations_batch(name, to_index)
        return len(to_index)

    # ------------------------------------------------------------------
    # CRUD: relations
    # ------------------------------------------------------------------

    def create_relation(self, source: str, target: str, label: str) -> bool:
        """Create a directed edge source --[label]--> target."""
        source = _norm_name(source)
        target = _norm_name(target)
        label = label.strip()
        if not source or not target or not label:
            return False
        with self._lock:
            # Ensure both entities exist
            for n in (source, target):
                k = n.lower()
                if k not in self._entities:
                    self._entities[k] = {
                        "name": n,
                        "type": "",
                        "observations": [],
                        "created_at": _now(),
                    }
            # Dedup: same source+target+label = same relation
            for r in self._relations:
                if (r["source"].lower() == source.lower()
                        and r["target"].lower() == target.lower()
                        and r["label"].lower() == label.lower()):
                    return False
            self._relations.append({
                "source": source,
                "target": target,
                "label": label,
            })
            self._save()
            return True

    def list_relations(self) -> List[dict]:
        with self._lock:
            return [dict(r) for r in self._relations]

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 8) -> List[dict]:
        """Semantic search over observations.

        Returns a list of {entity, text, score} dicts sorted by relevance.
        Falls back to keyword substring match when the vector index is
        unavailable.
        """
        if not query.strip():
            return []
        self._ensure_vectors()
        if self._vec_ready and self._ef is not None:
            return self._semantic_search(query, top_k)
        return self._keyword_search(query)

    def _semantic_search(self, query: str, top_k: int) -> List[dict]:
        try:
            table = self._ensure_table()
            if table is None:
                return self._keyword_search(query)
            qvec = self._ef.embed_query(query)
            results = table.search(qvec).limit(top_k).to_list()
            out = []
            for r in results:
                dist = float(r.get("_distance", 1.0))
                score = max(0.0, 1.0 - (dist * dist) / 2.0)
                out.append({
                    "entity": r.get("entity", ""),
                    "text": r.get("text", ""),
                    "score": round(score, 4),
                })
            return out
        except Exception as e:
            logger.debug("graph_memory: semantic search failed: %s", e)
            return self._keyword_search(query)

    def _keyword_search(self, query: str) -> List[dict]:
        q = query.lower()
        out = []
        with self._lock:
            for e in self._entities.values():
                for obs in e["observations"]:
                    if q in obs.lower():
                        out.append({"entity": e["name"], "text": obs, "score": 0.5})
        return out

    # ------------------------------------------------------------------
    # snapshot (for prompt injection)
    # ------------------------------------------------------------------

    def snapshot(self, max_items: int = 10) -> str:
        """Compact text snapshot of the most salient graph facts.

        Used for prompt injection: returns a short, readable digest of
        entities/observations/relations, capped at ``max_items`` items
        so it doesn't bloat the system prompt.
        """
        with self._lock:
            entities = list(self._entities.values())
            relations = list(self._relations)
        if not entities and not relations:
            return ""
        lines = []
        # Prioritise entities with the most observations
        entities.sort(key=lambda e: len(e.get("observations", [])), reverse=True)
        shown = 0
        for e in entities:
            if shown >= max_items:
                break
            obs = e.get("observations", [])
            if not obs:
                continue
            typ = f" ({e['type']})" if e.get("type") else ""
            lines.append(f"- {e['name']}{typ}: {'; '.join(obs[:3])}")
            shown += 1
        # A few relations
        for r in relations[:max_items // 2]:
            lines.append(f"- {r['source']} --[{r['label']}]--> {r['target']}")
        return "\n".join(lines) if lines else ""

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        with self._lock:
            self._entities = {}
            self._relations = []
            self._save()
        # Drop vector table
        if self._db is not None:
            try:
                if "observations" in self._db.table_names():
                    self._db.drop_table("observations")
            except Exception:
                pass
        self._table = None

    def close(self) -> None:
        """Release resources.

        The FastEmbed ONNX model is a process-wide shared singleton
        (get_or_create_embedding_function) — we only drop our reference,
        we don't release the session, because RAG or another caller may
        still be using it.  The LanceDB connection is private and safe
        to close.
        """
        self._db = None
        self._table = None
        self._ef = None
        self._vec_ready = False


# ---------------------------------------------------------------------------
# LLM-based extraction: pull entities / relations / observations from text
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = """\
Extract durable facts from the conversation as a knowledge graph. Return ONLY JSON:
{
  "entities": [{"name": "name", "type": "person|project|concept|tool|place|other", "observations": ["fact"]}],
  "relations": [{"source": "name", "target": "name", "label": "verb phrase"}]
}

Rules:
- Only concrete, durable facts about the user, their projects, preferences,
  decisions, or domain knowledge. Skip chitchat and transient state.
- Skip the assistant's own tools/capabilities (no entities for "rag_search", etc.).
- Observations are short self-contained sentences still useful weeks later.
- Canonical entity names ("PostgreSQL", not "postgres db").
- If nothing to persist, return {"entities": [], "relations": []}.
- Valid JSON only, no markdown fences or commentary.\
"""

_EXTRACT_USER = """\
Extract knowledge-graph facts from this conversation fragment.
Return JSON only.

--- Conversation ---
{conversation}
--- End ---
"""


def extract_facts_via_llm(
    llm: Any,
    conversation: str,
    max_chars: int = 4000,
) -> dict:
    """Call the LLM to extract entities/relations/observations from text.

    Returns a dict with 'entities' and 'relations' lists.  On any error
    (parse failure, LLM error) returns an empty graph — callers should
    treat extraction as best-effort, never blocking.

    Args:
        llm: an object with chat_stream(messages, tools, temperature)
             (same interface as LLMClient).
        conversation: the conversation text to extract from.
        max_chars: truncate overly long inputs to stay within the LLM's
                   context budget.
    """
    if not conversation.strip():
        return {"entities": [], "relations": []}

    # Truncate to a reasonable window so we don't blow the context budget.
    text = conversation[:max_chars]

    messages = [
        {"role": "system", "content": _EXTRACT_SYSTEM},
        {"role": "user", "content": _EXTRACT_USER.format(conversation=text)},
    ]
    try:
        out: List[str] = []
        for ev in llm.chat_stream(messages, tools=None, temperature=0.1):
            if ev.type == "content":
                out.append(ev.content)
            elif ev.type == "done":
                break
        raw = "".join(out).strip()
        # Tolerate markdown-fenced JSON from models that ignore instructions.
        if raw.startswith("```"):
            # Strip ```json ... ``` wrapper
            lines = raw.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {"entities": [], "relations": []}
        data.setdefault("entities", [])
        data.setdefault("relations", [])
        return data
    except Exception as e:
        logger.debug("graph_memory: extract_facts failed: %s", e)
        return {"entities": [], "relations": []}

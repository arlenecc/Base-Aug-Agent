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


def _normalize_vector(vec: List[float]) -> List[float]:
    """Normalize a vector to unit length.

    LanceDB's default L2 distance only maps to cosine similarity
    (``1 - d²/2``) when vectors are unit-normalized.  FastEmbed outputs are
    NOT normalized, so raw L2 distances are huge (~300+), making the
    ``1 - d²/2`` similarity collapse to 0 for every pair.  Normalize both on
    write and on query.
    """
    norm = 0.0
    for x in vec:
        norm += x * x
    norm = norm ** 0.5
    if norm == 0:
        return list(vec)
    return [x / norm for x in vec]


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
        # backfill 的独立非阻塞锁：防止并发重复回填（_vec_lock 保护的是
        # lazy init 本身，语义不同，混用会把回填串行进 init 持锁路径）。
        self._backfill_lock = threading.Lock()
        self._entities: Dict[str, dict] = {}  # name → entity
        self._relations: List[dict] = []
        self._db = None
        self._table = None
        self._ef = None
        self._vec_ready = False
        # 向量索引只探测一次；lancedb/fastembed 缺失时不反复重试 connect+import。
        self._vec_tried = False
        self._backfilled = False
        self._load()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            # 结构校验：手工编辑/其他程序写坏可能产出「合法 JSON 但结构错误」
            # 的文件（顶层 list、entities 为 list 等）——AttributeError 会在
            # 之后任意一次图操作时从很远的位置抛出，且没有任何恢复路径。
            # 不满足结构约定时走与解析失败相同的「警告 + 置空」处理。
            if not isinstance(data, dict):
                raise ValueError("graph json root is not an object")
            entities = data.get("entities", {})
            relations = data.get("relations", [])
            if not isinstance(entities, dict) or not isinstance(relations, list):
                raise ValueError("graph json entities/relations have wrong types")
            self._entities = {n.lower(): e for n, e in entities.items()}
            # Normalise: ensure every entity has observations list
            for e in self._entities.values():
                if not isinstance(e.get("observations"), list):
                    e["observations"] = []
            self._relations = relations
        except (OSError, ValueError) as e:
            logger.warning("graph_memory: failed to load %s: %s（按空图继续，坏文件保留在磁盘）", self._path, e)
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

        On first successful init, backfills any observations already present
        in the JSON graph but missing from the vector table (so facts stored
        before the vector index existed become searchable too).
        """
        if self._vec_ready or self._vec_tried:
            # init 已定型（成功/失败各一次）。backfill 的重试必须走独立
            # 轻量路径——放在早退 gate 之后不可达，日志里「下次调用将重试」
            # 的承诺从未成立，回填失败的历史事实在本进程内永远无法语义召回。
            self._retry_backfill()
            return
        with self._vec_lock:
            if self._vec_ready or self._vec_tried:  # double-check
                self._retry_backfill()
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
            finally:
                # 只尝试一次：lancedb/fastembed 缺失的环境下，旧的
                # 「失败不记忆」会让每次 search() 都重新 connect + import
                # （并且每次都把同样那条 debug 日志刷出来）。
                self._vec_tried = True
        if self._vec_ready:
            self._retry_backfill()

    def _retry_backfill(self) -> None:
        """Run the one-shot backfill with retry-on-failure (non-blocking)."""
        if getattr(self, "_backfilled", False):
            return
        if not self._vec_ready:
            return
        # 非阻塞：已有线程在回填（embed 慢）时后来的调用直接跳过，
        # 等它失败后下一次调用再试。
        if not self._backfill_lock.acquire(blocking=False):
            return
        try:
            try:
                self._backfill_vectors()
                self._backfilled = True
            except Exception as e:
                logger.warning("graph_memory: backfill failed (下次调用将重试): %s", e)
        finally:
            self._backfill_lock.release()

    def _backfill_vectors(self) -> None:
        """Index any observations present in the graph JSON but missing from
        the vector table (e.g. facts stored before the vector index existed)."""
        # Snapshot entities under lock.
        with self._lock:
            pairs = [
                (e["name"], obs)
                for e in self._entities.values()
                for obs in e.get("observations", [])
            ]
        if not pairs:
            return
        table = self._ensure_table()
        # Determine which ids are already indexed.
        existing_ids: set = set()
        if table is not None:
            try:
                # 只取 id 列：to_arrow() 会物化全部列（含 768 维向量列），
                # to_pylist() 再转成 Python 对象 —— 上万条 observation 时是
                # 数百 MB 的一次性峰值，只为拿一个 id 集合。
                rows = (
                    table.search()
                    .select(["id"])
                    .limit(max(table.count_rows(), 1))
                    .to_arrow()
                    .to_pylist()
                )
                for r in rows:
                    existing_ids.add(r.get("id", ""))
            except Exception as e:
                logger.debug("graph_memory: 读取已索引 id 失败 (%s)，按全量回填", e)
                existing_ids = set()
        # Collect missing (entity, observation) pairs.
        missing = [
            (name, obs)
            for name, obs in pairs
            if _hash(name + "|" + obs) not in existing_ids
        ]
        if not missing:
            return
        # Embed + insert missing observations in batches of 100.
        for i in range(0, len(missing), 100):
            batch = missing[i:i + 100]
            batch_texts = [obs for _, obs in batch]
            # embed 失败必须抛出：静默 return 会让调用方把「部分失败」标记为
            # 已完成（_backfilled=True），剩余批次的历史事实再也补不上。
            vecs = self._ef.embed_documents(batch_texts)
            records = [
                {
                    "id": _hash(name + "|" + obs),
                    "entity": name,
                    "text": obs,
                    "vector": _normalize_vector(vec),
                }
                for (name, obs), vec in zip(batch, vecs)
            ]
            with self._vec_lock:
                if table is None:
                    self._table = self._db.create_table("observations", records)
                    table = self._table
                else:
                    table.add(records)
        logger.info("graph_memory: backfilled %d observations into vector index", len(missing))

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
        self._ensure_vectors()
        if not self._vec_ready:
            return
        try:
            table = self._ensure_table()
            oid = _hash(entity_name + "|" + observation)
            # Observations are documents (retrieval targets), so embed with
            # the document prefix — matching _semantic_search which embeds
            # the query with the query prefix.
            vec = _normalize_vector(self._ef.embed_documents([observation])[0])
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

        Lazily initialises the vector index on first use (``_ensure_vectors``)
        — previously it bailed out if the index wasn't already ready, which
        meant the very first ``add_observations`` (before any ``search``)
        silently skipped indexing and the vector table was never built.
        """
        if not observations:
            return
        self._ensure_vectors()
        if not self._vec_ready:
            return
        try:
            vecs = self._ef.embed_documents(observations)
            if len(vecs) != len(observations):
                # embedding 返回数量不足（模型异常/截断）时不能静默用 zip 截断
                # ——被丢掉的内容已写进 JSON 图谱，会造成「图里有、向量库没有」
                # 的永久性检索缺口。这里宁可整批失败（外层有兜底），也不留
                # 半截索引。
                raise RuntimeError(
                    f"embedding returned {len(vecs)} vectors for "
                    f"{len(observations)} observations"
                )
            records = [
                {
                    "id": _hash(entity_name + "|" + obs),
                    "entity": entity_name,
                    "text": obs,
                    "vector": _normalize_vector(vec),
                }
                for obs, vec in zip(observations, vecs)
            ]
            with self._vec_lock:
                table = self._ensure_table()
                if table is None:
                    self._table = self._db.create_table("observations", records)
                    return
                # 单条 IN 删除而不是逐条 delete：典型 N+1 —— 一批 N 条记录
                # 就是 N 次 LanceDB 删除事务。
                ids = [rec["id"] for rec in records]
                if len(ids) == 1:
                    try:
                        table.delete(f"id = '{ids[0]}'")
                    except Exception:
                        pass
                elif ids:
                    try:
                        in_list = ", ".join("'" + i.replace("'", "''") + "'" for i in ids)
                        table.delete(f"id IN ({in_list})")
                    except Exception:
                        # 少数后端不支持 IN → 退回逐条删除，保证语义不丢。
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
            # 用 .get：损坏的历史数据可能缺 source/target 键，直接下标取会让
            # 整个 delete_entity 抛 KeyError —— 删除失败且图处于半删状态。
            self._relations = [
                r for r in self._relations
                if (r.get("source") or "").lower() != key
                and (r.get("target") or "").lower() != key
            ]
            self._save()
        # _deindex_observations 内部是 LanceDB 删除事务 + table_names() 文件
        # 系统探测（50-500ms 量级），必须放在图锁外——create_entity /
        # add_observations / batch_apply 都遵守「embedding/LanceDB 操作绝不
        # 持图锁」的约定（否则冻结每轮系统提示的 snapshot/搜索），唯独这里
        # 漏了，删除实体期间所有图读取被阻塞。
        self._deindex_observations(name)
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
    # batch write (fact extraction)
    # ------------------------------------------------------------------

    def batch_apply(self, entities: List[dict], relations: List[dict]) -> int:
        """一次写入一批实体的 observations + 关系，返回新增 observation 数。

        事实抽取一次会产出 ≤5 个实体 + ≤3 条关系。逐个调
        ``add_observations`` / ``create_relation`` 会让**每个**元素都触发
        一次 ``_save()`` —— 而每次 _save() 都要把整张图 JSON 序列化一遍再
        原子写盘（图越大越贵），也就是最多 8 次全量落盘换一批事实。

        这里在一把锁内完成所有内存更新，只 _save() 一次；embedding 仍按实体
        在锁外批量执行（ONNX 计算绝不能持图锁）。
        """
        # 先归一化入参，尽量少占用锁的时间。
        # name.lower() -> (name, entity_type, observations)
        merged: Dict[str, tuple] = {}
        for ent in entities or []:
            name = _norm_name((ent or {}).get("name", ""))
            if not name:
                continue
            key = name.lower()
            etype = ((ent or {}).get("type") or "").strip()
            obs = [o.strip() for o in (ent or {}).get("observations", []) if o and o.strip()]
            if key not in merged:
                merged[key] = (name, etype, list(obs))
                continue
            # 同一实体在抽取结果里重复出现 → 合并而不是丢弃后者，否则后一条
            # 里独有的 observation 会被静默丢掉。
            prev_name, prev_type, prev_obs = merged[key]
            for o in obs:
                if o not in prev_obs:
                    prev_obs.append(o)
            if etype and not prev_type:
                merged[key] = (prev_name, etype, prev_obs)
        wanted: List[tuple] = list(merged.values())

        rels: List[tuple] = []
        for rel in relations or []:
            src = _norm_name((rel or {}).get("source", ""))
            tgt = _norm_name((rel or {}).get("target", ""))
            lbl = ((rel or {}).get("label") or "").strip()
            if src and tgt and lbl:
                rels.append((src, tgt, lbl))

        to_index: List[tuple] = []  # (name, [new observations])
        added = 0
        with self._lock:
            for name, etype, obs in wanted:
                key = name.lower()
                e = self._entities.get(key)
                if e is None:
                    e = {
                        "name": name,
                        "type": etype,
                        "observations": [],
                        "created_at": _now(),
                    }
                    self._entities[key] = e
                elif etype and not e.get("type"):
                    e["type"] = etype
                if not isinstance(e.get("observations"), list):
                    e["observations"] = []
                fresh = []
                for o in obs:
                    if o not in e["observations"]:
                        e["observations"].append(o)
                        fresh.append(o)
                if fresh:
                    to_index.append((name, fresh))
                    added += len(fresh)

            for src, tgt, lbl in rels:
                for n in (src, tgt):
                    k = n.lower()
                    if k not in self._entities:
                        self._entities[k] = {
                            "name": n,
                            "type": "",
                            "observations": [],
                            "created_at": _now(),
                        }
                # 去重：source+target+label 相同视为同一条边。
                if any(
                    (r.get("source") or "").lower() == src.lower()
                    and (r.get("target") or "").lower() == tgt.lower()
                    and (r.get("label") or "").lower() == lbl.lower()
                    for r in self._relations
                ):
                    continue
                self._relations.append({"source": src, "target": tgt, "label": lbl})

            if added or rels:
                # 只落盘一次。
                self._save()

        # 锁外批量 embedding：每个实体一批 ONNX 前向。
        for name, fresh in to_index:
            self._index_observations_batch(name, fresh)
        return added

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
        return self._keyword_search(query, top_k)

    def _semantic_search(self, query: str, top_k: int) -> List[dict]:
        try:
            table = self._ensure_table()
            if table is None:
                return self._keyword_search(query, top_k)
            qvec = _normalize_vector(self._ef.embed_query(query))
            results = table.search(qvec).limit(top_k).to_list()
            out = []
            for r in results:
                dist = float(r.get("_distance", 1.0))
                score = max(0.0, 1.0 - (dist * dist) / 2.0)
                # 相似度阈值：过滤语义上不相关的结果（否则无关查询也会
                # 返回低分命中，破坏"无匹配返回空"的语义）。
                if score < 0.5:
                    continue
                out.append({
                    "entity": r.get("entity", ""),
                    "text": r.get("text", ""),
                    "score": round(score, 4),
                })
            return out
        except Exception as e:
            logger.debug("graph_memory: semantic search failed: %s", e)
            return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int = 8) -> List[dict]:
        """Keyword substring fallback.

        ``top_k`` **必须**生效：这是没有装 rag 可选依赖时的默认路径，若不限制
        条数，一次宽泛查询会把全部 observation 灌进上下文（其他检索工具都做了
        钳制，这里不能例外）。
        """
        q = query.lower()
        out = []
        with self._lock:
            for e in self._entities.values():
                for obs in e["observations"]:
                    if q in obs.lower():
                        out.append({"entity": e["name"], "text": obs, "score": 0.5})
                        if len(out) >= top_k:
                            return out
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
        # 真正关闭 LanceDB 连接：每次「应用配置」重建 Agent 都会走这里，
        # 只把引用置 None 会泄漏连接及其对象存储缓存。
        if self._db is not None:
            try:
                close_fn = getattr(self._db, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception as e:
                logger.debug("graph_memory: closing LanceDB connection failed: %s", e)
        self._db = None
        self._table = None
        self._ef = None
        self._vec_ready = False
        # 允许重新初始化：否则 close 后 _ensure_vectors 会提前返回，
        # 后续（重建的）store 永远拿不到向量能力。
        self._vec_tried = False
        # _backfilled 同样必须复位：它一旦为 True 就永久跳过 backfill，
        # 重建的 store 会漏掉「JSON 图里有、向量表里没有」的历史事实。
        self._backfilled = False


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
  decisions, or domain knowledge. Skip chitchat, transient state, file contents
  and raw tool output.
- Skip the assistant's own tools/capabilities (no entities for "rag_search", etc.).
- Observations are short self-contained sentences still useful weeks later.
- Canonical entity names ("PostgreSQL", not "postgres db").
- Write observations in the same language as the conversation (mismatched
  language makes later semantic recall fail).
- Extract at most 5 entities and 3 observations each — keep only the most
  durable. Exceeding this pollutes memory and inflates every later prompt.
- If nothing to persist, return {"entities": [], "relations": []}.
- Valid JSON only, no markdown fences or commentary.\
"""

_EXTRACT_USER = """\
Extract knowledge-graph facts from this conversation fragment (≤5 entities).

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
        # 类型收敛：模型输出脏 JSON（"observations": "likes Python" 字符串、
        # entities 混入非 dict 元素）会让 batch_apply 的迭代行为错乱——字符串
        # 被逐字符拆成单字「事实」永久污染图谱（去重机制无法自愈），非 dict
        # 元素抛 AttributeError 使整批抽取被丢弃。这里统一收敛到约定结构。
        ents = data.get("entities")
        data["entities"] = (
            [e for e in ents if isinstance(e, dict)] if isinstance(ents, list) else []
        )
        rels = data.get("relations")
        data["relations"] = (
            [r for r in rels if isinstance(r, dict)] if isinstance(rels, list) else []
        )
        for e in data["entities"]:
            obs = e.get("observations", [])
            if isinstance(obs, str):
                e["observations"] = [obs] if obs.strip() else []
            elif isinstance(obs, list):
                e["observations"] = [str(o) for o in obs if o is not None and str(o).strip()]
            else:
                e["observations"] = []
        return data
    except Exception as e:
        logger.debug("graph_memory: extract_facts failed: %s", e)
        return {"entities": [], "relations": []}

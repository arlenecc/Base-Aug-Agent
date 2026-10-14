"""work_memory and knowledge-graph memory tools.

WorkMemory stays a simple key-value scratchpad (unchanged).

LongTermMemory is now a knowledge graph (entities + relations + observations)
backed by GraphMemoryStore with LanceDB semantic search.  Three tools expose
the graph to the agent:

  - work_memory         — short-term scratchpad (set/get/list/clear)
  - memory_graph        — CRUD on the knowledge graph
  - memory_search       — semantic recall over observations
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, List, Optional

from ..config import AgentConfig
from ..memory import LongTermMemory, WorkMemory
from .base import Tool, ToolRegistry, ToolResult


# ---------------------------------------------------------------------------
# Singleton WorkMemory — avoids atexit-handler accumulation.
#
# Each WorkMemory wraps a _JsonStore that registers an atexit flush closure
# on construction.  If we created a new instance on every call, each tool
# invocation would append one more closure to the atexit registry, growing
# without bound for the lifetime of the process (memory leak).  The instance
# is cached per workspace path on the module level so it survives registry
# rebuilds (work memory data must persist across "应用配置" anyway).
# ---------------------------------------------------------------------------

_work_memory_by_path: dict = {}
_work_memory_lock = threading.Lock()
# 缓存上限：按 workspace 路径缓存的实例永不淘汰，用户反复切换工作目录
# （或与 tmp 目录配合的长跑测试）会让这个字典单调增长，每个条目都持有一个
# _JsonStore（含其数据与 atexit WeakSet 引用）。超过上限就淘汰最久未用的。
_WORK_MEMORY_MAX_ENTRIES = 32


def _work_memory(config: AgentConfig) -> WorkMemory:
    path = os.path.join(config.workspace, ".agent", "work_memory.json")
    wm = _work_memory_by_path.get(path)
    if wm is None:
        with _work_memory_lock:
            wm = _work_memory_by_path.get(path)
            if wm is None:
                # 先强制淘汰，再建新实例：避免刚创建的实例立刻被挤出去。
                if len(_work_memory_by_path) >= _WORK_MEMORY_MAX_ENTRIES:
                    for stale in list(_work_memory_by_path)[
                        : max(1, _WORK_MEMORY_MAX_ENTRIES // 4)
                    ]:
                        victim = _work_memory_by_path.pop(stale, None)
                        if victim is not None:
                            try:
                                victim.flush(force=True)
                            except Exception:
                                pass
                wm = WorkMemory(path=path)
                _work_memory_by_path[path] = wm
            else:
                # 命中即「最近使用」：dict 保持插入序，移到末尾实现简易 LRU。
                _work_memory_by_path.pop(path)
                _work_memory_by_path[path] = wm
    return wm


# ---------------------------------------------------------------------------
# Singleton LongTermMemory — avoids leaking ONNX/LanceDB resources.
#
# Each LongTermMemory wraps a GraphMemoryStore that lazily loads a FastEmbed
# ONNX model (~137MB).  If we created a new instance on every call, every
# tool invocation and every fact extraction would leak ~137MB of C++ heap
# that Python GC can't reclaim promptly.  The singleton is cached on the
# ToolRegistry so shutdown() can release it.
# ---------------------------------------------------------------------------

_LONG_MEMORY_SINGLETON_ATTR = "_long_memory_instance"


def _get_long_memory(registry: "ToolRegistry", config: AgentConfig) -> Optional[LongTermMemory]:
    """Return the singleton LongTermMemory, creating it on first access.

    Returns ``None`` when the registry has already been shut down: a late
    background thread (fact extraction) would otherwise build a brand-new
    ``GraphMemoryStore`` *after* ``shutdown()`` released the old one — that
    instance (~137MB ONNX + a LanceDB connection) would then have no owner
    and leak for the rest of the process.
    """
    if getattr(registry, "_shutdown", False):
        return None
    existing = getattr(registry, _LONG_MEMORY_SINGLETON_ATTR, None)
    if existing is not None:
        return existing
    # 首次创建是重活（懒加载 ~137MB ONNX + LanceDB 连接），agent 线程与 UI
    # 线程可能并发首次调用：不加锁会各建一个实例，后写者覆盖属性，先建的
    # 连接/模型泄漏且 shutdown() 收不回。与 _work_memory 相同的双重检查模式。
    with registry._singleton_lock:
        existing = getattr(registry, _LONG_MEMORY_SINGLETON_ATTR, None)
        if existing is not None:
            return existing
        emb = getattr(config, "rag_embedding_model", "nomic-ai/nomic-embed-text-v1.5-Q")
        lm = LongTermMemory(
            path=os.path.join(config.workspace, ".agent", "long_memory.json"),
            embedding_model=emb,
        )
        setattr(registry, _LONG_MEMORY_SINGLETON_ATTR, lm)
        return lm


def _long_memory(config: AgentConfig) -> LongTermMemory:
    """Standalone LongTermMemory for callers without a registry reference.

    Creates a fresh instance — callers are responsible for calling close()
    when done.  In the normal tool/agent path, use _get_long_memory() instead
    so the singleton is reused and properly released on shutdown().
    """
    emb = getattr(config, "rag_embedding_model", "nomic-ai/nomic-embed-text-v1.5-Q")
    return LongTermMemory(
        path=os.path.join(config.workspace, ".agent", "long_memory.json"),
        embedding_model=emb,
    )


# ---------------------------------------------------------------------------
# Work memory tool (unchanged)
# ---------------------------------------------------------------------------


class WorkMemoryTool(Tool):
    name = "work_memory"
    description = "Task scratchpad. op: set(key,value), get(key), list, clear."
    parameters = {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": ["set", "get", "list", "clear"]},
            "key": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["op"],
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def run(self, op: str, key: str = "", value: str = "") -> ToolResult:
        wm = _work_memory(self.config)
        if op == "set":
            if not key:
                return ToolResult(False, error="op 'set' requires 'key'")
            wm.set(key, value)
            return ToolResult(True, output=f"set {key}")
        if op == "get":
            if not key:
                return ToolResult(False, error="op 'get' requires 'key'")
            v = wm.get(key)
            return ToolResult(True, output=v if v is not None else "")
        if op == "list":
            return ToolResult(True, output=json.dumps(wm.list(), ensure_ascii=False))
        if op == "clear":
            wm.clear()
            return ToolResult(True, output="cleared")
        return ToolResult(False, error=f"unknown op: {op}")


# ---------------------------------------------------------------------------
# Knowledge-graph CRUD tool
# ---------------------------------------------------------------------------


class MemoryGraphTool(Tool):
    """CRUD operations on the long-term knowledge graph.

    Operations:
      - create_entity:  create/merge an entity (args: name, type?, observations?)
      - delete_entity:  remove an entity and its relations (arg: name)
      - add_observations: add facts to an entity (args: name, observations)
      - create_relation: create a directed edge (args: source, target, label)
      - list_entities:  list all entities (no extra args)
      - get_entity:     get one entity's full record (arg: name)
      - snapshot:       compact digest for the prompt (no extra args)
      - clear:          wipe the entire graph
    """
    name = "memory_graph"
    description = (
        "Long-term knowledge graph CRUD. op: create_entity(name,type?,observations?), "
        "delete_entity(name), add_observations(name,observations), "
        "create_relation(source,target,label), list_entities, get_entity(name), "
        "snapshot, clear."
    )
    parameters = {
        "type": "object",
        "properties": {
            "op": {
                "type": "string",
                "enum": [
                    "create_entity", "delete_entity", "add_observations",
                    "create_relation", "list_entities", "get_entity",
                    "snapshot", "clear",
                ],
            },
            "name": {"type": "string"},
            "entity_type": {"type": "string", "description": "person/project/concept/tool/place/other"},
            "observations": {"type": "array", "items": {"type": "string"}},
            "source": {"type": "string"},
            "target": {"type": "string"},
            "label": {"type": "string"},
        },
        "required": ["op"],
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def run(self, op: str, name: str = "", entity_type: str = "",
            observations: List[str] = None, source: str = "",
            target: str = "", label: str = "") -> ToolResult:
        lm = _get_long_memory(self.registry, self.config)
        if lm is None:
            return ToolResult(False, error="长期记忆已释放（registry 已关闭）")
        g = lm.graph
        # 模型可能把 observations 传成单个字符串——字符串会被 for 循环按
        # 字符迭代，一次调用就往图谱塞几十条单字垃圾观察并触发同等数量的
        # 嵌入计算。非 list[str] 的脏值统一归一/过滤。
        if observations is not None:
            if isinstance(observations, str):
                observations = [observations]
            elif isinstance(observations, (list, tuple)):
                observations = [str(o) for o in observations if o is not None]
            else:
                observations = None
        if op == "create_entity":
            if not name:
                return ToolResult(False, error="op 'create_entity' requires 'name'")
            e = g.create_entity(name, entity_type=entity_type, observations=observations or [])
            return ToolResult(True, output=json.dumps(e, ensure_ascii=False))
        if op == "delete_entity":
            if not name:
                return ToolResult(False, error="op 'delete_entity' requires 'name'")
            ok = g.delete_entity(name)
            return ToolResult(True, output="deleted" if ok else "not found")
        if op == "add_observations":
            if not name or not observations:
                return ToolResult(False, error="op 'add_observations' requires 'name' and 'observations'")
            added = g.add_observations(name, observations)
            return ToolResult(True, output=f"added {added} observation(s)")
        if op == "create_relation":
            if not source or not target or not label:
                return ToolResult(False, error="op 'create_relation' requires 'source', 'target', 'label'")
            ok = g.create_relation(source, target, label)
            return ToolResult(True, output="created" if ok else "duplicate or invalid")
        if op == "list_entities":
            entities = g.list_entities()
            # Compact: name + type + observation count.
            # 图谱可能积累数百实体，全量塞进上下文会挤掉真正要处理的内容；
            # 截断并提示用 get_entity 精确查询。
            _MAX_LIST = 200
            summary = [
                f"{e['name']} ({e.get('type','?')}) [{len(e.get('observations',[]))} obs]"
                for e in entities[:_MAX_LIST]
            ]
            if len(entities) > _MAX_LIST:
                summary.append(f"… (共 {len(entities)} 个实体，仅显示前 {_MAX_LIST} 个；"
                               f"用 get_entity 查询具体实体)")
            return ToolResult(True, output=json.dumps(summary, ensure_ascii=False))
        if op == "get_entity":
            if not name:
                return ToolResult(False, error="op 'get_entity' requires 'name'")
            e = g.get_entity(name)
            return ToolResult(True, output=json.dumps(e, ensure_ascii=False) if e else "not found")
        if op == "snapshot":
            snap = g.snapshot(max_items=10)
            return ToolResult(True, output=snap or "(empty)")
        if op == "clear":
            g.clear()
            return ToolResult(True, output="cleared")
        return ToolResult(False, error=f"unknown op: {op}")


# ---------------------------------------------------------------------------
# Semantic search tool
# ---------------------------------------------------------------------------


class MemorySearchTool(Tool):
    """Semantic search over long-term memory observations."""
    name = "memory_search"
    description = "Semantic search over long-term memory observations. Returns matching facts with entity and relevance score."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to recall — use the user's own natural phrasing; never translate or transliterate",
            },
            "top_k": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def run(self, query: str, top_k: int = 5) -> ToolResult:
        # 模型可能传 null 或超大 top_k：None 会让 embed 失败，大 top_k 会把
        # 全部观察灌进上下文。归一化 + 钳制（与 rag_search 一致）。
        query = (query or "").strip()
        if not query:
            return ToolResult(False, error="请提供要回忆的内容（query）")
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = 5
        top_k = max(1, min(top_k, 20))
        lm = _get_long_memory(self.registry, self.config)
        if lm is None:
            return ToolResult(False, error="长期记忆已释放（registry 已关闭）")
        results = lm.graph.search(query, top_k=top_k)
        return ToolResult(True, output=json.dumps(results, ensure_ascii=False))

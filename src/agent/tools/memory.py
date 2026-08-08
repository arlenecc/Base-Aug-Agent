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
from typing import Any, List

from ..config import AgentConfig
from ..memory import LongTermMemory, WorkMemory
from .base import Tool, ToolRegistry, ToolResult


def _work_memory(config: AgentConfig) -> WorkMemory:
    return WorkMemory(path=os.path.join(config.workspace, ".agent", "work_memory.json"))


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


def _get_long_memory(registry: "ToolRegistry", config: AgentConfig) -> LongTermMemory:
    """Return the singleton LongTermMemory, creating it on first access."""
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
        g = lm.graph
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
            # Compact: name + type + observation count
            summary = [
                f"{e['name']} ({e.get('type','?')}) [{len(e.get('observations',[]))} obs]"
                for e in entities
            ]
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
            "query": {"type": "string", "description": "What to recall"},
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
        lm = _get_long_memory(self.registry, self.config)
        results = lm.graph.search(query, top_k=top_k)
        return ToolResult(True, output=json.dumps(results, ensure_ascii=False))

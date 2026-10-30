"""Tool base class, registry, and exports."""
from __future__ import annotations

import inspect
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..config import AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    success: bool
    output: str = ""
    error: str = ""
    # When requires_input is set, the agent should call callbacks.ask_user and
    # feed the answer back (used by ask_user tool internally).
    requires_input: bool = False

    def to_message(self) -> str:
        if self.success:
            return self.output
        return f"ERROR: {self.error}" if self.error else "ERROR: unknown"


class Tool:
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}
    # destructive marks tools that *can* affect the system or files outside the
    # workspace. Workspace-locked file ops set this False (they cannot escape).
    destructive: bool = False

    def run(self, **kwargs) -> ToolResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def should_confirm(self, args: Dict[str, Any]) -> bool:
        """Return True if this invocation needs user confirmation before running.

        Default policy: confirm iff `self.destructive` is set. Override for
        finer-grained control (e.g. confirm only when args target paths outside
        the workspace). Operations confined to the workspace never need
        confirmation, since they cannot harm the system or other files.
        """
        return self.destructive

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, config: AgentConfig, callbacks: "Any", aux_llm_getter: "Any" = None):
        self.config = config
        self.callbacks = callbacks
        # 读取「当前」辅助模型后端的回调（可为 None）。RAG 检索结果会先送辅助
        # 模型归纳整理再交给主模型；辅助模型缺失时退化为原始分片。
        self._aux_llm_getter = aux_llm_getter
        self.config.ensure_workspace()
        self._tools: Dict[str, Tool] = {}
        # 保护 _tools / _mcp_clients：MCP 工具由后台线程注册，而 agent 线程
        # 会并发读（schemas/get/execute）。没有锁时后台线程写入期间读到的可能
        # 是「注册了一半」的工具表。
        self._tools_lock = threading.RLock()
        self._mcp_clients: list = []  # MCPClient instances, for cleanup
        self._mcp_threads: list = []  # 后台连接线程，shutdown 时要回收
        self._shutdown = False
        self._rag_engine = None  # lazy-initialized RAGEngine
        # 保护懒初始化的重资源单例（SkillRetriever / LongTermMemory）：
        # agent 线程与 UI 线程可能并发首次调用。
        self._singleton_lock = threading.RLock()
        self._retriever_lock = self._singleton_lock
        # 工具 schema 的版本号：工具表变动时递增，agent 侧据此避免每轮迭代
        # 重复序列化整套 schema 来算 token 估算的指纹。
        self._schema_version = 0
        # 状态版本号：状态型工具（task_plan）改写了「会注入系统提示」的持久
        # 数据后递增，agent 侧的 _system_prompt 缓存据此失效——否则同一次
        # run() 内更新的计划在后续迭代里不可见。
        self._state_version = 0
        self._register_defaults()

    # ------------------------------------------------------------------
    def _register_defaults(self) -> None:
        from .file_ops import FileReadTool, FileWriteTool, FileModifyTool
        from .code_run import CodeRunTool
        from .shell_run import ShellRunTool
        from .web import WebScanTool, WebExecJsTool
        from .interact import AskUserTool
        from .memory import WorkMemoryTool, MemoryGraphTool, MemorySearchTool
        from .planning import TaskPlanTool
        from .skill_search import SkillSearchTool, SkillLoadTool, SkillSemanticSearchTool

        tools_to_register = [
            FileReadTool(), FileWriteTool(), FileModifyTool(),
            CodeRunTool(),
            ShellRunTool(),
            WebScanTool(),
            AskUserTool(),
            WorkMemoryTool(),
            TaskPlanTool(),
            MemoryGraphTool(),
            MemorySearchTool(),
            SkillSearchTool(),
            SkillLoadTool(),
            SkillSemanticSearchTool(),
        ]

        # webexec_js: only register when browser_endpoint is configured.
        # Saves ~146 tokens of schema when no browser bridge is available
        # (the common case).  Without an endpoint the tool just returns
        # guidance text — no functional value.
        if getattr(self.config, "browser_endpoint", ""):
            tools_to_register.append(WebExecJsTool())

        for t in tools_to_register:
            t.bind(self.config, self)
            self._tools[t.name] = t
        self._invalidate_schema_cache()

        # Register RAG tools
        self._register_rag_tools()

        # Register MCP tools from configured servers
        self._register_mcp_tools()

    def _register_mcp_tools(self) -> None:
        """Connect to configured MCP servers and register their tools.

        **在后台线程连接**：握手 + tools/list 是同步阻塞的，一个起不来的
        server 要等满 30s 超时。旧实现在 ``ToolRegistry.__init__`` 里同步跑，
        而它在 UI 线程上被 ``_rebuild_agent()`` 调用 —— 配了 N 个 server 就是
        最长 30s × N 的界面冻结，且期间没有任何提示。

        现在只在这里创建 client（廉价）并起一个守护线程去连接；工具连好后
        才注册进 ``_tools``（读侧有锁保护）。代价是「配置生效后的第一轮对话」
        可能还看不到 MCP 工具——远好于界面卡死。工具被调用时若尚未连接，
        ``MCPClient.ensure_started()`` 会现场拉起。
        """
        mcp_servers = getattr(self.config, "mcp_servers", None) or []
        if not mcp_servers:
            return

        from .mcp_client import MCPClient

        clients = []
        for server_cfg in mcp_servers:
            if not isinstance(server_cfg, dict):
                continue
            name = server_cfg.get("name", "")
            command = server_cfg.get("command", "")
            if not name or not command:
                logger.warning("MCP server config missing name or command: %s", server_cfg)
                continue
            clients.append(MCPClient(
                name=name,
                command=command,
                args=server_cfg.get("args", []),
                env=server_cfg.get("env"),
            ))

        if not clients:
            return

        with self._tools_lock:
            self._mcp_clients.extend(clients)

        t = threading.Thread(
            target=self._mcp_connect_worker,
            args=(clients,),
            daemon=True,
            name="mcp-connect",
        )
        with self._tools_lock:
            self._mcp_threads.append(t)
        t.start()

    def _mcp_connect_worker(self, clients: list) -> None:
        """后台连接各 MCP server 并注册其工具（见 _register_mcp_tools）。"""
        from .mcp_client import MCPError
        from .mcp_tool import MCPTool

        for client in clients:
            if self._is_shutdown():
                return
            try:
                client.start()
            except MCPError as e:
                logger.warning("MCP server '%s' init failed: %s", client.name, e)
                continue
            except Exception as e:
                logger.warning("MCP server '%s' unexpected error: %s", client.name, e)
                continue

            try:
                tool_defs = client.list_tools()
            except Exception as e:
                logger.warning("MCP server '%s' tools/list failed: %s", client.name, e)
                continue

            with self._tools_lock:
                if self._shutdown:
                    # 连接期间 registry 已关闭：这个 client 已登记在
                    # _mcp_clients 里，交给 shutdown() 回收，不再注册工具。
                    return
                registered = 0
                for tool_def in tool_defs:
                    t = MCPTool(client, tool_def)
                    if t.name and t.name not in self._tools:
                        t.bind(self.config, self)
                        self._tools[t.name] = t
                        registered += 1
                if registered:
                    # 工具表变了 → 让 agent 侧的 schema 缓存失效，否则新注册
                    # 的 MCP 工具永远进不了模型可见的 schema。
                    self._invalidate_schema_cache()
            logger.info(
                "MCP server '%s' ready: %d tools registered", client.name, registered
            )

    def _is_shutdown(self) -> bool:
        with self._tools_lock:
            return self._shutdown

    def _register_rag_tools(self) -> None:
        """Initialize RAG engine and register RAG tools."""
        kb = getattr(self.config, "knowledge_base", "") or ""
        # If knowledge_base is not explicitly set, default to
        # <workspace>/knowledge_base so that RAG tools are available out of
        # the box when users place documents there (or have previously
        # synced knowledge).  Without this, the agent never registers RAG
        # tools and cannot discover the local knowledge base.
        if not kb:
            kb = os.path.join(self.config.workspace, "knowledge_base")
            logger.info(
                "RAG: knowledge_base not configured, defaulting to %s", kb
            )

        # RAG 是可选能力：整段初始化（建目录 + 构造引擎 + 注册工具）都必须
        # 包在 try 里。否则知识库路径不可创建/引擎构造失败时，异常会冒泡到
        # ToolRegistry.__init__ → Agent 构造失败 → 文件读写、shell、web 等
        # **所有**工具一起失效。
        try:
            # Create the directory if it doesn't exist so that RAGEngine
            # initialization doesn't fail on a missing path.
            os.makedirs(kb, exist_ok=True)

            from ..rag.engine import RAGEngine
            from .rag_tool import RagSearchTool, RagStatusTool, RagIngestTool, RagOutlineTool

            logger.info(
                "RAG: initializing engine (workspace=%s kb=%s embedding=%s)",
                self.config.workspace, kb,
                getattr(self.config, "rag_embedding_model",
                        "jinaai/jina-embeddings-v5-text-nano"),
            )
            engine = RAGEngine(
                workspace=self.config.workspace,
                knowledge_base=kb,
                embedding_model=getattr(self.config, "rag_embedding_model",
                                       "jinaai/jina-embeddings-v5-text-nano"),
                embedding_backend=getattr(self.config, "rag_embedding_backend", "mlx"),
                embedding_model_path=getattr(self.config, "rag_embedding_model_path", ""),
                embedding_threads=getattr(self.config, "rag_embedding_threads", 6),
                embedding_batch_size=getattr(self.config, "rag_embedding_batch_size", 8),
                embedding_max_length=getattr(self.config, "rag_embedding_max_length", 8192),
                chunk_size=getattr(self.config, "rag_chunk_size", 500),
                chunk_overlap=getattr(self.config, "rag_chunk_overlap", 50),
                semantic_chunking=getattr(self.config, "rag_semantic_chunking", True),
                rerank_model=getattr(self.config, "rag_rerank_model", "BAAI/bge-reranker-v2-m3-int8"),
                rerank_backend=getattr(self.config, "rag_rerank_backend", "fastembed"),
                rerank_model_path=getattr(self.config, "rag_rerank_model_path", ""),
                rerank_mlx_model_path=getattr(self.config, "rag_rerank_mlx_model_path", ""),
                rerank_jina_model_path=getattr(self.config, "rag_rerank_jina_model_path", ""),
                rerank_threads=getattr(self.config, "onnx_threads", 4),
                rerank_batch_size=getattr(self.config, "rag_rerank_batch_size", 16),
                rerank_max_length=getattr(self.config, "rag_rerank_max_length", 8192),
                rerank_enabled=getattr(self.config, "rag_rerank_enabled", True),
                matryoshka_enabled=getattr(self.config, "rag_matryoshka_enabled", True),
                matryoshka_dim=getattr(self.config, "rag_matryoshka_dim", 512),
                vector_dtype=getattr(self.config, "rag_vector_dtype", "int8"),
                pdf_engine=getattr(self.config, "rag_pdf_engine", "docling"),
                # OvisOCR3 + docling 动态倍率（2026-10 审查修复：此前漏转发，
                # agent 侧 ingest 用模块默认值，配置窗口里的自定义模型路径与
                # dynamic_scale 开关对它完全不生效）。
                ocr_backend=getattr(self.config, "rag_ocr_backend", "mlx"),
                ocr_model=getattr(self.config, "rag_ocr_model", ""),
                ocr_model_path=getattr(self.config, "rag_ocr_model_path", ""),
                docling_dynamic_scale=getattr(
                    self.config, "rag_docling_dynamic_scale", False),
                docling_vlm=getattr(self.config, "rag_docling_vlm", False),
                # PDF 大小限流并发（小文件并发路数 / 大文件阈值 MB）。内存吃紧
                # 时 rag_pdf_small_concurrency=1 可退回全串行。
                pdf_small_concurrency=getattr(
                    self.config, "rag_pdf_small_concurrency", 2),
                pdf_large_threshold_mb=getattr(
                    self.config, "rag_pdf_large_threshold_mb", 20.0),
                pdf_batch_threshold_mb=getattr(
                    self.config, "rag_pdf_batch_threshold_mb", 50.0),
                pdf_batch_pages=getattr(
                    self.config, "rag_pdf_batch_pages", 50),
            )
            self._rag_engine = engine

            registered = []
            for t in [RagSearchTool(engine, aux_llm_getter=self._aux_llm_getter), RagStatusTool(engine), RagIngestTool(engine),
                      RagOutlineTool(engine)]:
                if t.name not in self._tools:
                    t.bind(self.config, self)
                    self._tools[t.name] = t
                    registered.append(t.name)
            if registered:
                self._invalidate_schema_cache()
            logger.info("RAG: tools registered: %s", ", ".join(registered))
        except Exception as e:
            logger.warning(
                "RAG 工具不可用（其余工具不受影响）: %s: %s", type(e).__name__, e
            )
            return

        # NOTE: auto-ingest is intentionally NOT performed here.
        # _register_rag_tools() runs in ToolRegistry.__init__, which is called
        # from the UI main thread (via _rebuild_agent). Synchronous ingest would
        # block the UI for minutes on first run (model download + parsing +
        # embedding of all files). The UI provides SyncKnowledgeWorker (async,
        # on a background thread) for this purpose. For non-UI usage, call
        # engine.ingest() explicitly after construction.

    def get_rag_engine(self):
        """Return the RAG engine instance, or None if not configured."""
        return self._rag_engine

    def get_skill_retriever(self):
        """Return the shared SkillRetriever (hybrid skill search), lazily built.

        The retriever scans the workspace's skill roots for skills and builds a
        LanceDB table (embeddings + BM25).  **Both layouts are indexed together** —
        ``<workspace>/skills/*/SKILL.md`` (new) and ``<workspace>/.agent/skills/*``
        (legacy skill.json + prompt.md) — so a skill is findable no matter which
        directory it lives in.  Building is lazy: the first call triggers scanning
        + indexing (can be slow for thousands of skills), so it is cached on the
        registry and rebuilt on subsequent calls only if the skills directories
        changed.
        """
        # 缓存按根集合区分：「应用配置」切换 workspace 后，复用旧目录的 retriever
        # 会检索到错误工作区的技能。
        skills_dirs = [
            os.path.join(self.config.workspace, "skills"),
            os.path.join(self.config.workspace, ".agent", "skills"),
        ]
        cache_key = "\n".join(skills_dirs)
        cache = getattr(self, "_skill_retriever_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(self, "_skill_retriever_cache", cache)
        retriever = cache.get(cache_key)
        if retriever is not None:
            return retriever
        # 首次构建是重活（扫描 + LanceDB 建表 + embedding），且 agent 线程与
        # UI 线程可能并发首次调用：不加锁会各建一个 SkillRetriever，后写者
        # 覆盖属性，先建的 LanceDB 连接泄漏（shutdown 只回收缓存里的那个）。
        with self._retriever_lock:
            retriever = cache.get(cache_key)
            if retriever is not None:
                return retriever
            # 创建后复查关闭标志（与 _get_long_memory 同源竞态）：若 shutdown() 已
            # 在读取 _shutdown=False 之后置 True 并跳过 close，新建的 retriever
            # （LanceDB 连接）将无主泄漏。立即 close 丢弃。
            if getattr(self, "_shutdown", False):
                return None
            db_path = os.path.join(self.config.workspace, ".agent", "skill_index.lancedb")
            from ..rag.vector_store import get_or_create_embedding_function
            embedding_function = get_or_create_embedding_function(
                getattr(self.config, "rag_embedding_model", "nomic-ai/nomic-embed-text-v1.5-Q"),
                backend=getattr(self.config, "rag_embedding_backend", "fastembed"),
                model_path=getattr(self.config, "rag_embedding_model_path", ""),
                threads=getattr(self.config, "onnx_threads", 4),
                batch_size=getattr(self.config, "rag_embedding_batch_size", 8),
                max_length=getattr(self.config, "rag_embedding_max_length", 8192),
            )
            from ..skill_retriever import SkillRetriever
            retriever = SkillRetriever(
                skills_dir=skills_dirs,
                db_path=db_path,
                embedding_function=embedding_function,
            )
            if getattr(self, "_shutdown", False):
                try:
                    retriever.close()
                except Exception:
                    pass
                return None
            cache[cache_key] = retriever
            setattr(self, "_skill_retriever", retriever)
            return retriever

    def reload_rag(self) -> None:
        """Refresh the RAG engine's table handle so new sync data is visible.

        Call this after SyncKnowledgeWorker finishes — the agent's RAGEngine
        holds a stale LanceDB table snapshot that won't reflect newly written
        rows until the table is re-opened.
        """
        if self._rag_engine is not None:
            try:
                self._rag_engine.reload()
            except Exception as e:
                logger.warning("ToolRegistry.reload_rag error: %s", e)

    def shutdown(self) -> None:
        """Stop all MCP server subprocesses and release RAG engine resources."""
        # 标记关闭：后台连接线程看到后不再注册工具、不再新建子进程。
        with self._tools_lock:
            self._shutdown = True
            threads = list(self._mcp_threads)
            clients = list(self._mcp_clients)
            tools = list(self._tools.values())
        # 等后台线程收手（它连接的子进程随后由下面的 stop() 回收）。
        for t in threads:
            if t.is_alive():
                t.join(timeout=2.0)
        with self._tools_lock:
            self._mcp_threads.clear()
        for client in clients:
            try:
                client.stop()
            except Exception:
                pass
        with self._tools_lock:
            self._mcp_clients.clear()
        # Close shared httpx clients in web tools to release connection pools.
        for tool in tools:
            http_client = getattr(tool, "_http", None)
            if http_client is not None:
                try:
                    http_client.close()
                except Exception:
                    pass
        # Release the RAG engine's VectorStore (LanceDB connection + FastEmbed
        # ONNX model + BGE reranker). Without this, each _rebuild_agent() call
        # leaks ~600MB of C++ heap memory.
        if self._rag_engine is not None:
            try:
                logger.info("🔌 正在释放 RAG 引擎资源 (向量库连接 + 嵌入模型 + Reranker)...")
                self._rag_engine.close()
                logger.info("✅ RAG 引擎资源已释放")
            except Exception as e:
                logger.warning("⚠ RAG 引擎资源释放异常: %s", e)
            self._rag_engine = None
        # Release the singleton LongTermMemory's GraphMemoryStore (LanceDB +
        # FastEmbed ONNX model ~137MB).  Without this, each _rebuild_agent()
        # call leaks the model loaded for the knowledge graph.
        from .memory import _LONG_MEMORY_SINGLETON_ATTR
        lm = getattr(self, _LONG_MEMORY_SINGLETON_ATTR, None)
        if lm is not None:
            try:
                lm.close()
            except Exception as e:
                logger.warning("⚠ 长期记忆资源释放异常: %s", e)
            setattr(self, _LONG_MEMORY_SINGLETON_ATTR, None)
        # Release the SkillIndex singleton(s) (if any)
        setattr(self, "_skill_index", None)
        setattr(self, "_skill_index_cache", None)
        # Release the SkillRetriever (LanceDB connection). Note: it shares the
        # process-wide embedding function singleton, which we do NOT close here.
        cache = getattr(self, "_skill_retriever_cache", None)
        retrievers = list(cache.values()) if isinstance(cache, dict) else []
        legacy = getattr(self, "_skill_retriever", None)
        if legacy is not None and legacy not in retrievers:
            retrievers.append(legacy)
        for skill_retriever in retrievers:
            try:
                skill_retriever.close()
            except Exception as e:
                logger.warning("⚠ skill retriever 资源释放异常: %s", e)
        setattr(self, "_skill_retriever", None)
        setattr(self, "_skill_retriever_cache", None)

    def get(self, name: str) -> Optional[Tool]:
        with self._tools_lock:
            return self._tools.get(name)

    def names(self) -> List[str]:
        with self._tools_lock:
            return list(self._tools.keys())

    def schemas(self) -> List[Dict[str, Any]]:
        """返回工具 schema 列表（带缓存）。

        旧实现每次调用都重建全部 schema。而 agent 每轮迭代会调它两次
        （一次算 token 估算的 md5 指纹、一次真正发请求），MCP server 多时
        是几十 KB JSON 的重复序列化。缓存由 `_invalidate_schema_cache()`
        在工具表变动时失效。
        """
        cached = getattr(self, "_schemas_cache", None)
        if cached is not None:
            return cached
        # 快照：后台线程可能正在往 _tools 里注册 MCP 工具。
        with self._tools_lock:
            tools = list(self._tools.values())
        result = [t.schema() for t in tools]
        with self._tools_lock:
            self._schemas_cache = result
        return result

    def _invalidate_schema_cache(self) -> None:
        """工具表变动后调用（默认工具注册/MCP 后台注册）。"""
        self._schemas_cache = None
        self._schema_version += 1

    @property
    def schema_version(self) -> int:
        """工具 schema 版本号（变动即递增），供 agent 侧做指纹缓存键。"""
        return self._schema_version

    @property
    def state_version(self) -> int:
        """注入系统提示的持久状态（任务计划等）的版本号。"""
        return self._state_version

    def bump_state_version(self) -> None:
        """状态型工具改写了注入系统提示的数据后调用。

        Agent 的 ``_system_prompt`` 以 turn 为粒度缓存；不 bump 的话，
        同一次 ``run()`` 里 ``task_plan`` 刚创建/更新的计划在后续迭代的
        请求里不可见（缓存返回旧提示），「每轮都能看到计划进度」的
        设计意图就落空了。
        """
        self._state_version += 1

    # ------------------------------------------------------------------
    def execute(self, name: str, args: Dict[str, Any]) -> ToolResult:
        with self._tools_lock:
            tool = self._tools.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return ToolResult(
                success=False,
                error=f"Invalid arguments for {name}: expected a JSON object, "
                      f"got {type(args).__name__}",
            )

        # 按 JSON Schema 做一次宽容的类型归一：模型经常把数字写成字符串
        # （"top_k": "5"、"timeout": "30"）。直接传给 run() 会让工具内部
        # 的算术/比较炸掉，或者被 int(x or 5) 静默吞成默认值。这里只在
        # 转换无歧义时改写，**不**因为类型不符而拒绝调用（严格校验会让
        # 模型陷入「参数无效」的重试循环）。
        args = _coerce_arg_types(tool, args)

        # 签名级预检：把「模型传错参数」和「工具内部抛 TypeError」区分开。
        # 旧实现靠捕获 run() 抛出的 TypeError 一律报 "Invalid arguments"——
        # 工具内部代码出 bug（比如对 None 调 .strip()）时模型会收到"参数
        # 无效"，于是反复重试同样的调用而不是上报问题。
        try:
            inspect.signature(tool.run).bind(**args)
        except TypeError as e:
            return ToolResult(success=False, error=f"Invalid arguments for {name}: {e}")

        try:
            return tool.run(**args)
        except TypeError as e:
            # 签名已通过 → 这里的 TypeError 是工具内部缺陷，如实报告。
            logger.exception("tool %s raised TypeError internally (args=%r)", name, args)
            return ToolResult(
                success=False,
                error=f"{name} failed internally (not an argument problem): "
                      f"{type(e).__name__}: {e}",
            )
        except Exception as e:  # pragma: no cover - defensive
            return ToolResult(success=False, error=f"{type(e).__name__}: {e}")


def _coerce_arg_types(tool: Tool, args: Dict[str, Any]) -> Dict[str, Any]:
    """按工具的 JSON Schema 把参数值归一到声明类型（宽容，不拒绝）。

    只处理两种最常见的模型输出偏差：
      * ``"5"``（字符串）→ ``5``（integer/number）
      * ``"true"``/``"false"`` → bool
    无法安全转换时保持原值不动——严格校验会让模型陷入「参数无效」的
    重试循环，而工具内部大多已有自己的归一化（如 int(top_k or 5)）。
    """
    schema = getattr(tool, "parameters", None)
    if not isinstance(schema, dict):
        return args
    props = schema.get("properties")
    if not isinstance(props, dict):
        return args
    coerced: Dict[str, Any] = {}
    for key, value in args.items():
        spec = props.get(key)
        if not isinstance(spec, dict) or not isinstance(value, str):
            coerced[key] = value
            continue
        declared = spec.get("type")
        raw = value.strip()
        if declared == "integer" or declared == "number":
            try:
                coerced[key] = int(raw) if declared == "integer" else float(raw)
                continue
            except ValueError:
                pass
        elif declared == "boolean":
            low = raw.lower()
            if low in ("true", "1", "yes", "on"):
                coerced[key] = True
                continue
            if low in ("false", "0", "no", "off"):
                coerced[key] = False
                continue
        coerced[key] = value
    return coerced


def parse_args(raw: str) -> Dict[str, Any]:
    """Parse a tool-call arguments JSON string.

    Returns ``{}`` for empty input (a model may legitimately call a no-arg
    tool with ``""``). For non-empty but malformed JSON, raises ``ValueError``
    with a clear message so the agent can report it back to the model —
    silently returning ``{}`` here would turn a JSON typo into a confusing
    "missing required argument" error one layer up, which the model cannot
    easily diagnose.
    """
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in tool arguments ({e.msg} at pos {e.pos}): {raw[:200]!r}") from e

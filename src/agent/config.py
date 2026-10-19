"""Configuration for the agent."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Runtime configuration. Persisted to ~/.base-agent/config.json by the UI."""

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    # 主模型既可使用 OpenAI-compatible 网络端点，也可在应用进程内直接加载。
    # 本地后端均为懒加载，应用配置时不会在 UI 线程同步读入数 GB 权重。
    llm_backend: str = "openai"  # openai | mlx_lm | mlx_vlm | onnx | llama_cpp
    llm_model_path: str = "models/qwen3.5-4b-agentic-coder-v4-4bit-mlx"
    llm_vision_projection_path: str = ""  # GGUF VLM 的 mmproj/clip 文件
    llm_chat_format: str = ""
    llm_vlm_handler: str = "mtmd"
    local_model_threads: int = 4
    local_model_batch_size: int = 512
    local_gpu_layers: int = -1
    onnx_threads: int = 4
    temperature: float = 0.7
    max_tokens: int = 32768
    top_p: float = 0.95
    min_p: float = 0.05
    top_k: int = 20
    repetition_penalty: float = 1.0
    workspace: str = field(default_factory=lambda: os.path.expanduser("~/base-agent-workspace"))
    # Default to <workspace>/knowledge_base so RAG tools are available out of
    # the box.  Users can override via the UI or config.json.
    knowledge_base: str = ""
    request_timeout: float = 120.0
    max_iterations: int = 15
    max_history: int = 50  # max messages retained in conversation history
    # Context window budget (tokens). When the estimated prompt size reaches
    # 90% of this value, the agent proactively summarizes older messages and
    # persists the summary to memory.md. Also used to recover from
    # context_length_exceeded errors returned by the LLM API.
    # Default matches max_tokens (32768) so the shrink threshold tracks the
    # model's actual context window.
    max_context_tokens: int = 32768
    # Shrink ratio: trigger proactive shrink when estimated context size
    # exceeds this fraction of max_context_tokens.
    context_shrink_ratio: float = 0.9

    # RAG settings
    rag_chunk_size: int = 500       # tokens per chunk
    rag_chunk_overlap: int = 50     # 10% of chunk_size
    # 默认走本地 MLX 直载（实测 Embedding 3.7x、Reranker 2.5x 于 fastembed CPU）。
    # MLX 不可用时（非 Apple Silicon / 未安装 / 路径缺失）自动回退 fastembed，
    # 不会出现「默认配置直接不可用」。
    rag_embedding_backend: str = "mlx"  # mlx | fastembed | onnx | llama_cpp
    rag_embedding_model: str = "nomic-ai/nomic-embed-text-v1.5-Q"  # MLX 回退用
    rag_embedding_model_path: str = "models/nomic-embed-text-v1.5-mlx"
    # 子批扫档实测（MLX）：2→88、8→152、16→178、32→205、64→205、128→190 条/s。
    # 默认取拐点 16；下限 2 保留极端内存压力下的避让位，上限 64 之后无收益。
    rag_embedding_batch_size: int = 16
    rag_embedding_max_length: int = 512
    rag_rerank_backend: str = "mlx"  # mlx | fastembed | onnx | disabled
    rag_rerank_model: str = "BAAI/bge-reranker-v2-m3-int8"  # MLX 回退用
    rag_rerank_model_path: str = "models/bge-reranker-v2-m3-ONNX/model_quantized.onnx"
    # MLX 交叉编码器是目录而非 .onnx 文件，与上面的 ONNX 路径分开保存，
    # 这样在 UI 里切换后端不必重新粘贴路径。
    rag_rerank_mlx_model_path: str = "models/bge-reranker-v2-m3-mlx-mxfp8"
    rag_rerank_batch_size: int = 16
    rag_rerank_max_length: int = 512
    rag_rerank_enabled: bool = True
    rag_auto_ingest: bool = True
    # PDF 解析引擎（2026-10 新增，与 docling 并行的 marker-pdf 路线）：
    #   - "docling"：默认，版面理解 + ocrmac(Apple Vision) OCR
    #   - "marker"：marker-pdf 2.x（surya OCR，服务端进程架构）
    # 主引擎不可用（未安装/导入失败）时自动回退到另一引擎；单文件转换
    # 失败仍走 parsers.py 的轻量兜底，行为与 #133 之前一致。
    rag_pdf_engine: str = "docling"

    # Browser endpoint for webexec_js (Playwright-like bridge). Optional.
    browser_endpoint: str = ""

    # MCP servers – list of {name, command, args?, env?} dicts.
    # Example:
    #   [{"name": "filesystem", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]}]
    mcp_servers: list = field(default_factory=list)

    # 数值字段的类型，用于把 config.json 里的字符串/浮点值强制转换回正确类型。
    # 手工编辑过的 config.json 常出现 "32768"（字符串）或 32768.0（浮点），
    # 直接喂给 dataclass 会让后续算术/比较在运行时炸掉。
    _INT_FIELDS = (
        "max_tokens", "top_k", "max_iterations", "max_history",
        "max_context_tokens", "rag_chunk_size", "rag_chunk_overlap",
        "local_model_threads", "local_model_batch_size", "local_gpu_layers",
        "onnx_threads", "rag_embedding_batch_size", "rag_embedding_max_length",
        "rag_rerank_batch_size", "rag_rerank_max_length",
    )
    _FLOAT_FIELDS = ("temperature", "top_p", "min_p", "repetition_penalty",
                     "request_timeout", "context_shrink_ratio")
    _STR_FIELDS = (
        "base_url", "api_key", "model", "workspace", "knowledge_base",
        "llm_backend", "llm_model_path", "llm_vision_projection_path",
        "llm_chat_format", "llm_vlm_handler",
        "rag_embedding_backend", "rag_embedding_model", "rag_embedding_model_path",
        "rag_rerank_backend", "rag_rerank_model", "rag_rerank_model_path",
        "rag_rerank_mlx_model_path",
        "browser_endpoint", "rag_pdf_engine",
    )
    _BOOL_FIELDS = ("rag_rerank_enabled", "rag_auto_ingest")
    # 业务上不允许为空的字符串字段（空串 → 视为缺失，回退到默认值）。
    # knowledge_base / API key / mmproj / chat format 可以为空。
    _NON_EMPTY_STR_FIELDS = (
        "base_url", "model", "workspace", "llm_backend", "llm_model_path",
        "rag_embedding_backend", "rag_embedding_model", "rag_embedding_model_path",
        "rag_rerank_backend", "rag_rerank_model", "rag_rerank_model_path",
        "rag_rerank_mlx_model_path",
        "rag_pdf_engine",
    )

    @classmethod
    def _coerce(cls, name: str, value):
        """Force a raw JSON value into the declared field type.

        Returns ``None`` when the value cannot be coerced, so the caller can
        fall back to the dataclass default instead of storing a wrong type.
        """
        # null 一律视为「字段缺失」：str(None) 会产生字符串 "None"，随后
        # base_url + "/models" 之类的拼接会把它当真实值用；workspace = "None"
        # 更是会在当前目录创建一个名为 None 的目录。
        if value is None:
            return None
        try:
            if name in cls._INT_FIELDS:
                return int(value)
            if name in cls._FLOAT_FIELDS:
                return float(value)
            if name in cls._STR_FIELDS:
                s = str(value)
                # 关键路径字段不能是空串：workspace="" 会让 ensure_workspace()
                # 的 os.makedirs("") 抛 FileNotFoundError；base_url="" 会把请求
                # 打到 "/chat/completions"。这里统一按「字段缺失」处理，让调用
                # 方回退到 dataclass 默认值。
                if not s.strip() and name in cls._NON_EMPTY_STR_FIELDS:
                    return None
                return s
            if name in cls._BOOL_FIELDS:
                # bool("false") 是 True——手工编辑配置时常见的 "false"/"no"/"0"
                # 字符串会把用户显式的关闭悄悄变成打开。只接受明确的映射。
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)):
                    return bool(value)
                s = str(value).strip().lower()
                if s in ("true", "1", "yes", "on"):
                    return True
                if s in ("false", "0", "no", "off", ""):
                    return False
                return None
        except (TypeError, ValueError):
            return None
        return value

    @classmethod
    def load(cls, path: str) -> "AgentConfig":
        import json

        if not os.path.exists(path):
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("config root is not an object")
        except (OSError, json.JSONDecodeError, ValueError) as e:
            # 配置文件损坏（写一半崩溃 / 手工编辑出错）时不能让程序起不来：
            # 备份坏文件后用默认配置启动，用户重新「应用配置」即可恢复。
            logger.warning("config: %s 无法解析 (%s)，备份后使用默认配置", path, e)
            try:
                # 优先用固定名 .corrupt（既有行为/测试依赖）；若已存在则加
                # 时间戳，避免第二次损坏覆盖掉第一次的备份。
                backup = path + ".corrupt"
                if os.path.exists(backup):
                    backup = f"{backup}.{int(time.time())}"
                os.replace(path, backup)
            except OSError:
                pass
            return cls()

        known = {k for k in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {}
        for k, v in data.items():
            if k not in known:
                continue
            coerced = cls._coerce(k, v)
            if coerced is not None:
                kwargs[k] = coerced
        cfg = cls(**kwargs)
        cfg._migrate()
        return cfg

    def _migrate(self) -> None:
        """Normalize obsolete config values from older versions.

        Runs after loading from disk so that stale values saved before a
        migration don't break the current code. Silently rewrites the
        in-memory config; the next ``save()`` persists the fix.
        """
        # Pre-LanceDB era used sentence-transformers model names or
        # ChromaDB defaults that FastEmbed doesn't recognize. Map any
        # unsupported embedding model to the current default.
        _DEPRECATED_EMBEDDING_MODELS = {
            "all-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-small-zh-v1.5",
            "BAAI/bge-large-zh-v1.5",
            "",
        }
        if self.rag_embedding_model in _DEPRECATED_EMBEDDING_MODELS:
            self.rag_embedding_model = "nomic-ai/nomic-embed-text-v1.5-Q"
        # Non-quantized nomic-embed-text-v1.5 (~548MB) → quantized -Q (~137MB).
        # Same 768-dim embeddings, 4x smaller download, same quality.
        elif self.rag_embedding_model == "nomic-ai/nomic-embed-text-v1.5":
            self.rag_embedding_model = "nomic-ai/nomic-embed-text-v1.5-Q"
        # 重排序已迁移到 fastembed ONNX 栈：旧默认 bge-reranker-base 是
        # FlagEmbedding 时代的 fp32 全量模型（~1.1GB），默认值统一升级到
        # 官方 INT8 版（~570MB，中文精排更强）。仅重写旧默认值——手工显式
        # 配置的其他模型名不动。沿用 max_tokens=4096 的迁移先例。
        if self.rag_rerank_model == "BAAI/bge-reranker-base":
            self.rag_rerank_model = "BAAI/bge-reranker-v2-m3-int8"
        # Old LLMClient builds defaulted to max_tokens=4096. Only rewrite the
        # value when it is still that legacy default — an explicitly chosen
        # small budget (e.g. 4096 for a local 4K model) must survive; silently
        # bumping every sub-8192 value would fight the user on every start.
        if self.max_tokens == 4096:
            self.max_tokens = 32768
        # max_context_tokens was introduced at 32000; older configs that
        # saved a stale small value should also be bumped. Track max_tokens
        # so the shrink threshold follows the model's actual window.
        if self.max_context_tokens < 8192:
            self.max_context_tokens = self.max_tokens

        if self.llm_backend not in ("openai", "mlx_lm", "mlx_vlm", "onnx", "llama_cpp"):
            self.llm_backend = "openai"
        if self.rag_embedding_backend not in ("fastembed", "mlx", "onnx", "llama_cpp"):
            self.rag_embedding_backend = "fastembed"
        if self.rag_rerank_backend not in ("fastembed", "mlx", "onnx", "disabled"):
            self.rag_rerank_backend = "fastembed"
        self.local_model_threads = max(1, self.local_model_threads)
        self.local_model_batch_size = max(1, self.local_model_batch_size)
        self.onnx_threads = max(1, self.onnx_threads)
        # 批大小夹在 [2, 64]：低于 2 吞吐腰斩，高于 64 无收益且内存无谓增长。
        self.rag_embedding_batch_size = max(2, min(64, self.rag_embedding_batch_size))
        self.rag_embedding_max_length = max(8, self.rag_embedding_max_length)
        self.rag_rerank_batch_size = max(1, self.rag_rerank_batch_size)
        self.rag_rerank_max_length = max(8, self.rag_rerank_max_length)

    def save(self, path: str) -> None:
        import json

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # 原子写：先写临时文件再 os.replace。旧实现直接 open(path, "w")，
        # 写一半崩溃（断电/强杀）会留下截断的 JSON，下次启动 load() 直接抛
        # JSONDecodeError 导致程序无法启动。
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.__dict__, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            # 写失败时必须清掉临时文件：否则每次失败都在配置目录里留一个
            # .tmp 垃圾文件（json.dump 对不可序列化值失败是会发生的）。
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def ensure_workspace(self) -> str:
        os.makedirs(self.workspace, exist_ok=True)
        return self.workspace

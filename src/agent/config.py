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
    # 主模型两种工作方式（UI 顶层二选一，#140）：
    #   - "local"：进程内直载本地权重，模型类型由 llm_backend 细分
    #     （mlx_lm / mlx_vlm / onnx / llama_cpp）；
    #   - "openai"：OpenAI 兼容接口，此时 llm_backend 恒为 "openai"，
    #     本地路径字段被忽略。
    # 默认本地模式 + 默认加载 models/qwen3.5-4b-agentic-coder-v4-4bit-mlx。
    llm_mode: str = "local"  # local | openai
    # 主模型既可使用 OpenAI-compatible 网络端点，也可在应用进程内直接加载。
    # 本地后端均为懒加载，应用配置时不会在 UI 线程同步读入数 GB 权重。
    llm_backend: str = "mlx_lm"  # openai | mlx_lm | mlx_vlm | onnx | llama_cpp
    llm_model_path: str = "models/qwen3.5-4b-agentic-coder-v4-4bit-mlx"
    llm_vision_projection_path: str = ""  # GGUF VLM 的 mmproj/clip 文件
    llm_chat_format: str = ""
    llm_vlm_handler: str = "mtmd"
    # 思考模式（本地模式主模型）。走 chat template 的 enable_thinking：
    # True 时模板在 generation prompt 末尾预置 <think> 开标签，模型先输出
    # 思考再作答（思考过程显示在「思考过程」面板）；False 时直接作答。
    # 仅在模板支持该参数时生效，不支持的模板会自动退掉该 kwarg。
    llm_enable_thinking: bool = True
    # 辅助模型（#140）：只支持本地直载，供上下文摘要等轻量子任务使用，
    # 避免为一次摘要把主模型的推理队列占住。默认 Qwen3.5-2B-MLX-4bit。
    aux_llm_backend: str = "mlx_lm"  # mlx_lm | mlx_vlm | onnx | llama_cpp
    aux_llm_model_path: str = "models/Qwen3.5-2B-MLX-4bit"
    aux_llm_vision_projection_path: str = ""
    aux_llm_chat_format: str = ""
    aux_llm_vlm_handler: str = "mtmd"
    # 辅助模型的思考模式。**默认关闭**：辅助模型只跑摘要/归纳这类轻量子任务，
    # 要的是快与稳；开思考会多出一大段 reasoning 文本，显著拖慢摘要且对
    # 归纳质量无实质帮助。主模型默认开（llm_enable_thinking=True）。
    aux_llm_enable_thinking: bool = False
    # 辅助模型独立采样参数（不与主模型共用——主模型要的是生成质量，
    # 辅助模型要的是快与稳，温度/生成长度都不同）。
    aux_temperature: float = 0.5
    aux_max_tokens: int = 2048
    # 辅助模型的上下文窗口。只做摘要/归纳等轻量子任务，不需要主模型那么大的
    # 窗口，默认 16384——足够容纳主对话的历史分片，又不至于把 KV 缓存占满。
    aux_max_context_tokens: int = 16384
    aux_top_p: float = 0.95
    aux_min_p: float = 0.05
    aux_top_k: int = 20
    aux_repetition_penalty: float = 1.0
    aux_request_timeout: float = 120.0
    local_model_threads: int = 4
    local_model_batch_size: int = 512
    local_gpu_layers: int = -1
    # llama.cpp (GGUF) 专用：KV cache 大小（n_ctx）。llama.cpp 会在**加载时一次性
    # 预分配**整个 KV cache，随 n_ctx 线性膨胀——7B 模型 n_ctx=32768 约预占 17GB
    # 统一内存，直接把 16G 机器吃爆（表现就是「发句 hi 内存就冲到 20G+ 卡死」）。
    # 该值独立于 max_context_tokens（逻辑上下文窗口，MLX/API 后端按需增长 KV、
    # 不受影响），只约束 llama.cpp 的预分配上限。16384 对 7B 约 8.6GB、13B 约
    # 13GB，16G 机器可承受；内存更小的机器可调到 8192，显存更大的可调大。
    local_llm_n_ctx: int = 16384
    # MLX 专用：prefill 分块大小（一次前向喂多少 token）。越大越快（少几次
    # kernel 启动与同步），峰值内存越高；太小则长 prompt 的 prefill 被切成很多
    # 块、白白多付同步开销。2861 token 的预热前缀一次喂完完全没问题。
    local_llm_prompt_batch_size: int = 4096
    # 投机解码（speculative decoding）：用**已加载的辅助模型**当草稿模型给主模型
    # 起草，主模型一次性并行验证，不额外加载任何东西。草稿模型必须与主模型同
    # 词表（默认主/辅助都是 qwen3.5，vocab 248320，天然匹配）。
    #
    # **默认关闭**：mlx_lm 的投机解码要求 prompt cache 可裁剪，而 qwen3.5 是混合
    # 架构、线性层用 ArraysCache（不可裁剪），与本项目的 KV 前缀缓存互斥——同时
    # 用会直接抛
    #   ValueError: Speculative decoding requires a trimmable prompt cache (got {'ArraysCache'})
    # 实测：前缀缓存省掉的是每轮 ~2900 token 的 prefill（约 16s），远大于投机解码
    # 的生成收益。只有在不传 prompt_cache 的场合（纯 transformer 模型、或放弃前缀
    # 复用）才打开。
    llm_speculative: bool = False
    # 每轮让草稿模型先起草多少个 token（mlx_lm 的 num_draft_tokens）。太大反而
    # 会因为拒绝率高而浪费算力，2~4 是常见甜点区。
    llm_num_draft_tokens: int = 3
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
    # nomic-embed-text-v1.5 是旋转位置编码，训练长度 2048、靠 RoPE 外推到
    # n_positions=8192（本地 MLX 权重 config 即如此）；绝对位置嵌入模型（如
    # bge-small-zh，512）会在后端加载时按自己 config 夹紧。默认按 nomic 设 8192。
    rag_embedding_max_length: int = 8192
    rag_rerank_backend: str = "mlx"  # mlx | fastembed | onnx（UI 无 disabled 项）
    rag_rerank_model: str = "BAAI/bge-reranker-v2-m3-int8"  # MLX 回退用
    rag_rerank_model_path: str = "models/bge-reranker-v2-m3-ONNX/model_quantized.onnx"
    # MLX 交叉编码器是目录而非 .onnx 文件，与上面的 ONNX 路径分开保存，
    # 这样在 UI 里切换后端不必重新粘贴路径。
    rag_rerank_mlx_model_path: str = "models/bge-reranker-v2-m3-mlx-mxfp8"
    rag_rerank_batch_size: int = 16
    # bge-reranker-v2-m3 的位置嵌入上限是 8192（本地 MLX/ONNX 权重 config 均为
    # 8194，fastembed tokenizer 本身截断在 8192）。旧版 bge-reranker-base 只有
    # 512，各后端加载时会按自己的 config.json 夹紧，所以这里按 v2-m3 设上限。
    rag_rerank_max_length: int = 8192
    # 精排默认启用且 UI 不再暴露开关（下拉框已无 disabled）。这里保留字段，
    # 便于在配置文件里手动写 rag_rerank_enabled=false 临时关掉精排——
    # 此时 _rerank_backend 会被置为 disabled，检索退化为向量距离排序。
    rag_rerank_enabled: bool = True

    rag_auto_ingest: bool = True
    # PDF 解析引擎（2026-10 新增，与 docling 并行的 marker-pdf 路线）：
    #   - "docling"：默认，版面理解 + ocrmac(Apple Vision) OCR
    #   - "marker"：marker-pdf 2.x（surya OCR，服务端进程架构）
    #   - "ovis"：PyMuPDF + OvisOCR2 VLM（第三方案，同时是 PDF/图像的
    #     轻量兜底，2026-10 起取代 PyMuPDF+RapidOCR 组合）
    # 主引擎不可用（未安装/导入失败）时自动回退到另一引擎；单文件转换
    # 失败仍走轻量兜底，行为与 #133 之前一致。
    rag_pdf_engine: str = "docling"
    # OvisOCR2 模型配置（ovis 引擎 / PDF 兜底共用）：
    #   - rag_ocr_backend：mlx | onnx | gguf
    #       · mlx  → MLX-VLM 权重（默认 OvisOCR2-0.8B-MLX-4bit）
    #       · onnx → ONNX 多模态导出（含视觉+文本联合图），用 rag_ocr_model_path
    #       · gguf → llama.cpp GGUF 多模态模型（需 mmproj 投影文件才能看图）
    #   - rag_ocr_model / rag_ocr_model_path：MLX / ONNX 共用（目录或 .onnx 文件）
    #   - rag_ocr_gguf_path：GGUF 模型文件（GGUF 后端专用）
    #   - rag_ocr_mmproj_path：GGUF 多模态的 mmproj / clip 投影文件（必填才能读图）
    #   - rag_ocr_vlm_handler：llama.cpp VLM handler（mtmd/llava15/llava16/
    #     qwen25vl/minicpmv26/moondream/nanollava），默认 mtmd
    #   - rag_ocr_chat_format：llama.cpp chat_format（如 qwen2-vl，模型需要的
    #     话填；多数 mtmd handler 可留空）
    rag_ocr_backend: str = "mlx"
    rag_ocr_model: str = "OvisOCR2-0.8B-MLX-4bit"
    rag_ocr_model_path: str = "models/OvisOCR2-0.8B-MLX-4bit"
    rag_ocr_gguf_path: str = ""
    rag_ocr_mmproj_path: str = ""
    rag_ocr_vlm_handler: str = "mtmd"
    rag_ocr_chat_format: str = ""
    # docling 动态 OCR 倍率（逐页分析：表格页升 ocr_options.scale=2.5）。
    # 2026-10 实测（5 页中文扫描书，MPS + ocrmac，离线计时）：表格密集件
    # 耗时 +87%（20.0s vs 10.7s），而 char 级质量收益不稳定（输出字符数
    # 完全一致），且目录页装饰线会被误判成表格页——故默认关闭，全篇固定
    # scale=2.0。需要表格高保真的用户在配置文件里开 true。
    rag_docling_dynamic_scale: bool = False

    # PDF 转换按文件大小限流（docling / marker 两个转换器共用一套参数）：
    #   - rag_pdf_small_concurrency：小于阈值的「小文件」最多几路并发转换。
    #     2 = 同时持两个转换器实例（小文件吞吐约 2x），但 MPS 上两份模型
    #     权重常驻、并发 OCR 的峰值统一内存也约 2x。同步时若系统内存压力
    #     大 / 疯狂 swap，把它设成 1 即退回全串行（与加此功能前一致）。
    #   - rag_pdf_large_threshold_mb：大于等于该大小的文件视为「大文件」，
    #     独占单实例串行转换——大文件（扫描书）单册峰值内存就很高，并发
    #     两份会把统一内存撑爆。
    rag_pdf_small_concurrency: int = 2
    rag_pdf_large_threshold_mb: float = 20.0

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
        "local_llm_n_ctx", "local_llm_prompt_batch_size", "llm_num_draft_tokens",
        "onnx_threads","rag_embedding_batch_size", "rag_embedding_max_length",
        "rag_rerank_batch_size", "rag_rerank_max_length",
        "rag_pdf_small_concurrency",
        "aux_max_tokens", "aux_top_k", "aux_max_context_tokens",
    )
    _FLOAT_FIELDS = ("temperature", "top_p", "min_p", "repetition_penalty",
                     "request_timeout", "context_shrink_ratio",
                     "rag_pdf_large_threshold_mb",
                     "aux_temperature", "aux_top_p", "aux_min_p",
                     "aux_repetition_penalty", "aux_request_timeout")
    _STR_FIELDS = (
        "base_url", "api_key", "model", "workspace", "knowledge_base",
        "llm_mode",
        "llm_backend", "llm_model_path", "llm_vision_projection_path",
        "llm_chat_format", "llm_vlm_handler",
        "aux_llm_backend", "aux_llm_model_path",
        "aux_llm_vision_projection_path", "aux_llm_chat_format",
        "aux_llm_vlm_handler",
        "rag_embedding_backend", "rag_embedding_model", "rag_embedding_model_path",
        "rag_rerank_backend", "rag_rerank_model", "rag_rerank_model_path",
        "rag_rerank_mlx_model_path",
        "browser_endpoint", "rag_pdf_engine",
        "rag_ocr_backend", "rag_ocr_model", "rag_ocr_model_path",
        "rag_ocr_gguf_path", "rag_ocr_mmproj_path",
        "rag_ocr_vlm_handler", "rag_ocr_chat_format",
    )
    _BOOL_FIELDS = ("rag_rerank_enabled", "rag_auto_ingest",
                    "rag_docling_dynamic_scale",
                    "llm_enable_thinking", "aux_llm_enable_thinking",
                    "llm_speculative")
    # 业务上不允许为空的字符串字段（空串 → 视为缺失，回退到默认值）。
    # knowledge_base / API key / mmproj / chat format 可以为空。
    _NON_EMPTY_STR_FIELDS = (
        "base_url", "model", "workspace", "llm_mode",
        "llm_backend", "llm_model_path",
        "aux_llm_backend", "aux_llm_model_path",
        "rag_embedding_backend", "rag_embedding_model", "rag_embedding_model_path",
        "rag_rerank_backend", "rag_rerank_model", "rag_rerank_model_path",
        "rag_rerank_mlx_model_path",
        "rag_pdf_engine",
        "rag_ocr_backend", "rag_ocr_model", "rag_ocr_model_path",
        "rag_ocr_gguf_path", "rag_ocr_mmproj_path",
        "rag_ocr_vlm_handler", "rag_ocr_chat_format",
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
        # 兜底：未登记在上面的字段，按 dataclass 声明的类型强制转换。这样新增
        # int/float/bool 字段时不必同步维护上面的清单（遗漏会让 "32768" 字符串
        # 或 32768.0 浮点原样进 dataclass，后续算术/比较在运行时炸掉）。
        try:
            ftype = cls.__dataclass_fields__.get(name)
            if ftype is not None:
                tname = str(ftype.type)
                if "int" in tname and "float" not in tname:
                    return int(value)
                if "float" in tname:
                    return float(value)
                if tname == "bool" or "bool" in tname:
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
        # llm_mode 是后加字段（#140）：旧配置里没有它。此时不能直接用
        # dataclass 默认的 "local"——那会把一个一直用 OpenAI 接口的老用户
        # 静默切到本地 MLX。按旧配置里真实生效的 llm_backend 反推模式。
        if "llm_mode" not in data:
            cfg.llm_mode = "openai" if cfg.llm_backend == "openai" else "local"
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
        # 辅助模型默认温度 0.2 → 0.5：只重写仍是旧默认的值，用户显式调过的
        # 温度不动（沿用上面 max_tokens 的迁移先例）。
        if self.aux_temperature == 0.2:
            self.aux_temperature = 0.5
        # aux_max_context_tokens 是后加的字段：旧配置走 dataclass 默认 16384；
        # 这里只兜「被设成比生成长度还小」的非法值，避免 llama.cpp n_ctx 之类
        # 把窗口卡死。
        if self.aux_max_context_tokens < 2048:
            self.aux_max_context_tokens = 16384

        if self.llm_mode not in ("local", "openai"):
            self.llm_mode = "local"
        if self.llm_backend not in ("openai", "mlx_lm", "mlx_vlm", "onnx", "llama_cpp"):
            self.llm_backend = "openai"
        # 模式与后端必须自洽：OpenAI 接口模式下 llm_backend 只能是 openai；
        # 本地模式下绝不可能是 openai（否则路径字段被忽略、模型 silently
        # 走网络，用户在配置窗口看到的「本地模式」与实际行为相反）。
        if self.llm_mode == "openai":
            self.llm_backend = "openai"
        elif self.llm_backend == "openai":
            self.llm_mode = "local"
            self.llm_backend = "mlx_lm"
        # 辅助模型只支持本地直载：没有网络端点可配，非法值一律回退 mlx_lm。
        if self.aux_llm_backend not in ("mlx_lm", "mlx_vlm", "onnx", "llama_cpp"):
            self.aux_llm_backend = "mlx_lm"
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

# Base Agent

本地自主 Agent，基于 PyQt6 图形界面，支持 OpenAI 兼容的大语言模型，内置可插拔工具系统和本地 RAG 知识库。

## 功能特性

- **多模型兼容** — 支持所有 OpenAI 兼容 API（OpenAI / DeepSeek / Qwen / 本地 vLLM 等）
- **规划与反思** — 非平凡任务先建结构化计划（`task_plan`：目标 + 步骤 + 状态），计划持久化并注入系统提示，长任务/上下文收缩后目标不丢失；四道防停滞防线：空响应注入反思重试、相同调用连续失败附重试指引、连续多次失败强制复盘、迭代预算耗尽强制收尾总结——杜绝任务半路跑丢或静默终止
- **流式对话** — 实时显示推理过程（thinking trace）和生成内容，支持 token 速度实时统计（仅计输出 token，不含 prompt）；结构化 read 超时（流式 chunk 间隔上限 60s），本地模型生成中途卡住时自动报错而非永久卡死
- **工具调用** — 内置 16 个工具（13 通用 + 3 RAG；`webexec_js` 仅在配置浏览器端点时注册），支持模型自主决策和工具链式调用
- **知识图谱记忆** — 长期记忆采用知识图谱（实体 + 关系 + 观察），对话结束后自动用 LLM 抽取事实入图谱（异步 daemon 线程，不阻塞用户回复）；观察通过 FastEmbed + LanceDB 语义检索，精简注入 prompt 避免全量灌入拖慢响应
- **本地 RAG 知识库** — 自动解析文档、清洗、切片、向量化，BGE 重排序精准召回
- **增量同步** — 通过 manifest.json 记录文件签名（mtime + size + content_hash），未修改文件完全跳过；右键「同步知识」可强制全量重处理。manifest 写入走**跨进程文件锁（flock）+ 读改写合并**，多个入口（UI 同步 / agent 自动 ingest）并发时不会互相抹掉条目
- **上下文收缩** — 估计 prompt 达到 90% 上下文窗口时主动摘要旧消息；遇到 `context_length_exceeded` 错误时被动收缩重试；摘要持久化到 `workspace/memory.md`
- **依赖自动管理** — 同步知识库前自动扫描文件类型、检查并安装缺失的解析依赖
- **全链路日志** — 知识同步的每个关键节点（文件解析 → 清洗 → Markdown 转换 → 切片 → 向量化 → 入库 → 清理 → Manifest 保存）都通过 QTimer 轮询 + 线程安全日志缓冲区实时显示在右侧日志面板中；知识检索时展示向量检索候选数、BGE Reranker 精排过程和最终结果排名
- **协作式取消** — 同步过程中可随时停止，worker 在文件/批处理边界安全退出；已入库数据不丢失，Manifest 保持一致性。`code_run` 走独立子进程组，超时时 **SIGTERM → SIGKILL killpg** 真正杀掉整棵进程树（线程版只能等进程退出），同时立起取消标志文件供 `_cancelled()` 协作退出
- **内存管理** — Embedding 模型进程级单例共享（RAG + 知识图谱共用同一 FastEmbed 实例，~137MB 而非 ~274MB）；LongTermMemory 单例缓存避免重复加载；`ToolRegistry.shutdown()` 统一释放 RAG 引擎 + 知识图谱 LanceDB 连接 + ONNX 模型 + BGE reranker；ONNX Runtime InferenceSession 共享时不 release（避免破坏其他调用方）；RapidOCR 为**实例池（最多 2 份）**、docling/marker/ovis 转换器全局单例且同步结束后显式释放（docling 小文件并发数默认 1 = 单实例串行，为省 ~1.9GB 统一内存；需吞吐可把 `rag_pdf_small_concurrency` 调回 2）；转换器「取+用」在同一把锁内原子完成、释放前非阻塞抢推理锁（避免两代模型并存，#141/#144）；`_JsonStore` 用单一全局 atexit 回调 + WeakSet 统一 flush（避免每实例注册一条永不注销的回调）；窗口关闭时统一清理 QThread worker、logger handler、主/辅助 LLM 后端与 terminate 路径残留的 RAG 引擎；`VectorStore` 用**在途操作计数 + 条件变量**保证 `clear()`/`close()`/`reload()` 不与在途检索/写入并发；「应用配置」的重资源释放移到后台线程，不冻结界面
- **健壮性** — 子进程统一 `start_new_session` 独立进程组，超时/停止时用 `killpg` 回收整棵树（`shell_run` 不留孤儿、MCP 不留僵尸）；`code_run` 在**受限子进程**中执行且写路径防护覆盖 `builtins.open` / `io.open` / `os.*` 写类操作 / `pathlib.Path` / `shutil`，`rename`/`link`/`symlink` 的**目标路径**同样校验；确认弹窗默认拒绝（超时不会自动放行高风险操作）；`skill_load` 拒绝绝对路径/`..` 入口；`config.json` 原子写入 + 损坏自动回退默认配置；`web_scan` 限制 http/https、拦截云元数据地址（SSRF）并限制响应体大小
- **Prompt 优化** — SYSTEM_PROMPT 精简至 ~200 tok（去掉模型已知的协议说明与冗余反例，补充「语义检索 query 用原始语言、勿翻译/音译」与「工具报错后改参数重试、两次失败即换方法」两条高价值约束）；Knowledge base 状态条件注入（无 KB 时省 tok）；`webexec_js` 条件注册（无浏览器时省 146 tok/轮）；Tool descriptions 精简（`rag_search`/`rag_outline` 去除冗长重复解释）；Work memory 注入用 compact JSON；长期记忆不注入 prompt 而靠 `memory_search` 按需检索；记忆抽取 prompt 精简并加条数/语言约束
- **MCP 协议** — 对接通用 MCP Server，自动注册远程工具
- **确认机制** — 工作区内操作自动执行，`shell_run` / `code_run` 等高风险操作需用户确认；确认弹窗只显示**用途说明**（启发式优先：注释/docstring/命令名映射，零延迟；无注释时才用 LLM 兜底生成 ≤20 字用途），不贴代码全文，避免超长文本把按钮顶出屏幕
- **技能系统** — 自动识别用户意图匹配技能；支持目录化技能（`.agent/skills/` 下每个技能一个目录，含 `skill.json` 元数据 + `prompt.md` 指令），`SkillIndex` 维护 `_index.json` 索引，`skill_search` / `skill_load` 按需发现与加载，技能指令不进系统提示词，节省 token

## 界面布局

```
┌──────────────────────────────────────────────────────────────────┐
│ 对话 [附加图片…]                [清空对话] [终止对话] [日志] [配置] │ 顶栏
├─────────────────────────────────┬────────────────────────────────┤
│                                 │ 任务计划            1/3 完成    │
│                                 ├────────────────────────────────┤
│ 对话内容…                        │ 目标：…                        │
│                                 │ ▶ 2. 进行中的步骤（高亮标记）    │
│                                 │                                │
│ （一直顶到最上面）                │ 思考过程                       │
│                                 │ 推理过程实时显示…                │
│                                 │                                │
│              0.0 tok/s | 总计 0 tokens                            │
├─────────────────────────────────┴────────────────────────────────┤
│ 输入框                                               [发送]      │
├──────────────────────────────────────────────────────────────────┤
│ [状态栏]                                   [进度条]               │
└──────────────────────────────────────────────────────────────────┘
```

- **「配置」**（右上角）：打开独立配置窗口，模型后端 / 路径 / 知识库 / 采样参数都在里面
- **「日志」**（右上角）：打开独立日志窗口，RAG 同步、工具调用、检索日志都进这里
- **任务计划**（右上）：任务调用 `task_plan` 制定计划后显示目标与每一步状态（○ 待办 / ▶ 进行中 / ✓ 完成 / ✕ 阻塞），进行中的步骤加粗 + 底色标记，标题右侧显示 `已完成/总数`
- **tokens 计数**：`tok/s | 总计 N tokens`，在发送框上方居中

**按钮说明**：
- 「同步知识」（配置窗口）：增量同步知识库（右键可强制全量重处理）
- 「停止同步」（配置窗口）：仅在同步进行中启用，协作式取消（完成当前文件/批次后退出）
- 「终止对话」：终止当前 agent 对话（完成当前工具后退出）
- 「应用配置」（配置窗口）：在 agent 运行期间会被阻止，需先终止当前对话

## 快速开始

### 安装

```bash
# 从源码安装
git clone https://github.com/arlenecc/Base-Aug-Agent.git
cd Base-Aug-Agent

# 安装核心依赖（GUI + LLM + MCP）
pip install -e .

# 安装 RAG 依赖（知识库功能）
pip install -e ".[rag]"

# 安装全部可选依赖（测试 + RAG + OCR）
pip install -e ".[all]"
```

**依赖分组**：

| 分组 | 安装命令 | 包含内容 |
|------|---------|---------|
| 核心 | `pip install -e .` | PyQt6, httpx, beautifulsoup4 |
| RAG | `pip install -e ".[rag]"` | lancedb, fastembed, FlagEmbedding, python-docx, openpyxl, python-pptx, PyMuPDF, ebooklib, docling, chonkie, jieba |
| OCR | `pip install -e ".[ocr]"` | rapidocr-onnxruntime（图片型 PDF 识别） |
| 本地 MLX | `pip install -e ".[local-mlx]"` | mlx, mlx-lm, mlx-vlm, mlx-embeddings（仅 Apple Silicon） |
| 本地 ONNX | `pip install -e ".[local-onnx]"` | onnxruntime, transformers |
| 本地 GGUF | `pip install -e ".[local-gguf]"` | llama-cpp-python |
| 测试 | `pip install -e ".[test]"` | pytest, pytest-qt（`qapp` fixture，UI 测试必需） |
| 全部 | `pip install -e ".[all]"` | 以上所有 |

RAG 子系统依赖也可在首次同步知识库时通过界面自动检测并安装。

### 环境要求（macOS Intel）

RAG 依赖（`FlagEmbedding` / `torch` / `docling` / `chonkie` 等）需在 **Python 3.12** 环境安装——macOS Intel 上 `torch` 最高仅提供 3.12 的 wheel。若系统默认 `python3` 指向 3.13/3.14，会出现 `import FlagEmbedding` 失败、`docling` 无法安装等问题。

- **运行**：使用 3.12 解释器启动 `python main.py`（或用 3.12 创建 venv）。
- **打包**：`build.sh` 已自动优先使用 `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`。

> **BGE Reranker 的 numpy 兼容性**：macOS Intel 上 `torch` 最高 2.2.2（需 numpy 1.x），而 `chonkie`/`scipy`/`opencv` 新版要求 numpy 2.x，二者无法共存。当前默认使用 numpy 2.x 以保障 embedding/分块/检索/OCR 等核心功能，**BGE Reranker 在 numpy 2.x 下不可用**（`_get_reranker` 会提前检测并回退到向量距离排序，不影响检索）。如需启用 Reranker，需手动将 numpy 降至 1.26.x 并同步降级 chonkie（0.5.1）/scipy（1.13）/opencv（4.x）/transformers（4.48）。

### 启动

```bash
python main.py
```

### 配置

首次启动后在界面填写以下信息：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| Base URL | API 服务地址 | `https://api.openai.com/v1` |
| API Key | 密钥 | - |
| 模型 | 模型名称 | `gpt-4o-mini` |
| 上下文长度 | `max_tokens` | `32768` |
| 温度 | `temperature` | `0.7` |
| top_p | 核采样参数 | `0.95` |
| min_p | 最小概率阈值 | `0.05` |
| top_k | Top-K 采样 | `20` |
| 重复惩罚 | 重复惩罚系数 | `1.0` |
| 超时(秒) | 请求超时 | `120` |

配置自动持久化到 `~/.base-agent/config.json`。

## 内置工具

| 工具 | 功能 | 需确认 |
|------|------|--------|
| `file_read` | 读取文件 | ❌ |
| `file_write` | 写入文件 | ❌ |
| `file_modify` | 修改文件（搜索替换） | ❌ |
| `shell_run` | 执行 Shell 命令 | ✅ |
| `code_run` | 在受限子进程中执行 Python 代码（可强杀，写限工作区） | ✅ |
| `web_scan` | 抓取网页内容 | ❌ |
| `webexec_js` | 浏览器执行 JS（仅配置浏览器端点时注册） | ❌ |
| `ask_user` | 向用户提问 | ❌ |
| `task_plan` | 任务计划（目标 + 步骤 + 状态，注入系统提示防跑丢） | ❌ |
| `work_memory` | 短期工作记忆（键值对 scratchpad） | ❌ |
| `memory_graph` | 知识图谱 CRUD（实体/关系/观察） | ❌ |
| `memory_search` | 长期记忆语义检索 | ❌ |
| `skill_search` | 关键词搜索技能目录中的技能 | ❌ |
| `skill_semantic_search` | 混合检索技能（SKILL.md 向量 + BM25，nomic-embed-text） | ❌ |
| `skill_load` | 按路径加载技能完整指令（SKILL.md 或支持文件） | ❌ |
| `rag_search` | 搜索本地知识库（支持按文档名定向检索） | ❌ |
| `rag_outline` | 获取文档缩略版本（目录结构 + 章节标题） | ❌ |
| `rag_status` | 查看知识库状态 | ❌ |
| `rag_ingest` | 增量索引知识库 | ❌ |

## 本地模型直载（免网络推理）

除 OpenAI 兼容端点外，主模型、Embedding、Reranker 都可以直接在应用进程内加载本地权重，全程不发网络请求。三处选择都是**配置项**（`~/.base-agent/config.json`，界面顶部配置区可直接改）。

### 可选运行时

| 后端 | 配置项取值 | 依赖 extra | 适用模型格式 |
|------|-----------|-----------|-------------|
| MLX-LM | `llm_backend=mlx_lm` | `.[local-mlx]` | MLX 量化/半精度 LLM 目录（Apple Silicon） |
| MLX-VLM | `llm_backend=mlx_vlm` | `.[local-mlx]` | MLX 多模态目录（图/音/视频） |
| ONNX Runtime | `llm_backend=onnx`、Embedding/Reranker `onnx` | `.[local-onnx]` | 导出的 `.onnx` + 同目录 `tokenizer.json` |
| MLX Reranker | `rag_rerank_backend=mlx` | `.[local-mlx]` | MLX 交叉编码器目录（XLM-RoBERTa 序列分类，支持 mxfp8/int8/int4 量化） |
| llama.cpp | `llm_backend=llama_cpp`、Embedding `llama_cpp` | `.[local-gguf]` | `.gguf`（VLM 需额外配置 mmproj） |
| fastembed | Embedding/Reranker 默认 | `.[rag]` | fastembed 目录名（CPU 量化 ONNX，首次需下载） |

```bash
pip install -e ".[local-mlx]"    # mlx / mlx-lm / mlx-vlm / mlx-embeddings
pip install -e ".[local-onnx]"   # onnxruntime + transformers
pip install -e ".[local-gguf]"   # llama-cpp-python
```

### 关键配置项

| 配置键 | 默认 | 说明 |
|--------|------|------|
| `llm_mode` | `local` | `local` / `openai`——主模型顶层二选一（#140）。`local` = 进程内直载本地权重；`openai` = OpenAI 兼容接口（此时 `llm_backend` 恒为 `openai`，本地路径字段被忽略） |
| `llm_backend` | `mlx_lm` | 本地模式下的模型类型：`mlx_lm` / `mlx_vlm` / `onnx` / `llama_cpp`（UI 显示为 MLX-LM / MLX-VLM / ONNX / GGUF）；接口模式下恒为 `openai` |
| `llm_model_path` | `models/qwen3.5-4b-agentic-coder-v4-4bit-mlx` | 本地主模型目录或 `.onnx`/`.gguf` 文件 |
| `aux_llm_backend` / `aux_llm_model_path` | `mlx_lm` / `models/Qwen3.5-2B-MLX-4bit` | 辅助模型（#140）：只支持本地直载，供上下文摘要等轻量子任务使用，不占主模型推理队列 |
| `aux_temperature` / `aux_max_tokens` / `aux_top_p` / `aux_min_p` / `aux_top_k` / `aux_repetition_penalty` / `aux_request_timeout` | `0.2` / `2048` / `0.95` / `0.05` / `20` / `1.0` / `120` | 辅助模型独立采样参数（与主模型分开——主模型重质量，辅助模型重快与稳）。改完即生效，无需重新加载模型 |
| `rag_embedding_backend` | `mlx` | `mlx` / `fastembed` / `onnx` / `llama_cpp`（UI 显示 GGUF） |
| `rag_embedding_model_path` | `models/nomic-embed-text-v1.5-mlx` | 本地 Embedding 权重 |
| `rag_rerank_backend` | `mlx` | `mlx` / `fastembed` / `onnx`（UI 显示 GGUF；精排默认启用，无 disabled 选项） |
| `rag_rerank_model_path` | `models/bge-reranker-v2-m3-ONNX/model_quantized.onnx` | 本地 ONNX Reranker 文件 |
| `rag_rerank_mlx_model_path` | `models/bge-reranker-v2-m3-mlx-mxfp8` | 本地 MLX Reranker 目录（与 ONNX 分开保存，切换后端不用重贴路径） |
| `onnx_threads` / `local_model_threads` | `4` | CPU 推理线程数 |
| `rag_embedding_batch_size` / `rag_rerank_batch_size` | `16` / `16` | 批大小；Embedding 夹在 `[2, 64]`（MLX 实测 2→88、8→152、16→178、32→205、64→205、128→190 条/s） |
| `rag_embedding_max_length` | `8192` | 嵌入截断长度，按 nomic-embed-text-v1.5 的真实上限设（旋转式位置编码：`n_positions=8192`，训练长度仅 2048，靠 RoPE 外推）；绝对位置嵌入模型（如 bge-small-zh，512）按 config 自动夹回。各后端加载时按 `_model_max_length` 夹紧 |
| `rag_rerank_max_length` | `8192` | 精排截断长度，按 bge-reranker-v2-m3 的位置嵌入上限设；各后端按自己的 config.json 夹紧（旧版 base → 512）。长切片下按 token 预算自动收缩批大小（注意力 O(L²)） |

本地模型全部**懒加载**：应用配置时只读路径，权重在首次推理或点「加载模型」时才载入。配置窗口的「加载模型」在后台线程把权重读进内存备用，「卸载」释放它（agent 会换到一个全新的懒加载后端，不会撞上「后端已关闭」）。应用配置时若模型类型/路径没变，会**复用已加载权重的后端**，不会每次重新读数 GB。Embedding 权重在 RAG / 知识图谱 / 技能检索之间按「后端 + 路径 + 线程 + 批大小」共享同一份进程级单例，不会重复占用内存。本地模型没有原生 tool calling，`<think>` 与 `<tool_call>` 由 `_TaggedOutputParser` 从纯文本解析成标准事件。

### 测速对比

界面已移除测速入口，改用命令行跑，结果写入 `~/.base-agent/model_benchmark.json`：

```bash
python -m src.agent.model_benchmark --save     # 测速并保存
python -m src.agent.model_benchmark --show     # 查看已保存结果与建议
```

本机（Apple Silicon，batch=8/16，3 轮）实测：

| 环节 | 本地直载 | fastembed CPU | 结论 |
|------|---------|---------------|------|
| Embedding | MLX `nomic-embed-text-v1.5-mlx` 153–161 条/s | `nomic-ai/nomic-embed-text-v1.5-Q` 42–47 条/s | **3.4–3.7x**，向量余弦 0.962 → 建议切 `mlx` |
| Reranker | MLX `bge-reranker-v2-m3-mlx-mxfp8` 34–39 条/s | `BAAI/bge-reranker-v2-m3-int8` 14–15 条/s | **2.2–2.7x**，分数相关 0.997 → 建议切 `mlx` |
| Reranker | ONNX `bge-reranker-v2-m3-ONNX` 14.2 条/s | 同上 | 0.93–1.06x（持平），分数相关 1.000 → 免联网的等价方案 |

切换判据是「速度 + 一致性」双门槛：只快但向量/分数不一致时会提示先回归检索质量，而不是直接建议切换。

### 默认选择

RAG 的 Embedding 与 Reranker **默认走本地 MLX**（`nomic-embed-text-v1.5-mlx`、`bge-reranker-v2-m3-mlx-mxfp8`），即上表的两组提速结论直接生效；主模型仍是 `openai` 网络端点，不是被强制本地化。

MLX 仅在 Apple Silicon macOS 可用，因此默认路径带**自动回退**：非 macOS / 非 arm64 / 未装 MLX / 模型路径缺失时，日志会写明原因并回退到 `fastembed` CPU 量化版，RAG 不会因默认配置而不可用。回退用的目录名保留在 `rag_embedding_model` 与 `rag_rerank_model` 里。

## 本地知识库 (RAG)

### 工作流程

```
文档目录 → 统一解析 (docling → Markdown) → Hybrid 智能分片 (结构+语义, 100-800 tokens, 10% overlap)
→ 向量化 (nomic-embed-text-v1.5 / FastEmbed ONNX) → LanceDB 存储（向量 + 完整原文 + BM25 全文索引）
检索时: 向量相似度 top_k×4 + BM25 关键词 top_k×4 → 合并去重 → BGE Reranker 精排
       → 过滤相似度 < 0.7 → 返回最相关 ≤ 3 条
```

### 同步全链路日志

同步过程中，右侧日志面板会实时显示每个阶段的详细进度：

```
══════════════════════════════════════════
▶ 知识库同步开始: 目录=/data/kb 模式=增量同步 并发数=2
══════════════════════════════════════════
━━━ Step 1/6: 扫描知识库目录 ━━━
  扫描完成: 共发现 12 个文件 [.pdf(5), .docx(3), .md(4)]
━━━ Step 2/6: 增量对比（Manifest） ━━━
  ✓ 跳过(未修改): report.pdf
  → 待处理(新增): new_doc.docx
  对比结果: 总计12 文件, 3 个待处理, 9 个跳过
━━━ Step 3/6: 解析文档 + 清洗 + 切片 ━━━
  [Worker-123] 启动，负责 1 个文件
  [1/3] 开始处理: new_doc.docx
    ├─ 解析文件: new_doc.docx
    ├─ 解析完成: new_doc.docx (提取 12500 字符, 耗时 0.15s)
    ├─ 清洗文本: new_doc.docx (原始 12500 字符)
    ├─ 清洗完成: new_doc.docx → 11800 字符 (耗时 0.02s)
    ├─ 转换 Markdown: new_doc.docx
    ├─ Markdown 完成: new_doc.docx (11800 字符)
    ├─ 文档切片: new_doc.docx → 24 个切片 (每片 100-800 tokens, 重叠 10%, 耗时 0.01s)
    └─ ✅ 文件处理完成: new_doc.docx (24 切片, 总耗时 0.21s)
━━━ Step 4/6: 向量化 + 存入向量库 ━━━
  ├─ 向量化: 首批 24 个切片正在嵌入...
  ├─ 向量化完成: 首批 24 个切片
  ├─ 向量库写入: 首批 24 条记录
  └─ 向量化写入完成: 共 48 个切片 (2 批次)
━━━ Step 5/6: 清理已删除文件 ━━━
  🗑 已清理: old_draft.pdf
━━━ Step 6/6: 保存同步记录 ━━━
  同步记录已保存: 12 个文件条目
══════════════════════════════════════════
✅ 知识库同步完成 (耗时 3.52s)
   扫描文件: 12 | 新增/更新: 3 | 跳过(未修改): 9 | 清理删除: 1
   总字符数: 42000 | 向量切片: 48 | 错误: 0
══════════════════════════════════════════
```

### 检索日志

检索知识库时，日志面板也会展示详细的检索过程：

```
🔍 知识库检索开始: 查询="如何配置 RAG" 目标结果数=3 向量库总量=48
  ├─ 混合初检: 向量相似度 + BM25 关键词, 各取 top 12 候选
  ├─ 向量相似度搜索: 在 48 条向量中检索 top 12...
  ├─ BM25 检索: 12 条候选
  ├─ 混合候选 20 条 (向量 12 + BM25 12, 去重后 20)
  ├─ BGE Reranker 精排: 对 20 个候选进行交叉编码打分...
  ├─ BGE 重排序完成: 20 → 3 条结果
  │   [1] rag_setup.md (重排序分: 0.8523)
  │   [2] config_guide.docx (重排序分: 0.7641)
  │   [3] faq.pdf (重排序分: 0.6218)
  ├─ 相似度过滤: 丢弃 1 条 < 0.70 的结果, 保留 2 条
  └─ ✅ 检索完成: 返回 2 条结果 (耗时 0.48s)
```

### 关键设计

- **嵌入模型**：`nomic-ai/nomic-embed-text-v1.5-Q`（Q4 量化版，~137MB），通过 FastEmbed (ONNX Runtime) 本地推理，无需 GPU、无需 HuggingFace 网络、无需 PyTorch。768 维向量，量化后 4x 更小，质量无损。首次下载后离线可用。截断长度默认 **8192**（旋转式位置编码 `n_positions=8192` 外推上限，训练长度仅 2048），本地 MLX/ONNX 后端加载时按各自模型 config 夹紧（绝对位置嵌入模型自动降到其上限）。
- **向量数据库**：LanceDB（本地 Lance 列式格式），轻量、零配置、支持高效向量检索。向量归一化为单位长度后用 L2 距离模拟余弦相似度。每切片同时存入 embedding 向量、完整原文、来源文件名、切片编号，并开启 BM25 全文索引（原生 FTS + jieba 预分词）。
- **混合检索**：向量相似度 + BM25 关键词并行检索（各取 top_k×4 候选），按 (source, chunk_index) 合并去重，兼顾语义召回与精确关键词命中（中文经 jieba 分词）。
- **重排序**：BGE Reranker (`BAAI/bge-reranker-v2-m3-int8`)，默认启用（UI 已移除关闭入口），对混合候选精排，提升召回精度；依赖不可用时自动回退到距离排序。如需彻底关闭，在配置文件写 `rag_rerank_enabled: false`。截断长度 `rag_rerank_max_length` 默认 **8192**（v2-m3 的真实上限），各后端加载时按自己的 `max_position_embeddings` 夹紧——旧版 bge-reranker-base 只有 512，切到它时会自动夹回 512。
- **相似度过滤**：rerank 后过滤语义相似度 < 0.7 的结果（不相关的切片不发回模型），最终返回 ≤ 3 条最相关切片（含完整原文与来源信息）。

### Meta-context + Targeted RAG（动态按需检索）

针对"用户提到具体书名/文档"的场景，实现**先掌握全局结构、再按需提取细节**的动态 RAG：

```
文档解析 (docling → Markdown)
  └─ 提取目录结构（章节树 + 标题层级）→ 形成「缩略版本」
  └─ 存入 LanceDB documents 表（文档名 / 缩略版本 / 章节列表 / md 缓存路径）

用户提问「某本书里 XXX 是怎么说的？」
  ├─ ① rag_outline("书名") → 返回全书目录结构（掌握全局）
  ├─ ② 大模型规划需要哪些章节的细节
  ├─ ③ rag_search(query, source="书名") → 定向检索该书内的原文片段
  ├─ ④ 细节追加到 context，滚动累加
  └─ ⑤ 综合全局结构 + 按需细节 → 最终答案
```

- **缩略版本**：`rag_outline` 返回完整目录结构（标题层级），让大模型对全书有全局视野。**不生成逐章摘要**——逐章 LLM 摘要慢、费 token、且对本地模型易产生空结果，标题层级已足以支撑定向检索。
- **纯文本目录提取**：除标准 Markdown 标题（`#`~`######`）外，还支持从纯文本（无标题语法的 PDF/EPUB 解析结果）启发式提取目录：识别「目录/Contents」标记块、中文章节标题（`第X章`/`第X节`/`X卷`）、编号标题（`1.1`）、EPUB 链接（`[TITLE](#anchor)`）、全大写短行。
- **定向检索**：`rag_search` 支持 `source` 参数（文档名关键词），限定在某本书内检索，避免混入其它文档。
- **增量重刷**：同步时无论文件是否修改，都会扫描 Markdown 缓存，对缺记录或旧格式（含已废弃的「章节摘要」段/空目录）的文档重刷为纯目录结构。

- **增量同步**：`manifest.json` 记录每个文件的签名（mtime + size + content_hash），未修改文件完全跳过（不解析、不切片、不嵌入、不写入），已删除文件自动清理对应向量和 manifest 条目。`force=True` 可强制全量重处理。取消同步后仍会保存 Manifest 以确保下次同步的增量准确性。
- **流式 ingest**：worker 池（默认 2 线程）并行解析+切片，主线程通过有界队列（容量 = 2×workers）消费并增量写入向量库，背压机制避免内存堆积。OCR 引擎为**实例池（最多 2 份）**——并发限流仅包 RapidOCR 的推理调用（ocrmac/ANE 路径可并发），扫描版 epub 单书内以 2 线程并行 OCR 图片让 2 份实例同时开工；docling/marker/ovis 转换器「取+用」在同一把锁内原子完成，避免周期回收时两代模型并存（#141/#142/#144）。**扫描版 epub 串行**：同一时刻只 1 本扫描书在做 OCR（峰值内存数 GB），文字版 epub 保持 2 路并行。
- **Markdown 缓存**：解析结果缓存为 Markdown，基于源文件 mtime 判断是否需要重新解析，避免重复处理。
- **批量嵌入**：嵌入按批次进行（默认 200 条/批），避免大知识库一次性嵌入导致 OOM。每批次内按文本长度降序分桶后再拆子批，子批大小由自适应控制器实测吞吐动态调整（2-64，初始 16、步长 2：更快则继续上调、变慢回退；持续劣化自动降档避让，环境变快重新爬升；超长文本另由形状预算钳制条数，RSS 软墙 8GB 兜底），累计持 GIL 达到阈值才主动让出（不逐个 sleep），让 UI 保持响应。每批次嵌入前检查取消标志，及时响应停止请求。
- **协作式取消**：取消后 worker 在文件/批处理边界退出，已写入的向量数据保持完整；`_chunk_iter` 使用 0.5s 短超时确保取消后快速响应；`add_streaming` 在每批处理前检查取消标志避免无效计算。
- **资源释放**：同步完成后 `RAGEngine.close()` 显式释放 LanceDB 连接、FastEmbed ONNX 模型、BGE reranker、RapidOCR 实例池（2×~500MB≈1GB）、docling/marker 转换器（各 ~1.9GB）、OvisOCR2 VLM（~0.9GB）；`VectorStore.close()` 显式调用 ONNX `InferenceSession.release()` 立即回收 C++ 堆内存；窗口关闭时补关主/辅助 LLM 后端与 terminate 路径残留的 RAG 引擎，并移除 logger handler 防止悬空引用；`_JsonStore` 采用 0.5s 节流写盘避免高频 I/O。
- **文档元数据缓存**：`VectorStore` 对 documents 表内容做内存缓存，`get_document_digest` / `list_documents` 不再每次全表 `to_pylist()`；写入（upsert/delete/clear）时失效缓存。documents 表写入用 `_documents_lock` 串行化，避免多 worker 并发 ingest 时 delete+add 交错。
- **不存全文 Markdown**：documents 表只存「缩略版本 + 章节列表 + Markdown 缓存路径」，**不存全文**（全文已有 `rag/documents/*.md` 缓存，表内再存是第三份冗余，且会被缓存全量载入内存；该字段实际零消费者）。缓存按列读取并显式剔除 `markdown`，兼容旧数据。
- **删除即清理**：源文件从知识库移除后，同步时一并清理向量、documents 元数据行与 manifest 条目。
- **BM25 索引一致性**：写入（含单批写完、取消等提前出口）与删除都会失效 FTS 标记，`reload()` / `clear()` 也会重置——避免「混合检索静默退化成纯向量」或「已删文档仍被召回」。重建**推迟到下次检索**惰性执行（旧实现每次 `add()` 都全表重建，一次同步 N 批 = N 次全库建索引）。
- **Reranker 加载去重**：BGE reranker 加载用状态机（`idle/loading/done/failed`）标记，并发调用不会各自起加载线程（每线程 ~500MB PyTorch 权重）；超时后后台线程完成时仍写入结果，后续调用直接复用。

### 支持的文档格式

**统一解析引擎（docling）**：Word、Excel、PowerPoint、PDF、EPUB 通过 [docling](https://docling-project.github.io/) 统一解析，直接输出结构化 Markdown（含布局、表格、标题、图片 OCR）。docling 未安装时自动回退到按格式的轻量解析器。

**PDF 解析引擎三选一**（`rag_pdf_engine`，配置窗口单选钮）：`docling`（默认，版面理解 + ocrmac OCR）/ `marker`（marker-pdf 2.x，surya OCR）/ `ovis`（PyMuPDF + OvisOCR2-0.8B MLX-VLM 逐页转录，同时是所有引擎单文件失败时的轻量兜底）。docling 的 OCR 渲染倍率默认全篇固定 2.0；曾实现的逐页动态倍率因 +87% 耗时零收益已改为配置项 `rag_docling_dynamic_scale`（默认关闭）。

| 格式 | 扩展名 | 解析方式 |
|------|--------|------|
| Word | `.docx`, `.doc` | docling（回退 `python-docx`） |
| Excel | `.xlsx`, `.xls` | docling（回退 `openpyxl`） |
| PowerPoint | `.pptx`, `.ppt` | docling（回退 `python-pptx`） |
| PDF | `.pdf` | `rag_pdf_engine` 三选一（docling / marker / ovis），图片页走引擎自带 OCR 或 RapidOCR |
| EPUB | `.epub` | docling（回退 `ebooklib`） |
| MOBI/AZW | `.mobi`, `.azw3` | calibre（可选） |
| 纯文本 | `.txt`, `.md`, `.csv` | 无额外依赖 |
| HTML | `.html`, `.htm` | `beautifulsoup4` |

### Hybrid 智能分片

分片采用**结构感知 + 语义分块**的混合（Hybrid）策略：

```
Markdown 文本
  ├─ 按标题结构（#~######）+ 空行段落边界切分
  ├─ 超长章节（>800 tokens）→ chonkie SemanticChunker 语义细分
  │    （复用 nomic-embed-text 嵌入模型，按语义相似度边界切分）
  ├─ 合并过小块至 ≥100 tokens
  └─ 相邻块 10% overlap（80 tokens，句子边界对齐，保证跨块上下文连续）
```

- **大小**：语义块最小 100 tokens、最大 800 tokens、重叠 10%
- **嵌入模型**：`nomic-embed-text-v1.5-Q`（与向量存储共享同一份 ONNX 模型，不重复加载）
- **三级回退**：hybrid → 语义分块（chonkie）→ 递归分块（无模型依赖），任何环境都能工作

### 依赖自动管理

点击「同步知识」时会自动：

1. 扫描知识库目录，识别所有文件扩展名
2. 对照依赖表检查所需 Python 包是否已安装
3. 缺失的依赖自动 `pip install`（含版本约束）
4. 核心依赖（`lancedb`、`fastembed`）也会一并检查
5. 安装完成后自动开始同步

若自动安装失败（如网络问题），会弹出提示列出需手动安装的包。

### 使用方式

1. 在界面第三行「知识库」中输入文档目录路径（或点击 `…` 选择）
2. 点击「同步知识」执行增量同步（只处理新增/修改的文件）；右键点击可强制全量重处理
3. 右侧日志面板实时显示同步进度
4. 同步完成后，对话中模型会自动调用 `rag_search` 检索相关知识
5. 需要清理重复向量时，点击「同步知识」左侧的「向量去重」按钮（无需重解析文件）

### 同步期进度可见性（心跳日志）

大文件（尤其是上百页的扫描书）转换一次要几十秒到数分钟，期间日志面板会长时间不刷新。
为此 `ingest()` 起一条**心跳线程**，默认每 60 秒输出一行进度：

```
  ⏱ 同步进行中 (已 128s): 已完成 37/120 个文件, 处理中 4 个 [算法导论.pdf(96s), 剑桥中国史.epub(88s) ...] 等4个
```

行内会列出**当前仍在处理**的文件及其已耗时，据此可以直观区分三种「卡住」：

| 现象 | 含义 |
|------|------|
| 心跳持续出现、在办文件时间在涨 | 正常，只是这几本慢 |
| 心跳出现但「处理中 0 个」 | 在向量化 / 收尾阶段，没有文件在解析 |
| 心跳**不再出现** | 主线程卡死或进程已被系统杀掉 |

可用环境变量 `RAG_HEARTBEAT_S` 调整间隔（设 `0` 关闭），`RAG_MEM_LOG=1` 额外开启内存采样。

### 副本去重（三层，互不替代）

同一本书常以多种形式重复出现（不同目录下的副本、重新保存的 PDF、`docx→pdf` 导出）。
去重分三层，分别在不同时机生效：

| 层 | 时机 | 判据 | 作用 |
|----|------|------|------|
| 解析前 | 同步 Step 2.5 | **文件指纹**（大小 + 头/中/尾各 64KB 采样 sha256） | 同内容文件只解析一次，其余在 manifest 标记 `duplicate_of`，省掉整段解析/切片/向量化 |
| 入库后 | 同步 Step 5.5 | `content_hash`（解析产物） | 清掉新出现的副本向量与缓存，避免检索噪声 |
| 存量清理 | 点「向量去重」按钮 | 向量库内同文本分组 + `ingested_at` 排序 | 清理历史遗留重复，**保留入库时间最早的一份** |

「向量去重」是独立按钮而非同步的一部分：存量重复（本功能上线前已入库的副本）不会因新同步
被动清理，需要显式跑一次。老库缺 `ingested_at` 列时会自动迁移补列（旧行填 `0.0`，即视为最早）。

### PDF 转换的并发与内存护栏

PDF 转换器（docling/marker）实例重、非线程安全，大文件并发会撑爆内存。按**文件大小**限流：

- 小文件（< `rag_pdf_large_threshold_mb`，默认 20MB）：最多 `rag_pdf_small_concurrency`（默认 2）路并发
- 大文件（≥ 阈值）：独占，且它排队时会阻止新的小文件入场（防大文件被源源不断的小文件饿死）

转换器实例池按 OCR 渲染倍率分桶复用，每实例服满 50 篇主动退役重建，冲刷其原生缓存。
闸门的 `acquire()` 返回许可令牌、`release(permit)` 凭据归还——不能按 size 重新判定分类
（阈值热改会把读者计数减到写者分支，导致后续大文件永久阻塞，见问题表 #162）。

### Token 控制

每次 RAG 检索最多返回 3 条结果，单次消耗 ≤ 1500 tokens（在 32768 上下文窗口中占比 < 5%），确保响应速度不受影响。

## 知识图谱记忆

长期记忆采用**知识图谱**架构，模仿 Memory MCP Server 的设计模式，用本地 JSON 文件 + LanceDB 语义索引实现，无需外部数据库。

### 数据模型

```
Entity（实体）    = {name, type, observations: [str], created_at}
Relation（关系）  = {source: name, target: name, label: str}
Observation（观察）= 挂在实体上的自由文本事实
```

图以实体 **name**（大小写不敏感）为主键，创建同名实体会合并观察。

### 文件布局

```
workspace/.agent/
├── work_memory.json           # 短期工作记忆（键值对 scratchpad）
├── long_memory.graph.json     # 知识图谱 JSON（实体 + 关系）
├── long_memory.graph.json.vectors/  # LanceDB 观察向量索引
├── skills.json                # SkillManager 数据（扁平技能记录 + 请求计数）
└── skills/                    # 目录化技能（每个技能一个子目录）
    ├── _index.json            # SkillIndex 自动生成的索引表
    └── <skill-name>/
        ├── skill.json         # 元数据（name/description/keywords/tags/entry）
        └── prompt.md          # 技能指令全文
```

### 自动事实抽取（一次抽取，双写）

每次 agent 对话结束（无 tool_calls 的最终回复）后，**异步**调用 LLM 从最近几轮对话中抽取实体、关系和观察（daemon 线程，不阻塞用户回复）。**一次抽取，同时写入两份记忆**，避免重复抽取：

```
对话结束 → _maybe_extract_facts_async()
  ├─ daemon 线程启动，on_finished() 立即回调
  ├─ 收集最近 6 轮对话窗口（解析代词/项目指代）
  ├─ LLM 低温度(0.1)抽取 JSON: {entities: [...], relations: [...]}
  ├─ 写入长期记忆 GraphMemoryStore（去重 + 向量索引）
  └─ 同步写入短期记忆当天桶 __auto_facts__（只保留当天，跨天重置）
```

- **长期记忆**：全部事实持久化到知识图谱，靠 `memory_search` 按需语义检索。
- **短期记忆**：当天抽取的事实保留在 `work_memory` 的 `__auto_facts__` 键（`{"date": "YYYY-MM-DD", "facts": [...]}`），对话时直接注入上下文；跨天后自动清空（历史已在长期记忆）。
- **独立抽取客户端**：抽取线程使用独立的 `LLMClient` 实例（`_get_extract_llm()` 惰性创建，独立的 `httpx.Client` 连接池），避免与对话线程并发争用同一个非线程安全的 `httpx.Client`；`_fact_extract_lock` 非阻塞锁进一步防止多个抽取堆积。`Agent.close()` 显式释放该独立客户端。

### 语义检索

观察通过 `nomic-embed-text-v1.5-Q` 嵌入后存入 LanceDB，支持语义检索：

```
memory_search(query) → GraphMemoryStore.search()
  ├─ LanceDB 向量检索 top_k 条观察
  ├─ 返回 {entity, text, score}
  └─ 向量索引不可用时回退到关键词子串匹配
```

索引侧的关键设计：

- **批量嵌入**：`add_observations` / `create_entity` 收集新观察后调用 `embed_documents()` 一次性批量嵌入（单次 ONNX 前向），替代逐条 `embed_query()`（N 条观察从 N×10-50ms 降为一次批量计算）
- **非对称前缀**：观察作为检索目标用 document 前缀嵌入，查询用 query 前缀嵌入（nomic 模型要求）
- **注入防护**：删除向量时对实体名单引号做 SQL 转义，防止 filter 表达式被畸形/注入

### Embedding 模型共享

RAG 知识库和知识图谱记忆共用同一 FastEmbed ONNX 模型实例（进程级单例 `get_or_create_embedding_function()`），避免加载两份 ~137MB 模型。ONNX Runtime `session.run()` 线程安全，可并发调用。

### 精简 Prompt 注入

**旧方案**：全量注入所有工作记忆 + 所有长期记忆事实 → 记忆多时 prompt 膨胀，拖慢响应。

**新方案**：
- **SYSTEM_PROMPT**：精简至 ~200 tok，操作原则/确认策略/工具约定/记忆/技能/RAG 引导各一条、无冗余
- **Knowledge base 状态**：仅当 RAG 可用时注入（无 KB 时不注入），仅含文档名列表（不含路径），最多 8 个
- **工作记忆**：手动 scratchpad（compact JSON，value 截断 200 字符）+ 当天自动抽取事实（列表形式）
- **长期记忆**：不注入 prompt，靠 `memory_search` 工具按需检索（避免长期记忆膨胀拖慢每轮响应）
- **Tool descriptions**：精简（`rag_search` 的 query 描述从 ~150 字压到一句话，`rag_outline` 去除重复的「定向检索」解释）
- **记忆抽取 prompt**：`_EXTRACT_SYSTEM` 规则压缩去重

```
# Work memory
{key:"value[:200]…"}

# Long-term memory (snapshot)
- PostgreSQL (tool): 用 16.3 版本; 连接池 50
- Alice (person): 负责 API 层
- Alice --[works_on]--> API Gateway
```

### 工具

| 工具 | 操作 |
|------|------|
| `work_memory` | `set` / `get` / `list` / `clear`（短期 scratchpad） |
| `memory_graph` | `create_entity` / `delete_entity` / `add_observations` / `create_relation` / `list_entities` / `get_entity` / `snapshot` / `clear` |
| `memory_search` | 语义检索观察，返回 `{entity, text, score}` |

### 兼容性

旧的 `long_memory.json`（扁平事实列表）在首次加载时自动迁移到知识图谱（作为 "General" 实体的观察），迁移后旧文件清空。

## 技能系统

技能用于把高频任务固化为可复用的指令模板。系统包含三层：

- **SkillManager**（`skills.json`）：按关键词匹配用户意图，命中后把技能 prompt 注入系统提示；同一意图出现 ≥ 2 次时建议固化为技能
- **SkillIndex**（`.agent/skills/` + `_index.json`）：目录化技能注册表。每个技能一个目录，含 `skill.json`（元数据：name / description / keywords / tags / entry）和 `prompt.md`（指令全文）
- **SkillRetriever**（`skills/` + LanceDB）：对 **SKILL.md** 的混合语义检索。递归扫描 `workspace/skills` 下的所有 `SKILL.md`，解析 frontmatter（name/tags/description）与正文段落（When to Use / Examples），用 nomic-embed-text 分别对 description 和 examples 计算向量存入独立 LanceDB 表，并对「name+tags+description+examples 拼接文本」建 BM25 全文索引

工作方式：

```
用户请求
  ├─ SkillManager.match() 命中扁平技能 → prompt 注入系统提示
  │   （目录技能注册的扁平记录引导 agent 调用 skill_load 加载全文）
  └─ 复杂任务 → agent 主动调用 skill_semantic_search(query) 混合检索
                │  向量检索（description 0.6 + examples 0.4 融合，取 3 候选）
                │  + BM25 全文检索（取 3 候选）→ 去重 → 剔除相似度 < 0.5
                │  返回候选的 name+tags+description，由 agent 判断选用
                └─ skill_load(dir) 读取 SKILL.md 全文（或 references/ 支持文件）并遵循
```

关键设计：

- **按需加载**：技能全文不进系统提示词，仅在需要时通过 `skill_load` 读入，避免技能增多导致 prompt 膨胀
- **索引防抖**：扫描器在目录 mtime 变化或 300s 间隔后重建，LanceDB 表懒构建
- **混合检索**：向量相似度（description/examples 0.6:0.4 加权融合）+ BM25 关键词双通道，统一用余弦相似度排序，阈值 0.5 过滤
- **小批次索引（防内存爆炸）**：`build()` 以 `batch_size=8` 分批 embed + 写入，每批处理 8 个 skill 后立即释放；description/examples/combined 文本分别截断至 2000/3000/6000 字符。实测 2757 个 skill 全量索引：峰值 RSS 4.19 GB（此前一次性处理全部文本时达 65 GB 导致系统卡死），耗时 12.4 分钟
- **路径安全**：`skill_load` 对 `path` / `entry` 做 realpath 前缀检查，拒绝 `..` 等路径穿越
- **创建技能**：`SkillManager.create_dir_skill(name, keywords, prompt, ...)` 自动生成目录结构并注册扁平记录

## 上下文收缩

Agent 在长对话中会自动管理上下文窗口，避免超出模型 token 限制：

- **主动收缩**：当估计的 prompt 大小达到 `max_context_tokens × 90%` 时，自动摘要旧消息，保留最近 6 条消息（当前任务上下文）verbatim，旧消息压缩为单条摘要标记。Token 估算采用增量计数（O(1) 而非 O(N) 每次全量扫描），高频场景下性能更优。估算包含 **系统提示 + 历史 + tools schema**（工具多时 schema 可达数千 token，漏算会让主动收缩几乎不触发）。
- **注入上限**：系统提示的动态注入段（当天事实、命中技能 prompt）**不在收缩范围内**，因此各自设上限（事实 ≤20 条/2000 字符，技能 prompt ≤1500 字符），否则长会话下系统提示会单调膨胀且无自愈路径。
- **被动收缩**：遇到 LLM API 返回 `context_length_exceeded` 类错误时，自动收缩并重试同一轮（不消耗迭代次数），最多重试 3 次
- **持久化**：摘要写入 `workspace/memory.md`（带时间戳和触发原因），支持长期/跨会话回溯
- **摘要策略**：调用 LLM 生成要点摘要（用户需求、决策、已完成步骤、未完成子任务、关键路径/URL/数值）；LLM 不可用时回退到结构化摘要（用户请求 + 工具调用列表）
- **悬挂消息修复**：收缩时确保历史不以悬挂 `tool` 消息开头（否则 API 拒绝请求），自动将对应的 `assistant(tool_calls)` 拉回保留段

## 规划与反思

针对「任务半路跑丢、直接就停、不知道做什么就乱做」的问题，Agent 内置结构化规划 + 四道防停滞防线：

### 任务计划（task_plan）

```
用户下达非平凡任务
  ├─ ① 模型调用 task_plan create(goal, steps) 建计划
  │     计划持久化到 workspace/.agent/task_plan.json
  ├─ ② 计划注入系统提示（# Current task plan），每轮迭代可见
  ├─ ③ 每步工具结果回来后 task_plan update(index, status, note) 标记进度
  ├─ ④ 发现新步骤 → add；换任务 → 重新 create；完成 → clear
  └─ 换会话（清空对话）→ reset() 一并清除计划
```

- **目标不丢失**：计划在系统提示里，不参与上下文收缩（_shrink_context 只压 _history），长任务被摘要后模型依然锚定原始目标与进度。
- **字段上限**：goal ≤300 字符、步骤 ≤20 条、单步 ≤200 字符、note ≤200 字符——系统提示不在收缩范围内，必须防止计划本身膨胀。
- **宽容输入**：`"2"` → `2` 的 index 归一由 ToolRegistry 的类型归一承担。

### 四道防停滞防线（run() 内置看门狗）

| 防线 | 触发条件 | 动作 |
|------|---------|------|
| 1. 空响应重试 | 模型返回无内容且无工具调用 | 撤掉空 assistant 消息（部分服务端对 content=null 且无 tool_calls 的消息返回 400）、注入反思提醒后重试；每轮最多 2 次 |
| 2. 重复失败指引 | 相同 (工具, 参数) 连续失败 ≥2 次 | 在工具结果末尾附加重试指引（改参数/换方法/ask_user），不改历史结构 |
| 3. 连续失败复盘 | 连续 ≥4 次工具失败（不管参数） | 注入整段反思提醒（目标是什么/试了什么/为何失败），每轮最多 2 次 |
| 4. 预算耗尽收尾 | 迭代达到 max_iterations 且最后一轮仍是工具调用 | 注入收尾提醒并做一次**不带工具**的最终调用，强制总结已完成工作与剩余步骤；LLM 失败时回退最近一条非空回复 |

所有提醒以 user 角色注入（与上下文摘要标记同一模式），次数有上限，防止提醒本身变成新的循环。

## MCP 协议支持

在 `~/.base-agent/config.json` 中配置 MCP 服务器：

```json
{
  "mcp_servers": [
    {
      "name": "filesystem",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    }
  ]
}
```

启动后**在后台线程连接** MCP Server 并注册其提供的工具：

- 连接（握手 + `tools/list`）是同步阻塞的，一个起不来的 server 要等满 30s。放在 UI 线程会让「应用配置」/启动时冻结 30s×N 且无提示。
- 因此构造 `ToolRegistry` 时只创建 client（廉价）并起守护线程去连接；工具表读写有锁，读侧取快照。
- 工具被调用时尚未连上时，`MCPClient.ensure_started()` 会现场拉起；握手失败属于可重试的瞬时故障（不会被误判为永久停止）。
- 代价：配置生效后的**第一轮对话**可能还看不到 MCP 工具——远好于界面卡死。

## 项目结构

```
src/agent/
├── agent.py              # 核心 Agent 推理循环（含增量 token 计数、prompt 缓存、自动事实抽取、反思/防停滞看门狗）
├── planner.py            # 任务计划（目标 + 步骤 + 状态，持久化到 .agent/task_plan.json，注入系统提示）
├── config.py             # 配置管理
├── llm_client.py         # LLM 流式客户端
├── memory.py             # 工作记忆 / 长期记忆（知识图谱，节流写盘）
├── graph_memory.py       # 知识图谱存储引擎（Entity/Relation/Observation + 批量嵌入 + LanceDB 语义检索）
├── skills.py             # 技能管理（意图匹配 / 请求固化建议 / 目录技能创建）
├── skill_index.py        # 技能索引（扫描 .agent/skills/、维护 _index.json、加权检索）
├── skill_retriever.py    # SKILL.md 混合检索（扫描 + 解析 + LanceDB 向量/BM25 检索）
├── rag/
│   ├── engine.py            # RAG 引擎（编排：扫描→对比→解析→清洗→切片→向量化→清理→保存）
│   ├── docling_parser.py    # docling 统一解析器（Word/Excel/PPT/PDF/EPUB → Markdown）
│   ├── marker_parser.py     # marker-pdf 解析器（PDF 解析引擎第二方案）
│   ├── ovis_ocr.py          # OvisOCR2 多模态 OCR（PDF 引擎第三方案 + 单文件兜底）
│   ├── parsers.py           # 按格式的轻量解析器（docling 回退方案 + MOBI/HTML/文本）
│   ├── cleaner.py           # 文本清洗（去标签/URL/控制字符/零宽字符/空白归一化）
│   ├── chunker.py           # Token 级别递归切片 + 分片回退链编排
│   ├── hybrid_chunker.py    # Hybrid 分片（文档结构 + 语义分块）
│   ├── semantic_chunker.py  # chonkie SemanticChunker 语义分片（复用 nomic-embed-text）
│   ├── vector_store.py      # LanceDB 向量存储 + FastEmbed 嵌入 + BGE 重排序
│   └── deps.py              # 依赖检查与自动安装
├── tools/
│   ├── base.py           # 工具注册中心（MCP + RAG 生命周期管理）
│   ├── file_ops.py       # 文件读写工具
│   ├── shell_run.py      # Shell 执行
│   ├── code_run.py       # Python 代码执行（受限子进程沙箱 + killpg 强杀 + 写路径防护）
│   ├── web.py            # 网页抓取 / JS 执行
│   ├── interact.py       # 用户交互
│   ├── memory.py         # 记忆工具（work_memory / memory_graph / memory_search）
│   ├── skill_search.py   # 技能工具（skill_search / skill_load）
│   ├── rag_tool.py       # RAG 检索工具
│   ├── mcp_client.py     # MCP JSON-RPC 客户端
│   └── mcp_tool.py       # MCP 工具适配
└── ui/
    └── main_window.py    # PyQt6 主窗口（QTimer 日志轮询、信号槽、QThread worker）
```

## 运行测试

```bash
pip install -e ".[test]"

# 全部测试（含 RAG 端到端，需下载模型，较慢）
pytest tests/ -v

# 快速测试（排除需下载模型的 e2e 测试）
pytest tests/ -k "not rag_e2e and not rag_full_pipeline" -v
```

测试覆盖：Agent 推理循环、工具调用、上下文收缩（主动/被动/悬挂消息修复）、LLM 客户端、RAG 全流程（端到端 + 集成 + manifest 增量同步 + 异常文件处理）、RAG 工具自主发现与调用、依赖检查、技能系统（意图匹配 + SkillIndex 索引/检索/路径安全 + SKILL.md 混合检索）、UI Bridge、内存存储、文档目录提取（Markdown 标题 + 纯文本启发式）等。

近期补充的回归测试：首次 ingest 不再死锁（全新知识库）、`shell_run` 超时不留孤儿进程、`web_scan` 协议白名单、损坏 config 回退、`skill_load` 路径穿越防护、配置迁移不再改写用户显式设置、**「终止对话」在流式与工具链中途立即生效**、**向量化失败不再泄漏 worker 线程**、**`_JsonStore.reload` 不再丢弃防抖窗口内的写入**、**任务计划持久化/注入/重置**、**空响应反思重试、重复失败指引、连续失败复盘、预算耗尽强制收尾**。当前共 402 个用例（含 UI 测试）。

> UI 相关测试（`test_bridge.py` / `test_integration.py` / `test_ui_flow.py`）依赖 `pytest-qt` 提供的 `qapp` fixture；未安装时这三个文件的用例会在 setup 阶段报错，其余测试不受影响。

## 近期修复（全量代码复查）

按影响面排序，详见 [软件设计文档 §11](docs/软件设计文档.md)：

| # | 问题 | 影响 | 修复 |
|---|------|------|------|
| 1 | `upsert_document` 持 `_documents_lock` 后再取同一把非可重入锁 | **全新知识库首次同步永久卡死** | 锁改 `RLock` + 拆分 `_ensure_documents_table_locked()` |
| 2 | `InstallDepsWorker.finished` 遮盖 `QThread.finished` | `deleteLater` 在 `run()` 内销毁运行中的线程 | 改名 `deps_finished`，`deleteLater` 挂内置信号 |
| 3 | 确认弹窗 `wait()` 返回值被忽略 + 初值 `True` | 超时后 `shell_run`/`code_run` 被自动放行 | 初值 `False` + 检查 `wait()` 结果 + `reset()` |
| 4 | `skill_load` 未校验 `entry` | 绝对路径 / `..` 可读磁盘任意文件 | 显式拒绝 + `realpath` 前缀校验 |
| 5 | 语义切片无视注入的 embedding function | 离线环境卡在模型下载 | 全链路透传 `embedding_function` |
| 6 | 切片拼接凭空造字符 | chunk 文本不是原文子串，模型看到不存在的内容 | 取源码真实间隔 / 句末边界直接切片 |
| 7 | docling 锁惰性创建竞态 + `convert()` 无锁 | 并发加载两份模型、状态污染 | 模块级锁 + 串行化转换 + `release_converter()` |
| 8 | `shell_run` 只杀顶层 `sh` + 输出无上限 | 孤儿进程、大输出打满内存 | 独立进程组 `killpg` + 200KB 截断 |
| 9 | MCP `stop()` 以 `_running` 为条件提前 return | 僵尸进程永不回收 | 以 `_process` 为条件 + `killpg` + 关闭管道 |
| 10 | 配置损坏即无法启动、迁移改写用户设置、非原子写 | 启动崩溃 / 设置被反复改回 | 备份回退 + 仅迁移旧默认值 + 原子写 |
| 11 | 技能索引目录缺失时每次查询写盘、mtime 失效 | 无谓磁盘 I/O、索引最长 300s 陈旧 | 补更新时间戳 + 同时 stat `skill.json` |
| 12 | `skill_retriever.build()` 先删表后探测维度 | embedding 失败即丢失已有索引 | 先探测再重建 |
| 13 | `skills.json` 损坏记录拖垮 `match()`、请求表无界 | 整轮技能检索失败、文件膨胀 | 逐条校验跳过 + 淘汰上限 |
| 14 | `_JsonStore` 每实例注册 atexit 回调且从不注销 | 缓慢内存泄漏 | 单一全局回调 + WeakSet |
| 15 | `_describe_code_purpose` 用 `ev.get()` 访问 `StreamEvent` | LLM 兜底是死代码 | 按属性访问 |
| 16 | PDF/OCR 异常分支漏 `doc.close()`、加密 PDF 整份失败 | 句柄与解码缓存泄漏 | `try/finally` + 空密码解锁 |
| 17 | `VectorStore.close()` 未重置 `_fts_ready` | 重连后 BM25 静默失效 | 一并重置 |
| 18 | `web_scan` 无协议白名单、响应体全量入内存 | `file://` 读本地文件、OOM | http/https 白名单 + 8MB 截断 |
| 19 | 技能缓存未按 workspace 区分 | 切换工作区后读到旧技能 | 按目录路径缓存 |
| 20 | `chat_view` 无 `setMaximumBlockCount` | 长会话文档无限增长 | 限 5000 段 |
| 21 | 取消标志只在 confirm/ask_user 被检查 | **「终止对话」对流式/工具链不生效** | `is_cancelled()` 在迭代/流式/工具边界轮询，跳过的工具补合成结果保持历史一致 |
| 22 | `add_streaming` 异常后 worker 阻塞在 `q.put` | 每次同步失败泄漏 4 线程 + 文档内存 | 消费端失败置位 `_abort`，worker 立即退出 |
| 23 | manifest 在写库成功前提交 | 失败/取消的文件下次被跳过，内容**永久缺失** | `pending_manifest` 延迟到整条链路成功才并入 |
| 24 | `code_run` 劫持进程级 cwd/stdout | 超时泄漏线程永久破坏全局状态 | 主线程快照 + join 后恢复 |
| 25 | `code_run` 沙箱 `abspath` 不解析 symlink | workspace 内 symlink 写穿到系统文件 | `realpath` 前缀校验 |
| 26 | `file_read`/`code_run` 输出无上限 | 大文件 → OOM / 撑爆上下文 | 200K 上限（流式有界收集、按需读取） |
| 27 | MCP `tools/call` 复用 30s 超时 | 长任务必然超时且结果被丢弃 | 独立 300s 下限 |
| 28 | `_get_reranker` 无锁 check-then-act | 并发加载两份 ~500MB 模型；close 后复活 | 状态机加锁 + `_closed` 标志 |
| 29 | `_escape_like` 不转义单引号/通配符 | 畸形 SQL；含 `%`/`_` 的文件名匹配所有文档 | 引号翻倍 + `ESCAPE '\\'`（实测验证） |
| 30 | `_ensure_initialized` / 懒单例无锁 | 并发首调重复建连接/模型并泄漏 | 双重检查锁 |
| 31 | 依赖检查异常仍启动同步 | 缺依赖环境跑 RAG 引擎，掩盖真实错误 | 异常与取消分开，失败不启动 |
| 32 | `file_ops` 原子写固定 `.tmp` 名 | 并发互踩混合内容、失败残留 | `mkstemp` 唯一名 + finally 清理 |
| 33 | `graph_memory` 批量嵌入 `zip` 截断 | 图谱与向量库永久不一致 | 数量校验，整批失败 |
| 34 | `deps` 把 `config.json` 当重排序缓存命中 | 中断下载后跳过预下载 | 只认权重文件 |
| 35 | `skill_load` 读过期 retriever 属性 | 切 workspace 后校验发生在旧目录 | 走按目录缓存的 `get_skill_retriever()` |
| 36 | `file_modify` 无界读 + 二进制裸异常 | 大文件 OOM；二进制报难懂错误 | 10MB 上限 + 显式捕获，指引改用 file_write |
| 37 | `_atomic_write` 保留 mkstemp 的 0600 | 每次写文件悄悄变成「仅属主可读」 | 恢复原权限 / 按 umask 取 0644 |
| 38 | `mcp` 工具输出无上限 | 一次调用撑爆模型上下文 | 200K 截断 + 标注原始长度 |
| 39 | `mcp.stop()` 最终 `wait()` 无超时 | D 状态进程卡死 shutdown | 5s 超时放弃等待 |
| 40 | `memory_search` 无 top_k 钳制 / `list_entities` 全量 | 全部观察/实体灌进上下文 | top_k ≤ 20；列表截断到 200 |
| 41 | `rag_status` 文件列表全量 | 数百文件挤占上下文 | 截断到 50 并标注总数 |
| 42 | `execute()` 把工具内部 TypeError 误报为参数错误 | 工具出 bug 时模型反复重试而不是上报 | `inspect.signature` 预检区分两类错误 |
| 43 | `code_run` 泄漏线程可吞掉后续执行的 stdout | 并发执行时输出丢失 | 恢复权全部移交主线程，worker 不再碰全局状态 |
| 44 | `_JsonStore.reload()` 丢弃防抖窗口内的未落盘写入 | **0.5s 内 set 后 reload，刚固化的技能静默丢失** | reload 前先把脏数据强制写回 |
| 45 | `add_streaming` 单批写完提前 return，跳过 FTS 重建 | **小知识库同步后 BM25 静默失效**（混合检索退化成纯向量，无报错） | 提前出口同样重建索引 |
| 46 | `delete_by_source` 删向量后不失效 FTS | 已删除文档的内容仍被关键词检索召回并返回给 LLM | 删除后置 `_fts_ready = False` |
| 47 | `reload()` 只刷向量表句柄 | 同步后 `rag_outline` 读到旧 digest、BM25 看不到新切片 | 一并重置 documents 表/缓存/FTS 标志 |
| 48 | documents 元数据表存每篇文档**完整 markdown** 且全量进缓存 | 语料在磁盘存 3 份；进程常驻内存 ≈ 整个知识库；该字段**零消费者** | 不入库，改存 `md_path`（缓存按列读取，绝不加载 markdown） |
| 49 | 删除源文件只清向量与 manifest，不删文档元数据 | `rag_outline` 列出已无向量的文档 | `_cleanup_deleted_files` 同步 `delete_document` |
| 50 | `_fetch_worker` 成功后从不置 None | **获取模型后关闭窗口必崩**（`isRunning()` 抛 RuntimeError → PyQt abort） | 两个槽都清空引用；`closeEvent` 全部改用 `_is_worker_running()` |
| 51 | `_on_stop_sync` / `closeEvent` 直接调 `isRunning()` | 同上，`deleteLater` 后点击「停止同步」崩溃 | 统一走带 RuntimeError 保护的守卫 |
| 52 | `usage` 字段为 `null` 时 `None > 0` | 部分网关返回 null → **整轮 run() 崩溃，回复丢失** | `usage.get(x) or 0` 归一 |
| 53 | 事实抽取 prompt 无条数/语言约束 + 当天事实桶无上限 | 长会话系统提示单调膨胀（且系统提示**不在上下文收缩范围内**） | 抽取限 5 实体 × 3 观察；桶限 200 条；注入限 20 条/2000 字符 |
| 54 | 命中技能 prompt、当天事实注入零截断 | 同上，膨胀无自愈路径 | skill prompt ≤1500 字符并提示 `skill_load` |
| 55 | `_estimate_history_tokens` 不计 `tools` schema | 工具多时系统性低估 → 主动收缩几乎不触发，只剩被动收缩 | 计入 tools schema（带缓存） |
| 56 | `_get_extract_llm()` 阻塞等 `_fact_extract_lock` | **确认弹窗被抽取线程阻塞最长一个 LLM 请求（~120s）**，与「不延迟弹窗」意图相反 | 非阻塞取锁，取不到直接复用 `self.llm` |
| 57 | `memory_search` 关键词兜底路径完全忽略 `top_k` | 未装 rag 依赖时（默认路径）一次查询灌入全部观察 | `_keyword_search` 接收并生效 top_k |
| 58 | 技能分词用 `\w+` 对中文失效 | `"提取pdf表格"` 成单 token，中文关键词**永远匹配不上** | 补 CJK 整串 + bigram |
| 59 | `GraphMemoryStore.close()` 只置 None 从不 close | 每次「应用配置」泄漏一个 LanceDB 连接 | 显式 close + 重置 `_vec_tried` |
| 60 | 工具层 RAG 引擎初始化在 try 之外 | 知识库不可用时**所有工具（文件/shell/web）一起失效** | 整段包 try，失败仅跳过 RAG 工具 |
| 61 | `shell_run` 的 `timeout` 无上下限钳制 | 传 99999 → 阻塞 27h 且无法取消；传 0 → 立刻 kill | 钳制到 [1s, 600s] |
| 62 | `web_scan` 流式失败 `except Exception` 静默重试 | 真实超时后重复请求（最坏 2×30s）、已读内容丢弃 | 仅 `AttributeError`（无 stream 实现）才回退 |
| 63 | `_reranker_state` 打分失败后停在 `"done"` 而模型为 None | 状态机失真 | 同步置 `"failed"` |
| 64 | 每 chunk 重复计算 `_hash_source` | 写入侧纯浪费（同批同源） | 批次内缓存 source→hash |
| 65 | SYSTEM_PROMPT 含模型已知的协议说明与冗余反例 | 每轮浪费 ~100 tok | 精简；补充「语义检索 query 语言」与「工具报错重试」两条关键约束 |
| 66 | `code_run` 在进程内 `exec`，线程**杀不掉** | 死循环线程存活到进程退出，持续占满一个核 | 改为受限子进程，超时 SIGTERM→SIGKILL `killpg` 整棵树（实测无残留忙循环） |
| 67 | `code_run` 只 patch `builtins.open` | `pathlib.write_text` / `os.open` / `os.remove` / `shutil` 均可绕过写保护 | 子进程内覆盖 `builtins.open`、`io.open`、`os.*` 写类、`pathlib.Path`、`shutil` |
| 68 | `code_run` 未校验 `rename`/`replace`/`link`/`symlink` 的**目标** | 先在工作区写好再 rename 出去即可落盘到任意位置 | 源与目标双向校验（含 pathlib 的 `rename`/`replace`/`symlink_to`） |
| 69 | MCP 连接在 `ToolRegistry.__init__` 同步执行 | **UI 线程冻结 30s×N**（配多个 server 时） | 后台守护线程连接 + 工具表加锁；未连上时 `ensure_started()` 懒拉起 |
| 70 | MCP 握手失败调 `stop()` 置 `_stopped` | 一次瞬时失败 = 该 server 本进程内**永久无法连接** | 拆出 `_teardown(stopped=False)`，失败可重试；重连前先收掉存活的旧进程 |
| 71 | `_get_extract_llm()` 取不到锁/构造失败时退回 `self.llm` | 抽取线程与主线程流式请求共用同一 `httpx.Client`（流交错损坏） | 返回 `None` 并放弃本次抽取，绝不降级到共享连接池 |
| 72 | `close()`/`shutdown()` 后抽取线程仍可新建资源 | 专用 LLM 连接池、~137MB ONNX 长期记忆**无人回收** | `Agent._closed` 标志 + `_get_long_memory` 检查 `registry._shutdown` |
| 73 | `on_token_speed` / `on_usage` / `on_finished` 未隔离 | UI 回调一次异常 → `on_finished()` 永不执行，**界面永久忙碌** | 统一走 `_safe_callback`；`is_cancelled()` 也加隔离版 |
| 74 | `_on_send` 只看 `_busy` 就覆盖 `self._worker` | 旧 QThread 仍运行时丢引用 → Qt `qFatal`，**进程静默退出** | 发送前校验 `_is_worker_running(_worker)`，并在同步进行中拒绝发送 |
| 75 | `closeEvent` 用通配 `disconnect()` | 把精心连的 `finished→deleteLater` 一起断掉；`terminate()` 后无条件丢引用 | 改用 `blockSignals` + 统一 `_shutdown_worker()`，未确认结束**保留引用** |
| 76 | 「应用配置」在主线程跑 `tools.shutdown()` | 逐个 MCP `join(2s)` + 释放 ~600MB 模型 → **界面冻结十几秒** | 旧资源释放移到后台线程（`_release_old_agent_async`） |
| 77 | 「应用配置」与同步/依赖安装互不互斥 | 同步线程与主线程并发开关同一 LanceDB 目录与进程级单例 | 双方对称加守卫（agent 忙 / sync 忙 / deps 忙均拒绝） |
| 78 | 同步临时 logging handler 仅靠 `finally` 摘除 | `terminate()` 时不执行 → handler 永久残留、日志重复、闭包持有整个 worker | `QThread.finished` 槽无条件补摘；`setLevel` 旧值恢复 |
| 79 | manifest 全量覆盖写 | 两个入口并发时后写者用旧快照**抹掉**前者刚写入的条目 | `_ManifestLock`（flock + TLS 防自锁）+ 读改写合并；`removed` 集合区分「我删的」与「别人写的」 |
| 80 | `RAGEngine.clear()` 无锁删 manifest | 被并发写**复活** → 向量库空、记录满，检索永远返回空且不自愈 | 持 manifest 锁删除 |
| 81 | `clear()`/`close()` 与在途 `add`/`search` 并发 | 打到已 drop 的表或 `None` 连接（随机 AttributeError） | 在途计数 + 条件变量排空；`_begin_op` 同时检查 `_closing`；`reload()` 同样排空 |
| 82 | `SkillVectorStore._get_db()`/`_get_table()` 无锁 | 并发首调建立两个 LanceDB 连接，泄漏一个 | 双重检查加锁（锁改 RLock：两者会嵌套） |
| 83 | graph 向量回填失败前先置 `_backfilled` | 本进程内**永不重试**，历史事实一直检索不到 | 仅成功后置位，失败告警并允许重试 |
| 84 | 回填用 `to_arrow().to_pylist()` 取 id 集合 | 物化全部列（含 768 维向量）→ 数百 MB 峰值 | `search().select(["id"])` 列裁剪下推 |
| 85 | 流式每个 chunk 都做一次文档重排 + 压入 undo 栈 | 长回复主线程持续重排；undo 栈无上限增长 | 100ms 攒批写入；只读视图 `setUndoRedoEnabled(False)` |
| 86 | `_tool_schema_tokens` 每轮迭代重复序列化整套 schema | MCP 工具多时一轮 15 次迭代的几十 KB JSON ×2 | `ToolRegistry.schemas()` 加缓存，工具表变动时失效 |
| 87 | `_trim_history` 边界缺失 | 尾部连续 tool 消息时**历史被清空**，模型失去全部上下文 | 至少保留一条 |
| 88 | 文档目录提取把代码块内的 `#` 当标题 | shell/Python 注释污染目录 digest | 围栏代码块感知（与 hybrid_chunker 一致） |
| 89 | `shell_run` 子进程继承 stdin、严格解码 | `cat`/`ssh` 挂到超时；一个非法字节让该流**剩余输出静默丢失** | `stdin=DEVNULL` + `utf-8`/`errors="replace"` |
| 90 | `code_run` 超时后孙进程持有管道 → pump 线程永久挂起 | 每次调用白等 4s，泄漏两个线程 + 两个 fd | 短宽限后从父进程侧关闭管道唤醒；单行长度也封顶 |
| 91 | `web_scan` 可被诱导访问 `169.254.169.254` | SSRF 拿到云主机 IAM 凭据 | 拦截链路本地/云元数据地址（放行 127.0.0.0/8 不影响本地开发） |
| 92 | `graph_memory` 批量抽取逐元素落盘 | ≤8 次整图 JSON 序列化 + 原子写盘换一批事实 | `batch_apply()` 一把锁改完内存、只落盘一次 |
| 93 | `_system_prompt` 按 turn 缓存，`task_plan` 更新计划后缓存不失效 | 同一轮内规划变更后模型看不到最新计划，行为倒退 | `ToolRegistry` 加 `_state_version`，计划工具写操作 bump；提示词缓存键纳入版本号 |
| 94 | `_tool_schema_tokens` 缓存命中时仍全量 `json.dumps` schema 算指纹 | 命中路径零收益，MCP 工具多时每轮白做几十 KB 序列化 ×2 | `ToolRegistry._schema_version` 递增计数器替代 md5 指纹，命中时零序列化 |
| 95 | `code_run.run()` 内 `import builtins as _b` 未使用 | 死代码 | 删除 |
| 96 | UI 配置面板 `max_tokens` / `request_timeout` 无下限校验 | 填 0/负数 → 每轮对话立刻失败且报错难排查 | 钳制下限（max_tokens ≥ 1、timeout ≥ 1s），非法输入回退默认值 |
| 97 | 获取模型返回空列表时下拉框被清空 | 用户已手输的模型名丢失，需重新输入 | 空列表时保留当前文本 |
| 98 | code_run / shell_run 每次执行都弹确认 | 正常工作流（跑脚本、git、pip、装依赖）反复被打断 | 按内容风险评估：仅提权(sudo)、写系统目录(/etc 等)、磁盘/电源/内核操作、`curl\|sh`、工作区外 rm、代码含 subprocess/os.system/ctypes 逃逸调用时确认，其余直接执行 |
| 99 | 任务全部完成后计划残留在系统提示 | 模型无明确收尾信号，陈旧计划一直注入每轮提示 | 计划全部 done 时 render_plan 追加 "All steps are done → clear" 信号；系统提示规则 1 明确「全部完成后调用 task_plan clear」 |
| 100 | 渲染后无有效内容的图片页仍送 OCR（纯色占位图、扫描件空白纸页） | RapidOCR 对每页检测 2-5 秒后返回空，单文档可累计浪费 1 分钟以上 | 渲染后采样位图极差 ≤ 12 即判近纯色页，跳过 OCR 计入 blank（含噪点扫描页，实测 4 场景验证） |
| 101 | 图片页 OCR 零预处理：暗色主题（黑底白字）/低对比度模糊文字页 det 大面积检不出 | 电子书「一大片页面检测不到文字」 | 灰度渲染 + `_ocr_normalize` 归一化（暗页自动反色 + p2-p98 对比度拉伸），分位数在反色后重算（否则噪声被放大十余倍淹没文字）；直接传 ndarray 省 PNG 编解码。真实引擎实测 6 场景：暗主题/模糊/浅灰页检出恢复或改善，正常页无回归 |
| 102 | OCR 走 rapidocr-onnxruntime 1.x（CPU EP），未利用 Apple Silicon 的 ANE/GPU | 检测 arm64 macOS 后仍纯 CPU 推理 | 优先 import rapidocr 3.x（模型目录与 docling 共享、懒加载），Apple Silicon 上传 `EngineConfig.onnxruntime.use_coreml: True` 启用 CoreML EP（MLComputeUnits=ALL），失败自动回退 CPU EP；OCR 依赖检测改为 3.x/1.x 任一可用。基准实测 1.5x 加速、输出与 CPU EP 完全一致 |
| 103 | deps.py 依赖清单漏登记 chonkie / FlagEmbedding（docling 有动态检查） | 缺失时不检查、不自动安装也不提示，语义分块/重排序静默退化 | check_dependencies 增加两项可选检查；DepSpec 新增 `auto_install` 字段——chonkie 轻量随同步自动安装，FlagEmbedding（~2GB 含 PyTorch）仅提示手动安装，避免 2GB 下载被静默触发 |
| 104 | onnxruntime 无版本下限：fastembed 可能拉到旧版，旧版 wheel 无 CoreMLExecutionProvider | CoreML EP 静默回退 CPU EP | check_dependencies 增加 onnxruntime>=1.31 版本下限检查（低于则进 missing_required 由 ensure_rag_deps 自动升级），版本号解析失败不误报 |
| 105 | 重排序依赖 FlagEmbedding（连带 PyTorch ~2GB），模型 pytorch_model.bin ~1.1GB | 依赖链重、冷启动慢、与 transformers 版本兼容问题多（XLMRobertaTokenizer API 断裂等） | 改用 fastembed TextCrossEncoder（ONNX，复用嵌入同一条 fastembed/onnxruntime 栈）：模型换成官方 INT8 量化版 bge-reranker-v2-m3-int8（~570MB，中文精排优于 base）；fastembed 返回原始 logit，vector_store 侧 sigmoid 归一化保持 [0,1] 阈值语义；预下载改为 `TextCrossEncoder(lazy_load=True)` 幂等下载，缓存判定迁移到 fastembed 缓存目录；torch 兼容性探测段删除。pyproject 移除 FlagEmbedding/transformers，fastembed>=0.6 |
| 106 | shell_run 确认策略误报与逃逸：`2>/dev/null`、`>/dev/stderr` 等丢弃输出惯用法被当成「写 /dev 设备」高频弹确认；`rm "$HOME/x"`、`rm '/etc/passwd'` 引号包裹的逃逸路径绕过确认直接执行；`dd of=/dev/disk0`（写盘目标在 of= 参数，不走重定向）完全漏检 | 每条普通命令都可能打断用户；引号包裹的危险 rm 与 dd 写盘不受控 | 写系统路径正则对 /dev 里的 null/stdin/stdout/stderr/tty 加负向前瞻白名单；`_rm_targets_outside` 对 token 剥引号后再判逃逸；新增 `_RE_DD_WRITE` 检测 `of=/dev/*`（豁免 /dev/null）。回归：/dev 白名单不确认、引号逃逸与 dd 写盘确认、真写 /dev/disk0 仍确认 |
| 107 | config 不迁移 FlagEmbedding 时代的旧默认重排序模型 `bge-reranker-base`（fp32 ~1.1GB） | 旧配置用户继续拉 1.1GB fp32 模型（INT8 版 570MB 且中文更强） | `_migrate` 沿用 max_tokens=4096 迁移先例，旧默认值重写为 bge-reranker-v2-m3-int8；显式配置的其他模型名不动 |
| 108 | deps 预下载触发条件用 `report.reranker_model_missing`（pip 安装**前**判定） | fastembed 刚被 Step 1 从旧版升级的场景，模型下载被跳过、延迟一轮同步 | Step 3 改为安装完成后实时判定（`_is_importable("fastembed") and not _is_reranker_cached()`） |
| 109 | UI 配置面板 temperature / top_p / min_p / top_k / repetition_penalty 无范围钳制 | 输错（负数、超范围）被服务端直接拒绝，所有对话失败且报错难懂 | 与 max_tokens/request_timeout 同策略：temperature∈[0,2]、top_p∈(0,1]、min_p∈[0,1)、top_k≥0、repetition_penalty∈[1,2]，超范围值钳到边界 |
| 110 | 全局审查·工具层 7 项：memory 工具 observations 传字符串被逐字符入库；file_modify 的 old="" 使 replace("", new) 损坏整个文件；_atomic_write 用 os.umask 翻转读权限（进程级竞态）；web_fetch 的 SSRF 拦截漏掉整数/十六进制 IP 形式（http://2852039166/ 可绕过直达云元数据端点）；skill_search top_k 垃圾值裸异常；code_run 输出泵 readline 在 TextIOWrapper 内部缓冲整行（单行 500MB 的 print 内存峰值就是 500MB，截断只在之后）；MCP stdin 的 flush() 无超时且持有写锁（server 停止读取时 agent 线程永久卡死） | 单次工具调用即可污染长期记忆/损坏文件/挂死对话 | observations 归一化（字符串→单元素列表，元素过滤）；old 为空直接拒绝；新建文件直接 fchmod 0644 不再翻转 umask；SSRF 检测补 `int(host,0)` 归一；top_k try/except 归一；_pump 改 8192 分块按 \n 切行（行超限即丢弃，内存峰值恒定）；stdin 改 os.write+select 带超时分块写，超时报不健康（15s） |
| 111 | 全局审查·RAG 层 6 项：子目录文件的 digest 元数据删除键不匹配（doc_name 存 rel_path，清理只按 basename/绝对路径删）→ 孤儿行永久累积、rag_outline 列出已删文档；ensure_digests 重刷漏传 md_path → 同文档双行且重刷不生效；删除失败仍移除 manifest 条目 → 向量永久残留仍可被检索；增量对比 _file_signature 未捕 OSError（扫描后文件消失）→ 整个同步崩溃；.doc/.xls/.ppt 路由到必然失败的解析器（python-docx/openpyxl/python-pptx 只支持 OOXML）→ 依赖白装 + 报错难懂；MOBI 裸解析全量载入（数百 MB 电子书 ≈3× 峰值内存） | 数据一致性、同步稳定性 | 清理时补删 rel_path（doc_name 精确命中）；重刷补传 md_path；delete_by_source 失败改为抛出，失败文件保留 manifest 下次重试；签名步骤 per-file try/except 记入 errors；三个旧格式移出 SUPPORTED_EXTENSIONS/EXTENSION_DEPS 并给出「另存为 OOXML」明确提示；裸解析加 100MB 上限 |
| 112 | 全局审查·检索质量：chunker._merge_overlap 用外层分隔符拼接深层递归切片——外层按 "\n\n" 切、深层按句号切，相邻 sub_chunk 之间不存在外层 sep，拼接产物含原文没有的字符（与 semantic_chunker 已修的同类 bug） | 边界 chunk 被伪造内容污染，影响检索与模型阅读 | _gap_between 上移到 chunker.py 共用；merge 时在原文中定位相邻片段的真实间隔，定位失败退回旧 sep |
| 113 | 全局审查·技能层 3 项：skill_index 读 skill.json/prompt.md 未捕 UnicodeDecodeError（GBK 文件让 skill_search/skill_load 整体持续失败）；索引未做 CJK 切分（未空格的中文查询永远检索不到技能，skills.py 已修但此处未同步）；SkillVectorStore.build 中途失败留下半成品表（部分技能 5 分钟内不可见） | 一个编码不对的文件废掉整个技能系统 | 打开改 errors="replace" + except 扩展；_tokenize 对齐 skills.py 的 CJK run+bigram；build 失败即 drop 新表下次完整重建 |
| 114 | 全局审查·agent 核心 5 项：graph_memory backfill 重试在早退 gate 之后不可达（日志承诺的重试从未发生）且 embed 失败静默 return 后误标已完成；事实抽取 LLM 输出未做类型校验（observations 字符串逐字符入库、非 dict 元素炸整批）；delete_entity 持图锁执行 LanceDB 删除（违反本文件「embedding/LanceDB 不持图锁」不变量，删除期间冻结所有图读取）；confirm 路径快绕过防堆叠锁与后台抽取并发共用同一 httpx.Client（非线程安全，流交错损坏）；graph_memory/_JsonStore 的 _load 不防「合法 JSON 但结构错误」且无恢复路径 | 长期记忆污染且无法自愈、UI 卡顿、并发损坏 | backfill 移到独立非阻塞重试路径（_backfill_lock），embed 失败显式抛出；extract_facts_via_llm 返回前类型收敛（entities/relations 过滤 dict、observations 收敛 List[str]）；_deindex_observations 挪到图锁释放后；confirm 兜底在使用期间持锁（拿不到则退回启发式）；两处 _load 加结构校验，坏文件告警后按空图继续 |
| 115 | 全局审查·UI 层 4 项：窗口几何持久化的 isinstance(geo, bytes) 永假（QSettings 回读 QByteArray）——全平台静默失效；同步线程 terminate 后的临时 handler 兜底读 window 上不存在的属性（handler 引用挂在 worker 上），兜底是死代码——handler 闭包强引用 worker+RAG 引擎阻止回收；获取模型完成槽无条件解锁按钮/复位进度条（agent 运行中出现假空闲态）且失败弹窗阻塞流式渲染；reasoning 流未攒批（每个 chunk 五步光标操作，与 content 路径优化不对称） | 持久化失效、资源泄漏、UI 状态错乱、思考区越长越卡 | _as_bytes 兼容 QByteArray/bytes/str；兜底改为显式传 worker（finished 槽默认参数快照 + closeEvent terminate 路径显式调用）；两个槽按忙碌态合成规则设置；_on_reasoning 复用 _stream_buf 同款 100ms 攒批（发送/清空对话时同步清缓冲） |
| 116 | docling 对「每页文本层 + 满页背景图」的中文电子书跑全量管线（auto OCR 逐页位图推理 + 布局/表格模型，torch MPS 缓存只增不放）——实测 21 本书中 19 本属此类，单本 225-280 页耗时 37-74 分钟、进程 RSS 涨到 36GB 且不回落；扫描件判定 `_is_scan_pdf` 只认「前几页全无文本层」，捕捉不到混合型 | 同步 2 小时只完成解析、向量写入饿死（消费者空转），内存失控需重启应用回收 | 新增 `_pdf_is_image_heavy`：采样前 20 页，纯图页或嵌入图片占页面面积 ≥15% 的文本页占比 ≥30% 即路由到 PyMuPDF+RapidOCR 快速路径（共享单例引擎 + CoreML + 逐页有界内存，探测开销毫秒级）；docling 保留给纯文本 PDF 与 Office 格式，且管线强制 CPU 加速 + 限 4 线程（MPS 缓存问题兜底，版本差异 try/except 回退默认构造） |
| 117 | jieba 在 pyproject `[rag]` extra 里，但不在 rag/deps.py 运行时检查清单——环境缺失时同步前不会自动补装，BM25 中文预分词静默退化为空白切分（整句一个 token，召回大幅下降），且写入侧退化持久（text_fts 列已按坏分词落库） | 中文全文检索无感知劣化 | check_dependencies 补 jieba（与 chonkie 同级：可选 + 自动安装，纯 Python ~19MB 无重依赖）；查询侧装完即生效，历史向量需 force 重同步修复 |
| 118 | 语义切片嵌入走共享 nomic 适配器（BERT 推理，chonkie 窗口方案 = 2N 次 embed，中文逗号切句后 N 极大）——88k 字符章节 90-150s，长章节多的书切片阶段耗时分钟级 | 同步耗时被切片放大；nomic 与切片决策并非同空间需求（只比较相邻窗口相对相似度） | 新增 model2vec 静态嵌入快路径（potion-base-32M，~30MB，纯 numpy 查表）：88k 字符章节 **0.4s（~250 倍）**；解析链注入 EF（测试）→ potion → 共享 nomic → 递归兜底四级回退；pyproject [rag] 与 deps.py 同步声明，缺失时自动安装并热升级（60s 负缓存） |
| 119 | RapidOCR+CoreML 的原生缓存按输入形状/批次保留：真实 200dpi 扫描页实测 **~35MB/页滞留**（合成小图无法复现），400 页扫描书滞留 ~14GB 进程不释放，久不触碰被系统压缩至 17GB | 扫描书同步内存失控（重启后仍 16.95GB，叠加旧 MPS 问题曾 50GB 崩溃） | 每书每 60 页 OCR 强制重建引擎（冲刷 CoreML/ORT 原生缓存，重载 ~6s 相对每页 5-12s OCR 可忽略），峰值滞留钳制在 ~2GB；全书结束收缩 MuPDF store（满页解码位图缓存）+ `gc.collect()`；实测同 60 页滞留 **2114MB → 554MB**；周期回收代价换 OOM 安全，勿因合成实验不复现而移除 |
| 120 | ingest 文件分片按 `ceil(n/workers)` 连续切片：9 文件 4 worker 时第 4 片为空（3×3=9 已分完），第 4 个 worker 领空即退（日志可见「负责 0 个文件」）；且扫描书按文件名顺序扎在同一片时其余 worker 长时空转 | 白耗一个并发槽；同步尾延迟被最慢片拉长 | `_shard_files_balanced`：按文件大小降序 + 贪心分配给累计负载最小的 worker（LPT 调度），最大 4 个文件必属 4 个不同 worker，n≥workers 时空片不可能；新增 6 项分片测试 |
| 121 | 重排模型缓存路径三重错配：`_download_reranker_model` 未传 cache_dir → fastembed 默认写到**系统临时目录**（macOS 定期清理 → 每次同步重下 570MB）；`_is_reranker_cached` 却查 `define_cache_dir("local_cache")`（相对 CWD 的无意义路径）→ 永远判定未缓存 | 每次同步重复 570MB 下载；离线环境报「未缓存」 | 下载显式指定持久目录 `~/.cache/fastembed`（与嵌入模型同处）；检测兼容持久 + 临时两个缓存根，只认真实权重文件 |
| 122 | 嵌入吞吐瓶颈：fastembed 批内所有序列 **padding 到批内最长**，长度混杂批浪费 2-4 倍算力（32 条混杂批 309s ≈ 40 tok/s，全库 4000 切片 ≈ 8-19h，且子批间完全静默 → 误判卡死）；子批 32 的激活高水位 4.3GB。实测不可行的加速路：CoreML EP（仅 41% 节点可用 + 两次 SIGKILL）、torch-MPS（nomic 自定义代码不兼容）、线程调优（±10%）、fp32 换 int8（速度相同） | 嵌入是同步最大瓶颈；内存峰值被嵌入激活抬高 | **按文本长度降序分桶 + 12 条/子批**：195 真实切片实测 **69.8s（~1676 tok/s，~18 倍）**，全库 ≈ 25-30 分钟——无需换模型、无需表迁移；激活峰值 ~2.7GB；向量按原序回填（结果与混杂批一致）；子批级进度日志消除静默期；ORT threads=4 |
| 123 | #119 的周期回收被 worker 本地引用击穿：`ocr_engine` 缓存在每个 worker 的**循环级局部变量**里，一个 worker 触发回收只清自己和全局，其余 worker 仍抱旧引擎继续用——最多 4 代引擎的原生缓存并存（每代 ~1-2GB），回收形同虚设（实测仍 16G→12G 锯齿） | 回收不彻底，长扫描书同步内存居高不下 | OCR 引擎改为**每页从进程级单例取用、页末释放**（不做 worker 级缓存）：回收置 None 后所有 worker 下一页统一重建，任意时刻至多一代引擎存活 |
| 124 | 用户质疑「没有操作需要 11GB」：模型权重本身极小（嵌入 137MB + OCR ~50MB），大头全是框架层滞留——嵌入激活 arena 首批长切片顶到 2.7GB 只增不缩、OCR 原生缓存 60 页周期水位 ~2GB、macOS 分配器滞留+压缩计费 ~2GB | 11GB 平台期虽稳定但远超模型本体大小 | 旋钮收紧后按用户实测回调：嵌入子批 12（6 条省内存但吞吐损失明显）、OCR 回收 20 页（单代水位 ~0.7GB）；纯文本书的汇总日志依赖 `ocr_engine` 变量，循环前显式初始化 None 防未绑定 |
| 125 | 子批固定 12 是特定环境（独占 CPU、~900 tok 切片）下的平衡点，环境漂移后失准：OCR worker 抢占 CPU 时大子批吞吐下跌、独占 CPU 时小子批浪费 GEMM 效率；且内存/吞吐权衡无法用单一常量覆盖 | 子批大小拍死一个数，换环境就次优 | `_SubBatchTuner` 实测自调（初始 4-20、步长 2）：以算力速率 `ops/s = n × max_len²/dt` 归一批长度差异；爬山 +2 采纳需 ≥3% 提升、变慢回退；监控阶段连续慢窗口降 2 避让、速率超历史最优 10% 重新爬升；日志带 `[size=N]` 可观测；新增 11 项状态机测试 |
| 126 | #125 的 4-20 自调经合成基准与真实切片生产路径双重验证后方向性错误：真实切片上吞吐与内存都随子批增大单调变差（size2 = 2071 tok/s @ 1.28GB、size4 = 1864 @ 2.14GB、size12 = 1163 @ 4.99GB）——激活工作集 ∝ 子批×批内最长序列，超出 CPU 缓存后被内存带宽卡死；旧 tuner 起步 8 先向上爬，在最长切片阶段用最慢尺寸；首个窗口含模型加载导致基线虚低、热试探被误采纳 | 自调方向与真实硬件行为相反，且无内存兜底 | 范围修订为 1-5、初始 2、步长 1（#127/#128 用户定参，初始值即落在实测最优区 size2）；内存安全改用**形状预算** `_shape_cap_size`（激活 ∝ 条数×max_len²，预算 6M 单位 ≈ 2GB 激活，超长文本自动收缩条数）——原 5GB 绝对 RSS 阈值在真实同步进程（基线 6-10GB）首个子批即触发降档并永久卡死在下限（实测「子批很快降到 1」的根因），已弃用；首个预热窗口丢弃不计入判定；测试 13 项 |
| 129 | 「降到 1 再也不回去」有两个成因：① 降档是单向棘轮——触发只需连续 2 窗口 <75% 最优（一次内存尖峰的短暂变慢即可），回去却需 >110% 最优（平台期不可能达到）；② 旧 5GB 闩锁版 RSS 守卫（mem_pressed 永不解除）若应用未重启仍会生效 | 瞬时内存尖峰造成永久降档，吞吐腰斩 | monitor 新增恢复分支：速率回到历史最优 95% 且尺寸低于最优尺寸 → **直接步回最优尺寸**（不走路爬山——plateau 上试探必然持平被回退，回退锚会复现卡死）；留在 monitor 自校正，最优尺寸真变慢由降档路径接管，无循环；按用户定参加回 8GB RSS 软墙（无闩锁：超墙每子批 force_down 降 1 档、回墙下即停，恢复交给恢复分支）；爬升出口（采纳/回退/封顶）重锚 `_best_size`；测试扩到 17 项 |
| 130 | OCR 回收周期 20 页是凭水位倒推的估计值（~35MB/页 × 20 ≈ 0.7GB），热重建成本当时按冷启动 ~6s 估，偏保守 | 周期过紧白白增加重建频率，或过松累积缓存 | 系统扫描 N=10..100 × 真实 200dpi 页：热重建仅 1-2s/次（冷启动 3-4s 只发生一次），峰值足迹各 N 持平 ~5.2GB（压缩器吸收缓存增长，peak_cmprs 随 N 线性涨），N 对 400 页书总耗时影响仅 1-3%；定 N=50：单代水位 ~1.8GB、重建摊销 ~0.5%，速度与内存的实测平衡点 |
| 131 | 重排序三处与设计意图脱节：① 两个真实调用点都不传 rerank_model，实际吃 engine.py 旧默认 `BAAI/bge-reranker-base`（fp32 ~1.04GB 非量化），而非设计声明的 INT8 v2-m3（~570MB）；② `TextCrossEncoder` 未传 cache_dir，fastembed 默认下到系统临时目录（macOS 定期清理 → 重复下载，与 #121 同型）；③ 未显式传 providers，EP 选择依赖 fastembed 默认行为 | 加载的模型比预期大一倍且非量化；模型缓存被系统清理后重复下载 | engine.py 默认改为 `DEFAULT_RERANK_MODEL`（BAAI/bge-reranker-v2-m3-int8，与 deps.py 缓存判定、设计文档对齐）；TextCrossEncoder 显式传持久 cache_dir（~/.cache/fastembed）+ `providers=["CPUExecutionProvider"]`（与嵌入模型同款锁死）；目录映射核实：catalog 名 → onnx-community/bge-reranker-v2-m3-ONNX 的 onnx/model_int8.onnx；用已缓存小型 cross-encoder 实测验证 CPU 锁死生效 |
| 132 | 重排延迟实测：v2-m3-int8（568M）在本机 CPU 逐对 ~0.55s/对是地板（批并行无收益——CPU EP 单序列已吃满多核；threads ±10% 噪声；截断 512 对 top3 常规场景无收益，已放弃）；agent LLM 自主传 top_k=5 → 40 条候选 ≈ 23s，交互不可接受 | LLM 自选大 top_k 放大重排延迟 | rag_search 工具 top_k 封顶 3（LLM 传 5 被钳到 3 → 24 条 ≈ 14s），schema 描述同步「上限 3」；进一步降耗的备选杠杆（候选池减半 multiplier 4→2、异步重排）留观察后决定 |
| 136 | 精排截断长度写死 512（`rag_rerank_max_length` 默认值 + 各后端硬编码），而 bge-reranker-v2-m3 的真实上限是 8192（本地 MLX/ONNX 权重 `max_position_embeddings=8194`，fastembed tokenizer 本身就截断在 8192）：query+chunk 拼超过 512 tokens 的部分被静默丢弃。实测同一文档关键结论在尾部时，截断 512 得分 -8.81（低于 0.7 阈值会被过滤掉）vs 完整 8192 得分 +1.75——即「相关文档被截成不相关」。#132 曾以「对 top3 常规场景无收益」放弃放大，但那只覆盖了短 chunk | 长切片（默认 500 tokens + query 后常超 512）的尾部内容不参与精排打分，排序与阈值过滤双失真 | 默认改为 8192；各后端加载时按自己 config.json 的 `max_position_embeddings` 夹紧（`_model_max_length`，config 缺失退回 512）——旧版 bge-reranker-base（514）切过去自动夹回 512，不会把越界位置喂进网络；长序列下按 token 预算收缩批大小（`_rerank_batch_for`，8192 token/批：512→16、1024→8、2048→4、8192→1），避免注意力 O(L²) 把内存打爆；MLX/ONNX  rerank 按行切片保证覆盖与顺序；新增 4 项回归测试 |
| 137 | 嵌入截断长度同样写死 512（`rag_embedding_max_length` 默认值 + MLX/ONNX/向量库单例键硬编码），而 nomic-embed-text-v1.5 真实上限是 8192（旋转式位置编码 `n_positions=8192`，训练长度 2048 靠 RoPE 外推）——512 下长切片尾部语义被丢弃。reranker 的「按模型夹紧 + token 预算切批」机制只覆盖了精排，没覆盖嵌入 | 长切片（默认 500 tokens 切片、语义分块上限 800 tokens）嵌入时尾部不参与向量，召回与图谱抽取质量双失真 | 把同一套机制平移到嵌入：默认 8192；`_model_max_length` 增强为「旋转式取 n_positions、绝对位置取 max_position_embeddings-2」（nomic→8192、bge-small-zh→512、缺失→512）；MLX/ONNX 嵌入后端加载时夹紧并打日志；长切片下按 token 预算收缩批大小（`_embedding_batch_for`，与 reranker 同预算 8192 token/批）。实测对照：本地 MLX NomicBERT 图 vs fastembed 官方 ONNX 导出——短文本对齐 0.966（量化基线差），长文本 4056 tokens 下 fast↔mlx8192=0.887、fast↔mlx2048=0.730、mlx2048↔mlx8192=0.829，证明 8192 外推没有掉质量且显著找回被截断内容；config/engine/graph_memory/memory/tools 默认值同步 8192；新增 4 项回归测试 |
| 138 | #137 把嵌入收缩预算设成和 reranker 一样的「8192 token/批」，对 ONNX(CPU，物化注意力、内存 O(S²))偏激进——8192 时直接夹到 1，但 MLX 走 flash attention（内存 O(S)）实际远没到顶；而 ONNX 实际加载多为 int8 量化，即便误用 fp16 涨到 ~7GB 也可接受 | 长切片下 MLX 吞吐被不必要地压到单条、ONNX 也过保守 | 嵌入批收缩改为**后端感知**（`_EMBEDDING_SHAPE_BUDGET`）：`mlx=300M` → B=4@8192（flash 下 ~1GB），`onnx=150M` → B=2@8192（fp32 注意力 ~7GB 顶、int8 更低）；`VectorStore._shape_cap_size` 同步按 `self._embedding_backend` 取预算，使 RAG 入库主路径也受益；无/未知后端沿用 6M 保守值。两套约束（调优器形状预算 + 嵌入函数内部 `_embedding_batch_for`）现在用同一组后端预算，8192 时 MLX=4、ONNX=2，默认 500 切片均不受影响；回归测试更新为后端感知 |
| 139 | 上一轮后端感知预算仍沿袭 #126 的「内存安全优先于下限」语义：超长文本下 `budget // max_len²` 可能算成 0，子批被**夹到 1 条**。单条长切片会把吞吐打到最优的 50-69%（实测低于 4 条即明显掉），且模型加载 / ONNX 会话初始化的固定成本被单条摊得极薄——纯亏 | 超长切片（>1732 字符）入库时子批恒为 1，吞吐塌陷且调优器无法在可行域内爬山 | `_shape_cap_size` 与 `_embedding_batch_for` 的形状预算**下限改为 2 条**（与调优器探索下限 `_SUBBATCH_MIN` 对齐，两者不再互相打架）：`min(size, max(_SUBBATCH_MIN, budget // max_len²))`。请求本身不足 2 条时按请求返回，不为凑下限放大批次（单条直调仍为 1）。副作用：未知/默认后端在 >1732 字符切片上从 1 条变 2 条——int8 ONNX @8192 B=2 实测内存可接受；调优器 report() 收到的仍是钳制后实际条数 |
| 135 | 子批区间 1-5 / 初始 2 是 fastembed(ONNX CPU) 时代的标定（「子批增大吞吐单调变差」）。MLX 直载后曲线反转：短切片 2→88、8→152、16→178、32→205、64→205、128→190 条/s；长切片（~1200 字符）与子批数无关（1-16 均 13.2 条/s）。原区间把吞吐锁在最优的 50-69% | 默认后端换成 MLX 后自适应器仍按旧曲线取小批，白白损失一半吞吐 | 区间改为 **2-64、初始 16、步长 2**（用户定参），RSS 软墙保留 8GB；`rag_embedding_batch_size` 默认 8→16 并夹 `[2,64]`；形状预算仍可把单批压到 1（内存硬约束优先于下限）；子批测试改为常量驱动（改档位不改测试）；实测 96 条语料 batch 8→16：173→211 条/s（+22%） |
| 140 | 配置窗口原本是一长条平铺行：主模型后端、远程模型名、Base URL/API Key、采样参数、目录、Embedding、Reranker 全挤在一起，「本地 vs 接口」只是下拉框里的一项，加载/更换模型没有显式入口，也没有辅助模型 | 配置项语义混杂、找不到该点哪里；主模型一换就要重新读数 GB 权重；轻量子任务（上下文摘要）只能占用主模型的推理队列 | 配置窗口重构成**两个分区**：① **Agent 模型配置**——主模型模式二选一（本地 / OpenAI 兼容接口，默认本地）；本地模式两行（模型类型 MLX-LM/MLX-VLM/ONNX/GGUF + 本地路径 + 选择；采样参数 + **加载模型** + **卸载**），接口模式两行（Base URL / API Key / 模型 + 获取模型；采样参数 + 应用配置）；下方**辅助模型**同本地模式结构、独立采样参数。② **知识同步配置**——目录；Embedding（后端 + 模型名 + 本地路径 + 同步知识）；Reranker（后端 + 模型名 + 本地路径 + 停止同步）。`llama_cpp` 在 UI 上一律改名 **GGUF**，Embedding 下拉补齐 GGUF 项。启动即默认加载主/辅模型（后台线程，`BASE_AGENT_SKIP_MODEL_AUTOLOAD=1` 可跳过）；「加载模型」主动把权重读进内存并直接交给 agent，「卸载」释放后给 agent 换新懒加载后端；**按签名复用**已加载权重，应用配置不再重复读数 GB；加载期间配置变更则丢弃陈旧结果；辅助模型参数改为**实时视图**（`_AuxConfigView`），改完即生效无需重载。辅助模型接入 agent 的上下文摘要，失败自动回退主模型。默认主模型 `qwen3.5-4b-agentic-coder-v4-4bit-mlx`、辅助 `Qwen3.5-2B-MLX-4bit`；旧配置无 `llm_mode` 时按 `llm_backend` 反推，不会把 OpenAI 用户静默切到本地 |
| 141 | docling/marker 转换器「锁外取实例 + 锁内转换」与周期回收并发 | 两代转换器并存（各 ~1.9GB 统一内存），回收形同虚设 | 改 `RLock` + 「取+用」原子化（`_convert_with_scale` 同锁完成取实例与 convert）；marker 同款修复 + `is_marker_available()` 改 `find_spec` 轻量探测（旧版真 import 会拉起 ~2s 重依赖） |
| 142 | OCR 信号量包住整页「渲染+预处理+OCR」 | ocrmac（ANE）本可并发被整页串行化，白损吞吐 | 信号量移入 `_ocr_image_cached` 仅包 RapidOCR 的 `engine(img)`；`_extract_pdf` 外层包裹移除 |
| 143 | ovis 文本页一律走 HTML DOM 表格解析；MLX 加载后无内存上限 | 纯文本页整页 DOM 解析浪费；MLX 缓存随推理增长 | `_vector_table_lines(page)`（`get_drawings()` 数长直线，<4 条跳过表格填回）；引擎加载后调 `_cap_mlx_memory_limits()` |
| 144 | ovis `release_engine()` 与在途推理竞态：置 None 后紧接着重建 | 两代 ~0.9GB 引擎并存（#141 同源） | 非阻塞获取 `_infer_lock`，在途推理时跳过本次释放，留待下次回收 |
| 145 | 全跳过分支把全库总量当成本次写入量（`stats["chunks"] = store.count()`）；`_on_batch` 读 stats 不持锁；检索日志变量遮蔽 `source` 参数；`_cleanup_deleted_files` 返回注解错误 | UI「本次同步切片数」显示误导；极小概率读到撕裂的进度值 | 全跳过分支改为 0（与取消分支对齐）；读 stats 加锁；循环变量改名；返回注解修正 `Set[str]` |
| 146 | 生命周期兜底缺口：① 关窗不关辅助 LLM；② terminate() 杀掉的 sync worker 不走 finally，RAG 引擎 ~600MB 残留；③ 模型加载在途时同步前 gc 扫描会关掉加载中的后端；④ agent 侧 RAGEngine 少转发 4 个 OCR/动态倍率参数；⑤ SyncKnowledgeWorker 漏 `rerank_mlx_model_path` | 数 GB MLX 权重残留到进程退出；同步前 gc 扫描让「加载模型」随机失败；agent 路径 OCR 引擎选择静默失效 | closeEvent 补 `_aux_llm.close()` 与 sync worker 引擎兜底关闭（close 幂等）；gc 扫描加 `_model_busy` 守卫；`tools/base.py` 补 `ocr_backend/ocr_model/ocr_model_path/docling_dynamic_scale` 转发；SyncKnowledgeWorker 补 `rerank_mlx_model_path`；ovis `_extract_pdf` finally 补 `store_shrink(100)` |
| 147 | `agent.run()` 的 except 分支内调用 `_shrink_context`，其内部 `json.dumps` 抛异常会逸出 `run()` | UI 永久 busy（on_finished 永不执行） | 收缩调用包进独立 try/except，失败仅降级为上报错误 + 结束，绝不逸出 |
| 148 | `web.py` SSRF 只校验输入 URL，全局 `follow_redirects=True` 让 `302 → 链路本地/元数据地址` 绕过校验 | 云凭据 / 内网被窃取 | httpx 挂 `event_hooks["response"]` 对每个重定向 hop 重新过 SSRF 校验；`_is_ssrf_target` 补充拦截 `0.0.0.0/8` |
| 149 | `agent.py` 后台事实抽取线程复用本地后端（带 `_generation_lock`）的 `self.llm` | 抽取占用生成锁、阻塞用户下一条回复，并污染生成上下文 | 仅带 `_generation_lock` 的本地后端跳过后台抽取；测试替身等无状态对象仍复用 |
| 150 | `ToolRegistry.shutdown()` 与 `_get_long_memory`/`get_skill_retriever` 创建之间竞态 | 重建 Agent 时泄漏一个长期记忆/技能检索器单例（~137MB + 连接） | 两个 getter 创建后复查 `_shutdown`，已关闭则立即 close 丢弃 |
| 151 | `graph_memory.close()` 后守护抽取线程仍调写路径 → 重新 `lancedb.connect` 且永不回收 | 进程级 LanceDB 连接泄漏 | close 置 `_closed`，`_ensure_vectors` 顶部已关闭实例直接放弃重开 |
| 152 | `SkillRetriever.build()` 持锁后无条件 drop 重建，并发首次建表互相丢弃 | 全量 embed 被重复执行，浪费数分钟 | 持锁后再检查表/self._table，命中则复用 |
| 153 | 共享 FastEmbed ONNX 单例被 RAG/记忆/技能并发 embed | 偶发 `Ort::Exception` 或向量错位 | `EmbeddingFunction` 的 embed 调用统一加 `_embed_lock` 串行 |
| 154 | `config.py` 新增数值/布尔字段漏登记到强转清单，旧配置字符串原样进 dataclass | 重启即崩溃或算术运行时错 | `_coerce` 兜底按 dataclass 声明类型（从 `__dataclass_fields__`）强转 |
| 155 | `_rebuild_agent` 孤儿后端关闭守卫依赖 `_model_obj`，`LLMClient` 无此属性 | OpenAI 后端 httpx 连接池泄漏 | 守卫改为「有 `close()` 且未 `_closed`」 |
| 156 | 首检 `search()` 同步全量回填向量（模型下载时卡数分钟）；技能检索每次查询全目录 `os.walk`+`stat` | 首次语义检索阻塞调用线程；每次检索附带目录遍历 | 首检回填改为后台 daemon 线程异步执行（本次先走 keyword 兜底）；mtime 计算节流留待后续 |
| 157 | MLX 聊天/视觉后端加载时不调 MLX buffer cache 上限（默认≈95% RAM） | 本地模型对话时 cache 涨到系统上限 → 内存冲到 13G+ 卡死（权重仅 ~3G，差额全是 cache） | 两个 MLX 后端 `_ensure_loaded` 内 `mx.set_cache_limit(1GB)` 封顶 scratch buffer（不限权重体积，免大模型加载失败） |
| 158 | llama.cpp (GGUF) 后端用 `max_context_tokens`(默认 32768) 当 `n_ctx`，加载时一次性预分配 KV cache | 发句「hi」内存冲到 23G+ 卡死（权重~4G+KV~17G+嵌入~2G）；MLX/API 按需增长不受影响 | 新增 `local_llm_n_ctx`(默认 16384) 封顶预分配；后端暴露 `context_window`，agent 收缩阈值改用 `model_context_limit` 防喂超 |
| 159 | #157 把 MLX buffer cache 压到 1GB 过激，prefill 长上下文时大注意力矩阵每步反复释放重分配 | 内存不爆了但每句话都卡半天、且历史越长越卡 | 抽出统一 `apply_mlx_memory_policy()`：cache 封顶 `min(40% RAM,6GB)`、总内存封顶 `min(当前,80% RAM)`；聊天/VLM/RAG 三处复用同一策略 |
| 160 | RAG `list_sources()` 用 `to_arrow(columns=["source"])` 投影，本地 LanceDB 不支持该参数 → 兜底整表 `to_arrow().to_pylist()`（57 万行含向量） | `status()` 每轮调用，单次 200+ 秒 → 每句话卡几分钟、彩虹球 | 改用 `search().select(["source"])` 投影下推（~0.6s）；documents 表同款修正 |
| 161 | `AgentWorker.run()` 每条消息返回、下次 `start()` 换新 OS 线程；MLX KV 缓存线程局部 → 每轮作废 | 每轮全量 prefill ~3500 token（~25s），第一句最慢 | 改常驻 worker（同一线程）；加载后 `prime_prefix_cache` 只 prefill 不生成预热 system+tools 前缀，首条命中缓存 |
| 162 | 并发闸门 `release(size)` 按**当时的**阈值重新判定大/小文件 | 阈值在 acquire/release 之间被热改（引擎每次构造都会 configure）时，读者计数被减到写者分支 → `_readers` 只增不减 → **所有后续大文件转换永久阻塞**，同步卡死且无超时无日志 | `acquire()` 返回许可令牌固化分类，`release(permit)` 凭据归还；重复/多余归还记日志并忽略（防止 `_readers` 变负破坏大文件独占） |
| 163 | OCR 预处理把 Otsu **二值化**后的纯黑白图统一喂给 ocrmac 与 RapidOCR | 二值化利于 RapidOCR 的 det/rec，却抹掉 Apple Vision 依赖的抗锯齿笔画与灰度过渡 → ocrmac 读不出字 → 大量页面逐页回退 RapidOCR（日志上看似「切到 RapidOCR」） | 预处理按引擎分流：**ocrmac 走灰度（不二值化）**，仅 RapidOCR 兜底通路二值化；上游只送灰度原图 |
| 164 | ocrmac autorelease 补丁把**构造**也包进 `autorelease_pool`，且 `return inst` 位于 `with` 内 | 池在识别发生前就排空，构造期创建的 ObjC 对象被提前释放（悬垂引用），与「每页识别完再排空」的意图相反 | 只给 `recognize()` 套池，构造留在池外 |
| 165 | RapidOCR 单例快路径无锁读取 + 引擎引用在信号量外取用 | 释放与取用之间竞态 → 两代 ~500MB 引擎并存（历史「内存涨到 16GB」的残形态）；极端下释放掉 worker 正握着的引擎 → 原生段错误 | 快路径纳入锁；`_ocr_image_cached` 改为**在信号量内取引擎**，`release_ocr_engine()` 先抢信号量再释放 |
| 166 | `engine.ingest()` 的 Step 5~7（清理删除项/副本去重/保存 manifest/补刷目录）任一步抛异常 | 心跳与内存采样线程只在两条正常 return 停下 → 异常时线程带着闭包（`_inflight`/`stats`/`stats_lock`）活到进程结束，每 60s 向已结束的同步写日志 | 整段包进 `try/finally` 无条件停线程 |
| 167 | 全部文件未修改（增量全跳过）时用户点「停止同步」 | 引用未初始化的 `kept_files` → `UnboundLocalError`，第 5/6 步整段跳过（删除项不清理、manifest 不保存） | `kept_files` 提到顶层初始化 |
| 168 | docling/marker 构建转换器的**任何**异常都永久置 `_converter_unavailable` | 一次瞬时 MPS 显存抖动/surya SpawnError 就让引擎在整个进程生命周期内静默失效，全部 PDF 掉到轻量兜底且无法自愈 | 区分 `ImportError`（永久）与其它异常（60s 负缓存，之后自动重试恢复） |
| 169 | 同步中点关闭窗口 | `RAGEngine.close()` 内部会 join ingest worker（默认 deadline **600s**），而 `QThread.terminate()` 杀不掉卡在 docling 原生线程池里的线程 → 主线程同步阻塞最多 10 分钟，窗口不消失、事件循环不转，表现为彻底卡死 | 仅当线程**确认停止**才 `close()`；否则跳过（宁可让引擎随进程回收，也不在主线程 join 第三方原生线程） |
| 170 | 自定义信号槽（`deps_finished`/`sync_finished`/…）里把 worker 引用置 None + `finished→deleteLater` | 那些信号在 `run()` **内部** emit，emit 与线程真正结束之间有窗口；窗口内引用归零/DeferredDelete 被模态框的嵌套事件循环派发 → `QThread: Destroyed while thread is still running` qFatal → 进程 abort（无 Python traceback） | 新增 QThread 保活登记表 `_worker_refs` + `_retain_worker()`：创建即登记，仅线程确认结束后解除；业务槽照常置 None 但不再承担保活职责 |
| 171 | `closeEvent` 完全没纳管 `LoadModelWorker`；`_bridge.blockSignals` 放在方法末尾；漏停 `_stream_flush_timer`/`_reason_flush_timer` | 启动后几十秒内关窗 → 仍在跑的加载线程被遗留（退出时可能 abort）；bridge 信号屏蔽晚于 agent/llm 释放 → 跨线程 UAF；漏停定时器在关窗后 fire 打到已释放控件 | 纳管 LoadModelWorker、bridge 屏蔽提到最前、补停两个定时器 |
| 172 | 同步前 gc 扫描「孤儿 LLM 后端」的守卫只看 `_model_busy` 标志 | 该标志在 `load_done` 槽里就复位、而加载线程未必结束 → 扫描会关掉 worker 正在读取权重的后端（「加载模型」随机失败，重则原生层竞态） | 守卫加上「加载线程是否仍在 `isRunning()`」 |
| 173 | 分段转换的 `fitz.open()` 子文档（`_extract_pdf_by_scale`）与 pandoc epub 图片的 `Image.open()` 无异常保护 | `insert_pdf`/`save`/`np.asarray` 抛异常时不关闭 → 每失败一个文件泄漏一个 fitz.Document / PIL 句柄与解码缓存 | 分别补 `try/finally: sub.close()` 与 `with Image.open(...)` |
| 174 | `ovis_ocr.release_engine()` 的 `mx.clear_cache()` 写在 `if _engine is not None:` 内 | 该函数是每轮同步结束的收尾冲刷入口；第二轮以后（引擎已释放）直接跳过 → MLX 缓存单调增长，「同步跑一会就被系统杀」 | 移出 `if`，无条件冲刷 |
| 175 | `vector_store` 判断共享 EmbeddingFunction 用 `self._ef in _EF_SINGLETONS.values()` | `in` 走 `==`，触发对象自定义 `__eq__`；一旦比较抛异常就把共享单例误判为私有 → `session.release()` 砸掉 graph_memory 正在用的模型 | 改用 identity：`any(v is self._ef for v in _EF_SINGLETONS.values())` |

## 部署

详细部署文档见 [docs/deploy.md](docs/deploy.md)。

## License

MIT

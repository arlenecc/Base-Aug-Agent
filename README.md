# Base Agent

本地自主 Agent，基于 PyQt6 图形界面，支持 OpenAI 兼容的大语言模型，内置可插拔工具系统和本地 RAG 知识库。

## 功能特性

- **多模型兼容** — 支持所有 OpenAI 兼容 API（OpenAI / DeepSeek / Qwen / 本地 vLLM 等）
- **流式对话** — 实时显示推理过程（thinking trace）和生成内容，支持 token 速度实时统计（仅计输出 token，不含 prompt）
- **工具调用** — 内置 15 个工具（12 通用 + 3 RAG；`webexec_js` 仅在配置浏览器端点时注册），支持模型自主决策和工具链式调用
- **知识图谱记忆** — 长期记忆采用知识图谱（实体 + 关系 + 观察），对话结束后自动用 LLM 抽取事实入图谱（异步 daemon 线程，不阻塞用户回复）；观察通过 FastEmbed + LanceDB 语义检索，精简注入 prompt 避免全量灌入拖慢响应
- **本地 RAG 知识库** — 自动解析文档、清洗、切片、向量化，BGE 重排序精准召回
- **增量同步** — 通过 manifest.json 记录文件签名（mtime + size + content_hash），未修改文件完全跳过；右键「同步知识」可强制全量重处理
- **上下文收缩** — 估计 prompt 达到 90% 上下文窗口时主动摘要旧消息；遇到 `context_length_exceeded` 错误时被动收缩重试；摘要持久化到 `workspace/memory.md`
- **依赖自动管理** — 同步知识库前自动扫描文件类型、检查并安装缺失的解析依赖
- **全链路日志** — 知识同步的每个关键节点（文件解析 → 清洗 → Markdown 转换 → 切片 → 向量化 → 入库 → 清理 → Manifest 保存）都通过 QTimer 轮询 + 线程安全日志缓冲区实时显示在右侧日志面板中；知识检索时展示向量检索候选数、BGE Reranker 精排过程和最终结果排名
- **协作式取消** — 同步过程中可随时停止，worker 在文件/批处理边界安全退出；已入库数据不丢失，Manifest 保持一致性
- **内存管理** — Embedding 模型进程级单例共享（RAG + 知识图谱共用同一 FastEmbed 实例，~137MB 而非 ~274MB）；LongTermMemory 单例缓存避免重复加载；`ToolRegistry.shutdown()` 统一释放 RAG 引擎 + 知识图谱 LanceDB 连接 + ONNX 模型 + BGE reranker；ONNX Runtime InferenceSession 共享时不 release（避免破坏其他调用方）；OCR 引擎全局单例；窗口关闭时统一清理 QThread worker 和 logger handler
- **Prompt 优化** — SYSTEM_PROMPT 精简至 ~260 tok（较原来 -65%）；Knowledge base 指令条件注入（无 KB 时省 272 tok/轮）；`webexec_js` 条件注册（无浏览器时省 146 tok/轮）；Tool descriptions 精简；Work memory 注入用 compact JSON；总体每轮节省 ~787 token（-34%）
- **MCP 协议** — 对接通用 MCP Server，自动注册远程工具
- **确认机制** — 工作区内操作自动执行，`shell_run` / `code_run` 等高风险操作需用户确认
- **技能系统** — 自动识别用户意图匹配技能；支持目录化技能（`.agent/skills/` 下每个技能一个目录，含 `skill.json` 元数据 + `prompt.md` 指令），`SkillIndex` 维护 `_index.json` 索引，`skill_search` / `skill_load` 按需发现与加载，技能指令不进系统提示词，节省 token

## 界面布局

```
┌──────────────────────────────────────────────────────────────────┐
│ [Base URL] [API Key] [模型] [获取模型]                    Row 0  │
│ [上下文长度] [温度] [top_p] [min_p] [top_k] [重复惩罚] [超时] [应用配置] │
│ [工作目录] [知识库] [同步知识] [停止同步]                   Row 2  │
├──────────────────────────────┬───────────────────────────────────┤
│ [对话] — [清空对话] [终止对话] │ 思考过程 / 日志                    │
│                              │                                   │
│ 对话内容...                   │ 推理过程实时显示...                 │
│                              │ 知识同步进度实时显示...              │
│                              │ 检索过程日志实时显示...              │
├──────────────────────────────┴───────────────────────────────────┤
│ 输入框                                               [发送]      │
├──────────────────────────────────────────────────────────────────┤
│ [状态栏]                                   [进度条] [token 速度]  │
└──────────────────────────────────────────────────────────────────┘
```

**按钮说明**：
- 「同步知识」：增量同步知识库（右键可强制全量重处理）
- 「停止同步」：仅在同步进行中启用，协作式取消（完成当前文件/批次后退出）
- 「终止对话」：终止当前 agent 对话（完成当前工具后退出）
- 「应用配置」：在 agent 运行期间会被阻止，需先终止当前对话

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
| 核心 | `pip install -e .` | PyQt6, httpx, beautifulsoup4, mcp |
| RAG | `pip install -e ".[rag]"` | lancedb, fastembed, FlagEmbedding, python-docx, openpyxl, python-pptx, PyMuPDF, ebooklib |
| OCR | `pip install -e ".[ocr]"` | rapidocr-onnxruntime（图片型 PDF 识别） |
| 测试 | `pip install -e ".[test]"` | pytest, pytest-qt, pytest-cov, ruff |
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
| `code_run` | 执行 Python 代码 | ✅ |
| `web_scan` | 抓取网页内容 | ❌ |
| `webexec_js` | 浏览器执行 JS（仅配置浏览器端点时注册） | ❌ |
| `ask_user` | 向用户提问 | ❌ |
| `work_memory` | 短期工作记忆（键值对 scratchpad） | ❌ |
| `memory_graph` | 知识图谱 CRUD（实体/关系/观察） | ❌ |
| `memory_search` | 长期记忆语义检索 | ❌ |
| `skill_search` | 搜索/列出技能目录中的技能 | ❌ |
| `skill_load` | 按路径加载技能完整指令（prompt.md） | ❌ |
| `rag_search` | 搜索本地知识库（支持按文档名定向检索） | ❌ |
| `rag_outline` | 获取文档缩略版本（目录结构 + 章节标题） | ❌ |
| `rag_status` | 查看知识库状态 | ❌ |
| `rag_ingest` | 增量索引知识库 | ❌ |

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
▶ 知识库同步开始: 目录=/data/kb 模式=增量同步 并发数=4
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

- **嵌入模型**：`nomic-ai/nomic-embed-text-v1.5-Q`（Q4 量化版，~137MB），通过 FastEmbed (ONNX Runtime) 本地推理，无需 GPU、无需 HuggingFace 网络、无需 PyTorch。768 维向量，量化后 4x 更小，质量无损。首次下载后离线可用。
- **向量数据库**：LanceDB（本地 Lance 列式格式），轻量、零配置、支持高效向量检索。向量归一化为单位长度后用 L2 距离模拟余弦相似度。每切片同时存入 embedding 向量、完整原文、来源文件名、切片编号，并开启 BM25 全文索引（原生 FTS + jieba 预分词）。
- **混合检索**：向量相似度 + BM25 关键词并行检索（各取 top_k×4 候选），按 (source, chunk_index) 合并去重，兼顾语义召回与精确关键词命中（中文经 jieba 分词）。
- **重排序**：BGE Reranker (`BAAI/bge-reranker-base`)，对混合候选精排，提升召回精度。无重排序依赖时自动回退到距离排序。
- **相似度过滤**：rerank 后过滤语义相似度 < 0.7 的结果（不相关的切片不发回模型），最终返回 ≤ 3 条最相关切片（含完整原文与来源信息）。

### Meta-context + Targeted RAG（动态按需检索）

针对"用户提到具体书名/文档"的场景，实现**先掌握全局结构、再按需提取细节**的动态 RAG：

```
文档解析 (docling → Markdown)
  └─ 提取目录结构（章节树 + 标题层级）→ 形成「缩略版本」
  └─ 存入 LanceDB documents 表（文档名 / 缩略版本 / 完整 Markdown）

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
- **流式 ingest**：worker 池（默认 4 线程）并行解析+切片，主线程通过有界队列（容量 = 2×workers）消费并增量写入向量库，背压机制避免内存堆积。OCR 通过全局信号量串行执行，单例引擎避免多 worker 重复加载模型。
- **Markdown 缓存**：解析结果缓存为 Markdown，基于源文件 mtime 判断是否需要重新解析，避免重复处理。
- **批量嵌入**：嵌入按批次进行（默认 20 条/批），避免大知识库一次性嵌入导致 OOM。每批次内再拆分为 5 条/子批，子批次间主动释放 GIL 让 UI 保持响应。每批次嵌入前检查取消标志，及时响应停止请求。
- **协作式取消**：取消后 worker 在文件/批处理边界退出，已写入的向量数据保持完整；`_chunk_iter` 使用 0.5s 短超时确保取消后快速响应；`add_streaming` 在每批处理前检查取消标志避免无效计算。
- **资源释放**：同步完成后 `RAGEngine.close()` 显式释放 LanceDB 连接、FastEmbed ONNX 模型、BGE reranker、RapidOCR 引擎（~500MB，仅图片 PDF 入库时加载）；`VectorStore.close()` 显式调用 ONNX `InferenceSession.release()` 立即回收 C++ 堆内存；窗口关闭时移除 logger handler 防止悬空引用；`_JsonStore` 采用 0.5s 节流写盘避免高频 I/O。
- **文档元数据缓存**：`VectorStore` 对 documents 表内容做内存缓存，`get_document_digest` / `list_documents` 不再每次全表 `to_pylist()`（表内含完整 Markdown，反复全量扫描既慢又占内存）；写入（upsert/delete/clear）时失效缓存。documents 表写入用 `_documents_lock` 串行化，避免多 worker 并发 ingest 时 delete+add 交错。
- **Reranker 加载去重**：BGE reranker 加载用状态机（`idle/loading/done/failed`）标记，并发调用不会各自起加载线程（每线程 ~500MB PyTorch 权重）；超时后后台线程完成时仍写入结果，后续调用直接复用。

### 支持的文档格式

**统一解析引擎（docling）**：Word、Excel、PowerPoint、PDF、EPUB 通过 [docling](https://docling-project.github.io/) 统一解析，直接输出结构化 Markdown（含布局、表格、标题、图片 OCR）。docling 未安装时自动回退到按格式的轻量解析器。

| 格式 | 扩展名 | 解析方式 |
|------|--------|------|
| Word | `.docx`, `.doc` | docling（回退 `python-docx`） |
| Excel | `.xlsx`, `.xls` | docling（回退 `openpyxl`） |
| PowerPoint | `.pptx`, `.ppt` | docling（回退 `python-pptx`） |
| PDF | `.pdf` | docling（图片型/纯扫描走 `rapidocr-onnxruntime`） |
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
- **抽取串行化**：`_fact_extract_lock` 非阻塞锁，避免抽取线程与下一轮对话并发争用 LLM 的 `httpx.Client`（非线程安全）。

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
- **SYSTEM_PROMPT**：精简至 ~260 tok（-65%），合并冗余段落
- **Knowledge base 指令**：仅当 KB 有数据时注入（无 KB 省 272 tok/轮）
- **工作记忆**：手动 scratchpad（compact JSON，value 截断 200 字符）+ 当天自动抽取事实（列表形式）
- **长期记忆**：不注入 prompt，靠 `memory_search` 工具按需检索（避免长期记忆膨胀拖慢每轮响应）
- **总体每轮节省 ~787 token（-34%）**

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

技能用于把高频任务固化为可复用的指令模板。系统包含两层：

- **SkillManager**（`skills.json`）：按关键词匹配用户意图，命中后把技能 prompt 注入系统提示；同一意图出现 ≥ 2 次时建议固化为技能
- **SkillIndex**（`.agent/skills/` + `_index.json`）：目录化技能注册表。每个技能一个目录，含 `skill.json`（元数据：name / description / keywords / tags / entry）和 `prompt.md`（指令全文）

工作方式：

```
用户请求
  ├─ SkillManager.match() 命中扁平技能 → prompt 注入系统提示
  │   （目录技能注册的扁平记录引导 agent 调用 skill_load 加载全文）
  └─ 复杂任务 → agent 主动调用 skill_search(query) 检索技能目录
                └─ 命中后 skill_load(path) 读取 prompt.md 全文并遵循
```

关键设计：

- **按需加载**：技能全文不进系统提示词，仅在需要时通过 `skill_load` 读入，避免技能增多导致 prompt 膨胀
- **索引防抖**：`_index.json` 在目录 mtime 变化或 300s 间隔后重建，原子写（tmp + rename）
- **加权检索**：name ×3 / keywords ×2 / tags ×2 / description 子串 ×1
- **路径安全**：`skill_load` 对 `path` / `entry` 做 realpath 前缀检查，拒绝 `..` 等路径穿越
- **创建技能**：`SkillManager.create_dir_skill(name, keywords, prompt, ...)` 自动生成目录结构并注册扁平记录

## 上下文收缩

Agent 在长对话中会自动管理上下文窗口，避免超出模型 token 限制：

- **主动收缩**：当估计的 prompt 大小达到 `max_context_tokens × 90%` 时，自动摘要旧消息，保留最近 6 条消息（当前任务上下文）verbatim，旧消息压缩为单条摘要标记。Token 估算采用增量计数（O(1) 而非 O(N) 每次全量扫描），高频场景下性能更优。
- **被动收缩**：遇到 LLM API 返回 `context_length_exceeded` 类错误时，自动收缩并重试同一轮（不消耗迭代次数），最多重试 3 次
- **持久化**：摘要写入 `workspace/memory.md`（带时间戳和触发原因），支持长期/跨会话回溯
- **摘要策略**：调用 LLM 生成要点摘要（用户需求、决策、已完成步骤、未完成子任务、关键路径/URL/数值）；LLM 不可用时回退到结构化摘要（用户请求 + 工具调用列表）
- **悬挂消息修复**：收缩时确保历史不以悬挂 `tool` 消息开头（否则 API 拒绝请求），自动将对应的 `assistant(tool_calls)` 拉回保留段

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

启动后自动连接 MCP Server，注册其提供的工具。

## 项目结构

```
src/agent/
├── agent.py              # 核心 Agent 推理循环（含增量 token 计数、prompt 缓存、自动事实抽取）
├── config.py             # 配置管理
├── llm_client.py         # LLM 流式客户端
├── memory.py             # 工作记忆 / 长期记忆（知识图谱，节流写盘）
├── graph_memory.py       # 知识图谱存储引擎（Entity/Relation/Observation + 批量嵌入 + LanceDB 语义检索）
├── skills.py             # 技能管理（意图匹配 / 请求固化建议 / 目录技能创建）
├── skill_index.py        # 技能索引（扫描 .agent/skills/、维护 _index.json、加权检索）
├── rag/
│   ├── engine.py            # RAG 引擎（编排：扫描→对比→解析→清洗→切片→向量化→清理→保存）
│   ├── docling_parser.py    # docling 统一解析器（Word/Excel/PPT/PDF/EPUB → Markdown）
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
│   ├── code_run.py       # Python 代码执行（协作式取消 + 泄漏线程追踪）
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

测试覆盖：Agent 推理循环、工具调用、上下文收缩（主动/被动/悬挂消息修复）、LLM 客户端、RAG 全流程（端到端 + 集成 + manifest 增量同步 + 异常文件处理）、RAG 工具自主发现与调用、依赖检查、技能系统（意图匹配 + SkillIndex 索引/检索/路径安全）、UI Bridge、内存存储、文档目录提取（Markdown 标题 + 纯文本启发式）等。当前 311 测试用例。

## 部署

详细部署文档见 [docs/deploy.md](docs/deploy.md)。

## License

MIT

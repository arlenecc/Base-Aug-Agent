# Base Agent

本地自主 Agent，基于 PyQt6 图形界面，支持 OpenAI 兼容的大语言模型，内置可插拔工具系统和本地 RAG 知识库。

## 功能特性

- **多模型兼容** — 支持所有 OpenAI 兼容 API（OpenAI / DeepSeek / Qwen / 本地 vLLM 等）
- **流式对话** — 实时显示推理过程（thinking trace）和生成内容，支持 token 速度实时统计
- **工具调用** — 内置 13 个工具，支持模型自主决策和工具链式调用
- **本地 RAG 知识库** — 自动解析文档、清洗、切片、向量化，BGE 重排序精准召回
- **增量同步** — 通过 manifest.json 记录文件签名（mtime + size + content_hash），未修改文件完全跳过；右键「同步知识」可强制全量重处理
- **上下文收缩** — 估计 prompt 达到 90% 上下文窗口时主动摘要旧消息；遇到 `context_length_exceeded` 错误时被动收缩重试；摘要持久化到 `workspace/memory.md`
- **依赖自动管理** — 同步知识库前自动扫描文件类型、检查并安装缺失的解析依赖
- **全链路日志** — 知识同步的每个关键节点（文件解析 → 清洗 → Markdown 转换 → 切片 → 向量化 → 入库 → 清理 → Manifest 保存）都通过 QTimer 轮询 + 线程安全日志缓冲区实时显示在右侧日志面板中；知识检索时展示向量检索候选数、BGE Reranker 精排过程和最终结果排名
- **协作式取消** — 同步过程中可随时停止，worker 在文件/批处理边界安全退出；已入库数据不丢失，Manifest 保持一致性
- **内存管理** — VectorStore/RAGEngine 显式 `close()` 释放 LanceDB 连接 + FastEmbed ONNX 模型（~130MB）+ BGE reranker（~500MB）；ONNX Runtime InferenceSession 显式 `release()` 立即回收 C++ 堆内存；OCR 引擎全局单例；窗口关闭时统一清理 QThread worker 和 logger handler
- **MCP 协议** — 对接通用 MCP Server，自动注册远程工具
- **确认机制** — 工作区内操作自动执行，`shell_run` / `code_run` 等高风险操作需用户确认
- **技能系统** — 自动识别用户意图，匹配并激活预定义技能提示词

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
| `webexec_js` | 浏览器执行 JS | ❌ |
| `ask_user` | 向用户提问 | ❌ |
| `work_memory` | 短期工作记忆 | ❌ |
| `memory_extract` | 持久化记忆提取 | ❌ |
| `rag_search` | 搜索本地知识库 | ❌ |
| `rag_status` | 查看知识库状态 | ❌ |
| `rag_ingest` | 重新索引知识库 | ❌ |

## 本地知识库 (RAG)

### 工作流程

```
文档目录 → 文本提取 → 清洗 → Markdown 转换 → 切片 (500 tokens / 50 tokens overlap)
→ 向量化 (nomic-embed-text-v1.5 / FastEmbed ONNX) → LanceDB 存储
检索时: 初检 top_k×4 条 → BGE Reranker 精排 → 返回 top 3
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
    ├─ 文档切片: new_doc.docx → 24 个切片 (每片约 500 tokens, 重叠 50 tokens, 耗时 0.01s)
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
  ├─ 初检: 从 48 个向量中检索 top 12 候选
  ├─ 查询向量化: 将查询文本转为向量...
  ├─ 向量相似度搜索: 在 48 条向量中检索 top 12...
  ├─ 向量检索完成: 获得 12 个候选切片
  ├─ BGE Reranker 精排: 对 12 个候选进行交叉编码打分...
  ├─ BGE 重排序完成: 12 → 3 条结果
  │   [1] rag_setup.md (重排序分: 0.8523)
  │   [2] config_guide.docx (重排序分: 0.7641)
  │   [3] faq.pdf (重排序分: 0.6218)
  └─ ✅ 检索完成: 返回 3 条结果 (耗时 0.48s)
```

### 关键设计

- **嵌入模型**：`nomic-ai/nomic-embed-text-v1.5`，通过 FastEmbed (ONNX Runtime) 本地推理，无需 GPU、无需 HuggingFace 网络、无需 PyTorch。首次下载约 130MB（量化后），之后离线可用。
- **向量数据库**：LanceDB（本地 Lance 列式格式），轻量、零配置、支持高效向量检索。向量归一化为单位长度后用 L2 距离模拟余弦相似度。
- **重排序**：BGE Reranker (`BAAI/bge-reranker-base`)，对初检结果精排，提升召回精度。无重排序依赖时自动回退到距离排序。
- **增量同步**：`manifest.json` 记录每个文件的签名（mtime + size + content_hash），未修改文件完全跳过（不解析、不切片、不嵌入、不写入），已删除文件自动清理对应向量和 manifest 条目。`force=True` 可强制全量重处理。取消同步后仍会保存 Manifest 以确保下次同步的增量准确性。
- **流式 ingest**：worker 池（默认 4 线程）并行解析+切片，主线程通过有界队列（容量 = 2×workers）消费并增量写入向量库，背压机制避免内存堆积。OCR 通过全局信号量串行执行，单例引擎避免多 worker 重复加载模型。
- **Markdown 缓存**：解析结果缓存为 Markdown，基于源文件 mtime 判断是否需要重新解析，避免重复处理。
- **批量嵌入**：嵌入按批次进行（默认 20 条/批），避免大知识库一次性嵌入导致 OOM。每批次内再拆分为 5 条/子批，子批次间主动释放 GIL 让 UI 保持响应。每批次嵌入前检查取消标志，及时响应停止请求。
- **协作式取消**：取消后 worker 在文件/批处理边界退出，已写入的向量数据保持完整；`_chunk_iter` 使用 0.5s 短超时确保取消后快速响应；`add_streaming` 在每批处理前检查取消标志避免无效计算。
- **资源释放**：同步完成后 `RAGEngine.close()` 显式释放 LanceDB 连接、FastEmbed ONNX 模型、BGE reranker；`VectorStore.close()` 显式调用 ONNX `InferenceSession.release()` 立即回收 C++ 堆内存；窗口关闭时移除 logger handler 防止悬空引用；`_JsonStore` 采用 0.5s 节流写盘避免高频 I/O。

### 支持的文档格式

| 格式 | 扩展名 | 依赖 |
|------|--------|------|
| Word | `.docx`, `.doc` | `python-docx` |
| Excel | `.xlsx`, `.xls` | `openpyxl` |
| PowerPoint | `.pptx`, `.ppt` | `python-pptx` |
| PDF | `.pdf` | `PyMuPDF`（图片型需 `rapidocr-onnxruntime`） |
| EPUB | `.epub` | `ebooklib` |
| MOBI/AZW | `.mobi`, `.azw3` | calibre（可选） |
| 纯文本 | `.txt`, `.md`, `.csv` | 无额外依赖 |
| HTML | `.html`, `.htm` | `beautifulsoup4` |

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
├── agent.py              # 核心 Agent 推理循环（含增量 token 计数、prompt 缓存）
├── config.py             # 配置管理
├── llm_client.py         # LLM 流式客户端
├── memory.py             # 工作记忆 / 长期记忆（节流写盘）
├── skills.py             # 技能系统
├── rag/
│   ├── engine.py         # RAG 引擎（编排：扫描→对比→解析→清洗→切片→向量化→清理→保存）
│   ├── parsers.py        # 文档解析器（Word/Excel/PPT/PDF/EPUB/MOBI/HTML/文本）
│   ├── cleaner.py        # 文本清洗（去标签/URL/控制字符/零宽字符/空白归一化）
│   ├── chunker.py        # Token 级别切片（基于字符启发的 token 估算）
│   ├── vector_store.py   # LanceDB 向量存储 + FastEmbed 嵌入 + BGE 重排序
│   └── deps.py           # 依赖检查与自动安装
├── tools/
│   ├── base.py           # 工具注册中心（MCP + RAG 生命周期管理）
│   ├── file_ops.py       # 文件读写工具
│   ├── shell_run.py      # Shell 执行
│   ├── code_run.py       # Python 代码执行（协作式取消 + 泄漏线程追踪）
│   ├── web.py            # 网页抓取 / JS 执行
│   ├── interact.py       # 用户交互
│   ├── memory.py         # 记忆工具
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

测试覆盖：Agent 推理循环、工具调用、上下文收缩（主动/被动/悬挂消息修复）、LLM 客户端、RAG 全流程（端到端 + 集成 + manifest 增量同步 + 异常文件处理）、RAG 工具自主发现与调用、依赖检查、技能系统、UI Bridge、内存存储等。当前 100+ 测试用例。

## 部署

详细部署文档见 [docs/deploy.md](docs/deploy.md)。

## License

MIT

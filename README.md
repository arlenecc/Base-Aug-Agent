# Base Agent

本地自主 Agent，基于 PyQt6 图形界面，支持 OpenAI 兼容的大语言模型，内置可插拔工具系统和本地 RAG 知识库。

## 功能特性

- **多模型兼容** — 支持所有 OpenAI 兼容 API（OpenAI / DeepSeek / Qwen / 本地 vLLM 等）
- **流式对话** — 实时显示推理过程（thinking trace）和生成内容，支持 token 速度实时统计
- **工具调用** — 内置 13 个工具，支持模型自主决策和工具链式调用
- **本地 RAG 知识库** — 自动解析文档、清洗、切片、向量化，BGE 重排序精准召回
- **依赖自动管理** — 同步知识库前自动扫描文件类型、检查并安装缺失的解析依赖
- **MCP 协议** — 对接通用 MCP Server，自动注册远程工具
- **确认机制** — 工作区内操作自动执行，`shell_run` / `code_run` 等高风险操作需用户确认
- **技能系统** — 自动识别用户意图，匹配并激活预定义技能提示词

## 界面布局

```
┌──────────────────────────────────────────────────────────────────┐
│ [Base URL] [API Key] [模型] [获取模型]                    Row 0  │
│ [上下文长度] [温度] [top_p] [min_p] [top_k] [重复惩罚] [超时] [应用配置] │
│ [工作目录] [知识库] [同步知识]                              Row 2  │
├──────────────────────────────┬───────────────────────────────────┤
│ [对话] — [清空对话] [停止]    │ 思考过程 / 日志                    │
│                              │                                   │
│ 对话内容...                   │ 推理过程实时显示...                 │
│                              │                                   │
├──────────────────────────────┴───────────────────────────────────┤
│ 输入框                                               [发送]      │
├──────────────────────────────────────────────────────────────────┤
│ [状态栏]                                   [进度条] [token 速度]  │
└──────────────────────────────────────────────────────────────────┘
```

## 快速开始

### 安装

```bash
# 从源码安装
git clone https://github.com/arlenecc/Base-Aug-Agent.git
cd Base-Aug-Agent
pip install -e .
```

核心依赖：`PyQt6`、`httpx`、`beautifulsoup4`、`mcp`。RAG 子系统的依赖（`lancedb`、`fastembed` 等）会在首次同步知识库时自动检测并安装。

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
文档目录 → 文本提取 → 清洗 → 切片 (500 tokens / 10% overlap)
→ 向量化 (nomic-embed-text-v1.5 / FastEmbed ONNX) → LanceDB 存储
→ 检索时: 初检 12 条 → BGE Reranker 精排 → 返回 top 3
```

**关键设计**：
- **嵌入模型**：`nomic-ai/nomic-embed-text-v1.5`，通过 FastEmbed (ONNX Runtime) 本地推理，无需 GPU、无需 HuggingFace 网络、无需 PyTorch。首次下载约 130MB（量化后），之后离线可用。
- **向量数据库**：LanceDB（本地 Lance 列式格式），轻量、零配置、支持高效向量检索。向量归一化为单位长度后用 L2 距离模拟余弦相似度。
- **重排序**：可选 BGE Reranker (`BAAI/bge-reranker-base`)，对初检结果精排，提升召回精度。无重排序依赖时自动回退到距离排序。
- **Markdown 缓存**：解析结果缓存为 Markdown，基于源文件 mtime 判断是否需要重新解析，避免重复处理。
- **批量嵌入**：嵌入按批次进行（默认 100 条/批），避免大知识库一次性嵌入导致 OOM。

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
2. 对照依赖表（`EXTENSION_DEPS`）检查所需 Python 包是否已安装
3. 缺失的依赖自动 `pip install`（含版本约束，如 `numpy<2.0`）
4. 核心依赖（`lancedb`、`fastembed`）也会一并检查
5. 安装完成后自动开始同步

若自动安装失败（如网络问题），会弹出提示列出需手动安装的包。

### 使用方式

1. 在界面第三行「知识库」中输入文档目录路径（或点击 `…` 选择）
2. 点击「同步知识」，后台自动完成依赖检查、文档提取、清洗、切片、向量化
3. 同步完成后，对话中模型会自动调用 `rag_search` 检索相关知识

### Token 控制

每次 RAG 检索最多返回 3 条结果，单次消耗 ≤ 1500 tokens（在 32768 上下文窗口中占比 < 5%），确保响应速度不受影响。

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
├── agent.py              # 核心 Agent 推理循环
├── config.py             # 配置管理
├── llm_client.py         # LLM 流式客户端
├── memory.py             # 工作记忆 / 长期记忆
├── skills.py             # 技能系统
├── rag/
│   ├── engine.py         # RAG 引擎（编排）
│   ├── parsers.py        # 文档解析器
│   ├── cleaner.py        # 文本清洗
│   ├── chunker.py        # Token 级别切片
│   ├── vector_store.py   # LanceDB 向量存储 + BGE 重排序
│   └── deps.py           # 依赖检查与自动安装
├── tools/
│   ├── base.py           # 工具注册中心
│   ├── file_ops.py       # 文件读写工具
│   ├── shell_run.py      # Shell 执行
│   ├── code_run.py       # Python 代码执行
│   ├── web.py            # 网页抓取 / JS 执行
│   ├── interact.py       # 用户交互
│   ├── memory.py         # 记忆工具
│   ├── rag_tool.py       # RAG 检索工具
│   ├── mcp_client.py     # MCP JSON-RPC 客户端
│   └── mcp_tool.py       # MCP 工具适配
└── ui/
    └── main_window.py    # PyQt6 主窗口
```

## 运行测试

```bash
pip install -e ".[test]"

# 全部 185 个测试
pytest tests/ -v
```

测试覆盖：Agent 推理循环、工具调用、LLM 客户端、RAG 全流程（端到端 + 集成）、依赖检查、技能系统、UI Bridge、内存存储等。

## 部署

详细部署文档见 [docs/deploy.md](docs/deploy.md)。

## License

MIT

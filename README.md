# Base Agent

本地自主 Agent，基于 PyQt6 图形界面，支持 OpenAI 兼容的大语言模型，内置可插拔工具系统和本地 RAG 知识库。

## 内核架构（三态自动机 + 五级信息压缩）

底层控制流参照 GenericAgent 项目的内核（`agent_loop.py` / `agentmain.py` /
`llmcore.py` / `ga.py`）重构：**内核只控制流**，
历史交给 `Session`，能力交给 `code_run`/`shell_run`，经验交给 `memory`/`skills`。
详见 [docs/KERNEL_REFACTOR.md](docs/KERNEL_REFACTOR.md)。

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
   ┌────────────┐   │ tool_calls         ┌────────────┐       │
   │   THINK    │───┼───────────────────►│    ACT     │       │
   │  (决策态)   │   │                    │  (执行态)   │       │
   └─────┬──────┘   │                    └──────┬─────┘       │
         │          │                           │             │
         │ 无 tool  │                           │ next_prompt │
         │          │                           ├─────────────┘
         ▼          │                           │ 预算告罄 / 每 N 轮
   ┌────────────┐   │                    ┌──────▼──────────┐
   │   EXIT     │◄──┴────────────────────│   CONSOLIDATE   │
   │  (终止态)   │   压缩完成 → 回 THINK    │   (沉淀态)      │
   └────────────┘                        └─────────────────┘
```

**五级信息压缩**，把上下文硬性控制在 30k 以内：

| 级 | 名称 | 时机 | 手段 | 上限 |
|---|---|---|---|---|
| L0 | 原始捕获 | 工具返回瞬间 | 硬上限 head+tail | 单条 ≤ 20K 字符 |
| L1 | 工具级裁剪 | 注入本轮前 | 按 RAW 区余额 ÷ 并发数分配 `maxlen` | 单条 ≤ 6K 字符 |
| L2 | 回合摘要 | 每轮结束 | 强制 `<summary>` ≤30 字物理快照 | 每轮 ≤ 30 字 |
| L3 | 会话折叠 | 预算告罄 / 每 N 轮 | 旧轮 → 一行摘要（同类项合并） | 折叠区 ≤ 2.7k token |
| L4 | 记忆蒸馏 | 任务完成 / 检查点 | 沉淀进工作记忆 → 知识图谱 → SKILL.md | 系统提示 ≤ 3.9k token |

**30k 预算分配**：`SYS 3.9k / EARLIER 2.7k / RAW 12k / TOOLS 5.1k / RESERVE 6.3k`。
关键不变量是 **L2+L3 让历史恒定 O(1)** —— 旧实现把整条历史每轮重发，
上下文随轮次线性膨胀，只能靠超阈值后的被动摘要兜底。

## 功能特性

- **多模型兼容** — 支持所有 OpenAI 兼容 API（OpenAI / DeepSeek / Qwen / 本地 vLLM 等）
- **流式对话** — 实时显示推理过程（thinking trace）和生成内容，支持 token 速度实时统计（仅计输出 token，不含 prompt）；结构化 read 超时（流式 chunk 间隔上限 60s），本地模型生成中途卡住时自动报错而非永久卡死
- **工具调用** — 内置 21 个工具（原子 8 + 能力 12 + 1 个条件注册；另有 2 个原子命名别名，`webexec_js` / `web_execute_js` 仅在配置浏览器端点时注册），支持模型自主决策和工具链式调用
- **内核控制流** — 三态自动机（THINK / ACT / CONSOLIDATE）+ 五级信息压缩，上下文硬性控制在 30k 以内
- **知识图谱记忆** — 长期记忆采用知识图谱（实体 + 关系 + 观察），对话结束后自动用 LLM 抽取事实入图谱（异步 daemon 线程，不阻塞用户回复）；观察通过 FastEmbed + LanceDB 语义检索，精简注入 prompt 避免全量灌入拖慢响应
- **本地 RAG 知识库** — 自动解析文档、清洗、切片、向量化，BGE 重排序精准召回
- **增量同步** — 通过 manifest.json 记录文件签名（mtime + size + content_hash），未修改文件完全跳过；右键「同步知识」可强制全量重处理
- **依赖自动管理** — 同步知识库前自动扫描文件类型、检查并安装缺失的解析依赖
- **全链路日志** — 知识同步的每个关键节点（文件解析 → 清洗 → Markdown 转换 → 切片 → 向量化 → 入库 → 清理 → Manifest 保存）都通过 QTimer 轮询 + 线程安全日志缓冲区实时显示在右侧日志面板中；知识检索时展示向量检索候选数、BGE Reranker 精排过程和最终结果排名
- **失败自动恢复** — 计划步骤失败时**不停车**：内核依据失败信息自动换方案重试（反思门失败感知：失败后提醒预算 2→4 次、引用最近错误、明确禁止「停下来问用户是否继续」）；原样重试同一失败调用会被点名禁止；连续失败会要求先诊断再换方法；失败尝试不足 2 次就把步骤标 `blocked` 会被守卫拦回
- **协作式取消** — 同步过程中可随时停止，worker 在文件/批处理边界安全退出；已入库数据不丢失，Manifest 保持一致性；「终止对话」会立即中断并回收 `code_run`/`shell_run` 子进程（旧实现子进程要跑到自身超时，最长 600s）
- **内存管理** — Embedding 模型进程级单例共享（RAG + 知识图谱共用同一 FastEmbed 实例，~137MB 而非 ~274MB）；LongTermMemory 单例缓存避免重复加载；`ToolRegistry.shutdown()` 统一释放 RAG 引擎 + 知识图谱 LanceDB 连接 + ONNX 模型 + BGE reranker；ONNX Runtime InferenceSession 共享时不 release（避免破坏其他调用方）；OCR 引擎与 docling 转换器全局单例且同步结束后显式释放；`_JsonStore` 用单一全局 atexit 回调 + WeakSet 统一 flush（避免每实例注册一条永不注销的回调）；窗口关闭时统一清理 QThread worker 和 logger handler
- **健壮性** — 子进程统一 `start_new_session` 独立进程组，超时/停止时用 `killpg` 回收整棵树（`shell_run` 不留孤儿、MCP 不留僵尸）；确认弹窗默认拒绝（超时不会自动放行高风险操作）；`skill_load` 拒绝绝对路径/`..` 入口；`config.json` 原子写入 + 损坏自动回退默认配置；`web_scan` 限制 http/https 与响应体大小
- **Prompt 优化** — SYSTEM_PROMPT 精简至 ~200 tok（去掉模型已知的协议说明与冗余反例，补充「语义检索 query 用原始语言、勿翻译/音译」与「工具报错后改参数重试、两次失败即换方法」两条高价值约束）；Knowledge base 状态条件注入（无 KB 时省 tok）；`webexec_js` 条件注册（无浏览器时省 146 tok/轮）；Tool descriptions 精简（`rag_search`/`rag_outline` 去除冗长重复解释）；Work memory 注入用 compact JSON；长期记忆不注入 prompt 而靠 `memory_search` 按需检索；记忆抽取 prompt 精简并加条数/语言约束
- **MCP 协议** — 对接通用 MCP Server，自动注册远程工具
- **确认机制** — 工作区内操作自动执行，`shell_run` / `code_run` 等高风险操作需用户确认；确认弹窗只显示**用途说明**（启发式优先：注释/docstring/命令名映射，零延迟；无注释时才用 LLM 兜底生成 ≤20 字用途），不贴代码全文，避免超长文本把按钮顶出屏幕
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
| 核心 | `pip install -e .` | PyQt6, httpx, beautifulsoup4 |
| RAG | `pip install -e ".[rag]"` | lancedb, fastembed, FlagEmbedding, python-docx, openpyxl, python-pptx, PyMuPDF, ebooklib, docling, chonkie, jieba |
| OCR | `pip install -e ".[ocr]"` | rapidocr-onnxruntime（图片型 PDF 识别） |
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

工具分三层：**原子工具**（对齐 GenericAgent 的 9 个）→ **能力工具**（base-agent 原有）→ **外部协议**（MCP）。

### 第一层：原子工具

| 工具 | 功能 | 需确认 |
|------|------|--------|
| `code_run` | 执行 Python 代码 —— **能力核心**，可编程覆盖任意操作 | ✅ |
| `file_read` | 读取文件 | ❌ |
| `file_write` | 写入/追加文件 | ❌ |
| `file_patch` | 精确替换唯一文本块（`file_modify` 的原子命名别名） | ❌ |
| `web_scan` | 抓取网页内容 | ❌ |
| `web_execute_js` | 浏览器执行 JS（`webexec_js` 的原子命名别名，需配置端点） | ❌ |
| `update_working_checkpoint` | 短期工作记忆（每轮自动回注） | ❌ |
| `ask_user` | 向用户提问（中断自动机） | ❌ |
| `start_long_term_update` | 触发 L4 经验蒸馏 | ❌ |

### 第二层：能力工具

| 工具 | 功能 | 需确认 |
|------|------|--------|
| `shell_run` | 执行 Shell 命令 —— `code_run` 的系统侧延伸 | ✅ |
| `work_memory` | 短期草稿（键值对 scratchpad） | ❌ |
| `memory_graph` | 知识图谱 CRUD（实体/关系/观察） | ❌ |
| `memory_search` | 长期记忆语义检索 | ❌ |
| `skill_search` | 关键词搜索技能目录 | ❌ |
| `skill_semantic_search` | 混合检索技能（SKILL.md 向量 + BM25） | ❌ |
| `skill_load` | 加载技能完整指令 | ❌ |
| `skill_save` | **能力进化**：把成功流程固化为 SKILL.md（自动进索引） | ❌ |
| `rag_search` | 搜索本地知识库（支持按文档名定向检索） | ❌ |
| `rag_outline` | 获取文档缩略版本（目录结构 + 章节标题） | ❌ |
| `rag_status` | 查看知识库状态 | ❌ |
| `rag_ingest` | 增量索引知识库 | ❌ |

### 能力进化闭环

```
任务成功
  ├─ start_long_term_update → 蒸馏事实/偏好/坑 → memory_graph（经验）
  ├─ skill_save             → 固化可复用流程 → workspace/skills/<name>/SKILL.md（能力）
  └─ rag_ingest             → 外部语料灌入本地知识库（素材）
下次同类任务 → skill_semantic_search 命中 → skill_load → 直接复用
```

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
- **流式 ingest**：worker 池（默认 4 线程）并行解析+切片，主线程通过有界队列（容量 = 2×workers）消费并增量写入向量库，背压机制避免内存堆积。OCR 通过全局信号量串行执行，单例引擎避免多 worker 重复加载模型。
- **Markdown 缓存**：解析结果缓存为 Markdown，基于源文件 mtime 判断是否需要重新解析，避免重复处理。
- **批量嵌入**：嵌入按批次进行（默认 20 条/批），避免大知识库一次性嵌入导致 OOM。每批次内再拆分为 5 条/子批，子批次间主动释放 GIL 让 UI 保持响应。每批次嵌入前检查取消标志，及时响应停止请求。
- **协作式取消**：取消后 worker 在文件/批处理边界退出，已写入的向量数据保持完整；`_chunk_iter` 使用 0.5s 短超时确保取消后快速响应；`add_streaming` 在每批处理前检查取消标志避免无效计算。
- **资源释放**：同步完成后 `RAGEngine.close()` 显式释放 LanceDB 连接、FastEmbed ONNX 模型、BGE reranker、RapidOCR 引擎（~500MB，仅图片 PDF 入库时加载）；`VectorStore.close()` 显式调用 ONNX `InferenceSession.release()` 立即回收 C++ 堆内存；窗口关闭时移除 logger handler 防止悬空引用；`_JsonStore` 采用 0.5s 节流写盘避免高频 I/O。
- **文档元数据缓存**：`VectorStore` 对 documents 表内容做内存缓存，`get_document_digest` / `list_documents` 不再每次全表 `to_pylist()`；写入（upsert/delete/clear）时失效缓存。documents 表写入用 `_documents_lock` 串行化，避免多 worker 并发 ingest 时 delete+add 交错。
- **不存全文 Markdown**：documents 表只存「缩略版本 + 章节列表 + Markdown 缓存路径」，**不存全文**（全文已有 `rag/documents/*.md` 缓存，表内再存是第三份冗余，且会被缓存全量载入内存；该字段实际零消费者）。缓存按列读取并显式剔除 `markdown`，兼容旧数据。
- **删除即清理**：源文件从知识库移除后，同步时一并清理向量、documents 元数据行与 manifest 条目。
- **BM25 索引一致性**：写入（含单批写完的提前出口）与删除都会失效 FTS 标记并按需重建，`reload()` 也会重置——避免「混合检索静默退化成纯向量」或「已删文档仍被召回」。
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

## 上下文治理（30k 硬上限）

Agent 不再「超了才压」，而是**先分配后消费**：30k 预算按
`SYS 3.9k / EARLIER 2.7k / RAW 12k / TOOLS 5.1k / RESERVE 6.3k` 切分，
每一级压缩都在自己的配额内花钱。

- **L1 工具级裁剪**：单条工具结果按「RAW 区余额 ÷ 并发调用数」动态算出 `maxlen`，
  并发 4 个工具时每个自动只拿 1/4，不用改任何工具代码
- **L2 回合摘要**：每轮强制 `<summary>` ≤30 字物理快照，替换掉整轮原文
- **L3 会话折叠**：旧轮整轮折叠成一行摘要（连续相同动作用 `（N turns）` 塌缩）；
  折叠以「整轮」为单位，因此**结构上不可能**产生悬挂 `tool` 消息
- **L4 记忆蒸馏**：任务完成/检查点时异步沉淀进知识图谱与 SKILL.md
- **兜底收缩**：单个巨型工具输出（如整本 PDF）仍可能一次性击穿预算 —— 此时才启用
  LLM 摘要式收缩（旧的 `_shrink_context` 等价物），摘要持久化到 `workspace/memory.md`
- **注入上限**：系统提示的动态注入段（当天事实、命中技能 prompt）**不在压缩范围内**，
  因此各自设上限（事实 ≤20 条/2000 字符，技能 prompt ≤1500 字符）
- **被动收缩**：遇到 LLM API 返回 `context_length_exceeded` 类错误时进入兜底收缩；
  不重试同一个超长 prompt（重试只会再失败一次）

相关配置（`config.json`，均有默认值）：

| 配置 | 默认 | 说明 |
|---|---|---|
| `kernel_max_context_tokens` | 30000 | 整个 prompt 的硬上限 |
| `kernel_raw_turns` | 2 | 保留原文的最近轮数（调大更准、更贵） |
| `kernel_consolidate_every` | 4 | 每多少轮进入一次 CONSOLIDATE |
| `kernel_max_tool_chars` | 6000 | 单条工具结果的字符上限 |
| `kernel_max_digest_lines` | 40 | 折叠摘要区保留的最大行数 |

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
├── kernel/                  # 内核：只控制流
│   ├── outcome.py           # 三态枚举 / StepOutcome / TurnRecord / 转移表
│   ├── budget.py            # 30k 预算配额与核算
│   ├── compressor.py        # L0–L4 五级压缩 + 蒸馏 prompt
│   ├── session.py           # 历史托管：滚动折叠 + 工作记忆 + 锚点提示
│   └── loop.py              # KernelLoop：三态自动机主循环
├── agent.py              # 门面：保留 callbacks/skills/配置，把控制流委托给内核
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
│   ├── parsers.py           # 按格式的轻量解析器（docling 回退方案 + MOBI/HTML/文本）
│   ├── cleaner.py           # 文本清洗（去标签/URL/控制字符/零宽字符/空白归一化）
│   ├── chunker.py           # Token 级别递归切片 + 分片回退链编排
│   ├── hybrid_chunker.py    # Hybrid 分片（文档结构 + 语义分块）
│   ├── semantic_chunker.py  # chonkie SemanticChunker 语义分片（复用 nomic-embed-text）
│   ├── vector_store.py      # LanceDB 向量存储 + FastEmbed 嵌入 + BGE 重排序
│   └── deps.py              # 依赖检查与自动安装
├── tools/
│   ├── base.py           # 工具注册中心 + AliasTool（MCP + RAG 生命周期管理）
│   ├── checkpoint.py     # update_working_checkpoint / start_long_term_update
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

测试覆盖：Agent 推理循环、工具调用、LLM 客户端、RAG 全流程（端到端 + 集成 + manifest 增量同步 + 异常文件处理）、RAG 工具自主发现与调用、依赖检查、技能系统（意图匹配 + SkillIndex 索引/检索/路径安全 + SKILL.md 混合检索）、UI Bridge、内存存储、文档目录提取（Markdown 标题 + 纯文本启发式）等。

`tests/test_kernel.py` 覆盖内核专项：三态转移表、L0–L4 五级压缩原语、Session 折叠与
「无悬挂 tool 消息」不变量、30k 预算配额、**长任务下 prompt 峰值不越过 30k**、
9 个原子工具与能力工具的注册、`AliasTool` 委派、`update_working_checkpoint` /
`start_long_term_update` / `skill_save` 的行为与路径穿越防护。

近期补充的回归测试：首次 ingest 不再死锁（全新知识库）、`shell_run` 超时不留孤儿进程、`web_scan` 协议白名单、损坏 config 回退、`skill_load` 路径穿越防护、配置迁移不再改写用户显式设置、**「终止对话」在流式与工具链中途立即生效**、**向量化失败不再泄漏 worker 线程**、**`_JsonStore.reload` 不再丢弃防抖窗口内的写入**。

内核专项回归：`max_turns` 是「每次 run()」而非会话累计、reask 不掉最终轮、周期性 CONSOLIDATE 真的折叠、旧轮折叠后无悬挂 `tool` 消息、长任务 prompt 峰值不越过 30k。工具/RAG 专项：`skill_save` 后索引被标脏并能召回、`code_run` 超时关闭缓冲区、删除的知识库文件不再被 Step 7 复活、mtime 被保留的内容变更能被重新同步。

失败自动恢复专项（`tests/test_failure_recovery.py`，11 例）：步骤失败后想收尾被失败感知的反思门拦回、失败放宽提醒预算、原样重试被点名禁止、连续失败提示诊断换方案、防过早 blocked 守卫及其上限、成功后计数清零。复查修复专项（`tests/test_review_fixes.py`，19 例）：MCP 写队列、图谱确认/类型校验/clear 清队列、WorkMemory 淘汰强制落盘、plan patch 目标语义、抽取结果归一化、config 类型收敛与原子写、取消立即中断 code_run/shell_run、CJK 按字符截断、checkpoint 显式失败等。

当前共 502 个用例（489 通过；13 个 `test_rag_full_pipeline` 用例依赖 `python-pptx`，未安装该可选依赖时在 setup 阶段报环境错误，不影响其余测试）。

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
| 59 | `mcp.stop()` 先关管道再发 SIGTERM | 读线程阻塞在 `readline()` 时 `close()` 等到子进程退出（实测挂起 60s 的 server 卡 59s），且信号根本发不出去 | 先 `killpg` + 带超时 `wait`，进程退出后才关管道；仍存活则放弃关管道 |
| 60 | `call_tool` 只看 JSON-RPC `error`，忽略 `result.isError` | 工具失败（"permission denied"）被当成成功结果喂给模型 | `isError` 为真时抛 `MCPError`，错误正文一并带出 |
| 61 | MCP 工具不校验 `required`（`run(**kwargs)` 使签名预检失效） | 缺参原样发到远端，server 不校验时静默产生错误结果 | 按 `inputSchema.required` 本地校验，缺参不发请求 |
| 62 | skill 扫描 `os.walk` 不跟随链接 + `read_skill_dir` 用 realpath 判包含 | **`ln -s` 挂载的外部技能完全搜不到、也读不出来**（最常用挂载方式） | 自写 walker 跟随链接 + realpath 集合防环；`read_skill_dir` 只拦绝对路径与 `..` |
| 63 | SKILL.md frontmatter 只按单行解析 | `description: >-` 折叠块被解析成 `"-"`，技能进了索引但语义检索**永远召回不到** | 支持 `>` / `\|` 块标量；缩进行不再污染顶层字段 |
| 64 | skill BM25 索引缺 `replace=True` | 每次 sync 重建失败（`index already exists`），索引停在首建快照 | 与 RAG 侧一致：`replace=True` + simple tokenizer 参数 |
| 65 | `retrieve()` 用向量阈值统一过滤 BM25 候选 | 关键词精确命中（实测查询 "CI"：BM25 0.88、余弦 0.478）被整体丢弃 | BM25 命中不再受余弦阈值影响，排序仍用余弦 |
| 66 | `list_sources` 用 `to_arrow(columns=[...])`（lancedb 0.25 不支持该参数） | 两次调用都抛 `TypeError` → 永远回退全表**全列**物化（含 768 维向量），而它每轮对话都被 `_system_prompt()` 调一次：2 万切片 **7.5s**、内存 GB 级 | 只投影 `source` 列（2 万切片 7.47s → 35ms，结果一致） |
| 67 | LLM 调用失败即整轮终止 | 首字节前的网络抖动 / 429 / 502 让用户白等一场，只能手动重发 | 仅在「未收到任何 chunk」时退避重试 2 次；已吐内容、上下文过长、4xx 不重试 |
| 68 | 多个独立工具串行执行 | 多工具回合 wall time = Σ（每个 0.4s 的两个调用要 0.8s） | 只读白名单工具并发执行（UI 回调仍在主线程发出，历史保持原序）；写操作永不并发 |
| 69 | 工具执行期间无法响应取消 | `shell_run` 最长 600s、MCP `tools/call` 300s → 点「终止对话」要等它跑完 | 工具放到线程执行，主线程 0.1s 轮询取消，命中立即返回合成结果 |
| 70 | 历史里旧 tool 输出从不裁剪（单条上限 200K） | 上下文随轮次线性膨胀，且每轮整体重发 | 较早的 tool 结果裁剪为 head+tail 并标注（末尾 6 条不动）；需要完整内容可重新调用 |
| 71 | tools schema 每轮、每次迭代都重建 | 工具集固定后的无谓开销 | 缓存 schema 列表与 token 估算 |
| 72 | 迭代上限耗尽只写日志 | 回复戛然而止，模型以为任务已完成 | 追加一条 user 角色提示：先总结进展，再询问是否继续 |
| 73 | 「流未收到 done 事件」日志连写两行 | 日志噪音 | 去重 |
| 74 | `GraphMemoryStore.close()` 只置 None 从不 close | 每次「应用配置」泄漏一个 LanceDB 连接 | 显式 close + 重置 `_vec_tried` |
| 75 | 工具层 RAG 引擎初始化在 try 之外 | 知识库不可用时**所有工具（文件/shell/web）一起失效** | 整段包 try，失败仅跳过 RAG 工具 |
| 76 | `shell_run` 的 `timeout` 无上下限钳制 | 传 99999 → 阻塞 27h 且无法取消；传 0 → 立刻 kill | 钳制到 [1s, 600s] |
| 77 | `web_scan` 流式失败 `except Exception` 静默重试 | 真实超时后重复请求（最坏 2×30s）、已读内容丢弃 | 仅 `AttributeError`（无 stream 实现）才回退 |
| 78 | `_reranker_state` 打分失败后停在 `"done"` 而模型为 None | 状态机失真 | 同步置 `"failed"` |
| 79 | 每 chunk 重复计算 `_hash_source` | 写入侧纯浪费（同批同源） | 批次内缓存 source→hash |
| 80 | SYSTEM_PROMPT 含模型已知的协议说明与冗余反例 | 每轮浪费 ~100 tok | 精简；补充「语义检索 query 语言」与「工具报错重试」两条关键约束 |

### 第七轮：内核重构（三态自动机）

把 `agent.py` 的单体 400 行循环拆成「内核只控制流」的分层结构，详见
[docs/KERNEL_REFACTOR.md](docs/KERNEL_REFACTOR.md)。`Agent` 退化为门面，
历史交给 `Session`、能力交给 `code_run`/`shell_run`、经验交给 `memory`/`skills`。
旧行为全部保留，`run()` 签名与 `AgentCallbacks` 不变。

### 第八轮：内核实现复查

| # | 问题 | 影响 | 修复 |
|---|------|------|------|
| 81 | `max_turns` 用**会话累计**轮次判定 | 累计 15 轮后 Agent **彻底不再调 LLM** | 改为「本次 run()」的轮次计数 |
| 82 | reask 分支先回滚再写回 | 最终答复轮丢失，L4 蒸馏缺结论 | 不在回滚分支里 finish_turn |
| 83 | 周期性 CONSOLIDATE 传 `force_fold(0)` | **完全空操作**，turns 单调增长 | 传 `consolidate_every` |
| 84 | 预算漏算 `tool_calls` 参数 | 并发多工具静默突破 30k | 计入参数体 |
| 85 | `system`/`tools` 超支不从 RAW 扣 | 30k 只是记账，不真正约束 | 超支转嫁 RAW |
| 86 | `estimate_tokens` 密度过乐观 | 同上 | 收紧 3.0/1.5 字符/token |
| 87 | 上下文超长被当致命错误 | 兜底收缩永远走不到 | 单独分类并触发兜底 |
| 88 | 蒸馏线程无并发上限、退回共享 httpx | 并发蒸馏 + 非线程安全客户端 | 非阻塞锁 + 独立抽取客户端 |
| 89 | `skill_save` 用 `scan(force=True)` | 保存的技能**永远进不了索引** | 改用 `mark_stale()` |
| 90 | `code_run` 超时后不关缓冲区 | 泄漏线程往**真实** stdout 写 | 超时即 close，其下次写入自行退出 |
| 91 | 删除的知识库文件被 Step 7 复活 | 日志打「已清理」但仍在索引里 | 清理一并删 Markdown 缓存 |
| 92 | `content_hash` 只写不读 | `cp -p` / 解压导致**永久漏同步** | 判为「未修改」时再核一次源指纹 |
| 93 | `_on_fetch_models` 覆盖运行中 QThread | Qt6 `qFatal` **进程 abort** | 加并发守卫 |
| 94 | `closeEvent` 先 `terminate()` 再 `shutdown()` | **关窗口挂死** | 标记后跳过取锁的释放 |
| 95 | 「终止对话」无兜底 | 阻塞在长工具时所有按钮永久禁用 | 30s 看门狗强制收尾 |
| 96 | `_rebuild_agent` 失败留下半死 agent | MCP 全挂、httpx 已关 | 失败即清空引用 |
| 97 | 流式每 chunk 两次全文档光标操作 | 100 tok/s 时 UI 明显卡顿 | 40ms 合并刷新（~25fps） |
| 98 | 日志 handler 用模块级全局且不置空 | 多窗口误删、handler 常驻 | 用实例引用 + 置空 |
| 99 | docx/xlsx/pptx 句柄不关闭 | 4 worker × 数千文件累积句柄 | `try/finally` 显式关闭 |
| 100 | `skill_retriever._llm_pick` 用 `ev.get()` | LLM 择优是死代码 | 按 dataclass 属性访问 |

### 第九轮：失败自动恢复 + 全量复查（bug / 泄漏 / 性能 / 体验）

**失败自动恢复**（用户需求：计划中某项任务失败时，自动依据失败信息找方案解决并继续后续任务，而不是停下等用户输入「继续」）：

- 失败感知反思门：本次 run 发生过失败时，提醒预算 2→4 次，且提醒引用最近一次失败的**具体错误**，明确要求「读完整错误 → 换方法重试 → 多种方法都不可行才标 blocked」，禁止停下来问用户是否继续
- 原样重试检测：同名工具 + 相同参数再次失败 → 点名「禁止重复」，要求换参数/换工具/用 `code_run` 写诊断脚本
- 连续失败提示：连续失败 ≥2 次 → 要求先诊断再换方案，同一思路只试一次
- 防过早 blocked 守卫：失败尝试 <2 次就把步骤标 `blocked` → 提醒恢复 `doing` 再试（守卫有上限，模型坚持放弃时最终尊重）
- 系统提示 rule 8 与 `update_plan` 描述同步强化；回归测试 `tests/test_failure_recovery.py`

**自动续跑与任务边界（消灭「必须人工输入继续」）**：

- 自动续跑：两类停车不再甩回给用户——① MAX_TURNS 截断（长任务跑满轮次上限）；
  ② 模型提前收尾但计划仍有未完成步骤（反思门提醒额度用尽）。每段续跑都是
  完整的内核 run（取消/反思门/预算全部生效），最多续 2 段，额度用尽才交回用户
  （`agent._auto_continue` 可关闭）
- 反思门提醒补全出口：明确告诉模型「若该步骤的工作**实际已完成**，用
  `update_plan(op=patch, status=done)` 收尾即可，不要为响应提醒而重复已完成的工作」
  ——修复「实际都做完了但面板一直显示有未完成项，反复要求打继续」
- 任务边界 fork 提示：新请求到来而旧计划仍有 open 步骤时，run() 注入一次性提示，
  强制模型显式判断「新任务 → create 替换计划 / 继续旧任务 → 接着做」——修复
  「新任务的计划与输入无关（沿用了上一个任务的旧计划）」
- `update_plan` patch 用错 step_id 时，报错附上当前计划渲染，模型一次自纠

**复查修复**：

| # | 问题 | 影响 | 修复 |
|---|------|------|------|
| 101 | MCP `stdin.write()` 无超时且持锁 | server 不读 stdin 时 agent 线程**无限期挂死** | 专用写线程 + 有界队列（满/超时抛 MCPError），写失败立刻唤醒全部等待者 |
| 102 | RAG ingest 互斥锁是实例级 | UI 同步与 agent 的 `rag_ingest` 是两个 engine 实例，锁完全无效 | 模块级按 `rag_dir` 键控的跨实例锁 |
| 103 | `graph.clear()` 不清待重试队列 | 用户删掉的事实被向量检索**复活** | `clear()` 一并清空 `_pending_index`（带锁） |
| 104 | 事实抽取结果不校验内部结构 | 一个坏实体让**整轮抽取全部丢弃** | 逐条归一化：坏条目跳过、好条目保留 |
| 105 | `memory_graph` 的 clear/delete 不弹确认 | 模型一次误调用清空长期记忆 | `should_confirm` 对破坏性 op 返回 True |
| 106 | `observations` 传字符串被按字符迭代 | 生成几十条逐字符 observation 污染图谱 | 入口类型校验，明确报错 |
| 107 | WorkMemory LRU 淘汰不传 `force` | 防抖窗口内的写入被**静默丢弃** | 淘汰路径 `flush(force=True)` |
| 108 | `update_plan` patch 无 step_id 时污染 note | 第一步的结论被覆盖、目标步骤漏改 | 先筛目标集合再统一写入；只给 note 必须带 step_id |
| 109 | patch 原地改 `get_plan()` 活字典 | 校验失败时计划半改 | 深拷贝后改，一次性 `set_plan` |
| 110 | append 的 id 回退撞号 / create 强升 doing | patch 一次改中两步 / 显式状态被覆盖 | id 跳过已占用；仅全 pending 时提升 |
| 111 | `skill_save` 临时文件名只含 pid | agent 线程与蒸馏线程并发写同一 tmp → 产物损坏 | `mkstemp` 唯一名 + finally 清理 |
| 112 | `skill_search` list 无上限 | 技能多时撑爆上下文 | 截断 50 条 + 总数提示 |
| 113 | checkpoint 无 session 时静默成功 | 模型以为已写入，实际什么都没存 | 显式返回失败 |
| 114 | shell_run 取消后子进程继续跑 | 用户点终止后命令最长再跑 600s | 0.2s 粒度轮询取消标志，命中即杀进程组 |
| 115 | shell_run reader join 后无条件 close 管道 | 孙进程持管道时 **close 永久挂死**工具线程 | 仅 reader 已结束时才关管道 |
| 116 | shell_run 无条件对 pgid 发 SIGKILL | pgid 被 OS 回收复用时**误杀无关进程组** | 组主已退出时先信号 0 探测组是否仍存在 |
| 117 | code_run 取消后线程跑满 timeout | 占 CPU 且 sys.stdout 被持续替换 | 轮询取消立即收尾 + PermissionError/SystemExit 统一带 traceback |
| 118 | `ToolRegistry.shutdown()` 裸清单例 | 与懒初始化竞态 → 每次应用配置泄漏 ~137MB | 单例清空持 `_singleton_lock` |
| 119 | skill embedding 失败回退 `None` | SkillVectorStore 悄悄自建第二个默认模型单例 | 显式回退默认模型单例 |
| 120 | FastEmbed 懒加载无锁 | 首次并发各加载一份 ~137MB ONNX | 进程级加载锁 + 双重检查 |
| 121 | FTS 建索引跨实例并发 | 写入侧与检索侧对同一张表并发 replace | 模块级建索引锁 + 「已存在」容错 |
| 122 | reranker 后台加载/close 裸写共享状态 | 已 close 的实例「复活」持有 ~500MB 权重 | 状态写回/close 全部持 `_reranker_lock` 并复查 `_closed` |
| 123 | documents 表 doc_id 用 basename | 不同子目录同名文件互相覆盖/误删 | doc_id 改用 `md_path`（唯一），兼容清理旧影子行 |
| 124 | 语义切片负缓存 TTL 从未生效 | 离线时每个超长 section 各重试一次模型加载 | embed 真实失败记时间戳，同一 TTL 节流 |
| 125 | MOBI latin-1 回退被 `del data` 废掉 | 非 utf-8 MOBI 静默返回空 | 不再提前释放字节，回退可用 |
| 126 | 取消后仍执行 Step7 全量 digest 重刷 | 同步已取消还多跑几秒 | cancelled 时跳过 |
| 127 | `delete_by_source` 恒返回 1 | 与 docstring 语义不符 | 前后 count 差值尽力统计 |
| 128 | `rag_ingest` force 字符串 `"false"` 为真值 | 意外触发全量重建 | 严格布尔解析 |
| 129 | file_write 现场翻转进程 umask | 非线程安全，窗口内他线程建文件权限 0666 | 导入期缓存 umask |
| 130 | file_read 按字节数判截断 | 多字节 UTF-8（中文）被误判截断 | 按字符增量读取判定 |
| 131 | 图谱回填/去重全表物化（含向量列） | 记忆量大时内存与耗时线性恶化 | 列投影下推到存储层 |
| 132 | 图谱待重试队列无锁无上限 | 并发丢条目；嵌入持续失败时内存只增 | 专用锁 + 500 条上限 |
| 133 | skill 目录 mtime 探测每次检索两次全树 walk | 千级技能树每次查询 O(目录数) syscall | 60s TTL 缓存（`mark_stale` 显式标脏兜底） |
| 134 | config 字符串字段 `str()` 收敛垃圾值 | model 写成列表 → 存成 `"['x']"` | 非 str 回退默认 |
| 135 | config.save 固定 tmp 名 | 并发保存互踩 | mkstemp 唯一名 + 失败清理 |
| 136 | 强杀后流式状态残留 | **下一回合整条回复被吞**（用户看到空回复） | 回合开始统一复位 stripper/流式状态；强杀先 flush |
| 137 | 强杀丢弃运行中 QThread 引用 | wait 超时时 sip 析构 → **qFatal 进程 abort** | 超时挂孤儿列表，closeEvent 再统一 wait |
| 138 | `_worker_terminated` 永不复位 | 之后每次退出都跳过资源释放（泄漏 MCP/连接池） | `_rebuild_agent` 成功后复位 |
| 139 | 连点「终止」重置看门狗倒计时 | 30s 强杀可被无限推迟 | 仅在未激活时 start |
| 140 | 推理流逐 chunk 全视口重绘 | 推理模型 UI 明显卡顿 | 复用 40ms 合并刷新（与正文同节奏） |
| 141 | 工具入参不截断 | 一次大 file_write 把 MB 级文本常驻 QTextDocument | 与 tool-end 一致截断 400 字符 |
| 142 | 日志/流式输出无条件吸底 | 用户选中文本复制被反复打断、无法上翻 | 仅吸底时跟随；插入用临时光标不碰用户光标 |
| 143 | `_on_finished` 忽略 wait 结果 | 与 137 同构的析构 abort 风险 | 超时挂孤儿列表 |
| 144 | 计划面板跨线程读 Session | 渲染出半新半旧的中间态 | `render_plan_dict` 纯函数 + 信号快照渲染 |
| 145 | busy 时按 Enter 静默丢弃 | 像输入框坏了 | 状态栏提示「正在运行中」 |
| 146 | 清空对话无 busy 守卫、不清计数器 | 运行中清空破坏会话；token 计数残留 | 加守卫 + 复位计数 |
| 147 | Ctrl+Enter 也发送、按键事件未 accept | 误发送；事件向父控件意外传播 | 仅裸 Enter 发送 + `e.accept()` |

## 部署

详细部署文档见 [docs/deploy.md](docs/deploy.md)。

## License

MIT

# Base Agent 生产环境部署指南

## 1. 环境要求

| 组件 | 最低版本 | 说明 |
|------|----------|------|
| Python | 3.9+ | 推荐 3.10+ |
| 操作系统 | macOS / Linux | Windows 需额外适配 Qt 依赖 |
| 内存 | 4 GB+ | 加载嵌入模型和重排序模型需要额外内存 |
| 磁盘 | 2 GB+ | 模型缓存 + 向量数据库持久化存储 |

## 2. 安装

### 2.1 安装核心依赖

```bash
# 克隆项目
git clone <repo-url> base-agent
cd base-agent

# 安装核心包
pip install -e .
```

### 2.2 安装 RAG 依赖（按需）

```bash
# ===== RAG 核心依赖（必装）=====
pip install chromadb sentence-transformers

# ===== 文档解析依赖（按需安装）=====
# Word 文档
pip install python-docx

# Excel 表格
pip install openpyxl

# PowerPoint 演示文稿
pip install python-pptx

# PDF 文档（含 OCR 支持）
pip install PyMuPDF rapidocr-onnxruntime

# EPUB 电子书
pip install ebooklib

# ==== 重排序模型（可选，提升检索精度）=====
pip install FlagEmbedding

# ==== MOBI/AZW3 格式（可选，依赖 calibre 命令行工具）=====
# macOS: brew install calibre
# Linux: sudo apt install calibre
```

### 2.3 一键安装全部依赖

```bash
pip install -e ".[test]"
pip install chromadb sentence-transformers \
    python-docx openpyxl python-pptx \
    PyMuPDF rapidocr-onnxruntime \
    ebooklib FlagEmbedding
```

## 3. 模型下载（离线环境）

首次运行时，`sentence-transformers` 会自动从 HuggingFace 下载所需的嵌入模型。如果生产环境无法访问 HuggingFace，需要预先下载模型文件。

### 3.1 所需模型

| 模型 | 用途 | 大小 |
|------|------|------|
| `all-MiniLM-L6-v2` | 文本嵌入向量化 | ~90 MB |
| `BAAI/bge-reranker-base` | 检索结果重排序 | ~1.1 GB |

### 3.2 离线部署方案

```bash
# 在有网络的机器上预下载模型
pip install huggingface_hub
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir ./models/all-MiniLM-L6-v2
huggingface-cli download BAAI/bge-reranker-base --local-dir ./models/bge-reranker-base

# 将模型目录复制到生产服务器
scp -r ./models/ user@server:/path/to/models/

# 生产环境设置环境变量指向本地模型
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

模型缓存默认路径：`~/.cache/huggingface/hub/`，将模型文件放到该路径即可。

## 4. 配置

配置文件位于 `~/.base-agent/config.json`，首次启动应用后自动生成。也可手动创建：

```json
{
  "base_url": "https://api.openai.com/v1",
  "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "model": "gpt-4o-mini",
  "temperature": 0.7,
  "max_tokens": 32768,
  "top_p": 0.95,
  "min_p": 0.05,
  "top_k": 20,
  "repetition_penalty": 1.0,
  "workspace": "~/base-agent-workspace",
  "knowledge_base": "/data/knowledge-base",
  "request_timeout": 120.0,
  "max_iterations": 15,
  "rag_chunk_size": 500,
  "rag_chunk_overlap": 50,
  "rag_embedding_model": "all-MiniLM-L6-v2",
  "rag_rerank_model": "BAAI/bge-reranker-base",
  "rag_rerank_enabled": true,
  "rag_auto_ingest": true,
  "browser_endpoint": "",
  "mcp_servers": []
}
```

### 关键配置项说明

| 配置项 | 说明 | 推荐值 |
|--------|------|--------|
| `knowledge_base` | 知识库文档目录路径 | 指向存放原始文档的目录 |
| `rag_chunk_size` | 每个切片的 token 数 | 500 |
| `rag_chunk_overlap` | 切片间重叠 token 数（10%） | 50 |
| `rag_embedding_model` | 嵌入向量模型 | `all-MiniLM-L6-v2`（多语言轻量） |
| `rag_rerank_model` | 重排序模型 | `BAAI/bge-reranker-base` |
| `rag_rerank_enabled` | 是否启用重排序 | `true` |
| `rag_auto_ingest` | 是否启动时自动同步知识库 | `true`（生产环境建议 true） |

## 5. 知识库目录结构

将需要索引的文档放入 `knowledge_base` 指定的目录，支持子目录递归扫描：

```
/data/knowledge-base/
├── 技术文档/
│   ├── 架构设计.docx
│   ├── API文档.md
│   └── 部署手册.pdf
├── 产品资料/
│   ├── 需求文档.xlsx
│   ├── 用户手册.epub
│   └── 竞品分析.html
├── 培训材料/
│   └── 新人指南.pptx
└── 参考资料/
    ├── paper.pdf
    └── book.mobi
```

### 支持的文档格式

| 格式 | 扩展名 | 依赖 |
|------|--------|------|
| Word | `.docx`, `.doc` | `python-docx` |
| Excel | `.xlsx`, `.xls` | `openpyxl` |
| PowerPoint | `.pptx`, `.ppt` | `python-pptx` |
| PDF | `.pdf` | `PyMuPDF`（图片型 PDF 需 `rapidocr-onnxruntime`） |
| EPUB | `.epub` | `ebooklib` + `beautifulsoup4` |
| MOBI/AZW | `.mobi`, `.azw3`, `.azw` | calibre `ebook-convert`（可选） |
| 纯文本 | `.txt`, `.md`, `.rst`, `.csv`, `.tsv` | 无额外依赖 |
| HTML | `.html`, `.htm` | `beautifulsoup4` |

## 6. 启动服务

### 6.1 交互式 GUI 模式

```bash
cd base-agent
python -m src.agent.ui.main_window
```

启动后：
1. 在界面顶部的「知识库」输入框中填写知识库路径，或点击「…」按钮选择目录
2. 点击「同步知识」按钮，等待后台处理完成（弹出提示框）
3. 「同步知识」处理流程：文档扫描 → 文本提取 → 清洗 → Markdown 生成 → 切片 → 向量化 → 存入 ChromaDB
4. 同步完成后，在对话框中输入问题时，Agent 会自动调用 `rag_search` 工具检索知识库

### 6.2 无头模式（编程调用）

```python
from src.agent.rag.engine import RAGEngine

# 初始化引擎
engine = RAGEngine(
    workspace="/data/agent-workspace",
    knowledge_base="/data/knowledge-base",
)

# 同步知识库（首次运行或文档更新后）
stats = engine.ingest(force=True)
print(f"索引完成: {stats['files_found']} 文件 → {stats['chunks']} 切片")

# 检索知识库
results = engine.search("如何配置 RAG 服务", top_k=3)
for r in results:
    print(f"[{r['score']:.4f}] {r['source']}")
    print(r["text"][:200])
    print("---")

# 查看状态
status = engine.status()
print(f"已索引: {status['chunks_stored']} 切片, {len(status['sources'])} 文件")
```

### 6.3 定时同步脚本（cron）

```bash
# 创建同步脚本 /usr/local/bin/sync-knowledge.sh
cat > /usr/local/bin/sync-knowledge.sh << 'EOF'
#!/bin/bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /opt/base-agent
python3 -c "
from src.agent.rag.engine import RAGEngine
import json

engine = RAGEngine(
    workspace='/data/agent-workspace',
    knowledge_base='/data/knowledge-base',
)
stats = engine.ingest(force=True)
print(json.dumps(stats, ensure_ascii=False, indent=2))
"
EOF

chmod +x /usr/local/bin/sync-knowledge.sh

# 添加定时任务：每天凌晨 2 点同步
crontab -e << 'EOF'
0 2 * * * /usr/local/bin/sync-knowledge.sh >> /var/log/rag-sync.log 2>&1
EOF
```

## 7. 数据目录说明

启动后自动创建的目录结构：

```
~/base-agent-workspace/
└── rag/
    ├── documents/          # 清洗后的 Markdown 文件（缓存）
    │   ├── rag_intro.md
    │   ├── vector_db.txt.md
    │   └── ...
    └── vectors/            # ChromaDB 向量数据库（持久化）
        └── chroma.sqlite3
```

- `rag/documents/` — 清洗后的文本缓存，二次同步时跳过已处理的文件
- `rag/vectors/` — ChromaDB 持久化存储，包含所有向量索引

## 8. 生产环境 Checklist

### 部署前

- [ ] Python 3.9+ 已安装
- [ ] 所有 RAG 依赖已安装（见第 2 节）
- [ ] 嵌入模型已下载到本地缓存（见第 3 节）
- [ ] 重排序模型已下载（如需启用重排序）
- [ ] 配置文件 `~/.base-agent/config.json` 已就绪
- [ ] `knowledge_base` 目录存在且包含文档
- [ ] `workspace` 目录有写入权限

### 首次启动

- [ ] 点击「同步知识」或执行 `ingest(force=True)`
- [ ] 确认同步完成，查看到切片数和文件数统计
- [ ] 发送一条测试查询，确认 `rag_search` 工具能返回相关知识库内容

### 运维

- [ ] 文档更新后，重新执行「同步知识」
- [ ] 设置 cron 定时同步（见 6.3 节）
- [ ] 监控 `rag/vectors/` 目录磁盘使用
- [ ] 定期清理过期的 Markdown 缓存（`rag/documents/`）

## 9. 故障排查

### 问题：同步知识卡住不动

**原因**：HuggingFace 模型下载超时。

**解决**：设置离线模式环境变量，确保模型已预下载到本地。

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### 问题：搜索返回空结果

**检查清单**：
1. `knowledge_base` 路径是否正确，目录下是否有受支持的文档
2. 是否已执行「同步知识」（查看 `rag/vectors/` 目录下是否有 `chroma.sqlite3`）
3. 嵌入模型是否已下载（`~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/`）

### 问题：重排序不生效

**原因**：`FlagEmbedding` 未安装或 `BAAI/bge-reranker-base` 模型未下载。

**解决**：
```bash
pip install FlagEmbedding
# 确保模型已缓存到 ~/.cache/huggingface/hub/
```

如果不需要重排序，可在配置中设置 `"rag_rerank_enabled": false`，系统会自动回退到直接返回向量检索结果。

### 问题：PDF 文档提取为空

**原因**：PDF 为纯图片格式，且未安装 OCR 依赖。

**解决**：
```bash
pip install rapidocr-onnxruntime
```

### 问题：ChromaDB SQLite 锁冲突

**原因**：多个进程同时访问同一个向量数据库。

**解决**：确保同一时间只有一个进程在写入。ChromaDB 使用 SQLite 作为后端，不支持并发写入。
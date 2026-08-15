"""RAG tool — allows the agent to search the local knowledge base."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, TYPE_CHECKING

from .base import Tool, ToolResult, ToolRegistry

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..rag.engine import RAGEngine


class RagSearchTool(Tool):
    """Search the local knowledge base for relevant information."""

    name = "rag_search"
    description = (
        "搜索本地知识库获取相关信息。当需要查找文档、参考资料或专业知识时使用此工具。"
        "适合查询知识库中的文档内容，如技术文档、书籍、论文等。"
        "当需要限定在某本书/文档内检索时，用 source 参数传入书名或文档名关键词。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "搜索查询词。必须使用用户原始语言的关键词/短语，"
                    "例如用户问中文就用中文关键词（如「回归神 主旨 思想」），"
                    "用户问英文就用英文。严禁把中文音译成拼音或罗马化，"
                    "也不要翻译成其他语言。直接提取用户问题中的核心名词/概念即可。"
                ),
            },
            "top_k": {
                "type": "integer",
                "description": "返回结果数量，默认 3，范围 1-10（经过重排序后的最优结果）",
            },
            "source": {
                "type": "string",
                "description": (
                    "可选。限定检索的文档名/书名关键词（如「亲密关系」「人生的活法」）。"
                    "当 rag_outline 已返回某本书的结构、需要定向检索该书内某章节细节时，"
                    "传入该书名，只在该书内检索，避免混入其它文档。"
                ),
            },
        },
        "required": ["query"],
    }
    destructive = False

    def __init__(self, engine: "RAGEngine"):
        self._engine = engine

    def bind(self, config: Any, registry: ToolRegistry) -> None:
        pass

    def run(self, query: str = "", top_k: int = 3, source: str = "") -> ToolResult:
        if not query.strip():
            return ToolResult(success=False, error="请提供搜索查询词")

        # 限制 top_k 上限，防止模型传入过大值导致返回海量内容撑爆上下文窗口。
        top_k = max(1, min(int(top_k), 10))

        source = (source or "").strip()
        logger.info(
            "🔍 RAG 工具调用: rag_search(query=%r, top_k=%d, source=%r)",
            query[:80], top_k, source[:80],
        )
        try:
            results = self._engine.search_formatted(
                query, top_k=top_k, source=source or None,
            )
            return ToolResult(success=True, output=results)
        except Exception as e:
            logger.error("❌ RAG 搜索失败: %s", e, exc_info=True)
            return ToolResult(success=False, error=f"知识库搜索失败: {e}")


class RagOutlineTool(Tool):
    """Return a document's condensed outline (目录结构) for Meta-context RAG.

    当用户提到具体书名/文档名时，先调用本工具获取该文档的完整目录结构
    （标题层级），让模型先掌握全文结构，再规划需要深入检索哪些章节的细节。
    """

    name = "rag_outline"
    description = (
        "获取知识库中某本文档/书籍的缩略版本（完整目录结构与章节标题）。"
        "当用户明确提到具体书名、文档名，或需要从某本具体的书/文档中系统性地"
        "回答问题时，先调用本工具了解全文档结构，再决定用 rag_search 检索哪些"
        "章节的细节。参数 book_name 为书名或文档名的关键词。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "book_name": {
                "type": "string",
                "description": "书名或文档名的关键词（支持部分标题匹配）",
            },
        },
        "required": ["book_name"],
    }
    destructive = False

    def __init__(self, engine: "RAGEngine"):
        self._engine = engine

    def bind(self, config: Any, registry: ToolRegistry) -> None:
        pass

    def run(self, book_name: str = "", **kwargs) -> ToolResult:
        if not book_name.strip():
            # 未指定书名时，列出所有已建立缩略版本的文档。
            docs = self._engine.list_documents()
            if not docs:
                return ToolResult(success=False, error="知识库中暂无已建立缩略版本的文档")
            return ToolResult(
                success=True,
                output="知识库中已建立缩略版本的文档：\n" + "\n".join(f"  - {d}" for d in docs),
            )

        logger.info("📖 RAG 工具调用: rag_outline(book_name=%r)", book_name[:80])
        try:
            doc = self._engine.get_document_outline(book_name)
            if doc is None:
                return ToolResult(
                    success=False,
                    error=f"未找到文档「{book_name}」的缩略版本。可先调用 rag_outline（不带参数）查看可用文档列表。",
                )
            digest = doc.get("digest", "")
            if not digest:
                return ToolResult(success=False, error=f"文档「{doc.get('doc_name', book_name)}」暂无缩略版本")
            return ToolResult(success=True, output=digest)
        except Exception as e:
            logger.error("❌ RAG 缩略版本获取失败: %s", e, exc_info=True)
            return ToolResult(success=False, error=f"获取文档缩略版本失败: {e}")


class RagStatusTool(Tool):
    """Check the status of the local knowledge base."""

    name = "rag_status"
    description = (
        "查看本地知识库的状态，包括已索引的文档数量、来源文件列表等。"
    )
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    destructive = False

    def __init__(self, engine: "RAGEngine"):
        self._engine = engine

    def bind(self, config: Any, registry: ToolRegistry) -> None:
        pass

    def run(self, **kwargs) -> ToolResult:
        try:
            status = self._engine.status()
            lines = [
                f"知识库状态:",
                f"  知识库目录: {status['knowledge_base'] or '（未设置）'}",
                f"  RAG 目录: {status['rag_dir']}",
                f"  已存储切片数: {status['chunks_stored']}",
                f"  已索引文件数: {len(status['sources'])}",
            ]
            if status["sources"]:
                lines.append("  已索引文件:")
                for s in status["sources"]:
                    lines.append(f"    - {os.path.basename(s)}")
            return ToolResult(success=True, output="\n".join(lines))
        except Exception as e:
            return ToolResult(success=False, error=f"获取知识库状态失败: {e}")


class RagIngestTool(Tool):
    """Re-ingest the knowledge base directory."""

    name = "rag_ingest"
    description = (
        "重新索引知识库目录中的所有文档。当知识库中添加了新文件后使用此工具。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "force": {
                "type": "boolean",
                "description": "是否强制重新处理所有文件（默认 false，跳过已处理的文件）",
            },
        },
        "required": [],
    }
    destructive = False

    def __init__(self, engine: "RAGEngine"):
        self._engine = engine

    def bind(self, config: Any, registry: ToolRegistry) -> None:
        pass

    def run(self, force: bool = False, **kwargs) -> ToolResult:
        logger.info("📥 RAG 工具调用: rag_ingest(force=%s)", force)
        try:
            stats = self._engine.ingest(force=force)
            if "error" in stats:
                logger.warning("rag_ingest returned error: %s", stats["error"])
                return ToolResult(success=False, error=stats["error"])
            lines = [
                "知识库索引完成:",
                f"  扫描文件数: {stats.get('files_found', 0)}",
                f"  新提取文件: {stats.get('files_extracted', 0)}",
                f"  跳过(已处理): {stats.get('files_skipped', 0)}",
                f"  总字符数: {stats.get('total_chars', 0)}",
                f"  生成切片数: {stats.get('chunks', 0)}",
            ]
            errors = stats.get("errors", [])
            if errors:
                lines.append(f"  错误: {len(errors)} 个")
            return ToolResult(success=True, output="\n".join(lines))
        except Exception as e:
            logger.error("rag_ingest failed: %s", e, exc_info=True)
            return ToolResult(success=False, error=f"知识库索引失败: {e}")
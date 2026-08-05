"""RAG tool — allows the agent to search the local knowledge base."""
from __future__ import annotations

import os
from typing import Any, Dict, TYPE_CHECKING

from .base import Tool, ToolResult, ToolRegistry

if TYPE_CHECKING:
    from ..rag.engine import RAGEngine


class RagSearchTool(Tool):
    """Search the local knowledge base for relevant information."""

    name = "rag_search"
    description = (
        "搜索本地知识库获取相关信息。当需要查找文档、参考资料或专业知识时使用此工具。"
        "适合查询知识库中的文档内容，如技术文档、书籍、论文等。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索查询词，描述你需要查找的信息",
            },
            "top_k": {
                "type": "integer",
                "description": "返回结果数量，默认 3（经过重排序后的最优结果）",
            },
        },
        "required": ["query"],
    }
    destructive = False

    def __init__(self, engine: "RAGEngine"):
        self._engine = engine

    def bind(self, config: Any, registry: ToolRegistry) -> None:
        pass

    def run(self, query: str = "", top_k: int = 3) -> ToolResult:
        if not query.strip():
            return ToolResult(success=False, error="请提供搜索查询词")

        try:
            results = self._engine.search_formatted(query, top_k=top_k)
            return ToolResult(success=True, output=results)
        except Exception as e:
            return ToolResult(success=False, error=f"知识库搜索失败: {e}")


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
        try:
            stats = self._engine.ingest(force=force)
            if "error" in stats:
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
            return ToolResult(success=False, error=f"知识库索引失败: {e}")
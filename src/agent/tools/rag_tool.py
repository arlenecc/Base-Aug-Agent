"""RAG tool — allows the agent to search the local knowledge base."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import Tool, ToolResult, ToolRegistry

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..rag.engine import RAGEngine

# 归纳阈值已做成可配置项（AgentConfig：rag_condense_min_chars /
# rag_condense_max_chars / rag_condense_output_max_chars，见 config.py）。
# 以下常量仅作为配置缺失时的 getattr 兜底默认值。
# 给辅助模型的输入上限（字符）：辅助模型上下文有限，原始分片过长先截断。
_AUX_INPUT_MAX_CHARS = 12000
# 归纳后交给主模型的浓缩文本上限（字符）。
_CONDENSED_MAX_CHARS = 2000
# 触发归纳的最低结果总量（字符）：低于此值原样返回，不做归纳。
_CONDENSE_MIN_CHARS = 4000


class RagSearchTool(Tool):
    """Search the local knowledge base for relevant information."""

    name = "rag_search"
    description = "搜索本地知识库。查询文档内容、参考资料时使用；可用 source 限定在某本书/文档内检索。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索关键词/短语。用用户原始语言，提取问题核心概念，勿翻译或音译。",
            },
            "top_k": {
                "type": "integer",
                "description": "返回结果数，默认 3，上限 5。",
            },
            "source": {
                "type": "string",
                "description": "可选。限定检索的书名/文档名关键词。",
            },
        },
        "required": ["query"],
    }
    destructive = False

    def __init__(self, engine: "RAGEngine", aux_llm_getter: Optional[Any] = None):
        self._engine = engine
        # aux_llm_getter(): 读取「当前」辅助模型后端（可能为 None，例如辅助模型
        # 未加载、或主模型用 OpenAI 接口模式）。用 getter 而非直接存引用，才能
        # 在辅助模型运行期加载/切换后拿到最新实例。
        self._aux_llm_getter = aux_llm_getter

    def bind(self, config: Any, registry: ToolRegistry) -> None:
        # 归纳阈值等从 AgentConfig 读取（注册时 bind 必被调用）；未 bind 时
        # _condense_with_aux 里 getattr(self, "_config", None) 落到常量兜底。
        self._config = config

    def run(self, query: str = "", top_k: int = 3, source: str = "") -> ToolResult:
        # 模型可能传 null（JSON null → None），None.strip() 会抛异常并被
        # registry 兜底误报成 "Invalid arguments"，模型会原样重试。这里统一
        # 归一化成空串再判断。
        query = (query or "").strip()
        if not query:
            return ToolResult(success=False, error="请提供搜索查询词")

        # 限制 top_k 上限：候选数 = top_k × 4 × 2 路，全量送 reranker 精排。
        # 默认 jina(ONNX/CPU) 实测 ~49 对/s：top_k=5 → 40 对 ≈ 0.8s，可接受；
        # 旧默认逐对串行的 bge INT8 ~0.55s/对（#132）时 40 对要 23s，当时因此
        # 封顶 3——换 jina 后放宽到 5。
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = 3
        top_k = max(1, min(top_k, 5))

        source = (source or "").strip()
        logger.info(
            "🔍 RAG 工具调用: rag_search(query=%r, top_k=%d, source=%r)",
            query[:80], top_k, source[:80],
        )
        try:
            results = self._engine.search(query, top_k=top_k, source=source or None)
        except Exception as e:
            logger.error("❌ RAG 搜索失败: %s", e, exc_info=True)
            return ToolResult(success=False, error=f"知识库搜索失败: {e}")

        if not results:
            return ToolResult(success=True, output="（知识库中未找到相关内容）")

        # 结果总量 ≥ rag_condense_min_chars（默认 4000 字）才送辅助模型归纳
        # 浓缩（输出 ≤ rag_condense_output_max_chars）；低于阈值直接原样返回，
        # 避免丢细节、白等一次归纳。辅助模型不可用时同样退化为原始格式。
        condensed = self._condense_with_aux(query, results)
        if condensed is not None:
            return ToolResult(success=True, output=condensed)
        return ToolResult(success=True, output=self._engine.format_results(results))

    def _condense_with_aux(self, query: str, results: List[Dict[str, Any]]) -> Optional[str]:
        """把检索到的所有文本分片送辅助模型归纳，浓缩到 _CONDENSED_MAX_CHARS 以内。

        返回浓缩文本；辅助模型不可用 / 调用失败 / 返回空时返回 None，由调用方
        退化为原始格式化输出。
        """
        if self._aux_llm_getter is None:
            return None
        aux = self._aux_llm_getter()
        if aux is None:
            return None

        # 阈值从 AgentConfig 读取（bind 注入）；未注入时落到模块常量兜底。
        cfg = getattr(self, "_config", None)
        min_chars = int(getattr(cfg, "rag_condense_min_chars", _CONDENSE_MIN_CHARS) or 0)
        max_chars = int(getattr(cfg, "rag_condense_max_chars", _AUX_INPUT_MAX_CHARS)
                        or _AUX_INPUT_MAX_CHARS)
        out_cap = int(getattr(cfg, "rag_condense_output_max_chars", _CONDENSED_MAX_CHARS)
                      or _CONDENSED_MAX_CHARS)

        chunks = []
        for r in results:
            src = os.path.basename(r.get("source", "") or "")
            text = (r.get("text", "") or "").strip()
            if text:
                chunks.append(f"[来源: {src}]\n{text}")
        raw = "\n\n".join(chunks)
        if not raw.strip():
            return None
        if len(raw) < min_chars:
            # 低于最低阈值：内容本就不多，归纳反而丢细节、白等一次辅助模型
            # 生成——直接原样返回（阈值可配 rag_condense_min_chars，0 = 总是归纳）。
            logger.info(
                "  ├─ 检索结果 %d 字 < 阈值 %d 字，跳过辅助模型归纳（原样返回）",
                len(raw), min_chars,
            )
            return None
        if len(raw) > max_chars:
            raw = raw[:max_chars] + "\n…（原文过长，已截断）"

        prompt = (
            "以下是针对用户问题从知识库检索到的若干文本分片（已按相关度重排序）。"
            "请对全部内容进行归纳整理，保留核心观点、关键事实与重要信息，剔除冗余"
            f"和无关细节，并将全部内容浓缩精简到 {out_cap} 字以内。请保留每条要点的来源"
            "标注。直接输出整理后的内容，不要加前言、不要解释过程。\n\n"
            f"用户问题：{query}\n\n检索到的文本分片：\n{raw}"
        )
        messages = [
            {"role": "system", "content": "你是严谨的信息提炼助手，擅长把多段长文本浓缩为精炼、保真的要点。"},
            {"role": "user", "content": prompt},
        ]
        out: List[str] = []
        try:
            for ev in aux.chat_stream(messages, tools=None, temperature=0.2):
                if ev.type == "content":
                    out.append(ev.content)
                elif ev.type == "done":
                    break
        except Exception as e:
            logger.error("❌ RAG 辅助模型归纳整理失败，回退原始分片: %s", e)
            return None
        text = "".join(out).strip()
        if not text:
            return None
        # 保险：辅助模型未遵守字数限制时强制截断（按字符）。
        if len(text) > out_cap:
            text = text[:out_cap]
        return text


class RagOutlineTool(Tool):
    """Return a document's condensed outline (目录结构) for Meta-context RAG.

    当用户提到具体书名/文档名时，先调用本工具获取该文档的完整目录结构
    （标题层级），让模型先掌握全文结构，再规划需要深入检索哪些章节的细节。
    """

    name = "rag_outline"
    description = (
        "获取某文档/书籍的目录结构（章节标题）。用户提到具体书名时先调用，"
        "了解全文结构后再用 rag_search 检索章节细节。不带 book_name 时列出所有文档。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "book_name": {
                "type": "string",
                "description": "书名/文档名关键词（支持部分匹配）",
            },
        },
        "required": [],
    }
    destructive = False

    def __init__(self, engine: "RAGEngine"):
        self._engine = engine

    def bind(self, config: Any, registry: ToolRegistry) -> None:
        pass

    def run(self, book_name: str = "", **kwargs) -> ToolResult:
        book_name = (book_name or "").strip()
        if not book_name:
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
                sources = status["sources"]
                # 大知识库可能索引了成百上千个文件，全量列出既挤占上下文也
                # 没有决策价值——截断并提示总数。
                _MAX_SOURCES = 50
                lines.append(f"  已索引文件: 共 {len(sources)} 个")
                for s in sources[:_MAX_SOURCES]:
                    lines.append(f"    - {os.path.basename(s)}")
                if len(sources) > _MAX_SOURCES:
                    lines.append(f"    … （其余 {len(sources) - _MAX_SOURCES} 个省略）")
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
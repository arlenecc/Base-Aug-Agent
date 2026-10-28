"""Embedding 任务前缀策略 —— 按模型族统一判定（唯一事实源）。

背景：各嵌入模型族对「文档 / 查询」使用不同的任务前缀，缺前缀或错用前缀
会显著拉低检索质量。此前 nomic 的前缀在 MLX（model_backends）与 FastEmbed
（vector_store）两处各写一份，bge 的中文查询指令又硬编码在 MLX 的加载分支
里，GGUF 后端则完全没加前缀——字符串分散、改一处漏两处。本模块把
「模型名 → 任务前缀」收敛成一张表，任何后端（MLX / fastembed / ONNX /
llama_cpp）统一经此解析，按模型名自动识别模型族。

前缀约定（官方推荐格式，直接拼接在文本前；英文前缀保留尾部空格分隔词元）：
    nomic : document="search_document: "   query="search_query: "
    bge   : document=""（不加）             query="为这个句子生成表示以检索相关文档"
    jina  : document="passage: "           query="query: "
            另有任务前缀：clustering="clustering: "  classification="classification: "
    e5    : document="passage: "           query="query: "
    其它  : 所有前缀均为空串（不加，与未收敛前的未知模型行为一致）
"""

from __future__ import annotations

from typing import Dict

# 模型族 → 各任务前缀。document/query 用于检索；clustering/classification
# 目前仅 jina 族定义（nomic/bge/e5 官方未给出对应任务前缀，取空串）。
_FAMILY_PREFIXES: Dict[str, Dict[str, str]] = {
    "nomic": {
        "document": "search_document: ",
        "query": "search_query: ",
        "clustering": "",
        "classification": "",
    },
    "bge": {
        "document": "",
        "query": "为这个句子生成表示以检索相关文档",
        "clustering": "",
        "classification": "",
    },
    "jina": {
        "document": "passage: ",
        "query": "query: ",
        "clustering": "clustering: ",
        "classification": "classification: ",
    },
    "e5": {
        "document": "passage: ",
        "query": "query: ",
        "clustering": "",
        "classification": "",
    },
}

# 模型名/本地路径 → 模型族的关键字（按序匹配，命中即返回）。
# 匹配对象统一小写后做子串查找，兼容 HF 仓库名（"nomic-ai/nomic-embed-text-v1.5-Q"）
# 与本地模型目录（"models/bge-small-zh-v1.5-mlx"）两种形态。
_FAMILY_KEYWORDS = (
    ("nomic", "nomic"),
    ("bge", "bge"),
    ("jina", "jina"),
    ("e5", "e5"),
)

SUPPORTED_TASKS = ("document", "query", "clustering", "classification")


def detect_family(model: str) -> str:
    """按模型名/本地路径识别模型族；未识别返回空串（表示不加任何前缀）。"""
    text = (model or "").lower()
    for keyword, family in _FAMILY_KEYWORDS:
        if keyword in text:
            return family
    return ""


def task_prefix(model: str, task: str) -> str:
    """取 ``model`` 在 ``task`` 任务下的前缀；未知模型族/任务返回空串。

    task 取值：``"document"`` | ``"query"`` | ``"clustering"`` |
    ``"classification"``。
    """
    family = detect_family(model)
    return _FAMILY_PREFIXES.get(family, {}).get(task, "")


def document_prefix(model: str) -> str:
    """检索文档（入库侧）前缀。"""
    return task_prefix(model, "document")


def query_prefix(model: str) -> str:
    """检索查询（查询侧）前缀。"""
    return task_prefix(model, "query")


def prefixes(model: str) -> "tuple[str, str]":
    """(document_prefix, query_prefix) 一步取齐，供各后端构造时解包。"""
    return document_prefix(model), query_prefix(model)


def add_prefix(model: str, task: str, text: str) -> str:
    """给单条文本加 ``task`` 前缀（幂等：已带该前缀则原样返回）。

    各后端的 embed_documents/embed_query 统一走这里，保证「startswith
    防重复」的判断也只有一份实现。
    """
    prefix = task_prefix(model, task)
    if not prefix or text.startswith(prefix):
        return text
    return prefix + text


def apply_prefix(prefix: str, text: str) -> str:
    """按已解析好的 ``prefix`` 给单条文本加前缀（幂等，空前缀原样返回）。

    与 :func:`add_prefix` 的区别：前缀已由后端构造期解析进实例属性
    （且可能被加载期按实际模型族覆盖），运行时直接用属性值即可，
    不必再按模型名解析一次。
    """
    if not prefix or text.startswith(prefix):
        return text
    return prefix + text

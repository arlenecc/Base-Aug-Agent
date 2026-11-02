"""默认模型注册表：把「本地模型目录名」映射到 HuggingFace 仓库 id。

首次启动、默认配置所需的模型若本地缺失，下载器会按这里给出的 repo id 从
HuggingFace 镜像（默认 https://hf-mirror.com）拉取。

规则（按用户要求）：仓库名统一使用 ``arlenecc/<模型目录名>``。若某个模型实际
不在 ``arlenecc`` 组织下，请在 ``MODEL_REPO_OVERRIDES`` 里显式覆盖。
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import List, Tuple

# 默认镜像站（国内可直连）；可通过环境变量 BASEAGENT_HF_MIRROR 覆盖。
HF_MIRROR = os.environ.get("BASEAGENT_HF_MIRROR", "https://hf-mirror.com").rstrip("/")

# 例外覆盖表：键为「本地模型目录名」，值为完整 HF repo id。
# 默认所有模型都落在 arlenecc 组织下，故此处通常为空。
MODEL_REPO_OVERRIDES: dict[str, str] = {}

# 角色中文名（用于 UI 展示）。
ROLE_LABELS: dict[str, str] = {
    "llm": "主模型 (LLM)",
    "aux_llm": "辅助模型",
    "embedding": "嵌入模型",
    "reranker": "重排模型",
    "ocr": "OCR 模型",
}


def repo_id_for(local_name: str) -> str:
    """返回某本地模型目录名对应的 HF repo id。"""
    return MODEL_REPO_OVERRIDES.get(local_name, f"arlenecc/{local_name}")


def _local_name_of(path_field_value: str) -> str:
    return Path(path_field_value or "").name


def _default_dir_name(config, path_field: str) -> str:
    """读取某 path 字段的【类默认值】目录名（即该角色的默认模型名）。

    自动下载始终以「默认模型」为准，不随用户自定义的目录名变化
    （用户自定义路径只决定下载落点，不改变要下载哪个模型）。
    """
    f = next((x for x in dataclasses.fields(config) if x.name == path_field), None)
    if f is None or f.default is dataclasses.MISSING:
        return ""
    return _local_name_of(f.default or "")


def collect_needed_models(config) -> List[Tuple[str, str, str]]:
    """根据当前配置，收集「默认需要、且为本地后端」的模型清单。

    返回 ``[(role, local_name, repo_id), ...]``。仅包含本地后端（排除 OpenAI /
    API 类远端后端），且遵循各功能的启用开关。``local_name`` / ``repo_id`` 取该角色
    的**默认模型名**（不受用户自定义目录名影响），下载落点则由调用方按用户配置决定。

    默认会包含：主 LLM（local 模式）、embedding（本地后端）、reranker
    （rag_rerank_enabled）。aux_llm 仅在启用时包含；OCR 默认引擎为 docling
    （用 Apple Vision，不依赖 OvisOCR3），故默认不包含。
    """
    specs: List[Tuple[str, str, str]] = []

    def _add(role: str, path_field: str) -> None:
        raw = getattr(config, path_field, "") or ""
        if not raw:
            return
        # 以类默认值的目录名为准（即默认模型名），而不是用户可能改过的目录名。
        local_name = _default_dir_name(config, path_field) or _local_name_of(raw)
        specs.append((role, local_name, repo_id_for(local_name)))

    # 主模型：仅 local 模式、且非 openai 后端时需要本地权重。
    if getattr(config, "llm_mode", "local") == "local":
        backend = getattr(config, "llm_backend", "") or ""
        if backend and backend != "openai":
            _add("llm", "llm_model_path")

    # 嵌入：本地后端（mlx/onnx/...）才需要本地权重。
    emb_backend = getattr(config, "rag_embedding_backend", "") or ""
    if emb_backend and emb_backend != "api":
        _add("embedding", "rag_embedding_model_path")

    # 重排：启用且非 disabled/api。
    if getattr(config, "rag_rerank_enabled", False):
        rr_backend = getattr(config, "rag_rerank_backend", "") or ""
        if rr_backend and rr_backend not in ("disabled", "api"):
            _add("reranker", "rag_rerank_model_path")

    # 辅助模型：仅启用时。
    if getattr(config, "aux_llm_enabled", False):
        aux_backend = getattr(config, "aux_llm_backend", "") or ""
        if aux_backend and aux_backend != "openai":
            _add("aux_llm", "aux_llm_model_path")

    # OCR（OvisOCR3）：默认 PDF 引擎为 docling（Apple Vision），不依赖 OvisOCR3，
    # 故默认不包含；若用户显式切到 ovis 引擎且开启，可在此追加。
    if getattr(config, "rag_pdf_engine", "") == "ovis":
        _add("ocr", "rag_ocr_model_path")

    return specs

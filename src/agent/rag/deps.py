"""Dependency check and auto-install for RAG document parsing.

Scans the knowledge base directory for supported file types, checks whether
the required Python packages are installed, and offers to auto-install any
that are missing via pip.
"""
from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dependency registry: extension → list of (import_name, pip_spec)
# ---------------------------------------------------------------------------

@dataclass
class DepSpec:
    """A single dependency requirement."""
    import_name: str       # module to check with importlib
    pip_spec: str          # pip install spec (e.g., "python-docx", "numpy<2.0")
    label: str             # human-readable label
    required: bool = True  # if False, only warn (e.g., OCR is optional)


# Map each file extension to its required dependencies.
# The `required` flag distinguishes hard failures from optional enhancements.
EXTENSION_DEPS: Dict[str, List[DepSpec]] = {
    ".docx": [DepSpec("docx", "python-docx", "Word 文档解析")],
    ".doc":  [DepSpec("docx", "python-docx", "Word 文档解析")],
    ".xlsx": [DepSpec("openpyxl", "openpyxl", "Excel 表格解析")],
    ".xls":  [DepSpec("openpyxl", "openpyxl", "Excel 表格解析")],
    ".pptx": [DepSpec("pptx", "python-pptx", "PPT 幻灯片解析")],
    ".ppt":  [DepSpec("pptx", "python-pptx", "PPT 幻灯片解析")],
    ".pdf":  [
        DepSpec("fitz", "PyMuPDF", "PDF 文字解析"),
        DepSpec("rapidocr_onnxruntime", "rapidocr-onnxruntime", "图片型 PDF OCR 识别", required=False),
    ],
    ".epub": [
        DepSpec("ebooklib", "ebooklib", "EPUB 电子书解析"),
        DepSpec("bs4", "beautifulsoup4", "HTML 内容清洗"),
    ],
    # .mobi/.azw3 use calibre CLI or raw extraction — no hard Python deps
    ".mobi": [],
    ".azw3": [],
    ".azw":  [],
    # Plain text formats — no external deps
    ".txt": [],
    ".md": [],
    ".markdown": [],
    ".rst": [],
    ".csv": [],
    ".tsv": [],
    ".html": [DepSpec("bs4", "beautifulsoup4", "HTML 解析")],
    ".htm":  [DepSpec("bs4", "beautifulsoup4", "HTML 解析")],
}

# Core RAG infrastructure dependencies (always needed for vector search)
RAG_CORE_DEPS: List[DepSpec] = [
    DepSpec("lancedb", "lancedb", "向量数据库"),
    DepSpec("fastembed", "fastembed", "嵌入模型 (ONNX Runtime)"),
]

# Version constraints: packages that must be installed together with
# specific version pins to avoid conflicts.
VERSION_PINS: Dict[str, str] = {
    # FastEmbed (ONNX Runtime) 在 numpy 2.x 上有兼容性问题，需要 pin < 2.0
    "numpy": "numpy<2.0",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class DependencyReport:
    """Result of a dependency check."""
    missing_required: List[DepSpec] = field(default_factory=list)
    missing_optional: List[DepSpec] = field(default_factory=list)
    scanned_extensions: Set[str] = field(default_factory=set)

    @property
    def all_missing(self) -> List[DepSpec]:
        return self.missing_required + self.missing_optional

    @property
    def has_missing(self) -> bool:
        return bool(self.missing_required or self.missing_optional)

    @property
    def has_blocking(self) -> bool:
        """True if any *required* deps are missing."""
        return bool(self.missing_required)

    def summary(self) -> str:
        lines: List[str] = []
        if self.missing_required:
            lines.append("缺少必要依赖：")
            for d in self.missing_required:
                lines.append(f"  - {d.label} (pip install {d.pip_spec})")
        if self.missing_optional:
            lines.append("缺少可选依赖（部分功能受限）：")
            for d in self.missing_optional:
                lines.append(f"  - {d.label} (pip install {d.pip_spec})")
        if not lines:
            lines.append("所有依赖已就绪。")
        return "\n".join(lines)


def scan_extensions(kb_path: str) -> Set[str]:
    """Scan a knowledge base directory and return the set of file extensions present."""
    root = Path(kb_path)
    if not root.is_dir():
        logger.warning("RAG deps: knowledge base dir not found: %s", kb_path)
        return set()
    exts: Set[str] = set()
    for fp in root.rglob("*"):
        if fp.is_file() and not fp.name.startswith("."):
            exts.add(fp.suffix.lower())
    logger.info("RAG deps: scanned %s — found extensions: %s", kb_path, sorted(exts) if exts else "(none)")
    return exts


def check_dependencies(kb_path: str, include_core: bool = True) -> DependencyReport:
    """Check which dependencies are missing for the files in the knowledge base.

    Args:
        kb_path: Path to the knowledge base directory.
        include_core: Also check RAG core dependencies (chromadb, sentence-transformers).

    Returns:
        DependencyReport with missing required and optional deps.
    """
    report = DependencyReport()
    exts = scan_extensions(kb_path)
    report.scanned_extensions = exts

    # Collect all deps to check
    deps_to_check: List[DepSpec] = []
    if include_core:
        deps_to_check.extend(RAG_CORE_DEPS)
    for ext in exts:
        if ext in EXTENSION_DEPS:
            deps_to_check.extend(EXTENSION_DEPS[ext])

    # Deduplicate by pip_spec while preserving order
    seen: Set[str] = set()
    unique_deps: List[DepSpec] = []
    for d in deps_to_check:
        if d.pip_spec not in seen:
            seen.add(d.pip_spec)
            unique_deps.append(d)

    # Check each dependency
    for dep in unique_deps:
        if not _is_importable(dep.import_name):
            if dep.required:
                report.missing_required.append(dep)
            else:
                report.missing_optional.append(dep)

    if report.has_missing:
        logger.info(
            "RAG deps: check complete — %d required missing, %d optional missing",
            len(report.missing_required), len(report.missing_optional),
        )
    else:
        logger.info("RAG deps: all %d dependencies satisfied", len(unique_deps))
    return report


def install_packages(specs: List[str], progress_callback=None) -> Tuple[bool, str]:
    """Install pip packages in the current Python environment.

    Args:
        specs: List of pip install specs (e.g., ["python-docx", "numpy<2.0"]).
        progress_callback: Optional callable(str) for progress updates.

    Returns:
        (success, output_message)
    """
    if not specs:
        return True, "无需安装。"

    logger.info("RAG deps: installing %s", ", ".join(specs))

    # Apply version pins: if a spec doesn't already have a version constraint,
    # check if we need to pin it.
    final_specs: List[str] = []
    for spec in specs:
        # Extract package name (before any version specifier).
        # Handles: "numpy<2.0", "numpy>=1.0,<2.0", "numpy==1.20", "numpy[extra]"
        pkg_name = spec.split("<")[0].split(">")[0].split("=")[0].split("[")[0].strip().lower()
        if pkg_name in VERSION_PINS and "<" not in spec and ">" not in spec:
            final_specs.append(VERSION_PINS[pkg_name])
        else:
            final_specs.append(spec)

    cmd = [sys.executable, "-m", "pip", "install"] + final_specs
    if progress_callback:
        progress_callback(f"正在安装: {' '.join(final_specs)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5-minute timeout
        )
        if result.returncode == 0:
            msg = f"安装成功: {' '.join(final_specs)}"
            logger.info("RAG deps: install succeeded: %s", " ".join(final_specs))
            if progress_callback:
                progress_callback(msg)
            return True, msg
        else:
            error_tail = result.stderr.strip().split("\n")[-5:] if result.stderr else []
            msg = f"安装失败 (exit {result.returncode}):\n" + "\n".join(error_tail)
            logger.error("RAG deps: install failed (exit %d): %s", result.returncode, "\n".join(error_tail))
            if progress_callback:
                progress_callback(msg)
            return False, msg
    except subprocess.TimeoutExpired:
        msg = "安装超时（5 分钟），请手动执行: pip install " + " ".join(final_specs)
        logger.error("RAG deps: install timed out after 5 min: %s", " ".join(final_specs))
        if progress_callback:
            progress_callback(msg)
        return False, msg
    except Exception as e:
        msg = f"安装异常: {e}"
        logger.error("RAG deps: install exception: %s", e)
        if progress_callback:
            progress_callback(msg)
        return False, msg


def ensure_dependencies(
    kb_path: str,
    auto_install: bool = True,
    include_core: bool = True,
    progress_callback=None,
) -> DependencyReport:
    """Check and optionally auto-install missing dependencies.

    Args:
        kb_path: Path to knowledge base directory.
        auto_install: If True, install missing required deps automatically.
        include_core: Also check RAG core dependencies.
        progress_callback: Optional callable(str) for progress updates.

    Returns:
        DependencyReport describing what was missing (and what was installed).
    """
    report = check_dependencies(kb_path, include_core=include_core)

    if not report.has_blocking or not auto_install:
        logger.info("RAG deps: ensure_dependencies — no blocking deps to install (blocking=%s auto=%s)",
                    report.has_blocking, auto_install)
        return report

    # Collect pip specs for missing required deps
    specs_to_install: List[str] = []
    for dep in report.missing_required:
        specs_to_install.append(dep.pip_spec)

    logger.info("RAG deps: auto-installing %d required packages: %s",
                len(specs_to_install), ", ".join(specs_to_install))
    if progress_callback:
        progress_callback(f"检测到缺少 {len(specs_to_install)} 个必要依赖，开始自动安装…")

    success, msg = install_packages(specs_to_install, progress_callback)

    if success:
        # Re-check to update the report
        report = check_dependencies(kb_path, include_core=include_core)
        if report.has_blocking:
            logger.warning("RAG deps: post-install check — %d deps still missing",
                           len(report.missing_required))
        else:
            logger.info("RAG deps: all required deps installed successfully")
        if progress_callback:
            if report.has_blocking:
                progress_callback("部分依赖安装后仍无法导入，请手动检查。")
            else:
                progress_callback("所有必要依赖已安装成功。")
    else:
        if progress_callback:
            progress_callback(f"自动安装失败: {msg}")

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_importable(module_name: str) -> bool:
    """Check if a module can be imported without actually importing it globally."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False
    except Exception:
        # Some packages raise other exceptions on import (e.g., version warnings)
        # but are still usable. Treat as installed.
        return True

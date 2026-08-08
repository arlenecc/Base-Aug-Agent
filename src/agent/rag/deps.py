"""Dependency check, auto-install, and model pre-download for RAG.

Covers three layers:
1. Python packages (lancedb, fastembed, PyMuPDF, python-docx, etc.)
2. OCR engine (rapidocr-onnxruntime) — auto-detected if PDF has image pages
3. Embedding model (nomic-ai/nomic-embed-text-v1.5) — pre-downloaded from HF

Runs before every sync so users never hit missing-dependency errors mid-ingest.
"""
from __future__ import annotations

import importlib
import logging
import os
import subprocess
import warnings
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dependency registry
# ---------------------------------------------------------------------------


@dataclass
class DepSpec:
    """A single dependency requirement."""
    import_name: str       # module to check with importlib
    pip_spec: str          # pip install spec (e.g., "python-docx", "numpy<2.0")
    label: str             # human-readable label
    required: bool = True  # if False, only warn (e.g., OCR is optional unless needed)


# Map each file extension to its required dependencies.
EXTENSION_DEPS: Dict[str, List[DepSpec]] = {
    ".docx": [DepSpec("docx", "python-docx", "Word 文档解析")],
    ".doc":  [DepSpec("docx", "python-docx", "Word 文档解析")],
    ".xlsx": [DepSpec("openpyxl", "openpyxl", "Excel 表格解析")],
    ".xls":  [DepSpec("openpyxl", "openpyxl", "Excel 表格解析")],
    ".pptx": [DepSpec("pptx", "python-pptx", "PPT 幻灯片解析")],
    ".ppt":  [DepSpec("pptx", "python-pptx", "PPT 幻灯片解析")],
    ".pdf":  [
        DepSpec("fitz", "PyMuPDF", "PDF 文字解析"),
        # OCR is conditional: only required when the PDF actually has image pages.
        # check_dependencies() will upgrade it to required if image pages are detected.
        DepSpec("rapidocr_onnxruntime", "rapidocr-onnxruntime", "图片型 PDF OCR 识别", required=False),
    ],
    ".epub": [
        DepSpec("ebooklib", "ebooklib", "EPUB 电子书解析"),
        DepSpec("bs4", "beautifulsoup4", "HTML 内容清洗"),
    ],
    ".mobi": [],
    ".azw3": [],
    ".azw":  [],
    ".txt": [],
    ".md": [],
    ".markdown": [],
    ".rst": [],
    ".csv": [],
    ".tsv": [],
    ".html": [DepSpec("bs4", "beautifulsoup4", "HTML 解析")],
    ".htm":  [DepSpec("bs4", "beautifulsoup4", "HTML 解析")],
}

# Core RAG infrastructure (always checked, always required)
RAG_CORE_DEPS: List[DepSpec] = [
    DepSpec("lancedb", "lancedb", "向量数据库"),
    DepSpec("fastembed", "fastembed", "嵌入模型 (ONNX Runtime)"),
]

# Embedding model info for pre-download
# Embedding model: quantized ONNX variant — only 137MB (vs 548MB for FP32).
# FastEmbed uses model_quantized.onnx; we download only the required files
# instead of the full snapshot (which is ~1.6GB with all quantization formats).
EMBEDDING_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_MODEL_LABEL = "嵌入模型 nomic-embed-text-v1.5 (~137MB, 仅下载 ONNX 量化版)"

# FastEmbed model name — uses the -Q variant (quantized ONNX, model_quantized.onnx).
# Same 768-dim embeddings, 4x smaller than FP32, 12x smaller than full snapshot.
FASTEMBED_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5-Q"

# Files actually needed by FastEmbed for the quantized model.
# Only these files are downloaded — not the entire ~1.6GB repo snapshot.
_EMBEDDING_REQUIRED_FILES = [
    "onnx/model_quantized.onnx",   # quantized ONNX model (~137MB)
    "tokenizer.json",              # tokenizer (~0.7MB)
    "config.json",                 # model config (~3KB)
    "tokenizer_config.json",       # tokenizer config (~4KB)
    "vocab.txt",                   # vocabulary (~0.2MB)
]

# Reranker model (used at search time, not ingest — pre-download to avoid
# blocking the first search). ~500MB download, optional: search falls back
# to distance-based ranking if unavailable.
RERANK_MODEL_ID = "BAAI/bge-reranker-base"
RERANK_MODEL_LABEL = "重排序模型 bge-reranker-base (~500MB, 可选)"

# Files needed for the BGE reranker (minimal set for FlagEmbedding).
_RERANKER_REQUIRED_FILES = [
    "pytorch_model.bin",   # PyTorch weights (~500MB)
    "config.json",         # model config
    "tokenizer.json",      # tokenizer
    "tokenizer_config.json",
    "vocab.txt",
    "sentencepiece.bpe.model",
]

# Version pins: packages that need specific version constraints
VERSION_PINS: Dict[str, str] = {
    "numpy": "numpy<2.0",  # FastEmbed ONNX Runtime compat
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class DependencyReport:
    """Result of a full dependency check (packages + models)."""
    # Python packages
    missing_required: List[DepSpec] = field(default_factory=list)
    missing_optional: List[DepSpec] = field(default_factory=list)
    scanned_extensions: Set[str] = field(default_factory=set)

    # Models
    embedding_model_missing: bool = False
    reranker_model_missing: bool = False

    # OCR detection
    has_image_pdf: bool = False   # knowledge base contains image-based PDF pages
    ocr_installed: bool = False

    # Install results
    installed_packages: List[str] = field(default_factory=list)
    install_errors: List[str] = field(default_factory=list)

    @property
    def all_missing(self) -> List[DepSpec]:
        return self.missing_required + self.missing_optional

    @property
    def has_missing(self) -> bool:
        return bool(self.missing_required or self.missing_optional)

    @property
    def has_blocking(self) -> bool:
        """True if any *required* deps or the embedding model is missing."""
        return bool(self.missing_required) or self.embedding_model_missing

    def summary(self) -> str:
        """Human-readable summary of what's missing."""
        lines: List[str] = []
        if self.missing_required:
            lines.append("缺少必要 Python 包：")
            for d in self.missing_required:
                lines.append(f"  - {d.label} (pip install {d.pip_spec})")
        if self.embedding_model_missing:
            lines.append(f"缺少嵌入模型：{EMBEDDING_MODEL_LABEL}")
            lines.append(f"  → 模型 ID: {EMBEDDING_MODEL_ID} (仅需 {len(_EMBEDDING_REQUIRED_FILES)} 个文件)")
            lines.append(f"  → 手动下载命令：")
            lines.append(f"     export HF_ENDPOINT=https://hf-mirror.com")
            lines.append(f"     export HF_HUB_DISABLE_XET=1")
            lines.append(f'     python3 -c "from huggingface_hub import hf_hub_download; '
                         f"[hf_hub_download('{EMBEDDING_MODEL_ID}', f) for f in {_EMBEDDING_REQUIRED_FILES}]\"")

        if self.has_image_pdf and not self.ocr_installed:
            lines.append("知识库包含图片版 PDF，但 OCR 引擎未安装：")
            lines.append("  - rapidocr-onnxruntime (pip install rapidocr-onnxruntime)")
        if self.missing_optional:
            lines.append("缺少可选依赖（部分功能受限）：")
            for d in self.missing_optional:
                lines.append(f"  - {d.label} (pip install {d.pip_spec})")
        if self.installed_packages:
            lines.append(f"已自动安装 {len(self.installed_packages)} 个包：")
            for pkg in self.installed_packages:
                lines.append(f"  ✅ {pkg}")
        if self.install_errors:
            lines.append("安装失败：")
            for err in self.install_errors:
                lines.append(f"  ❌ {err}")
        if not lines:
            lines.append("所有依赖已就绪（Python 包 + 嵌入模型 + OCR 引擎）。")
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


# ---------------------------------------------------------------------------
# PDF image-page detection
# ---------------------------------------------------------------------------


def _pdf_has_image_pages(kb_path: str, max_pages_per_pdf: int = 10) -> bool:
    """Quickly sample PDF files to check if any have image-based pages.

    Only checks the first ``max_pages_per_pdf`` pages of each PDF to keep
    the pre-sync check fast. Returns True as soon as one image page is found.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        # Can't check without PyMuPDF — assume no image PDFs
        return False

    root = Path(kb_path)
    pdf_files = list(root.rglob("*.pdf")) + list(root.rglob("*.PDF"))
    if not pdf_files:
        return False

    logger.info("RAG deps: checking %d PDF(s) for image-based pages...", len(pdf_files))
    for pdf_path in pdf_files:
        if pdf_path.name.startswith("."):
            continue
        try:
            doc = fitz.open(str(pdf_path))
            pages_to_check = min(len(doc), max_pages_per_pdf)
            for i in range(pages_to_check):
                page = doc[i]
                text = page.get_text("text").strip()
                if not text and page.get_images(full=True):
                    doc.close()
                    logger.info(
                        "RAG deps: found image-based page in '%s' (page %d)",
                        pdf_path.name, i + 1,
                    )
                    return True
            doc.close()
        except Exception as e:
            logger.debug("RAG deps: failed to check PDF '%s': %s", pdf_path.name, e)

    return False


# ---------------------------------------------------------------------------
# Embedding model pre-download
# ---------------------------------------------------------------------------


def _get_fastembed_cache_dir() -> str:
    """Return the persistent cache directory for FastEmbed models.

    FastEmbed defaults to a system temp dir (/var/folders/.../T/) which
    macOS cleans regularly, causing the model to be re-downloaded.  We
    use ~/.cache/fastembed instead — same as vector_store.py.
    """
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "fastembed")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _is_embedding_model_cached() -> bool:
    """Check whether the embedding model is already cached locally.

    FastEmbed uses its OWN cache directory (NOT the standard HuggingFace
    cache).  We check the persistent cache dir (~/.cache/fastembed) that
    we configure in vector_store.py and _download_embedding_model().
    """
    try:
        from fastembed import TextEmbedding
        cache_dir = _get_fastembed_cache_dir()
        # lazy_load=True: don't load ONNX Runtime, just resolve the model path
        # Suppress FastEmbed's "model has been updated" UserWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model = TextEmbedding(
                model_name=FASTEMBED_MODEL_NAME,
                cache_dir=cache_dir,
                lazy_load=True,
            )
        # Check if the model directory actually has the ONNX file
        model_dir = getattr(model.model, "_model_dir", None)
        if model_dir and os.path.isdir(model_dir):
            # Check for the quantized ONNX file
            onnx_path = os.path.join(model_dir, "onnx", "model_quantized.onnx")
            if os.path.isfile(onnx_path):
                logger.debug("RAG deps: embedding model found in FastEmbed cache: %s", onnx_path)
                return True
        return False
    except Exception as e:
        logger.debug("RAG deps: FastEmbed cache check error: %s", e)
        return False


def _download_embedding_model(progress_callback=None) -> Tuple[bool, str]:
    """Pre-download the embedding model by instantiating FastEmbed.

    Previously this used a subprocess to call huggingface_hub.hf_hub_download,
    but that had two problems:
    1. It downloaded to the HF cache, but FastEmbed uses its OWN cache —
       so the model was downloaded but FastEmbed didn't see it.
    2. In a PyInstaller-frozen .app, sys.executable is the frozen binary
       and the subprocess can't import huggingface_hub.

    The fix: just instantiate TextEmbedding directly.  FastEmbed handles
    the download to its own cache directory, which is where it will look
    for the model at runtime.  This works in both dev and frozen envs.
    """
    logger.info("RAG deps: downloading embedding model via FastEmbed...")
    if progress_callback:
        progress_callback(f"正在下载{EMBEDDING_MODEL_LABEL}（FastEmbed 自动缓存）...")

    try:
        from fastembed import TextEmbedding
        # Instantiating TextEmbedding triggers the download to FastEmbed's
        # own cache directory.  Use the same persistent cache_dir as
        # vector_store.py to avoid re-downloads after macOS temp cleanup.
        cache_dir = _get_fastembed_cache_dir()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            TextEmbedding(model_name=FASTEMBED_MODEL_NAME, cache_dir=cache_dir)
        logger.info("RAG deps: embedding model downloaded and cached by FastEmbed")
        if progress_callback:
            progress_callback(f"✅ {EMBEDDING_MODEL_LABEL} 下载完成")
        return True, "下载成功"
    except Exception as e:
        msg = str(e)
        logger.error("RAG deps: embedding model download failed: %s", e)
        if progress_callback:
            progress_callback(f"❌ 模型下载失败: {msg}")
        return False, msg


# ---------------------------------------------------------------------------
# Package checks
# ---------------------------------------------------------------------------


def check_dependencies(kb_path: str, include_core: bool = True) -> DependencyReport:
    """Check Python packages, embedding model, and OCR engine availability.

    Args:
        kb_path: Path to the knowledge base directory.
        include_core: Also check RAG core dependencies (lancedb, fastembed).

    Returns:
        DependencyReport with full status of all dependencies.
    """
    report = DependencyReport()
    exts = scan_extensions(kb_path)
    report.scanned_extensions = exts

    # ---- Python packages ----
    deps_to_check: List[DepSpec] = []
    if include_core:
        deps_to_check.extend(RAG_CORE_DEPS)
    for ext in exts:
        if ext in EXTENSION_DEPS:
            deps_to_check.extend(EXTENSION_DEPS[ext])

    # Deduplicate by pip_spec
    seen: Set[str] = set()
    unique_deps: List[DepSpec] = []
    for d in deps_to_check:
        if d.pip_spec not in seen:
            seen.add(d.pip_spec)
            unique_deps.append(d)

    for dep in unique_deps:
        if not _is_importable(dep.import_name):
            if dep.required:
                report.missing_required.append(dep)
            else:
                report.missing_optional.append(dep)

    # ---- OCR engine: upgrade from optional to required if image PDFs exist ----
    if ".pdf" in exts:
        has_images = _pdf_has_image_pages(kb_path)
        report.has_image_pdf = has_images
        # Check if OCR is installed
        report.ocr_installed = _is_importable("rapidocr_onnxruntime")
        if has_images and not report.ocr_installed:
            # Move OCR from optional to required
            ocr_spec = DepSpec("rapidocr_onnxruntime", "rapidocr-onnxruntime", "图片型 PDF OCR 识别")
            if ocr_spec not in report.missing_required:
                report.missing_required.append(ocr_spec)

    # ---- Embedding model ----
    if include_core:
        report.embedding_model_missing = not _is_embedding_model_cached()

    # ---- Reranker model (optional, only warn) ----
    report.reranker_model_missing = not _is_reranker_cached()

    # Log summary
    if report.has_missing or report.embedding_model_missing:
        logger.info(
            "RAG deps: check complete — %d required pkgs missing, %d optional, "
            "embedding_model=%s, ocr=%s, image_pdf=%s",
            len(report.missing_required), len(report.missing_optional),
            "MISSING" if report.embedding_model_missing else "OK",
            "OK" if report.ocr_installed else "MISSING",
            report.has_image_pdf,
        )
    else:
        logger.info("RAG deps: all %d packages + embedding model + OCR are ready", len(unique_deps))

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

    # Apply version pins
    final_specs: List[str] = []
    for spec in specs:
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
    """Check and optionally auto-install ALL missing dependencies.

    This is the main entry point called by InstallDepsWorker before every sync.
    It handles:
    1. Python packages (auto-install via pip)
    2. Embedding model (pre-download from HuggingFace)
    3. OCR engine (auto-install if image-based PDFs are detected)

    Args:
        kb_path: Path to knowledge base directory.
        auto_install: If True, install missing deps automatically.
        include_core: Also check RAG core dependencies.
        progress_callback: Optional callable(str) for progress updates.

    Returns:
        DependencyReport describing what was missing and what was installed.
    """
    report = check_dependencies(kb_path, include_core=include_core)

    if not auto_install:
        return report

    installed_anything = False

    # ---- Step 1: Install missing Python packages ----
    if report.has_missing:
        specs_to_install: List[str] = []
        for dep in report.missing_required:
            specs_to_install.append(dep.pip_spec)
        for dep in report.missing_optional:
            specs_to_install.append(dep.pip_spec)

        if specs_to_install:
            logger.info("RAG deps: auto-installing %d packages: %s",
                        len(specs_to_install), ", ".join(specs_to_install))
            if progress_callback:
                progress_callback(f"检测到缺少 {len(specs_to_install)} 个依赖，开始自动安装…")

            success, msg = install_packages(specs_to_install, progress_callback)

            if success:
                report.installed_packages.extend(specs_to_install)
                installed_anything = True
                # Re-check to update the report
                report = check_dependencies(kb_path, include_core=include_core)
                # Preserve install info
                report.installed_packages = specs_to_install
                if report.has_blocking:
                    logger.warning("RAG deps: post-install — %d deps still missing",
                                   len(report.missing_required))
                else:
                    logger.info("RAG deps: all packages installed successfully")
                if progress_callback:
                    if report.has_blocking:
                        progress_callback("部分依赖安装后仍无法导入，请手动检查。")
                    else:
                        progress_callback("所有 Python 依赖已安装成功。")
            else:
                report.install_errors.append(msg)
                if progress_callback:
                    progress_callback(f"自动安装失败: {msg}")

    # ---- Step 2: Pre-download embedding model ----
    if report.embedding_model_missing:
        if progress_callback:
            progress_callback(f"嵌入模型未缓存，开始下载 {EMBEDDING_MODEL_LABEL}…")
        success, msg = _download_embedding_model(progress_callback)
        if success:
            report.embedding_model_missing = False
            installed_anything = True
        else:
            report.install_errors.append(f"嵌入模型下载失败: {msg}")

    # ---- Step 3: Pre-download reranker model (optional, non-blocking) ----
    if report.reranker_model_missing:
        if progress_callback:
            progress_callback(f"重排序模型未缓存，开始后台下载 {RERANK_MODEL_LABEL}…")
        success, msg = _download_reranker_model(progress_callback)
        if success:
            report.reranker_model_missing = False
            installed_anything = True
        else:
            # Reranker is optional — don't block sync if download fails
            report.install_errors.append(f"重排序模型下载失败（可选，不影响同步）: {msg}")

    if installed_anything and progress_callback:
        progress_callback("依赖检查完成。")

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_importable(module_name: str) -> bool:
    """Check if a module can be imported without importing it globally."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False
    except Exception:
        # Some packages raise non-ImportError on import but are still usable.
        return True


def _is_reranker_cached() -> bool:
    """Check if the BGE reranker model is cached locally."""
    try:
        from huggingface_hub import try_to_load_from_cache
        key_files = ["pytorch_model.bin", "model.safetensors", "config.json"]
        for kf in key_files:
            cached = try_to_load_from_cache(RERANK_MODEL_ID, kf)
            if cached is not None:
                return True
        return False
    except Exception:
        return False


def _download_reranker_model(progress_callback=None) -> Tuple[bool, str]:
    """Pre-download only the files FlagEmbedding needs for BGE reranker.

    Uses hf_hub_download for individual files instead of snapshot_download
    to avoid pulling the entire repo (~1GB+ with unused formats).
    """
    logger.info(
        "RAG deps: downloading reranker model files (%d files)...",
        len(_RERANKER_REQUIRED_FILES),
    )
    if progress_callback:
        progress_callback(f"正在下载{RERANK_MODEL_LABEL}（仅必需文件）...")

    env = os.environ.copy()
    env["HF_HUB_DISABLE_XET"] = "1"
    if "HF_ENDPOINT" not in env:
        env["HF_ENDPOINT"] = "https://huggingface.co"

    lines = [
        "from huggingface_hub import hf_hub_download",
        f"model_id = '{RERANK_MODEL_ID}'",
        f"files = {_RERANKER_REQUIRED_FILES!r}",
        "import sys",
        "for filename in files:",
        "    print(f'  Downloading {filename}...', file=sys.stderr)",
        "    local = hf_hub_download(repo_id=model_id, filename=filename)",
        "    print(f'    -> {local}', file=sys.stderr)",
        "print('ALL_FILES_DOWNLOADED')",
    ]
    script = "\n".join(lines)

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=900,  # 15 minutes for ~500MB
            env=env,
        )
        if result.returncode == 0 and "ALL_FILES_DOWNLOADED" in result.stdout:
            logger.info("RAG deps: reranker model files downloaded")
            if progress_callback:
                progress_callback(f"✅ {RERANK_MODEL_LABEL} 下载完成")
            return True, "下载成功"
        else:
            error_tail = result.stderr.strip().split("\n")[-3:] if result.stderr else ["unknown error"]
            msg = "\n".join(error_tail)
            logger.warning("RAG deps: reranker download failed: %s", msg)
            if progress_callback:
                progress_callback(f"⚠ 重排序模型下载失败（可选）: {msg}")
            return False, msg
    except subprocess.TimeoutExpired:
        msg = "下载超时（15 分钟）"
        logger.warning("RAG deps: reranker download timed out")
        if progress_callback:
            progress_callback(f"⚠ 重排序模型{msg}（可选，不影响同步）")
        return False, msg
    except Exception as e:
        msg = str(e)
        logger.warning("RAG deps: reranker download exception: %s", e)
        return False, msg

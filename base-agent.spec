# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for building base-agent into a platform-native bundle.

Usage:
    pyinstaller base-agent.spec

Produces:
    macOS : dist/BaseAgent.app  — application bundle (via BUNDLE)
    Win   : dist/BaseAgent/     — onedir folder with BaseAgent.exe (via COLLECT)
    Linux: dist/BaseAgent/      — onedir folder (via COLLECT)
"""

import os
import sys

# Icon: macOS uses .icns; Windows/Linux need .ico (fall back to none if absent).
if sys.platform == 'darwin':
    _ICON = 'BaseAgent.icns'
elif os.path.exists('BaseAgent.ico'):
    _ICON = 'BaseAgent.ico'
else:
    _ICON = None

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

block_cipher = None


# Helper to bundle an arbitrary directory tree as datas (PyInstaller 6.x no
# longer injects add_data_dir into the spec namespace, so define it here).
def add_data_dir(src, dest):
    out = []
    for root, _dirs, files in os.walk(src):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, src)
            out.append((full, os.path.join(dest, rel)))
    return out

# -----------------------------------------------------------------------
# Collect all data files and submodules for complex dependencies
# -----------------------------------------------------------------------

datas = []
binaries = []
hiddenimports = []

# --- PyQt6 ---
datas += collect_data_files('PyQt6')
hiddenimports += collect_submodules('PyQt6')

# --- httpx / httpcore ---
hiddenimports += collect_submodules('httpx')
hiddenimports += collect_submodules('httpcore')
hiddenimports += collect_submodules('h11')
hiddenimports += collect_submodules('certifi')

# --- BeautifulSoup4 ---
datas += collect_data_files('bs4')

# --- LanceDB (vector store) ---
hiddenimports += collect_submodules('lancedb')
datas += collect_data_files('lancedb')
# LanceDB uses pyarrow / lance under the hood
hiddenimports += collect_submodules('pyarrow')
datas += collect_data_files('pyarrow')

# --- FastEmbed (embedding model runner) ---
hiddenimports += collect_submodules('fastembed')
datas += collect_data_files('fastembed')
# onnxruntime is used by fastembed
hiddenimports += collect_submodules('onnxruntime')
datas += collect_data_files('onnxruntime')

# --- FlagEmbedding (reranker) ---
try:
    hiddenimports += collect_submodules('FlagEmbedding')
except Exception:
    pass

# --- Document parsers (optional RAG deps) ---
for mod in ['docx', 'openpyxl', 'pptx', 'fitz', 'ebooklib']:
    try:
        hiddenimports += collect_submodules(mod)
        datas += collect_data_files(mod)
    except Exception:
        pass

# --- docling (unified document parser, lazy-loaded) ---
try:
    hiddenimports += collect_submodules('docling')
    datas += collect_data_files('docling')
except Exception:
    pass

# --- chonkie (semantic chunking, lazy-loaded) ---
try:
    hiddenimports += collect_submodules('chonkie')
    datas += collect_data_files('chonkie')
except Exception:
    pass

# --- jieba (BM25 CJK tokenizer) ---
try:
    hiddenimports += collect_submodules('jieba')
    datas += collect_data_files('jieba')
except Exception:
    pass

# --- numpy (semantic chunker vector ops) ---
try:
    hiddenimports += collect_submodules('numpy')
except Exception:
    pass

# --- agent internal modules (ensure fully bundled; some are lazily imported) ---
hiddenimports += [
    'agent', 'agent.ui', 'agent.ui.main_window',
    'agent.rag', 'agent.rag.engine', 'agent.rag.vector_store',
    'agent.rag.parsers', 'agent.rag.deps',
]

# --- RapidOCR (optional OCR for scanned PDFs) ---
try:
    hiddenimports += collect_submodules('rapidocr_onnxruntime')
    datas += collect_data_files('rapidocr_onnxruntime')
except Exception:
    pass

# --- Package metadata (needed by some libs for version checks) ---
datas += copy_metadata('PyQt6')
datas += copy_metadata('httpx')
datas += copy_metadata('lancedb')
datas += copy_metadata('fastembed')
datas += copy_metadata('beautifulsoup4')
# docling / chonkie 为可选依赖（未安装时回退到按格式解析/递归分片），
# 用 try/except 保护，避免打包时因缺少元数据而失败。
for _pkg in ('docling', 'chonkie'):
    try:
        datas += copy_metadata(_pkg)
    except Exception:
        pass

# -----------------------------------------------------------------------
# 可选：将默认模型内嵌进 .app（默认不内嵌，体积过大）
# -----------------------------------------------------------------------
# 配置里默认相对路径 "models/<name>" 由 resolve_model_path 解析，打包后按
# 以下位置查找（详见 src/agent/model_backends.py::resolve_model_path）：
#   - 默认位："models"        → .app/Contents/MacOS/models          （候选④，推荐）
#   - 标准位："Contents/Resources/models" → .app/Contents/Resources/models（候选⑤）
# 模型（主模型 Qwen3.5-* + 辅助模型 jina/OvisOCR 等）可达数 GB，默认不内嵌
# 以控制包体。需要离线开箱即用时设置环境变量再构建：
#     BUNDLE_MODELS=1 python -m PyInstaller base-agent.spec
# 未内嵌时不影响打包/运行（运行时仍可在 UI 里手动指定模型路径，或把
# models/ 放到可执行文件同级目录）。
MODELS_SRC = "models"
# 辅助模型（嵌入 / 重排）：离线 RAG 必需，体积较小。
# 注：OvisOCR3(OCR) 不打包以控制体积，需要 OCR 时运行时另行指定模型路径。
AUX_MODEL_DIRS = [
    "jina-embeddings-v5-text-nano-mlx-int4",
    "jina-reranker-v2-base-multilingual-onnx",
]


def _bundle_model_dir(name):
    global datas
    src = os.path.join(MODELS_SRC, name)
    if not os.path.isdir(src):
        print(f"[spec] 警告: 未找到 {src}，跳过")
        return
    # 落入 Contents/MacOS/models/<name>（resolve 候选④命中）
    datas += add_data_dir(src, os.path.join("models", name))


if os.environ.get("BUNDLE_MODELS") == "1" and os.path.isdir(MODELS_SRC):
    # 全量内嵌（主模型 Qwen3.5 + 辅助模型）
    datas += add_data_dir(MODELS_SRC, "models")
    print(f"[spec] BUNDLE_MODELS=1：已内嵌全部 {MODELS_SRC}/（含主模型）")
elif os.environ.get("BUNDLE_AUX_MODELS") == "1" and os.path.isdir(MODELS_SRC):
    # 仅内嵌辅助模型（排除主模型 Qwen3.5，体积可控且离线 RAG/OCR 可用）
    for _d in AUX_MODEL_DIRS:
        _bundle_model_dir(_d)
    print(f"[spec] BUNDLE_AUX_MODELS=1：已内嵌辅助模型 {AUX_MODEL_DIRS}（不含主模型）")
else:
    print(f"[spec] 未内嵌模型（BUNDLE_MODELS/BUNDLE_AUX_MODELS 未开启）—— 运行时需在 UI 指定模型路径")

# -----------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------

a = Analysis(
    ['main.py'],
    pathex=['src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy unused modules to reduce bundle size
        'matplotlib',
        'numpy.tests',
        'pandas.tests',
        'pytest',
        'IPython',
        'notebook',
        'jupyter',
        'tkinter',
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# -----------------------------------------------------------------------
# Command-line executable (single file)
# -----------------------------------------------------------------------

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BaseAgent',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # Keep console for log output
    icon=_ICON,
)

if sys.platform == 'darwin':
    # -------------------------------------------------------------------
    # macOS .app bundle
    # -------------------------------------------------------------------
    app = BUNDLE(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        name='BaseAgent.app',
        icon=_ICON,
        bundle_identifier='com.baseagent.app',
        info_plist={
            'CFBundleDisplayName': 'BaseAgent',
            'CFBundleShortVersionString': '0.1.0',
            'CFBundleVersion': '1',
            'NSHighResolutionCapable': True,
            'NSMicrophoneUsageDescription': 'BaseAgent may need microphone access for voice input.',
            'LSMinimumSystemVersion': '12.0',
        },
    )
else:
    # -------------------------------------------------------------------
    # Windows / Linux onedir folder (BUNDLE is macOS-only)
    # -------------------------------------------------------------------
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        name='BaseAgent',
        icon=_ICON,
    )

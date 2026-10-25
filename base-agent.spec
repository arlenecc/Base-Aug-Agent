# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for building base-agent into a macOS .app bundle.

Usage:
    pyinstaller base-agent.spec

Produces:
    dist/BaseAgent.app       — macOS application bundle
    dist/BaseAgent           — command-line executable
"""

import os
import sys
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

block_cipher = None

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
hiddenimports += collect_submodules('FlagEmbedding')

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
# 可选：将默认模型内嵌进 .app（#用户需求：打包内置模型）
# -----------------------------------------------------------------------
# 配置里默认相对路径 "models/<name>" 由 resolve_model_path 解析，打包后按
# 以下位置查找（详见 src/agent/model_backends.py::resolve_model_path）：
#   - 默认位："models"        → .app/Contents/MacOS/models          （候选④，推荐）
#   - 标准位："Contents/Resources/models" → .app/Contents/Resources/models（候选⑤）
# 模型可达数 GB：onefile 模式每次启动都会解压、体积与启动成本高，建议用本
# spec 的 .app 目录(onedir)模式。models/ 不存在时跳过，不影响打包/运行
# （运行时仍可在 UI 里手动指定模型路径）。
MODELS_SRC = "models"
if os.path.isdir(MODELS_SRC):
    # 默认：落入 Contents/MacOS/models（与可执行文件同级，resolve 候选④命中）
    datas += add_data_dir(MODELS_SRC, "models")
    # 可选 macOS 标准位（resolve 候选⑤）：改用下一行代替上一行
    # datas += add_data_dir(MODELS_SRC, "Contents/Resources/models")
    print(f"[spec] 已将 {MODELS_SRC}/ 内嵌进 .app -> Contents/MacOS/models")
else:
    print(f"[spec] 未找到 {MODELS_SRC}/，跳过模型内嵌（运行时可在 UI 指定模型路径）")

# -----------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------

a = Analysis(
    ['main.py'],
    pathex=['src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
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
)

# -----------------------------------------------------------------------
# macOS .app bundle
# -----------------------------------------------------------------------

app = BUNDLE(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='BaseAgent.app',
    icon='BaseAgent.icns',
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

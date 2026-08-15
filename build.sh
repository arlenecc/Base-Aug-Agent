#!/bin/bash
# Build BaseAgent into a macOS .app bundle using PyInstaller.
#
# Usage:
#   ./build.sh          — build the app
#   ./build.sh clean    — remove build artifacts before building
#   ./build.sh dist     — create a zip distribution
#   ./build.sh dmg      — build the app and create a .dmg installer
#
# Output:
#   dist/BaseAgent.app  — the macOS application bundle
#   dist/BaseAgent.dmg  — (with `dmg` subcommand) disk image installer

set -euo pipefail

cd "$(dirname "$0")"

# RAG 依赖（FlagEmbedding/torch/transformers/docling/chonkie 等）安装在
# Python 3.12 环境（macOS Intel 上 torch 仅 3.12 有 wheel）。默认的 `python3`
# 可能指向 3.13/3.14，缺少这些依赖。这里优先使用 3.12 的解释器，保证打包
# 出的 .app 包含完整的 RAG 能力（BGE Reranker 等）。
PY312="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
if [[ -x "$PY312" ]]; then
    PYTHON_BIN="$PY312"
    PIP_BIN="${PY312%/python3}/pip3"
else
    PYTHON_BIN="$(command -v python3)"
    PIP_BIN="$(command -v pip3)"
fi
echo "🔧 使用 Python: $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"

if [[ "${1:-}" == "clean" ]]; then
    echo "🧹 Cleaning previous build artifacts..."
    rm -rf build/ dist/
fi

# Ensure PyInstaller is installed
if ! "$PYTHON_BIN" -c "import PyInstaller" 2>/dev/null; then
    echo "📦 Installing PyInstaller..."
    "$PIP_BIN" install pyinstaller
fi

echo "🔨 Building BaseAgent.app..."
PYTHONPATH=src "$PYTHON_BIN" -m PyInstaller base-agent.spec --noconfirm

echo ""
echo "✅ Build complete!"
echo "   App:  dist/BaseAgent.app"
echo "   Size: $(du -sh dist/BaseAgent.app | cut -f1)"
echo ""
echo "   To launch: open dist/BaseAgent.app"
echo "   To distribute: ./build.sh dist"

if [[ "${1:-}" == "dist" ]]; then
    echo ""
    echo "📦 Creating zip distribution..."
    cd dist
    zip -r -y BaseAgent.zip BaseAgent.app
    cd ..
    echo "   Archive: dist/BaseAgent.zip ($(du -sh dist/BaseAgent.zip | cut -f1))"
fi

create_dmg() {
    local app_path="dist/BaseAgent.app"
    local dmg_path="dist/BaseAgent.dmg"
    local staging="dist/dmg_staging"
    local vol_name="BaseAgent"

    if [[ ! -d "$app_path" ]]; then
        echo "❌ $app_path not found. Run ./build.sh first." >&2
        return 1
    fi

    echo ""
    echo "💿 Creating DMG installer..."

    # Clean up any previous artifacts
    rm -f "$dmg_path"
    rm -rf "$staging"
    mkdir -p "$staging"

    # Symlink to /Applications so users can drag-and-drop
    ln -s /Applications "$staging/Applications"
    # Copy the app (hdiutil needs real files, not symlinks to the app)
    cp -R "$app_path" "$staging/"

    # Create a read-only compressed DMG directly with hdiutil.
    # UDZO = zlib-compressed read-only, widely compatible on macOS.
    echo "   Creating compressed read-only DMG..."
    hdiutil create -srcfolder "$staging" \
        -volname "$vol_name" \
        -fs HFS+ \
        -format UDZO \
        -imagekey zlib-level=9 \
        "$dmg_path" -quiet

    # Clean up staging
    rm -rf "$staging"

    echo "   DMG:   $dmg_path ($(du -sh "$dmg_path" | cut -f1))"
}

if [[ "${1:-}" == "dmg" ]]; then
    create_dmg
    echo ""
    echo "🎉 Done! Distribute dist/BaseAgent.dmg"
fi

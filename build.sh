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

if [[ "${1:-}" == "clean" ]]; then
    echo "🧹 Cleaning previous build artifacts..."
    rm -rf build/ dist/
fi

# Ensure PyInstaller is installed
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "📦 Installing PyInstaller..."
    pip3 install pyinstaller
fi

echo "🔨 Building BaseAgent.app..."
PYTHONPATH=src python3 -m PyInstaller base-agent.spec --noconfirm

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

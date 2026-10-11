#!/usr/bin/env bash
# Builds an Apple Silicon, ad-hoc signed Macrophone.app into ./dist.
# Works with just the Xcode Command Line Tools — no Xcode project needed.
#
# Usage: scripts/build-app.sh            (version from ./VERSION)
#        VERSION=0.2.0 scripts/build-app.sh
set -euo pipefail

CDPATH= cd -- "$(dirname -- "$0")/.."
VERSION="${VERSION:-$(cat VERSION)}"
BUILD="${BUILD:-$(git rev-list --count HEAD 2>/dev/null || echo 1)}"
APP="dist/Macrophone.app"

[ -f build/whisper/src/libwhisper.a ] || scripts/build-whisper.sh

echo "▸ Building Macrophone $VERSION ($BUILD)"
swift build -c release --arch arm64 --product Macrophone -Xswiftc -gnone -Xlinker -dead_strip --quiet

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cp .build/arm64-apple-macosx/release/Macrophone "$APP/Contents/MacOS/Macrophone"
strip -x "$APP/Contents/MacOS/Macrophone"

sed -e "s/__VERSION__/$VERSION/" -e "s/__BUILD__/$BUILD/" Resources/Info.plist > "$APP/Contents/Info.plist"
printf 'APPL????' > "$APP/Contents/PkgInfo"

# Ad-hoc signature with a stable designated requirement, so rebuilding doesn't reset the
# Microphone / Input Monitoring / Accessibility grants.
codesign --force --sign - --requirements '=designated => identifier "com.ethanclawsie.macrophone"' --timestamp=none "$APP"
codesign --verify --strict "$APP"

echo "✓ $APP ($(du -sh "$APP" | cut -f1 | xargs), $(lipo -archs "$APP/Contents/MacOS/Macrophone"))"

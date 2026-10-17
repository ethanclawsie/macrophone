#!/usr/bin/env bash
# Publishes a release: builds the app, uploads a zip to GitHub Releases, and updates the
# Homebrew cask in ethanclawsie/homebrew-tap.
#
# Usage: scripts/release.sh 0.1.0
#
# Always runs gh/git as the personal account (keychain login) by ignoring any work
# GH_TOKEN / GITHUB_TOKEN environment variables.
set -euo pipefail

cd "$(dirname "$0")/.."
VERSION="${1:?usage: scripts/release.sh <version>}"
OWNER="ethanclawsie"
REPO="$OWNER/macrophone"
TAP_REPO="$OWNER/homebrew-tap"
ZIP="dist/Macrophone-$VERSION.zip"

gh_personal() { env -u GH_TOKEN -u GITHUB_TOKEN gh "$@"; }
git_personal() { env -u GH_TOKEN -u GITHUB_TOKEN git "$@"; }

[[ "$(gh_personal api user --jq .login)" == "$OWNER" ]] || { echo "✗ gh is not logged in as $OWNER"; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "✗ working tree not clean"; exit 1; }

echo "$VERSION" > VERSION
git add VERSION && git commit -qm "Release $VERSION" || true

VERSION="$VERSION" scripts/build-app.sh
rm -f "$ZIP"
ditto -c -k --keepParent dist/Macrophone.app "$ZIP"
SHA=$(shasum -a 256 "$ZIP" | cut -d' ' -f1)
echo "▸ $ZIP  sha256=$SHA"

git tag -f "v$VERSION"
git_personal push -q origin main
git_personal push -qf origin "v$VERSION"
gh_personal release create "v$VERSION" "$ZIP" --repo "$REPO" --title "Macrophone $VERSION" \
  --notes "Install: \`brew install --cask $OWNER/tap/macrophone\`" 2>/dev/null \
  || gh_personal release upload "v$VERSION" "$ZIP" --repo "$REPO" --clobber

# Update the cask in the tap.
TAP_DIR=$(mktemp -d)
trap 'rm -rf "$TAP_DIR"' EXIT
git_personal -c credential.helper= -c "credential.helper=!env -u GH_TOKEN -u GITHUB_TOKEN gh auth git-credential" \
  clone -q "https://github.com/$TAP_REPO.git" "$TAP_DIR"
mkdir -p "$TAP_DIR/Casks"
sed -e "s/__VERSION__/$VERSION/" -e "s/__SHA256__/$SHA/" packaging/macrophone.rb > "$TAP_DIR/Casks/macrophone.rb"
(
  cd "$TAP_DIR"
  git config credential.helper ""
  git config --add credential.helper "!env -u GH_TOKEN -u GITHUB_TOKEN gh auth git-credential"
  git add Casks/macrophone.rb
  git commit -qm "macrophone $VERSION" && git_personal push -q origin HEAD || echo "(cask unchanged)"
)

echo "✓ Released $VERSION — brew upgrade --cask macrophone"

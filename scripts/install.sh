#!/usr/bin/env sh
# install.sh — download a released `ski` binary and drop it on a PATH dir that
# scripts/ski-bootstrap.sh resolves (~/.local/bin by default).
#
#   curl -fsSL https://raw.githubusercontent.com/bcmyguest/skill-injector/main/scripts/install.sh | sh
#
# Env knobs:
#   SKI_VERSION   release tag to install (e.g. v0.1.0). Default: latest release.
#   SKI_BIN_DIR   install dir. Default: $HOME/.local/bin
#
# Linux x86_64 only — that is the single platform the release pipeline builds.
# The default binary embeds the ONNX runtime statically; the bge model and
# reranker still download on first run (needs network).
set -eu

REPO="bcmyguest/skill-injector"
TARGET="x86_64-unknown-linux-gnu"
BIN_DIR="${SKI_BIN_DIR:-$HOME/.local/bin}"

err() { echo "install.sh: $*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# --- platform guard --------------------------------------------------------
os="$(uname -s)"; arch="$(uname -m)"
[ "$os" = "Linux" ] || err "unsupported OS '$os' (release builds Linux only); build from source: cargo install --path ."
case "$arch" in
  x86_64|amd64) : ;;
  *) err "unsupported arch '$arch' (release builds x86_64 only); build from source: cargo install --path ." ;;
esac

have tar || err "'tar' is required"
if have curl; then dl() { curl -fsSL "$1" -o "$2"; }
elif have wget; then dl() { wget -qO "$2" "$1"; }
else err "need 'curl' or 'wget'"; fi

# --- resolve version -------------------------------------------------------
tag="${SKI_VERSION:-}"
if [ -z "$tag" ]; then
  api="https://api.github.com/repos/$REPO/releases/latest"
  tag="$(dl "$api" /dev/stdout | grep -m1 '"tag_name"' | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/')"
  [ -n "$tag" ] || err "could not resolve latest release tag; set SKI_VERSION"
fi

asset="ski-${TARGET}.tar.gz"
base="https://github.com/$REPO/releases/download/$tag"
echo "install.sh: installing ski $tag ($TARGET) -> $BIN_DIR"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
dl "$base/$asset" "$tmp/$asset" || err "download failed: $base/$asset"

# --- verify checksum (best effort) ----------------------------------------
if dl "$base/checksums.txt" "$tmp/checksums.txt" 2>/dev/null && have sha256sum; then
  want="$(grep " $asset\$" "$tmp/checksums.txt" | awk '{print $1}')"
  if [ -n "$want" ]; then
    got="$(sha256sum "$tmp/$asset" | awk '{print $1}')"
    [ "$want" = "$got" ] || err "checksum mismatch for $asset (want $want, got $got)"
    echo "install.sh: checksum ok"
  fi
fi

tar -xzf "$tmp/$asset" -C "$tmp"
[ -f "$tmp/ski" ] || err "archive did not contain 'ski'"

mkdir -p "$BIN_DIR"
install -m 755 "$tmp/ski" "$BIN_DIR/ski"
echo "install.sh: installed $BIN_DIR/ski"
"$BIN_DIR/ski" --version 2>/dev/null || true

case ":$PATH:" in
  *":$BIN_DIR:"*) : ;;
  *) echo "install.sh: NOTE add $BIN_DIR to PATH (or rely on ski-bootstrap.sh, which already checks it)" ;;
esac

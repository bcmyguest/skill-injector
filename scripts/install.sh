#!/usr/bin/env sh
# install.sh — download a released `ski` binary and drop it on a PATH dir that
# scripts/ski-bootstrap.sh resolves (~/.local/bin by default).
#
#   curl -fsSL https://raw.githubusercontent.com/bcmyguest/skill-injector/main/scripts/install.sh | sh
#
# Env knobs:
#   SKI_VERSION   release tag to install (e.g. v0.1.0). Default: latest release.
#   SKI_BIN_DIR   install dir. Default: $HOME/.local/bin
#   SKI_HOST      which host to wire after install: claude | opencode | both |
#                 none. Default: auto — wire every host detected on disk.
#   SKI_PREWARM   set to 0 to skip the post-wire model download + index build
#                 (it will then happen on first use instead).
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
SKI="$BIN_DIR/ski"
"$SKI" --version 2>/dev/null || true

case ":$PATH:" in
  *":$BIN_DIR:"*) : ;;
  *) echo "install.sh: NOTE add $BIN_DIR to PATH (or rely on ski-bootstrap.sh, which already checks it)" ;;
esac

# --- wire host adapters ----------------------------------------------------
# Binary alone does nothing; it has to be wired into a host. `ski init -g claude`
# merges the three hooks into ~/.claude/settings.json (the marketplace-free path);
# `ski init -g opencode` drops the bundled plugin into ~/.config/opencode/plugin.
# Both are additive and idempotent — safe to re-run.
wire_claude()   { "$SKI" init -g claude   || echo "install.sh: 'ski init -g claude' failed — wire it manually" >&2; }
wire_opencode() { "$SKI" init -g opencode || echo "install.sh: 'ski init -g opencode' failed — wire it manually" >&2; }

# Pre-warm one host's model cache + index NOW, so the one-time model download
# (~275 MB to ~/.config/ski/models) happens here — visibly, with the user
# watching an installer — instead of silently stalling their first prompt (or
# being killed by the host's hook timeout and re-attempted every prompt).
# Best-effort: an offline install still succeeds, injection self-heals later.
# Skippable with SKI_PREWARM=0. Both hosts share the model cache, so warming is
# once; the second host only needs its own (cheap) index build.
prewarm() {
  [ "${SKI_PREWARM:-1}" = 0 ] && return 0
  echo "install.sh: pre-downloading embedding models + building the skill index (one-time, ~275 MB)..."
  for h in "$@"; do
    "$SKI" index --host "$h" || \
      echo "install.sh: warning: pre-warm for '$h' failed (offline?) — the download will retry on first use" >&2
  done
}

case "${SKI_HOST:-auto}" in
  none)
    echo "install.sh: SKI_HOST=none — skipping host wiring (run 'ski init -g <host>' yourself)"
    ;;
  claude)   wire_claude;   prewarm claude ;;
  opencode) wire_opencode; prewarm opencode ;;
  both)
    wire_claude
    wire_opencode
    prewarm claude opencode
    ;;
  auto)
    hosts=""
    if [ -d "$HOME/.claude" ]; then wire_claude; hosts="claude"; fi
    if [ -d "$HOME/.config/opencode" ]; then wire_opencode; hosts="$hosts opencode"; fi
    if [ -z "$hosts" ]; then
      echo "install.sh: no host found on disk — run 'ski init -g claude' or 'ski init -g opencode' to wire one" >&2
    else
      # shellcheck disable=SC2086
      prewarm $hosts
    fi
    ;;
  *) err "unknown SKI_HOST '${SKI_HOST}' (use: claude | opencode | both | none)" ;;
esac

#!/usr/bin/env bash
# ski-bootstrap.sh — resolve the `ski` binary and run it with the given args,
# passing stdin straight through. The Claude Code adapter for skill-inject calls
# this from every hook (hook / observe / session-start).
#
# FAIL OPEN: if `ski` can't be found we exit 0 with no stdout, so a missing or
# not-yet-built install never blocks a prompt or a tool call. Set SKI_DEBUG=1 to
# get a one-line stderr hint when the binary is absent.
set -euo pipefail

resolve() {
  if command -v ski >/dev/null 2>&1; then
    command -v ski
    return 0
  fi
  for cand in "$HOME/.local/bin/ski" "$HOME/.cargo/bin/ski"; do
    if [[ -x "$cand" ]]; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  return 1
}

if ! bin="$(resolve)"; then
  if [[ -n "${SKI_DEBUG:-}" ]]; then
    echo "ski-bootstrap: 'ski' not found on PATH, ~/.local/bin, or ~/.cargo/bin." >&2
    echo "  install it from the plugin dir:  cargo install --path \"\${CLAUDE_PLUGIN_ROOT:-.}\"" >&2
  fi
  exit 0
fi

# ski owns its own fail-open contract (errors -> empty output, exit 0).
exec "$bin" "$@"

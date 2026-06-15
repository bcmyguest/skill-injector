#!/usr/bin/env bash
# version-sync.sh — assert the ski version is identical across the three
# manifests that must stay in lockstep on every release:
#   Cargo.toml                     -> package.version
#   .claude-plugin/plugin.json     -> .version
#   .claude-plugin/marketplace.json-> .metadata.version
#
# Usage:
#   scripts/version-sync.sh            # just check the three agree
#   scripts/version-sync.sh v0.1.0     # also assert they match this release tag
#                                      # (leading "v" is stripped before compare)
#
# Exit 0 if in lockstep (and, when a tag is given, matching it); 1 otherwise.
set -euo pipefail

cd "$(dirname "$0")/.."

cargo_v=$(grep -m1 '^version = ' Cargo.toml | sed -E 's/^version = "(.*)"/\1/')
plugin_v=$(jq -r '.version' .claude-plugin/plugin.json)
market_v=$(jq -r '.metadata.version' .claude-plugin/marketplace.json)

printf 'Cargo.toml:        %s\n' "$cargo_v"
printf 'plugin.json:       %s\n' "$plugin_v"
printf 'marketplace.json:  %s\n' "$market_v"

rc=0
if [ "$cargo_v" != "$plugin_v" ] || [ "$cargo_v" != "$market_v" ]; then
  echo "ERROR: version mismatch across manifests" >&2
  rc=1
fi

if [ "$#" -ge 1 ]; then
  tag_v="${1#v}"
  printf 'release tag:       %s\n' "$tag_v"
  if [ "$cargo_v" != "$tag_v" ]; then
    echo "ERROR: manifest version ($cargo_v) != release tag ($tag_v)" >&2
    rc=1
  fi
fi

if [ "$rc" -eq 0 ]; then
  echo "OK: versions in lockstep ($cargo_v)"
fi
exit "$rc"

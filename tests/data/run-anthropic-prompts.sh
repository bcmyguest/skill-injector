#!/usr/bin/env bash
# Run the anthropic/skills prompt corpus through ski's ranker and report
# top-1 accuracy. Read-only: builds nothing on disk, indexes in-memory per call.
#
#   ./run-anthropic-prompts.sh            # scoped: rank only against the 17 skills
#   ./run-anthropic-prompts.sh -g         # global: compete against every installed skill
#   ./run-anthropic-prompts.sh -v         # also print each predicted skill + score
#   PROMPTS=other.tsv ./run-anthropic-prompts.sh
#
# Scoped mode points ski at the anthropic-agent-skills marketplace via SKI_ROOTS,
# measuring how well the ranker discriminates *within* that library. Global mode
# uses ski's default roots, measuring real injection behaviour with every other
# installed skill as a distractor. Each prompt is scored with
# `ski why "<prompt>" --top 1`; a `*` mark means the top hit cleared
# min_similarity. Negatives pass when the top hit is UNstarred.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
crate="$(cd "$here/../.." && pwd)"
prompts="${PROMPTS:-$here/anthropic_skills_prompts.tsv}"
verbose=0 global=0
for a in "$@"; do
  case "$a" in
    -v) verbose=1 ;;
    -g|--global) global=1 ;;
  esac
done

anthropic_root="$HOME/.claude/plugins/marketplaces/anthropic-agent-skills"
if [[ "$global" == 0 ]]; then
  if [[ ! -d "$anthropic_root" ]]; then
    echo "anthropic-agent-skills not installed at $anthropic_root" >&2
    echo "install it, or run with -g to compete against your default roots." >&2
    exit 2
  fi
  export SKI_ROOTS="$anthropic_root"
  echo "scope: $anthropic_root (17-skill library only)"
else
  unset SKI_ROOTS || true
  echo "scope: ski default roots (global — every installed skill competes)"
fi

cargo="$(command -v cargo || echo "$HOME/.cargo/bin/cargo")"
# SKI_FEATURES=fastembed exercises the real bge embedder; unset uses bag-of-words.
feat_args=()
[[ -n "${SKI_FEATURES:-}" ]] && feat_args=(--features "$SKI_FEATURES")
ski() { ( cd "$crate" && "$cargo" run -q "${feat_args[@]}" -- "$@" ); }

# One warm-up build so `cargo run` chatter doesn't pollute the first result.
ski why warmup --top 1 >/dev/null 2>&1 || true

pass=0 fail=0 total=0
declare -a failures=()

while IFS=$'\t' read -r want kind prompt; do
  [[ -z "${want:-}" || "${want:0:1}" == "#" ]] && continue
  total=$((total + 1))

  # why output: header line, then hit lines "<mark> <name> score <n> ...".
  # mark is exactly one char (* or space) at col 0, space at col 1, name at col 2+.
  line="$(ski why "$prompt" --top 1 2>/dev/null | grep ' score ' | head -n1)"
  mark="${line:0:1}"
  rest="${line:2}"
  got="${rest%% *}"
  score="$(sed -n 's/.* score \(-\{0,1\}[0-9.]*\).*/\1/p' <<<"$line")"

  if [[ "$want" == "(none)" ]]; then
    # negative: pass when nothing cleared the threshold (top hit unstarred)
    if [[ "$mark" != "*" ]]; then ok=1; else ok=0; fi
  else
    if [[ "$mark" == "*" && "$got" == "$want" ]]; then ok=1; else ok=0; fi
  fi

  if [[ "$ok" == 1 ]]; then
    pass=$((pass + 1))
    [[ "$verbose" == 1 ]] && printf 'PASS [%-10s] %-22s %s\n' "$kind" "$got" "$prompt"
  else
    fail=$((fail + 1))
    failures+=("$(printf '[%-10s] want=%-22s got=%-22s (%s*=%s) :: %s' \
      "$kind" "$want" "${got:-<none>}" "${score:-?}" "$mark" "$prompt")")
  fi
done < "$prompts"

echo
echo "=== ski vs anthropic/skills =================================="
printf 'total %d   pass %d   fail %d   accuracy %d%%\n' \
  "$total" "$pass" "$fail" $(( total ? pass * 100 / total : 0 ))
if ((fail > 0)); then
  echo "--- misses ---------------------------------------------------"
  printf '%s\n' "${failures[@]}"
fi
exit 0

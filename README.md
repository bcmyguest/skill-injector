# skill-inject (`ski`)

Local, model-agnostic **automatic skill injection** for [Claude Code](https://docs.claude.com/en/docs/claude-code)
and [opencode](https://opencode.ai).

## Why this exists

Agent skill systems advertise every skill's description to the model and trust it to
notice the relevant one and invoke it. In practice that leaks:

- **Missed triggers.** When a skill's description shares no vocabulary with your prompt,
  the model overlooks it — the match is *semantic*, not lexical. The more skills you
  install, the more often the right one hides in the crowd.
- **Model-dependent.** "Pick the skill from a wall of descriptions" is something even
  frontier models do unevenly, and smaller or local models do worse. Which skill fires
  drifts with whatever model is driving the session.
- **Always-on cost.** Every description sits in the context window every turn, relevant
  or not.

`ski` replaces the guesswork with a deterministic local retriever: it embeds your prompt
on CPU, ranks it against your skill descriptions, and injects the matched skill **only
when one is genuinely relevant** — the same result no matter which model runs, with no
API call and nothing leaving your machine. The model still chooses which *files* a skill
points to; `ski` only guarantees the right skill is **in the room** when it matters.
Skills the model loads on its own are tracked and never re-injected.

### See it decide

`ski` scores every installed skill against your prompt and injects **only** the ones
above a fixed cutoff (`-2.50` below); a higher score is a stronger match. Real `ski why`
output against a live library of 57 skills (reproduce with `ski index` then
`ski why "<prompt>"`):

```text
$ ski why "clean up this messy CSV"
  xlsx              -0.59   <- injected (clear winner)
  pre-commit-setup  -3.72   <- skipped
```

`clean up this messy CSV` never says *spreadsheet* or *xlsx* — the match is on *meaning*,
not vocabulary, and it lands far ahead of every other skill. Keyword or description
matching can't bridge that gap, and a model scanning 57 descriptions can easily miss it.

```text
$ ski why "what time is the meeting tomorrow"
  handoff          -3.08   <- best skill, still below the cutoff
```

An off-topic prompt leaves every skill under the cutoff, so `ski` injects nothing — no
false positives, no context pollution.

This repo is a single Rust binary (`ski`) plus the thin host adapters that drive it,
packaged as a one-plugin Claude Code marketplace. See [DEVELOPING.md](./DEVELOPING.md)
for the dev workflow.

## Benchmarks

**100% local** — no API call, no token cost, nothing leaves your machine. The whole
pipeline (embed → retrieve → rerank) runs on CPU — around **half a second per prompt** on
the machine benchmarked below. Real samples, ranked against a live library of 57 skills:

| your prompt | skill `ski` injects | match score |
|---|---|---|
| `set up a python project with uv` | `uv-setup` | 2.76 |
| `scaffold a new react typescript frontend` | `react-ts-setup` | 3.38 |
| `how do I credit Claude in this git commit` | `git-attribution` | 1.21 |
| `make an animated gif for slack` | `slack-gif-creator` | 1.63 |
| `write a Word doc with a table of contents` | `docx` | 0.12 |
| `extract tables from a pdf` | `pdf` | 0.67 |

Every row is a real `ski why` result. A higher **match score** means a stronger match;
anything below `-2.50` is left out entirely (as in the off-topic example above).

| operation (cold — every hook is a fresh process) | time |
|---|---|
| rank + inject one prompt (`ski hook`) | **~0.54 s** median |
| full index rebuild (57 skills) | ~0.78 s |
| incremental reindex, no change | ~0.19 s |

`bge-small-en-v1.5` (384-dim) retrieval + `jina-reranker-v1-turbo-en` rerank, ~270 MB RAM.
Measured CPU-only on an AMD Ryzen AI MAX+ 395 — cold runs with model load included,
not warm microbenchmarks. Reproduce with `ski index` then `ski why "<your prompt>"`.

## Install (Claude Code)

The plugin is hooks-only and needs the `ski` binary on disk.

1. **Get the binary.** Easiest — install the latest prebuilt release into
   `~/.local/bin` (Linux x86_64):

   ```sh
   curl -fsSL https://raw.githubusercontent.com/bcmyguest/skill-injector/main/scripts/install.sh | sh
   ```

   `.deb` / `.rpm` packages are on the [Releases](https://github.com/bcmyguest/skill-injector/releases)
   page too. Or build from source — default build = real embedder + reranker
   (downloads the model once, then offline):

   ```sh
   cargo install --path .            # -> ~/.cargo/bin/ski
   ```

   Or the offline bag-of-words build (no deps, no model download):

   ```sh
   cargo install --path . --no-default-features
   ```

2. **Enable the plugin** from this marketplace:

   ```sh
   /plugin marketplace add bcmyguest/skill-injector   # or a local path to this repo
   /plugin install skill-inject@skill-inject
   ```

   **Can't use the marketplace?** `ski init -g claude` merges the same three hooks
   straight into `~/.claude/settings.json` (backing up any existing file to
   `settings.json.bak` first). It's additive and idempotent — it won't touch unrelated
   settings or double-wire a hook already present.

`hooks/hooks.json` wires three hooks through `scripts/ski-bootstrap.sh`, which resolves
`ski` from `PATH`, then `~/.local/bin`, then `~/.cargo/bin`:

| hook | matcher | command |
|---|---|---|
| `UserPromptSubmit` | — | `ski hook --host claude` (rank + inject) |
| `PostToolUse` | `Read\|Skill` | `ski observe --host claude` (record model-loaded skills) |
| `SessionStart` | `startup\|resume\|compact` | `ski session-start --host claude` (reindex; re-arm on compact) |

If `ski` isn't found, the bootstrap exits 0 with no output — a missing build never blocks
a prompt. Set `SKI_DEBUG=1` for an install hint on stderr.

For **opencode**, see [opencode/README.md](./opencode/README.md).

## Usage

```sh
ski init -g claude            # wire ski's hooks into ~/.claude/settings.json (or opencode)
ski index                     # build the index at $XDG_DATA_HOME/ski/index.json
ski why "credit Claude in this commit" --top 5   # ranked skills + scores (tuning aid)

# Hook hot-path (stdin event -> injection JSON on stdout):
echo '{"session_id":"s1","cwd":".","prompt":"credit Claude in this commit"}' \
  | ski hook --host claude     # -> {"hookSpecificOutput":{...,"additionalContext":...}}
echo '{"session_id":"s1","cwd":".","prompt":"set up a python project"}' \
  | ski hook --host opencode   # -> {"skills":[...],"inject":"..."}
```

The dedup ledger lives at `$XDG_STATE_HOME/ski/sessions/<session_id>.json`. The index is
per-host (Claude `index.json`, opencode `index-opencode.json`) so the two never clobber.
Downloaded embedder/reranker models cache once at `$XDG_CONFIG_HOME/ski/models` (default
`~/.config/ski/models`) — never in the working directory.
`SKI_ROOTS` (colon-separated) overrides the skill-discovery roots for both hosts.

## How it works

```
prompt ─▶ adapter (Claude hook / opencode plugin) ─▶ ski (Rust, one binary)
                                                        1. load index (skill vectors)
                                                        2. embed(prompt) locally
                                                        3. retrieve: cosine top-K (bge)
                                                        4. rerank: cross-encoder (JINA turbo)
                                                        5. gate: threshold + margin + deny/force
                                                        6. dedup vs per-session ledger
                                                        7. emit injection ─▶ adapter injects as context
```

- **Two-stage ranking.** A bge-small bi-encoder retrieves a candidate set; a JINA-turbo
  cross-encoder reranks it. Cheap O(1) query + cached vectors first, expensive pairwise
  scoring only on the short list. (Why not reranker-only: it's O(N) per prompt and loses
  the cosine early-out.)
- **Per-session dedup.** A skill injected by `ski` *or* loaded by the model itself is
  recorded in a session ledger and never re-injected — until compaction re-arms it.
- **Fail-open everywhere.** Bad stdin, a missing index, any IO error → no output, exit 0.
  A ranking problem never blocks your prompt.

## Embedding backends

- **Default (`fastembed`):** real embeddings via fastembed (ONNX). Retrieval with
  `bge-small-en-v1.5` (the query gets bge's retrieval-instruction prefix; descriptions
  don't), reranking with JINA turbo. `all-MiniLM-L6-v2-q` is the low-RAM alternative.
  Models download once and cache at `$XDG_CONFIG_HOME/ski/models` (default
  `~/.config/ski/models`).
- **`--no-default-features` (offline):** deterministic hashed bag-of-words. No deps, no
  network, no model — surface-token matching plus the keyword boost. Used for tests and
  as the fallback when no recognized model is configured.

The index is tagged with the embedder id, so switching backends/models triggers a full
reindex automatically.

## Build, test, lint

```sh
cargo build --release                # default: real embedder + reranker
cargo build --no-default-features    # offline: bag-of-words, no model download
cargo test --no-default-features     # unit + golden tests (offline, network-free)

cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
```

Golden tests run against the self-contained fixtures in
[`tests/fixtures/skills/`](tests/fixtures/skills) — they depend on nothing outside this
repo. `fmt` / `clippy` / `test` are also wired as `pre-commit` hooks
(`.pre-commit-config.yaml`).

## License

[GNU AGPL-3.0-or-later](LICENSE). Copyright (c) 2026 ski contributors. If you run a
modified version — including over a network — you must release your source under the
same terms.

**No-AI-training request (non-binding):** the AGPL governs your legal rights, but the
authors additionally ask that this project, in whole or in part, not be used as training,
fine-tuning, or evaluation data for machine-learning or AI systems.

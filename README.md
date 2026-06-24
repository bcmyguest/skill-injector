# skill-inject (`ski`)

Local, model-agnostic **automatic skill injection** for [Claude Code](https://docs.claude.com/en/docs/claude-code)
and [opencode](https://opencode.ai).

> A strong model often **won't use a skill it should** — on indirect prompts it
> hand-rolls the task instead of invoking the skill built for it. `ski` is a local,
> deterministic nudge that surfaces the right skill so the model actually reaches for it.

## Why this exists

Skill systems dump every skill's description into the model's context and hope it picks
the right one. That works until it doesn't:

- **The model skips skills it should use.** On indirect prompts — "clean up this messy
  CSV", "match our brand" — a capable host often just does the task by hand instead of
  reaching for the skill built for it.
- **It gets worse with more skills, and worse with weaker models.** Picking one skill out
  of a wall of descriptions is hard, and which one fires drifts with whatever model is
  driving the session.
- **Every description costs context, every turn** — relevant or not.

`ski` does the picking for you. It embeds your prompt on CPU, ranks it against your skill
descriptions, and injects the matching skill **only when one actually fits** — same
result on any model, no API call, nothing leaving your machine. The model still decides
what to *do* with the skill; `ski` just makes sure the right one is in the room. Skills
the model finds on its own are tracked and never injected twice.

### See it decide

`ski` scores every installed skill against your prompt and injects **only** the ones
above a fixed cutoff (`-2.50` below); a higher score is a stronger match. Real `ski why`
output against a live library of 57 skills:

```text
$ ski why "clean up this messy CSV"
  xlsx              -0.59   <- injected (clear winner)
  pre-commit-setup  -3.72   <- skipped
```

`clean up this messy CSV` never says *spreadsheet* or *xlsx* — the match is on *meaning*,
not vocabulary, and it lands far ahead of every other skill. Keyword or description
matching can't bridge that gap, and a model scanning 57 descriptions can easily miss it.

`ski` deliberately errs toward over-sending — a borderline skill is injected rather than
withheld, since a strong host simply ignores a skill it doesn't need but can't use one it
never saw, and when `ski` does inject it asks firmly ("invoke it now") rather than hedging.

```text
$ ski why "what time is the meeting tomorrow"
  handoff          -3.08   <- best skill, still below the cutoff
```

An off-topic prompt leaves every skill under the cutoff, so `ski` injects nothing — no
false positives, no context pollution.

This repo is a single Rust binary (`ski`) plus the thin host adapters that drive it,
packaged as a one-plugin Claude Code marketplace. See [DEVELOPING.md](./DEVELOPING.md)
for the dev workflow.

## Speed and examples

**Fast and entirely local** — no API call, no token cost, nothing leaves your machine.
The whole pipeline (embed → retrieve → rerank) runs on CPU in about **half a second per
prompt**:

| operation (cold — every hook is a fresh process) | time |
|---|---|
| rank + inject one prompt (`ski hook`) | **~0.61 s** median |
| full index rebuild (57 skills) | ~0.73 s |
| incremental reindex, no change | ~0.19 s |

`bge-small-en-v1.5` (384-dim) retrieval + `jina-reranker-v1-turbo-en` rerank, ~270 MB RAM.
Measured CPU-only on an AMD Ryzen AI MAX+ 395 — cold runs with model load included,
not warm microbenchmarks.

**It matches on meaning, not keywords.** Every row is a real `ski why` result against a
live library of 57 skills; a higher match score is a stronger match, and anything below
`-2.50` is left out entirely (as in the off-topic example above):

| your prompt | skill `ski` injects | match score |
|---|---|---|
| `set up a python project with uv` | `uv-setup` | 2.76 |
| `scaffold a new react typescript frontend` | `react-ts-setup` | 3.38 |
| `how do I credit Claude in this git commit` | `git-attribution` | 1.21 |
| `make an animated gif for slack` | `slack-gif-creator` | 1.63 |
| `write a Word doc with a table of contents` | `docx` | 0.12 |
| `extract tables from a pdf` | `pdf` | 0.67 |

Reproduce any of it with `ski index` then `ski why "<your prompt>"`.

## Install

One command installs the prebuilt binary into `~/.local/bin` (Linux x86_64) and wires
**every host it finds on disk** — Claude Code *and* opencode:

```sh
curl -fsSL https://raw.githubusercontent.com/bcmyguest/skill-injector/main/scripts/install.sh | sh
```

It auto-detects hosts (`~/.claude` → Claude hooks in `settings.json`; `~/.config/opencode`
→ the opencode plugin). Pin one with `SKI_HOST=claude|opencode|both|none`. The host wiring
is additive and idempotent — re-running is safe, and any existing Claude `settings.json`
is backed up to `settings.json.bak` first.

On its **first run** the prebuilt binary downloads its embedder + reranker weights
(~275 MB, once) to `~/.config/ski/models`; that first prompt blocks while the download
happens, and every run after is fully offline. (The `--no-default-features` build skips
this — it uses the bundled bag-of-words embedder and never touches the network.)

`.deb` / `.rpm` packages are on the [Releases](https://github.com/bcmyguest/skill-injector/releases)
page. To build from source instead (default build = real embedder + reranker, downloads
the model once then runs offline), then wire the host yourself:

```sh
cargo install --path .            # -> ~/.cargo/bin/ski (real embedder + reranker)
# ...or the offline bag-of-words build, no deps or model download:
# cargo install --path . --no-default-features
ski init -g claude               # then wire the host (or: opencode)
```

**Prefer the Claude marketplace?** Skip the host-wiring step and enable the plugin instead:

```sh
/plugin marketplace add bcmyguest/skill-injector   # or a local path to this repo
/plugin install skill-inject@skill-inject
```

However you install on Claude, three hooks get wired (`hooks/hooks.json` runs them through
`scripts/ski-bootstrap.sh`, which resolves `ski` from `PATH`, then `~/.local/bin`, then
`~/.cargo/bin`):

| hook | matcher | command |
|---|---|---|
| `UserPromptSubmit` | — | `ski hook --host claude` (rank + inject) |
| `PostToolUse` | `Read\|Skill` | `ski observe --host claude` (record model-loaded skills) |
| `SessionStart` | `startup\|resume\|compact` | `ski session-start --host claude` (reindex; re-arm on compact) |

If `ski` isn't found, the bootstrap exits 0 with no output — a missing build never blocks
a prompt. Set `SKI_DEBUG=1` for an install hint on stderr.

For opencode specifics (skill roots, the plugin event map), see
[opencode/README.md](./opencode/README.md).

## Usage

```sh
ski init -g claude            # wire ski's hooks into ~/.claude/settings.json (or opencode)
ski index                     # build the index at $XDG_DATA_HOME/ski/index.json
ski why "credit Claude in this commit" --top 5   # ranked skills + scores (tuning aid)

# Telemetry readout (needs telemetry = true, or SKI_TELEMETRY=1, while hooks ran):
ski history                   # aggregate: recommended vs. actually-used, top false positives/misses
ski history --tail 20         # last 20 events interleaved: recommendations (prompt + per-candidate
                              #   confidence + used?) and self-loads (acted-on-rec vs. RECALL MISS + prompt)
ski history --tail 20 --session conv-abc   # ...filtered to one conversation
ski clear                     # re-arm injection (wipe per-session dedup); --telemetry also wipes the log

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

## Configuration

Everything works with no config. An optional `~/.config/ski/config.toml`
(`$XDG_CONFIG_HOME/ski/config.toml`) overrides the compiled defaults — every key is
optional, and a missing or malformed file is ignored (fail open, never blocks a prompt).
The most-reached-for key is `deny`, to silence a skill that keeps surfacing.

```toml
# Silence / force specific skills (by their `name`):
deny  = ["example-skill"]   # never auto-injected
force = []                  # injected whenever a keyword hits, even below threshold

max_skills  = 2             # max skills injected per prompt
char_budget = 6000          # max total injected characters

# Reranker gate — JINA cross-encoder logits, where ~0 is the relevant/irrelevant
# boundary. Raise toward 0 to inject less; lower to inject more.
rerank_min = -2.5

# Opt-in JSONL telemetry (recommend/use events) for `ski history`; off by default.
# Equivalent to setting the SKI_TELEMETRY env var.
telemetry = false

# Stage-1 cosine thresholds. Normally left to per-embedder calibration; pin to override.
# min_similarity = 0.30
# score_margin   = 0.15

# model              = "bge-small-en-v1.5"   # the default; alts: "all-MiniLM-L6-v2-q", "bge-base-en-v1.5"
# inject_mode        = "directive"           # or "body"
# directive_strength = "auto"                # auto | soft | hard
# roots              = ["/abs/path/to/skills"]  # discovery roots; not tilde-expanded,
#                                               # and the SKI_ROOTS env var still wins
```

Advanced ranking knobs are also accepted: `keyword_boost`, `recall_floor`, `high_conf`,
`clear_gap`, `rerank_top_k`, `rerank_margin` (see `src/config.rs` for what each gates).

## How it works

```
prompt ─▶ adapter (Claude hook / opencode plugin) ─▶ ski (Rust, one binary)
                                                        1. prefilter (skip control payloads)
                                                        2. load index (skill vectors)
                                                        3. embed(prompt) locally
                                                        4. retrieve: cosine top-K (bge)
                                                        5. rerank: cross-encoder (JINA turbo)
                                                        6. gate: threshold + margin + deny/force + slash self-rec
                                                        7. dedup vs per-session ledger
                                                        8. emit injection ─▶ adapter injects as context
```

- **Two-stage ranking.** A bge-small bi-encoder retrieves a candidate set; a JINA-turbo
  cross-encoder reranks it. Cheap O(1) query + cached vectors first, expensive pairwise
  scoring only on the short list. (Why not reranker-only: it's O(N) per prompt and loses
  the cosine early-out.)
- **Prompt prefilter.** Host-generated control payloads (`<task-notification>`,
  `<system-reminder>` blocks) aren't user requests, so they skip injection outright
  rather than embedding into noise matches. A `/<name>` slash invocation is an explicit
  skill choice, so the skill it names is never recommended back — that self-recommendation
  was the single largest false positive in `ski history`.
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

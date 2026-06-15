# skill-inject (`ski`)

Local, model-agnostic **automatic skill injection** for [Claude Code](https://docs.claude.com/en/docs/claude-code)
and [opencode](https://opencode.ai). A hook embeds your prompt **locally**, ranks it
against your installed skill descriptions, and — when one is relevant — injects that
skill into context.

The model still chooses which *files* a skill points to; `ski` only guarantees the
right skill is **considered** when it matters. Skills the model loads on its own are
tracked and never re-injected.

This repo is a single Rust binary (`ski`) plus the thin host adapters that drive it,
packaged as a one-plugin Claude Code marketplace. See [PLAN.md](./PLAN.md) for the
full design and [DEVELOPING.md](./DEVELOPING.md) for the dev workflow.

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
  the cosine early-out — see PLAN.md.)
- **Per-session dedup.** A skill injected by `ski` *or* loaded by the model itself is
  recorded in a session ledger and never re-injected — until compaction re-arms it.
- **Fail-open everywhere.** Bad stdin, a missing index, any IO error → no output, exit 0.
  A ranking problem never blocks your prompt.

## Install (Claude Code)

The plugin is hooks-only and needs the `ski` binary on disk.

1. **Build the binary.** Default build = real embedder + reranker (downloads the
   model once, then offline):

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
`SKI_ROOTS` (colon-separated) overrides the skill-discovery roots for both hosts.

## Embedding backends

- **Default (`fastembed`):** real embeddings via fastembed (ONNX). Retrieval with
  `bge-small-en-v1.5` (the query gets bge's retrieval-instruction prefix; descriptions
  don't), reranking with JINA turbo. `all-MiniLM-L6-v2-q` is the low-RAM alternative.
  Models download once and cache.
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

See [LICENSE](LICENSE). All rights reserved; in particular, the contents may **not** be
used as training, fine-tuning, or evaluation data for machine-learning or AI systems.

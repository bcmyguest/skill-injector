# Developing `ski`

How to set up, build, lint, and test the `ski` crate (the engine behind
skill-inject). For the design, see [PLAN.md](./PLAN.md); for a usage overview, see
[README.md](./README.md).

## 1. Toolchain

`ski` targets stable Rust (developed on 1.96). Install via [rustup](https://rustup.rs)
and add the lint components:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # if you don't have rustup
rustup toolchain install stable
rustup component add clippy rustfmt
```

`pre-commit` runs the lint/test gate (see §5):

```sh
pipx install pre-commit        # or: uv tool install pre-commit / pip install pre-commit
```

Everything below runs from the repo root (the crate dir).

## 2. Layout

```
src/
  main.rs        # clap CLI: index / why / hook / observe / session-start
  lib.rs         # module wiring
  observe.rs     # PostToolUse: record model-loaded skills (Read SKILL.md / Skill)
  session_start.rs # SessionStart: incremental reindex + re-arm on compaction
  config.rs      # Config + defaults (model, thresholds, roots)
  text.rs        # tokenize + FNV hashes (deterministic — see "Determinism")
  skill.rs       # SKILL.md discovery + frontmatter parse
  embed/
    mod.rs       # Embedder trait + EmbedKind + build() backend selector
    bow.rs       # offline bag-of-words backend (`--no-default-features`)
    fast.rs      # fastembed (bge/MiniLM) backend — the default `fastembed` feature
  index.rs       # persisted, incrementally-refreshed embedding index
  rank.rs        # cosine + keyword hybrid scoring; select() guardrails
tests/
  golden.rs      # prompt -> expected skill, against the real repo skills
```

## 3. Build & run

```sh
cargo build --release            # default: real embedder + reranker (model downloads once)
cargo run -- index               # build index at $XDG_DATA_HOME/ski/index.json
cargo run -- why "set up pre-commit hooks" --top 5   # rank + scores (tuning aid)
```

Offline bag-of-words build (no deps, no model/network):

```sh
cargo build --no-default-features
```

## 4. Test

```sh
cargo test --no-default-features  # unit + golden tests (offline, network-free)
cargo test                        # default: same, against the real bge model (network on first run)
```

- **Unit tests** live next to the code (`#[cfg(test)] mod tests`): tokenizer,
  hashing, bow determinism/normalization, cosine, keyword scoring, frontmatter.
- **Golden tests** (`tests/golden.rs`) discover the self-contained fixture skills
  under `tests/fixtures/skills/` and assert the top-ranked skill for representative
  prompts (e.g. `"bootstrap a new python project with uv"` → `uv-setup`). Run them with
  `--no-default-features` to use the offline bag-of-words backend (no model, network-free,
  deterministic); they depend on nothing outside this repo. When you add or rename a
  fixture, update the golden cases; use
  `cargo run -- why "<prompt>"` to pick prompts and read the scores.

## 5. Lint & format

```sh
cargo fmt --all                       # apply formatting
cargo fmt --all -- --check            # CI check (no changes)
cargo clippy --all-targets -- -D warnings
```

Install the git hooks so this runs automatically on commit (config lives at the repo
root, scoped to `*.rs`):

```sh
pre-commit install
pre-commit run --all-files            # run on demand
```

The hooks are `ski-fmt`, `ski-clippy`, `ski-test`. They pin `--no-default-features`
(offline, no model download); run the default `fastembed` lane separately in CI.

## 6. Adding an embedding backend

1. Implement `embed::Embedder` (`id()` + `embed(texts, kind)`); honor `EmbedKind`
   if the model is asymmetric (bge prefixes queries, not documents).
2. Register it in `embed::build()` so a config `model` id selects it.
3. Pick a unique `id()` — the index is tagged with it, so a change forces a full,
   automatic reindex. Don't reuse another backend's tag.

## 7. Determinism (important)

Persisted embeddings and content hashes must reproduce byte-for-byte across runs
and machines. Use the fixed FNV hashes in `text.rs` — **never** `std`'s
`DefaultHasher`/`RandomState` (seeded/unspecified) for anything that lands in the
index or on disk.

## 8. Conventions

- Keep an offline, dependency-light lane behind `--no-default-features` (bag-of-words,
  no ONNX/network) — it's what the tests and pre-commit hooks run. Heavy deps (ONNX) stay
  gated behind the `fastembed` feature, which is now on by default.
- Fail open on the hook path: a ranking error must never block the user's prompt.
- Match the surrounding style; `cargo fmt` + clippy `-D warnings` is the bar.

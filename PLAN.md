# skill-inject — Plan

> Working name: **skill-inject** (binary `ski`). Name is provisional — see Open Questions.

Local-first, model-agnostic **automatic skill injection** for Claude Code and
opencode. A hook embeds the user prompt locally, ranks it against skill
descriptions, and — when a skill is relevant — either **tells the model to load it**
(`directive` mode, v1 default) or **injects the `SKILL.md` body directly**
(`body` mode), with **no cloud call for the decision**. We also track which skills
the **model loaded on its own** so we never re-inject a choice it already made. The
model still chooses which *files* a skill points to; we only guarantee the skill
itself is considered when relevant.

---

## 1. Goal / Non-goals

### Goal
- Make relevant skills appear in context **automatically**, driven by a **local**
  semantic match (no API call to Claude/Haiku for the decision).
- **Respect the model's own choices:** track skills the model already loaded
  (native chooser / `use_skill`) and never re-inject them (§4.8).
- Work for both Claude Code (strong native chooser, used as fallback/refinement)
  and opencode with **local models** (weak chooser — primary motivation).
- One ranking engine, two thin adapters. Single offline binary.
- rtk-style setup: `ski init -g` / `ski init -g opencode`.

### Non-goals
- Not a vector DB / RAG-over-codebase tool. N(skills) is tiny (10s–100s); cosine
  over an in-memory array beats running qdrant **or chroma** — both add a
  Python/server dependency that breaks the single-binary, offline design. If we ever
  outgrow in-memory (1000s+ skills), step to an **embedded Rust ANN**
  (`usearch` / `hnsw_rs`): in-process, no server — still not a DB. (See [[2-background]].)
- Not replacing progressive disclosure — augmenting it for weak choosers.
- Not authoring skills, not editing skill files.

---

## 2. Background — what exists, and the gap {#2-background}

| Project | Decision | Injection | Gap vs. us |
|---|---|---|---|
| Claude Code skills (native) | model reads always-present `name`+`description`, decides | model self-loads `SKILL.md` | reliable ~20% w/o nudge; needs the model to decide |
| `joshuadavidthomas/opencode-agent-skills` | local semantic *similarity* | injects `<available-skills>` + **prompt encouraging** agent to call `use_skill` | still routes through a **tool call + model agency** — not automatic content injection |
| `jefflester/claude-skills-supercharged` | **Haiku** intent analysis + keyword fallback | `UserPromptSubmit` injects skill, once/conversation, 1h intent cache | decision uses a **cloud model**; we want it fully local |

**Our differentiator:** local embedding decision **+ direct content injection** (no
`use_skill`, no model agency in the loop). Borrow the good parts: supercharged's
inject-once tracking and intent cache; opencode-agent-skills' SDK injection
plumbing.

---

## 3. Architecture

```
                       ┌─────────────────────────────┐
   user prompt ───────▶│  ADAPTER (per host)         │
                       │   • Claude: UserPromptSubmit │
                       │     hook  → binary (stdin)   │
                       │   • opencode: TS plugin →     │
                       │     binary (spawn)           │
                       └──────────────┬──────────────┘
                                      │ JSON {prompt, session_id, cwd, host}
                                      ▼
                       ┌─────────────────────────────┐
                       │  ski core (Rust, 1 binary)   │
                       │                              │
                       │  1. load INDEX (skill vecs)  │◀── INDEX cache (persistent,
                       │  2. embed(prompt)  fastembed │      invalidated by mtime/hash)
                       │  3. hybrid score:            │
                       │       cosine + keyword boost │
                       │  4. threshold + top-K        │
                       │  5. session dedup            │◀── SESSION state (per session_id)
                       │  6. read SKILL.md bodies     │
                       │  7. emit injection JSON      │──▶ DECISION cache (TTL, optional)
                       └──────────────┬──────────────┘
                                      │ JSON {inject: "<text>", skills:[...]}
                                      ▼
                       ADAPTER injects as context (additionalContext / synthetic msg)
```

**Why Rust single binary:** matches rtk; `fastembed-rs` (ONNX) gives all-MiniLM
embeddings with no Python/venv and no running service. The same binary is the hook,
the indexer, and the installer. opencode's TS plugin just spawns it.

---

## 4. Components

### 4.1 Core binary `ski`
Subcommands:
- `ski init [-g] [opencode]` — install/configure host(s), download model, build index. (§9)
- `ski index [--rebuild]` — discover skills, embed descriptions, write index. Incremental by file hash. **Auto-runs on every SessionStart** (incremental, hash-gated → cheap) since skills drift over time.
- `ski hook --host <claude|opencode>` — read hook event JSON on **stdin**, write injection JSON on **stdout**. The hot path.
- `ski observe --host <claude|opencode>` — record skills the **model** loaded itself (§4.8) into session state, so we don't re-inject them.
- `ski session-start --host <claude|opencode>` — incremental reindex + re-arm session state (clear `loaded` on compaction).
- `ski why <prompt>` — debug: print ranked skills + scores (no injection). Tuning aid.
- `ski stats` — optional later: injection counts / hit rate (rtk-`gain` vibe).

### 4.2 Skill discovery
Scan roots, parse `SKILL.md` YAML frontmatter (`name`, `description`, optional
`keywords`/`aliases`). Roots:
- Claude: `~/.claude/skills/`, plugin skills `~/.claude/plugins/**/skills/`, project `.claude/skills/`.
- opencode: equivalent skill dirs + project.
- Honor a config `extra_roots`.

### 4.3 Embedding
`fastembed-rs` (ONNX via `ort`; optional CUDA exec-provider on GPU boxes).
**Default: `bge-small-en-v1.5`** (384-dim, stronger recall — sensible default given
GPU / ample RAM, e.g. the Strix Halo box). **Lite alt: `all-MiniLM-L6-v2-q`**
(`model = "..."` or `ski init --lite`) for low-RAM / CPU-only machines. Both 384-dim;
the index is **model-tagged**, so switching model forces a full reindex.

bge is **asymmetric**: prefix the *query* (prompt) with bge's retrieval instruction
(`"Represent this sentence for searching relevant passages: "`) but embed skill
descriptions **without** prefix. (MiniLM is symmetric — no prefix either side.)

Model downloaded on `init` → `~/.local/share/ski/models/`, then fully offline.
Optional `--features embed-model` `include_bytes!` build for a zero-download binary.

### 4.4 Index store
`~/.local/share/ski/index.json` (+ optional `vectors.bin` of packed f32):
```jsonc
{
  "model": "all-MiniLM-L6-v2-q",
  "dim": 384,
  "skills": [
    {
      "id": "git-tools/git-attribution",
      "name": "git-attribution",
      "description": "Kernel-style AI commit attribution ...",
      "path": "/home/b/.claude/plugins/.../SKILL.md",
      "host": "claude",
      "keywords": ["commit", "attribution", "assisted-by"],
      "hash": "sha256:...",            // SKILL.md content hash → cache invalidation
      "embedding": [0.013, -0.21, ...] // 384 f32 of the *description*
    }
  ]
}
```
Reindex only entries whose `hash` changed. (Index = the **big cache**; computed once.)

### 4.5 Decision pipeline (the hot path)
1. Parse stdin event → `{prompt, session_id, cwd, host}`.
2. Load index; if any skill-file mtime newer than index, incremental reindex.
3. `q = embed(prompt)`.
4. **Hybrid score** per skill:
   `score = cosine(q, skill.embedding) + Σ keyword_boost(prompt, skill.keywords)`
   Exact keyword/alias substring → fixed boost (or `force` flag → always inject).
5. Keep `score ≥ min_similarity`, sort desc, cap `max_skills`, respect `char_budget`.
6. **Session dedup:** drop skills already in this session's state — whether **we**
   injected them or the **model loaded them itself** (§4.8). Cleared on compaction.
7. Read surviving `SKILL.md` bodies → build injection text per `inject_mode` & `directive_strength`.
8. Append injected ids to session state; write decision cache.
9. Emit injection JSON.

### 4.6 Session state
`~/.local/state/ski/sessions/<session_id>.json`:
```json
{ "loaded": { "git-tools/git-attribution": "ski", "python-dev/uv-setup": "model" },
  "updated": "2026-06-14T12:00:00Z" }
```
Dedup skips **any** id in `loaded`, regardless of source. `ski` = we injected it;
`model` = the model loaded it on its own (§4.8) — we must not re-inject a decision
the model already made. **Re-arm on compaction** (Claude `SessionStart` matcher
`compact`; opencode `session.compacted`): clear `loaded` so skills can be
re-injected into the fresh summary.

### 4.8 Detect model-loaded skills
The model may pull a skill itself (Claude's native chooser; opencode `use_skill`).
Record those so we don't double-inject:
- **Claude:** `PostToolUse` on `Read`/`Skill` → if the path matches `**/SKILL.md`,
  derive the skill id and mark `loaded[id] = "model"`.
- **opencode:** observe the skill-load / tool-exec event → same.

`ski observe` does this. Especially important in `directive` mode: once the model
acts on our directive and reads the skill, we stop re-prompting it.

### 4.7 Caches (two layers — keep them distinct)
- **Index cache** (skill-description embeddings): persistent, invalidated by file
  hash. *Mandatory, big win.*
- **Decision cache** (`hash(prompt + index_version)` → skill ids, TTL ~1h):
  *optional.* Locally the embed is already ~ms, so this mainly skips re-IO on
  identical re-submits. Mirrors supercharged's intent cache but cheaper to justify.
- **Prompt cache (Anthropic):** we do **not** set `cache-control`. `additionalContext`
  appends *after* the cached system/skills prefix, so injection is cache-safe by
  construction. Re-injecting the same body every turn would bloat context → that is
  exactly what §4.6 session dedup prevents. (opencode local models: no prompt cache;
  same anti-bloat reasoning.)

---

## 5. Config

`~/.config/ski/config.toml` (project `.ski.toml` overrides):
```toml
model            = "bge-small-en-v1.5"  # default; "all-MiniLM-L6-v2-q" = lite (low-RAM/CPU). custom ONNX = later
min_similarity   = 0.35       # cosine threshold — TUNE with `ski why`
max_skills       = 2
char_budget      = 6000       # cap total injected chars
inject_mode      = "directive" # v1 default: tell model to load (keep agency). "body" = inject SKILL.md
directive_strength = "auto"   # auto | soft | hard  (see §6)
decision_cache_ttl = "1h"
extra_roots      = []
deny  = []                    # skill ids never auto-injected
force = []                    # skill ids always injected when keyword hit
```

---

## 6. Per-host / per-model intensity

| Host | Default `inject_mode` | `directive_strength` | Rationale |
|---|---|---|---|
| Claude Code | `directive` | `soft` ("A relevant skill exists below — load it if applicable.") | strong native chooser; a nudge is enough |
| opencode + local model | `directive` | `hard` ("You MUST load and follow the skill below for this task.") | weak chooser → explicit |

`directive_strength = "auto"` resolves from `--host` + a `local_model` config flag.
**v1 ships `directive` everywhere** (keep model agency, observe follow-through);
flip to `body` — globally or per-skill — once eval shows local models ignore the
directive. `body` is the eventual auto-inject target, not the v1 default.

---

## 7. Code sketches

### 7.1 Core: rank (Rust)
```rust
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0, 0.0, 0.0);
    for i in 0..a.len() { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    dot / (na.sqrt() * nb.sqrt() + 1e-8)
}

fn rank(prompt: &str, index: &Index, cfg: &Config) -> Vec<Hit> {
    let model = TextEmbedding::try_new(
        InitOptions::new(cfg.embedding_model())   // bge-small-en-v1.5 (default) | MiniLM-q (lite)
            .with_cache_dir(cfg.model_dir())
    ).unwrap();
    // bge is asymmetric: prefix the query, NOT the skill descriptions (§4.3)
    let q = &model.embed(vec![format!("{}{}", cfg.query_prefix(), prompt)], None).unwrap()[0];

    let mut hits: Vec<Hit> = index.skills.iter().map(|s| {
        let kw = s.keywords.iter()
            .filter(|k| prompt.to_lowercase().contains(&k.to_lowercase()))
            .count() as f32 * cfg.keyword_boost;
        Hit { id: s.id.clone(), score: cosine(q, &s.embedding) + kw }
    }).collect();

    hits.retain(|h| h.score >= cfg.min_similarity || cfg.force.contains(&h.id));
    hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    hits.truncate(cfg.max_skills);
    hits
}
```

### 7.2 Core: hook IO (`ski hook`)
```jsonc
// stdin (normalized by adapter)
{ "host": "claude", "session_id": "abc", "cwd": "/repo", "prompt": "fix the commit attribution" }

// stdout
{ "skills": ["git-tools/git-attribution"],
  "inject": "<skill name=\"git-attribution\">\n...SKILL.md body...\n</skill>" }
```

### 7.3 Claude adapter — `plugins/skill-inject/hooks/hooks.json`
```json
{
  "hooks": {
    "UserPromptSubmit": [
      { "hooks": [ { "type": "command",
        "command": "ski hook --host claude" } ] }
    ],
    "PostToolUse": [
      { "matcher": "Read|Skill",
        "hooks": [ { "type": "command",
          "command": "ski observe --host claude" } ] }
    ],
    "SessionStart": [
      { "matcher": "startup|resume|compact",
        "hooks": [ { "type": "command",
          "command": "ski session-start --host claude" } ] }
    ]
  }
}
```
`ski session-start` = incremental reindex + re-arm; `ski observe` records skills the
model loaded itself (§4.8). The `UserPromptSubmit` wrapper emits Claude's contract:
```json
{ "hookSpecificOutput": {
    "hookEventName": "UserPromptSubmit",
    "additionalContext": "<skill ...>...</skill>" } }
```
> Packaging: ship a tiny `scripts/ski-bootstrap.sh` (rtk-style) that resolves the
> `ski` binary (PATH → `~/.local/bin` → `cargo install` hint) so the plugin works
> even before a global `init`.

### 7.4 opencode adapter — TS plugin
```ts
import type { Plugin } from "@opencode-ai/plugin"

export const SkillInject: Plugin = async ({ $, client, directory }) => {
  return {
    // confirm exact event name against opencode 1.17.x (see Open Questions)
    "chat.message": async ({ message }, _out) => {
      if (message.role !== "user") return
      const payload = JSON.stringify({
        host: "opencode", session_id: message.sessionID,
        cwd: directory, prompt: message.text,
      })
      const res = await $`ski hook --host opencode`.stdin(payload).text()
      const { inject } = JSON.parse(res)
      if (inject) {
        // inject as synthetic context message (same plumbing as opencode-agent-skills)
        await client.session.message.create({
          sessionID: message.sessionID,
          parts: [{ type: "text", text: inject, synthetic: true }],
        })
      }
    },
  }
}
```
> Reference `joshuadavidthomas/opencode-agent-skills` for the exact injection call;
> opencode's "inject AI-visible message" support is maturing (issue #17412).
> Also wire `session.start → ski session-start` (reindex / re-arm) and the
> skill-load event `→ ski observe` (§4.8) to mirror the Claude adapter.

---

## 8. Guardrails
- `min_similarity` threshold + `max_skills` cap + `char_budget` → no context flooding.
- **Relative margin gate** (`score_margin`): drop any skill more than `score_margin`
  below the single best-scoring skill, measured against the global top **before**
  session dedup. Suppresses the weak tail (noisy embedders inject only near-peers)
  and makes re-submitting a handled prompt fall silent instead of scraping lower
  matches once the strong ones are deduped.
- `deny` list; `force` list for must-haves on keyword hit.
- Hybrid (embedding **+** keyword) so exact tool/command names aren't missed by
  embeddings alone.
- Session dedup → never re-inject a skill already loaded (by us **or the model**, §4.8)
  within a session; the only re-arm is compaction.
- Fail-open: any error in `ski hook` → emit empty injection, never block the prompt.

---

## 9. Init (rtk-style)
`ski init -g`:
1. Ensure binary on PATH (or copy to `~/.local/bin`).
2. Download embedding model → `~/.local/share/ski/models/`.
3. Run `ski index` (and re-run on every SessionStart thereafter — §4.1).
4. **Claude:** merge `UserPromptSubmit` + `PostToolUse` (observe model loads) +
   `SessionStart` (reindex & re-arm) hooks into `~/.claude/settings.json`
   (idempotent, like `install-plugins.sh`).

`ski init -g opencode`:
1–3 same.
4. Add plugin entry to `~/.config/opencode/opencode.json` `plugin[]` + the
   `@opencode-ai/plugin` dep; drop the TS adapter file.

Idempotent; `--auto-patch` for non-interactive. In **this repo**, also shipped as a
marketplace plugin (`plugins/skill-inject/`) whose `hooks.json` points at `ski`.

---

## 10. Milestones
1. ✅ **DONE — Core rank + golden tests.** Rust `ski` crate: skill discovery +
   `SKILL.md` frontmatter parse + embedding index + hybrid rank (cosine + keyword),
   exposed as `ski index` / `ski why`. `Embedder` trait with an offline
   bag-of-words backend (default; zero network/model — so build/test/CI run
   anywhere) and a `fastembed` backend (bge-small / MiniLM) behind a cargo feature.
   Golden tests assert `prompt → skill` on the real repo skills (`"…credit Claude in
   this commit"` → `git-attribution`, `"bootstrap a new python project with uv"` →
   `uv-setup`, etc.); verified top-1 on all 56 installed skills. fmt + clippy
   (`-D warnings`) clean; pre-commit hooks wired at repo root. `ski why` is the
   tuning harness. (`fastembed` feature compiles but isn't built in the offline lane
   — verify + tune `min_similarity` against bge before milestone 2 ships injection.)
2. ✅ **DONE — Hook path.** `ski hook --host <claude|opencode>`: stdin event →
   load-or-build index → embed → hybrid rank → `select` (deny + threshold +
   relative margin gate + force + session dedup + `max_skills` cap) →
   `inject::build` (directive/body, host-aware
   strength, `char_budget`) → host contract on stdout (Claude `additionalContext` /
   opencode `{skills,inject}`). Per-session dedup ledger at
   `$XDG_STATE_HOME/ski/sessions/<id>.json` (`loaded: {id: "ski"|"model"}`); a skill
   is never re-injected in a session. Fails open everywhere (empty injection, exit 0).
   Decision cache deferred to milestone 6 (low local payoff). fmt + clippy + tests
   green. *(min_similarity + score_margin still tuned for bow — re-tune for bge.)*
3. **Claude adapter** — hooks.json + bootstrap + `additionalContext`. End-to-end on Claude Code.
4. **opencode adapter** — TS plugin spawning binary; confirm injection event/API.
5. **`ski init`** — global install for both hosts; package as marketplace plugin here.
6. **Polish** — `ski stats`, per-model intensity, compaction re-arm.

---

## 11. Decisions & open questions

**Resolved**
- **Name = `ski`** (product `skill-inject`).
- **Embedding model = `bge-small-en-v1.5`** default — stronger recall for GPU/RAM-rich boxes; `all-MiniLM-L6-v2-q` lite fallback for constrained machines (§4.3). Other local models / custom ONNX = later.
- **`inject_mode` default = `directive`** — keep model agency for v1; revisit `body` after measuring local-model follow-through (§6).
- **Model bundling = download-on-init** — cache to `~/.local/share/ski/models/`; offline `include_bytes!` build stays a feature flag.
- **Reindex on every SessionStart** — skills drift; incremental + hash-gated so it's cheap (§4.1).
- **Track model-loaded skills** — yes; never re-inject the model's own choices (§4.8).

**Open**
1. **Prompt context for embedding** — prompt-only (lean v1) vs prompt + last assistant turn (better recall, noisier).
2. **opencode hook event** — exact event name + injection API in opencode 1.17.x (verify vs `opencode-agent-skills`).
3. **Decision cache** — defer to milestone 6 (low local payoff).

# Review backlog — 2026-06-24

Findings from a deep three-axis review (correctness / style+architecture / docs
usability) of `ski` at v0.8.0. Toolchain was clean (`build`, `clippy
--all-targets --all-features`, `fmt --check`); `cargo test` = 147 pass but
`tests/phrase.rs` flaked under the full parallel run (see S1).

Status legend: **DONE** · **NEXT** · **OPEN** · **WON'T-FIX**

---

## Correctness

- **C1 — DONE.** `hook.rs:331` corrupt `index.json` permanently bricks the hook.
  `Index::load(&path)?` propagated instead of falling through to rebuild → zero
  injections every prompt until manual cleanup. Asymmetric with
  `session_start::reindex` (self-heals). Fixed: `.ok().flatten()` + fall through.
- **C2 — DONE (torn-read half).** `session.rs:103-110` save was a non-atomic,
  unlocked read-modify-write; `fs::write` over the live file let a concurrent
  reader observe a half-written file (~3% torn reads under load → dedup silently
  reset → re-injection). Fixed: write temp + atomic `rename`. **Residual OPEN:**
  the *lost-update* window (two writers racing the read-modify-write drop one
  mark, ~95% under contention) remains — needs an advisory file lock or
  load-merge-save. Documented as best-effort for now. → see C2b.
- **C2b — OPEN.** Lost-update race on concurrent session writes. Advisory file
  lock (e.g. `flock`) around load→mutate→save, or merge-on-write. Lower priority
  than the torn read; effect is a missed dedup, not corruption.
- **C3 — OPEN (MED).** `skill.rs:97` one non-UTF8 `SKILL.md` kills the whole
  library. `read_to_string?` bubbles through `discover` → hook fails open with 0
  injections for *all* skills; `index`/`why` abort exit 1 with no offending path.
  Fix: `fs::read` + `from_utf8_lossy`, or `continue` past the bad file; include
  the path in the error.
- **C4 — OPEN (MED).** `rank.rs:222-226` a NaN score silently mis-ranks
  (`partial_cmp ... unwrap_or(Equal)` → NaN compares Equal to all, can land rank
  0). No crash, wrong output. Fix: filter non-finite before sort, or `total_cmp`
  treating NaN as worst.
- **C5 — OPEN (LOW).** `config.rs:375-386` one mistyped TOML field silently
  discards the *whole* config (incl. `deny`). `toml::from_str(..).ok()`. Keep
  fail-open but emit a one-line stderr warning on parse error so the user knows
  the file was ignored.
- **C6 — OPEN (LOW).** `rank.rs:38-50` `cosine` zips to the shorter vector → a
  query/entry dimension mismatch is silently truncated, not rejected. The
  `model == id()` guard normally prevents it; a hand-edited/same-id index slips
  through. Fix: assert/skip on `a.len() != b.len()`.
- **C7 — OPEN (LOW).** `skill.rs:63-93` no depth cap in `collect` → unbounded
  recursion on a pathologically deep real tree. (Symlink loops are safe — bounded
  by kernel `ELOOP`.) Fix: depth cap mirroring `PROJECT_WALK_LEVELS`.
- **C8 — OPEN (LOW, channel off by default).** `lexical.rs:144` single-skill
  library bypasses the dominance margin (`second = get(1).unwrap_or(0.0)` →
  `top - second == top`); only `lexical_min` gates it. Fix: treat "no runner-up"
  as non-dominant, or require ≥2 skills for the fast-path.
- **C9 — OPEN (trivial/rare).** A leading BOM makes line 1 ≠ `---` → skill file
  silently skipped (`skill.rs` frontmatter parse). Strip BOM before the `---`
  check.

## Style / architecture

- **S1 — DONE.** `tests/phrase.rs:28` `temp_root()` keyed on `process::id()` only;
  both `#[test]`s in the one binary shared the path and test1's `remove_dir_all`
  wiped test2 mid-read → flaky `discover` panic under parallel run. Fixed:
  per-test label + nanos suffix (mirrors `skill.rs:378` idiom).
- **S2 — NEXT.** `cmd_why` (main.rs:154-234) does **not** run the same math as the
  hook: `rank_all` (no file/project/context channels) vs the hook's
  `rank_all_ctx`. `ski why` can star a skill the hook wouldn't inject — breaks the
  README's "Reproduce any of it." Gate logic is also re-implemented inline. Fix:
  extract `pipeline::decide(...) -> Decision` shared by hook / why / eval. (This
  is the deferred big item; subsumes part of S3 and S5.)
- **S3 — OPEN (HIGH).** `hook.rs:72-269` `decide()` is a ~200-line god-function
  (IO, filtering, init, 3 channel assemblies, 3-way stage dispatch, 2 telemetry
  exits, inject, dedup). Only leaf helpers are testable. Decompose into named
  stages (`load_pipeline` / `gather_context` / `run_stages` / `emit`). Pairs with
  S2.
- **S4 — OPEN (MED).** `history.rs:83-208` four near-identical JSONL scanners +
  a fifth in `aggregate` → the log is parsed 4-6× per `history` run. Fix: one
  `parse_events() -> Vec<Event>` pass; derive the per-session maps + aggregate
  from it (~120 lines removed).
- **S5 — OPEN (MED).** Score formula `cos+ctx+file+project+kw+ph` written in 3
  places; `why`'s display (main.rs:199) **and** `rerank::passes` (rerank.rs:132)
  both **omit the `project` term**. Drift bug on any channel add. Fix:
  `Hit::stage1_score()` + `breakdown()`, single-sourced. (Tied to S2.)
- **S6 — OPEN (MED).** `config.rs:391-473` 28 fields × 4 sites
  (`Config`/`FileConfig`/`apply`/`base`) = shotgun surgery per knob. Fix: a
  declarative `overlay!` macro or serde-flatten layer.
- **S7 — OPEN (MED).** `hook.rs:351-401` `select` / `select_reranked` /
  `select_lexical` repeat the same tail (deny filter → Rec map → `should_recommend`
  → `take`). Factor a shared `finalize(...)`; callers differ only in pre-filter.
- **S8 — OPEN (LOW).** `lib.rs:8-26` every module is `pub`; ~10 are internal-only.
  Tighten the genuinely-internal ones to `pub(crate)`.
- **S9 — OPEN (LOW-MED).** `config.rs:195-323` `base()` is 128 lines, ~90 of them
  tuning-history comments that bury the default values. Move the narratives to a
  `TUNING.md` / module docs; keep one-line "why" per default.
- **S10 — OPEN (LOW).** `rank.rs:154-159` `prompt_top` computed by iterating all
  skills, then recomputed again in the hook (hook.rs:136). Return it from
  `rank_all_ctx` so the hook doesn't redo it.
- **S11 — OPEN (LOW).** Inline magic numbers: cosine-map anchors
  (`confidence.rs:60` — 0.5/0.47/0.99) and recency decay base
  (`context.rs:159` — 0.5). Name them as consts for discoverability.
- **S12 — OPEN (MED).** The three fail-open hooks (hook.rs:61, observe.rs:45,
  session_start.rs:33) swallow errors with **zero trace** — when injection
  silently stops there's nothing to debug with. Add a `SKI_DEBUG`-gated
  `eprintln!` of the swallowed error at each `unwrap_or_default`/`let _ =` site.
- **S13 — NOTE (LOW, documented).** `config.rs:329-332` `Default for Config`
  silently means "Claude-host config" (Claude roots). Documented; footgun, not a
  defect.

## Docs usability

- **D1 — DONE.** `README.md:184` `rerank_min = -1.5` → shipped default is `-2.5`
  (`config.rs:292`; live `ski why` prints `threshold -2.50`). Fixed → `-2.5`.
- **D2 — DONE.** First-run model **download** (~275 MB, network-blocking, cached
  to `~/.config/ski/models`) was undocumented; README only mentioned "~270 MB
  RAM," which a reader conflates with the whole footprint. Fixed: explicit
  first-run note in Install.
- **D3 — WON'T-FIX (editorial).** The refreshed README dropped main's "misses ~2
  in 5 / doesn't reliably beat the native chooser" limitation line. This is a
  *deliberate* reframe (v0.8.0 revival: the injection premise was validated, the
  "no value" verdict applied to the old retriever/tuning, not the idea). Leaving
  the current honest framing ("a strong model often won't use a skill it should",
  "errs toward over-sending") as-is. Revisit only if the user wants the
  native-chooser caveat back.
- **D4 — OPEN (LOW).** README `ski why` blocks (lines 35-39, 50-52) show a
  simplified 2-column form; real output has a `threshold` header, a `*` marker,
  a 5-channel breakdown, and a `lexical(BM25)` block. Annotate "(simplified)" or
  show real output.
- **D5 — OPEN (MED).** Zero-skills new-user dead-end: with no skills installed,
  `ski index` prints `indexed 0 skills` and `ski hook` emits nothing, with no
  pointer to *why*. No "verify it works" step after install. Add a first-run /
  verify line and note skills must exist under the discovery roots.
- **D6 — OPEN (LOW).** `ski init` real signature is `init [OPTIONS] <HOST>` with
  `-g` **required**; README always shows `-g` but never says it's mandatory.
  `ski init claude` (no `-g`) errors "per-project install is not implemented yet."
  One-word note.
- **D7 — OPEN (LOW/security).** The curl one-liner pulls *latest* (TOFU, moving
  target). `SKI_VERSION` / `SKI_BIN_DIR` exist (install.sh:8-9) but are
  undocumented. Document a pinned form: `SKI_VERSION=v0.8.0 curl … | sh`.
- **D8 — OPEN (security/MED).** `install.sh:54-62` checksum is fetched from the
  *same* release URL as the binary (integrity, not authenticity — no signature)
  **and** is silently skipped if `checksums.txt` 404s or `sha256sum` is absent.
  Print a visible "checksum unavailable — skipping verification" warning on skip;
  consider signing.
- **D9 — OPEN (LOW).** `lexical_min` / `lexical_margin` config keys exist
  (config.rs:148,153) but aren't in the README "advanced ranking knobs" list.
  (Opt-in / off by default.)
- **D10 — OPEN (LOW).** `ski history --compare` is real (main.rs:90-93) but
  absent from the README Usage block (only `--tail`/`--session` shown).
  Tuning/research aid; low priority.

---

### This pass fixed
C1, C2 (torn-read), S1, D1, D2.

### Recommended next
S2 (single-source the ranker so `why` == hook), which also unblocks S3 and S5.

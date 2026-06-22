# Injection phrasing, dedup, and telemetry

Three related improvements to the hot path, all shipped without touching the
embedder, reranker, or `Cargo.toml` (no new deps; telemetry is gated by an env
var, not a config field, to stay out of `config.rs`).

A single unifying primitive ties them together: **confidence** — one number in
`[0,1]` derived from whichever stage produced a hit (cosine bi-encoder or
cross-encoder reranker). Phrasing scales with it, dedup gates on it, telemetry
records it.

---

## 0. Confidence (`src/confidence.rs`, new)

Two scoring scales reach the injector: stage-1 cosine (`~0.3–0.9`, anisotropic)
and stage-2 reranker logits (`~-10..+10`). Neither is directly comparable, so we
map both onto one calibrated-ish `[0,1]` axis.

```rust
pub enum Stage { Cosine, Rerank }
pub enum Band  { High, Medium, Low }

pub const HIGH: f32 = 0.80; // >= HIGH is the High band and the repeat-recommend bar
pub const LOW:  f32 = 0.55; // <  LOW  is the Low (tentative) band
const COSINE_SPAN: f32 = 0.45; // bge strong matches sit ~0.45 above the floor

pub fn of(score: f32, stage: Stage, cfg: &Config) -> f32 {
    match stage {
        // JINA-turbo logits are ~calibrated; sigmoid -> probability.
        Stage::Rerank => 1.0 / (1.0 + (-score).exp()),
        // Cosine has no probabilistic meaning; piecewise [floor, floor+span] -> [.5,.97].
        Stage::Cosine => {
            let t = ((score - cfg.min_similarity) / COSINE_SPAN).clamp(0.0, 1.0);
            (0.5 + 0.47 * t).clamp(0.0, 0.99)
        }
    }
}

pub fn band(conf: f32) -> Band { /* HIGH / LOW cutoffs */ }
```

The cosine mapping is an explicit heuristic (documented as such); the reranker
mapping is principled (the cross-encoder is trained with a sigmoid objective).
`HIGH`/`LOW`/`COSINE_SPAN` are module constants for now — promote to `Config`
later if they need per-deployment tuning (kept out of `config.rs` to avoid
colliding with the in-flight rerank-calibration work).

---

## 1. Shorter, confidence-scaled phrasing (`src/inject.rs`)

**Before** — a ~430-char header every turn plus, per skill, a name, description,
a full `Skill`-tool sentence, and `(source: /long/path)`:

```
The following skills are likely relevant to this request. Invoke each relevant
one with the `Skill` tool ... Read the listed path directly only if ...:

- **git-attribution** — How to attribute AI assistance in git commits.
  If relevant, invoke it: `Skill` with skill `git-attribution` (source: /home/.../SKILL.md)
```

**After** — a one-line header, and per skill a distinctive `SkillRecommendation`
token, the description, and a verb scaled by band. The `(source: path)` fallback
is dropped (both hosts have a `Skill` mechanism); the raw confidence decimal is
**not** shown either (see the credibility note below):

```
ski matched these skills to your request — a dedicated retrieval+rerank pass,
separate from and complementary to the host's own skill selection. Invoke
fitting ones by name via the `Skill` tool; do not Read the files:

- SkillRecommendation(`git-attribution`): How to attribute AI assistance in git commits. — invoke it.
- SkillRecommendation(`pdf`): Work with PDF files. — invoke it if it fits.
- SkillRecommendation(`xlsx`): Spreadsheet tasks. — consider invoking it.
```

Verb by `(strength, band)`:

| band   | soft (Claude)            | hard (opencode)                         |
|--------|--------------------------|-----------------------------------------|
| High   | `— invoke it.`           | `— you MUST invoke it before responding.`|
| Medium | `— invoke it if it fits.`| `— invoke it before responding if it fits.`|
| Low    | `— consider invoking it.`| `— invoke it before responding if it fits.`|

`build` takes `&[Rec]` (`id` + `confidence`) instead of `&[Hit]`; the header/line
templates read the band. The distinctive `SkillRecommendation(` token also makes
telemetry/grep able to spot a recommendation echoed back in a transcript.

**Credibility note (2026-06-22).** The phrasing was retuned for trust once the
reranker floor (`rerank_min`, see the config) was tightened to abstain on
low-confidence noise. Because every injected line now clears a precision gate,
the language stops hedging: the old `— possibly relevant.` (which undersold
matches the model *did* act on) became `— consider invoking it.`, the header
reframes ski as a dedicated pass that complements the host's built-in chooser
(which misses skills), and the raw decimal — which only invited the model to
anchor on a mid value and discount a real match — is dropped from the
model-facing line while still being recorded in telemetry.

---

## 2. Score-aware dedup (`src/session.rs`, `src/hook.rs`)

Each session entry grows from a bare `Source` to a `Record { source, confidence }`
(backward-compatible deserialize: an old bare `"ski"` string still loads, as
`confidence = 0`).

Two rules, exactly as specified:

1. **A used skill is never recommended again.** If the model loaded it
   (`Source::Model`, recorded by `ski observe`), it is permanently suppressed.
2. **A recommended-but-unused skill is re-recommended only when it newly reaches
   HIGH confidence.** We showed it once below HIGH; we get one more nudge if a
   later prompt makes it clearly relevant. Once shown at HIGH, never again (the
   model saw the strongest signal and passed).

```rust
pub fn should_recommend(&self, id: &str, new_conf: f32, high: f32) -> bool {
    match self.loaded.get(id) {
        None                                     => true,           // never seen
        Some(r) if r.source == Source::Model     => false,          // used -> never
        Some(r)                                  => new_conf >= high && r.confidence < high,
    }
}
```

`should_recommend` replaces the old unconditional `!is_loaded` filter inside
`select` / `select_reranked`. The relative-margin gate still runs against the
global best *before* dedup, so a repeat prompt whose strong match is already
loaded still falls silent instead of scraping its weak tail. After injection,
each shown id is recorded with the confidence we displayed
(`mark_recommended`), so next turn's `r.confidence < high` test is accurate.

---

## 3. Opt-in telemetry (`src/telemetry.rs` new, `src/history.rs` new)

Off by default. Enabled with `SKI_TELEMETRY=1` (truthy: `1|true|yes|on`) in the
hook's environment (e.g. the `env` block of `settings.json`). Append-only JSONL
at `$XDG_STATE_HOME/ski/telemetry.jsonl`, two event kinds:

```jsonc
// hook, per injection:
{"ts":1718.., "kind":"recommend", "session":"s1", "prompt":"commit this",
 "stage":"rerank",
 "candidates":[{"id":"git-attribution","confidence":0.91}],
 "injected":  [{"id":"git-attribution","confidence":0.91}]}

// observe, when the model loads a skill itself:
{"ts":1718.., "kind":"use", "session":"s1", "skill":"git-attribution", "via":"skill"}
```

Joining the two by `session` + `skill` answers the question the handoff asked —
*was a recommendation acted on?* — and surfaces both failure modes:

- **false positive**: recommended, never used (phrasing/threshold too eager).
- **recall miss**: used, never recommended (calibration gap → free hard-negative
  / positive data for the rerank tuning work).

`ski history` aggregates the log:

```
$ ski history
events: 412 recommend, 138 use across 47 sessions
recommended:        251   used-after-rec: 96 (38%)   false positives: 155 (62%)
recall misses (used, never recommended): 42
top false positives:  template-skill ×19, internal-comms ×11, ...
top recall misses:     pdf ×7, uv-develop ×5, ...
```

`ski clear` wipes per-session dedup state (handy for re-testing dedup);
`ski clear --telemetry` also truncates the log.

---

## Status / effectiveness

See the handoff. In short: all three implemented behind tests; phrasing cuts
per-injection overhead roughly in half; dedup and telemetry are logic-verified by
unit tests. End-to-end "did the model behave better" needs live runs with
`SKI_TELEMETRY=1`, which the telemetry exists to measure.

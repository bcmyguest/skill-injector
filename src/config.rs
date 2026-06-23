//! Runtime configuration. Compiled defaults ([`Config::base`]), overlaid by an
//! optional user file (`~/.config/ski/config.toml`, see [`FileConfig`]) loaded
//! through [`Config::load`]. The file is the escape hatch: silence a noisy skill
//! with `deny`, pin `rerank_min`, widen `max_skills`, etc. without a rebuild.

use crate::embed::Embedder;
use crate::hook::Host;
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// How a matched skill is delivered to the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InjectMode {
    /// Tell the model a relevant skill exists and let it load the file (keeps
    /// model agency; the v1 default).
    Directive,
    /// Inject the `SKILL.md` body straight into context.
    Body,
}

/// Forcefulness of a `directive`-mode injection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strength {
    /// Resolve from the host (Claude -> soft, opencode -> hard).
    Auto,
    /// A nudge — enough for a strong native chooser.
    Soft,
    /// An imperative — for weak local choosers.
    Hard,
}

#[derive(Debug, Clone)]
pub struct Config {
    /// Embedding model id. Recognized by the fastembed backend; otherwise the
    /// offline bag-of-words backend is used regardless of this value.
    pub model: String,
    /// Minimum hybrid score for a skill to be eligible for injection.
    pub min_similarity: f32,
    /// Max gap below the single best-scoring skill a skill may fall and still be
    /// injected. Suppresses the weak tail: when the top match is strong, only
    /// near-peers ride along; when only weak matches exist (or the leader was
    /// already injected this session), nothing clears the gate. Tuned alongside
    /// `min_similarity` per embedder.
    pub score_margin: f32,
    /// Max skills injected per prompt.
    pub max_skills: usize,
    /// Max total injected characters (budget; enforced in the hook path).
    pub char_budget: usize,
    /// Added to a skill's score per matching keyword.
    pub keyword_boost: f32,
    /// Added to a skill's score per matched trigger phrase (see
    /// [`crate::rank::phrase_score`]). Higher than `keyword_boost`: a full
    /// multi-token phrase match is stronger, higher-precision evidence than a
    /// single keyword token.
    pub phrase_boost: f32,
    /// Filesystem roots scanned for `SKILL.md` files.
    pub roots: Vec<PathBuf>,
    /// How matched skills are injected.
    pub inject_mode: InjectMode,
    /// Forcefulness of directive-mode injections.
    pub directive_strength: Strength,
    /// Skill ids never auto-injected.
    pub deny: Vec<String>,
    /// Skill ids injected whenever a keyword hits, even below `min_similarity`.
    pub force: Vec<String>,

    // --- Query-side context enrichment (see `crate::rank::rank_all_ctx`). A vague
    // follow-up prompt is disambiguated by signals from the turns before it. Two
    // channels share the recent-prompt window (`context_depth`):
    //   * the file-type channel (`file_boost`) — a document file named in the prompt
    //     or a recent turn (`.xlsx`, `.pdf`, ...) boosts its skill. High-precision
    //     and **on by default**: it adds no false-inject on any eval corpus and
    //     doubles recall on multi-turn document follow-ups.
    //   * the dense blend (`context_weight`) — a recency-weighted context vector
    //     mixed into stage-1. **Off by default**: it lifts multi-turn recall but
    //     admits a topic-switch false-inject no scalar floor separates from genuine
    //     vague follow-ups. Set `context_weight > 0` to opt in. ---
    /// How many recent prompts to retain as conversational context (0 = disabled).
    pub context_depth: usize,
    /// Max weight the context channel can add to a skill's score. The *effective*
    /// weight scales from this (a fully vague prompt) down to 0 (a confident,
    /// specific prompt) — see [`crate::rank::context_weight`]. Cosine-space, tuned
    /// per embedder; 0.0 disables the blend.
    pub context_weight: f32,
    /// Prompt best-self-cosine at/below which a prompt counts as *fully* vague
    /// (context applied at full `context_weight`).
    pub vague_lo: f32,
    /// Prompt best-self-cosine at/above which a prompt counts as confident
    /// (context suppressed entirely). Between `vague_lo` and this, context scales
    /// linearly.
    pub vague_hi: f32,
    /// Score added to a skill when a file of its type is referenced in the prompt
    /// or recent context (e.g. a `.xlsx` boosts `xlsx`; see
    /// [`crate::context::file_ids`]). High-precision and *not* vagueness-gated — a
    /// named file is unambiguous. 0.0 disables the channel.
    pub file_boost: f32,
    /// Score added to a skill whose ecosystem matches the working directory's
    /// project manifest (a `Cargo.toml` boosts the canonical rust skill, a
    /// `go.mod` the go skill, etc.; see [`crate::context::project_ids`]). Unlike a
    /// named file, this is an *ambient* signal present on every turn, so it is the
    /// weakest channel and is **gated on the skill's own cosine clearing
    /// `min_similarity`** in [`crate::rank::rank_all_ctx`] — it can only break ties
    /// among already-plausible skills, never rescue an irrelevant one below the
    /// floor. **Off by default** (0.0) pending live-data tuning; set low (≤ ~0.1)
    /// when enabling. A project type that maps to several skills (python, the JS
    /// frameworks) is deliberately left unmapped, so this is near-inert on a
    /// single-ecosystem-per-skill corpus.
    pub project_boost: f32,

    // --- Stage-2 reranking (see `crate::rerank`). The thresholds below are on the
    // cross-encoder's logit scale, unrelated to the cosine thresholds above, and
    // are *not* touched by `calibrate_to`. ---
    /// Stage-1 score below which a prompt is treated as having no relevant skill,
    /// so the (costly) reranker is skipped entirely.
    pub recall_floor: f32,
    /// Stage-1 score above which the top match may be a confident lone winner.
    pub high_conf: f32,
    /// Minimum stage-1 gap from the top match to the runner-up for the top to
    /// count as a *lone* winner (and thus skip reranking).
    pub clear_gap: f32,
    /// How many stage-1 candidates are handed to the reranker.
    pub rerank_top_k: usize,
    /// Minimum reranker logit for a skill to be injected.
    pub rerank_min: f32,
    /// Max reranker-logit gap below the best reranked skill for a peer to ride along.
    pub rerank_margin: f32,

    /// Confidence (`[0,1]`) at/above which a *lone* near-certain match is escalated
    /// from a directive pointer to a full body inject — the `SKILL.md` is inlined
    /// directly so the model can't skip the Skill-tool round-trip. Only fires in
    /// `inject_mode = directive` and only when exactly one skill is selected (two
    /// co-relevant peers mean we are *less* certain, so they stay directives). Set
    /// deliberately high: in practice this is reached only by a cross-encoder-
    /// confirmed match (the cosine→confidence map caps below it for bge), so a
    /// fluky stage-1 hit never triggers a body dump. Raise above `1.0` to disable.
    pub body_inject_min: f32,

    // --- Stage-1.5 lexical channel (see `crate::lexical`). BM25 over the full skill
    // description, a high-precision fast-path that injects a *dominant* lexical
    // winner directly, skipping the reranker — it rescues indirect prompts whose
    // discriminating vocabulary lives in the description prose but whose bi-encoder
    // cosine is muddy and whose reranker logit falls below the abstention floor.
    // Only fires when stage-1 has no confident lone dense winner. These thresholds
    // are on BM25's own scale, unrelated to the cosine/logit thresholds above and
    // untouched by `calibrate_to`. ---
    /// Minimum absolute BM25 score for the top description match to be a lexical
    /// winner. `<= 0` disables the channel entirely.
    pub lexical_min: f32,
    /// Minimum BM25 gap from the top description match to the runner-up for the top
    /// to count as *dominant* (and thus inject directly, skipping the reranker). The
    /// margin is what keeps the fast-path high-precision: a cluster of near-equal
    /// descriptions abstains and defers to the reranker.
    pub lexical_margin: f32,

    /// Append opt-in JSONL telemetry events (see [`crate::telemetry`]). Off by
    /// default. Enabled by this field *or* a truthy `SKI_TELEMETRY` env var —
    /// either one turns it on, so the env var still works without a config file.
    pub telemetry: bool,
}

impl Config {
    /// Adopt the active embedder's score thresholds. Cosine distributions are a
    /// property of the embedding space, not user preference, so `min_similarity`
    /// and `score_margin` follow the embedder that actually ran (bge vs the
    /// offline bag-of-words fallback). Other fields are left untouched.
    pub fn calibrate_to(&mut self, embedder: &dyn Embedder) {
        self.min_similarity = embedder.min_similarity();
        self.score_margin = embedder.score_margin();
    }

    /// Config scoped to `host`: discovery `roots` (and, via
    /// [`crate::paths::index_path`], the on-disk index) cover only that host's
    /// skill library. Keeps an injected skill name resolvable in the host that
    /// receives it — a Claude-only id never injects into opencode and vice versa.
    pub fn for_host(host: Host) -> Self {
        Self {
            roots: host_roots(host),
            ..Self::base()
        }
    }

    /// Host-scoped config with the user file ([`FileConfig`]) overlaid, returned
    /// alongside the parsed file. The file is returned so a caller that calibrates
    /// can re-assert the cosine pins afterward: [`Config::calibrate_to`] overwrites
    /// `min_similarity`/`score_margin` from the embedder and would otherwise clobber
    /// a user-set value. Callers that never calibrate can ignore the [`FileConfig`].
    pub fn load(host: Host) -> (Self, FileConfig) {
        let file = FileConfig::load();
        let mut cfg = Self::for_host(host);
        file.apply(&mut cfg);
        (cfg, file)
    }

    /// Every field except `roots`, which [`Config::for_host`] fills per host.
    fn base() -> Self {
        Self {
            model: "bge-small-en-v1.5".into(),
            min_similarity: 0.30,
            score_margin: 0.15,
            max_skills: 2,
            char_budget: 6000,
            keyword_boost: 0.15,
            phrase_boost: 0.20,
            roots: Vec::new(), // overwritten by `for_host`.
            inject_mode: InjectMode::Directive,
            directive_strength: Strength::Auto,
            deny: Vec::new(),
            force: Vec::new(),
            // The file-type channel is on by default (high-precision, zero added
            // false-inject across every eval corpus). The dense blend stays off
            // (`context_weight 0.0`): it lifts multi-turn recall but admits a
            // topic-switch false-inject no scalar floor separates. `context_depth`
            // keeps the rolling prompt window the file channel needs to see a file
            // attached a turn or two back; `vague_lo`/`hi` are bge-cosine-space
            // gates, live only for the dense blend once it is opted into.
            context_depth: 3,
            context_weight: 0.0,
            vague_lo: 0.55,
            vague_hi: 0.65,
            file_boost: 0.3,
            // Ambient project signal off until tuned on live data (see field docs).
            project_boost: 0.0,
            // Reranker gate + thresholds, calibrated against the JINA turbo
            // reranker (see `examples/rerank_probe`). Stage-1 top-1 accuracy: 76%
            // stage-1 only -> 88% with reranking.
            //
            // `recall_floor` skips the reranker when nothing is plausibly relevant.
            // bge is anisotropic (unrelated prompts still cosine ~0.5), which
            // compresses the usable range: 0.50 skips clearly-irrelevant prompts
            // without dropping real-but-weak matches. `high_conf` is effectively
            // disabled (2.0): a confidence-based skip measurably *hurt* accuracy,
            // because the bi-encoder is confidently wrong on the confusable pairs
            // the reranker exists to fix. It is retained as a tunable, not removed.
            //
            // `rerank_min` is tuned on the realistic ~120-skill index (17 anthropic
            // + highest-installed community skills from skills.sh; see
            // `tests/data/popular_skills_prompts.tsv`) — the artificially narrow
            // 17-skill anthropic library is *not* the tuning authority (its indirect
            // prompts have no good match and score like noise, which would pull the
            // floor too low). The earlier -2.5/-1.5 floors admitted a band of
            // negative-logit candidates — sigmoid(-1.5) ~= 0.18 confidence — that the
            // cross-encoder is itself signalling are *not* matches. Live telemetry
            // confirmed it: of the recommendations injected in the 0.18-0.24
            // confidence band ("commit and push" -> caveman-commit, "continue" ->
            // pickup, "/goal ..." -> skill-creator), essentially none were ever acted
            // on. Injecting them is the dominant usage-rate drag and erodes the whole
            // channel's credibility (the model learns to ignore *every*
            // SkillRecommendation, including the strong ones), so the floor abstains
            // on them. -1.1 (sigmoid ~= 0.25) is the precise recall-preserving
            // precision maximum on the realistic corpus: a strict improvement over
            // -1.5 there (recall held at 41/43 = 95%, false injects 3 -> 1 of 64);
            // tightening past it (-1.0) starts dropping real positives. The cost is
            // some recall on *indirect* prompts whose logit sits in -1.5..-1.1 — a
            // zone where genuine weak matches and live noise overlap and no scalar
            // separates them (cross-encoder pairs score independently of the index) —
            // but those are largely cases the host's own skill chooser covers anyway.
            // Larger embedders/rerankers (bge-base, jina-v2) tie this at higher cost,
            // so the gate is the lever. Sweep via `SKI_RERANK_MIN` in `examples/eval`.
            //
            // `rerank_min` alone can't catch every false inject: on a richer corpus
            // a no-match prompt's reranked logit interleaves with genuine weak
            // matches, so no scalar separates them. The complementary lever is the
            // stage-1 *agreement* gate in `rerank::passes` — a reranked skill's
            // bi-encoder score must sit within a small slack of `min_similarity`,
            // i.e. the reranker may reorder the retrieved-relevant set but not
            // resurrect a skill stage-1 judged irrelevant. That cut false injects a
            // further ~67% (3 -> 1 on the 52-negative realistic corpus) at no extra
            // compute, holding recall at 95%. See `rerank::AGREEMENT_SLACK`, `examples/eval`.
            recall_floor: 0.50,
            high_conf: 2.0,
            clear_gap: 0.12,
            rerank_top_k: 12,
            rerank_min: -1.1,
            rerank_margin: 2.0,
            // Body-escalate only a lone, cross-encoder-confirmed near-certain match
            // (sigmoid(2.45) ~= 0.92). High enough that a stage-1 cosine hit — whose
            // confidence maps to <= ~0.85 for bge — never triggers it, so only the
            // reranker's strongest verdicts inline the full SKILL.md.
            body_inject_min: 0.92,
            // Lexical fast-path off until tuned on the eval corpus (see `crate::lexical`
            // and `examples/eval`). Sweep via `SKI_LEXICAL_MIN` / `SKI_LEXICAL_MARGIN`;
            // set `lexical_min > 0` (with a validated margin) to enable.
            lexical_min: 0.0,
            lexical_margin: 0.0,
            telemetry: false,
        }
    }
}

impl Default for Config {
    /// The Claude-scoped config. `ski index`/`why` (and the eval harness) default
    /// here; the hot paths build [`Config::for_host`] from their `--host` flag.
    fn default() -> Self {
        Self::for_host(Host::Claude)
    }
}

/// User overrides parsed from `~/.config/ski/config.toml`. Every field is
/// optional; an absent field — or an absent/malformed file — leaves the compiled
/// default untouched. Parsing fails open: a malformed file yields an empty
/// overlay (all defaults) rather than an error, so a bad config can never block
/// injection. Unknown keys are ignored (a typo drops one field, not the file).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct FileConfig {
    pub model: Option<String>,
    pub min_similarity: Option<f32>,
    pub score_margin: Option<f32>,
    pub max_skills: Option<usize>,
    pub char_budget: Option<usize>,
    pub keyword_boost: Option<f32>,
    pub phrase_boost: Option<f32>,
    pub roots: Option<Vec<PathBuf>>,
    pub inject_mode: Option<String>,
    pub directive_strength: Option<String>,
    pub deny: Option<Vec<String>>,
    pub force: Option<Vec<String>>,
    pub context_depth: Option<usize>,
    pub context_weight: Option<f32>,
    pub vague_lo: Option<f32>,
    pub vague_hi: Option<f32>,
    pub file_boost: Option<f32>,
    pub project_boost: Option<f32>,
    pub recall_floor: Option<f32>,
    pub high_conf: Option<f32>,
    pub clear_gap: Option<f32>,
    pub rerank_top_k: Option<usize>,
    pub rerank_min: Option<f32>,
    pub rerank_margin: Option<f32>,
    pub body_inject_min: Option<f32>,
    pub lexical_min: Option<f32>,
    pub lexical_margin: Option<f32>,
    pub telemetry: Option<bool>,
}

impl FileConfig {
    /// Parse the user config, or an empty (all-default) overlay when the file is
    /// missing or unparseable.
    pub fn load() -> Self {
        std::fs::read_to_string(crate::paths::config_path())
            .ok()
            .and_then(|raw| Self::parse(&raw))
            .unwrap_or_default()
    }

    /// Pure TOML parse, shared by [`load`](Self::load) and tests. `None` on
    /// malformed input.
    fn parse(raw: &str) -> Option<Self> {
        toml::from_str(raw).ok()
    }

    /// Overlay every present field onto `cfg`. `roots` is ignored while the
    /// `SKI_ROOTS` env override is active (env wins, for evals/tooling). Unknown
    /// `inject_mode`/`directive_strength` strings are ignored, keeping the default.
    pub fn apply(&self, cfg: &mut Config) {
        if let Some(v) = &self.model {
            cfg.model = v.clone();
        }
        self.apply_cosine(cfg);
        if let Some(v) = self.max_skills {
            cfg.max_skills = v;
        }
        if let Some(v) = self.char_budget {
            cfg.char_budget = v;
        }
        if let Some(v) = self.keyword_boost {
            cfg.keyword_boost = v;
        }
        if let Some(v) = self.phrase_boost {
            cfg.phrase_boost = v;
        }
        if let Some(v) = &self.roots {
            if std::env::var_os("SKI_ROOTS").is_none() {
                cfg.roots = v.clone();
            }
        }
        if let Some(m) = self.inject_mode.as_deref().and_then(parse_inject_mode) {
            cfg.inject_mode = m;
        }
        if let Some(s) = self.directive_strength.as_deref().and_then(parse_strength) {
            cfg.directive_strength = s;
        }
        if let Some(v) = &self.deny {
            cfg.deny = v.clone();
        }
        if let Some(v) = &self.force {
            cfg.force = v.clone();
        }
        if let Some(v) = self.context_depth {
            cfg.context_depth = v;
        }
        if let Some(v) = self.context_weight {
            cfg.context_weight = v;
        }
        if let Some(v) = self.vague_lo {
            cfg.vague_lo = v;
        }
        if let Some(v) = self.vague_hi {
            cfg.vague_hi = v;
        }
        if let Some(v) = self.file_boost {
            cfg.file_boost = v;
        }
        if let Some(v) = self.project_boost {
            cfg.project_boost = v;
        }
        if let Some(v) = self.recall_floor {
            cfg.recall_floor = v;
        }
        if let Some(v) = self.high_conf {
            cfg.high_conf = v;
        }
        if let Some(v) = self.clear_gap {
            cfg.clear_gap = v;
        }
        if let Some(v) = self.rerank_top_k {
            cfg.rerank_top_k = v;
        }
        if let Some(v) = self.rerank_min {
            cfg.rerank_min = v;
        }
        if let Some(v) = self.rerank_margin {
            cfg.rerank_margin = v;
        }
        if let Some(v) = self.body_inject_min {
            cfg.body_inject_min = v;
        }
        if let Some(v) = self.lexical_min {
            cfg.lexical_min = v;
        }
        if let Some(v) = self.lexical_margin {
            cfg.lexical_margin = v;
        }
        if let Some(v) = self.telemetry {
            cfg.telemetry = v;
        }
    }

    /// Re-assert just the cosine thresholds. [`Config::calibrate_to`] overwrites
    /// `min_similarity`/`score_margin` from the embedder, so a user pin must be
    /// applied *after* calibration to survive.
    pub fn apply_cosine(&self, cfg: &mut Config) {
        if let Some(v) = self.min_similarity {
            cfg.min_similarity = v;
        }
        if let Some(v) = self.score_margin {
            cfg.score_margin = v;
        }
    }
}

fn parse_inject_mode(s: &str) -> Option<InjectMode> {
    match s.trim().to_ascii_lowercase().as_str() {
        "directive" => Some(InjectMode::Directive),
        "body" => Some(InjectMode::Body),
        _ => None,
    }
}

fn parse_strength(s: &str) -> Option<Strength> {
    match s.trim().to_ascii_lowercase().as_str() {
        "auto" => Some(Strength::Auto),
        "soft" => Some(Strength::Soft),
        "hard" => Some(Strength::Hard),
        _ => None,
    }
}

/// Discovery roots for `host`. `SKI_ROOTS` (colon-separated) overrides for any
/// host — it lets evals/tools scope discovery to one skill library without a
/// config file (e.g. `SKI_ROOTS=~/.claude/plugins/marketplaces/anthropic-agent-skills`).
fn host_roots(host: Host) -> Vec<PathBuf> {
    if let Some(raw) = std::env::var_os("SKI_ROOTS") {
        let roots: Vec<PathBuf> = std::env::split_paths(&raw)
            .filter(|p| !p.as_os_str().is_empty())
            .collect();
        if !roots.is_empty() {
            return roots;
        }
    }
    match host {
        Host::Claude => {
            let mut v = Vec::new();
            if let Some(h) = std::env::var_os("HOME").map(PathBuf::from) {
                v.push(h.join(".claude/skills"));
                v.push(h.join(".claude/plugins"));
            }
            v.push(PathBuf::from(".claude/skills"));
            v
        }
        Host::Opencode => opencode_roots(),
    }
}

/// opencode declares its skill directories in `opencode.json` (`skills.paths`),
/// not a fixed directory, so its roots are read from the global config rather
/// than guessed. Absolute paths are used as-is; relative paths resolve against
/// the process cwd, which the hook subprocess inherits from opencode's project
/// dir. Project-local `opencode.json` overrides are a later milestone (the hook
/// does not yet consume the event's `cwd`).
fn opencode_roots() -> Vec<PathBuf> {
    let Some(cfg_path) = opencode_config_path() else {
        return Vec::new();
    };
    let Ok(raw) = std::fs::read_to_string(&cfg_path) else {
        return Vec::new();
    };
    parse_opencode_paths(&raw, std::env::current_dir().ok().as_deref())
}

/// Location of opencode's global config (`$XDG_CONFIG_HOME/opencode/opencode.json`,
/// default `~/.config/opencode/opencode.json`).
fn opencode_config_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("opencode").join("opencode.json"))
}

/// Pull `skills.paths` out of an opencode config blob, resolving relative entries
/// against `cwd`. A missing key or malformed JSON yields no roots (fail open: no
/// injection rather than a wrong-host one). Pure core of [`opencode_roots`].
fn parse_opencode_paths(raw: &str, cwd: Option<&Path>) -> Vec<PathBuf> {
    let Ok(json) = serde_json::from_str::<serde_json::Value>(raw) else {
        return Vec::new();
    };
    let Some(paths) = json
        .get("skills")
        .and_then(|s| s.get("paths"))
        .and_then(|p| p.as_array())
    else {
        return Vec::new();
    };
    paths
        .iter()
        .filter_map(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| {
            let p = PathBuf::from(s);
            match cwd {
                Some(cwd) if p.is_relative() => cwd.join(p),
                _ => p,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::{bow::BowEmbedder, EmbedKind, Embedder};

    /// Stands in for a dense embedder with its own (non-default) thresholds.
    struct StubEmbedder;
    impl Embedder for StubEmbedder {
        fn id(&self) -> String {
            "stub".into()
        }
        fn embed(&self, _: &[String], _: EmbedKind) -> anyhow::Result<Vec<Vec<f32>>> {
            Ok(vec![])
        }
        fn min_similarity(&self) -> f32 {
            0.64
        }
        fn score_margin(&self) -> f32 {
            0.12
        }
    }

    #[test]
    fn calibrate_adopts_embedder_thresholds() {
        let mut cfg = Config::default();
        cfg.calibrate_to(&StubEmbedder);
        assert_eq!(cfg.min_similarity, 0.64);
        assert_eq!(cfg.score_margin, 0.12);
    }

    #[test]
    fn claude_roots_are_claude_scoped() {
        // Skip if an outer `SKI_ROOTS` override is active (it shadows both hosts).
        if std::env::var_os("SKI_ROOTS").is_some() {
            return;
        }
        let claude = host_roots(Host::Claude);
        assert!(claude
            .iter()
            .any(|p| p.to_string_lossy().contains(".claude/skills")));
        assert!(!claude
            .iter()
            .any(|p| p.to_string_lossy().contains("opencode")));
    }

    #[test]
    fn opencode_paths_parsed_and_resolved() {
        let json = r#"{"skills":{"paths":[".opencode/skills","/abs/repo"],"urls":[]}}"#;
        let roots = parse_opencode_paths(json, Some(Path::new("/proj")));
        assert_eq!(
            roots,
            vec![
                PathBuf::from("/proj/.opencode/skills"),
                PathBuf::from("/abs/repo"),
            ]
        );
    }

    #[test]
    fn opencode_paths_tolerate_missing_key_and_bad_json() {
        assert!(parse_opencode_paths("{}", None).is_empty());
        assert!(parse_opencode_paths(r#"{"skills":{}}"#, None).is_empty());
        assert!(parse_opencode_paths("not json", None).is_empty());
    }

    #[test]
    fn file_overlay_applies_present_fields_only() {
        let raw = r#"
            max_skills = 5
            rerank_min = -0.5
            deny = ["noisy-skill"]
            inject_mode = "body"
            directive_strength = "hard"
            telemetry = true
        "#;
        let file = FileConfig::parse(raw).unwrap();
        let mut cfg = Config::default();
        let (orig_model, orig_budget) = (cfg.model.clone(), cfg.char_budget);
        assert!(!cfg.telemetry); // off by default
        file.apply(&mut cfg);
        assert_eq!(cfg.max_skills, 5);
        assert_eq!(cfg.rerank_min, -0.5);
        assert_eq!(cfg.deny, ["noisy-skill"]);
        assert_eq!(cfg.inject_mode, InjectMode::Body);
        assert_eq!(cfg.directive_strength, Strength::Hard);
        assert!(cfg.telemetry); // enabled via config.toml
                                // Untouched fields keep their defaults.
        assert_eq!(cfg.model, orig_model);
        assert_eq!(cfg.char_budget, orig_budget);
    }

    #[test]
    fn cosine_pin_survives_calibration() {
        // A user pin must win even though calibrate_to runs after the overlay.
        let file = FileConfig::parse("min_similarity = 0.80").unwrap();
        let mut cfg = Config::default();
        file.apply(&mut cfg);
        cfg.calibrate_to(&StubEmbedder); // would set 0.64
        file.apply_cosine(&mut cfg); // re-assert the pin
        assert_eq!(cfg.min_similarity, 0.80);
        assert_eq!(cfg.score_margin, 0.12); // unpinned -> embedder value
    }

    #[test]
    fn malformed_file_is_empty_overlay() {
        assert!(FileConfig::parse("this is not = = toml").is_none());
    }

    #[test]
    fn unknown_keys_are_ignored() {
        let file = FileConfig::parse("bogus_key = 1\nmax_skills = 3").unwrap();
        let mut cfg = Config::default();
        file.apply(&mut cfg);
        assert_eq!(cfg.max_skills, 3);
    }

    #[test]
    fn bad_enum_string_keeps_default() {
        let file = FileConfig::parse(r#"inject_mode = "nonsense""#).unwrap();
        let mut cfg = Config::default();
        file.apply(&mut cfg);
        assert_eq!(cfg.inject_mode, InjectMode::Directive); // unchanged
    }

    #[test]
    fn calibrate_to_bow_uses_trait_defaults() {
        // The bag-of-words embedder doesn't override the trait defaults.
        let mut cfg = Config {
            min_similarity: 0.99,
            score_margin: 0.99,
            ..Default::default()
        };
        cfg.calibrate_to(&BowEmbedder::new());
        assert_eq!(cfg.min_similarity, 0.30);
        assert_eq!(cfg.score_margin, 0.15);
    }
}

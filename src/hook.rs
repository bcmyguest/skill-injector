//! `ski hook` — the hot path. Reads a hook event on stdin, decides which skills
//! to inject, writes the host's injection contract on stdout.
//!
//! **Fail open is the contract.** Any error — bad stdin, missing index, IO
//! failure — results in an empty injection and exit 0, never a blocked prompt.
//!
//! Output by host:
//! - Claude (`UserPromptSubmit`): `{hookSpecificOutput:{hookEventName,
//!   additionalContext}}`, or nothing when there's no injection.
//! - opencode: `{skills:[...], inject:"..."}` always (the TS adapter parses it).

use crate::confidence::{self, Stage};
use crate::config::{Config, InjectMode, Strength};
use crate::embed::{self, EmbedKind};
use crate::index::{self, Index};
use crate::inject::Rec;
use crate::rank::Hit;
use crate::session::Session;
use crate::{context, inject, paths, pipeline, rank, skill, telemetry};
use serde::Deserialize;
use std::io::Read;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Host {
    Claude,
    Opencode,
}

impl FromStr for Host {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "claude" => Ok(Host::Claude),
            "opencode" => Ok(Host::Opencode),
            other => anyhow::bail!("unknown host '{other}' (expected 'claude' or 'opencode')"),
        }
    }
}

/// The normalized hook event. Claude's `UserPromptSubmit` payload already uses
/// these field names; the opencode adapter sends the same shape.
#[derive(Debug, Default, Deserialize)]
struct RawEvent {
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    session_id: String,
    #[serde(default)]
    cwd: String,
}

#[derive(Debug, Default)]
struct Decision {
    inject: String,
    skills: Vec<String>,
}

/// Run the hook for `host`. Always exits 0 (fail open).
pub fn run(host: Host) -> anyhow::Result<()> {
    let decision = decide(host).unwrap_or_default();
    let out = match host {
        Host::Claude => render_claude(&decision),
        Host::Opencode => render_opencode(&decision),
    };
    if !out.is_empty() {
        println!("{out}");
    }
    Ok(())
}

fn decide(host: Host) -> anyhow::Result<Decision> {
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf)?;
    let event: RawEvent = serde_json::from_str(&buf).unwrap_or_default();
    if event.prompt.trim().is_empty() {
        return Ok(Decision::default());
    }
    // Host-generated control payloads (task notifications, injected reminders) are
    // not user requests; embedding them as a query only surfaces noise matches the
    // model never acts on. Skip them outright. See `is_control_prompt`.
    if is_control_prompt(&event.prompt) {
        return Ok(Decision::default());
    }
    // A leading `/<name>` is an explicit skill invocation by the user; we must not
    // re-recommend the very skill they just ran (it always reads as an unused
    // false positive). Captured here, filtered out of `selected` below.
    let invoked_skill = slash_command_id(&event.prompt);

    let (mut cfg, file) = Config::load(host);
    telemetry::init(cfg.telemetry); // config.toml can enable telemetry (or the env var).
    let embedder = embed::build(&cfg.model)?;
    cfg.calibrate_to(embedder.as_ref());
    file.apply_cosine(&mut cfg); // user pin wins over embedder calibration.
    let idx = load_or_build_index(&cfg, embedder.as_ref(), host)?;
    if idx.skills.is_empty() {
        return Ok(Decision::default());
    }

    let session_path = paths::session_path(&event.session_id);
    let mut session = Session::load(&session_path);

    let query = embedder
        .embed(std::slice::from_ref(&event.prompt), EmbedKind::Query)?
        .remove(0);
    // Conversational context (Goal 3): the prior-turn window disambiguates a vague
    // prompt. Built from the *previous* turns before the current prompt is pushed
    // below; inert (no vector, no enrichment) unless the feature is enabled.
    let cvec = context::vector(embedder.as_ref(), &session.recent_prompts, &cfg).unwrap_or(None);
    // File-type channel: a file named in the prompt (or a recent turn that attached
    // one) boosts its skill — the directly-attributable context signal. Empty/no-IO
    // when the channel is off.
    let file_ids = if cfg.file_boost > 0.0 {
        let file_text = format!("{} {}", session.recent_prompts.join(" "), event.prompt);
        context::file_ids(&file_text)
    } else {
        std::collections::BTreeSet::new()
    };
    // Project-type channel: the working directory's manifest (`Cargo.toml`, ...)
    // boosts its ecosystem's skill — an ambient signal, so gated downstream on the
    // skill's own cosine. Empty/no-IO when the channel is off.
    let project_ids = if cfg.project_boost > 0.0 {
        context::project_ids(&event.cwd)
    } else {
        std::collections::BTreeSet::new()
    };
    let hits = rank::rank_all_ctx(
        &query,
        cvec.as_deref(),
        &file_ids,
        &project_ids,
        &event.prompt,
        &idx,
        &cfg,
    );
    let prompt_top = hits.iter().map(|h| h.cosine).fold(0.0_f32, f32::max);
    let rerank_query = context::rerank_query(
        &event.prompt,
        prompt_top,
        &session.recent_prompts,
        !file_ids.is_empty(),
        &cfg,
    );
    // Append this turn to the rolling window now that context has been built from
    // the prior turns. Persisted immediately (best-effort) so a later vague turn
    // sees it even if this turn injects nothing. No-op/no-IO when the feature is off.
    if cfg.context_depth > 0 {
        session.push_prompt(&event.prompt, cfg.context_depth);
        let _ = session.save(&session_path);
    }

    // With telemetry on, remember the active prompt now (before any early return)
    // so a self-load later in this conversation — including after a prompt that
    // injected nothing — can be tied back to it as a recall miss. One extra write
    // per prompt, paid only by telemetry users.
    if telemetry::enabled() {
        session.last_prompt = event.prompt.clone();
        let _ = session.save(&session_path);
    }
    // Stage 1.5 + 2 cascade — single-sourced in `pipeline`, shared with `ski why`
    // and `examples/eval`: a dominant lexical (BM25) winner injects directly unless
    // stage-1 has a confident lone dense winner; else the cross-encoder arbitrates
    // the ambiguous middle; else the cheap stage-1 cosine result stands. The winning
    // stage sets which confidence mapping the recs carry.
    let plan = pipeline::decide(&hits, &idx, &event.prompt, &rerank_query, &cfg);
    let stage = plan.stage;
    // `considered` snapshots the top of the winning stage's ranking *before* the gate
    // (id + raw stage score: BM25 for lexical, cosine-blend for stage 1, reranker
    // logit for stage 2). Logged on every prompt — including abstentions — so a later
    // native pick can be measured against where ski actually ranked it.
    let considered = match &plan.lexical {
        Some(win) => vec![(win.id.clone(), win.score)],
        None => top_considered(&plan.rows),
    };
    let mut selected = finalize(&plan.passed, stage, &cfg, &session);
    // Drop the skill the user invoked by slash command — recommending it back is
    // pure noise (the dominant false positive in telemetry: `/pickup` -> pickup).
    if let Some(invoked) = &invoked_skill {
        selected.retain(|r| &r.id != invoked);
    }
    if selected.is_empty() {
        // Nothing cleared the gate (or dedup/deny/slash removal emptied it). Record
        // the considered ranking anyway so a native pick on this prompt can be
        // scored against where ski ranked it — the abstention case is the whole
        // point of always logging.
        telemetry::record_recommend(
            &event.session_id,
            &event.prompt,
            stage,
            &considered,
            &[],
            &[],
            Some("below_gate"),
        );
        return Ok(Decision::default());
    }

    let strength = resolve_strength(cfg.directive_strength, host);
    // Escalate a lone, near-certain match from a directive pointer to a full body
    // inject: inline the SKILL.md so the model can't skip the Skill-tool round-trip.
    // Two co-relevant peers mean we are less certain, so they stay directives.
    let mode = inject_mode(&selected, &cfg);
    let (text, ids) = inject::build(&selected, &idx, mode, strength, cfg.char_budget);
    if text.is_empty() {
        telemetry::record_recommend(
            &event.session_id,
            &event.prompt,
            stage,
            &considered,
            &selected,
            &[],
            Some("empty_text"),
        );
        return Ok(Decision::default());
    }

    // Record each injected id at the confidence we displayed, so next turn's
    // score-aware dedup is accurate.
    let injected: Vec<(String, f32)> = ids
        .iter()
        .map(|id| (id.clone(), confidence_of(&selected, id)))
        .collect();
    for (id, conf) in &injected {
        session.mark_recommended(id, *conf);
    }
    let _ = session.save(&session_path); // best-effort: state IO never blocks.

    // Successful inject: `abstained` is None and `considered` carries the pre-gate
    // ranking so the injected ids can be located within it during analysis.
    telemetry::record_recommend(
        &event.session_id,
        &event.prompt,
        stage,
        &considered,
        &selected,
        &injected,
        None,
    );

    Ok(Decision {
        inject: text,
        skills: ids,
    })
}

/// How many top-ranked skills to snapshot into a `recommend` event's `considered`
/// list. Deep enough to locate a native pick that ski ranked but abstained on;
/// anything past this reads as "ski never surfaced it" in the comparison.
const CONSIDER_K: usize = 10;

/// Top-`CONSIDER_K` of a ranking as `(id, raw stage score)` for telemetry — the
/// chooser's pre-gate view. Hits arrive already sorted by descending score.
fn top_considered(hits: &[Hit]) -> Vec<(String, f32)> {
    hits.iter()
        .take(CONSIDER_K)
        .map(|h| (h.id.clone(), h.score))
        .collect()
}

/// Host-generated control payloads that arrive on the prompt channel but aren't
/// user requests — task-completion notifications and injected reminder blocks.
/// They are detected by a leading control tag so a genuine prompt that merely
/// quotes one in prose still injects normally. Embedding these as a query only
/// produces noise matches (telemetry: a `<task-notification>` blob surfaced
/// `skill-development`/`claude-automation-recommender`, both unused).
fn is_control_prompt(prompt: &str) -> bool {
    let p = prompt.trim_start();
    p.starts_with("<task-notification") || p.starts_with("<system-reminder")
}

/// The skill id a leading `/command` invokes, if the prompt is one. `/pickup` or
/// `/pickup keep going` -> `Some("pickup")`; plain prose or a bare `/` -> `None`.
/// A slash command is one leading token of command-name characters; anything else
/// (a path like `/etc/hosts`, a fraction) bails so only real invocations match.
fn slash_command_id(prompt: &str) -> Option<String> {
    let rest = prompt.trim_start().strip_prefix('/')?;
    let name = rest.split_whitespace().next()?;
    let ok = !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | ':'));
    // A namespaced command (`/plugin:skill`) maps to its trailing skill segment.
    ok.then(|| name.rsplit(':').next().unwrap_or(name).to_string())
}

/// The confidence of `id` within `recs` (the value we displayed for it).
fn confidence_of(recs: &[Rec], id: &str) -> f32 {
    recs.iter()
        .find(|r| r.id == id)
        .map(|r| r.confidence)
        .unwrap_or(0.0)
}

/// Load the persisted index; build it on first run so the hook works before an
/// explicit `ski index`. Rebuilds when the stored index was made by a different
/// embedder — its vectors live in another space (and often another dimension), so
/// cosine against a query from the current embedder would be meaningless. This
/// makes switching embedders (e.g. bag-of-words -> bge) self-healing in the hot
/// path rather than only on the next `SessionStart`.
fn load_or_build_index(
    cfg: &Config,
    embedder: &dyn embed::Embedder,
    host: Host,
) -> anyhow::Result<Index> {
    let path = paths::index_path(host);
    // Swallow a load error (corrupt/truncated index) and fall through to a
    // rebuild rather than propagating — a bad index file must not brick the hook
    // on every prompt. Mirrors the self-healing read in `session_start::reindex`.
    if let Some(idx) = Index::load(&path).ok().flatten() {
        if idx.model == embedder.id() {
            return Ok(idx);
        }
    }
    let skills = skill::discover(&cfg.roots)?;
    let idx = index::build(&skills, embedder, None)?;
    let _ = idx.save(&path);
    Ok(idx)
}

/// Apply the caller-side guardrails to the gate survivors from [`pipeline::decide`]:
/// drop denied skills, attach the winning stage's confidence, drop any the session's
/// score-aware dedup rejects, and cap at `max_skills`. The stage gate itself — the
/// absolute floor / relative margin / reranker thresholds / lexical dominance —
/// already ran in `pipeline`, measured against the global best *before* this dedup,
/// so re-injecting a prompt whose strong matches are already loaded falls silent
/// rather than scraping the weak tail.
///
/// Uniform across stages: for the lexical winner the `Stage::Lexical` mapping is a
/// fixed High-band confidence ([`confidence::LEXICAL_CONF`]) that ignores the score;
/// the reranker uses its logit; stage-1 uses its cosine blend.
fn finalize(passed: &[Hit], stage: Stage, cfg: &Config, session: &Session) -> Vec<Rec> {
    passed
        .iter()
        .filter(|h| !cfg.deny.contains(&h.id))
        .map(|h| Rec {
            confidence: confidence::of(h.score, stage, cfg),
            id: h.id.clone(),
        })
        .filter(|r| session.should_recommend(&r.id, r.confidence, confidence::HIGH))
        .take(cfg.max_skills)
        .collect()
}

/// Pick the inject shape for the chosen `recs`. Normally `cfg.inject_mode`, but a
/// lone match at/above `cfg.body_inject_min` confidence is escalated to
/// [`InjectMode::Body`] — the full `SKILL.md` is inlined so a near-certain skill
/// is applied, not merely pointed at. Requires `directive` mode (an explicit
/// `body` config already inlines everything) and exactly one rec (co-relevant
/// peers signal lower certainty and a heavier dump, so they stay directives).
fn inject_mode(recs: &[Rec], cfg: &Config) -> InjectMode {
    if cfg.inject_mode == InjectMode::Directive
        && recs.len() == 1
        && recs[0].confidence >= cfg.body_inject_min
    {
        InjectMode::Body
    } else {
        cfg.inject_mode
    }
}

/// Resolve [`Strength::Auto`] from the host: Claude has a strong native chooser
/// (a nudge suffices); opencode's local models need an imperative.
fn resolve_strength(strength: Strength, host: Host) -> Strength {
    match strength {
        Strength::Auto => match host {
            Host::Claude => Strength::Soft,
            Host::Opencode => Strength::Hard,
        },
        other => other,
    }
}

fn render_claude(d: &Decision) -> String {
    if d.inject.is_empty() {
        return String::new();
    }
    serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": d.inject,
        }
    })
    .to_string()
}

fn render_opencode(d: &Decision) -> String {
    serde_json::json!({ "skills": d.skills, "inject": d.inject }).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::session::Source;

    fn hit(id: &str, score: f32, keyword: f32) -> Hit {
        Hit {
            id: id.to_string(),
            name: id.to_string(),
            cosine: score - keyword,
            context: 0.0,
            file: 0.0,
            project: 0.0,
            keyword,
            phrase: 0.0,
            score,
        }
    }

    #[test]
    fn host_parse() {
        assert_eq!("claude".parse::<Host>().unwrap(), Host::Claude);
        assert_eq!("OpenCode".parse::<Host>().unwrap(), Host::Opencode);
        assert!("bogus".parse::<Host>().is_err());
    }

    #[test]
    fn raw_event_parses_claude_and_opencode_shapes() {
        let claude = r#"{"session_id":"s1","cwd":"/r","prompt":"hi","transcript_path":"/t"}"#;
        let ev: RawEvent = serde_json::from_str(claude).unwrap();
        assert_eq!(ev.prompt, "hi");
        assert_eq!(ev.session_id, "s1");

        let oc = r#"{"host":"opencode","session_id":"s2","cwd":"/r","prompt":"yo"}"#;
        let ev: RawEvent = serde_json::from_str(oc).unwrap();
        assert_eq!(ev.prompt, "yo");
        assert_eq!(ev.session_id, "s2");
    }

    #[test]
    fn strength_resolution() {
        assert_eq!(
            resolve_strength(Strength::Auto, Host::Claude),
            Strength::Soft
        );
        assert_eq!(
            resolve_strength(Strength::Auto, Host::Opencode),
            Strength::Hard
        );
        // Explicit settings pass through unchanged.
        assert_eq!(
            resolve_strength(Strength::Hard, Host::Claude),
            Strength::Hard
        );
    }

    /// The stage-1 cosine path as the hook runs it: gate in `pipeline`, then the
    /// caller-side `finalize` (deny/confidence/dedup/cap).
    fn select_cosine(hits: &[Hit], cfg: &Config, session: &Session) -> Vec<Rec> {
        finalize(
            &pipeline::cosine_passed(hits, cfg),
            Stage::Cosine,
            cfg,
            session,
        )
    }

    #[test]
    fn select_threshold_and_cap() {
        let cfg = Config::default(); // min 0.30, margin 0.15, max 2
        let session = Session::default();
        let hits = vec![
            hit("a", 0.90, 0.0),
            hit("b", 0.85, 0.0),
            hit("c", 0.84, 0.0), // within margin but over the cap
            hit("d", 0.10, 0.0), // below threshold
        ];
        let got: Vec<String> = select_cosine(&hits, &cfg, &session)
            .into_iter()
            .map(|h| h.id)
            .collect();
        assert_eq!(got, ["a", "b"]); // capped at 2, d dropped
    }

    #[test]
    fn select_skips_loaded_and_denied() {
        let cfg = Config {
            deny: vec!["a".to_string()],
            ..Default::default()
        };
        let mut session = Session::default();
        session.mark("b", Source::Model);
        let hits = vec![
            hit("a", 0.90, 0.0),
            hit("b", 0.85, 0.0),
            hit("c", 0.80, 0.0),
        ];
        let got: Vec<String> = select_cosine(&hits, &cfg, &session)
            .into_iter()
            .map(|h| h.id)
            .collect();
        assert_eq!(got, ["c"]); // a denied, b already loaded, c within margin
    }

    #[test]
    fn select_margin_drops_weak_tail() {
        let cfg = Config::default(); // margin 0.15
        let session = Session::default();
        // Both clear the 0.30 floor, but b is far below the 0.90 leader.
        let hits = vec![hit("a", 0.90, 0.0), hit("b", 0.50, 0.0)];
        let got: Vec<String> = select_cosine(&hits, &cfg, &session)
            .into_iter()
            .map(|h| h.id)
            .collect();
        assert_eq!(got, ["a"]);
    }

    #[test]
    fn select_repeat_falls_silent() {
        // The strong match was already recommended at high confidence, so its
        // repeat is suppressed; the rest are a weak tail measured against the
        // (still-global) leader, so nothing rides along either.
        let cfg = Config::default();
        let mut session = Session::default();
        session.mark_recommended("a", 0.95);
        let hits = vec![hit("a", 0.90, 0.0), hit("b", 0.50, 0.0)];
        assert!(select_cosine(&hits, &cfg, &session).is_empty());
    }

    #[test]
    fn select_repeats_on_rise_into_high() {
        // A skill shown earlier at medium confidence is re-recommended when a
        // later prompt makes it a strong match.
        let cfg = Config::default();
        let mut session = Session::default();
        session.mark_recommended("a", 0.60); // earlier: medium
        let hits = vec![hit("a", 0.90, 0.0)]; // now: cosine 0.90 -> high
        let got: Vec<String> = select_cosine(&hits, &cfg, &session)
            .into_iter()
            .map(|r| r.id)
            .collect();
        assert_eq!(got, ["a"]);
    }

    #[test]
    fn select_keeps_co_relevant_cluster() {
        let cfg = Config::default(); // margin 0.15
        let session = Session::default();
        let hits = vec![hit("a", 0.90, 0.0), hit("b", 0.80, 0.0)];
        let got: Vec<String> = select_cosine(&hits, &cfg, &session)
            .into_iter()
            .map(|h| h.id)
            .collect();
        assert_eq!(got, ["a", "b"]);
    }

    #[test]
    fn select_force_bypasses_threshold_on_keyword() {
        let cfg = Config {
            force: vec!["x".to_string()],
            ..Default::default()
        };
        let session = Session::default();
        // x is below threshold but forced with a keyword hit; y just below, no force.
        let hits = vec![hit("x", 0.1, 0.15), hit("y", 0.2, 0.0)];
        let got: Vec<String> = select_cosine(&hits, &cfg, &session)
            .into_iter()
            .map(|h| h.id)
            .collect();
        assert_eq!(got, ["x"]);
    }

    fn rec(id: &str, confidence: f32) -> Rec {
        Rec {
            id: id.to_string(),
            confidence,
        }
    }

    #[test]
    fn lone_near_certain_match_escalates_to_body() {
        let cfg = Config::default(); // directive mode, body_inject_min 0.92
        assert_eq!(inject_mode(&[rec("a", 0.95)], &cfg), InjectMode::Body);
    }

    #[test]
    fn body_escalation_needs_high_confidence() {
        let cfg = Config::default();
        // A High-band but not near-certain match stays a directive pointer.
        assert_eq!(inject_mode(&[rec("a", 0.85)], &cfg), InjectMode::Directive);
    }

    #[test]
    fn body_escalation_needs_a_lone_match() {
        let cfg = Config::default();
        // Two co-relevant near-certain peers stay directives (less certain; a
        // double body dump is too heavy).
        assert_eq!(
            inject_mode(&[rec("a", 0.95), rec("b", 0.95)], &cfg),
            InjectMode::Directive
        );
    }

    #[test]
    fn body_escalation_disabled_above_one() {
        let cfg = Config {
            body_inject_min: 1.1, // the documented "off" setting
            ..Default::default()
        };
        assert_eq!(inject_mode(&[rec("a", 0.99)], &cfg), InjectMode::Directive);
    }

    #[test]
    fn explicit_body_mode_is_unchanged() {
        let cfg = Config {
            inject_mode: InjectMode::Body,
            ..Default::default()
        };
        // A weak, multi-skill selection still inlines when the user pinned body mode.
        assert_eq!(
            inject_mode(&[rec("a", 0.2), rec("b", 0.2)], &cfg),
            InjectMode::Body
        );
    }

    #[test]
    fn control_prompts_detected() {
        assert!(is_control_prompt(
            "<task-notification>\n<task-id>x</task-id>\n</task-notification>"
        ));
        assert!(is_control_prompt(
            "  <system-reminder>foo</system-reminder>"
        ));
        // Genuine prompts that only mention a tag in prose still inject.
        assert!(!is_control_prompt(
            "explain the <task-notification> payload"
        ));
        assert!(!is_control_prompt("set up a python project"));
    }

    #[test]
    fn slash_command_id_extracts_name() {
        assert_eq!(slash_command_id("/pickup"), Some("pickup".into()));
        assert_eq!(
            slash_command_id("/pickup keep going"),
            Some("pickup".into())
        );
        assert_eq!(slash_command_id("  /handoff now"), Some("handoff".into()));
        // Namespaced command -> trailing skill segment.
        assert_eq!(
            slash_command_id("/caveman:caveman-commit"),
            Some("caveman-commit".into())
        );
        // Not slash commands.
        assert_eq!(slash_command_id("commit and push"), None);
        assert_eq!(slash_command_id("/etc/hosts is a path"), None);
        assert_eq!(slash_command_id("/"), None);
    }

    #[test]
    fn render_claude_empty_is_silent() {
        assert_eq!(render_claude(&Decision::default()), "");
    }

    #[test]
    fn render_claude_wraps_context() {
        let d = Decision {
            inject: "ctx".to_string(),
            skills: vec!["a".to_string()],
        };
        let v: serde_json::Value = serde_json::from_str(&render_claude(&d)).unwrap();
        assert_eq!(v["hookSpecificOutput"]["hookEventName"], "UserPromptSubmit");
        assert_eq!(v["hookSpecificOutput"]["additionalContext"], "ctx");
    }

    #[test]
    fn render_opencode_always_json() {
        let v: serde_json::Value = serde_json::from_str(&render_opencode(&Decision::default()))
            .expect("opencode output is always valid JSON");
        assert_eq!(v["inject"], "");
        assert!(v["skills"].as_array().unwrap().is_empty());
    }
}

//! In-process precision/recall harness for the injection decision.
//!
//! Builds the index once (one model load, unlike the per-prompt subprocess in
//! `tests/data/run-anthropic-prompts.sh`), runs the *real* two-stage decision
//! (stage-1 cosine, or stage-2 rerank when ambiguous) for every labelled prompt,
//! and reports a confusion matrix: recall on positives, false-positive rate on
//! negatives. Per-prompt score dumps (`-v`) expose the distributions the gate is
//! tuned against.
//!
//! Usage (point SKI_ROOTS at the eval index — colon-separated union is fine):
//!   SKI_ROOTS="/var/tmp/ski-eval/.claude/skills:$HOME/.claude/plugins/marketplaces/anthropic-agent-skills" \
//!     cargo run --example eval -- tests/data/popular_skills_prompts.tsv -v
//!
//! Labels: `<expected-skill-id>\t<kind>\t<prompt>`, `(none)` expects no injection.
//! `borderline` rows are reported but excluded from the headline FP/recall (they
//! are observe-only by design).

use ski::config::Config;
use ski::embed::{self, EmbedKind};
use ski::hook::Host;
use ski::rank::Hit;
use ski::{index, rank, rerank, skill};

struct Case {
    want: String, // "(none)" for a negative
    kind: String,
    prompt: String,
}

fn parse_cases(raw: &str) -> Vec<Case> {
    raw.lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .filter_map(|l| {
            let mut it = l.splitn(3, '\t');
            let want = it.next()?.trim().to_string();
            let kind = it.next()?.trim().to_string();
            let prompt = it.next()?.trim().to_string();
            if prompt.is_empty() {
                return None;
            }
            Some(Case { want, kind, prompt })
        })
        .collect()
}

/// The skills the hook would inject for `hits` — mirrors `hook::select` /
/// `hook::select_reranked` minus session dedup. Returns `(stage, injected_ids)`.
fn decide(
    hits: &[Hit],
    idx: &index::Index,
    prompt: &str,
    cfg: &Config,
) -> (&'static str, Vec<Hit>) {
    let reranked = rerank::is_ambiguous(hits, cfg)
        .then(|| rerank::rerank(hits, idx, prompt, cfg))
        .flatten();
    match reranked {
        Some(r) => {
            let kept: Vec<Hit> = rerank::passes(&r, cfg)
                .into_iter()
                .filter(|h| !cfg.deny.contains(&h.id))
                .take(cfg.max_skills)
                .collect();
            ("rerank", kept)
        }
        None => {
            let top = hits.first().map(|h| h.score).unwrap_or(0.0);
            let kept: Vec<Hit> = hits
                .iter()
                .filter(|h| !cfg.deny.contains(&h.id))
                .filter(|h| h.score >= cfg.min_similarity && h.score >= top - cfg.score_margin)
                .take(cfg.max_skills)
                .cloned()
                .collect();
            ("stage1", kept)
        }
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let verbose = args.iter().any(|a| a == "-v" || a == "--verbose");
    let path = args
        .iter()
        .find(|a| !a.starts_with('-'))
        .cloned()
        .unwrap_or_else(|| "tests/data/popular_skills_prompts.tsv".to_string());

    let raw = std::fs::read_to_string(&path)?;
    let cases = parse_cases(&raw);

    let (mut cfg, file) = Config::load(Host::Claude);
    // A/B affordance: override the phrase-channel boost (0.0 disables it) so the
    // same corpus can be scored with and without the channel in one rebuild.
    if let Ok(v) = std::env::var("SKI_PHRASE_BOOST") {
        cfg.phrase_boost = v.parse().expect("SKI_PHRASE_BOOST must be a float");
    }
    let skills = skill::discover(&cfg.roots)?;
    let embedder = embed::build(&cfg.model)?;
    cfg.calibrate_to(embedder.as_ref());
    file.apply_cosine(&mut cfg);
    let idx = index::build(&skills, embedder.as_ref(), None)?;
    eprintln!(
        "index: {} skills via {} | rerank_min {:.2} margin {:.2} | min_sim {:.2}",
        idx.skills.len(),
        idx.model,
        cfg.rerank_min,
        cfg.rerank_margin,
        cfg.min_similarity,
    );

    // Confusion counters. `borderline` rows are tallied separately (observe-only).
    let (mut tp, mut fn_, mut fp, mut tn) = (0u32, 0u32, 0u32, 0u32);
    let (mut n_pos, mut n_neg) = (0u32, 0u32);
    let mut fp_rows: Vec<String> = Vec::new();
    let mut fn_rows: Vec<String> = Vec::new();

    for c in &cases {
        let query = embedder
            .embed(std::slice::from_ref(&c.prompt), EmbedKind::Query)?
            .remove(0);
        let hits = rank::rank_all(&query, &c.prompt, &idx, &cfg);
        let (stage, injected) = decide(&hits, &idx, &c.prompt, &cfg);
        let ids: Vec<String> = injected.iter().map(|h| h.id.clone()).collect();
        let is_neg = c.want == "(none)";
        let observe_only = c.kind == "borderline";

        if verbose {
            let top: Vec<String> = hits
                .iter()
                .take(4)
                .map(|h| format!("{}={:.3}", h.id, h.score))
                .collect();
            let inj: Vec<String> = injected
                .iter()
                .map(|h| {
                    format!(
                        "{}=L{:.2}/cos{:.3}+kw{:.2}+ph{:.2}",
                        h.id, h.score, h.cosine, h.keyword, h.phrase
                    )
                })
                .collect();
            eprintln!(
                "[{:<10}] {:<7} inject=[{}]  top: {}  :: {}",
                c.kind,
                stage,
                inj.join(", "),
                top.join(", "),
                c.prompt,
            );
        }

        if observe_only {
            continue;
        }
        if is_neg {
            n_neg += 1;
            if injected.is_empty() {
                tn += 1;
            } else {
                fp += 1;
                fp_rows.push(format!(
                    "  FP [{:<10}] inject=[{}] :: {}",
                    c.kind,
                    ids.join(", "),
                    c.prompt
                ));
            }
        } else {
            n_pos += 1;
            if ids.iter().any(|id| id == &c.want) {
                tp += 1;
            } else {
                fn_ += 1;
                fn_rows.push(format!(
                    "  FN [{:<10}] want={} got=[{}] :: {}",
                    c.kind,
                    c.want,
                    ids.join(", "),
                    c.prompt
                ));
            }
        }
    }

    println!("\n=== eval: {} ===", path);
    println!(
        "positives {n_pos}: recall {tp}/{n_pos} ({:.0}%)   misses {fn_}",
        pct(tp, n_pos)
    );
    println!(
        "negatives {n_neg}: false-inject {fp}/{n_neg} ({:.0}%)   clean {tn}",
        pct(fp, n_neg)
    );
    if !fn_rows.is_empty() {
        println!("--- recall misses ---");
        fn_rows.iter().for_each(|r| println!("{r}"));
    }
    if !fp_rows.is_empty() {
        println!("--- false injections ---");
        fp_rows.iter().for_each(|r| println!("{r}"));
    }
    Ok(())
}

fn pct(n: u32, d: u32) -> f32 {
    if d == 0 {
        0.0
    } else {
        100.0 * n as f32 / d as f32
    }
}

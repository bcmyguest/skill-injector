//! In-process precision/recall harness for the injection decision.
//!
//! Builds the index once (one model load, unlike the per-prompt subprocess in
//! `tests/data/run-anthropic-prompts.sh`), runs the *real* two-stage decision
//! (stage-1 cosine, or stage-2 rerank when ambiguous) for every labelled prompt,
//! and reports a confusion matrix: recall on positives, false-positive rate on
//! negatives. It also reports the stage-1 retrieval ceiling — recall@`rerank_top_k`
//! and top-1 over positives, before any rerank/threshold gating — so you can tell
//! whether a miss is a retrieval failure (gold never reached the reranker) or a
//! ranking failure (gold was retrieved but the gate dropped it). Per-prompt score
//! dumps (`-v`) expose the distributions the gate is tuned against.
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
use ski::{context, index, lexical, rank, rerank, skill};

struct Case {
    want: String, // "(none)" for a negative
    kind: String,
    prompt: String,
    /// Optional prior-turn context (oldest-first), from a 4th `|`-separated TSV
    /// column. Empty for the single-prompt corpora.
    context: Vec<String>,
    /// Optional working directory, from a 5th TSV column, exercising the ambient
    /// project-type channel (`SKI_PROJECT_BOOST`). Empty for the prompt-only corpora.
    cwd: String,
}

fn parse_cases(raw: &str) -> Vec<Case> {
    raw.lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .filter_map(|l| {
            let mut it = l.splitn(5, '\t');
            let want = it.next()?.trim().to_string();
            let kind = it.next()?.trim().to_string();
            let prompt = it.next()?.trim().to_string();
            let context = it
                .next()
                .map(|c| {
                    c.split('|')
                        .map(|p| p.trim().to_string())
                        .filter(|p| !p.is_empty())
                        .collect()
                })
                .unwrap_or_default();
            let cwd = it.next().map(|c| c.trim().to_string()).unwrap_or_default();
            if prompt.is_empty() {
                return None;
            }
            Some(Case {
                want,
                kind,
                prompt,
                context,
                cwd,
            })
        })
        .collect()
}

/// The skills the hook would inject for `hits` — mirrors `hook::decide` (lexical
/// fast-path, then stage-1 cosine / stage-2 rerank) minus session dedup. Returns
/// `(stage, injected_hits)`.
fn decide(
    hits: &[Hit],
    idx: &index::Index,
    prompt: &str,
    rerank_query: &str,
    cfg: &Config,
) -> (&'static str, Vec<Hit>) {
    // Stage 1.5: a dominant lexical winner injects directly (unless stage-1 already
    // has a confident lone dense winner), skipping the reranker.
    if !rerank::confident_winner(hits, cfg) {
        if let Some(win) = lexical::dominant(prompt, idx, cfg) {
            let kept: Vec<Hit> = hits
                .iter()
                .filter(|h| h.id == win.id && !cfg.deny.contains(&h.id))
                .take(cfg.max_skills)
                .cloned()
                .collect();
            return ("lexical", kept);
        }
    }
    let reranked = rerank::is_ambiguous(hits, cfg)
        .then(|| rerank::rerank(hits, idx, rerank_query, cfg))
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
    // Context enrichment (Goal 3) is off by default; these env knobs activate and
    // tune it for one run, mirroring SKI_PHRASE_BOOST, so the same corpus can be
    // scored with and without conversational context.
    if let Ok(v) = std::env::var("SKI_CONTEXT_DEPTH") {
        cfg.context_depth = v.parse().expect("SKI_CONTEXT_DEPTH must be a usize");
    }
    if let Ok(v) = std::env::var("SKI_CONTEXT_WEIGHT") {
        cfg.context_weight = v.parse().expect("SKI_CONTEXT_WEIGHT must be a float");
    }
    if let Ok(v) = std::env::var("SKI_VAGUE_LO") {
        cfg.vague_lo = v.parse().expect("SKI_VAGUE_LO must be a float");
    }
    if let Ok(v) = std::env::var("SKI_VAGUE_HI") {
        cfg.vague_hi = v.parse().expect("SKI_VAGUE_HI must be a float");
    }
    if let Ok(v) = std::env::var("SKI_FILE_BOOST") {
        cfg.file_boost = v.parse().expect("SKI_FILE_BOOST must be a float");
    }
    if let Ok(v) = std::env::var("SKI_PROJECT_BOOST") {
        cfg.project_boost = v.parse().expect("SKI_PROJECT_BOOST must be a float");
    }
    // Reranker-gate sweep knobs: tune the stage-2 abstention floor/margin for one
    // run without editing config.toml (these are on the logit scale, untouched by
    // `calibrate_to`).
    if let Ok(v) = std::env::var("SKI_RERANK_MIN") {
        cfg.rerank_min = v.parse().expect("SKI_RERANK_MIN must be a float");
    }
    if let Ok(v) = std::env::var("SKI_RERANK_MARGIN") {
        cfg.rerank_margin = v.parse().expect("SKI_RERANK_MARGIN must be a float");
    }
    // Lexical fast-path (BM25 over description) sweep knobs: `lexical_min <= 0`
    // disables it, so the same corpus can be scored with and without the channel.
    if let Ok(v) = std::env::var("SKI_LEXICAL_MIN") {
        cfg.lexical_min = v.parse().expect("SKI_LEXICAL_MIN must be a float");
    }
    if let Ok(v) = std::env::var("SKI_LEXICAL_MARGIN") {
        cfg.lexical_margin = v.parse().expect("SKI_LEXICAL_MARGIN must be a float");
    }
    let skills = skill::discover(&cfg.roots)?;
    let embedder = embed::build(&cfg.model)?;
    cfg.calibrate_to(embedder.as_ref());
    file.apply_cosine(&mut cfg);
    let idx = index::build(&skills, embedder.as_ref(), None)?;
    eprintln!(
        "index: {} skills via {} | rerank_min {:.2} margin {:.2} | min_sim {:.2} | lexical_min {:.2} margin {:.2}",
        idx.skills.len(),
        idx.model,
        cfg.rerank_min,
        cfg.rerank_margin,
        cfg.min_similarity,
        cfg.lexical_min,
        cfg.lexical_margin,
    );

    // Confusion counters. `borderline` rows are tallied separately (observe-only).
    let (mut tp, mut fn_, mut fp, mut tn) = (0u32, 0u32, 0u32, 0u32);
    let (mut n_pos, mut n_neg) = (0u32, 0u32);
    let mut fp_rows: Vec<String> = Vec::new();
    let mut fn_rows: Vec<String> = Vec::new();
    // Stage-1 retrieval ceiling (pre-rerank), over positives only: recall@k is the
    // fraction whose gold skill survives into the top-`rerank_top_k` candidates the
    // reranker is fed (`rerank::rerank` takes exactly that many); top-1 is the
    // fraction already ranked first by hybrid score. recall@k ~100% means retrieval
    // is not the bottleneck and the problem is ranking within the retrieved set.
    let (mut recall_at_k, mut stage1_top1) = (0u32, 0u32);
    let mut recall_miss_rows: Vec<String> = Vec::new();

    for c in &cases {
        let query = embedder
            .embed(std::slice::from_ref(&c.prompt), EmbedKind::Query)?
            .remove(0);
        let cvec = context::vector(embedder.as_ref(), &c.context, &cfg)?;
        // File-type channel: scan this turn's prompt AND its prior context for named
        // files (a `.xlsx` etc.), mapping each to its skill.
        let file_text = format!("{} {}", c.context.join(" "), c.prompt);
        let file_ids = context::file_ids(&file_text);
        // Ambient project-type channel: the case's cwd (5th column) maps to its
        // ecosystem skill. Empty when the channel is off or no cwd is given.
        let project_ids = if cfg.project_boost > 0.0 {
            context::project_ids(&c.cwd)
        } else {
            std::collections::BTreeSet::new()
        };
        let hits = rank::rank_all_ctx(
            &query,
            cvec.as_deref(),
            &file_ids,
            &project_ids,
            &c.prompt,
            &idx,
            &cfg,
        );
        // The reranker reads text: enrich its query with the recent window when the
        // prompt is vague (same gate that lets the context vector contribute).
        let prompt_top = hits.iter().map(|h| h.cosine).fold(0.0_f32, f32::max);
        let rerank_query = context::rerank_query(
            &c.prompt,
            prompt_top,
            &c.context,
            !file_ids.is_empty(),
            &cfg,
        );
        let (stage, injected) = decide(&hits, &idx, &c.prompt, &rerank_query, &cfg);
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
                        "{}=L{:.2}/cos{:.3}+ctx{:.2}+file{:.2}+proj{:.2}+kw{:.2}+ph{:.2}",
                        h.id, h.score, h.cosine, h.context, h.file, h.project, h.keyword, h.phrase
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
            // Stage-1 ceiling: where does the gold skill land in the full hybrid
            // ranking, before any rerank/threshold gating?
            let rank = hits.iter().position(|h| h.id == c.want);
            if rank == Some(0) {
                stage1_top1 += 1;
            }
            if rank.is_some_and(|r| r < cfg.rerank_top_k) {
                recall_at_k += 1;
            } else {
                recall_miss_rows.push(format!(
                    "  R@k MISS [{:<10}] want={} stage-1 rank={} :: {}",
                    c.kind,
                    c.want,
                    rank.map_or_else(|| "absent".to_string(), |r| r.to_string()),
                    c.prompt
                ));
            }
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
    println!(
        "stage-1 (pre-rerank, k={}): recall@k {recall_at_k}/{n_pos} ({:.0}%)   top-1 {stage1_top1}/{n_pos} ({:.0}%)",
        cfg.rerank_top_k,
        pct(recall_at_k, n_pos),
        pct(stage1_top1, n_pos),
    );
    if !recall_miss_rows.is_empty() {
        println!(
            "--- stage-1 recall@k misses (gold below top-{}) ---",
            cfg.rerank_top_k
        );
        recall_miss_rows.iter().for_each(|r| println!("{r}"));
    }
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

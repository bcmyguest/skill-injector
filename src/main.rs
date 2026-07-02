//! `ski` CLI. Milestones 1–3 implement `index`, `why`, `hook`, `observe`, and
//! `session-start`.

use anyhow::Result;
use clap::{Parser, Subcommand};
use ski::config::Config;
use ski::embed::{self, EmbedKind};
use ski::hook::{self, Host};
use ski::index::{self, Index};
use ski::{
    context, history, init, lexical, observe, paths, pipeline, rank, rerank, session_start, skill,
};

#[derive(Parser)]
#[command(
    name = "ski",
    version,
    about = "skill-inject: local semantic skill auto-injection"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// (Re)build the persistent skill index.
    Index {
        /// Ignore the existing index and re-embed everything.
        #[arg(long)]
        rebuild: bool,
        /// Which host's skill library to index ('claude' or 'opencode').
        #[arg(long, default_value = "claude")]
        host: String,
    },
    /// Rank skills against a prompt and print scores (tuning aid).
    Why {
        /// The prompt (all trailing words are joined).
        #[arg(required = true)]
        prompt: Vec<String>,
        /// How many ranked skills to show.
        #[arg(long, default_value_t = 10)]
        top: usize,
        /// Which host's skill library to rank against ('claude' or 'opencode').
        #[arg(long, default_value = "claude")]
        host: String,
    },
    /// UserPromptSubmit hot-path: decide which skills to inject + emit the
    /// host's injection contract. Driven by the hooks, not run by hand.
    #[command(hide = true)]
    Hook {
        #[arg(long)]
        host: String,
    },
    /// PostToolUse: record skills the model loaded itself. Driven by the hooks,
    /// not run by hand.
    #[command(hide = true)]
    Observe {
        #[arg(long)]
        host: String,
    },
    /// SessionStart: incremental reindex + re-arm session state on compaction.
    /// Driven by the hooks, not run by hand.
    #[command(hide = true)]
    SessionStart {
        #[arg(long)]
        host: String,
    },
    /// Install ski's hooks/plugin for a host into your user config (the
    /// marketplace-free setup path).
    Init {
        /// Which host to set up ('claude' or 'opencode').
        host: String,
        /// Install user-wide (required; per-project install is not yet supported).
        #[arg(short = 'g', long)]
        global: bool,
    },
    /// Read the opt-in telemetry log (recommendations vs. actual use). Default is
    /// the aggregate readout; `--tail N` lists recent calls individually —
    /// recommendations (prompt, per-candidate confidence, used?) and self-loads
    /// (acted-on-rec vs. recall miss). `--compare` shows ski's ranking vs the
    /// native chooser's actual pick per prompt (agreed / near-miss / buried /
    /// absent) — where ski could get an edge. Empty unless hooks ran with
    /// `SKI_TELEMETRY=1`.
    History {
        /// List the most recent N events individually (recommendations and
        /// self-loads, newest last) instead of the aggregate.
        #[arg(long)]
        tail: Option<usize>,
        /// When listing, only events whose session id contains this substring.
        #[arg(long)]
        session: Option<String>,
        /// Show ski's ranking vs the native chooser's pick per prompt, classified
        /// by where ski ranked what the model actually used.
        #[arg(long)]
        compare: bool,
    },
    /// Wipe per-session dedup state (re-arm injection for testing).
    Clear {
        /// Also truncate the telemetry log.
        #[arg(long)]
        telemetry: bool,
    },
}

fn main() -> Result<()> {
    // Rust ignores SIGPIPE, so `ski why ... | head` used to panic with a
    // broken-pipe backtrace once head closed the pipe. Restore the default
    // die-quietly disposition, like other well-behaved CLI tools.
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Index { rebuild, host } => cmd_index(host.parse::<Host>()?, rebuild),
        Cmd::Why { prompt, top, host } => cmd_why(host.parse::<Host>()?, &prompt.join(" "), top),
        Cmd::Hook { host } => hook::run(host.parse::<Host>()?),
        Cmd::Observe { host } => observe::run(host.parse::<Host>()?),
        Cmd::SessionStart { host } => session_start::run(host.parse::<Host>()?),
        Cmd::Init { host, global } => init::run(host.parse::<Host>()?, global),
        Cmd::History {
            tail,
            session,
            compare,
        } => history::run(tail, session.as_deref(), compare),
        Cmd::Clear { telemetry } => history::clear(telemetry),
    }
}

fn cmd_index(host: Host, rebuild: bool) -> Result<()> {
    let (cfg, _file) = Config::load(host);
    let index_path = paths::index_path(host);
    let discovery = skill::discover_all(&cfg.roots);
    let embedder = embed::build(&cfg.model)?;
    let prev = if rebuild {
        None
    } else {
        Index::load(&index_path)?
    };
    let idx = index::build(&discovery.skills, embedder.as_ref(), prev.as_ref())?;
    idx.save(&index_path)?;
    println!(
        "indexed {} skills ({} dims) via '{}' -> {}",
        idx.skills.len(),
        idx.dim,
        idx.model,
        index_path.display()
    );
    report_skipped(&discovery.skipped);
    if idx.skills.is_empty() {
        eprintln!(
            "note: no skills found. Discovery roots for this host: {}",
            format_roots(&cfg.roots)
        );
        eprintln!("      install skills there, or point `roots` in config.toml / SKI_ROOTS at your library.");
    }
    Ok(())
}

/// One stderr line per unusable `SKILL.md` (capped), so "my skill never
/// injects" is diagnosable instead of silent.
fn report_skipped(skipped: &[(std::path::PathBuf, String)]) {
    const SHOW: usize = 10;
    if skipped.is_empty() {
        return;
    }
    eprintln!(
        "note: skipped {} SKILL.md file(s) with unusable frontmatter:",
        skipped.len()
    );
    for (path, reason) in skipped.iter().take(SHOW) {
        eprintln!("  {}: {reason}", path.display());
    }
    if skipped.len() > SHOW {
        eprintln!("  ... and {} more", skipped.len() - SHOW);
    }
}

fn format_roots(roots: &[std::path::PathBuf]) -> String {
    roots
        .iter()
        .map(|r| r.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn cmd_why(host: Host, prompt: &str, top: usize) -> Result<()> {
    let (mut cfg, file) = Config::load(host);
    let discovery = skill::discover_all(&cfg.roots);
    report_skipped(&discovery.skipped);
    let skills = discovery.skills;
    if skills.is_empty() {
        println!("no skills found in roots: {}", format_roots(&cfg.roots));
        return Ok(());
    }
    let embedder = embed::build(&cfg.model)?;
    cfg.calibrate_to(embedder.as_ref());
    file.apply_cosine(&mut cfg); // user pin wins over embedder calibration.

    // Reuse the persisted index instead of re-embedding the whole library on
    // every invocation: `why` is the interactive tuning aid, and paying the full
    // embed cost per call made it needlessly slow. Unchanged skills keep their
    // cached vectors (same id+hash+model, exactly like the hook); the refreshed
    // index is persisted back (best-effort) so the next `why`/hook reuses it too.
    let index_path = paths::index_path(host);
    let prev = Index::load(&index_path).ok().flatten();
    let idx = index::build(&skills, embedder.as_ref(), prev.as_ref())?;
    let _ = idx.save(&index_path);
    let query = embedder
        .embed(&[prompt.to_string()], EmbedKind::Query)?
        .remove(0);

    // Build the same channel inputs the hook does for a turn-1 prompt, so `why`
    // reproduces the live decision rather than a context-free approximation: the
    // file-type channel from the prompt text, the ambient project channel from the
    // working directory, and the context-enriched rerank query. There is no
    // conversation history here, so the context-blend vector is absent — exactly the
    // hook's first turn.
    let file_ids = if cfg.file_boost > 0.0 {
        context::file_ids(prompt)
    } else {
        std::collections::BTreeSet::new()
    };
    let project_ids = if cfg.project_boost > 0.0 {
        std::env::current_dir()
            .ok()
            .map(|d| context::project_ids(&d.to_string_lossy()))
            .unwrap_or_default()
    } else {
        std::collections::BTreeSet::new()
    };
    let hits = rank::rank_all_ctx(&query, None, &file_ids, &project_ids, prompt, &idx, &cfg);
    let prompt_top = hits.iter().map(|h| h.cosine).fold(0.0_f32, f32::max);
    let rerank_query = context::rerank_query(prompt, prompt_top, &[], !file_ids.is_empty(), &cfg);
    // Whether stage-1 has a confident lone dense winner (suppresses the lexical
    // fast-path), for the lexical block's verdict below.
    let dense_confident = rerank::confident_winner(&hits, &cfg);

    // The exact decision the hook would make, via the shared pipeline. A `*` marks a
    // row that would actually inject (cleared the winning stage's gate — for the
    // reranker that means `passes`, i.e. reranker thresholds *and* stage-1 agreement,
    // not just `rerank_min`; for the lexical fast-path, the dominant BM25 winner).
    let plan = pipeline::decide(&hits, &idx, prompt, &rerank_query, &cfg);
    // Star exactly what the hook would inject: gate survivors minus deny, capped at
    // `max_skills`. (The hook also applies session dedup, which `why` has no session
    // for — so a star is "would inject on a fresh conversation".)
    let injectable: std::collections::HashSet<&str> = plan
        .passed
        .iter()
        .filter(|h| !cfg.deny.contains(&h.id))
        .take(cfg.max_skills)
        .map(|h| h.id.as_str())
        .collect();

    println!(
        "stage {}  threshold {:.2}  prompt: {prompt:?}",
        pipeline::stage_label(plan.stage, &idx.model),
        plan.threshold
    );
    for h in plan.rows.iter().take(top) {
        let mark = if injectable.contains(h.id.as_str()) {
            "*"
        } else {
            " "
        };
        // Stage-1 channel attribution from the single-sourced breakdown, so `why`
        // can never omit a channel the score includes (it previously dropped
        // `project`). On a reranked row `h.score` is the logit; the breakdown still
        // shows the preserved stage-1 channels behind it.
        let parts = h
            .breakdown()
            .iter()
            .map(|(label, v)| format!("{label} {v:.3}"))
            .collect::<Vec<_>>()
            .join(" + ");
        println!("{mark} {:<26} score {:.3}  ({parts})", h.name, h.score);
    }

    // Lexical (BM25-over-description) channel detail: a dominant winner injects
    // directly, skipping the reranker (unless stage-1 has a confident lone dense
    // winner). Shown as a tuning aid — the top BM25 scores and whether the dominance
    // gate fires at the active `lexical_min` / `lexical_margin`.
    let lex = lexical::scores(prompt, &idx);
    if let Some(top) = lex.first() {
        let second = lex.get(1).map(|l| l.score).unwrap_or(0.0);
        let fires = lexical::dominant(prompt, &idx, &cfg).is_some();
        let verdict = if cfg.lexical_min <= 0.0 {
            "off".to_string()
        } else if dense_confident {
            "deferred (confident dense winner)".to_string()
        } else if fires {
            format!("FIRES -> {}", top.id)
        } else {
            "no dominant winner".to_string()
        };
        println!(
            "\nlexical(BM25): min {:.2} margin {:.2} -> {verdict}",
            cfg.lexical_min, cfg.lexical_margin,
        );
        println!(
            "  top gap {:.3} (#1 {:.3} - #2 {:.3})",
            top.score - second,
            top.score,
            second
        );
        for l in lex.iter().take(5) {
            println!("  {:<26} bm25 {:.3}", l.id, l.score);
        }
    }
    Ok(())
}

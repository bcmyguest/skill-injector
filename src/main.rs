//! `ski` CLI. Milestones 1–3 implement `index`, `why`, `hook`, `observe`, and
//! `session-start`.

use anyhow::Result;
use clap::{Parser, Subcommand};
use ski::config::Config;
use ski::embed::{self, EmbedKind};
use ski::hook::{self, Host};
use ski::index::{self, Index};
use ski::{observe, paths, rank, rerank, session_start, skill};

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
        prompt: Vec<String>,
        /// How many ranked skills to show.
        #[arg(long, default_value_t = 10)]
        top: usize,
        /// Which host's skill library to rank against ('claude' or 'opencode').
        #[arg(long, default_value = "claude")]
        host: String,
    },
    /// [stub, milestone 2] hook hot-path: decide + emit injection.
    Hook {
        #[arg(long)]
        host: String,
    },
    /// PostToolUse: record skills the model loaded itself.
    Observe {
        #[arg(long)]
        host: String,
    },
    /// SessionStart: incremental reindex + re-arm session state on compaction.
    SessionStart {
        #[arg(long)]
        host: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Index { rebuild, host } => {
            let host = host.parse::<Host>()?;
            cmd_index(&Config::for_host(host), host, rebuild)
        }
        Cmd::Why { prompt, top, host } => {
            let host = host.parse::<Host>()?;
            cmd_why(&Config::for_host(host), &prompt.join(" "), top)
        }
        Cmd::Hook { host } => hook::run(host.parse::<Host>()?),
        Cmd::Observe { host } => observe::run(host.parse::<Host>()?),
        Cmd::SessionStart { host } => session_start::run(host.parse::<Host>()?),
    }
}

fn cmd_index(cfg: &Config, host: Host, rebuild: bool) -> Result<()> {
    let index_path = paths::index_path(host);
    let skills = skill::discover(&cfg.roots)?;
    let embedder = embed::build(&cfg.model)?;
    let prev = if rebuild {
        None
    } else {
        Index::load(&index_path)?
    };
    let idx = index::build(&skills, embedder.as_ref(), prev.as_ref())?;
    idx.save(&index_path)?;
    println!(
        "indexed {} skills ({} dims) via '{}' -> {}",
        idx.skills.len(),
        idx.dim,
        idx.model,
        index_path.display()
    );
    Ok(())
}

fn cmd_why(cfg: &Config, prompt: &str, top: usize) -> Result<()> {
    let mut cfg = cfg.clone();
    let skills = skill::discover(&cfg.roots)?;
    if skills.is_empty() {
        println!("no skills found in roots: {:?}", cfg.roots);
        return Ok(());
    }
    let embedder = embed::build(&cfg.model)?;
    cfg.calibrate_to(embedder.as_ref());
    let idx = index::build(&skills, embedder.as_ref(), None)?;
    let query = embedder
        .embed(&[prompt.to_string()], EmbedKind::Query)?
        .remove(0);
    let hits = rank::rank_all(&query, prompt, &idx, &cfg);

    // Mirror the hook's decision so `why` (and the eval that drives it) reflects
    // the real pipeline: stage-1 cosine, or stage-2 reranker logits when the gate
    // fires. The `*` mark means the row cleared the threshold for whichever stage
    // produced it.
    let (rows, threshold, stage) = match rerank::is_ambiguous(&hits, &cfg)
        .then(|| rerank::rerank(&hits, &idx, prompt, &cfg))
        .flatten()
    {
        Some(reranked) => (reranked, cfg.rerank_min, "rerank:turbo".to_string()),
        None => (hits, cfg.min_similarity, format!("stage1:{}", idx.model)),
    };

    println!("stage {stage}  threshold {threshold:.2}  prompt: {prompt:?}");
    for h in rows.iter().take(top) {
        let mark = if h.score >= threshold { "*" } else { " " };
        println!(
            "{mark} {:<26} score {:.3}  (cos {:.3} + kw {:.3})",
            h.name, h.score, h.cosine, h.keyword
        );
    }
    Ok(())
}

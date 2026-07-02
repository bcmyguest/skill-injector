//! Stage-1.5 lexical channel: BM25 over full skill descriptions.
//!
//! The dense bi-encoder embeds the (short, curated) description and the tiny
//! cross-encoder reranker both miss lexically-obvious *indirect* matches — a
//! prompt like "turn this sales spreadsheet into a chart" whose gold skill's
//! description literally contains "spreadsheet", "chart", "formulas" but whose
//! cosine sits in the muddy ~0.6 band and whose reranker logit falls below the
//! abstention floor (so ski says nothing; the host's own chooser misses it too).
//! BM25 over the description *prose* ranks those #1 reliably where the embeddings
//! do not, because the discriminating vocabulary is literally in the prose.
//!
//! This is a high-precision **fast-path**, not another additive score term: a
//! *dominant* BM25 winner (clears [`Config::lexical_min`] in absolute score AND
//! beats the runner-up by [`Config::lexical_margin`]) is injected directly,
//! skipping the reranker — mirroring the existing "confident lone winner skips
//! rerank" gate ([`crate::rerank::confident_winner`]) but keyed on lexical
//! certainty, which is reliable exactly where the bi-encoder cosine is not. A
//! plain stage-1 score boost would not work: the reranker overwrites stage-1
//! `score` with its logit and the agreement gate in [`crate::rerank::passes`]
//! then rejects the still-sub-floor cosine.
//!
//! The fast-path only fires when stage-1 has *no* confident lone dense winner
//! (the caller gates on [`crate::rerank::confident_winner`]); a strong dense
//! match is never overridden. Off by default ([`Config::lexical_min`] `<= 0`).

use crate::config::Config;
use crate::index::Index;
use crate::rank::cmp_score_desc;
use crate::text::content_tokens;
use std::collections::HashMap;

/// BM25 term-frequency saturation. Standard Lucene/Elasticsearch default.
const K1: f32 = 1.2;
/// BM25 length-normalization. Standard default.
const B: f32 = 0.75;
/// Minimum distinct query content-tokens that must appear in the winning
/// description for a lexical dominance to count. BM25 rewards a lone high-IDF term
/// heavily, so a single rare word shared with an otherwise-unrelated description
/// ("anthropic" -> brand-guidelines, "keyboard" -> accessibility) spikes a dominant
/// score that is pure noise. A genuine indirect match overlaps the description on
/// *several* terms ("chart" + "spreadsheet" + "formula"); requiring two independent
/// hits is the precision gate that separates the rescues from the false injections.
const MIN_TERM_OVERLAP: usize = 2;

/// A skill's BM25 score against the prompt.
#[derive(Clone, Debug)]
pub struct Lex {
    pub id: String,
    pub score: f32,
}

/// BM25(prompt, description) for every skill, sorted by descending score.
///
/// Corpus statistics (document frequency, average length) are computed inline
/// from the in-memory index every call — the descriptions are already stored, so
/// no index-schema change is needed, and the corpus (tens to low-hundreds of
/// short descriptions) is cheap to tokenize. Both prompt and descriptions are
/// tokenized with [`content_tokens`] (stopword-stripped), so glue words neither
/// inflate a score nor pad a document's length.
pub fn scores(prompt: &str, idx: &Index) -> Vec<Lex> {
    // Query terms as a set: a short prompt's repeats carry no extra signal, and
    // de-duping keeps a doubled word from double-counting one description's hit.
    let mut q: Vec<String> = content_tokens(prompt);
    q.sort();
    q.dedup();
    if q.is_empty() || idx.skills.is_empty() {
        return Vec::new();
    }

    // Per-document term frequencies and lengths (content tokens only).
    let docs: Vec<HashMap<String, u32>> = idx
        .skills
        .iter()
        .map(|e| {
            let mut tf: HashMap<String, u32> = HashMap::new();
            for t in content_tokens(&e.description) {
                *tf.entry(t).or_insert(0) += 1;
            }
            tf
        })
        .collect();
    let lens: Vec<f32> = docs
        .iter()
        .map(|d| d.values().sum::<u32>() as f32)
        .collect();
    let n = docs.len() as f32;
    let avgdl = (lens.iter().sum::<f32>() / n).max(1.0);

    // Document frequency, only for the query terms (all the IDF we need).
    let idf: HashMap<&str, f32> = q
        .iter()
        .map(|t| {
            let df = docs.iter().filter(|d| d.contains_key(t)).count() as f32;
            // Lucene BM25 IDF: ln(1 + (N - df + 0.5)/(df + 0.5)) — always >= 0, so a
            // term in most descriptions never drags a score negative.
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            (t.as_str(), idf)
        })
        .collect();

    let mut out: Vec<Lex> = idx
        .skills
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let dl = lens[i];
            let mut score = 0.0f32;
            for t in &q {
                let f = *docs[i].get(t).unwrap_or(&0) as f32;
                if f == 0.0 {
                    continue;
                }
                let denom = f + K1 * (1.0 - B + B * dl / avgdl);
                score += idf[t.as_str()] * (f * (K1 + 1.0)) / denom;
            }
            Lex {
                id: e.id.clone(),
                score,
            }
        })
        .collect();
    out.sort_by(|a, b| cmp_score_desc(a.score, b.score));
    out
}

/// The dominant lexical winner, if one exists: the top-scoring skill, provided it
/// clears `cfg.lexical_min` in absolute BM25 *and* beats the runner-up by at least
/// `cfg.lexical_margin`. `None` when the channel is off (`lexical_min <= 0`), the
/// prompt has no content tokens, or no skill stands clearly apart — the margin is
/// what makes this high-precision, so a cluster of near-equal descriptions abstains
/// and defers to the reranker.
pub fn dominant(prompt: &str, idx: &Index, cfg: &Config) -> Option<Lex> {
    if cfg.lexical_min <= 0.0 {
        return None;
    }
    let ranked = scores(prompt, idx);
    let top = ranked.first()?;
    if top.score < cfg.lexical_min {
        return None;
    }
    // Dominance is *over peers*: with no runner-up (a single-skill library) the
    // margin gate would pass vacuously and the fast-path would fire on the sole
    // skill for any two-term overlap. No peers -> no dominance -> defer.
    let second = ranked.get(1).map(|l| l.score)?;
    if top.score - second < cfg.lexical_margin {
        return None;
    }
    // Precision guard: reject a dominance carried by a single high-IDF term. Count
    // the distinct query content-tokens that actually appear in the winner's
    // description; a real indirect match overlaps on at least `MIN_TERM_OVERLAP`.
    let mut q: Vec<String> = content_tokens(prompt);
    q.sort();
    q.dedup();
    let win_terms: std::collections::HashSet<String> = idx
        .skills
        .iter()
        .find(|e| e.id == top.id)
        .map(|e| content_tokens(&e.description).into_iter().collect())
        .unwrap_or_default();
    let overlap = q.iter().filter(|t| win_terms.contains(*t)).count();
    if overlap < MIN_TERM_OVERLAP {
        return None;
    }
    Some(top.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Entry;

    fn entry(id: &str, description: &str) -> Entry {
        Entry {
            id: id.to_string(),
            name: id.to_string(),
            description: description.to_string(),
            path: String::new(),
            keywords: Vec::new(),
            trigger_phrases: Vec::new(),
            hash: String::new(),
            embedding: Vec::new(),
        }
    }

    fn index_of(entries: Vec<Entry>) -> Index {
        Index {
            model: "test".into(),
            dim: 0,
            skills: entries,
        }
    }

    fn cfg(min: f32, margin: f32) -> Config {
        Config {
            lexical_min: min,
            lexical_margin: margin,
            ..Default::default()
        }
    }

    #[test]
    fn ranks_description_vocabulary_match_first() {
        // The xlsx-style case: the indirect prompt's vocabulary lives in the gold
        // description's prose, not in the others'.
        let idx = index_of(vec![
            entry(
                "xlsx",
                "create and edit spreadsheets, compute formulas, build charts",
            ),
            entry("pdf", "merge split and extract text from pdf documents"),
            entry(
                "docx",
                "create and edit word documents with headings and tables",
            ),
        ]);
        let ranked = scores(
            "turn this sales spreadsheet into a chart with formulas",
            &idx,
        );
        assert_eq!(ranked[0].id, "xlsx");
        assert!(ranked[0].score > ranked[1].score);
    }

    #[test]
    fn dominant_requires_absolute_floor() {
        // BM25 has no stemming, so the query token must appear verbatim in the
        // description (singular "spreadsheet" here, matching the prompt).
        let idx = index_of(vec![
            entry("xlsx", "edit a spreadsheet, charts and formulas"),
            entry("pdf", "pdf documents"),
        ]);
        // A real match exists and stands apart, but an absurd floor rejects it.
        assert!(dominant("edit my spreadsheet", &idx, &cfg(100.0, 0.5)).is_none());
        // A reachable floor accepts it.
        let win = dominant("edit my spreadsheet", &idx, &cfg(0.5, 0.5)).unwrap();
        assert_eq!(win.id, "xlsx");
    }

    #[test]
    fn dominant_requires_margin_over_runner_up() {
        // Two descriptions share the query term equally -> no clear winner -> abstain.
        let idx = index_of(vec![
            entry("a", "process the report data"),
            entry("b", "process the report data"),
        ]);
        assert!(dominant("process the report", &idx, &cfg(0.1, 0.5)).is_none());
    }

    #[test]
    fn dominant_rejects_single_term_match() {
        // The false-injection pattern: one rare query word ("anthropic") hits one
        // otherwise-unrelated description and BM25 spikes it into a dominant winner.
        // A single-term overlap must abstain even though the floor and margin pass.
        let idx = index_of(vec![
            entry(
                "brand-guidelines",
                "apply anthropic brand colors to artifacts",
            ),
            entry("pdf", "merge and split pdf documents"),
        ]);
        assert!(dominant(
            "who founded anthropic and in what year",
            &idx,
            &cfg(0.1, 0.1)
        )
        .is_none());
        // Two real overlapping terms clear the guard.
        let idx2 = index_of(vec![
            entry("xlsx", "edit a spreadsheet, charts and formulas"),
            entry("pdf", "pdf documents"),
        ]);
        let win = dominant("edit the spreadsheet formulas", &idx2, &cfg(0.1, 0.1)).unwrap();
        assert_eq!(win.id, "xlsx");
    }

    #[test]
    fn single_skill_library_is_never_dominant() {
        // With no runner-up the margin gate would pass vacuously; the sole skill
        // must not ride the fast-path on any two-term overlap.
        let idx = index_of(vec![entry(
            "xlsx",
            "edit a spreadsheet, charts and formulas",
        )]);
        assert!(dominant("edit the spreadsheet formulas", &idx, &cfg(0.1, 0.1)).is_none());
    }

    #[test]
    fn dominant_off_when_min_non_positive() {
        let idx = index_of(vec![entry("xlsx", "spreadsheets charts")]);
        assert!(dominant("spreadsheet", &idx, &cfg(0.0, 0.5)).is_none());
    }

    #[test]
    fn empty_prompt_or_index_is_none() {
        let idx = index_of(vec![entry("xlsx", "spreadsheets")]);
        assert!(scores("", &idx).is_empty());
        assert!(scores("the an of to", &idx).is_empty()); // all stopwords
        assert!(scores("spreadsheet", &index_of(vec![])).is_empty());
    }
}

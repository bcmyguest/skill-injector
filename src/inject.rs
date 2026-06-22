//! Turn ranked hits into the text injected into the model's context.
//!
//! Two shapes (`Config::inject_mode`):
//! - **directive** — a short pointer to the skill (name + description + path)
//!   that tells the model to invoke it via the `Skill` tool, not to read the
//!   file. Forcefulness set by [`Strength`].
//! - **body** — the `SKILL.md` content inlined directly, no model agency.
//!
//! Either way the total stays under `char_budget`: blocks are added until the
//! next one would overflow (the first block is always allowed so a single large
//! skill still gets injected).

use crate::confidence::{self, Band};
use crate::config::{InjectMode, Strength};
use crate::index::{Entry, Index};
use std::fs;

/// A skill chosen for injection: its id and the confidence we'll display. The
/// hook computes these (stage-appropriate confidence + dedup) and hands them to
/// [`build`]; tests construct them directly.
#[derive(Clone, Debug)]
pub struct Rec {
    pub id: String,
    pub confidence: f32,
}

/// Build the injection text for `recs` and return it alongside the ids actually
/// injected (after the char budget is applied). `strength` must already be
/// resolved (not [`Strength::Auto`]); `Auto` is treated as `Soft`.
pub fn build(
    recs: &[Rec],
    index: &Index,
    mode: InjectMode,
    strength: Strength,
    char_budget: usize,
) -> (String, Vec<String>) {
    let mut blocks: Vec<String> = Vec::new();
    let mut ids: Vec<String> = Vec::new();
    let mut used = 0usize;

    for r in recs {
        let Some(entry) = index.get(&r.id) else {
            continue;
        };
        let block = match mode {
            InjectMode::Directive => directive_block(entry, strength, r.confidence),
            InjectMode::Body => body_block(entry),
        };
        if !blocks.is_empty() && used + block.len() > char_budget {
            break;
        }
        used += block.len();
        blocks.push(block);
        ids.push(r.id.clone());
    }

    if blocks.is_empty() {
        return (String::new(), ids);
    }

    let header = match mode {
        InjectMode::Directive => {
            "ski matched these skills to your request — a dedicated retrieval+rerank pass, \
             separate from and complementary to the host's own skill selection. Invoke \
             fitting ones by name via the `Skill` tool; do not Read the files:"
        }
        InjectMode::Body => "Skill instructions relevant to this request are included below:",
    };
    (format!("{header}\n\n{}", blocks.join("\n\n")), ids)
}

/// One directive line: a distinctive `SkillRecommendation(name)` token, the
/// description, then a verb scaled by confidence band (and harder under
/// [`Strength::Hard`] for weak local choosers).
///
/// The raw confidence is deliberately **not** shown to the model: every line
/// here has already cleared a precision gate, so a bare decimal (`0.36`) only
/// invites the model to anchor on it and discount a genuine match. The band —
/// expressed in the verb's forcefulness — is the honest, right-grained signal;
/// the exact value still rides into telemetry via [`Rec::confidence`].
fn directive_block(entry: &Entry, strength: Strength, confidence: f32) -> String {
    let verb = match (strength, confidence::band(confidence)) {
        (Strength::Hard, Band::High) => "you MUST invoke it before responding.",
        (Strength::Hard, _) => "invoke it before responding if it fits.",
        (_, Band::High) => "invoke it.",
        (_, Band::Medium) => "invoke it if it fits.",
        (_, Band::Low) => "consider invoking it.",
    };
    format!(
        "- SkillRecommendation(`{}`): {} — {}",
        entry.name, entry.description, verb
    )
}

fn body_block(entry: &Entry) -> String {
    let body = fs::read_to_string(&entry.path)
        .map(|c| strip_frontmatter(&c).to_string())
        .unwrap_or_else(|_| entry.description.clone());
    format!("<skill name=\"{}\">\n{}\n</skill>", entry.name, body.trim())
}

/// Drop a leading `--- ... ---` YAML frontmatter block, returning the body.
fn strip_frontmatter(content: &str) -> &str {
    let trimmed = content.trim_start();
    let Some(rest) = trimmed.strip_prefix("---") else {
        return content;
    };
    // The opening fence must be its own line, and we need a closing fence.
    if !rest.starts_with('\n') && !rest.starts_with("\r\n") {
        return content;
    }
    match rest.find("\n---") {
        Some(end) => {
            let after = &rest[end + "\n---".len()..];
            // Skip to the end of the closing fence line, then to the body.
            after
                .find('\n')
                .map(|nl| after[nl + 1..].trim_start_matches(['\n', '\r']))
                .unwrap_or("")
        }
        None => content,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, name: &str, path: &str) -> Entry {
        Entry {
            id: id.to_string(),
            name: name.to_string(),
            description: "does a thing".to_string(),
            path: path.to_string(),
            keywords: vec![],
            trigger_phrases: vec![],
            hash: "0".to_string(),
            embedding: vec![],
        }
    }

    fn index_of(entries: Vec<Entry>) -> Index {
        Index {
            model: "test".to_string(),
            dim: 0,
            skills: entries,
        }
    }

    fn rec(id: &str, confidence: f32) -> Rec {
        Rec {
            id: id.to_string(),
            confidence,
        }
    }

    #[test]
    fn directive_soft_vs_hard() {
        let idx = index_of(vec![entry("a", "alpha", "/p/SKILL.md")]);
        let (soft, _) = build(
            &[rec("a", 0.91)],
            &idx,
            InjectMode::Directive,
            Strength::Soft,
            6000,
        );
        let (hard, _) = build(
            &[rec("a", 0.91)],
            &idx,
            InjectMode::Directive,
            Strength::Hard,
            6000,
        );
        // The distinctive token is shown; the raw confidence and source path are not.
        assert!(soft.contains("SkillRecommendation(`alpha`)"));
        assert!(!soft.contains("0.91"));
        assert!(!soft.contains("/p/SKILL.md"));
        assert!(!soft.contains("MUST"));
        assert!(hard.contains("MUST")); // high-confidence hard directive
    }

    #[test]
    fn directive_verb_scales_with_band() {
        let idx = index_of(vec![entry("a", "alpha", "/p/SKILL.md")]);
        let soft = |c| {
            build(
                &[rec("a", c)],
                &idx,
                InjectMode::Directive,
                Strength::Soft,
                6000,
            )
            .0
        };
        assert!(soft(0.95).contains("— invoke it."));
        assert!(soft(0.70).contains("invoke it if it fits."));
        assert!(soft(0.40).contains("consider invoking it."));
    }

    #[test]
    fn char_budget_caps_but_allows_first() {
        let idx = index_of(vec![
            entry("a", "alpha", "/p/a/SKILL.md"),
            entry("b", "bravo", "/p/b/SKILL.md"),
        ]);
        // Budget of 1 still emits the first block, never the second.
        let (text, ids) = build(
            &[rec("a", 0.9), rec("b", 0.9)],
            &idx,
            InjectMode::Directive,
            Strength::Soft,
            1,
        );
        assert_eq!(ids, ["a"]);
        assert!(text.contains("alpha") && !text.contains("bravo"));
    }

    #[test]
    fn unknown_id_skipped() {
        let idx = index_of(vec![entry("a", "alpha", "/p/SKILL.md")]);
        let (_, ids) = build(
            &[rec("missing", 0.9), rec("a", 0.9)],
            &idx,
            InjectMode::Directive,
            Strength::Soft,
            6000,
        );
        assert_eq!(ids, ["a"]);
    }

    #[test]
    fn empty_recs_yield_empty() {
        let idx = index_of(vec![]);
        let (text, ids) = build(&[], &idx, InjectMode::Directive, Strength::Soft, 6000);
        assert!(text.is_empty() && ids.is_empty());
    }

    #[test]
    fn strip_frontmatter_removes_yaml() {
        let md = "---\nname: x\ndescription: y\n---\n\nReal body here.\n";
        assert_eq!(strip_frontmatter(md), "Real body here.\n");
    }

    #[test]
    fn strip_frontmatter_passthrough_without_block() {
        let md = "no frontmatter\njust text\n";
        assert_eq!(strip_frontmatter(md), md);
    }

    #[test]
    fn strip_frontmatter_handles_unterminated() {
        let md = "---\nname: x\nno closing fence\n";
        assert_eq!(strip_frontmatter(md), md);
    }
}

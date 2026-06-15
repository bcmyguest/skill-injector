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

use crate::config::{InjectMode, Strength};
use crate::index::{Entry, Index};
use crate::rank::Hit;
use std::fs;

/// Build the injection text for `hits` and return it alongside the ids actually
/// injected (after the char budget is applied). `strength` must already be
/// resolved (not [`Strength::Auto`]); `Auto` is treated as `Soft`.
pub fn build(
    hits: &[Hit],
    index: &Index,
    mode: InjectMode,
    strength: Strength,
    char_budget: usize,
) -> (String, Vec<String>) {
    let mut blocks: Vec<String> = Vec::new();
    let mut ids: Vec<String> = Vec::new();
    let mut used = 0usize;

    for h in hits {
        let Some(entry) = index.get(&h.id) else {
            continue;
        };
        let block = match mode {
            InjectMode::Directive => directive_block(entry, strength),
            InjectMode::Body => body_block(entry),
        };
        if !blocks.is_empty() && used + block.len() > char_budget {
            break;
        }
        used += block.len();
        blocks.push(block);
        ids.push(h.id.clone());
    }

    if blocks.is_empty() {
        return (String::new(), ids);
    }

    let header = match mode {
        InjectMode::Directive => {
            "The following skills are likely relevant to this request. Invoke each \
             relevant one with the `Skill` tool using its name below — do NOT just \
             read the file, as reading bypasses skill loading and tracking. (Pass the \
             bare name, or your harness's `plugin:name` form if it requires one.) Read \
             the listed path directly only if your environment has no `Skill` tool:"
        }
        InjectMode::Body => "Skill instructions relevant to this request are included below:",
    };
    (format!("{header}\n\n{}", blocks.join("\n\n")), ids)
}

fn directive_block(entry: &Entry, strength: Strength) -> String {
    match strength {
        Strength::Hard => format!(
            "- **{}** — {}\n  You MUST invoke this skill before responding: `Skill` with skill `{}` (source: {})",
            entry.name, entry.description, entry.name, entry.path
        ),
        // Soft, and Auto defensively (callers resolve Auto upstream).
        _ => format!(
            "- **{}** — {}\n  If relevant, invoke it: `Skill` with skill `{}` (source: {})",
            entry.name, entry.description, entry.name, entry.path
        ),
    }
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

    fn hit(id: &str) -> Hit {
        Hit {
            id: id.to_string(),
            name: id.to_string(),
            cosine: 0.5,
            keyword: 0.0,
            score: 0.5,
        }
    }

    #[test]
    fn directive_soft_vs_hard() {
        let idx = index_of(vec![entry("a", "alpha", "/p/SKILL.md")]);
        let (soft, _) = build(
            &[hit("a")],
            &idx,
            InjectMode::Directive,
            Strength::Soft,
            6000,
        );
        let (hard, _) = build(
            &[hit("a")],
            &idx,
            InjectMode::Directive,
            Strength::Hard,
            6000,
        );
        assert!(soft.contains("alpha") && soft.contains("/p/SKILL.md"));
        assert!(!soft.contains("MUST"));
        assert!(hard.contains("MUST"));
    }

    #[test]
    fn char_budget_caps_but_allows_first() {
        let idx = index_of(vec![
            entry("a", "alpha", "/p/a/SKILL.md"),
            entry("b", "bravo", "/p/b/SKILL.md"),
        ]);
        // Budget of 1 still emits the first block, never the second.
        let (text, ids) = build(
            &[hit("a"), hit("b")],
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
            &[hit("missing"), hit("a")],
            &idx,
            InjectMode::Directive,
            Strength::Soft,
            6000,
        );
        assert_eq!(ids, ["a"]);
    }

    #[test]
    fn empty_hits_yield_empty() {
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

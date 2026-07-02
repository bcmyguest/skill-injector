//! Skill discovery and `SKILL.md` frontmatter parsing.

use crate::text::{fnv1a_64, tokenize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct Skill {
    /// Unique id (the skill's declared `name`).
    pub id: String,
    pub name: String,
    pub description: String,
    /// First few prose lines of the body — dense topical signal that disambiguates
    /// confusable descriptions without the dilution of the full document. Embedded
    /// alongside `description`; see [`Skill::doc_text`].
    pub body_head: String,
    /// Keywords for the hybrid keyword boost: explicit `keywords`/`aliases`
    /// frontmatter, plus tokens derived from the name.
    pub keywords: Vec<String>,
    /// Multi-word trigger phrases mined from the description's quoted spans (the
    /// literal wording a skill says to invoke it on, e.g. `"find that online"`).
    /// Each is normalized to its content tokens; the ranker boosts a skill when a
    /// prompt contains all of a phrase's tokens. See [`extract_phrases`].
    pub trigger_phrases: Vec<String>,
    pub path: PathBuf,
    /// Content hash for index cache invalidation.
    pub hash: String,
}

impl Skill {
    /// Text fed to the document embedder: the curated description plus the body
    /// head. Keeping them together gives the bi-encoder more topical signal than
    /// the one-line description alone.
    pub fn doc_text(&self) -> String {
        if self.body_head.is_empty() {
            self.description.clone()
        } else {
            format!("{}\n{}", self.description, self.body_head)
        }
    }
}

/// Deepest subdirectory nesting `collect` will descend into per root, mirroring
/// `context::PROJECT_WALK_LEVELS`. Bounds the walk against a pathologically deep
/// real tree; a symlink loop is already safe (kernel `ELOOP`), this guards the
/// non-symlink case.
const MAX_WALK_DEPTH: usize = 12;

/// Walk `roots` and parse every `SKILL.md` found. A single unreadable or
/// malformed file is skipped (with a `SKI_DEBUG`-gated note) rather than
/// aborting discovery for every other skill.
pub fn discover(roots: &[PathBuf]) -> anyhow::Result<Vec<Skill>> {
    let mut files = Vec::new();
    for r in roots {
        collect(r, &mut files, 0);
    }
    files.sort();
    files.dedup();

    let mut out = Vec::new();
    for f in &files {
        match parse_file(f) {
            Ok(Some(s)) => out.push(s),
            Ok(None) => {}
            Err(e) => {
                if std::env::var_os("SKI_DEBUG").is_some() {
                    eprintln!("ski: skipping unreadable skill file {}: {e}", f.display());
                }
            }
        }
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out.dedup_by(|a, b| a.id == b.id);
    Ok(out)
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>, depth: usize) {
    if depth >= MAX_WALK_DEPTH {
        return;
    }
    let Ok(rd) = fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            // Skip build/VCS dirs, and the test/example/template trees that ship
            // inside checked-out skill repos: their `SKILL.md` files are fixtures
            // and placeholders, not installed skills, and indexing them injects
            // pure noise (e.g. a repo's `tests/fixtures/skills/*` cloned under
            // `~/.claude/plugins`).
            let skip = matches!(
                p.file_name().and_then(|s| s.to_str()),
                Some(
                    ".git"
                        | "target"
                        | "node_modules"
                        | "tests"
                        | "fixtures"
                        | "examples"
                        | "template"
                        | "templates"
                )
            );
            if !skip {
                collect(&p, out, depth + 1);
            }
        } else if p.file_name().and_then(|s| s.to_str()) == Some("SKILL.md") {
            out.push(p);
        }
    }
}

/// Parse a single `SKILL.md`. Returns `None` if it lacks a usable frontmatter.
/// Reads bytes and lossily decodes them rather than `read_to_string`, so a
/// single non-UTF8 file degrades to mojibake (and likely fails the frontmatter
/// check) instead of erroring the whole library out of discovery.
pub fn parse_file(path: &Path) -> anyhow::Result<Option<Skill>> {
    let bytes = fs::read(path)?;
    let content = String::from_utf8_lossy(&bytes).into_owned();
    let Some((name, description, mut keywords)) = parse_frontmatter(&content) else {
        return Ok(None);
    };
    if name.is_empty() || description.is_empty() || is_placeholder(&description) {
        return Ok(None);
    }
    for tok in tokenize(&name) {
        if !keywords.contains(&tok) {
            keywords.push(tok);
        }
    }
    let hash = format!("{:016x}", fnv1a_64(content.as_bytes()));
    let trigger_phrases = extract_phrases(&description);
    Ok(Some(Skill {
        id: name.clone(),
        name,
        description,
        body_head: body_head(&content, 8, 600),
        keywords,
        trigger_phrases,
        path: path.to_path_buf(),
        hash,
    }))
}

/// Minimum content tokens (stopwords excluded) for a quoted span to qualify as a
/// trigger phrase. Two is the floor: a full two-token match (e.g. `connect mysql`,
/// `screen reader support`) requires *both* discriminative tokens present, which
/// stays high-precision on realistic prompts while covering the many two-word
/// triggers skills actually ship. Single-token spans ("set up" → `set`, "report",
/// "the file") collapse below this and are dropped — they are common-word noise
/// that belongs to the dense/keyword channels, not here.
const MIN_PHRASE_TOKENS: usize = 2;

/// Upper bound on content tokens. A quoted span longer than this is a sentence or
/// a wholly-quoted description, not a trigger phrase — reject it so the channel
/// stays a *phrase* matcher and never demands a paragraph-length token overlap.
const MAX_PHRASE_TOKENS: usize = 10;

/// Mine multi-word trigger phrases from a skill description. Scans the *already
/// unquoted* description for inner quoted spans (single or double quotes, ASCII or
/// curly), keeps those with [`MIN_PHRASE_TOKENS`]..=[`MAX_PHRASE_TOKENS`] content
/// tokens, and returns each normalized to a space-joined string of its content
/// tokens (the form the ranker matches against a prompt). De-duplicated,
/// order-preserving.
///
/// Runs on the parsed description, never the raw YAML line, so a wholly
/// double-quoted `description:` value does not surface its entire text as one
/// phrase — only the genuinely inner quotes remain.
///
/// A straight `'` only opens/closes a span at a word boundary (preceded/followed
/// by a non-alphanumeric or the string edge), so apostrophes in contractions and
/// possessives — `don't`, `user's` — are not mistaken for quotes.
pub fn extract_phrases(description: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let chars: Vec<char> = description.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if let Some(close) = opens_quote(&chars, i) {
            // Find the matching close at a word boundary.
            if let Some(end) = find_close(&chars, i + 1, close) {
                let span: String = chars[i + 1..end].iter().collect();
                let toks = crate::text::content_tokens(&span);
                if (MIN_PHRASE_TOKENS..=MAX_PHRASE_TOKENS).contains(&toks.len()) {
                    let phrase = toks.join(" ");
                    if !out.contains(&phrase) {
                        out.push(phrase);
                    }
                }
                i = end + 1;
                continue;
            }
        }
        let _ = c;
        i += 1;
    }
    out
}

/// If position `i` is an opening quote, return the char that closes it. A straight
/// quote must sit at a left word boundary to count (else it is an apostrophe).
fn opens_quote(chars: &[char], i: usize) -> Option<char> {
    let c = chars[i];
    let close = match c {
        '\u{201c}' => '\u{201d}', // “ ”
        '\u{2018}' => '\u{2019}', // ‘ ’
        '"' | '\'' => c,          // straight quotes close themselves
        _ => return None,
    };
    let boundary = i == 0 || !chars[i - 1].is_alphanumeric();
    boundary.then_some(close)
}

/// Index of the closing quote `close` at or after `from`, requiring a right word
/// boundary for straight quotes so contraction apostrophes do not close early.
fn find_close(chars: &[char], from: usize, close: char) -> Option<usize> {
    let straight = close == '"' || close == '\'';
    (from..chars.len()).find(|&j| {
        chars[j] == close && (!straight || chars.get(j + 1).is_none_or(|n| !n.is_alphanumeric()))
    })
}

/// Pull the first `max_lines` non-blank body lines (after the frontmatter),
/// capped at `max_chars`. Markdown heading/list markers are stripped so the
/// embedder sees prose, not punctuation. Empty when there is no body.
fn body_head(content: &str, max_lines: usize, max_chars: usize) -> String {
    let mut lines = content.lines();
    // Skip the leading `--- ... ---` frontmatter block, if present.
    if lines.next().map(|l| l.trim()) == Some("---") {
        for l in lines.by_ref() {
            if l.trim() == "---" {
                break;
            }
        }
    }
    let mut out: Vec<String> = Vec::new();
    for l in lines {
        let t = l
            .trim()
            .trim_start_matches(['#', '-', '*', '>', ' '])
            .trim();
        if t.is_empty() {
            continue;
        }
        out.push(t.to_string());
        if out.len() >= max_lines {
            break;
        }
    }
    let joined = out.join(" ");
    match joined.char_indices().nth(max_chars) {
        Some((i, _)) => joined[..i].to_string(),
        None => joined,
    }
}

/// Extract `name`, `description`, and `keywords`/`aliases` from a leading
/// `--- ... ---` YAML frontmatter block. Intentionally minimal: handles the
/// single-line `key: value` and inline-list shapes our skills use, not the full
/// YAML grammar (no block scalars / nested maps).
pub fn parse_frontmatter(content: &str) -> Option<(String, String, Vec<String>)> {
    // A leading UTF-8 BOM (U+FEFF) is not whitespace to `str::trim`, so an
    // untouched line 1 would never equal "---"; strip it before the check.
    let content = content.strip_prefix('\u{FEFF}').unwrap_or(content);
    let mut lines = content.lines();
    if lines.next()?.trim() != "---" {
        return None;
    }
    let (mut name, mut description, mut keywords) = (String::new(), String::new(), Vec::new());
    for line in lines {
        let t = line.trim_end();
        if t.trim() == "---" {
            break;
        }
        if let Some(v) = t.strip_prefix("name:") {
            name = unquote(v.trim());
        } else if let Some(v) = t.strip_prefix("description:") {
            description = unquote(v.trim());
        } else if let Some(v) = t.strip_prefix("keywords:") {
            keywords = parse_list(v.trim());
        } else if let Some(v) = t.strip_prefix("aliases:") {
            keywords.extend(parse_list(v.trim()));
        }
    }
    Some((name, description, keywords))
}

/// Whether a description is the unfilled skeleton from a `template/SKILL.md`
/// (e.g. "Replace with description of the skill…"). Such files are scaffolding,
/// not installed skills, so they must never be indexed or injected.
fn is_placeholder(description: &str) -> bool {
    description
        .trim_start()
        .to_ascii_lowercase()
        .starts_with("replace with")
}

fn unquote(s: &str) -> String {
    let s = s.trim();
    let bytes = s.as_bytes();
    if bytes.len() >= 2
        && ((bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[bytes.len() - 1] == b'\''))
    {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

fn parse_list(s: &str) -> Vec<String> {
    s.trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .map(|x| unquote(x.trim()).to_ascii_lowercase())
        .filter(|x| !x.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_frontmatter() {
        let md = "---\nname: git-attribution\ndescription: Credit AI in commits.\n---\nbody\n";
        let (name, desc, _) = parse_frontmatter(md).unwrap();
        assert_eq!(name, "git-attribution");
        assert_eq!(desc, "Credit AI in commits.");
    }

    #[test]
    fn parses_quotes_and_keywords() {
        let md = "---\nname: \"x\"\ndescription: 'd'\nkeywords: [Foo, bar]\n---\n";
        let (name, desc, kw) = parse_frontmatter(md).unwrap();
        assert_eq!(name, "x");
        assert_eq!(desc, "d");
        assert_eq!(kw, ["foo", "bar"]);
    }

    #[test]
    fn rejects_without_frontmatter() {
        assert!(parse_frontmatter("no frontmatter here").is_none());
    }

    #[test]
    fn detects_template_placeholder() {
        assert!(is_placeholder(
            "Replace with description of the skill and when Claude should use it."
        ));
        assert!(is_placeholder("  replace WITH something"));
        assert!(!is_placeholder("Credit AI assistance in git commits."));
    }

    #[test]
    fn extracts_multiword_trigger_phrases() {
        // Inner-quoted spans in the (already unquoted) description that carry >=3
        // content tokens become trigger phrases, normalized to their content tokens.
        let desc = "Use when the user says \"find that page online\" or asks to \"search the public web archive\".";
        let ph = extract_phrases(desc);
        assert!(ph.contains(&"find page online".to_string()), "got {ph:?}");
        // "the" is a stopword and dropped; the rest survive as content tokens.
        assert!(
            ph.contains(&"search public web archive".to_string()),
            "got {ph:?}"
        );
    }

    #[test]
    fn ignores_short_and_common_quoted_spans() {
        // Single-word or all-stopword quotes are noise, not triggers, and must not
        // become phrases (they would over-fire the lexical channel).
        let desc = "Triggers include 'report', 'memo', 'set up', and \"the file\".";
        assert!(
            extract_phrases(desc).is_empty(),
            "short/common quotes leaked: {:?}",
            extract_phrases(desc)
        );
    }

    #[test]
    fn extraction_ignores_yaml_outer_quoting() {
        // A description whose YAML value is wholly double-quoted must not yield the
        // entire description as one giant "phrase": extraction runs on the parsed,
        // unquoted value, and the only real triggers are the inner single quotes.
        let md = "---\nname: docx\ndescription: \"Edit Word docs. Triggers include any mention of 'word document export'.\"\n---\nbody\n";
        let s = parse_file_from_str(md);
        assert!(
            s.trigger_phrases
                .iter()
                .all(|p| p.split_whitespace().count() <= 4),
            "outer YAML quote captured as phrase: {:?}",
            s.trigger_phrases
        );
        assert!(s
            .trigger_phrases
            .contains(&"word document export".to_string()));
    }

    /// Test helper: parse a SKILL.md from a string via a temp file.
    fn parse_file_from_str(md: &str) -> Skill {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!(
            "ski-phrase-{}-{}",
            std::process::id(),
            fnv1a_64(md.as_bytes())
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("SKILL.md");
        let mut f = fs::File::create(&path).unwrap();
        write!(f, "{md}").unwrap();
        let s = parse_file(&path).unwrap().unwrap();
        let _ = fs::remove_dir_all(&dir);
        s
    }

    #[test]
    fn parse_file_rejects_placeholder_skill() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("ski-tpl-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("SKILL.md");
        let mut f = fs::File::create(&path).unwrap();
        write!(
            f,
            "---\nname: template-skill\ndescription: Replace with description of the skill.\n---\nbody\n"
        )
        .unwrap();
        assert!(parse_file(&path).unwrap().is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_file_tolerates_non_utf8_bytes() {
        // A non-UTF8 SKILL.md must not error `parse_file` (which would otherwise
        // bubble through `discover` and blank out the whole library) — it lossily
        // decodes, and since the mangled frontmatter check then fails, it degrades
        // to `Ok(None)` (skipped) rather than `Err`.
        let dir = std::env::temp_dir().join(format!("ski-nonutf8-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("SKILL.md");
        fs::write(&path, [0xff, 0xfe, b'-', b'-', b'-', 0x00]).unwrap();
        assert!(parse_file(&path).is_ok());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn discover_skips_unreadable_file_instead_of_aborting() {
        // One bad path among several must not blank out the rest of the library:
        // discover() should skip the unreadable entry and still return the others.
        let dir = std::env::temp_dir().join(format!("ski-discover-skip-{}", std::process::id()));
        let good = dir.join("good");
        fs::create_dir_all(&good).unwrap();
        fs::write(
            good.join("SKILL.md"),
            "---\nname: good-skill\ndescription: A perfectly fine skill.\n---\nbody\n",
        )
        .unwrap();
        // A directory named SKILL.md can never be opened as a file -> read error.
        let bad = dir.join("bad");
        fs::create_dir_all(&bad).unwrap();
        fs::create_dir_all(bad.join("SKILL.md")).unwrap();

        let found = discover(std::slice::from_ref(&dir)).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, "good-skill");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_frontmatter_strips_leading_bom() {
        let md = "\u{FEFF}---\nname: x\ndescription: d\n---\n";
        let (name, desc, _) = parse_frontmatter(md).unwrap();
        assert_eq!(name, "x");
        assert_eq!(desc, "d");
    }

    #[test]
    fn collect_bounds_recursion_depth() {
        // Build a chain of nested dirs deeper than MAX_WALK_DEPTH with a SKILL.md
        // at the bottom; it must not be found (and, more importantly, must not
        // blow the stack on a real pathological tree).
        let root = std::env::temp_dir().join(format!("ski-deep-{}", std::process::id()));
        let mut dir = root.clone();
        for i in 0..MAX_WALK_DEPTH + 5 {
            dir = dir.join(format!("d{i}"));
        }
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("SKILL.md"),
            "---\nname: too-deep\ndescription: unreachable.\n---\n",
        )
        .unwrap();
        let mut out = Vec::new();
        collect(&root, &mut out, 0);
        assert!(out.is_empty(), "found a file past the depth cap: {out:?}");
        let _ = fs::remove_dir_all(&root);
    }
}

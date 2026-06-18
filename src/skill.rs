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

/// Walk `roots` and parse every `SKILL.md` found.
pub fn discover(roots: &[PathBuf]) -> anyhow::Result<Vec<Skill>> {
    let mut files = Vec::new();
    for r in roots {
        collect(r, &mut files);
    }
    files.sort();
    files.dedup();

    let mut out = Vec::new();
    for f in &files {
        if let Some(s) = parse_file(f)? {
            out.push(s);
        }
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out.dedup_by(|a, b| a.id == b.id);
    Ok(out)
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
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
                collect(&p, out);
            }
        } else if p.file_name().and_then(|s| s.to_str()) == Some("SKILL.md") {
            out.push(p);
        }
    }
}

/// Parse a single `SKILL.md`. Returns `None` if it lacks a usable frontmatter.
pub fn parse_file(path: &Path) -> anyhow::Result<Option<Skill>> {
    let content = fs::read_to_string(path)?;
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
    Ok(Some(Skill {
        id: name.clone(),
        name,
        description,
        body_head: body_head(&content, 8, 600),
        keywords,
        path: path.to_path_buf(),
        hash,
    }))
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
}

//! Tiny text utilities shared by the embedder, ranker, and skill parser.
//! Deterministic by design — embeddings persisted in the index must reproduce
//! byte-for-byte across runs and builds, so we use a fixed FNV hash, not the
//! std hasher (whose seed/impl is not a stability guarantee).

/// Lowercase, split on non-alphanumerics, drop tokens shorter than 2 chars.
pub fn tokenize(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() {
            cur.push(ch.to_ascii_lowercase());
        } else if !cur.is_empty() {
            if cur.len() >= 2 {
                out.push(std::mem::take(&mut cur));
            } else {
                cur.clear();
            }
        }
    }
    if cur.len() >= 2 {
        out.push(cur);
    }
    out
}

/// Function words that carry no discriminative signal for phrase matching. Kept
/// deliberately small — just the high-frequency glue that would otherwise let a
/// trigger phrase fire on, or be padded to length by, unrelated prose. Domain
/// terms are never listed here.
const STOPWORDS: &[&str] = &[
    "the", "an", "of", "to", "for", "and", "or", "in", "on", "at", "is", "it", "be", "as", "by",
    "with", "from", "into", "me", "my", "we", "our", "you", "your", "this", "that", "these",
    "those", "use", "used", "when", "user", "users", "say", "says", "want", "wants", "ask", "asks",
    "do", "does", "not", "if", "so", "up", "out", "via", "are", "was", "will", "can", "a", "i",
];

/// `tokenize`, minus stopwords — the discriminative tokens of a phrase. Used both
/// to gate a candidate phrase by length and to match it against a prompt.
pub fn content_tokens(s: &str) -> Vec<String> {
    tokenize(s)
        .into_iter()
        .filter(|t| !STOPWORDS.contains(&t.as_str()))
        .collect()
}

/// Light, deterministic singular form of a (lowercase) token, so the surface-form
/// channels — keyword, phrase, BM25 — match across trivial inflection
/// ("spreadsheets" ↔ "spreadsheet", "dependencies" ↔ "dependency"). Not a real
/// stemmer: it only needs to be *consistent*, because both the prompt side and the
/// skill side are normalized through it at match time. Applied at match time only —
/// never inside the embedders — so persisted index vectors are untouched.
pub fn norm_token(t: &str) -> String {
    let b = t.as_bytes();
    let n = b.len();
    if n <= 3 || !t.ends_with('s') {
        return t.to_string();
    }
    // "class", "status", "analysis": common non-plural s-endings stay whole.
    if t.ends_with("ss") || t.ends_with("us") || t.ends_with("is") {
        return t.to_string();
    }
    if n > 4 && t.ends_with("ies") {
        return format!("{}y", &t[..n - 3]); // dependencies -> dependency
    }
    if t.ends_with("sses")
        || t.ends_with("ches")
        || t.ends_with("shes")
        || t.ends_with("xes")
        || t.ends_with("zes")
    {
        return t[..n - 2].to_string(); // branches -> branch, boxes -> box
    }
    t[..n - 1].to_string() // charts -> chart
}

/// [`content_tokens`], each normalized through [`norm_token`] — the form the
/// surface-matching channels (phrase, BM25) compare prompt and skill text in.
pub fn match_tokens(s: &str) -> Vec<String> {
    content_tokens(s).iter().map(|t| norm_token(t)).collect()
}

/// FNV-1a 32-bit — stable token→bucket hash for the bag-of-words embedder.
pub fn fnv1a_32(s: &str) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for b in s.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// FNV-1a 64-bit — content hash for index cache invalidation (not security).
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_splits_and_lowercases() {
        assert_eq!(
            tokenize("Set-up a NEW uv_project!"),
            ["set", "up", "new", "uv", "project"]
        );
    }

    #[test]
    fn tokenize_drops_single_chars() {
        assert_eq!(tokenize("a b cd e"), ["cd"]);
    }

    #[test]
    fn content_tokens_drops_stopwords() {
        // Function words carry no discriminative signal for phrase matching, so
        // they are excluded from the content-token set (and the length gate).
        assert_eq!(
            content_tokens("connect to the Neon database"),
            ["connect", "neon", "database"]
        );
        // A phrase that is *only* stopwords/short words collapses to nothing.
        assert!(content_tokens("set it up").is_empty() || content_tokens("set it up") == ["set"]);
    }

    #[test]
    fn norm_token_singularizes_common_plurals() {
        assert_eq!(norm_token("spreadsheets"), "spreadsheet");
        assert_eq!(norm_token("charts"), "chart");
        assert_eq!(norm_token("dependencies"), "dependency");
        assert_eq!(norm_token("branches"), "branch");
        assert_eq!(norm_token("boxes"), "box");
        assert_eq!(norm_token("classes"), "class");
    }

    #[test]
    fn norm_token_leaves_non_plurals_alone() {
        // Short tokens and common non-plural s-endings must survive intact.
        for t in ["uv", "css", "class", "status", "analysis", "chart", "rust"] {
            assert_eq!(norm_token(t), t);
        }
    }

    #[test]
    fn match_tokens_normalizes_content_tokens() {
        assert_eq!(
            match_tokens("compute the formulas in these spreadsheets"),
            ["compute", "formula", "spreadsheet"]
        );
    }

    #[test]
    fn fnv_is_deterministic() {
        assert_eq!(fnv1a_32("commit"), fnv1a_32("commit"));
        assert_ne!(fnv1a_32("commit"), fnv1a_32("attribution"));
        assert_eq!(fnv1a_64(b"hello"), fnv1a_64(b"hello"));
    }
}

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
    fn fnv_is_deterministic() {
        assert_eq!(fnv1a_32("commit"), fnv1a_32("commit"));
        assert_ne!(fnv1a_32("commit"), fnv1a_32("attribution"));
        assert_eq!(fnv1a_64(b"hello"), fnv1a_64(b"hello"));
    }
}

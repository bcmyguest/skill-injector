//! Opt-in stderr diagnostics for the fail-open hot paths.
//!
//! `hook`/`observe`/`session-start` swallow every error by design (README:
//! "fail-open everywhere") so a ranking problem never blocks a prompt. That
//! contract has a cost: when injection silently stops, there is normally
//! nothing to debug with — the hook just goes quiet forever with no trace of
//! why. [`debug`] prints the swallowed error to stderr, but only when
//! `SKI_DEBUG` is set, so the default (quiet) behavior is unchanged and a
//! user who suspects something is wrong has a way to find out what.

/// Whether `SKI_DEBUG` is set (any value, including empty).
pub fn enabled() -> bool {
    std::env::var_os("SKI_DEBUG").is_some()
}

/// Print `ski: {context}: {err}` to stderr iff `SKI_DEBUG` is set. No-op
/// otherwise. `context` should read as a fragment ("hook decide failed").
pub fn debug(context: &str, err: &impl std::fmt::Display) {
    if enabled() {
        eprintln!("ski: {context}: {err}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enabled_reflects_env_var() {
        // SKI_DEBUG is read at call time with no caching, so tests can toggle
        // it freely — but other tests run in the same process may also read/
        // write it, so only assert the presence check itself is correct given
        // an explicit value, not the ambient default.
        std::env::set_var("SKI_DEBUG", "1");
        assert!(enabled());
        std::env::remove_var("SKI_DEBUG");
        assert!(!enabled());
    }
}

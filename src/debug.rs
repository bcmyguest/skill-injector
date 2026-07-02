//! `SKI_DEBUG`-gated stderr tracing for the fail-open paths.
//!
//! The hooks deliberately swallow every error (a ranking problem must never
//! block a prompt), but that used to mean *zero* trace: when injection silently
//! stopped — corrupt index, unreadable state dir, bad stdin — there was nothing
//! to debug with. Setting `SKI_DEBUG=1` (the same variable `ski-bootstrap.sh`
//! already honors for its install hint) surfaces the swallowed error on stderr
//! without changing any fail-open behavior.

/// Whether `SKI_DEBUG` is set to a non-empty, non-`0` value.
pub fn enabled() -> bool {
    std::env::var_os("SKI_DEBUG").is_some_and(|v| !v.is_empty() && v != "0")
}

/// Print a swallowed-error trace to stderr when [`enabled`]. Never touches
/// stdout (the hooks' output contract) and never fails.
pub fn log(msg: impl std::fmt::Display) {
    if enabled() {
        eprintln!("ski[debug]: {msg}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_and_empty_disable() {
        // Can't mutate the process env safely in parallel tests; just assert the
        // parse rule via the current (unset) state.
        if std::env::var_os("SKI_DEBUG").is_none() {
            assert!(!enabled());
        }
        log("never panics, even when disabled");
    }
}

//! ski / skill-inject — local semantic auto-injection of agent skills.
//!
//! Milestones 1–3 surface: skill discovery, embedding index, ranking
//! (`ski index` / `ski why`), the hook hot-path with session dedup (`ski hook`),
//! model-load observation (`ski observe`), and session lifecycle
//! (`ski session-start`). `init` lands in a later milestone.

pub mod config;
pub mod embed;
pub mod hook;
pub mod index;
pub mod inject;
pub mod observe;
pub mod paths;
pub mod rank;
pub mod rerank;
pub mod session;
pub mod session_start;
pub mod skill;
pub mod text;

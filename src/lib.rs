//! ski / skill-inject — local semantic auto-injection of agent skills.
//!
//! Milestones 1–3 surface: skill discovery, embedding index, ranking
//! (`ski index` / `ski why`), the hook hot-path with session dedup (`ski hook`),
//! model-load observation (`ski observe`), session lifecycle
//! (`ski session-start`), and host setup (`ski init`).

pub mod confidence;
pub mod config;
pub mod context;
pub mod embed;
pub mod history;
pub mod hook;
pub mod index;
pub mod init;
pub mod inject;
pub mod observe;
pub mod paths;
pub mod rank;
pub mod rerank;
pub mod session;
pub mod session_start;
pub mod skill;
pub mod telemetry;
pub mod text;

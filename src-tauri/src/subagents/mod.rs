//! Bounded, typed subagents.
//!
//! A subagent is a **narrower** worker a run may hand a piece of work to: read
//! these four pages and say what is on them; find the passages about seal wear;
//! check these three figures. It is not a second copy of the parent with the
//! same reach.
//!
//! ```text
//! agents/
//!   document-extractor.md      profile: role, tools, limits, isolation, schema
//!   knowledge-retriever.md
//!   calculation-checker.md
//!   artifact-reviewer.md
//!   code-worker.md
//! ```
//!
//! ## The one property everything here exists to hold
//!
//! **A child is never more capable than its parent.** Every field of
//! [`inherit::EffectivePolicy`] is produced by narrowing an
//! [`inherit::InheritedPolicy`] — sets by filtering the parent's, scalars by
//! `min` — and a child never constructs its own. A profile is a *request*, and
//! a request that asks for more than the parent holds simply does not get it.
//!
//! That covers the tools, the classification ceiling, the workspace, the
//! network (always none), the approval requirement and the depth. There is no
//! expression in this module through which a child reaches a wider value.
//!
//! ## Why the profiles are Markdown
//!
//! Same reason as skills: an operator can read, review and diff a file. And the
//! same consequence — a profile is untrusted input, so it is compiled by
//! [`profile::compile`] against hard ceilings and only ever used to narrow.
//!
//! ## What a child is handed
//!
//! A [`packet::ChildTaskPacket`]: an objective in the parent's words, and
//! **references** to inputs. Not the parent's transcript, and not the passages
//! the parent retrieved. A child that needs a passage fetches it itself, under
//! its own clearance, and may legitimately get less than the parent did.
//!
//! ## What comes back
//!
//! A [`result::ChildResult`] whose status the manager sets from what actually
//! happened. A worker cannot report success for work it did not finish, because
//! the status is not a field it fills in — which is requirement 8, and the
//! failure it prevents is a timed-out retrieval being folded in as a complete
//! one.
//!
//! ## Deliberately absent in this phase
//!
//! No agent teams, no remote workers, no unattended plant actions. A child
//! cannot spawn a child ([`profile::ceiling::MAX_DEPTH`] is 1), and every
//! shipped profile declares `max-children: 0`.

pub mod certification;
pub mod inherit;
pub mod manager;
pub mod packet;
pub mod profile;
pub mod result;
pub mod workers;

use std::path::Path;

pub use inherit::{EffectivePolicy, InheritRefusal, InheritedPolicy};
pub use manager::{ChildWorker, SpawnRefusal, Spawned, SubagentManager, MAX_CONCURRENT_READERS};
pub use packet::{derive_idempotency_key, ChildTaskPacket, InputRef};
pub use profile::{
    AgentProfile, Isolation, Limits, MemoryScope, ProfileError, SchemaKind, WritePolicy,
};
pub use result::{ChildResult, ChildStatus, EvidenceRef, Finding};

/// A profile that could not be compiled, kept so it can be reported.
#[derive(Debug, Clone)]
pub struct RejectedProfile {
    pub file: String,
    pub error: ProfileError,
}

/// Everything found in a profiles directory.
#[derive(Debug, Clone, Default)]
pub struct LoadedProfiles {
    pub profiles: Vec<AgentProfile>,
    /// Files that looked like profiles and did not compile.
    ///
    /// Reported rather than dropped, for the same reason a quarantined skill is
    /// still listed: an operator with a broken profile needs to see it is
    /// there, not conclude it was never installed.
    pub rejected: Vec<RejectedProfile>,
}

impl LoadedProfiles {
    pub fn get(&self, name: &str) -> Option<&AgentProfile> {
        self.profiles.iter().find(|profile| profile.name == name)
    }
}

/// Compiles every `*.md` in a directory into a profile.
///
/// A missing directory is an empty set rather than a failure: a deployment with
/// no subagent profiles is a deployment that does not use subagents.
pub fn load_profiles(directory: &Path) -> LoadedProfiles {
    let mut loaded = LoadedProfiles::default();

    let Ok(entries) = std::fs::read_dir(directory) else {
        return loaded;
    };

    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        // A README beside the profiles is documentation, not a broken profile.
        if stem.eq_ignore_ascii_case("readme") {
            continue;
        }

        let file = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(stem)
            .to_string();

        match std::fs::read_to_string(&path) {
            Ok(source) => {
                let sha256 = crate::skills::registry::sha256_of(&source);
                match profile::compile(&source, stem, &sha256) {
                    Ok(profile) => loaded.profiles.push(profile),
                    Err(error) => loaded.rejected.push(RejectedProfile { file, error }),
                }
            }
            Err(error) => loaded.rejected.push(RejectedProfile {
                file,
                error: ProfileError::Malformed {
                    detail: error.to_string(),
                },
            }),
        }
    }

    loaded.profiles.sort_by(|a, b| a.name.cmp(&b.name));
    loaded
}

#[cfg(test)]
mod tests;

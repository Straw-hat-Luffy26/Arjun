//! Agent Skills: reusable instructions an operator installs, not code.
//!
//! A skill is a directory holding a `SKILL.md` and, optionally, references,
//! scripts and assets. It tells the model how this organisation does a
//! particular job — how an approval note is worded here, which figures a seal
//! assessment needs, what to do when a scan will not read.
//!
//! ```text
//! skills/
//!   trusted.json                      operator-maintained; nothing is trusted without it
//!   inspection-approval-note/
//!     SKILL.md                        frontmatter + instructions
//!     references/                     material the skill may quote
//!     scripts/                        programs it may name (never run from here)
//!     assets/                         templates and fixtures
//! ```
//!
//! ## The one thing to understand about this module
//!
//! **A skill is untrusted data.** It is a file somebody put on the machine, and
//! its contents reach a model. That makes it exactly the same class of input as
//! a scanned document: it may be wrong, it may be out of date, and it may have
//! been written to persuade the model to do something the person running the
//! task did not ask for.
//!
//! So a skill is guidance and never enforcement. Concretely:
//!
//! - Its `allowed-tools` list can only **narrow** what the run already
//!   permitted ([`narrowing`]). There is no expression in this codebase by
//!   which a skill's contents reach the plan, the clearance, the workspace, the
//!   sandbox tier or the network policy.
//! - Its `metadata.approval-class` is a *description an operator reads*. Whether
//!   an action needs approval is decided per tool in `orchestrator::tools`, and
//!   consulted from `policy::PolicyGateway`, every time.
//! - Sentences in its body — including sentences addressed to the model, in the
//!   imperative, about permissions — are text. They pass through the same
//!   gateway as anything else the model was told.
//!
//! Nothing here changes what a run may do. It changes what the model *knows*,
//! and that distinction is the whole of the module's security posture.
//!
//! ## Where each requirement lives
//!
//! - [`frontmatter`] — a strict, small YAML subset. Refuses anchors, aliases,
//!   tags and merge keys by name, because a skill file is untrusted input and
//!   full YAML is a large grammar.
//! - [`manifest`] — validation: lowercase-hyphenated names, length caps,
//!   required fields, and the name matching its parent directory.
//! - [`registry`] — metadata-only discovery, the trust list, quarantine, and
//!   loading a body only after the checks pass. Hot reload swaps a snapshot,
//!   and a run holds an `Arc` to the definition it started with.
//! - [`containment`] — a reference, script or asset resolves inside the skill
//!   or not at all.
//! - [`narrowing`] — the intersection, and the reasons it can only be one.
//!
//! ## What is deliberately absent
//!
//! There is no installer. Nothing here fetches from GitHub, npm, PyPI or any
//! other remote source, and adding one would put an unreviewed file on the
//! machine in the one place whose contents reach the model. Skills arrive the
//! way any other operational material arrives: somebody puts them there.
//!
//! Scripts under `scripts/` are **named, not executed**. A skill can tell a
//! model that a script exists and what it does; running it is `execute_code`,
//! which goes through the gateway, the sandbox assessment and an approval like
//! anything else.

pub mod containment;
pub mod frontmatter;
pub mod manifest;
pub mod narrowing;
pub mod registry;

pub use manifest::{
    is_valid_name, ApprovalClass, NetworkNeed, Quarantine, SkillCard, SkillManifest,
    CAPABILITY_PAGE,
};
pub use narrowing::{narrow, Narrowed};
pub use registry::{
    LoadRefusal, LoadedSkill, Signature, SkillContext, SkillRegistry, SkillUse, Snapshot,
    TrustList, TrustedSkill,
};

#[cfg(test)]
mod tests;

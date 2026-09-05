//! A run's history, written while it happens and readable after a restart.
//!
//! ## What was wrong with the old arrangement
//!
//! [`super::tasks`] writes one JSON file per run, once, when the run ends. Good
//! for reviewing a finished task; nothing at all for the two cases that matter
//! most:
//!
//! - **A window remounts mid-run.** The trace lived in React state, so it was
//!   gone. The run carried on in the backend with nobody watching it.
//! - **The process dies mid-run.** Nothing was ever written, so the run left no
//!   trace of having existed. The next start showed a task list with a hole in
//!   it exactly where the interesting thing happened.
//!
//! Both are the same missing thing: durable state *during* a run rather than
//! only after one.
//!
//! ## The shape
//!
//! - [`model`] — the event types, the envelope, and the redaction every payload
//!   passes through on the way in.
//! - [`machine`] — the explicit run states, and which events may legally move
//!   between them.
//! - [`store`] — ordered, atomic, append-only writes to SQLite; snapshots;
//!   recovery of runs the process took down with it.
//! - [`projection`] — folding events into the state a screen draws, so opening
//!   the Tasks list does not replay every event of every run.
//! - [`idempotency`] — doing a side-effecting tool call at most once, even
//!   across the restart that is the whole reason it could happen twice.
//!
//! ## What is deliberately *not* here
//!
//! The task record. A finished run still writes its JSON file exactly as
//! before, and [`super::tasks`] still reads it. That is not legacy left lying
//! around — the record holds the answer, the evidence and the working, which
//! is what somebody reviewing a finished task wants and none of which belongs
//! in an event stream. The two answer different questions, and old records
//! stay readable because nothing about them changed.

/// How many times the process may pick a run back up by itself before it stops
/// trying and asks a person.
///
/// Recovery has to be bounded or a run that fails during recovery is recovered
/// again, fails again, and the loop is only visible as a growing event log. The
/// ceiling is low on purpose: a run that could not be picked up twice is not
/// going to be picked up on the third attempt for any reason this side can act
/// on.
pub const MAX_RECOVERY_ATTEMPTS: u32 = 2;

pub mod approvals;
pub mod checkpoint;
pub mod context;
pub mod children;
pub mod operations;
pub mod idempotency;
pub mod lease;
pub mod machine;
mod migrations;
pub mod model;
pub mod projection;
pub mod store;

pub use approvals::{ApprovalInvalid, ApprovalStatus, DurableApproval};
pub use lease::{Held, Lease, DEFAULT_LEASE_SECONDS, HEARTBEAT_SECONDS};
pub use checkpoint::{
    NotResumable, Resumability, RunCheckpoint, WorldNow, CHECKPOINT_SCHEMA_VERSION,
};
pub use idempotency::{
    args_fingerprint, derive_key, is_side_effecting, EffectLookup, EffectStatus, KeyConflict,
    RecordedOutcome,
};
pub use machine::{advance, RunState, Transition};
pub use model::{
    canonical, digest, is_confidential, payload_hash, redact, EventDraft, TaskEvent, TaskEventType,
    UnreadableEvent, SCHEMA_VERSION, SYSTEM_ACTOR,
};
pub use projection::{fold, ActivityRecord, TaskSnapshot, UnknownEffect};
pub use store::{AppendError, EventPage, TaskEventLog};

#[cfg(test)]
mod checkpoint_store_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod context_tests;

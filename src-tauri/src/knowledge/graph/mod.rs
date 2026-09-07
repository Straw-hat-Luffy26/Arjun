//! Notebooks: a named library of documents, and the graph derived from it.
//!
//! A conversation's attachments vanish with the conversation. Nothing in ARJUN
//! has been able to hold "these forty documents belong together" and answer a
//! question that spans them — every file is read on its own, so a supplier named
//! in a contract and the same supplier named in a maintenance log are two
//! unrelated strings.
//!
//! A notebook is that missing container. It owns nothing but references: the
//! text still lives in [`crate::agent_runtime::documents::DocumentStore`],
//! content-addressed, and a notebook holds the sha256 of each document it
//! includes. Adding a document to a notebook copies no bytes.
//!
//! - [`store`]: notebooks and their membership, owner-scoped in SQL.
//!
//! ## Why the owner check is in the SQL
//!
//! The ten `memory_engine::api::*` commands were deleted from this application
//! because every one of them proved that *somebody* was signed in and then read,
//! wrote or deleted the memory of *everybody* — the session was checked, and the
//! query it authorised was not scoped to the person who passed the check. That
//! is the exact shape of the bug, and it is invisible in review because the
//! command looks correct: it calls `require_session` on the first line.
//!
//! So no query here is written without `owner_user_id` in its `WHERE`, the way
//! [`crate::knowledge::index`] binds clearance into the statement rather than
//! filtering rows it already fetched. A store method that takes no owner is a
//! store method that cannot be safely called from a command.

#[cfg(test)]
mod journey_tests;
pub mod persist;
pub mod relations;
pub mod render;
pub mod statistical;
pub mod store;
pub mod typing;

pub use persist::{node_id, EvidenceRow, GraphEdge, GraphNode, GraphView};
pub use relations::{RelationStats, Triplet, VerifiedRelation, RELATION_VERSION};
pub use statistical::{extract, DraftEdge, DraftNode, ExtractionStats, GraphDraft, EXTRACTOR_VERSION};
pub use store::{Notebook, NotebookDocument, NotebookStore};

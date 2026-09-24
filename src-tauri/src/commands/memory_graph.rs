//! The commands behind the agent-memory graph view.
//!
//! Read-only, and thin for the same reason [`super::knowledge`] is: the rule
//! about what a person may see lives in
//! [`crate::knowledge::graph::runtime_store`], where it is applied before an
//! edge is traversed or a count is taken. These commands establish who is
//! asking and hand that session down. A second copy of an access check is a
//! second thing that can disagree with the first.
//!
//! ## The subscription, and why nothing is pushed
//!
//! A window learns that the graph moved from [`MEMORY_GRAPH_EVENT`], whose
//! payload is one integer: the new head revision. It then calls
//! [`memory_graph_changes`] *as itself* and is given only what it may see.
//!
//! The alternative — emitting the change with its content — is what the
//! instruction "never broadcast private payloads to every window/user" rules
//! out, and it is worth being precise about why, because Tauri's `emit` makes
//! it the easy thing to do. `AppHandle::emit` delivers to every window in the
//! process. A deployment where an administrator and an employee each have a
//! window open would, on every commit, hand both of them the same payload; the
//! employee's window would then be responsible for not drawing what it was
//! already given. That is a permission check on the wrong side of the wire.
//! Sending a revision number instead means the private half never leaves the
//! backend without a session attached to the request.
//!
//! A revision number is not itself a disclosure: it says something moved, which
//! is what a reader needs in order to go and ask.

use std::collections::HashMap;
use std::sync::Arc;

use serde::Serialize;
use tauri::State;

use crate::agent_runtime::context_manifest::ContextManifest;
use crate::commands::agent::TaskEvents;
use crate::commands::governance::{require_session, CurrentSession};
use crate::knowledge::graph::runtime_feed::{
    ChangeBatch, InContextItem, MemorySnapshot, MAX_BATCH,
};
use crate::knowledge::graph::runtime_memory::MemoryScope;
use crate::knowledge::graph::runtime_store::MemoryGraph;

/// The wake signal. Carries [`GraphMoved`] and nothing else.
pub const MEMORY_GRAPH_EVENT: &str = "memory-graph:moved";

/// "The graph is at revision N." The entire payload of the broadcast.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphMoved {
    pub revision: i64,
}

/// The graph as this person may see it, with the cursor that continues it.
///
/// `run_id` is optional and additive: naming a run marks the items that run's
/// context actually carried. Without one the graph is still the graph — the
/// view is useful for looking at what a task knows whether or not a turn is in
/// flight, and refusing to answer without a run would make it useless between
/// turns.
#[tauri::command]
pub async fn memory_graph_snapshot(
    scope: MemoryScope,
    project_id: Option<String>,
    run_id: Option<String>,
    graph: State<'_, Arc<MemoryGraph>>,
    events: State<'_, TaskEvents>,
    session: State<'_, CurrentSession>,
) -> Result<MemorySnapshot, String> {
    let signed_in = require_session(&session)?;
    let mut snapshot = graph
        .snapshot_at(&signed_in, &scope, project_id.as_deref())
        .map_err(|error| error.explain())?;

    if let Some(run_id) = run_id.as_deref() {
        // A checkpoint that cannot be read is not an error for this screen. The
        // graph is the answer to the question asked; the context highlight is a
        // second, weaker fact layered on it, and losing it should not blank the
        // view. It is logged rather than swallowed silently.
        match events.checkpoint(run_id) {
            Ok(Some(checkpoint)) => {
                if let Some(manifest) = checkpoint.manifest.as_ref() {
                    let (in_context, revision) = in_context_from(manifest, &snapshot);
                    snapshot.in_context = in_context;
                    snapshot.context_revision = revision;
                }
            }
            Ok(None) => {}
            Err(error) => log::warn!(
                "[MEMORY-GRAPH] run {run_id}: the checkpoint could not be read, so the view \
                 cannot mark what the turn carried: {}",
                error.explain()
            ),
        }
    }

    Ok(snapshot)
}

/// Everything after `cursor`, as this person may see it.
///
/// `limit` is clamped to [`MAX_BATCH`] rather than trusted. A caller asking for
/// everything is asking for a message big enough to stall the window it is
/// drawn in, and the honest answer is a batch plus `hasMore`.
#[tauri::command]
pub async fn memory_graph_changes(
    scope: MemoryScope,
    project_id: Option<String>,
    cursor: i64,
    limit: Option<usize>,
    graph: State<'_, Arc<MemoryGraph>>,
    session: State<'_, CurrentSession>,
) -> Result<ChangeBatch, String> {
    let signed_in = require_session(&session)?;
    graph
        .changes_since(
            &signed_in,
            &scope,
            project_id.as_deref(),
            cursor,
            limit.unwrap_or(MAX_BATCH),
        )
        .map_err(|error| error.explain())
}

/// What the turn carried, narrowed to what this reader may see.
///
/// ## Two filters, both load-bearing
///
/// The manifest names every item the compiler put in front of the model,
/// including ones this reader is not cleared for — it is a record of what the
/// *model* read, not of what a person may. So the selection is intersected with
/// the authorised snapshot. Without that, a person with narrower clearance than
/// the agent would see a highlight, and a count, for an item they cannot open.
///
/// The revision comparison is the second. An item the model read at revision 3
/// that now stands at 4 is still highlighted — the turn did carry it — but
/// `current` is false, because presenting the corrected text as the thing the
/// model saw would misrepresent the answer the person is about to trust.
fn in_context_from(
    manifest: &ContextManifest,
    snapshot: &MemorySnapshot,
) -> (Vec<InContextItem>, Option<i64>) {
    let Some(binding) = manifest.graph.as_ref() else {
        return (Vec::new(), None);
    };
    let visible: HashMap<&str, u64> = snapshot
        .items
        .iter()
        .map(|item| (item.item_id.as_str(), item.revision))
        .collect();

    let selected = binding
        .selected
        .iter()
        .filter_map(|chosen| {
            let now = visible.get(chosen.item_id.as_str())?;
            Some(InContextItem {
                item_id: chosen.item_id.clone(),
                revision: chosen.revision,
                reason: chosen.reason.clone(),
                current: *now == chosen.revision,
            })
        })
        .collect();

    (selected, Some(binding.graph_revision))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_runtime::context_manifest::{GraphBinding, HistoryBinding, SelectedItem};
    use crate::knowledge::graph::runtime_memory::tests::{item, operator};
    use crate::knowledge::graph::runtime_memory::{MemoryItem, MemoryKind};

    fn manifest_with(selected: Vec<SelectedItem>) -> ContextManifest {
        let mut manifest = ContextManifest::new(
            "run-1",
            "attempt-1",
            "conversation-1",
            "message-1",
            "model-a",
            8192,
            Vec::new(),
            None,
            HistoryBinding {
                carried: 0,
                dropped: 0,
                tokens: 0,
                pinned: Vec::new(),
                omitted_pins: Vec::new(),
            },
        );
        manifest.graph = Some(GraphBinding {
            requested_revision: None,
            graph_revision: 9,
            selected,
        });
        manifest
    }

    fn snapshot_of(items: Vec<MemoryItem>) -> MemorySnapshot {
        MemorySnapshot {
            items,
            edges: Vec::new(),
            cursor: 9,
            in_context: Vec::new(),
            context_revision: None,
        }
    }

    /// The headline rule: a manifest entry this reader may not see is not
    /// highlighted, and is not counted either.
    #[test]
    fn an_item_the_reader_cannot_see_is_not_marked_in_context() {
        let visible = item(MemoryKind::Fact, operator());
        let snapshot = snapshot_of(vec![visible.clone()]);
        let manifest = manifest_with(vec![
            SelectedItem {
                item_id: visible.item_id.clone(),
                revision: 1,
                reason: "mandatory".into(),
                scope: None,
                precedence: None,
            },
            SelectedItem {
                item_id: "mi-not-for-you".into(),
                revision: 1,
                reason: "recall".into(),
                scope: None,
                precedence: None,
            },
        ]);

        let (marked, revision) = in_context_from(&manifest, &snapshot);
        assert_eq!(marked.len(), 1, "an unauthorised item was highlighted");
        assert_eq!(marked[0].item_id, visible.item_id);
        assert_eq!(revision, Some(9));
    }

    /// An item corrected since the turn was compiled is still in context, and
    /// is not pretended to be the current one.
    #[test]
    fn an_item_that_moved_on_is_marked_not_current() {
        let mut held = item(MemoryKind::Fact, operator());
        held.revision = 4;
        let snapshot = snapshot_of(vec![held.clone()]);
        let manifest = manifest_with(vec![SelectedItem {
            item_id: held.item_id.clone(),
            revision: 3,
            reason: "mandatory".into(),
            scope: None,
            precedence: None,
        }]);

        let (marked, _) = in_context_from(&manifest, &snapshot);
        assert_eq!(marked.len(), 1);
        assert!(!marked[0].current, "a stale selection was reported current");
        assert_eq!(marked[0].revision, 3, "the revision the model read was lost");
    }

    #[test]
    fn an_item_at_the_revision_it_was_read_at_is_current() {
        let held = item(MemoryKind::Fact, operator());
        let snapshot = snapshot_of(vec![held.clone()]);
        let manifest = manifest_with(vec![SelectedItem {
            item_id: held.item_id.clone(),
            revision: held.revision,
            reason: "mandatory".into(),
            scope: None,
            precedence: None,
        }]);

        let (marked, _) = in_context_from(&manifest, &snapshot);
        assert!(marked[0].current);
    }

    /// A manifest from before the graph existed carries no binding. That is a
    /// real state — a turn compiled by an older build — and it reports nothing
    /// rather than guessing.
    #[test]
    fn a_manifest_with_no_graph_binding_highlights_nothing() {
        let held = item(MemoryKind::Fact, operator());
        let snapshot = snapshot_of(vec![held]);
        let mut manifest = manifest_with(Vec::new());
        manifest.graph = None;

        let (marked, revision) = in_context_from(&manifest, &snapshot);
        assert!(marked.is_empty());
        assert_eq!(revision, None);
    }
}

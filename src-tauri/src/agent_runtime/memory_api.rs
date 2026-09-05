//! The only way a model can reach memory.
//!
//! ## Why this is a separate module from [`super::memory`]
//!
//! [`super::memory`] is the store and the policy. This is the *boundary* — the
//! two RPC methods the agent runtime may call, and the translation between what
//! a model asked for and what Rust is prepared to do about it.
//!
//! Keeping them apart matters because the store's API is deliberately more
//! capable than anything a model should reach. `MemoryStore::remember` can write
//! any scope, any classification, any ACL. That is right for a store the
//! application drives; it would be catastrophic as a tool schema. So the model's
//! surface is two verbs, and everything that decides anything about them is
//! filled in on this side.
//!
//! ## What the model does not get to choose
//!
//! Not "what the model chooses is validated" — *the model does not supply it at
//! all*, which is a stronger property and a shorter argument:
//!
//! - **Identity.** From the signed-in session. There is no user argument.
//! - **Project.** Derived from that session. There is no project argument, so a
//!   cross-project read is not a request that gets refused — it is a request
//!   that cannot be expressed.
//! - **Classification and ACL.** From the source material, in the store.
//! - **Approval.** From the approval queue, verified against the exact content
//!   by [`super::memory::ApprovalBinding`].
//!
//! The one thing a model does choose is *which scope to read* and *which key to
//! promote*, and both are checked against a fixed list before anything is
//! touched.
//!
//! ## What reaches the durable record
//!
//! Hashes and counts. A recall that returned four items records that four items
//! were returned from a named scope; it does not record what they said. The
//! record has to be readable by somebody who is not cleared for the material it
//! describes, and the only way to guarantee that is for the values never to be
//! in it.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::events;
use super::memory::{
    ApprovalBinding, MemoryError, MemoryItem, MemoryKind, MemoryScope, MemorySource, Remember,
};
use super::protocol::{code, WireError};
use super::RuntimeDeps;
use crate::identity::Session;
use crate::policy::Classification;

/// The scopes a model may name, and nothing else.
///
/// A closed list rather than a free string. An unknown scope is refused by the
/// parser before any of this module's logic runs, so a typo and an attack look
/// the same from here — which is the point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RequestedScope {
    /// This run's own state.
    Run,
    /// The project's approved terminology, templates and stable facts.
    Workspace,
    /// The signed-in person's own non-sensitive preferences.
    User,
}

impl RequestedScope {
    fn parse(raw: &str) -> Result<Self, WireError> {
        match raw {
            "run" => Ok(Self::Run),
            "workspace" => Ok(Self::Workspace),
            "user" => Ok(Self::User),
            other => Err(WireError::new(
                code::BAD_PARAMS,
                format!("{other:?} is not a memory scope. Use \"run\", \"workspace\" or \"user\"."),
            )),
        }
    }
}

/// Longest key a model may name.
///
/// A key is an identifier, not a payload. Without a bound, `key` is a channel
/// for putting arbitrary text into a durable event by way of a refusal message.
const MAX_KEY_CHARS: usize = 128;

/// Checks a key is an identifier and not a smuggled document.
fn parse_key(raw: Option<&str>) -> Result<String, WireError> {
    let key = raw.unwrap_or_default().trim();
    if key.is_empty() {
        return Err(WireError::new(
            code::BAD_PARAMS,
            "A memory key is required.".to_string(),
        ));
    }
    if key.chars().count() > MAX_KEY_CHARS {
        return Err(WireError::new(
            code::BAD_PARAMS,
            format!("That memory key is longer than the {MAX_KEY_CHARS} characters allowed."),
        ));
    }
    if !key
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':'))
    {
        return Err(WireError::new(
            code::BAD_PARAMS,
            "A memory key may hold only letters, digits, dashes, underscores, dots and colons."
                .to_string(),
        ));
    }
    Ok(key.to_string())
}

impl RuntimeDeps {
    /// The project this session works within.
    ///
    /// Derived from the signed-in person, never from the run and never from the
    /// model. A person's department is the only project-shaped fact identity
    /// carries today; when a richer project model arrives this is the one place
    /// that changes.
    ///
    /// `None` means the person belongs to no project, and that is *narrowing*,
    /// not widening: they see workspace memory confined to no project, which is
    /// none of it.
    pub(super) fn project_of(&self, session: &Session) -> Option<String> {
        session.user.department.clone()
    }

    /// The store scope for a requested scope, filled in from Rust-side facts.
    pub(super) fn scope_for(
        &self,
        requested: RequestedScope,
        run_id: &str,
        session: &Session,
    ) -> MemoryScope {
        match requested {
            RequestedScope::Run => MemoryScope::Run {
                run_id: run_id.to_string(),
            },
            RequestedScope::Workspace => MemoryScope::Workspace {
                // `unwrap_or_default` yields an empty project id, which matches
                // nothing rather than everything — see `project_of`.
                project_id: self.project_of(session).unwrap_or_default(),
            },
            RequestedScope::User => MemoryScope::User {
                user_id: session.user.id.clone(),
            },
        }
    }
}

/// What a recall returns to the model.
///
/// The value is included — that is the point of recalling — but nothing else
/// about the item is. Its ACL, its source document and its approval binding stay
/// on this side: a model that could read them could report them, and a model
/// that can report an ACL can describe the shape of what it is not allowed to
/// see.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RecalledItem {
    pub key: String,
    pub value: String,
    pub kind: MemoryKind,
}

impl From<&MemoryItem> for RecalledItem {
    fn from(item: &MemoryItem) -> Self {
        Self {
            key: item.key.clone(),
            value: item.value.clone(),
            kind: item.kind,
        }
    }
}

/// `memory.recall_authorized` — everything in one scope this person may read.
///
/// Named for what it does rather than what it is about: there is no unauthorized
/// recall to contrast it with, and a caller reading the method list should not
/// have to wonder whether there is.
pub fn recall_authorized(params: Value, deps: &Arc<RuntimeDeps>) -> Result<Value, WireError> {
    let session = deps.session()?;
    let run_id = params
        .get("runId")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    let requested = RequestedScope::parse(
        params
            .get("scope")
            .and_then(Value::as_str)
            .unwrap_or_default(),
    )
    .map_err(|error| {
        remember_memory_event(
            deps,
            &run_id,
            events::TaskEventType::MemoryRefused,
            json!({ "operation": "recall", "because": "unknown scope" }),
        );
        error
    })?;

    let scope = deps.scope_for(requested, &run_id, &session);
    if deps.events.snapshot(&run_id).map_err(|error| WireError::new(code::INTERNAL, error))?
        .is_some_and(|snapshot| snapshot.actor != session.user.id) {
        return Err(WireError::new(code::REFUSED, "This task's memory belongs to another operator."));
    }
    if params.get("transcriptSeq").is_some() {
        if requested != RequestedScope::Run { return Err(WireError::new(code::REFUSED,"Transcript retrieval is confined to the current run.")); }
        return super::context_api::read_transcript(&params,deps);
    }
    let project = deps.project_of(&session);

    // Expiry is swept before the read rather than only filtered during it, so a
    // lapsed item stops occupying the store as well as stopping being returned.
    // A failure to sweep is not a failure to read: the read filters expiry too.
    let _ = deps.memory.expire(&scope);

    let items = deps.memory.recall(&scope, &session, project.as_deref());
    let recalled: Vec<RecalledItem> = items.iter().filter(|item| match &item.source {
        MemorySource::Document { document_sha256, page, classification } => deps.index
            .region(&session, document_sha256, *page, *page, 1)
            .is_ok_and(|hits| hits.iter().any(|hit| hit.classification == *classification)),
        _ => true,
    }).map(RecalledItem::from).collect();

    remember_memory_event(
        deps,
        &run_id,
        events::TaskEventType::MemoryRecalled,
        json!({
            "scope": scope_label(&scope),
            // Counts and key hashes only. What was recalled is the material this
            // record must be readable without.
            "count": recalled.len(),
            "keyHashes": recalled
                .iter()
                .map(|item| events::digest(&item.key))
                .collect::<Vec<_>>(),
        }),
    );

    Ok(json!({
        "scope": scope_label(&scope),
        "items": recalled,
        "note": "Held on this machine for this scope only. Cite documents by their evidence marker, not from memory.",
    }))
}

/// `memory.promote_approved` — copy one run-scope fact into the project's memory.
///
/// The one operation that makes something a later run will read, which is why it
/// is the one operation that requires a person. Every part of the decision is
/// re-derived here from the stored item and the approval record; the model
/// supplies a key and an approval id and nothing else.
pub fn promote_approved(params: Value, deps: &Arc<RuntimeDeps>) -> Result<Value, WireError> {
    let session = deps.session()?;
    let run_id = params
        .get("runId")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    let refuse = |deps: &Arc<RuntimeDeps>, because: &str, message: String| -> WireError {
        remember_memory_event(
            deps,
            &run_id,
            events::TaskEventType::MemoryRefused,
            json!({ "operation": "promote", "because": because }),
        );
        WireError::new(code::REFUSED, message)
    };

    let key = parse_key(params.get("key").and_then(Value::as_str))?;
    let approval_id = params
        .get("approvalId")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    if approval_id.is_empty() {
        return Err(refuse(
            deps,
            "no approval named",
            "Promoting a fact into the project's memory needs the id of an approval a person \
             granted for it. Ask for approval first."
                .to_string(),
        ));
    }

    // The item as this run actually holds it. Read from the store rather than
    // from the call, so the value promoted is the value recorded — a model
    // cannot promote one thing having shown a person another.
    let source_scope = MemoryScope::Run {
        run_id: run_id.clone(),
    };
    let held = deps
        .memory
        .recall_one(
            &source_scope,
            &key,
            &session,
            deps.project_of(&session).as_deref(),
        )
        .ok_or_else(|| {
            refuse(
                deps,
                "no such run-scope item",
                format!("This run holds nothing under {key:?} to promote."),
            )
        })?;

    let Some(item) = deps.approvals.find(&approval_id) else {
        return Err(refuse(
            deps,
            "approval not found",
            format!("No approval {approval_id} exists."),
        ));
    };
    let Some(decision) = item.decision.clone() else {
        return Err(refuse(
            deps,
            "approval still pending",
            format!("Approval {approval_id} has not been decided yet."),
        ));
    };
    if !decision.approved() {
        return Err(refuse(
            deps,
            "approval was refused",
            format!("Approval {approval_id} was refused, so nothing is promoted."),
        ));
    }
    // The approval must belong to this run. Otherwise an approval granted for
    // one task would authorise a promotion in another, which is the sort of
    // reuse that is invisible in a queue of similar-looking requests.
    if item.request.task_id != run_id {
        return Err(refuse(
            deps,
            "approval belongs to another run",
            format!("Approval {approval_id} was granted for a different task."),
        ));
    }

    let project = deps.project_of(&session).ok_or_else(|| {
        refuse(
            deps,
            "no project for this person",
            "You are not assigned to a project, so there is no project memory to promote into."
                .to_string(),
        )
    })?;

    let request = Remember {
        scope: MemoryScope::Workspace {
            project_id: project.clone(),
        },
        // Promotion produces a stable project fact. The run-scope kinds do not
        // survive the crossing, and the model does not get to pick.
        kind: MemoryKind::ProjectFact,
        key: key.clone(),
        value: held.value.clone(),
        classification: held.classification,
        source: held.source.clone(),
        approval: None,
        expires_at: None,
    };
    let bound = ApprovalBinding::bind(&approval_id, decision.decided_by(), &request);
    let request = Remember {
        approval: Some(bound.clone()),
        ..request
    };

    let stored = deps.memory.remember(request).map_err(|error| {
        let because = match &error {
            MemoryError::Conflict { .. } => "conflicting memory source".to_string(),
            MemoryError::ApprovalDoesNotCover { field, .. } => {
                format!("approval does not cover the {field}")
            }
            MemoryError::PromotionNeedsApproval { .. } => "promotion needs approval".to_string(),
            MemoryError::SensitivePreference { .. } => "sensitive key".to_string(),
            MemoryError::WrongScope { .. } => "wrong scope for this kind".to_string(),
            MemoryError::NotFound { .. } => "not found".to_string(),
            // A storage failure is not a policy refusal, and conflating the two
            // would hide a disk problem behind a message about permissions.
            MemoryError::Storage { .. } => "the store could not be written".to_string(),
        };
        refuse(deps, &because, error.to_string())
    })?;

    remember_memory_event(
        deps,
        &run_id,
        events::TaskEventType::MemoryPromoted,
        json!({
            "scope": scope_label(&stored.scope),
            "keyHash": bound.key_hash,
            "valueHash": bound.value_hash,
            "sourceHash": bound.source_hash,
            "sourceClassification": bound.source_classification,
            "approvalId": bound.approval_id,
            "approver": bound.approver,
            "policyVersion": bound.policy_version,
        }),
    );

    Ok(json!({
        "promoted": true,
        "key": stored.key,
        "scope": scope_label(&stored.scope),
        "note": "Recorded for this project under the approval you were granted. Changing the value later needs a new approval.",
    }))
}

/// How a scope reads in a record, without naming the person.
///
/// A user scope is written as `user` rather than `user:kiran`: the event
/// envelope already carries the actor, and repeating the id inside the payload
/// puts a second copy of it in a record read by people who do not need it.
fn scope_label(scope: &MemoryScope) -> String {
    match scope {
        MemoryScope::Run { .. } => "run".to_string(),
        MemoryScope::Workspace { project_id } => format!("workspace:{project_id}"),
        MemoryScope::User { .. } => "user".to_string(),
    }
}

/// Writes one memory event into the run's durable history.
///
/// Separate from `RuntimeDeps::remember` only so every payload in this module
/// goes through one place that can be read for what it carries. Everything here
/// is a count, a hash, a scope label or a fixed reason string.
fn remember_memory_event(
    deps: &Arc<RuntimeDeps>,
    run_id: &str,
    event_type: events::TaskEventType,
    payload: Value,
) {
    if run_id.is_empty() {
        // No run to attribute it to. The health probe belongs to no run, and an
        // event keyed on an empty id would be a row nothing can read back.
        return;
    }
    deps.remember(run_id, event_type, payload);
}

/// Records something into this run's own memory.
///
/// Not reachable by a model: it is called from the tool path when Rust has
/// established a fact worth carrying, and the caller supplies the classification
/// and source from material it has already read.
pub fn remember_for_run(
    deps: &Arc<RuntimeDeps>,
    run_id: &str,
    kind: MemoryKind,
    key: &str,
    value: &str,
    classification: Classification,
    source: MemorySource,
) -> Result<MemoryItem, MemoryError> {
    deps.memory.remember(Remember {
        scope: MemoryScope::Run {
            run_id: run_id.to_string(),
        },
        kind,
        key: key.to_string(),
        value: value.to_string(),
        classification,
        source,
        approval: None,
        expires_at: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_three_known_scopes_parse() {
        assert_eq!(RequestedScope::parse("run").unwrap(), RequestedScope::Run);
        assert_eq!(
            RequestedScope::parse("workspace").unwrap(),
            RequestedScope::Workspace
        );
        assert_eq!(RequestedScope::parse("user").unwrap(), RequestedScope::User);
    }

    #[test]
    fn an_unknown_scope_is_refused_by_name() {
        // Including the ones a model would plausibly invent by analogy.
        for attempt in ["global", "workspace:project-b", "RUN", "", "admin"] {
            let refusal = RequestedScope::parse(attempt).expect_err("must be refused");
            assert_eq!(refusal.code, code::BAD_PARAMS, "{attempt:?} was accepted");
        }
    }

    #[test]
    fn a_key_must_be_an_identifier_rather_than_a_payload() {
        assert!(parse_key(Some("unit-price")).is_ok());
        assert!(parse_key(Some("sop.4_2:thickness")).is_ok());

        // A whole document pasted into the key would otherwise reach a durable
        // event by way of the refusal message.
        assert!(parse_key(Some(&"x".repeat(MAX_KEY_CHARS + 1))).is_err());
        assert!(parse_key(Some("")).is_err());
        assert!(parse_key(None).is_err());
        // Path separators and quotes, which is how a key becomes a filename or
        // a fragment of something else.
        for bad in ["../etc/passwd", "a/b", "a\\b", "a b", "key\"; drop"] {
            assert!(parse_key(Some(bad)).is_err(), "{bad:?} was accepted");
        }
    }

    #[test]
    fn a_scope_label_never_carries_the_person() {
        let label = scope_label(&MemoryScope::User {
            user_id: "kiran".to_string(),
        });
        assert_eq!(label, "user");
        assert!(!label.contains("kiran"));
    }
}

//! What a skill declares about itself, once it has been checked.
//!
//! ## The distinction this module exists to hold
//!
//! A `SKILL.md` is a file somebody wrote. Everything in it — including the part
//! that says which tools the skill needs — is a **claim**, not a permission.
//! This module turns claims into a [`SkillManifest`] only when they are
//! well-formed, and a manifest is still only a description. Nothing here grants
//! anything; see [`super::narrowing`] for what a validated manifest is allowed
//! to *do*, which is narrow and never widen.
//!
//! ## Why a card and a manifest are different types
//!
//! [`SkillCard`] is what discovery produces and what `capability.search`
//! returns: a few hundred bytes, enough to decide whether a skill is worth
//! loading. [`SkillManifest`] is the full validated frontmatter. The types are
//! separate so that a caller holding a card cannot accidentally be holding a
//! whole skill — which is the mistake requirement 4 exists to prevent, and the
//! kind a type system is good at preventing permanently.

use serde::{Deserialize, Serialize};

use crate::orchestrator::tools::ToolName;
use crate::policy::Classification;

use super::frontmatter::{Document, Node};

/// Longest a skill name may be.
pub const MAX_NAME: usize = 64;
/// Longest a description may be. Long enough for three sentences; short enough
/// that a hundred cards still fit in a search result a person can read.
pub const MAX_DESCRIPTION: usize = 1024;

/// What a skill says it needs from the network.
///
/// There is deliberately no `external` variant. A skill that wanted to reach
/// the internet would be refused by the broker anyway, and offering the word in
/// the vocabulary would suggest the answer depends on what the skill asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum NetworkNeed {
    /// Touches nothing. The only value a skill may hold in Work mode.
    None,
    /// Wants the loopback inference endpoint the router already chose.
    ///
    /// Declared for honesty rather than permission: every run reaches that
    /// endpoint regardless, and a skill saying so does not change what it may
    /// do. Still quarantined in Work mode, because a skill that felt the need
    /// to ask is one an operator should look at before trusting.
    Loopback,
}

impl NetworkNeed {
    fn parse(raw: &str) -> Option<Self> {
        Some(match raw.trim() {
            "none" => NetworkNeed::None,
            "loopback" => NetworkNeed::Loopback,
            _ => return None,
        })
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            NetworkNeed::None => "none",
            NetworkNeed::Loopback => "loopback",
        }
    }
}

/// Who has to say yes before this skill's actions happen.
///
/// A claim the skill makes about itself, recorded so an operator can see it
/// before loading. The actual decision is the policy gateway's, every time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ApprovalClass {
    /// Nothing this skill does leaves a trace outside the task.
    None,
    /// A reviewer signs off before its side effects happen.
    Reviewer,
}

impl ApprovalClass {
    fn parse(raw: &str) -> Option<Self> {
        Some(match raw.trim() {
            "none" => ApprovalClass::None,
            "reviewer" => ApprovalClass::Reviewer,
            _ => return None,
        })
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            ApprovalClass::None => "none",
            ApprovalClass::Reviewer => "reviewer",
        }
    }
}

/// Why a skill is not available.
///
/// Every variant is a decision an operator can act on, so each carries what
/// would resolve it rather than only what is wrong.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "reason")]
pub enum Quarantine {
    /// The frontmatter could not be read.
    Malformed { detail: String },
    /// `name` does not match the directory the file is in.
    ///
    /// Checked because the directory is what an operator sees and audits, and
    /// a skill whose declared name differs from its location is a skill that
    /// would be listed under one name and loaded from another.
    NameMismatch { declared: String, directory: String },
    /// A required field is absent.
    MissingField { field: String },
    /// A field is present and unusable.
    InvalidField { field: String, detail: String },
    /// It names a tool this build does not have.
    UnknownTool { tool: String },
    /// It wants the network, and this is Work mode.
    RequiresNetwork { need: String },
    /// It declares a binary that is not on this machine.
    MissingBinary { binary: String },
    /// It is for a different version of ARJUN.
    Incompatible { requires: String, running: String },
    /// Its content hash is not in the operator's trust list.
    Unsigned { sha256: String },
    /// Its content hash is in the trust list under a *different* value, which
    /// means the file changed after it was trusted.
    Tampered { expected: String, found: String },
}

impl Quarantine {
    /// One line for the operator, saying what would resolve it.
    pub fn explain(&self) -> String {
        match self {
            Quarantine::Malformed { detail } => {
                format!("Its SKILL.md could not be read: {detail}")
            }
            Quarantine::NameMismatch { declared, directory } => format!(
                "It calls itself {declared:?} but lives in a directory called {directory:?}. \
                 Rename one to match the other."
            ),
            Quarantine::MissingField { field } => {
                format!("Its frontmatter has no {field:?}, which is required.")
            }
            Quarantine::InvalidField { field, detail } => {
                format!("Its {field:?} is not usable: {detail}")
            }
            Quarantine::UnknownTool { tool } => format!(
                "It asks for a tool called {tool:?}, which this build does not have. A skill \
                 cannot introduce a tool."
            ),
            Quarantine::RequiresNetwork { need } => format!(
                "It declares network use ({need}), and Work mode permits none. It would be \
                 available in Provisioning mode, where no confidential material may be handled."
            ),
            Quarantine::MissingBinary { binary } => {
                format!("It needs {binary:?}, which is not on this machine.")
            }
            Quarantine::Incompatible { requires, running } => format!(
                "It is written for ARJUN {requires} and this is {running}."
            ),
            Quarantine::Unsigned { sha256 } => format!(
                "Its contents are not in the trust list. Review the skill and add \
                 {} to skills/trusted.json to allow it.",
                &sha256[..16.min(sha256.len())]
            ),
            Quarantine::Tampered { expected, found } => format!(
                "Its contents changed after it was trusted: the trust list expects {}… and the \
                 file hashes to {}…. Review the change before trusting it again.",
                &expected[..16.min(expected.len())],
                &found[..16.min(found.len())]
            ),
        }
    }
}

/// What a skill declares, validated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillManifest {
    pub name: String,
    pub description: String,
    pub version: String,
    pub license: String,
    pub author: String,
    /// The ARJUN version requirement, verbatim.
    pub requires_arjun: String,
    pub requires_binaries: Vec<String>,
    pub network: NetworkNeed,
    /// The most sensitive material this skill is written for. A skill is only
    /// offered to somebody cleared for it.
    pub classification: Classification,
    /// Tools the skill says it needs. A *request*, and the ceiling of what it
    /// may use — never an addition to what the run already permits.
    pub allowed_tools: Vec<ToolName>,
    pub approval_class: ApprovalClass,
    /// Everything else the author wrote, as scalars. Carried so an operator can
    /// read it; never consulted for a decision.
    pub metadata: std::collections::BTreeMap<String, String>,
    /// SHA-256 of the whole SKILL.md, frontmatter and body.
    pub sha256: String,
}

/// How many skill cards one `capability.search` may put in front of a model.
///
/// Not a style choice, and the reasoning is in [`SkillCard::for_model`]: even
/// slimmed, sixty-five cards do not belong in an 8192-token window. Twelve is
/// what fits an eighth of it with room to spare, and the response says how
/// many matched and how to narrow, so nothing is hidden — only paged.
///
/// The operator's listing (`skill_search`) is not paged. A desktop screen has
/// no context window.
pub const CAPABILITY_PAGE: usize = 12;

/// The concise form: what discovery keeps and what `capability.search` returns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillCard {
    pub name: String,
    pub description: String,
    pub version: String,
    pub classification: Classification,
    pub allowed_tools: Vec<String>,
    pub approval_class: ApprovalClass,
    pub network: NetworkNeed,
    pub sha256: String,
    /// Absent when the skill is available.
    pub quarantined: Option<Quarantine>,
    /// Whether this skill came from an outside collection rather than being
    /// written for this product.
    ///
    /// Read from the manifest's own `imported-from` metadata, which the import
    /// sets and nothing else does. It is on the card for two reasons: an
    /// operator looking at the Skills screen should be able to tell where a
    /// skill came from, and `capability.search` puts the skills written for
    /// this deployment on the first page — see the ordering there.
    pub imported: bool,
}

impl SkillCard {
    pub fn of(manifest: &SkillManifest, quarantined: Option<Quarantine>) -> Self {
        Self {
            name: manifest.name.clone(),
            description: manifest.description.clone(),
            version: manifest.version.clone(),
            classification: manifest.classification,
            allowed_tools: manifest
                .allowed_tools
                .iter()
                .map(|tool| tool.as_str().to_string())
                .collect(),
            approval_class: manifest.approval_class,
            network: manifest.network,
            sha256: manifest.sha256.clone(),
            quarantined,
            imported: manifest.metadata.contains_key("imported-from"),
        }
    }

    /// What a *model* is told about this skill.
    ///
    /// Deliberately less than the card. The full card carries a sha256, a
    /// version and a classification — evidence for the person reading the
    /// Skills screen, and nothing a model can act on. Sixty-five full cards
    /// serialise to about 32 KB, roughly eight thousand tokens, which is the
    /// entire window of the smallest model this product serves
    /// (`ai_engine::manager::DEFAULT_WORKING_CONTEXT`). One unprompted
    /// `capability.search` would leave no room for the person's own documents.
    ///
    /// What survives is what changes a decision: which skill to ask for, what
    /// it would let the run do, and whether using it stops for a reviewer.
    pub fn for_model(&self) -> serde_json::Value {
        serde_json::json!({
            "name": self.name,
            "description": self.description,
            "tools": self.allowed_tools,
            "needsApproval": self.approval_class == ApprovalClass::Reviewer,
        })
    }

    /// A card for a directory whose `SKILL.md` did not validate.
    ///
    /// There is no manifest to describe it, and it is still listed — an
    /// operator with a broken skill needs to see that it is there and what is
    /// wrong with it. Hiding it looks exactly like the skill was never
    /// installed, and sends somebody looking for a file that is right in front
    /// of them.
    ///
    /// Named from the directory, because the declared name is one of the things
    /// that may be missing or wrong. Classified `Internal` — the broadest
    /// clearance — because a folder name and a parse error are operational
    /// facts rather than confidential ones, and because the alternative is a
    /// broken skill nobody can see.
    pub fn unreadable(directory: &str, quarantined: Quarantine) -> Self {
        Self {
            name: directory.to_string(),
            description: String::new(),
            version: String::new(),
            classification: Classification::Internal,
            allowed_tools: Vec::new(),
            approval_class: ApprovalClass::None,
            network: NetworkNeed::None,
            sha256: String::new(),
            quarantined: Some(quarantined),
            // Nothing is known about a directory that did not validate,
            // including where it came from.
            imported: false,
        }
    }

    pub fn is_available(&self) -> bool {
        self.quarantined.is_none()
    }
}

/// True for a lowercase, hyphen-separated name with no leading or trailing
/// hyphen and no run of two.
pub fn is_valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_NAME
        && name.starts_with(|c: char| c.is_ascii_lowercase())
        && name.ends_with(|c: char| c.is_ascii_lowercase() || c.is_ascii_digit())
        && !name.contains("--")
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

/// Reads a validated manifest out of a parsed frontmatter block.
///
/// `directory` is the name of the folder the file was found in, and the name
/// must match it. `sha256` is over the whole file, so the manifest carries the
/// identity of the exact bytes that produced it.
pub fn validate(
    document: &Document,
    directory: &str,
    sha256: &str,
) -> Result<SkillManifest, Quarantine> {
    let required = |field: &str| -> Result<String, Quarantine> {
        document
            .scalar(field)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .ok_or_else(|| Quarantine::MissingField {
                field: field.to_string(),
            })
    };

    let name = required("name")?;
    if !is_valid_name(&name) {
        return Err(Quarantine::InvalidField {
            field: "name".to_string(),
            detail: format!(
                "{name:?} must be lowercase letters, digits and single hyphens, at most \
                 {MAX_NAME} characters, starting with a letter"
            ),
        });
    }
    if name != directory {
        return Err(Quarantine::NameMismatch {
            declared: name,
            directory: directory.to_string(),
        });
    }

    let description = required("description")?;
    if description.chars().count() > MAX_DESCRIPTION {
        return Err(Quarantine::InvalidField {
            field: "description".to_string(),
            detail: format!(
                "it is {} characters, above the {MAX_DESCRIPTION} character limit",
                description.chars().count()
            ),
        });
    }

    let version = required("version")?;
    if parse_version(&version).is_none() {
        return Err(Quarantine::InvalidField {
            field: "version".to_string(),
            detail: format!("{version:?} is not a `major.minor.patch` version"),
        });
    }

    let license = required("license")?;
    let author = required("author")?;

    let compatibility = document
        .map("compatibility")
        .ok_or_else(|| Quarantine::MissingField {
            field: "compatibility".to_string(),
        })?;
    let requires_arjun = compatibility
        .get("arjun")
        .and_then(Node::as_scalar)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| Quarantine::MissingField {
            field: "compatibility.arjun".to_string(),
        })?
        .to_string();
    let requires_binaries = match compatibility.get("requires-binaries") {
        None => Vec::new(),
        Some(Node::List(items)) => items.clone(),
        Some(_) => {
            return Err(Quarantine::InvalidField {
                field: "compatibility.requires-binaries".to_string(),
                detail: "it must be a list, written one item per line".to_string(),
            })
        }
    };

    let network_raw = required("network")?;
    let network = NetworkNeed::parse(&network_raw).ok_or_else(|| Quarantine::InvalidField {
        field: "network".to_string(),
        detail: format!("{network_raw:?} must be `none` or `loopback`"),
    })?;

    let classification_raw = required("classification")?;
    let classification = parse_classification(&classification_raw).ok_or_else(|| {
        Quarantine::InvalidField {
            field: "classification".to_string(),
            detail: format!(
                "{classification_raw:?} is not one of: internal, processDiagram, financial, \
                 vendorNegotiation, unreleasedDesign, internalCorrespondence, businessStrategy"
            ),
        }
    })?;

    let declared_tools = document
        .list("allowed-tools")
        .ok_or_else(|| Quarantine::MissingField {
            field: "allowed-tools".to_string(),
        })?;
    if declared_tools.is_empty() {
        return Err(Quarantine::InvalidField {
            field: "allowed-tools".to_string(),
            detail: "a skill with no tools cannot do anything; omit the skill instead".to_string(),
        });
    }
    let mut allowed_tools = Vec::new();
    for raw in declared_tools {
        // A name this build does not know is quarantined rather than skipped.
        // Skipping would let a skill list a tool that a future build adds and
        // be silently more capable after an upgrade nobody reviewed.
        let tool = ToolName::from_str(raw.trim()).ok_or_else(|| Quarantine::UnknownTool {
            tool: raw.clone(),
        })?;
        if !allowed_tools.contains(&tool) {
            allowed_tools.push(tool);
        }
    }

    let metadata_map = document.map("metadata");
    let approval_raw = metadata_map
        .and_then(|fields| fields.get("approval-class"))
        .and_then(Node::as_scalar)
        .unwrap_or("none");
    let approval_class =
        ApprovalClass::parse(approval_raw).ok_or_else(|| Quarantine::InvalidField {
            field: "metadata.approval-class".to_string(),
            detail: format!("{approval_raw:?} must be `none` or `reviewer`"),
        })?;

    let metadata = metadata_map
        .map(|fields| {
            fields
                .iter()
                .filter_map(|(key, node)| {
                    node.as_scalar().map(|value| (key.clone(), value.to_string()))
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(SkillManifest {
        name,
        description,
        version,
        license,
        author,
        requires_arjun,
        requires_binaries,
        network,
        classification,
        allowed_tools,
        approval_class,
        metadata,
        sha256: sha256.to_string(),
    })
}

/// `major.minor.patch` as three numbers.
pub fn parse_version(raw: &str) -> Option<(u32, u32, u32)> {
    let mut parts = raw.trim().split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((major, minor, patch))
}

/// Whether `running` satisfies a requirement like `>=0.1.0`, `0.1.0` or `*`.
///
/// A deliberately tiny grammar. Anything it does not understand is treated as
/// unsatisfied, so an unfamiliar requirement quarantines the skill rather than
/// being read as permissive.
pub fn satisfies(requirement: &str, running: &str) -> bool {
    let Some(running) = parse_version(running) else {
        return false;
    };
    let requirement = requirement.trim();
    if requirement == "*" {
        return true;
    }
    if let Some(rest) = requirement.strip_prefix(">=") {
        return parse_version(rest).is_some_and(|wanted| running >= wanted);
    }
    if let Some(rest) = requirement.strip_prefix('=') {
        return parse_version(rest).is_some_and(|wanted| running == wanted);
    }
    parse_version(requirement).is_some_and(|wanted| running == wanted)
}

/// Reads a classification from its camelCase wire name.
fn parse_classification(raw: &str) -> Option<Classification> {
    serde_json::from_value(serde_json::Value::String(raw.trim().to_string())).ok()
}

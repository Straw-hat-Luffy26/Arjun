//! Choosing which skills a run should have, before it starts.
//!
//! ## The gap this closes
//!
//! Sixty-five skills are installed, hash-pinned and loadable, and for artifact
//! generation every one of them was inert. `planning.rs` does not mention
//! skills; nothing selects one; `skill.load` is reachable **only if the model
//! thinks to call it**. A skill on disk that no run ever loads is a file, not a
//! capability.
//!
//! This selects them deterministically, from two things a run knows before it
//! begins: the format it is going to produce, and the words the person used.
//!
//! ## Why deterministic, and in Rust
//!
//! Because it has to be the same every time and it has to be testable. A model
//! asked "which skills do you want" gives a different answer on a re-run, which
//! makes "did the skill reach the model" unanswerable — and that question is
//! the whole point of the instrumentation around this.
//!
//! ## What this does not do
//!
//! It does not grant anything. Selection is a *proposal*: every name here still
//! goes through [`super::SkillRegistry::load`], which checks the trust list, the
//! hash on disk now, the signed-in person's clearance, the ARJUN version and the
//! sovereignty mode, and then through [`super::narrowing::narrow`], which can
//! only ever remove tools from a run. A skill this module picked and the
//! registry refused is simply not loaded, and the refusal is recorded.
//!
//! Quarantined skills are never proposed: `skill.load` would refuse them, so
//! offering one spends a step to earn a refusal.

use std::collections::BTreeMap;

use super::{SkillCard, SkillContext, SkillRegistry};

/// Why a skill was chosen. Carried so the record can say more than "a skill was
/// used" — an operator asking *why this one* has an answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reason {
    /// The run is producing this format and the skill declares itself for it.
    OutputFormat(String),
    /// The request's own words matched the skill's name or description.
    Domain(String),
    /// A source this run may read is of a kind the skill is written for.
    ///
    /// The kind comes from the backend's own extraction record, so this fires
    /// on what a document turned out to *be* rather than on what it was called.
    InputKind(String),
    /// The run is answering from a notebook's selected sources.
    NotebookResearch,
}

impl Reason {
    pub fn explain(&self) -> String {
        match self {
            Reason::OutputFormat(format) => {
                format!("the run produces a .{format} and this skill is written for that format")
            }
            Reason::Domain(term) => format!("the request mentions {term:?}"),
            Reason::InputKind(kind) => {
                format!("a source this run may read was extracted as {kind:?}")
            }
            Reason::NotebookResearch => {
                "the question is scoped to a notebook's selected sources".to_string()
            }
        }
    }
}

/// What this run is about to *read*, as the backend resolved it.
///
/// Deliberately not filenames. Every field here is the outcome of work that
/// already happened -- an extraction kind written when the document was read, a
/// readiness state worked out from what is on disk, a capability asked of the
/// model registry. A selector fed filenames would choose `source-pdf` for a
/// `.pdf` that turned out to be a scan nothing could read, and
/// `source-spreadsheet` for a `.xls` that is really XML.
#[derive(Debug, Clone, Default)]
pub struct ReadingContext {
    /// Extraction kinds of the sources in scope: `pdf-text`, `pdf-scan`,
    /// `xlsx`, `docx`, `pptx`, `image`, `text`, `pasted`.
    pub source_kinds: Vec<String>,
    /// True when the question is scoped to a notebook.
    pub notebook: bool,
    /// At least one source needs OCR or vision to be read at all.
    pub needs_vision: bool,
    /// This machine has a vision or document-OCR model available.
    pub vision_available: bool,
    /// At least one source is a legacy binary Office file.
    pub legacy_office: bool,
}

impl ReadingContext {
    /// Whether there is anything here for an input skill to be chosen for.
    pub fn is_empty(&self) -> bool {
        !self.notebook && self.source_kinds.is_empty()
    }
}

/// How many input-reading skills one run may carry.
///
/// Two, plus the coordinator. A mixed notebook -- a PDF, a workbook and a deck
/// -- must not load every reading skill's body into the window; the two whose
/// sources dominate the scope are the ones whose guidance changes the answer,
/// and the coordinator knows to hand the rest off.
pub const MAX_INPUT_SKILLS: usize = 2;

/// Picks the skills for what this run is going to *read*.
///
/// Separate from [`select`], which picks for what the run will *produce*. They
/// are different questions with different budgets: a run can be reading three
/// formats and writing none.
pub fn select_for_reading(
    reading: &ReadingContext,
    registry: &SkillRegistry,
    context: &SkillContext<'_>,
) -> Vec<Selection> {
    if reading.is_empty() {
        return Vec::new();
    }

    let available: Vec<SkillCard> = registry
        .search("", context)
        .into_iter()
        .filter(SkillCard::is_available)
        .collect();

    let mut chosen: Vec<Selection> = Vec::new();

    // 1. The coordinator, whenever a notebook is in scope. It decides whether
    //    this is a lookup, an overview, a comparison or a calculation, and
    //    those need different evidence out of the same sources.
    if reading.notebook {
        if let Some(card) = available
            .iter()
            .find(|card| card.inputs.iter().any(|kind| kind == "notebook"))
        {
            chosen.push(Selection {
                name: card.name.clone(),
                reason: Reason::NotebookResearch,
            });
        }
    }

    // 2. Reading skills, by how many of the sources in scope they cover.
    //
    //    Counted rather than first-match: a notebook of six workbooks and one
    //    PDF should reach the spreadsheet skill first, because that is where
    //    nearly every answer is going to come from.
    let mut coverage: BTreeMap<String, (usize, SkillCard, String)> = BTreeMap::new();
    for kind in &reading.source_kinds {
        let lowered = kind.to_lowercase();
        for card in &available {
            if chosen.iter().any(|s| s.name == card.name) {
                continue;
            }
            if !card.inputs.iter().any(|declared| *declared == lowered) {
                continue;
            }
            let entry = coverage
                .entry(card.name.clone())
                .or_insert((0, card.clone(), lowered.clone()));
            entry.0 += 1;
        }
    }

    // A legacy binary Office file reaches the conversion skill regardless of
    // how many sources are like it: it is the difference between reading the
    // document and reporting that nothing on this machine can open it.
    if reading.legacy_office {
        if let Some(card) = available
            .iter()
            .filter(|card| !chosen.iter().any(|s| s.name == card.name))
            .find(|card| card.inputs.iter().any(|kind| kind == "legacy-office"))
        {
            coverage.insert(
                card.name.clone(),
                (usize::MAX, card.clone(), "legacy-office".to_string()),
            );
        }
    }

    let mut ranked: Vec<(usize, SkillCard, String)> = coverage.into_values().collect();
    // Most sources covered first; ties broken by name so a re-run picks the
    // same skills.
    ranked.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.name.cmp(&b.1.name)));

    for (_, card, kind) in ranked.into_iter().take(MAX_INPUT_SKILLS) {
        chosen.push(Selection {
            name: card.name.clone(),
            reason: Reason::InputKind(kind),
        });
    }

    chosen
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    pub name: String,
    pub reason: Reason,
}

/// How many skills one run may carry.
///
/// Each loaded skill's body goes into the model's context, so this is a context
/// budget rather than a preference. Three is enough for a format skill and two
/// domain skills, which is the shape nearly every real request has.
pub const MAX_SELECTED: usize = 3;

/// Words too common to mean anything. Matching on these would select a skill
/// for every request that contains the word "the".
const STOP_WORDS: &[&str] = &[
    "the", "and", "for", "with", "from", "this", "that", "into", "make", "create", "write",
    "produce", "generate", "please", "need", "want", "about", "using", "give", "show", "report",
    "file", "document", "have", "been", "will", "would", "should", "could", "what", "when",
    "where", "which", "there", "their", "them", "then", "than", "some", "more", "most", "also",
    "only", "very", "such", "each", "other", "over", "under", "after", "before",
];

fn terms(request: &str) -> Vec<String> {
    request
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '-')
        .filter(|word| word.len() >= 4)
        .filter(|word| !STOP_WORDS.contains(word))
        .map(str::to_string)
        .collect()
}

/// Picks the skills for a run.
///
/// `output_format` is the extension the run is going to write, without the dot.
/// `request` is the person's own words — not the composed prompt, for the same
/// reason routing uses the question rather than the attachment: a hundred
/// kilobytes of scanned text matches every keyword there is.
pub fn select(
    request: &str,
    output_format: Option<&str>,
    registry: &SkillRegistry,
    context: &SkillContext<'_>,
) -> Vec<Selection> {
    // The registry's own search applies clearance and the operating mode, so
    // what comes back is already what this person may see.
    let available: Vec<SkillCard> = registry
        .search("", context)
        .into_iter()
        .filter(SkillCard::is_available)
        .collect();

    let mut chosen: Vec<Selection> = Vec::new();

    // 1. The format skill. At most one: two documents' worth of house style is
    //    a contradiction, not twice the guidance.
    if let Some(format) = output_format.map(|f| f.trim_start_matches('.').to_lowercase()) {
        if let Some(card) = available
            .iter()
            .find(|card| card.formats.iter().any(|declared| *declared == format))
        {
            chosen.push(Selection {
                name: card.name.clone(),
                reason: Reason::OutputFormat(format),
            });
        }
    }

    // 2. Domain skills, by how much of the request they match.
    //
    //    Scored rather than first-match: a request about a pump and a vendor
    //    should reach the skill that covers both before either of the ones that
    //    cover half of it.
    let terms = terms(request);
    let mut scored: Vec<(usize, &SkillCard, String)> = Vec::new();
    for card in &available {
        if chosen.iter().any(|s| s.name == card.name) {
            continue;
        }
        let haystack = format!("{} {}", card.name, card.description).to_lowercase();
        let mut hits = 0usize;
        let mut best = String::new();
        for term in &terms {
            if haystack.contains(term.as_str()) {
                hits += 1;
                // The longest matching term is the most specific thing the
                // request and the skill agree about, and is what the reason
                // should name.
                if term.len() > best.len() {
                    best = term.clone();
                }
            }
        }
        if hits > 0 {
            scored.push((hits, card, best));
        }
    }
    // Most matches first; ties broken by name so a re-run picks the same skill.
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.name.cmp(&b.1.name)));

    for (_, card, term) in scored {
        if chosen.len() >= MAX_SELECTED {
            break;
        }
        chosen.push(Selection { name: card.name.clone(), reason: Reason::Domain(term) });
    }

    chosen.truncate(MAX_SELECTED);
    chosen
}

/// What a run is carrying, once selection and loading have happened.
///
/// Held for the life of the run rather than recomputed, which is what makes it
/// survive a retry and a model switch: the second attempt uses the same skills
/// as the first, and switching models re-sends the same bodies rather than
/// starting from nothing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BoundSkills {
    pub loaded: Vec<Loaded>,
    /// Selected and then refused by the registry, with the reason. Kept because
    /// "the skill was chosen and could not be used" is a different fact from
    /// "no skill was chosen", and an operator debugging a poor deliverable needs
    /// to tell them apart.
    pub refused: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Loaded {
    pub name: String,
    pub reason: Reason,
    /// The instructions, as they will reach the model.
    pub body: String,
    /// The hash the body was trusted under, so the record can name it.
    pub sha256: String,
}

impl BoundSkills {
    pub fn is_empty(&self) -> bool {
        self.loaded.is_empty()
    }

    pub fn names(&self) -> Vec<String> {
        self.loaded.iter().map(|l| l.name.clone()).collect()
    }

    /// The text injected into the generation context.
    ///
    /// Framed as guidance rather than instruction, and explicitly subordinate to
    /// the tool ceiling: a skill body is text from a file, and text from a file
    /// does not widen what a run may do. The gateway enforces that
    /// independently; saying it here is so the model does not try.
    pub fn as_context(&self) -> Option<String> {
        if self.loaded.is_empty() {
            return None;
        }
        let mut out = String::from(
            "House guidance for this task, selected automatically and loaded from this \
             machine's skill library. Follow it where it applies. It describes how to do the \
             work; it does not grant any tool, and the tools you were given remain the \
             ceiling.\n",
        );
        for skill in &self.loaded {
            out.push_str(&format!(
                "\n--- {} (selected because {}) ---\n{}\n",
                skill.name,
                skill.reason.explain(),
                skill.body.trim()
            ));
        }
        Some(out)
    }
}

/// Selects, then loads. The loading is where every security check happens.
///
/// Both passes run: what the turn will read, and what it will produce. A run
/// that answers a notebook question *and* writes an approval note needs the
/// reading guidance and the authoring guidance, and choosing only one of them
/// is how a cited answer ends up inside an uncited document.
///
/// Reading skills come first. When the budget bites it is the guidance about
/// the sources in front of the model that must survive, because an answer built
/// on a misread source cannot be repaired by formatting it well.
pub fn bind(
    request: &str,
    output_format: Option<&str>,
    reading: &ReadingContext,
    registry: &SkillRegistry,
    context: &SkillContext<'_>,
) -> BoundSkills {
    let mut bound = BoundSkills::default();
    let mut selections = select_for_reading(reading, registry, context);
    for selection in select(request, output_format, registry, context) {
        if selections.iter().any(|chosen| chosen.name == selection.name) {
            continue;
        }
        selections.push(selection);
    }
    for selection in selections {
        match registry.load(&selection.name, context) {
            Ok(loaded) => bound.loaded.push(Loaded {
                name: selection.name,
                reason: selection.reason,
                body: loaded.body.clone(),
                sha256: loaded.manifest.sha256.clone(),
            }),
            Err(refusal) => bound.refused.push((selection.name, refusal.explain())),
        }
    }
    bound
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{Role, Session, User};
    use crate::orchestrator::tools::ToolName;
    use crate::sovereignty::mode::OperatingMode;

    fn shipped() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("src-tauri has a parent")
            .join("skills")
    }

    fn session() -> Session {
        Session::open(User::new("priya", "Priya Sharma", vec![Role::Employee]))
    }

    fn context(session: &Session) -> SkillContext<'_> {
        SkillContext { session, mode: OperatingMode::Work, run_permits: ToolName::ALL }
    }

    #[test]
    fn the_request_decides_which_domain_skills_are_chosen() {
        let registry = SkillRegistry::open(shipped());
        let session = session();
        let chosen = select(
            "Read the P&ID and trace the line from the cooling water pump",
            None,
            &registry,
            &context(&session),
        );

        assert!(!chosen.is_empty(), "a plant request must reach a plant skill");
        assert!(
            chosen.iter().all(|s| matches!(s.reason, Reason::Domain(_))),
            "no format was named, so nothing may be chosen for a format: {chosen:?}"
        );
    }

    /// Determinism. A selector that answers differently on a re-run makes
    /// "which skill did this run use" unanswerable.
    #[test]
    fn the_same_request_selects_the_same_skills_every_time() {
        let registry = SkillRegistry::open(shipped());
        let session = session();
        let request = "Write the hazard and operability study for the new transfer line";

        let first = select(request, Some("docx"), &registry, &context(&session));
        for _ in 0..5 {
            assert_eq!(
                select(request, Some("docx"), &registry, &context(&session)),
                first,
                "selection must not vary between runs"
            );
        }
    }

    #[test]
    fn nothing_is_selected_for_a_request_that_matches_nothing() {
        let registry = SkillRegistry::open(shipped());
        let session = session();
        // Only stop words and short words, so nothing can match.
        let chosen = select("do it for me now", None, &registry, &context(&session));
        assert!(chosen.is_empty(), "{chosen:?}");
    }

    #[test]
    fn no_more_than_the_context_budget_is_ever_selected() {
        let registry = SkillRegistry::open(shipped());
        let session = session();
        let chosen = select(
            "inspection vendor calculation safety document deck workbook diagram approval \
             hazard equipment multimodal sandbox verification evaluation",
            Some("docx"),
            &registry,
            &context(&session),
        );
        assert!(chosen.len() <= MAX_SELECTED, "{} selected", chosen.len());
    }

    /// Selection is a proposal. Loading is where clearance, the trust list and
    /// the hash on disk are checked, and `bind` goes through it.
    #[test]
    fn binding_loads_through_the_registry_so_every_check_still_runs() {
        let registry = SkillRegistry::open(shipped());
        let session = session();
        let bound = bind(
            "Read the P&ID and trace the line from the cooling water pump",
            None,
            &ReadingContext::default(),
            &registry,
            &context(&session),
        );

        assert!(!bound.is_empty(), "{:?}", bound.refused);
        for skill in &bound.loaded {
            assert!(!skill.body.trim().is_empty(), "{} loaded an empty body", skill.name);
            assert_eq!(
                skill.sha256.len(),
                64,
                "a loaded skill carries the hash it was trusted under"
            );
        }
    }

    #[test]
    fn the_injected_context_names_each_skill_and_why_it_was_chosen() {
        let registry = SkillRegistry::open(shipped());
        let session = session();
        let bound = bind(
            "Read the P&ID and trace the line from the cooling water pump",
            None,
            &ReadingContext::default(),
            &registry,
            &context(&session),
        );
        let text = bound.as_context().expect("something was selected");

        for skill in &bound.loaded {
            assert!(text.contains(&skill.name), "{} is not in the injected text", skill.name);
        }
        assert!(
            text.contains("does not grant any tool"),
            "the injected text must say a skill cannot widen the run"
        );
    }

    // -- Choosing skills for what a run will read ----------------------
    //
    // The other half of selection. These are regressions for the failure the
    // input pass exists to prevent: nine source-reading skills sitting on disk,
    // hash-pinned and loadable, and inert in every run because nothing ever
    // chose one.

    fn reading(kinds: &[&str], notebook: bool) -> ReadingContext {
        ReadingContext {
            source_kinds: kinds.iter().map(|k| (*k).to_string()).collect(),
            notebook,
            ..Default::default()
        }
    }

    #[test]
    fn a_notebook_question_reaches_the_research_coordinator() {
        let registry = SkillRegistry::open(shipped());
        let session = session();
        let chosen = select_for_reading(&reading(&[], true), &registry, &context(&session));

        assert!(
            chosen.iter().any(|s| s.name == "notebook-grounded-research"),
            "a notebook question must reach the coordinator: {chosen:?}"
        );
    }

    #[test]
    fn the_extraction_kind_decides_which_reader_is_chosen() {
        let registry = SkillRegistry::open(shipped());
        let session = session();

        for (kind, expected) in [
            ("pdf-text", "source-pdf"),
            ("pdf-scan", "source-pdf"),
            ("xlsx", "source-spreadsheet"),
            ("csv", "source-spreadsheet"),
            ("docx", "source-word"),
            ("pptx", "source-presentation"),
            ("image", "source-image-drawing"),
            ("pasted", "source-copied-text"),
        ] {
            let chosen =
                select_for_reading(&reading(&[kind], false), &registry, &context(&session));
            assert!(
                chosen.iter().any(|s| s.name == expected),
                "a {kind:?} source must reach {expected}: {chosen:?}"
            );
        }
    }

    /// The rule the whole design rests on: activation follows what a document
    /// turned out to *be*, not what it was called.
    #[test]
    fn a_filename_alone_never_chooses_a_reader() {
        let registry = SkillRegistry::open(shipped());
        let session = session();

        // Nothing has been extracted, so there is no kind, so no reader is
        // chosen -- however many files the request names.
        let chosen =
            select_for_reading(&ReadingContext::default(), &registry, &context(&session));
        assert!(chosen.is_empty(), "{chosen:?}");

        // A `.xls` whose content turned out to be an XML spreadsheet reaches the
        // spreadsheet reader through its resolved kind, not its name -- the name
        // alone would have routed it to a legacy converter.
        let chosen =
            select_for_reading(&reading(&["xlsx"], false), &registry, &context(&session));
        assert!(
            chosen.iter().any(|s| s.name == "source-spreadsheet"),
            "{chosen:?}"
        );
        assert!(
            !chosen.iter().any(|s| s.name == "legacy-office-conversion"),
            "nothing here needs converting: {chosen:?}"
        );
    }

    /// A parser that cannot open a container is what identifies a legacy file.
    #[test]
    fn a_source_no_parser_can_open_reaches_the_conversion_skill() {
        let registry = SkillRegistry::open(shipped());
        let session = session();
        let mut scope = reading(&["pdf-text"], true);
        scope.legacy_office = true;

        let chosen = select_for_reading(&scope, &registry, &context(&session));
        assert!(
            chosen.iter().any(|s| s.name == "legacy-office-conversion"),
            "{chosen:?}"
        );
    }

    /// A mixed notebook must not load every reading skill body into one turn.
    #[test]
    fn a_mixed_notebook_stays_within_the_context_budget() {
        let registry = SkillRegistry::open(shipped());
        let session = session();
        let chosen = select_for_reading(
            &reading(
                &["pdf-text", "xlsx", "docx", "pptx", "image", "pasted", "txt"],
                true,
            ),
            &registry,
            &context(&session),
        );

        assert!(
            chosen.len() <= MAX_INPUT_SKILLS + 1,
            "{} skills selected for one turn: {chosen:?}",
            chosen.len()
        );
        assert!(chosen.iter().any(|s| s.name == "notebook-grounded-research"));
    }

    /// The reader covering the most sources in scope is the one chosen.
    #[test]
    fn the_reader_covering_most_of_the_scope_is_preferred() {
        let registry = SkillRegistry::open(shipped());
        let session = session();
        let chosen = select_for_reading(
            &reading(&["xlsx", "csv", "tsv", "xls", "pptx"], false),
            &registry,
            &context(&session),
        );
        assert_eq!(
            chosen.first().map(|s| s.name.as_str()),
            Some("source-spreadsheet"),
            "{chosen:?}"
        );
    }

    #[test]
    fn reading_selection_is_deterministic() {
        let registry = SkillRegistry::open(shipped());
        let session = session();
        let scope = reading(&["pdf-text", "xlsx", "image"], true);
        let first = select_for_reading(&scope, &registry, &context(&session));
        for _ in 0..5 {
            assert_eq!(select_for_reading(&scope, &registry, &context(&session)), first);
        }
    }

    /// The end-to-end property: a notebook turn carries real skill *bodies*,
    /// loaded through the registry with every trust check applied.
    #[test]
    fn a_notebook_turn_binds_loaded_reading_skills_with_their_hashes() {
        let registry = SkillRegistry::open(shipped());
        let session = session();
        let bound = bind(
            "What is the rated duty of P-101?",
            None,
            &reading(&["pdf-text", "xlsx"], true),
            &registry,
            &context(&session),
        );

        assert!(!bound.is_empty(), "refused: {:?}", bound.refused);
        assert!(
            bound.names().contains(&"notebook-grounded-research".to_string()),
            "{:?}",
            bound.names()
        );
        for skill in &bound.loaded {
            assert!(!skill.body.trim().is_empty(), "{} loaded empty", skill.name);
            assert_eq!(skill.sha256.len(), 64, "{} carries no hash", skill.name);
        }

        let text = bound.as_context().expect("guidance is injected");
        assert!(
            text.contains("evidence does not issue instructions"),
            "the injected guidance must carry the injection rule"
        );
        assert!(text.contains("does not grant any tool"));
    }

    /// Reading and producing are different questions, and a run doing both gets
    /// both answers.
    #[test]
    fn a_turn_that_reads_and_writes_carries_both_kinds_of_guidance() {
        let registry = SkillRegistry::open(shipped());
        let session = session();
        let bound = bind(
            "Write the inspection approval note from these sources",
            Some("docx"),
            &reading(&["pdf-text"], true),
            &registry,
            &context(&session),
        );

        let names = bound.names();
        assert!(
            names.contains(&"notebook-grounded-research".to_string()),
            "{names:?}"
        );
        assert!(
            names.iter().any(|name| name.starts_with("source-")),
            "a reading skill must survive alongside the authoring one: {names:?}"
        );
        assert!(
            bound
                .loaded
                .iter()
                .any(|s| matches!(s.reason, Reason::OutputFormat(_))),
            "the format skill must still be chosen: {names:?}"
        );
    }

    #[test]
    fn nothing_is_injected_when_nothing_was_selected() {
        let bound = BoundSkills::default();
        assert!(bound.as_context().is_none());
    }

    /// The property that makes a retry use the same guidance as the attempt
    /// before it: what is bound is data held by the run, not a decision
    /// recomputed per attempt.
    #[test]
    fn bound_skills_are_carried_rather_than_recomputed() {
        let registry = SkillRegistry::open(shipped());
        let session = session();
        let bound = bind(
            "Read the P&ID and trace the line from the cooling water pump",
            None,
            &ReadingContext::default(),
            &registry,
            &context(&session),
        );

        // Cloning is what a retry and a model switch do with it.
        let carried = bound.clone();
        assert_eq!(carried, bound);
        assert_eq!(carried.names(), bound.names());
        assert_eq!(carried.as_context(), bound.as_context());
    }
}

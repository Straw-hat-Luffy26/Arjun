//! Turning what a model wrote into something a later turn can reuse.
//!
//! ## Why this exists
//!
//! "One model creates a diagram, another creates code" almost never means two
//! files. It means two fenced blocks inside two assistant messages: a mermaid
//! block the chat renders with its own layout, and a python block the chat
//! renders with syntax highlighting. Both are *display* of message text.
//! Nothing on the backend ever captured either, so when a third turn was asked
//! for "a PDF with that diagram and that code" there was nothing to retrieve —
//! not because retrieval was broken, but because no artifact had ever existed.
//!
//! This is the capture. It reads a *finished* assistant message and names the
//! blocks in it worth keeping.
//!
//! ## Why finished, and why not streaming
//!
//! A fence still arriving is a fence with no closing marker and a half-written
//! body. Capturing during the stream would record every prefix of the same
//! block as its own artifact, and the list would fill with a dozen truncated
//! versions of one diagram. So this runs once, over the completed text, and the
//! content hash in [`super::conversation_store`] absorbs the rest: a retry that
//! produces byte-identical output reuses the version it already has rather than
//! adding another.
//!
//! ## What is deliberately not captured
//!
//! Short fragments and shell snippets. A three-line shell block showing how to
//! run something is not an artifact somebody will later ask to put in a PDF, and
//! an inventory full of them makes the real ones harder to find. The threshold
//! is a stated constant rather than a per-language tuning, because a rule
//! somebody can read is worth more here than a rule that is slightly better.

use super::conversation_store::ArtifactKind;

/// The shortest fenced block worth keeping as its own artifact.
///
/// Two lines of shell is an instruction; twenty lines of Python is a thing
/// somebody will refer back to. The cut is deliberately low — the cost of
/// keeping one line too many is an extra inventory row, and the cost of keeping
/// one too few is the failure this module exists to fix.
pub const MIN_CAPTURED_LINES: usize = 3;

/// Languages whose fenced blocks are diagrams rather than code.
const DIAGRAM_LANGUAGES: &[&str] = &["mermaid", "graphviz", "dot", "plantuml"];

/// Languages that are usually a command to run rather than a thing to keep.
///
/// Still captured when long, because a forty-line shell script is a script.
const TRANSIENT_LANGUAGES: &[&str] = &["bash", "sh", "shell", "zsh", "console", "text", ""];

/// One block worth keeping, lifted out of a message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedBlock {
    pub kind: ArtifactKind,
    /// The fence's language tag, lowercased. Empty when the fence had none.
    pub language: String,
    /// The block's body, exactly as written — no trimming of interior lines, no
    /// re-indentation, no normalisation. This is the artifact's content, and a
    /// reader asking for "the same code" means these bytes.
    pub content: String,
    /// A title for the inventory, derived from the content rather than invented.
    pub title: String,
    pub mime: String,
    /// Position in the message, so two blocks in one message get stable ids.
    pub index: usize,
}

impl CapturedBlock {
    pub fn lines(&self) -> usize {
        self.content.lines().count()
    }
}

/// Finds the fenced blocks in a finished assistant message.
///
/// Fences are matched by their own opening run length, so a block containing a
/// three-backtick line inside a four-backtick fence survives intact — which
/// matters, because a model explaining Markdown does exactly that.
pub fn capture(message: &str) -> Vec<CapturedBlock> {
    let mut captured = Vec::new();
    let mut lines = message.lines();
    let mut index = 0usize;

    while let Some(line) = lines.next() {
        let trimmed = line.trim_start();
        let Some(fence) = opening_fence(trimmed) else {
            continue;
        };
        // The opening fence's indentation, so an indented block keeps its own
        // relative indentation and loses only the wrapper's.
        let indent = line.len() - trimmed.len();
        let language = trimmed[fence.len()..]
            .trim()
            .to_lowercase()
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .to_string();

        let mut body: Vec<String> = Vec::new();
        let mut closed = false;
        for inner in lines.by_ref() {
            let inner_trimmed = inner.trim_start();
            if inner_trimmed.starts_with(&fence) && inner_trimmed[fence.len()..].trim().is_empty() {
                closed = true;
                break;
            }
            // Strip only the wrapper's indentation, and only where it is there.
            let strippable = inner.len() >= indent
                && inner.chars().take(indent).all(char::is_whitespace);
            body.push(if strippable {
                inner[indent..].to_string()
            } else {
                inner.to_string()
            });
        }

        // An unclosed fence is a message that was cut off. Skipped rather than
        // captured: half a function recorded as "the code" is worse than no
        // artifact, because a later turn would reuse it believing it complete.
        if !closed {
            continue;
        }

        let content = body.join("\n");
        if !worth_keeping(&language, &content) {
            continue;
        }

        let kind = if DIAGRAM_LANGUAGES.contains(&language.as_str()) {
            ArtifactKind::DiagramSource
        } else {
            ArtifactKind::Code
        };

        captured.push(CapturedBlock {
            title: title_for(&language, &content, index),
            mime: mime_for(&language, kind),
            kind,
            language,
            content,
            index,
        });
        index += 1;
    }

    captured
}

/// The fence that opens a block, when a line opens one.
fn opening_fence(trimmed: &str) -> Option<String> {
    let ticks = trimmed.chars().take_while(|c| *c == '`').count();
    (ticks >= 3).then(|| "`".repeat(ticks))
}

/// Whether a block is an artifact or an aside.
fn worth_keeping(language: &str, content: &str) -> bool {
    if content.trim().is_empty() {
        return false;
    }
    let lines = content.lines().filter(|l| !l.trim().is_empty()).count();
    if lines < MIN_CAPTURED_LINES {
        return false;
    }
    // A long shell script is a script; a short one is an instruction.
    if TRANSIENT_LANGUAGES.contains(&language) && lines < MIN_CAPTURED_LINES * 3 {
        return false;
    }
    true
}

/// A name for the inventory, taken from the content rather than invented.
///
/// A model that wrote `def build_report(` gets "build_report"; a mermaid graph
/// gets "Diagram 1". Nothing here guesses at what the code *does* — a title is a
/// label, and a label that overstates is worse than a dull one.
fn title_for(language: &str, content: &str, index: usize) -> String {
    for line in content.lines().take(40) {
        if let Some(name) = declared_name(line.trim()) {
            return name;
        }
    }
    if DIAGRAM_LANGUAGES.contains(&language) {
        return format!("Diagram {}", index + 1);
    }
    if language.is_empty() {
        format!("Block {}", index + 1)
    } else {
        format!("{language} block {}", index + 1)
    }
}

/// A definition's name, for the handful of forms that are unambiguous.
fn declared_name(line: &str) -> Option<String> {
    const PREFIXES: &[&str] = &[
        "pub fn ",
        "pub struct ",
        "export function ",
        "export class ",
        "def ",
        "class ",
        "fn ",
        "function ",
        "struct ",
        "interface ",
    ];
    for prefix in PREFIXES {
        if let Some(rest) = line.strip_prefix(prefix) {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if name.len() >= 2 {
                return Some(name);
            }
        }
    }
    None
}

/// The media type an artifact of this language is stored under.
fn mime_for(language: &str, kind: ArtifactKind) -> String {
    if kind == ArtifactKind::DiagramSource {
        return match language {
            "mermaid" => "text/vnd.mermaid".to_string(),
            "dot" | "graphviz" => "text/vnd.graphviz".to_string(),
            other => format!("text/x-{other}"),
        };
    }
    match language {
        "python" | "py" => "text/x-python".to_string(),
        "rust" | "rs" => "text/x-rust".to_string(),
        "typescript" | "ts" | "tsx" => "text/x-typescript".to_string(),
        "javascript" | "js" | "jsx" => "text/javascript".to_string(),
        "json" => "application/json".to_string(),
        "sql" => "application/sql".to_string(),
        "" => "text/plain".to_string(),
        other => format!("text/x-{other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(lines: usize) -> String {
        (1..=lines)
            .map(|n| format!("line_{n} = {n}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The scenario this module exists for.
    #[test]
    fn a_diagram_and_some_code_in_two_messages_both_become_artifacts() {
        let from_model_a =
            "Here is the flow:\n\n```mermaid\ngraph TD\n  A[Inlet] --> B[Pump]\n  B --> C[Header]\n```\n";
        let from_model_b = format!("And the code:\n\n```python\n{}\n```\n", body(6));

        let diagram = capture(from_model_a);
        let code = capture(&from_model_b);

        assert_eq!(diagram.len(), 1, "{diagram:?}");
        assert_eq!(diagram[0].kind, ArtifactKind::DiagramSource);
        assert!(diagram[0].content.contains("A[Inlet] --> B[Pump]"));

        assert_eq!(code.len(), 1, "{code:?}");
        assert_eq!(code[0].kind, ArtifactKind::Code);
        assert_eq!(code[0].language, "python");
    }

    /// "The same code" means these bytes. Indentation, blank lines and order are
    /// the content, not formatting this may tidy.
    #[test]
    fn the_body_is_preserved_exactly() {
        let source = "def f():\n    if True:\n\n        return {\n            'a': 1,\n        }";
        let message = format!("```python\n{source}\n```\n");
        let captured = capture(&message);

        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].content, source);
        assert!(captured[0].content.contains("            'a': 1,"));
        assert!(
            captured[0].content.contains("\n\n"),
            "the blank line inside the function survives"
        );
    }

    #[test]
    fn an_indented_fence_loses_only_the_wrapper_indentation() {
        let message =
            "1. Do this:\n\n   ```python\n   def f():\n       return 1\n   x = f()\n   ```\n";
        let captured = capture(message);
        assert_eq!(captured.len(), 1, "{captured:?}");
        assert_eq!(captured[0].content, "def f():\n    return 1\nx = f()");
    }

    /// A message cut off mid-block must not become an artifact a later turn
    /// reuses believing it complete.
    #[test]
    fn an_unclosed_fence_is_not_captured() {
        let message = format!("```python\n{}\n", body(10));
        assert!(capture(&message).is_empty());
    }

    #[test]
    fn a_longer_fence_can_contain_a_shorter_one() {
        let message = "````markdown\nHere is how you write it:\n\n```python\nprint(1)\n```\n\nThat is all.\n````\n";
        let captured = capture(message);

        assert_eq!(captured.len(), 1, "{captured:?}");
        assert!(
            captured[0].content.contains("```python"),
            "the inner fence is content: {:?}",
            captured[0].content
        );
    }

    #[test]
    fn short_asides_are_not_captured() {
        assert!(capture("```bash\nnpm install\n```").is_empty());
        assert!(capture("```\nok\n```").is_empty());
        assert!(capture("```python\nx = 1\n```").is_empty());
    }

    #[test]
    fn a_real_shell_script_is_captured() {
        let script = (1..=12)
            .map(|n| format!("echo step {n}"))
            .collect::<Vec<_>>()
            .join("\n");
        let captured = capture(&format!("```bash\n{script}\n```"));
        assert_eq!(captured.len(), 1, "{captured:?}");
    }

    #[test]
    fn two_blocks_in_one_message_get_distinct_positions() {
        let message = format!(
            "First:\n\n```python\n{}\n```\n\nThen:\n\n```mermaid\ngraph TD\n A-->B\n B-->C\n```\n",
            body(5)
        );
        let captured = capture(&message);

        assert_eq!(captured.len(), 2, "{captured:?}");
        assert_eq!(captured[0].index, 0);
        assert_eq!(captured[1].index, 1);
        assert_eq!(captured[1].kind, ArtifactKind::DiagramSource);
    }

    /// A title is a label taken from the content, never a guess about meaning.
    #[test]
    fn the_title_comes_from_the_content() {
        let captured = capture(&format!(
            "```python\ndef build_report(rows):\n{}\n```",
            body(4)
        ));
        assert_eq!(captured[0].title, "build_report");

        let diagram = capture("```mermaid\ngraph TD\n A-->B\n B-->C\n```");
        assert_eq!(diagram[0].title, "Diagram 1");
    }

    #[test]
    fn the_language_decides_the_media_type() {
        assert_eq!(
            capture(&format!("```python\n{}\n```", body(5)))[0].mime,
            "text/x-python"
        );
        assert_eq!(
            capture("```mermaid\ngraph TD\n A-->B\n B-->C\n```")[0].mime,
            "text/vnd.mermaid"
        );
    }

    /// Prose around the fences is not part of any artifact.
    #[test]
    fn only_the_fenced_bodies_are_captured() {
        let message = format!(
            "I have written the pipeline below. Note that it assumes UTF-8.\n\n```python\n{}\n```\n\nLet me know if you want it changed.\n",
            body(5)
        );
        let captured = capture(&message);
        assert_eq!(captured.len(), 1);
        assert!(!captured[0].content.contains("Let me know"));
        assert!(!captured[0].content.contains("assumes UTF-8"));
    }

    /// Unicode in source is content, and survives capture untouched.
    #[test]
    fn unicode_in_source_is_preserved() {
        let source =
            "# note: 温度 ≥ 100 °C\nvalue = \"café — naïve\"\nprint(value)\nassert value";
        let captured = capture(&format!("```python\n{source}\n```"));
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].content, source);
    }

    #[test]
    fn a_message_with_no_fences_produces_nothing() {
        assert!(capture("Just a sentence about a diagram and some code.").is_empty());
        assert!(capture("").is_empty());
    }
}

/// The journey this change set exists for, over the real store.
///
/// Two models, two messages, two artifacts, and a third turn that finds both by
/// id without reading either earlier run's workspace. Kept here rather than in
/// the store's own tests because it is the *seam* that was broken: the capture
/// and the store were each fine in isolation, and nothing joined them.
#[cfg(test)]
mod cross_model_journey {
    use super::*;
    use crate::artifacts::conversation_store::{
        ArtifactKind, ConversationArtifacts, NewArtifact, Producer,
    };

    const ALICE: &str = "user-alice";

    /// What `commands::agent::register_message_artifacts` does, in miniature.
    fn register(
        store: &ConversationArtifacts,
        conversation: &str,
        message_id: &str,
        run_id: &str,
        model_id: &str,
        message: &str,
    ) {
        for block in capture(message) {
            store
                .record(NewArtifact {
                    artifact_id: Some(format!("art-{message_id}-{}", block.index)),
                    conversation_id: conversation.to_string(),
                    owner_user_id: ALICE.to_string(),
                    message_id: Some(message_id.to_string()),
                    run_id: Some(run_id.to_string()),
                    producer: Producer {
                        model_id: Some(model_id.to_string()),
                        tool: None,
                        agent: None,
                    },
                    kind: block.kind,
                    mime: block.mime.clone(),
                    title: block.title.clone(),
                    filename: None,
                    complete: true,
                    derived_from: None,
                    renders: None,
                    language: (!block.language.is_empty()).then(|| block.language.clone()),
                    render_requires: Vec::new(),
                    content: block.content.into_bytes(),
                })
                .expect("recorded");
        }
    }

    const DIAGRAM: &str =
        "Here is the flow:\n\n```mermaid\ngraph TD\n  A[Inlet] --> B[Pump]\n  B --> C[Header]\n```\n";

    const REVISED: &str = "Revised:\n\n```mermaid\ngraph TD\n  A[Inlet] --> B[Pump]\n  B --> D[Filter]\n  D --> C[Header]\n```\n";

    fn code_message() -> String {
        let body = (1..=8)
            .map(|n| format!("def step_{n}():\n    return {n}"))
            .collect::<Vec<_>>()
            .join("\n");
        format!("And the code:\n\n```python\n{body}\n```\n")
    }

    #[test]
    fn a_diagram_from_one_model_and_code_from_another_are_both_reachable_later() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let store = ConversationArtifacts::open(dir.path()).expect("the store opens");

        // Model A, run 1.
        register(&store, "conv-1", "msg-a", "run-1", "qwen2.5-vl-7b", DIAGRAM);
        // Model B, run 2 -- a different model, a different run, and a workspace
        // it could never have read from.
        register(
            &store,
            "conv-1",
            "msg-b",
            "run-2",
            "qwen2.5-coder-7b",
            &code_message(),
        );

        // Run 3 asks what the conversation holds.
        let listed = store.list(ALICE, "conv-1").expect("listed");
        assert_eq!(listed.len(), 2, "{listed:?}");

        let diagram = listed
            .iter()
            .find(|a| a.kind == ArtifactKind::DiagramSource)
            .expect("the diagram is there");
        let code = listed
            .iter()
            .find(|a| a.kind == ArtifactKind::Code)
            .expect("the code is there");

        assert_eq!(diagram.producer.model_id.as_deref(), Some("qwen2.5-vl-7b"));
        assert_eq!(code.producer.model_id.as_deref(), Some("qwen2.5-coder-7b"));
        assert_ne!(diagram.run_id, code.run_id, "produced by different runs");

        // And run 3 can read both, exactly as written.
        let (_, diagram_bytes) = store.read(ALICE, &diagram.reference()).unwrap().unwrap();
        let (_, code_bytes) = store.read(ALICE, &code.reference()).unwrap().unwrap();
        assert!(String::from_utf8_lossy(&diagram_bytes).contains("A[Inlet] --> B[Pump]"));
        assert!(String::from_utf8_lossy(&code_bytes).contains("def step_8():"));
    }

    /// Restart. The store is reopened on the same directory and both are still
    /// there, at the same hashes.
    #[test]
    fn both_artifacts_survive_a_restart_with_their_hashes_intact() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let (diagram_hash, code_hash) = {
            let store = ConversationArtifacts::open(dir.path()).expect("opens");
            register(&store, "conv-1", "msg-a", "run-1", "model-a", DIAGRAM);
            register(&store, "conv-1", "msg-b", "run-2", "model-b", &code_message());
            let listed = store.list(ALICE, "conv-1").unwrap();
            (
                listed
                    .iter()
                    .find(|a| a.kind == ArtifactKind::DiagramSource)
                    .unwrap()
                    .sha256
                    .clone(),
                listed
                    .iter()
                    .find(|a| a.kind == ArtifactKind::Code)
                    .unwrap()
                    .sha256
                    .clone(),
            )
        };

        let reopened = ConversationArtifacts::open(dir.path()).expect("reopens");
        let listed = reopened.list(ALICE, "conv-1").expect("listed");
        assert_eq!(listed.len(), 2);
        assert!(listed.iter().any(|a| a.sha256 == diagram_hash));
        assert!(listed.iter().any(|a| a.sha256 == code_hash));
    }

    /// A replayed or resumed run re-registering the same message must not
    /// double the list.
    #[test]
    fn replaying_the_same_message_does_not_duplicate_its_artifacts() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let store = ConversationArtifacts::open(dir.path()).expect("opens");

        register(&store, "conv-1", "msg-a", "run-1", "model-a", DIAGRAM);
        register(&store, "conv-1", "msg-a", "run-1-retry", "model-a", DIAGRAM);

        let listed = store.list(ALICE, "conv-1").expect("listed");
        assert_eq!(listed.len(), 1, "{listed:?}");
        assert_eq!(listed[0].version, 1, "and no second version was minted");
    }

    /// A later message that revises the diagram becomes its own artifact, and
    /// the original stays readable -- which is what "the revised diagram with
    /// the original code" needs.
    #[test]
    fn a_revised_diagram_does_not_disturb_the_original_code() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let store = ConversationArtifacts::open(dir.path()).expect("opens");

        register(&store, "conv-1", "msg-a", "run-1", "model-a", DIAGRAM);
        register(&store, "conv-1", "msg-b", "run-2", "model-b", &code_message());
        register(&store, "conv-1", "msg-c", "run-3", "model-a", REVISED);

        let listed = store.list(ALICE, "conv-1").expect("listed");
        let diagrams: Vec<_> = listed
            .iter()
            .filter(|a| a.kind == ArtifactKind::DiagramSource)
            .collect();
        assert_eq!(diagrams.len(), 2, "two distinct diagrams: {diagrams:?}");

        let original = diagrams
            .iter()
            .find(|a| a.message_id.as_deref() == Some("msg-a"))
            .expect("the original diagram");
        let (_, bytes) = store.read(ALICE, &original.reference()).unwrap().unwrap();
        let text = String::from_utf8_lossy(&bytes).to_string();
        assert!(text.contains("B --> C[Header]"), "{text}");
        assert!(!text.contains("Filter"), "the original is untouched: {text}");

        let code = listed
            .iter()
            .find(|a| a.kind == ArtifactKind::Code)
            .expect("the code is there");
        let (_, code_bytes) = store.read(ALICE, &code.reference()).unwrap().unwrap();
        assert!(String::from_utf8_lossy(&code_bytes).contains("def step_1():"));
    }

    /// A conversation's artifacts are that conversation's.
    #[test]
    fn another_conversations_artifacts_are_not_offered() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let store = ConversationArtifacts::open(dir.path()).expect("opens");

        register(&store, "conv-1", "msg-a", "run-1", "model-a", DIAGRAM);
        register(&store, "conv-2", "msg-z", "run-9", "model-a", &code_message());

        let listed = store.list(ALICE, "conv-1").expect("listed");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].kind, ArtifactKind::DiagramSource);
    }
}

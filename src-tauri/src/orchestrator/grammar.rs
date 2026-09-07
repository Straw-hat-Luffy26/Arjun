//! Making a malformed tool call impossible rather than merely unlikely.
//!
//! A quantised model asked to emit JSON gets it wrong a meaningful fraction of
//! the time — a trailing comma, a missing brace, an invented field, a tool name
//! that does not exist. On a single reply that is an annoyance. Across a
//! multi-step agent loop it compounds: at even a few percent per call, a plan of
//! a dozen steps is closer to a coin toss than to a process.
//!
//! Retrying does not fix it either. A model that produced malformed output once
//! is likely to produce it again, and the retries burn the step budget that
//! exists to stop the task running away.
//!
//! So the output is constrained at the sampler. `llama.cpp` accepts a GBNF
//! grammar and will only sample tokens that keep the output on a valid path —
//! which makes "the model emitted a broken tool call" not a failure to handle
//! but a state that cannot occur.
//!
//! ## Two turns, not one
//!
//! Constraining is not free: a grammar applied to *reasoning* measurably hurts
//! the quality of that reasoning, and this product's whole point is tasks where
//! the thinking matters. So a step is two turns —
//!
//! 1. an unconstrained turn, where the model works out what to do; then
//! 2. a constrained turn, where it emits only the call.
//!
//! The cost is one extra pass; the benefit is thinking that is not deformed by
//! a grammar and a call that cannot be malformed. [`ToolGrammar::preamble`] is
//! what introduces the second turn.
//!
//! ## The grammar is narrower than the catalogue
//!
//! It is built from the tools *this task* was given, not from every tool that
//! exists. A model cannot ask for something outside its plan, because there is
//! no token path that spells it.

use super::tools::{spec_for, ArgumentKind, ToolName};

/// A GBNF grammar admitting exactly the valid calls for a set of tools.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolGrammar {
    pub gbnf: String,
    /// The tools this grammar admits, for the audit record.
    pub tools: Vec<ToolName>,
}

impl ToolGrammar {
    /// What to say to the model on the constrained turn.
    ///
    /// Short on purpose. The grammar does the enforcing; this only has to tell
    /// the model what is now expected of it, and a long instruction competes for
    /// attention with the task itself.
    pub fn preamble(&self) -> String {
        let names: Vec<&str> = self.tools.iter().map(|t| t.as_str()).collect();
        format!(
            "Now emit exactly one tool call as JSON, and nothing else. \
             Available tools: {}.",
            names.join(", ")
        )
    }
}

/// The rule name for one tool's call shape.
///
/// GBNF rule names admit letters, digits and hyphens — not the dot a namespaced
/// tool name carries, nor the underscore. Both are folded to a hyphen, which
/// keeps names distinct because no two tools differ only in that punctuation.
/// The tool's real name still appears verbatim inside the rule body, where it is
/// a quoted JSON string and any character is fine; only the identifier is
/// rewritten. Getting this wrong does not fail a test — it emits a grammar
/// llama.cpp rejects at parse time, which surfaces as a constrained turn that
/// produces nothing at all.
fn rule_name(tool: ToolName) -> String {
    format!("call-{}", tool.as_str().replace(['_', '.'], "-"))
}

/// The GBNF fragment matching one argument's value.
fn value_rule(kind: ArgumentKind) -> &'static str {
    match kind {
        // A path is a string as far as the grammar is concerned. Constraining
        // its *shape* here would be a second, weaker copy of the scope check the
        // gateway already does properly — and a grammar that half-enforced a
        // security rule would invite someone to trust it for the whole job.
        ArgumentKind::Text | ArgumentKind::Path => "string",
        ArgumentKind::Integer => "integer",
        ArgumentKind::Object => "object",
    }
}

/// Builds a grammar admitting exactly the calls these tools accept.
///
/// Returns `None` for an empty tool set. A grammar admitting nothing would make
/// the model unable to emit anything at all, and the sampler would run to its
/// token limit producing nothing — far worse than the caller simply not
/// constraining a turn that has no tools to offer.
pub fn build(tools: &[ToolName]) -> Option<ToolGrammar> {
    if tools.is_empty() {
        return None;
    }

    let mut rules = String::new();

    // The root is an alternation over complete call shapes rather than a generic
    // "tool name plus arguments". Splitting them would let the model pair one
    // tool's name with another tool's arguments, which is valid JSON, passes a
    // shape check, and means nothing.
    let alternatives: Vec<String> = tools.iter().map(|t| rule_name(*t)).collect();
    rules.push_str(&format!("root ::= {}\n\n", alternatives.join(" | ")));

    for tool in tools {
        let spec = spec_for(*tool);
        let mut rule = format!(
            "{} ::= \"{{\" ws \"\\\"tool\\\"\" ws \":\" ws \"\\\"{}\\\"\" ws \",\" ws \
             \"\\\"arguments\\\"\" ws \":\" ws \"{{\" ws",
            rule_name(*tool),
            tool.as_str()
        );

        if spec.arguments.is_empty() {
            rule.push_str(" \"}\" ws \"}\"\n");
        } else {
            for (i, argument) in spec.arguments.iter().enumerate() {
                if i > 0 {
                    rule.push_str(" \",\" ws");
                }
                rule.push_str(&format!(
                    " \"\\\"{}\\\"\" ws \":\" ws {} ws",
                    argument.name,
                    value_rule(argument.kind)
                ));
            }
            rule.push_str(" \"}\" ws \"}\"\n");
        }

        rules.push_str(&rule);
    }

    // Shared terminals. `string` permits escapes so a Windows path or a quoted
    // phrase inside a query is expressible — a grammar that could not express a
    // backslash would make half the paths on this platform unsayable.
    rules.push_str(
        "\n\
         string ::= \"\\\"\" char* \"\\\"\"\n\
         char ::= [^\"\\\\] | \"\\\\\" ([\"\\\\/bfnrt] | \"u\" hex hex hex hex)\n\
         hex ::= [0-9a-fA-F]\n\
         integer ::= \"-\"? ([0-9] | [1-9] [0-9]*)\n\
         object ::= \"{\" ws (member (ws \",\" ws member)*)? ws \"}\"\n\
         member ::= string ws \":\" ws value\n\
         value ::= string | integer | object | array | \"true\" | \"false\" | \"null\"\n\
         array ::= \"[\" ws (value (ws \",\" ws value)*)? ws \"]\"\n\
         ws ::= [ \\t\\n]*\n",
    );

    Some(ToolGrammar {
        gbnf: rules,
        tools: tools.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_grammar_admits_only_the_tools_it_was_given() {
        let grammar = build(&[ToolName::SearchDocuments, ToolName::RunCalculation]).unwrap();

        assert!(grammar.gbnf.contains("knowledge.search_authorized"));
        assert!(grammar.gbnf.contains("calculation.evaluate_with_units"));

        // Nothing else exists as a spellable path.
        for absent in [
            "sandbox.run_code",
            "workspace.write_text",
            "artifact.create_approval_note",
        ] {
            assert!(
                !grammar.gbnf.contains(absent),
                "{absent} should not be reachable in this grammar"
            );
        }
    }

    /// A model cannot ask for a tool outside its plan, because no token path
    /// spells it.
    #[test]
    fn a_tool_outside_the_plan_is_unspellable() {
        let grammar = build(&[ToolName::SearchDocuments]).unwrap();
        assert!(!grammar.gbnf.contains("sandbox.run_code"));
    }

    #[test]
    fn every_declared_argument_appears_in_its_rule() {
        for tool in ToolName::ALL {
            let grammar = build(&[*tool]).unwrap();
            for argument in spec_for(*tool).arguments {
                assert!(
                    grammar.gbnf.contains(&format!("\\\"{}\\\"", argument.name)),
                    "{} is missing {:?} in its grammar",
                    tool.as_str(),
                    argument.name
                );
            }
        }
    }

    /// Splitting the name and the arguments would let a model pair one tool's
    /// name with another's arguments: valid JSON, passes a shape check, means
    /// nothing.
    #[test]
    fn each_tool_has_its_own_complete_call_shape() {
        let grammar = build(&[ToolName::SearchDocuments, ToolName::ExecuteCode]).unwrap();

        assert!(grammar.gbnf.contains("call-knowledge-search-authorized ::="));
        assert!(grammar.gbnf.contains("call-sandbox-run-code ::="));
        assert!(grammar
            .gbnf
            .starts_with("root ::= call-knowledge-search-authorized | call-sandbox-run-code"));
    }

    /// A rule name llama.cpp cannot parse produces a constrained turn that
    /// emits nothing at all — a failure that looks like a hung model rather
    /// than a malformed grammar, which is why it is asserted rather than left
    /// to be noticed.
    #[test]
    fn no_rule_name_carries_a_character_gbnf_cannot_spell() {
        for tool in ToolName::ALL {
            let rule = rule_name(*tool);
            assert!(
                rule.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'),
                "{} produces the unusable rule name {rule}",
                tool.as_str()
            );
        }
    }

    /// Folding `.` and `_` to `-` must not make two tools share a rule name:
    /// the second would silently replace the first in the alternation.
    #[test]
    fn every_tool_keeps_its_own_rule_name() {
        let names: std::collections::BTreeSet<String> =
            ToolName::ALL.iter().map(|t| rule_name(*t)).collect();
        assert_eq!(names.len(), ToolName::ALL.len());
    }

    #[test]
    fn argument_kinds_map_to_the_right_terminal() {
        // `execute_code` takes two strings; `create_docx` takes an object.
        let code = build(&[ToolName::ExecuteCode]).unwrap();
        assert!(code.gbnf.contains("\\\"source\\\"\" ws \":\" ws string"));

        let docx = build(&[ToolName::CreateDocx]).unwrap();
        assert!(docx.gbnf.contains("\\\"content\\\"\" ws \":\" ws object"));
    }

    /// A grammar that cannot express a backslash makes half the paths on Windows
    /// unsayable, so the string rule has to admit escape sequences.
    ///
    /// Asserted on fragments that carry no backslashes of their own — writing the
    /// escape sequence out in the test would need its own layer of escaping, and
    /// getting *that* wrong is how a test ends up asserting nothing.
    #[test]
    fn strings_can_express_escapes_and_therefore_windows_paths() {
        let grammar = build(&[ToolName::ReadScopedFile]).unwrap();

        assert!(grammar.gbnf.contains("char ::="));
        // The JSON escape set, which is what makes a backslash expressible.
        assert!(grammar.gbnf.contains("bfnrt"));
        // And the unicode escape form.
        assert!(grammar.gbnf.contains(r#""u" hex hex hex hex"#));
    }

    /// An empty grammar would leave the sampler with no legal token at all.
    #[test]
    fn no_tools_produces_no_grammar_rather_than_an_empty_one() {
        assert_eq!(build(&[]), None);
    }

    #[test]
    fn every_grammar_defines_the_terminals_its_rules_reference() {
        for tool in ToolName::ALL {
            let gbnf = build(&[*tool]).unwrap().gbnf;
            for terminal in ["string ::=", "ws ::=", "object ::=", "integer ::="] {
                assert!(gbnf.contains(terminal), "{} is missing {terminal}", tool.as_str());
            }
        }
    }

    // ── The second turn ──────────────────────────────────────────────────

    #[test]
    fn the_preamble_names_exactly_the_available_tools() {
        let grammar = build(&[ToolName::SearchDocuments, ToolName::RunCalculation]).unwrap();
        let preamble = grammar.preamble();

        assert!(preamble.contains("knowledge.search_authorized"));
        assert!(preamble.contains("calculation.evaluate_with_units"));
        assert!(!preamble.contains("sandbox.run_code"));
    }

    /// It competes for attention with the task, so it stays short.
    ///
    /// The ceiling is against the whole catalogue, which is the worst case and
    /// not the usual one: a grammar is built from the plan's permitted tools,
    /// and a plan permits a handful. Namespaced names cost roughly ten
    /// characters each over the old flat ones — paid back in the model reaching
    /// for the right tool, which is worth more than the tokens.
    /// The ceiling was 600 when the catalogue held twenty tools, and is 800
    /// now that the seven notebook tools bring it to twenty-seven. A looser
    /// number but a tighter budget: 722 characters over 27 tools is 26.7
    /// each, against 30 before. The measured figure is in the failure
    /// message, so the next person to add a tool can see whether they spent
    /// more than their share rather than guessing at a new ceiling.
    #[test]
    fn the_preamble_is_brief() {
        let grammar = build(ToolName::ALL).unwrap();
        assert!(
            grammar.preamble().len() < 800,
            "the preamble should not crowd out the task itself (it is {})",
            grammar.preamble().len()
        );
    }

    #[test]
    fn the_grammar_records_which_tools_it_admits() {
        let grammar = build(&[ToolName::SearchDocuments]).unwrap();
        assert_eq!(grammar.tools, vec![ToolName::SearchDocuments]);
    }
}

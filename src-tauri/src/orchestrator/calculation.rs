//! Arithmetic that is actually correct, with the units checked.
//!
//! ARJUN design rule 27: *"the orchestrator should use a deterministic calculator ... The
//! model may explain the result, but the calculation engine should be the source
//! of numerical truth."*
//!
//! That instruction is not caution for its own sake. A language model asked for
//! `(8.2 - 9.0) / 9.0 * 100` will usually produce something close to `-8.9`, and
//! *close* is the problem: a wall-thickness deviation that is wrong in the second
//! decimal place still reads as an answer, gets copied into an approval note, and
//! nothing downstream can tell. So the number comes from here, and the model's
//! job is to say what it means.
//!
//! Since P08 the engine itself is `crate::calculation`; this module keeps the
//! record shape its readers were built on.
//!
//! ## Units are checked, not carried along
//!
//! `8.2 mm - 9.0 mm` is a length. `8.2 mm - 9.0 kg` is a mistake, and it is
//! caught rather than silently producing `-0.8` of nothing. This matters more in
//! a refinery than almost anywhere: the arithmetic in an inspection report is
//! simple, and the errors that survive review are the dimensional ones.
//!
//! ## Everything is on the record
//!
//! A [`CalculationRecord`] carries the expression, the parsed inputs, every
//! intermediate step, the result with its unit, and the rounding applied. PS
//! step 27 asks for exactly that, and it is what lets a reviewer check a number
//! without redoing it — and what the artifact validator later reconciles the
//! document against.

use serde::{Deserialize, Serialize};

pub use crate::calculation::eval::Step;

/// Everything ARJUN design rule 27 asks a calculation to record, in the shape
/// the workbook, the verifier and the run record have always read.
///
/// Since P08 this is a projection of a [`crate::calculation::CalcRecord`]:
/// `id` names the full record (typed inputs, sources, substitutions, engine
/// version, tolerance), which is what a citation `[C:calc-…]` resolves to.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CalculationRecord {
    /// Exactly what was asked, unmodified.
    pub expression: String,
    /// The quantities read out of it, with their units, as written.
    pub inputs: Vec<String>,
    pub steps: Vec<Step>,
    /// The result as a number, before any rounding for display.
    pub value: f64,
    pub unit: String,
    /// The result as it should be written down.
    pub formatted: String,
    /// How `formatted` was produced from `value`.
    pub rounding: String,
    /// True when the engine computed this rather than a model.
    ///
    /// Always true here. It exists so a record's provenance is on the record
    /// itself, rather than being something a reader has to know.
    pub deterministic: bool,
    /// The full record's content address (`calc-…`).
    #[serde(default)]
    pub id: String,
    /// Engine, version and unit-table digest.
    #[serde(default)]
    pub engine: String,
    /// `computed`, or `provisional` when an input has no source.
    #[serde(default)]
    pub status: String,
}

impl CalculationRecord {
    /// The projection of a full record that has a result.
    pub fn from_record(record: &crate::calculation::CalcRecord) -> Option<Self> {
        let first = record.first()?;
        let inputs = if record.inputs.is_empty() {
            crate::calculation::expr::parse(
                crate::calculation::expr::split_equation(&record.equation).map_or(record.equation.as_str(), |(_, rhs)| rhs),
            )
            .map(|node| node.literals())
            .unwrap_or_default()
        } else {
            // Each with its source, as the record holds it.
            record.inputs.iter().map(|input| input.line()).collect()
        };
        Some(Self {
            expression: record.equation.clone(),
            inputs,
            steps: record.steps.clone(),
            value: first.value,
            unit: first.unit.clone(),
            formatted: first.display.clone(),
            rounding: record.rounding_rule.clone(),
            deterministic: true,
            id: record.id.clone(),
            engine: record.engine.describe(),
            status: record.status.as_str().to_string(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CalculationError {
    pub message: String,
    /// Where in the expression, when known.
    pub position: Option<usize>,
}

/// Evaluates an expression and records how it got there.
///
/// Never calls a model, and never will. The whole value of this function is
/// that its answer does not depend on one. A bare expression has no named
/// inputs: its numbers with units are recorded as values without a source,
/// and the full record says so (`crate::calculation::evaluate` takes typed,
/// sourced inputs).
pub fn evaluate(expression: &str) -> Result<CalculationRecord, CalculationError> {
    evaluate_record(expression).1
}

/// [`evaluate`], and the full record it projects.
pub fn evaluate_record(expression: &str) -> (crate::calculation::CalcRecord, Result<CalculationRecord, CalculationError>) {
    let record = crate::calculation::evaluate(expression, Vec::new(), &crate::calculation::Options::default(), None);
    let projected = match (&record.error, CalculationRecord::from_record(&record)) {
        (_, Some(projection)) => Ok(projection),
        (Some(error), None) => Err(CalculationError { message: error.message.clone(), position: error.at }),
        (None, None) => Err(CalculationError { message: "The calculation produced no result.".into(), position: None }),
    };
    (record, projected)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value_of(expression: &str) -> f64 {
        evaluate(expression).expect("should evaluate").value
    }

    #[test]
    fn plain_arithmetic_is_correct() {
        assert_eq!(value_of("2 + 3"), 5.0);
        assert_eq!(value_of("10 - 4"), 6.0);
        assert_eq!(value_of("6 * 7"), 42.0);
        assert_eq!(value_of("9 / 3"), 3.0);
    }

    #[test]
    fn multiplication_binds_tighter_than_addition() {
        assert_eq!(value_of("2 + 3 * 4"), 14.0);
        assert_eq!(value_of("(2 + 3) * 4"), 20.0);
    }

    #[test]
    fn negation_works_at_the_front_and_inside_brackets() {
        assert_eq!(value_of("-5 + 2"), -3.0);
        assert_eq!(value_of("3 * (-2)"), -6.0);
    }

    /// The calculation from the problem statement's own worked example.
    #[test]
    fn the_wall_thickness_deviation_is_exact() {
        let record = evaluate("(8.2 mm - 9.0 mm) / 9.0 mm * 100").unwrap();

        assert!((record.value - -8.888_888_888_9).abs() < 1e-9);
        assert_eq!(record.formatted, "-8.889");
        // The units cancel, so the answer is a plain ratio rather than "mm".
        assert!(record.unit.is_empty());
    }

    #[test]
    fn a_length_minus_a_length_is_a_length() {
        let record = evaluate("9.0 mm - 8.2 mm").unwrap();
        assert_eq!(record.unit, "mm");
        assert_eq!(record.formatted, "0.8 mm");
    }

    /// The error class that survives human review.
    #[test]
    fn adding_incompatible_units_is_refused_rather_than_computed() {
        let error = evaluate("8.2 mm + 9.0 kg").unwrap_err();
        assert!(error.message.contains("units do not match"), "{}", error.message);
        assert!(error.message.contains("mm"));
        assert!(error.message.contains("kg"));
    }

    #[test]
    fn subtracting_a_bare_number_from_a_measurement_is_refused() {
        let error = evaluate("8.2 mm - 9.0").unwrap_err();
        assert!(error.message.contains("units do not match"));
        assert!(error.message.contains("a plain number"));
    }

    #[test]
    fn multiplying_units_produces_a_compound_unit() {
        assert_eq!(evaluate("3 mm * 4 mm").unwrap().unit, "mm²");
        assert_eq!(evaluate("10 kg / 2 mm").unwrap().unit, "kg/mm");
    }

    #[test]
    fn dividing_like_units_cancels_them() {
        let record = evaluate("8.2 mm / 9.0 mm").unwrap();
        assert!(record.unit.is_empty());
        assert!(record.value < 1.0);
    }

    // ── The record ───────────────────────────────────────────────────────

    #[test]
    fn every_input_is_recorded_with_its_unit() {
        let record = evaluate("(8.2 mm - 9.0 mm) / 9.0 mm * 100").unwrap();
        // Exactly as written: `9.0`, not `9`.
        assert_eq!(record.inputs, vec!["8.2 mm", "9.0 mm", "9.0 mm", "100"]);
    }

    /// A reviewer must be able to follow the working without redoing it.
    #[test]
    fn intermediate_steps_are_recorded_in_order() {
        let record = evaluate("(8.2 mm - 9.0 mm) / 9.0 mm * 100").unwrap();

        assert_eq!(record.steps[0].description, "8.2 mm - 9.0 mm");
        assert_eq!(record.steps[0].result, "-0.8 mm");
        assert_eq!(record.steps[1].description, "-0.8 mm / 9.0 mm");
        assert_eq!(record.steps.last().unwrap().result, "-8.8888888889");
    }

    #[test]
    fn the_record_states_its_rounding_and_keeps_the_exact_value() {
        let record = evaluate("1 / 3").unwrap();
        assert_eq!(record.formatted, "0.3333");
        assert!((record.value - 0.333_333_333_333).abs() < 1e-9, "the exact value is kept");
        assert!(record.rounding.contains("significant figures"));
    }

    /// A record's provenance is on the record, not something a reader must know.
    #[test]
    fn every_record_says_it_was_computed_deterministically() {
        assert!(evaluate("1 + 1").unwrap().deterministic);
    }

    // ── Refusing rather than guessing ────────────────────────────────────

    #[test]
    fn division_by_zero_is_refused() {
        assert!(evaluate("5 / 0").unwrap_err().message.contains("Division by zero"));
    }

    #[test]
    fn an_empty_expression_is_refused() {
        assert!(evaluate("   ").unwrap_err().message.contains("nothing to calculate"));
    }

    #[test]
    fn an_unclosed_bracket_is_refused() {
        assert!(evaluate("(1 + 2").unwrap_err().message.contains("never closed"));
    }

    /// A partial reading is worse than a refusal, because it produces a number.
    #[test]
    fn trailing_nonsense_is_refused_rather_than_partly_evaluated() {
        let error = evaluate("2 + 2 more please").unwrap_err();
        assert!(error.message.contains("does not read as a unit"), "{}", error.message);
    }

    /// Prose containing a number is not a calculation, and must not become one.
    #[test]
    fn a_sentence_is_not_mistaken_for_an_expression() {
        for prose in [
            "the wall is 8 mm thick",
            "2 and then some",
            "measured 8.2 mm at four points",
        ] {
            assert!(evaluate(prose).is_err(), "{prose:?} should not evaluate");
        }
    }

    /// But a genuine trailing token still reports as unreadable.
    #[test]
    fn a_stray_symbol_after_a_complete_expression_is_refused() {
        let error = evaluate("2 + 2 )").unwrap_err();
        assert!(error.message.contains("Could not make sense of"), "{}", error.message);
    }

    #[test]
    fn an_error_says_where_it_happened() {
        let error = evaluate("8.2 mm + 9.0 kg").unwrap_err();
        assert!(error.position.is_some());
    }

    #[test]
    fn a_percent_sign_reads_as_a_unit() {
        let record = evaluate("5% + 3%").unwrap();
        assert_eq!(record.unit, "%");
        assert_eq!(record.formatted, "8 %");
    }

    /// Floating-point noise must never reach a document.
    #[test]
    fn the_written_result_carries_no_floating_point_noise() {
        assert_eq!(evaluate("0.1 + 0.2").unwrap().formatted, "0.3");
        assert_eq!(evaluate("3 * 1.1").unwrap().formatted, "3.3");
    }
}

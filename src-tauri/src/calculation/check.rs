//! Independent checking: re-read every input from where it came from, then
//! recompute by two paths, and never ask a model whether it agrees.
//!
//! A checker that re-evaluated the stored numbers would only prove the engine
//! is deterministic. This one goes back to each input's source — the memory
//! item at its current revision, the passage, the other calculation — and
//! looks for the value there. A source that has moved on makes the
//! calculation **stale**, and the checker recomputes with what the source now
//! says so the reader sees the difference. A value the source does not
//! contain is reported as such. Only when every input is found where the
//! record says, and both recomputations agree within the operation's stated
//! tolerance, is the record **verified**.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::decimal::Decimal;
use super::eval::{Evaluator, QKind, Quantity};
use super::expr;
use super::inputs::{CalcInput, InputSource, InputStatus};
use super::ops::{self, Options};
use super::record::{CalcRecord, Operation, RecordStatus};
use super::units::{lookup, parse_unit, UnitMap};

/// What a source says now.
#[derive(Debug, Clone, PartialEq)]
pub enum SourceState {
    /// Unchanged since it was cited; `text` is what it says.
    Current { revision: Option<u64>, text: String },
    /// Superseded, corrected or moved to another revision. `text` is what now
    /// stands in its place, where there is something, and `now` the source a
    /// recomputation should cite instead.
    Changed { revision: Option<u64>, text: Option<String>, why: String, now: Option<InputSource> },
    /// Withdrawn, revoked, or not readable by this reader. Worded the same.
    Gone { why: String },
    /// This checker has no way to re-read that kind of source.
    Unreadable { why: String },
}

/// The sources a checker can go back to. Implemented by the runtime over
/// the memory graph, the knowledge index and the calculation store, under
/// the checker's own clearance.
pub trait SourceReader {
    fn memory(&self, item_id: &str, pinned: Option<u64>) -> SourceState;
    fn evidence(&self, handle: &str) -> SourceState;
    fn document(&self, sha256: &str, locator: Option<&str>) -> SourceState;
    /// A calculation, and why it is stale when it is.
    fn calculation(&self, calc_id: &str) -> Option<(CalcRecord, Option<String>)>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CheckVerdict {
    /// Every input found at its source, and both recomputations agree.
    Verified,
    /// An input's source has changed since the calculation was made.
    Stale,
    /// A source does not contain the value the record used.
    SourceDisagrees,
    /// The calculation rests on an unsourced, assumed or missing input.
    Unresolved,
    /// A source could not be re-read here, so nothing can be confirmed.
    Unverifiable,
    /// The recomputation disagrees with the record.
    Mismatch,
    /// The original refused, and the recomputation refuses the same way.
    RefusalReproduced,
}

impl CheckVerdict {
    pub fn as_str(self) -> &'static str {
        match self {
            CheckVerdict::Verified => "verified",
            CheckVerdict::Stale => "stale",
            CheckVerdict::SourceDisagrees => "source disagrees",
            CheckVerdict::Unresolved => "unresolved",
            CheckVerdict::Unverifiable => "unverifiable",
            CheckVerdict::Mismatch => "mismatch",
            CheckVerdict::RefusalReproduced => "refusal reproduced",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InputCheck {
    pub input: String,
    pub as_recorded: String,
    pub source: String,
    pub finding: String,
    /// What the source says now, when it differs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_value: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CheckReport {
    pub calc_id: String,
    pub verdict: CheckVerdict,
    pub inputs: Vec<InputCheck>,
    /// The recomputation from the inputs as recorded.
    pub recomputed: Option<String>,
    /// Whether the recomputation produced the same record id and results.
    pub agreement: String,
    /// The second path: every input in SI base units first.
    pub si_path: Option<String>,
    /// When an input changed: the calculation redone with what the source
    /// now says. A new record, never an edit of the old one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corrected: Option<CalcRecord>,
    pub detail: String,
}

impl CheckReport {
    pub fn render(&self) -> String {
        let mut out = format!("Check of {}: {}\n", self.calc_id, self.verdict.as_str().to_uppercase());
        for input in &self.inputs {
            out.push_str(&format!("  {} = {} [{}]: {}", input.input, input.as_recorded, input.source, input.finding));
            if let Some(now) = &input.current_value {
                out.push_str(&format!(" (the source now says {now})"));
            }
            out.push('\n');
        }
        if let Some(recomputed) = &self.recomputed {
            out.push_str(&format!("Recomputed from the recorded inputs: {recomputed}. {}\n", self.agreement));
        }
        if let Some(si) = &self.si_path {
            out.push_str(&format!("Recomputed in SI base units: {si}\n"));
        }
        if let Some(corrected) = &self.corrected {
            let display = corrected.first().map(|r| r.display.clone()).unwrap_or_else(|| corrected.status.as_str().to_string());
            out.push_str(&format!("With the corrected input(s): {} = {display} ({}, a new record).\n", corrected.id, corrected.status.as_str()));
        }
        out.push_str(&self.detail);
        out
    }
}

/// Numbers with their units in a source's text: `9.0 mm`, `8.20mm`, `0.354 in`.
fn quantities_in(text: &str) -> Vec<(String, String)> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let starts = chars[i].is_ascii_digit() && (i == 0 || !(chars[i - 1].is_alphanumeric() || chars[i - 1] == '.'));
        if !starts {
            i += 1;
            continue;
        }
        let negative = i > 0 && (chars[i - 1] == '-' || chars[i - 1] == '−') && (i < 2 || !chars[i - 2].is_alphanumeric());
        let start = i;
        while i < chars.len() && (chars[i].is_ascii_digit() || (chars[i] == '.' && chars.get(i + 1).is_some_and(|c| c.is_ascii_digit()))) {
            i += 1;
        }
        // A date is a value too.
        if i - start == 4 && chars.get(i) == Some(&'-') {
            let window: String = chars[start..(start + 10).min(chars.len())].iter().collect();
            if window.len() == 10 && window.as_bytes()[7] == b'-' && window[5..7].bytes().all(|b| b.is_ascii_digit()) && window[8..].bytes().all(|b| b.is_ascii_digit()) {
                out.push((window, String::new()));
                i = start + 10;
                continue;
            }
        }
        let number: String = chars[start..i].iter().collect();
        let mut j = i;
        while j < chars.len() && chars[j] == ' ' {
            j += 1;
        }
        let unit_start = j;
        while j < chars.len() && (super::units::is_unit_char(chars[j]) || matches!(chars[j], '/' | '^' | '²' | '³') || (j > unit_start && chars[j].is_ascii_digit() && chars[j - 1] == '^')) {
            j += 1;
        }
        let unit: String = chars[unit_start..j].iter().collect::<String>().trim_end_matches(['/', '^']).to_string();
        let unit = if parse_unit(&unit).is_ok() { unit } else { String::new() };
        out.push((if negative { format!("-{number}") } else { number }, unit));
    }
    out
}

/// Whether `text` states `input`'s value: the same number (as a decimal) in
/// a unit that converts to the same amount.
fn states(text: &str, input: &CalcInput) -> bool {
    let Some(value) = input.value.as_deref() else { return false };
    let Ok(wanted) = input.quantity() else { return false };
    quantities_in(text).iter().any(|(number, unit)| {
        if wanted.kind == QKind::Date {
            return number == value;
        }
        let Some(found) = Decimal::parse(number) else { return false };
        // A bare number in the source is read in the input's own unit.
        let candidate = CalcInput { value: Some(found.to_plain()), unit: if unit.is_empty() { input.unit.clone() } else { unit.clone() }, ..input.clone() };
        match candidate.quantity() {
            Ok(q) if q.dim() == wanted.dim() && q.kind == wanted.kind => {
                let (a, b) = (q.si(), wanted.si());
                (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(f64::MIN_POSITIVE)
            }
            _ => false,
        }
    })
}

/// The value of the same kind the replacement text now states, if exactly
/// one reading of that dimension is there.
fn restated(text: &str, input: &CalcInput) -> Option<String> {
    let wanted = input.quantity().ok()?;
    let candidates: Vec<String> = quantities_in(text)
        .into_iter()
        .filter_map(|(number, unit)| {
            let unit = if unit.is_empty() { input.unit.clone() } else { unit };
            let candidate = CalcInput { value: Some(number.clone()), unit: unit.clone(), ..input.clone() };
            let q = candidate.quantity().ok()?;
            (q.dim() == wanted.dim() && q.kind == wanted.kind).then(|| if unit.is_empty() { number } else { format!("{number} {unit}") })
        })
        .collect();
    match candidates.as_slice() {
        [only] => Some(only.clone()),
        _ => None,
    }
}

/// Reruns an operation exactly as recorded, over `inputs`.
pub fn rerun(record: &CalcRecord, inputs: Vec<CalcInput>, resolver: Option<ops::Resolver>) -> CalcRecord {
    let options = Options::from_record(&record.options);
    let text = |key: &str| record.options.get(key).and_then(|v| v.as_str()).map(str::to_string);
    match record.operation {
        Operation::Evaluate => ops::evaluate(&record.equation, inputs, &options, resolver),
        Operation::ValidateDimensions => ops::validate_dimensions(&record.equation, inputs, options.result_unit.as_deref()),
        Operation::Solve => {
            let equations: Vec<String> = record.equation.split("; ").map(str::to_string).collect();
            let unknowns: Vec<String> = record
                .options
                .get("unknowns")
                .and_then(|u| u.as_array())
                .map(|u| u.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
                .unwrap_or_default();
            ops::solve(&equations, &unknowns, inputs, &options, resolver)
        }
        Operation::Compare => ops::compare(
            &text("value").unwrap_or_default(),
            &text("relation").unwrap_or_default(),
            &text("limit").unwrap_or_default(),
            text("tolerance").as_deref(),
            inputs,
            &options,
            resolver,
        ),
        Operation::Sensitivity => ops::sensitivity(Some(&record.equation), None, inputs, &options, resolver),
    }
}

/// The second path for an evaluation: every input converted to SI base
/// units before any arithmetic, so a unit-conversion slip in the first path
/// cannot agree with itself.
fn si_path(record: &CalcRecord) -> Option<Result<f64, String>> {
    if record.operation != Operation::Evaluate {
        return None;
    }
    let body = match expr::split_equation(&record.equation) {
        Some((_, rhs)) => rhs.to_string(),
        None => record.equation.clone(),
    };
    let node = expr::parse(&body).ok()?;
    let mut env = BTreeMap::new();
    for input in &record.inputs {
        let Ok(q) = input.quantity() else { continue };
        let converted = match q.kind {
            QKind::Linear => {
                let mut units = UnitMap::default();
                for (symbol, exponent) in ["m", "kg", "s", "K", "mol", "A"].iter().zip(q.dim()) {
                    if exponent != 0 {
                        units.add(lookup(symbol).expect("SI base unit"), exponent as i32);
                    }
                }
                Quantity { value: q.si(), units, kind: QKind::Linear, text: None }
            }
            QKind::AbsoluteTemperature => {
                let mut units = UnitMap::default();
                units.add(lookup("K").expect("kelvin"), 1);
                Quantity { value: q.si(), units, kind: QKind::AbsoluteTemperature, text: None }
            }
            _ => q,
        };
        env.insert(input.id.clone(), converted);
    }
    Some(Evaluator::quiet(&env).eval(&node).map(|q| q.si()).map_err(|e| e.describe()))
}

/// Checks `record` against its sources.
pub fn check(record: &CalcRecord, reader: &dyn SourceReader) -> CheckReport {
    let mut findings = Vec::new();
    let mut stale = false;
    let mut disagrees = false;
    let mut unresolved = !matches!(record.status, RecordStatus::Computed | RecordStatus::Refused);
    let mut unreadable = false;
    let mut corrected_inputs = record.inputs.clone();
    let mut corrected_any = false;

    for (index, input) in record.inputs.iter().enumerate() {
        let (finding, current) = match (&input.source, input.status()) {
            (_, InputStatus::Missing) => {
                unresolved = true;
                ("MISSING: no value".to_string(), None)
            }
            (_, InputStatus::Unsourced) => {
                unresolved = true;
                ("UNSOURCED: nothing to re-read".to_string(), None)
            }
            (InputSource::Assumption { reason }, _) => {
                unresolved = true;
                (format!("ASSUMED ({reason}): an assumption is stated, not verified"), None)
            }
            (InputSource::Requester, _) => ("taken as stated by the requester; a statement has no source to re-read".to_string(), None),
            (InputSource::Calculation { calculation_id }, _) => match reader.calculation(calculation_id) {
                None => {
                    stale = true;
                    (format!("{calculation_id} is gone or not readable"), None)
                }
                Some((referenced, why_stale)) => {
                    let now = referenced.first().map(|r| r.raw.clone());
                    if let Some(why) = why_stale {
                        stale = true;
                        (format!("{calculation_id} is stale: {why}"), None)
                    } else if now.as_deref() == input.value.as_deref() {
                        (format!("matches {calculation_id}'s result exactly"), None)
                    } else {
                        disagrees = true;
                        (format!("{calculation_id}'s result is not the value recorded"), now)
                    }
                }
            },
            (source, _) => {
                let state = match source {
                    InputSource::Memory { item_id, revision } => reader.memory(item_id, *revision),
                    InputSource::Evidence { handle } => reader.evidence(handle),
                    InputSource::Document { sha256, locator } => reader.document(sha256, locator.as_deref()),
                    _ => SourceState::Unreadable { why: "not a re-readable source".into() },
                };
                match state {
                    SourceState::Current { text, .. } if states(&text, input) => ("found at the source, unchanged".to_string(), None),
                    SourceState::Current { text, .. } => {
                        disagrees = true;
                        ("the source does not state this value".to_string(), restated(&text, input))
                    }
                    SourceState::Changed { text, why, now: replacement, .. } => {
                        stale = true;
                        let now = text.as_deref().and_then(|t| restated(t, input));
                        if let Some(now) = &now {
                            let mut words = now.splitn(2, ' ');
                            corrected_inputs[index].value = words.next().map(str::to_string);
                            corrected_inputs[index].unit = words.next().unwrap_or_default().to_string();
                            // Cited where the value now comes from; without a
                            // replacement to cite, the value is unsourced.
                            corrected_inputs[index].source = replacement.unwrap_or(InputSource::Unsourced);
                            corrected_any = true;
                        }
                        (format!("CHANGED: {why}"), now)
                    }
                    SourceState::Gone { why } => {
                        stale = true;
                        (format!("GONE: {why}"), None)
                    }
                    SourceState::Unreadable { why } => {
                        unreadable = true;
                        (format!("could not be re-read here: {why}"), None)
                    }
                }
            }
        };
        findings.push(InputCheck {
            input: input.id.clone(),
            as_recorded: input.written(),
            source: input.source.describe(),
            finding,
            current_value: current,
        });
    }

    // Recompute from the inputs exactly as recorded.
    let resolve = |id: &str| reader.calculation(id).map(|(record, _)| record);
    let again = rerun(record, record.inputs.clone(), Some(&resolve));
    let same_results = again.results.len() == record.results.len()
        && again.results.iter().zip(&record.results).all(|(a, b)| {
            let tolerance = record.tolerance.relative.unwrap_or(0.0).max(1e-12) * a.si_value.abs().max(b.si_value.abs());
            a.display == b.display && (a.si_value - b.si_value).abs() <= tolerance.max(f64::MIN_POSITIVE)
        });
    let same_refusal = record.status == RecordStatus::Refused
        && again.status == RecordStatus::Refused
        && again.error.as_ref().map(|e| e.code) == record.error.as_ref().map(|e| e.code);
    let agreement = if again.id != record.id {
        "The recomputation is a different record (the inputs or engine differ), so the two cannot be compared.".to_string()
    } else if same_refusal {
        "It refuses again, with the same error.".to_string()
    } else if same_results {
        format!("Same record id and results, within {}.", record.tolerance.describe())
    } else {
        "The results DIFFER from the record.".to_string()
    };
    let recomputed = match again.first() {
        Some(result) => Some(result.display.clone()),
        None => again.error.as_ref().map(|e| format!("refused ({})", e.describe())),
    };
    let si = si_path(record).map(|outcome| match (outcome, record.first()) {
        (Ok(value), Some(result)) => {
            let tolerance = 1e-9 * value.abs().max(result.si_value.abs()).max(f64::MIN_POSITIVE);
            if (value - result.si_value).abs() <= tolerance {
                format!("{} {} — agrees within relative 1e-9", super::eval::trim_number(value), result.si_unit)
            } else {
                format!("{} {} — DISAGREES with {} {}", super::eval::trim_number(value), result.si_unit, super::eval::trim_number(result.si_value), result.si_unit)
            }
        }
        (Ok(value), None) => format!("{} — but the record has no result", super::eval::trim_number(value)),
        (Err(error), _) => format!("refused ({error})"),
    });
    let si_disagrees = si.as_deref().is_some_and(|s| s.contains("DISAGREES") || (record.status.has_result() && s.starts_with("refused")));

    let verdict = if record.status == RecordStatus::Refused {
        if same_refusal { CheckVerdict::RefusalReproduced } else { CheckVerdict::Mismatch }
    } else if stale {
        CheckVerdict::Stale
    } else if disagrees {
        CheckVerdict::SourceDisagrees
    } else if !(same_results && again.id == record.id) || si_disagrees {
        CheckVerdict::Mismatch
    } else if unresolved {
        CheckVerdict::Unresolved
    } else if unreadable {
        CheckVerdict::Unverifiable
    } else {
        CheckVerdict::Verified
    };

    let corrected = (corrected_any && stale).then(|| rerun(record, corrected_inputs, Some(&resolve)));
    let detail = match verdict {
        CheckVerdict::Verified => "Every input was found at its source and both recomputations agree. Verified by recomputation, not by opinion.".to_string(),
        CheckVerdict::Stale => "An input's source has changed since this was calculated. Anything built on this figure is stale until it is recalculated from the current source.".to_string(),
        CheckVerdict::SourceDisagrees => "A source does not state the value the calculation used. The figure cannot be relied on until the input is corrected or its source cited.".to_string(),
        CheckVerdict::Unresolved => format!(
            "The arithmetic reproduces, but the calculation rests on unresolved input(s){}. It is not verified.",
            if record.unresolved.is_empty() { String::new() } else { format!(": {}", record.unresolved.join("; ")) }
        ),
        CheckVerdict::Unverifiable => "A source could not be re-read here, so the inputs cannot be confirmed. It is not verified.".to_string(),
        CheckVerdict::Mismatch => "The recomputation does not reproduce the record. Do not use this figure.".to_string(),
        CheckVerdict::RefusalReproduced => "The calculation was refused, and is refused again for the same reason. There is no figure to use.".to_string(),
    };
    CheckReport { calc_id: record.id.clone(), verdict, inputs: findings, recomputed, agreement, si_path: si, corrected, detail }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calculation::ops::evaluate;

    #[test]
    fn quantities_are_found_in_source_text() {
        let found = quantities_in("Minimum allowable thickness: 9.0 mm; measured 8.20mm at point C on 2026-08-12, -5 degC.");
        assert!(found.contains(&("9.0".into(), "mm".into())), "{found:?}");
        assert!(found.contains(&("8.20".into(), "mm".into())), "{found:?}");
        assert!(found.contains(&("2026-08-12".into(), String::new())), "{found:?}");
        assert!(found.contains(&("-5".into(), "degC".into())), "{found:?}");
    }

    #[test]
    fn a_value_is_stated_in_any_unit_that_converts_to_it() {
        let input = CalcInput::parse_line("t = 9.0 mm [M:mi-1@1]").unwrap();
        assert!(states("the minimum is 9.00 mm", &input));
        assert!(states("the minimum is 0.9 cm", &input));
        assert!(!states("the minimum is 9.5 mm", &input));
        assert!(!states("the minimum is 9.0 kg", &input));
        assert_eq!(restated("corrected: the minimum is 8.8 mm (was 9.0 mm)", &input), None, "two readings is not one");
        assert_eq!(restated("corrected: the minimum is 8.8 mm", &input).as_deref(), Some("8.8 mm"));
    }

    struct Fixed(SourceState);
    impl SourceReader for Fixed {
        fn memory(&self, _: &str, _: Option<u64>) -> SourceState {
            self.0.clone()
        }
        fn evidence(&self, _: &str) -> SourceState {
            self.0.clone()
        }
        fn document(&self, _: &str, _: Option<&str>) -> SourceState {
            self.0.clone()
        }
        fn calculation(&self, _: &str) -> Option<(CalcRecord, Option<String>)> {
            None
        }
    }

    fn record() -> CalcRecord {
        evaluate(
            "loss = (t_min - t_meas) / t_min",
            vec![
                CalcInput::parse_line("t_min = 9.0 mm [M:mi-min@1]").unwrap(),
                CalcInput::parse_line("t_meas = 8.2 mm [M:mi-meas@1]").unwrap(),
            ],
            &Options { result_unit: Some("%".into()), rounding: None },
            None,
        )
    }

    #[test]
    fn inputs_found_at_their_sources_verify() {
        let reader = Fixed(SourceState::Current { revision: Some(1), text: "minimum 9.0 mm; measured 8.2 mm".into() });
        let report = check(&record(), &reader);
        assert_eq!(report.verdict, CheckVerdict::Verified, "{}", report.render());
        assert!(report.si_path.unwrap().contains("agrees"));
    }

    #[test]
    fn a_changed_source_makes_it_stale_and_recomputes_with_the_new_value() {
        let reader = Fixed(SourceState::Changed {
            revision: Some(2),
            text: Some("corrected reading: 7.9 mm".into()),
            why: "superseded".into(),
            now: Some(InputSource::Memory { item_id: "mi-fix".into(), revision: Some(1) }),
        });
        let report = check(&record(), &reader);
        assert_eq!(report.verdict, CheckVerdict::Stale);
        let corrected = report.corrected.expect("recomputed with the corrected value");
        assert_ne!(corrected.id, record().id);
    }

    #[test]
    fn a_source_that_does_not_state_the_value_is_said() {
        let reader = Fixed(SourceState::Current { revision: Some(1), text: "minimum 9.5 mm".into() });
        assert_eq!(check(&record(), &reader).verdict, CheckVerdict::SourceDisagrees);
    }
}

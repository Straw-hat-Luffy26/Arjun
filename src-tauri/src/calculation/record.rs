//! The calculation record: what was computed, from what, by which engine, and
//! how to recompute it. Immutable once written, and named by its content.
//!
//! The id is a hash of the operation, the engine and unit-table version, the
//! equation, every input exactly as given (value text, unit, source,
//! uncertainty, assumptions) and the options. The same computation asked
//! twice is the same record; a corrected input is a different record, never an
//! edit of the old one. Results are not hashed: they are a function of what
//! is, and a checker proves that by recomputing them.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::decimal::Rounding;
use super::error::CalcError;
use super::eval::Step;
use super::inputs::{CalcInput, InputStatus};
use super::units;

pub const ENGINE_NAME: &str = "arjun-calc";
/// Bumped whenever a rule changes what a computation returns.
pub const ENGINE_VERSION: &str = "2.0.0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Engine {
    pub name: String,
    pub version: String,
    /// Digest of the unit table the factors came from.
    pub units: String,
}

impl Engine {
    pub fn current() -> Self {
        Self { name: ENGINE_NAME.into(), version: ENGINE_VERSION.into(), units: units::table_digest() }
    }

    pub fn describe(&self) -> String {
        format!("{} {} (units {})", self.name, self.version, self.units)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Evaluate,
    ValidateDimensions,
    Solve,
    Compare,
    Sensitivity,
}

impl Operation {
    pub fn as_str(self) -> &'static str {
        match self {
            Operation::Evaluate => "evaluate",
            Operation::ValidateDimensions => "validate_dimensions",
            Operation::Solve => "solve",
            Operation::Compare => "compare",
            Operation::Sensitivity => "sensitivity",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RecordStatus {
    /// Every input sourced; the result stands on them.
    Computed,
    /// Computed, but resting on an unsourced or assumed input.
    Provisional,
    /// An input has no value, so there is no number.
    Unresolved,
    /// Refused with a structured error; no number.
    Refused,
}

impl RecordStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            RecordStatus::Computed => "computed",
            RecordStatus::Provisional => "provisional",
            RecordStatus::Unresolved => "unresolved",
            RecordStatus::Refused => "refused",
        }
    }

    pub fn has_result(self) -> bool {
        matches!(self, RecordStatus::Computed | RecordStatus::Provisional)
    }
}

/// One result, raw and as written down.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CalcResult {
    pub name: String,
    /// The shortest decimal that reads back as `value`: the exact binary result.
    pub raw: String,
    pub value: f64,
    /// How the unit reads (`mm²`, `°C`).
    pub unit: String,
    /// The same unit in a form the input grammar parses (`mm^2`, `degC`).
    pub unit_code: String,
    /// Rounded by the record's rule, with its unit.
    pub display: String,
    pub si_value: f64,
    pub si_unit: String,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncertainty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncertainty_display: Option<String>,
}

/// What agreement means for this operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tolerance {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relative: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub absolute: Option<f64>,
    pub basis: String,
}

impl Tolerance {
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        if let Some(r) = self.relative {
            parts.push(format!("relative {r:e}"));
        }
        if let Some(a) = self.absolute {
            parts.push(format!("absolute {a:e} (SI)"));
        }
        if parts.is_empty() {
            self.basis.clone()
        } else {
            format!("{} — {}", parts.join(", "), self.basis)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DimensionReport {
    /// The dimension of the result, in words and SI base units.
    pub result: String,
    pub result_si: String,
    /// Each input's dimension, as read from its unit.
    pub inputs: Vec<String>,
    pub problems: Vec<CalcError>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Verdict {
    Holds,
    Fails,
    /// The uncertainty straddles the limit, or the two are equal within the
    /// engine's own tolerance on a strict relation.
    Indeterminate,
}

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Holds => "holds",
            Verdict::Fails => "fails",
            Verdict::Indeterminate => "indeterminate",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Comparison {
    pub value: String,
    pub relation: String,
    pub limit: String,
    pub verdict: Verdict,
    /// How far inside (positive) or outside (negative) the limit, in its unit.
    pub margin: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub margin_percent_of_limit: Option<String>,
    pub basis: String,
}

/// An engineering conclusion, which exists only with a supplied standard.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Conclusion {
    /// As supplied: name, edition and clause.
    pub standard: String,
    /// The input the criterion came from, and its source.
    pub criterion: String,
    pub statement: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SensitivityRow {
    pub input: String,
    /// ∂result/∂input, in result units per input unit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derivative: Option<String>,
    /// (∂result/∂input)·(input/result), dimensionless, on absolute (SI) values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elasticity: Option<f64>,
    /// The result at the input's low and high values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub swing: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SolverReport {
    /// `linear`, `linear system` or `bracketed root (bisection)`.
    pub family: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CalcRecord {
    pub id: String,
    pub operation: Operation,
    pub engine: Engine,
    pub equation: String,
    pub inputs: Vec<CalcInput>,
    /// Options that change the result (result unit, rounding, relation, …).
    pub options: serde_json::Value,
    pub substitutions: Vec<String>,
    pub steps: Vec<Step>,
    pub results: Vec<CalcResult>,
    pub rounding: Rounding,
    pub rounding_rule: String,
    pub tolerance: Tolerance,
    pub assumptions: Vec<String>,
    /// Inputs that are unsourced or missing, each with why.
    pub unresolved: Vec<String>,
    pub status: RecordStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<CalcError>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<DimensionReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comparison: Option<Comparison>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sensitivity: Vec<SensitivityRow>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solver: Option<SolverReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conclusion: Option<Conclusion>,
}

/// The content address: `calc-` and sixteen hex digits.
pub fn record_id(operation: Operation, engine: &Engine, equation: &str, inputs: &[CalcInput], options: &serde_json::Value) -> String {
    let canonical = serde_json::json!({
        "operation": operation.as_str(),
        "engine": engine,
        "equation": equation.trim(),
        "inputs": inputs,
        "options": options,
    });
    let digest = Sha256::digest(serde_json::to_vec(&canonical).unwrap_or_default());
    format!("calc-{}", digest[..8].iter().map(|b| format!("{b:02x}")).collect::<String>())
}

impl CalcRecord {
    pub fn first(&self) -> Option<&CalcResult> {
        self.results.first()
    }

    /// The hash of the whole record as stored, for tamper checks.
    pub fn sha256(&self) -> String {
        let digest = Sha256::digest(serde_json::to_vec(self).unwrap_or_default());
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }

    pub fn citation(&self) -> String {
        format!("[C:{}]", self.id)
    }

    /// The record as a model reads it: everything needed to quote the result
    /// exactly and to say what it rests on.
    pub fn render(&self) -> String {
        let mut out = format!(
            "{} · {} · {} · {}\n",
            self.id,
            self.operation.as_str(),
            self.status.as_str(),
            self.engine.describe()
        );
        out.push_str(&format!("Equation: {}\n", self.equation));
        if !self.inputs.is_empty() {
            out.push_str("Inputs:\n");
            for input in &self.inputs {
                let status = match input.status() {
                    InputStatus::Sourced => "sourced",
                    InputStatus::Assumed => "ASSUMED",
                    InputStatus::Unsourced => "UNSOURCED",
                    InputStatus::Missing => "MISSING",
                };
                out.push_str(&format!("  {} ({status})\n", input.line()));
            }
        }
        for substitution in &self.substitutions {
            out.push_str(&format!("Substituted: {substitution}\n"));
        }
        if !self.steps.is_empty() {
            out.push_str("Working:\n");
            for step in self.steps.iter().take(24) {
                out.push_str(&format!("  {} = {}\n", step.description, step.result));
            }
            if self.steps.len() > 24 {
                out.push_str(&format!("  … {} more step(s) in the record\n", self.steps.len() - 24));
            }
        }
        if let Some(solver) = &self.solver {
            out.push_str(&format!("Solver: {} — {}\n", solver.family, solver.detail));
        }
        for result in &self.results {
            out.push_str(&format!(
                "Result {}: {}{} (raw {}{}; SI {}{})\n",
                result.name,
                result.display,
                result.uncertainty_display.as_deref().map(|u| format!(" {u}")).unwrap_or_default(),
                result.raw,
                if result.unit.is_empty() { String::new() } else { format!(" {}", result.unit) },
                super::eval::trim_number(result.si_value),
                if result.si_unit.is_empty() { String::new() } else { format!(" {}", result.si_unit) }
            ));
        }
        if let Some(dimensions) = &self.dimensions {
            out.push_str(&format!("Dimensions: result is {} ({})\n", dimensions.result, dimensions.result_si));
            for problem in &dimensions.problems {
                out.push_str(&format!("  problem: {}\n", problem.describe()));
            }
        }
        if let Some(comparison) = &self.comparison {
            out.push_str(&format!(
                "Comparison: {} {} {} → {} (margin {}{}). {}\n",
                comparison.value,
                comparison.relation,
                comparison.limit,
                comparison.verdict.as_str().to_uppercase(),
                comparison.margin,
                comparison.margin_percent_of_limit.as_deref().map(|p| format!(", {p} of the limit")).unwrap_or_default(),
                comparison.basis
            ));
        }
        for row in &self.sensitivity {
            let mut parts = Vec::new();
            if let Some(d) = &row.derivative {
                parts.push(format!("∂/∂{} = {d}", row.input));
            }
            if let Some(e) = row.elasticity {
                parts.push(format!("elasticity {}", super::eval::trim_number((e * 1e4).round() / 1e4)));
            }
            if let Some(s) = &row.swing {
                parts.push(s.clone());
            }
            if let Some(n) = &row.note {
                parts.push(n.clone());
            }
            out.push_str(&format!("Sensitivity {}: {}\n", row.input, parts.join("; ")));
        }
        match &self.conclusion {
            Some(conclusion) => out.push_str(&format!("Conclusion: {} (criterion {}; standard as supplied: {})\n", conclusion.statement, conclusion.criterion, conclusion.standard)),
            None if self.operation == Operation::Compare => out.push_str(
                "No engineering conclusion: that needs a limit sourced from a supplied standard with its edition. This is a numerical comparison only.\n",
            ),
            None => {}
        }
        out.push_str(&format!("Rounding: {}. Tolerance: {}.\n", self.rounding_rule, self.tolerance.describe()));
        if !self.assumptions.is_empty() {
            out.push_str(&format!("Assumptions: {}\n", self.assumptions.join("; ")));
        }
        if !self.unresolved.is_empty() {
            out.push_str(&format!("Unresolved: {}\n", self.unresolved.join("; ")));
        }
        if let Some(error) = &self.error {
            let heading = if self.status == RecordStatus::Unresolved { "Unresolved" } else { "Refused" };
            out.push_str(&format!("{heading} — {}. No number was produced.\n", error.describe().trim_end_matches('.')));
        }
        if self.status.has_result() {
            out.push_str(&format!(
                "Cite it as {} and quote the result exactly as displayed; do not recompute or re-round it.",
                self.citation()
            ));
            if self.status == RecordStatus::Provisional {
                out.push_str(" It is PROVISIONAL: say which inputs are unresolved wherever it is used.");
            }
        }
        out
    }
}

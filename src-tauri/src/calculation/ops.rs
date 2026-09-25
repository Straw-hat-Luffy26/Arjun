//! The five operation families, and what each one promises.
//!
//! | Operation | Engine | Tolerance | Rounding |
//! |---|---|---|---|
//! | evaluate | the expression evaluator | relative 1e-12 (binary64, ≤ 256 operations) | display only; default 4 s.f., half away from zero |
//! | validate_dimensions | shapes only, no values | exact (integer exponents) | none |
//! | solve — one linear unknown or a square linear system (≤ 6) | probed coefficients, Gaussian elimination with partial pivoting | residual relative 1e-9, checked; condition number ≤ 1e10 | as evaluate |
//! | solve — one unknown in a bracket | bisection, ≤ 200 halvings | bracket width relative 1e-12, residual checked against the ends | as evaluate |
//! | compare | both sides in SI | engine relative 1e-12, plus the caller's stated tolerance | margin as evaluate |
//! | sensitivity | central differences, relative step 1e-6 | O(h²) truncation; local only | as evaluate |
//!
//! Anything else — a nonlinear system, an unbracketed nonlinear equation,
//! an equation in more unknowns than equations — is refused as unsupported.
//! Nothing is ever handed to a model to finish.

use std::collections::BTreeMap;

use serde_json::json;

use super::decimal::{self, Decimal, Rounding};
use super::error::{CalcError, ErrorCode};
use super::eval::{infer, trim_number, Evaluator, QKind, Quantity, Shape};
use super::expr::{self, Node, ParseProblem};
use super::inputs::{CalcInput, InputSource, InputStatus};
use super::record::*;
use super::units::{self, describe_dim, parse_unit, si_label, Kind, UnitMap};

/// Finds a calculation record by id, for inputs taken from another result.
pub type Resolver<'a> = &'a dyn Fn(&str) -> Option<CalcRecord>;

pub const EVALUATE_RELATIVE: f64 = 1e-12;
pub const LINEAR_RESIDUAL: f64 = 1e-9;
pub const BRACKET_RELATIVE: f64 = 1e-12;
pub const MAX_BISECTIONS: usize = 200;
pub const SENSITIVITY_STEP: f64 = 1e-6;
pub const MAX_UNKNOWNS: usize = 6;
pub const CONDITION_LIMIT: f64 = 1e10;

#[derive(Debug, Clone, Default)]
pub struct Options {
    /// Express the result in this unit.
    pub result_unit: Option<String>,
    /// How the result is written down. `None` is four significant figures
    /// with trailing zeros dropped.
    pub rounding: Option<Rounding>,
}

impl Options {
    fn json(&self) -> serde_json::Value {
        json!({ "resultUnit": self.result_unit, "round": self.rounding })
    }

    /// The options a record was computed under, read back for a rerun.
    pub fn from_record(options: &serde_json::Value) -> Options {
        Options {
            result_unit: options.get("resultUnit").and_then(|u| u.as_str()).map(str::to_string),
            rounding: options.get("round").and_then(|r| serde_json::from_value(r.clone()).ok()),
        }
    }
}

// ── Shared preparation ───────────────────────────────────────────────────

struct Prepared {
    env: BTreeMap<String, Quantity>,
    shapes: BTreeMap<String, Shape>,
    unresolved: Vec<String>,
    missing: Vec<String>,
    assumptions: Vec<String>,
    provisional: bool,
}

/// Resolves inputs taken from other calculations, then reads every input.
fn prepare(inputs: &mut [CalcInput], resolver: Option<Resolver>) -> Result<Prepared, CalcError> {
    let mut prepared = Prepared {
        env: BTreeMap::new(),
        shapes: BTreeMap::new(),
        unresolved: Vec::new(),
        missing: Vec::new(),
        assumptions: Vec::new(),
        provisional: false,
    };
    for input in inputs.iter_mut() {
        if let InputSource::Calculation { calculation_id } = &input.source {
            let referenced = resolver.and_then(|find| find(calculation_id));
            match referenced {
                Some(record) if record.status.has_result() && !record.results.is_empty() => {
                    let first = &record.results[0];
                    if input.value.is_none() {
                        input.value = Some(first.raw.clone());
                        input.unit = first.unit_code.clone();
                    }
                    if record.status == RecordStatus::Provisional {
                        prepared.provisional = true;
                        prepared.unresolved.push(format!("{} rests on {calculation_id}, which is itself provisional", input.id));
                    }
                }
                Some(record) if input.value.is_none() => {
                    prepared.missing.push(format!("{}: {calculation_id} is {} and has no result", input.id, record.status.as_str()));
                }
                None if input.value.is_none() => {
                    prepared.missing.push(format!("{}: {calculation_id} is not a calculation this reader can open", input.id));
                }
                _ => {}
            }
        }
        match input.status() {
            InputStatus::Missing => {
                if !prepared.missing.iter().any(|m| m.starts_with(&format!("{}:", input.id))) {
                    prepared.missing.push(format!("{}: no value", input.id));
                }
            }
            InputStatus::Unsourced => {
                prepared.provisional = true;
                let why = if input.standard.is_some() {
                    "names a standard but not where its value was read"
                } else {
                    "has no source"
                };
                prepared.unresolved.push(format!("{} = {} {why}", input.id, input.written()));
            }
            InputStatus::Assumed => {
                prepared.provisional = true;
                if let InputSource::Assumption { reason } = &input.source {
                    prepared.assumptions.push(format!("{} = {} is assumed: {reason}", input.id, input.written()));
                }
            }
            InputStatus::Sourced => {}
        }
        if let Some(standard) = &input.standard {
            prepared.assumptions.push(format!("{} is taken from {standard}, as supplied", input.id));
        }
        for note in &input.assumptions {
            prepared.assumptions.push(format!("{}: {note}", input.id));
        }
        if input.value.is_some() {
            let quantity = input.quantity()?;
            prepared.shapes.insert(input.id.clone(), Shape { dim: quantity.dim(), kind: quantity.kind });
            prepared.env.insert(input.id.clone(), quantity);
        } else {
            let unit = parse_unit(&input.unit).map_err(|p| CalcError::new(ErrorCode::UnknownUnit, format!("input {}: {}", input.id, p.explain())))?;
            prepared.shapes.insert(input.id.clone(), Shape { dim: unit.dim(), kind: unit.kind().into() });
        }
    }
    Ok(prepared)
}

fn parse_error(problem: ParseProblem) -> CalcError {
    let code = if problem.unit {
        if problem.message.contains("is not accepted") {
            ErrorCode::AmbiguousUnit
        } else {
            ErrorCode::UnknownUnit
        }
    } else if problem.message.contains("longer than") || problem.message.contains("more than") || problem.message.contains("deeper") {
        ErrorCode::Bounds
    } else {
        ErrorCode::Parse
    };
    CalcError { code, message: problem.message, at: problem.at }
}

fn base(operation: Operation, equation: &str, inputs: &[CalcInput], options: serde_json::Value, rounding: Rounding) -> CalcRecord {
    let engine = Engine::current();
    CalcRecord {
        id: record_id(operation, &engine, equation, inputs, &options),
        operation,
        engine,
        equation: equation.trim().to_string(),
        inputs: inputs.to_vec(),
        options,
        substitutions: Vec::new(),
        steps: Vec::new(),
        results: Vec::new(),
        rounding,
        rounding_rule: rounding.describe(),
        tolerance: Tolerance { relative: Some(EVALUATE_RELATIVE), absolute: None, basis: "IEEE-754 binary64 arithmetic, at most 256 operations each rounded once; display rounding is separate".into() },
        assumptions: Vec::new(),
        unresolved: Vec::new(),
        status: RecordStatus::Computed,
        error: None,
        dimensions: None,
        comparison: None,
        sensitivity: Vec::new(),
        solver: None,
        conclusion: None,
    }
}

fn refuse(mut record: CalcRecord, error: CalcError) -> CalcRecord {
    record.status = RecordStatus::Refused;
    record.results.clear();
    record.comparison = None;
    record.conclusion = None;
    record.error = Some(error);
    record
}

fn apply(record: &mut CalcRecord, prepared: &Prepared) {
    record.unresolved.extend(prepared.unresolved.iter().cloned());
    record.assumptions.extend(prepared.assumptions.iter().cloned());
}

fn is_identifier(text: &str) -> bool {
    text.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_') && text.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Every symbol is an input; inputs nobody used are said.
fn check_symbols(nodes: &[&Node], known: &[String], inputs: &[CalcInput], record: &mut CalcRecord) -> Result<(), CalcError> {
    let mut used: Vec<String> = Vec::new();
    for node in nodes {
        for symbol in node.symbols() {
            if !used.contains(&symbol) {
                used.push(symbol);
            }
        }
    }
    for symbol in &used {
        if !known.contains(symbol) {
            return Err(CalcError::new(
                ErrorCode::UnknownSymbol,
                format!("{symbol:?} is not one of the inputs. Define it in `inputs` (for example \"{symbol} = 9.0 mm [M:mi-…#r1]\")."),
            ));
        }
    }
    for input in inputs {
        if !used.contains(&input.id) {
            record.assumptions.push(format!("input {} is not used by the equation", input.id));
        }
    }
    Ok(())
}

/// Numbers written straight into an expression. A plain number is read as a
/// formula constant; one with a unit or a date is a value with no source.
fn literal_values(node: &Node, out: &mut Vec<String>) {
    match node {
        Node::Number { text, unit_text, .. } if !unit_text.is_empty() => out.push(format!("{text} {unit_text}")),
        Node::Date { text, .. } => out.push(text.clone()),
        Node::Neg { inner, .. } => literal_values(inner, out),
        Node::Binary { left, right, .. } => {
            literal_values(left, out);
            literal_values(right, out);
        }
        Node::Power { base, .. } => literal_values(base, out),
        Node::Call { args, .. } => args.iter().for_each(|a| literal_values(a, out)),
        _ => {}
    }
}

fn uses_year(nodes: &[&Node], inputs: &[CalcInput], result_unit: Option<&str>) -> bool {
    fn walk(node: &Node) -> bool {
        match node {
            Node::Number { unit, .. } => unit.parts.iter().any(|(u, _)| u.symbol == "a"),
            Node::Neg { inner, .. } => walk(inner),
            Node::Binary { left, right, .. } => walk(left) || walk(right),
            Node::Power { base, .. } => walk(base),
            Node::Call { args, .. } => args.iter().any(walk),
            _ => false,
        }
    }
    let in_unit = |text: &str| parse_unit(text).is_ok_and(|u| u.parts.iter().any(|(u, _)| u.symbol == "a"));
    nodes.iter().any(|n| walk(n)) || inputs.iter().any(|i| in_unit(&i.unit)) || result_unit.is_some_and(in_unit)
}

const JULIAN_YEAR: &str = "a year (a, yr) is the Julian year, 365.25 d";

/// The unit written so the input grammar reads it back.
fn unit_code(units: &UnitMap) -> String {
    units
        .parts
        .iter()
        .map(|(u, p)| if *p == 1 { u.symbol.to_string() } else { format!("{}^{p}", u.symbol) })
        .collect::<Vec<_>>()
        .join("*")
}

fn result_of(name: &str, q: &Quantity, rounding: Rounding, trim: bool) -> CalcResult {
    if q.kind == QKind::Date {
        let date = q.format();
        return CalcResult {
            name: name.to_string(),
            raw: date.clone(),
            value: q.value,
            unit: String::new(),
            unit_code: String::new(),
            display: date,
            si_value: q.value,
            si_unit: "d since 1970-01-01".into(),
            kind: "date".into(),
            uncertainty: None,
            uncertainty_display: None,
        };
    }
    let unit = q.label();
    let number = decimal::display(q.value, rounding, trim).unwrap_or_else(|| q.value.to_string());
    CalcResult {
        name: name.to_string(),
        raw: decimal::raw(q.value),
        value: q.value,
        unit: unit.clone(),
        unit_code: unit_code(&q.units),
        display: if unit.is_empty() { number } else { format!("{number} {unit}") },
        si_value: q.si(),
        si_unit: match q.kind {
            QKind::AbsoluteTemperature => "K".into(),
            QKind::Gauge => "Pa (gauge)".into(),
            _ => si_label(q.dim()),
        },
        kind: match q.kind {
            QKind::Linear => "linear",
            QKind::AbsoluteTemperature => "absoluteTemperature",
            QKind::Gauge => "gauge",
            QKind::Date => "date",
        }
        .into(),
        uncertainty: None,
        uncertainty_display: None,
    }
}

/// Expresses `q` in `target`, refusing a conversion that changes meaning.
fn convert(q: Quantity, target: &str) -> Result<Quantity, CalcError> {
    let target = target.trim();
    if q.kind == QKind::Date {
        return if target.is_empty() || target == "date" {
            Ok(q)
        } else {
            Err(CalcError::new(ErrorCode::DateMisuse, format!("the result is a date, and cannot be expressed in {target}")))
        };
    }
    let unit = parse_unit(target).map_err(|p| {
        let code = if matches!(p, units::UnitProblem::Ambiguous { .. }) { ErrorCode::AmbiguousUnit } else { ErrorCode::UnknownUnit };
        CalcError::new(code, format!("result unit: {}", p.explain()))
    })?;
    if unit.dim() != q.dim() {
        return Err(CalcError::new(
            ErrorCode::DimensionMismatch,
            format!("the result is {} and {target} is {}: the units do not match", describe_dim(q.dim()), describe_dim(unit.dim())),
        ));
    }
    let map = unit.as_map();
    let (value, kind) = match (q.kind, unit.kind()) {
        (QKind::AbsoluteTemperature, Kind::AbsoluteTemperature) => (q.si() / unit.factor() - unit.offset(), QKind::AbsoluteTemperature),
        (QKind::AbsoluteTemperature, _) => {
            return Err(CalcError::new(ErrorCode::AffineMisuse, "the result is an absolute temperature; ask for degC, degF or K"))
        }
        (QKind::Linear, Kind::AbsoluteTemperature) if unit.offset() == 0.0 => (q.si() / unit.factor(), QKind::Linear),
        (QKind::Linear, Kind::AbsoluteTemperature) => {
            return Err(CalcError::new(ErrorCode::AffineMisuse, "the result is a temperature difference; write the unit as delta_degC, delta_degF or K"))
        }
        (QKind::Gauge, Kind::Gauge) => (q.si() / unit.factor(), QKind::Gauge),
        (QKind::Gauge, _) | (_, Kind::Gauge) => {
            return Err(CalcError::new(ErrorCode::GaugeMisuse, "gauge and absolute pressures do not convert without a stated atmosphere; use absolute(p, p_atm)"))
        }
        _ => (q.si() / unit.factor(), QKind::Linear),
    };
    Ok(Quantity { value, units: map, kind, text: None })
}

/// Evaluates `node` against `env`, then expresses it in `result_unit`.
fn compute(node: &Node, env: &BTreeMap<String, Quantity>, result_unit: Option<&str>) -> Result<Quantity, CalcError> {
    let q = Evaluator::quiet(env).eval(node)?;
    match result_unit {
        Some(unit) => convert(q, unit),
        None => Ok(q),
    }
}

/// ∂result/∂input for one input, by central difference, in display units.
struct Derivative {
    display: f64,
    si: f64,
}

fn derivative(node: &Node, env: &BTreeMap<String, Quantity>, id: &str, result_unit: Option<&str>) -> Result<Derivative, CalcError> {
    let held = env.get(id).expect("an input in the environment");
    let x = held.value;
    let h = SENSITIVITY_STEP * x.abs().max(1.0);
    let at = |value: f64| -> Result<Quantity, CalcError> {
        let mut probe = env.clone();
        probe.insert(id.to_string(), Quantity { value, text: None, ..held.clone() });
        compute(node, &probe, result_unit)
    };
    let (plus, minus) = (at(x + h)?, at(x - h)?);
    let input_factor = match held.kind {
        QKind::AbsoluteTemperature | QKind::Gauge => held.units.single().map_or(1.0, |u| u.factor),
        _ => held.units.factor(),
    };
    Ok(Derivative {
        display: (plus.value - minus.value) / (2.0 * h),
        si: (plus.si() - minus.si()) / (2.0 * h * input_factor),
    })
}

/// First-order propagation of the inputs' stated uncertainties, assuming
/// they are independent: √Σ(∂f/∂xᵢ · uᵢ)².
fn propagate(node: &Node, env: &BTreeMap<String, Quantity>, inputs: &[CalcInput], result_unit: Option<&str>) -> Result<Option<f64>, CalcError> {
    let mut sum = 0.0;
    let mut any = false;
    for input in inputs {
        if !env.contains_key(&input.id) || !node.symbols().contains(&input.id) {
            continue;
        }
        if let Some(u) = input.absolute_uncertainty()? {
            let d = derivative(node, env, &input.id, result_unit)?;
            sum += (d.display * u).powi(2);
            any = true;
        }
    }
    Ok(any.then(|| sum.sqrt()))
}

const PROPAGATION: &str = "uncertainty propagated to first order from the inputs' stated ± values, assuming they are independent; no coverage factor applied";

fn with_uncertainty(mut result: CalcResult, u: Option<f64>) -> CalcResult {
    if let Some(u) = u {
        let shown = decimal::display(u, Rounding::Significant(2), true).unwrap_or_else(|| u.to_string());
        result.uncertainty = Some(u);
        result.uncertainty_display = Some(if result.unit.is_empty() { format!("±{shown}") } else { format!("±{shown} {}", result.unit) });
    }
    result
}

// ── evaluate ─────────────────────────────────────────────────────────────

/// `expression`, or `name = expression`, over `inputs`.
pub fn evaluate(expression: &str, inputs: Vec<CalcInput>, options: &Options, resolver: Option<Resolver>) -> CalcRecord {
    let rounding = options.rounding.unwrap_or(Rounding::DEFAULT);
    let mut inputs = inputs;
    let prepared = prepare(&mut inputs, resolver);
    let mut record = base(Operation::Evaluate, expression, &inputs, options.json(), rounding);
    let prepared = match prepared {
        Ok(p) => p,
        Err(error) => return refuse(record, error),
    };
    apply(&mut record, &prepared);
    let (name, body) = match expr::split_equation(expression) {
        Some((lhs, rhs)) if is_identifier(lhs.trim()) => (lhs.trim().to_string(), rhs),
        Some(_) => {
            return refuse(
                record,
                CalcError::new(ErrorCode::Unsupported, "evaluate takes an expression, or `name = expression`; to find an unknown in an equation use calculation.solve"),
            )
        }
        None => ("result".to_string(), expression),
    };
    let node = match expr::parse(body) {
        Ok(node) => node,
        Err(problem) => return refuse(record, parse_error(problem)),
    };
    let known: Vec<String> = inputs.iter().map(|i| i.id.clone()).collect();
    if let Err(error) = check_symbols(&[&node], &known, &inputs, &mut record) {
        return refuse(record, error);
    }
    let mut literals = Vec::new();
    literal_values(&node, &mut literals);
    if !literals.is_empty() {
        record.unresolved.push(format!(
            "{} written into the expression with no source",
            literals.join(", ")
        ));
    }
    record.substitutions.push(node.render(&|symbol| inputs.iter().find(|i| i.id == symbol).map(CalcInput::written)));
    if uses_year(&[&node], &inputs, options.result_unit.as_deref()) {
        record.assumptions.push(JULIAN_YEAR.into());
    }

    if !prepared.missing.is_empty() {
        // No value to evaluate with, but the units can still be wrong: a
        // mismatch refuses now rather than when the value arrives.
        let mut problems = Vec::new();
        infer(&node, &prepared.shapes, &mut problems);
        if let Some(first) = problems.into_iter().next() {
            return refuse(record, first);
        }
        record.status = RecordStatus::Unresolved;
        record.unresolved.extend(prepared.missing.iter().cloned());
        record.error = Some(CalcError::new(ErrorCode::MissingInput, format!("no number without {}", prepared.missing.join(", "))));
        return record;
    }

    let mut evaluator = Evaluator::new(&prepared.env);
    let q = match evaluator.eval(&node) {
        Ok(q) => q,
        Err(error) => return refuse(record, error),
    };
    record.steps = evaluator.steps;
    record.assumptions.extend(evaluator.notes);
    let q = match &options.result_unit {
        Some(unit) => match convert(q, unit) {
            Ok(q) => q,
            Err(error) => return refuse(record, error),
        },
        None => q,
    };
    let uncertainty = match propagate(&node, &prepared.env, &inputs, options.result_unit.as_deref()) {
        Ok(u) => u,
        Err(error) => return refuse(record, error),
    };
    if uncertainty.is_some() {
        record.assumptions.push(PROPAGATION.into());
    }
    record.results = vec![with_uncertainty(result_of(&name, &q, rounding, options.rounding.is_none()), uncertainty)];
    record.status = if prepared.provisional || !literals.is_empty() { RecordStatus::Provisional } else { RecordStatus::Computed };
    record
}

// ── validate_dimensions ──────────────────────────────────────────────────

/// Dimensions only: every addition, function argument, both sides of an
/// equation and the requested result unit. Inputs need units, not values.
pub fn validate_dimensions(text: &str, inputs: Vec<CalcInput>, result_unit: Option<&str>) -> CalcRecord {
    let mut inputs = inputs;
    let prepared = prepare(&mut inputs, None);
    let mut record = base(Operation::ValidateDimensions, text, &inputs, json!({ "resultUnit": result_unit }), Rounding::DEFAULT);
    record.tolerance = Tolerance { relative: None, absolute: None, basis: "exact: integer exponent arithmetic on the base dimensions, no numerical tolerance".into() };
    record.rounding_rule = "not applicable: no number is computed".into();
    let prepared = match prepared {
        Ok(p) => p,
        Err(error) => return refuse(record, error),
    };
    apply(&mut record, &prepared);
    record.unresolved.extend(prepared.missing.iter().cloned());
    let sides: Vec<&str> = match expr::split_equation(text) {
        Some((lhs, rhs)) => vec![lhs, rhs],
        None => vec![text],
    };
    let mut nodes = Vec::new();
    for side in &sides {
        match expr::parse(side) {
            Ok(node) => nodes.push(node),
            Err(problem) => return refuse(record, parse_error(problem)),
        }
    }
    let known: Vec<String> = inputs.iter().map(|i| i.id.clone()).collect();
    let node_refs: Vec<&Node> = nodes.iter().collect();
    if let Err(error) = check_symbols(&node_refs, &known, &inputs, &mut record) {
        return refuse(record, error);
    }
    let mut problems = Vec::new();
    let shapes: Vec<Shape> = nodes.iter().map(|n| infer(n, &prepared.shapes, &mut problems)).collect();
    if shapes.len() == 2 && shapes[0] != shapes[1] && problems.is_empty() {
        problems.push(CalcError::new(
            ErrorCode::DimensionMismatch,
            format!("the two sides differ: the left is {} and the right is {}", shapes[0].describe(), shapes[1].describe()),
        ));
    }
    let result = shapes[0];
    if let Some(unit) = result_unit {
        match parse_unit(unit) {
            Ok(expected) if expected.dim() != result.dim => problems.push(CalcError::new(
                ErrorCode::DimensionMismatch,
                format!("the result is {} and {unit} is {}", result.describe(), describe_dim(expected.dim())),
            )),
            Ok(_) => {}
            Err(p) => problems.push(CalcError::new(ErrorCode::UnknownUnit, format!("result unit: {}", p.explain()))),
        }
    }
    record.dimensions = Some(DimensionReport {
        result: result.describe(),
        result_si: if result.kind == QKind::Date { "date".into() } else { si_label(result.dim) },
        inputs: inputs
            .iter()
            .map(|i| format!("{}: {}", i.id, prepared.shapes.get(&i.id).map(|s| s.describe()).unwrap_or_else(|| "unknown".into())))
            .collect(),
        problems: problems.clone(),
    });
    if let Some(first) = problems.into_iter().next() {
        let mut refused = refuse(record, first);
        // The report keeps every problem; the error is the first.
        refused.status = RecordStatus::Refused;
        return refused;
    }
    record
}

// ── solve ────────────────────────────────────────────────────────────────

struct Unknown {
    symbol: String,
    units: UnitMap,
    kind: QKind,
    unit_text: String,
    bracket: Option<(f64, f64)>,
}

/// `t mm`, `t mm in 1..20`, `x in 0..5`.
fn parse_unknown(text: &str) -> Result<Unknown, CalcError> {
    let invalid = |message: String| CalcError::new(ErrorCode::InvalidInput, format!("unknown {text:?}: {message}"));
    let (head, bracket) = match text.rsplit_once(" in ") {
        Some((head, range)) => {
            let (lo, hi) = range.split_once("..").ok_or_else(|| invalid("a bracket is written lo..hi".into()))?;
            let lo = Decimal::parse(lo).map(|d| d.to_f64()).ok_or_else(|| invalid(format!("{lo:?} is not a number")))?;
            let hi = Decimal::parse(hi).map(|d| d.to_f64()).ok_or_else(|| invalid(format!("{hi:?} is not a number")))?;
            if lo >= hi {
                return Err(invalid("a bracket's low end must be below its high end".into()));
            }
            (head, Some((lo, hi)))
        }
        None => (text, None),
    };
    let mut words = head.trim().splitn(2, char::is_whitespace);
    let symbol = words.next().unwrap_or_default().to_string();
    if !is_identifier(&symbol) {
        return Err(invalid("it needs a name".into()));
    }
    let unit_text = words.next().unwrap_or_default().trim().to_string();
    let unit = parse_unit(&unit_text).map_err(|p| CalcError::new(ErrorCode::UnknownUnit, format!("unknown {symbol}: {}", p.explain())))?;
    if unit.kind() == Kind::Gauge {
        return Err(invalid("solve for an absolute pressure, not a gauge one".into()));
    }
    Ok(Unknown { symbol, units: unit.as_map(), kind: unit.kind().into(), unit_text, bracket })
}

/// Equations in as many unknowns: one linear unknown, a linear system of up
/// to six, or one unknown inside a stated bracket.
pub fn solve(equations: &[String], unknowns: &[String], inputs: Vec<CalcInput>, options: &Options, resolver: Option<Resolver>) -> CalcRecord {
    let rounding = options.rounding.unwrap_or(Rounding::DEFAULT);
    let mut inputs = inputs;
    let prepared = prepare(&mut inputs, resolver);
    let equation_text = equations.join("; ");
    let mut record = base(
        Operation::Solve,
        &equation_text,
        &inputs,
        json!({ "unknowns": unknowns, "round": options.rounding }),
        rounding,
    );
    let prepared = match prepared {
        Ok(p) => p,
        Err(error) => return refuse(record, error),
    };
    apply(&mut record, &prepared);

    if equations.is_empty() || equations.len() > MAX_UNKNOWNS {
        return refuse(record, CalcError::new(ErrorCode::Bounds, format!("solve takes 1 to {MAX_UNKNOWNS} equations")));
    }
    if unknowns.len() != equations.len() {
        return refuse(
            record,
            CalcError::new(
                ErrorCode::Unsupported,
                format!("{} equation(s) in {} unknown(s): this engine solves square systems only, one equation per unknown", equations.len(), unknowns.len()),
            ),
        );
    }
    let unknowns: Vec<Unknown> = match unknowns.iter().map(|u| parse_unknown(u)).collect() {
        Ok(u) => u,
        Err(error) => return refuse(record, error),
    };
    for unknown in &unknowns {
        if inputs.iter().any(|i| i.id == unknown.symbol) {
            return refuse(record, CalcError::new(ErrorCode::InvalidInput, format!("{} is both an input and an unknown", unknown.symbol)));
        }
    }
    let mut sides: Vec<(Node, Node)> = Vec::new();
    for equation in equations {
        let Some((lhs, rhs)) = expr::split_equation(equation) else {
            return refuse(record, CalcError::new(ErrorCode::Parse, format!("{equation:?} is not an equation: it needs one `=`")));
        };
        match (expr::parse(lhs), expr::parse(rhs)) {
            (Ok(l), Ok(r)) => sides.push((l, r)),
            (Err(problem), _) | (_, Err(problem)) => return refuse(record, parse_error(problem)),
        }
    }
    let mut known: Vec<String> = inputs.iter().map(|i| i.id.clone()).collect();
    known.extend(unknowns.iter().map(|u| u.symbol.clone()));
    let node_refs: Vec<&Node> = sides.iter().flat_map(|(l, r)| [l, r]).collect();
    if let Err(error) = check_symbols(&node_refs, &known, &inputs, &mut record) {
        return refuse(record, error);
    }
    record.assumptions.retain(|a| !unknowns.iter().any(|u| a == &format!("input {} is not used by the equation", u.symbol)));
    let mut literals = Vec::new();
    node_refs.iter().for_each(|n| literal_values(n, &mut literals));
    if !literals.is_empty() {
        record.unresolved.push(format!("{} written into the equations with no source", literals.join(", ")));
    }
    if uses_year(&node_refs, &inputs, None) {
        record.assumptions.push(JULIAN_YEAR.into());
    }

    // Dimensions, with each unknown's shape from its unit.
    let mut shapes = prepared.shapes.clone();
    for unknown in &unknowns {
        shapes.insert(unknown.symbol.clone(), Shape { dim: unknown.units.dim(), kind: unknown.kind });
    }
    let mut problems = Vec::new();
    for (i, (lhs, rhs)) in sides.iter().enumerate() {
        let l = infer(lhs, &shapes, &mut problems);
        let r = infer(rhs, &shapes, &mut problems);
        if l.dim != r.dim {
            problems.push(CalcError::new(
                ErrorCode::DimensionMismatch,
                format!("equation {}: the left side is {} and the right is {}", i + 1, l.describe(), r.describe()),
            ));
        }
    }
    if let Some(first) = problems.into_iter().next() {
        return refuse(record, first);
    }
    if !prepared.missing.is_empty() {
        record.status = RecordStatus::Unresolved;
        record.unresolved.extend(prepared.missing.iter().cloned());
        record.error = Some(CalcError::new(ErrorCode::MissingInput, format!("no solution without {}", prepared.missing.join(", "))));
        return record;
    }

    let residual = |x: &[f64]| -> Result<(Vec<f64>, Vec<f64>), CalcError> {
        let mut env = prepared.env.clone();
        for (unknown, value) in unknowns.iter().zip(x) {
            env.insert(unknown.symbol.clone(), Quantity { value: *value, units: unknown.units.clone(), kind: unknown.kind, text: None });
        }
        let mut r = Vec::new();
        let mut scale = Vec::new();
        for (lhs, rhs) in &sides {
            let mut evaluator = Evaluator::quiet(&env);
            let (l, rr) = (evaluator.eval(lhs)?, evaluator.eval(rhs)?);
            r.push(l.si() - rr.si());
            scale.push(l.si().abs().max(rr.si().abs()));
        }
        Ok((r, scale))
    };

    let n = unknowns.len();
    let solution: Vec<f64> = if let (1, Some((lo, hi))) = (n, unknowns[0].bracket) {
        match bisect(&|x| residual(&[x]).map(|(r, _)| r[0]), lo, hi, &unknowns[0]) {
            Ok((root, detail)) => {
                record.solver = Some(SolverReport { family: "bracketed root (bisection)".into(), detail });
                record.tolerance = Tolerance {
                    relative: Some(BRACKET_RELATIVE),
                    absolute: None,
                    basis: format!("bisection to a bracket narrower than {BRACKET_RELATIVE:e} of the root (at most {MAX_BISECTIONS} halvings); the root is the only one the bracket isolates, not necessarily the only one"),
                };
                vec![root]
            }
            Err(error) => return refuse(record, error),
        }
    } else {
        match linear(&residual, n, &unknowns) {
            Ok((x, detail)) => {
                record.solver = Some(SolverReport { family: if n == 1 { "linear".into() } else { "linear system".into() }, detail });
                record.tolerance = Tolerance {
                    relative: Some(LINEAR_RESIDUAL),
                    absolute: None,
                    basis: format!("coefficients probed from the equations and checked for linearity; Gaussian elimination with partial pivoting; condition number at most {CONDITION_LIMIT:e}; every equation's residual checked below {LINEAR_RESIDUAL:e} of its size"),
                };
                x
            }
            Err(error) => return refuse(record, error),
        }
    };

    let solved: BTreeMap<String, String> = unknowns
        .iter()
        .zip(&solution)
        .map(|(u, x)| {
            let q = Quantity { value: *x, units: u.units.clone(), kind: u.kind, text: None };
            (u.symbol.clone(), q.format())
        })
        .collect();
    for (lhs, rhs) in &sides {
        let substitute = |symbol: &str| {
            inputs.iter().find(|i| i.id == symbol).map(CalcInput::written).or_else(|| solved.get(symbol).cloned())
        };
        record.substitutions.push(format!("{} = {}", lhs.render(&substitute), rhs.render(&substitute)));
    }
    record.results = unknowns
        .iter()
        .zip(&solution)
        .map(|(u, x)| {
            let q = Quantity { value: *x, units: u.units.clone(), kind: u.kind, text: None };
            result_of(&u.symbol, &q, rounding, options.rounding.is_none())
        })
        .collect();
    record.status = if prepared.provisional || !literals.is_empty() { RecordStatus::Provisional } else { RecordStatus::Computed };
    record
}

type Residual<'a> = dyn Fn(&[f64]) -> Result<(Vec<f64>, Vec<f64>), CalcError> + 'a;

fn linear(residual: &Residual, n: usize, unknowns: &[Unknown]) -> Result<(Vec<f64>, String), CalcError> {
    let names = unknowns.iter().map(|u| u.symbol.as_str()).collect::<Vec<_>>().join(", ");
    let probe_failed = |error: CalcError| {
        CalcError::new(
            ErrorCode::Nonlinear,
            format!("the equations could not be probed for linearity in {names} ({}). Give one unknown a bracket (`x unit in lo..hi`) to solve it as a bracketed root", error.describe()),
        )
    };
    let x0 = vec![1.0; n];
    let (r0, scale0) = residual(&x0).map_err(probe_failed)?;
    let mut columns: Vec<Vec<f64>> = Vec::new();
    for j in 0..n {
        let mut x = x0.clone();
        x[j] += 1.0;
        let (rj, _) = residual(&x).map_err(probe_failed)?;
        let column: Vec<f64> = rj.iter().zip(&r0).map(|(a, b)| a - b).collect();
        x[j] += 1.0;
        let (r2, _) = residual(&x).map_err(probe_failed)?;
        for i in 0..n {
            let tolerance = LINEAR_RESIDUAL * r0[i].abs().max(r2[i].abs()).max(scale0[i]).max(f64::MIN_POSITIVE);
            if (r2[i] - r0[i] - 2.0 * column[i]).abs() > tolerance {
                return Err(CalcError::new(
                    ErrorCode::Nonlinear,
                    format!(
                        "equation {} is not linear in {}. {}",
                        i + 1,
                        unknowns[j].symbol,
                        if n == 1 { "Give a bracket (`x unit in lo..hi`) to solve it as a bracketed root." } else { "Nonlinear systems are not a supported family." }
                    ),
                ));
            }
        }
        columns.push(column);
    }
    if n > 1 {
        let joint: Vec<f64> = x0.iter().map(|v| v + 1.0).collect();
        let (rj, _) = residual(&joint).map_err(probe_failed)?;
        for i in 0..n {
            let predicted: f64 = r0[i] + columns.iter().map(|c| c[i]).sum::<f64>();
            let tolerance = LINEAR_RESIDUAL * rj[i].abs().max(r0[i].abs()).max(scale0[i]).max(f64::MIN_POSITIVE);
            if (rj[i] - predicted).abs() > tolerance {
                return Err(CalcError::new(ErrorCode::Nonlinear, format!("equation {} couples the unknowns nonlinearly; nonlinear systems are not a supported family", i + 1)));
            }
        }
    }
    let a: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| columns[j][i]).collect()).collect();
    let b: Vec<f64> = r0.iter().map(|v| -v).collect();
    let (dx, condition) = gauss(a, b)?;
    let x: Vec<f64> = x0.iter().zip(&dx).map(|(a, b)| a + b).collect();
    let (r, scale) = residual(&x).map_err(|error| CalcError::new(ErrorCode::Domain, format!("the solution falls outside where the equations are defined: {}", error.describe())))?;
    let largest = scale.iter().cloned().fold(0.0, f64::max);
    for i in 0..n {
        if r[i].abs() > LINEAR_RESIDUAL * scale[i].max(largest * 1e-3).max(f64::MIN_POSITIVE) {
            return Err(CalcError::new(
                ErrorCode::NotConverged,
                format!("equation {} is not satisfied to {LINEAR_RESIDUAL:e} at the computed solution (residual {} SI); the system may be nearly singular", i + 1, trim_number(r[i])),
            ));
        }
    }
    Ok((x, format!("condition number {}; residuals checked", decimal::display(condition, Rounding::Significant(3), true).unwrap_or_default())))
}

/// Gaussian elimination with row scaling and partial pivoting, returning the
/// solution and an estimate of the condition number (∞-norm).
fn gauss(a: Vec<Vec<f64>>, b: Vec<f64>) -> Result<(Vec<f64>, f64), CalcError> {
    let n = a.len();
    let mut m: Vec<Vec<f64>> = Vec::new();
    for (row, rhs) in a.into_iter().zip(b) {
        let size = row.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
        if size == 0.0 {
            return Err(CalcError::new(ErrorCode::Singular, "an equation does not depend on any unknown, so the system has no unique solution"));
        }
        let mut scaled: Vec<f64> = row.iter().map(|v| v / size).collect();
        scaled.push(rhs / size);
        // The identity, for the inverse.
        scaled.extend((0..n).map(|k| if k == m.len() { 1.0 } else { 0.0 }));
        m.push(scaled);
    }
    let a_norm = m.iter().map(|row| row[..n].iter().map(|v| v.abs()).sum::<f64>()).fold(0.0, f64::max);
    for col in 0..n {
        let pivot_row = (col..n).max_by(|i, j| m[*i][col].abs().total_cmp(&m[*j][col].abs())).expect("rows");
        if m[pivot_row][col].abs() <= 1e-12 {
            return Err(CalcError::new(
                ErrorCode::Singular,
                format!("the equations do not determine the unknowns: the system is singular (rank {col} of {n}), so there is either no solution or infinitely many"),
            ));
        }
        m.swap(col, pivot_row);
        for row in 0..n {
            if row != col {
                let factor = m[row][col] / m[col][col];
                if factor != 0.0 {
                    for k in col..m[row].len() {
                        m[row][k] -= factor * m[col][k];
                    }
                }
            }
        }
    }
    let x: Vec<f64> = (0..n).map(|i| m[i][n] / m[i][i]).collect();
    // The inverse of the row-scaled system, whose pivots were chosen: its
    // condition number is the one that says how much the answer can move.
    let inverse_norm = (0..n).map(|i| (0..n).map(|k| (m[i][n + 1 + k] / m[i][i]).abs()).sum::<f64>()).fold(0.0, f64::max);
    let condition = a_norm * inverse_norm;
    if !condition.is_finite() || condition > CONDITION_LIMIT {
        return Err(CalcError::new(
            ErrorCode::IllConditioned,
            format!("the system is ill-conditioned (condition number about {}): the inputs cannot pin the answer down", decimal::display(condition, Rounding::Significant(2), true).unwrap_or_else(|| "infinite".into())),
        ));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(CalcError::new(ErrorCode::NonFinite, "the solution is not finite"));
    }
    Ok((x, condition))
}

fn bisect(f: &dyn Fn(f64) -> Result<f64, CalcError>, lo: f64, hi: f64, unknown: &Unknown) -> Result<(f64, String), CalcError> {
    let at_end = |x: f64| {
        f(x).map_err(|error| CalcError::new(ErrorCode::Domain, format!("the equation cannot be evaluated at the bracket end {} {}: {}", trim_number(x), unknown.unit_text, error.describe())))
    };
    let (mut a, mut b) = (lo, hi);
    let (mut fa, fb) = (at_end(a)?, at_end(b)?);
    let ends = fa.abs().max(fb.abs());
    if fa == 0.0 {
        return Ok((a, "the low end of the bracket is a root".into()));
    }
    if fb == 0.0 {
        return Ok((b, "the high end of the bracket is a root".into()));
    }
    if fa.signum() == fb.signum() {
        return Err(CalcError::new(
            ErrorCode::NoSignChange,
            format!(
                "the equation has the same sign at both ends of {}..{} {}, so the bracket isolates no root (there may be none, or an even number)",
                trim_number(lo),
                trim_number(hi),
                unknown.unit_text
            ),
        ));
    }
    let mut iterations = 0;
    let mut mid = 0.5 * (a + b);
    while iterations < MAX_BISECTIONS {
        iterations += 1;
        mid = 0.5 * (a + b);
        let fm = f(mid).map_err(|error| CalcError::new(ErrorCode::NotConverged, format!("the equation is undefined inside the bracket at {} {}: {}", trim_number(mid), unknown.unit_text, error.describe())))?;
        if fm == 0.0 || (b - a).abs() <= BRACKET_RELATIVE * mid.abs().max(1.0) {
            break;
        }
        if fm.signum() == fa.signum() {
            a = mid;
            fa = fm;
        } else {
            b = mid;
        }
    }
    if (b - a).abs() > BRACKET_RELATIVE * mid.abs().max(1.0) * 2.0 {
        return Err(CalcError::new(ErrorCode::NotConverged, format!("the bracket did not narrow to {BRACKET_RELATIVE:e} in {MAX_BISECTIONS} halvings")));
    }
    let residual = f(mid).unwrap_or(f64::INFINITY);
    if residual.abs() > 1e-6 * ends {
        return Err(CalcError::new(
            ErrorCode::NotConverged,
            "the sign change inside the bracket is a discontinuity, not a root: the equation does not approach zero there",
        ));
    }
    Ok((mid, format!("{iterations} halvings; final bracket width {} {}", trim_number((b - a).abs()), unknown.unit_text)))
}

// ── compare ──────────────────────────────────────────────────────────────

fn has_edition(standard: &str) -> bool {
    let lower = standard.to_ascii_lowercase();
    let year = standard
        .as_bytes()
        .windows(4)
        .enumerate()
        .any(|(i, w)| {
            w.iter().all(u8::is_ascii_digit)
                && standard.as_bytes().get(i + 4).is_none_or(|c| !c.is_ascii_digit())
                && (i == 0 || !standard.as_bytes()[i - 1].is_ascii_digit())
                && (1900..=2099).contains(&std::str::from_utf8(w).unwrap_or("0").parse::<u32>().unwrap_or(0))
        });
    year || ["edition", "rev ", "rev.", "revision", "addendum"].iter().any(|t| lower.contains(t))
}

/// Whether `value` stands in `relation` to `limit`, and by how much.
///
/// An engineering conclusion is drawn only when the limit is an input
/// sourced from a supplied standard that names its edition, and the value
/// rests on no unresolved input. Otherwise the answer is a numerical
/// comparison, and the record says that is all it is.
pub fn compare(value: &str, relation: &str, limit: &str, tolerance: Option<&str>, inputs: Vec<CalcInput>, options: &Options, resolver: Option<Resolver>) -> CalcRecord {
    let rounding = options.rounding.unwrap_or(Rounding::DEFAULT);
    let mut inputs = inputs;
    // A calculation's result compared directly.
    let reference = value.trim().trim_start_matches('[').trim_end_matches(']').trim_start_matches("C:");
    let value_expression = if reference.starts_with("calc-") && !reference.contains(char::is_whitespace) {
        let id = if inputs.iter().any(|i| i.id == "value") { "compared".to_string() } else { "value".to_string() };
        inputs.insert(0, CalcInput {
            id: id.clone(),
            value: None,
            unit: String::new(),
            source: InputSource::Calculation { calculation_id: reference.to_string() },
            standard: None,
            uncertainty: None,
            assumptions: Vec::new(),
        });
        id
    } else {
        value.trim().to_string()
    };
    let prepared = prepare(&mut inputs, resolver);
    let equation = format!("{value_expression} {relation} {limit}");
    let mut record = base(
        Operation::Compare,
        &equation,
        &inputs,
        json!({ "value": value_expression, "relation": relation.trim(), "limit": limit.trim(), "tolerance": tolerance, "round": options.rounding }),
        rounding,
    );
    let prepared = match prepared {
        Ok(p) => p,
        Err(error) => return refuse(record, error),
    };
    apply(&mut record, &prepared);
    let relation = match relation.trim() {
        "<=" | "≤" => "<=",
        "<" => "<",
        ">=" | "≥" => ">=",
        ">" => ">",
        "==" | "=" => "==",
        other => {
            return refuse(record, CalcError::new(ErrorCode::Unsupported, format!("{other:?} is not a relation this engine compares; use <=, <, >=, > or ==")))
        }
    };
    let (value_node, limit_node) = match (expr::parse(&value_expression), expr::parse(limit)) {
        (Ok(v), Ok(l)) => (v, l),
        (Err(problem), _) | (_, Err(problem)) => return refuse(record, parse_error(problem)),
    };
    let known: Vec<String> = inputs.iter().map(|i| i.id.clone()).collect();
    if let Err(error) = check_symbols(&[&value_node, &limit_node], &known, &inputs, &mut record) {
        return refuse(record, error);
    }
    let mut literals = Vec::new();
    literal_values(&value_node, &mut literals);
    literal_values(&limit_node, &mut literals);
    if !literals.is_empty() {
        record.unresolved.push(format!("{} written in with no source", literals.join(", ")));
    }
    if !prepared.missing.is_empty() {
        record.status = RecordStatus::Unresolved;
        record.unresolved.extend(prepared.missing.iter().cloned());
        record.error = Some(CalcError::new(ErrorCode::MissingInput, format!("no comparison without {}", prepared.missing.join(", "))));
        return record;
    }
    let mut evaluator = Evaluator::new(&prepared.env);
    let (v, l) = match (evaluator.eval(&value_node), evaluator.eval(&limit_node)) {
        (Ok(v), Ok(l)) => (v, l),
        (Err(error), _) | (_, Err(error)) => return refuse(record, error),
    };
    record.steps = evaluator.steps;
    record.assumptions.extend(evaluator.notes);
    if v.dim() != l.dim() || (v.kind == QKind::Date) != (l.kind == QKind::Date) || (v.kind == QKind::Gauge) != (l.kind == QKind::Gauge) {
        return refuse(
            record,
            CalcError::new(ErrorCode::DimensionMismatch, format!("{} and {} are not the same kind of quantity, so they do not compare", v.written(), l.written())),
        );
    }
    // Uncertainties, in SI, combined in quadrature.
    let scale_of = |q: &Quantity| match q.kind {
        QKind::AbsoluteTemperature | QKind::Gauge => q.units.single().map_or(1.0, |u| u.factor),
        QKind::Date => 1.0,
        QKind::Linear => q.units.factor(),
    };
    let u_value = match propagate(&value_node, &prepared.env, &inputs, None) {
        Ok(u) => u.unwrap_or(0.0) * scale_of(&v),
        Err(error) => return refuse(record, error),
    };
    let u_limit = match propagate(&limit_node, &prepared.env, &inputs, None) {
        Ok(u) => u.unwrap_or(0.0) * scale_of(&l),
        Err(error) => return refuse(record, error),
    };
    let u = (u_value.powi(2) + u_limit.powi(2)).sqrt();
    let stated = match tolerance.map(str::trim).filter(|t| !t.is_empty()) {
        None => 0.0,
        Some(text) => {
            let text = text.trim_start_matches('±').trim_start_matches("+/-").trim();
            if let Some(percent) = text.strip_suffix('%') {
                match Decimal::parse(percent.trim()) {
                    Some(p) => l.si().abs() * p.to_f64() / 100.0,
                    None => return refuse(record, CalcError::new(ErrorCode::InvalidInput, format!("tolerance {text:?} is not a percentage"))),
                }
            } else {
                let mut parts = text.splitn(2, char::is_whitespace);
                let number = parts.next().and_then(Decimal::parse).map(|d| d.to_f64());
                let unit = parse_unit(parts.next().unwrap_or_default());
                match (number, unit) {
                    (Some(n), Ok(unit)) if unit.dim() == l.dim() && n >= 0.0 => n * unit.factor(),
                    _ => {
                        return refuse(
                            record,
                            CalcError::new(ErrorCode::InvalidInput, format!("tolerance {text:?} must be a non-negative amount in the limit's units, or a percentage")),
                        )
                    }
                }
            }
        }
    };
    let (vs, ls) = (v.si(), l.si());
    let noise = EVALUATE_RELATIVE * vs.abs().max(ls.abs());
    let verdict = if relation == "==" {
        let gap = (vs - ls).abs();
        if gap + u <= stated + noise {
            Verdict::Holds
        } else if gap - u > stated + noise {
            Verdict::Fails
        } else {
            Verdict::Indeterminate
        }
    } else {
        // Positive margin: inside the limit.
        let margin = if relation.starts_with('<') { ls - vs } else { vs - ls };
        let strict = relation.len() == 1;
        if margin - u > noise {
            Verdict::Holds
        } else if margin + u < -noise {
            Verdict::Fails
        } else if u == 0.0 && margin.abs() <= noise {
            if strict { Verdict::Indeterminate } else { Verdict::Holds }
        } else {
            Verdict::Indeterminate
        }
    };
    let margin_si = if relation.starts_with('>') { vs - ls } else { ls - vs };
    let limit_scale = scale_of(&l);
    let margin_in_limit = margin_si / limit_scale;
    let margin_label = match l.kind {
        QKind::AbsoluteTemperature => match l.units.single().map(|u| u.symbol) {
            Some("degC") => "Δ°C".to_string(),
            Some("degF") => "Δ°F".to_string(),
            _ => "K".to_string(),
        },
        QKind::Gauge => l.label().trim_end_matches('g').to_string(),
        QKind::Date => "d".into(),
        QKind::Linear => l.label(),
    };
    let margin_number = decimal::display(margin_in_limit, rounding, options.rounding.is_none()).unwrap_or_default();
    let mut basis = format!("compared in SI; engine tolerance relative {EVALUATE_RELATIVE:e}");
    if stated > 0.0 {
        basis.push_str(&format!("; stated tolerance {}", tolerance.unwrap_or_default().trim()));
    }
    if u > 0.0 {
        basis.push_str("; the inputs' stated uncertainties, propagated and combined in quadrature, are applied to both sides (no coverage factor)");
    }
    record.comparison = Some(Comparison {
        value: v.written(),
        relation: relation.to_string(),
        limit: l.written(),
        verdict,
        margin: if margin_label.is_empty() { margin_number } else { format!("{margin_number} {margin_label}") },
        margin_percent_of_limit: (l.kind == QKind::Linear && ls != 0.0)
            .then(|| format!("{}%", decimal::display(100.0 * margin_si / ls.abs(), Rounding::Significant(3), true).unwrap_or_default())),
        basis,
    });
    record.results = vec![result_of(&value_expression, &v, rounding, options.rounding.is_none()), result_of(limit.trim(), &l, rounding, options.rounding.is_none())];

    // The conclusion, only on a supplied, versioned standard.
    let criterion = inputs.iter().find(|i| limit.trim() == i.id);
    let value_resolved = !prepared.provisional && literals.is_empty();
    match criterion {
        Some(input) if input.status() == InputStatus::Sourced && input.standard.as_deref().is_some_and(has_edition) && value_resolved => {
            let standard = input.standard.clone().unwrap_or_default();
            let statement = match verdict {
                Verdict::Holds => format!("{} {relation} {} holds: the criterion of {standard} is met", v.written(), l.written()),
                Verdict::Fails => format!("{} {relation} {} fails: the criterion of {standard} is not met", v.written(), l.written()),
                Verdict::Indeterminate => format!("{} {relation} {} cannot be decided within the stated uncertainty; no conclusion against {standard} is drawn", v.written(), l.written()),
            };
            record.conclusion = Some(Conclusion { standard, criterion: format!("{} [{}]", input.id, input.source.describe()), statement });
        }
        Some(input) if input.standard.as_deref().is_some_and(|s| !has_edition(s)) => {
            record.assumptions.push(format!("the standard for {} is named without an edition, so no conclusion is drawn against it", input.id));
        }
        _ => {}
    }
    record.status = if prepared.provisional || !literals.is_empty() { RecordStatus::Provisional } else { RecordStatus::Computed };
    record
}

// ── sensitivity ──────────────────────────────────────────────────────────

/// How the result moves with each input, near the stated values.
pub fn sensitivity(expression: Option<&str>, calculation: Option<&str>, inputs: Vec<CalcInput>, options: &Options, resolver: Option<Resolver>) -> CalcRecord {
    let rounding = options.rounding.unwrap_or(Rounding::DEFAULT);
    let (expression, mut inputs, result_unit) = match (expression.map(str::trim).filter(|e| !e.is_empty()), calculation) {
        (Some(expression), None) => (expression.to_string(), inputs, options.result_unit.clone()),
        (None, Some(id)) => {
            let id = id.trim().trim_start_matches("C:");
            match resolver.and_then(|find| find(id)) {
                Some(found) if found.operation == Operation::Evaluate => {
                    let result_unit = found.options.get("resultUnit").and_then(|u| u.as_str()).map(str::to_string);
                    (found.equation.clone(), found.inputs.clone(), result_unit)
                }
                Some(found) => {
                    let record = base(Operation::Sensitivity, id, &[], json!({}), rounding);
                    return refuse(record, CalcError::new(ErrorCode::Unsupported, format!("{id} is a {} record; sensitivity is computed for evaluations", found.operation.as_str())));
                }
                None => {
                    let record = base(Operation::Sensitivity, id, &[], json!({}), rounding);
                    return refuse(record, CalcError::new(ErrorCode::InvalidInput, format!("{id} is not a calculation this reader can open")));
                }
            }
        }
        _ => {
            let record = base(Operation::Sensitivity, expression.unwrap_or_default(), &inputs, json!({}), rounding);
            return refuse(record, CalcError::new(ErrorCode::InvalidInput, "give either an expression with its inputs, or a calculation id — one of the two"));
        }
    };
    let prepared = prepare(&mut inputs, resolver);
    let mut record = base(
        Operation::Sensitivity,
        &expression,
        &inputs,
        json!({ "resultUnit": result_unit, "round": options.rounding, "from": calculation }),
        rounding,
    );
    record.tolerance = Tolerance {
        relative: Some(SENSITIVITY_STEP),
        absolute: None,
        basis: format!("central differences with a step of {SENSITIVITY_STEP:e} of each input (at least {SENSITIVITY_STEP:e} of one unit); truncation error of order step²; the derivatives describe the result near the stated values only"),
    };
    let prepared = match prepared {
        Ok(p) => p,
        Err(error) => return refuse(record, error),
    };
    apply(&mut record, &prepared);
    let body = match expr::split_equation(&expression) {
        Some((lhs, rhs)) if is_identifier(lhs.trim()) => rhs.to_string(),
        _ => expression.clone(),
    };
    let node = match expr::parse(&body) {
        Ok(node) => node,
        Err(problem) => return refuse(record, parse_error(problem)),
    };
    let known: Vec<String> = inputs.iter().map(|i| i.id.clone()).collect();
    if let Err(error) = check_symbols(&[&node], &known, &inputs, &mut record) {
        return refuse(record, error);
    }
    if !prepared.missing.is_empty() {
        record.status = RecordStatus::Unresolved;
        record.unresolved.extend(prepared.missing.iter().cloned());
        record.error = Some(CalcError::new(ErrorCode::MissingInput, format!("no sensitivity without {}", prepared.missing.join(", "))));
        return record;
    }
    let base_value = match compute(&node, &prepared.env, result_unit.as_deref()) {
        Ok(q) => q,
        Err(error) => return refuse(record, error),
    };
    let result_label = base_value.label();
    let mut rows = Vec::new();
    for input in &inputs {
        let Some(held) = prepared.env.get(&input.id) else { continue };
        if !node.symbols().contains(&input.id) {
            continue;
        }
        if held.kind == QKind::Date {
            rows.push(SensitivityRow { input: input.id.clone(), derivative: None, elasticity: None, swing: None, note: Some("a date is not varied".into()) });
            continue;
        }
        match derivative(&node, &prepared.env, &input.id, result_unit.as_deref()) {
            Ok(d) => {
                let x_si = held.si();
                let elasticity = (base_value.si() != 0.0).then(|| d.si * x_si / base_value.si()).filter(|e| e.is_finite());
                let per = format!(
                    "{}{}",
                    if result_label.is_empty() { "" } else { result_label.as_str() },
                    if held.label().is_empty() { String::new() } else { format!(" per {}", held.label()) }
                );
                let swing_at = |value: f64| -> Option<String> {
                    let mut probe = prepared.env.clone();
                    probe.insert(input.id.clone(), Quantity { value, text: None, ..held.clone() });
                    compute(&node, &probe, result_unit.as_deref()).ok().map(|q| result_of("", &q, rounding, options.rounding.is_none()).display)
                };
                let swing = match input.absolute_uncertainty() {
                    Ok(Some(u)) if u > 0.0 => {
                        let (low, high) = (swing_at(held.value - u), swing_at(held.value + u));
                        Some(format!(
                            "at {} {}: {}; at +{}: {}",
                            input.id,
                            input.uncertainty.as_ref().map(|u| u.describe()).unwrap_or_default().replacen('±', "-", 1),
                            low.unwrap_or_else(|| "undefined".into()),
                            input.uncertainty.as_ref().map(|u| u.describe()).unwrap_or_default().trim_start_matches('±'),
                            high.unwrap_or_else(|| "undefined".into())
                        ))
                    }
                    _ if held.value != 0.0 => {
                        let (low, high) = (swing_at(held.value * 0.99), swing_at(held.value * 1.01));
                        Some(format!("±1% probe: {} to {}", low.unwrap_or_else(|| "undefined".into()), high.unwrap_or_else(|| "undefined".into())))
                    }
                    _ => None,
                };
                rows.push(SensitivityRow {
                    input: input.id.clone(),
                    derivative: Some(format!("{} {}", decimal::display(d.display, Rounding::Significant(4), true).unwrap_or_default(), per).trim().to_string()),
                    elasticity,
                    swing,
                    note: None,
                });
            }
            Err(error) => rows.push(SensitivityRow {
                input: input.id.clone(),
                derivative: None,
                elasticity: None,
                swing: None,
                note: Some(format!("could not be probed: {}", error.describe())),
            }),
        }
    }
    rows.sort_by(|a, b| {
        b.elasticity
            .map(f64::abs)
            .unwrap_or(-1.0)
            .total_cmp(&a.elasticity.map(f64::abs).unwrap_or(-1.0))
            .then_with(|| a.input.cmp(&b.input))
    });
    record.sensitivity = rows;
    let uncertainty = match propagate(&node, &prepared.env, &inputs, result_unit.as_deref()) {
        Ok(u) => u,
        Err(error) => return refuse(record, error),
    };
    if uncertainty.is_some() {
        record.assumptions.push(PROPAGATION.into());
    }
    record.results = vec![with_uncertainty(result_of("result", &base_value, rounding, options.rounding.is_none()), uncertainty)];
    let mut literals = Vec::new();
    literal_values(&node, &mut literals);
    record.status = if prepared.provisional || !literals.is_empty() { RecordStatus::Provisional } else { RecordStatus::Computed };
    record
}

/// The dimension of a unit, for tests and reports.
pub fn dim_of(unit: &str) -> Option<units::Dim> {
    parse_unit(unit).ok().map(|u| u.dim())
}

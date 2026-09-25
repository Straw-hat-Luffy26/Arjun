//! A calculation's inputs, typed: the value as written, its unit, where it
//! came from, how sure anyone is of it, and what was assumed.
//!
//! A model passes each input as one line, because a nested object per input
//! costs a small context window more than the arithmetic is worth:
//!
//! ```text
//! t_meas = 8.20 mm ±0.05 mm [M:mi-7#r2] [note: governing reading, point C]
//! t_min  = 9.0 mm [E3]
//! p_atm  = 101.325 kPa [assumed: standard atmosphere; site elevation not given]
//! t_req  = 6.0 mm [standard: ASME B31.3-2022 §304.1.2] [S:ab12cd34@p41]
//! t_cor  = ? mm
//! ```
//!
//! Brackets: `[M:item@rev]` or `[M:item#rN]` a shared-memory item;
//! `[E3]` this run's evidence marker; `[ev:<chunk>]` a knowledge passage;
//! `[S:<sha256>@<locator>]` source bytes; `[C:calc-…]` another calculation's
//! result; `[user]` stated by the person asking; `[assumed: why]`;
//! `[standard: name edition clause]`; `[note: …]`. `?` is a value nobody has.
//!
//! An input with no source is **unresolved**, and so is one with no value.
//! Neither is quietly treated as known: an unsourced value can still be
//! computed with, but the result is provisional and says which input made it
//! so; a missing value produces no number at all.

use serde::{Deserialize, Serialize};

use super::decimal::Decimal;
use super::error::{CalcError, ErrorCode};
use super::eval::{QKind, Quantity};
use super::expr::{days_from_civil, Func};
use super::units::parse_unit;

pub const MAX_INPUTS: usize = 24;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum InputSource {
    /// A shared-memory item, at the revision it was read.
    Memory { item_id: String, revision: Option<u64> },
    /// `E3` (this run's evidence table) or `ev:<chunk>` (a knowledge passage).
    Evidence { handle: String },
    Document { sha256: String, locator: Option<String> },
    /// Another calculation's first result.
    Calculation { calculation_id: String },
    /// Stated by the person who asked.
    Requester,
    Assumption { reason: String },
    Unsourced,
}

impl InputSource {
    pub fn describe(&self) -> String {
        match self {
            InputSource::Memory { item_id, revision: Some(r) } => format!("M:{item_id}@{r}"),
            InputSource::Memory { item_id, revision: None } => format!("M:{item_id}"),
            InputSource::Evidence { handle } => handle.clone(),
            InputSource::Document { sha256, locator: Some(l) } => format!("S:{}@{l}", &sha256[..sha256.len().min(12)]),
            InputSource::Document { sha256, locator: None } => format!("S:{}", &sha256[..sha256.len().min(12)]),
            InputSource::Calculation { calculation_id } => format!("C:{calculation_id}"),
            InputSource::Requester => "stated by the requester".into(),
            InputSource::Assumption { reason } => format!("assumed: {reason}"),
            InputSource::Unsourced => "no source".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum Uncertainty {
    /// ± this much, in `unit` (the input's own unit when empty).
    Absolute { value: String, unit: String },
    /// ± this percentage of the value.
    Relative { percent: String },
}

impl Uncertainty {
    pub fn describe(&self) -> String {
        match self {
            Uncertainty::Absolute { value, unit } if unit.is_empty() => format!("±{value}"),
            Uncertainty::Absolute { value, unit } => format!("±{value} {unit}"),
            Uncertainty::Relative { percent } => format!("±{percent}%"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum InputStatus {
    Sourced,
    /// Explicitly assumed, with a reason.
    Assumed,
    /// A value with nothing behind it.
    Unsourced,
    /// No value at all.
    Missing,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CalcInput {
    /// The symbol the expression uses.
    pub id: String,
    /// Exactly as written (`8.20`, `2026-08-12`), or `None` for `?`.
    pub value: Option<String>,
    #[serde(default)]
    pub unit: String,
    pub source: InputSource,
    /// The standard, edition and clause a value was taken from, as supplied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub standard: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncertainty: Option<Uncertainty>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assumptions: Vec<String>,
}

impl CalcInput {
    pub fn status(&self) -> InputStatus {
        if self.value.is_none() {
            return InputStatus::Missing;
        }
        match self.source {
            InputSource::Unsourced => InputStatus::Unsourced,
            InputSource::Assumption { .. } => InputStatus::Assumed,
            _ => InputStatus::Sourced,
        }
    }

    /// `8.20 mm`, or `? mm`.
    pub fn written(&self) -> String {
        let value = self.value.as_deref().unwrap_or("?");
        if self.unit.is_empty() {
            value.to_string()
        } else {
            format!("{value} {}", self.unit)
        }
    }

    /// The whole line back, as a model would write it.
    pub fn line(&self) -> String {
        let mut out = format!("{} = {}", self.id, self.written());
        if let Some(u) = &self.uncertainty {
            out.push_str(&format!(" {}", u.describe()));
        }
        match &self.source {
            InputSource::Unsourced => {}
            InputSource::Requester => out.push_str(" [user]"),
            InputSource::Assumption { reason } => out.push_str(&format!(" [assumed: {reason}]")),
            other => out.push_str(&format!(" [{}]", other.describe())),
        }
        if let Some(standard) = &self.standard {
            out.push_str(&format!(" [standard: {standard}]"));
        }
        for note in &self.assumptions {
            out.push_str(&format!(" [note: {note}]"));
        }
        out
    }

    /// The value as a quantity, or why it is not one.
    pub fn quantity(&self) -> Result<Quantity, CalcError> {
        let invalid = |message: String| CalcError::new(ErrorCode::InvalidInput, format!("input {}: {message}", self.id));
        let Some(text) = self.value.as_deref() else {
            return Err(CalcError::new(ErrorCode::MissingInput, format!("input {} has no value (it was given as ?).", self.id)));
        };
        if let Some(day) = parse_date(text) {
            if !self.unit.is_empty() {
                return Err(invalid("a date takes no unit".into()));
            }
            return Ok(Quantity { value: day as f64, units: Default::default(), kind: QKind::Date, text: Some(text.to_string()) });
        }
        let decimal = Decimal::parse(text).ok_or_else(|| invalid(format!("{text:?} is not a number written out")))?;
        let unit = parse_unit(&self.unit).map_err(|problem| {
            let code = match problem {
                super::units::UnitProblem::Ambiguous { .. } => ErrorCode::AmbiguousUnit,
                _ => ErrorCode::UnknownUnit,
            };
            CalcError::new(code, format!("input {}: {}", self.id, problem.explain()))
        })?;
        Ok(Quantity { value: decimal.to_f64(), units: unit.as_map(), kind: unit.kind().into(), text: Some(self.written()) })
    }

    /// ± in the input's own unit, as a magnitude.
    pub fn absolute_uncertainty(&self) -> Result<Option<f64>, CalcError> {
        let Some(uncertainty) = &self.uncertainty else { return Ok(None) };
        let invalid = |message: String| CalcError::new(ErrorCode::InvalidInput, format!("input {}: {message}", self.id));
        let own = self.quantity()?;
        if own.kind == QKind::Date {
            return Err(invalid("an uncertainty on a date is not supported; give the interval's uncertainty in days instead".into()));
        }
        let magnitude = match uncertainty {
            Uncertainty::Relative { percent } => {
                let p = Decimal::parse(percent).map(|d| d.to_f64()).ok_or_else(|| invalid(format!("{percent:?} is not a percentage")))?;
                own.value.abs() * p / 100.0
            }
            Uncertainty::Absolute { value, unit } => {
                let u = Decimal::parse(value).map(|d| d.to_f64()).ok_or_else(|| invalid(format!("{value:?} is not a number")))?;
                if unit.is_empty() || unit == &self.unit {
                    u
                } else {
                    let unit_expr = parse_unit(unit).map_err(|p| invalid(p.explain()))?;
                    if unit_expr.dim() != own.dim() {
                        return Err(CalcError::new(
                            ErrorCode::DimensionMismatch,
                            format!("input {}: its uncertainty is in {unit}, which is not the same kind of quantity as {}", self.id, self.unit),
                        ));
                    }
                    // A difference: factors only, never offsets.
                    u * unit_expr.factor() / own.units.factor()
                }
            }
        };
        if !magnitude.is_finite() || magnitude < 0.0 {
            return Err(invalid("an uncertainty must be a non-negative number".into()));
        }
        Ok(Some(magnitude))
    }

    /// One line of the input grammar (see the module header).
    pub fn parse_line(line: &str) -> Result<CalcInput, CalcError> {
        let invalid = |message: String| CalcError::new(ErrorCode::InvalidInput, format!("{line:?}: {message}"));
        let (id, rest) = line.split_once('=').ok_or_else(|| invalid("an input is written `name = value unit [source]`".into()))?;
        let id = id.trim().to_string();
        let valid_id = id.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            && id.len() <= 32;
        if !valid_id {
            return Err(invalid(format!("{id:?} is not a usable name: letters, digits and _ only, starting with a letter")));
        }
        if Func::ALL.iter().any(|(name, _)| *name == id) {
            return Err(invalid(format!("{id:?} is the name of a function")));
        }

        // Brackets first: everything from the first `[` on.
        let (body, brackets) = match rest.find('[') {
            Some(at) => (&rest[..at], &rest[at..]),
            None => (rest, ""),
        };
        let mut source: Option<InputSource> = None;
        let mut standard = None;
        let mut assumptions = Vec::new();
        let mut remaining = brackets.trim();
        while !remaining.is_empty() {
            let Some(inner_end) = remaining.find(']') else {
                return Err(invalid("a [ is never closed".into()));
            };
            if !remaining.starts_with('[') {
                return Err(invalid(format!("unexpected {remaining:?} between sources")));
            }
            let inner = remaining[1..inner_end].trim();
            remaining = remaining[inner_end + 1..].trim_start();
            let lower = inner.to_ascii_lowercase();
            let set = |source: &mut Option<InputSource>, value: InputSource| -> Result<(), CalcError> {
                if source.is_some() {
                    return Err(invalid("an input has one source; put the others in [note: …]".into()));
                }
                *source = Some(value);
                Ok(())
            };
            if let Some(text) = strip_label(inner, &["standard"]) {
                if text.is_empty() {
                    return Err(invalid("[standard: …] names nothing".into()));
                }
                standard = Some(text.to_string());
            } else if let Some(text) = strip_label(inner, &["note", "notes"]) {
                assumptions.push(text.to_string());
            } else if let Some(text) = strip_label(inner, &["assumed", "assumption", "assume"]) {
                if text.is_empty() {
                    return Err(invalid("an assumption says why: [assumed: …]".into()));
                }
                set(&mut source, InputSource::Assumption { reason: text.to_string() })?;
            } else if matches!(lower.as_str(), "user" | "requester" | "stated" | "given") {
                set(&mut source, InputSource::Requester)?;
            } else {
                set(&mut source, parse_reference(inner).ok_or_else(|| invalid(format!("[{inner}] is not a source this engine reads")))?)?;
            }
        }

        // The value, its unit and any uncertainty.
        let body = body.trim().replace("+/-", "±").replace("+-", "±");
        let (value_part, uncertainty_part) = match body.split_once('±') {
            Some((v, u)) => (v.trim().to_string(), Some(u.trim().to_string())),
            None => (body.clone(), None),
        };
        let mut words = value_part.splitn(2, char::is_whitespace);
        let value_text = words.next().unwrap_or_default().trim().to_string();
        let unit = words.next().unwrap_or_default().trim().to_string();
        if value_text.is_empty() {
            return Err(invalid("no value; write ? for one nobody has".into()));
        }
        let value = if value_text == "?" { None } else { Some(value_text) };
        let uncertainty = match uncertainty_part {
            None => None,
            Some(text) if text.ends_with('%') => Some(Uncertainty::Relative { percent: text.trim_end_matches('%').trim().to_string() }),
            Some(text) => {
                let mut parts = text.splitn(2, char::is_whitespace);
                Some(Uncertainty::Absolute {
                    value: parts.next().unwrap_or_default().to_string(),
                    unit: parts.next().unwrap_or_default().trim().to_string(),
                })
            }
        };
        let input = CalcInput {
            id,
            value,
            unit,
            source: source.unwrap_or(InputSource::Unsourced),
            standard,
            uncertainty,
            assumptions,
        };
        input.validate()?;
        Ok(input)
    }

    /// Everything that can be checked without evaluating anything.
    pub fn validate(&self) -> Result<(), CalcError> {
        if self.value.is_some() {
            self.quantity()?;
            self.absolute_uncertainty()?;
        } else if !self.unit.is_empty() {
            parse_unit(&self.unit).map_err(|p| CalcError::new(ErrorCode::UnknownUnit, format!("input {}: {}", self.id, p.explain())))?;
        }
        Ok(())
    }
}

fn strip_label<'a>(inner: &'a str, labels: &[&str]) -> Option<&'a str> {
    let (label, rest) = inner.split_once(':')?;
    labels.contains(&label.trim().to_ascii_lowercase().as_str()).then(|| rest.trim())
}

/// `M:item@3`, `M:item#r3`, `mi-…#r3`, `E3`, `ev:123`, `S:sha@p4`, `C:calc-…`.
pub fn parse_reference(inner: &str) -> Option<InputSource> {
    let inner = inner.trim();
    if let Some(number) = inner.strip_prefix('E') {
        if number.parse::<u32>().is_ok_and(|n| n > 0) {
            return Some(InputSource::Evidence { handle: inner.to_string() });
        }
    }
    if let Some(chunk) = inner.strip_prefix("ev:") {
        return (!chunk.is_empty()).then(|| InputSource::Evidence { handle: inner.to_string() });
    }
    if inner.starts_with("calc-") {
        return Some(InputSource::Calculation { calculation_id: inner.to_string() });
    }
    if inner.starts_with("mi-") {
        return memory(inner);
    }
    let (kind, value) = inner.split_once(':')?;
    match kind {
        "M" => memory(value),
        "C" => value.starts_with("calc-").then(|| InputSource::Calculation { calculation_id: value.to_string() }),
        "S" => {
            let (sha, locator) = match value.split_once('@') {
                Some((sha, locator)) => (sha, Some(locator.to_string())),
                None => (value, None),
            };
            (sha.len() >= 8 && sha.chars().all(|c| c.is_ascii_hexdigit()))
                .then(|| InputSource::Document { sha256: sha.to_ascii_lowercase(), locator })
        }
        _ => None,
    }
}

fn memory(value: &str) -> Option<InputSource> {
    let (id, revision) = if let Some((id, rev)) = value.rsplit_once("#r") {
        (id, Some(rev.parse::<u64>().ok()?))
    } else if let Some((id, rev)) = value.rsplit_once('@') {
        (id, Some(rev.parse::<u64>().ok()?))
    } else {
        (value, None)
    };
    (!id.is_empty()).then(|| InputSource::Memory { item_id: id.to_string(), revision })
}

fn parse_date(text: &str) -> Option<i64> {
    let bytes = text.as_bytes();
    if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return None;
    }
    days_from_civil(text[..4].parse().ok()?, text[5..7].parse().ok()?, text[8..10].parse().ok()?)
}

/// Reads a tool's `inputs`: each a line of the grammar, or (from Rust
/// callers) the record itself. Ids must be unique and at most
/// [`MAX_INPUTS`].
pub fn read_inputs(values: &[serde_json::Value]) -> Result<Vec<CalcInput>, CalcError> {
    if values.len() > MAX_INPUTS {
        return Err(CalcError::new(ErrorCode::Bounds, format!("at most {MAX_INPUTS} inputs per calculation; split it into steps")));
    }
    let mut out: Vec<CalcInput> = Vec::new();
    for value in values {
        let input = match value {
            serde_json::Value::String(line) => CalcInput::parse_line(line)?,
            other => {
                let input: CalcInput = serde_json::from_value(other.clone())
                    .map_err(|e| CalcError::new(ErrorCode::InvalidInput, format!("an input could not be read: {e}")))?;
                input.validate()?;
                input
            }
        };
        if out.iter().any(|held| held.id == input.id) {
            return Err(CalcError::new(ErrorCode::InvalidInput, format!("the input {} is defined twice", input.id)));
        }
        out.push(input);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_line_reads_into_a_typed_input() {
        let input = CalcInput::parse_line("t_meas = 8.20 mm ±0.05 mm [M:mi-7#r2] [note: governing reading, point C]").unwrap();
        assert_eq!(input.value.as_deref(), Some("8.20"));
        assert_eq!(input.unit, "mm");
        assert_eq!(input.source, InputSource::Memory { item_id: "mi-7".into(), revision: Some(2) });
        assert_eq!(input.uncertainty, Some(Uncertainty::Absolute { value: "0.05".into(), unit: "mm".into() }));
        assert_eq!(input.assumptions, vec!["governing reading, point C"]);
        assert_eq!(input.status(), InputStatus::Sourced);
        assert_eq!(input.quantity().unwrap().written(), "8.20 mm");
        assert_eq!(input.absolute_uncertainty().unwrap(), Some(0.05));
    }

    #[test]
    fn unsourced_assumed_and_missing_inputs_say_so() {
        assert_eq!(CalcInput::parse_line("x = 3 m").unwrap().status(), InputStatus::Unsourced);
        assert_eq!(CalcInput::parse_line("x = 3 m [assumed: nominal]").unwrap().status(), InputStatus::Assumed);
        assert_eq!(CalcInput::parse_line("x = ? m").unwrap().status(), InputStatus::Missing);
        // A standard named without where its value was read is not a source.
        let limit = CalcInput::parse_line("t_req = 6.0 mm [standard: ASME B31.3-2022 §304.1.2]").unwrap();
        assert_eq!(limit.status(), InputStatus::Unsourced);
        assert_eq!(limit.standard.as_deref(), Some("ASME B31.3-2022 §304.1.2"));
    }

    #[test]
    fn every_source_form_is_read() {
        assert_eq!(parse_reference("E3"), Some(InputSource::Evidence { handle: "E3".into() }));
        assert_eq!(parse_reference("ev:42"), Some(InputSource::Evidence { handle: "ev:42".into() }));
        assert_eq!(parse_reference("M:mi-1@4"), Some(InputSource::Memory { item_id: "mi-1".into(), revision: Some(4) }));
        assert_eq!(parse_reference("C:calc-0123abcd"), Some(InputSource::Calculation { calculation_id: "calc-0123abcd".into() }));
        assert!(matches!(parse_reference("S:ABCDEF12@p4"), Some(InputSource::Document { .. })));
        assert_eq!(parse_reference("wikipedia"), None);
    }

    #[test]
    fn a_bad_input_is_refused_by_name() {
        assert_eq!(CalcInput::parse_line("x = 3 months [user]").unwrap_err().code, ErrorCode::AmbiguousUnit);
        assert_eq!(CalcInput::parse_line("x = 3 widgets").unwrap_err().code, ErrorCode::UnknownUnit);
        assert_eq!(CalcInput::parse_line("x = three m").unwrap_err().code, ErrorCode::InvalidInput);
        assert_eq!(CalcInput::parse_line("sqrt = 3").unwrap_err().code, ErrorCode::InvalidInput);
        assert_eq!(CalcInput::parse_line("x = 3 m [M:a] [E1]").unwrap_err().code, ErrorCode::InvalidInput);
        assert_eq!(CalcInput::parse_line("x = 3 m ±2 kg").unwrap_err().code, ErrorCode::DimensionMismatch);
        let twice = read_inputs(&[serde_json::json!("x = 1 m"), serde_json::json!("x = 2 m")]).unwrap_err();
        assert!(twice.message.contains("twice"));
    }

    #[test]
    fn a_relative_uncertainty_and_a_date_are_read() {
        let input = CalcInput::parse_line("q = 250 kW ±2% [user]").unwrap();
        assert!((input.absolute_uncertainty().unwrap().unwrap() - 5.0).abs() < 1e-12);
        let date = CalcInput::parse_line("inspected = 2026-08-12 [E1]").unwrap();
        assert_eq!(date.quantity().unwrap().kind, QKind::Date);
    }
}

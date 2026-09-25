//! Structured refusals. A calculation that cannot be done says which kind of
//! "cannot" it is, where, and why — and produces no number.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    /// The text is not an expression this engine reads.
    Parse,
    UnknownUnit,
    /// A unit that exists but means two things (`month`, `gal`, bare `C`).
    AmbiguousUnit,
    /// A symbol no input defines.
    UnknownSymbol,
    /// An input named with no value (`?`).
    MissingInput,
    DimensionMismatch,
    /// Absolute temperatures added, or a Celsius value scaled.
    AffineMisuse,
    /// A gauge pressure used where only an absolute one means anything.
    GaugeMisuse,
    /// A date used as a number, or moved by a part of a day.
    DateMisuse,
    DivisionByZero,
    /// Outside the function's domain: `sqrt(-1)`, `ln(0)`.
    Domain,
    /// The arithmetic overflowed or produced NaN.
    NonFinite,
    /// A linear system with no unique solution.
    Singular,
    /// A linear system whose answer the data cannot pin down.
    IllConditioned,
    /// A bracket in which the function does not change sign.
    NoSignChange,
    NotConverged,
    /// An equation that is not linear in its unknowns, with no bracket given.
    Nonlinear,
    /// Outside the documented operation families.
    Unsupported,
    InvalidInput,
    /// Longer, deeper or wider than the engine evaluates.
    Bounds,
}

impl ErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::Parse => "parse",
            ErrorCode::UnknownUnit => "unknown_unit",
            ErrorCode::AmbiguousUnit => "ambiguous_unit",
            ErrorCode::UnknownSymbol => "unknown_symbol",
            ErrorCode::MissingInput => "missing_input",
            ErrorCode::DimensionMismatch => "dimension_mismatch",
            ErrorCode::AffineMisuse => "affine_misuse",
            ErrorCode::GaugeMisuse => "gauge_misuse",
            ErrorCode::DateMisuse => "date_misuse",
            ErrorCode::DivisionByZero => "division_by_zero",
            ErrorCode::Domain => "domain",
            ErrorCode::NonFinite => "non_finite",
            ErrorCode::Singular => "singular",
            ErrorCode::IllConditioned => "ill_conditioned",
            ErrorCode::NoSignChange => "no_sign_change",
            ErrorCode::NotConverged => "not_converged",
            ErrorCode::Nonlinear => "nonlinear",
            ErrorCode::Unsupported => "unsupported",
            ErrorCode::InvalidInput => "invalid_input",
            ErrorCode::Bounds => "bounds",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CalcError {
    pub code: ErrorCode,
    pub message: String,
    /// Character position in the expression, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<usize>,
}

impl CalcError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self { code, message: message.into(), at: None }
    }

    pub fn at(code: ErrorCode, at: usize, message: impl Into<String>) -> Self {
        Self { code, message: message.into(), at: Some(at) }
    }

    /// `division_by_zero: …`, the form a model reads.
    pub fn describe(&self) -> String {
        format!("{}: {}", self.code.as_str(), self.message)
    }
}

impl std::fmt::Display for CalcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.describe())
    }
}

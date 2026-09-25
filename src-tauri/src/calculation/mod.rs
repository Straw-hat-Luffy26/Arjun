//! The deterministic calculation engine (plan §5.7, P08).
//!
//! Every number a deliverable carries comes from here, never from a model's
//! arithmetic. A calculation is an immutable, content-addressed
//! [`CalcRecord`]: its typed inputs (value as written, unit, source,
//! uncertainty, assumptions), the equation, the substitutions, the working,
//! the engine and unit-table version, the raw and displayed results, the
//! rounding rule and the tolerance that applies. Five operation families are
//! supported, each with its tolerance and rounding stated in [`ops`]; anything
//! outside them is refused with a structured [`CalcError`], as are invalid
//! units, dimension mismatches, domain errors, singular systems and
//! non-finite results.
//!
//! [`check`] re-reads each input from its source and recomputes by two paths.
//! [`store`] keeps records append-only. Publication into the shared memory
//! graph, and the tools that expose all of this, live in the runtime
//! (`agent_runtime::calculation_tools`).

pub mod check;
pub mod decimal;
pub mod error;
pub mod eval;
pub mod expr;
pub mod inputs;
pub mod ops;
pub mod record;
pub mod store;
pub mod units;

pub use check::{check, CheckReport, CheckVerdict, SourceReader, SourceState};
pub use error::{CalcError, ErrorCode};
pub use inputs::{read_inputs, CalcInput, InputSource, InputStatus, Uncertainty};
pub use ops::{compare, evaluate, sensitivity, solve, validate_dimensions, Options};
pub use record::{CalcRecord, CalcResult, Operation, RecordStatus, Verdict};
pub use store::CalculationStore;

#[cfg(test)]
mod tests;

//! Known-answer tests for the operation families, and the P00 fixture pack's
//! six calculation cases run through the engine.
//!
//! The fixture cases read their inputs from
//! `fixtures/agent-system/v1/sources/calc/unit-cases.json` and their expected
//! answers from `…/expected/expected-answers.json`, so the numbers here are
//! the pack's, not this file's. What this file supplies is the formulation —
//! the SOP's definition of wall loss against the minimum allowable thickness
//! — which in a run is the model's job, and which P10's journey grades.

use serde_json::Value;

use super::*;
use crate::calculation::check::{check, CheckVerdict, SourceReader, SourceState};
use crate::calculation::decimal::Rounding;
use crate::calculation::record::Verdict;

fn pack(path: &str) -> Value {
    let full = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("fixtures/agent-system/v1").join(path);
    serde_json::from_str(&std::fs::read_to_string(&full).unwrap_or_else(|e| panic!("{}: {e}", full.display()))).unwrap()
}

fn case<'a>(cases: &'a Value, id: &str) -> &'a Value {
    cases["cases"].as_array().unwrap().iter().find(|c| c["id"] == id).unwrap_or_else(|| panic!("no case {id}"))
}

fn line(id: &str, quantity: &str, source: &str) -> CalcInput {
    CalcInput::parse_line(&format!("{id} = {quantity} [{source}]")).unwrap()
}

fn within(actual: f64, expected: &Value) -> bool {
    let target = expected["expected"].as_f64().unwrap();
    let tolerance = &expected["tolerance"];
    let allowed = match tolerance["kind"].as_str().unwrap() {
        "relative" => tolerance["value"].as_f64().unwrap() * target.abs(),
        _ => tolerance["value"].as_f64().unwrap(),
    };
    (actual - target).abs() <= allowed
}

fn percent() -> Options {
    Options { result_unit: Some("%".into()), rounding: None }
}

// ── The P00 fixture pack ────────────────────────────────────────────────

#[test]
fn pack_calc_01_wall_loss_against_the_minimum() {
    let (inputs, expected) = (pack("sources/calc/unit-cases.json"), pack("expected/expected-answers.json"));
    let given = &case(&inputs, "calc-01-wall-loss-mm")["inputs"];
    let record = evaluate(
        "wall_loss = (t_min - t_meas) / t_min",
        vec![line("t_min", given["minimum"].as_str().unwrap(), "user"), line("t_meas", given["measured"].as_str().unwrap(), "user")],
        &percent(),
        None,
    );
    println!("--- calc-01 ---\n{}", record.render());
    let result = record.first().unwrap();
    assert!(within(result.value, case(&expected, "calc-01-wall-loss-mm")), "{}", record.render());
    assert_eq!(result.display, "8.889 %");
    assert_eq!(record.status, RecordStatus::Computed);
}

#[test]
fn pack_calc_02_mixed_units_are_converted_not_subtracted_raw() {
    let (inputs, expected) = (pack("sources/calc/unit-cases.json"), pack("expected/expected-answers.json"));
    let given = &case(&inputs, "calc-02-mixed-units-inch")["inputs"];
    let record = evaluate(
        "(t_min - t_meas) / t_min",
        vec![line("t_min", given["minimum"].as_str().unwrap(), "user"), line("t_meas", given["measured"].as_str().unwrap(), "user")],
        &percent(),
        None,
    );
    println!("--- calc-02 ---\n{}", record.render());
    let result = record.first().unwrap();
    assert!(within(result.value, case(&expected, "calc-02-mixed-units-inch")), "{}", record.render());
    // The wrong answer the case exists to reject: 0.354 - 8.2 taken raw.
    assert!(result.value > 0.0 && result.value < 10.0);
}

#[test]
fn pack_calc_03_the_basis_decides_the_figure() {
    let (inputs, expected) = (pack("sources/calc/unit-cases.json"), pack("expected/expected-answers.json"));
    let given = &case(&inputs, "calc-03-percent-of-what")["inputs"];
    let nominal = evaluate(
        "(t_nom - t_meas) / t_nom",
        vec![line("t_nom", given["nominal"].as_str().unwrap(), "user"), line("t_meas", given["measured"].as_str().unwrap(), "user")],
        &Options { result_unit: Some("%".into()), rounding: Some(Rounding::Places(1)) },
        None,
    );
    // The claimed arithmetic is right: 31.7% against the nominal thickness.
    assert_eq!(nominal.first().unwrap().display, "31.7 %");
    // The SOP's basis is the minimum allowable thickness (9.0 mm, calc-01).
    let sop = evaluate(
        "(t_min - t_meas) / t_min",
        vec![line("t_min", "9.0 mm", "user"), line("t_meas", given["measured"].as_str().unwrap(), "user")],
        &Options { result_unit: Some("%".into()), rounding: Some(Rounding::Places(1)) },
        None,
    );
    let must_state = case(&expected, "calc-03-percent-of-what")["must_state"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect::<Vec<_>>();
    assert!(must_state.iter().any(|s| s.contains("8.9%")));
    println!("--- calc-03 (nominal basis) ---\n{}\n--- calc-03 (the SOP's basis) ---\n{}", nominal.render(), sop.render());
    assert_eq!(sop.first().unwrap().display, "8.9 %");
}

#[test]
fn pack_calc_04_ninety_days_is_not_three_months() {
    let (inputs, expected) = (pack("sources/calc/unit-cases.json"), pack("expected/expected-answers.json"));
    let given = &case(&inputs, "calc-04-replacement-window")["inputs"];
    let record = evaluate(
        "last_day = inspected + window",
        vec![line("inspected", given["inspection_date"].as_str().unwrap(), "user"), line("window", given["window"].as_str().unwrap(), "user")],
        &Options::default(),
        None,
    );
    println!("--- calc-04 ---\n{}", record.render());
    assert_eq!(record.first().unwrap().display, case(&expected, "calc-04-replacement-window")["expected"].as_str().unwrap());
    // Three months is refused as a duration rather than guessed at.
    assert_eq!(CalcInput::parse_line("window = 3 months [user]").unwrap_err().code, ErrorCode::AmbiguousUnit);
}

#[test]
fn pack_calc_05_the_rate_uses_the_actual_interval() {
    let (inputs, expected) = (pack("sources/calc/unit-cases.json"), pack("expected/expected-answers.json"));
    let given = &case(&inputs, "calc-05-corrosion-rate")["inputs"];
    let record = evaluate(
        "rate = (t_from - t_to) / (d_to - d_from)",
        vec![
            line("t_from", given["from"]["thickness"].as_str().unwrap(), "user"),
            line("t_to", given["to"]["thickness"].as_str().unwrap(), "user"),
            line("d_from", given["from"]["date"].as_str().unwrap(), "user"),
            line("d_to", given["to"]["date"].as_str().unwrap(), "user"),
        ],
        &Options { result_unit: Some("mm/a".into()), rounding: None },
        None,
    );
    println!("--- calc-05 ---\n{}", record.render());
    let result = record.first().unwrap();
    assert!(within(result.value, case(&expected, "calc-05-corrosion-rate")), "{}", record.render());
    assert!(record.assumptions.iter().any(|a| a.contains("365.25")), "the year convention is stated: {:?}", record.assumptions);
}

#[test]
fn pack_calc_06_a_zero_minimum_is_refused_with_no_number() {
    let (inputs, expected) = (pack("sources/calc/unit-cases.json"), pack("expected/expected-answers.json"));
    let given = &case(&inputs, "calc-06-division-by-zero")["inputs"];
    let record = evaluate(
        "(t_min - t_meas) / t_min",
        vec![line("t_min", given["minimum"].as_str().unwrap(), "user"), line("t_meas", given["measured"].as_str().unwrap(), "user")],
        &percent(),
        None,
    );
    println!("--- calc-06 ---\n{}", record.render());
    assert_eq!(record.status, RecordStatus::Refused);
    assert!(record.results.is_empty());
    let error = record.error.as_ref().unwrap();
    assert_eq!(error.code, ErrorCode::DivisionByZero);
    assert!(error.message.contains("not usable"), "{}", error.message);
    let rendered = record.render();
    for forbidden in case(&expected, "calc-06-division-by-zero")["must_not_produce"].as_array().unwrap() {
        let forbidden = forbidden.as_str().unwrap();
        if matches!(forbidden, "Infinity" | "NaN") {
            assert!(!rendered.contains(forbidden), "{rendered}");
        }
    }
    assert!(!rendered.contains("Result"), "no result line: {rendered}");
}

// ── Units and dimensions ─────────────────────────────────────────────────

#[test]
fn unit_conversions_are_exact_to_the_table() {
    let convert = |expression: &str, unit: &str| evaluate(expression, vec![], &Options { result_unit: Some(unit.into()), rounding: Some(Rounding::Significant(9)) }, None);
    assert_eq!(convert("1 bar", "psi").first().unwrap().display, "14.5037738 psi");
    assert_eq!(convert("100 degC", "degF").first().unwrap().display, "212.000000 °F");
    assert_eq!(convert("0.354 in", "mm").first().unwrap().display, "8.99160000 mm");
    assert_eq!(convert("1 kWh", "MJ").first().unwrap().display, "3.60000000 MJ");
    assert_eq!(convert("20 degC", "K").first().unwrap().display, "293.150000 K");
    // A temperature difference is not a temperature.
    assert_eq!(convert("80 degC - 20 degC", "degF").error.unwrap().code, ErrorCode::AffineMisuse);
    assert_eq!(convert("80 degC - 20 degC", "delta_degF").first().unwrap().display, "108.000000 Δ°F");
}

#[test]
fn a_pressure_and_a_temperature_are_a_dimensional_mismatch_everywhere() {
    let inputs = vec![line("p", "10 bar", "user"), line("t", "300 K", "user")];
    let evaluated = evaluate("p + t", inputs.clone(), &Options::default(), None);
    assert_eq!(evaluated.status, RecordStatus::Refused);
    assert_eq!(evaluated.error.as_ref().unwrap().code, ErrorCode::DimensionMismatch);
    assert!(evaluated.error.as_ref().unwrap().message.contains("a pressure"));

    let checked = validate_dimensions("p = t * 2", inputs.clone(), None);
    assert_eq!(checked.status, RecordStatus::Refused);
    let report = checked.dimensions.unwrap();
    assert!(report.problems[0].message.contains("a pressure") && report.problems[0].message.contains("temperature"), "{:?}", report.problems);

    // And the shape check works with no values at all.
    let unvalued = vec![CalcInput::parse_line("p = ? bar").unwrap(), CalcInput::parse_line("v = ? m^3").unwrap(), CalcInput::parse_line("n = ? mol").unwrap(), CalcInput::parse_line("r = ? J/(mol·K)").unwrap(), CalcInput::parse_line("t = ? K").unwrap()];
    let ideal_gas = validate_dimensions("p * v = n * r * t", unvalued, Some("J"));
    assert!(ideal_gas.dimensions.as_ref().unwrap().problems.is_empty(), "{:?}", ideal_gas.dimensions);
    assert_eq!(ideal_gas.status, RecordStatus::Computed);
}

// ── Boundaries, domains, non-finite ──────────────────────────────────────

#[test]
fn zero_and_negative_boundary_inputs_are_computed_or_refused_never_clipped() {
    let rate = |value: &str| {
        evaluate(
            "remaining = (t_meas - t_min) / rate",
            vec![line("t_meas", "8.2 mm", "user"), line("t_min", "9.0 mm", "user"), line("rate", value, "user")],
            &Options { result_unit: Some("a".into()), rounding: None },
            None,
        )
    };
    // A zero rate has no remaining life: refused, not "infinite".
    assert_eq!(rate("0 mm/a").error.unwrap().code, ErrorCode::DivisionByZero);
    // Below the minimum the remaining life is negative, and said so.
    let negative = rate("0.58 mm/a");
    assert!(negative.first().unwrap().value < 0.0);
    assert_eq!(negative.first().unwrap().display, "-1.379 a");
    // And a negative rate is arithmetic, not an error: the check against a
    // criterion is what catches it.
    let compared = compare(
        "remaining",
        ">=",
        "zero",
        None,
        vec![line("remaining", "-1.379 a", "user"), line("zero", "0 a", "user")],
        &Options::default(),
        None,
    );
    assert_eq!(compared.comparison.unwrap().verdict, Verdict::Fails);
}

#[test]
fn an_invalid_mathematical_domain_is_a_structured_refusal() {
    for (expression, code) in [
        ("sqrt(x)", ErrorCode::Domain),
        ("ln(y)", ErrorCode::Domain),
        ("log10(x)", ErrorCode::Domain),
        ("tan(z)", ErrorCode::Domain),
        ("exp(w)", ErrorCode::NonFinite),
        ("ln(t)", ErrorCode::DimensionMismatch),
    ] {
        let record = evaluate(
            expression,
            vec![line("x", "-4", "user"), line("y", "0", "user"), line("z", "90 deg", "user"), line("w", "1000", "user"), line("t", "2 mm", "user")],
            &Options::default(),
            None,
        );
        assert_eq!(record.status, RecordStatus::Refused, "{expression}");
        assert_eq!(record.error.as_ref().unwrap().code, code, "{expression}: {:?}", record.error);
        assert!(record.results.is_empty());
    }
}

// ── Rounding and uncertainty ─────────────────────────────────────────────

#[test]
fn rounding_is_stated_and_the_raw_result_is_kept() {
    let record = evaluate("x", vec![line("x", "2.675", "user")], &Options { result_unit: None, rounding: Some(Rounding::Places(2)) }, None);
    assert_eq!(record.first().unwrap().display, "2.68");
    assert_eq!(record.first().unwrap().raw, "2.675");
    assert_eq!(record.rounding_rule, "2 decimal places, half away from zero");
    let kept = evaluate("x * 1", vec![line("x", "8.8", "user")], &Options { result_unit: None, rounding: Some(Rounding::Places(2)) }, None);
    assert_eq!(kept.first().unwrap().display, "8.80", "a stated rule keeps its zeros");
    let default = evaluate("1 / 3", vec![], &Options::default(), None);
    assert_eq!(default.first().unwrap().display, "0.3333");
    assert_eq!(default.first().unwrap().raw, "0.3333333333333333");
}

#[test]
fn an_uncertain_input_carries_its_uncertainty_to_the_result_and_to_the_verdict() {
    let inputs = || vec![line("t_min", "9.0 mm", "user"), CalcInput::parse_line("t_meas = 8.2 mm ±0.05 mm [user]").unwrap()];
    let record = evaluate("(t_min - t_meas) / t_min", inputs(), &percent(), None);
    let result = record.first().unwrap();
    // ∂/∂t_meas = -100/9.0 % per mm; × 0.05 mm = 0.5556 %.
    assert!((result.uncertainty.unwrap() - 0.555_555_555_6).abs() < 1e-6, "{:?}", result.uncertainty);
    assert_eq!(result.uncertainty_display.as_deref(), Some("±0.56 %"));
    assert!(record.assumptions.iter().any(|a| a.contains("first order")));

    // 8.2 ± 0.05 mm against 8.22 mm: the interval straddles the limit.
    let straddles = compare("t_meas", ">=", "limit", None, vec![CalcInput::parse_line("t_meas = 8.2 mm ±0.05 mm [user]").unwrap(), line("limit", "8.22 mm", "user")], &Options::default(), None);
    assert_eq!(straddles.comparison.unwrap().verdict, Verdict::Indeterminate);
    let clear = compare("t_meas", ">=", "limit", None, vec![CalcInput::parse_line("t_meas = 8.2 mm ±0.05 mm [user]").unwrap(), line("limit", "8.0 mm", "user")], &Options::default(), None);
    assert_eq!(clear.comparison.unwrap().verdict, Verdict::Holds);
}

#[test]
fn unsourced_assumed_and_missing_inputs_stay_explicit() {
    let unsourced = evaluate("a * 2", vec![CalcInput::parse_line("a = 3 m").unwrap()], &Options::default(), None);
    assert_eq!(unsourced.status, RecordStatus::Provisional);
    assert!(unsourced.unresolved[0].contains("no source"));
    assert!(unsourced.render().contains("PROVISIONAL"));
    let missing = evaluate("a * b", vec![line("a", "3 m", "user"), CalcInput::parse_line("b = ? m").unwrap()], &Options::default(), None);
    assert_eq!(missing.status, RecordStatus::Unresolved);
    assert!(missing.results.is_empty());
    let literal = evaluate("(9.0 mm - 8.2 mm) / 9.0 mm", vec![], &Options::default(), None);
    assert_eq!(literal.status, RecordStatus::Provisional, "numbers typed into an expression have no source");
}

// ── A changed source value ───────────────────────────────────────────────

struct Graph {
    min: SourceState,
}
impl SourceReader for Graph {
    fn memory(&self, item: &str, _: Option<u64>) -> SourceState {
        match item {
            "mi-min" => self.min.clone(),
            _ => SourceState::Current { revision: Some(1), text: "Governing reading at C: 8.2 mm".into() },
        }
    }
    fn evidence(&self, _: &str) -> SourceState {
        SourceState::Unreadable { why: "not in this test".into() }
    }
    fn document(&self, _: &str, _: Option<&str>) -> SourceState {
        SourceState::Unreadable { why: "not in this test".into() }
    }
    fn calculation(&self, _: &str) -> Option<(CalcRecord, Option<String>)> {
        None
    }
}

#[test]
fn a_changed_source_value_changes_the_record_id_and_makes_the_old_one_stale() {
    let inputs = |min: &str| vec![line("t_min", min, "M:mi-min@1"), line("t_meas", "8.2 mm", "M:mi-meas@1")];
    let original = evaluate("(t_min - t_meas) / t_min", inputs("9.0 mm"), &percent(), None);
    let again = evaluate("(t_min - t_meas) / t_min", inputs("9.0 mm"), &percent(), None);
    assert_eq!(original.id, again.id, "the same computation is the same record");
    let corrected = evaluate("(t_min - t_meas) / t_min", inputs("8.8 mm"), &percent(), None);
    assert_ne!(original.id, corrected.id, "a corrected input is a new record, not an edit");

    let current = check(&original, &Graph { min: SourceState::Current { revision: Some(1), text: "Minimum allowable thickness 9.0 mm".into() } });
    assert_eq!(current.verdict, CheckVerdict::Verified, "{}", current.render());

    let changed = check(
        &original,
        &Graph {
            min: SourceState::Changed {
                revision: Some(2),
                text: Some("Minimum allowable thickness corrected to 8.8 mm".into()),
                why: "superseded by a correction".into(),
                now: Some(InputSource::Memory { item_id: "mi-min".into(), revision: Some(2) }),
            },
        },
    );
    assert_eq!(changed.verdict, CheckVerdict::Stale);
    let redone = changed.corrected.expect("recomputed with the source's current value");
    let cited_now = evaluate(
        "(t_min - t_meas) / t_min",
        vec![line("t_min", "8.8 mm", "M:mi-min@2"), line("t_meas", "8.2 mm", "M:mi-meas@1")],
        &percent(),
        None,
    );
    assert_eq!(redone.id, cited_now.id, "the recomputation cites the source as it now stands");
    assert_ne!(redone.id, corrected.id, "…not the old revision with a new number");
    assert_eq!(redone.first().unwrap().display, "6.818 %");
}

// ── solve ────────────────────────────────────────────────────────────────

#[test]
fn a_linear_unknown_is_solved_exactly_and_checked() {
    // Barlow: P = 2 S t / D, for t.
    let record = solve(
        &["p = 2 * s * t / d".to_string()],
        &["t mm".to_string()],
        vec![line("p", "10 MPa", "user"), line("s", "138 MPa", "user"), line("d", "219.1 mm", "user")],
        &Options::default(),
        None,
    );
    assert_eq!(record.status, RecordStatus::Computed, "{}", record.render());
    assert_eq!(record.first().unwrap().display, "7.938 mm");
    assert_eq!(record.solver.as_ref().unwrap().family, "linear");
}

#[test]
fn a_linear_system_solves_and_a_singular_or_ill_conditioned_one_refuses() {
    let system = solve(
        &["2 * x + y = a".to_string(), "x - y = b".to_string()],
        &["x m".to_string(), "y m".to_string()],
        vec![line("a", "7 m", "user"), line("b", "-1 m", "user")],
        &Options::default(),
        None,
    );
    assert_eq!(system.results.iter().map(|r| r.display.clone()).collect::<Vec<_>>(), vec!["2 m", "3 m"], "{}", system.render());

    let singular = solve(
        &["x + y = a".to_string(), "2 * x + 2 * y = b".to_string()],
        &["x m".to_string(), "y m".to_string()],
        vec![line("a", "1 m", "user"), line("b", "3 m", "user")],
        &Options::default(),
        None,
    );
    assert_eq!(singular.error.unwrap().code, ErrorCode::Singular);

    let ill = solve(
        &["x + y = a".to_string(), "x + 1.00000000001 * y = b".to_string()],
        &["x m".to_string(), "y m".to_string()],
        vec![line("a", "1 m", "user"), line("b", "1 m", "user")],
        &Options::default(),
        None,
    );
    assert!(matches!(ill.error.as_ref().unwrap().code, ErrorCode::IllConditioned | ErrorCode::Singular), "{:?}", ill.error);
}

#[test]
fn a_nonlinear_unknown_needs_a_bracket_and_a_bracket_needs_a_sign_change() {
    let unbracketed = solve(&["x^3 = two".to_string()], &["x".to_string()], vec![line("two", "2", "user")], &Options::default(), None);
    assert_eq!(unbracketed.error.unwrap().code, ErrorCode::Nonlinear);
    let bracketed = solve(&["x^3 = two".to_string()], &["x in 0..2".to_string()], vec![line("two", "2", "user")], &Options { result_unit: None, rounding: Some(Rounding::Significant(7)) }, None);
    assert_eq!(bracketed.first().unwrap().display, "1.259921", "{}", bracketed.render());
    assert!(bracketed.solver.unwrap().family.contains("bisection"));
    let no_root = solve(&["x^2 = minus".to_string()], &["x in 0..2".to_string()], vec![line("minus", "-1", "user")], &Options::default(), None);
    assert_eq!(no_root.error.unwrap().code, ErrorCode::NoSignChange);
    let pole = solve(&["1 / x = zero".to_string()], &["x in -1..1".to_string()], vec![line("zero", "0", "user")], &Options::default(), None);
    assert!(matches!(pole.error.as_ref().unwrap().code, ErrorCode::NotConverged | ErrorCode::DivisionByZero), "{:?}", pole.error);
    let not_square = solve(&["x + y = a".to_string()], &["x m".to_string(), "y m".to_string()], vec![line("a", "1 m", "user")], &Options::default(), None);
    assert_eq!(not_square.error.unwrap().code, ErrorCode::Unsupported);
}

// ── compare and the engineering conclusion ───────────────────────────────

#[test]
fn a_conclusion_needs_a_sourced_limit_from_a_standard_with_its_edition() {
    let measured = || line("t_meas", "8.2 mm", "M:mi-meas@1");
    let with_edition = compare(
        "t_meas",
        ">=",
        "t_req",
        None,
        vec![measured(), CalcInput::parse_line("t_req = 6.0 mm [standard: ASME B31.3-2022 §304.1.2] [S:ab12cd34ef@p41]").unwrap()],
        &Options::default(),
        None,
    );
    let conclusion = with_edition.conclusion.as_ref().expect("a conclusion against a versioned, sourced standard");
    assert!(conclusion.statement.contains("is met") && conclusion.standard.contains("2022"));
    assert_eq!(with_edition.comparison.as_ref().unwrap().margin, "2.2 mm");

    let no_edition = compare("t_meas", ">=", "t_req", None, vec![measured(), CalcInput::parse_line("t_req = 6.0 mm [standard: ASME B31.3] [S:ab12cd34ef@p41]").unwrap()], &Options::default(), None);
    assert!(no_edition.conclusion.is_none());
    assert!(no_edition.assumptions.iter().any(|a| a.contains("without an edition")));

    let unsourced = compare("t_meas", ">=", "t_req", None, vec![measured(), CalcInput::parse_line("t_req = 6.0 mm [standard: ASME B31.3-2022 §304.1.2]").unwrap()], &Options::default(), None);
    assert!(unsourced.conclusion.is_none(), "a standard named without where its value was read is not a criterion");
    assert!(unsourced.render().contains("No engineering conclusion"));

    let literal = compare("t_meas", ">=", "6.0 mm", None, vec![measured()], &Options::default(), None);
    assert!(literal.conclusion.is_none());
    assert_eq!(literal.status, RecordStatus::Provisional);
}

#[test]
fn equality_needs_a_tolerance_and_strict_equality_at_the_limit_is_indeterminate() {
    let within = compare("a", "==", "b", Some("±0.1 mm"), vec![line("a", "8.25 mm", "user"), line("b", "8.2 mm", "user")], &Options::default(), None);
    assert_eq!(within.comparison.unwrap().verdict, Verdict::Holds);
    let outside = compare("a", "==", "b", Some("±0.01 mm"), vec![line("a", "8.25 mm", "user"), line("b", "8.2 mm", "user")], &Options::default(), None);
    assert_eq!(outside.comparison.unwrap().verdict, Verdict::Fails);
    let at_limit = compare("a", "<", "b", None, vec![line("a", "9.0 mm", "user"), line("b", "0.9 cm", "user")], &Options::default(), None);
    assert_eq!(at_limit.comparison.unwrap().verdict, Verdict::Indeterminate);
    let mismatch = compare("a", "<", "b", None, vec![line("a", "10 bar", "user"), line("b", "300 K", "user")], &Options::default(), None);
    assert_eq!(mismatch.error.unwrap().code, ErrorCode::DimensionMismatch);
}

// ── sensitivity ──────────────────────────────────────────────────────────

#[test]
fn sensitivity_ranks_inputs_by_elasticity_and_reports_swings() {
    let record = sensitivity(
        Some("(t_min - t_meas) / t_min"),
        None,
        vec![line("t_min", "9.0 mm", "user"), CalcInput::parse_line("t_meas = 8.2 mm ±0.05 mm [user]").unwrap()],
        &percent(),
        None,
    );
    assert_eq!(record.status, RecordStatus::Computed, "{}", record.render());
    // f = 1 - t_meas/t_min: elasticity wrt t_meas = -(t_meas/t_min)/f = -10.25.
    let meas = record.sensitivity.iter().find(|r| r.input == "t_meas").unwrap();
    assert!((meas.elasticity.unwrap() + 10.25).abs() < 1e-4, "{:?}", meas.elasticity);
    assert!(meas.swing.as_ref().unwrap().contains("8.333 %"), "{:?}", meas.swing);
    // Both inputs move the result equally hard, in opposite directions:
    // wrt t_min the elasticity is +(t_meas/t_min)/f = +10.25.
    let min = record.sensitivity.iter().find(|r| r.input == "t_min").unwrap();
    assert!((min.elasticity.unwrap() - 10.25).abs() < 1e-4, "{:?}", min.elasticity);
    let magnitudes: Vec<f64> = record.sensitivity.iter().map(|r| r.elasticity.unwrap().abs()).collect();
    assert!(magnitudes.windows(2).all(|w| w[0] >= w[1]), "ranked by |elasticity|: {magnitudes:?}");
    assert!(record.first().unwrap().uncertainty.is_some());
}

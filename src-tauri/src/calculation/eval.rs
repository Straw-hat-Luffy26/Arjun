//! Evaluating an expression tree, with the units checked at every step.
//!
//! Values are kept in the units they were written in, not converted to SI:
//! an engineer who wrote `mm` reads `mm` back, and `9.0 mm - 8.2 mm` does not
//! pick up conversion noise it never needed. Two different units of the same
//! dimension meet by conversion (`1 m + 1 mm` is `1.001 m`), and a quotient
//! whose units cancel is a plain number (`2 m / 500 mm` is `4`).
//!
//! [`infer`] walks the same tree with dimensions only — no values — which is
//! what `calculation.validate_dimensions` reports, and which works when some
//! inputs have no value yet.

use std::collections::BTreeMap;

use super::decimal;
use super::error::{CalcError, ErrorCode};
use super::expr::{civil_from_days, BinOp, Func, Node};
use super::units::{describe_dim, lookup, Dim, Kind, UnitDef, UnitMap, DIMENSIONLESS, PRESSURE, TEMPERATURE};

/// What kind of thing a value is, beyond its dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QKind {
    Linear,
    AbsoluteTemperature,
    Gauge,
    /// Days since 1970-01-01.
    Date,
}

impl From<Kind> for QKind {
    fn from(kind: Kind) -> Self {
        match kind {
            Kind::Linear => QKind::Linear,
            Kind::AbsoluteTemperature => QKind::AbsoluteTemperature,
            Kind::Gauge => QKind::Gauge,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Quantity {
    /// In `units`; for a date, days since 1970-01-01.
    pub value: f64,
    pub units: UnitMap,
    pub kind: QKind,
    /// The value exactly as a source wrote it, while it is still that value.
    pub text: Option<String>,
}

const TIME: Dim = [0, 0, 1, 0, 0, 0];

impl Quantity {
    pub fn plain(value: f64) -> Self {
        Self { value, units: UnitMap::default(), kind: QKind::Linear, text: None }
    }

    pub fn dim(&self) -> Dim {
        self.units.dim()
    }

    /// The single unit of a temperature or gauge value.
    fn scale(&self) -> Option<&'static UnitDef> {
        self.units.single()
    }

    /// In SI: kelvin for a temperature, pascals above atmosphere for a gauge
    /// pressure, days for a date.
    pub fn si(&self) -> f64 {
        match self.kind {
            QKind::Date => self.value,
            QKind::AbsoluteTemperature => {
                let unit = self.scale();
                (self.value + unit.map_or(0.0, |u| u.offset)) * unit.map_or(1.0, |u| u.factor)
            }
            QKind::Linear | QKind::Gauge => self.value * self.units.factor(),
        }
    }

    pub fn label(&self) -> String {
        self.units.label()
    }

    /// How a person reads it: `-0.8 mm`, `2026-11-10`, `293.15 K`.
    pub fn format(&self) -> String {
        if self.kind == QKind::Date {
            return civil_from_days(self.value.round() as i64);
        }
        let number = trim_number(self.value);
        let label = self.label();
        if label.is_empty() {
            number
        } else {
            format!("{number} {label}")
        }
    }

    /// The source text while the value is unchanged, else [`Self::format`].
    pub fn written(&self) -> String {
        self.text.clone().unwrap_or_else(|| self.format())
    }

    fn describe(&self) -> String {
        match self.kind {
            QKind::Date => "a date".to_string(),
            QKind::AbsoluteTemperature => format!("a temperature in {}", self.label()),
            QKind::Gauge => format!("a gauge pressure in {}", self.label()),
            QKind::Linear if self.units.is_empty() => "a plain number".to_string(),
            QKind::Linear => format!("a value in {}", self.label()),
        }
    }
}

/// A sum that cancels to within the engine's stated tolerance (relative
/// 1e-12 of the larger operand) is zero, not the rounding noise of binary
/// arithmetic: `68 °F - 20 °C` is `0 Δ°F`, not `1.02e-13 Δ°F`.
fn cancelled(result: f64, a: f64, b: f64) -> f64 {
    if result != 0.0 && result.abs() <= 1e-12 * a.abs().max(b.abs()) {
        0.0
    } else {
        result
    }
}

/// Ten decimals, no trailing zeros or floating-point noise; the shortest
/// round-trip form where ten decimals would lose the value.
pub fn trim_number(value: f64) -> String {
    if !value.is_finite() {
        return value.to_string();
    }
    if value != 0.0 && (value.abs() < 1e-6 || value.abs() >= 1e15) {
        return decimal::raw(value);
    }
    let mut text = format!("{value:.10}");
    while text.contains('.') && (text.ends_with('0') || text.ends_with('.')) {
        text.pop();
    }
    if text.is_empty() || text == "-" || text == "-0" {
        text = "0".to_string();
    }
    text
}

/// One step of the working.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Step {
    pub description: String,
    pub result: String,
}

/// Evaluates trees against a fixed set of named values.
pub struct Evaluator<'a> {
    env: &'a BTreeMap<String, Quantity>,
    record: bool,
    pub steps: Vec<Step>,
    /// Conversions the engine made on the reader's behalf, said aloud.
    pub notes: Vec<String>,
}

impl<'a> Evaluator<'a> {
    pub fn new(env: &'a BTreeMap<String, Quantity>) -> Self {
        Self { env, record: true, steps: Vec::new(), notes: Vec::new() }
    }

    /// The same, without keeping the working: for solving and probing.
    pub fn quiet(env: &'a BTreeMap<String, Quantity>) -> Self {
        Self { env, record: false, steps: Vec::new(), notes: Vec::new() }
    }

    pub fn eval(&mut self, node: &Node) -> Result<Quantity, CalcError> {
        let result = self.eval_inner(node)?;
        if !result.value.is_finite() {
            return Err(CalcError::at(
                ErrorCode::NonFinite,
                node.at(),
                "The result is not a finite number. Check for a division by something very small or an overflow.",
            ));
        }
        Ok(result)
    }

    fn eval_inner(&mut self, node: &Node) -> Result<Quantity, CalcError> {
        match node {
            Node::Number { text, value, unit, unit_text, .. } => Ok(Quantity {
                value: value.to_f64(),
                units: unit.as_map(),
                kind: unit.kind().into(),
                text: Some(if unit_text.is_empty() { text.clone() } else { format!("{text} {unit_text}") }),
            }),
            Node::Date { text, day, .. } => Ok(Quantity {
                value: *day as f64,
                units: UnitMap::default(),
                kind: QKind::Date,
                text: Some(text.clone()),
            }),
            Node::Symbol { name, at } => self.env.get(name).cloned().ok_or_else(|| {
                CalcError::at(
                    ErrorCode::UnknownSymbol,
                    *at,
                    format!("{name:?} is not one of the inputs. Define it in `inputs` (for example \"{name} = 9.0 mm [M:mi-…#r1]\") or write the value with its unit."),
                )
            }),
            Node::Neg { inner, at } => {
                let q = self.eval(inner)?;
                match q.kind {
                    QKind::Date => Err(CalcError::at(ErrorCode::DateMisuse, *at, "A date cannot be negated.")),
                    QKind::AbsoluteTemperature | QKind::Gauge if !matches!(**inner, Node::Number { .. }) => Err(CalcError::at(
                        ErrorCode::AffineMisuse,
                        *at,
                        format!("Negating {} has no single meaning; write the negative value itself.", q.describe()),
                    )),
                    _ => Ok(Quantity { value: -q.value, text: q.text.map(|t| format!("-{t}")), ..q }),
                }
            }
            Node::Binary { op, left, right, at } => {
                let l = self.eval(left)?;
                let r = self.eval(right)?;
                let result = match op {
                    BinOp::Add | BinOp::Sub => self.add(&l, &r, *op == BinOp::Sub, *at)?,
                    BinOp::Mul | BinOp::Div => self.mul(&l, &r, *op == BinOp::Div, *at)?,
                };
                self.step(format!("{} {} {}", l.written(), op.symbol(), r.written()), &result);
                Ok(result)
            }
            Node::Power { base, exponent, exponent_text, at } => {
                let b = self.eval(base)?;
                let result = self.power(&b, *exponent, *at, false)?;
                self.step(format!("({})^{exponent_text}", b.written()), &result);
                Ok(result)
            }
            Node::Call { func, args, at } => {
                let values = args.iter().map(|a| self.eval(a)).collect::<Result<Vec<_>, _>>()?;
                let result = self.call(*func, &values, *at)?;
                self.step(
                    format!("{}({})", func.name(), values.iter().map(Quantity::written).collect::<Vec<_>>().join(", ")),
                    &result,
                );
                Ok(result)
            }
        }
    }

    fn step(&mut self, description: String, result: &Quantity) {
        if self.record {
            self.steps.push(Step { description, result: result.format() });
        }
    }

    fn note(&mut self, text: String) {
        if !self.notes.contains(&text) {
            self.notes.push(text);
        }
    }

    fn add(&mut self, l: &Quantity, r: &Quantity, sub: bool, at: usize) -> Result<Quantity, CalcError> {
        let verb = if sub { "subtract" } else { "add" };
        let mismatch = || {
            CalcError::at(
                ErrorCode::DimensionMismatch,
                at,
                format!(
                    "Cannot {verb} {} and {}: the units do not match ({} against {}).",
                    l.describe(),
                    r.describe(),
                    describe_dim(l.dim()),
                    describe_dim(r.dim())
                ),
            )
        };
        let sign = if sub { -1.0 } else { 1.0 };
        let q = |value: f64, units: UnitMap, kind: QKind| Quantity { value, units, kind, text: None };
        match (l.kind, r.kind) {
            (QKind::Date, QKind::Date) => {
                if !sub {
                    return Err(CalcError::at(ErrorCode::DateMisuse, at, "Adding two dates has no meaning; subtract them for the days between."));
                }
                Ok(q(l.value - r.value, unit_map("d"), QKind::Linear))
            }
            (QKind::Date, _) | (_, QKind::Date) => {
                let (date, duration) = if l.kind == QKind::Date { (l, r) } else { (r, l) };
                if r.kind == QKind::Date && sub {
                    return Err(CalcError::at(ErrorCode::DateMisuse, at, "A duration minus a date has no meaning."));
                }
                if duration.kind != QKind::Linear || duration.dim() != TIME {
                    return Err(CalcError::at(
                        ErrorCode::DateMisuse,
                        at,
                        format!("A date moves by a duration in days (d), and {} is not one.", duration.describe()),
                    ));
                }
                let days = duration.si() / 86_400.0;
                if (days - days.round()).abs() > 1e-9 {
                    return Err(CalcError::at(
                        ErrorCode::DateMisuse,
                        at,
                        format!(
                            "{} is {} days, not a whole number of days, so the result is not a date. Give the interval in whole days.",
                            duration.written(),
                            trim_number(days)
                        ),
                    ));
                }
                Ok(q(date.value + sign * days.round(), UnitMap::default(), QKind::Date))
            }
            _ if l.dim() != r.dim() => Err(mismatch()),
            (QKind::AbsoluteTemperature, QKind::AbsoluteTemperature) => {
                if !sub {
                    return Err(CalcError::at(
                        ErrorCode::AffineMisuse,
                        at,
                        format!(
                            "Adding two temperatures ({} + {}) has no physical meaning. Subtract them for a difference, or add a difference written as delta_degC, delta_degF or delta_K.",
                            l.written(),
                            r.written()
                        ),
                    ));
                }
                let scale = l.scale().expect("an absolute temperature has one unit");
                let delta = match scale.symbol {
                    "degC" => "delta_degC",
                    "degF" => "delta_degF",
                    _ => "K",
                };
                Ok(q(cancelled(l.si() - r.si(), l.si(), r.si()) / scale.factor, unit_map(delta), QKind::Linear))
            }
            (QKind::AbsoluteTemperature, QKind::Linear) => {
                let scale = l.scale().expect("an absolute temperature has one unit");
                Ok(q(l.value + sign * r.si() / scale.factor, l.units.clone(), QKind::AbsoluteTemperature))
            }
            (QKind::Linear, QKind::AbsoluteTemperature) => {
                if sub {
                    return Err(CalcError::at(ErrorCode::AffineMisuse, at, "A temperature difference minus a temperature has no meaning."));
                }
                let scale = r.scale().expect("an absolute temperature has one unit");
                Ok(q(r.value + l.si() / scale.factor, r.units.clone(), QKind::AbsoluteTemperature))
            }
            (QKind::Gauge, QKind::Gauge) => {
                if !sub {
                    return Err(CalcError::at(
                        ErrorCode::GaugeMisuse,
                        at,
                        "Adding two gauge pressures counts the atmosphere twice. Subtract them for a difference, or make one absolute with absolute(p, p_atm).",
                    ));
                }
                let scale = l.scale().expect("a gauge pressure has one unit");
                Ok(q(cancelled(l.si() - r.si(), l.si(), r.si()) / scale.factor, unit_map(absolute_counterpart(scale.symbol)), QKind::Linear))
            }
            (QKind::Gauge, QKind::Linear) => {
                let scale = l.scale().expect("a gauge pressure has one unit");
                Ok(q(l.value + sign * r.si() / scale.factor, l.units.clone(), QKind::Gauge))
            }
            (QKind::Linear, QKind::Gauge) => {
                if sub {
                    return Err(CalcError::at(ErrorCode::GaugeMisuse, at, "A pressure minus a gauge pressure has no single meaning; make the gauge value absolute first with absolute(p, p_atm)."));
                }
                let scale = r.scale().expect("a gauge pressure has one unit");
                Ok(q(r.value + l.si() / scale.factor, r.units.clone(), QKind::Gauge))
            }
            (QKind::Linear, QKind::Linear) => {
                let converted = r.value * r.units.factor() / l.units.factor();
                Ok(q(cancelled(l.value + sign * converted, l.value, converted), l.units.clone(), QKind::Linear))
            }
            _ => Err(mismatch()),
        }
    }

    /// A value about to be multiplied, divided or raised to a power.
    fn for_product(&mut self, q: &Quantity, other_dimensionless: bool, at: usize, what: &str) -> Result<Quantity, CalcError> {
        match q.kind {
            QKind::Linear => Ok(q.clone()),
            QKind::Date => Err(CalcError::at(ErrorCode::DateMisuse, at, format!("A date cannot be {what}; subtract two dates for a duration first."))),
            QKind::Gauge => Err(CalcError::at(
                ErrorCode::GaugeMisuse,
                at,
                format!(
                    "{} is a gauge pressure, relative to a local atmosphere this calculation does not state, so it cannot be {what}. Make it absolute with absolute(p, p_atm) using a sourced atmospheric pressure.",
                    q.written()
                ),
            )),
            QKind::AbsoluteTemperature => {
                let scale = q.scale().expect("an absolute temperature has one unit");
                if scale.offset == 0.0 {
                    return Ok(Quantity { kind: QKind::Linear, ..q.clone() });
                }
                if other_dimensionless {
                    return Err(CalcError::at(
                        ErrorCode::AffineMisuse,
                        at,
                        format!(
                            "Scaling {} by a plain number is ambiguous: on its own scale it means one thing and as an absolute temperature another. Convert it to K yourself, or scale a temperature difference.",
                            q.written()
                        ),
                    ));
                }
                let kelvin = q.si();
                self.note(format!("{} entered a product as {} K (absolute temperature).", q.written(), trim_number(kelvin)));
                Ok(Quantity { value: kelvin, units: unit_map("K"), kind: QKind::Linear, text: None })
            }
        }
    }

    fn mul(&mut self, l: &Quantity, r: &Quantity, div: bool, at: usize) -> Result<Quantity, CalcError> {
        let what = if div { "divided" } else { "multiplied" };
        let l = self.for_product(l, r.dim() == DIMENSIONLESS && r.kind == QKind::Linear, at, what)?;
        let r = self.for_product(r, l.dim() == DIMENSIONLESS && l.kind == QKind::Linear, at, what)?;
        if div && r.value == 0.0 {
            return Err(CalcError::at(
                ErrorCode::DivisionByZero,
                at,
                format!("Division by zero: the divisor {} is zero, so there is no result. The input is not usable as given.", r.written()),
            ));
        }
        let mut units = l.units.clone();
        for (unit, power) in &r.units.parts {
            units.add(unit, if div { -power } else { *power });
        }
        let raw = if div { l.value / r.value } else { l.value * r.value };
        let multiplier = units.simplify();
        Ok(Quantity { value: raw * multiplier, units, kind: QKind::Linear, text: None })
    }

    fn power(&mut self, base: &Quantity, n: f64, at: usize, sqrt: bool) -> Result<Quantity, CalcError> {
        let base = self.for_product(base, false, at, "raised to a power")?;
        let integer = n.fract() == 0.0;
        if base.value < 0.0 && !integer {
            return Err(CalcError::at(
                ErrorCode::Domain,
                at,
                if sqrt {
                    format!("The square root of a negative number ({}) is not real.", base.written())
                } else {
                    format!("A negative number ({}) has no real power {n}.", base.written())
                },
            ));
        }
        if base.value == 0.0 && n < 0.0 {
            return Err(CalcError::at(ErrorCode::DivisionByZero, at, "Zero raised to a negative power is a division by zero."));
        }
        let scaled_ok = base.units.parts.iter().all(|(_, p)| ((*p as f64) * n).fract() == 0.0);
        let (value, units) = if scaled_ok {
            let mut units = UnitMap::default();
            for (unit, p) in &base.units.parts {
                units.add(unit, ((*p as f64) * n) as i32);
            }
            (base.value.powf(n), units)
        } else {
            // Written units do not divide (sqrt of mm·m); try in SI base units.
            let dim = base.dim();
            if dim.iter().any(|e| ((*e as f64) * n).fract() != 0.0) {
                return Err(CalcError::at(
                    ErrorCode::DimensionMismatch,
                    at,
                    format!("{} to the power {n} has no unit: its dimension ({}) does not divide.", base.written(), describe_dim(dim)),
                ));
            }
            let mut units = UnitMap::default();
            for (symbol, exponent) in ["m", "kg", "s", "K", "mol", "A"].iter().zip(dim) {
                if exponent != 0 {
                    units.add(lookup(symbol).expect("SI base unit"), ((exponent as f64) * n) as i32);
                }
            }
            (base.si().powf(n), units)
        };
        Ok(Quantity { value, units, kind: QKind::Linear, text: None })
    }

    fn plain_argument(&mut self, func: Func, q: &Quantity, at: usize) -> Result<f64, CalcError> {
        let q = self.for_product(q, false, at, &format!("passed to {}()", func.name()))?;
        if q.dim() != DIMENSIONLESS {
            return Err(CalcError::at(
                ErrorCode::DimensionMismatch,
                at,
                format!("{}() needs a plain number (or an angle), and was given {}.", func.name(), q.describe()),
            ));
        }
        Ok(q.value * q.units.factor())
    }

    fn call(&mut self, func: Func, args: &[Quantity], at: usize) -> Result<Quantity, CalcError> {
        let domain = |message: String| CalcError::at(ErrorCode::Domain, at, message);
        match func {
            Func::Sqrt => self.power(&args[0], 0.5, at, true),
            Func::Abs => {
                let q = self.for_product(&args[0], false, at, "passed to abs()")?;
                Ok(Quantity { value: q.value.abs(), text: None, ..q })
            }
            Func::Min | Func::Max => {
                let first = &args[0];
                let mut best = first.clone();
                for other in &args[1..] {
                    if other.dim() != first.dim() || other.kind != first.kind {
                        return Err(CalcError::at(
                            ErrorCode::DimensionMismatch,
                            at,
                            format!("{}() compares like with like, and was given {} and {}.", func.name(), first.describe(), other.describe()),
                        ));
                    }
                    let better = if func == Func::Min { other.si() < best.si() } else { other.si() > best.si() };
                    if better {
                        best = other.clone();
                    }
                }
                // Expressed in the first argument's unit.
                let value = match best.kind {
                    QKind::Linear | QKind::Gauge => best.si() / first.units.factor(),
                    QKind::AbsoluteTemperature => {
                        let scale = first.scale().expect("one unit");
                        best.si() / scale.factor - scale.offset
                    }
                    QKind::Date => best.value,
                };
                Ok(Quantity { value, units: first.units.clone(), kind: first.kind, text: None })
            }
            Func::Ln | Func::Log10 => {
                let x = self.plain_argument(func, &args[0], at)?;
                if x <= 0.0 {
                    return Err(domain(format!("{}() is defined only for positive numbers, and was given {}.", func.name(), trim_number(x))));
                }
                Ok(Quantity::plain(if func == Func::Ln { x.ln() } else { x.log10() }))
            }
            Func::Exp => {
                let x = self.plain_argument(func, &args[0], at)?;
                let y = x.exp();
                if !y.is_finite() {
                    return Err(CalcError::at(ErrorCode::NonFinite, at, format!("exp({}) overflows.", trim_number(x))));
                }
                Ok(Quantity::plain(y))
            }
            Func::Sin | Func::Cos | Func::Tan => {
                let x = self.plain_argument(func, &args[0], at)?;
                if func == Func::Tan && x.cos().abs() < 1e-12 {
                    return Err(domain("tan() is undefined at an odd multiple of 90°.".into()));
                }
                Ok(Quantity::plain(match func {
                    Func::Sin => x.sin(),
                    Func::Cos => x.cos(),
                    _ => x.tan(),
                }))
            }
            Func::Absolute => {
                let (gauge, atmosphere) = (&args[0], &args[1]);
                if gauge.kind != QKind::Gauge {
                    return Err(CalcError::at(ErrorCode::GaugeMisuse, at, format!("absolute() takes a gauge pressure first (barg, psig, kPag), and was given {}.", gauge.describe())));
                }
                if atmosphere.kind != QKind::Linear || atmosphere.dim() != PRESSURE {
                    return Err(CalcError::at(ErrorCode::GaugeMisuse, at, format!("absolute() takes an absolute atmospheric pressure second, and was given {}.", atmosphere.describe())));
                }
                let scale = gauge.scale().expect("a gauge pressure has one unit");
                let units = unit_map(absolute_counterpart(scale.symbol));
                Ok(Quantity { value: (gauge.si() + atmosphere.si()) / units.factor(), units, kind: QKind::Linear, text: None })
            }
        }
    }
}

fn unit_map(symbol: &str) -> UnitMap {
    let mut map = UnitMap::default();
    map.add(lookup(symbol).expect("a unit the engine itself names"), 1);
    map
}

fn absolute_counterpart(gauge: &str) -> &'static str {
    match gauge {
        "barg" => "bar",
        "kPag" => "kPa",
        "MPag" => "MPa",
        "psig" => "psi",
        _ => "Pa",
    }
}

// ── Dimensions only ──────────────────────────────────────────────────────

/// A value's shape without its value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub dim: Dim,
    pub kind: QKind,
}

impl Shape {
    pub fn describe(&self) -> String {
        match self.kind {
            QKind::Date => "a date".into(),
            QKind::AbsoluteTemperature => "an absolute temperature".into(),
            QKind::Gauge => "a gauge pressure".into(),
            QKind::Linear => describe_dim(self.dim),
        }
    }
}

/// Walks `node` with shapes only, collecting every problem rather than
/// stopping at the first. The shape returned is the best reading when there
/// were problems (the left side of a mismatch).
pub fn infer(node: &Node, shapes: &BTreeMap<String, Shape>, problems: &mut Vec<CalcError>) -> Shape {
    let linear = |dim: Dim| Shape { dim, kind: QKind::Linear };
    match node {
        Node::Number { unit, .. } => Shape { dim: unit.dim(), kind: unit.kind().into() },
        Node::Date { .. } => Shape { dim: DIMENSIONLESS, kind: QKind::Date },
        Node::Symbol { name, at } => match shapes.get(name) {
            Some(shape) => *shape,
            None => {
                problems.push(CalcError::at(ErrorCode::UnknownSymbol, *at, format!("{name:?} is not one of the inputs.")));
                linear(DIMENSIONLESS)
            }
        },
        Node::Neg { inner, at } => {
            let shape = infer(inner, shapes, problems);
            if shape.kind == QKind::Date {
                problems.push(CalcError::at(ErrorCode::DateMisuse, *at, "A date cannot be negated."));
            }
            shape
        }
        Node::Binary { op, left, right, at } => {
            let l = infer(left, shapes, problems);
            let r = infer(right, shapes, problems);
            match op {
                BinOp::Add | BinOp::Sub => {
                    let sub = *op == BinOp::Sub;
                    let verb = if sub { "subtract" } else { "add" };
                    match (l.kind, r.kind) {
                        (QKind::Date, QKind::Date) if sub => linear(TIME),
                        (QKind::Date, QKind::Linear) if r.dim == TIME => l,
                        (QKind::Linear, QKind::Date) if !sub && l.dim == TIME => r,
                        (QKind::Date, _) | (_, QKind::Date) => {
                            problems.push(CalcError::at(ErrorCode::DateMisuse, *at, format!("Cannot {verb} {} and {}.", l.describe(), r.describe())));
                            l
                        }
                        _ if l.dim != r.dim => {
                            problems.push(CalcError::at(
                                ErrorCode::DimensionMismatch,
                                *at,
                                format!("Cannot {verb} {} and {}: the units do not match.", l.describe(), r.describe()),
                            ));
                            l
                        }
                        (QKind::AbsoluteTemperature, QKind::AbsoluteTemperature) => {
                            if !sub {
                                problems.push(CalcError::at(ErrorCode::AffineMisuse, *at, "Adding two absolute temperatures has no physical meaning."));
                            }
                            linear(TEMPERATURE)
                        }
                        (QKind::Gauge, QKind::Gauge) => {
                            if !sub {
                                problems.push(CalcError::at(ErrorCode::GaugeMisuse, *at, "Adding two gauge pressures counts the atmosphere twice."));
                            }
                            linear(PRESSURE)
                        }
                        (QKind::Linear, QKind::AbsoluteTemperature | QKind::Gauge) if sub => {
                            problems.push(CalcError::at(ErrorCode::AffineMisuse, *at, format!("Cannot subtract {} from a difference.", r.describe())));
                            r
                        }
                        (QKind::Linear, other) => Shape { dim: l.dim, kind: other },
                        _ => l,
                    }
                }
                BinOp::Mul | BinOp::Div => {
                    for (side, other) in [(l, r), (r, l)] {
                        match side.kind {
                            QKind::Date => problems.push(CalcError::at(ErrorCode::DateMisuse, *at, "A date cannot be multiplied or divided.")),
                            QKind::Gauge => problems.push(CalcError::at(ErrorCode::GaugeMisuse, *at, "A gauge pressure cannot be multiplied or divided; make it absolute with absolute(p, p_atm).")),
                            QKind::AbsoluteTemperature if other.dim == DIMENSIONLESS && other.kind == QKind::Linear => {
                                // Ambiguous only on a scale with an offset; K is
                                // fine, and the shape cannot tell them apart,
                                // so this is left to evaluation.
                            }
                            _ => {}
                        }
                    }
                    let mut dim = l.dim;
                    for (slot, e) in dim.iter_mut().zip(r.dim) {
                        *slot += if *op == BinOp::Div { -e } else { e };
                    }
                    linear(dim)
                }
            }
        }
        Node::Power { base, exponent, at, .. } => {
            let b = infer(base, shapes, problems);
            if matches!(b.kind, QKind::Date | QKind::Gauge) {
                problems.push(CalcError::at(ErrorCode::DimensionMismatch, *at, format!("{} cannot be raised to a power.", b.describe())));
            }
            if b.dim.iter().any(|e| ((*e as f64) * exponent).fract() != 0.0) {
                problems.push(CalcError::at(ErrorCode::DimensionMismatch, *at, format!("{} to the power {exponent} has no unit.", b.describe())));
                return linear(DIMENSIONLESS);
            }
            let mut dim = b.dim;
            for slot in dim.iter_mut() {
                *slot = ((*slot as f64) * exponent) as i8;
            }
            linear(dim)
        }
        Node::Call { func, args, at } => {
            let shapes_of: Vec<Shape> = args.iter().map(|a| infer(a, shapes, problems)).collect();
            match func {
                Func::Sqrt => {
                    let b = shapes_of[0];
                    if b.dim.iter().any(|e| e % 2 != 0) {
                        problems.push(CalcError::at(ErrorCode::DimensionMismatch, *at, format!("The square root of {} has no unit.", b.describe())));
                        return linear(DIMENSIONLESS);
                    }
                    let mut dim = b.dim;
                    dim.iter_mut().for_each(|e| *e /= 2);
                    linear(dim)
                }
                Func::Abs => shapes_of[0],
                Func::Min | Func::Max => {
                    let first = shapes_of[0];
                    if shapes_of.iter().any(|s| *s != first) {
                        problems.push(CalcError::at(ErrorCode::DimensionMismatch, *at, format!("{}() compares like with like.", func.name())));
                    }
                    first
                }
                Func::Ln | Func::Log10 | Func::Exp | Func::Sin | Func::Cos | Func::Tan => {
                    if shapes_of[0].dim != DIMENSIONLESS || shapes_of[0].kind != QKind::Linear {
                        problems.push(CalcError::at(
                            ErrorCode::DimensionMismatch,
                            *at,
                            format!("{}() needs a plain number, and was given {}.", func.name(), shapes_of[0].describe()),
                        ));
                    }
                    linear(DIMENSIONLESS)
                }
                Func::Absolute => {
                    if shapes_of[0].kind != QKind::Gauge || shapes_of[1].kind != QKind::Linear || shapes_of[1].dim != PRESSURE {
                        problems.push(CalcError::at(ErrorCode::GaugeMisuse, *at, "absolute() takes a gauge pressure and an absolute atmospheric pressure."));
                    }
                    linear(PRESSURE)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calculation::expr::parse;

    fn eval(text: &str) -> Result<Quantity, CalcError> {
        let env = BTreeMap::new();
        Evaluator::new(&env).eval(&parse(text).map_err(|p| CalcError::new(ErrorCode::Parse, p.message))?)
    }

    #[test]
    fn different_units_of_one_dimension_convert_and_keep_the_first_units_name() {
        let q = eval("1 m + 1 mm").unwrap();
        assert_eq!(q.format(), "1.001 m");
        assert_eq!(eval("0.354 in - 8.2 mm").unwrap().label(), "in");
        let ratio = eval("2 m / 500 mm").unwrap();
        assert!(ratio.units.is_empty());
        assert!((ratio.value - 4.0).abs() < 1e-12);
    }

    #[test]
    fn a_pressure_and_a_temperature_do_not_add() {
        let error = eval("10 bar + 300 K").unwrap_err();
        assert_eq!(error.code, ErrorCode::DimensionMismatch);
        assert!(error.message.contains("a pressure") && error.message.contains("a temperature"), "{}", error.message);
    }

    #[test]
    fn absolute_temperatures_subtract_to_a_difference_and_never_add() {
        let delta = eval("80 degC - 20 degC").unwrap();
        assert_eq!(delta.format(), "60 Δ°C");
        assert_eq!(eval("20 degC + 10 degC").unwrap_err().code, ErrorCode::AffineMisuse);
        assert_eq!(eval("20 degC + 10 delta_degC").unwrap().format(), "30 °C");
        assert_eq!(eval("68 degF - 20 degC").unwrap().format(), "0 Δ°F");
        assert_eq!(eval("2 * 20 degC").unwrap_err().code, ErrorCode::AffineMisuse);
        // In a product it is absolute, and the record says so.
        let env = BTreeMap::new();
        let mut evaluator = Evaluator::new(&env);
        let q = evaluator.eval(&parse("8.314 J/(mol·K) * 20 degC").unwrap()).unwrap();
        assert!((q.si() - 8.314 * 293.15).abs() < 1e-9);
        assert!(evaluator.notes[0].contains("293.15 K"));
    }

    #[test]
    fn gauge_pressure_needs_a_stated_atmosphere_to_become_absolute() {
        assert_eq!(eval("10 barg * 2").unwrap_err().code, ErrorCode::GaugeMisuse);
        assert_eq!(eval("10 barg + 5 barg").unwrap_err().code, ErrorCode::GaugeMisuse);
        assert_eq!(eval("12 barg - 10 barg").unwrap().format(), "2 bar");
        let absolute = eval("absolute(10 barg, 1.01325 bar)").unwrap();
        assert_eq!(absolute.format(), "11.01325 bar");
    }

    #[test]
    fn dates_move_by_whole_days_only() {
        assert_eq!(eval("2026-08-12 + 90 d").unwrap().format(), "2026-11-10");
        assert_eq!(eval("2026-08-12 - 2024-03-18").unwrap().format(), "877 d");
        assert_eq!(eval("2026-08-12 + 1.5 d").unwrap_err().code, ErrorCode::DateMisuse);
        assert_eq!(eval("2026-08-12 + 1 a").unwrap_err().code, ErrorCode::DateMisuse);
        assert_eq!(eval("2026-08-12 * 2").unwrap_err().code, ErrorCode::DateMisuse);
    }

    #[test]
    fn a_domain_error_is_refused_and_named() {
        assert_eq!(eval("sqrt(-4 mm^2)").unwrap_err().code, ErrorCode::Domain);
        assert_eq!(eval("ln(0)").unwrap_err().code, ErrorCode::Domain);
        assert_eq!(eval("ln(2 mm)").unwrap_err().code, ErrorCode::DimensionMismatch);
        assert_eq!(eval("5 mm / 0 mm").unwrap_err().code, ErrorCode::DivisionByZero);
        assert_eq!(eval("exp(1000)").unwrap_err().code, ErrorCode::NonFinite);
        assert_eq!(eval("tan(90 deg)").unwrap_err().code, ErrorCode::Domain);
        assert_eq!(eval("(1e200 * 1e200)").unwrap_err().code, ErrorCode::NonFinite);
    }

    #[test]
    fn powers_and_roots_carry_their_units() {
        assert_eq!(eval("(3 mm)^2").unwrap().format(), "9 mm²");
        assert_eq!(eval("sqrt(16 m^2)").unwrap().format(), "4 m");
        assert_eq!(eval("sqrt(2 mm)").unwrap_err().code, ErrorCode::DimensionMismatch);
        assert!((eval("sin(30 deg)").unwrap().value - 0.5).abs() < 1e-12);
    }

    #[test]
    fn dimensions_are_inferred_without_values_and_every_problem_is_listed() {
        let mut shapes = BTreeMap::new();
        shapes.insert("p".to_string(), Shape { dim: PRESSURE, kind: QKind::Linear });
        shapes.insert("t".to_string(), Shape { dim: TEMPERATURE, kind: QKind::AbsoluteTemperature });
        let mut problems = Vec::new();
        // p + t, ln(p), and a pressure plus the plain number ln() returns.
        infer(&parse("p + t + ln(p)").unwrap(), &shapes, &mut problems);
        assert_eq!(problems.len(), 3, "{problems:?}");
        let mut none = Vec::new();
        let shape = infer(&parse("p * 2 m^2").unwrap(), &shapes, &mut none);
        assert!(none.is_empty());
        assert_eq!(shape.dim, [1, 1, -2, 0, 0, 0]);
    }
}

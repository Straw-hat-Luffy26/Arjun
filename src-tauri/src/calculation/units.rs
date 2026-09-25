//! The units the engine knows, and nothing else.
//!
//! A unit this table does not list is an error, not a new dimension. The engine
//! used to treat any word after a number as a unit of its own, so `1 m + 1 mm`
//! was refused as mismatched while `8.2 mmm + 1 mmm` computed happily. A short
//! table that is right beats a permissive parser that is sometimes wrong.
//!
//! ## Three kinds of quantity that are not plain magnitudes
//!
//! - **Absolute temperature** in °C or °F has an offset: `20 °C + 10 °C` is not
//!   30 °C of anything. Differences are written `delta_degC` (`Δ°C`), and a
//!   temperature that enters a product is converted to kelvin and the record
//!   says so.
//! - **Gauge pressure** (`barg`, `psig`, `kPag`) is relative to a local
//!   atmosphere the engine does not know. It adds and subtracts with pressure
//!   differences; making it absolute takes `absolute(p, p_atm)` with a sourced
//!   atmospheric pressure.
//! - **Dates** (`2026-08-12`) subtract to a duration in days and take whole
//!   days added; a month is not a duration and is refused.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

/// Exponents over the base dimensions, in this order.
pub const BASES: [&str; 6] = ["L", "M", "T", "Θ", "N", "I"];

pub type Dim = [i8; 6];

pub const DIMENSIONLESS: Dim = [0; 6];
const LENGTH: Dim = [1, 0, 0, 0, 0, 0];
const MASS: Dim = [0, 1, 0, 0, 0, 0];
const TIME: Dim = [0, 0, 1, 0, 0, 0];
pub const TEMPERATURE: Dim = [0, 0, 0, 1, 0, 0];
const AMOUNT: Dim = [0, 0, 0, 0, 1, 0];
const CURRENT: Dim = [0, 0, 0, 0, 0, 1];
const AREA: Dim = [2, 0, 0, 0, 0, 0];
const VOLUME: Dim = [3, 0, 0, 0, 0, 0];
const FREQUENCY: Dim = [0, 0, -1, 0, 0, 0];
const FORCE: Dim = [1, 1, -2, 0, 0, 0];
pub const PRESSURE: Dim = [-1, 1, -2, 0, 0, 0];
const ENERGY: Dim = [2, 1, -2, 0, 0, 0];
const POWER: Dim = [2, 1, -3, 0, 0, 0];
const VOLTAGE: Dim = [2, 1, -3, 0, 0, -1];
const DYNAMIC_VISCOSITY: Dim = [-1, 1, -1, 0, 0, 0];

/// What adding and subtracting mean for a unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A magnitude: adds, scales, converts by a factor.
    Linear,
    /// A temperature on a scale with an offset.
    AbsoluteTemperature,
    /// A pressure measured from the local atmosphere.
    Gauge,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnitDef {
    /// The canonical symbol, as the engine writes it back.
    pub symbol: &'static str,
    /// Other ways it may be written.
    pub aliases: &'static [&'static str],
    pub dim: Dim,
    /// SI value of one of this unit: `si = (value + offset) × factor`.
    pub factor: f64,
    pub offset: f64,
    pub kind: Kind,
    /// How the engine writes it (`°C`, `mm`).
    pub label: &'static str,
}

const fn unit(symbol: &'static str, aliases: &'static [&'static str], dim: Dim, factor: f64, label: &'static str) -> UnitDef {
    UnitDef { symbol, aliases, dim, factor, offset: 0.0, kind: Kind::Linear, label }
}

/// Pounds-force per square inch, exactly (lbf = 4.4482216152605 N, in = 0.0254 m).
const PSI: f64 = 4.448_221_615_260_5 / (0.0254 * 0.0254);

/// The table. Its digest is part of the engine's identity, so a changed
/// factor changes every record id computed with it.
pub const UNITS: &[UnitDef] = &[
    // Length
    unit("m", &["metre", "meter", "metres", "meters"], LENGTH, 1.0, "m"),
    unit("mm", &[], LENGTH, 1e-3, "mm"),
    unit("cm", &[], LENGTH, 1e-2, "cm"),
    unit("km", &[], LENGTH, 1e3, "km"),
    unit("um", &["µm", "μm", "micron", "microns"], LENGTH, 1e-6, "µm"),
    unit("in", &["inch", "inches"], LENGTH, 0.0254, "in"),
    unit("ft", &["foot", "feet"], LENGTH, 0.3048, "ft"),
    // Volume with names of their own
    unit("L", &["litre", "liter", "litres", "liters"], VOLUME, 1e-3, "L"),
    unit("mL", &["ml"], VOLUME, 1e-6, "mL"),
    unit("ha", &["hectare", "hectares"], AREA, 1e4, "ha"),
    // Mass
    unit("kg", &[], MASS, 1.0, "kg"),
    unit("g", &["gram", "grams"], MASS, 1e-3, "g"),
    unit("t", &["tonne", "tonnes"], MASS, 1e3, "t"),
    unit("lb", &["lbm", "lbs"], MASS, 0.453_592_37, "lb"),
    // Time
    unit("s", &["sec", "second", "seconds"], TIME, 1.0, "s"),
    unit("min", &["minute", "minutes"], TIME, 60.0, "min"),
    unit("h", &["hr", "hour", "hours"], TIME, 3600.0, "h"),
    unit("d", &["day", "days"], TIME, 86_400.0, "d"),
    unit("wk", &["week", "weeks"], TIME, 604_800.0, "wk"),
    // The Julian year, 365.25 d. Stated in every record that uses it.
    unit("a", &["yr", "year", "years"], TIME, 31_557_600.0, "a"),
    unit("Hz", &[], FREQUENCY, 1.0, "Hz"),
    unit("rpm", &[], FREQUENCY, 1.0 / 60.0, "rpm"),
    // Temperature: absolute scales and differences
    UnitDef { symbol: "K", aliases: &["kelvin"], dim: TEMPERATURE, factor: 1.0, offset: 0.0, kind: Kind::AbsoluteTemperature, label: "K" },
    UnitDef { symbol: "degC", aliases: &["°C", "℃", "celsius"], dim: TEMPERATURE, factor: 1.0, offset: 273.15, kind: Kind::AbsoluteTemperature, label: "°C" },
    UnitDef { symbol: "degF", aliases: &["°F", "℉", "fahrenheit"], dim: TEMPERATURE, factor: 5.0 / 9.0, offset: 459.67, kind: Kind::AbsoluteTemperature, label: "°F" },
    unit("delta_K", &["ΔK"], TEMPERATURE, 1.0, "ΔK"),
    unit("delta_degC", &["Δ°C", "ΔdegC"], TEMPERATURE, 1.0, "Δ°C"),
    unit("delta_degF", &["Δ°F", "ΔdegF"], TEMPERATURE, 5.0 / 9.0, "Δ°F"),
    // Pressure (absolute, or a difference)
    unit("Pa", &[], PRESSURE, 1.0, "Pa"),
    unit("kPa", &["kPaa"], PRESSURE, 1e3, "kPa"),
    unit("MPa", &["MPaa"], PRESSURE, 1e6, "MPa"),
    unit("GPa", &[], PRESSURE, 1e9, "GPa"),
    unit("bar", &["bara"], PRESSURE, 1e5, "bar"),
    unit("mbar", &[], PRESSURE, 1e2, "mbar"),
    unit("psi", &["psia"], PRESSURE, PSI, "psi"),
    unit("atm", &[], PRESSURE, 101_325.0, "atm"),
    unit("mmHg", &[], PRESSURE, 133.322_387_415, "mmHg"),
    // Gauge pressure
    UnitDef { symbol: "barg", aliases: &[], dim: PRESSURE, factor: 1e5, offset: 0.0, kind: Kind::Gauge, label: "barg" },
    UnitDef { symbol: "kPag", aliases: &[], dim: PRESSURE, factor: 1e3, offset: 0.0, kind: Kind::Gauge, label: "kPag" },
    UnitDef { symbol: "MPag", aliases: &[], dim: PRESSURE, factor: 1e6, offset: 0.0, kind: Kind::Gauge, label: "MPag" },
    UnitDef { symbol: "psig", aliases: &[], dim: PRESSURE, factor: PSI, offset: 0.0, kind: Kind::Gauge, label: "psig" },
    // Force, energy, power
    unit("N", &["newton", "newtons"], FORCE, 1.0, "N"),
    unit("kN", &[], FORCE, 1e3, "kN"),
    unit("MN", &[], FORCE, 1e6, "MN"),
    unit("lbf", &[], FORCE, 4.448_221_615_260_5, "lbf"),
    unit("kgf", &[], FORCE, 9.806_65, "kgf"),
    unit("J", &["joule", "joules"], ENERGY, 1.0, "J"),
    unit("kJ", &[], ENERGY, 1e3, "kJ"),
    unit("MJ", &[], ENERGY, 1e6, "MJ"),
    unit("GJ", &[], ENERGY, 1e9, "GJ"),
    unit("Wh", &[], ENERGY, 3600.0, "Wh"),
    unit("kWh", &[], ENERGY, 3.6e6, "kWh"),
    unit("MWh", &[], ENERGY, 3.6e9, "MWh"),
    unit("Btu", &["BTU"], ENERGY, 1_055.055_852_62, "Btu"),
    unit("W", &["watt", "watts"], POWER, 1.0, "W"),
    unit("kW", &[], POWER, 1e3, "kW"),
    unit("MW", &[], POWER, 1e6, "MW"),
    unit("V", &["volt", "volts"], VOLTAGE, 1.0, "V"),
    unit("kV", &[], VOLTAGE, 1e3, "kV"),
    unit("A", &["amp", "amps", "ampere"], CURRENT, 1.0, "A"),
    unit("mA", &[], CURRENT, 1e-3, "mA"),
    unit("mol", &[], AMOUNT, 1.0, "mol"),
    unit("kmol", &[], AMOUNT, 1e3, "kmol"),
    unit("cP", &["cp", "centipoise"], DYNAMIC_VISCOSITY, 1e-3, "cP"),
    // Dimensionless
    unit("%", &["percent", "pct"], DIMENSIONLESS, 0.01, "%"),
    unit("ppm", &[], DIMENSIONLESS, 1e-6, "ppm"),
    unit("rad", &["radian", "radians"], DIMENSIONLESS, 1.0, "rad"),
    unit("deg", &["°", "degree", "degrees"], DIMENSIONLESS, std::f64::consts::PI / 180.0, "°"),
];

/// Words that look like units and are refused, with the reason.
const REFUSED: &[(&str, &str)] = &[
    ("month", "a month is not a fixed duration (28 to 31 days); give the interval in days"),
    ("months", "a month is not a fixed duration (28 to 31 days); give the interval in days"),
    ("mo", "a month is not a fixed duration (28 to 31 days); give the interval in days"),
    ("gal", "US and imperial gallons differ by 20%; give the volume in L or m^3"),
    ("gallon", "US and imperial gallons differ by 20%; give the volume in L or m^3"),
    ("ton", "short, long and metric tons differ; give the mass in t (tonne) or kg"),
    ("tons", "short, long and metric tons differ; give the mass in t (tonne) or kg"),
    ("cal", "the thermochemical and IT calories differ; give the energy in J or kJ"),
    ("hp", "mechanical and metric horsepower differ; give the power in W or kW"),
    ("C", "C alone is ambiguous (coulomb or degrees Celsius); write degC for a temperature"),
    ("F", "F alone is ambiguous (farad or degrees Fahrenheit); write degF for a temperature"),
    ("kgf/cm2", "write kgf/cm^2"),
];

/// The version of this table, for the engine's identity.
pub fn table_digest() -> String {
    let mut hasher = Sha256::new();
    for unit in UNITS {
        hasher.update(format!(
            "{}|{}|{:?}|{:e}|{:e}|{:?};",
            unit.symbol,
            unit.aliases.join(","),
            unit.dim,
            unit.factor,
            unit.offset,
            unit.kind
        ));
    }
    let digest = hasher.finalize();
    digest[..4].iter().map(|b| format!("{b:02x}")).collect()
}

/// Looks a single unit symbol up, aliases included.
pub fn lookup(symbol: &str) -> Result<&'static UnitDef, UnitProblem> {
    if let Some(found) = UNITS.iter().find(|u| u.symbol == symbol || u.aliases.contains(&symbol)) {
        return Ok(found);
    }
    if let Some((_, why)) = REFUSED.iter().find(|(word, _)| *word == symbol) {
        return Err(UnitProblem::Ambiguous { unit: symbol.to_string(), why: why.to_string() });
    }
    Err(UnitProblem::Unknown { unit: symbol.to_string() })
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnitProblem {
    Unknown { unit: String },
    Ambiguous { unit: String, why: String },
    Malformed { text: String, why: String },
}

impl UnitProblem {
    pub fn explain(&self) -> String {
        match self {
            UnitProblem::Unknown { unit } => format!(
                "{unit:?} does not read as a unit here: it is not one this engine knows. Use a \
                 listed unit (mm, m, in, kg, s, h, d, a, bar, barg, kPa, psi, degC, delta_degC, \
                 K, N, J, W, %, …) or write the quantity another way."
            ),
            UnitProblem::Ambiguous { unit, why } => format!("{unit:?} is not accepted: {why}."),
            UnitProblem::Malformed { text, why } => format!("the unit {text:?} could not be read: {why}."),
        }
    }
}

/// A unit expression: symbols with integer powers, in the order written.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UnitExpr {
    pub parts: Vec<(&'static UnitDef, i32)>,
}

impl UnitExpr {
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    pub fn dim(&self) -> Dim {
        let mut dim = DIMENSIONLESS;
        for (unit, power) in &self.parts {
            for (slot, exponent) in dim.iter_mut().zip(unit.dim) {
                *slot += exponent * (*power as i8);
            }
        }
        dim
    }

    /// SI value of one of this compound unit, ignoring offsets.
    pub fn factor(&self) -> f64 {
        self.parts.iter().map(|(unit, power)| unit.factor.powi(*power)).product()
    }

    /// The kind of a quantity carrying exactly this unit.
    ///
    /// Only a lone temperature or gauge symbol at the first power is absolute
    /// or gauge; inside a compound (`J/(kg·K)`) it is a plain dimension.
    pub fn kind(&self) -> Kind {
        match self.parts.as_slice() {
            [(unit, 1)] => unit.kind,
            _ => Kind::Linear,
        }
    }

    pub fn offset(&self) -> f64 {
        match self.parts.as_slice() {
            [(unit, 1)] => unit.offset,
            _ => 0.0,
        }
    }

    pub fn label(&self) -> String {
        label_of(self.parts.iter().map(|(u, p)| (u.label, *p)))
    }

    /// The same symbols as a power map, for arithmetic.
    pub fn as_map(&self) -> UnitMap {
        let mut map = UnitMap::default();
        for (unit, power) in &self.parts {
            map.add(unit, *power);
        }
        map
    }
}

/// A product of unit symbols with integer powers, keeping first-seen order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UnitMap {
    pub parts: Vec<(&'static UnitDef, i32)>,
}

impl UnitMap {
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    pub fn add(&mut self, unit: &'static UnitDef, power: i32) {
        if power == 0 {
            return;
        }
        if let Some(slot) = self.parts.iter_mut().find(|(held, _)| held.symbol == unit.symbol) {
            slot.1 += power;
        } else {
            self.parts.push((unit, power));
        }
        self.parts.retain(|(_, p)| *p != 0);
    }

    pub fn dim(&self) -> Dim {
        UnitExpr { parts: self.parts.clone() }.dim()
    }

    pub fn factor(&self) -> f64 {
        UnitExpr { parts: self.parts.clone() }.factor()
    }

    pub fn label(&self) -> String {
        label_of(self.parts.iter().map(|(u, p)| (u.label, *p)))
    }

    pub fn single(&self) -> Option<&'static UnitDef> {
        match self.parts.as_slice() {
            [(unit, 1)] => Some(unit),
            _ => None,
        }
    }

    /// Folds same-dimension symbols together, returning the multiplier the
    /// value needs: `m/mm` becomes a plain 1000, `kPa·bar⁻¹` 0.01.
    ///
    /// Two symbols merge when their dimensions are identical; the one written
    /// first is kept. Dimensionless symbols (`%`, `ppm`, angles) are folded
    /// into the number whenever anything else is present, so `50% × 200 kg`
    /// is `100 kg` and not `10000 %·kg`.
    pub fn simplify(&mut self) -> f64 {
        let mut multiplier = 1.0;
        let mut i = 0;
        while i < self.parts.len() {
            let (keep, _) = self.parts[i];
            let mut j = i + 1;
            while j < self.parts.len() {
                let (other, power) = self.parts[j];
                if other.dim == keep.dim && other.dim != DIMENSIONLESS && other.symbol != keep.symbol {
                    multiplier *= (other.factor / keep.factor).powi(power);
                    self.parts[i].1 += power;
                    self.parts.remove(j);
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
        self.parts.retain(|(_, p)| *p != 0);
        let has_dimensional = self.parts.iter().any(|(u, _)| u.dim != DIMENSIONLESS);
        let dimensionless_count = self.parts.iter().filter(|(u, _)| u.dim == DIMENSIONLESS).count();
        if has_dimensional || dimensionless_count > 1 || self.parts.iter().any(|(u, p)| u.dim == DIMENSIONLESS && *p != 1) {
            for (unit, power) in self.parts.iter().filter(|(u, _)| u.dim == DIMENSIONLESS) {
                multiplier *= unit.factor.powi(*power);
            }
            self.parts.retain(|(u, _)| u.dim != DIMENSIONLESS);
        }
        // Everything cancelled to a pure number: say it in plain numbers.
        if !self.parts.is_empty() && self.dim() == DIMENSIONLESS && self.parts.iter().all(|(u, _)| u.dim != DIMENSIONLESS) {
            multiplier *= self.factor();
            self.parts.clear();
        }
        multiplier
    }
}

fn superscript(power: i32) -> String {
    match power {
        2 => "²".to_string(),
        3 => "³".to_string(),
        n => format!("^{n}"),
    }
}

fn label_of(parts: impl Iterator<Item = (&'static str, i32)>) -> String {
    let parts: Vec<(&str, i32)> = parts.collect();
    if parts.is_empty() {
        return String::new();
    }
    let render = |(label, power): (&str, i32)| -> String {
        let magnitude = power.abs();
        if magnitude == 1 {
            label.to_string()
        } else {
            format!("{label}{}", superscript(magnitude))
        }
    };
    let top: Vec<String> = parts.iter().filter(|(_, p)| *p > 0).map(|&(l, p)| render((l, p))).collect();
    let bottom: Vec<String> = parts.iter().filter(|(_, p)| *p < 0).map(|&(l, p)| render((l, p))).collect();
    let top = if top.is_empty() { "1".to_string() } else { top.join("·") };
    match bottom.len() {
        0 => top,
        1 => format!("{top}/{}", bottom[0]),
        _ => format!("{top}/({})", bottom.join("·")),
    }
}

/// Parses a unit expression: `mm`, `kg/m^3`, `W/(m·K)`, `m³/h`, `N*m`.
///
/// `·`, `*` multiply; `/` divides by the one symbol after it, or by a
/// bracketed group. A product after a division without brackets (`J/kg*K`)
/// is refused as ambiguous rather than guessed.
pub fn parse_unit(text: &str) -> Result<UnitExpr, UnitProblem> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(UnitExpr::default());
    }
    if let Some((_, why)) = REFUSED.iter().find(|(word, _)| *word == text) {
        return Err(UnitProblem::Ambiguous { unit: text.to_string(), why: why.to_string() });
    }
    let malformed = |why: &str| UnitProblem::Malformed { text: text.to_string(), why: why.to_string() };
    let chars: Vec<char> = text.chars().collect();
    let mut position = 0;
    let mut parts: Vec<(&'static UnitDef, i32)> = Vec::new();
    let mut after_division = false;

    let push = |parts: &mut Vec<(&'static UnitDef, i32)>, unit: &'static UnitDef, power: i32| {
        if let Some(slot) = parts.iter_mut().find(|(held, _)| held.symbol == unit.symbol) {
            slot.1 += power;
        } else {
            parts.push((unit, power));
        }
    };

    let mut sign = 1;
    loop {
        if position >= chars.len() {
            return Err(malformed("it ends with an operator"));
        }
        if chars[position] == '(' {
            if sign != -1 {
                return Err(malformed("a bracket belongs after a division, as in W/(m·K)"));
            }
            let close = chars[position..].iter().position(|c| *c == ')').map(|at| at + position)
                .ok_or_else(|| malformed("a bracket is never closed"))?;
            let inner: String = chars[position + 1..close].iter().collect();
            for (unit, power) in parse_unit(&inner)?.parts {
                push(&mut parts, unit, -power);
            }
            position = close + 1;
        } else {
            let start = position;
            while position < chars.len() && is_unit_char(chars[position]) {
                position += 1;
            }
            if start == position {
                return Err(malformed("a unit symbol was expected"));
            }
            let symbol: String = chars[start..position].iter().collect();
            let mut power: i32 = 1;
            if position < chars.len() && (chars[position] == '²' || chars[position] == '³') {
                power = if chars[position] == '²' { 2 } else { 3 };
                position += 1;
            } else if position < chars.len() && chars[position] == '^' {
                position += 1;
                let exponent_start = position;
                if position < chars.len() && chars[position] == '-' {
                    position += 1;
                }
                while position < chars.len() && chars[position].is_ascii_digit() {
                    position += 1;
                }
                let raw: String = chars[exponent_start..position].iter().collect();
                power = raw.parse().map_err(|_| malformed("^ needs a whole-number power"))?;
                if power == 0 || power.abs() > 6 {
                    return Err(malformed("a unit power must be between -6 and 6, and not 0"));
                }
            } else if position < chars.len() && chars[position].is_ascii_digit() {
                // `m3` and `cm2` are common and unambiguous: a trailing digit
                // on a length symbol is its power.
                let digit_start = position;
                while position < chars.len() && chars[position].is_ascii_digit() {
                    position += 1;
                }
                let raw: String = chars[digit_start..position].iter().collect();
                power = raw.parse().map_err(|_| malformed("unreadable power"))?;
                if !(2..=3).contains(&power) {
                    return Err(malformed("write a power as ^n"));
                }
            }
            let unit = lookup(&symbol)?;
            push(&mut parts, unit, power * sign);
        }
        if position >= chars.len() {
            break;
        }
        match chars[position] {
            // A second division (`kg/m/s`) reads as kg/(m·s), which is the
            // only reading anybody means by it.
            '/' => {
                sign = -1;
                after_division = true;
            }
            '*' | '·' | '.' => {
                if after_division {
                    return Err(malformed("a product after a division is ambiguous; bracket the denominator, as in J/(kg·K)"));
                }
                sign = 1;
            }
            other => return Err(malformed(&format!("{other:?} is not part of a unit"))),
        }
        position += 1;
    }
    parts.retain(|(_, p)| *p != 0);
    Ok(UnitExpr { parts })
}

pub fn is_unit_char(c: char) -> bool {
    c.is_alphabetic() || matches!(c, '%' | '°' | 'µ' | 'μ' | '℃' | '℉' | 'Δ' | '_')
}

/// A dimension written for a person: `pressure`, `length`, or its exponents.
pub fn describe_dim(dim: Dim) -> String {
    let named: &[(Dim, &str)] = &[
        (DIMENSIONLESS, "a plain number"),
        (LENGTH, "a length"),
        (MASS, "a mass"),
        (TIME, "a time"),
        (TEMPERATURE, "a temperature"),
        (AMOUNT, "an amount of substance"),
        (CURRENT, "a current"),
        (AREA, "an area"),
        (VOLUME, "a volume"),
        (FREQUENCY, "a frequency"),
        (FORCE, "a force"),
        (PRESSURE, "a pressure"),
        (ENERGY, "an energy"),
        (POWER, "a power"),
        (VOLTAGE, "a voltage"),
        (DYNAMIC_VISCOSITY, "a dynamic viscosity"),
        ([1, 0, -1, 0, 0, 0], "a velocity"),
        ([-3, 1, 0, 0, 0, 0], "a density"),
        ([3, 0, -1, 0, 0, 0], "a volumetric flow"),
        ([0, 1, -1, 0, 0, 0], "a mass flow"),
        ([1, 0, -2, 0, 0, 0], "an acceleration"),
    ];
    if let Some((_, name)) = named.iter().find(|(d, _)| *d == dim) {
        return name.to_string();
    }
    format!("a quantity of dimension {}", si_label(dim))
}

/// The SI base-unit label for a dimension: `kg·m⁻¹·s⁻²` written `kg/(m·s²)`.
pub fn si_label(dim: Dim) -> String {
    const SI: [&str; 6] = ["m", "kg", "s", "K", "mol", "A"];
    let order = [1usize, 0, 2, 3, 4, 5];
    label_of(order.iter().filter(|i| dim[**i] != 0).map(|i| (SI[*i], dim[*i] as i32)))
}

/// Exponent-by-exponent, for a report.
pub fn dim_map(dim: Dim) -> BTreeMap<&'static str, i8> {
    BASES.iter().zip(dim).filter(|(_, e)| *e != 0).map(|(b, e)| (*b, e)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_compound_unit_is_parsed_with_its_powers() {
        let density = parse_unit("kg/m^3").unwrap();
        assert_eq!(density.dim(), [-3, 1, 0, 0, 0, 0]);
        assert_eq!(density.label(), "kg/m³");
        let conductivity = parse_unit("W/(m·K)").unwrap();
        assert_eq!(conductivity.dim(), [1, 1, -3, -1, 0, 0]);
        assert_eq!(parse_unit("m³/h").unwrap().dim(), [3, 0, -1, 0, 0, 0]);
        assert_eq!(parse_unit("N*m").unwrap().dim(), ENERGY);
        assert_eq!(parse_unit("m3").unwrap().dim(), VOLUME);
        assert!((parse_unit("psi").unwrap().factor() - 6894.757_293_168).abs() < 1e-6);
    }

    #[test]
    fn an_ambiguous_or_unknown_unit_is_refused_with_its_reason() {
        assert!(matches!(parse_unit("J/kg*K"), Err(UnitProblem::Malformed { .. })));
        assert!(matches!(parse_unit("months"), Err(UnitProblem::Ambiguous { .. })));
        assert!(matches!(parse_unit("gal"), Err(UnitProblem::Ambiguous { .. })));
        assert!(matches!(parse_unit("C"), Err(UnitProblem::Ambiguous { .. })));
        assert!(matches!(parse_unit("widgets"), Err(UnitProblem::Unknown { .. })));
        assert!(parse_unit("m^0").is_err());
    }

    #[test]
    fn a_lone_temperature_is_absolute_and_inside_a_compound_it_is_a_dimension() {
        assert_eq!(parse_unit("degC").unwrap().kind(), Kind::AbsoluteTemperature);
        assert_eq!(parse_unit("°C").unwrap().kind(), Kind::AbsoluteTemperature);
        assert_eq!(parse_unit("J/(kg·K)").unwrap().kind(), Kind::Linear);
        assert_eq!(parse_unit("delta_degC").unwrap().kind(), Kind::Linear);
        assert_eq!(parse_unit("barg").unwrap().kind(), Kind::Gauge);
    }

    #[test]
    fn same_dimension_symbols_fold_together() {
        let mut map = UnitMap::default();
        map.add(lookup("m").unwrap(), 1);
        map.add(lookup("mm").unwrap(), -1);
        assert_eq!(map.simplify(), 1000.0);
        assert!(map.is_empty());

        let mut percent_of_mass = UnitMap::default();
        percent_of_mass.add(lookup("%").unwrap(), 1);
        percent_of_mass.add(lookup("kg").unwrap(), 1);
        assert_eq!(percent_of_mass.simplify(), 0.01);
        assert_eq!(percent_of_mass.label(), "kg");

        let mut percent = UnitMap::default();
        percent.add(lookup("%").unwrap(), 1);
        assert_eq!(percent.simplify(), 1.0);
        assert_eq!(percent.label(), "%");
    }

    #[test]
    fn the_table_has_a_stable_digest() {
        assert_eq!(table_digest(), table_digest());
        assert_eq!(table_digest().len(), 8);
        for unit in UNITS {
            assert!(unit.factor.is_finite() && unit.factor > 0.0, "{}", unit.symbol);
        }
    }
}

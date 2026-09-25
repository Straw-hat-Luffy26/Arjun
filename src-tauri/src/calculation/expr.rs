//! The expression language: small on purpose, bounded, and never `eval`.
//!
//! ```text
//! equation := expr ( '=' expr )?
//! expr     := term (( '+' | '-' ) term)*
//! term     := unary (( '*' | '/' | '×' | '÷' ) unary)*
//! unary    := ( '-' | '+' ) unary | power
//! power    := primary ( '^' exponent )?
//! primary  := number unit? | date | symbol | function '(' expr (',' expr)* ')' | '(' expr ')'
//! ```
//!
//! A unit follows its number (`8.2 mm`, `1500 kg/m^3`); a bare word elsewhere
//! is an input's symbol. There is no implicit multiplication: `2 x` is refused
//! rather than read as `2 * x`, because the same shape is how a sentence
//! containing a number reads. Functions are a fixed list. Every parse is
//! bounded in length, nodes and depth, so no input makes the engine do
//! unbounded work.

use super::decimal::Decimal;
use super::units::{is_unit_char, parse_unit, UnitExpr};

pub const MAX_CHARS: usize = 1000;
pub const MAX_NODES: usize = 256;
pub const MAX_DEPTH: usize = 32;
pub const MAX_ARGS: usize = 8;
pub const MAX_EXPONENT: f64 = 12.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinOp {
    pub fn symbol(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
        }
    }

    fn precedence(self) -> u8 {
        match self {
            BinOp::Add | BinOp::Sub => 1,
            BinOp::Mul | BinOp::Div => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Func {
    Sqrt,
    Abs,
    Min,
    Max,
    Ln,
    Log10,
    Exp,
    Sin,
    Cos,
    Tan,
    /// `absolute(gauge, atmosphere)`: a gauge pressure made absolute with a
    /// stated atmospheric pressure.
    Absolute,
}

impl Func {
    pub const ALL: &'static [(&'static str, Func)] = &[
        ("sqrt", Func::Sqrt),
        ("abs", Func::Abs),
        ("min", Func::Min),
        ("max", Func::Max),
        ("ln", Func::Ln),
        ("log10", Func::Log10),
        ("exp", Func::Exp),
        ("sin", Func::Sin),
        ("cos", Func::Cos),
        ("tan", Func::Tan),
        ("absolute", Func::Absolute),
    ];

    pub fn name(self) -> &'static str {
        Func::ALL.iter().find(|(_, f)| *f == self).map(|(n, _)| *n).unwrap_or("?")
    }

    fn arity(self) -> (usize, usize) {
        match self {
            Func::Min | Func::Max => (2, MAX_ARGS),
            Func::Absolute => (2, 2),
            _ => (1, 1),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Number { text: String, value: Decimal, unit: UnitExpr, unit_text: String, at: usize },
    /// Days since 1970-01-01.
    Date { text: String, day: i64, at: usize },
    Symbol { name: String, at: usize },
    Neg { inner: Box<Node>, at: usize },
    Binary { op: BinOp, left: Box<Node>, right: Box<Node>, at: usize },
    Power { base: Box<Node>, exponent: f64, exponent_text: String, at: usize },
    Call { func: Func, args: Vec<Node>, at: usize },
}

impl Node {
    pub fn at(&self) -> usize {
        match self {
            Node::Number { at, .. }
            | Node::Date { at, .. }
            | Node::Symbol { at, .. }
            | Node::Neg { at, .. }
            | Node::Binary { at, .. }
            | Node::Power { at, .. }
            | Node::Call { at, .. } => *at,
        }
    }

    /// Every symbol this refers to, in first-use order.
    pub fn symbols(&self) -> Vec<String> {
        let mut out = Vec::new();
        self.collect_symbols(&mut out);
        out
    }

    fn collect_symbols(&self, out: &mut Vec<String>) {
        match self {
            Node::Symbol { name, .. } => {
                if !out.contains(name) {
                    out.push(name.clone());
                }
            }
            Node::Neg { inner, .. } => inner.collect_symbols(out),
            Node::Binary { left, right, .. } => {
                left.collect_symbols(out);
                right.collect_symbols(out);
            }
            Node::Power { base, .. } => base.collect_symbols(out),
            Node::Call { args, .. } => args.iter().for_each(|a| a.collect_symbols(out)),
            Node::Number { .. } | Node::Date { .. } => {}
        }
    }

    /// Literal numbers as written, with units: the inputs a bare expression
    /// carries.
    pub fn literals(&self) -> Vec<String> {
        let mut out = Vec::new();
        self.collect_literals(&mut out);
        out
    }

    fn collect_literals(&self, out: &mut Vec<String>) {
        match self {
            Node::Number { text, unit_text, .. } => {
                out.push(if unit_text.is_empty() { text.clone() } else { format!("{text} {unit_text}") })
            }
            Node::Date { text, .. } => out.push(text.clone()),
            Node::Symbol { .. } => {}
            Node::Neg { inner, .. } => inner.collect_literals(out),
            Node::Binary { left, right, .. } => {
                left.collect_literals(out);
                right.collect_literals(out);
            }
            Node::Power { base, .. } => base.collect_literals(out),
            Node::Call { args, .. } => args.iter().for_each(|a| a.collect_literals(out)),
        }
    }

    fn precedence(&self) -> u8 {
        match self {
            Node::Binary { op, .. } => op.precedence(),
            Node::Neg { .. } => 3,
            Node::Power { .. } => 4,
            Node::Number { unit, .. } if !unit.is_empty() => 2,
            Node::Number { value, .. } if value.is_negative() => 3,
            _ => 5,
        }
    }

    /// Written back out, with `substitute` replacing each symbol.
    ///
    /// A substituted value that has a unit reads like a product (`8.2 mm`),
    /// and one that is negative like a negation; each is bracketed where its
    /// position needs it and nowhere else.
    pub fn render(&self, substitute: &dyn Fn(&str) -> Option<String>) -> String {
        self.render_at(substitute, 0, false)
    }

    fn render_at(&self, substitute: &dyn Fn(&str) -> Option<String>, need: u8, strict: bool) -> String {
        let (text, own) = match self {
            Node::Number { text, unit_text, .. } => (
                if unit_text.is_empty() { text.clone() } else { format!("{text} {unit_text}") },
                self.precedence(),
            ),
            Node::Date { text, .. } => (text.clone(), 5),
            Node::Symbol { name, .. } => match substitute(name) {
                Some(value) => {
                    let own = if value.starts_with('-') {
                        3
                    } else if value.contains(' ') {
                        2
                    } else {
                        5
                    };
                    (value, own)
                }
                None => (name.clone(), 5),
            },
            Node::Neg { inner, .. } => (format!("-{}", inner.render_at(substitute, 3, false)), 3),
            Node::Binary { op, left, right, .. } => {
                let p = op.precedence();
                let strict_right = matches!(op, BinOp::Sub | BinOp::Div);
                let right_text = right.render_at(substitute, p, strict_right);
                // `a - -3 mm` reads as a typo; a negative right operand is
                // always bracketed.
                let right_text = if right_text.starts_with('-') { format!("({right_text})") } else { right_text };
                (format!("{} {} {right_text}", left.render_at(substitute, p, false), op.symbol()), p)
            }
            Node::Power { base, exponent_text, .. } => {
                (format!("{}^{exponent_text}", base.render_at(substitute, 5, true)), 4)
            }
            Node::Call { func, args, .. } => (
                format!(
                    "{}({})",
                    func.name(),
                    args.iter().map(|a| a.render_at(substitute, 0, false)).collect::<Vec<_>>().join(", ")
                ),
                5,
            ),
        };
        if own < need || (strict && own == need) {
            format!("({text})")
        } else {
            text
        }
    }
}

/// Why an expression could not be read.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseProblem {
    pub message: String,
    pub at: Option<usize>,
    /// True when the problem is a unit, which is reported as its own code.
    pub unit: bool,
}

impl ParseProblem {
    fn at(at: usize, message: impl Into<String>) -> Self {
        Self { message: message.into(), at: Some(at), unit: false }
    }
}

/// `lhs = rhs`, split on a lone `=`.
pub fn split_equation(text: &str) -> Option<(&str, &str)> {
    let bytes = text.as_bytes();
    let mut found = None;
    for (i, b) in bytes.iter().enumerate() {
        if *b == b'=' {
            let before = i.checked_sub(1).map(|j| bytes[j]);
            let after = bytes.get(i + 1).copied();
            if matches!(before, Some(b'<' | b'>' | b'=' | b'!')) || after == Some(b'=') {
                continue;
            }
            if found.is_some() {
                return None;
            }
            found = Some(i);
        }
    }
    found.map(|i| (&text[..i], &text[i + 1..]))
}

/// Parses one expression, bounded.
pub fn parse(text: &str) -> Result<Node, ParseProblem> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(ParseProblem { message: "There is nothing to calculate.".into(), at: None, unit: false });
    }
    if trimmed.chars().count() > MAX_CHARS {
        return Err(ParseProblem {
            message: format!("The expression is longer than {MAX_CHARS} characters; split it into named steps."),
            at: None,
            unit: false,
        });
    }
    let mut parser = Parser { chars: trimmed.chars().collect(), position: 0, nodes: 0, depth: 0 };
    let node = parser.expression()?;
    parser.skip_spaces();
    if parser.position < parser.chars.len() {
        let rest: String = parser.chars[parser.position..].iter().collect();
        return Err(ParseProblem::at(parser.position, format!("Could not make sense of {rest:?}.")));
    }
    Ok(node)
}

struct Parser {
    chars: Vec<char>,
    position: usize,
    nodes: usize,
    depth: usize,
}

impl Parser {
    fn skip_spaces(&mut self) {
        while self.position < self.chars.len() && self.chars[self.position].is_whitespace() {
            self.position += 1;
        }
    }

    fn peek(&mut self) -> Option<char> {
        self.skip_spaces();
        self.chars.get(self.position).copied()
    }

    fn count(&mut self, at: usize) -> Result<(), ParseProblem> {
        self.nodes += 1;
        if self.nodes > MAX_NODES {
            return Err(ParseProblem::at(at, format!("The expression has more than {MAX_NODES} parts; split it into named steps.")));
        }
        Ok(())
    }

    fn enter(&mut self, at: usize) -> Result<(), ParseProblem> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(ParseProblem::at(at, format!("The expression nests deeper than {MAX_DEPTH} levels.")));
        }
        Ok(())
    }

    fn expression(&mut self) -> Result<Node, ParseProblem> {
        self.enter(self.position)?;
        let mut left = self.term()?;
        while let Some(c) = self.peek() {
            let op = match c {
                '+' => BinOp::Add,
                '-' | '−' => BinOp::Sub,
                _ => break,
            };
            let at = self.position;
            self.position += 1;
            let right = self.term()?;
            self.count(at)?;
            left = Node::Binary { op, left: Box::new(left), right: Box::new(right), at };
        }
        self.depth -= 1;
        Ok(left)
    }

    fn term(&mut self) -> Result<Node, ParseProblem> {
        let mut left = self.unary()?;
        while let Some(c) = self.peek() {
            let op = match c {
                '*' | '×' | '·' => BinOp::Mul,
                '/' | '÷' => BinOp::Div,
                _ => break,
            };
            let at = self.position;
            self.position += 1;
            let right = self.unary()?;
            self.count(at)?;
            left = Node::Binary { op, left: Box::new(left), right: Box::new(right), at };
        }
        Ok(left)
    }

    fn unary(&mut self) -> Result<Node, ParseProblem> {
        match self.peek() {
            Some('-') | Some('−') => {
                let at = self.position;
                self.position += 1;
                self.enter(at)?;
                let inner = self.unary()?;
                self.depth -= 1;
                self.count(at)?;
                Ok(Node::Neg { inner: Box::new(inner), at })
            }
            Some('+') => {
                self.position += 1;
                self.unary()
            }
            _ => self.power(),
        }
    }

    fn power(&mut self) -> Result<Node, ParseProblem> {
        let base = self.primary()?;
        if self.peek() == Some('^') {
            let at = self.position;
            self.position += 1;
            let (exponent, exponent_text) = self.exponent()?;
            self.count(at)?;
            return Ok(Node::Power { base: Box::new(base), exponent, exponent_text, at });
        }
        Ok(base)
    }

    /// A literal exponent: `2`, `-1`, `0.5`, `(1/3)`, `(-2)`. Never an
    /// expression — a power whose exponent depends on an input has no single
    /// dimension.
    fn exponent(&mut self) -> Result<(f64, String), ParseProblem> {
        let at = self.position;
        let bracketed = self.peek() == Some('(');
        if bracketed {
            self.position += 1;
        }
        self.skip_spaces();
        let start = self.position;
        if matches!(self.chars.get(self.position), Some('-') | Some('−')) {
            self.position += 1;
        }
        while self.position < self.chars.len() && (self.chars[self.position].is_ascii_digit() || self.chars[self.position] == '.') {
            self.position += 1;
        }
        let numerator_text: String = self.chars[start..self.position].iter().collect::<String>().replace('−', "-");
        let mut value: f64 = numerator_text
            .parse()
            .map_err(|_| ParseProblem::at(at, "A power must be a number written out, such as ^2 or ^(1/3)."))?;
        let mut text = numerator_text.clone();
        if bracketed {
            self.skip_spaces();
            if self.chars.get(self.position) == Some(&'/') {
                self.position += 1;
                self.skip_spaces();
                let d_start = self.position;
                while self.position < self.chars.len() && self.chars[self.position].is_ascii_digit() {
                    self.position += 1;
                }
                let denominator: String = self.chars[d_start..self.position].iter().collect();
                let d: f64 = denominator
                    .parse()
                    .ok()
                    .filter(|d: &f64| *d > 0.0)
                    .ok_or_else(|| ParseProblem::at(at, "A fractional power needs a positive whole denominator, such as ^(1/3)."))?;
                value /= d;
                text = format!("({numerator_text}/{denominator})");
            } else {
                text = format!("({numerator_text})");
            }
            self.skip_spaces();
            if self.chars.get(self.position) != Some(&')') {
                return Err(ParseProblem::at(at, "A bracketed power was never closed."));
            }
            self.position += 1;
        }
        if value.abs() > MAX_EXPONENT {
            return Err(ParseProblem::at(at, format!("A power larger than {MAX_EXPONENT} is outside what this engine evaluates.")));
        }
        Ok((value, text))
    }

    fn primary(&mut self) -> Result<Node, ParseProblem> {
        let at = match self.peek() {
            None => {
                return Err(ParseProblem { message: "The expression ended unexpectedly.".into(), at: None, unit: false })
            }
            Some(_) => self.position,
        };
        self.count(at)?;
        let c = self.chars[at];
        if c == '(' {
            self.position += 1;
            let inner = self.expression()?;
            if self.peek() != Some(')') {
                return Err(ParseProblem::at(self.position, "A bracket was opened and never closed."));
            }
            self.position += 1;
            return Ok(inner);
        }
        if let Some(date) = self.date(at)? {
            return Ok(date);
        }
        if c.is_ascii_digit() || c == '.' {
            return self.number(at);
        }
        if c.is_alphabetic() || c == '_' {
            let start = self.position;
            while self.position < self.chars.len() && (self.chars[self.position].is_alphanumeric() || self.chars[self.position] == '_') {
                self.position += 1;
            }
            let name: String = self.chars[start..self.position].iter().collect();
            if name.chars().count() > 32 {
                return Err(ParseProblem::at(start, format!("The name {name:?} is longer than 32 characters.")));
            }
            if self.peek() == Some('(') {
                let func = Func::ALL
                    .iter()
                    .find(|(n, _)| *n == name)
                    .map(|(_, f)| *f)
                    .ok_or_else(|| ParseProblem::at(start, format!("{name}() is not a function this engine has. It has: sqrt, abs, min, max, ln, log10, exp, sin, cos, tan and absolute(gauge, atmosphere).")))?;
                self.position += 1;
                let mut args = Vec::new();
                loop {
                    args.push(self.expression()?);
                    match self.peek() {
                        Some(',') => {
                            self.position += 1;
                        }
                        Some(')') => {
                            self.position += 1;
                            break;
                        }
                        _ => return Err(ParseProblem::at(self.position, format!("{name}( was never closed."))),
                    }
                    if args.len() >= MAX_ARGS {
                        return Err(ParseProblem::at(start, format!("{name}() takes at most {MAX_ARGS} arguments.")));
                    }
                }
                let (least, most) = func.arity();
                if args.len() < least || args.len() > most {
                    return Err(ParseProblem::at(start, format!("{name}() takes {} argument(s), and was given {}.", if least == most { least.to_string() } else { format!("{least} to {most}") }, args.len())));
                }
                return Ok(Node::Call { func, args, at: start });
            }
            return Ok(Node::Symbol { name, at: start });
        }
        Err(ParseProblem::at(at, format!("Did not expect {c:?} here.")))
    }

    /// `2026-08-12`, exactly: four digits, two, two, and not followed by a digit.
    fn date(&mut self, at: usize) -> Result<Option<Node>, ParseProblem> {
        let window: String = self.chars[at..(at + 10).min(self.chars.len())].iter().collect();
        let shape = window.len() == 10
            && window.char_indices().all(|(i, c)| if i == 4 || i == 7 { c == '-' } else { c.is_ascii_digit() });
        if !shape || self.chars.get(at + 10).is_some_and(|c| c.is_ascii_digit()) {
            return Ok(None);
        }
        let year: i64 = window[..4].parse().unwrap_or(0);
        let month: u32 = window[5..7].parse().unwrap_or(0);
        let day: u32 = window[8..10].parse().unwrap_or(0);
        let Some(days) = days_from_civil(year, month, day) else {
            return Err(ParseProblem::at(at, format!("{window} is not a calendar date.")));
        };
        self.position = at + 10;
        Ok(Some(Node::Date { text: window, day: days, at }))
    }

    fn number(&mut self, at: usize) -> Result<Node, ParseProblem> {
        let start = self.position;
        while self.position < self.chars.len() && (self.chars[self.position].is_ascii_digit() || self.chars[self.position] == '.') {
            self.position += 1;
        }
        // An exponent: `1e-3`, `2.5E6`. Only when a digit follows, so `2 eV`
        // would not be misread (and is not a unit here anyway).
        if matches!(self.chars.get(self.position), Some('e') | Some('E')) {
            let mut probe = self.position + 1;
            if matches!(self.chars.get(probe), Some('-') | Some('+')) {
                probe += 1;
            }
            if self.chars.get(probe).is_some_and(|c| c.is_ascii_digit()) {
                self.position = probe;
                while self.position < self.chars.len() && self.chars[self.position].is_ascii_digit() {
                    self.position += 1;
                }
            }
        }
        let text: String = self.chars[start..self.position].iter().collect();
        let value = Decimal::parse(&text).ok_or_else(|| ParseProblem::at(start, format!("{text:?} is not a number.")))?;

        // The unit, if one follows.
        let mut probe = self.position;
        while probe < self.chars.len() && self.chars[probe].is_whitespace() {
            probe += 1;
        }
        if probe >= self.chars.len() || !is_unit_char(self.chars[probe]) {
            return Ok(Node::Number { text, value, unit: UnitExpr::default(), unit_text: String::new(), at });
        }
        let unit_start = probe;
        let mut end = probe;
        loop {
            while end < self.chars.len() && is_unit_char(self.chars[end]) {
                end += 1;
            }
            // A power: ^n, ², ³, or digits straight after (`m3`).
            if end < self.chars.len() && (self.chars[end] == '²' || self.chars[end] == '³') {
                end += 1;
            } else if end < self.chars.len() && self.chars[end] == '^' {
                let mut look = end + 1;
                if matches!(self.chars.get(look), Some('-')) {
                    look += 1;
                }
                if self.chars.get(look).is_some_and(|c| c.is_ascii_digit()) {
                    end = look;
                    while end < self.chars.len() && self.chars[end].is_ascii_digit() {
                        end += 1;
                    }
                }
            } else if end < self.chars.len() && self.chars[end].is_ascii_digit() {
                while end < self.chars.len() && self.chars[end].is_ascii_digit() {
                    end += 1;
                }
            }
            // A compound continues only with no space: `kg/m^3`, `W/(m·K)`.
            let next = self.chars.get(end).copied();
            let after = self.chars.get(end + 1).copied();
            match (next, after) {
                (Some('/' | '*' | '·'), Some(a)) if is_unit_char(a) => {
                    end += 1;
                }
                (Some('/'), Some('(')) => {
                    let close = self.chars[end..].iter().position(|c| *c == ')').map(|i| i + end);
                    match close {
                        Some(close) => {
                            end = close + 1;
                            break;
                        }
                        None => break,
                    }
                }
                _ => break,
            }
        }
        let unit_text: String = self.chars[unit_start..end].iter().collect();

        // A unit is followed by an operator, a bracket, a comma or the end. A
        // second bare word after it means this is prose that happens to
        // contain a number — "2 and then some" — not "8.2 mm".
        let mut after = end;
        while after < self.chars.len() && self.chars[after].is_whitespace() {
            after += 1;
        }
        if self.chars.get(after).is_some_and(|c| c.is_alphabetic()) {
            return Err(ParseProblem {
                message: format!(
                    "{unit_text:?} does not read as a unit here. A calculation should be an expression \
                     such as `(8.2 mm - 9.0 mm) / 9.0 mm * 100`, not a sentence."
                ),
                at: Some(unit_start),
                unit: true,
            });
        }
        let unit = parse_unit(&unit_text).map_err(|problem| ParseProblem {
            message: problem.explain(),
            at: Some(unit_start),
            unit: true,
        })?;
        self.position = end;
        Ok(Node::Number { text, value, unit, unit_text, at })
    }
}

/// Days since 1970-01-01 for a proleptic Gregorian date (H. Hinnant's
/// algorithm), or `None` for a date that does not exist.
pub fn days_from_civil(year: i64, month: u32, day: u32) -> Option<i64> {
    if !(1..=12).contains(&month) || day == 0 || !(1..=9999).contains(&year) {
        return None;
    }
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let length = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month as usize - 1];
    if day > length {
        return None;
    }
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let m = month as i64;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + day as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146_097 + doe - 719_468)
}

/// The date `days` after 1970-01-01, as `YYYY-MM-DD`.
pub fn civil_from_days(days: i64) -> String {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn units_follow_their_numbers_and_operators_need_spaces_or_digits() {
        let node = parse("1500 kg/m^3 * 2 m^3").unwrap();
        assert_eq!(node.literals(), vec!["1500 kg/m^3", "2 m^3"]);
        let division = parse("10 kg / 2 s").unwrap();
        assert!(matches!(division, Node::Binary { op: BinOp::Div, .. }));
        assert_eq!(parse("20 W/(m·K)").unwrap().literals(), vec!["20 W/(m·K)"]);
    }

    #[test]
    fn symbols_functions_and_powers_parse() {
        let node = parse("sqrt(t_min^2 + 4 * x) - max(a, b, 3 mm)").unwrap();
        assert_eq!(node.symbols(), vec!["t_min", "x", "a", "b"]);
        assert!(parse("unknownfn(2)").unwrap_err().message.contains("not a function"));
        assert!(parse("x^y").is_err(), "a power is a literal");
        assert!(parse("x^(1/3)").is_ok());
        assert!(parse("x^20").unwrap_err().message.contains("larger than"));
    }

    #[test]
    fn a_date_is_a_literal_and_an_impossible_one_is_refused() {
        assert!(matches!(parse("2026-08-12").unwrap(), Node::Date { .. }));
        assert!(parse("2026-02-30").unwrap_err().message.contains("not a calendar date"));
        assert_eq!(civil_from_days(days_from_civil(2026, 8, 12).unwrap() + 90), "2026-11-10");
        assert_eq!(days_from_civil(2026, 8, 12).unwrap() - days_from_civil(2024, 3, 18).unwrap(), 877);
        assert_eq!(days_from_civil(1970, 1, 1), Some(0));
    }

    #[test]
    fn a_sentence_is_not_an_expression() {
        for prose in ["the wall is 8 mm thick", "2 and then some", "measured 8.2 mm at four points", "2 x"] {
            assert!(parse(prose).is_err(), "{prose:?}");
        }
    }

    #[test]
    fn the_bounds_refuse_rather_than_work_forever() {
        let long = "1 + ".repeat(300) + "1";
        assert!(parse(&long).is_err());
        let deep = "(".repeat(40) + "1" + &")".repeat(40);
        assert!(parse(&deep).unwrap_err().message.contains("deeper"));
    }

    #[test]
    fn a_rendering_brackets_only_where_the_position_needs_it() {
        let node = parse("(t_min - t_meas) / t_min * 100").unwrap();
        let text = node.render(&|name| match name {
            "t_min" => Some("9.0 mm".to_string()),
            "t_meas" => Some("8.2 mm".to_string()),
            _ => None,
        });
        assert_eq!(text, "(9.0 mm - 8.2 mm) / (9.0 mm) * 100");
        let negative = parse("a - b").unwrap().render(&|n| (n == "b").then(|| "-3 mm".to_string()));
        assert_eq!(negative, "a - (-3 mm)");
    }

    #[test]
    fn an_equation_splits_on_a_lone_equals() {
        assert_eq!(split_equation("a = b + 1"), Some(("a ", " b + 1")));
        assert_eq!(split_equation("a <= b"), None);
        assert_eq!(split_equation("a = b = c"), None);
    }
}

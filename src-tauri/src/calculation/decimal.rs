//! Exact decimals, as written, and rounding that does what a reader expects.
//!
//! A source says `8.20 mm`. Parsed to `f64` that is `8.199999999999999289…`,
//! and written back it is `8.2` — the trailing zero, which said the measurement
//! was taken to a hundredth, is gone. So an input's value is kept here as the
//! digits it was written with, and only converted to binary for arithmetic.
//!
//! Rounding for display works on decimal digits, too. Binary rounding of
//! `2.675` to two places gives `2.67`, because the double nearest `2.675` is
//! just below it; nobody reading a calculation sheet expects that. The result
//! of a computation is first written as the shortest decimal that reads back
//! to the same double — Rust's own `{:e}` formatting guarantees that — and that
//! decimal is rounded, half away from zero.

use serde::{Deserialize, Serialize};

/// A decimal number: `digits × 10^exponent`, with its sign.
///
/// The digits are exactly those written, trailing zeros included: `8.20` is
/// `[8, 2, 0] × 10^-2`, and says something `8.2` does not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decimal {
    negative: bool,
    digits: Vec<u8>,
    exponent: i32,
}

/// How a result is written down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "rule", content = "n")]
pub enum Rounding {
    /// This many significant figures.
    Significant(u8),
    /// This many digits after the decimal point.
    Places(u8),
}

impl Rounding {
    /// The default: four significant figures.
    ///
    /// Enough for the tolerances in an inspection report, and short enough
    /// that a reader is not misled into thinking an input was that precise.
    pub const DEFAULT: Rounding = Rounding::Significant(4);

    /// `4sf`, `3 sig`, `2dp`, `0 dp`. Bounded to twelve.
    pub fn parse(text: &str) -> Option<Rounding> {
        let compact: String = text.chars().filter(|c| !c.is_whitespace()).collect::<String>().to_ascii_lowercase();
        let split = compact.find(|c: char| !c.is_ascii_digit())?;
        let (number, rule) = compact.split_at(split);
        let n: u8 = number.parse().ok()?;
        match rule {
            "sf" | "sig" | "s.f." | "significant" if (1..=12).contains(&n) => Some(Rounding::Significant(n)),
            "dp" | "d.p." | "places" | "decimals" if n <= 12 => Some(Rounding::Places(n)),
            _ => None,
        }
    }

    pub fn describe(self) -> String {
        match self {
            Rounding::Significant(n) => format!("{n} significant figures, half away from zero"),
            Rounding::Places(n) => format!("{n} decimal places, half away from zero"),
        }
    }
}

impl Decimal {
    /// `[-+]digits[.digits][e[-+]digits]`, nothing else: no spaces, no
    /// thousands separators, no `inf`/`nan`.
    pub fn parse(text: &str) -> Option<Decimal> {
        let text = text.trim();
        let (negative, body) = match text.as_bytes().first()? {
            b'-' => (true, &text[1..]),
            b'+' => (false, &text[1..]),
            _ => (false, text),
        };
        let (mantissa, exponent_part) = match body.find(['e', 'E']) {
            Some(at) => (&body[..at], Some(&body[at + 1..])),
            None => (body, None),
        };
        let (whole, fraction) = match mantissa.split_once('.') {
            Some((whole, fraction)) => (whole, fraction),
            None => (mantissa, ""),
        };
        if whole.is_empty() && fraction.is_empty() {
            return None;
        }
        if !whole.bytes().all(|b| b.is_ascii_digit()) || !fraction.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let mut exponent: i32 = match exponent_part {
            Some(raw) if !raw.is_empty() => raw.parse::<i32>().ok().filter(|e| e.abs() <= 400)?,
            Some(_) => return None,
            None => 0,
        };
        exponent -= fraction.len() as i32;
        let mut digits: Vec<u8> = whole.bytes().chain(fraction.bytes()).map(|b| b - b'0').collect();
        // Leading zeros carry nothing. Trailing ones do, and stay.
        while digits.len() > 1 && digits[0] == 0 {
            digits.remove(0);
        }
        if digits.len() > 40 {
            return None;
        }
        Some(Decimal { negative: negative && digits.iter().any(|d| *d != 0), digits, exponent })
    }

    /// The shortest decimal that reads back as `value`.
    pub fn from_f64(value: f64) -> Option<Decimal> {
        if !value.is_finite() {
            return None;
        }
        if value == 0.0 {
            return Some(Decimal { negative: false, digits: vec![0], exponent: 0 });
        }
        Decimal::parse(&format!("{value:e}"))
    }

    pub fn to_f64(&self) -> f64 {
        self.to_scientific().parse().unwrap_or(f64::NAN)
    }

    pub fn is_zero(&self) -> bool {
        self.digits.iter().all(|d| *d == 0)
    }

    pub fn is_negative(&self) -> bool {
        self.negative
    }

    /// Significant figures as written: `9.0` has two, `8.20` three, `0.050`
    /// two. Trailing zeros of a whole number without a point (`100`) are not
    /// counted, which is the usual reading and the conservative one.
    pub fn significant_figures(&self) -> usize {
        let first = self.digits.iter().position(|d| *d != 0);
        let Some(first) = first else { return 1 };
        let mut last = self.digits.len();
        if self.exponent >= 0 {
            while last > first + 1 && self.digits[last - 1] == 0 {
                last -= 1;
            }
        }
        last - first
    }

    /// Rounded half away from zero, on the decimal digits.
    pub fn round(&self, rule: Rounding) -> Decimal {
        if self.is_zero() {
            return match rule {
                Rounding::Places(n) => Decimal { negative: false, digits: vec![0; n as usize + 1], exponent: -(n as i32) },
                Rounding::Significant(_) => Decimal { negative: false, digits: vec![0], exponent: 0 },
            };
        }
        // The exponent of the last digit to keep.
        let keep_exponent = match rule {
            Rounding::Places(n) => -(n as i32),
            Rounding::Significant(n) => {
                let first = self.digits.iter().position(|d| *d != 0).unwrap_or(0);
                let magnitude = self.exponent + (self.digits.len() - 1 - first) as i32;
                magnitude - (n as i32 - 1)
            }
        };
        if keep_exponent <= self.exponent {
            // Nothing to cut; pad with zeros so the rule is visible.
            let mut digits = self.digits.clone();
            digits.extend(std::iter::repeat_n(0, (self.exponent - keep_exponent) as usize));
            return Decimal { negative: self.negative, digits, exponent: keep_exponent }.normalised();
        }
        let cut = (keep_exponent - self.exponent) as usize;
        let (kept, dropped): (Vec<u8>, Vec<u8>) = if cut >= self.digits.len() {
            let mut padded = vec![0; cut - self.digits.len()];
            padded.extend(&self.digits);
            (vec![0], padded)
        } else {
            (self.digits[..self.digits.len() - cut].to_vec(), self.digits[self.digits.len() - cut..].to_vec())
        };
        let mut kept = if kept.is_empty() { vec![0] } else { kept };
        if dropped.first().is_some_and(|d| *d >= 5) {
            // Half away from zero: a 5 rounds the magnitude up whatever follows.
            let mut i = kept.len();
            loop {
                if i == 0 {
                    kept.insert(0, 1);
                    break;
                }
                i -= 1;
                if kept[i] == 9 {
                    kept[i] = 0;
                } else {
                    kept[i] += 1;
                    break;
                }
            }
        }
        let rounded = Decimal { negative: self.negative, digits: kept, exponent: keep_exponent };
        // Carrying can add a digit: 9.9996 at four figures is 10.00, not 10.000.
        match rule {
            Rounding::Significant(n) if rounded.significant_digits_len() > n as usize => {
                let mut trimmed = rounded.clone();
                trimmed.digits.pop();
                trimmed.exponent += 1;
                trimmed.normalised()
            }
            _ => rounded.normalised(),
        }
    }

    fn significant_digits_len(&self) -> usize {
        let first = self.digits.iter().position(|d| *d != 0).unwrap_or(self.digits.len() - 1);
        self.digits.len() - first
    }

    fn normalised(mut self) -> Decimal {
        while self.digits.len() > 1 && self.digits[0] == 0 && (self.digits.len() as i32 + self.exponent) > 1 {
            self.digits.remove(0);
        }
        if self.is_zero() {
            self.negative = false;
        }
        self
    }

    /// Written out in full, keeping every digit (`8.20`, `-0.0050`, `1200`).
    ///
    /// Scientific notation only outside `1e-6 ≤ |x| < 1e15`, where a plain
    /// rendering would be a line of zeros nobody can count.
    pub fn to_plain(&self) -> String {
        let magnitude = self.exponent + self.digits.len() as i32 - 1;
        if !self.is_zero() && !(-6..15).contains(&magnitude) {
            return self.to_scientific();
        }
        let digits: String = self.digits.iter().map(|d| (b'0' + d) as char).collect();
        let body = if self.exponent >= 0 {
            format!("{digits}{}", "0".repeat(self.exponent as usize))
        } else {
            let point = digits.len() as i32 + self.exponent;
            if point > 0 {
                format!("{}.{}", &digits[..point as usize], &digits[point as usize..])
            } else {
                format!("0.{}{digits}", "0".repeat((-point) as usize))
            }
        };
        if self.negative {
            format!("-{body}")
        } else {
            body
        }
    }

    /// [`Self::to_plain`] without trailing zeros after the point.
    pub fn to_trimmed(&self) -> String {
        let plain = self.to_plain();
        if plain.contains('e') || !plain.contains('.') {
            return plain;
        }
        let trimmed = plain.trim_end_matches('0').trim_end_matches('.');
        if trimmed == "-0" || trimmed.is_empty() {
            "0".to_string()
        } else {
            trimmed.to_string()
        }
    }

    pub fn to_scientific(&self) -> String {
        let digits: String = self.digits.iter().map(|d| (b'0' + d) as char).collect();
        let first = digits.find(|c| c != '0').unwrap_or(digits.len() - 1);
        let significant = &digits[first..];
        let exponent = self.exponent + significant.len() as i32 - 1;
        let mantissa = if significant.len() > 1 {
            format!("{}.{}", &significant[..1], &significant[1..])
        } else {
            significant.to_string()
        };
        format!("{}{mantissa}e{exponent}", if self.negative { "-" } else { "" })
    }
}

/// A computed value, rounded for display by `rule`.
///
/// `trim` drops trailing zeros (the engine's default rendering, `0.3` rather
/// than `0.3000`); a caller who asked for a rule keeps them, because `8.80`
/// at two places is what they asked to see.
pub fn display(value: f64, rule: Rounding, trim: bool) -> Option<String> {
    let rounded = Decimal::from_f64(value)?.round(rule);
    Some(if trim { rounded.to_trimmed() } else { rounded.to_plain() })
}

/// The shortest round-trip rendering, for the record's raw result.
pub fn raw(value: f64) -> String {
    match Decimal::from_f64(value) {
        Some(decimal) => decimal.to_plain(),
        None => value.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(text: &str) -> Decimal {
        Decimal::parse(text).expect(text)
    }

    #[test]
    fn a_value_keeps_the_digits_it_was_written_with() {
        assert_eq!(d("8.20").to_plain(), "8.20");
        assert_eq!(d("-0.0050").to_plain(), "-0.0050");
        assert_eq!(d("1200").to_plain(), "1200");
        assert_eq!(d("9.0").to_f64(), 9.0);
        assert_eq!(d("1.5e3").to_plain(), "1500");
    }

    #[test]
    fn significant_figures_are_read_from_what_was_written() {
        assert_eq!(d("9.0").significant_figures(), 2);
        assert_eq!(d("8.20").significant_figures(), 3);
        assert_eq!(d("0.050").significant_figures(), 2);
        assert_eq!(d("100").significant_figures(), 1);
    }

    #[test]
    fn rounding_is_decimal_and_half_away_from_zero() {
        // The double nearest 2.675 is below it; decimal rounding still gives 2.68.
        assert_eq!(display(2.675, Rounding::Places(2), false).unwrap(), "2.68");
        assert_eq!(display(-2.675, Rounding::Places(2), false).unwrap(), "-2.68");
        assert_eq!(display(0.125, Rounding::Significant(2), false).unwrap(), "0.13");
        assert_eq!(display(8.888888888888889, Rounding::Significant(4), true).unwrap(), "8.889");
        assert_eq!(display(9.99996, Rounding::Significant(4), false).unwrap(), "10.00");
        assert_eq!(display(0.30000000000000004, Rounding::Significant(4), true).unwrap(), "0.3");
        assert_eq!(display(8.8, Rounding::Places(2), false).unwrap(), "8.80");
        assert_eq!(display(1234.5, Rounding::Significant(2), true).unwrap(), "1200");
        assert_eq!(display(0.0, Rounding::Places(2), false).unwrap(), "0.00");
    }

    #[test]
    fn the_raw_value_is_the_shortest_round_trip() {
        assert_eq!(raw(0.1 + 0.2), "0.30000000000000004");
        assert_eq!(raw(8.88888888888889), "8.88888888888889");
        assert_eq!(raw(1e-9), "1e-9");
    }

    #[test]
    fn nothing_that_is_not_a_plain_decimal_parses() {
        for bad in ["", "1,000", "inf", "NaN", "1.2.3", "--1", "e5", "1e", "0x10"] {
            assert!(Decimal::parse(bad).is_none(), "{bad:?}");
        }
    }

    #[test]
    fn a_rounding_rule_is_parsed_and_bounded() {
        assert_eq!(Rounding::parse("4sf"), Some(Rounding::Significant(4)));
        assert_eq!(Rounding::parse("2 dp"), Some(Rounding::Places(2)));
        assert_eq!(Rounding::parse("0dp"), Some(Rounding::Places(0)));
        assert_eq!(Rounding::parse("0sf"), None);
        assert_eq!(Rounding::parse("40sf"), None);
        assert_eq!(Rounding::parse("round nicely"), None);
    }
}

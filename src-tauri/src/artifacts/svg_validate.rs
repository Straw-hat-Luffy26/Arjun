//! Reading a produced SVG back, to find out whether it is really a drawing.
//!
//! ## Why this exists
//!
//! A diagram was checked by `Kind::Diagram => report(true, "Present, N
//! byte(s).")` — the same "the file exists" check a PDF got. An SVG whose root
//! element never closed, or that declared a `viewBox` and then drew everything
//! outside it, or that referenced a marker it never defined, passed that and
//! was shown to somebody as a finished drawing.
//!
//! ## What is checked
//!
//! - the document is well-formed: every element that opens is closed, in order
//! - the root is `<svg>` and carries a `viewBox` with four usable numbers
//! - the shapes and labels it actually contains
//! - every internal reference resolves — `marker-end="url(#arw)"` against the
//!   `<marker id="arw">` the writer emits in `<defs>`
//! - every `<text>` element has text in it
//! - nothing is drawn outside the `viewBox`
//! - the drawing has an accessible name
//!
//! ## What is not checked
//!
//! Whether the drawing is *legible*: whether two labels overlap, whether the
//! edges cross more than they need to, whether the layout reads well. Those are
//! real questions and this cannot answer them honestly, so it does not pretend
//! to. The two that geometry can reach are in [`quality`]; the rest are named
//! as a gap rather than faked here.
//!
//! ## The parser
//!
//! A small well-formedness scanner rather than a dependency. The writers emit a
//! known subset — no namespaces beyond the default, no CDATA, no entities
//! beyond the five `esc` produces — and a scanner for that subset is about a
//! hundred and fifty lines. A general XML crate would be more capable and would
//! also be a new dependency in a product that ships an SBOM for an air-gapped
//! deployment, which is a cost this does not need to pay.
//!
//! A file this cannot parse is reported as one it cannot parse, never as one
//! that passed.

use std::collections::BTreeSet;
use std::path::Path;

/// What reading an SVG back found.
#[derive(Debug, Clone, Default)]
pub struct SvgCheck {
    /// Whether the document is well-formed and rooted in `<svg>`.
    pub parses: bool,
    /// `viewBox` as four numbers, if it has a usable one.
    pub view_box: Option<(f64, f64, f64, f64)>,
    /// Drawn elements: rects, paths, lines, circles, polygons.
    pub shapes: usize,
    /// `<text>` elements carrying text.
    pub labels: usize,
    /// Whether the root carries an accessible name.
    pub has_accessible_name: bool,
    pub problems: Vec<String>,
}

impl SvgCheck {
    pub fn is_sound(&self) -> bool {
        self.problems.is_empty() && self.parses
    }
}

pub fn check_svg(path: &Path) -> SvgCheck {
    match std::fs::read_to_string(path) {
        Ok(text) => check_svg_text(&text),
        Err(error) => SvgCheck {
            problems: vec![format!("the file could not be read: {error}")],
            ..Default::default()
        },
    }
}

/// One element as the scanner saw it.
#[derive(Debug, Clone)]
struct Element {
    name: String,
    attributes: Vec<(String, String)>,
    /// Text directly inside it.
    text: String,
}

impl Element {
    fn attribute(&self, key: &str) -> Option<&str> {
        self.attributes.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }
}

/// Shapes that put ink on the page.
const SHAPES: &[&str] = &["rect", "path", "line", "circle", "ellipse", "polygon", "polyline"];

/// Elements whose contents are not markup and must be skipped wholesale.
const OPAQUE: &[&str] = &["style", "script"];

/// Scans the document. `Err` only when it is not well-formed.
fn scan(text: &str) -> Result<Vec<Element>, String> {
    let chars: Vec<char> = text.chars().collect();
    let mut elements: Vec<Element> = Vec::new();
    // (name, index into `elements`) for everything currently open.
    let mut open: Vec<(String, usize)> = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] != '<' {
            let start = i;
            while i < chars.len() && chars[i] != '<' {
                i += 1;
            }
            // Character data belongs to the innermost open element.
            if let Some((_, index)) = open.last() {
                let run: String = chars[start..i].iter().collect();
                elements[*index].text.push_str(&run);
            }
            continue;
        }

        // A comment.
        if chars[i..].starts_with(&['<', '!', '-', '-'][..]) {
            let rest: String = chars[i..].iter().collect();
            let end = rest.find("-->").ok_or("a comment is never closed")?;
            i += rest[..end + 3].chars().count();
            continue;
        }
        // A declaration or processing instruction.
        if i + 1 < chars.len() && (chars[i + 1] == '!' || chars[i + 1] == '?') {
            while i < chars.len() && chars[i] != '>' {
                i += 1;
            }
            i += 1;
            continue;
        }

        // A closing tag.
        if i + 1 < chars.len() && chars[i + 1] == '/' {
            let mut j = i + 2;
            let start = j;
            while j < chars.len() && chars[j] != '>' {
                j += 1;
            }
            if j >= chars.len() {
                return Err("a closing tag is never terminated".to_string());
            }
            let name = chars[start..j].iter().collect::<String>().trim().to_string();
            match open.pop() {
                Some((expected, _)) if expected == name => {}
                Some((expected, _)) => {
                    return Err(format!(
                        "</{name}> closes while <{expected}> is still open, so the document is \
                         not well formed"
                    ))
                }
                None => return Err(format!("</{name}> closes an element that was never opened")),
            }
            i = j + 1;
            continue;
        }

        // An opening tag.
        let mut j = i + 1;
        let start = j;
        while j < chars.len() && !chars[j].is_whitespace() && chars[j] != '>' && chars[j] != '/' {
            j += 1;
        }
        let name: String = chars[start..j].iter().collect();
        if name.is_empty() {
            return Err("a tag has no name".to_string());
        }

        // Attributes. A quoted value may contain `>` — `aria-label` holds
        // arbitrary text — so the scan tracks quoting rather than looking for
        // the next angle bracket.
        let mut attributes = Vec::new();
        let mut self_closing = false;
        loop {
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j >= chars.len() {
                return Err(format!("<{name}> is never terminated"));
            }
            if chars[j] == '/' {
                self_closing = true;
                j += 1;
                continue;
            }
            if chars[j] == '>' {
                j += 1;
                break;
            }
            let key_start = j;
            while j < chars.len()
                && chars[j] != '='
                && !chars[j].is_whitespace()
                && chars[j] != '>'
                && chars[j] != '/'
            {
                j += 1;
            }
            let key: String = chars[key_start..j].iter().collect();
            if key.is_empty() {
                return Err(format!("<{name}> has a malformed attribute"));
            }
            let mut k = j;
            while k < chars.len() && chars[k].is_whitespace() {
                k += 1;
            }
            if k < chars.len() && chars[k] == '=' {
                k += 1;
                while k < chars.len() && chars[k].is_whitespace() {
                    k += 1;
                }
                if k >= chars.len() {
                    return Err(format!("<{name}> has an attribute with no value"));
                }
                let quote = chars[k];
                if quote != '"' && quote != '\'' {
                    return Err(format!("<{name}>'s {key:?} attribute is not quoted"));
                }
                k += 1;
                let value_start = k;
                while k < chars.len() && chars[k] != quote {
                    k += 1;
                }
                if k >= chars.len() {
                    return Err(format!("<{name}>'s {key:?} attribute is never closed"));
                }
                attributes.push((key, chars[value_start..k].iter().collect::<String>()));
                j = k + 1;
            } else {
                attributes.push((key, String::new()));
            }
        }

        let index = elements.len();
        elements.push(Element { name: name.clone(), attributes, text: String::new() });

        if !self_closing {
            if OPAQUE.contains(&name.as_str()) {
                // Skip to the matching close without interpreting anything
                // between: a stylesheet contains `{`, `>` and quotes that are
                // not markup, and reading them as markup would reject every
                // drawing this product writes.
                let rest: String = chars[j..].iter().collect();
                let close = format!("</{name}>");
                let end = rest.find(&close).ok_or_else(|| format!("<{name}> is never closed"))?;
                elements[index].text.push_str(&rest[..end]);
                j += rest[..end + close.len()].chars().count();
            } else {
                open.push((name, index));
            }
        }
        i = j;
    }

    if let Some((unclosed, _)) = open.pop() {
        return Err(format!("<{unclosed}> is never closed"));
    }
    Ok(elements)
}

/// Well-formedness only, for a caller that is not checking an SVG.
///
/// Returns how many elements the document holds. `artifacts::text_formats`
/// checks XML with this rather than with a second scanner: two parsers for the
/// same subset eventually disagree about the same document, and then the bug is
/// in whichever one the caller happened to use.
pub fn scan_for_wellformedness(text: &str) -> Result<usize, String> {
    scan(text).map(|elements| elements.len())
}

fn numbers(value: &str) -> Vec<f64> {
    value
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse().ok())
        .collect()
}

pub fn check_svg_text(text: &str) -> SvgCheck {
    let mut check = SvgCheck::default();

    if text.trim().is_empty() {
        check.problems.push("the file is empty".to_string());
        return check;
    }

    let elements = match scan(text) {
        Ok(elements) => elements,
        Err(problem) => {
            check.problems.push(problem);
            return check;
        }
    };

    let Some(root) = elements.first() else {
        check.problems.push("the file contains no elements".to_string());
        return check;
    };
    if root.name != "svg" {
        check.problems.push(format!("the root element is <{}>, not <svg>", root.name));
        return check;
    }
    check.parses = true;

    // The viewBox. Without it a drawing has no coordinate system and scales to
    // whatever the viewer guesses.
    match root.attribute("viewBox") {
        Some(raw) => {
            let values = numbers(raw);
            if values.len() != 4 {
                check.problems.push(format!("the viewBox {raw:?} is not four numbers"));
            } else if values[2] <= 0.0 || values[3] <= 0.0 {
                check.problems.push(format!(
                    "the viewBox has a width of {} and a height of {}, so it encloses nothing",
                    values[2], values[3]
                ));
            } else {
                check.view_box = Some((values[0], values[1], values[2], values[3]));
            }
        }
        None => check
            .problems
            .push("the drawing has no viewBox, so it has no coordinate system".to_string()),
    }

    check.has_accessible_name = root
        .attribute("aria-label")
        .is_some_and(|v| !v.trim().is_empty())
        || elements.iter().any(|e| e.name == "title" && !e.text.trim().is_empty());
    if !check.has_accessible_name {
        check.problems.push(
            "the drawing has no accessible name: no aria-label on the root and no <title>"
                .to_string(),
        );
    }

    // Identifiers, and the references that must resolve against them.
    let mut ids: BTreeSet<&str> = BTreeSet::new();
    for element in &elements {
        if let Some(id) = element.attribute("id") {
            if !ids.insert(id) {
                check.problems.push(format!("the id {id:?} is defined more than once"));
            }
        }
    }

    let mut empty_labels = 0usize;
    let mut outside: Vec<String> = Vec::new();
    for element in &elements {
        if SHAPES.contains(&element.name.as_str()) {
            check.shapes += 1;
        }
        if element.name == "text" {
            if element.text.trim().is_empty() {
                empty_labels += 1;
            } else {
                check.labels += 1;
            }
        }

        for (key, value) in &element.attributes {
            let target = if value.starts_with("url(#") && value.ends_with(')') {
                Some(&value[5..value.len() - 1])
            } else if (key == "href" || key == "xlink:href") && value.starts_with('#') {
                Some(&value[1..])
            } else {
                None
            };
            if let Some(target) = target {
                if !ids.contains(target) {
                    check.problems.push(format!(
                        "<{}> references {target:?}, which nothing in the drawing defines",
                        element.name
                    ));
                }
            }
        }

        // Geometry, against the viewBox. A rotated label's own coordinates are
        // in the rotated frame, so it cannot be judged this way; nor can
        // anything inside `<defs>`, which is never drawn where it is declared.
        if let Some((min_x, min_y, width, height)) = check.view_box {
            let transformed = element.attribute("transform").is_some();
            let in_defs = element.name == "marker" || element.name == "defs";
            if !transformed && !in_defs {
                if let (Some(Ok(x)), Some(Ok(y))) = (
                    element.attribute("x").map(str::parse::<f64>),
                    element.attribute("y").map(str::parse::<f64>),
                ) {
                    if x < min_x - 1.0
                        || y < min_y - 1.0
                        || x > min_x + width + 1.0
                        || y > min_y + height + 1.0
                    {
                        outside.push(format!("<{}> at ({x:.0}, {y:.0})", element.name));
                    }
                }
            }
        }
    }

    if empty_labels > 0 {
        check.problems.push(format!("{empty_labels} <text> element(s) carry no text"));
    }
    if !outside.is_empty() {
        check.problems.push(format!(
            "drawn outside the viewBox, where no viewer will show it: {}",
            outside.join(", ")
        ));
    }
    if check.shapes == 0 && check.labels == 0 {
        check.problems.push("the drawing contains no shapes and no labels".to_string());
    }

    check
}

/// Quality checks: a drawing that is valid but not worth showing.
pub mod quality {
    use super::SvgCheck;

    pub fn inspect(check: &SvgCheck) -> Vec<String> {
        let mut notes = Vec::new();
        // A diagram of shapes with nothing named is a picture of boxes.
        if check.shapes > 2 && check.labels == 0 {
            notes.push(format!("draws {} shapes and labels none of them", check.shapes));
        }
        if let Some((_, _, width, height)) = check.view_box {
            if width < 32.0 || height < 32.0 {
                notes.push(format!("the canvas is {width:.0} by {height:.0}, too small to read"));
            }
        }
        notes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::chart::{ChartKind, ChartSpec, Series};
    use crate::artifacts::diagram::{Direction, DiagramSpec, Edge, Node, Shape};

    fn diagram() -> String {
        let spec = DiagramSpec {
            title: "Cooling water loop".to_string(),
            direction: Direction::Across,
            nodes: vec![
                Node {
                    id: "p1".to_string(),
                    label: "Cooling water pump".to_string(),
                    shape: Shape::Box,
                    tag: Some("P-101".to_string()),
                },
                Node {
                    id: "v1".to_string(),
                    label: "Control valve".to_string(),
                    shape: Shape::Valve,
                    tag: Some("PV-2201".to_string()),
                },
            ],
            edges: vec![Edge {
                from: "p1".to_string(),
                to: "v1".to_string(),
                label: Some("CW supply".to_string()),
            }],
        };
        crate::artifacts::diagram::render_svg(&spec).expect("renders")
    }

    fn chart() -> String {
        let spec = ChartSpec {
            kind: ChartKind::Bar,
            title: "Throughput by unit".to_string(),
            categories: vec!["Unit 1".to_string(), "Unit 2".to_string()],
            series: vec![Series {
                name: "March".to_string(),
                values: vec![12.0, 18.0],
            }],
            value_label: "tonnes per hour".to_string(),
        };
        crate::artifacts::chart::render_svg(&spec).expect("renders")
    }

    #[test]
    fn a_diagram_this_product_wrote_reads_back_sound() {
        let check = check_svg_text(&diagram());
        assert!(check.is_sound(), "{:?}", check.problems);
        assert!(check.view_box.is_some());
        assert!(check.shapes > 0, "no shapes were found");
        assert!(check.labels > 0, "no labels were found");
        assert!(check.has_accessible_name);
    }

    #[test]
    fn a_chart_this_product_wrote_reads_back_sound() {
        let check = check_svg_text(&chart());
        assert!(check.is_sound(), "{:?}", check.problems);
        assert!(check.shapes > 0);
    }

    /// The marker the diagram writer defines in `<defs>` and points at from
    /// every edge. If either side is renamed without the other, the arrowheads
    /// silently stop rendering and the drawing still "opens".
    #[test]
    fn an_internal_reference_that_resolves_is_accepted() {
        let svg = diagram();
        assert!(svg.contains("url(#arw)"), "the fixture must reference a marker");
        assert!(check_svg_text(&svg).is_sound());
    }

    #[test]
    fn a_reference_to_something_undefined_is_rejected() {
        let check = check_svg_text(&diagram().replace("url(#arw)", "url(#missing)"));
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("missing")), "{:?}", check.problems);
    }

    #[test]
    fn an_unclosed_element_is_rejected() {
        let check = check_svg_text("<svg viewBox=\"0 0 90 90\" aria-label=\"x\"><g></svg>");
        assert!(!check.is_sound());
        assert!(
            check
                .problems
                .iter()
                .any(|p| p.contains("not well formed") || p.contains("never closed")),
            "{:?}",
            check.problems
        );
    }

    #[test]
    fn a_document_that_is_not_svg_is_rejected() {
        let check = check_svg_text("<html><body>not a drawing</body></html>");
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("not <svg>")));
    }

    #[test]
    fn a_drawing_with_no_view_box_is_rejected() {
        let check = check_svg_text("<svg aria-label=\"x\"><rect x=\"1\" y=\"1\"/></svg>");
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("viewBox")));
    }

    #[test]
    fn a_view_box_that_encloses_nothing_is_rejected() {
        let check = check_svg_text("<svg viewBox=\"0 0 0 0\" aria-label=\"x\"><rect/></svg>");
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("encloses nothing")));
    }

    #[test]
    fn an_empty_label_is_rejected() {
        let check = check_svg_text(
            "<svg viewBox=\"0 0 90 90\" aria-label=\"x\"><text x=\"5\" y=\"5\"></text></svg>",
        );
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("no text")), "{:?}", check.problems);
    }

    /// Ink outside the viewBox is ink nobody will ever see.
    #[test]
    fn something_drawn_off_the_canvas_is_rejected() {
        let check = check_svg_text(
            "<svg viewBox=\"0 0 100 100\" aria-label=\"x\">\
             <text x=\"5000\" y=\"20\">off the page</text></svg>",
        );
        assert!(!check.is_sound());
        assert!(
            check.problems.iter().any(|p| p.contains("outside the viewBox")),
            "{:?}",
            check.problems
        );
    }

    #[test]
    fn a_duplicate_identifier_is_rejected() {
        let check = check_svg_text(
            "<svg viewBox=\"0 0 90 90\" aria-label=\"x\"><rect id=\"a\"/><rect id=\"a\"/></svg>",
        );
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("more than once")));
    }

    #[test]
    fn a_drawing_with_no_accessible_name_is_rejected() {
        let check = check_svg_text("<svg viewBox=\"0 0 90 90\"><rect x=\"1\" y=\"1\"/></svg>");
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("accessible name")));
    }

    /// The writers put a stylesheet in the document. Its braces and quotes are
    /// not markup, and a scanner that read them as markup would reject every
    /// real drawing.
    #[test]
    fn a_stylesheet_is_not_read_as_markup() {
        let check = check_svg_text(
            "<svg viewBox=\"0 0 90 90\" aria-label=\"x\">\
             <style>.ink{fill:currentColor} text{font-family:\"Segoe UI\"}</style>\
             <text x=\"5\" y=\"5\">label</text></svg>",
        );
        assert!(check.is_sound(), "{:?}", check.problems);
        assert_eq!(check.labels, 1);
    }

    #[test]
    fn quality_notices_a_diagram_that_labels_nothing() {
        let check = SvgCheck {
            parses: true,
            shapes: 6,
            labels: 0,
            view_box: Some((0.0, 0.0, 400.0, 300.0)),
            has_accessible_name: true,
            problems: Vec::new(),
        };
        assert!(quality::inspect(&check).iter().any(|n| n.contains("labels none")));
    }
}

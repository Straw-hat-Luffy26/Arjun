//! Block, process, flow and engineering diagrams, drawn as SVG.
//!
//! ## Why this is not Mermaid, and not a knowledge graph
//!
//! ARJUN already draws knowledge graphs: `knowledge.build_graph` emits Mermaid
//! and the chat surface lays it out with a force simulation. That is the right
//! picture for "what is connected to what" over terms nobody chose, and the
//! wrong one for a diagram somebody asked for by name. A block diagram has an
//! author's order - inlet before pump before header - and a force layout
//! discards exactly that, arranging by attraction instead of by intent. Asking
//! for a process flow and receiving a floating cloud of labelled circles is not
//! a different style of the same answer; it is a different answer.
//!
//! So this lays out by rank: an edge means "then", ranks advance in the
//! direction the diagram is read, and a node sits after everything that feeds
//! it. The result is stable, deterministic, and says what the author meant.
//!
//! ## Engineering shapes
//!
//! PS 26117 treats engineering drawings as *input* - scanned P&IDs read through
//! OCR and vision. Drawing them is ARJUN's addition, and it is deliberately
//! modest: the shapes here are the common process-drawing vocabulary (vessel,
//! valve, instrument, equipment) so a sketch of a line reads as a process
//! sketch rather than as boxes. It is not a P&ID authoring tool and does not
//! pretend to be one; nothing here is drawn to ISA-5.1.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

/// How a node is drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// The default. A step, a component, a system.
    Box,
    /// A softer box, for something that is a stage rather than a thing.
    Rounded,
    /// A vessel, tank or drum.
    Vessel,
    /// A valve. Drawn as the two triangles a process drawing uses.
    Valve,
    /// An instrument or measurement point. A circle, as on a P&ID.
    Instrument,
    /// A decision. Only meaningful in a flowchart.
    Decision,
}

impl Shape {
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_lowercase().as_str() {
            "box" | "block" | "component" | "system" | "equipment" => Some(Self::Box),
            "rounded" | "stage" | "step" | "process" => Some(Self::Rounded),
            "vessel" | "tank" | "drum" | "column" => Some(Self::Vessel),
            "valve" => Some(Self::Valve),
            "instrument" | "gauge" | "sensor" | "transmitter" => Some(Self::Instrument),
            "decision" | "choice" => Some(Self::Decision),
            _ => None,
        }
    }
}

/// Which way the diagram is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Left to right. The usual for a process line.
    Across,
    /// Top to bottom. The usual for a flowchart or an architecture stack.
    Down,
}

impl Direction {
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_lowercase().as_str() {
            "lr" | "across" | "left-to-right" | "horizontal" => Some(Self::Across),
            "td" | "tb" | "down" | "top-to-bottom" | "vertical" => Some(Self::Down),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    /// What edges refer to it by.
    pub id: String,
    pub label: String,
    pub shape: Shape,
    /// An equipment tag - `PV-2201`, `TK-101`. Drawn under the label, in the
    /// smaller type a drawing uses, and left out entirely when absent rather
    /// than drawn as an empty line.
    pub tag: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub from: String,
    pub to: String,
    /// What the connection is. Drawn on the line when present.
    pub label: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DiagramSpec {
    pub title: String,
    pub direction: Direction,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

const NODE_W: f64 = 168.0;
const NODE_H: f64 = 56.0;
const GAP_ALONG: f64 = 92.0;
const GAP_ACROSS: f64 = 34.0;
const MARGIN: f64 = 28.0;
const TITLE_H: f64 = 42.0;

fn esc(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(ch),
        }
    }
    out
}

/// Splits a label onto at most two lines so it fits its box.
fn wrap(label: &str) -> Vec<String> {
    const PER_LINE: usize = 22;
    if label.chars().count() <= PER_LINE {
        return vec![label.to_string()];
    }
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for word in label.split_whitespace() {
        if current.is_empty() {
            current = word.to_string();
        } else if current.chars().count() + 1 + word.chars().count() <= PER_LINE {
            current.push(' ');
            current.push_str(word);
        } else {
            lines.push(std::mem::take(&mut current));
            current = word.to_string();
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.len() > 2 {
        let head = lines[0].clone();
        let rest = lines[1..].join(" ");
        let clipped: String = rest.chars().take(PER_LINE - 1).collect();
        return vec![head, format!("{clipped}…")];
    }
    lines
}

/// Ranks each node after everything that feeds it.
///
/// Longest path from a source, computed by relaxation rather than recursion so
/// a cycle cannot blow the stack. A cycle has no correct ranking; the loop
/// stops after as many passes as there are nodes and the back edge is simply
/// drawn returning to an earlier rank, which is what a reader expects a
/// recycle line to look like.
fn rank_nodes(spec: &DiagramSpec) -> BTreeMap<String, usize> {
    let ids: BTreeSet<&str> = spec.nodes.iter().map(|n| n.id.as_str()).collect();
    let mut rank: BTreeMap<String, usize> = spec.nodes.iter().map(|n| (n.id.clone(), 0)).collect();

    for _ in 0..spec.nodes.len() {
        let mut moved = false;
        for edge in &spec.edges {
            if !ids.contains(edge.from.as_str()) || !ids.contains(edge.to.as_str()) {
                continue;
            }
            let want = rank[&edge.from] + 1;
            if rank[&edge.to] < want {
                rank.insert(edge.to.clone(), want);
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
    rank
}

/// Draws the diagram, or says why it cannot.
pub fn render_svg(spec: &DiagramSpec) -> Result<String, String> {
    if spec.title.trim().is_empty() {
        return Err("A diagram needs a title. Nothing was drawn.".to_string());
    }
    if spec.nodes.is_empty() {
        return Err("A diagram needs at least one block. Nothing was drawn.".to_string());
    }

    let known: BTreeSet<&str> = spec.nodes.iter().map(|n| n.id.as_str()).collect();
    for edge in &spec.edges {
        // Refused rather than dropped. A diagram quietly missing a connection
        // is a diagram that says the wrong thing about the plant.
        if !known.contains(edge.from.as_str()) {
            return Err(format!(
                "The connection {:?} -> {:?} starts at a block that is not in the diagram. \
                 Nothing was drawn.",
                edge.from, edge.to
            ));
        }
        if !known.contains(edge.to.as_str()) {
            return Err(format!(
                "The connection {:?} -> {:?} ends at a block that is not in the diagram. \
                 Nothing was drawn.",
                edge.from, edge.to
            ));
        }
    }

    let rank = rank_nodes(spec);
    let mut lanes: BTreeMap<usize, Vec<&Node>> = BTreeMap::new();
    for node in &spec.nodes {
        lanes.entry(rank[&node.id]).or_default().push(node);
    }
    let depth = lanes.len();
    let widest = lanes.values().map(Vec::len).max().unwrap_or(1);

    let (width, height) = match spec.direction {
        Direction::Across => (
            MARGIN * 2.0 + depth as f64 * NODE_W + (depth.saturating_sub(1)) as f64 * GAP_ALONG,
            TITLE_H
                + MARGIN * 2.0
                + widest as f64 * NODE_H
                + (widest.saturating_sub(1)) as f64 * GAP_ACROSS,
        ),
        Direction::Down => (
            MARGIN * 2.0 + widest as f64 * NODE_W + (widest.saturating_sub(1)) as f64 * GAP_ACROSS,
            TITLE_H
                + MARGIN * 2.0
                + depth as f64 * NODE_H
                + (depth.saturating_sub(1)) as f64 * GAP_ALONG,
        ),
    };

    // Where each node's box sits.
    let mut at: BTreeMap<&str, (f64, f64)> = BTreeMap::new();
    for (layer, nodes) in &lanes {
        let count = nodes.len();
        for (index, node) in nodes.iter().enumerate() {
            let (x, y) = match spec.direction {
                Direction::Across => {
                    let block =
                        count as f64 * NODE_H + (count.saturating_sub(1)) as f64 * GAP_ACROSS;
                    let top = TITLE_H + (height - TITLE_H - block) / 2.0;
                    (
                        MARGIN + *layer as f64 * (NODE_W + GAP_ALONG),
                        top + index as f64 * (NODE_H + GAP_ACROSS),
                    )
                }
                Direction::Down => {
                    let block =
                        count as f64 * NODE_W + (count.saturating_sub(1)) as f64 * GAP_ACROSS;
                    let left = (width - block) / 2.0;
                    (
                        left + index as f64 * (NODE_W + GAP_ACROSS),
                        TITLE_H + MARGIN + *layer as f64 * (NODE_H + GAP_ALONG),
                    )
                }
            };
            at.insert(node.id.as_str(), (x, y));
        }
    }

    let mut svg = String::with_capacity(4096);
    write!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width:.0} {height:.0}\" \
         width=\"100%\" role=\"img\" aria-label=\"{}\">",
        esc(&spec.title)
    )
    .expect("write");

    svg.push_str(
        "<defs><marker id=\"arw\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" \
         markerWidth=\"7\" markerHeight=\"7\" orient=\"auto\">\
         <path d=\"M0,0 L10,5 L0,10 z\" fill=\"currentColor\" fill-opacity=\".55\"/>\
         </marker></defs>\
         <style>\
         .ink{fill:currentColor}\
         .edge{stroke:currentColor;stroke-opacity:.55;fill:none}\
         .shape{stroke:currentColor;stroke-opacity:.45;fill:none}\
         text{font-family:system-ui,-apple-system,Segoe UI,sans-serif}\
         </style>",
    );

    write!(
        svg,
        "<text class=\"ink\" x=\"{MARGIN}\" y=\"26\" font-size=\"15\" font-weight=\"600\">{}</text>",
        esc(&spec.title)
    )
    .expect("write");

    // Edges first, so a line never sits on top of the box it points at.
    for edge in &spec.edges {
        let (fx, fy) = at[edge.from.as_str()];
        let (tx, ty) = at[edge.to.as_str()];
        let (x1, y1, x2, y2) = match spec.direction {
            Direction::Across => (fx + NODE_W, fy + NODE_H / 2.0, tx - 7.0, ty + NODE_H / 2.0),
            Direction::Down => (fx + NODE_W / 2.0, fy + NODE_H, tx + NODE_W / 2.0, ty - 7.0),
        };
        // A gentle curve, so two edges between the same ranks stay apart.
        let (cx, cy) = ((x1 + x2) / 2.0, (y1 + y2) / 2.0);
        write!(
            svg,
            "<path class=\"edge\" d=\"M{x1:.1},{y1:.1} Q{cx:.1},{y1:.1} {cx:.1},{cy:.1} \
             T{x2:.1},{y2:.1}\" marker-end=\"url(#arw)\"/>"
        )
        .expect("write");
        if let Some(label) = &edge.label {
            write!(
                svg,
                "<text class=\"ink\" x=\"{cx:.1}\" y=\"{:.1}\" font-size=\"10\" \
                 text-anchor=\"middle\" fill-opacity=\".75\">{}</text>",
                cy - 5.0,
                esc(label)
            )
            .expect("write");
        }
    }

    for node in &spec.nodes {
        let (x, y) = at[node.id.as_str()];
        let cx = x + NODE_W / 2.0;
        let cy = y + NODE_H / 2.0;
        match node.shape {
            Shape::Box => write!(
                svg,
                "<rect class=\"shape\" x=\"{x:.1}\" y=\"{y:.1}\" width=\"{NODE_W}\" \
                 height=\"{NODE_H}\"/>"
            )
            .expect("write"),
            Shape::Rounded => write!(
                svg,
                "<rect class=\"shape\" x=\"{x:.1}\" y=\"{y:.1}\" width=\"{NODE_W}\" \
                 height=\"{NODE_H}\" rx=\"14\"/>"
            )
            .expect("write"),
            Shape::Vessel => write!(
                svg,
                "<rect class=\"shape\" x=\"{x:.1}\" y=\"{y:.1}\" width=\"{NODE_W}\" \
                 height=\"{NODE_H}\" rx=\"{:.1}\"/>",
                NODE_H / 2.0
            )
            .expect("write"),
            Shape::Instrument => write!(
                svg,
                "<circle class=\"shape\" cx=\"{cx:.1}\" cy=\"{cy:.1}\" r=\"{:.1}\"/>",
                NODE_H / 2.0
            )
            .expect("write"),
            Shape::Decision => write!(
                svg,
                "<polygon class=\"shape\" points=\"{cx:.1},{y:.1} {:.1},{cy:.1} {cx:.1},{:.1} \
                 {x:.1},{cy:.1}\"/>",
                x + NODE_W,
                y + NODE_H
            )
            .expect("write"),
            Shape::Valve => write!(
                svg,
                "<polygon class=\"shape\" points=\"{x:.1},{y:.1} {cx:.1},{cy:.1} {x:.1},{:.1}\"/>\
                 <polygon class=\"shape\" points=\"{:.1},{y:.1} {cx:.1},{cy:.1} {:.1},{:.1}\"/>",
                y + NODE_H,
                x + NODE_W,
                x + NODE_W,
                y + NODE_H
            )
            .expect("write"),
        }

        let lines = wrap(&node.label);
        let has_tag = node
            .tag
            .as_deref()
            .map(str::trim)
            .is_some_and(|t| !t.is_empty());
        let block = lines.len() as f64 * 14.0 + if has_tag { 13.0 } else { 0.0 };
        let mut text_y = cy - block / 2.0 + 11.0;
        for line in &lines {
            write!(
                svg,
                "<text class=\"ink\" x=\"{cx:.1}\" y=\"{text_y:.1}\" font-size=\"12\" \
                 text-anchor=\"middle\">{}</text>",
                esc(line)
            )
            .expect("write");
            text_y += 14.0;
        }
        if let Some(tag) = &node.tag {
            let tag = tag.trim();
            if !tag.is_empty() {
                write!(
                    svg,
                    "<text class=\"ink\" x=\"{cx:.1}\" y=\"{text_y:.1}\" font-size=\"10\" \
                     text-anchor=\"middle\" fill-opacity=\".65\" \
                     font-family=\"ui-monospace,monospace\">{}</text>",
                    esc(tag)
                )
                .expect("write");
            }
        }
    }

    svg.push_str("</svg>");
    Ok(svg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, label: &str, shape: Shape) -> Node {
        Node {
            id: id.to_string(),
            label: label.to_string(),
            shape,
            tag: None,
        }
    }

    fn line() -> DiagramSpec {
        DiagramSpec {
            title: "Unit Four feed line".to_string(),
            direction: Direction::Across,
            nodes: vec![
                node("inlet", "Feed inlet", Shape::Box),
                node("pump", "Charge pump", Shape::Box),
                node("drum", "Surge drum", Shape::Vessel),
            ],
            edges: vec![
                Edge {
                    from: "inlet".into(),
                    to: "pump".into(),
                    label: None,
                },
                Edge {
                    from: "pump".into(),
                    to: "drum".into(),
                    label: Some("40 m3/h".into()),
                },
            ],
        }
    }

    #[test]
    fn every_block_and_connection_is_drawn() {
        let svg = render_svg(&line()).expect("diagram");
        assert!(svg.contains("Feed inlet"));
        assert!(svg.contains("Charge pump"));
        assert!(svg.contains("Surge drum"));
        assert_eq!(svg.matches("<path class=\"edge\"").count(), 2, "{svg}");
        assert!(svg.contains("40 m3/h"));
    }

    /// The property that makes this a diagram rather than a graph: order.
    #[test]
    fn a_block_is_placed_after_everything_that_feeds_it() {
        let ranks = rank_nodes(&line());
        assert_eq!(ranks["inlet"], 0);
        assert_eq!(ranks["pump"], 1);
        assert_eq!(ranks["drum"], 2);
    }

    #[test]
    fn a_recycle_line_does_not_hang() {
        let mut spec = line();
        spec.edges.push(Edge {
            from: "drum".into(),
            to: "pump".into(),
            label: Some("recycle".into()),
        });
        let svg = render_svg(&spec).expect("diagram");
        assert_eq!(svg.matches("<path class=\"edge\"").count(), 3, "{svg}");
    }

    #[test]
    fn every_engineering_shape_draws_something_distinct() {
        for (shape, marker) in [
            (Shape::Box, "<rect"),
            (Shape::Rounded, "rx=\"14\""),
            (Shape::Vessel, "rx=\"28"),
            (Shape::Instrument, "<circle"),
            (Shape::Decision, "<polygon"),
            (Shape::Valve, "<polygon"),
        ] {
            let spec = DiagramSpec {
                title: "Shapes".to_string(),
                direction: Direction::Across,
                nodes: vec![node("a", "A", shape)],
                edges: vec![],
            };
            let svg = render_svg(&spec).expect("diagram");
            assert!(
                svg.contains(marker),
                "{shape:?} did not draw {marker}: {svg}"
            );
        }
    }

    #[test]
    fn an_equipment_tag_is_drawn_under_its_label() {
        let mut spec = line();
        spec.nodes[1].tag = Some("P-101A".to_string());
        let svg = render_svg(&spec).expect("diagram");
        assert!(svg.contains("P-101A"), "{svg}");
        assert!(svg.contains("ui-monospace"));
    }

    #[test]
    fn an_absent_tag_draws_no_empty_line() {
        let svg = render_svg(&line()).expect("diagram");
        assert!(!svg.contains("ui-monospace"), "{svg}");
    }

    /// A connection to a block that is not there is refused, not dropped.
    #[test]
    fn a_dangling_connection_is_refused() {
        let mut spec = line();
        spec.edges.push(Edge {
            from: "pump".into(),
            to: "flare".into(),
            label: None,
        });
        let problem = render_svg(&spec).unwrap_err();
        assert!(
            problem.contains("ends at a block that is not in the diagram"),
            "{problem}"
        );
        assert!(problem.contains("Nothing was drawn"));
    }

    #[test]
    fn it_refuses_an_empty_diagram() {
        let mut spec = line();
        spec.nodes.clear();
        assert!(render_svg(&spec)
            .unwrap_err()
            .contains("at least one block"));
    }

    #[test]
    fn a_label_cannot_break_out_of_the_drawing() {
        let mut spec = line();
        spec.nodes[0].label = "</svg><script>alert(1)</script>".to_string();
        let svg = render_svg(&spec).expect("diagram");
        assert!(!svg.contains("<script>"), "{svg}");
        assert_eq!(svg.matches("</svg>").count(), 1);
    }

    #[test]
    fn the_same_diagram_is_drawn_the_same_way_every_time() {
        assert_eq!(render_svg(&line()).unwrap(), render_svg(&line()).unwrap());
    }

    #[test]
    fn direction_and_shape_are_read_from_ordinary_words() {
        assert_eq!(Direction::parse("LR"), Some(Direction::Across));
        assert_eq!(Direction::parse("top-to-bottom"), Some(Direction::Down));
        assert_eq!(Direction::parse("sideways"), None);
        assert_eq!(Shape::parse("tank"), Some(Shape::Vessel));
        assert_eq!(Shape::parse("transmitter"), Some(Shape::Instrument));
        assert_eq!(Shape::parse("banana"), None);
    }

    #[test]
    fn a_long_label_is_wrapped_rather_than_overflowing() {
        let lines = wrap("Regenerative heat exchanger train downstream of the reboiler");
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert!(lines.iter().all(|l| l.chars().count() <= 23), "{lines:?}");
    }
}

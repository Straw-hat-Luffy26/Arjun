//! Charts, drawn as SVG.
//!
//! ## Why this is written here rather than taken from a crate
//!
//! `plotters` is the obvious candidate and was evaluated. It is a good library
//! and the wrong fit twice over. It draws to a raster backend by default, and a
//! PNG cannot be read by the chat surface, styled by the theme, or scaled by
//! the reader - where an SVG is text, renders inline, and follows the light and
//! dark palette like every other surface. And it arrives with a transitive
//! footprint this product pays for in its SBOM and its air-gapped install, for
//! drawing rectangles and lines.
//!
//! The same reasoning already governs [`super::docx`], [`super::xlsx`] and
//! [`super::pptx`], which write OOXML by hand with no dependency at all. A bar
//! chart is a simpler format than a Word document.
//!
//! ## What it will not do
//!
//! It refuses rather than inventing. A series with no points, a chart with no
//! series, a category count that does not match the values - each is an error
//! with a sentence saying what was wrong. Nothing here substitutes a default,
//! draws an empty axis and calls it a chart, or silently truncates a series to
//! fit. A picture that looks like data and is not is the worst thing this
//! module could produce.

use std::fmt::Write as _;

/// How the series should be drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChartKind {
    /// Discrete categories side by side. The default for "compare these".
    Bar,
    /// A line through the points, for something measured over an ordered axis.
    Line,
}

impl ChartKind {
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_lowercase().as_str() {
            "bar" | "column" | "histogram" => Some(Self::Bar),
            "line" | "trend" | "series" => Some(Self::Line),
            _ => None,
        }
    }
}

/// One named run of values, one per category.
#[derive(Debug, Clone)]
pub struct Series {
    pub name: String,
    pub values: Vec<f64>,
}

/// Everything needed to draw one chart.
#[derive(Debug, Clone)]
pub struct ChartSpec {
    pub kind: ChartKind,
    pub title: String,
    pub categories: Vec<String>,
    pub series: Vec<Series>,
    /// What the numbers are. Drawn on the value axis, so a reader is never left
    /// guessing whether a bar is rupees, tonnes or percent.
    pub value_label: String,
}

/// Colours that stay legible on both themes.
///
/// Fixed rather than read from CSS variables: an SVG written to a file is
/// opened outside the app as often as inside it, and a `var(--accent)` that
/// resolves to nothing there draws an invisible chart.
const PALETTE: [&str; 6] = [
    "#4c8dff", "#f2a541", "#4ec9a5", "#e0607e", "#a78bfa", "#8fbf3f",
];

const WIDTH: f64 = 720.0;
const HEIGHT: f64 = 420.0;
const LEFT: f64 = 78.0;
const RIGHT: f64 = 24.0;
const TOP: f64 = 54.0;
const BOTTOM: f64 = 74.0;

/// Escapes text for XML content and attributes.
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

/// Rounds a number for display without pretending to precision it lacks.
fn tidy(value: f64) -> String {
    if value == value.trunc() && value.abs() < 1e15 {
        format!("{}", value as i64)
    } else {
        let text = format!("{value:.2}");
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// A round step for the value axis.
fn axis_step(span: f64) -> f64 {
    if span <= 0.0 {
        return 1.0;
    }
    let raw = span / 4.0;
    let magnitude = 10f64.powf(raw.log10().floor());
    let normalised = raw / magnitude;
    let stepped = if normalised <= 1.0 {
        1.0
    } else if normalised <= 2.0 {
        2.0
    } else if normalised <= 5.0 {
        5.0
    } else {
        10.0
    };
    stepped * magnitude
}

/// Draws the chart, or says why it cannot.
pub fn render_svg(spec: &ChartSpec) -> Result<String, String> {
    if spec.title.trim().is_empty() {
        return Err("A chart needs a title. Nothing was drawn.".to_string());
    }
    if spec.categories.is_empty() {
        return Err("A chart needs at least one category. Nothing was drawn.".to_string());
    }
    if spec.series.is_empty() {
        return Err("A chart needs at least one series. Nothing was drawn.".to_string());
    }
    for series in &spec.series {
        if series.values.len() != spec.categories.len() {
            return Err(format!(
                "Series {:?} has {} value(s) for {} categories. Every series must have one \
                 value per category; nothing was drawn.",
                series.name,
                series.values.len(),
                spec.categories.len()
            ));
        }
        if let Some(bad) = series.values.iter().find(|v| !v.is_finite()) {
            return Err(format!(
                "Series {:?} contains {bad}, which cannot be plotted. Nothing was drawn.",
                series.name
            ));
        }
    }

    let all: Vec<f64> = spec
        .series
        .iter()
        .flat_map(|s| s.values.iter().copied())
        .collect();
    let high = all.iter().copied().fold(f64::MIN, f64::max);
    let low = all.iter().copied().fold(f64::MAX, f64::min).min(0.0);
    // A flat series still has to draw something with a readable axis.
    let span = if (high - low).abs() < f64::EPSILON {
        high.abs().max(1.0)
    } else {
        high - low
    };
    let step = axis_step(span);
    let top_value = (high / step).ceil() * step;
    let base_value = (low / step).floor() * step;
    let range = (top_value - base_value).max(f64::EPSILON);

    let plot_w = WIDTH - LEFT - RIGHT;
    let plot_h = HEIGHT - TOP - BOTTOM;
    let y_of = |value: f64| TOP + plot_h - ((value - base_value) / range) * plot_h;

    let mut svg = String::with_capacity(4096);
    write!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {WIDTH} {HEIGHT}\" \
         width=\"100%\" role=\"img\" aria-label=\"{}\">",
        esc(&spec.title)
    )
    .expect("write to string");

    // `currentColor` for ink, so the chart follows the theme it is dropped into
    // and still has a colour when opened on its own.
    svg.push_str(
        "<style>\
         .ink{fill:currentColor}\
         .grid{stroke:currentColor;stroke-opacity:.16}\
         .axis{stroke:currentColor;stroke-opacity:.45}\
         text{font-family:system-ui,-apple-system,Segoe UI,sans-serif}\
         </style>",
    );

    write!(
        svg,
        "<text class=\"ink\" x=\"{LEFT}\" y=\"28\" font-size=\"16\" font-weight=\"600\">{}</text>",
        esc(&spec.title)
    )
    .expect("write");

    // Value axis: gridline, tick and label at each step.
    let mut value = base_value;
    while value <= top_value + step / 2.0 {
        let y = y_of(value);
        write!(
            svg,
            "<line class=\"grid\" x1=\"{LEFT}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\"/>\
             <text class=\"ink\" x=\"{:.1}\" y=\"{:.1}\" font-size=\"11\" text-anchor=\"end\" \
             fill-opacity=\".7\">{}</text>",
            WIDTH - RIGHT,
            LEFT - 8.0,
            y + 4.0,
            esc(&tidy(value))
        )
        .expect("write");
        value += step;
    }

    if !spec.value_label.trim().is_empty() {
        write!(
            svg,
            "<text class=\"ink\" transform=\"translate(16 {:.1}) rotate(-90)\" font-size=\"11\" \
             text-anchor=\"middle\" fill-opacity=\".7\">{}</text>",
            TOP + plot_h / 2.0,
            esc(&spec.value_label)
        )
        .expect("write");
    }

    let zero_y = y_of(0.0f64.max(base_value));
    write!(
        svg,
        "<line class=\"axis\" x1=\"{LEFT}\" y1=\"{zero_y:.1}\" x2=\"{:.1}\" y2=\"{zero_y:.1}\"/>",
        WIDTH - RIGHT
    )
    .expect("write");

    let slot = plot_w / spec.categories.len() as f64;

    // Category labels, under their slot.
    for (index, category) in spec.categories.iter().enumerate() {
        let centre = LEFT + slot * (index as f64 + 0.5);
        write!(
            svg,
            "<text class=\"ink\" x=\"{centre:.1}\" y=\"{:.1}\" font-size=\"11\" \
             text-anchor=\"middle\" fill-opacity=\".7\">{}</text>",
            HEIGHT - BOTTOM + 18.0,
            esc(category)
        )
        .expect("write");
    }

    match spec.kind {
        ChartKind::Bar => {
            let count = spec.series.len() as f64;
            let group = slot * 0.72;
            let width = group / count;
            for (s, series) in spec.series.iter().enumerate() {
                let colour = PALETTE[s % PALETTE.len()];
                for (index, value) in series.values.iter().enumerate() {
                    let x = LEFT + slot * index as f64 + (slot - group) / 2.0 + width * s as f64;
                    let y = y_of(*value);
                    let (top, height) = if *value >= 0.0 {
                        (y, zero_y - y)
                    } else {
                        (zero_y, y - zero_y)
                    };
                    write!(
                        svg,
                        "<rect x=\"{x:.1}\" y=\"{top:.1}\" width=\"{:.1}\" height=\"{:.1}\" \
                         fill=\"{colour}\"><title>{}: {}</title></rect>",
                        (width - 2.0).max(1.0),
                        height.max(0.5),
                        esc(&series.name),
                        esc(&tidy(*value))
                    )
                    .expect("write");
                }
            }
        }
        ChartKind::Line => {
            for (s, series) in spec.series.iter().enumerate() {
                let colour = PALETTE[s % PALETTE.len()];
                let points: Vec<String> = series
                    .values
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        format!(
                            "{:.1},{:.1}",
                            LEFT + slot * (index as f64 + 0.5),
                            y_of(*value)
                        )
                    })
                    .collect();
                write!(
                    svg,
                    "<polyline fill=\"none\" stroke=\"{colour}\" stroke-width=\"2\" \
                     stroke-linejoin=\"round\" points=\"{}\"/>",
                    points.join(" ")
                )
                .expect("write");
                for (index, value) in series.values.iter().enumerate() {
                    write!(
                        svg,
                        "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"3\" fill=\"{colour}\">\
                         <title>{}: {}</title></circle>",
                        LEFT + slot * (index as f64 + 0.5),
                        y_of(*value),
                        esc(&series.name),
                        esc(&tidy(*value))
                    )
                    .expect("write");
                }
            }
        }
    }

    // Legend, only when there is more than one series to tell apart.
    if spec.series.len() > 1 {
        let mut x = LEFT;
        for (s, series) in spec.series.iter().enumerate() {
            let colour = PALETTE[s % PALETTE.len()];
            write!(
                svg,
                "<rect x=\"{x:.1}\" y=\"{:.1}\" width=\"10\" height=\"10\" rx=\"2\" \
                 fill=\"{colour}\"/><text class=\"ink\" x=\"{:.1}\" y=\"{:.1}\" font-size=\"11\" \
                 fill-opacity=\".8\">{}</text>",
                HEIGHT - 26.0,
                x + 15.0,
                HEIGHT - 17.0,
                esc(&series.name)
            )
            .expect("write");
            x += 15.0 + 8.0 * series.name.chars().count() as f64 + 22.0;
        }
    }

    svg.push_str("</svg>");
    Ok(svg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(kind: ChartKind) -> ChartSpec {
        ChartSpec {
            kind,
            title: "Throughput by unit".to_string(),
            categories: vec!["Unit One".to_string(), "Unit Four".to_string()],
            series: vec![Series {
                name: "Tonnes".to_string(),
                values: vec![120.0, 96.0],
            }],
            value_label: "tonnes/day".to_string(),
        }
    }

    #[test]
    fn a_bar_chart_draws_one_rectangle_per_value() {
        let svg = render_svg(&spec(ChartKind::Bar)).expect("chart");
        assert_eq!(svg.matches("<rect").count(), 2, "{svg}");
        assert!(svg.contains("Throughput by unit"));
        assert!(svg.contains("Unit Four"));
        assert!(svg.contains("tonnes/day"));
    }

    #[test]
    fn a_line_chart_draws_a_polyline_through_its_points() {
        let svg = render_svg(&spec(ChartKind::Line)).expect("chart");
        assert_eq!(svg.matches("<polyline").count(), 1, "{svg}");
        assert_eq!(svg.matches("<circle").count(), 2, "{svg}");
    }

    /// Bars must be proportional, or the picture lies about the data.
    #[test]
    fn a_bar_twice_the_value_is_twice_the_height() {
        let mut s = spec(ChartKind::Bar);
        s.series[0].values = vec![50.0, 100.0];
        let svg = render_svg(&s).expect("chart");

        let heights: Vec<f64> = svg
            .match_indices("height=\"")
            .filter_map(|(at, _)| {
                let rest = &svg[at + 8..];
                rest.split('"').next()?.parse::<f64>().ok()
            })
            .collect();
        let bars: Vec<f64> = heights.into_iter().filter(|h| *h > 1.0).collect();
        assert_eq!(bars.len(), 2, "{bars:?}");
        let ratio = bars[1] / bars[0];
        assert!(
            (ratio - 2.0).abs() < 0.02,
            "a value twice as large drew a bar {ratio}x as tall"
        );
    }

    #[test]
    fn a_legend_appears_only_when_there_is_something_to_tell_apart() {
        let one = render_svg(&spec(ChartKind::Bar)).expect("chart");
        assert!(!one.contains("rx=\"2\""), "a single series drew a legend");

        let mut two = spec(ChartKind::Bar);
        two.series.push(Series {
            name: "Target".to_string(),
            values: vec![130.0, 110.0],
        });
        let svg = render_svg(&two).expect("chart");
        assert!(svg.contains("Target"));
        assert!(svg.contains("rx=\"2\""));
    }

    /// Every refusal, because a chart drawn from bad data is worse than none.
    #[test]
    fn it_refuses_rather_than_drawing_something_untrue() {
        let mut empty = spec(ChartKind::Bar);
        empty.series.clear();
        assert!(render_svg(&empty).unwrap_err().contains("one series"));

        let mut untitled = spec(ChartKind::Bar);
        untitled.title = "  ".to_string();
        assert!(render_svg(&untitled).unwrap_err().contains("needs a title"));

        let mut short = spec(ChartKind::Bar);
        short.series[0].values = vec![1.0];
        let problem = render_svg(&short).unwrap_err();
        assert!(problem.contains("1 value(s) for 2 categories"), "{problem}");

        let mut nan = spec(ChartKind::Bar);
        nan.series[0].values = vec![1.0, f64::NAN];
        assert!(render_svg(&nan).unwrap_err().contains("cannot be plotted"));
    }

    /// A category called `</svg>` must not end the drawing.
    #[test]
    fn a_label_cannot_break_out_of_the_document() {
        let mut hostile = spec(ChartKind::Bar);
        hostile.categories[0] = "</svg><script>alert(1)</script>".to_string();
        let svg = render_svg(&hostile).expect("chart");
        assert!(!svg.contains("<script>"), "{svg}");
        assert!(svg.ends_with("</svg>"));
        assert_eq!(svg.matches("</svg>").count(), 1);
    }

    #[test]
    fn negative_values_hang_below_the_zero_line() {
        let mut s = spec(ChartKind::Bar);
        s.series[0].values = vec![-40.0, 80.0];
        let svg = render_svg(&s).expect("chart");
        assert_eq!(svg.matches("<rect").count(), 2, "{svg}");
        assert!(svg.contains("-40"));
    }

    #[test]
    fn the_same_data_always_draws_the_same_chart() {
        let a = render_svg(&spec(ChartKind::Bar)).expect("chart");
        let b = render_svg(&spec(ChartKind::Bar)).expect("chart");
        assert_eq!(a, b);
    }

    #[test]
    fn a_kind_is_read_from_the_words_a_person_uses() {
        assert_eq!(ChartKind::parse("bar"), Some(ChartKind::Bar));
        assert_eq!(ChartKind::parse("Column"), Some(ChartKind::Bar));
        assert_eq!(ChartKind::parse("line"), Some(ChartKind::Line));
        assert_eq!(ChartKind::parse("trend"), Some(ChartKind::Line));
        assert_eq!(ChartKind::parse("pie"), None);
    }
}

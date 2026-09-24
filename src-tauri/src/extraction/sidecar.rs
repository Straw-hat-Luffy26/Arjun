//! The geometry half of extraction: PyMuPDF, through the page rasteriser.
//!
//! `sidecars/document_sidecar/render_pages.py` already renders pages for the
//! artifact reviewer (P04). P06 adds four modes to the same script — layout,
//! crop, skew, probe image — so the analyst needs nothing installed that the
//! reviewer does not: the dependency is `page-rasteriser` in
//! [`crate::deployment`], resolved and reported the same way.
//!
//! Every call is a short-lived process bounded by [`SIDECAR_TIMEOUT`], run off
//! the async executor by the caller. The script prints exactly one JSON object;
//! an `error` key is a refusal the caller repeats, and anything else that is not
//! JSON is reported with the last line of stderr.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;

use super::regions::{BBox, PageSpace};

/// Long enough for a table search on a dense page; short enough that a hung
/// parser is noticed inside one tool call.
pub const SIDECAR_TIMEOUT: Duration = Duration::from_secs(45);

/// How the script is reached on this machine.
#[derive(Debug, Clone)]
pub struct Sidecar {
    pub python: String,
    pub script: PathBuf,
}

/// One text or image block of a page, as the layout engine reports it.
#[derive(Debug, Clone, Deserialize)]
pub struct LayoutBlock {
    pub bbox: Vec<f64>,
    pub kind: String,
    #[serde(default)]
    pub text: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LayoutCell {
    pub row: u32,
    pub col: u32,
    #[serde(default)]
    pub bbox: Option<Vec<f64>>,
    #[serde(default)]
    pub text: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LayoutTable {
    pub bbox: Vec<f64>,
    pub rows: u32,
    pub cols: u32,
    #[serde(default)]
    pub cells: Vec<LayoutCell>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LayoutPage {
    pub page: u32,
    pub width: f64,
    pub height: f64,
    #[serde(default)]
    pub rotation: i32,
    #[serde(default)]
    pub text_chars: u32,
    #[serde(default)]
    pub blocks: Vec<LayoutBlock>,
    #[serde(default)]
    pub tables: Vec<LayoutTable>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LayoutAnswer {
    pub version: String,
    pub pages: u32,
    pub coord_space: String,
    pub layout: Vec<LayoutPage>,
}

impl LayoutAnswer {
    pub fn space(&self) -> Result<PageSpace, String> {
        PageSpace::parse(&self.coord_space).ok_or_else(|| {
            format!(
                "the layout engine reported coordinates in {:?}, which this build does not map",
                self.coord_space
            )
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CropAnswer {
    pub page: u32,
    pub width: u32,
    pub height: u32,
    pub bbox: Vec<f64>,
    pub coord_space: String,
    pub pixels_per_unit: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkewAnswer {
    pub page: u32,
    pub degrees: Option<f64>,
    pub method: String,
    #[serde(default)]
    pub dark_pixels: u64,
}

impl Sidecar {
    /// The script and interpreter this deployment resolves.
    pub fn resolve() -> Result<Self, String> {
        let script = crate::deployment::require_path("page-rasteriser")?;
        Ok(Self {
            python: crate::deployment::program("python"),
            script,
        })
    }

    fn run(&self, args: &[String]) -> Result<serde_json::Value, String> {
        let mut command = crate::system_analyzer::process_utils::create_hidden_command(&self.python);
        command.arg(&self.script).args(args);
        let output = crate::artifacts::render::run_bounded(command, SIDECAR_TIMEOUT)
            .map_err(|error| format!("the page analyser did not run: {error}"))?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let answer: serde_json::Value = serde_json::from_str(stdout.trim()).map_err(|_| {
            format!(
                "the page analyser did not answer in JSON (exit {:?}): {}",
                output.status.code(),
                String::from_utf8_lossy(&output.stderr)
                    .lines()
                    .last()
                    .unwrap_or_default()
            )
        })?;
        if let Some(error) = answer.get("error").and_then(|e| e.as_str()) {
            return Err(error.to_string());
        }
        Ok(answer)
    }

    fn parse<T: serde::de::DeserializeOwned>(value: serde_json::Value, what: &str) -> Result<T, String> {
        serde_json::from_value(value)
            .map_err(|error| format!("the page analyser's {what} answer was not understood: {error}"))
    }

    pub fn layout(&self, source: &Path, from_page: u32, to_page: u32) -> Result<LayoutAnswer, String> {
        let answer = self.run(&[
            "--layout".into(),
            source.display().to_string(),
            "--from".into(),
            from_page.to_string(),
            "--to".into(),
            to_page.to_string(),
        ])?;
        Self::parse(answer, "layout")
    }

    /// Renders exactly `bbox` of `page` into `target`. `None` is the whole page.
    pub fn crop(
        &self,
        source: &Path,
        target: &Path,
        page: u32,
        bbox: Option<&BBox>,
        dpi: u32,
    ) -> Result<CropAnswer, String> {
        let mut args = vec![
            "--crop".into(),
            source.display().to_string(),
            target.display().to_string(),
            "--page".into(),
            page.to_string(),
            "--dpi".into(),
            dpi.to_string(),
        ];
        if let Some(bbox) = bbox {
            args.push("--bbox".into());
            args.push(format!("{},{},{},{}", bbox.x0, bbox.y0, bbox.x1, bbox.y1));
        }
        let answer = self.run(&args)?;
        Self::parse(answer, "crop")
    }

    pub fn skew(&self, source: &Path, page: u32) -> Result<SkewAnswer, String> {
        let answer = self.run(&[
            "--skew".into(),
            source.display().to_string(),
            "--page".into(),
            page.to_string(),
        ])?;
        Self::parse(answer, "skew")
    }

    /// Draws `token` into a PNG, for the vision-readiness probe.
    pub fn probe_image(&self, target: &Path, token: &str) -> Result<(), String> {
        self.run(&[
            "--probe-image".into(),
            target.display().to_string(),
            "--text".into(),
            token.to_string(),
        ])
        .map(|_| ())
    }
}

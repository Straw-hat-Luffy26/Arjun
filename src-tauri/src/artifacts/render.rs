//! Turning a produced file into pages a reviewer can look at.
//!
//! ## The two adapters, and why these two
//!
//! | Adapter | Does | Runtime | Licence | Provisioned by |
//! |---|---|---|---|---|
//! | `office` | `.docx` / `.pptx` / `.xlsx` / `.svg` → PDF | LibreOffice, headless | MPL-2.0 | the operator or the offline deployment pack; `ARJUN_SOFFICE` points at it |
//! | `rasteriser` | PDF → one PNG per page, with its text and whether it is blank | PyMuPDF, via `sidecars/document_sidecar/render_pages.py` | AGPL-3.0 (already a dependency of the document sidecar) | already installed for `attachment_extract.py` |
//!
//! Nothing new is compiled in. LibreOffice is the one open layout engine that
//! reads all three OOXML formats; PyMuPDF is what the document sidecar already
//! uses to rasterise scanned pages for OCR, so the second adapter adds a script,
//! not a dependency.
//!
//! ## Pinned
//!
//! Each adapter names the version series it was qualified against
//! ([`QUALIFIED_OFFICE_SERIES`], [`QUALIFIED_RASTERISER_SERIES`]). Every render
//! records the exact version that produced it. A version outside the qualified
//! series is reported **unavailable**, with the version found: a layout engine
//! nobody has checked this product's files against is not evidence that they
//! render.
//!
//! ## Unavailable is a state, not a pass
//!
//! A machine without LibreOffice gets [`RenderState::Unavailable`] and the
//! reason, and the validation ladder records `render_checked: unavailable`. It
//! never becomes a skipped rung that reads as green.
//!
//! ## What a renderer is not allowed to do
//!
//! - **Fetch.** [`super::package::PackageReport::render_refusals`] is asked
//!   first; a package with an external image, OLE link, attached template,
//!   frame or remote field code is refused before LibreOffice starts. An SVG
//!   with an `href` or `url(…)` that leaves the file is refused the same way.
//!   Refusing is the guarantee: with nothing external in the file there is
//!   nothing to fetch.
//! - **Run macros.** A macro-bearing package is refused by the same check; the
//!   throwaway LibreOffice profile each render uses additionally disables macro
//!   execution and sets macro security to its highest level.
//! - **Outlive its budget.** Each process is killed at [`RENDER_TIMEOUT`].
//! - **Touch anything shared.** Each render runs in its own directory with its
//!   own profile, and the input is a copy of the stored bytes.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::package::{self, DetectedFormat};

/// LibreOffice series this product's writers were rendered and checked with.
pub const QUALIFIED_OFFICE_SERIES: &[&str] = &["24.2", "24.8", "25.2", "25.8"];
/// PyMuPDF series the rasteriser script was checked with.
pub const QUALIFIED_RASTERISER_SERIES: &[&str] = &["1.24", "1.25", "1.26", "1.27", "1.28"];
/// The wall clock one renderer process may use.
pub const RENDER_TIMEOUT: Duration = Duration::from_secs(120);
/// Pages rasterised per call. The script enforces the same number.
pub const MAX_PAGES_PER_RENDER: u32 = 50;

/// One adapter, as found on this machine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Adapter {
    pub name: String,
    pub available: bool,
    /// The exact version reported by the program itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Whether that version is in the qualified series.
    pub qualified: bool,
    pub qualified_series: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unavailable_because: Option<String>,
    pub licence: String,
    pub provisioning: String,
    #[serde(skip)]
    program: Option<PathBuf>,
    #[serde(skip)]
    script: Option<PathBuf>,
}

impl Adapter {
    /// Available and at a qualified version.
    pub fn usable(&self) -> bool {
        self.available && self.qualified
    }

    /// Why it cannot be used, when it cannot.
    pub fn why_not(&self) -> Option<String> {
        if !self.available {
            return Some(
                self.unavailable_because
                    .clone()
                    .unwrap_or_else(|| format!("{} is not available", self.name)),
            );
        }
        if !self.qualified {
            return Some(format!(
                "{} {} is installed, and only the {} series have been qualified",
                self.name,
                self.version.as_deref().unwrap_or("of unknown version"),
                self.qualified_series.join(", ")
            ));
        }
        None
    }

    fn unavailable(name: &str, why: String, licence: &str, provisioning: &str, series: &[&str]) -> Self {
        Adapter {
            name: name.to_string(),
            available: false,
            version: None,
            qualified: false,
            qualified_series: series.iter().map(|s| s.to_string()).collect(),
            unavailable_because: Some(why),
            licence: licence.to_string(),
            provisioning: provisioning.to_string(),
            program: None,
            script: None,
        }
    }
}

/// Both adapters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Inventory {
    pub office: Adapter,
    pub rasteriser: Adapter,
}

const OFFICE_LICENCE: &str = "MPL-2.0 (LibreOffice)";
const OFFICE_PROVISIONING: &str =
    "External. Install LibreOffice from the offline deployment pack, or set ARJUN_SOFFICE to its \
     soffice executable (soffice.com on Windows).";
const RASTERISER_LICENCE: &str = "AGPL-3.0 (PyMuPDF), already required by the document sidecar";
const RASTERISER_PROVISIONING: &str =
    "Bundled script sidecars/document_sidecar/render_pages.py; PyMuPDF is installed with the \
     document sidecar's Python packages.";

fn series_of(version: &str) -> String {
    version.split('.').take(2).collect::<Vec<_>>().join(".")
}

/// Asks the machine which adapters it has. Spawns each program once.
pub fn probe() -> Inventory {
    Inventory {
        office: probe_office(),
        rasteriser: probe_rasteriser(),
    }
}

/// The inventory, probed once per process.
pub fn inventory() -> &'static Inventory {
    static CACHED: std::sync::OnceLock<Inventory> = std::sync::OnceLock::new();
    CACHED.get_or_init(probe)
}

fn office_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(path) = crate::deployment::resolve_id("office-renderer").path() {
        out.push(path.to_path_buf());
    }
    #[cfg(target_os = "windows")]
    {
        // `soffice.exe` is a GUI-subsystem program and prints nothing to a
        // pipe; `soffice.com` beside it is the console one.
        for base in [r"C:\Program Files\LibreOffice\program", r"C:\Program Files (x86)\LibreOffice\program"] {
            out.push(PathBuf::from(base).join("soffice.com"));
        }
        out.push(PathBuf::from("soffice.com"));
    }
    #[cfg(target_os = "macos")]
    out.push(PathBuf::from("/Applications/LibreOffice.app/Contents/MacOS/soffice"));
    out.push(PathBuf::from(crate::deployment::program("office-renderer")));
    out.into_iter()
        .map(|path| {
            // An override naming soffice.exe is swapped for its console twin.
            if path.file_name().is_some_and(|n| n.eq_ignore_ascii_case("soffice.exe")) {
                let console = path.with_file_name("soffice.com");
                if console.exists() {
                    return console;
                }
            }
            path
        })
        .collect()
}

fn probe_office() -> Adapter {
    let mut tried = Vec::new();
    for candidate in office_candidates() {
        let mut command = crate::system_analyzer::process_utils::create_hidden_command(&candidate);
        command.arg("--version");
        match run_bounded(command, Duration::from_secs(30)) {
            Ok(output) => {
                let banner = String::from_utf8_lossy(&output.stdout).to_string();
                // "LibreOffice 24.2.7.2 420(Build:2)"
                let version = banner
                    .split_whitespace()
                    .skip_while(|word| !word.eq_ignore_ascii_case("libreoffice"))
                    .nth(1)
                    .map(str::to_string);
                let Some(version) = version else {
                    tried.push(format!("{} answered without a version", candidate.display()));
                    continue;
                };
                let qualified = QUALIFIED_OFFICE_SERIES.contains(&series_of(&version).as_str());
                return Adapter {
                    name: "LibreOffice".to_string(),
                    available: true,
                    version: Some(version),
                    qualified,
                    qualified_series: QUALIFIED_OFFICE_SERIES.iter().map(|s| s.to_string()).collect(),
                    unavailable_because: None,
                    licence: OFFICE_LICENCE.to_string(),
                    provisioning: OFFICE_PROVISIONING.to_string(),
                    program: Some(candidate),
                    script: None,
                };
            }
            Err(error) => tried.push(format!("{}: {error}", candidate.display())),
        }
    }
    Adapter::unavailable(
        "LibreOffice",
        format!("no LibreOffice answered (tried {})", tried.join("; ")),
        OFFICE_LICENCE,
        OFFICE_PROVISIONING,
        QUALIFIED_OFFICE_SERIES,
    )
}

fn probe_rasteriser() -> Adapter {
    let unavailable = |why: String| {
        Adapter::unavailable("PyMuPDF", why, RASTERISER_LICENCE, RASTERISER_PROVISIONING, QUALIFIED_RASTERISER_SERIES)
    };
    let resolution = crate::deployment::resolve_id("page-rasteriser");
    let Some(script) = resolution.path().map(Path::to_path_buf) else {
        return unavailable(format!("the page rasteriser script was not found: {resolution:?}"));
    };
    let python = crate::deployment::program("python");
    let mut command = crate::system_analyzer::process_utils::create_hidden_command(&python);
    command.arg(&script).arg("--probe");
    let output = match run_bounded(command, Duration::from_secs(60)) {
        Ok(output) => output,
        Err(error) => return unavailable(format!("{python} could not run the rasteriser: {error}")),
    };
    let answer: serde_json::Value = match serde_json::from_slice(&output.stdout) {
        Ok(value) => value,
        Err(_) => {
            return unavailable(format!(
                "the rasteriser probe did not answer in JSON (exit {:?}): {}",
                output.status.code(),
                String::from_utf8_lossy(&output.stderr).lines().last().unwrap_or_default()
            ))
        }
    };
    if let Some(error) = answer.get("error").and_then(|e| e.as_str()) {
        return unavailable(error.to_string());
    }
    let version = answer.get("version").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
    let qualified = QUALIFIED_RASTERISER_SERIES.contains(&series_of(&version).as_str());
    Adapter {
        name: "PyMuPDF".to_string(),
        available: true,
        version: Some(version),
        qualified,
        qualified_series: QUALIFIED_RASTERISER_SERIES.iter().map(|s| s.to_string()).collect(),
        unavailable_because: None,
        licence: RASTERISER_LICENCE.to_string(),
        provisioning: RASTERISER_PROVISIONING.to_string(),
        program: Some(PathBuf::from(python)),
        script: Some(script),
    }
}

/// What rendering did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RenderState {
    /// Pages were laid out and rasterised.
    Rendered,
    /// Laid out to PDF, and the rasteriser that would show the pages is not
    /// usable here. The PDF exists; nobody has looked at a page of it.
    PdfOnly,
    /// An adapter this format needs is not usable on this machine.
    Unavailable,
    /// The file is not safe to hand a renderer.
    Refused,
    /// A renderer ran and did not produce pages.
    Failed,
    /// The format has no pages (plain text).
    Unsupported,
}

/// One rendered page.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PageRender {
    pub page: u32,
    /// The PNG's file name inside the render directory.
    pub image: String,
    pub image_sha256: String,
    pub width: u32,
    pub height: u32,
    pub blank: bool,
    pub text_characters: usize,
    #[serde(default, skip_serializing)]
    pub text: String,
}

/// The adapter and version that produced a render.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RendererUsed {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderOutcome {
    pub state: RenderState,
    pub detail: String,
    #[serde(default)]
    pub problems: Vec<String>,
    pub renderers: Vec<RendererUsed>,
    /// Pages the laid-out document has. Zero when nothing was laid out.
    pub total_pages: u32,
    pub pages: Vec<PageRender>,
    /// The intermediate PDF's hash, when one was produced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pdf_sha256: Option<String>,
}

impl RenderOutcome {
    fn with(state: RenderState, detail: impl Into<String>, problems: Vec<String>) -> Self {
        RenderOutcome {
            state,
            detail: detail.into(),
            problems,
            renderers: Vec::new(),
            total_pages: 0,
            pages: Vec::new(),
            pdf_sha256: None,
        }
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// An SVG's references that leave the file.
fn svg_external_references(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let lower = text.to_ascii_lowercase();
    for marker in ["href=\"", "href='", "src=\"", "src='", "url("] {
        let mut rest = lower.as_str();
        while let Some(at) = rest.find(marker) {
            let value = rest[at + marker.len()..].trim_start_matches(['"', '\'', ' ']);
            let internal = value.starts_with('#') || value.starts_with("data:");
            if !internal {
                out.push(value.chars().take(60).collect());
            }
            rest = &rest[at + marker.len()..];
        }
    }
    out
}

/// Renders `bytes` into `out_dir`: laid out to PDF where it is not one, then
/// rasterised for pages `from..=to`.
pub fn render(
    bytes: &[u8],
    format: DetectedFormat,
    out_dir: &Path,
    from: u32,
    to: u32,
    inventory: &Inventory,
) -> RenderOutcome {
    if from == 0 || to < from {
        return RenderOutcome::with(RenderState::Failed, "the page range must start at 1 and run forward", Vec::new());
    }
    let to = to.min(from + MAX_PAGES_PER_RENDER - 1);
    if let Err(error) = std::fs::create_dir_all(out_dir) {
        return RenderOutcome::with(RenderState::Failed, format!("the render directory could not be made: {error}"), Vec::new());
    }

    let mut renderers = Vec::new();
    let pdf: Vec<u8> = match format {
        DetectedFormat::Pdf => bytes.to_vec(),
        DetectedFormat::Docx | DetectedFormat::Pptx | DetectedFormat::Xlsx | DetectedFormat::Svg => {
            let refusals = if format == DetectedFormat::Svg {
                svg_external_references(&String::from_utf8_lossy(bytes))
                    .into_iter()
                    .map(|target| format!("the drawing references {target}, outside the file"))
                    .collect()
            } else {
                package::inspect(bytes, &package::LIMITS).render_refusals()
            };
            if !refusals.is_empty() {
                return RenderOutcome::with(
                    RenderState::Refused,
                    "The file was not handed to a renderer, because rendering it could fetch or run something.",
                    refusals,
                );
            }
            if let Some(why) = inventory.office.why_not() {
                return RenderOutcome::with(RenderState::Unavailable, why, Vec::new());
            }
            match office_to_pdf(bytes, format, out_dir, &inventory.office) {
                Ok(pdf) => {
                    renderers.push(RendererUsed {
                        name: inventory.office.name.clone(),
                        version: inventory.office.version.clone().unwrap_or_default(),
                    });
                    pdf
                }
                Err(error) => return RenderOutcome::with(RenderState::Failed, error, Vec::new()),
            }
        }
        DetectedFormat::Text => {
            return RenderOutcome::with(
                RenderState::Unsupported,
                "Plain text has no pages to render; it is read, not laid out.",
                Vec::new(),
            )
        }
        other => {
            return RenderOutcome::with(
                RenderState::Refused,
                format!("A {} is not something this renders.", other.label()),
                Vec::new(),
            )
        }
    };
    let pdf_sha256 = sha256_hex(&pdf);
    let pdf_path = out_dir.join("rendered.pdf");
    if let Err(error) = std::fs::write(&pdf_path, &pdf) {
        return RenderOutcome::with(RenderState::Failed, format!("the PDF could not be kept: {error}"), Vec::new());
    }

    if let Some(why) = inventory.rasteriser.why_not() {
        let mut outcome = RenderOutcome::with(
            RenderState::PdfOnly,
            format!("Laid out to PDF; no page was rasterised because {why}."),
            Vec::new(),
        );
        outcome.renderers = renderers;
        outcome.pdf_sha256 = Some(pdf_sha256);
        return outcome;
    }

    let (Some(python), Some(script)) = (&inventory.rasteriser.program, &inventory.rasteriser.script) else {
        return RenderOutcome::with(RenderState::Unavailable, "the rasteriser has no program recorded", Vec::new());
    };
    let mut command = crate::system_analyzer::process_utils::create_hidden_command(python);
    command
        .arg(script)
        .arg(&pdf_path)
        .arg(out_dir)
        .args(["--from", &from.to_string(), "--to", &to.to_string()]);
    let output = match run_bounded(command, RENDER_TIMEOUT) {
        Ok(output) => output,
        Err(error) => return RenderOutcome::with(RenderState::Failed, format!("the rasteriser did not finish: {error}"), Vec::new()),
    };
    let answer: serde_json::Value = match serde_json::from_slice(&output.stdout) {
        Ok(answer) => answer,
        Err(_) => {
            return RenderOutcome::with(
                RenderState::Failed,
                format!("the rasteriser did not answer in JSON (exit {:?})", output.status.code()),
                Vec::new(),
            )
        }
    };
    if let Some(error) = answer.get("error").and_then(|e| e.as_str()) {
        let state = if output.status.code() == Some(3) { RenderState::Unavailable } else { RenderState::Failed };
        return RenderOutcome::with(state, error.to_string(), Vec::new());
    }
    renderers.push(RendererUsed {
        name: "PyMuPDF".to_string(),
        version: answer.get("version").and_then(|v| v.as_str()).unwrap_or("unknown").to_string(),
    });

    let total_pages = answer.get("pages").and_then(|p| p.as_u64()).unwrap_or(0) as u32;
    let mut pages = Vec::new();
    let mut problems = Vec::new();
    for page in answer.get("rendered").and_then(|r| r.as_array()).cloned().unwrap_or_default() {
        let number = page.get("page").and_then(|p| p.as_u64()).unwrap_or(0) as u32;
        let image = page.get("image").and_then(|i| i.as_str()).unwrap_or_default().to_string();
        // Only a bare file name inside this directory is accepted back.
        if image.is_empty() || image.contains(['/', '\\']) || image.contains("..") {
            problems.push(format!("page {number} came back with an unusable image name"));
            continue;
        }
        let Ok(png) = std::fs::read(out_dir.join(&image)) else {
            problems.push(format!("page {number}'s image was not written"));
            continue;
        };
        if !png.starts_with(b"\x89PNG") {
            problems.push(format!("page {number}'s image is not a PNG"));
            continue;
        }
        let text = page.get("text").and_then(|t| t.as_str()).unwrap_or_default().to_string();
        let blank = page.get("blank").and_then(|b| b.as_bool()).unwrap_or(false);
        if blank {
            problems.push(format!("page {number} renders blank"));
        }
        pages.push(PageRender {
            page: number,
            image,
            image_sha256: sha256_hex(&png),
            width: page.get("width").and_then(|w| w.as_u64()).unwrap_or(0) as u32,
            height: page.get("height").and_then(|h| h.as_u64()).unwrap_or(0) as u32,
            blank,
            text_characters: text.chars().filter(|c| !c.is_whitespace()).count(),
            text,
        });
    }
    if total_pages == 0 {
        problems.push("the laid-out document has no pages".to_string());
    }
    let state = if pages.is_empty() { RenderState::Failed } else { RenderState::Rendered };
    RenderOutcome {
        state,
        detail: format!(
            "Laid out to {total_pages} page(s); rasterised {} of them (pages {from}-{}).",
            pages.len(),
            to.min(total_pages.max(from))
        ),
        problems,
        renderers,
        total_pages,
        pages,
        pdf_sha256: Some(pdf_sha256),
    }
}

/// The macro and link settings every throwaway profile starts with.
const PROFILE_SETTINGS: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<oor:items xmlns:oor="http://openoffice.org/2001/registry" xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<item oor:path="/org.openoffice.Office.Common/Security/Scripting"><prop oor:name="MacroSecurityLevel" oor:op="fuse"><value>3</value></prop></item>
<item oor:path="/org.openoffice.Office.Common/Security/Scripting"><prop oor:name="DisableMacrosExecution" oor:op="fuse"><value>true</value></prop></item>
<item oor:path="/org.openoffice.Office.Common/Security/Scripting"><prop oor:name="BlockUntrustedRefererLinks" oor:op="fuse"><value>true</value></prop></item>
</oor:items>
"#;

fn file_url(path: &Path) -> String {
    let text = path.display().to_string().replace('\\', "/");
    if text.starts_with('/') {
        format!("file://{text}")
    } else {
        format!("file:///{text}")
    }
}

fn office_to_pdf(bytes: &[u8], format: DetectedFormat, out_dir: &Path, office: &Adapter) -> Result<Vec<u8>, String> {
    let program = office.program.as_ref().ok_or("LibreOffice has no program recorded")?;
    let extension = match format {
        DetectedFormat::Docx => "docx",
        DetectedFormat::Pptx => "pptx",
        DetectedFormat::Xlsx => "xlsx",
        DetectedFormat::Svg => "svg",
        _ => return Err("only Office files and drawings are laid out".to_string()),
    };
    let work = out_dir.join("work");
    let profile = work.join("profile");
    let user = profile.join("user");
    std::fs::create_dir_all(&user).map_err(|e| format!("the renderer profile could not be made: {e}"))?;
    std::fs::write(user.join("registrymodifications.xcu"), PROFILE_SETTINGS)
        .map_err(|e| format!("the renderer profile could not be written: {e}"))?;
    let input = work.join(format!("input.{extension}"));
    std::fs::write(&input, bytes).map_err(|e| format!("the input could not be staged: {e}"))?;

    let mut command = crate::system_analyzer::process_utils::create_hidden_command(program);
    command
        .arg("--headless")
        .arg("--norestore")
        .arg("--nolockcheck")
        .arg("--nodefault")
        .arg("--nologo")
        .arg(format!("-env:UserInstallation={}", file_url(&profile)))
        .arg("--convert-to")
        .arg("pdf")
        .arg("--outdir")
        .arg(&work)
        .arg(&input)
        .env("HOME", &work)
        .current_dir(&work);
    let started = Instant::now();
    let output = run_bounded(command, RENDER_TIMEOUT)?;
    let produced = work.join("input.pdf");
    let pdf = std::fs::read(&produced).map_err(|_| {
        format!(
            "LibreOffice ran for {:.1}s and produced no PDF (exit {:?}): {}",
            started.elapsed().as_secs_f32(),
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
                .lines()
                .filter(|l| !l.contains("javaldx"))
                .last()
                .unwrap_or("no message")
        )
    })?;
    if !pdf.starts_with(b"%PDF-") {
        return Err("LibreOffice wrote something that is not a PDF".to_string());
    }
    // The staged copy and the profile are not kept: the stored bytes are the
    // record, and the profile is per-render by design.
    let _ = std::fs::remove_dir_all(&work);
    Ok(pdf)
}

/// Runs a process with piped output, killing it at `timeout`.
pub fn run_bounded(mut command: Command, timeout: Duration) -> Result<std::process::Output, String> {
    command.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command.spawn().map_err(|e| format!("it could not be started: {e}"))?;
    let mut stdout = child.stdout.take();
    let mut stderr = child.stderr.take();
    let out_reader = std::thread::spawn(move || {
        let mut buffer = Vec::new();
        if let Some(pipe) = stdout.as_mut() {
            let _ = pipe.take(8 * 1024 * 1024).read_to_end(&mut buffer);
        }
        buffer
    });
    let err_reader = std::thread::spawn(move || {
        let mut buffer = Vec::new();
        if let Some(pipe) = stderr.as_mut() {
            let _ = pipe.take(1024 * 1024).read_to_end(&mut buffer);
        }
        buffer
    });
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() >= deadline => {
                // On Windows `soffice.com` starts `soffice.bin` and waits on
                // it; killing only the parent would leave the renderer running.
                #[cfg(target_os = "windows")]
                {
                    let _ = crate::system_analyzer::process_utils::create_hidden_command("taskkill")
                        .args(["/F", "/T", "/PID", &child.id().to_string()])
                        .output();
                }
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("it was stopped after {}s", timeout.as_secs()));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(25)),
            Err(error) => return Err(format!("it could not be waited on: {error}")),
        }
    };
    Ok(std::process::Output {
        status,
        stdout: out_reader.join().unwrap_or_default(),
        stderr: err_reader.join().unwrap_or_default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nothing_here() -> Inventory {
        Inventory {
            office: Adapter::unavailable("LibreOffice", "not installed in this test".into(), OFFICE_LICENCE, OFFICE_PROVISIONING, QUALIFIED_OFFICE_SERIES),
            rasteriser: Adapter::unavailable("PyMuPDF", "not installed in this test".into(), RASTERISER_LICENCE, RASTERISER_PROVISIONING, QUALIFIED_RASTERISER_SERIES),
        }
    }

    #[test]
    fn a_missing_renderer_is_reported_unavailable_and_never_rendered() {
        let dir = tempfile::tempdir().unwrap();
        let bytes = crate::artifacts::content::tests_support::sample_docx(dir.path());
        let outcome = render(&bytes, DetectedFormat::Docx, &dir.path().join("r"), 1, 5, &nothing_here());
        assert_eq!(outcome.state, RenderState::Unavailable);
        assert!(outcome.detail.contains("not installed"), "{}", outcome.detail);
        assert!(outcome.pages.is_empty());
    }

    #[test]
    fn an_unqualified_version_is_not_usable() {
        let mut office = nothing_here().office;
        office.available = true;
        office.version = Some("7.3.7.2".into());
        office.qualified = QUALIFIED_OFFICE_SERIES.contains(&series_of("7.3.7.2").as_str());
        assert!(!office.usable());
        assert!(office.why_not().unwrap().contains("7.3.7.2"));
    }

    #[test]
    fn an_svg_that_reaches_outside_itself_is_refused_before_any_renderer() {
        let dir = tempfile::tempdir().unwrap();
        let svg = br##"<svg viewBox="0 0 10 10"><image href="https://tracker.example/x.png"/><use href="#a"/></svg>"##;
        let outcome = render(svg, DetectedFormat::Svg, dir.path(), 1, 1, &nothing_here());
        assert_eq!(outcome.state, RenderState::Refused, "{outcome:?}");
        assert!(outcome.problems[0].contains("tracker.example"));
    }

    #[test]
    fn plain_text_has_no_pages_and_says_so() {
        let dir = tempfile::tempdir().unwrap();
        let outcome = render(b"notes", DetectedFormat::Text, dir.path(), 1, 1, &nothing_here());
        assert_eq!(outcome.state, RenderState::Unsupported);
    }

    /// Real adapters, when this machine has them. When it does not, the test
    /// asserts the honest state instead of passing silently: the inventory
    /// must say *why* each is unusable.
    #[test]
    fn the_real_adapters_render_a_produced_deck_page_for_slide_or_say_why_not() {
        let inventory = probe();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("deck.pptx");
        let deck = crate::artifacts::doc_model::Deck {
            title: "Render check".into(),
            classification: "Internal".into(),
            slides: vec![crate::artifacts::doc_model::SlideModel {
                heading: "Findings".into(),
                bullets: vec!["Point C is below the minimum.".into()],
                table: None,
                notes: None,
            }],
        };
        crate::artifacts::pptx::write_deck_model(&path, &deck, false).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let outcome = render(&bytes, DetectedFormat::Pptx, &dir.path().join("render"), 1, 10, &inventory);
        eprintln!("RENDER-INVENTORY {}", serde_json::to_string(&inventory).unwrap());
        eprintln!("RENDER-OUTCOME {:?} {} {:?}", outcome.state, outcome.detail, outcome.renderers);
        if inventory.office.usable() && inventory.rasteriser.usable() {
            assert_eq!(outcome.state, RenderState::Rendered, "{outcome:?}");
            assert_eq!(outcome.total_pages, 2, "one page per slide, title slide included");
            assert!(outcome.pages.iter().all(|p| !p.blank && p.text_characters > 0), "{:?}", outcome.pages);
            assert!(outcome.pages[1].text.contains("Findings"));
            assert!(dir.path().join("render").join(&outcome.pages[0].image).is_file());
        } else {
            assert!(
                matches!(outcome.state, RenderState::Unavailable | RenderState::PdfOnly),
                "{outcome:?}"
            );
            assert!(inventory.office.why_not().is_some() || inventory.rasteriser.why_not().is_some());
        }
    }
}

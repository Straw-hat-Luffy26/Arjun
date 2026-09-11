//! Commands behind the document scan view.
//!
//! The slider in the UI has to describe what it will actually do — which
//! weight file a stop loads, and how much of the page the model will be
//! allowed to see. Those numbers live in [`crate::ai_engine::ocr_profile`],
//! and this command hands them to the frontend rather than letting the UI
//! keep a second copy. A slider whose labels disagree with the profiles that
//! run is worse than no slider: it reports a configuration nobody is using.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::agent_runtime::cancellation::CancelToken;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};

use crate::ai_engine::ocr_profile::{
    to_page, CoordSpace, OcrDetent, OcrTier, PageBox, PageGeometry,
};
use crate::ai_engine::ocr_repetition::degenerate_tail_start;
use crate::ai_engine::ocr_spans::{OcrEvent, RawBox};
use crate::ai_engine::ocr_stream::stream_ocr;
use crate::agent_runtime::stages::StageTag;
use crate::registry::ModelRegistry;
use crate::serving::ModelServers;

/// One slider stop, as the UI needs to render it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrDetentInfo {
    pub detent: OcrDetent,
    pub label: String,
    pub tier: OcrTier,
    pub tier_label: String,
    pub max_image_tokens: u32,
    pub max_decode_tokens: u32,
}

/// The stops, fastest first.
pub fn detent_info() -> Vec<OcrDetentInfo> {
    OcrDetent::ALL
        .iter()
        .map(|detent| {
            let profile = detent.profile();
            OcrDetentInfo {
                detent: *detent,
                label: detent.label().to_string(),
                tier: profile.tier,
                tier_label: profile.tier.label().to_string(),
                max_image_tokens: profile.max_image_tokens,
                max_decode_tokens: profile.max_decode_tokens,
            }
        })
        .collect()
}

#[tauri::command]
pub fn get_ocr_detents() -> Vec<OcrDetentInfo> {
    detent_info()
}

/// The coordinate convention this build's model reports in.
///
/// **Measured, not assumed.** The same page was read at 1000x1400 and at
/// 731x1024; the emitted boxes were identical to within a pixel
/// (`title [77, 51, 723, 86]` vs `[77, 52, 723, 86]`). Boxes that do not move
/// with the input size are normalised, not input pixels — that comparison is
/// the whole discriminator, and it is encoded as a test in
/// [`crate::ai_engine::ocr_profile`].
///
/// Cross-checked against known ink positions: the page's bottom-right marker
/// sits at y=1299 of 1400, and the model reported y=923, where normalisation
/// predicts 1299/1400*999 = 927.
const CALIBRATED_COORD_SPACE: Option<CoordSpace> = Some(CoordSpace::Normalised);

/// Width and height from a PNG's IHDR chunk.
///
/// Read directly rather than pulling in an image decoder: the overlay only
/// needs the page's dimensions, and the header carries them at a fixed offset
/// (8-byte signature, 4-byte length, `IHDR`, then two big-endian u32s).
fn png_dimensions(path: &std::path::Path) -> Result<(u32, u32), String> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("could not read {}: {e}", path.display()))?;
    if bytes.len() < 24 || &bytes[..8] != b"\x89PNG\r\n\x1a\n" || &bytes[12..16] != b"IHDR" {
        return Err(format!("{} is not a PNG", path.display()));
    }
    let w = u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    let h = u32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
    if w == 0 || h == 0 {
        return Err(format!("{} reports a zero dimension", path.display()));
    }
    Ok((w, h))
}

/// A rendered page, ready for the scan view to display.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PageImage {
    /// A `data:` URI.
    ///
    /// The page lives in app data, which the webview cannot reach by path,
    /// and enabling Tauri's asset protocol would open a filesystem route for
    /// the sake of one image. The CSP already allows `data:` for images, and
    /// the vision bridge sends pages to the model the same way — so the page
    /// is handed over as bytes rather than as a path.
    pub data_url: String,
    /// The overlay's coordinate space. Read from the file rather than assumed,
    /// so a page that is not 1000x1400 still gets boxes in the right place.
    pub width: u32,
    pub height: u32,
}

/// Loads one rendered page for display.
#[tauri::command]
pub fn get_page_image(
    app: AppHandle,
    document_sha256: String,
    page: u32,
) -> Result<PageImage, String> {
    let path = page_image_path(&app, &document_sha256, page)?;
    let (width, height) = png_dimensions(&path)?;
    let bytes =
        std::fs::read(&path).map_err(|e| format!("could not read {}: {e}", path.display()))?;
    Ok(PageImage {
        data_url: format!(
            "data:image/png;base64,{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &bytes)
        ),
        width,
        height,
    })
}

/// The largest attachment that will be read.
///
/// A page scan is a few hundred kilobytes; anything approaching this is not a
/// document. Bounded here rather than at the model, because a refusal the user
/// can read beats a request that dies inside llama-server.
const MAX_ATTACHMENT_BYTES: usize = 24 * 1024 * 1024;

/// One file the user attached to a chat turn.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatAttachment {
    /// The name it arrived under, for display and for the prompt. Never used
    /// as a path — the bytes are addressed by their own hash.
    pub name: String,
    pub mime: String,
    /// Base64 of the file itself. The bytes cross the boundary, not a path:
    /// the webview has no filesystem the backend could re-open.
    pub data_base64: String,
}

/// What reading one attachment produced.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AttachmentRead {
    pub name: String,
    /// Content address of the stored bytes. Two users attaching the same file
    /// converge on one copy, and the id cannot be confused between runs.
    pub sha256: String,
    /// What the OCR model actually read. Never synthesised.
    pub text: String,
    /// Which local path handled it: `image`, `pdf-scan`, `pdf-text`, `docx`…
    pub kind: String,
    /// How many pages were read. One for an image.
    pub pages: u32,
    /// The OCR model that read it, or `None` when no model was needed
    /// because the file already carried its text.
    ///
    /// Reported rather than inferred from `kind`: the routing reasons shown
    /// to the person name this model, and a label that disagrees with what
    /// ran is worse than no label.
    pub ocr_model_id: Option<String>,
    /// The slider stop the read actually ran at.
    pub ocr_detent: Option<OcrDetent>,
    /// The same text, still split by page.
    ///
    /// `text` above is the assembled blob the prompt is composed from, and the
    /// assembly is lossy in the one way that matters here: the page numbers
    /// survive only as `--- page 7 of 40 ---` headings inside a string, which is
    /// not something anything downstream can index by.
    ///
    /// This is what [`crate::agent_runtime::documents`] stores, so a page the
    /// budget left out of the prompt can be asked for later by its number. A
    /// reader with no pages of its own — a spreadsheet, a deck, a text file —
    /// records its whole output as page 1, which is what its page count says it
    /// is. Skipped in the wire form: it is a second copy of `text`, it is large,
    /// and nothing across the boundary reads it.
    #[serde(skip)]
    pub page_text: std::collections::BTreeMap<u32, String>,
    /// True when the reader itself stopped early — a workbook past its row cap.
    ///
    /// Distinct from the context budget's truncation, which happens later and
    /// for a different reason. A document cut by its *reader* is not made whole
    /// by reading its pages back, and the store carries the difference.
    pub truncated: bool,
}

fn attachment_extension(mime: &str) -> Option<&'static str> {
    match mime {
        "image/png" => Some("png"),
        "image/jpeg" | "image/jpg" => Some("jpg"),
        "image/webp" => Some("webp"),
        _ => None,
    }
}

/// What kind of handling a file needs.
///
/// Routing by type is the whole design. Forcing a PDF through OCR would
/// render pages to read text that is already in the file — slower, lossier,
/// and pointless GPU work. Only images and scans reach the vision model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttachmentKind {
    /// Straight to the OCR model, unchanged from the path that already works.
    Image(&'static str),
    /// Through the local extractor first; it decides whether OCR is needed.
    Document(&'static str),
}

/// What will happen to a file, decided before a byte of it is sent anywhere.
///
/// The composer shows this so "an OCR model will read this" is visible while
/// the person is still typing, rather than being discovered afterwards from a
/// progress line. It is the same [`validate_attachment`] the run itself uses,
/// so a plan that says a file is unreadable and a run that accepts it cannot
/// disagree.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AttachmentPlan {
    pub name: String,
    /// `image` | `document` | `rejected`.
    pub route: &'static str,
    /// True when a vision model has to look at the page. A PDF carrying its
    /// own text layer is `false` — rendering it to pixels to read back text
    /// the file was already holding is pointless GPU work.
    ///
    /// For a document this is *possible* OCR, not certain: whether a PDF is a
    /// scan is only known once the extractor has opened it.
    pub needs_ocr: bool,
    /// Why, in the person's terms. Shown verbatim.
    pub explanation: String,
    /// Set when the file cannot be read at all; the same sentence the run
    /// would have refused with.
    pub refusal: Option<String>,
}

/// The plan for one file, from its name and MIME type alone.
pub fn plan_attachment(name: &str, mime: &str) -> AttachmentPlan {
    // A length of one: the size gate is a separate question and the composer
    // has not weighed the bytes. Only the type decision is being made here.
    match validate_attachment(name, mime, 1) {
        Ok(AttachmentKind::Image(_)) => AttachmentPlan {
            name: name.to_string(),
            route: "image",
            needs_ocr: true,
            explanation: format!(
                "{name} is an image, so the document-OCR model reads it on this device before the answer is composed."
            ),
            refusal: None,
        },
        Ok(AttachmentKind::Document(ext)) => {
            let scan_possible = ext == "pdf";
            AttachmentPlan {
                name: name.to_string(),
                route: "document",
                needs_ocr: scan_possible,
                explanation: if scan_possible {
                    format!(
                        "{name} is a PDF. Its text layer is read directly; if there is none it is a scan, and each page goes to the document-OCR model."
                    )
                } else {
                    format!(
                        "{name} carries its own text, so it is extracted locally and no model is needed to read it."
                    )
                },
                refusal: None,
            }
        }
        Err(refusal) => AttachmentPlan {
            name: name.to_string(),
            route: "rejected",
            needs_ocr: false,
            explanation: refusal.clone(),
            refusal: Some(refusal),
        },
    }
}

/// One file the composer is about to send, named and typed.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AttachmentDescriptor {
    pub name: String,
    pub mime: String,
}

/// What the attached files will be routed to, before the turn is sent.
#[tauri::command]
pub fn preview_attachment_routing(files: Vec<AttachmentDescriptor>) -> Vec<AttachmentPlan> {
    files
        .iter()
        .map(|f| plan_attachment(&f.name, &f.mime))
        .collect()
}

/// Extensions the local extractor can turn into text without any model.
///
/// `pptx` is here because a board presentation is one of the document kinds
/// this product exists to handle, and it was the one Office format the
/// extractor could not open — a deck attached to a chat was refused outright
/// with a message listing every other format.
const DOCUMENT_SUFFIXES: &[&str] = &[
    "pdf", "txt", "md", "markdown", "csv", "json", "log", "tsv", "docx", "xlsx", "pptx",
];

/// Everything about an attachment that can be judged before touching disk.
///
/// Split out so the limits are testable without a running app: the refusal
/// messages are what the person actually sees, and a silent acceptance here
/// is how an unreadable file becomes an answer about a document nobody read.
fn validate_attachment(name: &str, mime: &str, len: usize) -> Result<AttachmentKind, String> {
    let suffix = name
        .rsplit('.')
        .next()
        .filter(|s| *s != name)
        .unwrap_or("")
        .to_ascii_lowercase();

    // MIME first, because the browser knows better than the name; the suffix
    // is the fallback for picks where the browser reported nothing.
    let kind = if let Some(ext) = attachment_extension(mime) {
        AttachmentKind::Image(ext)
    } else if let Some(ext) = DOCUMENT_SUFFIXES.iter().find(|e| **e == suffix) {
        AttachmentKind::Document(ext)
    } else {
        return Err(format!(
            "{name} is a {mime} — ARJUN reads PDF, Word, Excel, PowerPoint, text, Markdown, CSV, JSON and image files."
        ));
    };

    if len == 0 {
        return Err(format!("{name} is empty."));
    }
    if len > MAX_ATTACHMENT_BYTES {
        return Err(format!(
            "{name} is {:.1} MB; the limit is {} MB.",
            len as f64 / 1_048_576.0,
            MAX_ATTACHMENT_BYTES / 1_048_576
        ));
    }
    Ok(kind)
}

/// Where the one-shot document extractor lives.
///
/// Delegated to [`crate::deployment`], which tries the installer's resource
/// directory before the checkout. The candidate list this replaced walked
/// outwards from `current_dir()` — the repository root when a developer runs
/// `tauri dev`, and whatever directory Windows felt like when the installed app
/// starts from a Start menu shortcut. So the extractor was found on every
/// machine that had the source tree, and on no machine that had only the
/// installer.
fn extractor_script() -> Result<PathBuf, String> {
    crate::deployment::require_path("document-extractor")
}

/// How the extractor handled one page of a PDF.
///
/// One of these per page, in page order, every page present. `page` is the
/// number the person sees on the document, which is why the merge below keys
/// on it rather than on a position in a list: when only some pages are
/// rendered, a position in the rendered subset is not a page number, and
/// labelling OCR output by position reported page 5 of a scan as page 1.
#[derive(Debug, Clone, Deserialize)]
struct PageDetail {
    page: u32,
    /// `text` — read from the PDF's own text layer.
    /// `ocr` — rendered to an image for the OCR model.
    /// `unread` — neither was possible; `why` says what happened.
    source: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    image: Option<PathBuf>,
    /// Sub-threshold text the page did carry, if any.
    ///
    /// Used only when OCR comes back empty. A page holding "Figure 3
    /// continued" over an illegible scan should report that much.
    #[serde(default, rename = "layerText")]
    layer_text: String,
    #[serde(default)]
    why: String,
}

/// What the extractor reported about one file.
#[derive(Debug, Clone, Deserialize)]
struct Extracted {
    kind: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    pages: u32,
    /// Present for PDFs only, and then for every page.
    ///
    /// Supersedes the old `pageImages` list, which said which pages had been
    /// rendered and nothing about the rest — so a page that was neither read
    /// nor rendered simply did not appear anywhere, and the difference between
    /// "blank" and "skipped" was unrecoverable on this side.
    #[serde(default, rename = "pageDetail")]
    page_detail: Vec<PageDetail>,
    #[serde(default)]
    truncated: bool,
    #[serde(default)]
    error: Option<String>,
}

/// The pages the extractor already read, and the ones it could not.
///
/// Split out from the read loop so it can be tested without an application: it
/// is the half of the merge that decides which pages exist at all, and the
/// reported bug was a page that existed and appeared in neither list.
fn seed_from_detail(
    detail: &[PageDetail],
) -> (std::collections::BTreeMap<u32, String>, Vec<(u32, String)>) {
    let mut by_page = std::collections::BTreeMap::new();
    let mut unread = Vec::new();
    for page in detail {
        match page.source.as_str() {
            "text" if !page.text.trim().is_empty() => {
                by_page.insert(page.page, page.text.trim().to_string());
            }
            // A text page with nothing on it is a page nothing was made of,
            // which is the thing this whole change exists to stop losing.
            "text" => unread.push((page.page, "it has nothing on it".to_string())),
            "unread" => unread.push((page.page, page.why.clone())),
            _ => {}
        }
    }
    (by_page, unread)
}

/// What one page is worth once the model has had its turn.
///
/// `Ok` is the text to keep for the page; `Err` is the reason it counts as
/// unread. The empty case is the one that matters: a rendered page the model
/// made nothing of used to be dropped here without a word — the same silence
/// as the whole-file misclassification, one page down. A blank scan and a page
/// that was never looked at produce identical output if neither is mentioned.
fn settle_ocr_page(ocr_text: &str, layer_text: &str) -> Result<String, String> {
    if !ocr_text.trim().is_empty() {
        return Ok(ocr_text.trim().to_string());
    }
    if !layer_text.trim().is_empty() {
        // The model saw nothing, but the page did carry something.
        return Ok(layer_text.trim().to_string());
    }
    Err("it was read but nothing could be made out on it".to_string())
}

/// Turns the read pages into the text the model sees.
///
/// Two rules, both from the failure this replaced. Pages come out in document
/// order carrying their own numbers, so a citation to "page 4" means page 4 of
/// the file the person is holding. And every page that could not be read is
/// named, because an answer written over a document with a silent hole in it
/// is indistinguishable from one written over the whole thing.
fn assemble_pdf_text(
    by_page: std::collections::BTreeMap<u32, String>,
    mut unread: Vec<(u32, String)>,
    pages: u32,
) -> String {
    let mut parts: Vec<String> = by_page
        .iter()
        .map(|(page, text)| {
            if pages > 1 {
                format!("--- page {page} of {pages} ---\n{text}")
            } else {
                text.clone()
            }
        })
        .collect();

    if !unread.is_empty() {
        unread.sort_by_key(|(page, _)| *page);
        unread.dedup_by_key(|(page, _)| *page);
        let listed = unread
            .iter()
            .map(|(page, why)| format!("page {page} ({why})"))
            .collect::<Vec<_>>()
            .join(", ");
        // Said plainly rather than letting an answer describe a document as
        // though all of it had been read.
        parts.push(format!(
            "({} of the {} pages could not be read: {}. Nothing above comes from them, so do \
             not describe what is on them.)",
            unread.len(),
            pages,
            listed
        ));
    }
    parts.join("\n\n")
}

/// What the UI shows while a document is being read.
///
/// Every field is a fact the backend actually has. `page`/`pages` are filled
/// in only once the extractor has counted them, so "Reading page 2 of 6" is
/// never a guess — before that, the phase alone is shown.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AttachmentProgress {
    pub name: String,
    /// `reading` | `preparing` | `extracting` | `understanding` | `done`
    pub phase: &'static str,
    pub page: Option<u32>,
    pub pages: Option<u32>,
    /// Which local path handled it, so the chip can say how it was read.
    pub kind: Option<String>,
    /// The turn this read belongs to.
    ///
    /// This channel used to carry only a filename, which was enough while one
    /// window read one file at a time and wrong the moment it did not: a
    /// progress line has to land on the turn that asked for the read, and a
    /// name cannot say which turn that is. Carried as the caller's own ids
    /// because the read happens before the run has one of its own.
    pub correlation_id: Option<String>,
    pub message_id: Option<String>,
    pub conversation_id: Option<String>,
}

fn progress(
    app: &AppHandle,
    tag: &StageTag,
    name: &str,
    phase: &'static str,
    page: Option<u32>,
    pages: Option<u32>,
    kind: Option<String>,
) {
    let _ = app.emit(
        "attachment:progress",
        AttachmentProgress {
            name: name.to_string(),
            phase,
            page,
            pages,
            kind,
            correlation_id: tag.correlation_id.clone(),
            message_id: tag.message_id.clone(),
            conversation_id: tag.conversation_id.clone(),
        },
    );
}

/// What the OCR model is doing to an attachment right now, character by
/// character.
///
/// A phase label ("Understanding document…") says a model is busy; it does
/// not show the reading. These events carry the model's own output as it
/// arrives, so the person watches the page being transcribed and can tell a
/// good read from a bad one while it happens rather than afterwards.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase", tag = "event")]
pub enum AttachmentOcrEvent {
    /// The model committed to a region and said what kind of thing it is.
    #[serde(rename_all = "camelCase")]
    Region {
        name: String,
        page: u32,
        index: usize,
        /// `title`, `text`, `table`, `figure`, `footer` — the model's label.
        label: String,
    },
    /// Transcribed characters. `index` is the region they belong to, or
    /// `None` for a line the model did not open a region for.
    #[serde(rename_all = "camelCase")]
    Text {
        name: String,
        page: u32,
        index: Option<usize>,
        delta: String,
    },
    /// One page finished, with what it cost. Emitted per page so a six-page
    /// scan shows six completions rather than one at the very end.
    #[serde(rename_all = "camelCase")]
    Page {
        name: String,
        page: u32,
        pages: u32,
        model_id: String,
        detent: OcrDetent,
        characters: usize,
        elapsed_ms: u64,
        /// True when the read stopped because it ran out of decode budget
        /// rather than because the model finished.
        ///
        /// It is the difference between "this is the page" and "this is as
        /// much as fitted", and a looping read produces the second while
        /// looking exactly like the first.
        hit_decode_cap: bool,
        /// True when the read degenerated into repetition and was cut.
        ///
        /// Distinct from `hit_decode_cap`, and the more common of the two now
        /// that the loop is stopped early: a page cut by the repetition guard
        /// never reaches the decode cap, so reporting only the cap would make
        /// every stopped loop look like a clean read. `characters` counts what
        /// survived the cut, not what the model emitted.
        looped: bool,
    },
}

/// Sends one already-stored image to the OCR model and returns what it read.
///
/// This is the path verified end to end against the real model, so it is
/// reused verbatim for both a directly attached image and a rendered page of
/// a scanned PDF rather than reimplemented for each.
///
/// `detent` is the slider stop the person chose. It used to be hard-coded to
/// `Detailed`, which made the chat path unable to trade accuracy for speed at
/// all — the slider existed only on the scan screen.
#[allow(clippy::too_many_arguments)]
async fn ocr_one_image(
    app: &AppHandle,
    registry: &ModelRegistry,
    servers: &ModelServers,
    image: &std::path::Path,
    detent: OcrDetent,
    // The attachment this page belongs to, for the events.
    name: &str,
    page: u32,
    pages: u32,
    // The turn's stop signal.
    //
    // This used to be a fresh `Arc::new(AtomicBool::new(false))` built at
    // the call below — a flag nothing else held a reference to and nothing
    // could ever set. The read loop dutifully tested it on every chunk and
    // it was false every time, so the code read as though a page could be
    // stopped while in fact no page ever could.
    cancel: &crate::agent_runtime::cancellation::CancelToken,
) -> Result<String, String> {
    // Cheap, and first: a turn stopped while an earlier page was decoding
    // must not start this one. `stream_ocr` checks again, but this also
    // skips the VRAM admission and the server start below, which are the
    // slow parts of getting a page read.
    cancel.check()?;
    let profile = detent.profile();
    let model_id = ocr_model_id(detent);
    let entry = registry
        .find(model_id)
        .ok_or_else(|| format!("{model_id} is not in the registry, so images cannot be read."))?
        .clone();

    // Budgeted against free VRAM with the model's own layer count, and any
    // other server released only if this one will not otherwise fit. See
    // `serving::admission`.
    let plan = crate::serving::admission::admit(&servers, &entry, registry.models_dir())
        .await
        .map_err(|error| error.to_string())?
        .plan;
    let endpoint = servers
        .endpoint_for(&entry, registry.models_dir(), &plan)
        .await
        .map_err(|e| e.to_string())?;

    crate::serving::probe::check_loopback(&endpoint.base_url).map_err(|outcome| {
        format!(
            "refusing to send an attachment off-machine: {}",
            outcome.explain(&endpoint.base_url)
        )
    })?;

    // arjun-egress-ok: loopback only, enforced by the check above.
    let client = ocr_http_client().clone();
    let text = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let sink = text.clone();
    let emitter = app.clone();
    let file = name.to_string();
    let started = std::time::Instant::now();
    let summary = stream_ocr(
        &client,
        &endpoint.base_url,
        &endpoint.served_model_id,
        image,
        &profile,
        cancel,
        move |event| match event {
            OcrEvent::Text { index, delta } => {
                if let Ok(mut t) = sink.lock() {
                    t.push_str(&delta);
                }
                let _ = emitter.emit(
                    "attachment:ocr",
                    AttachmentOcrEvent::Text {
                        name: file.clone(),
                        page,
                        index,
                        delta,
                    },
                );
            }
            OcrEvent::Region { index, label, .. } => {
                let _ = emitter.emit(
                    "attachment:ocr",
                    AttachmentOcrEvent::Region {
                        name: file.clone(),
                        page,
                        index,
                        label,
                    },
                );
            }
        },
    )
    .await
    .map_err(|e| format!("reading the page failed: {e:#}"))?;
    let raw = text.lock().map(|t| t.clone()).unwrap_or_default();
    // Measured again over the assembled text rather than reusing the stream's
    // offset. The two count different things: the guard inside `stream_ocr`
    // watches the model's raw output, headers and boxes included, while this
    // is what the span parser kept — so an offset from one does not address
    // the other, and cutting at it would slice mid-word.
    let looped_from = degenerate_tail_start(&raw);
    let read = match looped_from {
        Some(at) => raw.chars().take(at).collect::<String>(),
        None => raw,
    };
    // Either signal is enough to call the read incomplete. The stream guard
    // can fire on repetition that lives entirely in the region headers, which
    // the parser strips before this text is assembled; the tail scan can fire
    // on a page that ran to the decode cap before the guard was reached.
    let looped = looped_from.is_some() || summary.looped_at.is_some();
    if looped {
        log::warn!(
            "[OCR] {name} page {page} degenerated into repetition and was cut short; \
             {} characters kept",
            read.chars().count()
        );
    }
    // `hit_decode_cap` used to be dropped on the floor here, and dropping it
    // is how a page that filled its entire token budget with one repeated
    // character reached the answer looking like an ordinary read. A page that
    // stopped because it ran out of budget did not finish, and the reader has
    // to be told which of the two happened.
    let _ = app.emit(
        "attachment:ocr",
        AttachmentOcrEvent::Page {
            name: name.to_string(),
            page,
            pages,
            model_id: model_id.to_string(),
            detent,
            characters: read.chars().count(),
            elapsed_ms: started.elapsed().as_millis() as u64,
            hit_decode_cap: summary.hit_decode_cap,
            looped,
        },
    );

    // A page stopped part-way through is not a page that was read.
    //
    // `stream_ocr` returns whatever text arrived alongside `cancelled: true`,
    // and this function used to take the text and drop the flag — so pressing
    // Stop while page 7 was decoding wrote page 7's half-transcription into
    // `by_page` as an ordinary page, then into `DocumentStore` as the durable
    // text for that page. Pages 8 onward were correctly listed as unread; page
    // 7 quietly lied, and (because the store is first-write-wins) went on
    // lying to every later turn.
    //
    // Reported as an error so the caller's `unread` list picks it up, which is
    // the same treatment every other unreadable page gets.
    if summary.cancelled {
        return Err(
            "the turn was stopped while this page was being read, so what arrived is only part              of it"
                .to_string(),
        );
    }

    Ok(read)
}

/// The HTTP client used to talk to the local OCR server.
///
/// Two things it fixes, both of which were "no bound at all":
///
/// - **Timeouts.** `Client::new()` has none. The two waits in `stream_ocr` are
///   cancellable, but only by an explicit `CancelToken` — so a server that
///   accepted the request, sent headers and then stalled forever was caught by
///   nothing except the person pressing Stop. A page is minutes of honest work,
///   so the read timeout is generous; what it rules out is *never*.
/// - **Reuse.** It was constructed per page, so a forty-page scan built forty
///   clients and forty connection pools to the same loopback address.
///
/// `read_timeout` rather than `timeout`: the whole-request bound would cut a
/// page that is streaming correctly but slowly, which is the normal case on a
/// small GPU. This bounds the gap *between* bytes instead.
fn ocr_http_client() -> &'static reqwest::Client {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .read_timeout(std::time::Duration::from_secs(120))
            .build()
            // A client that will not build is a programming error in the two
            // constants above, not a condition a caller can do anything about.
            .expect("the OCR http client builds from constants")
    })
}

/// Runs the local one-shot extractor over a stored document.
///
/// A separate short-lived process rather than the long-lived document
/// sidecar: this needs no state between calls, and a crash in a PDF parser
/// then takes nothing else down with it.
/// How many pages of a scanned document are rendered for reading.
///
/// The extractor has always had a `--max-pages` flag and a default of 12, and
/// nothing here ever passed it — so every scan was silently cut at twelve pages
/// while this module's own comments described reading "a forty-page drawing
/// set". Pages past the cap land in the unread list, which is honest at the
/// prompt, but they never reach `page_text` either, so `document.read_pages(31)`
/// answers "no such page" for the life of the document.
///
/// Sixty rather than twelve. A page is minutes of GPU time, so this is still a
/// bound and not a licence — but it is the number at which a drawing set is
/// genuinely unusual rather than the number at which an ordinary inspection
/// report is cut in half.
const MAX_RENDERED_PAGES: u32 = 60;

fn run_extractor(path: &std::path::Path, out_dir: &std::path::Path) -> Result<Extracted, String> {
    let script = extractor_script()?;
    let python = crate::deployment::dependency("python");
    let output = crate::system_analyzer::process_utils::create_hidden_command(
        crate::deployment::program("python"),
    )
        .arg(&script)
        .arg(path)
        .arg(out_dir)
        .arg("--max-pages")
        .arg(MAX_RENDERED_PAGES.to_string())
        .output()
        // The spawn is the probe. A failure here is almost always a machine
        // with no interpreter rather than a broken extractor, so the remedy
        // for the interpreter is what the user needs to read.
        .map_err(|e| {
            format!(
                "the document extractor could not be started: {e}. {}",
                python.remedy
            )
        })?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with('{'))
        .ok_or_else(|| {
            format!(
                "the document extractor returned nothing usable: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )
        })?;
    let parsed: Extracted = serde_json::from_str(line)
        .map_err(|e| format!("the document extractor returned unreadable output: {e}"))?;
    if let Some(error) = parsed.error {
        return Err(error);
    }
    Ok(parsed)
}

/// Decodes one attachment, stores it content-addressed, and turns it into
/// text a language model can reason about.
///
/// The routing is the design. A PDF that already carries a text layer is read
/// by parsing it, not by rendering pages and asking a vision model to read
/// back text the file was holding all along. Only images and genuine scans
/// reach the OCR model. The composer cannot influence this — the frontend can
/// only hand over bytes.
pub async fn read_attachment(
    app: &AppHandle,
    registry: &ModelRegistry,
    servers: &ModelServers,
    attachment: &ChatAttachment,
    detent: OcrDetent,
    tag: &StageTag,
    cancel: &crate::agent_runtime::cancellation::CancelToken,
) -> Result<AttachmentRead, String> {
    // Before the file is even decoded. Reading a 24 MB base64 blob and hashing
    // it is not free, and a turn stopped before this one started should spend
    // none of it.
    cancel.check()?;
    let bytes = base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        &attachment.data_base64,
    )
    .map_err(|e| format!("{} could not be decoded: {e}", attachment.name))?;
    let kind = validate_attachment(&attachment.name, &attachment.mime, bytes.len())?;

    let sha256 = {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(&bytes);
        format!("{:x}", h.finalize())
    };

    let base = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("no app data directory: {e}"))?
        .join("documents")
        .join("attachments")
        .join(&sha256);
    std::fs::create_dir_all(&base)
        .map_err(|e| format!("could not store {}: {e}", attachment.name))?;

    progress(app, tag, &attachment.name, "reading", None, None, None);

    // What the answer will say about how this file was read. Filled in by
    // whichever branch actually runs, so it reports the path taken rather
    // than the path expected.
    let read_kind: String;
    let mut read_pages: u32 = 1;
    let mut ocr_model: Option<String> = None;
    // The pages, kept apart from the blob the prompt is composed from.
    //
    // Every branch below already has them — a PDF as `by_page`, everything else
    // as a single blob that is its own page 1. They were being merged into one
    // string and the numbers thrown away, which is why nothing could ask for
    // page 31 afterwards. See `AttachmentRead::page_text`.
    let mut page_text: std::collections::BTreeMap<u32, String> =
        std::collections::BTreeMap::new();
    let mut reader_truncated = false;

    let text = match kind {
        AttachmentKind::Image(ext) => {
            let stored = base.join(format!("page-1.{ext}"));
            if !stored.exists() {
                std::fs::write(&stored, &bytes)
                    .map_err(|e| format!("could not store {}: {e}", attachment.name))?;
            }
            progress(
                app,
                tag,
                &attachment.name,
                "understanding",
                Some(1),
                Some(1),
                Some("image".into()),
            );
            read_kind = "image".into();
            ocr_model = Some(ocr_model_id(detent).to_string());
            let read = ocr_one_image(
                app,
                registry,
                servers,
                &stored,
                detent,
                &attachment.name,
                1,
                1,
                cancel,
            )
            .await?;
            // An image is one page, and it is page 1. Recorded even when the
            // model read nothing: the store drops empty pages itself, in one
            // place, rather than each branch here deciding separately.
            page_text.insert(1, read.trim().to_string());
            read
        }
        AttachmentKind::Document(ext) => {
            let stored = base.join(format!("source.{ext}"));
            if !stored.exists() {
                std::fs::write(&stored, &bytes)
                    .map_err(|e| format!("could not store {}: {e}", attachment.name))?;
            }
            progress(app, tag, &attachment.name, "preparing", None, None, None);
            // Off the async executor.
            //
            // `run_extractor` is `std::process::Command::output()` — fully
            // synchronous and unbounded — and it was being called straight from
            // this `async fn`. For the whole of a pypdf parse plus a PyMuPDF
            // rasterisation of every page, tens of seconds on a large scan, it
            // held a Tokio worker thread and stalled unrelated Tauri commands
            // that happened to be scheduled on it.
            let extracted = {
                let stored = stored.clone();
                let base = base.clone();
                tokio::task::spawn_blocking(move || run_extractor(&stored, &base))
                    .await
                    // The task panicked rather than returning an error, which
                    // is a bug in the extractor wrapper and not something the
                    // person attaching a file can act on — but they still need
                    // to be told the document was not read.
                    .map_err(|e| format!("the document extractor did not finish: {e}"))??
            };
            let pages = extracted.pages.max(1);

            read_kind = extracted.kind.clone();
            read_pages = pages;

            if extracted.page_detail.is_empty() {
                // Not a PDF: a spreadsheet, a Word file, a deck, plain text.
                // One blob, no pages to put in order, and no model was used.
                progress(
                    app,
                    tag,
                    &attachment.name,
                    "extracting",
                    None,
                    Some(pages),
                    Some(extracted.kind.clone()),
                );
                let mut blob = extracted.text;
                // One blob, so one page — and the page count is corrected to
                // say so.
                //
                // `extracted.pages` is the *reader's* count, and for two
                // formats it is not a count of what was stored: `extract_xlsx`
                // reports `len(sheets)` and `extract_pptx` reports
                // `len(slides)`, while both hand back a single blob that lands
                // here as page 1. `doc_pipeline::measure` then computed
                // `pages_failed = [2..=n]`, so a perfectly-read five-sheet
                // workbook was described to the model as "1 of 5 pages read; no
                // text from pages 2-5" — a false alarm on every multi-sheet
                // workbook and every deck, permanently.
                //
                // The reader's own count is not lost: it is still what the
                // progress events reported while the file was being read. What
                // is corrected is the number attached to the stored text, which
                // is the one everything downstream reasons about.
                read_pages = 1;
                // Stored before the truncation note is appended, so what is
                // kept is the reader's output rather than the reader's output
                // plus a sentence about the reader.
                page_text.insert(1, blob.trim().to_string());
                reader_truncated = extracted.truncated;
                if extracted.truncated {
                    // The spreadsheet and deck readers have always reported
                    // this, and nothing here has ever repeated it: a workbook
                    // cut off at its row cap read as a workbook that ended
                    // there. Silence about what was dropped is the same failure
                    // the page merge below exists to fix.
                    blob.push_str(
                        "\n\n(this file was longer than the reader's limit, so it was cut off \
                         here — do not describe it as complete)",
                    );
                }
                blob
            } else {
                // A PDF, handled page by page.
                //
                // The merge happens here rather than in the extractor because
                // only this side has the OCR output. Keyed on the real page
                // number throughout, so a document whose second page is a scan
                // reports it as page 2 rather than as page 1 of the rendered
                // subset.
                let (mut by_page, mut unread) = seed_from_detail(&extracted.page_detail);

                let to_read: Vec<&PageDetail> = extracted
                    .page_detail
                    .iter()
                    .filter(|detail| detail.source == "ocr")
                    .collect();

                if to_read.is_empty() {
                    // Every page had a text layer. No model was needed.
                    progress(
                        app,
                        tag,
                        &attachment.name,
                        "extracting",
                        None,
                        Some(pages),
                        Some(extracted.kind.clone()),
                    );
                } else {
                    ocr_model = Some(ocr_model_id(detent).to_string());
                    // The counter counts the pages actually going to the model,
                    // which is the work being waited on. The page *labels*
                    // below carry the document's own numbering.
                    let queued = to_read.len() as u32;
                    for (index, detail) in to_read.iter().enumerate() {
                        // No further page after a Stop.
                        //
                        // A forty-page drawing set is forty of these, each
                        // minutes long, so this is the check that decides
                        // whether Stop means "in a moment" or "when the whole
                        // document has been read". The pages already done are
                        // kept and the rest are listed as unread, which is what
                        // they are — not blank, which would be a claim about
                        // the document nobody verified.
                        if cancel.is_cancelled() {
                            for remaining in to_read.iter().skip(index) {
                                unread.push((
                                    remaining.page,
                                    "the turn was stopped before this page was read".to_string(),
                                ));
                            }
                            break;
                        }
                        let Some(image) = detail.image.as_ref() else {
                            unread.push((
                                detail.page,
                                "it was marked for reading but no image was produced".to_string(),
                            ));
                            continue;
                        };
                        progress(
                            app,
                            tag,
                            &attachment.name,
                            "understanding",
                            Some(index as u32 + 1),
                            Some(queued),
                            Some(extracted.kind.clone()),
                        );
                        // One page's failure costs that page, and nothing else.
                        //
                        // This was `ocr_one_image(...).await?`, and the `?` left
                        // `read_attachment` entirely — which left `drive_run`
                        // entirely, because the call site there propagates too.
                        // So a transport blip, a 500, or `admission` reporting
                        // `WontFit` on page 20 of 40 discarded pages 1-19 —
                        // minutes of GPU time — wrote nothing to the document
                        // store, and gave the person a failed turn instead of a
                        // partial answer.
                        //
                        // The cancellation branch a few lines up already had
                        // this right: keep what was read, name the rest as
                        // unread. The error path simply had no equivalent.
                        let read = match ocr_one_image(
                            app,
                            registry,
                            servers,
                            image,
                            detent,
                            &attachment.name,
                            detail.page,
                            pages,
                            cancel,
                        )
                        .await
                        {
                            Ok(read) => read,
                            Err(reason) => {
                                log::warn!(
                                    "[ocr] {}: page {} could not be read, and the rest of the                                      document carries on: {reason}",
                                    attachment.name,
                                    detail.page
                                );
                                unread.push((detail.page, reason));
                                continue;
                            }
                        };
                        match settle_ocr_page(&read, &detail.layer_text) {
                            Ok(text) => {
                                by_page.insert(detail.page, text);
                            }
                            Err(reason) => unread.push((detail.page, reason)),
                        }
                    }
                }

                // Taken before `assemble_pdf_text` consumes the map. The
                // assembled blob keeps the page numbers only as headings inside
                // a string; this keeps them as numbers.
                reader_truncated = extracted.truncated;
                page_text.extend(by_page.iter().map(|(page, text)| (*page, text.clone())));
                assemble_pdf_text(by_page, unread, pages)
            }
        }
    };

    progress(app, tag, &attachment.name, "done", None, None, None);
    Ok(AttachmentRead {
        name: attachment.name.clone(),
        sha256,
        text: text.trim().to_string(),
        kind: read_kind,
        pages: read_pages,
        ocr_detent: ocr_model.as_ref().map(|_| detent),
        ocr_model_id: ocr_model,
        page_text,
        truncated: reader_truncated,
    })
}

/// Which weight file a stop loads.
///
/// One place, so the routing explanation the person reads and the server the
/// request goes to cannot name different models.
pub const fn ocr_model_id(detent: OcrDetent) -> &'static str {
    match detent.profile().tier {
        OcrTier::High => "unlimited-ocr-q6-k",
        OcrTier::Fast => "unlimited-ocr-q4-k-m",
    }
}

/// The stop signal for the page the scan view is reading.
///
/// Holds a [`CancelToken`] rather than a bare flag, so the wait for the
/// first byte of a page is interruptible rather than merely pollable — the
/// same reason the chat path carries one. A boolean can only be *checked*,
/// and the read spends most of its time blocked on a socket where nothing
/// checks anything.
///
/// Replaced rather than reset at the start of each scan: a token that has
/// been cancelled stays cancelled, which is the honest shape for a signal,
/// so a fresh page gets a fresh one.
#[derive(Default)]
pub struct ScanCancel(pub Mutex<CancelToken>);

/// What the UI receives on `ocr:span`.
///
/// Carries the model's own numbers alongside the mapped ones. Keeping the raw
/// box means a future change to the emitted format shows up as a mismatch
/// that can be asserted on, instead of an overlay that quietly drifts.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase", tag = "event")]
enum SpanPayload {
    /// The inner `rename_all` is load-bearing and easy to lose: the container
    /// attribute renames *variants*, not their fields, so without this the UI
    /// receives `page_box` while `ocr.service.ts` reads `pageBox` — and the
    /// overlay silently never draws.
    #[serde(rename_all = "camelCase")]
    Region {
        index: usize,
        label: String,
        bbox: Option<RawBox>,
        page_box: Option<PageBox>,
    },
    #[serde(rename_all = "camelCase")]
    Text { index: Option<usize>, delta: String },
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct StatusPayload {
    page: u32,
    state: &'static str,
    tokens: u32,
    elapsed_ms: u64,
    /// Only ever a measured figure. Absent rather than estimated.
    tokens_per_second: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ErrorPayload {
    page: u32,
    reason: String,
}

/// Where a rasterised page lives.
///
/// Rendering a PDF to page images happens upstream of this command; it is not
/// done here. A missing file is reported as exactly that rather than being
/// silently treated as an empty page.
fn page_image_path(app: &AppHandle, sha256: &str, page: u32) -> Result<PathBuf, String> {
    let base = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("no app data directory: {e}"))?;
    page_image_in(&base.join("documents"), sha256, page)
}

/// Where a page image lives, or why there is none.
///
/// Split from the Tauri command so the search order can be tested without an
/// app handle, and because the order is the whole content of this function.
///
/// Two locations, likeliest first. `attachments/<sha>/` is where
/// [`read_attachment`] has the extractor write the pages it rasterised for the
/// OCR model — every page of a scan, and no page of a PDF that arrived with its
/// own text layer. `pages/<sha>/` is where this used to look, and *only* where
/// it looked: nothing in the product has ever written that directory, so every
/// call failed and the scan view could open no document at all. From the
/// outside that is indistinguishable from the OCR model being broken, which is
/// how it was reported. It stays as a fallback rather than being deleted — a
/// machine carrying one of these directories from an earlier build keeps
/// working.
///
/// A page with no image is not a failure of the reader, and the message no
/// longer implies one. It is what a page carrying usable text looks like: the
/// extractor parsed it and never rasterised it, so there is nothing for a
/// vision model to look at. The old wording — "rasterise the document before
/// reading it" — named no command anybody can run.
fn page_image_in(documents_dir: &Path, sha256: &str, page: u32) -> Result<PathBuf, String> {
    let name = format!("page-{page}.png");
    let attachment = documents_dir.join("attachments").join(sha256).join(&name);
    if attachment.exists() {
        return Ok(attachment);
    }
    let legacy = documents_dir.join("pages").join(sha256).join(&name);
    if legacy.exists() {
        return Ok(legacy);
    }
    Err(format!(
        "page {page} of {sha256} has no rendered image, so there is nothing for the OCR model to \
         look at. A page is rasterised only when it carries no usable text layer — a PDF \
         that came with its own text was parsed instead, and that text is already in the \
         document. Looked in {} and {}.",
        attachment.display(),
        legacy.display()
    ))
}

/// Reads one page, streaming regions to the UI as the model finds them.
#[tauri::command]
pub async fn scan_page(
    app: AppHandle,
    registry: State<'_, Arc<ModelRegistry>>,
    servers: State<'_, Arc<ModelServers>>,
    cancel: State<'_, ScanCancel>,
    document_sha256: String,
    page: u32,
    detent: OcrDetent,
) -> Result<(), String> {
    // A fresh token here rather than at the end of the previous run: a run
    // that failed or was dropped must not leave the next one pre-cancelled.
    let token = CancelToken::never();
    match cancel.0.lock() {
        Ok(mut held) => *held = token.clone(),
        // A poisoned lock costs this scan its Stop button and nothing else.
        // Refusing to scan at all because an unrelated thread panicked
        // while holding a token would be the worse trade.
        Err(_) => log::error!(
            "[ocr] the scan cancellation slot is poisoned; this page cannot be stopped"
        ),
    }

    let profile = detent.profile();
    let model_id = match profile.tier {
        OcrTier::High => "unlimited-ocr-q6-k",
        OcrTier::Fast => "unlimited-ocr-q4-k-m",
    };

    let image = page_image_path(&app, &document_sha256, page)?;
    let (page_w, page_h) = png_dimensions(&image)?;
    // Under the measured convention the model normalises against its own
    // input, so the page's own size is both the source and the target space.
    let geometry = PageGeometry {
        page_width: page_w,
        page_height: page_h,
        input_width: page_w,
        input_height: page_h,
    };

    let entry = registry
        .find(model_id)
        .ok_or_else(|| {
            format!("{model_id} is not in the registry. Merge config/ocr-model-registry.json.")
        })?
        .clone();

    // Budgeted against free VRAM with the model's own layer count, and any
    // other server released only if this one will not otherwise fit. See
    // `serving::admission`.
    let plan = crate::serving::admission::admit(&servers, &entry, registry.models_dir())
        .await
        .map_err(|error| error.to_string())?
        .plan;

    let endpoint = servers
        .endpoint_for(&entry, registry.models_dir(), &plan)
        .await
        .map_err(|e| e.to_string())?;

    let _ = app.emit(
        "ocr:status",
        StatusPayload {
            page,
            state: "reading",
            tokens: 0,
            elapsed_ms: 0,
            tokens_per_second: None,
        },
    );

    // Enforced, not assumed. A managed endpoint is always loopback, but an
    // operator can point a registry entry at an external server, and this
    // command must not become the one place a document leaves the machine.
    // The annotation below is only honest because of this check.
    crate::serving::probe::check_loopback(&endpoint.base_url).map_err(|outcome| {
        format!(
            "refusing to send a document off-machine: {}",
            outcome.explain(&endpoint.base_url)
        )
    })?;

    // arjun-egress-ok: loopback only. The check above rejects any
    // non-loopback base URL, so the only host this client can address is the
    // local llama.cpp server ARJUN itself started. Sovereignty: no remote.
    let client = ocr_http_client().clone();
    let emitter = app.clone();
    let result = stream_ocr(
        &client,
        &endpoint.base_url,
        &endpoint.served_model_id,
        &image,
        &profile,
        &token,
        move |event| {
            let payload = match event {
                OcrEvent::Region { index, label, bbox } => SpanPayload::Region {
                    index,
                    label,
                    bbox,
                    // Mapped only when the convention has actually been
                    // measured; otherwise the UI draws no overlay rather
                    // than a plausible-looking wrong one.
                    page_box: bbox.and_then(|raw| {
                        CALIBRATED_COORD_SPACE.map(|space| to_page(raw, space, geometry))
                    }),
                },
                OcrEvent::Text { index, delta } => SpanPayload::Text { index, delta },
            };
            let _ = emitter.emit("ocr:span", payload);
        },
    )
    .await;

    match result {
        Ok(summary) => {
            let seconds = summary.elapsed_ms as f64 / 1000.0;
            let _ = app.emit(
                "ocr:status",
                StatusPayload {
                    page,
                    state: if summary.cancelled { "failed" } else { "done" },
                    tokens: summary.tokens,
                    elapsed_ms: summary.elapsed_ms,
                    // Measured, or absent. Never a plausible constant.
                    tokens_per_second: if seconds > 0.0 && summary.tokens > 0 {
                        Some(summary.tokens as f64 / seconds)
                    } else {
                        None
                    },
                },
            );
            if let Some(at) = summary.looped_at {
                // Said plainly, because the alternative is a page that looks
                // read. The guard stopped the model here; everything after
                // this point would have been repetition.
                let _ = app.emit(
                    "ocr:error",
                    ErrorPayload {
                        page,
                        reason: format!(
                            "The model began repeating itself after {at} characters, so \
                             the read was stopped there. What is shown above is the part \
                             of the page that was read; the rest was not. A sharper scan \
                             of this page, or a stop further right, usually fixes it."
                        ),
                    },
                );
            } else if summary.hit_decode_cap {
                // The page stopped because it ran out of budget. On the Fast
                // tier that is the signature of the loop the DRY substitute
                // is supposed to prevent, so it is surfaced, not hidden.
                let _ = app.emit(
                    "ocr:error",
                    ErrorPayload {
                        page,
                        reason: "Generation stopped at the decode limit rather than \
                                 finishing. On the Fast tier this usually means the \
                                 repetition guard did not hold; try a higher stop."
                            .to_string(),
                    },
                );
            }
            Ok(())
        }
        Err(error) => {
            let reason = format!("{error:#}");
            let _ = app.emit(
                "ocr:status",
                StatusPayload {
                    page,
                    state: "failed",
                    tokens: 0,
                    elapsed_ms: 0,
                    tokens_per_second: None,
                },
            );
            let _ = app.emit(
                "ocr:error",
                ErrorPayload {
                    page,
                    reason: reason.clone(),
                },
            );
            Err(reason)
        }
    }
}

/// Stops the page currently being read. Safe to call when nothing is running.
#[tauri::command]
pub fn cancel_scan(cancel: State<'_, ScanCancel>) {
    if let Ok(held) = cancel.0.lock() {
        held.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scan view could open nothing at all.
    ///
    /// `page_image_in` looked only in `documents/pages/<sha>/`, and no code
    /// path in the product writes that directory — the extractor puts the
    /// pages it rasterises in the attachment store. Every scan therefore
    /// failed with "has not been rendered to an image yet", which reads as the
    /// OCR model being broken rather than as the image never having been put
    /// where the reader looked.
    #[test]
    fn a_rasterised_attachment_page_is_found_where_the_extractor_wrote_it() {
        let dir = tempfile::tempdir().unwrap();
        let documents = dir.path().join("documents");
        let sha = "c8cf6b8d";
        let pages = documents.join("attachments").join(sha);
        std::fs::create_dir_all(&pages).unwrap();
        std::fs::write(pages.join("page-2.png"), b"a rendered page").unwrap();

        assert_eq!(
            page_image_in(&documents, sha, 2).unwrap(),
            pages.join("page-2.png")
        );
    }

    /// A page nobody rasterised is not a broken reader.
    ///
    /// A PDF carrying its own text layer is parsed, never rendered, so it has
    /// no page image and never will have one. The refusal has to say that
    /// rather than instruct the reader to "rasterise the document", which
    /// names no command anybody can run.
    #[test]
    fn a_page_with_a_text_layer_is_refused_with_the_reason_it_has_no_image() {
        let dir = tempfile::tempdir().unwrap();
        let reason = page_image_in(&dir.path().join("documents"), "deadbeef", 1).unwrap_err();

        assert!(
            reason.contains("no usable text layer"),
            "the refusal has to explain why the page was never rendered: {reason}"
        );
        assert!(
            !reason.contains("Rasterise the document"),
            "and must not name an action that does not exist: {reason}"
        );
    }

    /// A photograph has no text layer, so a model has to look at it. This is
    /// the case the composer's hint exists for.
    #[test]
    fn an_image_is_planned_for_the_ocr_model() {
        let plan = plan_attachment("scan.png", "image/png");
        assert_eq!(plan.route, "image");
        assert!(plan.needs_ocr);
        assert!(plan.refusal.is_none());
        assert!(
            plan.explanation.contains("document-OCR model"),
            "the hint has to name what will read it: {}",
            plan.explanation
        );
    }

    /// The whole point of routing by type: a spreadsheet already carries its
    /// text, and claiming an OCR model will read it would be a claim about
    /// work that never happens.
    #[test]
    fn a_spreadsheet_is_planned_without_any_model() {
        let plan = plan_attachment("readings.xlsx", "");
        assert_eq!(plan.route, "document");
        assert!(!plan.needs_ocr);
        assert!(
            plan.explanation.contains("no model is needed"),
            "it must say no model is needed: {}",
            plan.explanation
        );
    }

    /// A PDF is the honest "maybe": whether it is a scan is only known once
    /// the extractor opens it, and the wording says so rather than promising
    /// one path.
    #[test]
    fn a_pdf_is_planned_as_possibly_needing_ocr() {
        let plan = plan_attachment("drawing.pdf", "application/pdf");
        assert_eq!(plan.route, "document");
        assert!(plan.needs_ocr);
        assert!(plan.explanation.contains("text layer"));
    }

    /// The plan and the run must refuse the same files. A plan that accepts
    /// what the run rejects is how a person gets a hint and then an error.
    #[test]
    fn an_unreadable_file_is_refused_with_the_same_sentence_the_run_uses() {
        let plan = plan_attachment("weird.bin", "application/octet-stream");
        assert_eq!(plan.route, "rejected");
        assert!(!plan.needs_ocr);
        let refusal = plan.refusal.expect("a rejected file carries its reason");
        let from_run = validate_attachment("weird.bin", "application/octet-stream", 1)
            .expect_err("the run rejects it too");
        assert_eq!(refusal, from_run);
    }

    /// The slider stop and the weight file are one decision. If these ever
    /// disagree, the routing explanation names a model the request did not
    /// go to.
    #[test]
    fn each_stop_names_the_weight_file_its_profile_asks_for() {
        assert_eq!(ocr_model_id(OcrDetent::Fastest), "unlimited-ocr-q4-k-m");
        assert_eq!(ocr_model_id(OcrDetent::Fast), "unlimited-ocr-q4-k-m");
        assert_eq!(ocr_model_id(OcrDetent::Detailed), "unlimited-ocr-q6-k");
        assert_eq!(ocr_model_id(OcrDetent::Maximum), "unlimited-ocr-q6-k");
    }

    #[test]
    fn the_measured_coordinate_space_maps_a_known_box_onto_the_page() {
        // Real numbers from the calibration run: the page's bottom-right
        // marker has ink at y=1299 of 1400 and the model reported y=923.
        // If the convention were ever mis-set to InputPixels this lands at
        // 923 instead of ~1294 and the assertion fails.
        let space = CALIBRATED_COORD_SPACE.expect("calibrated by the Phase 0 gate");
        assert_eq!(space, CoordSpace::Normalised);
        let geometry = crate::ai_engine::ocr_profile::PageGeometry {
            page_width: 1000,
            page_height: 1400,
            input_width: 1000,
            input_height: 1400,
        };
        let mapped = to_page(
            RawBox {
                x1: 615,
                y1: 923,
                x2: 866,
                y2: 956,
            },
            space,
            geometry,
        );
        assert!(mapped.in_bounds, "the marker must land on the page");
        assert!(
            (mapped.y1 - 1294).abs() <= 12,
            "expected ~1294 for the footer marker, got {}",
            mapped.y1
        );
        assert!((mapped.x1 - 616).abs() <= 12, "got x1={}", mapped.x1);
    }

    #[test]
    fn a_span_payload_matches_the_typescript_contract() {
        let region = SpanPayload::Region {
            index: 0,
            label: "title".into(),
            bbox: None,
            page_box: None,
        };
        let json = serde_json::to_string(&region).expect("serialises");
        assert!(json.contains(r#""event":"region""#), "got {json}");
        assert!(json.contains(r#""pageBox""#), "got {json}");
    }

    #[test]
    fn cancelling_is_sticky_until_the_next_scan_clears_it() {
        let slot = ScanCancel::default();
        assert!(!slot.0.lock().unwrap().is_cancelled());
        slot.0.lock().unwrap().cancel();
        assert!(slot.0.lock().unwrap().is_cancelled());

        // A fresh scan installs a fresh token, which is what un-cancels the
        // view — the token itself never goes back to uncancelled.
        *slot.0.lock().unwrap() = CancelToken::never();
        assert!(!slot.0.lock().unwrap().is_cancelled());
    }

    /// A document stopped part-way through says which pages nobody read.
    ///
    /// The page loop breaks on cancellation and pushes every remaining page
    /// onto the unread list, which is what this assembles. The distinction is
    /// load-bearing: a page nobody looked at is not a blank page, and a model
    /// shown a document that simply stops answers as though it had read the
    /// whole thing.
    #[test]
    fn a_scan_stopped_midway_names_the_pages_it_did_not_read() {
        let mut by_page = std::collections::BTreeMap::new();
        by_page.insert(1, "page one".to_string());
        by_page.insert(2, "page two".to_string());
        // Pages 3-5 were never started, as the loop leaves them on a Stop.
        let unread = vec![
            (3, "the turn was stopped before this page was read".to_string()),
            (4, "the turn was stopped before this page was read".to_string()),
            (5, "the turn was stopped before this page was read".to_string()),
        ];

        let merged = assemble_pdf_text(by_page, unread, 5);

        assert!(merged.contains("page one"), "what was read is kept: {merged}");
        assert!(merged.contains("page two"));
        // And what was not is named, rather than silently absent.
        assert!(merged.contains("3"), "{merged}");
        assert!(merged.contains("stopped"), "{merged}");
        assert!(
            !merged.contains("page three"),
            "a page nobody read must not appear as content"
        );
    }

    #[test]
    fn an_attachment_of_an_unreadable_type_is_refused_by_name() {
        let err = validate_attachment("clip.mp4", "video/mp4", 10).unwrap_err();
        assert!(
            err.contains("clip.mp4"),
            "the person must see which file: {err}"
        );
        assert!(err.contains("PDF"), "and what would work: {err}");
    }

    #[test]
    fn an_empty_attachment_is_refused_rather_than_read_as_a_blank_page() {
        assert!(validate_attachment("scan.png", "image/png", 0)
            .unwrap_err()
            .contains("empty"));
    }

    #[test]
    fn an_oversized_attachment_is_refused_before_it_reaches_the_model() {
        // A refusal the person can read beats a request that dies inside
        // llama-server with no explanation.
        let err =
            validate_attachment("huge.png", "image/png", MAX_ATTACHMENT_BYTES + 1).unwrap_err();
        assert!(err.contains("limit"), "got {err}");
    }

    #[test]
    fn the_readable_image_types_map_to_the_extensions_the_model_accepts() {
        assert_eq!(
            validate_attachment("a.png", "image/png", 9).unwrap(),
            AttachmentKind::Image("png")
        );
        assert_eq!(
            validate_attachment("a.jpg", "image/jpeg", 9).unwrap(),
            AttachmentKind::Image("jpg")
        );
        assert_eq!(
            validate_attachment("a.webp", "image/webp", 9).unwrap(),
            AttachmentKind::Image("webp")
        );
    }

    /// The regression for the PDF refusal: a PDF used to be rejected because
    /// the only accepted types were the three the vision model reads. It must
    /// now route to the extractor, not to OCR.
    /// A board presentation is one of the document kinds this product exists
    /// for, and it was the one Office format the extractor could not open.
    #[test]
    fn a_presentation_is_read_locally_rather_than_refused() {
        assert_eq!(
            validate_attachment(
                "capex-approval.pptx",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                4096
            )
            .unwrap(),
            AttachmentKind::Document("pptx")
        );

        let plan = plan_attachment("capex-approval.pptx", "");
        assert_eq!(plan.route, "document");
        assert!(
            !plan.needs_ocr,
            "a deck carries its own text; sending slides to a vision model would be              slower and lossier than reading the XML"
        );
        assert!(plan.refusal.is_none());
    }

    fn detail(page: u32, source: &str, text: &str, why: &str) -> PageDetail {
        PageDetail {
            page,
            source: source.to_string(),
            text: text.to_string(),
            image: None,
            layer_text: String::new(),
            why: why.to_string(),
        }
    }

    /// The reported case: a digital cover in front of a scanned page.
    ///
    /// The extractor used to weigh the *whole file's* text against one
    /// threshold, so the cover carried the document over the line on its own.
    /// It came back `pdf-text` with no page images and `truncated: false`, and
    /// page two was gone with nothing to say it had ever been there. The
    /// merged text now has to contain both pages, in order, under their own
    /// numbers.
    #[test]
    fn a_digital_cover_does_not_swallow_the_scanned_page_behind_it() {
        let (mut by_page, unread) = seed_from_detail(&[
            detail(1, "text", "Q3 Seal Inspection Report", ""),
            detail(2, "ocr", "", ""),
        ]);
        // What the OCR model gave back for the page that needed it.
        by_page.insert(2, "Measured 8.2 mm against a 9.0 mm minimum".to_string());

        let merged = assemble_pdf_text(by_page, unread, 2);

        assert!(merged.contains("Q3 Seal Inspection Report"), "{merged}");
        assert!(merged.contains("Measured 8.2 mm"), "the scanned page was lost: {merged}");
        assert!(
            merged.find("page 1 of 2").unwrap() < merged.find("page 2 of 2").unwrap(),
            "pages came back out of order: {merged}"
        );
    }

    /// Page numbers are the document's, not the rendered subset's.
    ///
    /// When only some pages go to OCR, indexing by position in the rendered
    /// list labelled page 5 of a scan as "page 1" — a citation that points at
    /// the wrong page is worse than one that is missing.
    #[test]
    fn a_page_carries_the_number_it_has_in_the_document() {
        let (mut by_page, unread) = seed_from_detail(&[
            detail(1, "text", "Cover", ""),
            detail(2, "text", "Contents", ""),
            detail(3, "ocr", "", ""),
        ]);
        by_page.insert(3, "The drawing".to_string());

        let merged = assemble_pdf_text(by_page, unread, 3);
        assert!(merged.contains("--- page 3 of 3 ---\nThe drawing"), "{merged}");
        assert!(!merged.contains("page 1 of 3 ---\nThe drawing"), "{merged}");
    }

    /// Every page nothing was made of is named.
    #[test]
    fn unread_pages_are_named_rather_than_dropped() {
        let (by_page, unread) = seed_from_detail(&[
            detail(1, "text", "Cover", ""),
            detail(2, "unread", "", "the limit of 12 rendered pages was reached"),
            detail(3, "unread", "", "the page could not be rendered: broken xref"),
        ]);

        let merged = assemble_pdf_text(by_page, unread, 3);
        assert!(merged.contains("2 of the 3 pages could not be read"), "{merged}");
        assert!(merged.contains("page 2 (the limit of 12 rendered pages"), "{merged}");
        assert!(merged.contains("page 3 (the page could not be rendered"), "{merged}");
        assert!(
            merged.contains("do not describe what is on them"),
            "a hole the model is not warned about is one it will fill in: {merged}"
        );
    }

    /// A page whose text layer is empty is a page nothing was made of.
    ///
    /// It used to vanish in the join that dropped empty parts, which is the
    /// same silence one page down.
    #[test]
    fn a_blank_page_is_reported_not_skipped() {
        let (by_page, unread) =
            seed_from_detail(&[detail(1, "text", "Cover", ""), detail(2, "text", "   ", "")]);
        assert_eq!(by_page.len(), 1);
        let merged = assemble_pdf_text(by_page, unread, 2);
        assert!(merged.contains("page 2 (it has nothing on it)"), "{merged}");
    }

    /// The reported case, one page down: a scanned page with nothing on it.
    ///
    /// Rendering it and getting nothing back is not the same as it not being
    /// there, and only one of those two is worth an answer written over it.
    #[test]
    fn a_scanned_page_the_model_makes_nothing_of_is_marked_unread() {
        let reason = settle_ocr_page("   ", "").unwrap_err();
        assert!(reason.contains("nothing could be made out"), "{reason}");

        let (by_page, mut unread) = seed_from_detail(&[
            detail(1, "text", "Q3 Seal Inspection Report", ""),
            detail(2, "ocr", "", ""),
        ]);
        unread.push((2, reason));
        let merged = assemble_pdf_text(by_page, unread, 2);

        assert!(merged.contains("Q3 Seal Inspection Report"), "{merged}");
        assert!(
            merged.contains("page 2 (it was read but nothing could be made out on it)"),
            "the empty scan went unmentioned, which is the reported bug: {merged}"
        );
    }

    /// A page the model failed on still reports the little it did carry.
    #[test]
    fn a_page_the_model_failed_on_falls_back_to_its_text_layer() {
        assert_eq!(settle_ocr_page("", "Figure 3 continued").unwrap(), "Figure 3 continued");
        // The model's reading wins when there is one.
        assert_eq!(settle_ocr_page("The full caption", "Figure 3").unwrap(), "The full caption");
    }

    /// A single-page document is not labelled, because there is nothing to
    /// disambiguate and the label would only be noise.
    #[test]
    fn a_one_page_document_gets_no_page_labels() {
        let (by_page, unread) = seed_from_detail(&[detail(1, "text", "All of it", "")]);
        assert_eq!(assemble_pdf_text(by_page, unread, 1), "All of it");
    }

    #[test]
    fn a_pdf_routes_to_the_document_extractor_not_to_ocr() {
        assert_eq!(
            validate_attachment("report.pdf", "application/pdf", 4096).unwrap(),
            AttachmentKind::Document("pdf")
        );
    }

    #[test]
    fn text_native_formats_route_to_the_extractor_so_no_model_reads_them() {
        for (name, mime) in [
            ("notes.txt", "text/plain"),
            ("readme.md", "text/markdown"),
            ("rows.csv", "text/csv"),
            ("config.json", "application/json"),
        ] {
            match validate_attachment(name, mime, 64) {
                Ok(AttachmentKind::Document(_)) => {}
                other => panic!("{name} should be a document, got {other:?}"),
            }
        }
    }

    #[test]
    fn office_formats_route_to_the_extractor() {
        assert_eq!(
            validate_attachment(
                "a.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                64
            )
            .unwrap(),
            AttachmentKind::Document("docx")
        );
        assert_eq!(
            validate_attachment(
                "a.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                64
            )
            .unwrap(),
            AttachmentKind::Document("xlsx")
        );
    }

    #[test]
    fn an_image_still_routes_to_ocr_so_the_verified_path_is_untouched() {
        // Guards the working Unlimited-OCR path against a routing change.
        assert!(matches!(
            validate_attachment("scan.png", "image/png", 4096).unwrap(),
            AttachmentKind::Image(_)
        ));
    }

    #[test]
    fn a_browser_that_reports_no_mime_still_routes_by_extension() {
        // Some picks arrive with an empty `type`; the suffix is the fallback.
        assert_eq!(
            validate_attachment("report.pdf", "", 4096).unwrap(),
            AttachmentKind::Document("pdf")
        );
    }

    #[test]
    fn the_ui_receives_four_stops_in_slider_order() {
        let stops = detent_info();
        assert_eq!(stops.len(), 4);
        assert_eq!(stops[0].detent, OcrDetent::Fastest);
        assert_eq!(stops[3].detent, OcrDetent::Maximum);
    }

    #[test]
    fn every_stop_reports_the_numbers_its_profile_will_actually_use() {
        // The whole reason this command exists rather than a hardcoded table
        // in the frontend.
        for stop in detent_info() {
            let profile = stop.detent.profile();
            assert_eq!(stop.max_image_tokens, profile.max_image_tokens);
            assert_eq!(stop.max_decode_tokens, profile.max_decode_tokens);
            assert_eq!(stop.tier, profile.tier);
        }
    }

    #[test]
    fn the_payload_is_camel_case_for_the_typescript_side() {
        // `src/services/ocr.service.ts` declares maxImageTokens / tierLabel;
        // a rename here would break the slider silently at runtime.
        let json = serde_json::to_string(&detent_info()[0]).expect("serialises");
        assert!(json.contains("\"maxImageTokens\""), "got {json}");
        assert!(json.contains("\"tierLabel\""), "got {json}");
        assert!(json.contains("\"maxDecodeTokens\""), "got {json}");
        assert!(
            json.contains("\"fastest\""),
            "detent must serialise camelCase: {json}"
        );
    }
}

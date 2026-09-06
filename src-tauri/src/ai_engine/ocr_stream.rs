//! Driving one OCR page as a stream.
//!
//! [`super::vision_bridge`] asks a question and waits for the answer. That is
//! right for the indexing pipeline, which wants a finished description, and
//! wrong for the scan view, which exists to show the model working. So this
//! module runs the same OpenAI-compatible endpoint with `stream: true` and
//! turns the token stream into the region events the UI draws.
//!
//! ## Three layers, deliberately separate
//!
//! 1. [`SseDecoder`] pulls `data:` payloads out of the byte stream.
//! 2. [`delta_text`] digs the content out of one payload.
//! 3. [`super::ocr_spans::SpanParser`] turns that text into regions.
//!
//! Each is a pure function over chunks, so all three are testable without a
//! server — which matters here, because every one of them can be defeated by
//! an unlucky chunk boundary and every one of them fails *silently* when it
//! is. A dropped SSE frame looks exactly like a page with less text on it.

use std::path::Path;

use crate::agent_runtime::cancellation::CancelToken;

use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use futures_util::StreamExt;
use serde_json::{json, Value};

use super::ocr_profile::OcrProfile;
use super::ocr_repetition::RepetitionGuard;
use super::ocr_spans::{OcrEvent, RawBox, SpanParser};

/// One server-sent event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SseFrame {
    Data(String),
    /// The `[DONE]` sentinel. Distinguished from the connection simply
    /// ending, which is how a truncated stream is told from a finished one.
    Done,
}

/// Incremental SSE framing.
///
/// Frames are separated by a blank line, and a frame can be split across any
/// number of reads. Holding the tail is the whole job.
#[derive(Debug, Default)]
pub struct SseDecoder {
    buffer: String,
}

impl SseDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn feed(&mut self, chunk: &str) -> Vec<SseFrame> {
        // Normalised so a server using CRLF cannot slip past the blank-line
        // search below.
        self.buffer.push_str(&chunk.replace("\r\n", "\n"));
        let mut frames = Vec::new();
        while let Some(at) = self.buffer.find("\n\n") {
            let block: String = self.buffer.drain(..at).collect();
            self.buffer.drain(..2);
            if let Some(frame) = parse_block(&block) {
                frames.push(frame);
            }
        }
        frames
    }

    /// Whatever is left when the connection closes. A well-behaved server
    /// ends with a blank line and this returns nothing.
    pub fn finish(&mut self) -> Vec<SseFrame> {
        let block: String = self.buffer.drain(..).collect();
        parse_block(&block).into_iter().collect()
    }
}

fn parse_block(block: &str) -> Option<SseFrame> {
    let mut data = String::new();
    for line in block.lines() {
        // Comments (`: keep-alive`) and other fields are not errors; they are
        // simply not data.
        if let Some(rest) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest.trim_start());
        }
    }
    if data.is_empty() {
        return None;
    }
    if data.trim() == "[DONE]" {
        return Some(SseFrame::Done);
    }
    Some(SseFrame::Data(data))
}

/// The content delta inside one chat-completion chunk.
///
/// Returns `None` for the shapes that legitimately carry no text — the first
/// chunk announcing a role, and the last one carrying only a finish reason.
pub fn delta_text(payload: &Value) -> Option<String> {
    let choice = payload.get("choices")?.as_array()?.first()?;
    let text = choice
        .get("delta")
        .and_then(|d| d.get("content"))
        // Non-streaming servers, and llama.cpp's own completion shape, put it
        // elsewhere. Accepting both costs nothing and saves a silently empty
        // page if the endpoint is not the one we assumed.
        .or_else(|| choice.get("message").and_then(|m| m.get("content")))
        .or_else(|| choice.get("text"))?
        .as_str()?;
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

/// Whether the server said it stopped because it ran out of budget.
fn hit_length_cap(payload: &Value) -> bool {
    payload
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|c| c.first())
        .and_then(|c| c.get("finish_reason"))
        .and_then(|r| r.as_str())
        == Some("length")
}

/// What the run cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamSummary {
    pub tokens: u32,
    pub elapsed_ms: u64,
    /// True when generation stopped at the decode cap rather than finishing.
    ///
    /// Combined with a repeated n-gram this is the signature of the loop the
    /// missing no-repeat-ngram sampler causes, which is why it is reported
    /// rather than folded into a duration.
    pub hit_decode_cap: bool,
    pub cancelled: bool,
    /// Where the read degenerated into repetition, in characters of decoded
    /// text, when it did.
    ///
    /// `Some` means the stream was **aborted deliberately** and everything
    /// beyond this offset is the model repeating itself rather than the page.
    /// The caller cuts there and says so. See
    /// [`crate::ai_engine::ocr_repetition`] for why the sampler cannot
    /// prevent this and why the decision is taken client-side instead.
    pub looped_at: Option<usize>,
}

/// The single DRY sequence breaker sent for OCR, chosen so it never matches.
///
/// U+001F UNIT SEPARATOR. A vision model transcribing a page emits the text it
/// sees; it has no reason to produce a C0 control character, and a page cannot
/// contain one to be read. So the breaker list is never triggered and DRY
/// accumulates across the whole decode — including across the newlines,
/// colons, quotes and asterisks that llama.cpp would otherwise break on, which
/// is the behaviour `no_repeat_ngram_size` has and the default breaker list
/// destroys.
///
/// Present rather than absent, and one element rather than none, because the
/// server accepts neither of the alternatives. See `request_body`.
const DRY_NEVER_MATCHES: &str = "\u{001F}";

/// Builds the request body for one page.
///
/// Split out so the argument shape can be asserted without a server: the
/// image has to precede the prompt, and the sampler settings have to survive
/// into the payload or Q4_K_M loops.
pub fn request_body(
    model_id: &str,
    mime: &str,
    encoded_image: &str,
    profile: &OcrProfile,
) -> Value {
    json!({
        "model": model_id,
        "messages": [{
            "role": "user",
            "content": [
                { "type": "image_url",
                  "image_url": { "url": format!("data:{};base64,{}", mime, encoded_image) } },
                { "type": "text", "text": profile.prompt() },
            ],
        }],
        "max_tokens": profile.max_decode_tokens,
        "temperature": profile.temperature,
        "stream": true,
        // llama.cpp accepts DRY per request. These stand in for Baidu's
        // no_repeat_ngram processor; see ocr_profile for why they are not
        // optional.
        "dry_multiplier": profile.dry_multiplier,
        "dry_allowed_length": profile.dry_allowed_length,
        "dry_penalty_last_n": profile.dry_penalty_last_n,
        // One sentinel that cannot occur in decoded text, and this is the
        // load-bearing part.
        //
        // llama.cpp defaults `dry_sequence_breakers` to the four characters
        // newline, colon, double-quote and asterisk,
        // and resets the repetition match at every one of them. A read that
        // degenerates into an endless asterisk-and-newline ladder is therefore
        // built *entirely* out of
        // sequence breakers: DRY never accumulates a match, never applies a
        // penalty, and — at temperature 0, where there is no sampling noise to
        // knock the decode out of the cycle — the loop runs to the decode cap.
        // That was observed on a real page: the whole 16,384-token budget came
        // back as asterisks with every DRY setting above in force.
        //
        // The processor this stands in for has no notion of breakers at all.
        // `no_repeat_ngram_size=35` bans a repeated 35-gram wherever it falls,
        // punctuation included, so suppressing the list is not a liberty — it
        // is the closer reproduction of the reference implementation.
        //
        // It cannot be expressed as `[]`. The server validates the field and
        // rejects an empty array outright:
        //
        //     400 Bad Request — "Field 'dry_sequence_breakers': Error:
        //     dry_sequence_breakers must be a non-empty array of strings"
        //
        // and omitting the field is worse than either, because that restores
        // the very defaults above. So the list is one control character that no
        // OCR decode can emit. DRY tokenises each breaker and resets on a
        // match; a breaker that never matches is a list of no breakers, which
        // is what this needs, expressed in the only shape the server accepts.
        "dry_sequence_breakers": [DRY_NEVER_MATCHES],
    })
}

/// Streams one page, calling `on_event` as regions and text arrive.
///
/// `cancel` is checked between chunks: a page the operator abandoned should
/// stop costing GPU time immediately, and the summary says it was cancelled
/// rather than reporting a short page as a complete one.
/// The summary for a page that was stopped before it produced anything.
///
/// A distinct shape from an empty read, and the distinction is load-bearing:
/// `assemble_pdf_text` lists a page that yielded nothing as unread, and a page
/// nobody looked at because the turn was stopped is exactly that — not a blank
/// page, which is a claim about the document.
fn stopped_before_reading(elapsed_ms: u64) -> StreamSummary {
    StreamSummary {
        tokens: 0,
        elapsed_ms,
        hit_decode_cap: false,
        cancelled: true,
        looped_at: None,
    }
}

pub async fn stream_ocr<F>(
    client: &reqwest::Client,
    base_url: &str,
    model_id: &str,
    image_path: &Path,
    profile: &OcrProfile,
    cancel: &CancelToken,
    mut on_event: F,
) -> Result<StreamSummary>
where
    F: FnMut(OcrEvent),
{
    // Before anything is read off disk or sent anywhere. A page whose turn was
    // stopped while an earlier page was still decoding must not start at all.
    if cancel.is_cancelled() {
        return Ok(stopped_before_reading(0));
    }
    let bytes = tokio::fs::read(image_path)
        .await
        .with_context(|| format!("could not read page image {}", image_path.display()))?;
    let mime = match image_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        // The model's own tooling accepts PNG and JPEG only; refusing here
        // gives a better message than a server-side decode failure.
        other => bail!("unsupported page image type: {:?}", other),
    };

    let body = request_body(model_id, mime, &BASE64.encode(&bytes), profile);
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));

    let started = std::time::Instant::now();
    // The header wait, which had no cancellation at all.
    //
    // A vision model handed a dense A1 drawing spends tens of seconds on the
    // image before it sends a single byte, and this `await` is where all of
    // that time is spent. A flag tested in the loop below is not reached until
    // the model has already done the work it was being asked to stop doing —
    // so Stop pressed here used to take effect only once the page had finished.
    //
    // Dropping the future closes the socket, which is what actually stops
    // llama-server: the same mechanism the repetition guard relies on.
    let response = tokio::select! {
        biased;
        () = cancel.cancelled() => return Ok(stopped_before_reading(started.elapsed().as_millis() as u64)),
        sent = client.post(&url).json(&body).send() => {
            sent.with_context(|| format!("OCR request to {url} failed"))?
        }
    };

    if !response.status().is_success() {
        let status = response.status();
        let detail = response.text().await.unwrap_or_default();
        bail!("OCR request to {url} returned {status}: {detail}");
    }

    let mut sse = SseDecoder::new();
    let mut spans = SpanParser::new();
    let mut stream = response.bytes_stream();
    let mut tokens: u32 = 0;
    let mut hit_decode_cap = false;
    let mut cancelled = false;
    // Watches the decoded text rather than the token stream, because the
    // failure it exists for is invisible at token level. See the module.
    let mut repetition = RepetitionGuard::new();

    // The chunk wait.
    //
    // This was `while let Some(next) = stream.next().await` with the flag
    // tested *inside* the body — that is, only ever after a chunk had already
    // arrived. A model that has stopped producing, or is thinking between
    // tokens, leaves this parked on the socket with the check unreachable, so a
    // page that stalled could not be stopped at all.
    //
    // Selecting on the token means the wait itself is interruptible. `biased`
    // so cancellation is polled first: with the default random order a stream
    // that always has a chunk ready could starve the cancel arm indefinitely.
    'outer: loop {
        let next = tokio::select! {
            biased;
            () = cancel.cancelled() => {
                cancelled = true;
                break;
            }
            item = stream.next() => item,
        };
        let Some(next) = next else { break };
        let chunk = next.context("OCR stream broke mid-page")?;
        let text = String::from_utf8_lossy(&chunk).to_string();
        for frame in sse.feed(&text) {
            match frame {
                SseFrame::Done => break 'outer,
                SseFrame::Data(payload) => {
                    let value: Value = match serde_json::from_str(&payload) {
                        Ok(v) => v,
                        // A malformed frame is skipped rather than aborting a
                        // page that is otherwise readable.
                        Err(_) => continue,
                    };
                    if hit_length_cap(&value) {
                        hit_decode_cap = true;
                    }
                    if let Some(delta) = delta_text(&value) {
                        tokens += 1;
                        for event in spans.feed(&delta) {
                            on_event(event);
                        }
                        // Tested after the events are emitted, so the scan
                        // view keeps what was read up to the turn instead of
                        // losing the last window to the abort.
                        if repetition.feed(&delta).is_some() {
                            // Leaving the loop drops `stream`, which closes
                            // the connection, which is what actually stops
                            // llama-server generating. Without that the
                            // server decodes to the cap with nobody reading
                            // and the page still costs the minutes this guard
                            // exists to save.
                            break 'outer;
                        }
                    }
                }
            }
        }
    }

    for frame in sse.finish() {
        if let SseFrame::Data(payload) = frame {
            if let Ok(value) = serde_json::from_str::<Value>(&payload) {
                if let Some(delta) = delta_text(&value) {
                    for event in spans.feed(&delta) {
                        on_event(event);
                    }
                }
            }
        }
    }
    for event in spans.finish() {
        on_event(event);
    }

    Ok(StreamSummary {
        tokens,
        elapsed_ms: started.elapsed().as_millis() as u64,
        hit_decode_cap,
        cancelled,
        looped_at: repetition.tripped_at(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai_engine::ocr_profile::OcrDetent;

    /// Shaped like a real llama-server stream, carrying the region format the
    /// model actually emits (`label [x, y, x, y]text`, no markers).
    const STREAM: &str = concat!(
        "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"title [77, 51, 723, 86]\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"REPORT\\n\"}}]}\n\n",
        "data: [DONE]\n\n"
    );

    fn frames(chunks: &[&str]) -> Vec<SseFrame> {
        let mut d = SseDecoder::new();
        let mut out: Vec<SseFrame> = chunks.iter().flat_map(|c| d.feed(c)).collect();
        out.extend(d.finish());
        out
    }

    #[test]
    fn a_whole_stream_decodes_to_its_frames() {
        let got = frames(&[STREAM]);
        assert_eq!(got.len(), 4);
        assert_eq!(got[3], SseFrame::Done);
    }

    /// The same hazard as the span parser, one layer down: a frame split
    /// across reads must not vanish.
    #[test]
    fn no_read_boundary_can_drop_a_frame() {
        let expected = frames(&[STREAM]);
        for split in 1..STREAM.len() {
            if !STREAM.is_char_boundary(split) {
                continue;
            }
            let (a, b) = STREAM.split_at(split);
            assert_eq!(frames(&[a, b]), expected, "split at {split} lost a frame");
        }
    }

    #[test]
    fn crlf_framing_decodes_the_same_as_lf() {
        let crlf = STREAM.replace('\n', "\r\n");
        assert_eq!(frames(&[&crlf]), frames(&[STREAM]));
    }

    #[test]
    fn keep_alive_comments_are_not_data() {
        let got = frames(&[": ping\n\ndata: {\"a\":1}\n\n"]);
        assert_eq!(got, vec![SseFrame::Data("{\"a\":1}".into())]);
    }

    #[test]
    fn a_role_only_chunk_yields_no_text() {
        let v: Value =
            serde_json::from_str(r#"{"choices":[{"delta":{"role":"assistant"}}]}"#).unwrap();
        assert_eq!(delta_text(&v), None);
    }

    #[test]
    fn a_content_chunk_yields_its_text() {
        let v: Value = serde_json::from_str(r#"{"choices":[{"delta":{"content":"hi"}}]}"#).unwrap();
        assert_eq!(delta_text(&v).as_deref(), Some("hi"));
    }

    #[test]
    fn a_non_streaming_shape_is_still_read() {
        // Guards against a silently empty page if the endpoint answers in the
        // message shape rather than the delta shape.
        let v: Value =
            serde_json::from_str(r#"{"choices":[{"message":{"content":"hi"}}]}"#).unwrap();
        assert_eq!(delta_text(&v).as_deref(), Some("hi"));
    }

    #[test]
    fn a_length_finish_reason_is_recognised_as_the_decode_cap() {
        let v: Value =
            serde_json::from_str(r#"{"choices":[{"delta":{},"finish_reason":"length"}]}"#).unwrap();
        assert!(hit_length_cap(&v));
        let stop: Value =
            serde_json::from_str(r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#).unwrap();
        assert!(!hit_length_cap(&stop));
    }

    /// The three layers composed: bytes in, regions out.
    #[test]
    fn the_full_pipeline_turns_a_byte_stream_into_regions() {
        let mut sse = SseDecoder::new();
        let mut spans = SpanParser::new();
        let mut events = Vec::new();
        for frame in sse.feed(STREAM) {
            if let SseFrame::Data(p) = frame {
                let v: Value = serde_json::from_str(&p).unwrap();
                if let Some(delta) = delta_text(&v) {
                    events.extend(spans.feed(&delta));
                }
            }
        }
        events.extend(spans.finish());

        let regions: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, OcrEvent::Region { .. }))
            .collect();
        assert_eq!(regions.len(), 1);
        let text: String = events
            .iter()
            .filter_map(|e| match e {
                OcrEvent::Text { delta, .. } => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        // The parser is line-oriented, so a region's text carries its newline.
        assert_eq!(text.trim_end(), "REPORT");
    }

    /// End-to-end over a **real** llama-server response.
    ///
    /// Captured from Unlimited-OCR Q6_K with the patched projector reading a
    /// page whose ink positions are known by construction. This is the test
    /// that would have caught the two defects the standalone gate exposed:
    /// the model emits `label [x, y, x, y]text` rather than the documented
    /// `<|det|>` markers, and its coordinates are normalised rather than
    /// input pixels. Both were previously assumed, and both assumptions were
    /// wrong.
    #[test]
    fn a_real_captured_stream_produces_regions_that_land_on_the_ink() {
        use crate::ai_engine::ocr_profile::{to_page, CoordSpace, PageGeometry};

        let raw = include_str!("../../tests/fixtures/unlimited-ocr-real-stream.sse");
        let mut sse = SseDecoder::new();
        let mut spans = SpanParser::new();
        let mut events = Vec::new();
        let mut saw_done = false;
        for frame in sse.feed(raw) {
            match frame {
                SseFrame::Done => saw_done = true,
                SseFrame::Data(p) => {
                    let v: Value = serde_json::from_str(&p).expect("server frame is JSON");
                    if let Some(delta) = delta_text(&v) {
                        events.extend(spans.feed(&delta));
                    }
                }
            }
        }
        events.extend(spans.finish());
        assert!(saw_done, "the capture must contain the [DONE] sentinel");

        let regions: Vec<(String, RawBoxLocal)> = events
            .iter()
            .filter_map(|e| match e {
                OcrEvent::Region {
                    label, bbox: Some(b), ..
                } => Some((label.clone(), RawBoxLocal(b.x1, b.y1, b.x2, b.y2))),
                _ => None,
            })
            .collect();
        assert!(
            regions.len() >= 5,
            "expected the page's regions, got {regions:?}"
        );

        let text: String = events
            .iter()
            .filter_map(|e| match e {
                OcrEvent::Text { delta, .. } => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        assert!(text.contains("QUARTERLY FIELD REPORT"), "got {text:?}");
        assert!(text.contains("AUTH-7731"), "the corner marker must be read");
        assert!(text.contains("412 MOhm"), "table cells must survive");

        // The page is 1000x1400 and its corner marker's ink starts at
        // y=1299, x=620. Mapping the model's box under the measured
        // convention has to land there.
        let geometry = PageGeometry {
            page_width: 1000,
            page_height: 1400,
            input_width: 1000,
            input_height: 1400,
        };
        let (_, corner) = regions
            .iter()
            .find(|(l, _)| l == "footer")
            .expect("the corner marker is its own region");
        let mapped = to_page(
            RawBox {
                x1: corner.0,
                y1: corner.1,
                x2: corner.2,
                y2: corner.3,
            },
            CoordSpace::Normalised,
            geometry,
        );
        assert!(mapped.in_bounds, "mapped corner fell off the page: {mapped:?}");
        assert!(
            (mapped.y1 - 1299).abs() <= 25,
            "corner marker y should be ~1299, got {}",
            mapped.y1
        );
        assert!(
            (mapped.x1 - 620).abs() <= 25,
            "corner marker x should be ~620, got {}",
            mapped.x1
        );
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct RawBoxLocal(i32, i32, i32, i32);

    #[test]
    fn the_request_puts_the_image_before_the_prompt_and_keeps_the_sampler() {
        let profile = OcrDetent::Fastest.profile();
        let body = request_body("unlimited-ocr-q4-k-m", "image/png", "AAAA", &profile);
        let parts = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "image_url");
        assert_eq!(parts[1]["type"], "text");
        // The prompt measured to work on this build; see ocr_profile::prompt.
        assert_eq!(parts[1]["text"], "Free OCR.");
        assert_eq!(body["stream"], true);
        assert_eq!(body["temperature"], 0.0);
        assert_eq!(body["dry_allowed_length"], 35);
        assert_eq!(body["max_tokens"], profile.max_decode_tokens);
    }

    /// llama-server rejects the two obvious ways of saying "no breakers".
    ///
    /// An empty array is refused by the field validator:
    ///
    ///     400 — dry_sequence_breakers must be a non-empty array of strings
    ///
    /// and omitting the key restores the `['\n', ':', '"', '*']` default that
    /// makes an all-asterisk decode invisible to DRY. Every OCR request failed
    /// on the first of those for as long as the empty array was sent, so the
    /// shape is asserted rather than left to the comment beside it.
    #[test]
    fn the_dry_breaker_list_is_non_empty_and_cannot_match_a_transcription() {
        for detent in OcrDetent::ALL {
            let body = request_body("m", "image/png", "AAAA", &detent.profile());
            let breakers = body["dry_sequence_breakers"]
                .as_array()
                .expect("dry_sequence_breakers must be present, or llama.cpp uses its defaults");

            assert!(
                !breakers.is_empty(),
                "{detent:?} sent an empty breaker list, which the server refuses with a 400"
            );
            assert!(
                breakers.iter().all(|b| b.is_string()),
                "{detent:?} sent a non-string breaker; the server requires an array of strings"
            );

            // The point of the list is that it never fires. Anything a page can
            // actually contain would reset the repetition match and reinstate
            // the loop this setting exists to stop.
            for breaker in breakers {
                let text = breaker.as_str().unwrap();
                assert!(
                    text.chars().all(|c| c.is_control() && !c.is_whitespace()),
                    "{detent:?} sent {text:?} as a breaker — a transcription can emit that"
                );
            }
        }
    }
}

/// Stopping a page that is not producing anything.
///
/// The two waits this module spends nearly all of its time in, driven against a
/// real socket. Both used to be uninterruptible:
///
/// - The **header wait** had no cancellation at all. A vision model handed a
///   dense drawing spends tens of seconds on the image before it sends a byte,
///   and `send().await` simply sat there.
/// - The **chunk wait** tested the flag only *after* a chunk had arrived, so a
///   model that stalled mid-page left the check unreachable.
///
/// Each test is bounded by a timeout, and the timeout elapsing **is** the
/// failure: a regression here does not produce a wrong value, it produces a
/// hang, and a test that hangs for ever reports nothing.
#[cfg(test)]
mod cancellation_tests {
    use super::*;
    use crate::agent_runtime::cancellation::CancelToken;
    use crate::ai_engine::ocr_profile::OcrDetent;
    use std::time::Duration;
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;

    /// A page image on disk, which `stream_ocr` reads before it connects.
    fn a_png() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("temp dir");
        // The 8-byte signature is all this needs: the file is base64'd whole
        // and never decoded on this side.
        std::fs::write(dir.path().join("page-1.png"), b"\x89PNG\r\n\x1a\n")
            .expect("write the page");
        dir
    }

    /// A server that accepts the connection and then does exactly nothing.
    ///
    /// Stands in for a model still looking at the image: connected, no headers.
    async fn a_server_that_never_answers() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            // Held rather than dropped: dropping the socket would close the
            // connection and turn this into a transport error, which is a
            // different test.
            let mut held = Vec::new();
            while let Ok((socket, _)) = listener.accept().await {
                held.push(socket);
            }
        });
        format!("http://{addr}")
    }

    /// A server that sends headers and one frame, then stalls for ever.
    ///
    /// Stands in for a model that produced some of a page and stopped.
    async fn a_server_that_stalls_mid_page() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            let mut held = Vec::new();
            while let Ok((mut socket, _)) = listener.accept().await {
                let head = concat!(
                    "HTTP/1.1 200 OK\r\n",
                    "content-type: text/event-stream\r\n",
                    "transfer-encoding: chunked\r\n\r\n",
                );
                let _ = socket.write_all(head.as_bytes()).await;
                let body = "data: {\"choices\":[{\"delta\":{\"content\":\"half a page\"}}]}\n\n";
                let _ = socket
                    .write_all(format!("{:x}\r\n{}\r\n", body.len(), body).as_bytes())
                    .await;
                let _ = socket.flush().await;
                // No terminating chunk. The stream stays open and silent.
                held.push(socket);
            }
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn a_stop_interrupts_the_wait_for_the_first_byte() {
        let dir = a_png();
        let base = a_server_that_never_answers().await;
        let token = CancelToken::never();

        let stopper = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(150)).await;
            stopper.cancel();
        });

        let summary = tokio::time::timeout(
            Duration::from_secs(10),
            stream_ocr(
                &reqwest::Client::new(),
                &base,
                "test-model",
                &dir.path().join("page-1.png"),
                &OcrDetent::Fast.profile(),
                &token,
                |_| {},
            ),
        )
        .await
        .expect("the header wait must be interruptible, not merely pollable")
        .expect("a stopped page is not an error");

        assert!(summary.cancelled, "the page must report that it was stopped");
        assert_eq!(summary.tokens, 0, "nothing was read");
    }

    #[tokio::test]
    async fn a_stop_interrupts_a_stream_that_has_gone_quiet() {
        let dir = a_png();
        let base = a_server_that_stalls_mid_page().await;
        let token = CancelToken::never();

        let stopper = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(300)).await;
            stopper.cancel();
        });

        let seen = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let sink = seen.clone();
        let summary = tokio::time::timeout(
            Duration::from_secs(10),
            stream_ocr(
                &reqwest::Client::new(),
                &base,
                "test-model",
                &dir.path().join("page-1.png"),
                &OcrDetent::Fast.profile(),
                &token,
                move |event| {
                    if let OcrEvent::Text { delta, .. } = event {
                        if let Ok(mut held) = sink.lock() {
                            held.push_str(&delta);
                        }
                    }
                },
            ),
        )
        .await
        .expect("the chunk wait must be interruptible, not checked only on arrival")
        .expect("a stopped page is not an error");

        assert!(summary.cancelled);
        // Whatever arrived before the stop is kept. A stopped read is not a
        // failed one, and the text already decoded is still evidence.
        assert!(
            seen.lock().unwrap().contains("half a page"),
            "the text that did arrive must survive the stop"
        );
    }

    /// A page whose turn was already stopped does no work at all: no disk read,
    /// no socket, no model. This is what stops page 12 of 40 from starting.
    #[tokio::test]
    async fn a_page_of_an_already_stopped_turn_never_connects() {
        let dir = a_png();
        let summary = tokio::time::timeout(
            Duration::from_secs(5),
            stream_ocr(
                &reqwest::Client::new(),
                // Nothing is listening here. Reaching the socket at all would
                // be a connection error rather than a clean cancellation, so
                // this address is the assertion.
                "http://127.0.0.1:1",
                "test-model",
                &dir.path().join("page-1.png"),
                &OcrDetent::Fast.profile(),
                &CancelToken::cancelled_now(),
                |_| {},
            ),
        )
        .await
        .expect("an already-stopped page returns at once")
        .expect("a stopped page is not an error");

        assert!(summary.cancelled);
        assert_eq!(summary.tokens, 0);
    }
}

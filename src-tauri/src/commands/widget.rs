//! Serving a model-written widget its own origin.
//!
//! ## Why this module exists at all
//!
//! The build spec's widget lane is: the model emits HTML, CSS and JS, and it
//! renders live in the message inside a sandboxed iframe with its own CSP. The
//! obvious implementation is `<iframe srcdoc="…" sandbox="allow-scripts">` with
//! the widget's policy in a `<meta>` tag, and it does not work.
//!
//! A `srcdoc` document **inherits the embedding page's CSP**. This was measured
//! rather than assumed, with a positive control, before any of this was
//! written:
//!
//! | Embedder `script-src` | Inline script inside a sandboxed `srcdoc` frame |
//! |---|---|
//! | `'self' 'unsafe-inline'` | ran |
//! | `'self'` | blocked |
//!
//! ARJUN ships the second. Every local scheme behaves this way — `about:blank`,
//! `blob:` and `data:` inherit too — so there is no variant of "assemble it on
//! the client" that escapes it. The only way to give a widget a policy of its
//! own is to serve it from an origin of its own, with real response headers.
//! That is what this is.
//!
//! The alternative was adding `'unsafe-inline'` to the application's own
//! `script-src`, which would license inline script across the entire surface to
//! buy it inside one iframe. For a product whose thesis is that nothing runs or
//! reaches the network without a reviewed decision, that trade is not available.
//!
//! ## What the widget can do, and what bounds it
//!
//! The response carries `default-src 'none'`. No network of any kind: no fetch,
//! no XHR, no WebSocket, no font, no stylesheet, no image except `data:`. The
//! frame is mounted with `sandbox="allow-scripts"` and deliberately *without*
//! `allow-same-origin`, so it runs at an opaque origin and cannot reach the
//! parent DOM, cookies or storage; without `allow-popups` or
//! `allow-top-navigation`, so it cannot open or steer anything.
//!
//! The spec asks for the CSP to be "locked to an explicit CDN allowlist". Here
//! that allowlist is empty, and that is the correct value rather than a
//! shortcut: this application installs air gapped, so a widget that could reach
//! a CDN would be a widget that could reach anything.
//!
//! ## Why the HTML is not sanitised on the way in
//!
//! `svgSanitize.ts` allowlists elements because an SVG fence is injected into
//! the *application's own document*, where a `<script>` would run with the
//! surface's privileges. None of that is true here. This document is never part
//! of the application's DOM; it is a separate origin with no network and no
//! access outward, so script inside it is the feature rather than the hazard.
//!
//! Adding an HTML allowlist on top would be the weaker half of a boundary the
//! browser already enforces, and it would fail the day someone found a tag
//! nobody listed. What is bounded instead is size, count and lifetime — the
//! resources a runaway widget could actually consume.

use std::sync::{Arc, Mutex};

use tauri::http::{Request, Response};

/// The largest document that will be served.
///
/// A widget is a picture or a small explainer. Anything past this is either a
/// model that has lost its place or an attempt to exhaust memory, and both are
/// better refused with a sentence than rendered.
pub const MAX_WIDGET_BYTES: usize = 512 * 1024;

/// How many prepared widgets are kept.
///
/// One conversation can hold many, and each is held only so its frame can fetch
/// it. The oldest is dropped when the cap is reached: a widget scrolled far out
/// of view that is asked for again re-prepares, which costs one round trip and
/// bounds what a long session can retain.
pub const MAX_WIDGETS: usize = 32;

/// The prepared documents, oldest first.
///
/// A `Vec` rather than a map because eviction is by age and the count is small;
/// a linked hash map would be a dependency taken on to order 32 items.
#[derive(Default)]
pub struct WidgetStore {
    inner: Mutex<Vec<(String, String)>>,
}

/// Shared handle, as the other stores in this application are shared.
pub type Widgets = Arc<WidgetStore>;

impl WidgetStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Keeps a document under its id, evicting the oldest past the cap.
    fn keep(&self, id: String, document: String) {
        let Ok(mut kept) = self.inner.lock() else {
            return;
        };
        kept.retain(|(existing, _)| existing != &id);
        kept.push((id, document));
        while kept.len() > MAX_WIDGETS {
            kept.remove(0);
        }
    }

    /// The document for an id, if it is still held.
    pub fn get(&self, id: &str) -> Option<String> {
        let kept = self.inner.lock().ok()?;
        kept.iter()
            .find(|(existing, _)| existing == id)
            .map(|(_, document)| document.clone())
    }

    /// How many are held.
    pub fn len(&self) -> usize {
        self.inner.lock().map(|kept| kept.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The policy every widget document is served under.
///
/// One constant, so there is exactly one answer to "what may a widget do" and
/// it is greppable. `frame-ancestors` is left unset so the application can
/// embed it; everything a widget could otherwise reach is denied by name rather
/// than by omission, because a directive that is merely absent is easy to
/// mistake for one that is set.
const WIDGET_CSP: &str = "default-src 'none'; \
     style-src 'unsafe-inline'; \
     script-src 'unsafe-inline'; \
     img-src data:; \
     font-src data:; \
     media-src data:; \
     connect-src 'none'; \
     form-action 'none'; \
     base-uri 'none'; \
     object-src 'none'";

/// Accepts a widget document and returns the id its frame should load.
///
/// Refuses rather than truncates. A widget cut off at the cap would render as a
/// half-drawn picture with nothing indicating anything was missing, which is
/// the class of quiet wrongness this repository has a standing rule against.
///
/// The body of the command, separated from the `#[tauri::command]` wrapper so
/// the tests exercise the same code the application runs rather than a copy of
/// it that can drift.
pub fn prepare(widgets: &Widgets, document: &str) -> Result<String, String> {
    if document.trim().is_empty() {
        return Err("A widget needs a document; this one was empty.".to_string());
    }
    if document.len() > MAX_WIDGET_BYTES {
        return Err(format!(
            "That widget is {} KiB. The limit is {} KiB, so it was not prepared rather than \
             served half-drawn.",
            document.len() / 1024,
            MAX_WIDGET_BYTES / 1024
        ));
    }

    let id = uuid::Uuid::new_v4().to_string();
    widgets.keep(id.clone(), document.to_string());
    Ok(id)
}

#[tauri::command]
pub fn widget_prepare(
    widgets: tauri::State<'_, Widgets>,
    document: String,
) -> Result<String, String> {
    prepare(&widgets, &document)
}

/// The id a request is asking for: the last non-empty path segment.
///
/// Tolerant of the two shapes one registration produces — `widget://localhost/
/// <id>` on macOS and Linux, `http://widget.localhost/<id>` on Windows —
/// because the frontend builds one URL and the platform decides how it arrives.
fn requested_id(uri: &str) -> Option<String> {
    let without_query = uri.split(['?', '#']).next().unwrap_or(uri);
    without_query
        .rsplit('/')
        .find(|segment| !segment.is_empty())
        .map(|segment| segment.to_string())
}

/// Serves a prepared widget, or says plainly that it is gone.
///
/// A 404 carries a readable document rather than an empty body: the frame will
/// render whatever it is handed, and an empty one is exactly the blank box the
/// display half of the build spec is about. This one says what happened.
pub fn serve(widgets: &Widgets, request: &Request<Vec<u8>>) -> Response<Vec<u8>> {
    let Some(id) = requested_id(&request.uri().to_string()) else {
        return respond(400, missing_document("that request named no widget"));
    };

    match widgets.get(&id) {
        Some(document) => respond(200, document),
        None => respond(404, missing_document("this widget is no longer held in memory")),
    }
}

fn respond(status: u16, body: String) -> Response<Vec<u8>> {
    Response::builder()
        .status(status)
        .header("Content-Type", "text/html; charset=utf-8")
        .header("Content-Security-Policy", WIDGET_CSP)
        // No caching: an id is fetched once, by one frame, and a cached copy
        // would outlive the eviction this module relies on.
        .header("Cache-Control", "no-store")
        .header("X-Content-Type-Options", "nosniff")
        .body(body.into_bytes())
        .unwrap_or_else(|_| Response::new(Vec::new()))
}

/// What a frame shows when there is nothing to show it.
///
/// Styled inline because the policy above forbids a stylesheet, and phrased for
/// a reader rather than a developer.
fn missing_document(reason: &str) -> String {
    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Widget unavailable</title>\
         </head><body style=\"margin:0;padding:10px;font:12px system-ui,sans-serif;\
         color:#888\">This interactive view could not be loaded — {reason}. Ask again to \
         rebuild it.</body></html>"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> Widgets {
        Arc::new(WidgetStore::new())
    }

    #[test]
    fn a_prepared_widget_can_be_fetched_by_its_id() {
        let widgets = store();
        let id = prepare(&widgets, "<p>hello</p>").unwrap();
        assert_eq!(widgets.get(&id).as_deref(), Some("<p>hello</p>"));
    }

    #[test]
    fn an_empty_document_is_refused_rather_than_served_blank() {
        let widgets = store();
        let error = prepare(&widgets, "   ").unwrap_err();
        assert!(error.contains("empty"), "{error}");
        assert!(widgets.is_empty());
    }

    #[test]
    fn a_document_past_the_cap_is_refused_not_truncated() {
        // Half a picture, with nothing saying it is half, is worse than a
        // sentence saying it was not built.
        let widgets = store();
        let huge = "x".repeat(MAX_WIDGET_BYTES + 1);
        let error = prepare(&widgets, &huge).unwrap_err();
        assert!(error.contains("limit"), "{error}");
        assert!(widgets.is_empty());
    }

    #[test]
    fn the_oldest_widget_is_dropped_once_the_cap_is_reached() {
        let widgets = store();
        let first = prepare(&widgets, "<p>first</p>").unwrap();
        for n in 0..MAX_WIDGETS {
            prepare(&widgets, &format!("<p>{n}</p>")).unwrap();
        }
        assert_eq!(widgets.len(), MAX_WIDGETS);
        assert!(
            widgets.get(&first).is_none(),
            "the first widget should have been evicted"
        );
    }

    #[test]
    fn two_widgets_never_share_an_id() {
        let widgets = store();
        let one = prepare(&widgets, "<p>one</p>").unwrap();
        let two = prepare(&widgets, "<p>two</p>").unwrap();
        assert_ne!(one, two);
        assert_eq!(widgets.get(&one).as_deref(), Some("<p>one</p>"));
        assert_eq!(widgets.get(&two).as_deref(), Some("<p>two</p>"));
    }

    #[test]
    fn a_widget_that_is_gone_is_answered_with_words_not_an_empty_body() {
        // The frame renders whatever it is handed. An empty body is the blank
        // box that the display half of the spec exists to remove.
        let document = missing_document("this widget is no longer held in memory");
        assert!(document.contains("could not be loaded"));
        assert!(document.contains("no longer held in memory"));
    }

    #[test]
    fn the_id_is_read_from_either_platforms_url_shape() {
        assert_eq!(requested_id("widget://localhost/abc-123").as_deref(), Some("abc-123"));
        assert_eq!(
            requested_id("http://widget.localhost/abc-123").as_deref(),
            Some("abc-123")
        );
        assert_eq!(
            requested_id("http://widget.localhost/abc-123?v=2").as_deref(),
            Some("abc-123")
        );
    }

    #[test]
    fn the_policy_denies_the_network_in_every_form_a_widget_could_ask() {
        // The list is the security boundary, so it is asserted rather than
        // left to review. An air-gapped product has no CDN to allow, so the
        // spec's allowlist is empty on purpose — and empty is checkable.
        assert!(WIDGET_CSP.contains("default-src 'none'"));
        assert!(WIDGET_CSP.contains("connect-src 'none'"));
        assert!(WIDGET_CSP.contains("form-action 'none'"));
        assert!(WIDGET_CSP.contains("base-uri 'none'"));
        assert!(WIDGET_CSP.contains("object-src 'none'"));
        assert!(
            !WIDGET_CSP.contains("http"),
            "no host may be reachable from a widget"
        );
    }
}

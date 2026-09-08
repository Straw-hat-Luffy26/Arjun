//! Large documents, end to end, against a model server that is really running.
//!
//! ## What these prove that the unit tests do not
//!
//! Every other test around this work uses a budget that is a number in the
//! test. These use the window a `llama-server` was *started* with, ask that
//! server to count the tokens, and then send it the composed turn. The claim
//! being tested is the one an operator cares about:
//!
//! > A document many times larger than the model's context window is fully
//! > processed, stays queryable, and answering from it does not produce
//! > `400 ... exceeds the available context size`.
//!
//! ## Running them
//!
//! They skip unless a server is reachable, and the skip says so rather than
//! passing quietly:
//!
//! ```text
//! llama-server --model <any gguf> --port 61354 --ctx-size 4096
//! ARJUN_TEST_MODEL_BASE_URL=http://127.0.0.1:61354/v1 cargo test --lib -- large_document
//! ```
//!
//! A deliberately small `--ctx-size` is the point. At 4 096 tokens a twelve-page
//! report is already over the window, which makes "larger than the context" the
//! ordinary case rather than something needing a 200-page fixture to reach.
//!
//! ## What is measured
//!
//! [`Report`] carries the figures, and every one of them is counted rather than
//! estimated: pages from the reader, passages from the cut, tokens from the
//! server's own tokeniser, and the HTTP status from the completion the server
//! actually answered.

use std::collections::{BTreeMap, HashMap};

use super::doc_pipeline::{self, Candidate, Completeness};
use super::documents::{DocumentStore, NewExtraction, Sighting};
use crate::knowledge::chunking::Chunk;

/// Where the server is, unless the environment says otherwise.
const DEFAULT_BASE_URL: &str = "http://127.0.0.1:61354/v1";

const OWNER: &str = "validation-owner";
const OTHER_OWNER: &str = "validation-other";
const CONVERSATION: &str = "c-validation";
const OTHER_CONVERSATION: &str = "c-validation-other";

/// Held back for the reply, and for the chat template's own scaffolding.
///
/// Deliberately *not* imported from `commands::agent`. A test that read the
/// same constant from the same place as the code could not catch the code
/// changing it; these are the numbers this validation holds the pipeline to.
const REPLY_RESERVE: u32 = 512;
const TEMPLATE_OVERHEAD: u32 = 64;

/// Where the server is, refused unless it is on this machine.
///
/// The override exists so the validation can point at a server on another port,
/// not at a service somewhere else: this test sends whole documents to whatever
/// it is given, and a URL from the environment is exactly the shape of mistake
/// that turns a local validation into an exfiltration. Checked here rather than
/// trusted, the same way [`crate::serving::probe`] checks it.
/// One test at a time against the model server.
///
/// `llama-server` serves a small number of slots — one, by default — and both
/// of the tests below send whole documents to it. Run in parallel they queue
/// behind each other and time out, which reads as a retrieval failure and is
/// not one: each passes alone and they failed only together. The same lock the
/// cross-process suite uses on the same server, for the same reason.
fn model_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<tokio::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

fn base_url() -> String {
    let url = std::env::var("ARJUN_TEST_MODEL_BASE_URL")
        .unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
    let host = url
        .split("//")
        .nth(1)
        .unwrap_or_default()
        .split('/')
        .next()
        .unwrap_or_default()
        .rsplit_once(':')
        .map(|(host, _)| host.to_string())
        .unwrap_or_default();
    assert!(
        matches!(host.as_str(), "127.0.0.1" | "localhost" | "[::1]"),
        "ARJUN_TEST_MODEL_BASE_URL must be a loopback address; {url} is not one, and this test \
         sends whole documents to it"
    );
    url
}

/// A client per call site. Cheap enough at this volume, and it keeps the
/// timeout explicit where a 200-page turn is being sent.
///
/// arjun-egress-ok: loopback only, enforced by `base_url` above, which refuses
/// any host that is not this machine before a request is ever built. Proxies
/// are removed for the same reason they are removed in `serving::probe`: an
/// inherited proxy variable would turn a loopback request into one that leaves.
fn client() -> reqwest::Client {
    reqwest::Client::builder() // arjun-egress-ok: loopback only, enforced by base_url above
        .timeout(std::time::Duration::from_secs(120))
        .no_proxy()
        .build()
        .expect("an HTTP client can be built")
}

/// Whether a server is there to test against.
async fn reachable(base_url: &str) -> bool {
    let root = base_url.trim_end_matches("/v1").trim_end_matches('/');
    client()
        .get(format!("{root}/health"))
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

/// The window the server was started with, asked of the server.
async fn served_window(base_url: &str) -> Option<u32> {
    crate::serving::probe::served_context_tokens(base_url).await
}

/// One synthetic document, built to a shape a real one has.
struct Fixture {
    name: &'static str,
    /// Page number to text, exactly as `read_attachment` produces it.
    pages: BTreeMap<u32, String>,
    /// A distinctive code on an early page, a middle page and the last page.
    /// Each is unguessable, so finding one proves it was retrieved rather than
    /// produced.
    early: (u32, String),
    middle: (u32, String),
    late: (u32, String),
    /// The question asked of [`Self::middle`], phrased so exactly one answer is
    /// correct.
    ///
    /// Per fixture rather than generic, because a real page carries more than
    /// one code and "the reference code on page 13" does not name which. That
    /// ambiguity produced a wrong answer from a turn that held the right
    /// passage — a defect in the question, and worth fixing rather than
    /// tolerating.
    question: String,
}

impl Fixture {
    /// A real content address, so the store's own id validation is exercised
    /// rather than bypassed by a hand-written string.
    fn sha(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(self.name.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    fn last_page(&self) -> u32 {
        self.pages.keys().copied().max().unwrap_or(0)
    }

    fn chunks(&self) -> Vec<Chunk> {
        let borrowed: Vec<(u32, &str)> = self
            .pages
            .iter()
            .map(|(page, text)| (*page, text.as_str()))
            .collect();
        crate::knowledge::chunking::chunk_pages(&self.sha(), &borrowed)
    }

    fn completeness(&self, chunks: &[Chunk]) -> Completeness {
        let owned: Vec<(u32, String)> = self
            .pages
            .iter()
            .map(|(page, text)| (*page, text.clone()))
            .collect();
        doc_pipeline::measure(
            self.last_page(),
            &owned,
            chunks,
            chunks.len() as u32,
            false,
        )
    }

    fn extraction(&self) -> NewExtraction {
        NewExtraction {
            sha256: self.sha(),
            name: self.name.to_string(),
            kind: "pdf-text".to_string(),
            pages: self.last_page(),
            truncated: false,
            page_text: self.pages.clone(),
            sighting: Sighting {
                owner_user_id: OWNER.to_string(),
                conversation_id: CONVERSATION.to_string(),
                message_id: format!("m-{}", self.name),
                run_id: format!("r-{}", self.name),
                at: chrono::Utc::now().to_rfc3339(),
                ocr_model_id: None,
                ocr_detent: None,
            },
        }
    }
}

/// Prose that reads like a real page and compresses like one.
fn filler(page: u32, lines: usize) -> String {
    let mut out = String::new();
    for line in 0..lines {
        out.push_str(&format!(
            "Section {page}.{line} records the routine inspection findings for this area. \
             No deviation from the applicable procedure was observed during the walkdown.\n"
        ));
    }
    out
}

/// The eight shapes the validation covers, smallest first.
fn fixtures() -> Vec<Fixture> {
    let mut all = Vec::new();

    // 1. Small: comfortably inside any window.
    {
        let mut pages = BTreeMap::new();
        pages.insert(
            1,
            "Purchase order PO-88421.\nSupplier: Vaskar Engineering.".to_string(),
        );
        pages.insert(2, "Delivery within 30 days. Contact: works office.".to_string());
        all.push(Fixture {
            name: "small-order",
            early: (1, "PO-88421".to_string()),
            middle: (1, "PO-88421".to_string()),
            late: (2, "30 days".to_string()),
            question: "What is the purchase order number? Quote it exactly.".to_string(),
            pages,
        });
    }

    // 2. Near the limit: a few thousand tokens, close to a 4K window.
    {
        let mut pages = BTreeMap::new();
        for page in 1..=6u32 {
            pages.insert(page, filler(page, 18));
        }
        pages.insert(1, format!("{}\nRegister reference NL-3391-A.", filler(1, 16)));
        pages.insert(3, format!("{}\nMid-book marker QM-5507.", filler(3, 16)));
        pages.insert(6, format!("{}\nClosing entry ZC-9120.", filler(6, 16)));
        all.push(Fixture {
            name: "near-limit-register",
            early: (1, "NL-3391-A".to_string()),
            middle: (3, "QM-5507".to_string()),
            late: (6, "ZC-9120".to_string()),
            question: "What is the mid-book marker? Quote it exactly.".to_string(),
            pages,
        });
    }

    // 3. Over the limit: several times any small window.
    {
        let mut pages = BTreeMap::new();
        for page in 1..=40u32 {
            pages.insert(page, filler(page, 20));
        }
        pages.insert(2, format!("{}\nFirst-page datum FD-2210.", filler(2, 18)));
        pages.insert(20, format!("{}\nCentre datum CD-4471.", filler(20, 18)));
        pages.insert(40, format!("{}\nFinal datum LD-8802.", filler(40, 18)));
        all.push(Fixture {
            name: "over-limit-manual",
            early: (2, "FD-2210".to_string()),
            middle: (20, "CD-4471".to_string()),
            late: (40, "LD-8802".to_string()),
            question: "What is the centre datum? Quote it exactly.".to_string(),
            pages,
        });
    }

    // 4. A multi-page PDF with real headings, so the heading trail is exercised.
    {
        let mut pages = BTreeMap::new();
        for page in 1..=12u32 {
            pages.insert(
                page,
                format!(
                    "{page} Inspection Area {page}\n\n\
                     {page}.1 Scope\n{}\n\
                     {page}.2 Findings\n{}\n",
                    filler(page, 6),
                    filler(page, 6)
                ),
            );
        }
        pages.insert(
            1,
            "1 Inspection Area 1\n\n1.1 Scope\nOpening tag OT-1104 applies to this survey.\n"
                .to_string(),
        );
        pages.insert(
            7,
            "7 Inspection Area 7\n\n7.2 Findings\nMidspan tag MT-6620 was recorded here.\n"
                .to_string(),
        );
        pages.insert(
            12,
            "12 Inspection Area 12\n\n12.2 Findings\nClosing tag CT-9931 completes the survey.\n"
                .to_string(),
        );
        all.push(Fixture {
            name: "multi-page-survey",
            early: (1, "OT-1104".to_string()),
            middle: (7, "MT-6620".to_string()),
            late: (12, "CT-9931".to_string()),
            question: "What is the midspan tag? Quote it exactly.".to_string(),
            pages,
        });
    }

    // 5. Very large: two hundred pages, far past anything a window holds.
    {
        let mut pages = BTreeMap::new();
        for page in 1..=200u32 {
            pages.insert(page, filler(page, 12));
        }
        pages.insert(3, format!("{}\nVolume opener VO-7001.", filler(3, 10)));
        pages.insert(101, format!("{}\nDeep centre DC-5150.", filler(101, 10)));
        pages.insert(200, format!("{}\nVolume close VC-3300.", filler(200, 10)));
        all.push(Fixture {
            name: "very-large-volume",
            early: (3, "VO-7001".to_string()),
            middle: (101, "DC-5150".to_string()),
            late: (200, "VC-3300".to_string()),
            question: "What is the deep centre code? Quote it exactly.".to_string(),
            pages,
        });
    }

    // 6. Tables, which the cut must keep whole.
    {
        let mut pages = BTreeMap::new();
        for page in 1..=10u32 {
            pages.insert(
                page,
                format!(
                    "{page} Thickness Survey {page}\n\n\
                     | Point | Nominal | Measured | Status |\n\
                     |-------|---------|----------|--------|\n\
                     | P{page}-1 | 9.5 mm | 9.1 mm | Accept |\n\
                     | P{page}-2 | 9.5 mm | 8.8 mm | Accept |\n\
                     | P{page}-3 | 9.5 mm | 8.2 mm | Review |\n\n{}",
                    filler(page, 4)
                ),
            );
        }
        pages.insert(
            1,
            "1 Thickness Survey 1\n\n\
             | Point | Nominal | Measured | Status |\n\
             |-------|---------|----------|--------|\n\
             | TA-101 | 9.5 mm | 9.4 mm | Accept |\n"
                .to_string(),
        );
        pages.insert(
            5,
            "5 Thickness Survey 5\n\n\
             | Point | Nominal | Measured | Status |\n\
             |-------|---------|----------|--------|\n\
             | TB-505 | 12.0 mm | 7.7 mm | Replace |\n"
                .to_string(),
        );
        pages.insert(
            10,
            "10 Thickness Survey 10\n\n\
             | Point | Nominal | Measured | Status |\n\
             |-------|---------|----------|--------|\n\
             | TC-910 | 6.0 mm | 5.9 mm | Accept |\n"
                .to_string(),
        );
        all.push(Fixture {
            name: "table-survey",
            early: (1, "TA-101".to_string()),
            middle: (5, "TB-505".to_string()),
            late: (10, "TC-910".to_string()),
            question: "Which survey point has the status Replace? Quote the point exactly."
                .to_string(),
            pages,
        });
    }

    // 7. Repeated sections: the same heading over and over, one unique line in
    //    each of three places. This is where a prefix strategy fails hardest —
    //    every page looks like page one.
    {
        let mut pages = BTreeMap::new();
        for page in 1..=30u32 {
            pages.insert(
                page,
                format!(
                    "4 Daily Log\n\n4.1 Shift Notes\n\
                     Routine shift. Nothing to report. Handover accepted.\n\
                     Routine shift. Nothing to report. Handover accepted.\n{}",
                    filler(page, 4)
                ),
            );
        }
        pages.insert(
            2,
            "4 Daily Log\n\n4.1 Shift Notes\nIncident reference IR-2002 was raised.\n".to_string(),
        );
        pages.insert(
            15,
            "4 Daily Log\n\n4.1 Shift Notes\nIncident reference IR-1515 was raised.\n".to_string(),
        );
        pages.insert(
            30,
            "4 Daily Log\n\n4.1 Shift Notes\nIncident reference IR-3030 was raised.\n".to_string(),
        );
        all.push(Fixture {
            name: "repeated-log",
            early: (2, "IR-2002".to_string()),
            middle: (15, "IR-1515".to_string()),
            late: (30, "IR-3030".to_string()),
            question: "What incident reference was raised on page 15? Quote it exactly."
                .to_string(),
            pages,
        });
    }

    // 8. OCR-heavy: page furniture, broken lines and stray characters, as a
    //    vision model leaves them.
    {
        let mut pages = BTreeMap::new();
        for page in 1..=25u32 {
            pages.insert(
                page,
                format!(
                    "REV C          SHEET {page} OF 25          DRG-4471\n\
                     ---------------------------------------------\n\
                     NOTE 1  ALL DIMENSIONS IN MILLIMETRES UNLESS\n\
                     NOTED  OTHERWISE.  WELD  PROCEDURE  PER  WPS-12.\n\
                     ITEM  QTY  DESCRIPTION\n\
                     001   2    FLANGE  ASSY\n\
                     002   8    STUD  M20\n{}",
                    filler(page, 3)
                ),
            );
        }
        pages.insert(
            1,
            "REV C   SHEET 1 OF 25   DRG-4471\nTITLE BLOCK REFERENCE TB-0001 APPLIES.\n"
                .to_string(),
        );
        pages.insert(
            13,
            "REV C   SHEET 13 OF 25   DRG-4471\nGASKET TORQUE GT-4713 IS 47 NM.\n".to_string(),
        );
        pages.insert(
            25,
            "REV C   SHEET 25 OF 25   DRG-4471\nFINAL NOTE FN-2525 SIGNED OFF.\n".to_string(),
        );
        all.push(Fixture {
            name: "ocr-drawing-set",
            early: (1, "TB-0001".to_string()),
            middle: (13, "GT-4713".to_string()),
            late: (25, "FN-2525".to_string()),
            // Named by what it is, not by which page it is on: this page also
            // carries the drawing number DRG-4471, and "the reference code on
            // page 13" is genuinely ambiguous between the two.
            question: "What is the gasket torque tag? Quote it exactly.".to_string(),
            pages,
        });
    }

    all
}

/// Everything measured for one document. Printed as the validation table.
struct Report {
    name: &'static str,
    pages_total: u32,
    pages_extracted: u32,
    pages_failed: usize,
    chunks_total: u32,
    chunks_processed: u32,
    chunks_failed: u32,
    extracted_tokens: u32,
    /// Passages this turn could afford.
    chunks_in_turn: usize,
    /// What the server's own tokeniser counted for the composed turn.
    context_used: u32,
    ceiling: u32,
    /// Left out of *this turn*. Never means "lost" — see [`Self::content_lost`].
    omitted_from_turn: u32,
    /// Whether the passage holding the answer actually reached the model.
    ///
    /// Reported separately from [`Self::answered_correctly`] because they fail
    /// for opposite reasons and have opposite remedies. This being false is a
    /// retrieval defect and is this work's problem. This being true while the
    /// answer is wrong is a 4B model at 4K declining to quote, which is not.
    answer_in_prompt: bool,
    /// Lost from the *record*. Must always be false.
    content_lost: bool,
    http_status: u16,
    answered_correctly: bool,
}

/// The prompt shape `commands::agent::compose_prompt_from_selection` produces.
fn render(
    question: &str,
    selection: &doc_pipeline::Selection,
    counted: &HashMap<String, Completeness>,
) -> String {
    let body = doc_pipeline::render(selection, counted);
    if body.trim().is_empty() {
        return question.to_string();
    }
    format!("<attachments>\n{body}</attachments>\n\n{question}")
}

/// The composed turn for one document and one question, fitted the way
/// `commands::agent::fit_documents_to_window` fits it: estimate first, then
/// halve and re-choose until the *server's* count is under the ceiling.
async fn compose(
    fixture: &Fixture,
    chunks: &[Chunk],
    completeness: &Completeness,
    question: &str,
    window: u32,
    base_url: &str,
) -> (String, doc_pipeline::Selection, u32) {
    let sha = fixture.sha();
    let candidates = vec![Candidate {
        sha256: &sha,
        name: fixture.name,
        chunks,
        pages: fixture.last_page(),
        pinned: false,
    }];
    let counted: HashMap<String, Completeness> =
        [(sha.clone(), completeness.clone())].into_iter().collect();

    let fixed = doc_pipeline::estimate_tokens(question)
        .saturating_add(REPLY_RESERVE)
        .saturating_add(TEMPLATE_OVERHEAD);
    let mut budget = window.saturating_sub(fixed);
    let ceiling = window.saturating_sub(REPLY_RESERVE + TEMPLATE_OVERHEAD);

    let mut selection = doc_pipeline::select(question, &candidates, budget);
    let mut prompt = render(question, &selection, &counted);
    let mut used = 0u32;
    for _ in 0..4 {
        used = crate::serving::probe::count_tokens(base_url, &prompt)
            .await
            .expect("the server counts tokens");
        if used <= ceiling {
            break;
        }
        budget /= 2;
        selection = doc_pipeline::select(question, &candidates, budget);
        prompt = render(question, &selection, &counted);
    }
    (prompt, selection, used)
}

/// Asks the real server, and returns the status rather than unwrapping it.
///
/// The status is the point: a turn that overflows comes back 400 with
/// `exceeds the available context size`, and this validation exists to show
/// that it does not.
async fn ask(base_url: &str, prompt: &str) -> (u16, String) {
    let response = client()
        .post(format!("{}/chat/completions", base_url.trim_end_matches('/')))
        .json(&serde_json::json!({
            "model": "arjun-validation",
            "messages": [
                {
                    "role": "system",
                    "content": "Answer only from the passages given. Quote the reference code exactly."
                },
                { "role": "user", "content": prompt }
            ],
            "max_tokens": 160,
            "temperature": 0.0,
        }))
        .send()
        .await
        .expect("the server answers");
    let status = response.status().as_u16();
    let body = response.text().await.unwrap_or_default();
    if status != 200 {
        return (status, body);
    }
    let text = serde_json::from_str::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| {
            v.pointer("/choices/0/message/content")
                .and_then(|c| c.as_str())
                .map(str::to_string)
        })
        .unwrap_or_default();
    (status, text)
}

/// The headline validation: eight document shapes, a real server, and the
/// numbers to show for it.
#[tokio::test]
async fn every_document_shape_is_processed_whole_and_answered_without_overflowing() {
    let _serialised = model_lock().lock().await;
    let base_url = base_url();
    if !reachable(&base_url).await {
        eprintln!(
            "skipping: no model server at {base_url}. Start one with \
             `llama-server --model <gguf> --port 61354 --ctx-size 4096`."
        );
        return;
    }
    let Some(window) = served_window(&base_url).await else {
        eprintln!("skipping: {base_url} would not report its context size");
        return;
    };
    eprintln!("\nValidating against a server holding {window} tokens.\n");

    let dir = tempfile::tempdir().expect("a temp dir");
    let store = DocumentStore::open(dir.path()).expect("the store opens");
    let mut reports = Vec::new();

    for fixture in fixtures() {
        let sha = fixture.sha();
        let chunks = fixture.chunks();
        let completeness = fixture.completeness(&chunks);

        // ── Stored whole, before anything decides what fits ───────────────
        store.record(fixture.extraction()).expect("stored");

        // ── No page missing, and the order is the document's ──────────────
        let held = store
            .get(&sha, OWNER, Some(CONVERSATION))
            .expect("readable")
            .expect("present");
        let stored_pages: Vec<u32> = held.page_text.iter().map(|p| p.page).collect();
        let mut expected_pages: Vec<u32> = fixture
            .pages
            .iter()
            .filter(|(_, text)| !text.trim().is_empty())
            .map(|(page, _)| *page)
            .collect();
        expected_pages.sort_unstable();
        assert_eq!(
            stored_pages, expected_pages,
            "{}: pages were dropped or reordered on the way into the store",
            fixture.name
        );
        assert!(
            held.completeness.fully_processed(),
            "{}: the store does not consider the document fully processed: {}",
            fixture.name,
            held.completeness.summary()
        );

        // ── The turn, fitted to the window the server really holds ────────
        let (prompt, selection, used) = compose(
            &fixture,
            &chunks,
            &completeness,
            &fixture.question,
            window,
            &base_url,
        )
        .await;
        let ceiling = window.saturating_sub(REPLY_RESERVE + TEMPLATE_OVERHEAD);

        // ── Reading order preserved in what the model sees ────────────────
        let ordinals: Vec<u32> = selection.chosen.iter().map(|c| c.ordinal).collect();
        let mut sorted = ordinals.clone();
        sorted.sort_unstable();
        assert_eq!(
            ordinals, sorted,
            "{}: passages reached the model out of order",
            fixture.name
        );

        // ── The server answers, and does not refuse for size ──────────────
        let (status, answer) = ask(&base_url, &prompt).await;
        assert_eq!(
            status, 200,
            "{}: the server refused the composed turn: {answer}",
            fixture.name
        );
        assert!(
            !answer.contains("exceeds the available context size"),
            "{}: the turn overflowed the window: {answer}",
            fixture.name
        );

        // ── Retrieval reaches early, middle and late, by content ──────────
        for (label, (page, code)) in [
            ("early", &fixture.early),
            ("middle", &fixture.middle),
            ("late", &fixture.late),
        ] {
            let found = store
                .search(code, OWNER, CONVERSATION, 6)
                .expect("search runs");
            assert!(
                found.hits.iter().any(|hit| hit.text.contains(code)),
                "{}: the {label} reference {code} on page {page} was not retrievable",
                fixture.name
            );
            assert!(
                found
                    .hits
                    .iter()
                    .filter(|hit| hit.text.contains(code))
                    .all(|hit| hit.sha256 == sha),
                "{}: searching for {code} returned another document's passage",
                fixture.name
            );
        }

        reports.push(Report {
            name: fixture.name,
            pages_total: completeness.pages_total,
            pages_extracted: completeness.pages_extracted,
            pages_failed: completeness.pages_failed.len(),
            chunks_total: completeness.chunks_total,
            chunks_processed: completeness.chunks_stored,
            chunks_failed: completeness
                .chunks_total
                .saturating_sub(completeness.chunks_stored),
            extracted_tokens: completeness.extracted_tokens,
            chunks_in_turn: selection.chosen.len(),
            context_used: used,
            ceiling,
            omitted_from_turn: selection.omitted_chunks(),
            answer_in_prompt: prompt.contains(&fixture.middle.1),
            content_lost: !held.completeness.fully_processed(),
            http_status: status,
            answered_correctly: answer.contains(&fixture.middle.1),
        });
    }

    // ── The table the validation is for ───────────────────────────────────
    eprintln!(
        "{:<22}{:>6}{:>6}{:>6}{:>8}{:>6}{:>6}{:>11}{:>9}{:>10}{:>9}{:>9}{:>6}{:>10}{:>8}",
        "document",
        "pages",
        "read",
        "fail",
        "chunks",
        "proc",
        "fail",
        "doc-tokens",
        "in-turn",
        "not-shown",
        "ctx-used",
        "ceiling",
        "http",
        "in-prompt",
        "answer"
    );
    for r in &reports {
        eprintln!(
            "{:<22}{:>6}{:>6}{:>6}{:>8}{:>6}{:>6}{:>11}{:>9}{:>10}{:>9}{:>9}{:>6}{:>10}{:>8}",
            r.name,
            r.pages_total,
            r.pages_extracted,
            r.pages_failed,
            r.chunks_total,
            r.chunks_processed,
            r.chunks_failed,
            r.extracted_tokens,
            r.chunks_in_turn,
            r.omitted_from_turn,
            r.context_used,
            r.ceiling,
            r.http_status,
            if r.answer_in_prompt { "yes" } else { "NO" },
            if r.answered_correctly { "yes" } else { "no" }
        );
    }
    eprintln!();

    // ── The claims, asserted rather than eyeballed ────────────────────────
    for r in &reports {
        assert_eq!(r.chunks_failed, 0, "{}: a passage was not stored", r.name);
        assert_eq!(r.pages_failed, 0, "{}: a page produced no text", r.name);
        assert!(
            !r.content_lost,
            "{}: content was lost from the record",
            r.name
        );
        assert_eq!(r.http_status, 200, "{}: the server refused", r.name);
        assert!(
            r.context_used <= r.ceiling,
            "{}: the turn used {} tokens against a {}-token ceiling",
            r.name,
            r.context_used,
            r.ceiling
        );
    }

    // At least one document must genuinely have been larger than the window, or
    // this whole test proved nothing about large documents.
    assert!(
        reports
            .iter()
            .any(|r| r.extracted_tokens > window && r.omitted_from_turn > 0),
        "no fixture was actually larger than the {window}-token window"
    );

    // ── The claim this work is actually responsible for ───────────────────
    //
    // Retrieval put the answering passage in front of the model, for every
    // document shape, from a window that could not hold any of the large ones.
    // This is asserted without tolerance: a miss here is a selection defect.
    for r in &reports {
        assert!(
            r.answer_in_prompt,
            "{}: the passage holding the answer never reached the model — {} of {} passages \
             were selected, using {} of {} tokens",
            r.name,
            r.chunks_in_turn,
            r.chunks_total,
            r.context_used,
            r.ceiling
        );
    }

    // And the model answered from it, for every shape. Asserted without
    // tolerance now that each question has exactly one correct answer: a miss
    // here means either the passage stopped reaching the model or the passages
    // around it stopped making sense, and both are this work's problem.
    for r in &reports {
        assert!(
            r.answered_correctly,
            "{}: the answering passage was in the turn and the model still did not use it",
            r.name
        );
    }
}

/// Two distant pages in one answer.
///
/// The property a prefix could never have: a question whose answer needs page 3
/// and page 200 of the same document, from a window that holds neither the
/// document nor a tenth of it.
#[tokio::test]
async fn an_answer_can_need_two_pages_two_hundred_apart() {
    let _serialised = model_lock().lock().await;
    let base_url = base_url();
    if !reachable(&base_url).await {
        eprintln!("skipping: no model server at {base_url}");
        return;
    }
    let Some(window) = served_window(&base_url).await else {
        eprintln!("skipping: {base_url} would not report its context size");
        return;
    };

    let fixture = fixtures()
        .into_iter()
        .find(|f| f.name == "very-large-volume")
        .expect("the very large fixture exists");
    let chunks = fixture.chunks();
    let completeness = fixture.completeness(&chunks);

    let question = format!(
        "Two codes appear in this document: {} and {}. State both, exactly as written.",
        fixture.early.1, fixture.late.1
    );
    let (prompt, selection, used) =
        compose(&fixture, &chunks, &completeness, &question, window, &base_url).await;

    // Both passages reached the model, from opposite ends of two hundred pages.
    assert!(
        prompt.contains(&fixture.early.1),
        "the opening code did not reach the model"
    );
    assert!(
        prompt.contains(&fixture.late.1),
        "the closing code did not reach the model"
    );
    assert!(
        selection.omitted_chunks() > 0,
        "the fixture fitted the window, so this proves nothing"
    );

    let (status, answer) = ask(&base_url, &prompt).await;
    assert_eq!(status, 200, "the server refused: {answer}");
    eprintln!(
        "\ntwo distant pages: {used} tokens used, {} of {} passages in the turn\nanswer: {answer}\n",
        selection.chosen.len(),
        completeness.chunks_total
    );
    assert!(
        answer.contains(&fixture.early.1) && answer.contains(&fixture.late.1),
        "the model was given both codes and returned: {answer}"
    );
}

/// The isolation claims, over the same documents.
///
/// No server needed: this is about who may read what, and that is decided
/// entirely on this side.
#[test]
fn documents_do_not_leak_between_owners_conversations_or_each_other() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let store = DocumentStore::open(dir.path()).expect("the store opens");

    let all = fixtures();
    for fixture in &all {
        store.record(fixture.extraction()).expect("stored");
    }

    for fixture in &all {
        let code = &fixture.middle.1;

        // Another owner sees nothing at all — not even that a document exists.
        let theirs = store
            .search(code, OTHER_OWNER, CONVERSATION, 6)
            .expect("search runs");
        assert!(
            theirs.hits.is_empty() && theirs.documents_searched == 0,
            "{}: another owner could search this document",
            fixture.name
        );

        // Another conversation of the same owner sees nothing either.
        let elsewhere = store
            .search(code, OWNER, OTHER_CONVERSATION, 6)
            .expect("search runs");
        assert!(
            elsewhere.hits.is_empty(),
            "{}: another conversation could search this document",
            fixture.name
        );

        // And a hit for one document's code never carries another's text.
        let mine = store
            .search(code, OWNER, CONVERSATION, 6)
            .expect("search runs");
        let matching: Vec<_> = mine
            .hits
            .iter()
            .filter(|hit| hit.text.contains(code))
            .collect();
        assert!(
            !matching.is_empty(),
            "{}: the owner could not find their own code {code}",
            fixture.name
        );
        assert!(
            matching.iter().all(|hit| hit.sha256 == fixture.sha()),
            "{}: a search for {code} returned a passage from another document",
            fixture.name
        );
    }
}

/// The completeness check, stated as the property it is.
///
/// Every page in, every page out, every passage stored — measured from the
/// record on disk rather than from the absence of an error.
#[test]
fn the_record_can_prove_every_page_and_passage_was_processed() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let store = DocumentStore::open(dir.path()).expect("the store opens");

    for fixture in fixtures() {
        store.record(fixture.extraction()).expect("stored");
        let held = store
            .get(&fixture.sha(), OWNER, Some(CONVERSATION))
            .expect("readable")
            .expect("present");

        assert_eq!(
            held.completeness.pages_total, held.completeness.pages_extracted,
            "{}: {} of {} pages produced text",
            fixture.name, held.completeness.pages_extracted, held.completeness.pages_total
        );
        assert!(
            held.completeness.pages_failed.is_empty(),
            "{}: pages {:?} produced nothing",
            fixture.name,
            held.completeness.pages_failed
        );
        assert_eq!(
            held.completeness.chunks_stored, held.completeness.chunks_total,
            "{}: passages were cut but not stored",
            fixture.name
        );
        assert_eq!(
            held.chunks.len() as u32,
            held.completeness.chunks_total,
            "{}: the count and the passages disagree",
            fixture.name
        );
        assert!(
            held.completeness.fully_processed(),
            "{}: {}",
            fixture.name,
            held.completeness.summary()
        );

        // Every stored passage points at a page the document actually has, and
        // they are in reading order. A citation that names a page the document
        // does not have is worse than no citation.
        let pages: Vec<u32> = held.chunks.iter().map(|c| c.page).collect();
        assert!(
            pages.windows(2).all(|w| w[0] <= w[1]),
            "{}: passages are not in page order",
            fixture.name
        );
        assert!(
            pages.iter().all(|page| fixture.pages.contains_key(page)),
            "{}: a passage cites a page the document does not have",
            fixture.name
        );
    }
}

/// What the per-turn document listing costs now that chunks are stored.
///
/// `describe_conversation_documents` calls `for_conversation` on every turn, so
/// this read is on the hot path. Storing the cut beside the page text roughly
/// doubles the bytes it parses, and a doubling on a hot path is worth a number
/// rather than a shrug.
///
/// Not an assertion about wall-clock time — that would be a flaky test on a
/// loaded machine. It prints, and it fails only if the cost has become absurd.
#[test]
fn the_per_turn_document_listing_stays_cheap() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let store = DocumentStore::open(dir.path()).expect("the store opens");

    // Ten documents, one of them the 200-page volume: a heavier conversation
    // than a person is likely to build.
    for fixture in fixtures() {
        store.record(fixture.extraction()).expect("stored");
    }
    for extra in 0..4 {
        let mut pages = BTreeMap::new();
        for page in 1..=200u32 {
            pages.insert(page, filler(page, 12));
        }
        let fixture = Fixture {
            name: Box::leak(format!("bulk-{extra}").into_boxed_str()),
            early: (1, "x".into()),
            middle: (2, "y".into()),
            late: (3, "z".into()),
            question: String::new(),
            pages,
        };
        store.record(fixture.extraction()).expect("stored");
    }

    let started = std::time::Instant::now();
    let listed = store
        .for_conversation(OWNER, CONVERSATION)
        .expect("the listing reads");
    let took = started.elapsed();

    let chunks: usize = listed.iter().map(|d| d.chunks.len()).sum();
    let chars: usize = listed
        .iter()
        .map(|d| d.page_text.iter().map(|p| p.text.len()).sum::<usize>())
        .sum();
    eprintln!(
        "\nper-turn listing: {} documents, {chunks} passages, {chars} characters of page text, \
         {} ms\n",
        listed.len(),
        took.as_millis()
    );

    assert!(
        took < std::time::Duration::from_secs(2),
        "listing {} documents took {took:?}, which is long enough to be felt between pressing \
         enter and the first token",
        listed.len()
    );
}

/// Records written before chunks existed are still searchable.
///
/// ## Why this is not a synthetic test
///
/// Schema-1 records — page text, no chunks, no completeness — are on this
/// machine right now, written by earlier builds. A migration that worked on a
/// fixture and failed on those would mean a document read last week silently
/// stopped being retrievable this week, which is the exact class of failure
/// this work exists to remove.
///
/// The real records are **copied** into a temporary store first. Nothing here
/// touches the application's own directory: the point is to read what earlier
/// builds actually wrote, not to modify it.
///
/// Skips, loudly, when there is no such directory — on a fresh machine or in
/// CI there is nothing to migrate and nothing to prove.
#[test]
fn a_record_written_before_chunks_existed_is_migrated_on_read() {
    let Some(home) = std::env::var_os("APPDATA") else {
        eprintln!("skipping: no APPDATA, so no application store to read");
        return;
    };
    let real = std::path::Path::new(&home)
        .join("com.arjun.workbench")
        .join("documents")
        .join("extractions");
    if !real.is_dir() {
        eprintln!("skipping: no extractions at {}", real.display());
        return;
    }

    let dir = tempfile::tempdir().expect("a temp dir");
    let copied_root = dir.path().join("documents").join("extractions");
    std::fs::create_dir_all(&copied_root).expect("the copy target");

    let mut legacy = 0usize;
    // (sha, owner, conversation) from the record's own first sighting, so the
    // read below goes through the ordinary owner-filtered path.
    let mut copied: Vec<(String, String, String)> = Vec::new();
    for entry in std::fs::read_dir(&real).expect("the real store lists").flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Ok(bytes) = std::fs::read(&path) else { continue };
        // Only the ones this is about: written without a `chunks` field.
        let Ok(parsed) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            continue;
        };
        // A schema-1 record has no `chunks` key at all.
        //
        // An empty `chunks` array is something else entirely: a *current*
        // record whose reader produced nothing — an image with no text, a scan
        // that OCR'd to nothing, a docx that read empty. `DocumentStore::get`
        // deliberately leaves those alone, because `rebuild` only runs when
        // there is page text to cut, and cutting nothing would produce nothing.
        //
        // Conflating the two was this test's own defect: it collected modern
        // empty records, correctly found nothing migrated in them, and reported
        // that as a migration failure.
        if parsed.pointer("/document/chunks").is_some() {
            continue;
        }
        // And without page text there is nothing to cut whatever the schema
        // says, so such a record cannot demonstrate a migration either way.
        let has_text = parsed
            .pointer("/document/pageText")
            .and_then(|t| t.as_array())
            .is_some_and(|pages| !pages.is_empty());
        if !has_text {
            continue;
        }
        let sha = parsed
            .pointer("/document/sha256")
            .and_then(|s| s.as_str())
            .unwrap_or_default()
            .to_string();
        let owner = parsed
            .pointer("/document/seen/0/ownerUserId")
            .and_then(|s| s.as_str())
            .unwrap_or_default()
            .to_string();
        let conversation = parsed
            .pointer("/document/seen/0/conversationId")
            .and_then(|s| s.as_str())
            .unwrap_or_default()
            .to_string();
        if sha.is_empty() || owner.is_empty() || conversation.is_empty() {
            continue;
        }
        legacy += 1;
        let name = path.file_name().expect("a file name");
        std::fs::write(copied_root.join(name), &bytes).expect("the copy is written");
        copied.push((sha, owner, conversation));
    }
    if legacy == 0 {
        eprintln!(
            "skipping: no schema-1 record on this machine — every record carries a chunks field"
        );
        return;
    }

    let store = DocumentStore::open(dir.path()).expect("the copied store opens");
    let mut migrated = 0usize;
    for (sha, owner, conversation) in &copied {
        // The ordinary read, owner-filtered, exactly as a tool call makes it.
        let Ok(Some(document)) = store.get(sha, owner, Some(conversation)) else {
            continue;
        };
        if document.page_text.is_empty() {
            // A record whose reader produced nothing has nothing to cut, and
            // that is not a migration failure.
            continue;
        }
        assert!(
            !document.chunks.is_empty(),
            "{}: a record with {} page(s) of text was read back with no passages, so it is \
             invisible to search",
            document.name,
            document.page_text.len()
        );
        assert_eq!(
            document.completeness.chunks_total as usize,
            document.chunks.len(),
            "{}: the migrated count and the migrated passages disagree",
            document.name
        );
        assert!(
            document
                .chunks
                .iter()
                .all(|chunk| document.page_text.iter().any(|page| page.page == chunk.page)),
            "{}: a migrated passage cites a page the record does not have",
            document.name
        );
        eprintln!(
            "migrated: {} — {} page(s) of text into {} passage(s)",
            document.name,
            document.page_text.len(),
            document.chunks.len()
        );
        migrated += 1;
    }
    assert!(
        migrated > 0,
        "{legacy} legacy record(s) were copied and none of them migrated"
    );
}

/// The same migration, proved without depending on what is on this machine.
///
/// The test above reads the records earlier builds actually wrote, which is the
/// only way to prove the migration against real history. But on a machine whose
/// store has already been migrated it finds nothing and skips — and a test that
/// skips proves nothing. That is not hypothetical: it is the state of every
/// record in this developer's store today, all of them `schemaVersion: 2`.
///
/// So this one constructs the case instead. It writes a genuine schema-1 record
/// — page text and **no `chunks` key at all** — and reads it back through the
/// ordinary owner-filtered path.
///
/// The record is written as literal JSON rather than by serialising
/// `ExtractedDocument`, and that is the whole point: the current struct always
/// emits `chunks` and `completeness`, so a serialised fixture would arrive
/// already migrated and the test would pass without the migration ever running.
#[test]
fn a_schema_one_record_gains_its_passages_when_it_is_read() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let root = dir.path().join("documents").join("extractions");
    std::fs::create_dir_all(&root).expect("the store root");

    let sha = "1".repeat(64);
    let legacy = serde_json::json!({
        "schemaVersion": 1,
        "document": {
            "sha256": sha,
            "name": "inspection-report.pdf",
            "kind": "pdf-text",
            "pages": 2,
            "truncated": false,
            "extractedAt": "2026-01-01T00:00:00Z",
            "pageText": [
                { "page": 1, "text": filler(1, 40) },
                { "page": 2, "text": filler(2, 40) }
            ],
            "seen": [{
                "ownerUserId": "u-1",
                "conversationId": "c-1",
                "messageId": "m-1",
                "runId": "r-1",
                "at": "2026-01-01T00:00:00Z"
            }]
        }
    });
    std::fs::write(
        root.join(format!("{sha}.json")),
        serde_json::to_vec(&legacy).expect("the fixture serialises"),
    )
    .expect("the legacy record is written");

    let store = DocumentStore::open(dir.path()).expect("the store opens");
    let document = store
        .get(&sha, "u-1", Some("c-1"))
        .expect("the read succeeds")
        .expect("the record is visible to the owner who attached it");

    assert!(
        !document.chunks.is_empty(),
        "a schema-1 record with {} page(s) of text was read back with no passages, so it is          invisible to search",
        document.page_text.len()
    );
    assert_eq!(
        document.completeness.chunks_total as usize,
        document.chunks.len(),
        "the migrated count and the migrated passages disagree"
    );
    assert!(
        document
            .chunks
            .iter()
            .all(|chunk| document.page_text.iter().any(|page| page.page == chunk.page)),
        "a migrated passage cites a page the record does not have"
    );
}

/// A record with no page text is left alone, and that is not a failure.
///
/// The counterpart to the test above, and the case that made the machine-backed
/// test fail: an image that carried no text, a scan that OCR'd to nothing. Such
/// a record has an empty `chunks` array and nothing to cut, so `rebuild` must
/// not run and must not invent a passage from nothing.
#[test]
fn a_record_with_no_text_is_not_given_invented_passages() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let root = dir.path().join("documents").join("extractions");
    std::fs::create_dir_all(&root).expect("the store root");

    let sha = "2".repeat(64);
    let empty = serde_json::json!({
        "schemaVersion": 2,
        "document": {
            "sha256": sha,
            "name": "photograph.jpg",
            "kind": "image",
            "pages": 1,
            "truncated": false,
            "extractedAt": "2026-01-01T00:00:00Z",
            "pageText": [],
            "chunks": [],
            "seen": [{
                "ownerUserId": "u-1",
                "conversationId": "c-1",
                "messageId": "m-1",
                "runId": "r-1",
                "at": "2026-01-01T00:00:00Z"
            }]
        }
    });
    std::fs::write(
        root.join(format!("{sha}.json")),
        serde_json::to_vec(&empty).expect("the fixture serialises"),
    )
    .expect("the record is written");

    let store = DocumentStore::open(dir.path()).expect("the store opens");
    let document = store
        .get(&sha, "u-1", Some("c-1"))
        .expect("the read succeeds")
        .expect("the record is visible to the owner who attached it");

    assert!(
        document.chunks.is_empty(),
        "a record with no page text was given {} passage(s), which cite text that does not exist",
        document.chunks.len()
    );
}

/// The cut loses nothing.
///
/// Selection decides what one turn carries; chunking decides what *exists* to
/// be selected from, and if it dropped text on the way there would be no
/// recovering it — search would never find the passage and no page range would
/// contain it, because the store's own listing is derived from the same cut.
///
/// So: every non-trivial word of every page survives into a passage. "At least
/// once" rather than "exactly once", because a section that had to be split
/// carries [`crate::knowledge::chunking`]'s overlap into the next piece, and
/// that duplication is the point of it.
///
/// ## Text and heading trail together
///
/// A heading is not in `chunk.text`. It is consumed into `section_path`, which
/// is where it is more useful: it labels every passage beneath it, it is what
/// turns a passage into a citation, and
/// [`crate::agent_runtime::doc_pipeline`] searches it alongside the body. The
/// first version of this test looked only at `text` and reported `"Scope"` as
/// lost when it was sitting in the trail of the four passages under it — so
/// the haystack is both fields, which is exactly what a search sees.
#[test]
fn every_word_of_every_page_survives_the_cut() {
    for fixture in fixtures() {
        let chunks = fixture.chunks();
        let haystack: String = chunks
            .iter()
            .map(|chunk| {
                format!("{} {}", chunk.section_path.join(" "), chunk.text).to_lowercase()
            })
            .collect::<Vec<_>>()
            .join(" ");

        for (page, text) in &fixture.pages {
            for word in text.split_whitespace() {
                // Table pipes and rules are layout, not content, and the cut is
                // allowed to normalise them.
                let word = word.trim_matches(|c: char| !c.is_alphanumeric());
                if word.len() < 3 {
                    continue;
                }
                assert!(
                    haystack.contains(&word.to_lowercase()),
                    "{}: the word {word:?} from page {page} is in no passage, so nothing can \
                     ever retrieve it",
                    fixture.name
                );
            }
        }
    }
}

/// A heading is retained on every passage beneath it.
///
/// The property the test above depends on, asserted directly rather than left
/// implicit: a passage reading "Replace within 90 days" is nearly useless on
/// its own, and the same passage carrying `["4 Inspection", "4.2 Wall
/// Thickness"]` is evidence. If headings stopped being retained, the test
/// above would still pass — the words would be in the body instead — and the
/// citations would quietly become worthless.
#[test]
fn a_heading_labels_the_passages_beneath_it() {
    let fixture = fixtures()
        .into_iter()
        .find(|f| f.name == "multi-page-survey")
        .expect("the survey fixture has numbered headings");

    let chunks = fixture.chunks();
    let under_a_heading = chunks
        .iter()
        .filter(|chunk| !chunk.section_path.is_empty())
        .count();
    assert!(
        under_a_heading > 0,
        "a document full of numbered sections produced no heading trails at all"
    );

    // The trail is a trail: `1.1 Scope` sits under `1 Inspection Area 1`.
    let nested = chunks
        .iter()
        .find(|chunk| chunk.section_path.len() > 1)
        .unwrap_or_else(|| panic!("no passage carries more than one level of heading"));
    assert!(
        nested.section_path[0].contains("Inspection Area"),
        "the outermost heading is not outermost: {:?}",
        nested.section_path
    );
}

/// Page association survives the cut, for every page of every shape.
///
/// A passage that cites the wrong page is worse than one that cites none: it
/// sends a person to a page that does not say what they were told it says.
#[test]
fn every_passage_is_attributed_to_the_page_its_text_came_from() {
    for fixture in fixtures() {
        for chunk in fixture.chunks() {
            let page_text = fixture
                .pages
                .get(&chunk.page)
                .unwrap_or_else(|| panic!("{}: passage cites absent page {}", fixture.name, chunk.page));
            // The first substantial word of the passage must appear on the page
            // it claims. Checking one word rather than the whole passage,
            // because a split section legitimately carries overlap from the
            // piece before it.
            let Some(first) = chunk
                .text
                .split_whitespace()
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
                .find(|w| w.len() >= 4)
            else {
                continue;
            };
            assert!(
                page_text.to_lowercase().contains(&first.to_lowercase()),
                "{}: a passage attributed to page {} opens with {first:?}, which is not on that \
                 page",
                fixture.name,
                chunk.page
            );
        }
    }
}

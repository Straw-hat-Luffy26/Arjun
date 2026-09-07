//! What the OCR models read, kept after the turn that read it.
//!
//! ## The failure this exists to remove
//!
//! Reading a document is the most expensive thing this product does. A scanned
//! forty-page drawing set goes page by page through a vision model on the
//! operator's own GPU, and it takes minutes. Until now the entire result of that
//! work lived for exactly as long as one prompt string:
//!
//! - [`crate::ai_engine::ocr_budget`] decides how much of the text the window
//!   can afford. A large document is `Chunked` — the beginning goes in and the
//!   rest is dropped — or `ReferenceOnly`, where none of it does.
//! - Whatever was dropped was dropped *permanently*. Nothing held it. The text
//!   existed only inside the composed prompt.
//! - The next turn read nothing at all, because the next turn is a new run and
//!   attachments belong to the request that carried them.
//!
//! So "what does page 31 say?" could not be answered even though page 31 had
//! been read, by a model, four seconds earlier. And it could not be recovered by
//! replaying the conversation either: the assistant's visible answers only ever
//! contained what the model chose to write about the pages it was shown, which
//! by construction excludes the pages it was not.
//!
//! This is the store that keeps it. The read happens once; the pages stay.
//!
//! ## Isolation
//!
//! Content-addressed, so two people attaching the same file converge on one
//! copy — and that is exactly why the owner check cannot be the file path. The
//! record carries a list of [`Sighting`]s, one per time the document entered a
//! conversation, each naming the owner, the conversation, the message and the
//! run. Every read is filtered against that list: a caller sees a document only
//! if *they* have attached it, and a caller asking within a conversation sees it
//! only if it was attached in *that* conversation. A document on disk that the
//! asker has never attached is indistinguishable from one that is not there.
//!
//! ## Bounds
//!
//! [`MAX_PAGES_PER_READ`] and [`MAX_READ_BYTES`] cap what one call returns. The
//! page cap is deliberately the same ten pages `knowledge.load_evidence_region`
//! enforces: a model that has learned one rule about how much of a document it
//! may pull at a time should not have to learn a second, different one for a
//! document that arrived by paperclip rather than off the shelf.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::doc_pipeline::{self, Completeness};
use crate::ai_engine::ocr_profile::OcrDetent;
use crate::knowledge::chunking::{chunk_pages, Chunk};

/// The most pages one retrieval call may return.
///
/// Matches `knowledge.load_evidence_region`. A wider range is refused rather
/// than quietly trimmed, so the model always knows what it did and did not get.
pub const MAX_PAGES_PER_READ: u32 = 10;

/// The ceiling on one retrieval call's text, in bytes.
///
/// Ten pages of a dense table can be far more than ten pages of prose, and the
/// point of retrieval is to put *less* in the window than the whole document
/// would. A read that hits this says so on the way out.
pub const MAX_READ_BYTES: usize = 24 * 1024;

/// How many times one document's arrival is remembered.
///
/// Append-only lists that nothing bounds are how a store that was fine for a
/// month becomes a megabyte of JSON re-read on every tool call. The most recent
/// arrivals are the ones a run is asking about.
const MAX_SIGHTINGS: usize = 64;

/// The most passages one search returns.
///
/// Small on purpose. Search exists so a model can reach a page the turn could
/// not afford, and a search that returns twenty passages has re-created the
/// problem it was called to solve.
pub const MAX_SEARCH_HITS: usize = 6;

/// Version 2 added [`ExtractedDocument::chunks`] and
/// [`ExtractedDocument::completeness`].
///
/// A version 1 file still loads — both fields default — and is re-chunked on
/// read rather than being left unsearchable. See [`DocumentStore::read_file`].
const SCHEMA_VERSION: u32 = 2;

/// One page of a document, as it was read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PageText {
    /// The document's own page number, one-based.
    ///
    /// Not an index into the pages that happened to need OCR. A document whose
    /// second page is a scan reports it as page 2, so the number the model cites
    /// is the number a person turns to.
    pub page: u32,
    pub text: String,
}

/// One time a document entered a conversation, and under whose account.
///
/// The provenance record and the isolation boundary in one structure: what is
/// remembered about how the text came to exist is the same thing that decides
/// who may read it back.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Sighting {
    pub owner_user_id: String,
    pub conversation_id: String,
    /// The assistant cell of the turn the document was attached to.
    pub message_id: String,
    pub run_id: String,
    /// RFC 3339, UTC.
    pub at: String,
    /// The OCR model that read it, or `None` when the file carried its own text
    /// and no model was needed. Reported, never inferred from the file type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ocr_model_id: Option<String>,
    /// The slider stop the read actually ran at.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ocr_detent: Option<OcrDetent>,
}

/// A document this machine has read, and everything it read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtractedDocument {
    /// Content address of the bytes. The document's identity, and the only one:
    /// a file name is what somebody called it, and two of them can differ while
    /// naming the same drawing.
    pub sha256: String,
    /// The name it arrived under, for display. Never used as a path.
    pub name: String,
    /// Which local path handled it: `image`, `pdf-scan`, `pdf-text`, `docx`…
    pub kind: String,
    pub pages: u32,
    /// True when the reader itself stopped early — a workbook past its row cap,
    /// a deck past its slide cap. Carried so a retrieval can repeat it: a
    /// document that was cut by the *reader* is not made whole by reading its
    /// pages back.
    pub truncated: bool,
    /// RFC 3339, UTC. When the text below was produced.
    pub extracted_at: String,
    /// Every page that produced text, in page order.
    pub page_text: Vec<PageText>,
    /// The document cut into retrievable passages, in reading order.
    ///
    /// Derived from [`Self::page_text`] by [`crate::knowledge::chunking`] and
    /// kept beside it rather than recomputed per turn: chunking a forty-page
    /// scan is cheap but not free, and every turn of a conversation would pay
    /// it. Rebuilt whenever the page text changes, so the two cannot disagree.
    ///
    /// This is the field that makes a document larger than the window
    /// survivable. The window decides what one *turn* can afford; this decides
    /// what exists, and it is not the same decision.
    #[serde(default)]
    pub chunks: Vec<Chunk>,
    /// What was read, what was not, and how much of it there is.
    ///
    /// The completeness check. Written at extraction time from work that
    /// actually ran, so "all 42 pages were processed" can be answered from the
    /// record rather than assumed from the absence of an error.
    #[serde(default)]
    pub completeness: Completeness,
    /// Newest last. See [`Sighting`].
    pub seen: Vec<Sighting>,
}

impl ExtractedDocument {
    /// Whether this owner has ever attached this document.
    ///
    /// The isolation check. `conversation_id` narrows it further, which is what
    /// a run uses: a run may read the documents of the conversation it is in,
    /// and a document the same person attached to a different thread is not part
    /// of this one.
    pub fn visible_to(&self, owner_user_id: &str, conversation_id: Option<&str>) -> bool {
        self.seen.iter().any(|sighting| {
            sighting.owner_user_id == owner_user_id
                && match conversation_id {
                    Some(id) => sighting.conversation_id == id,
                    None => true,
                }
        })
    }

    /// The highest page number that produced text.
    pub fn last_page_with_text(&self) -> u32 {
        self.page_text.iter().map(|p| p.page).max().unwrap_or(0)
    }

    /// Re-cuts the document and re-counts it.
    ///
    /// Called after any change to [`Self::page_text`], and on read for a record
    /// written before chunks existed. Deriving rather than storing what the
    /// caller passed is deliberate: the chunks are a *function* of the page
    /// text, and the one way they could ever be wrong is by being computed from
    /// something else.
    fn rebuild(&mut self) {
        let pages: Vec<(u32, &str)> = self
            .page_text
            .iter()
            .map(|page| (page.page, page.text.as_str()))
            .collect();
        self.chunks = chunk_pages(&self.sha256, &pages);
        let owned: Vec<(u32, String)> = self
            .page_text
            .iter()
            .map(|page| (page.page, page.text.clone()))
            .collect();
        // `stored` is the chunk count itself: these chunks are written in the
        // same file, in the same atomic rename, as the page text they came
        // from. There is no separate index that could fall behind, so any
        // number other than "all of them" here would be a fiction.
        let stored = self.chunks.len() as u32;
        self.completeness = doc_pipeline::measure(
            self.pages,
            &owned,
            &self.chunks,
            stored,
            self.truncated,
        );
    }
}

/// An extraction on its way into the store.
#[derive(Debug, Clone)]
pub struct NewExtraction {
    pub sha256: String,
    pub name: String,
    pub kind: String,
    pub pages: u32,
    pub truncated: bool,
    /// Page number to text. A single-blob reader (a spreadsheet, a deck) records
    /// its whole output as page 1, which is what its page count says it is.
    pub page_text: BTreeMap<u32, String>,
    pub sighting: Sighting,
}

/// What one bounded read produced, and what it did not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PageRead {
    pub sha256: String,
    pub name: String,
    pub from_page: u32,
    pub to_page: u32,
    /// The document's total page count, so the model can ask for the next range
    /// without guessing whether one exists.
    pub pages: u32,
    pub found: Vec<PageText>,
    /// Pages in the requested range that produced no text.
    ///
    /// Named rather than omitted. "This page is blank" and "nobody could read
    /// this page" lead to opposite conclusions about whether a clause exists,
    /// and a range that silently returns four pages when five were asked for
    /// says neither.
    pub unread: Vec<u32>,
    /// True when [`MAX_READ_BYTES`] stopped this call before the range ended.
    ///
    /// Deliberately separate from [`Self::source_truncated`]. These were one
    /// field, and conflating them made the renderer tell the model that a
    /// *complete* document had been cut short at extraction time whenever a
    /// call merely hit its own size ceiling — which is both false and the
    /// opposite of useful, because the recovery for this one is to ask for a
    /// narrower range and the recovery for that one is that there is none.
    pub truncated: bool,
    /// True when the reader itself stopped early when the file was first read.
    ///
    /// A property of the document, not of this call. Pages beyond that point
    /// may not exist in the store at all, and no narrower range will find them.
    pub source_truncated: bool,
}

/// One passage a search found.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChunkHit {
    pub sha256: String,
    pub name: String,
    pub chunk_id: String,
    /// The page this passage starts on — what a citation points at.
    pub page: u32,
    /// The headings above it, outermost first. What turns a passage into
    /// evidence rather than a sentence.
    pub section_path: Vec<String>,
    pub text: String,
}

/// What one search over a conversation's documents produced.
///
/// Carries what was searched as well as what was found, because "nothing
/// matched" and "there was nothing to match against" are different answers and
/// a model given only the first will apologise for the wrong thing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchOutcome {
    pub query: String,
    pub hits: Vec<ChunkHit>,
    pub documents_searched: u32,
    pub chunks_searched: u32,
    /// True when [`MAX_READ_BYTES`] cut the result short of
    /// [`MAX_SEARCH_HITS`].
    pub truncated: bool,
}

/// Where the extractions live on disk.
pub struct DocumentStore {
    root: PathBuf,
}

/// The on-disk envelope, versioned like every other store here.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DocumentFile {
    schema_version: u32,
    document: ExtractedDocument,
}

impl DocumentStore {
    /// Open the store at `<app_data_dir>/documents/extractions/`.
    ///
    /// A sibling of `documents/attachments/`, where `commands::ocr` already puts
    /// the bytes under the same content address. Kept apart from it because the
    /// bytes are the input and this is the output, and a future change that
    /// re-reads a document at a different detent must be able to replace one
    /// without touching the other.
    pub fn open(app_data_dir: &Path) -> std::io::Result<Self> {
        let root = app_data_dir.join("documents").join("extractions");
        std::fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    /// The file for one content address.
    ///
    /// The hash is checked rather than trusted. It reaches this from
    /// `commands::ocr`, which computes it — but it also reaches the read path
    /// from a *model's* tool call, and a model-supplied string used unchecked as
    /// a filename is a path traversal with extra steps.
    fn file_path(&self, sha256: &str) -> Option<PathBuf> {
        let looks_like_a_hash = sha256.len() == 64
            && sha256
                .chars()
                .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c));
        looks_like_a_hash.then(|| self.root.join(format!("{sha256}.json")))
    }

    fn read_file(&self, sha256: &str) -> std::io::Result<Option<ExtractedDocument>> {
        let Some(path) = self.file_path(sha256) else {
            return Ok(None);
        };
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        // A record this build cannot parse is treated as absent rather than as
        // an error. The alternative is a chat turn failing outright because a
        // file written by a different version is on disk, and the document is
        // re-readable — the bytes are still in `documents/attachments/`.
        Ok(serde_json::from_slice::<DocumentFile>(&bytes)
            .ok()
            .map(|file| file.document)
            .map(|mut document| {
                // Migration, done on read rather than by a pass over the store.
                //
                // A schema-1 record has pages and no chunks, which would make
                // it invisible to search — the failure mode being that a
                // document read yesterday silently stops being retrievable
                // today. Re-cutting costs microseconds and cannot be forgotten.
                // Not written back here: this is a read path, and the next
                // `record` persists it.
                if document.chunks.is_empty() && !document.page_text.is_empty() {
                    document.rebuild();
                }
                document
            }))
    }

    fn write_file(&self, document: &ExtractedDocument) -> std::io::Result<()> {
        let Some(final_path) = self.file_path(&document.sha256) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "a document id that is not a sha-256 hash",
            ));
        };
        let envelope = DocumentFile {
            schema_version: SCHEMA_VERSION,
            document: document.clone(),
        };
        let bytes = serde_json::to_vec_pretty(&envelope)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let tmp_path = final_path.with_extension("json.tmp");
        {
            use std::io::Write;
            let mut file = std::fs::File::create(&tmp_path)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
        }
        std::fs::rename(&tmp_path, &final_path)?;
        Ok(())
    }

    /// Stores one extraction, or records that a known document arrived again.
    ///
    /// The text is content-addressed, so re-attaching the same file does not
    /// re-store it — but it *does* add a sighting, because the second arrival is
    /// in a different conversation or turn and that is what decides who may read
    /// it. Pages already held are kept: a second read at a lower detent must not
    /// be able to overwrite a better one with a worse one, and a read that
    /// produced text for a page the first read could not is folded in.
    pub fn record(&self, extraction: NewExtraction) -> std::io::Result<ExtractedDocument> {
        let now = chrono::Utc::now().to_rfc3339();
        let mut document =
            self.read_file(&extraction.sha256)?
                .unwrap_or_else(|| ExtractedDocument {
                    sha256: extraction.sha256.clone(),
                    name: extraction.name.clone(),
                    kind: extraction.kind.clone(),
                    pages: extraction.pages,
                    truncated: extraction.truncated,
                    extracted_at: now.clone(),
                    page_text: Vec::new(),
                    chunks: Vec::new(),
                    completeness: Completeness::default(),
                    seen: Vec::new(),
                });

        let mut pages: BTreeMap<u32, String> = std::mem::take(&mut document.page_text)
            .into_iter()
            .map(|page| (page.page, page.text))
            .collect();
        for (number, text) in extraction.page_text {
            if text.trim().is_empty() {
                continue;
            }
            // A page nobody has read yet is filled in; one that is already held
            // is left alone. See the doc comment: the first read to succeed on a
            // page wins, so a later cheaper pass cannot degrade it.
            pages.entry(number).or_insert(text);
        }
        document.page_text = pages
            .into_iter()
            .map(|(page, text)| PageText { page, text })
            .collect();
        document.pages = document.pages.max(extraction.pages);
        document.truncated = document.truncated || extraction.truncated;

        // Same turn twice is one arrival. A run that retries does not need two
        // rows saying the same document reached the same cell.
        let already = document.seen.iter().any(|s| {
            s.owner_user_id == extraction.sighting.owner_user_id
                && s.conversation_id == extraction.sighting.conversation_id
                && s.message_id == extraction.sighting.message_id
        });
        if !already {
            document.seen.push(extraction.sighting);
            if document.seen.len() > MAX_SIGHTINGS {
                let excess = document.seen.len() - MAX_SIGHTINGS;
                document.seen.drain(0..excess);
            }
        }

        // Cut and counted before it is written, so the chunks in the file and
        // the page text in the file are always the same read of the same
        // document. See [`ExtractedDocument::rebuild`].
        document.rebuild();

        self.write_file(&document)?;
        Ok(document)
    }

    /// One document, if this owner has attached it.
    ///
    /// `Ok(None)` for a document that is not there and for one that is but
    /// belongs to somebody else — the caller cannot tell the two apart, which is
    /// the point.
    pub fn get(
        &self,
        sha256: &str,
        owner_user_id: &str,
        conversation_id: Option<&str>,
    ) -> std::io::Result<Option<ExtractedDocument>> {
        let Some(document) = self.read_file(sha256)? else {
            return Ok(None);
        };
        if !document.visible_to(owner_user_id, conversation_id) {
            return Ok(None);
        }
        Ok(Some(document))
    }

    /// Reads a bounded page range back, for a caller that may see the document.
    ///
    /// Refuses a range wider than [`MAX_PAGES_PER_READ`] rather than trimming
    /// it, so the model always knows what it asked for and what it got. Stops on
    /// [`MAX_READ_BYTES`] and says that it did.
    pub fn pages(
        &self,
        sha256: &str,
        owner_user_id: &str,
        conversation_id: Option<&str>,
        from_page: u32,
        to_page: u32,
    ) -> Result<PageRead, String> {
        if from_page == 0 {
            return Err("Pages are numbered from 1.".to_string());
        }
        if to_page < from_page {
            return Err(format!(
                "The range {from_page}-{to_page} ends before it starts."
            ));
        }
        let width = to_page - from_page + 1;
        if width > MAX_PAGES_PER_READ {
            return Err(format!(
                "That is {width} pages and at most {MAX_PAGES_PER_READ} may be read at once. \
                 Ask for the pages you actually need."
            ));
        }

        let document = self
            .get(sha256, owner_user_id, conversation_id)
            .map_err(|error| format!("that document could not be read back: {error}"))?
            .ok_or_else(|| {
                // Deliberately one sentence for both "no such document" and
                // "not yours". Distinguishing them would be a way to find out
                // what somebody else has attached.
                "No document with that id has been attached to this conversation.".to_string()
            })?;

        let mut found = Vec::new();
        let mut unread = Vec::new();
        let mut bytes = 0usize;
        let mut truncated = false;

        for page in from_page..=to_page {
            let Some(held) = document
                .page_text
                .iter()
                .find(|candidate| candidate.page == page)
            else {
                unread.push(page);
                continue;
            };
            if bytes + held.text.len() > MAX_READ_BYTES {
                truncated = true;
                // The rest of the range is not silently absent: it is reported
                // as unread, which is the honest description of a page this call
                // did not return.
                unread.extend(page..=to_page);
                break;
            }
            bytes += held.text.len();
            found.push(held.clone());
        }

        Ok(PageRead {
            sha256: document.sha256,
            name: document.name,
            from_page,
            to_page,
            pages: document.pages,
            found,
            unread,
            truncated,
            source_truncated: document.truncated,
        })
    }

    /// The documents attached to one conversation, newest arrival first.
    ///
    /// Scanned rather than indexed. The set is one person's chat attachments,
    /// which is tens of files and not thousands, and an index is a second thing
    /// to keep correct for a saving nobody would measure.
    pub fn for_conversation(
        &self,
        owner_user_id: &str,
        conversation_id: &str,
    ) -> std::io::Result<Vec<ExtractedDocument>> {
        let mut found = Vec::new();
        let entries = match std::fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(found),
            Err(error) => return Err(error),
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let Some(document) = self.read_file(stem)? else {
                continue;
            };
            if document.visible_to(owner_user_id, Some(conversation_id)) {
                found.push(document);
            }
        }
        let latest_in = |d: &ExtractedDocument| {
            d.seen
                .iter()
                .filter(|s| s.conversation_id == conversation_id)
                .map(|s| s.at.clone())
                .max()
                .unwrap_or_default()
        };
        found.sort_by(|a, b| latest_in(b).cmp(&latest_in(a)));
        Ok(found)
    }

    /// Finds passages by content across the documents of one conversation.
    ///
    /// ## Why this exists alongside [`Self::pages`]
    ///
    /// A page range is the right tool when the model knows where to look. It is
    /// useless when it does not, and after a turn that could only afford a
    /// third of a forty-page document, not knowing is the normal case: the
    /// prompt says "pages 4-31 are not shown", and `read_pages` can only walk
    /// them ten at a time, hoping. Searching asks the question the model
    /// actually has — *where does this document talk about gasket torque* — and
    /// costs one call instead of three.
    ///
    /// ## Isolation
    ///
    /// Exactly [`Self::for_conversation`]'s: owner and conversation both, so a
    /// search can only reach what this person attached to this thread. There is
    /// no cross-conversation search and there is deliberately no way to ask for
    /// one — content-addressed storage means the same bytes may be somebody
    /// else's document, and a query is a fine way to find out what it says.
    ///
    /// ## Bounds
    ///
    /// [`MAX_SEARCH_HITS`] passages and [`MAX_READ_BYTES`] of text, the same
    /// ceiling a page read has. Ranking is
    /// [`crate::agent_runtime::doc_pipeline::rank_chunks`] — the scorer the turn
    /// itself used, so a passage the turn ranked highly is the passage a search
    /// for the same words returns.
    pub fn search(
        &self,
        query: &str,
        owner_user_id: &str,
        conversation_id: &str,
        limit: usize,
    ) -> Result<SearchOutcome, String> {
        let documents = self
            .for_conversation(owner_user_id, conversation_id)
            .map_err(|error| format!("the documents could not be read back: {error}"))?;

        let chunks_searched: u32 = documents.iter().map(|d| d.chunks.len() as u32).sum();
        let documents_searched = documents.len() as u32;

        // Ranked per document and then merged, rather than over one flat list.
        // Term rarity is a property of a document, and pooling two unrelated
        // documents makes a word common in one look rare because the other
        // never uses it.
        let mut scored: Vec<(f64, ChunkHit)> = Vec::new();
        for document in &documents {
            for (score, chunk) in doc_pipeline::rank_chunks(query, &document.chunks) {
                scored.push((
                    score,
                    ChunkHit {
                        sha256: document.sha256.clone(),
                        name: document.name.clone(),
                        chunk_id: chunk.id.clone(),
                        page: chunk.page,
                        section_path: chunk.section_path.clone(),
                        text: chunk.text.clone(),
                    },
                ));
            }
        }
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.page.cmp(&b.1.page))
        });

        let wanted = limit.clamp(1, MAX_SEARCH_HITS);
        let mut hits = Vec::new();
        let mut bytes = 0usize;
        let mut truncated = false;
        for (_, hit) in scored.into_iter().take(wanted) {
            if bytes + hit.text.len() > MAX_READ_BYTES {
                truncated = true;
                break;
            }
            bytes += hit.text.len();
            hits.push(hit);
        }

        Ok(SearchOutcome {
            query: query.to_string(),
            hits,
            documents_searched,
            chunks_searched,
            truncated,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sighting(owner: &str, conversation: &str, message: &str) -> Sighting {
        Sighting {
            owner_user_id: owner.to_string(),
            conversation_id: conversation.to_string(),
            message_id: message.to_string(),
            run_id: "r1".to_string(),
            at: chrono::Utc::now().to_rfc3339(),
            ocr_model_id: Some("unlimited-ocr-q6-k".to_string()),
            ocr_detent: Some(OcrDetent::Detailed),
        }
    }

    fn sha(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    fn extraction(
        sha256: &str,
        pages: &[(u32, &str)],
        owner: &str,
        conversation: &str,
    ) -> NewExtraction {
        NewExtraction {
            sha256: sha256.to_string(),
            name: "drawing.pdf".to_string(),
            kind: "pdf-scan".to_string(),
            pages: pages.len() as u32,
            truncated: false,
            page_text: pages
                .iter()
                .map(|(page, text)| (*page, (*text).to_string()))
                .collect(),
            sighting: sighting(owner, conversation, "a-1"),
        }
    }

    fn store() -> (tempfile::TempDir, DocumentStore) {
        let dir = tempfile::tempdir().expect("a temp dir");
        let store = DocumentStore::open(dir.path()).expect("the store opens");
        (dir, store)
    }

    /// The whole point: a page that budgeting left out is still readable.
    #[test]
    fn a_page_that_never_reached_the_prompt_can_still_be_read() {
        let (_dir, store) = store();
        let id = sha(0xab);
        store
            .record(extraction(
                &id,
                &[(1, "page one"), (2, "page two"), (31, "the clause on 31")],
                "owner-1",
                "c1",
            ))
            .expect("recorded");

        let read = store
            .pages(&id, "owner-1", Some("c1"), 31, 31)
            .expect("the page reads back");
        assert_eq!(read.found.len(), 1);
        assert_eq!(read.found[0].text, "the clause on 31");
        assert_eq!(read.found[0].page, 31, "the document's own numbering");
    }

    #[test]
    fn a_page_with_no_text_is_named_as_unread_rather_than_omitted() {
        let (_dir, store) = store();
        let id = sha(0x11);
        store
            .record(extraction(&id, &[(1, "one"), (3, "three")], "owner-1", "c1"))
            .expect("recorded");
        let read = store.pages(&id, "owner-1", Some("c1"), 1, 3).unwrap();
        assert_eq!(read.found.len(), 2);
        assert_eq!(read.unread, vec![2]);
    }

    #[test]
    fn another_owner_cannot_read_the_same_bytes() {
        let (_dir, store) = store();
        let id = sha(0x22);
        store
            .record(extraction(&id, &[(1, "confidential")], "owner-1", "c1"))
            .expect("recorded");

        assert!(store.get(&id, "owner-2", None).unwrap().is_none());
        let refused = store.pages(&id, "owner-2", None, 1, 1).unwrap_err();
        assert!(
            refused.contains("No document with that id"),
            "the refusal must not confirm the document exists: {refused}"
        );
    }

    #[test]
    fn a_document_from_another_conversation_is_not_in_this_one() {
        let (_dir, store) = store();
        let id = sha(0x33);
        store
            .record(extraction(&id, &[(1, "text")], "owner-1", "c1"))
            .expect("recorded");

        assert!(store.get(&id, "owner-1", Some("c2")).unwrap().is_none());
        // The same person, in the thread they actually attached it to.
        assert!(store.get(&id, "owner-1", Some("c1")).unwrap().is_some());
    }

    #[test]
    fn re_attaching_in_a_second_conversation_adds_a_sighting_not_a_second_copy() {
        let (_dir, store) = store();
        let id = sha(0x44);
        store
            .record(extraction(&id, &[(1, "text")], "owner-1", "c1"))
            .unwrap();
        let mut again = extraction(&id, &[(1, "text")], "owner-1", "c2");
        again.sighting.message_id = "a-2".to_string();
        let document = store.record(again).unwrap();

        assert_eq!(document.seen.len(), 2);
        assert_eq!(document.page_text.len(), 1, "one copy of the text");
        assert!(store.get(&id, "owner-1", Some("c1")).unwrap().is_some());
        assert!(store.get(&id, "owner-1", Some("c2")).unwrap().is_some());
    }

    #[test]
    fn the_same_turn_recorded_twice_is_one_arrival() {
        let (_dir, store) = store();
        let id = sha(0x55);
        store
            .record(extraction(&id, &[(1, "text")], "owner-1", "c1"))
            .unwrap();
        let document = store
            .record(extraction(&id, &[(1, "text")], "owner-1", "c1"))
            .unwrap();
        assert_eq!(document.seen.len(), 1);
    }

    /// A later read must not be able to replace a page with a worse version of
    /// itself — a cheaper detent, or an empty string.
    #[test]
    fn a_second_read_fills_gaps_and_never_overwrites() {
        let (_dir, store) = store();
        let id = sha(0x66);
        store
            .record(extraction(&id, &[(1, "the good read")], "owner-1", "c1"))
            .unwrap();
        let mut second = extraction(
            &id,
            &[(1, "a worse read"), (2, "new page")],
            "owner-1",
            "c1",
        );
        second.sighting.message_id = "a-2".to_string();
        let document = store.record(second).unwrap();

        let page_one = document.page_text.iter().find(|p| p.page == 1).unwrap();
        assert_eq!(page_one.text, "the good read");
        assert!(document.page_text.iter().any(|p| p.page == 2));
    }

    #[test]
    fn an_empty_page_is_not_stored_as_text() {
        let (_dir, store) = store();
        let id = sha(0x77);
        let document = store
            .record(extraction(&id, &[(1, "real"), (2, "   ")], "owner-1", "c1"))
            .unwrap();
        assert_eq!(document.page_text.len(), 1);
        assert_eq!(document.page_text[0].page, 1);
    }

    #[test]
    fn a_range_wider_than_the_cap_is_refused_rather_than_trimmed() {
        let (_dir, store) = store();
        let id = sha(0x88);
        store
            .record(extraction(&id, &[(1, "text")], "owner-1", "c1"))
            .unwrap();
        let refused = store.pages(&id, "owner-1", Some("c1"), 1, 40).unwrap_err();
        assert!(refused.contains("at most"), "{refused}");
        assert!(refused.contains("40 pages"), "{refused}");
    }

    #[test]
    fn a_backwards_or_zero_range_is_refused() {
        let (_dir, store) = store();
        let id = sha(0x99);
        store
            .record(extraction(&id, &[(1, "text")], "owner-1", "c1"))
            .unwrap();
        assert!(store.pages(&id, "owner-1", None, 0, 1).is_err());
        assert!(store.pages(&id, "owner-1", None, 5, 2).is_err());
    }

    #[test]
    fn a_read_that_hits_the_byte_ceiling_says_so() {
        let (_dir, store) = store();
        let id = sha(0xaa);
        let big = "z".repeat(MAX_READ_BYTES / 2);
        let pages: Vec<(u32, &str)> = vec![(1, big.as_str()), (2, big.as_str()), (3, big.as_str())];
        store.record(extraction(&id, &pages, "owner-1", "c1")).unwrap();

        let read = store.pages(&id, "owner-1", Some("c1"), 1, 3).unwrap();
        assert!(read.truncated, "the ceiling was hit and not reported");
        assert!(read.found.len() < 3);
        assert!(
            read.unread.contains(&3),
            "a page this call did not return must be named: {:?}",
            read.unread
        );
    }

    /// A model-supplied id is not a filename. `../` must not escape the store.
    #[test]
    fn an_id_that_is_not_a_hash_reaches_no_file() {
        let (_dir, store) = store();
        for hostile in [
            "../../conversations/secret",
            "..",
            "",
            "ABCDEF",
            &"g".repeat(64),
            &"a".repeat(63),
        ] {
            assert!(
                store.file_path(hostile).is_none(),
                "{hostile} was accepted as a document id"
            );
            assert!(store.get(hostile, "owner-1", None).unwrap().is_none());
        }
    }

    #[test]
    fn a_conversation_lists_only_its_own_documents() {
        let (_dir, store) = store();
        store
            .record(extraction(&sha(0xb1), &[(1, "a")], "owner-1", "c1"))
            .unwrap();
        store
            .record(extraction(&sha(0xb2), &[(1, "b")], "owner-1", "c2"))
            .unwrap();
        store
            .record(extraction(&sha(0xb3), &[(1, "c")], "owner-2", "c1"))
            .unwrap();

        let listed = store.for_conversation("owner-1", "c1").unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].sha256, sha(0xb1));
    }

    #[test]
    fn the_sighting_list_is_bounded() {
        let (_dir, store) = store();
        let id = sha(0xcc);
        for turn in 0..(MAX_SIGHTINGS + 20) {
            let mut arrival = extraction(&id, &[(1, "text")], "owner-1", "c1");
            arrival.sighting.message_id = format!("a-{turn}");
            store.record(arrival).unwrap();
        }
        let document = store.get(&id, "owner-1", Some("c1")).unwrap().unwrap();
        assert_eq!(document.seen.len(), MAX_SIGHTINGS);
        assert_eq!(
            document.seen.last().unwrap().message_id,
            format!("a-{}", MAX_SIGHTINGS + 19),
            "the newest arrivals are the ones kept"
        );
    }

    /// Two different truncations, with two different remedies.
    ///
    /// "this call hit its size limit" is recovered by asking for a narrower
    /// range; "the reader cut this file short" cannot be recovered at all. They
    /// were one field, and the renderer printed the second sentence for the
    /// first case — telling the model a complete document was incomplete, and
    /// withholding the one recovery that would have worked.
    #[test]
    fn a_size_limited_call_does_not_claim_the_document_was_cut_short() {
        let (_dir, store) = store();
        let id = sha(0xe1);
        let big = "z".repeat(MAX_READ_BYTES / 2);
        let pages: Vec<(u32, &str)> = vec![(1, big.as_str()), (2, big.as_str()), (3, big.as_str())];
        store.record(extraction(&id, &pages, "owner-1", "c1")).unwrap();

        let read = store.pages(&id, "owner-1", Some("c1"), 1, 3).unwrap();
        assert!(read.truncated, "this call did hit its ceiling");
        assert!(
            !read.source_truncated,
            "the document itself was read whole and must not be reported as cut short"
        );
    }

    #[test]
    fn a_document_the_reader_cut_short_says_so_on_a_read_that_fitted() {
        let (_dir, store) = store();
        let id = sha(0xe2);
        let mut arrival = extraction(&id, &[(1, "the part that was read")], "owner-1", "c1");
        arrival.truncated = true;
        store.record(arrival).unwrap();

        let read = store.pages(&id, "owner-1", Some("c1"), 1, 1).unwrap();
        assert!(read.source_truncated);
        assert!(!read.truncated, "this call did not hit any ceiling");
    }

    #[test]
    fn provenance_survives_the_round_trip() {
        let (_dir, store) = store();
        let id = sha(0xdd);
        store
            .record(extraction(&id, &[(1, "text")], "owner-1", "c1"))
            .unwrap();
        let document = store.get(&id, "owner-1", Some("c1")).unwrap().unwrap();
        let seen = &document.seen[0];
        assert_eq!(seen.ocr_model_id.as_deref(), Some("unlimited-ocr-q6-k"));
        assert_eq!(seen.ocr_detent, Some(OcrDetent::Detailed));
        assert_eq!(seen.conversation_id, "c1");
        assert_eq!(seen.message_id, "a-1");
        assert_eq!(document.kind, "pdf-scan");
        assert_eq!(document.last_page_with_text(), 1);
    }
}

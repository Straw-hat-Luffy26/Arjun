//! The organisation's own manuals, SOPs and past correspondence.
//!
//! PS 26117 asks the assistant to ground itself in the site's own documents
//! "through a local knowledge base connector" — a connector, not a file
//! uploader. A refinery's procedures live on a network share, not on somebody's
//! desktop, and a product that can only read what was dragged into it will be
//! kept permanently out of date by that friction alone.
//!
//! - [`chunking`]: cutting documents at the boundaries they actually have, so a
//!   retrieved passage still knows which procedure it came from.
//! - [`connector`]: reading a local folder or an internal share, read-only.
//! - [`index`]: finding passages, and never returning one the asker may not see.
//! - [`hybrid`]: combining keyword and vector search into one honest ranking.
//! - [`ingest`]: the pipeline from a file on a share to a searchable passage.
//! - [`evidence`]: handing passages to a model as data, never as instructions.
//! - [`multimodal`]: image regions, tables, and document-type metadata for
//!   multimodal retrieval. The same SQL applies the same clearance, so a
//!   region the asker cannot see is not returned.

pub mod chunking;
pub mod embedding;
pub mod collections;
pub mod connector;
pub mod evidence;
pub mod graph;
pub mod hybrid;
pub mod index;
pub mod ingest;
pub mod multimodal;
pub mod notebook_retrieval;

pub use chunking::{chunk_document, Chunk, ChunkKind};
pub use connector::{discover, plan_sync, Collection, SourceKind, SyncPlan};
pub use embedding::LocalEmbedder;
pub use hybrid::{reciprocal_rank_fusion, Embedder, Hybrid, HybridResults};
pub use index::{KnowledgeIndex, Retrieval, SearchResult};
pub use evidence::{present, EvidenceBlock, PresentedPassage};
pub use collections::CollectionStore;
pub use graph::{
    Assertion, AssertionProvenance, AssertionStatus, EvidenceManifest, Note, NoteKind, Notebook,
    NotebookDocument, NotebookStore, ResearchScope, RetrievalMode,
};
pub use notebook_retrieval::{NotebookRetrieval, ResolvedScope};
pub use ingest::{ingest_collection, DocumentReader, IngestOutcome};
pub use multimodal::{
    BBox, DocumentMeta, ImageRegion, Method as MultimodalMethod, MultimodalIndex,
    NewRegion, NewTable, RegionKind, TableChunk,
};

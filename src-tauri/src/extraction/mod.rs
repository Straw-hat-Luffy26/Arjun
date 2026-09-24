//! The Document & Vision Analyst's tool service (plan P06).
//!
//! Reads authorised documents into [`regions::EvidenceRegion`]s — boxes on
//! pages, with what was read there and how — and keeps transcription and
//! interpretation apart:
//!
//! - **Embedded text** first. A PDF that carries a text layer is read by
//!   parsing it ([`sidecar`]), never by asking a model to read back what the
//!   file already holds.
//! - **Local OCR** for scans ([`ocr`]): Unlimited-OCR on a loopback
//!   `llama-server`, over bounded crops, with the card reserved through the
//!   shared scheduler, every read checked for loops, truncation and malformed
//!   spans, and cached by source, crop, settings and OCR version.
//! - **Vision inference** only from a model an actual image call has proved
//!   can see ([`vision`]), bound to a projector whose header was verified
//!   ([`projector`]). Its output is a labelled proposal, never an observation.

pub mod ocr;
pub mod regions;
pub mod sidecar;
pub mod fields;
pub mod projector;
pub mod service;
pub mod tables;
pub mod vision;

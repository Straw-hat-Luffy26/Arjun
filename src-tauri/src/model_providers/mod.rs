//! Model Providers Module (Phase 4)
//! Generic architecture for interacting with different model sources.

pub mod provider;
pub mod registry;
pub mod ollama_library;
pub mod local;

pub use provider::*;
pub use registry::*;

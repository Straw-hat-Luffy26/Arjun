//! Tauri commands for the benchmark module.
//!
//! The commands are thin wrappers; the work happens in [`crate::benchmarks`].
//!
//! ## What is deliberately absent
//!
//! There used to be a `synthetic_benchmark` command. It returned a fixed row —
//! 38 tok/s, 220 ms TTFT, 3800 MiB, 100% accuracy — stamped with the current
//! time, and the System Health page called it whenever no real run had been
//! recorded, rendering it in the same grid as measured values under the words
//! "Last measured". It is gone, and nothing replaces it: a page with no
//! benchmark says it has no benchmark.
//!
//! This repository has made that mistake once before — `scripts/bench.py`
//! returned a hardcoded 38 tok/s when its binding failed to import, and the
//! constant was published as a measurement. The rule in `CLAUDE.md` is the one
//! that came out of it: measure, or fail loudly.

use std::path::PathBuf;

use serde::Serialize;
use tauri::State;

use crate::benchmarks::{self, BenchmarkResult};
use crate::commands::governance::{require_permission, require_session, CurrentSession};
use crate::identity::Permission;

/// What the System Health page reads.
///
/// Every row that reaches this type was measured. There is no `synthetic` flag
/// because there is no longer a source of unmeasured rows — a flag inviting the
/// UI to render one "greyed out" is how the last constant got on screen.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkRow {
    pub model_id: String,
    pub prompt_tokens: u64,
    pub reply_tokens: u64,
    pub ttft_ms: u64,
    pub total_ms: u64,
    pub tokens_per_second: f64,
    pub vram_peak_mib: u64,
    pub accuracy_pct: f64,
    pub at: String,
    pub hardware_tier: String,
}

impl From<BenchmarkResult> for BenchmarkRow {
    fn from(r: BenchmarkResult) -> Self {
        Self {
            model_id: r.model_id,
            prompt_tokens: r.prompt_tokens,
            reply_tokens: r.reply_tokens,
            ttft_ms: r.ttft_ms,
            total_ms: r.total_ms,
            tokens_per_second: r.tokens_per_second,
            vram_peak_mib: r.vram_peak_mib,
            accuracy_pct: r.accuracy_pct,
            at: r.at,
            hardware_tier: r.hardware_tier,
        }
    }
}

/// Runs a measured benchmark against the loaded model.
///
/// **Not wired, and it refuses rather than pretending.**
///
/// What stood here started a `BenchTimer`, called `finish` on the very next
/// line without asking the model for anything, and wrote the result to the
/// durable history as a real row. Its own comment said so: *"In a real wiring,
/// this is where the agent.service call would be issued."* Everything the row
/// reported was therefore a function of the caller's own arguments and the
/// microseconds between two statements — `accuracy_pct` came straight from the
/// caller, and `tokens_per_second` was the caller's `reply_tokens` divided by
/// approximately nothing.
///
/// Recording that as measured is worse than the synthetic row this module also
/// removed, because a synthetic row at least said what it was. So the command
/// refuses until the generation call is actually made here. `BenchTimer` and
/// `benchmarks::record` are ready for it; what is missing is the model call
/// between `start` and `finish`.
#[tauri::command]
pub async fn run_benchmark(
    _app_data_dir: State<'_, PathBuf>,
    session: State<'_, CurrentSession>,
    model_id: String,
    _prompt_tokens: u64,
    _reply_tokens: u64,
    _accuracy_pct: f64,
) -> Result<BenchmarkRow, String> {
    // A real benchmark runs the live model. That is a model-management
    // operation — it loads the model, exercises it, and writes to the model
    // history. The matrix puts it under `ImportModel`. Checked before the
    // refusal so an unauthorised caller is told the same thing here as
    // anywhere else.
    require_permission(&session, Permission::ImportModel)?;
    Err(format!(
        "The measured benchmark is not wired up, so no figure is reported for {model_id}. \
         Nothing was recorded: a number produced without asking the model would be \
         indistinguishable from one that was measured."
    ))
}

/// Reads the most recent measured rows, newest first.
#[tauri::command]
pub async fn recent_benchmarks(
    app_data_dir: State<'_, PathBuf>,
    session: State<'_, CurrentSession>,
    limit: Option<usize>,
) -> Result<Vec<BenchmarkRow>, String> {
    require_session(&session)?;
    let rows = benchmarks::recent(&app_data_dir, limit.unwrap_or(10).min(64))
        .map_err(|e| e.to_string())?;
    Ok(rows.into_iter().map(BenchmarkRow::from).collect())
}

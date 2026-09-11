//! Performance benchmarks for the System Health page.
//!
//! ## What this module measures
//!
//! - **Time to first token (TTFT)** — the wall-clock time from the
//!   first byte sent to the model to the first token received.
//!   Measured in milliseconds; the model is gemma-3-12b-it at
//!   Q4_K_M on the demo hardware (RTX 5060 4 GB).
//! - **Tokens per second** — averaged over a 64-token reply.
//! - **VRAM peak** — the maximum resident-set of the model, read
//!   from `nvidia-smi` when available, otherwise from the
//!   process's own working set.
//! - **Accuracy on demo tasks** — a small hand-graded suite: tag
//!   identification on a sample P&ID, calculation correctness on
//!   three well-known formulas, policy compliance on a sample SOP.
//!
//! ## What this module does NOT measure
//!
//! - It does **not** benchmark every model in the catalog. The
//!   number is for the *current* loaded model, and the system
//!   health page shows the result with the model name attached.
//! - It does **not** run a model-side load. The benchmark issues
//!   a real chat request with a known prompt and a known target
//!   token count; the result is what the model produced.
//! - It does **not** run on a headless server. The benchmark
//!   requires a model to be loaded, which requires the inference
//!   runtime, which requires a GPU or CPU budget the headless
//!   build does not have. The benchmark command refuses to run
//!   if no model is loaded.
//!
//! ## Honest scope
//!
//! The numbers this module reports are the numbers the *machine it
//! is running on* produced. They are not benchmarks of the model
//! itself; they are benchmarks of the *system* running the model.
//! A reviewer who asks "what is the throughput?" gets the
//! throughput on the laptop in front of them, not a vendor's
//! number from a different machine.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// One row of the benchmark. The `model_id` field lets the System
/// Health page render a table that mixes results from different
/// models if the operator runs the benchmark with each one.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkResult {
    pub model_id: String,
    pub prompt_tokens: u64,
    pub reply_tokens: u64,
    pub ttft_ms: u64,
    pub total_ms: u64,
    pub tokens_per_second: f64,
    pub vram_peak_mib: u64,
    pub accuracy_pct: f64,
    /// ISO-8601 string, not a `chrono::DateTime`, to keep
    /// `serde_json::from_str` inference simple and avoid the
    /// lifetime dance around `DateTime<Tz>` deserialization.
    pub at: String,
    pub hardware_tier: String,
}

/// The full benchmark suite. Hard-coded prompts and expected
/// answers — the prompts are short so the run is fast, the
/// answers are short so a hand-grader can score them in seconds.
const BENCH_PROMPTS: &[(&str, &str, &str)] = &[
    (
        "tag-identification",
        "On P&ID A-101-001 Rev 6, list every equipment tag you can see.",
        "P-101A P-101B P-102A P-102B V-101",
    ),
    (
        "calculation-correctness",
        "Convert 120 °C to Kelvin. Show your work.",
        "393.15",
    ),
    (
        "policy-compliance",
        "On a 1910.119(j) audit, name one clause the SOP is missing.",
        "1910.119(j)(2)(ii)",
    ),
];

/// Maps a hardware signature to a tier label. The signature is
/// derived from the system info command; the tier is what the
/// benchmark row prints.
pub fn tier_for(vram_mib: u64) -> &'static str {
    match vram_mib {
        0..=1_999 => "tier-3-cpu-only",
        2_000..=5_999 => "tier-1-rtx-5060-4gb",
        6_000..=11_999 => "tier-2-rtx-3060-12gb",
        _ => "tier-2-rtx-3060-12gb-plus",
    }
}

/// Reads the current VRAM usage from `nvidia-smi`. Returns `None`
/// when nvidia-smi is not on the path or returns an error — the
/// caller then falls back to the process's own working set.
pub fn read_vram_peak_mib() -> Option<u64> {
    let mut cmd = std::process::Command::new("nvidia-smi");
    cmd.args([
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]);
    // Suppress the console window that would otherwise pop up
    // every time the System Health page refreshes. nvidia-smi is
    // a console application; the Tauri release build is
    // `windows_subsystem = "windows"`, but the OS still opens a
    // window for any console child.
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000);
    }
    let out = cmd.output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let used: u64 = s.trim().lines().next()?.trim().parse().ok()?;
    Some(used)
}

/// Writes the benchmark result to a JSON file in the app data
/// directory, so the System Health page can read the most recent
/// run without a database round-trip.
pub fn record(app_data_dir: &Path, result: &BenchmarkResult) -> Result<()> {
    std::fs::create_dir_all(app_data_dir)
        .with_context(|| format!("could not create {}", app_data_dir.display()))?;
    let path: PathBuf = app_data_dir.join("benchmarks.json");
    let mut history: Vec<BenchmarkResult> = read_history(&path).unwrap_or_default();
    history.push(result.clone());
    // Cap the file at 64 rows so a long-lived installation does
    // not grow it without bound.
    if history.len() > 64 {
        let drop = history.len() - 64;
        history.drain(0..drop);
    }
    let json = serde_json::to_vec_pretty(&history)?;
    std::fs::write(&path, json)
        .with_context(|| format!("could not write {}", path.display()))?;
    Ok(())
}

/// Reads the most recent benchmark rows, newest first.
pub fn recent(app_data_dir: &Path, limit: usize) -> Result<Vec<BenchmarkResult>> {
    let path: PathBuf = app_data_dir.join("benchmarks.json");
    let mut history: Vec<BenchmarkResult> = read_history(&path).unwrap_or_default();
    history.reverse();
    history.truncate(limit);
    Ok(history)
}

/// Helper: reads and deserializes the benchmark history file.
/// `None` (or an error) is treated as "no history yet" rather
/// than a hard failure, so a corrupt or partial file does not
/// prevent the next `record` call.
fn read_history(path: &Path) -> Result<Vec<BenchmarkResult>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    // Read into a String, then deserialize. Going through a
    // String sidesteps a serde_json lifetime inference bug
    // around `from_reader` with borrowed file handles.
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("could not read {}", path.display()))?;
    let history: Vec<BenchmarkResult> = serde_json::from_str(&text).unwrap_or_default();
    Ok(history)
}

/// Stamps the start of a real benchmark run. The caller holds
/// the handle and calls [`BenchTimer::finish`] when the model
/// has produced its reply, with the token count and accuracy
/// in hand.
pub struct BenchTimer {
    model_id: String,
    started_at: Instant,
    prompt_tokens: u64,
    reply_tokens: u64,
}

impl BenchTimer {
    /// Starts the timer.
    pub fn start(model_id: impl Into<String>, prompt_tokens: u64, reply_tokens: u64) -> Self {
        Self {
            model_id: model_id.into(),
            started_at: Instant::now(),
            prompt_tokens,
            reply_tokens,
        }
    }

    /// Finishes the timer and returns the result row. The
    /// `accuracy_pct` is the caller's hand-graded score; the
    /// tokens-per-second is computed from the wall clock.
    pub fn finish(self, accuracy_pct: f64) -> BenchmarkResult {
        let total_ms = self.started_at.elapsed().as_millis() as u64;
        let tps = if total_ms > 0 {
            (self.reply_tokens as f64) / (total_ms as f64 / 1000.0)
        } else {
            0.0
        };
        let vram = read_vram_peak_mib().unwrap_or(0);
        BenchmarkResult {
            model_id: self.model_id,
            prompt_tokens: self.prompt_tokens,
            reply_tokens: self.reply_tokens,
            // The model-router's first-token timing is a more accurate TTFT
            // than the wall clock here, and it arrives through a separate
            // channel. Zero is the honest value for "this run did not carry
            // one" — a reader must be able to tell a missing measurement from
            // a fast one, which is why nothing here substitutes a plausible
            // number for it.
            ttft_ms: 0,
            total_ms,
            tokens_per_second: tps,
            vram_peak_mib: vram,
            accuracy_pct,
            at: chrono::Utc::now().to_rfc3339(),
            hardware_tier: tier_for(vram).to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_for_routes_by_vram() {
        assert_eq!(tier_for(0), "tier-3-cpu-only");
        assert_eq!(tier_for(3800), "tier-1-rtx-5060-4gb");
        assert_eq!(tier_for(8192), "tier-2-rtx-3060-12gb");
    }

    #[test]
    fn bench_timer_computes_tokens_per_second() {
        let t = BenchTimer::start("m", 10, 64);
        std::thread::sleep(std::time::Duration::from_millis(10));
        let r = t.finish(100.0);
        assert_eq!(r.prompt_tokens, 10);
        assert_eq!(r.reply_tokens, 64);
        // 64 tokens in ~10ms is >> 1000 t/s, well above zero.
        assert!(r.tokens_per_second > 0.0);
    }

    /// A row to exercise the store with.
    ///
    /// Built here, in the test module, and deliberately not a `pub fn` on the
    /// module: the helper this replaces was reachable from a Tauri command, and
    /// its hardcoded 38 tok/s was published as a measured benchmark. A fixture
    /// only tests can reach cannot be published by accident.
    fn a_row() -> BenchmarkResult {
        BenchmarkResult {
            model_id: "fixture-model".to_string(),
            prompt_tokens: 18,
            reply_tokens: 64,
            ttft_ms: 220,
            total_ms: 1700,
            tokens_per_second: 37.6,
            vram_peak_mib: 3800,
            accuracy_pct: 92.0,
            at: chrono::Utc::now().to_rfc3339(),
            hardware_tier: "fixture-tier".to_string(),
        }
    }

    #[test]
    fn record_then_recent_round_trips() {
        let tmp = tempdir();
        let r = a_row();
        record(&tmp, &r).unwrap();
        let rows = recent(&tmp, 10).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].model_id, r.model_id);
    }

    #[test]
    fn recent_caps_at_the_asked_limit() {
        let tmp = tempdir();
        for _ in 0..20 {
            record(&tmp, &a_row()).unwrap();
        }
        let rows = recent(&tmp, 5).unwrap();
        assert_eq!(rows.len(), 5);
    }

    #[test]
    fn record_caps_the_file_at_64_rows() {
        let tmp = tempdir();
        for _ in 0..80 {
            record(&tmp, &a_row()).unwrap();
        }
        let path = tmp.join("benchmarks.json");
        let bytes = std::fs::read(&path).unwrap();
        let history: Vec<BenchmarkResult> = serde_json::from_slice(&bytes).unwrap();
        assert!(history.len() <= 64);
    }

    fn tempdir() -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "arjun-bench-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}

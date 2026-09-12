//! Knowing whether a long piece of work is *working* or *stuck*.
//!
//! ## The problem with a stopwatch
//!
//! Every bound on generation in this product used to be wall-clock: the
//! artifact tools had ten to a hundred and twenty seconds each, and when the
//! time was up the call was killed. That is the right answer for a tool that
//! hangs and the wrong one for a tool that is busy — and the two are
//! indistinguishable from outside, which is why a forty-page report and a
//! two-row table were given the same budget and only the report lost.
//!
//! A stopwatch also cannot be tuned. Raise it and a genuinely wedged call holds
//! the run for that much longer; lower it and real work dies. The number is
//! wrong in both directions at once.
//!
//! ## What replaces it
//!
//! Progress. A phase that is *advancing* — bytes written, blocks rendered,
//! checks completed, tokens produced — is not stalled, however long it has been
//! going. A phase that has reported nothing for its own stall window is stuck,
//! however recently it started.
//!
//! Two things follow that a stopwatch could not give:
//!
//! - **Different phases get different windows.** Rendering emits bytes
//!   continuously and should be declared stuck quickly. Composing waits on a
//!   model that may think for a minute before its first token, so it gets far
//!   longer before silence means anything.
//! - **Budgets scale with the work.** The size of the content is known before
//!   rendering starts, so a hundred-section document is allowed proportionally
//!   longer than a one-section note — as a ceiling on total time, not as
//!   permission to go silent.
//!
//! ## What this does not replace
//!
//! The run deadline. [`crate::orchestrator::plan::Budget::max_duration`] is
//! still thirty minutes and is still absolute: a task that reports progress
//! forever is a task nobody is waiting for any more. This module decides when
//! silence means stuck; the deadline decides when *enough* is enough.

use std::time::{Duration, Instant};

/// What a generation is doing.
///
/// The phases are distinct because their silence means different things, not
/// for reporting. A renderer that has emitted nothing for ten seconds has
/// stopped; a model that has emitted nothing for ten seconds is thinking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    /// Asking the model for the content. Silence here is normal and can be
    /// long: reasoning happens before the first token.
    Composing,
    /// Writing the file. Bytes should appear steadily.
    Rendering,
    /// Re-opening and checking it. Bounded work over a file already on disk.
    Validating,
    /// Correcting the model in memory. Pure computation, and fast.
    Repairing,
    /// Writing it again after a failed check. Rendering by another name, and
    /// given the same window.
    Regenerating,
}

impl Phase {
    pub const ALL: &'static [Phase] = &[
        Phase::Composing,
        Phase::Rendering,
        Phase::Validating,
        Phase::Repairing,
        Phase::Regenerating,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Phase::Composing => "composing",
            Phase::Rendering => "rendering",
            Phase::Validating => "validating",
            Phase::Repairing => "repairing",
            Phase::Regenerating => "regenerating",
        }
    }

    /// How long this phase may report nothing before it is treated as stuck.
    ///
    /// These are silence windows, not time limits. A phase reporting progress
    /// every few seconds runs as long as the work takes.
    pub const fn stall_window(self) -> Duration {
        match self {
            // A local model on a small GPU can spend well over a minute on the
            // prompt before the first token. Four minutes matches the runtime's
            // own stall guard, which is the number observed not to fire during
            // real thinking.
            Phase::Composing => Duration::from_secs(4 * 60),
            // A writer emits continuously. Thirty seconds of nothing is a
            // writer that has stopped, not one that is busy.
            Phase::Rendering | Phase::Regenerating => Duration::from_secs(30),
            // Reading a file back is bounded work; a minute is generous for the
            // largest artifact this product produces.
            Phase::Validating => Duration::from_secs(60),
            // In-memory list manipulation. Ten seconds is already absurd.
            Phase::Repairing => Duration::from_secs(10),
        }
    }
}

/// A unit of progress. Named so a heartbeat says what moved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Advance {
    /// Model output.
    Tokens(u64),
    /// Bytes written to the artifact.
    Bytes(u64),
    /// Blocks, rows, slides or pages rendered.
    Blocks(u64),
    /// Validation steps completed.
    Checks(u64),
}

impl Advance {
    fn amount(self) -> u64 {
        match self {
            Advance::Tokens(n) | Advance::Bytes(n) | Advance::Blocks(n) | Advance::Checks(n) => n,
        }
    }
}

/// The total-time ceiling for a piece of work, scaled by how big it is.
///
/// Separate from the stall window: this bounds a task that is progressing but
/// will plainly never finish, where the window bounds one that has stopped.
///
/// Linear in the content, because rendering is: twice the sections is twice the
/// XML. Clamped at both ends — a one-line note still gets a workable floor, and
/// nothing is allowed near the run deadline, because a single tool consuming
/// the whole run leaves nothing for the answer.
pub fn budget_for(units: u64) -> Duration {
    /// Enough for the smallest real artifact on the slowest machine this runs
    /// on.
    const FLOOR: u64 = 120;
    /// A second per unit of content.
    const PER_UNIT: u64 = 1;
    /// A third of the thirty-minute run deadline. A tool that wants more than
    /// this is a task that should have been split.
    const CEILING: u64 = 10 * 60;

    Duration::from_secs((FLOOR.saturating_add(units.saturating_mul(PER_UNIT))).clamp(FLOOR, CEILING))
}

/// Tracks one piece of generation.
///
/// Deliberately synchronous and not clock-injected beyond `Instant::now()`: the
/// callers are synchronous renderers, and a tracker that needed an async
/// runtime to be useful would not be usable from inside a writer.
#[derive(Debug)]
pub struct Tracker {
    phase: Phase,
    phase_started: Instant,
    last_advance: Instant,
    started: Instant,
    budget: Duration,
    /// What has moved, by kind, for the record.
    tokens: u64,
    bytes: u64,
    blocks: u64,
    checks: u64,
}

/// Why a tracker says to stop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stop {
    /// Nothing has moved for this phase's window.
    Stalled { phase: Phase, silent_for: Duration },
    /// Progressing, but past the total ceiling for work this size.
    OverBudget { elapsed: Duration, budget: Duration },
}

impl Stop {
    pub fn explain(&self) -> String {
        match self {
            Stop::Stalled { phase, silent_for } => format!(
                "Stopped: {} reported nothing for {} seconds, which is past the {} seconds a \
                 {} phase is allowed to be silent. It is stuck rather than slow.",
                phase.name(),
                silent_for.as_secs(),
                phase.stall_window().as_secs(),
                phase.name()
            ),
            Stop::OverBudget { elapsed, budget } => format!(
                "Stopped: still making progress after {} seconds, against a budget of {} for \
                 work of this size. Producing this needs to be split into smaller pieces.",
                elapsed.as_secs(),
                budget.as_secs()
            ),
        }
    }
}

impl Tracker {
    /// `units` is how much content there is — sections, rows, slides. It sets
    /// the total-time ceiling and nothing else.
    pub fn new(phase: Phase, units: u64) -> Self {
        let now = Instant::now();
        Self {
            phase,
            phase_started: now,
            last_advance: now,
            started: now,
            budget: budget_for(units),
            tokens: 0,
            bytes: 0,
            blocks: 0,
            checks: 0,
        }
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }

    pub fn budget(&self) -> Duration {
        self.budget
    }

    /// Moves to a new phase. The silence clock restarts, because a phase that
    /// has just begun has not been silent.
    pub fn enter(&mut self, phase: Phase) {
        self.phase = phase;
        let now = Instant::now();
        self.phase_started = now;
        self.last_advance = now;
    }

    /// Records that something moved. This is the heartbeat.
    pub fn advance(&mut self, advance: Advance) {
        let amount = advance.amount();
        match advance {
            Advance::Tokens(_) => self.tokens += amount,
            Advance::Bytes(_) => self.bytes += amount,
            Advance::Blocks(_) => self.blocks += amount,
            Advance::Checks(_) => self.checks += amount,
        }
        // Even a zero-sized advance is a heartbeat: a renderer reporting "I am
        // on block 4" having written nothing since block 3 is still alive.
        self.last_advance = Instant::now();
    }

    /// How long since anything moved.
    pub fn silent_for(&self) -> Duration {
        self.last_advance.elapsed()
    }

    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    pub fn phase_elapsed(&self) -> Duration {
        self.phase_started.elapsed()
    }

    /// Whether to stop, and why. `None` means carry on.
    ///
    /// Silence is checked first: a task that has stopped should be reported as
    /// stuck rather than as over budget, because the two call for different
    /// actions from whoever reads it.
    pub fn should_stop(&self) -> Option<Stop> {
        let silent = self.silent_for();
        if silent >= self.phase.stall_window() {
            return Some(Stop::Stalled { phase: self.phase, silent_for: silent });
        }
        let elapsed = self.elapsed();
        if elapsed >= self.budget {
            return Some(Stop::OverBudget { elapsed, budget: self.budget });
        }
        None
    }

    /// One line for the log, so a slow run can be watched rather than guessed
    /// at.
    pub fn heartbeat(&self) -> String {
        format!(
            "phase={} elapsed={}s silent={}s tokens={} bytes={} blocks={} checks={} budget={}s",
            self.phase.name(),
            self.elapsed().as_secs(),
            self.silent_for().as_secs(),
            self.tokens,
            self.bytes,
            self.blocks,
            self.checks,
            self.budget.as_secs()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property the whole module exists for.
    #[test]
    fn a_task_that_keeps_moving_is_never_stopped_for_being_slow() {
        let mut tracker = Tracker::new(Phase::Rendering, 10_000);
        // Far more work than any of the old wall-clock ceilings allowed: each
        // advance resets the silence clock, so none of it is a stall.
        for block in 0..1_000 {
            tracker.advance(Advance::Blocks(1));
            assert!(
                tracker.should_stop().is_none(),
                "stopped at block {block}: {:?}",
                tracker.should_stop()
            );
        }
        assert_eq!(tracker.blocks, 1_000);
    }

    /// And the property that keeps it honest.
    #[test]
    fn silence_past_the_phase_window_is_a_stall() {
        let mut tracker = Tracker::new(Phase::Repairing, 1);
        // Repairing allows ten seconds. Reach back past it.
        tracker.last_advance = Instant::now() - Duration::from_secs(11);

        match tracker.should_stop() {
            Some(Stop::Stalled { phase, .. }) => assert_eq!(phase, Phase::Repairing),
            other => panic!("expected a stall, got {other:?}"),
        }
    }

    /// A model thinking is not a model stuck. The same eleven seconds of
    /// silence that stalls a repair is nothing at all while composing.
    #[test]
    fn the_same_silence_means_different_things_in_different_phases() {
        let mut composing = Tracker::new(Phase::Composing, 1);
        composing.last_advance = Instant::now() - Duration::from_secs(11);
        assert!(
            composing.should_stop().is_none(),
            "a model may be quiet for eleven seconds without being stuck"
        );

        let mut rendering = Tracker::new(Phase::Rendering, 1);
        rendering.last_advance = Instant::now() - Duration::from_secs(31);
        assert!(rendering.should_stop().is_some(), "a writer may not");
    }

    #[test]
    fn entering_a_phase_restarts_the_silence_clock() {
        let mut tracker = Tracker::new(Phase::Repairing, 1);
        tracker.last_advance = Instant::now() - Duration::from_secs(11);
        assert!(tracker.should_stop().is_some());

        tracker.enter(Phase::Validating);
        assert!(
            tracker.should_stop().is_none(),
            "a phase that has just begun has not been silent"
        );
        assert_eq!(tracker.phase(), Phase::Validating);
    }

    /// Large work gets a larger budget. This is the half a fixed ceiling could
    /// never express.
    #[test]
    fn a_bigger_artifact_is_given_longer() {
        let small = budget_for(1);
        let large = budget_for(600);
        assert!(large > small, "{large:?} is not longer than {small:?}");
    }

    #[test]
    fn the_smallest_work_still_gets_a_workable_floor() {
        assert!(budget_for(0) >= Duration::from_secs(120));
    }

    /// No single tool may consume the run.
    #[test]
    fn no_amount_of_content_reaches_the_run_deadline() {
        let deadline = Duration::from_secs(30 * 60);
        assert!(budget_for(u64::MAX) < deadline);
        assert!(budget_for(1_000_000) < deadline);
    }

    /// Progressing forever is still bounded — by the budget, and the message
    /// says which of the two reasons it was.
    #[test]
    fn progress_that_never_finishes_is_stopped_by_the_budget_not_the_window() {
        let mut tracker = Tracker::new(Phase::Rendering, 1);
        // Advancing right now, so it is not silent; but started long ago.
        tracker.advance(Advance::Bytes(1));
        tracker.started = Instant::now() - Duration::from_secs(10_000);

        match tracker.should_stop() {
            Some(Stop::OverBudget { budget, .. }) => assert_eq!(budget, budget_for(1)),
            other => panic!("expected an over-budget stop, got {other:?}"),
        }
    }

    #[test]
    fn every_phase_has_a_window_and_a_name() {
        for phase in Phase::ALL {
            assert!(!phase.name().is_empty());
            assert!(phase.stall_window() > Duration::ZERO, "{phase:?}");
        }
    }

    #[test]
    fn the_heartbeat_says_what_moved() {
        let mut tracker = Tracker::new(Phase::Validating, 4);
        tracker.advance(Advance::Checks(3));
        let line = tracker.heartbeat();
        assert!(line.contains("phase=validating"), "{line}");
        assert!(line.contains("checks=3"), "{line}");
    }

    #[test]
    fn a_stall_explains_itself_in_words_somebody_can_act_on() {
        let stop = Stop::Stalled { phase: Phase::Rendering, silent_for: Duration::from_secs(45) };
        let text = stop.explain();
        assert!(text.contains("rendering"), "{text}");
        assert!(text.contains("stuck rather than slow"), "{text}");
    }
}

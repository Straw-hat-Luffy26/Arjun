//! Stopping a turn, at any point in it.
//!
//! ## What Stop used to reach, and what it did not
//!
//! `agent_abort_run` did exactly one thing: send `run.abort` to the agent
//! runtime. That is the *last* stage of a turn. Everything before it — reading
//! attachments, running the OCR model over forty pages, routing, admitting a
//! model to VRAM, loading weights, spawning the runtime — happens before the
//! runtime has heard of the run, and during all of it Stop reached nothing.
//!
//! Those are not the fast stages. A scanned drawing set is minutes of vision
//! model on the operator's own GPU, and it is precisely the stage somebody
//! presses Stop during. The button reported success, the record showed a
//! cancellation, and the machine carried on reading the document.
//!
//! ## Two ids, one token
//!
//! A turn is addressed by two identities in its lifetime: the correlation id
//! the composer mints before the request goes out, and the run id the core
//! mints once the turn is accepted. The surface holds the first until
//! `plan_ready` teaches it the second.
//!
//! So a token is registered under *both*, and either one cancels it. That is
//! what makes Stop work during preparation — the only window in which the
//! surface has nothing but a correlation id, and also the window in which the
//! expensive stages run.
//!
//! ## The startup race
//!
//! Stop can arrive before the run has registered anything: the composer sends
//! the request, the person presses Stop, and the command that would register
//! the token has not reached that line yet. Cancelling an id nobody knows is
//! therefore *remembered* rather than discarded, and a registration that finds
//! its id already named comes back cancelled. A run cannot start by outrunning
//! the person who stopped it.
//!
//! ## Why a `Notify` and not only a flag
//!
//! A flag can only be *checked*, and a stage that is blocked cannot check
//! anything. The OCR read waits on a socket for tens of seconds before the
//! first token of a page arrives; a boolean tested at the top of the loop is
//! read once and then not again until the model has already done the work.
//! [`CancelToken::cancelled`] is a future, so a waiting stage can `select!` on
//! it and be interrupted rather than polled.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::Notify;

/// How many ids cancelled-before-registration are remembered.
///
/// Each entry is a small string held until a run claims it or the process
/// ends, and the set only grows when somebody stops a turn that never started.
/// Bounded anyway: an unbounded set fed by a caller that can name ids is a slow
/// leak with a name on it.
const MAX_REMEMBERED_CANCELLATIONS: usize = 256;

/// One turn's stop signal.
///
/// Cloneable and cheap: every stage of a turn holds the same one, and
/// cancelling it once stops all of them.
#[derive(Clone)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
    notify: Arc<Notify>,
}

impl Default for CancelToken {
    /// A token nothing will ever cancel. See [`CancelToken::never`].
    ///
    /// Spelled out rather than derived so the default is a *decision* somebody
    /// wrote down: an uncancelled token is the right default, and a derive
    /// would leave that looking accidental.
    fn default() -> Self {
        Self::never()
    }
}

impl CancelToken {
    /// A token nothing will ever cancel.
    ///
    /// For callers with no run to attach to — a scan started from the scan
    /// screen, a test exercising a stage in isolation. Named rather than
    /// spelled `AtomicBool::new(false)` at each call site, because that is
    /// exactly the shape the defect had: an unreachable flag that reads as
    /// working cancellation.
    pub fn never() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
            notify: Arc::new(Notify::new()),
        }
    }

    /// A token that is already cancelled.
    pub fn cancelled_now() -> Self {
        let token = Self::never();
        token.cancel();
        token
    }

    /// Whether somebody has asked for this turn to stop.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }

    /// Asks for this turn to stop, and wakes everything waiting on it.
    ///
    /// Idempotent: a second Stop on a turn already stopping is not an error,
    /// and `notify_waiters` on a token with nothing waiting is a no-op.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
        self.notify.notify_waiters();
    }

    /// Resolves when this turn is cancelled, and never otherwise.
    ///
    /// The point of the whole module. A stage blocked on a socket cannot test a
    /// boolean; it can `select!` on this.
    ///
    /// The flag is re-checked *after* registering interest and again after
    /// waking, which is what closes the race where cancellation lands between
    /// the caller's own check and its `await`. `Notify::notified()` only wakes
    /// for a `notify_waiters` that happens after the future is created, so
    /// without the first check a token cancelled a microsecond earlier would
    /// leave the caller waiting for a signal that had already been sent.
    pub async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }
        let waiting = self.notify.notified();
        if self.is_cancelled() {
            return;
        }
        waiting.await;
    }

    /// `Err` with a sentence a person reads, when this turn has been stopped.
    ///
    /// The wording is deliberately the same everywhere: a turn stopped during
    /// OCR and a turn stopped during model loading are the same event to the
    /// person who pressed the button, and two different sentences would invite
    /// them to wonder which stage mattered.
    pub fn check(&self) -> Result<(), String> {
        if self.is_cancelled() {
            return Err(STOPPED.to_string());
        }
        Ok(())
    }
}

/// What a stopped turn says, wherever it was stopped.
pub const STOPPED: &str = "Stopped, because somebody stopped it.";

/// Every turn that can currently be stopped, and every turn stopped too early.
#[derive(Default)]
pub struct RunCancellations {
    inner: Mutex<State>,
}

#[derive(Default)]
struct State {
    /// Id to token. One token is reachable under several ids.
    live: HashMap<String, CancelToken>,
    /// Ids cancelled before anything registered them. See the module header.
    remembered: HashSet<String>,
}

impl RunCancellations {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a turn under every id that can name it, and hands back its
    /// token.
    ///
    /// Comes back **already cancelled** when any of the ids was stopped before
    /// this call — the startup race. The caller is expected to check the token
    /// before doing any work, which it must do anyway.
    ///
    /// A poisoned lock yields a token nobody can cancel rather than a panic: a
    /// turn that cannot be stopped is bad, and a turn that cannot start because
    /// an unrelated thread panicked while holding a map is worse.
    pub fn register(&self, ids: &[&str]) -> CancelToken {
        let token = CancelToken::never();
        let Ok(mut state) = self.inner.lock() else {
            log::error!("[cancel] the cancellation table is poisoned; this turn cannot be stopped");
            return token;
        };
        let mut already = false;
        for id in ids {
            if id.is_empty() {
                continue;
            }
            if state.remembered.remove(*id) {
                already = true;
            }
            state.live.insert((*id).to_string(), token.clone());
        }
        drop(state);
        if already {
            token.cancel();
        }
        token
    }

    /// Adds one more id for an existing token.
    ///
    /// Used when the run mints its own id partway through: the turn is already
    /// registered under the correlation id, and this makes the new id reach the
    /// same token rather than a second one.
    pub fn also_known_as(&self, id: &str, token: &CancelToken) {
        if id.is_empty() {
            return;
        }
        let Ok(mut state) = self.inner.lock() else {
            return;
        };
        if state.remembered.remove(id) {
            drop(state);
            token.cancel();
            return;
        }
        state.live.insert(id.to_string(), token.clone());
    }

    /// Asks the turn named by `id` to stop.
    ///
    /// Returns whether a live turn was found. `false` means the id named
    /// nothing registered — either the turn has already finished, or it has not
    /// reached its registration yet. The second case is why the id is
    /// remembered: see the module header.
    pub fn cancel(&self, id: &str) -> bool {
        let Ok(mut state) = self.inner.lock() else {
            return false;
        };
        if let Some(token) = state.live.get(id).cloned() {
            drop(state);
            token.cancel();
            return true;
        }
        // Nothing registered under this id. Remembered so a registration that
        // arrives a moment later comes back cancelled.
        if state.remembered.len() >= MAX_REMEMBERED_CANCELLATIONS {
            // Evicting the oldest would need an order this does not keep;
            // refusing the new one is the honest alternative, and it only
            // happens after 256 stops that never matched a run.
            log::warn!(
                "[cancel] too many cancellations for turns that never started; {id} is not remembered"
            );
            return false;
        }
        state.remembered.insert(id.to_string());
        false
    }

    /// Whether this id names a turn that has been asked to stop.
    pub fn is_cancelled(&self, id: &str) -> bool {
        let Ok(state) = self.inner.lock() else {
            return false;
        };
        state
            .live
            .get(id)
            .map(CancelToken::is_cancelled)
            .unwrap_or_else(|| state.remembered.contains(id))
    }

    /// Drops every id a finished turn was reachable under.
    ///
    /// Called when the turn ends, however it ends. A table that only grew would
    /// hold a token per turn for the life of the process, and — worse — a later
    /// turn that happened to reuse an id would find a stale cancelled token.
    pub fn forget(&self, ids: &[&str]) {
        let Ok(mut state) = self.inner.lock() else {
            return;
        };
        for id in ids {
            state.live.remove(*id);
            state.remembered.remove(*id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_token_is_not_cancelled() {
        assert!(!CancelToken::never().is_cancelled());
        assert!(CancelToken::never().check().is_ok());
    }

    #[test]
    fn cancelling_is_visible_and_says_so() {
        let token = CancelToken::never();
        token.cancel();
        assert!(token.is_cancelled());
        assert_eq!(token.check().unwrap_err(), STOPPED);
    }

    /// Every stage holds the same token, so one Stop reaches all of them.
    #[test]
    fn a_clone_shares_the_signal() {
        let token = CancelToken::never();
        let held_by_a_stage = token.clone();
        token.cancel();
        assert!(held_by_a_stage.is_cancelled());
    }

    /// The whole point of two ids: the surface holds the correlation id for the
    /// entire preparation stage, which is the expensive one.
    #[test]
    fn either_id_stops_the_same_turn() {
        let table = RunCancellations::new();
        let token = table.register(&["correlation-1", "run-1"]);

        assert!(table.cancel("correlation-1"));
        assert!(token.is_cancelled());

        let second = table.register(&["correlation-2", "run-2"]);
        assert!(table.cancel("run-2"));
        assert!(second.is_cancelled());
    }

    /// The startup race. A turn must not be able to start by outrunning the
    /// person who stopped it.
    #[test]
    fn a_stop_that_arrives_before_the_turn_registers_still_lands() {
        let table = RunCancellations::new();

        // Nothing registered yet, so there is nothing to find.
        assert!(!table.cancel("correlation-1"));

        // …and the turn that registers a moment later is already stopped.
        let token = table.register(&["correlation-1", "run-1"]);
        assert!(token.is_cancelled());
    }

    #[test]
    fn a_remembered_cancellation_is_consumed_by_the_turn_it_named() {
        let table = RunCancellations::new();
        table.cancel("correlation-1");
        assert!(table.register(&["correlation-1"]).is_cancelled());
        // A *second* turn under the same id is not pre-cancelled: the first one
        // consumed it.
        assert!(!table.register(&["correlation-1"]).is_cancelled());
    }

    #[test]
    fn a_run_id_learned_later_reaches_the_same_token() {
        let table = RunCancellations::new();
        let token = table.register(&["correlation-1"]);
        table.also_known_as("run-1", &token);

        assert!(table.cancel("run-1"));
        assert!(token.is_cancelled());
    }

    /// The same race, one stage later: the run mints its id and only then
    /// learns somebody had already stopped it under that id.
    #[test]
    fn learning_an_id_that_was_already_stopped_cancels_immediately() {
        let table = RunCancellations::new();
        let token = table.register(&["correlation-1"]);
        table.cancel("run-1");

        table.also_known_as("run-1", &token);
        assert!(token.is_cancelled());
    }

    #[test]
    fn forgetting_a_turn_leaves_nothing_behind() {
        let table = RunCancellations::new();
        let token = table.register(&["correlation-1", "run-1"]);
        drop(token);
        table.forget(&["correlation-1", "run-1"]);

        // A later turn reusing an id must not inherit the earlier one's state.
        assert!(!table.cancel("run-1"), "the id names nothing now");
        assert!(!table.register(&["correlation-1"]).is_cancelled());
    }

    #[test]
    fn an_empty_id_registers_nothing() {
        let table = RunCancellations::new();
        let token = table.register(&["", "run-1"]);
        assert!(!table.cancel(""), "an empty id must not name a turn");
        assert!(!token.is_cancelled());
    }

    #[test]
    fn the_remembered_set_is_bounded() {
        let table = RunCancellations::new();
        for i in 0..(MAX_REMEMBERED_CANCELLATIONS + 50) {
            table.cancel(&format!("never-started-{i}"));
        }
        let state = table.inner.lock().expect("lock");
        assert!(state.remembered.len() <= MAX_REMEMBERED_CANCELLATIONS);
    }

    /// A stage blocked on a socket cannot test a boolean. This is the future it
    /// waits on instead.
    #[tokio::test]
    async fn waiting_on_a_token_wakes_when_it_is_cancelled() {
        let token = CancelToken::never();
        let waiting = token.clone();
        let handle = tokio::spawn(async move {
            waiting.cancelled().await;
            true
        });
        token.cancel();
        assert!(handle.await.expect("the waiter finished"));
    }

    /// The race the double check exists for: cancellation lands between a
    /// caller's own check and its `await`. Without the check after registering
    /// interest, this waits for a signal that has already been sent.
    #[tokio::test]
    async fn waiting_on_an_already_cancelled_token_returns_at_once() {
        let token = CancelToken::cancelled_now();
        // Would hang rather than fail if the check were missing, so it is
        // bounded: the timeout elapsing *is* the failure.
        tokio::time::timeout(std::time::Duration::from_secs(2), token.cancelled())
            .await
            .expect("a cancelled token must not make a waiter wait");
    }
}

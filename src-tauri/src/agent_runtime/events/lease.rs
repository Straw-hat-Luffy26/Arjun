//! Who is allowed to advance a run, and for how long.
//!
//! ## The failure this exists for
//!
//! Until resumption could actually execute, only one thing ever drove a run:
//! the command that started it. That is no longer true. `agent_resume_run` now
//! drives one too, and startup recovery will. So there are three ways for two
//! things to be working the same run at once:
//!
//! - an operator resumes a run the previous process is *still* running, because
//!   the application was force-quit and the old process outlived the window;
//! - two windows both offer the recovery flow and both are clicked;
//! - startup recovery picks up a run an operator is resuming by hand.
//!
//! Every one of them ends the same way: two loops appending to one event
//! stream, two sets of tool calls, and a side effect performed twice under
//! different idempotency keys because each attempt derives its own.
//!
//! The event log's `UNIQUE (run_id, seq)` stops the *history* being corrupted
//! and does nothing about the *work* being done twice, which is the part that
//! writes files.
//!
//! ## Why a lease and not a mutex
//!
//! A mutex would be enough if the contenders were threads. They are not: the
//! contender this must defend against is a **process that no longer exists**,
//! and a lock held by a dead process is held forever. So ownership expires, and
//! a holder that wants to keep it says so periodically.
//!
//! ## Why a fencing token
//!
//! Expiry alone is not sufficient and the reason is the classic one. A holder
//! can stall — a long GPU load, a swapped-out process — past its own expiry,
//! have the lease taken by somebody else, then wake up and carry on believing
//! it still holds it. It cannot tell that time passed.
//!
//! So every acquisition increments a token, and a holder presents its token
//! with anything it does. The woken-up straggler presents an old one and is
//! refused, without needing to have noticed anything. The token is monotonic
//! per run and never reused, which is the whole property.
//!
//! ## What this deliberately does not do
//!
//! It does not stop a person doing something; it stops two *workers* doing the
//! same thing. An operator who cannot resume a run because a lease is held is
//! told who holds it and when it lapses, rather than being refused blankly.

use chrono::{DateTime, Duration, Utc};
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

/// How long a freshly taken lease is good for.
///
/// Long enough that an ordinary slow step — a cold model load has been measured
/// at over two minutes — does not lapse mid-run, and short enough that a
/// machine which lost power is not locked out for the rest of the session.
pub const DEFAULT_LEASE_SECONDS: i64 = 300;

/// How often a holder should renew. A third of the term, so two renewals can be
/// missed before ownership is actually at risk.
pub const HEARTBEAT_SECONDS: i64 = 100;

/// A held claim on a run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Lease {
    pub run_id: String,
    pub owner: String,
    /// Monotonic per run. Presented with every write the holder makes.
    pub fence_token: i64,
    pub acquired_at: String,
    pub expires_at: String,
}

impl Lease {
    /// Whether this lease is still good at `now`.
    ///
    /// An unparseable expiry counts as lapsed: a claim nothing can read the
    /// term of is not a claim anybody should be held to.
    pub fn live_at(&self, now: DateTime<Utc>) -> bool {
        DateTime::parse_from_rfc3339(&self.expires_at)
            .map(|deadline| now < deadline.with_timezone(&Utc))
            .unwrap_or(false)
    }
}

/// Why a run could not be claimed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Held {
    pub owner: String,
    pub expires_at: String,
}

impl Held {
    pub fn explain(&self) -> String {
        format!(
            "Something else is already working this run ({}), and its claim lasts until {}. \
             Continuing from here as well would run the task twice.",
            self.owner, self.expires_at
        )
    }
}

pub(super) fn prepare(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS run_leases (
            run_id      TEXT PRIMARY KEY,
            owner       TEXT NOT NULL,
            fence_token INTEGER NOT NULL,
            acquired_at TEXT NOT NULL,
            expires_at  TEXT NOT NULL
        );
        ",
    )
}

/// Claims a run, if nothing live holds it.
///
/// The read that decides whether an existing claim has lapsed and the write
/// that replaces it happen in one transaction, so two callers racing produce
/// one winner and one [`Held`], never two winners.
pub(super) fn acquire(
    conn: &Connection,
    run_id: &str,
    owner: &str,
    term: Duration,
    now: DateTime<Utc>,
) -> rusqlite::Result<Result<Lease, Held>> {
    if term <= Duration::zero() {
        return Err(rusqlite::Error::InvalidQuery);
    }
    let tx = rusqlite::Transaction::new_unchecked(conn, rusqlite::TransactionBehavior::Immediate)?;

    let existing: Option<(String, i64, String)> = tx
        .query_row(
            "SELECT owner, fence_token, expires_at FROM run_leases WHERE run_id = ?1",
            [run_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?;

    let previous_token = match &existing {
        Some((holder, token, expires_at)) => {
            let live = DateTime::parse_from_rfc3339(expires_at)
                .map(|deadline| now < deadline.with_timezone(&Utc))
                .unwrap_or(false);
            // Reacquisition is never renewal, even for the same process. Two
            // command handlers in one process are still competing workers.
            if live {
                tx.commit()?;
                return Ok(Err(Held {
                    owner: holder.clone(),
                    expires_at: expires_at.clone(),
                }));
            }
            *token
        }
        None => 0,
    };

    // Incremented on every acquisition, including a reclaim by the same owner.
    // A straggler holding the old token is fenced out even when the new holder
    // has the same name.
    let fence_token = previous_token.checked_add(1).ok_or(rusqlite::Error::InvalidQuery)?;
    let lease = Lease {
        run_id: run_id.to_string(),
        owner: owner.to_string(),
        fence_token,
        acquired_at: now.to_rfc3339(),
        expires_at: (now + term).to_rfc3339(),
    };

    tx.execute(
        "INSERT OR REPLACE INTO run_leases
            (run_id, owner, fence_token, acquired_at, expires_at)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        rusqlite::params![
            lease.run_id,
            lease.owner,
            lease.fence_token,
            lease.acquired_at,
            lease.expires_at
        ],
    )?;
    tx.commit()?;
    Ok(Ok(lease))
}

/// Extends a claim the caller still holds.
///
/// `false` means the caller does not hold it any more — either it lapsed and
/// somebody else took it, or this is a straggler presenting a stale token. Both
/// mean the same thing to the caller: stop.
pub(super) fn renew(
    conn: &Connection,
    run_id: &str,
    owner: &str,
    fence_token: i64,
    term: Duration,
    now: DateTime<Utc>,
) -> rusqlite::Result<bool> {
    if term <= Duration::zero() { return Err(rusqlite::Error::InvalidQuery); }
    let changed = conn.execute(
        "UPDATE run_leases SET expires_at = ?4
          WHERE run_id = ?1 AND owner = ?2 AND fence_token = ?3
            AND julianday(expires_at) > julianday(?5)",
        rusqlite::params![run_id, owner, fence_token, (now + term).to_rfc3339(), now.to_rfc3339()],
    )?;
    Ok(changed > 0)
}

/// Gives a claim up.
///
/// Token-checked for the same reason renewal is: a straggler that wakes up and
/// tidies away a lease somebody else now holds would hand the run to a third
/// party mid-step.
pub(super) fn release(
    conn: &Connection,
    run_id: &str,
    owner: &str,
    fence_token: i64,
) -> rusqlite::Result<bool> {
    let changed = conn.execute(
        // Keep the token tombstone. Deleting the row would reuse token 1 on
        // the next acquisition and let an old holder impersonate the new one.
        "UPDATE run_leases SET expires_at = acquired_at
          WHERE run_id = ?1 AND owner = ?2 AND fence_token = ?3
            AND expires_at != acquired_at",
        rusqlite::params![run_id, owner, fence_token],
    )?;
    Ok(changed > 0)
}

/// Who holds this run at `now`, if anybody. A lapsed claim reads as nobody.
pub(super) fn holder(
    conn: &Connection,
    run_id: &str,
    now: DateTime<Utc>,
) -> rusqlite::Result<Option<Lease>> {
    let lease: Option<Lease> = conn
        .query_row(
            "SELECT run_id, owner, fence_token, acquired_at, expires_at
               FROM run_leases WHERE run_id = ?1",
            [run_id],
            |row| {
                Ok(Lease {
                    run_id: row.get(0)?,
                    owner: row.get(1)?,
                    fence_token: row.get(2)?,
                    acquired_at: row.get(3)?,
                    expires_at: row.get(4)?,
                })
            },
        )
        .optional()?;
    Ok(lease.filter(|lease| lease.live_at(now)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn database() -> Connection {
        let conn = Connection::open_in_memory().expect("an in-memory database");
        prepare(&conn).expect("the lease table is prepared");
        conn
    }

    fn term() -> Duration {
        Duration::seconds(DEFAULT_LEASE_SECONDS)
    }

    #[test]
    fn an_unclaimed_run_can_be_claimed() {
        let conn = database();
        let lease = acquire(&conn, "run-1", "worker-a", term(), Utc::now())
            .expect("the claim is answered")
            .expect("nothing else holds it");
        assert_eq!(lease.owner, "worker-a");
        assert_eq!(lease.fence_token, 1);
    }

    /// The property the whole module exists for.
    #[test]
    fn two_workers_cannot_both_hold_one_run() {
        let conn = database();
        let now = Utc::now();
        acquire(&conn, "run-1", "worker-a", term(), now)
            .expect("answered")
            .expect("first claim succeeds");

        let refused = acquire(&conn, "run-1", "worker-b", term(), now)
            .expect("answered")
            .expect_err("the second claim must be refused");
        assert_eq!(refused.owner, "worker-a");
    }

    /// A process that died holds nothing once its term is up.
    #[test]
    fn a_lapsed_claim_can_be_taken_by_somebody_else() {
        let conn = database();
        let now = Utc::now();
        acquire(&conn, "run-1", "worker-a", Duration::seconds(60), now)
            .expect("answered")
            .expect("claimed");

        let later = now + Duration::seconds(61);
        let lease = acquire(&conn, "run-1", "worker-b", term(), later)
            .expect("answered")
            .expect("the lapsed claim is available");
        assert_eq!(lease.owner, "worker-b");
        assert_eq!(lease.fence_token, 2, "the token must move on");
    }

    /// The fencing property: a straggler that wakes after losing the lease
    /// cannot write, even though it still believes it is the holder.
    #[test]
    fn a_stale_token_cannot_renew_or_release() {
        let conn = database();
        let now = Utc::now();
        let first = acquire(&conn, "run-1", "worker-a", Duration::seconds(60), now)
            .expect("answered")
            .expect("claimed");

        let later = now + Duration::seconds(61);
        acquire(&conn, "run-1", "worker-b", term(), later)
            .expect("answered")
            .expect("taken over");

        assert!(
            !renew(&conn, "run-1", "worker-a", first.fence_token, term(), later)
                .expect("answered"),
            "a straggler must not be able to renew"
        );
        assert!(
            !release(&conn, "run-1", "worker-a", first.fence_token).expect("answered"),
            "a straggler must not be able to release somebody else's claim"
        );

        let held = holder(&conn, "run-1", later)
            .expect("answered")
            .expect("still held");
        assert_eq!(held.owner, "worker-b");
    }

    #[test]
    fn a_holder_can_renew_and_release_its_own_claim() {
        let conn = database();
        let now = Utc::now();
        let lease = acquire(&conn, "run-1", "worker-a", Duration::seconds(60), now)
            .expect("answered")
            .expect("claimed");

        assert!(renew(
            &conn,
            "run-1",
            "worker-a",
            lease.fence_token,
            Duration::seconds(600),
            now
        )
        .expect("answered"));

        // Still held well past the original term, because it was renewed.
        assert!(holder(&conn, "run-1", now + Duration::seconds(120))
            .expect("answered")
            .is_some());

        assert!(release(&conn, "run-1", "worker-a", lease.fence_token).expect("answered"));
        assert!(holder(&conn, "run-1", now).expect("answered").is_none());
    }

    /// Two windows in the same process must not both drive a run.
    #[test]
    fn an_owner_cannot_reacquire_a_live_run_instead_of_renewing() {
        let conn = database();
        let now = Utc::now();
        acquire(&conn, "run-1", "worker-a", term(), now)
            .expect("answered")
            .expect("claimed");

        assert!(acquire(&conn, "run-1", "worker-a", term(), now).unwrap().is_err());
    }

    #[test]
    fn release_never_reuses_a_fencing_token() {
        let conn = database();
        let now = Utc::now();
        let first = acquire(&conn, "r", "w", term(), now).unwrap().unwrap();
        assert!(release(&conn, "r", "w", first.fence_token).unwrap());
        let second = acquire(&conn, "r", "w", term(), now).unwrap().unwrap();
        assert!(second.fence_token > first.fence_token);
        assert!(!release(&conn, "r", "w", first.fence_token).unwrap());
    }

    #[test]
    fn an_expired_worker_cannot_renew_even_before_takeover() {
        let conn = database();
        let now = Utc::now();
        let first = acquire(&conn, "r", "w", Duration::seconds(1), now).unwrap().unwrap();
        assert!(!renew(&conn, "r", "w", first.fence_token, term(), now + Duration::seconds(1)).unwrap());
    }

    #[test]
    fn a_lapsed_claim_reads_as_nobody_holding_it() {
        let conn = database();
        let now = Utc::now();
        acquire(&conn, "run-1", "worker-a", Duration::seconds(60), now)
            .expect("answered")
            .expect("claimed");
        assert!(holder(&conn, "run-1", now + Duration::seconds(61))
            .expect("answered")
            .is_none());
    }

    #[test]
    fn an_unreadable_expiry_counts_as_lapsed() {
        let lease = Lease {
            run_id: "run-1".into(),
            owner: "worker-a".into(),
            fence_token: 1,
            acquired_at: Utc::now().to_rfc3339(),
            expires_at: "not a timestamp".into(),
        };
        assert!(!lease.live_at(Utc::now()));
    }
}

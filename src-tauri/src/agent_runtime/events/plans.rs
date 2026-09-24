//! Where the orchestrator's plan versions and dispatched jobs are kept.
//!
//! Beside the task events rather than in a file of their own, because the
//! three are read together: the plan names jobs, jobs name the `subagent_*`
//! events their children left, and recovery after a crash has to see all of
//! them in one consistent state. One database, one lock, one backup.

use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::store::TaskEventLog;
use crate::agent_runtime::task_plan::TaskPlan;

/// Why a plan version was not written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanWriteError {
    /// Somebody else wrote the version this one meant to follow.
    Conflict { current: u32 },
    Storage(String),
}

impl std::fmt::Display for PlanWriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanWriteError::Conflict { current } => {
                write!(f, "the plan moved on to version {current} while this change was being made")
            }
            PlanWriteError::Storage(detail) => write!(f, "the plan could not be written: {detail}"),
        }
    }
}

/// One row of a plan's history, without its body.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanVersionSummary {
    pub version: u32,
    pub author: String,
    pub reason: String,
    pub created_at: String,
    pub body_sha256: String,
}

/// How a dispatched job stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    /// Recorded and waiting for a lane, a model or a prerequisite.
    Queued,
    Running,
    Completed,
    Partial,
    Blocked,
    Failed,
    TimedOut,
    Cancelled,
    Refused,
    /// The process that ran it went away before it settled.
    Interrupted,
}

impl JobStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            JobStatus::Queued => "queued",
            JobStatus::Running => "running",
            JobStatus::Completed => "completed",
            JobStatus::Partial => "partial",
            JobStatus::Blocked => "blocked",
            JobStatus::Failed => "failed",
            JobStatus::TimedOut => "timed_out",
            JobStatus::Cancelled => "cancelled",
            JobStatus::Refused => "refused",
            JobStatus::Interrupted => "interrupted",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        Some(match raw {
            "queued" => JobStatus::Queued,
            "running" => JobStatus::Running,
            "completed" => JobStatus::Completed,
            "partial" => JobStatus::Partial,
            "blocked" => JobStatus::Blocked,
            "failed" => JobStatus::Failed,
            "timed_out" => JobStatus::TimedOut,
            "cancelled" => JobStatus::Cancelled,
            "refused" => JobStatus::Refused,
            "interrupted" => JobStatus::Interrupted,
            _ => return None,
        })
    }

    /// From a child's ending, as `ChildStatus::as_str` spells it.
    pub fn from_child(raw: &str) -> Self {
        JobStatus::parse(raw).unwrap_or(JobStatus::Failed)
    }

    pub fn is_live(self) -> bool {
        matches!(self, JobStatus::Queued | JobStatus::Running)
    }
}

/// One job the orchestrator dispatched.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobRecord {
    pub job_id: String,
    pub run_id: String,
    pub step_id: String,
    pub role: String,
    /// `readOnly` or `writer`, as `DelegationMode::as_str` spells it.
    pub mode: String,
    /// 1 for the first attempt at this step.
    pub attempt: u32,
    /// Assigned before dispatch, so the child can be named — and stopped —
    /// while it runs.
    pub child_id: String,
    pub status: JobStatus,
    /// The registry definition resolved at dispatch.
    pub definition_id: String,
    pub definition_version: Option<u64>,
    pub definition_origin: String,
    pub model_id: Option<String>,
    pub parent_model_id: Option<String>,
    /// What was decided about the card for this job, in a line. See
    /// `delegation::lease_decision`.
    pub lease: String,
    pub objective_sha256: String,
    pub deadline_at: String,
    pub created_at: String,
    pub settled_at: Option<String>,
    /// A compact summary of the child's result: status, counts, published
    /// ids, artifact versions, missing items. Never finding text.
    pub result: Option<Value>,
    pub result_hash: Option<String>,
}

impl TaskEventLog {
    /// Writes the next version of a run's plan.
    ///
    /// The version must be exactly one past what is stored; anything else is a
    /// writer that read an older plan, and is refused rather than allowed to
    /// overwrite what happened in between.
    pub fn append_plan(&self, plan: &TaskPlan) -> Result<(), PlanWriteError> {
        let body = serde_json::to_string(plan)
            .map_err(|error| PlanWriteError::Storage(error.to_string()))?;
        let mut conn = self
            .lock()
            .map_err(|error| PlanWriteError::Storage(error.to_string()))?;
        let transaction = conn
            .transaction()
            .map_err(|error| PlanWriteError::Storage(error.to_string()))?;
        let current: u32 = transaction
            .query_row(
                "SELECT COALESCE(MAX(version), 0) FROM task_plan_versions WHERE run_id = ?1",
                params![plan.run_id],
                |row| row.get(0),
            )
            .map_err(|error| PlanWriteError::Storage(error.to_string()))?;
        if plan.version != current + 1 {
            return Err(PlanWriteError::Conflict { current });
        }
        transaction
            .execute(
                "INSERT INTO task_plan_versions
                     (run_id, version, author, reason, created_at, body, body_sha256)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    plan.run_id,
                    plan.version,
                    plan.author.as_str(),
                    plan.reason,
                    plan.created_at,
                    body,
                    plan.digest(),
                ],
            )
            .map_err(|error| PlanWriteError::Storage(error.to_string()))?;
        transaction
            .commit()
            .map_err(|error| PlanWriteError::Storage(error.to_string()))
    }

    /// The newest version of a run's plan, re-checked against its stored hash.
    pub fn latest_plan(&self, run_id: &str) -> Result<Option<TaskPlan>, String> {
        let conn = self.lock().map_err(|error| error.to_string())?;
        let row: Option<(String, String)> = conn
            .query_row(
                "SELECT body, body_sha256 FROM task_plan_versions
                  WHERE run_id = ?1 ORDER BY version DESC LIMIT 1",
                params![run_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(|error| error.to_string())?;
        drop(conn);
        let Some((body, stored_hash)) = row else {
            return Ok(None);
        };
        let plan: TaskPlan = serde_json::from_str(&body)
            .map_err(|error| format!("the stored plan for {run_id} could not be read: {error}"))?;
        if plan.digest() != stored_hash {
            return Err(format!(
                "the stored plan for {run_id} does not match the hash it was written with; it \
                 is not used"
            ));
        }
        Ok(Some(plan))
    }

    /// One version of a run's plan.
    pub fn plan_at(&self, run_id: &str, version: u32) -> Result<Option<TaskPlan>, String> {
        let conn = self.lock().map_err(|error| error.to_string())?;
        let body: Option<String> = conn
            .query_row(
                "SELECT body FROM task_plan_versions WHERE run_id = ?1 AND version = ?2",
                params![run_id, version],
                |row| row.get(0),
            )
            .optional()
            .map_err(|error| error.to_string())?;
        body.map(|body| serde_json::from_str(&body).map_err(|error| error.to_string()))
            .transpose()
    }

    /// Every version's header, oldest first.
    pub fn plan_history(&self, run_id: &str) -> Result<Vec<PlanVersionSummary>, String> {
        let conn = self.lock().map_err(|error| error.to_string())?;
        let mut statement = conn
            .prepare(
                "SELECT version, author, reason, created_at, body_sha256
                   FROM task_plan_versions WHERE run_id = ?1 ORDER BY version",
            )
            .map_err(|error| error.to_string())?;
        let rows = statement
            .query_map(params![run_id], |row| {
                Ok(PlanVersionSummary {
                    version: row.get(0)?,
                    author: row.get(1)?,
                    reason: row.get(2)?,
                    created_at: row.get(3)?,
                    body_sha256: row.get(4)?,
                })
            })
            .map_err(|error| error.to_string())?;
        rows.collect::<Result<Vec<_>, _>>().map_err(|error| error.to_string())
    }

    /// Records a job before it is dispatched.
    pub fn record_job(&self, job: &JobRecord) -> Result<(), String> {
        let conn = self.lock().map_err(|error| error.to_string())?;
        conn.execute(
            "INSERT INTO delegated_jobs
                 (job_id, run_id, step_id, role, mode, attempt, child_id, status,
                  definition_id, definition_version, definition_origin, model_id,
                  parent_model_id, lease, objective_sha256, deadline_at, created_at,
                  settled_at, result, result_hash)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15,
                     ?16, ?17, ?18, ?19, ?20)",
            params![
                job.job_id,
                job.run_id,
                job.step_id,
                job.role,
                job.mode,
                job.attempt,
                job.child_id,
                job.status.as_str(),
                job.definition_id,
                job.definition_version.map(|v| v as i64),
                job.definition_origin,
                job.model_id,
                job.parent_model_id,
                job.lease,
                job.objective_sha256,
                job.deadline_at,
                job.created_at,
                job.settled_at,
                job.result.as_ref().map(Value::to_string),
                job.result_hash,
            ],
        )
        .map_err(|error| error.to_string())?;
        Ok(())
    }

    /// Moves a queued job to running. `false` when it was no longer queued.
    pub fn start_job(&self, job_id: &str) -> Result<bool, String> {
        let conn = self.lock().map_err(|error| error.to_string())?;
        let changed = conn
            .execute(
                "UPDATE delegated_jobs SET status = 'running'
                  WHERE job_id = ?1 AND status = 'queued'",
                params![job_id],
            )
            .map_err(|error| error.to_string())?;
        Ok(changed == 1)
    }

    /// Records a job's ending. `false` when it had already ended: the first
    /// ending stands.
    pub fn settle_job(
        &self,
        job_id: &str,
        status: JobStatus,
        result: &Value,
        result_hash: &str,
        at: &str,
    ) -> Result<bool, String> {
        let conn = self.lock().map_err(|error| error.to_string())?;
        let changed = conn
            .execute(
                "UPDATE delegated_jobs
                    SET status = ?2, result = ?3, result_hash = ?4, settled_at = ?5
                  WHERE job_id = ?1 AND status IN ('queued', 'running')",
                params![job_id, status.as_str(), result.to_string(), result_hash, at],
            )
            .map_err(|error| error.to_string())?;
        Ok(changed == 1)
    }

    pub fn job(&self, job_id: &str) -> Result<Option<JobRecord>, String> {
        let conn = self.lock().map_err(|error| error.to_string())?;
        conn.query_row(&format!("{JOB_COLUMNS} WHERE job_id = ?1"), params![job_id], job_from_row)
            .optional()
            .map_err(|error| error.to_string())
    }

    pub fn jobs_for_run(&self, run_id: &str) -> Result<Vec<JobRecord>, String> {
        let conn = self.lock().map_err(|error| error.to_string())?;
        let mut statement = conn
            .prepare(&format!("{JOB_COLUMNS} WHERE run_id = ?1 ORDER BY created_at, job_id"))
            .map_err(|error| error.to_string())?;
        let rows = statement
            .query_map(params![run_id], job_from_row)
            .map_err(|error| error.to_string())?;
        rows.collect::<Result<Vec<_>, _>>().map_err(|error| error.to_string())
    }

    /// The most recent jobs across every run, newest first, for the Agents
    /// page.
    pub fn recent_jobs(&self, limit: usize) -> Result<Vec<JobRecord>, String> {
        let conn = self.lock().map_err(|error| error.to_string())?;
        let mut statement = conn
            .prepare(&format!("{JOB_COLUMNS} ORDER BY created_at DESC, job_id LIMIT ?1"))
            .map_err(|error| error.to_string())?;
        let rows = statement
            .query_map(params![limit as i64], job_from_row)
            .map_err(|error| error.to_string())?;
        rows.collect::<Result<Vec<_>, _>>().map_err(|error| error.to_string())
    }

    /// Marks every job left queued or running by a process that is gone.
    ///
    /// Run once at start, before any job of this process exists, so nothing
    /// it marks can be one that is actually running. Returns what it marked,
    /// so the plans that name them can be settled.
    pub fn interrupt_live_jobs(&self, at: &str) -> Result<Vec<JobRecord>, String> {
        let live: Vec<JobRecord> = {
            let conn = self.lock().map_err(|error| error.to_string())?;
            let mut statement = conn
                .prepare(&format!("{JOB_COLUMNS} WHERE status IN ('queued', 'running')"))
                .map_err(|error| error.to_string())?;
            let rows = statement
                .query_map([], job_from_row)
                .map_err(|error| error.to_string())?;
            rows.collect::<Result<Vec<_>, _>>().map_err(|error| error.to_string())?
        };
        for job in &live {
            self.settle_job(
                &job.job_id,
                JobStatus::Interrupted,
                &serde_json::json!({
                    "status": "interrupted",
                    "detail": "the process running this job stopped before it settled",
                }),
                "",
                at,
            )?;
        }
        Ok(live)
    }
}

const JOB_COLUMNS: &str = "SELECT job_id, run_id, step_id, role, mode, attempt, child_id, status,
        definition_id, definition_version, definition_origin, model_id, parent_model_id, lease,
        objective_sha256, deadline_at, created_at, settled_at, result, result_hash
   FROM delegated_jobs";

fn job_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<JobRecord> {
    let status: String = row.get(7)?;
    let result: Option<String> = row.get(18)?;
    Ok(JobRecord {
        job_id: row.get(0)?,
        run_id: row.get(1)?,
        step_id: row.get(2)?,
        role: row.get(3)?,
        mode: row.get(4)?,
        attempt: row.get(5)?,
        child_id: row.get(6)?,
        status: JobStatus::parse(&status).unwrap_or(JobStatus::Failed),
        definition_id: row.get(8)?,
        definition_version: row.get::<_, Option<i64>>(9)?.map(|v| v as u64),
        definition_origin: row.get(10)?,
        model_id: row.get(11)?,
        parent_model_id: row.get(12)?,
        lease: row.get(13)?,
        objective_sha256: row.get(14)?,
        deadline_at: row.get(15)?,
        created_at: row.get(16)?,
        settled_at: row.get(17)?,
        result: result.and_then(|text| serde_json::from_str(&text).ok()),
        result_hash: row.get(19)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_runtime::task_plan::{Author, TaskPlan};

    fn plan(version: u32) -> TaskPlan {
        TaskPlan {
            run_id: "run-p".into(),
            version,
            goal: "a goal".into(),
            constraints: Vec::new(),
            corrections: Vec::new(),
            steps: Vec::new(),
            decisions: Vec::new(),
            questions: Vec::new(),
            author: Author::Model,
            reason: format!("v{version}"),
            created_at: "2026-09-24T00:00:00Z".into(),
        }
    }

    fn job(id: &str, status: JobStatus) -> JobRecord {
        JobRecord {
            job_id: id.into(),
            run_id: "run-p".into(),
            step_id: "s1".into(),
            role: "knowledge-retriever".into(),
            mode: "readOnly".into(),
            attempt: 1,
            child_id: format!("child-{id}"),
            status,
            definition_id: "knowledge-retriever".into(),
            definition_version: Some(3),
            definition_origin: "registry".into(),
            model_id: Some("m".into()),
            parent_model_id: Some("m".into()),
            lease: "shares".into(),
            objective_sha256: "0".repeat(64),
            deadline_at: "2026-09-24T00:10:00Z".into(),
            created_at: "2026-09-24T00:00:00Z".into(),
            settled_at: None,
            result: None,
            result_hash: None,
        }
    }

    #[test]
    fn versions_are_consecutive_and_a_writer_that_read_an_old_one_is_refused() {
        let log = TaskEventLog::in_memory().unwrap();
        log.append_plan(&plan(1)).unwrap();
        log.append_plan(&plan(2)).unwrap();
        assert_eq!(log.append_plan(&plan(2)), Err(PlanWriteError::Conflict { current: 2 }));
        assert_eq!(log.append_plan(&plan(4)), Err(PlanWriteError::Conflict { current: 2 }));
        assert_eq!(log.latest_plan("run-p").unwrap().unwrap().version, 2);
        assert_eq!(log.plan_history("run-p").unwrap().len(), 2);
        assert_eq!(log.plan_at("run-p", 1).unwrap().unwrap().reason, "v1");
    }

    #[test]
    fn a_stored_plan_version_cannot_be_rewritten_or_deleted() {
        let log = TaskEventLog::in_memory().unwrap();
        log.append_plan(&plan(1)).unwrap();
        let conn = log.lock().unwrap();
        let update = conn.execute("UPDATE task_plan_versions SET reason = 'rewritten'", []);
        assert!(update.unwrap_err().to_string().contains("append-only"));
        let delete = conn.execute("DELETE FROM task_plan_versions", []);
        assert!(delete.unwrap_err().to_string().contains("append-only"));
    }

    #[test]
    fn a_job_settles_once_and_a_restart_interrupts_only_live_ones() {
        let log = TaskEventLog::in_memory().unwrap();
        log.record_job(&job("j1", JobStatus::Queued)).unwrap();
        log.record_job(&job("j2", JobStatus::Queued)).unwrap();
        assert!(log.start_job("j1").unwrap());
        assert!(!log.start_job("j1").unwrap(), "started twice");
        let result = serde_json::json!({"status": "completed"});
        assert!(log.settle_job("j1", JobStatus::Completed, &result, "h", "t").unwrap());
        assert!(
            !log.settle_job("j1", JobStatus::Failed, &result, "h2", "t2").unwrap(),
            "a second ending overwrote the first"
        );
        assert_eq!(log.job("j1").unwrap().unwrap().status, JobStatus::Completed);

        let interrupted = log.interrupt_live_jobs("t3").unwrap();
        assert_eq!(interrupted.iter().map(|j| j.job_id.as_str()).collect::<Vec<_>>(), vec!["j2"]);
        assert_eq!(log.job("j2").unwrap().unwrap().status, JobStatus::Interrupted);
        assert_eq!(log.job("j1").unwrap().unwrap().status, JobStatus::Completed);
        assert_eq!(log.jobs_for_run("run-p").unwrap().len(), 2);
        assert_eq!(log.job("j1").unwrap().unwrap().definition_version, Some(3));
    }
}

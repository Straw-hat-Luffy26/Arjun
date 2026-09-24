//! Frames on the wire to the agent runtime.
//!
//! Mirrors `agent-runtime/src/protocol.ts`. The two files are one contract
//! written twice, which is the cost of the boundary falling between languages.
//! The round-trip tests at the bottom encode what the TypeScript side actually
//! emits rather than what this side hopes it emits, so a drift in either
//! direction fails here instead of surfacing as a hang at run time.
//!
//! Newline-delimited JSON, one frame per line. See the TypeScript module for
//! why that framing and not a length prefix.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A parsed line, before classification.
///
/// Deliberately a flat struct with optional fields rather than an untagged
/// enum: untagged deserialisation reports "data did not match any variant",
/// which for a cross-language wire format is close to the least useful thing it
/// could say. Parsing permissively and classifying explicitly means a malformed
/// frame can be described precisely enough to fix.
#[derive(Debug, Clone, Deserialize)]
pub struct RawFrame {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub method: Option<String>,
    #[serde(default)]
    pub params: Option<Value>,
    #[serde(default)]
    pub result: Option<Value>,
    #[serde(default)]
    pub error: Option<WireError>,
}

/// The classified shape of a frame.
#[derive(Debug, Clone)]
pub enum Frame {
    /// Awaiting a reply, correlated by `id`.
    Request {
        id: String,
        method: String,
        params: Value,
    },
    /// A successful reply to something this side sent.
    Result { id: String, result: Value },
    /// A failed reply to something this side sent.
    Error { id: String, error: WireError },
    /// One-way. Never replied to.
    Notification { method: String, params: Value },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireError {
    pub code: String,
    pub message: String,
}

impl WireError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

/// Codes both ends branch on. Mirrors `ErrorCode` in the TypeScript module.
pub mod code {
    pub const UNKNOWN_METHOD: &str = "unknown_method";
    pub const BAD_PARAMS: &str = "bad_params";
    pub const REFUSED: &str = "refused";
    pub const TOOL_FAILED: &str = "tool_failed";
    pub const INTERNAL: &str = "internal";
    /// A model round that must not go ahead: no GPU lease, a model that would
    /// not load, mandatory state or a rendered request that does not fit. The
    /// runtime fails the run on this code instead of calling the model anyway.
    /// See `agent_runtime::rounds`.
    pub const ROUND_REFUSED: &str = "round_refused";
}

#[derive(Debug, thiserror::Error)]
pub enum FrameError {
    #[error("frame is not JSON: {0}")]
    NotJson(#[from] serde_json::Error),
    /// A frame carrying none of the four recognised combinations. Fatal: the
    /// two ends disagree about the channel, and continuing past that produces
    /// confusing failures far from the cause.
    #[error("frame matches no known shape")]
    UnknownShape,
}

impl Frame {
    pub fn parse(line: &str) -> Result<Self, FrameError> {
        let raw: RawFrame = serde_json::from_str(line)?;
        Self::classify(raw)
    }

    fn classify(raw: RawFrame) -> Result<Self, FrameError> {
        // Order mirrors the TypeScript peer: a request carries both `id` and
        // `method`, so it has to be recognised before the id-only shapes.
        match (raw.id, raw.method) {
            (Some(id), Some(method)) => Ok(Frame::Request {
                id,
                method,
                params: raw.params.unwrap_or(Value::Null),
            }),
            (Some(id), None) => {
                if let Some(error) = raw.error {
                    Ok(Frame::Error { id, error })
                } else if let Some(result) = raw.result {
                    Ok(Frame::Result { id, result })
                } else {
                    // An id with neither result nor error. Read as a null result
                    // rather than refused: a handler returning nothing is
                    // ordinary, and JSON cannot distinguish "returned undefined"
                    // from "field omitted".
                    Ok(Frame::Result {
                        id,
                        result: Value::Null,
                    })
                }
            }
            (None, Some(method)) => Ok(Frame::Notification {
                method,
                params: raw.params.unwrap_or(Value::Null),
            }),
            (None, None) => Err(FrameError::UnknownShape),
        }
    }
}

/// Frames this side sends.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum Outgoing {
    Request {
        id: String,
        method: String,
        params: Value,
    },
    Result {
        id: String,
        result: Value,
    },
    Error {
        id: String,
        error: WireError,
    },
    Notification {
        method: String,
        params: Value,
    },
}

impl Outgoing {
    /// Serialises to one line, newline included.
    ///
    /// Infallible by construction: every variant holds a `Value`, which always
    /// serialises. Returning a `String` rather than a `Result` keeps the write
    /// path free of an error case that cannot occur.
    pub fn encode(&self) -> String {
        let mut line =
            serde_json::to_string(self).expect("Outgoing holds only serialisable values");
        line.push('\n');
        line
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The literals here are what `agent-runtime/src/protocol.ts` emits.
    #[test]
    fn parses_the_four_shapes_the_typescript_side_emits() {
        let request =
            Frame::parse(r#"{"id":"1","method":"tool.authorize","params":{"tool":"x"}}"#).unwrap();
        assert!(matches!(request, Frame::Request { ref method, .. } if method == "tool.authorize"));

        let result = Frame::parse(r#"{"id":"1","result":{"ok":true}}"#).unwrap();
        assert!(matches!(result, Frame::Result { ref id, .. } if id == "1"));

        let error = Frame::parse(r#"{"id":"1","error":{"code":"refused","message":"no"}}"#).unwrap();
        assert!(matches!(error, Frame::Error { ref error, .. } if error.code == "refused"));

        let note = Frame::parse(r#"{"method":"run.event","params":{"n":1}}"#).unwrap();
        assert!(matches!(note, Frame::Notification { ref method, .. } if method == "run.event"));
    }

    #[test]
    fn a_request_is_not_mistaken_for_a_result() {
        // Both carry `id`; only the ordering of the checks keeps them apart.
        let frame = Frame::parse(r#"{"id":"7","method":"health","params":null}"#).unwrap();
        assert!(matches!(frame, Frame::Request { .. }));
    }

    #[test]
    fn an_id_with_no_payload_is_a_null_result_not_a_failure() {
        match Frame::parse(r#"{"id":"3"}"#).unwrap() {
            Frame::Result { result, .. } => assert_eq!(result, Value::Null),
            other => panic!("expected a null result, got {other:?}"),
        }
    }

    #[test]
    fn a_frame_with_neither_id_nor_method_is_refused() {
        let err = Frame::parse(r#"{"hello":"world"}"#).unwrap_err();
        assert!(matches!(err, FrameError::UnknownShape));
    }

    #[test]
    fn non_json_is_refused_rather_than_skipped() {
        assert!(matches!(
            Frame::parse("not json"),
            Err(FrameError::NotJson(_))
        ));
    }

    #[test]
    fn encoding_terminates_with_exactly_one_newline() {
        let line = Outgoing::Notification {
            method: "run.event".into(),
            params: serde_json::json!({ "n": 1 }),
        }
        .encode();
        assert!(line.ends_with('\n'));
        assert_eq!(line.matches('\n').count(), 1);
    }

    #[test]
    fn payload_newlines_are_escaped_so_framing_survives_document_text() {
        // A tool result carrying a two-line extract must not become two frames.
        let line = Outgoing::Result {
            id: "1".into(),
            result: serde_json::json!({ "text": "line one\nline two" }),
        }
        .encode();
        assert_eq!(line.matches('\n').count(), 1);

        match Frame::parse(line.trim_end()).unwrap() {
            Frame::Result { result, .. } => assert_eq!(result["text"], "line one\nline two"),
            other => panic!("expected a result, got {other:?}"),
        }
    }
}

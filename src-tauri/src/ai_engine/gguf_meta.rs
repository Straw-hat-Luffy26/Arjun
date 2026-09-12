//! GGUF header metadata, read before the model is loaded.
//!
//! Deciding where a model's weights go has to happen *before* a `LlamaModel`
//! exists, and `llama-cpp-2` only exposes `n_layer()`, `n_params()` and
//! `meta_val_str()` on an already-loaded model. This reads the same figures
//! straight from the file's key-value header.
//!
//! Only the header is touched. Arrays are seeked past rather than read — the
//! tokenizer vocabulary alone is typically 128k strings — so this costs a few
//! kilobytes of I/O on a 12 GB file.
//!
//! Two planner inputs depend on it:
//!
//! - `block_count`, which [`vram_planner`](super::vram_planner) otherwise guesses
//!   at 32 (`ASSUMED_LAYERS_FALLBACK`). gpt-oss-20b has 24.
//! - the exact KV cache cost, which the size-banded estimate gets badly wrong for
//!   MoE: it bands on *file size*, but a MoE file is large because of experts
//!   while its KV cost is driven by attention. For gpt-oss-20b the band returns
//!   256 KB/token against a real 48 KB — a 5× over-estimate that would consume an
//!   entire 4 GB card's weight budget on its own.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// v1 used 32-bit lengths throughout and is not produced by any current
/// converter. Refusing it is better than misreading it as v2.
const MIN_VERSION: u32 = 2;

/// Guards against allocating or looping against a corrupt length.
const MAX_KV_COUNT: u64 = 1 << 20;
const MAX_STRING_BYTES: u64 = 64 << 20;

/// KV cache elements are f16 in llama.cpp regardless of weight quantization.
const KV_BYTES_PER_ELEMENT: u64 = 2;

/// gate, up and down projections per expert. A fused `gate_up` tensor is
/// `embedding_length × (expert_ff_length × 2)` — the same parameter total, so
/// this holds for both tensor layouts.
const PROJECTIONS_PER_EXPERT: u64 = 3;

/// Ceiling on the share of a file attributed to routed experts.
///
/// The expert figure is computed from header geometry while the total may come
/// from a different key; a disagreement must not claim the whole file is experts
/// and leave nothing resident.
const MAX_EXPERT_FRACTION: f64 = 0.95;

/// The header figures the planner needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufMetadata {
    pub architecture: String,
    pub block_count: u32,
    pub embedding_length: u32,
    /// 0 for a dense model.
    pub expert_count: u32,
    /// Experts consulted per token — 4 of 32 for gpt-oss. This is what makes
    /// expert offload viable: only this share crosses PCIe per token.
    pub expert_used_count: u32,
    pub expert_ff_length: u32,
    pub head_count_kv: u32,
    pub key_length: u32,
    pub value_length: u32,
    /// From `general.parameter_count`, which not every converter writes.
    pub parameter_count: Option<u64>,
    /// Training context length, from `<arch>.context_length`.
    ///
    /// `None` when the converter did not write the key. Read because the
    /// registry otherwise recorded a flat 8192 for every model it found on
    /// disk, whatever the model was — which is the number the context meter
    /// showed and the number the agent loop compacted against. A model trained
    /// for 128k was being compacted at 8k, and one trained for 4k was told it
    /// had twice the room it has.
    pub context_length: Option<u32>,
    /// Whether this model's own chat template exposes a reasoning switch.
    ///
    /// Read from `tokenizer.chat_template` by looking for the `enable_thinking`
    /// variable the template branches on. That is the model telling us, in its
    /// own words, that it emits a separable reasoning block and will honour a
    /// request to turn it off.
    ///
    /// Derived rather than matched against a list of families. The previous
    /// answer to "does this model reason?" was a regular expression over the
    /// model id (`qwen-?3`, `nemotron-3`), which is wrong twice over: it says
    /// nothing about a model nobody has added to the list, and it keeps saying
    /// yes about a fine-tune whose template no longer has the switch.
    ///
    /// `false` for a model whose template does not mention it, which covers
    /// both a model that never reasons and one that always does. Neither wants
    /// the kwarg sent.
    pub supports_toggled_reasoning: bool,
    /// Whether this model produces a reasoning block **at all**.
    ///
    /// A different question from [`Self::supports_toggled_reasoning`], and
    /// conflating the two suppressed streaming across the whole product.
    ///
    /// The switch answers "may the `enable_thinking` kwarg be sent?". This
    /// answers "will reasoning come back?", and a model that always reasons —
    /// a DeepSeek-R1 distill, any model with a think tag baked into its
    /// template — answers no to the first and yes to the second.
    ///
    /// That mattered far beyond the Thinking panel. The runtime set
    /// `thinkingLevel` from the switch, and `thinkingLevel: "off"` makes the
    /// transport drop every reasoning delta *and* makes the reasoning-tag
    /// partitioner hold the visible answer back with them — so the whole
    /// answer arrived as a single `text_delta` and nothing streamed. Prose
    /// merely appeared abruptly; a long code block appeared to hang, because
    /// the reader watched an empty space for the entire time the model spent
    /// writing it.
    pub emits_reasoning: bool,
}

impl GgufMetadata {
    pub fn is_moe(&self) -> bool {
        self.expert_count > 0
    }

    /// Exact f16 KV cache cost per token.
    ///
    /// `layers × kv_heads × (key_dim + value_dim) × 2`. The key and value
    /// dimensions carry the factor of two that the size-banded estimate spells
    /// out separately.
    pub fn kv_bytes_per_token(&self) -> u64 {
        u64::from(self.block_count)
            * u64::from(self.head_count_kv)
            * (u64::from(self.key_length) + u64::from(self.value_length))
            * KV_BYTES_PER_ELEMENT
    }

    /// Routed-expert parameters across every layer.
    pub fn expert_params(&self) -> u64 {
        u64::from(self.block_count)
            .saturating_mul(u64::from(self.expert_count))
            .saturating_mul(PROJECTIONS_PER_EXPERT)
            .saturating_mul(u64::from(self.embedding_length))
            .saturating_mul(u64::from(self.expert_ff_length))
    }

    /// Parameters actually used per token.
    ///
    /// Everything outside the routed experts, plus the share of experts the
    /// router consults. Returns `None` when the total is unknown, since the
    /// non-expert remainder cannot be derived without it.
    pub fn active_params(&self, total_params: Option<u64>) -> Option<u64> {
        let total = total_params.or(self.parameter_count)?;
        let experts = self.expert_params();

        if experts == 0 || self.expert_count == 0 {
            return Some(total);
        }
        // A disagreement between header geometry and the stated total must not
        // underflow into a nonsense figure.
        let dense = total.saturating_sub(experts);
        let used = experts / u64::from(self.expert_count) * u64::from(self.expert_used_count);

        Some(dense.saturating_add(used))
    }

    /// Bytes of routed-expert weight inside a file of `model_bytes`.
    ///
    /// Derived as a share of the real file size rather than summed from tensor
    /// dimensions, because per-tensor byte sizes are not recoverable from the
    /// header across mixed quantizations. This assumes experts are quantized
    /// comparably to the rest of the model, which holds for the targets
    /// (gpt-oss-20b is uniformly MXFP4).
    ///
    /// Returns 0 when the file is dense or the geometry is unusable, which the
    /// caller reads as "no expert offload available".
    pub fn expert_bytes(&self, model_bytes: u64, total_params: Option<u64>) -> u64 {
        let experts = self.expert_params();
        let total = total_params.or(self.parameter_count).unwrap_or(0);
        if experts == 0 || total == 0 {
            return 0;
        }

        let fraction = (experts as f64 / total as f64).min(MAX_EXPERT_FRACTION);
        (model_bytes as f64 * fraction) as u64
    }
}

/// Reads the header of the GGUF file at `path`.
pub fn read_gguf_metadata(path: &Path) -> Result<GgufMetadata> {
    let file = File::open(path)
        .with_context(|| format!("could not open GGUF file '{}'", path.display()))?;
    let mut reader = BufReader::new(file);
    parse_gguf_metadata(&mut reader)
        .with_context(|| format!("could not read GGUF header of '{}'", path.display()))
}

/// Parses a GGUF header from any seekable stream.
///
/// Split from [`read_gguf_metadata`] so the format handling is testable against
/// synthetic headers without writing multi-gigabyte fixtures.
pub fn parse_gguf_metadata<R: Read + Seek>(r: &mut R) -> Result<GgufMetadata> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).context("file is too short to be a GGUF")?;
    if &magic != GGUF_MAGIC {
        bail!(
            "not a GGUF file: expected magic 'GGUF', found {:?}",
            String::from_utf8_lossy(&magic)
        );
    }

    let version = read_u32(r)?;
    if version < MIN_VERSION {
        bail!("unsupported GGUF version {version}; {MIN_VERSION} is the minimum");
    }

    let _tensor_count = read_u64(r)?;
    let kv_count = read_u64(r)?;
    if kv_count > MAX_KV_COUNT {
        bail!("GGUF header claims {kv_count} metadata entries, which is not credible");
    }

    let mut kv: HashMap<String, Scalar> = HashMap::new();
    for _ in 0..kv_count {
        let key = read_string(r)?;
        let value_type = read_u32(r)?;
        // Only the token list is worth looking inside. Every other array in a
        // GGUF header is numeric or is not about what the model can emit.
        let scan = key == "tokenizer.ggml.tokens";
        match read_value(r, value_type, scan)? {
            ArrayOrScalar::Scalar(value) => {
                kv.insert(key, value);
            }
            ArrayOrScalar::ScannedVocabulary { has_reasoning_token } => {
                if has_reasoning_token {
                    kv.insert(
                        VOCABULARY_HAS_REASONING_TOKEN.to_string(),
                        Scalar::Bool(true),
                    );
                }
            }
            ArrayOrScalar::Skipped => {}
        }
    }

    from_kv(&kv)
}

/// Tag openers that mean "what follows is reasoning".
///
/// These are the ones the runtime's own partitioner recognises
/// (`REASONING_TAG_NAMES` in `markdown-core`), and keeping the two in step is
/// the whole point: the partitioner strips these out of the visible answer, and
/// if this side has not declared the model as reasoning, the transport drops
/// what the partitioner stripped and holds the answer back with it.
///
/// Prefixes, not whole tags: `<think` covers `<think>` and `<thinking>`, and a
/// tag carrying attributes still matches.
const REASONING_OPENERS: &[&str] =
    &["<think", "<thought", "<reasoning", "<internal", "<antthinking"];

/// The key under which the vocabulary scan records what it saw.
///
/// Not a real GGUF key. The token list is stepped over rather than read — a
/// production vocabulary is 130k entries and this parse runs for every model on
/// the shelf — so the scanner cannot hand back the tokens themselves. It
/// records this one bit instead, in the same map, so `from_kv` stays a pure
/// function of the parsed header.
const VOCABULARY_HAS_REASONING_TOKEN: &str = "arjun.vocabulary_has_reasoning_token";

fn from_kv(kv: &HashMap<String, Scalar>) -> Result<GgufMetadata> {
    let architecture = kv
        .get("general.architecture")
        .and_then(Scalar::as_str)
        .ok_or_else(|| anyhow!("GGUF header has no general.architecture"))?
        .to_string();

    let get = |suffix: &str| {
        kv.get(&format!("{architecture}.{suffix}"))
            .and_then(Scalar::as_u32)
    };

    let block_count = get("block_count")
        .ok_or_else(|| anyhow!("GGUF header has no {architecture}.block_count"))?;
    let embedding_length = get("embedding_length").unwrap_or(0);
    let head_count = get("attention.head_count").unwrap_or(0);

    // A GGUF may store head_count_kv as a per-layer array, which is skipped
    // during parsing. Falling back to head_count over-states the KV cost rather
    // than under-stating it, which is the safe direction — see vram_planner.
    let head_count_kv = get("attention.head_count_kv").unwrap_or(head_count);

    // llama.cpp applies the same default when these keys are absent.
    let default_head_dim = embedding_length.checked_div(head_count).unwrap_or(0);
    let key_length = get("attention.key_length").unwrap_or(default_head_dim);
    let value_length = get("attention.value_length").unwrap_or(default_head_dim);
    let expert_count = get("expert_count").unwrap_or(0);
    let expert_used_count = get("expert_used_count").unwrap_or(0);
    let expert_ff_length = get("expert_feed_forward_length").unwrap_or(0);
    let context_length = get("context_length");

    // Substring rather than a template parse. The question is only whether the
    // template branches on the variable at all; rendering it would mean
    // shipping a Jinja engine to answer a yes-or-no.
    let chat_template = kv.get("tokenizer.chat_template").and_then(Scalar::as_str);
    let supports_toggled_reasoning = chat_template
        .map(|template| template.contains("enable_thinking"))
        .unwrap_or(false);
    // A template that opens a reasoning block is a model that reasons, switch
    // or no switch. Checked as well as the switch rather than instead of it: a
    // Qwen3 template has both and a DeepSeek-R1 distill has only the tag, and
    // reading only the switch is what made the second look like a model that
    // never reasons.
    //
    // The openers are the ones the runtime's own reasoning-tag partitioner
    // recognises (`REASONING_TAG_NAMES` in `markdown-core`), and keeping the
    // two in step is the whole point. The partitioner strips these tags out of
    // the visible answer; if this side has not declared the model as reasoning,
    // the transport drops what the partitioner stripped and holds the answer
    // back with it. A tag one side treats as reasoning and the other does not
    // is precisely the disagreement that loses the text.
    //
    // Prefixes, not whole tags: `<think` covers `<think>` and `<thinking>`, and
    // a tag carrying attributes still matches.
    let emits_reasoning = supports_toggled_reasoning
        || chat_template
            .map(|template| {
                REASONING_OPENERS
                    .iter()
                    .any(|opener| template.contains(opener))
            })
            .unwrap_or(false)
        // The vocabulary, for a model whose header carries no template at all.
        //
        // NVIDIA-Nemotron3-Nano-4B is exactly that: no
        // `tokenizer.chat_template` key, llama.cpp guesses a profile, and
        // `<think>` / `</think>` sit in `tokenizer.ggml.tokens` as special
        // tokens. It reasons on every turn. Reading only the template called
        // it a non-reasoning model, which set `thinkingLevel: "off"` in the
        // runtime, which dropped its reasoning *and held the visible answer
        // back with it* — twelve seconds of blank screen and then the whole
        // reply at once.
        //
        // A model cannot emit a token that is not in its vocabulary, so this
        // is evidence of the same kind as the template, not a guess.
        || kv.contains_key(VOCABULARY_HAS_REASONING_TOKEN);

    // Every `get` is done, so the closure's borrow of `architecture` has ended
    // and it can be moved into the result.
    Ok(GgufMetadata {
        architecture,
        supports_toggled_reasoning,
        emits_reasoning,
        block_count,
        embedding_length,
        expert_count,
        expert_used_count,
        expert_ff_length,
        head_count_kv,
        key_length,
        value_length,
        parameter_count: kv.get("general.parameter_count").and_then(Scalar::as_u64),
        // Deliberately not defaulted here. A caller that needs a number when the
        // key is missing has to choose one and say why; a default invented in
        // the parser would be indistinguishable from a value the file stated.
        context_length,
    })
}

// ─── Value decoding ─────────────────────────────────────────────────────────

/// A scalar metadata value. Arrays are skipped rather than represented — none of
/// the figures this module needs is stored as one.
#[derive(Debug, Clone, PartialEq)]
enum Scalar {
    U(u64),
    I(i64),
    F(f64),
    Bool(bool),
    Str(String),
}

impl Scalar {
    fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U(v) => u32::try_from(*v).ok(),
            Self::I(v) => u32::try_from(*v).ok(),
            _ => None,
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U(v) => Some(*v),
            Self::I(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// What one metadata value turned out to be.
///
/// Arrays are not materialised — see [`skip_array`] — so they cannot come back
/// as a `Scalar`. The vocabulary is the one array whose *contents* change a
/// decision, and it reports the single bit it was scanned for.
enum ArrayOrScalar {
    Scalar(Scalar),
    Skipped,
    ScannedVocabulary { has_reasoning_token: bool },
}

/// Reads one value, stepping over arrays.
///
/// `scan_vocabulary` asks for the one exception: the token list is walked and
/// each entry checked against [`REASONING_OPENERS`], because a model whose
/// header carries no chat template still declares what it can emit here.
fn read_value<R: Read + Seek>(
    r: &mut R,
    value_type: u32,
    scan_vocabulary: bool,
) -> Result<ArrayOrScalar> {
    let scalar = match value_type {
        0 => Scalar::U(u64::from(read_n::<_, 1>(r)?[0])),
        1 => Scalar::I(i64::from(read_n::<_, 1>(r)?[0] as i8)),
        2 => Scalar::U(u64::from(u16::from_le_bytes(read_n::<_, 2>(r)?))),
        3 => Scalar::I(i64::from(i16::from_le_bytes(read_n::<_, 2>(r)?))),
        4 => Scalar::U(u64::from(read_u32(r)?)),
        5 => Scalar::I(i64::from(i32::from_le_bytes(read_n::<_, 4>(r)?))),
        6 => Scalar::F(f64::from(f32::from_le_bytes(read_n::<_, 4>(r)?))),
        7 => Scalar::Bool(read_n::<_, 1>(r)?[0] != 0),
        8 => Scalar::Str(read_string(r)?),
        9 => {
            let has_reasoning_token = skip_array(r, scan_vocabulary)?;
            return Ok(if scan_vocabulary {
                ArrayOrScalar::ScannedVocabulary { has_reasoning_token }
            } else {
                ArrayOrScalar::Skipped
            });
        }
        10 => Scalar::U(read_u64(r)?),
        11 => Scalar::I(i64::from_le_bytes(read_n::<_, 8>(r)?)),
        12 => Scalar::F(f64::from_le_bytes(read_n::<_, 8>(r)?)),
        other => bail!("unknown GGUF value type {other}"),
    };
    Ok(ArrayOrScalar::Scalar(scalar))
}

/// Byte width of a fixed-size value type, or `None` for strings and arrays.
fn scalar_width(value_type: u32) -> Option<u64> {
    match value_type {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

/// Steps over an array, returning whether a reasoning token was seen in it.
///
/// The return is always `false` unless `scan` was asked for. Scanning still
/// does not build the vocabulary: only entries short enough to *be* a tag are
/// read, and everything else is seeked over exactly as before. A real
/// vocabulary is around 130k entries and this parse runs for every model on the
/// shelf at startup, so the cost of the scan is a bounded read of a few dozen
/// bytes per short token, not a 2 MB allocation.
fn skip_array<R: Read + Seek>(r: &mut R, scan: bool) -> Result<bool> {
    let element_type = read_u32(r)?;
    let len = read_u64(r)?;

    match scalar_width(element_type) {
        Some(width) => {
            let bytes = width
                .checked_mul(len)
                .ok_or_else(|| anyhow!("GGUF array length {len} overflows"))?;
            seek_forward(r, bytes)?;
            Ok(false)
        }
        // Strings are variable-length, so each has to be stepped over.
        None if element_type == 8 => {
            let mut found = false;
            let mut token = Vec::new();
            for _ in 0..len {
                let bytes = read_u64(r)?;
                if bytes > MAX_STRING_BYTES {
                    bail!("GGUF string of {bytes} bytes is not credible");
                }
                // `<antthinking` is the longest opener at twelve bytes. A
                // token longer than this ceiling cannot be one of them, and is
                // stepped over unread like every other entry.
                if scan && !found && bytes <= MAX_REASONING_TOKEN_BYTES {
                    token.clear();
                    token.resize(bytes as usize, 0);
                    r.read_exact(&mut token)
                        .context("GGUF header ended inside the vocabulary")?;
                    // Lossy: a token that is not valid UTF-8 is not a tag.
                    let text = String::from_utf8_lossy(&token);
                    found = REASONING_OPENERS
                        .iter()
                        .any(|opener| text.starts_with(opener));
                } else {
                    seek_forward(r, bytes)?;
                }
            }
            Ok(found)
        }
        None if element_type == 9 => bail!("nested GGUF arrays are not supported"),
        None => bail!("unknown GGUF array element type {element_type}"),
    }
}

/// Longest reasoning opener plus room for `>` and an attribute or two.
///
/// A bound rather than a guess: it is what keeps the vocabulary scan from
/// reading a 130k-entry token list into memory.
const MAX_REASONING_TOKEN_BYTES: u64 = 32;

fn seek_forward<R: Seek>(r: &mut R, bytes: u64) -> Result<()> {
    let offset = i64::try_from(bytes)
        .map_err(|_| anyhow!("GGUF skip of {bytes} bytes overflows a file offset"))?;
    r.seek(SeekFrom::Current(offset))
        .context("GGUF header ended unexpectedly")?;
    Ok(())
}

fn read_n<R: Read, const N: usize>(r: &mut R) -> Result<[u8; N]> {
    let mut buf = [0u8; N];
    r.read_exact(&mut buf).context("GGUF header ended unexpectedly")?;
    Ok(buf)
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    Ok(u32::from_le_bytes(read_n::<_, 4>(r)?))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    Ok(u64::from_le_bytes(read_n::<_, 8>(r)?))
}

fn read_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64(r)?;
    if len > MAX_STRING_BYTES {
        bail!("GGUF string of {len} bytes is not credible");
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf).context("GGUF header ended unexpectedly")?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // ─── Synthetic header construction ──────────────────────────────────────

    fn gguf_string(s: &str) -> Vec<u8> {
        let mut out = (s.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(s.as_bytes());
        out
    }

    fn kv_str(key: &str, value: &str) -> Vec<u8> {
        let mut out = gguf_string(key);
        out.extend_from_slice(&8u32.to_le_bytes());
        out.extend(gguf_string(value));
        out
    }

    fn kv_u32(key: &str, value: u32) -> Vec<u8> {
        let mut out = gguf_string(key);
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&value.to_le_bytes());
        out
    }

    fn kv_u64(key: &str, value: u64) -> Vec<u8> {
        let mut out = gguf_string(key);
        out.extend_from_slice(&10u32.to_le_bytes());
        out.extend_from_slice(&value.to_le_bytes());
        out
    }

    /// A string array, as the tokenizer vocabulary is stored.
    fn kv_string_array(key: &str, values: &[&str]) -> Vec<u8> {
        let mut out = gguf_string(key);
        out.extend_from_slice(&9u32.to_le_bytes());
        out.extend_from_slice(&8u32.to_le_bytes());
        out.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in values {
            out.extend(gguf_string(v));
        }
        out
    }

    /// A reasoning switch is the model's own statement, not a guess from its
    /// name.
    ///
    /// The previous answer to "does this model reason?" was a regular
    /// expression over the model id. It said nothing about a model nobody had
    /// added to the pattern, and it kept saying yes about a fine-tune whose
    /// template no longer had the switch. The template is the model telling us
    /// directly.
    #[test]
    fn a_template_that_branches_on_enable_thinking_reports_the_switch() {
        let mut reader = header(vec![
            kv_str("general.architecture", "llama"),
            kv_u32("llama.block_count", 32),
            kv_str(
                "tokenizer.chat_template",
                "{% if enable_thinking %}<think>{% endif %}",
            ),
        ]);
        let meta = parse_gguf_metadata(&mut reader).expect("parses");
        assert!(meta.supports_toggled_reasoning);
        assert!(meta.emits_reasoning, "a switchable model still reasons");
    }

    /// The case that suppressed streaming everywhere.
    ///
    /// A DeepSeek-R1 distill reasons on every turn and has no switch to do it
    /// with. Read only for the switch it looks like a model that never
    /// reasons, which set `thinkingLevel: "off"`, which made the partitioner
    /// hold the visible answer back until the turn ended.
    #[test]
    fn a_model_that_always_reasons_is_recognised_even_with_no_switch() {
        let mut reader = header(vec![
            kv_str("general.architecture", "qwen2"),
            kv_u32("qwen2.block_count", 28),
            kv_str(
                "tokenizer.chat_template",
                "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}<think>",
            ),
        ]);
        let meta = parse_gguf_metadata(&mut reader).expect("parses");
        assert!(
            !meta.supports_toggled_reasoning,
            "there is no switch, so the kwarg must not be sent"
        );
        assert!(
            meta.emits_reasoning,
            "it reasons on every turn, so its reasoning must be forwarded and its \
             answer must stream"
        );
    }

    /// Every opener the runtime's partitioner strips must be recognised here.
    ///
    /// The two lists are the same list, held in two languages, and the failure
    /// when they disagree is silent: the partitioner removes a tag this side
    /// never declared, the transport drops what it removed, and the visible
    /// answer is held back with it until the turn ends. Which is a run that
    /// produces nothing for four minutes and is stopped as stuck.
    #[test]
    fn every_reasoning_opener_the_partitioner_knows_is_recognised_here() {
        for opener in ["<think>", "<thinking>", "<thought>", "<reasoning>", "<internal>"] {
            let mut reader = header(vec![
                kv_str("general.architecture", "llama"),
                kv_u32("llama.block_count", 32),
                kv_str("tokenizer.chat_template", &format!("{{{{ x }}}}{opener}")),
            ]);
            let meta = parse_gguf_metadata(&mut reader).expect("parses");
            assert!(
                meta.emits_reasoning,
                "{opener} is stripped by the partitioner but was not declared as reasoning"
            );
        }
    }

    /// A tag carrying attributes is still that tag.
    #[test]
    fn an_opener_with_attributes_is_still_a_reasoning_opener() {
        let mut reader = header(vec![
            kv_str("general.architecture", "llama"),
            kv_u32("llama.block_count", 32),
            kv_str("tokenizer.chat_template", "{{ x }}<thinking level=\"high\">"),
        ]);
        let meta = parse_gguf_metadata(&mut reader).expect("parses");
        assert!(meta.emits_reasoning);
    }

    #[test]
    fn a_template_without_the_switch_reports_none() {
        let mut reader = header(vec![
            kv_str("general.architecture", "gemma3"),
            kv_u32("gemma3.block_count", 48),
            kv_str("tokenizer.chat_template", "{{ messages[0].content }}"),
        ]);
        let meta = parse_gguf_metadata(&mut reader).expect("parses");
        assert!(!meta.supports_toggled_reasoning);
        assert!(
            !meta.emits_reasoning,
            "a template mentioning neither is a model that does not reason"
        );
    }

    /// The case that shipped broken: a reasoning model with no chat template.
    ///
    /// NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf, on this machine, has **no**
    /// `tokenizer.chat_template` key at all \u2014 llama.cpp guesses a profile for
    /// it \u2014 while carrying `<think>` and `</think>` in
    /// `tokenizer.ggml.tokens` as special tokens. It reasons on every turn:
    /// answering "hi" cost 67 output tokens for a nine-token reply.
    ///
    /// Reading only the template made `emits_reasoning` false, which set
    /// `thinkingLevel: "off"` in the runtime, which dropped every reasoning
    /// delta *and* held the visible answer back with them. The operator saw no
    /// thinking and no streaming \u2014 the whole reply appeared at once after
    /// twelve seconds of blank space.
    ///
    /// A model cannot emit a token that is not in its vocabulary, and a
    /// vocabulary does not carry `<think>` by accident.
    #[test]
    fn reasoning_tokens_in_the_vocabulary_are_a_reasoning_model() {
        let mut reader = header(vec![
            kv_str("general.architecture", "nemotron_h"),
            kv_u32("nemotron_h.block_count", 42),
            kv_string_array(
                "tokenizer.ggml.tokens",
                &["[INST]", "<|im_start|>", "<think>", "</think>", "<tool_call>"],
            ),
        ]);
        let meta = parse_gguf_metadata(&mut reader).expect("parses");
        assert!(
            !meta.supports_toggled_reasoning,
            "there is no switch, and saying there is would send a kwarg its template cannot read"
        );
        assert!(
            meta.emits_reasoning,
            "a vocabulary carrying <think> is a model that reasons, template or no template"
        );
    }

    /// The vocabulary is scanned, not loaded. A real one is 130k entries and
    /// this runs on every model on the shelf at startup.
    #[test]
    fn scanning_the_vocabulary_does_not_read_past_it() {
        let mut reader = header(vec![
            kv_str("general.architecture", "llama"),
            kv_string_array("tokenizer.ggml.tokens", &["<think>", "a", "b"]),
            kv_u32("llama.block_count", 32),
        ]);
        let meta = parse_gguf_metadata(&mut reader).expect("parses");
        assert!(meta.emits_reasoning);
        assert_eq!(
            meta.block_count, 32,
            "the key after the vocabulary must still be found"
        );
    }

    /// An ordinary vocabulary says nothing either way.
    #[test]
    fn a_vocabulary_without_reasoning_tokens_reports_none() {
        let mut reader = header(vec![
            kv_str("general.architecture", "llama"),
            kv_u32("llama.block_count", 32),
            kv_string_array("tokenizer.ggml.tokens", &["hello", "world", "<|end|>"]),
        ]);
        let meta = parse_gguf_metadata(&mut reader).expect("parses");
        assert!(!meta.emits_reasoning);
    }

    /// A header with no template at all is not a reasoning model. Absent has
    /// to read as "no", or every model whose header this cannot parse would be
    /// asked for a reasoning block it will never send.
    #[test]
    fn a_header_with_no_template_reports_no_switch() {
        let mut reader = header(vec![
            kv_str("general.architecture", "llama"),
            kv_u32("llama.block_count", 32),
        ]);
        let meta = parse_gguf_metadata(&mut reader).expect("parses");
        assert!(!meta.supports_toggled_reasoning);
        assert!(!meta.emits_reasoning);
    }

    fn header(entries: Vec<Vec<u8>>) -> Cursor<Vec<u8>> {
        let mut out = GGUF_MAGIC.to_vec();
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        out.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for e in entries {
            out.extend(e);
        }
        Cursor::new(out)
    }

    /// gpt-oss-20b's real geometry: 24 layers, 8 KV heads, 64-wide K and V.
    fn gpt_oss_entries() -> Vec<Vec<u8>> {
        vec![
            kv_str("general.architecture", "gpt-oss"),
            kv_u64("general.parameter_count", 20_900_000_000),
            kv_u32("gpt-oss.block_count", 24),
            kv_u32("gpt-oss.embedding_length", 2880),
            kv_u32("gpt-oss.attention.head_count", 64),
            kv_u32("gpt-oss.attention.head_count_kv", 8),
            kv_u32("gpt-oss.attention.key_length", 64),
            kv_u32("gpt-oss.attention.value_length", 64),
            kv_u32("gpt-oss.expert_count", 32),
            kv_u32("gpt-oss.expert_used_count", 4),
            kv_u32("gpt-oss.expert_feed_forward_length", 2880),
        ]
    }

    // ─── Tests ──────────────────────────────────────────────────────────────

    #[test]
    fn reads_the_geometry_a_moe_plan_needs() {
        let meta = parse_gguf_metadata(&mut header(gpt_oss_entries())).unwrap();

        assert_eq!(meta.architecture, "gpt-oss");
        assert_eq!(meta.block_count, 24, "the planner otherwise assumes 32");
        assert_eq!(meta.expert_count, 32);
        assert!(meta.is_moe());
    }

    /// The figure the size-banded estimate gets 5× wrong for MoE.
    #[test]
    fn kv_cost_per_token_is_exact() {
        let meta = parse_gguf_metadata(&mut header(gpt_oss_entries())).unwrap();

        // 24 layers × 8 KV heads × (64 + 64) × 2 bytes
        assert_eq!(meta.kv_bytes_per_token(), 49_152);

        // At the working context this is a few hundred MB, not the ~2 GB the
        // banded estimate would charge a 4 GB card.
        assert!(meta.kv_bytes_per_token() * 8192 < 512 * 1024 * 1024);
    }

    /// The window is read from the file, and its absence is reported as absence.
    ///
    /// The registry recorded a flat 8192 for every model on disk before this key
    /// was parsed, which is the window the context meter showed and the window
    /// the agent loop compacted against. Returning `None` rather than a default
    /// keeps the choice of fallback at the call site, where it can be explained.
    #[test]
    fn the_trained_context_length_is_read_and_its_absence_is_not_invented() {
        let stated = parse_gguf_metadata(&mut header(vec![
            kv_str("general.architecture", "llama"),
            kv_u32("llama.block_count", 32),
            kv_u32("llama.embedding_length", 4096),
            kv_u32("llama.attention.head_count", 32),
            kv_u32("llama.context_length", 131_072),
        ]))
        .unwrap();
        assert_eq!(stated.context_length, Some(131_072));

        let silent = parse_gguf_metadata(&mut header(vec![
            kv_str("general.architecture", "llama"),
            kv_u32("llama.block_count", 32),
            kv_u32("llama.embedding_length", 4096),
            kv_u32("llama.attention.head_count", 32),
        ]))
        .unwrap();
        assert_eq!(
            silent.context_length, None,
            "a converter that wrote no key must not be reported as having stated one"
        );
    }

    #[test]
    fn a_dense_model_reports_no_experts() {
        let meta = parse_gguf_metadata(&mut header(vec![
            kv_str("general.architecture", "llama"),
            kv_u32("llama.block_count", 32),
            kv_u32("llama.embedding_length", 4096),
            kv_u32("llama.attention.head_count", 32),
            kv_u32("llama.attention.head_count_kv", 8),
        ]))
        .unwrap();

        assert!(!meta.is_moe());
        assert_eq!(meta.expert_params(), 0);
        assert_eq!(meta.expert_bytes(8 * 1024 * 1024 * 1024, None), 0);
    }

    #[test]
    fn head_dimension_defaults_to_embedding_over_heads_when_absent() {
        // llama.cpp applies the same default; 4096 / 32 = 128.
        let meta = parse_gguf_metadata(&mut header(vec![
            kv_str("general.architecture", "llama"),
            kv_u32("llama.block_count", 32),
            kv_u32("llama.embedding_length", 4096),
            kv_u32("llama.attention.head_count", 32),
            kv_u32("llama.attention.head_count_kv", 8),
        ]))
        .unwrap();

        assert_eq!(meta.key_length, 128);
        assert_eq!(meta.value_length, 128);
        assert_eq!(meta.kv_bytes_per_token(), 32 * 8 * 256 * 2);
    }

    #[test]
    fn expert_bytes_are_a_share_of_the_real_file_size() {
        let meta = parse_gguf_metadata(&mut header(gpt_oss_entries())).unwrap();
        let file_bytes = 12_800_000_000u64;

        let experts = meta.expert_bytes(file_bytes, None);

        // Experts dominate a MoE file, but never all of it — the attention stack
        // has to stay resident for the split to be worth making.
        assert!(experts > file_bytes / 2, "got {experts} of {file_bytes}");
        assert!(experts < file_bytes, "got {experts} of {file_bytes}");
    }

    #[test]
    fn expert_share_is_capped_when_the_geometry_disagrees_with_the_total() {
        // A parameter_count far too small would otherwise claim the entire file
        // is experts, leaving nothing resident.
        let mut entries = gpt_oss_entries();
        entries[1] = kv_u64("general.parameter_count", 1_000_000);

        let meta = parse_gguf_metadata(&mut header(entries)).unwrap();
        let file_bytes = 12_800_000_000u64;

        assert!(meta.expert_bytes(file_bytes, None) <= (file_bytes as f64 * 0.95) as u64);
    }

    #[test]
    fn a_caller_supplied_total_wins_over_the_header() {
        let meta = parse_gguf_metadata(&mut header(gpt_oss_entries())).unwrap();
        let file_bytes = 12_800_000_000u64;

        let from_header = meta.expert_bytes(file_bytes, None);
        let from_caller = meta.expert_bytes(file_bytes, Some(20_900_000_000 * 2));

        assert!(
            from_caller < from_header,
            "doubling the total should halve the expert share ({from_caller} vs {from_header})"
        );
    }

    /// The vocabulary is 128k+ strings; reading it would defeat the point of
    /// touching only the header.
    #[test]
    fn string_arrays_are_stepped_over_rather_than_read() {
        let mut entries = gpt_oss_entries();
        entries.insert(
            1,
            kv_string_array("tokenizer.ggml.tokens", &["hello", "world", "<|end|>"]),
        );

        let meta = parse_gguf_metadata(&mut header(entries)).unwrap();

        assert_eq!(meta.block_count, 24, "keys after the array must still be found");
        assert_eq!(meta.expert_count, 32);
    }

    #[test]
    fn numeric_arrays_are_stepped_over_too() {
        let mut entries = gpt_oss_entries();
        let mut arr = gguf_string("tokenizer.ggml.token_type");
        arr.extend_from_slice(&9u32.to_le_bytes());
        arr.extend_from_slice(&5u32.to_le_bytes()); // INT32 elements
        arr.extend_from_slice(&4u64.to_le_bytes());
        arr.extend_from_slice(&[0u8; 16]);
        entries.insert(1, arr);

        let meta = parse_gguf_metadata(&mut header(entries)).unwrap();

        assert_eq!(meta.block_count, 24);
    }

    /// Only a fraction of the experts fires per token — the reason offloading
    /// them to system RAM is affordable at all.
    #[test]
    fn active_params_count_only_the_experts_the_router_uses() {
        let meta = parse_gguf_metadata(&mut header(gpt_oss_entries())).unwrap();
        let total = 20_900_000_000u64;

        let active = meta.active_params(Some(total)).unwrap();

        assert!(active < total / 2, "4 of 32 experts fire, got {active} of {total}");
        assert!(active > 0);
    }

    #[test]
    fn a_dense_model_has_every_parameter_active() {
        let meta = parse_gguf_metadata(&mut header(vec![
            kv_str("general.architecture", "llama"),
            kv_u32("llama.block_count", 32),
        ]))
        .unwrap();

        assert_eq!(meta.active_params(Some(8_000_000_000)), Some(8_000_000_000));
    }

    #[test]
    fn active_params_are_unknown_without_a_total() {
        let mut entries = gpt_oss_entries();
        entries.remove(1); // general.parameter_count

        let meta = parse_gguf_metadata(&mut header(entries)).unwrap();
        assert_eq!(meta.active_params(None), None);
    }

    #[test]
    fn a_file_that_is_not_gguf_is_rejected() {
        let mut cursor = Cursor::new(b"NOTGGUF and then some".to_vec());

        let err = parse_gguf_metadata(&mut cursor).unwrap_err().to_string();
        assert!(err.contains("not a GGUF file"), "got: {err}");
    }

    #[test]
    fn a_truncated_header_is_an_error_rather_than_a_panic() {
        let full = header(gpt_oss_entries()).into_inner();

        for cut in [2, 8, 20, 40, 60] {
            let mut cursor = Cursor::new(full[..cut.min(full.len())].to_vec());
            assert!(
                parse_gguf_metadata(&mut cursor).is_err(),
                "a header cut at {cut} bytes must not parse"
            );
        }
    }

    #[test]
    fn a_header_without_an_architecture_is_rejected() {
        let mut cursor = header(vec![kv_u32("llama.block_count", 32)]);

        let err = parse_gguf_metadata(&mut cursor).unwrap_err().to_string();
        assert!(err.contains("general.architecture"), "got: {err}");
    }

    #[test]
    fn a_header_without_a_block_count_is_rejected() {
        // Without it there is no per-layer expert size, so no N can be computed.
        let mut cursor = header(vec![kv_str("general.architecture", "llama")]);

        let err = parse_gguf_metadata(&mut cursor).unwrap_err().to_string();
        assert!(err.contains("block_count"), "got: {err}");
    }

    #[test]
    fn an_absurd_metadata_count_is_refused_before_looping() {
        let mut out = GGUF_MAGIC.to_vec();
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&u64::MAX.to_le_bytes());

        let err = parse_gguf_metadata(&mut Cursor::new(out)).unwrap_err().to_string();
        assert!(err.contains("not credible"), "got: {err}");
    }

    #[test]
    fn version_1_is_refused_rather_than_misread() {
        let mut out = GGUF_MAGIC.to_vec();
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());

        let err = parse_gguf_metadata(&mut Cursor::new(out)).unwrap_err().to_string();
        assert!(err.contains("version 1"), "got: {err}");
    }

    #[test]
    fn every_scalar_type_round_trips() {
        // A converter may store block_count as any integer width.
        for (type_tag, bytes) in [
            (0u32, vec![24u8]),
            (2, 24u16.to_le_bytes().to_vec()),
            (4, 24u32.to_le_bytes().to_vec()),
            (5, 24i32.to_le_bytes().to_vec()),
            (10, 24u64.to_le_bytes().to_vec()),
            (11, 24i64.to_le_bytes().to_vec()),
        ] {
            let mut entry = gguf_string("llama.block_count");
            entry.extend_from_slice(&type_tag.to_le_bytes());
            entry.extend_from_slice(&bytes);

            let meta = parse_gguf_metadata(&mut header(vec![
                kv_str("general.architecture", "llama"),
                entry,
            ]))
            .unwrap();

            assert_eq!(meta.block_count, 24, "value type {type_tag} did not decode");
        }
    }
}

/// What a model's header says about how it must be run.
///
/// The two facts the serving path needs, and the two it used to guess: how
/// many layers there are to offload, and whether the model has a reasoning
/// switch to honour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ModelCapabilities {
    /// Transformer blocks, or `None` when the header could not be read.
    ///
    /// `None` is not zero and not a default. The planner scales the offload
    /// fraction by this count, so a made-up number silently leaves layers on
    /// the CPU that the plan believed were on the GPU — the planner is told
    /// "unknown" and applies its own documented assumption instead.
    pub layers: Option<u32>,
    /// Whether the chat template branches on `enable_thinking`.
    pub supports_toggled_reasoning: bool,
    /// Whether the model produces a reasoning block at all. See
    /// [`GgufMetadata::emits_reasoning`] — this is the one that decides
    /// whether the answer streams.
    pub emits_reasoning: bool,
    /// The trained context window, where the header states one.
    pub context_length: Option<u32>,
}

/// Reads a model's capabilities once per file, then remembers them.
///
/// Memoised because both the admission path and the run-parameter builder need
/// the same answers on every turn, and the answers cannot change: a GGUF
/// header is fixed for the life of the file. Without this, asking twice per
/// message meant opening a multi-gigabyte file twice per message to re-read a
/// few kilobytes that had not moved.
///
/// A file that cannot be read is remembered as unreadable rather than retried,
/// for the same reason — a missing or corrupt model does not become readable
/// by being opened again mid-conversation, and the caller's fallbacks are
/// documented for exactly this case.
pub fn capabilities(weights: &Path) -> ModelCapabilities {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<HashMap<PathBuf, ModelCapabilities>>> =
        std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));

    if let Ok(guard) = cache.lock() {
        if let Some(hit) = guard.get(weights) {
            return *hit;
        }
    }

    let found = match read_gguf_metadata(weights) {
        Ok(meta) => ModelCapabilities {
            layers: Some(meta.block_count).filter(|count| *count > 0),
            supports_toggled_reasoning: meta.supports_toggled_reasoning,
            emits_reasoning: meta.emits_reasoning,
            context_length: meta.context_length,
        },
        Err(error) => {
            log::warn!(
                "[gguf] {} could not be read, so its layer count and reasoning support are \
                 unknown and the defaults apply: {error:#}",
                weights.display()
            );
            ModelCapabilities::default()
        }
    };

    if let Ok(mut guard) = cache.lock() {
        guard.insert(weights.to_path_buf(), found);
    }
    found
}

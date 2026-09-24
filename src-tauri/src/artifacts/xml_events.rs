//! A pull scanner over the XML inside an Office package.
//!
//! ## Why not a crate
//!
//! The same reason [`super::svg_validate`] gives: this product ships an SBOM
//! into air-gapped deployments, and the subset an OOXML part uses — elements,
//! attributes, text, the five predefined entities and numeric character
//! references, the odd comment or processing instruction — is small enough to
//! scan in a page of code a reviewer can read.
//!
//! ## What it refuses
//!
//! A `<!DOCTYPE`. OOXML never carries one, and a document type declaration is
//! where entity expansion lives — the "billion laughs" and external-entity
//! attacks both start there. A part that declares one is refused rather than
//! expanded, which is the only safe reading of a file this product did not
//! write.
//!
//! Anything malformed is an error naming the byte offset. A scanner that
//! guessed its way past a broken tag would report content the file does not
//! hold.

/// One thing the scanner saw.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// An opening tag. `empty` is true for `<a/>`, which has no `End`.
    Start {
        name: String,
        attributes: Vec<(String, String)>,
        empty: bool,
    },
    End { name: String },
    /// Character data, entity references already decoded.
    Text(String),
}

impl Event {
    /// The attribute's decoded value, when this is a start tag carrying it.
    pub fn attribute(&self, key: &str) -> Option<&str> {
        match self {
            Event::Start { attributes, .. } => attributes
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str()),
            _ => None,
        }
    }

    /// The local name — `w:p` → `p` — for a start or end tag.
    pub fn local_name(&self) -> Option<&str> {
        match self {
            Event::Start { name, .. } | Event::End { name } => Some(local(name)),
            Event::Text(_) => None,
        }
    }
}

/// The part of a qualified name after its prefix.
pub fn local(name: &str) -> &str {
    name.rsplit_once(':').map(|(_, rest)| rest).unwrap_or(name)
}

/// Decodes `&amp;`, `&lt;`, `&gt;`, `&quot;`, `&apos;` and `&#..;` / `&#x..;`.
///
/// An unknown named entity is an error: OOXML defines none, and one here means
/// the text relies on a declaration this scanner refuses to read.
pub fn unescape(raw: &str) -> Result<String, String> {
    if !raw.contains('&') {
        return Ok(raw.to_string());
    }
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        let after = &rest[at + 1..];
        let end = after
            .find(';')
            .ok_or_else(|| "an entity reference is not terminated".to_string())?;
        let name = &after[..end];
        match name {
            "amp" => out.push('&'),
            "lt" => out.push('<'),
            "gt" => out.push('>'),
            "quot" => out.push('"'),
            "apos" => out.push('\''),
            numeric if numeric.starts_with('#') => {
                let code = if let Some(hex) = numeric[1..].strip_prefix(['x', 'X']) {
                    u32::from_str_radix(hex, 16)
                } else {
                    numeric[1..].parse::<u32>()
                }
                .map_err(|_| format!("&{numeric}; is not a character reference"))?;
                out.push(
                    char::from_u32(code)
                        .ok_or_else(|| format!("&{numeric}; is not a character"))?,
                );
            }
            other => return Err(format!("the entity &{other}; is not defined")),
        }
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// Scans a whole part into events.
///
/// Well-formedness is checked as it goes: every end tag must close the element
/// that is open, and every element opened must be closed by the end.
pub fn scan(text: &str) -> Result<Vec<Event>, String> {
    Ok(scan_spans(text)?.into_iter().map(|spanned| spanned.event).collect())
}

/// One event and the bytes of the source it came from.
///
/// For a tag, `start..end` covers the whole tag including `<` and `>`. For
/// text, it covers the raw (still escaped) characters — which is what a
/// targeted edit splices, so every byte outside it survives unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spanned {
    pub event: Event,
    pub start: usize,
    pub end: usize,
}

/// [`scan`], keeping where each event was in the source.
pub fn scan_spans(text: &str) -> Result<Vec<Spanned>, String> {
    let bytes = text.as_bytes();
    let mut events = Vec::new();
    let mut open: Vec<String> = Vec::new();
    let mut i = 0usize;
    let mut text_start = 0usize;

    let flush = |events: &mut Vec<Spanned>, from: usize, to: usize| -> Result<(), String> {
        if to > from {
            let raw = &text[from..to];
            if !raw.is_empty() {
                events.push(Spanned {
                    event: Event::Text(unescape(raw).map_err(|e| format!("at byte {from}: {e}"))?),
                    start: from,
                    end: to,
                });
            }
        }
        Ok(())
    };

    while i < bytes.len() {
        if bytes[i] != b'<' {
            i += 1;
            continue;
        }
        flush(&mut events, text_start, i)?;
        let rest = &text[i..];
        if rest.starts_with("<?") {
            let end = rest
                .find("?>")
                .ok_or_else(|| format!("a processing instruction at byte {i} never ends"))?;
            i += end + 2;
        } else if rest.starts_with("<!--") {
            let end = rest
                .find("-->")
                .ok_or_else(|| format!("a comment at byte {i} never ends"))?;
            i += end + 3;
        } else if rest.starts_with("<![CDATA[") {
            let end = rest
                .find("]]>")
                .ok_or_else(|| format!("a CDATA section at byte {i} never ends"))?;
            events.push(Spanned { event: Event::Text(rest[9..end].to_string()), start: i, end: i + end + 3 });
            i += end + 3;
        } else if rest.starts_with("<!") {
            // DOCTYPE and anything else declarative. See the module note.
            return Err(format!(
                "a document type declaration at byte {i} was refused: OOXML never carries one, \
                 and it is where entity expansion attacks start"
            ));
        } else {
            let end = find_tag_end(rest).ok_or_else(|| format!("a tag at byte {i} never ends"))?;
            let inner = &rest[1..end];
            if let Some(name) = inner.strip_prefix('/') {
                let name = name.trim();
                match open.pop() {
                    Some(expected) if expected == name => {}
                    Some(expected) => {
                        return Err(format!(
                            "</{name}> at byte {i} closes <{expected}>, which is not the element open"
                        ))
                    }
                    None => return Err(format!("</{name}> at byte {i} closes nothing")),
                }
                events.push(Spanned { event: Event::End { name: name.to_string() }, start: i, end: i + end + 1 });
            } else {
                let empty = inner.ends_with('/');
                let body = if empty { &inner[..inner.len() - 1] } else { inner };
                let (name, attributes) =
                    parse_tag(body).map_err(|e| format!("in the tag at byte {i}: {e}"))?;
                if !empty {
                    open.push(name.clone());
                }
                events.push(Spanned { event: Event::Start { name, attributes, empty }, start: i, end: i + end + 1 });
            }
            i += end + 1;
        }
        text_start = i;
    }
    flush(&mut events, text_start, bytes.len())?;
    if let Some(unclosed) = open.last() {
        return Err(format!("<{unclosed}> is never closed"));
    }
    Ok(events)
}

/// The index of the `>` that ends the tag at the start of `rest`, skipping any
/// `>` inside a quoted attribute value.
fn find_tag_end(rest: &str) -> Option<usize> {
    let mut quote: Option<u8> = None;
    for (index, byte) in rest.bytes().enumerate().skip(1) {
        match (quote, byte) {
            (Some(q), b) if b == q => quote = None,
            (Some(_), _) => {}
            (None, b'"') | (None, b'\'') => quote = Some(byte),
            (None, b'>') => return Some(index),
            (None, b'<') => return None,
            _ => {}
        }
    }
    None
}

fn parse_tag(body: &str) -> Result<(String, Vec<(String, String)>), String> {
    let body = body.trim();
    let name_end = body
        .find(|c: char| c.is_whitespace())
        .unwrap_or(body.len());
    let name = body[..name_end].to_string();
    if name.is_empty() {
        return Err("the tag has no name".to_string());
    }
    let mut attributes = Vec::new();
    let mut rest = body[name_end..].trim_start();
    while !rest.is_empty() {
        let eq = rest
            .find('=')
            .ok_or_else(|| format!("the attribute {:?} has no value", rest))?;
        let key = rest[..eq].trim().to_string();
        let after = rest[eq + 1..].trim_start();
        let quote = after
            .chars()
            .next()
            .filter(|c| *c == '"' || *c == '\'')
            .ok_or_else(|| format!("the value of {key} is not quoted"))?;
        let close = after[1..]
            .find(quote)
            .ok_or_else(|| format!("the value of {key} is not closed"))?;
        let value = unescape(&after[1..1 + close])?;
        attributes.push((key, value));
        rest = after[close + 2..].trim_start();
    }
    Ok((name, attributes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_paragraph_reads_as_start_text_end() {
        let events = scan(r#"<?xml version="1.0"?><w:p a="1"><w:t xml:space="preserve">A &amp; B</w:t><w:br/></w:p>"#)
            .expect("scans");
        // p, t, text, /t, br (empty), /p
        assert_eq!(events.len(), 6);
        assert_eq!(events[0].local_name(), Some("p"));
        assert_eq!(events[0].attribute("a"), Some("1"));
        assert_eq!(events[2], Event::Text("A & B".to_string()));
        assert!(matches!(&events[3], Event::End { name } if name == "w:t"));
        assert!(matches!(&events[4], Event::Start { empty: true, .. }));
        assert!(matches!(&events[5], Event::End { name } if name == "w:p"));
    }

    #[test]
    fn a_doctype_is_refused_rather_than_expanded() {
        let error = scan(r#"<!DOCTYPE lol [<!ENTITY a "aaaa">]><r>&a;</r>"#).expect_err("refused");
        assert!(error.contains("document type"), "{error}");
    }

    #[test]
    fn a_mismatched_end_tag_is_an_error() {
        assert!(scan("<a><b></a></b>").is_err());
        assert!(scan("<a>").is_err());
        assert!(scan("</a>").is_err());
    }

    #[test]
    fn character_references_decode_and_unknown_entities_do_not() {
        assert_eq!(unescape("&#65;&#x42;").unwrap(), "AB");
        assert!(unescape("&nbsp;").is_err());
    }

    #[test]
    fn a_gt_inside_an_attribute_does_not_end_the_tag() {
        let events = scan(r#"<a title="x > y">t</a>"#).expect("scans");
        assert_eq!(events[0].attribute("title"), Some("x > y"));
    }
}

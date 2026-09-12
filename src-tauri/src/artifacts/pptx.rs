//! The briefing deck — the same content, in the form a review meeting uses.
//!
//! The deck is the artifact people are most tempted to let a model write freely,
//! and the one where that goes worst: slides invite confident phrasing, and a
//! bullet has no room for the hedging that makes a claim honest. So the deck is
//! the most tightly templated of the three. The section order is fixed, the
//! evidence slide is not optional, and bullets are capped — a slide that has run
//! out of room says how many points it left behind rather than silently dropping
//! them.
//!
//! PowerPoint is stricter than Word about what it will open: a deck needs a
//! slide master, a layout and a theme even when nothing references them. They
//! are written here in their minimal valid form, once, and shared by every deck.

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::ooxml::{escape, placeholders_in, read_part, write_parts};

/// The most bullets one slide carries before the rest are summarised away.
///
/// Six is what fits at a readable size on a projector. The cap exists so a long
/// findings list degrades into "and 4 further findings — see the note" rather
/// than into six-point type nobody in the room can read.
const BULLETS_PER_SLIDE: usize = 6;

/// One slide's worth of content. The model supplies these; it does not decide
/// how many there are or what order they come in.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Slide {
    pub heading: String,
    pub bullets: Vec<String>,
}

/// The fixed shape of a briefing deck.
///
/// Every deck has these sections in this order. `Evidence` is last because it is
/// what a reviewer turns back to, and required because a deck without it is an
/// assertion rather than a briefing.
pub const BRIEFING_SECTIONS: &[&str] = &["Findings", "Recommendation", "Assumptions", "Evidence"];

/// What could not be built, before a file existed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeckError {
    pub message: String,
    pub missing: Vec<String>,
}

fn text_body(heading: &str, bullets: &[String], overflow: usize) -> String {
    let mut paragraphs = String::new();

    for bullet in bullets {
        paragraphs.push_str(&format!(
            "<a:p><a:r><a:rPr lang=\"en-IN\" sz=\"1800\" dirty=\"0\"/><a:t>{}</a:t></a:r></a:p>",
            escape(bullet)
        ));
    }

    if overflow > 0 {
        paragraphs.push_str(&format!(
            "<a:p><a:r><a:rPr lang=\"en-IN\" sz=\"1400\" i=\"1\" dirty=\"0\"/><a:t>and {overflow} \
             further point{} — see the accompanying note</a:t></a:r></a:p>",
            if overflow == 1 { "" } else { "s" }
        ));
    }

    format!(
        r#"<p:sp><p:nvSpPr><p:cNvPr id="2" name="Title"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="628650" y="365125"/><a:ext cx="10515600" cy="1325563"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-IN" sz="3200" b="1" dirty="0"/><a:t>{}</a:t></a:r></a:p></p:txBody></p:sp>
<p:sp><p:nvSpPr><p:cNvPr id="3" name="Body"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="628650" y="1825625"/><a:ext cx="10515600" cy="4351338"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/>{paragraphs}</p:txBody></p:sp>"#,
        escape(heading)
    )
}

fn slide_xml(heading: &str, bullets: &[String], overflow: usize) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/>{}</p:spTree></p:cSld>
<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"#,
        text_body(heading, bullets, overflow)
    )
}

const SLIDE_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>"#;

const SLIDE_LAYOUT: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
<p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree></p:cSld>
<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>"#;

const SLIDE_LAYOUT_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>"#;

const SLIDE_MASTER: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:cSld><p:bg><p:bgPr><a:solidFill><a:schemeClr val="bg1"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>
<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree></p:cSld>
<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
</p:sldMaster>"#;

const SLIDE_MASTER_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>"#;

/// The minimum theme PowerPoint will accept. Nothing here is a design choice —
/// the deck's appearance comes from the slides.
const THEME: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="ARJUN">
<a:themeElements>
<a:clrScheme name="ARJUN"><a:dk1><a:sysClr val="windowText" lastClr="000000"/></a:dk1><a:lt1><a:sysClr val="window" lastClr="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="1B1B1B"/></a:dk2><a:lt2><a:srgbClr val="F2F2F2"/></a:lt2><a:accent1><a:srgbClr val="C8892B"/></a:accent1><a:accent2><a:srgbClr val="8C6A2F"/></a:accent2><a:accent3><a:srgbClr val="4F5B62"/></a:accent3><a:accent4><a:srgbClr val="6B7B85"/></a:accent4><a:accent5><a:srgbClr val="9AA5AC"/></a:accent5><a:accent6><a:srgbClr val="3A3A3A"/></a:accent6><a:hlink><a:srgbClr val="0563C1"/></a:hlink><a:folHlink><a:srgbClr val="954F72"/></a:folHlink></a:clrScheme>
<a:fontScheme name="ARJUN"><a:majorFont><a:latin typeface="Calibri Light"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont><a:minorFont><a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme>
<a:fmtScheme name="ARJUN">
<a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>
<a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="12700"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="19050"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>
<a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>
<a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>
</a:fmtScheme></a:themeElements></a:theme>"#;

fn presentation_xml(slide_count: usize) -> String {
    let ids: String = (0..slide_count)
        .map(|index| {
            format!(
                "<p:sldId id=\"{}\" r:id=\"rId{}\"/>",
                256 + index,
                index + 1
            )
        })
        .collect();

    format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId{master}"/></p:sldMasterIdLst>
<p:sldIdLst>{ids}</p:sldIdLst>
<p:sldSz cx="12192000" cy="6858000"/><p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>"#,
        master = slide_count + 1
    )
}

fn presentation_rels(slide_count: usize) -> String {
    let mut relationships = String::new();
    for index in 0..slide_count {
        relationships.push_str(&format!(
            "<Relationship Id=\"rId{}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide\" Target=\"slides/slide{}.xml\"/>",
            index + 1,
            index + 1
        ));
    }
    relationships.push_str(&format!(
        "<Relationship Id=\"rId{}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster\" Target=\"slideMasters/slideMaster1.xml\"/>",
        slide_count + 1
    ));
    relationships.push_str(&format!(
        "<Relationship Id=\"rId{}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme\" Target=\"theme/theme1.xml\"/>",
        slide_count + 2
    ));

    format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{relationships}</Relationships>"#
    )
}

fn content_types(slide_count: usize) -> String {
    let slides: String = (1..=slide_count)
        .map(|n| format!("<Override PartName=\"/ppt/slides/slide{n}.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.slide+xml\"/>"))
        .collect();

    format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
{slides}</Types>"#
    )
}

const ROOT_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>"#;

/// Renders a [`crate::artifacts::doc_model::Deck`] into a `.pptx`.
///
/// ## Why this exists beside `write_deck`
///
/// `write_deck` renders one deck: the four sections in [`BRIEFING_SECTIONS`],
/// in that order, and it refuses anything else. Every presentation this product
/// has produced is Findings / Recommendation / Assumptions / Evidence. A
/// shutdown briefing, a vendor comparison or a training pack cannot be
/// expressed at all.
///
/// This renders a deck the model composed. The narrative is its own, and the
/// density limits that `write_deck` enforced by silently truncating to
/// `BULLETS_PER_SLIDE` are enforced in the model instead — where an overflowing
/// slide is *split* rather than cut short, so nothing is lost.
///
/// The title slide is still built here rather than supplied, so classification
/// and draft standing appear on every deck whatever the model wrote.
pub fn write_deck_model(
    path: &Path,
    deck: &crate::artifacts::doc_model::Deck,
    is_draft: bool,
) -> Result<(), DeckError> {
    let problems = deck.problems();
    if !problems.is_empty() {
        return Err(DeckError {
            message: format!(
                "The deck is not fit to render: {}. Nothing was written.",
                problems.join("; ")
            ),
            missing: Vec::new(),
        });
    }

    let mut cover = vec![format!("Classification: {}", deck.classification)];
    if is_draft {
        cover.insert(
            0,
            "DRAFT — not verified. Do not act on this deck until it has been reviewed.".to_string(),
        );
    }

    // (heading, bullets, notes). Overflow is zero throughout: the model's own
    // repair splits a long slide, so nothing reaches here that does not fit.
    let mut slides: Vec<(String, Vec<String>, Option<String>)> =
        vec![(deck.title.clone(), cover, None)];

    for slide in &deck.slides {
        let mut bullets = slide.bullets.clone();
        // A table on a slide is rendered as its rows. A real graphic frame
        // would be better and is a larger piece of work; this is honest about
        // being text, and it is legible.
        if let Some(crate::artifacts::doc_model::Block::Table { header, rows, caption }) =
            &slide.table
        {
            if let Some(caption) = caption.as_deref().filter(|c| !c.trim().is_empty()) {
                bullets.push(caption.to_string());
            }
            bullets.push(header.join(" | "));
            for row in rows {
                bullets.push(row.join(" | "));
            }
        }
        slides.push((slide.heading.clone(), bullets, slide.notes.clone()));
    }

    let count = slides.len();
    let notes: Vec<(usize, String)> = slides
        .iter()
        .enumerate()
        .filter_map(|(index, (_, _, notes))| {
            notes
                .as_deref()
                .filter(|n| !n.trim().is_empty())
                .map(|n| (index + 1, n.to_string()))
        })
        .collect();

    let mut parts: Vec<(String, String)> = vec![
        ("[Content_Types].xml".to_string(), model_content_types(count, &notes)),
        ("_rels/.rels".to_string(), ROOT_RELS.to_string()),
        ("ppt/presentation.xml".to_string(), presentation_xml(count)),
        ("ppt/_rels/presentation.xml.rels".to_string(), presentation_rels(count)),
        ("ppt/slideMasters/slideMaster1.xml".to_string(), SLIDE_MASTER.to_string()),
        (
            "ppt/slideMasters/_rels/slideMaster1.xml.rels".to_string(),
            SLIDE_MASTER_RELS.to_string(),
        ),
        ("ppt/slideLayouts/slideLayout1.xml".to_string(), SLIDE_LAYOUT.to_string()),
        (
            "ppt/slideLayouts/_rels/slideLayout1.xml.rels".to_string(),
            SLIDE_LAYOUT_RELS.to_string(),
        ),
        ("ppt/theme/theme1.xml".to_string(), THEME.to_string()),
    ];

    for (index, (heading, bullets, _)) in slides.iter().enumerate() {
        let n = index + 1;
        parts.push((format!("ppt/slides/slide{n}.xml"), slide_xml(heading, bullets, 0)));
        // A slide carrying notes needs a relationship to the notes part, so
        // the two rels differ by slide.
        let has_notes = notes.iter().any(|(slide, _)| *slide == n);
        parts.push((
            format!("ppt/slides/_rels/slide{n}.xml.rels"),
            if has_notes {
                model_slide_rels_with_notes(n)
            } else {
                SLIDE_RELS.to_string()
            },
        ));
    }

    // Speaker notes: where the detail that will not fit on the slide belongs.
    for (slide, text) in &notes {
        parts.push((format!("ppt/notesSlides/notesSlide{slide}.xml"), notes_xml(text)));
        parts.push((
            format!("ppt/notesSlides/_rels/notesSlide{slide}.xml.rels"),
            model_notes_rels(*slide),
        ));
    }

    let borrowed: Vec<(&str, String)> =
        parts.iter().map(|(name, body)| (name.as_str(), body.clone())).collect();
    write_parts(path, &borrowed).map_err(|e| DeckError {
        message: format!("The deck could not be written: {e}"),
        missing: Vec::new(),
    })
}

fn model_content_types(slides: usize, notes: &[(usize, String)]) -> String {
    let extra: String = notes
        .iter()
        .map(|(slide, _)| {
            format!(
                "<Override PartName=\"/ppt/notesSlides/notesSlide{slide}.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml\"/>"
            )
        })
        .collect();
    content_types(slides).replace("</Types>", &format!("{extra}</Types>"))
}

fn model_slide_rels_with_notes(slide: usize) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide{slide}.xml"/>
</Relationships>"#
    )
}

fn model_notes_rels(slide: usize) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="../slides/slide{slide}.xml"/>
</Relationships>"#
    )
}

fn notes_xml(text: &str) -> String {
    let paragraphs: String = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| format!("<a:p><a:r><a:t>{}</a:t></a:r></a:p>", escape(line)))
        .collect();
    format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:notes xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:cSld><p:spTree>
<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
<p:grpSpPr/>
<p:sp><p:nvSpPr><p:cNvPr id="2" name="Notes Placeholder"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr>
<p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/>{paragraphs}</p:txBody></p:sp>
</p:spTree></p:cSld></p:notes>"#
    )
}

/// Writes a briefing deck: a title slide, then one slide per fixed section.
///
/// `sections` supplies bullets by heading. A missing required section is an
/// error before a file exists — the same rule the approval note follows, for the
/// same reason.
pub fn write_deck(
    path: &Path,
    title: &str,
    classification: &str,
    sections: &[Slide],
    is_draft: bool,
) -> Result<(), DeckError> {
    if title.trim().is_empty() {
        return Err(DeckError {
            message: "A deck needs a title. Nothing was written.".to_string(),
            missing: vec!["title".to_string()],
        });
    }

    let missing: Vec<String> = BRIEFING_SECTIONS
        .iter()
        .filter(|required| {
            !sections
                .iter()
                .any(|s| s.heading.eq_ignore_ascii_case(required) && !s.bullets.is_empty())
        })
        .map(|s| s.to_string())
        .collect();

    if !missing.is_empty() {
        return Err(DeckError {
            message: format!(
                "The briefing template requires {} that {} not supplied: {}. Nothing was written.",
                if missing.len() == 1 { "a section" } else { "sections" },
                if missing.len() == 1 { "was" } else { "were" },
                missing.join(", ")
            ),
            missing,
        });
    }

    // The title slide is built here, not supplied, so classification and draft
    // standing appear on every deck whatever the model wrote.
    let mut cover = vec![format!("Classification: {classification}")];
    if is_draft {
        cover.insert(
            0,
            "DRAFT — not verified. Do not act on this deck until it has been reviewed.".to_string(),
        );
    }

    let mut slides: Vec<(String, Vec<String>, usize)> = vec![(title.to_string(), cover, 0)];

    for required in BRIEFING_SECTIONS {
        let Some(section) = sections
            .iter()
            .find(|s| s.heading.eq_ignore_ascii_case(required))
        else {
            continue;
        };

        let overflow = section.bullets.len().saturating_sub(BULLETS_PER_SLIDE);
        let shown: Vec<String> = section.bullets.iter().take(BULLETS_PER_SLIDE).cloned().collect();
        slides.push((section.heading.clone(), shown, overflow));
    }

    let count = slides.len();
    let mut parts: Vec<(String, String)> = vec![
        ("[Content_Types].xml".to_string(), content_types(count)),
        ("_rels/.rels".to_string(), ROOT_RELS.to_string()),
        ("ppt/presentation.xml".to_string(), presentation_xml(count)),
        ("ppt/_rels/presentation.xml.rels".to_string(), presentation_rels(count)),
        ("ppt/slideMasters/slideMaster1.xml".to_string(), SLIDE_MASTER.to_string()),
        (
            "ppt/slideMasters/_rels/slideMaster1.xml.rels".to_string(),
            SLIDE_MASTER_RELS.to_string(),
        ),
        ("ppt/slideLayouts/slideLayout1.xml".to_string(), SLIDE_LAYOUT.to_string()),
        (
            "ppt/slideLayouts/_rels/slideLayout1.xml.rels".to_string(),
            SLIDE_LAYOUT_RELS.to_string(),
        ),
        ("ppt/theme/theme1.xml".to_string(), THEME.to_string()),
    ];

    for (index, (heading, bullets, overflow)) in slides.iter().enumerate() {
        let n = index + 1;
        parts.push((
            format!("ppt/slides/slide{n}.xml"),
            slide_xml(heading, bullets, *overflow),
        ));
        parts.push((
            format!("ppt/slides/_rels/slide{n}.xml.rels"),
            SLIDE_RELS.to_string(),
        ));
    }

    let borrowed: Vec<(&str, String)> =
        parts.iter().map(|(name, body)| (name.as_str(), body.clone())).collect();

    write_parts(path, &borrowed).map_err(|e| DeckError {
        message: format!("The deck could not be written: {e}"),
        missing: Vec::new(),
    })
}

/// What re-opening a produced deck found.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeckCheck {
    pub opens: bool,
    pub slides: usize,
    /// Section headings found, in order.
    pub headings: Vec<String>,
    pub problems: Vec<String>,
}

impl DeckCheck {
    pub fn is_sound(&self) -> bool {
        self.opens && self.problems.is_empty()
    }
}

/// Re-opens a produced deck and checks it is what it claims to be.
pub fn check_deck(path: &Path) -> DeckCheck {
    let presentation = match read_part(path, "ppt/presentation.xml") {
        Ok(body) => body,
        Err(error) => {
            return DeckCheck {
                opens: false,
                slides: 0,
                headings: Vec::new(),
                problems: vec![format!("{}: {error}", path.display())],
            }
        }
    };

    let mut problems = Vec::new();
    let slides = presentation.matches("<p:sldId ").count();

    for required in [
        "ppt/slideMasters/slideMaster1.xml",
        "ppt/slideLayouts/slideLayout1.xml",
        "ppt/theme/theme1.xml",
    ] {
        if read_part(path, required).is_err() {
            problems.push(format!("the deck is missing {required} — PowerPoint will refuse it"));
        }
    }

    let mut headings = Vec::new();
    for n in 1..=slides {
        match read_part(path, &format!("ppt/slides/slide{n}.xml")) {
            Ok(slide) => {
                if let Some(heading) = first_run(&slide) {
                    headings.push(heading);
                }
                for marker in placeholders_in(&slide) {
                    problems.push(format!(
                        "slide {n} still contains the placeholder {marker:?}"
                    ));
                }
            }
            Err(_) => problems.push(format!("slide {n} is referenced but not present")),
        }
    }

    // The title slide is index 0; the sections follow it in fixed order.
    for required in BRIEFING_SECTIONS {
        if !headings.iter().any(|h| h.eq_ignore_ascii_case(required)) {
            problems.push(format!("the {required:?} slide is missing"));
        }
    }

    DeckCheck { opens: true, slides, headings, problems }
}

fn first_run(slide: &str) -> Option<String> {
    let start = slide.find("<a:t>")?;
    let rest = &slide[start + 5..];
    let end = rest.find("</a:t>")?;
    Some(rest[..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    fn section(heading: &str, bullets: &[&str]) -> Slide {
        Slide {
            heading: heading.to_string(),
            bullets: bullets.iter().map(|b| b.to_string()).collect(),
        }
    }

    fn complete() -> Vec<Slide> {
        vec![
            section("Findings", &["Measured 8.2 mm against a minimum of 9.0 mm [E1]."]),
            section("Recommendation", &["Replace PV-2201 within 90 days."]),
            section("Assumptions", &["None."]),
            section("Evidence", &["Maintenance SOP rev C, page 4."]),
        ]
    }

    #[test]
    fn a_complete_deck_is_written_and_reopens_soundly() {
        let dir = temp();
        let path = dir.path().join("brief.pptx");

        write_deck(&path, "PV-2201 wall thickness", "Inspection report", &complete(), false)
            .unwrap();

        let check = check_deck(&path);
        assert!(check.is_sound(), "{:?}", check.problems);
        // Title slide plus the four fixed sections.
        assert_eq!(check.slides, 5);
        assert_eq!(check.headings[0], "PV-2201 wall thickness");
    }

    /// A deck without evidence is an assertion rather than a briefing.
    #[test]
    fn a_missing_section_refuses_and_writes_nothing() {
        let dir = temp();
        let path = dir.path().join("brief.pptx");

        let sections: Vec<Slide> =
            complete().into_iter().filter(|s| s.heading != "Evidence").collect();

        let error = write_deck(&path, "T", "C", &sections, false).unwrap_err();
        assert_eq!(error.missing, vec!["Evidence"]);
        assert!(!path.exists());
    }

    #[test]
    fn a_section_present_but_empty_counts_as_missing() {
        let dir = temp();
        let mut sections = complete();
        sections[2].bullets.clear();

        let error =
            write_deck(&dir.path().join("b.pptx"), "T", "C", &sections, false).unwrap_err();
        assert_eq!(error.missing, vec!["Assumptions"]);
    }

    #[test]
    fn a_deck_without_a_title_refuses() {
        let dir = temp();
        let error =
            write_deck(&dir.path().join("b.pptx"), "   ", "C", &complete(), false).unwrap_err();
        assert_eq!(error.missing, vec!["title"]);
    }

    /// Sections come out in the template's order, not the caller's.
    #[test]
    fn section_order_is_the_templates_not_the_callers() {
        let dir = temp();
        let path = dir.path().join("brief.pptx");

        let mut shuffled = complete();
        shuffled.reverse();

        write_deck(&path, "T", "C", &shuffled, false).unwrap();
        let check = check_deck(&path);
        assert_eq!(&check.headings[1..], BRIEFING_SECTIONS);
    }

    /// A long list degrades into a stated count, not six-point type.
    #[test]
    fn a_slide_that_runs_out_of_room_says_how_many_points_it_left_behind() {
        let dir = temp();
        let path = dir.path().join("brief.pptx");

        let many: Vec<&str> = vec!["finding"; 9];
        let mut sections = complete();
        sections[0] = section("Findings", &many);

        write_deck(&path, "T", "C", &sections, false).unwrap();

        let slide = read_part(&path, "ppt/slides/slide2.xml").unwrap();
        assert!(slide.contains("and 3 further points"));
    }

    #[test]
    fn one_point_over_reads_as_singular() {
        let dir = temp();
        let path = dir.path().join("brief.pptx");

        let many: Vec<&str> = vec!["finding"; 7];
        let mut sections = complete();
        sections[0] = section("Findings", &many);

        write_deck(&path, "T", "C", &sections, false).unwrap();
        let slide = read_part(&path, "ppt/slides/slide2.xml").unwrap();
        assert!(slide.contains("and 1 further point —"));
    }

    #[test]
    fn a_draft_deck_carries_its_warning_on_the_title_slide() {
        let dir = temp();
        let path = dir.path().join("brief.pptx");

        write_deck(&path, "T", "C", &complete(), true).unwrap();
        let cover = read_part(&path, "ppt/slides/slide1.xml").unwrap();
        assert!(cover.contains("DRAFT — not verified"));
    }

    #[test]
    fn every_deck_states_its_classification_whatever_the_model_wrote() {
        let dir = temp();
        let path = dir.path().join("brief.pptx");

        write_deck(&path, "T", "P&ID / process diagram", &complete(), false).unwrap();
        let cover = read_part(&path, "ppt/slides/slide1.xml").unwrap();
        assert!(cover.contains("Classification: P&amp;ID / process diagram"));
    }

    #[test]
    fn model_content_containing_xml_still_produces_a_readable_deck() {
        let dir = temp();
        let path = dir.path().join("brief.pptx");

        let mut sections = complete();
        sections[0] = section("Findings", &["thickness < 9.0 mm & \"severe\" <a:t>injected</a:t>"]);

        write_deck(&path, "T & <U>", "C", &sections, false).unwrap();
        assert!(check_deck(&path).is_sound());
    }

    #[test]
    fn a_placeholder_left_on_a_slide_fails_the_post_render_check() {
        let dir = temp();
        let path = dir.path().join("brief.pptx");

        let mut sections = complete();
        sections[1] = section("Recommendation", &["TBD"]);

        write_deck(&path, "T", "C", &sections, false).unwrap();

        let check = check_deck(&path);
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("placeholder")));
    }

    #[test]
    fn a_file_that_is_not_a_deck_is_reported_rather_than_panicking() {
        let dir = temp();
        let path = dir.path().join("broken.pptx");
        std::fs::write(&path, "not a zip").unwrap();

        let check = check_deck(&path);
        assert!(!check.opens);
        assert!(!check.is_sound());
    }
}

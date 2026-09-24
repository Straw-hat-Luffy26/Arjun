#!/usr/bin/env python3
"""Labelled documents for the Document & Vision Analyst (plan P06).

Writes eight documents into a directory and prints one JSON manifest on stdout
naming, for each, where every labelled string actually is: its page, its box in
that page's own coordinate space, and its text. The boxes are measured by
PyMuPDF from the page that was drawn (`search_for` on the source page), not
typed in by hand, so a test comparing an extracted box against them is
comparing against where the ink is.

    python p06_fixtures.py <out_dir>

Documents
---------
typed.pdf        two typed pages with a text layer, and a ruled table
scan.pdf         two pages with no text layer: the same kind of content, as images
skewed.pdf       one scanned page rotated by SKEW degrees
handwriting.png  jittered italic writing, plus a scribble nothing can read
table-scan.pdf   one scanned page holding only a ruled table
photo.jpg        a JPEG "photograph" of an equipment nameplate
pid.png          a P&ID sketch with three instrument tags, one of them smudged
long.pdf         one scanned page used for the looping-output case

Coordinates: `pdf-points` for a PDF page (the scan pages are image-only pages
of the same point size as the page they were drawn from), `image-pixels` for
PNG and JPEG. The same conventions `render_pages.py --layout` reports.

Nothing here reaches the network. PyMuPDF only.
"""

import json
import math
import os
import random
import sys

SKEW = 4.0
SCAN_SCALE = 2.0  # 144 dpi, which is what a desk scanner's "text" mode gives


def load():
    try:
        import pymupdf as fitz  # noqa: F401
    except Exception:  # noqa: BLE001
        import fitz  # noqa: F401
    return fitz


def truth_of(page, strings, scale=1.0):
    found = []
    for text in strings:
        rects = page.search_for(text)
        if not rects:
            raise SystemExit("fixture text not found on its own page: %r" % text)
        r = rects[0]
        found.append({"text": text, "bbox": [round(v * scale, 2) for v in (r.x0, r.y0, r.x1, r.y1)]})
    return found


def draw_table(fitz, page, x, y, rows, widths, height=22):
    shape = page.new_shape()
    total = sum(widths)
    for r in range(len(rows) + 1):
        shape.draw_line((x, y + r * height), (x + total, y + r * height))
    cx = x
    shape.draw_line((cx, y), (cx, y + len(rows) * height))
    for w in widths:
        cx += w
        shape.draw_line((cx, y), (cx, y + len(rows) * height))
    shape.finish(color=(0, 0, 0), width=0.8)
    shape.commit()
    for r, row in enumerate(rows):
        cx = x
        for c, text in enumerate(row):
            page.insert_text((cx + 5, y + r * height + 15), text, fontsize=10, fontname="helv")
            cx += widths[c]


def typed_page_one(fitz, doc, tag, pressure):
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 80), "INSPECTION REPORT", fontsize=18, fontname="helv")
    page.insert_text((72, 120), "Tag: %s" % tag, fontsize=12, fontname="helv")
    page.insert_text((72, 140), "Design pressure: %s" % pressure, fontsize=12, fontname="helv")
    page.insert_text((72, 160), "Inspector: A. Rao", fontsize=12, fontname="helv")
    return page


def image_only_copy(fitz, source_doc, out_path, rotate=0.0):
    """Every page of `source_doc` as a picture on an otherwise empty page."""
    scan = fitz.open()
    for page in source_doc:
        matrix = fitz.Matrix(SCAN_SCALE, SCAN_SCALE)
        if rotate:
            matrix = matrix.prerotate(rotate)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        target = scan.new_page(width=page.rect.width, height=page.rect.height)
        target.insert_image(target.rect, pixmap=pixmap, keep_proportion=False)
    scan.save(out_path, deflate=True, garbage=3)
    scan.close()


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    fitz = load()
    manifest = {"generator": "p06_fixtures.py", "pymupdf": str(getattr(fitz, "VersionBind", "?")), "documents": {}}

    # -- typed.pdf ---------------------------------------------------------
    doc = fitz.open()
    p1 = typed_page_one(fitz, doc, "PT-2201", "10 bar")
    draw_table(fitz, p1, 72, 200, [["Point", "mm", "Status"], ["A", "9.4", "ok"], ["C", "8.2", "below"]], [120, 80, 100])
    # Measured before the next page exists: PyMuPDF retires a page object once
    # another page is added.
    truth = [dict(page=1, **t) for t in truth_of(p1, ["Tag: PT-2201", "Design pressure: 10 bar", "9.4"])]
    p2 = doc.new_page(width=595, height=842)
    p2.insert_text((72, 90), "Material: SS316", fontsize=12, fontname="helv")
    p2.insert_text((72, 110), "Line size: 4 in", fontsize=12, fontname="helv")
    truth += [dict(page=2, **t) for t in truth_of(p2, ["Material: SS316"])]
    doc.save(os.path.join(out, "typed.pdf"))
    doc.close()
    manifest["documents"]["typed"] = {"file": "typed.pdf", "pages": 2, "coordSpace": "pdf-points", "truth": truth}

    # -- scan.pdf ----------------------------------------------------------
    source = fitz.open()
    s1 = typed_page_one(fitz, source, "PT-3301", "16 bar")
    truth = [dict(page=1, **t) for t in truth_of(s1, ["INSPECTION REPORT", "Tag: PT-3301", "Design pressure: 16 bar"])]
    s2 = source.new_page(width=595, height=842)
    s2.insert_text((72, 90), "Material: CS A106", fontsize=12, fontname="helv")
    truth += [dict(page=2, **t) for t in truth_of(s2, ["Material: CS A106"])]
    image_only_copy(fitz, source, os.path.join(out, "scan.pdf"))
    source.close()
    manifest["documents"]["scan"] = {"file": "scan.pdf", "pages": 2, "coordSpace": "pdf-points", "truth": truth}

    # -- skewed.pdf --------------------------------------------------------
    source = fitz.open()
    k1 = typed_page_one(fitz, source, "PT-4401", "25 bar")
    for i in range(20):
        k1.insert_text((72, 200 + i * 22), "Weld %02d visually inspected, no indication recorded." % (i + 1), fontsize=11, fontname="helv")
    truth = [dict(page=1, **t) for t in truth_of(k1, ["Tag: PT-4401"])]
    image_only_copy(fitz, source, os.path.join(out, "skewed.pdf"), rotate=SKEW)
    source.close()
    manifest["documents"]["skewed"] = {
        "file": "skewed.pdf", "pages": 1, "coordSpace": "pdf-points", "skewDegrees": SKEW,
        "truth": truth, "note": "truth is the unrotated position; the page is rotated about its origin before scanning",
    }

    # -- handwriting.png ---------------------------------------------------
    rng = random.Random(26117)
    source = fitz.open()
    h = source.new_page(width=420, height=220)
    x = 30
    for ch in "Checked by R. Menon":
        y = 70 + rng.uniform(-2.5, 2.5)
        h.insert_text((x, y), ch, fontsize=20, fontname="tiit", morph=(fitz.Point(x, y), fitz.Matrix(rng.uniform(-6, 6))))
        x += 11 if ch != " " else 7
    x = 30
    for ch in "Date 12/09":
        y = 120 + rng.uniform(-2.5, 2.5)
        h.insert_text((x, y), ch, fontsize=20, fontname="tiit", morph=(fitz.Point(x, y), fitz.Matrix(rng.uniform(-6, 6))))
        x += 11 if ch != " " else 7
    scribble = h.new_shape()
    points = [fitz.Point(240 + i * 6, 170 + 10 * math.sin(i * 1.7) + rng.uniform(-4, 4)) for i in range(25)]
    scribble.draw_polyline(points)
    scribble.finish(color=(0.1, 0.1, 0.3), width=2)
    scribble.commit()
    scale = 3.0
    h.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).save(os.path.join(out, "handwriting.png"))
    manifest["documents"]["handwriting"] = {
        "file": "handwriting.png", "pages": 1, "coordSpace": "image-pixels", "size": [int(420 * scale), int(220 * scale)],
        "truth": [
            {"page": 1, "text": "Checked by R. Menon", "bbox": [round(v * scale, 2) for v in (28, 50, 245, 78)]},
            {"page": 1, "text": "Date 12/09", "bbox": [round(v * scale, 2) for v in (28, 100, 140, 128)]},
            {"page": 1, "text": None, "bbox": [round(v * scale, 2) for v in (236, 150, 392, 190)], "unreadable": True},
        ],
    }
    source.close()

    # -- table-scan.pdf ----------------------------------------------------
    source = fitz.open()
    t = source.new_page(width=595, height=842)
    t.insert_text((72, 80), "INSULATION RESISTANCE", fontsize=14, fontname="helv")
    draw_table(fitz, t, 72, 110, [["Phase", "Reading", "Limit"], ["R", "412 MOhm", "100 MOhm"], ["Y", "388 MOhm", "100 MOhm"]], [110, 120, 120])
    truth = [dict(page=1, **x) for x in truth_of(t, ["Phase", "388 MOhm"])]
    truth.append({"page": 1, "text": "table", "bbox": [72.0, 110.0, 422.0, 176.0]})
    image_only_copy(fitz, source, os.path.join(out, "table-scan.pdf"))
    source.close()
    manifest["documents"]["table"] = {"file": "table-scan.pdf", "pages": 1, "coordSpace": "pdf-points", "truth": truth}

    # -- photo.jpg ---------------------------------------------------------
    source = fitz.open()
    ph = source.new_page(width=400, height=300)
    for i in range(60):
        shade = 0.25 + 0.5 * i / 60
        ph.draw_rect(fitz.Rect(0, i * 5, 400, i * 5 + 5), color=None, fill=(shade, shade * 0.95, shade * 0.9))
    ph.draw_rect(fitz.Rect(90, 100, 320, 190), color=(0.2, 0.2, 0.2), fill=(0.85, 0.85, 0.8), width=2)
    ph.insert_text((105, 135), "PUMP P-101", fontsize=18, fontname="helv")
    ph.insert_text((105, 165), "7.5 kW  2900 rpm", fontsize=14, fontname="helv")
    scale = 2.0
    truth = [dict(page=1, **x) for x in truth_of(ph, ["PUMP P-101", "7.5 kW"], scale)]
    ph.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).save(os.path.join(out, "photo.jpg"), jpg_quality=88)
    source.close()
    manifest["documents"]["photo"] = {
        "file": "photo.jpg", "pages": 1, "coordSpace": "image-pixels", "size": [800, 600], "truth": truth,
        "nameplate": [round(v * scale, 2) for v in (90, 100, 320, 190)],
    }

    # -- pid.png -----------------------------------------------------------
    source = fitz.open()
    pid = source.new_page(width=600, height=300)
    line = pid.new_shape()
    line.draw_line((40, 150), (560, 150))
    for cx in (140, 300, 460):
        line.draw_line((cx, 150), (cx, 110))
        line.draw_circle((cx, 90), 20)
    line.finish(color=(0, 0, 0), width=1.5)
    line.commit()
    tags = ["PT-101", "FV-102", "TT-103"]
    for cx, tag in zip((140, 300, 460), tags):
        pid.insert_text((cx - 22, 94), tag, fontsize=10, fontname="helv")
    scale = 2.0
    truth = [dict(page=1, **x) for x in truth_of(pid, tags, scale)]
    # The third tag is smudged: a grey blot over most of it, after the text.
    pid.draw_rect(fitz.Rect(436, 82, 478, 98), color=None, fill=(0.45, 0.45, 0.45))
    pid.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).save(os.path.join(out, "pid.png"))
    for entry in truth:
        if entry["text"] == "TT-103":
            entry["unreadable"] = True
    source.close()
    manifest["documents"]["pid"] = {"file": "pid.png", "pages": 1, "coordSpace": "image-pixels", "size": [1200, 600], "truth": truth}

    # -- long.pdf ----------------------------------------------------------
    source = fitz.open()
    lp = source.new_page(width=595, height=842)
    lp.insert_text((72, 80), "SITE SURVEY", fontsize=16, fontname="helv")
    lp.insert_text((72, 110), "Tag: PT-5501", fontsize=12, fontname="helv")
    truth = [dict(page=1, **x) for x in truth_of(lp, ["SITE SURVEY", "Tag: PT-5501"])]
    image_only_copy(fitz, source, os.path.join(out, "long.pdf"))
    source.close()
    manifest["documents"]["long"] = {"file": "long.pdf", "pages": 1, "coordSpace": "pdf-points", "truth": truth}

    sys.stdout.write(json.dumps(manifest))


if __name__ == "__main__":
    main()

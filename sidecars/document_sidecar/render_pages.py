#!/usr/bin/env python3
"""Rasterise a PDF's pages for visual review, and read what each page says.

Why this exists
---------------
A produced Word document, deck or workbook is validated in stages (see
`src-tauri/src/artifacts/validation.rs`). Re-opening the package proves the
file is well formed; it does not prove it *renders* — that a page is not blank,
that a deck has as many pages as it has slides, that a reviewer looking at it
sees what the XML says. That needs a real layout engine, and then a picture of
what it laid out.

The layout engine is LibreOffice, driven from Rust (`artifacts::render`), which
turns the Office file into a PDF. This script is the second half: it opens that
PDF with PyMuPDF — the library `attachment_extract.py` already uses to render
scanned pages for OCR, so nothing new is installed — and writes one PNG per
page, with each page's text and whether the page is blank.

Nothing here reaches the network. MuPDF resolves no remote resource and runs no
PDF JavaScript while rendering.

Contract
--------
    python render_pages.py --probe
    python render_pages.py <pdf> <out_dir> [--from N] [--to M] [--width W]

Prints one JSON object on stdout.

    --probe      {"renderer": "pymupdf", "version": "1.28.2"}
    render       {"renderer": "pymupdf", "version": "...", "pages": 12,
                  "rendered": [{"page": 1, "image": "page-1.png",
                                "width": 1240, "height": 1754,
                                "blank": false, "text": "..."}]}
    on failure   {"error": "..."}

Exit status: 0 success, 3 PyMuPDF is not installed (the adapter is
*unavailable*, which the caller reports as such), 1 anything else.

At most MAX_PAGES pages are rendered per call; the caller asks for the rest.

Document analysis (P06)
-----------------------
The same engine answers the Document & Vision Analyst's geometry questions, so
nothing new is installed: PyMuPDF only, no PIL, no numpy.

    python render_pages.py --layout <file> [--from N] [--to M]
    python render_pages.py --crop <file> <out_png> --page N [--bbox x0,y0,x1,y1]
                           [--dpi D]
    python render_pages.py --skew <file> --page N
    python render_pages.py --probe-image <out_png> --text TOKEN

`<file>` is a PDF or an image (PNG/JPEG). **Coordinates** are in the page's own
space and every answer names it: `pdf-points` for a PDF page, `image-pixels`
for an image (PyMuPDF sizes an image page at 72/96 of its pixels, which is
converted back here so a box means the pixels a person sees).

    layout  {"pages": N, "coordSpace": ..., "layout": [{"page", "width",
             "height", "rotation", "textChars", "blocks": [{"bbox", "kind",
             "text"}], "tables": [{"bbox", "rows", "cols", "cells":
             [{"row", "col", "bbox", "text"}]}]}]}
    crop    {"page", "image", "width", "height", "bbox", "coordSpace",
             "pixelsPerUnit"}   -- the image is exactly the bbox, rendered
    skew    {"page", "degrees", "method", "darkPixels"}   -- measured by the
             projection profile of dark pixels; "unmeasured" when too few
    probe-image {"image", "width", "height"}

The skew is a measurement of the image, not a correction: nothing here rotates
a page, because the stored original is what every citation points at.
"""

import json
import os
import sys

MAX_PAGES = 50
DEFAULT_WIDTH = 1240
MAX_WIDTH = 2480


def out(payload, code=0):
    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()
    sys.exit(code)


def load():
    try:
        import pymupdf as fitz  # noqa: F401  (1.24+ spelling)
    except Exception:  # noqa: BLE001
        try:
            import fitz  # noqa: F401
        except Exception as error:  # noqa: BLE001
            out({"error": "PyMuPDF is not installed: %s" % error}, 3)
    return fitz


def version_of(fitz):
    version = getattr(fitz, "VersionBind", None) or getattr(fitz, "__version__", None)
    return str(version) if version else "unknown"


def argument(args, name, default):
    if name in args:
        index = args.index(name)
        try:
            return int(args[index + 1])
        except (IndexError, ValueError):
            out({"error": "%s needs a whole number" % name}, 1)
    return default


MAX_CROP_PIXELS = 4000 * 4000
MAX_TABLES_PER_PAGE = 12
MAX_BLOCKS_PER_PAGE = 400


def open_document(fitz, source):
    try:
        document = fitz.open(source)
    except Exception as error:  # noqa: BLE001 - reported, not raised
        out({"error": "the file could not be opened: %s" % error}, 1)
    if document.needs_pass:
        out({"error": "the PDF is password protected"}, 1)
    return document


def space_of(fitz, document, source):
    """The page's coordinate space, and the factor from PyMuPDF's units to it."""
    if document.is_pdf:
        return "pdf-points", 1.0
    # An image: PyMuPDF's rect is the pixel size at 72/96. Measured from the
    # image itself rather than assumed, so a JPEG stored at another resolution
    # still maps to its own pixels.
    try:
        image = fitz.Pixmap(source)
        page = document.load_page(0)
        factor = image.width / page.rect.width if page.rect.width else 1.0
        return "image-pixels", factor
    except Exception:  # noqa: BLE001 - fall back to PyMuPDF's own units
        return "image-points", 1.0


def rounded(values, factor):
    return [round(v * factor, 2) for v in values]


def page_range(args, total):
    first = argument(args, "--from", 1)
    last = argument(args, "--to", min(total, first + MAX_PAGES - 1))
    if first < 1 or last < first:
        out({"error": "the page range must start at 1 and run forward"}, 1)
    return first, min(last, total, first + MAX_PAGES - 1)


def layout(fitz, source, args):
    document = open_document(fitz, source)
    space, factor = space_of(fitz, document, source)
    total = document.page_count
    first, last = page_range(args, total)
    pages = []
    try:
        for number in range(first, last + 1):
            page = document.load_page(number - 1)
            data = page.get_text("dict")
            blocks = []
            chars = 0
            for block in data.get("blocks", [])[:MAX_BLOCKS_PER_PAGE]:
                if block.get("type") == 0:
                    text = "\n".join(
                        "".join(span.get("text", "") for span in line.get("spans", []))
                        for line in block.get("lines", [])
                    ).strip()
                    chars += len(text)
                    if text:
                        blocks.append({"bbox": rounded(block["bbox"], factor), "kind": "text", "text": text})
                else:
                    blocks.append({"bbox": rounded(block["bbox"], factor), "kind": "image", "text": ""})
            tables = []
            if document.is_pdf and hasattr(page, "find_tables"):
                try:
                    # PyMuPDF prints an advisory to stdout from here, and stdout
                    # carries exactly one JSON object; so its chatter goes to
                    # stderr instead.
                    import contextlib
                    with contextlib.redirect_stdout(sys.stderr):
                        found = page.find_tables().tables[:MAX_TABLES_PER_PAGE]
                except Exception:  # noqa: BLE001 - a page without tables is not an error
                    found = []
                for table in found:
                    extracted = table.extract()
                    cells = []
                    for r, row in enumerate(table.rows):
                        for c, cell in enumerate(row.cells):
                            text = ""
                            if r < len(extracted) and c < len(extracted[r]):
                                text = (extracted[r][c] or "").strip()
                            cells.append({
                                "row": r,
                                "col": c,
                                "bbox": rounded(cell, factor) if cell else None,
                                "text": text,
                            })
                    tables.append({
                        "bbox": rounded(table.bbox, factor),
                        "rows": table.row_count,
                        "cols": table.col_count,
                        "cells": cells,
                    })
            pages.append({
                "page": number,
                "width": round(page.rect.width * factor, 2),
                "height": round(page.rect.height * factor, 2),
                "rotation": page.rotation,
                "textChars": chars,
                "blocks": blocks,
                "tables": tables,
            })
    finally:
        document.close()
    out({"renderer": "pymupdf", "version": version_of(fitz), "pages": total, "coordSpace": space, "layout": pages})


def parse_bbox(raw):
    try:
        values = [float(part) for part in raw.split(",")]
    except ValueError:
        out({"error": "--bbox takes four numbers: x0,y0,x1,y1"}, 1)
    if len(values) != 4 or values[2] <= values[0] or values[3] <= values[1]:
        out({"error": "--bbox must be x0,y0,x1,y1 with x1 > x0 and y1 > y0"}, 1)
    return values


def crop(fitz, source, target, args):
    document = open_document(fitz, source)
    space, factor = space_of(fitz, document, source)
    number = argument(args, "--page", 1)
    if number < 1 or number > document.page_count:
        out({"error": "page %d is not in this document of %d page(s)" % (number, document.page_count)}, 1)
    try:
        page = document.load_page(number - 1)
        full = [0.0, 0.0, page.rect.width * factor, page.rect.height * factor]
        if "--bbox" in args:
            box = parse_bbox(args[args.index("--bbox") + 1])
        else:
            box = full
        # Clamped to the page, and said so by the returned bbox.
        box = [max(box[0], 0.0), max(box[1], 0.0), min(box[2], full[2]), min(box[3], full[3])]
        if box[2] <= box[0] or box[3] <= box[1]:
            out({"error": "the region lies outside page %d" % number}, 1)
        if space == "image-pixels":
            # The stored pixels, one for one: an image is not re-sampled.
            pixels_per_unit = 1.0
        else:
            dpi = min(max(argument(args, "--dpi", 200), 72), 400)
            pixels_per_unit = dpi / 72.0
        clip = fitz.Rect(*(v / factor for v in box))
        scale = pixels_per_unit * factor
        width = (box[2] - box[0]) * pixels_per_unit
        height = (box[3] - box[1]) * pixels_per_unit
        if width * height > MAX_CROP_PIXELS:
            out({"error": "that region would be %dx%d pixels, above the %d pixel limit; ask for a smaller region or a lower --dpi" % (width, height, MAX_CROP_PIXELS)}, 1)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
        pixmap.save(target)
        out({
            "page": number,
            "image": os.path.basename(target),
            "width": pixmap.width,
            "height": pixmap.height,
            "bbox": [round(v, 2) for v in box],
            "coordSpace": space,
            "pixelsPerUnit": round(pixmap.width / (box[2] - box[0]), 6),
        })
    finally:
        document.close()


def skew(fitz, source, args):
    """Estimates the page's skew from the projection profile of its dark pixels.

    Downsampled to about 600 pixels wide and sheared rather than rotated: for
    the few degrees a scanner introduces, a shear is the same measurement and
    costs a histogram per angle over the dark pixels only.
    """
    document = open_document(fitz, source)
    number = argument(args, "--page", 1)
    if number < 1 or number > document.page_count:
        out({"error": "page %d is not in this document" % number}, 1)
    try:
        page = document.load_page(number - 1)
        scale = 600.0 / page.rect.width if page.rect.width else 1.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY, alpha=False)
    finally:
        document.close()
    width, height, samples = pixmap.width, pixmap.height, pixmap.samples
    dark = []
    for y in range(height):
        row = samples[y * width:(y + 1) * width]
        for x in range(0, width):
            if row[x] < 110:
                dark.append((x, y))
    if len(dark) < 200:
        out({"page": number, "degrees": None, "method": "unmeasured", "darkPixels": len(dark)})
    import math
    best_angle, best_score = 0.0, -1.0
    steps = [a / 4.0 for a in range(-40, 41)]  # -10..10 degrees, quarter-degree steps
    for angle in steps:
        slope = math.tan(math.radians(angle))
        bins = {}
        for x, y in dark:
            key = int(y - x * slope)
            bins[key] = bins.get(key, 0) + 1
        score = sum(count * count for count in bins.values())
        if score > best_score:
            best_score, best_angle = score, angle
    out({"page": number, "degrees": best_angle, "method": "projection-profile", "darkPixels": len(dark)})


def probe_image(fitz, target, args):
    if "--text" not in args:
        out({"error": "--probe-image needs --text TOKEN"}, 1)
    token = args[args.index("--text") + 1]
    document = fitz.open()
    page = document.new_page(width=520, height=160)
    page.insert_text((24, 100), token, fontsize=44, fontname="helv")
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    pixmap.save(target)
    document.close()
    out({"image": os.path.basename(target), "width": pixmap.width, "height": pixmap.height})


def main():
    args = sys.argv[1:]
    fitz = load()
    if args[:1] == ["--probe"]:
        out({"renderer": "pymupdf", "version": version_of(fitz)})
    if args[:1] == ["--layout"] and len(args) >= 2:
        layout(fitz, args[1], args[2:])
    if args[:1] == ["--crop"] and len(args) >= 3:
        crop(fitz, args[1], args[2], args[3:])
    if args[:1] == ["--skew"] and len(args) >= 2:
        skew(fitz, args[1], args[2:])
    if args[:1] == ["--probe-image"] and len(args) >= 2:
        probe_image(fitz, args[1], args[2:])
    if len(args) < 2:
        out({"error": "usage: render_pages.py <pdf> <out_dir> [--from N] [--to M] [--width W]"}, 1)

    source, out_dir = args[0], args[1]
    first = argument(args, "--from", 1)
    last = argument(args, "--to", first + MAX_PAGES - 1)
    width = min(max(argument(args, "--width", DEFAULT_WIDTH), 200), MAX_WIDTH)
    if first < 1 or last < first:
        out({"error": "the page range must start at 1 and run forward"}, 1)
    last = min(last, first + MAX_PAGES - 1)

    if not os.path.isdir(out_dir):
        out({"error": "the output directory does not exist"}, 1)
    try:
        document = fitz.open(source)
    except Exception as error:  # noqa: BLE001 - reported, not raised
        out({"error": "the PDF could not be opened: %s" % error}, 1)

    try:
        if document.needs_pass:
            out({"error": "the PDF is password protected"}, 1)
        total = document.page_count
        rendered = []
        for number in range(first, min(last, total) + 1):
            page = document.load_page(number - 1)
            rect = page.rect
            scale = width / rect.width if rect.width else 1.0
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            name = "page-%d.png" % number
            pixmap.save(os.path.join(out_dir, name))
            # A page with one colour on it is blank, whatever its text layer
            # claims. Both are computed in C. `color_count()` returns a count
            # by default and a dict only when asked for colours, so the
            # fallback handles both rather than guessing "not blank".
            if hasattr(pixmap, "is_unicolor"):
                blank = bool(pixmap.is_unicolor)
            else:
                counted = pixmap.color_count()
                blank = (len(counted) if isinstance(counted, dict) else int(counted)) <= 1
            rendered.append({
                "page": number,
                "image": name,
                "width": pixmap.width,
                "height": pixmap.height,
                "blank": blank,
                "text": page.get_text("text"),
            })
    finally:
        document.close()

    out({"renderer": "pymupdf", "version": version_of(fitz), "pages": total, "rendered": rendered})


if __name__ == "__main__":
    main()

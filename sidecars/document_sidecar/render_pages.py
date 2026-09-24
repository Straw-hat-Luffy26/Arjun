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


def main():
    args = sys.argv[1:]
    fitz = load()
    if args[:1] == ["--probe"]:
        out({"renderer": "pymupdf", "version": version_of(fitz)})
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

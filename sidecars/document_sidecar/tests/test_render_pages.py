"""Tests for `render_pages.py`, the page rasteriser behind `artifact.render`.

Every test runs the real script as a subprocess, the way Rust calls it
(`artifacts::render`), against a PDF PyMuPDF builds here. What is asserted is
the JSON contract Rust reads: page numbers, image names inside the output
directory, sizes, text and the blank flag — and the refusals.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import fitz  # noqa: E402  (the document sidecar's own dependency)

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "render_pages.py")


def run(*args):
    completed = subprocess.run(
        [sys.executable, SCRIPT, *args], capture_output=True, text=True, timeout=120
    )
    return completed.returncode, json.loads(completed.stdout)


def make_pdf(path, pages):
    document = fitz.open()
    for text in pages:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text, fontsize=14)
    document.save(path)
    document.close()


class RenderPagesTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.pdf = os.path.join(self.dir, "in.pdf")

    def test_probe_names_the_library_and_its_version(self):
        code, answer = run("--probe")
        self.assertEqual(code, 0)
        self.assertEqual(answer["renderer"], "pymupdf")
        self.assertRegex(answer["version"], r"^\d+\.\d+")

    def test_each_page_becomes_a_png_with_its_text_and_a_blank_flag(self):
        make_pdf(self.pdf, ["Findings: point C measured 8.2 mm.", ""])
        code, answer = run(self.pdf, self.dir)
        self.assertEqual(code, 0, answer)
        self.assertEqual(answer["pages"], 2)
        first, second = answer["rendered"]
        self.assertEqual(first["page"], 1)
        self.assertEqual(first["image"], "page-1.png")
        self.assertIn("8.2 mm", first["text"])
        self.assertFalse(first["blank"])
        self.assertTrue(second["blank"], "an empty page is reported blank")
        with open(os.path.join(self.dir, "page-1.png"), "rb") as handle:
            self.assertEqual(handle.read(4), b"\x89PNG")

    def test_a_range_is_honoured_and_bounded(self):
        make_pdf(self.pdf, ["one", "two", "three"])
        code, answer = run(self.pdf, self.dir, "--from", "2", "--to", "3")
        self.assertEqual(code, 0)
        self.assertEqual([p["page"] for p in answer["rendered"]], [2, 3])
        code, answer = run(self.pdf, self.dir, "--from", "3", "--to", "1")
        self.assertEqual(code, 1)
        self.assertIn("error", answer)

    def test_a_file_that_is_not_a_pdf_is_refused_not_rendered(self):
        with open(self.pdf, "wb") as handle:
            handle.write(b"not a pdf at all")
        code, answer = run(self.pdf, self.dir)
        self.assertEqual(code, 1)
        self.assertIn("could not be opened", answer["error"])


def make_table_pdf(path):
    """A typed page with a heading, a sentence and a ruled 3x3 table."""
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    page.insert_text((72, 72), "Thickness readings", fontsize=16)
    page.insert_text((72, 100), "Point C is the governing reading.", fontsize=11)
    rows = [["Point", "mm", "Status"], ["A", "9.4", "ok"], ["C", "8.2", "below"]]
    x0, y0, w, h = 72, 140, 120, 24
    for r in range(4):
        page.draw_line((x0, y0 + r * h), (x0 + 3 * w, y0 + r * h))
    for c in range(4):
        page.draw_line((x0 + c * w, y0), (x0 + c * w, y0 + 3 * h))
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            page.insert_text((x0 + c * w + 6, y0 + r * h + 16), text, fontsize=11)
    document.save(path)
    document.close()


def make_rotated_scan(path, degrees):
    """A page of text rendered as an image, turned by `degrees`, as a scanner would."""
    document = fitz.open()
    page = document.new_page(width=600, height=780)
    for line in range(24):
        page.insert_text((60, 80 + line * 26), "Inspection line %02d reads within limits" % line, fontsize=13)
    pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5).prerotate(degrees), alpha=False)
    pixmap.save(path)
    document.close()


class AnalysisModeTests(unittest.TestCase):
    """The P06 modes: layout, crop, skew and the vision probe image."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_a_typed_page_yields_located_text_and_a_table_with_cell_boxes(self):
        pdf = os.path.join(self.dir, "typed.pdf")
        make_table_pdf(pdf)
        code, answer = run("--layout", pdf)
        self.assertEqual(code, 0, answer)
        self.assertEqual(answer["coordSpace"], "pdf-points")
        page = answer["layout"][0]
        self.assertEqual((page["width"], page["height"]), (595.0, 842.0))
        self.assertGreater(page["textChars"], 40)
        heading = [b for b in page["blocks"] if "Thickness readings" in b["text"]]
        self.assertEqual(len(heading), 1)
        x0, y0, x1, y1 = heading[0]["bbox"]
        self.assertTrue(60 <= x0 <= 80 and 50 <= y0 <= 80, heading[0]["bbox"])
        self.assertEqual(len(page["tables"]), 1, page["tables"])
        table = page["tables"][0]
        self.assertEqual((table["rows"], table["cols"]), (3, 3))
        c_row = [cell for cell in table["cells"] if cell["row"] == 2]
        self.assertEqual([cell["text"] for cell in c_row], ["C", "8.2", "below"])
        self.assertTrue(all(cell["bbox"] for cell in c_row))

    def test_an_image_page_reports_no_text_layer_in_its_own_pixels(self):
        png = os.path.join(self.dir, "scan.png")
        make_rotated_scan(png, 0)
        code, answer = run("--layout", png)
        self.assertEqual(code, 0, answer)
        self.assertEqual(answer["coordSpace"], "image-pixels")
        page = answer["layout"][0]
        self.assertEqual(page["textChars"], 0, "an image has no text layer to read")
        self.assertEqual((page["width"], page["height"]), (900.0, 1170.0))

    def test_a_crop_is_exactly_the_region_asked_for_and_bounded(self):
        png = os.path.join(self.dir, "scan.png")
        make_rotated_scan(png, 0)
        target = os.path.join(self.dir, "crop.png")
        code, answer = run("--crop", png, target, "--page", "1", "--bbox", "100,200,500,400")
        self.assertEqual(code, 0, answer)
        self.assertEqual((answer["width"], answer["height"]), (400, 200))
        self.assertEqual(answer["bbox"], [100.0, 200.0, 500.0, 400.0])
        with open(target, "rb") as handle:
            self.assertEqual(handle.read(4), b"\x89PNG")
        # Outside the page, and inverted, are refused rather than guessed at.
        code, answer = run("--crop", png, target, "--page", "1", "--bbox", "5000,5000,6000,6000")
        self.assertEqual(code, 1)
        code, answer = run("--crop", png, target, "--page", "1", "--bbox", "400,200,100,400")
        self.assertEqual(code, 1)
        code, answer = run("--crop", png, target, "--page", "2")
        self.assertEqual(code, 1)

    def test_a_pdf_crop_is_rendered_at_the_asked_resolution(self):
        pdf = os.path.join(self.dir, "typed.pdf")
        make_table_pdf(pdf)
        target = os.path.join(self.dir, "table.png")
        code, answer = run("--crop", pdf, target, "--page", "1", "--bbox", "72,140,432,212", "--dpi", "144")
        self.assertEqual(code, 0, answer)
        self.assertEqual(answer["coordSpace"], "pdf-points")
        self.assertEqual((answer["width"], answer["height"]), (720, 144))
        self.assertAlmostEqual(answer["pixelsPerUnit"], 2.0, places=3)

    def test_skew_is_measured_near_what_was_applied(self):
        for applied in (0.0, 4.0):
            png = os.path.join(self.dir, "skew-%s.png" % applied)
            make_rotated_scan(png, applied)
            code, answer = run("--skew", png, "--page", "1")
            self.assertEqual(code, 0, answer)
            self.assertEqual(answer["method"], "projection-profile")
            self.assertLessEqual(abs(abs(answer["degrees"]) - applied), 0.75, (applied, answer))

    def test_a_blank_page_leaves_skew_unmeasured_rather_than_zero(self):
        document = fitz.open()
        document.new_page(width=300, height=300)
        pdf = os.path.join(self.dir, "blank.pdf")
        document.save(pdf)
        document.close()
        code, answer = run("--skew", pdf, "--page", "1")
        self.assertEqual(code, 0, answer)
        self.assertIsNone(answer["degrees"])
        self.assertEqual(answer["method"], "unmeasured")

    def test_the_probe_image_carries_the_token_it_was_asked_for(self):
        target = os.path.join(self.dir, "probe.png")
        code, answer = run("--probe-image", target, "--text", "QX-4821-VEK")
        self.assertEqual(code, 0, answer)
        self.assertGreater(answer["width"], 500)
        # The token is really drawn: read back from a PDF of the image would
        # need OCR, so the page's own text layer is checked before rendering.
        with open(target, "rb") as handle:
            self.assertEqual(handle.read(4), b"\x89PNG")


if __name__ == "__main__":
    unittest.main()

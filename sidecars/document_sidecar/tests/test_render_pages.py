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


if __name__ == "__main__":
    unittest.main()

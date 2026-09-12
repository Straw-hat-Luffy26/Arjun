"""Tests for the per-page PDF routing in `attachment_extract`.

The bug these are written against
---------------------------------
`extract_pdf` used to weigh the *whole file's* text against one threshold and
call the result `pdf-text` or `pdf-scan`. Two things followed, both silent:

  - A digital cover sheet in front of scanned pages carried the whole file
    over the line on its own. The scans were never rendered, never OCR'd and
    never mentioned; `truncated` came back `False`.
  - The threshold was `MIN_TEXT_CHARS_PER_PAGE * min(total, 3)`, capped at
    three pages' worth however long the document was. One readable page
    anywhere in a hundred-page scan declared the whole thing text.

Every test below builds a real PDF with PyMuPDF and runs the real extractor.
Nothing is stubbed, because the thing under test is a judgement about what is
actually on a page, and a stub would only assert the judgement back at itself.
"""

import json
import os
import shutil
import subprocess
import struct
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # noqa: E402

from attachment_extract import (  # noqa: E402
    MIN_TEXT_CHARS_PER_PAGE,
    cfb_contents,
    cfb_directory_names,
    extract_pdf,
    ooxml_family,
    refuse_cfb,
    sniff,
)


# Comfortably over the per-page threshold, so a page carrying it is one the
# text layer can be trusted for.
COVER_TEXT = "Q3 Seal Inspection Report for the maintenance review board"

assert len(COVER_TEXT) >= MIN_TEXT_CHARS_PER_PAGE, "the fixture must clear the threshold"


def text_page(doc, body):
    """A digitally produced page: a real text layer, no image."""
    page = doc.new_page()
    page.insert_text((72, 100), body, fontsize=11)


def scanned_page(doc):
    """A scanned page: an image and no text layer at all."""
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, 600, 800), False)
    pix.clear_with(200)
    page.insert_image(page.rect, pixmap=pix)


class PerPageRouting(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.out = os.path.join(self.dir, "out")
        os.makedirs(self.out, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def build(self, *pages):
        """Writes a PDF from a list of "text"/"scan" markers and returns it."""
        doc = fitz.open()
        for index, kind in enumerate(pages):
            if kind == "text":
                text_page(doc, "%s (page %d)" % (COVER_TEXT, index + 1))
            else:
                scanned_page(doc)
        path = os.path.join(self.dir, "doc.pdf")
        doc.save(path)
        doc.close()
        return path

    def extract(self, *pages, **kwargs):
        return extract_pdf(self.build(*pages), self.out, kwargs.get("max_pages", 12))

    # -- The reported failure --------------------------------------------

    def test_a_digital_cover_does_not_swallow_the_scan_behind_it(self):
        """The exact case from the report: page 1 digital, page 2 scanned."""
        result = self.extract("text", "scan")

        self.assertEqual(result["kind"], "pdf-mixed")
        self.assertEqual(result["pages"], 2)
        # The scanned page must actually be queued for the model.
        self.assertEqual(len(result["pageImages"]), 1, "the scanned page was not rendered")
        sources = [d["source"] for d in result["pageDetail"]]
        self.assertEqual(sources, ["text", "ocr"])

    def test_the_rendered_image_is_named_for_its_real_page(self):
        """A citation to page 2 has to mean page 2 of the file."""
        result = self.extract("text", "scan")
        ocr = [d for d in result["pageDetail"] if d["source"] == "ocr"][0]
        self.assertEqual(ocr["page"], 2)
        self.assertTrue(os.path.basename(ocr["image"]).endswith("page-2.png"), ocr["image"])
        self.assertTrue(os.path.exists(ocr["image"]))

    def test_one_readable_page_does_not_declare_a_long_scan_readable(self):
        """The capped threshold, which never grew with the document.

        Twenty pages, one of them digital. The old rule compared the cover's
        text against three pages' worth and stopped there.
        """
        result = self.extract("text", *["scan"] * 19, max_pages=25)

        self.assertEqual(result["kind"], "pdf-mixed")
        self.assertEqual(len(result["pageImages"]), 19,
                         "nineteen scanned pages should all be queued")

    # -- Every page accounted for ----------------------------------------

    def test_every_page_appears_in_the_detail_exactly_once(self):
        """A page missing from the record cannot be told from a blank one
        afterwards, which is how the original bug hid."""
        result = self.extract("text", "scan", "text", "scan", "scan")
        numbers = [d["page"] for d in result["pageDetail"]]
        self.assertEqual(numbers, [1, 2, 3, 4, 5])

    def test_pages_over_the_render_limit_are_marked_unread(self):
        result = self.extract(*["scan"] * 5, max_pages=2)

        self.assertEqual(len(result["pageImages"]), 2)
        self.assertEqual(result["unreadPages"], [3, 4, 5])
        self.assertTrue(result["truncated"],
                        "dropping three pages is exactly what truncated means")
        for detail in result["pageDetail"][2:]:
            self.assertEqual(detail["source"], "unread")
            self.assertIn("limit", detail["why"])

    def test_a_fully_read_document_is_not_marked_truncated(self):
        result = self.extract("text", "scan")
        self.assertEqual(result["unreadPages"], [])
        self.assertFalse(result["truncated"])

    # -- The unmixed cases still behave ----------------------------------

    def test_an_all_text_pdf_needs_no_model(self):
        result = self.extract("text", "text", "text")
        self.assertEqual(result["kind"], "pdf-text")
        self.assertEqual(result["pageImages"], [])
        self.assertEqual(result["unreadPages"], [])
        self.assertTrue(all(d["source"] == "text" for d in result["pageDetail"]))

    def test_an_all_scan_pdf_goes_entirely_to_the_model(self):
        result = self.extract("scan", "scan")
        self.assertEqual(result["kind"], "pdf-scan")
        self.assertEqual(len(result["pageImages"]), 2)
        self.assertTrue(all(d["source"] == "ocr" for d in result["pageDetail"]))

    def test_a_page_below_the_threshold_keeps_its_text_as_a_fallback(self):
        """A page with a stray header over an unreadable scan.

        It goes to the model, but the little text it did carry is kept so the
        caller can report it if OCR comes back with nothing.
        """
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 100), "Fig 3", fontsize=9)
        pix = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, 600, 800), False)
        pix.clear_with(200)
        page.insert_image(page.rect, pixmap=pix)
        path = os.path.join(self.dir, "stray.pdf")
        doc.save(path)
        doc.close()

        result = extract_pdf(path, self.out, 12)
        self.assertEqual(result["kind"], "pdf-scan")
        self.assertEqual(result["pageDetail"][0]["layerText"], "Fig 3")


class Contract(unittest.TestCase):
    """The script is spawned as a process, so its stdout is the real contract."""

    def test_stdout_is_one_json_object_the_caller_can_parse(self):
        directory = tempfile.mkdtemp()
        try:
            doc = fitz.open()
            text_page(doc, COVER_TEXT)
            scanned_page(doc)
            path = os.path.join(directory, "mixed.pdf")
            doc.save(path)
            doc.close()

            script = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "attachment_extract.py",
            )
            out = subprocess.run(
                [sys.executable, script, path, os.path.join(directory, "out")],
                capture_output=True, text=True, check=True,
            ).stdout
            parsed = json.loads(out.strip().splitlines()[-1])

            self.assertEqual(parsed["kind"], "pdf-mixed")
            self.assertEqual(parsed["pages"], 2)
            # The field names the Rust side deserialises.
            for field in ("kind", "text", "pages", "pageImages",
                          "pageDetail", "unreadPages", "truncated"):
                self.assertIn(field, parsed)
        finally:
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()


def ole_file(entries, sector_shift=9, directory_sector=0, body=b""):
    """A structurally real OLE2 file.

    Built to the actual layout rather than by pasting a signature in front of
    some bytes, because the thing under test is directory *parsing*: a fixture
    carrying only the header would pass a byte scanner and fail a parser, which
    is exactly the difference these tests exist to pin.
    """
    sector_size = 1 << sector_shift
    header = bytearray(b"\x00" * 512)
    header[0:8] = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    struct.pack_into("<H", header, 0x1E, sector_shift)
    struct.pack_into("<I", header, 0x30, directory_sector)

    directory = bytearray()
    for name in entries:
        entry = bytearray(b"\x00" * 128)
        encoded = name.encode("utf-16-le")
        entry[0:len(encoded)] = encoded
        # Length in bytes, including the terminating null pair.
        struct.pack_into("<H", entry, 64, len(encoded) + 2)
        directory += entry

    target = (directory_sector + 1) * sector_size
    out = bytearray(header)
    out += body
    if len(out) < target:
        out += b"\x00" * (target - len(out))
    return bytes(out[:target] + directory)


class ContentDecidesTheReader(unittest.TestCase):
    """Routing follows what a file is, not what it is called."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)

    def write(self, name, payload):
        path = os.path.join(self.dir, name)
        with open(path, "wb") as handle:
            handle.write(payload)
        return path

    def test_a_pdf_is_recognised_whatever_it_is_called(self):
        path = self.write("report.doc", b"%PDF-1.7\n1 0 obj\n")
        self.assertEqual(sniff(path), "pdf")

    def test_a_pdf_with_junk_before_its_header_is_still_a_pdf(self):
        """Acrobat and PyMuPDF both tolerate this; anchoring at offset 0 did
        not, and sent the file to the text reader to be latin-1 decoded."""
        path = self.write("report.pdf", b"\n\n%PDF-1.4\n1 0 obj\n")
        self.assertEqual(sniff(path), "pdf")

    def test_a_legacy_office_file_is_recognised_from_its_directory(self):
        path = self.write("minutes.doc", ole_file(["Root Entry", "WordDocument"]))
        self.assertEqual(sniff(path), "cfb")
        self.assertEqual(cfb_contents(path), ("doc", "Word"))

    def test_the_directory_is_found_where_the_header_says_it_is(self):
        """Not in the first 8 KiB.

        MS-CFB places no constraint on where the directory sits, and Office
        writers routinely put it after the stream data. A fixed-window scan
        missed the names on any document big enough to matter.
        """
        path = self.write(
            "big.xls",
            ole_file(["Root Entry", "Workbook"], directory_sector=40, body=b"\x11" * 40000),
        )
        self.assertGreater(os.path.getsize(path), 8192)
        self.assertIn("Workbook", cfb_directory_names(path))
        self.assertEqual(cfb_contents(path), ("xls", "Excel"))

    def test_a_word_document_mentioning_bookmark_is_not_called_excel(self):
        """The false positive the raw-bytes search produced.

        `Book` as UTF-16LE is a prefix of "Bookmark", so a Word document whose
        body contained that word was confidently reported as a legacy Excel
        file -- a wrong format, stated with certainty.
        """
        body = "Bookmark and Booking and Bookkeeping ".encode("utf-16-le") * 200
        path = self.write(
            "minutes.doc",
            ole_file(["Root Entry", "WordDocument"], directory_sector=30, body=body),
        )
        self.assertEqual(cfb_contents(path), ("doc", "Word"))
        self.assertEqual(refuse_cfb(path)["detectedFormat"], "doc")

    def test_renaming_a_legacy_file_to_docx_does_not_make_it_readable(self):
        """The rule: renaming an extension is not conversion."""
        path = self.write("capex.xlsx", ole_file(["Root Entry", "Workbook"]))

        self.assertEqual(sniff(path), "cfb")
        refusal = refuse_cfb(path)
        self.assertEqual(refusal["reason"], "missing-parser")
        self.assertEqual(refusal["detectedFormat"], "xls")
        self.assertIn("Excel", refusal["error"])
        self.assertIn("renaming the extension does not", refusal["error"])

    def test_an_unidentified_ole2_container_is_not_called_a_pre_2007_office_file(self):
        """OLE2 is also `.msg`, `.msi` and `Thumbs.db`.

        Claiming "pre-2007 Office file" from the header alone was a confident
        answer to a question the code had never asked.
        """
        path = self.write("mail.msg", ole_file(["Root Entry", "__nameid_version1.0"]))
        refusal = refuse_cfb(path)

        self.assertEqual(refusal["reason"], "unidentified-container")
        self.assertEqual(refusal["detectedFormat"], "ole2")
        self.assertNotIn("pre-2007", refusal["error"])

    def test_a_password_protected_modern_file_is_named_as_such(self):
        """An encrypted .docx is OLE2 wrapping an encrypted zip. Telling
        somebody it is pre-2007 sends them to fix the wrong thing."""
        path = self.write(
            "secret.docx",
            ole_file(["Root Entry", "EncryptionInfo", "EncryptedPackage"]),
        )
        refusal = refuse_cfb(path)

        self.assertEqual(refusal["reason"], "password-protected")
        self.assertIn("password-protected", refusal["error"])
        self.assertNotIn("pre-2007", refusal["error"])

    def test_a_legacy_refusal_never_returns_scraped_text(self):
        """No plausible-looking fallback.

        Scraping printable strings out of a binary yields fragments in file
        order with no structure and no locators. A summary built from them reads
        as grounded and is not.
        """
        secret = "The rated duty is 12 bar ".encode("utf-16-le") * 50
        path = self.write(
            "old.xls",
            ole_file(["Root Entry", "Workbook"], directory_sector=10, body=secret),
        )
        refusal = refuse_cfb(path)

        self.assertNotIn("text", refusal)
        self.assertNotIn("12 bar", refusal["error"])

    def test_an_ooxml_package_is_identified_by_its_own_parts(self):
        path = os.path.join(self.dir, "deck.bin")
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("[Content_Types].xml", "<Types/>")
            archive.writestr("ppt/presentation.xml", "<p:presentation/>")
        self.assertEqual(sniff(path), "zip")
        self.assertEqual(ooxml_family(path), "pptx")

    def test_a_plain_zip_is_not_mistaken_for_a_document(self):
        path = os.path.join(self.dir, "archive.docx")
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("notes.txt", "hello")
        self.assertIsNone(ooxml_family(path))

    def test_an_unrecognised_binary_is_not_reported_as_text(self):
        """The regression that made every unknown file a successful read.

        `sniff` returning "text" as its fallback meant an executable came back
        as `{"kind": "text", ...}` full of mojibake. "Unknown" is the honest
        answer, and the caller refuses on it.
        """
        path = self.write("tool.exe", b"MZ\x90\x00\x03" + b"\x00" * 64)
        self.assertEqual(sniff(path), "unknown")

    def test_an_image_is_not_sent_to_the_document_extractor(self):
        for name, payload in [
            ("scan.pdf", b"\x89PNG\r\n\x1a\n" + b"\x00" * 16),
            ("photo.bin", b"BM" + b"\x00" * 20),
            ("shot.dat", b"RIFF\x24\x00\x00\x00WEBPVP8 "),
        ]:
            self.assertEqual(sniff(self.write(name, payload)), "image", name)

    def test_a_text_file_with_a_byte_order_mark_is_not_taken_for_an_image(self):
        for name, payload in [
            ("utf8.txt", b"\xef\xbb\xbfhello"),
            ("utf16le.txt", b"\xff\xfeh\x00i\x00"),
            ("utf16be.txt", b"\xfe\xffh\x00i\x00"),
            ("rows.csv", b"tag,duty\nP-101,12 bar\n"),
        ]:
            self.assertEqual(sniff(self.write(name, payload)), "unknown", name)


class TheExtractorRefusesRatherThanInvents(unittest.TestCase):
    """End to end, through the real subprocess, for the cases that used to come
    back as successful reads of nothing."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)

    def run_extractor(self, name, payload):
        path = os.path.join(self.dir, name)
        with open(path, "wb") as handle:
            handle.write(payload)
        script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "attachment_extract.py",
        )
        done = subprocess.run(
            [sys.executable, script, path, os.path.join(self.dir, "out")],
            capture_output=True, text=True, check=False,
        )
        lines = [l for l in done.stdout.splitlines() if l.strip().startswith("{")]
        self.assertTrue(lines, "the extractor printed no JSON: %s" % done.stderr)
        return json.loads(lines[-1])

    def test_an_executable_is_refused_rather_than_decoded(self):
        result = self.run_extractor("tool.exe", b"MZ\x90\x00\x03" + b"\x00" * 64)
        self.assertIn("error", result)
        self.assertNotIn("text", result)

    def test_an_empty_unknown_file_does_not_claim_a_page_was_read(self):
        result = self.run_extractor("nothing.bin", b"")
        self.assertIn("error", result)

    def test_a_text_file_still_reads(self):
        result = self.run_extractor("rows.csv", b"tag,duty\nP-101,12 bar\n")
        self.assertEqual(result.get("kind"), "text")
        self.assertIn("12 bar", result["text"])

    def test_an_ole2_file_carries_a_machine_readable_reason(self):
        result = self.run_extractor("old.doc", ole_file(["Root Entry", "WordDocument"]))
        self.assertEqual(result.get("reason"), "missing-parser")
        self.assertEqual(result.get("detectedFormat"), "doc")

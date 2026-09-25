"""Dispatches document requests to the best engine available on this machine.

Engines are tried in order of preference and the first available one wins. The
choice is reported with every result, so a deployment running on the fallback
never looks like one running on the real thing.

## Multimodal engine selection

A P&ID is a particular kind of document. The P&ID engine has tag detection
and a layout-aware reader that the generic Docling / text-layer engines do
not bother with, so the router checks the document type *first* and, when
it is a P&ID, runs the P&ID engine. The check is on the cheap text layer
only — a P&ID whose page is image-only is reported as unread, not routed to
the wrong engine.
"""

import os
from typing import Any, Dict, List, Optional, Type

import escalation
import injection
from engines.base import DocumentEngine
from engines.doc_type import detect as detect_type
from engines.docling_engine import DoclingEngine
from engines.pid_engine import PidEngine
from engines.text_layer import TextLayerEngine

#: Best first. Adding an engine is adding a class here. The P&ID engine is
#: added with a sentinel preference so it can be selected by document type
#: rather than by ordering.
ENGINE_PREFERENCE: List[Type[DocumentEngine]] = [DoclingEngine, TextLayerEngine]

#: Engines that are selected by document type, not by the global preference
#: list. Each is matched against the verdict of the document-type detector.
TYPE_ROUTED_ENGINES: Dict[str, Type[DocumentEngine]] = {
    "pid": PidEngine,
}

#: Refuse anything larger rather than exhausting memory on a laptop. A refinery
#: drawing set can be very large, and failing clearly beats failing by swapping.
MAX_FILE_BYTES = 512 * 1024 * 1024

#: Read directly, without a document engine. The type-routed engines (Docling,
#: the text layer, the P&ID reader) all expect a PDF.
PLAIN_TEXT_EXTENSIONS = {".txt", ".md", ".html", ".htm"}

#: Tags after which HTML text starts a new line, so words from two paragraphs
#: are not glued together into one token.
_HTML_BREAKS = {"p", "br", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "td"}


def read_plain_text(path: str, extension: str):
    """A text file as pages: split at form feeds, which is how a text export
    marks a page break, and otherwise one page.

    HTML is reduced to its text with the standard library's parser, so a tag
    is never indexed as if it were a word. Undecodable bytes are replaced and
    said so, rather than the file being refused or silently altered.
    """
    from html.parser import HTMLParser

    from engines.base import EngineCapabilities, ExtractionResult, PageResult

    with open(path, "rb") as handle:
        raw = handle.read()
    text = raw.decode("utf-8", errors="replace")
    result = ExtractionResult(
        engine="plain-text",
        engine_version="1",
        capabilities=EngineCapabilities(
            ocr=False, layout=False, tables=False, formulas=False, handwriting=False,
            pid_symbols=False, image_captioning=False,
        ),
    )
    if "�" in text:
        result.warnings.append(
            "Some bytes in this file are not UTF-8 and were replaced; the passages "
            "around them may read oddly."
        )
    if extension in (".html", ".htm"):

        class _Text(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.parts: List[str] = []
                self.skip = 0

            def handle_starttag(self, tag, attrs):  # noqa: ANN001
                if tag in ("script", "style"):
                    self.skip += 1
                elif tag in _HTML_BREAKS:
                    self.parts.append("\n")

            def handle_endtag(self, tag):  # noqa: ANN001
                if tag in ("script", "style") and self.skip:
                    self.skip -= 1

            def handle_data(self, data):  # noqa: ANN001
                if not self.skip:
                    self.parts.append(data)

        parser = _Text()
        parser.feed(text)
        text = "".join(parser.parts)
    pages = text.split("\f") if "\f" in text else [text]
    for index, page_text in enumerate(pages, start=1):
        stripped = page_text.strip()
        result.pages.append(
            PageResult(
                page=index,
                text=stripped,
                confidence=1.0 if stripped else 0.0,
                needs_review=not stripped,
                review_reason=None if stripped else "This page of the file is empty.",
            )
        )
    return result


class DocumentRouter:
    def __init__(self) -> None:
        self._engine: Optional[DocumentEngine] = None
        self._considered: List[Dict[str, Any]] = []
        self._select()

    def _select(self) -> None:
        for engine_class in ENGINE_PREFERENCE:
            try:
                available = engine_class.available()
            except Exception:  # noqa: BLE001 — a broken engine must not stop the rest
                available = False

            self._considered.append(
                {"engine": engine_class.name, "available": available}
            )

            if available and self._engine is None:
                self._engine = engine_class()

    def status(self) -> Dict[str, Any]:
        """What this sidecar can do, and what it fell back to."""
        if self._engine is None:
            return {
                "ready": False,
                "engine": None,
                "considered": self._considered,
                "detail": (
                    "No document engine is available. Install pypdf for basic text "
                    "extraction, or Docling for layout and tables."
                ),
            }

        degraded = self._engine.name != ENGINE_PREFERENCE[0].name
        return {
            "ready": True,
            "engine": self._engine.name,
            "engineVersion": self._engine.version,
            "capabilities": self._engine.capabilities().__dict__,
            "considered": self._considered,
            "degraded": degraded,
            "detail": (
                "Running on the fallback text-layer engine. Scanned pages cannot be read "
                "until Docling or a document vision model is installed."
                if degraded
                else "Running on the preferred engine."
            ),
        }

    def dispatch(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if method == "health_check":
            return self.status()

        if method == "extract":
            return self._extract(params)

        if method == "classify":
            return self._classify(params)

        raise ValueError(f"unknown method {method!r}")

    def _classify(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Classify a document without extracting it.

        Used by callers that want to know what kind of document this is before
        deciding how to ingest it. Reads the text layer, runs the detector
        over it, and returns the verdict. The actual file is not opened for
        layout, so the call is cheap and the verdict is best-effort.
        """
        path = params.get("path")
        if not path:
            raise ValueError("classify requires a 'path'")
        if not os.path.isfile(path):
            raise ValueError(f"no file at {path!r}")

        # Pull just enough text to detect. pypdf is what the fallback engine
        # uses, so it is available wherever classify is called.
        import pypdf

        try:
            reader = pypdf.PdfReader(path)
        except Exception as exc:  # noqa: BLE001
            return {
                "ready": True,
                "label": "unknown",
                "confidence": 0.0,
                "abstained": True,
                "abstentionReason": f"the file could not be opened as a PDF: {exc}",
                "signals": [],
            }

        if getattr(reader, "is_encrypted", False):
            return {
                "ready": True,
                "label": "unknown",
                "confidence": 0.0,
                "abstained": True,
                "abstentionReason": "the PDF is encrypted and could not be classified",
                "signals": [],
            }

        # Read the first few pages — a detector that needs the whole document
        # is not a cheap detector. Cap at 10 to keep the cost bounded.
        try:
            texts: List[str] = []
            for page in list(reader.pages)[:10]:
                try:
                    texts.append(page.extract_text() or "")
                except Exception:  # noqa: BLE001
                    continue
        except Exception:  # noqa: BLE001
            texts = []

        joined = "\n\n".join(texts)
        verdict = detect_type(joined)
        return {
            "ready": True,
            "label": verdict.label if not verdict.abstained else "unknown",
            "confidence": verdict.confidence,
            "abstained": verdict.abstained,
            "abstentionReason": verdict.abstention_reason,
            "signals": verdict.signals,
        }

    def _extract(self, params: Dict[str, Any]) -> Dict[str, Any]:
        path = params.get("path")
        if not path:
            raise ValueError("extract requires a 'path'")

        if not os.path.isfile(path):
            raise ValueError(f"no file at {path!r}")

        size = os.path.getsize(path)
        if size > MAX_FILE_BYTES:
            raise ValueError(
                f"the file is {size / 1024 / 1024:.0f} MB, above the "
                f"{MAX_FILE_BYTES / 1024 / 1024:.0f} MB limit for a single document"
            )

        # Plain text needs no engine, and reading it through one would be a
        # PDF parser refusing a Markdown file. A knowledge collection is mostly
        # PDFs and a fair number of notes, and the notes must not wait on
        # Docling being installed.
        extension = os.path.splitext(path)[1].lower()
        if extension in PLAIN_TEXT_EXTENSIONS:
            result = read_plain_text(path, extension)
        else:
            if self._engine is None:
                raise ValueError(
                    "No document engine is available on this machine, so nothing can be read."
                )
            # Pick the engine: the document type may override the global
            # preference for documents that have a dedicated engine (P&ID,
            # today). The type-routed engine is used only if it is available on
            # this machine and the document is plausibly of that type.
            result = self._select_engine_for(path).extract(path)

        payload = result.to_dict()
        payload["sourcePath"] = path
        payload["sourceBytes"] = size

        # Runs on every document, at ingest, before anything downstream sees the
        # text. The scan flags; it never edits — removing an offending line
        # would hide evidence of an attack and alter a record the organisation
        # may need intact.
        payload["injectionScan"] = injection.scan_pages(payload["pages"])

        # Two-tier reading: the cheap pass has run; this decides which pages it
        # could not settle. With no second engine installed the plan is still
        # produced, so the gap is visible rather than looking like a document
        # that simply had little on it.
        plan = escalation.plan(payload["pages"], payload["capabilities"])
        payload["escalation"] = plan.to_dict()
        payload["warnings"].extend(
            escalation.describe_unmet(plan, available=self._available_capabilities())
        )
        return payload

    def _select_engine_for(self, path: str) -> DocumentEngine:
        """Choose the engine for ``path``.

        The default is the highest-preference engine that is available. When
        the document type detector says this looks like a P&ID — and the P&ID
        engine is available — the P&ID engine is used instead, since its
        tag detection is what makes the extraction useful.

        The detection is done on a cheap read of the first pages. If it
        abstains, the global preference stands.
        """
        # Cheap read: just the first page's text. The detector is robust
        # enough that one page is enough to recognise a P&ID — a P&ID's
        # title block alone carries the signals.
        try:
            verdict = self._classify({"path": path})
            if not verdict.get("abstained", True) and verdict.get("label") in TYPE_ROUTED_ENGINES:
                engine_class = TYPE_ROUTED_ENGINES[verdict["label"]]
                if engine_class.available():
                    return engine_class()
        except Exception:  # noqa: BLE001
            # Detector failure is not a reason to refuse. Fall through to
            # the default preference.
            pass

        # At this point we have already proven that ``self._engine`` is not
        # None in the calling context, but the type-checker does not know
        # that. Re-assert it so the type is unambiguous.
        assert self._engine is not None
        return self._engine

    def _available_capabilities(self) -> List[str]:
        """Capabilities any installed engine could supply for a second pass."""
        supplied: List[str] = []
        for engine_class in ENGINE_PREFERENCE:
            try:
                if not engine_class.available():
                    continue
            except Exception:  # noqa: BLE001
                continue
            caps = engine_class().capabilities()
            if caps.ocr:
                supplied.append("ocr")
            if caps.layout:
                supplied.append("vision")
        return sorted(set(supplied))

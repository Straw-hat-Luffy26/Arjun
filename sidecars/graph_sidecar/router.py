"""Dispatch for the graph sidecar.

One method, `graph.extract_triplets`. Kept separate from `main.py` so the frame
loop can be read without the model in the way, and separate from `rebel.py` so
the model can be loaded lazily -- the process starts in milliseconds and pays
the 1.6 GB load only when the first extraction request arrives. A sidecar that
loaded REBEL at import would make every ARJUN launch several seconds slower for
a feature most sessions never touch.
"""

from __future__ import annotations

from typing import Any

from rebel import RebelExtractor


class GraphActionRouter:
    """Routes one method to the extractor, holding it across requests."""

    def __init__(self, model_dir: str | None = None) -> None:
        self._model_dir = model_dir
        self._extractor: RebelExtractor | None = None

    def _ensure(self) -> RebelExtractor:
        if self._extractor is None:
            self._extractor = RebelExtractor(self._model_dir)
        return self._extractor

    def dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "graph.extract_triplets":
            return self._extract(params)
        if method == "graph.ping":
            # Answered without loading the model. Lets the Rust side confirm the
            # process is alive and the interpreter has its dependencies before
            # committing to a load it may have to wait seconds for.
            return {"ok": True, "loaded": self._extractor is not None}
        raise ValueError(f"unknown method: {method}")

    def _extract(self, params: dict[str, Any]) -> dict[str, Any]:
        chunks = params.get("chunks") or []
        if not isinstance(chunks, list):
            raise ValueError("chunks must be a list of {id, text} objects")

        triplets = self._ensure().extract(chunks)
        return {
            "triplets": [triplet.to_json() for triplet in triplets],
            # Reported so the Rust side can distinguish "the model found nothing
            # in these passages" from "the passages never arrived". Those need
            # different messages and look identical in an empty result.
            "chunksSeen": len([c for c in chunks if (c.get("text") or "").strip()]),
        }

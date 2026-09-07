"""Relation extraction with Babelscape/rebel-large.

REBEL is a BART sequence-to-sequence model fine-tuned to emit relation triplets
as *text*. It does not answer questions and it has no chat template: given a
passage it produces a single linearised string naming the relations it found,
in a fixed grammar:

    <triplet> Pump PV-2201 <subj> Acme Pumps Ltd <obj> manufacturer
    <triplet> Acme Pumps Ltd <subj> Pune <obj> headquarters location

Read as: for each `<triplet>`, the head entity, then one or more
(tail, relation) pairs hanging off it. A head may carry several pairs, which is
why the decoder below is a small state machine rather than a regular expression.

## Why this runs on CPU

REBEL is 400M parameters. On CPU one passage costs a few hundred milliseconds,
which is the right price for background indexing and buys something the GPU
path cannot: it never contends with the chat model. The typing pass in
`knowledge/graph/typing.rs` has to borrow whichever llama-server endpoint is
already warm, and apologises for it in its own doc comment. This pass borrows
nothing.

## What this module does not do

It does not decide whether a triplet is true. Every gate -- was this chunk in
the request, do the subject and object actually occur in it, is this an edge the
statistical pass already found -- lives in `knowledge/graph/relations.rs` on the
Rust side, next to the data needed to check it. This module's whole
responsibility is: text in, triplets out, honestly labelled with the chunk they
came from.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Iterable, Iterator

# The Hugging Face repository this expects to have been downloaded.
REPO_ID = "Babelscape/rebel-large"

# BART's encoder is a hard 1024 positions. A passage longer than that is not
# truncated silently -- truncation drops the tail of the text while the model
# still reports triplets, so a relation stated in the dropped half never appears
# and nothing says why. Long chunks are split instead, at sentence boundaries,
# by `split_for_encoder`.
MAX_INPUT_TOKENS = 1024

# Beam search rather than sampling, because two runs over an unchanged document
# must produce the same graph. A rebuild that reshuffles edges is
# indistinguishable from a document that changed.
NUM_BEAMS = 3
MAX_OUTPUT_TOKENS = 256


@dataclass(frozen=True)
class Triplet:
    """One relation, and the passage it was read out of."""

    chunk_id: str
    subject: str
    relation: str
    object: str

    def to_json(self) -> dict:
        return asdict(self)


def decode_triplets(generated: str, chunk_id: str) -> list[Triplet]:
    """Parses REBEL's linearised output into triplets.

    Kept free of the model so it can be tested against recorded output without
    the weights present -- the same reason `typing::verify` on the Rust side is
    separated from `request_typing`.

    Malformed spans are skipped rather than raising. The model occasionally
    emits a `<triplet>` with no `<obj>`, and one bad span in a passage is not a
    reason to lose the good ones beside it.
    """
    triplets: list[Triplet] = []

    # These arrive as ordinary text because the caller decodes with
    # `skip_special_tokens=False` -- the tags *are* the output format.
    text = generated.replace("<s>", "").replace("</s>", "").replace("<pad>", "")

    head = tail = relation = ""
    state = "head"
    current = ""

    def flush() -> None:
        if head and relation and tail:
            triplets.append(Triplet(chunk_id, head.strip(), relation.strip(), tail.strip()))

    for token in text.split():
        if token == "<triplet>":
            # A new head entity. Whatever the previous head accumulated is
            # complete, because a relation label runs until the next tag.
            if state == "relation":
                flush()
            head = tail = relation = ""
            state = "head"
            current = ""
        elif token == "<subj>":
            # First `<subj>` after a `<triplet>` closes the head entity. Every
            # later one closes the *previous* (tail, relation) pair and opens
            # the next -- one head can carry several, which is the case a
            # regular expression gets wrong.
            if state == "head":
                head = current
            elif state == "relation":
                flush()
            tail = relation = ""
            state = "tail"
            current = ""
        elif token == "<obj>":
            # Despite the name, what follows `<obj>` is the relation label; the
            # text before it was the tail entity.
            if state == "tail":
                tail = current
            state = "relation"
            current = ""
        elif state == "relation":
            relation = (relation + " " + token) if relation else token
        else:
            current = (current + " " + token) if current else token

    if state == "relation":
        flush()

    # Deduplicate, preserving order. REBEL restates relations -- a passage
    # naming one supplier twice yields the identical triplet twice, and that is
    # one edge, not two.
    seen: set[tuple[str, str, str]] = set()
    unique: list[Triplet] = []
    for triplet in triplets:
        key = (triplet.subject, triplet.relation, triplet.object)
        if key in seen:
            continue
        seen.add(key)
        unique.append(triplet)
    return unique


def split_for_encoder(text: str, max_chars: int = 2400) -> list[str]:
    """Splits a passage into encoder-sized pieces at sentence boundaries.

    `max_chars` is a deliberately conservative stand-in for 1024 BART tokens.
    English runs roughly four characters per token, so 2400 characters leaves
    room for the tag-dense text this corpus is full of -- `PV-2201`,
    `ASME B31.3` -- which tokenises far worse than prose.

    Splitting mid-sentence would be worse than truncating. The Rust gate
    requires that a triplet's subject and object both appear in the chunk that
    was sent; a sentence cut in half puts the subject in one piece and the
    object in the other, and both then fail verification. The relation was real
    and gets reported as a fabrication.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    current = ""
    for sentence in _sentences(text):
        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current.strip())
            current = sentence
        else:
            current = (current + " " + sentence) if current else sentence
    if current.strip():
        pieces.append(current.strip())

    # A single sentence longer than the window is rare but real -- a table row
    # flattened into prose, say. Passing it whole and letting the tokenizer
    # truncate beats cutting it at an arbitrary character, because at least the
    # head of it stays intact.
    return [piece for piece in pieces if piece]


def _sentences(text: str) -> Iterator[str]:
    """Sentence boundaries, without a natural-language dependency.

    Splits after a full stop, question mark or exclamation mark followed by
    whitespace. Deliberately naive: an abbreviation such as "No. 3" produces one
    extra split, which costs a little recall at that boundary and nothing else.
    A wrong split is cheap here; a missing dependency in an air-gapped install
    is not.
    """
    start = 0
    for index, char in enumerate(text):
        if char in ".?!" and index + 1 < len(text) and text[index + 1].isspace():
            yield text[start : index + 1]
            start = index + 1
    tail = text[start:]
    if tail.strip():
        yield tail


class RebelExtractor:
    """Holds the model, loaded once.

    Loading REBEL costs several seconds and 1.6 GB of reads. A pass over forty
    documents that reloaded per call would spend longer loading than
    extracting, so the sidecar process is long-lived and this object outlives
    every request it serves.
    """

    def __init__(self, model_dir: str | None = None) -> None:
        self._model_dir = model_dir or os.environ.get("ARJUN_REBEL_DIR")
        if not self._model_dir:
            raise RuntimeError(
                "No model directory. Set ARJUN_REBEL_DIR to the directory holding "
                "Babelscape/rebel-large, or pass one to RebelExtractor."
            )
        if not os.path.isdir(self._model_dir):
            raise RuntimeError(f"Not a directory: {self._model_dir}")

        # Imported here, not at module scope, so `decode_triplets` and
        # `split_for_encoder` stay testable with neither torch nor the weights
        # installed.
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_dir)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_dir)
        self._model.eval()

    def extract(self, chunks: Iterable[dict]) -> list[Triplet]:
        """Extracts triplets from `[{"id", "text"}, ...]`.

        A chunk that produces nothing is not an error and is not reported as
        one. Most passages in a specification state no relation at all.
        """
        import torch

        results: list[Triplet] = []
        for chunk in chunks:
            chunk_id = chunk.get("id") or ""
            text = chunk.get("text") or ""
            if not chunk_id or not text.strip():
                continue

            for piece in split_for_encoder(text):
                encoded = self._tokenizer(
                    piece,
                    max_length=MAX_INPUT_TOKENS,
                    truncation=True,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    generated = self._model.generate(
                        **encoded,
                        max_length=MAX_OUTPUT_TOKENS,
                        num_beams=NUM_BEAMS,
                        do_sample=False,
                    )
                decoded = self._tokenizer.batch_decode(
                    generated, skip_special_tokens=False
                )[0]
                results.extend(decode_triplets(decoded, chunk_id))

        # Deduplicate across the pieces of one chunk.
        seen: set[tuple[str, str, str, str]] = set()
        unique: list[Triplet] = []
        for triplet in results:
            key = (triplet.chunk_id, triplet.subject, triplet.relation, triplet.object)
            if key in seen:
                continue
            seen.add(key)
            unique.append(triplet)
        return unique

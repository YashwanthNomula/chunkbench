"""Chunking strategies for chunkbench.

Three strategies, one interface: each takes raw text and returns a list of
:class:`Chunk` with exact character offsets back into the source document.

* ``fixed``     - sliding character window with overlap (the naive baseline)
* ``recursive`` - split on the strongest boundary that fits (paragraph, line,
                  sentence, word), then greedily merge back up to ``size``
* ``semantic``  - sentence-level greedy grouping that closes a chunk early on
                  topic shift (cosine similarity of word-count vectors)
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .retrieval import STOPWORDS, tokenize


@dataclass
class Chunk:
    """One chunk of a document."""

    doc_id: str
    idx: int
    text: str
    start: int  # char offset into the source document
    end: int    # char offset into the source document


def _check_size_overlap(size: int, overlap: int) -> None:
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")


# ---------------------------------------------------------------------------
# fixed: sliding character window
# ---------------------------------------------------------------------------

def fixed_chunks(text: str, size: int = 500, overlap: int = 50,
                 doc_id: str = "") -> list[Chunk]:
    """Split text into fixed-size character windows with ``overlap`` reuse."""
    _check_size_overlap(size, overlap)
    chunks: list[Chunk] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + size, n)
        piece = text[start:end]
        if piece.strip():
            chunks.append(Chunk(doc_id, len(chunks), piece, start, end))
        if end == n:
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# recursive: split on the strongest boundary that fits, then merge
# ---------------------------------------------------------------------------

_RECURSIVE_SEPS = ["\n\n", "\n", ". ", "? ", "! "]


def _split_spans(text: str, size: int) -> list[tuple[int, int]]:
    """Recursively split into (start, end) spans, strongest boundary first."""
    spans = [(0, len(text))]
    for sep in _RECURSIVE_SEPS:
        refined: list[tuple[int, int]] = []
        for s, e in spans:
            span = text[s:e]
            if len(span) > size and sep in span:
                pos = s
                for part in span.split(sep):
                    pe = pos + len(part)
                    refined.append((pos, pe))
                    pos = pe + len(sep)
            else:
                refined.append((s, e))
        spans = [(s, e) for s, e in refined if text[s:e].strip()]
    # hard-split anything still oversize (no usable boundary)
    final: list[tuple[int, int]] = []
    for s, e in spans:
        if e - s > size:
            for j in range(s, e, size):
                final.append((j, min(j + size, e)))
        else:
            final.append((s, e))
    return final


def recursive_chunks(text: str, size: int = 500, overlap: int = 50,
                     doc_id: str = "") -> list[Chunk]:
    """Boundary-aware chunks: never cut mid-paragraph when it can be avoided.

    ``overlap`` is accepted for API parity and validated, but recursive
    chunking does not need it: clean boundaries are the overlap substitute.
    """
    _check_size_overlap(size, overlap)
    spans = _split_spans(text, size)
    # greedy merge: adjacent spans merge while the merged chunk fits in size
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and e - merged[-1][0] <= size:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [Chunk(doc_id, i, text[s:e], s, e) for i, (s, e) in enumerate(merged)]


# ---------------------------------------------------------------------------
# semantic: sentence grouping with topic-shift detection
# ---------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentences_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Split into sentences, tracking exact (start, end) offsets."""
    out: list[tuple[str, int, int]] = []
    pos = 0
    for sent in _SENT_SPLIT.split(text.strip()):
        sent = sent.strip()
        if not sent:
            continue
        i = text.find(sent, pos)
        if i == -1:  # defensive: should not happen
            continue
        out.append((sent, i, i + len(sent)))
        pos = i + len(sent)
    return out


def _cosine(a: Counter, b: Counter) -> float:
    dot = sum(a[t] * b.get(t, 0) for t in a)
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def semantic_chunks(text: str, target: int = 500, threshold: float = 0.15,
                    min_chars: int = 200, doc_id: str = "",
                    size: int | None = None, overlap: int = 0) -> list[Chunk]:
    """Group sentences into chunks, closing early on topic shift.

    Sentences accumulate until ``target`` chars; a chunk also closes when the
    next sentence is dissimilar (cosine < ``threshold``) and the chunk already
    holds at least ``min_chars``. ``size``/``overlap`` are accepted so every
    strategy shares one call signature.
    """
    if target <= 0:
        raise ValueError("target must be positive")
    sents = _sentences_with_spans(text)
    chunks: list[Chunk] = []
    cur: list[tuple[str, int, int]] = []
    cur_vec: Counter = Counter()

    def flush() -> None:
        nonlocal cur, cur_vec
        if not cur:
            return
        s = cur[0][1]
        e = cur[-1][2]
        chunks.append(Chunk(doc_id, len(chunks), text[s:e], s, e))
        cur = []
        cur_vec = Counter()

    for sent, s, e in sents:
        vec = Counter(t for t in tokenize(sent) if t not in STOPWORDS)
        cur_len = (cur[-1][2] - cur[0][1]) if cur else 0
        if cur and cur_len + 1 + len(sent) > target:
            flush()
        elif cur and cur_len >= min_chars and _cosine(cur_vec, vec) < threshold:
            flush()
        cur.append((sent, s, e))
        cur_vec.update(vec)
    flush()
    return chunks

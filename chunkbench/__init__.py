"""chunkbench: benchmark chunking strategies for RAG retrieval."""
from __future__ import annotations

from functools import partial

from . import chunkers

__version__ = "1.0.0"

_fixed = partial(chunkers.fixed_chunks, size=500, overlap=50)
_fixed.chunkbench_params = {"size": 500, "overlap": 50}  # type: ignore[attr-defined]

_recursive = partial(chunkers.recursive_chunks, size=500, overlap=50)
_recursive.chunkbench_params = {"size": 500, "overlap": 50}  # type: ignore[attr-defined]

_semantic = partial(chunkers.semantic_chunks, target=500, threshold=0.15)
_semantic.chunkbench_params = {"target": 500, "threshold": 0.15}  # type: ignore[attr-defined]

STRATEGIES = {
    "fixed": (
        _fixed,
        "Sliding 500-char window with 50-char overlap. The naive baseline.",
    ),
    "recursive": (
        _recursive,
        "Split on the strongest boundary that fits (paragraph/line/sentence), "
        "then merge back up to 500 chars.",
    ),
    "semantic": (
        _semantic,
        "Sentence grouping that closes a chunk early on topic shift "
        "(cosine similarity < 0.15).",
    ),
}

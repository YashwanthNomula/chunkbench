"""Tiny pure-Python TF-IDF retrieval used to score every chunking strategy.

Identical retrieval for every strategy: the benchmark measures chunking, not
the ranker.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .chunkers import Chunk

_WORD = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """
    a an the and or but if then else when at by for with about into through
    during before after above below to from up down in out on off over under
    again further once here there all any both each few more most other some
    such no nor not only own same so than too very can will just don should
    now is are was were be been being have has had having do does did doing
    would could ought i you he she it we they them his her its our your their
    this that these those am as of s t d ll m re ve y
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with stopwords removed."""
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS]


class TfidfIndex:
    """In-memory TF-IDF index over chunks, cosine-ranked."""

    def __init__(self, chunks: list["Chunk"]):
        self.chunks = chunks
        self._doc_vecs: list[dict[str, float]] = []
        n = len(chunks)
        df: Counter = Counter()
        tfs: list[Counter] = []
        for ch in chunks:
            tf = Counter(tokenize(ch.text))
            tfs.append(tf)
            for term in tf:
                df[term] += 1
        self._idf = {
            term: math.log((n + 1) / (freq + 1)) + 1.0 for term, freq in df.items()
        }
        for tf in tfs:
            vec = {
                term: (1.0 + math.log(c)) * self._idf[term]
                for term, c in tf.items()
            }
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self._doc_vecs.append({t: v / norm for t, v in vec.items()})

    def search(self, query: str, k: int = 5) -> list[tuple["Chunk", float]]:
        """Return up to ``k`` (chunk, score) pairs, best first."""
        qtf = Counter(tokenize(query))
        if not qtf:
            return []
        qvec = {
            term: (1.0 + math.log(c)) * self._idf.get(term, 0.0)
            for term, c in qtf.items()
        }
        qnorm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
        qvec = {t: v / qnorm for t, v in qvec.items()}
        scored = []
        for chunk, dvec in zip(self.chunks, self._doc_vecs):
            score = sum(qvec[t] * dvec.get(t, 0.0) for t in qvec)
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

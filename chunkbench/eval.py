"""Benchmark harness: chunk every doc, retrieve, score against ground truth."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .chunkers import Chunk
from .retrieval import TfidfIndex


@dataclass
class StrategyResult:
    strategy: str
    params: dict
    recall_at: dict[int, float]
    mrr: float
    n_chunks: int
    avg_chunk_chars: float
    rows: list[dict] = field(default_factory=list)  # per-question detail


def _is_hit(chunk: Chunk, gold_doc: str, must_contain: list[str]) -> bool:
    if chunk.doc_id != gold_doc:
        return False
    lowered = chunk.text.lower()
    return all(phrase.lower() in lowered for phrase in must_contain)


def evaluate(
    docs: list[dict],
    questions: list[dict],
    strategies: dict[str, Callable[..., list[Chunk]]],
    ks: tuple[int, ...] = (1, 3, 5),
    top_k: int = 10,
) -> list[StrategyResult]:
    """Run every strategy over the corpus and score retrieval quality."""
    results: list[StrategyResult] = []
    for name, fn in strategies.items():
        chunks = [c for d in docs for c in fn(d["text"], doc_id=d["id"])]
        index = TfidfIndex(chunks)
        rows = []
        for q in questions:
            ranked = index.search(q["text"], k=top_k)
            rank = None
            for i, (chunk, _score) in enumerate(ranked, start=1):
                if _is_hit(chunk, q["gold_doc"], q["must_contain"]):
                    rank = i
                    break
            rows.append(
                {
                    "question": q["text"],
                    "gold_doc": q["gold_doc"],
                    "rank": rank,
                    "hit": rank is not None,
                }
            )
        n = len(questions)
        recall_at = {
            k: sum(1 for r in rows if r["rank"] is not None and r["rank"] <= k) / n
            for k in ks
        }
        mrr = sum(1.0 / r["rank"] if r["rank"] else 0.0 for r in rows) / n
        results.append(
            StrategyResult(
                strategy=name,
                params=getattr(fn, "chunkbench_params", {}),
                recall_at=recall_at,
                mrr=mrr,
                n_chunks=len(chunks),
                avg_chunk_chars=(
                    sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0.0
                ),
                rows=rows,
            )
        )
    return results

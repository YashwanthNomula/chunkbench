"""Terminal and JSON reports for chunkbench results."""
from __future__ import annotations

import json

from .eval import StrategyResult


def render_terminal(results: list[StrategyResult], ks: tuple[int, ...]) -> str:
    ks = tuple(ks)
    header = (
        f"{'strategy':<10}" + "".join(f"{f'recall@{k}':>10}" for k in ks)
        + f"{'MRR':>8}{'chunks':>8}{'avg chars':>10}"
    )
    lines = [header, "-" * len(header)]
    for r in sorted(results, key=lambda x: x.recall_at[ks[-1]], reverse=True):
        row = (
            f"{r.strategy:<10}"
            + "".join(f"{r.recall_at[k]:>10.3f}" for k in ks)
            + f"{r.mrr:>8.3f}{r.n_chunks:>8}{r.avg_chunk_chars:>10.0f}"
        )
        lines.append(row)
    lines.append("-" * len(header))

    best = max(results, key=lambda x: (x.recall_at[ks[-1]], x.mrr))
    lines.append(f"winner: {best.strategy} "
                 f"(recall@{ks[-1]} {best.recall_at[ks[-1]]:.3f}, "
                 f"MRR {best.mrr:.3f}, {best.n_chunks} chunks)")

    misses: dict[str, list[str]] = {}
    for r in results:
        missed = [row["question"] for row in r.rows if row["rank"] is None]
        if missed:
            misses[r.strategy] = missed
    if misses:
        lines.append("")
        lines.append("questions no strategy retrieved in top 10:")
        for strat, qs in misses.items():
            for q in qs:
                lines.append(f"  [{strat}] {q}")
    return "\n".join(lines)


def render_json(results: list[StrategyResult]) -> str:
    payload = [
        {
            "strategy": r.strategy,
            "params": r.params,
            "recall_at": {str(k): v for k, v in r.recall_at.items()},
            "mrr": r.mrr,
            "n_chunks": r.n_chunks,
            "avg_chunk_chars": r.avg_chunk_chars,
            "rows": r.rows,
        }
        for r in results
    ]
    return json.dumps(payload, indent=2)

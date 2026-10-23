"""Grid sweep: benchmark window sizes x overlap fractions for chunkbench.

The roadmap asked for exactly this: run the benchmark across a grid of
chunk sizes and overlaps and report the best config per corpus. Overlaps
are fractions of the window size (0 = no overlap, 0.25 = 25% overlap) so
the grid stays meaningful as the window grows.
"""
from __future__ import annotations

import json
import math

from . import chunkers
from .eval import StrategyResult, evaluate

# How many (strategy, size, overlap) runs we allow before asking the user
# to narrow the grid. Each run re-chunks the whole corpus and re-indexes.
MAX_RUNS = 200


def parse_int_list(spec: str) -> list[int]:
    """Parse a comma-separated list of positive ints, fail-closed."""
    parts = [p.strip() for p in spec.split(",")]
    if not parts or any(p == "" for p in parts):
        raise ValueError(f"empty grid value in {spec!r}")
    values = []
    for p in parts:
        if not p.isdigit():
            raise ValueError(f"not a positive integer: {p!r}")
        values.append(int(p))
    if any(v <= 0 for v in values):
        raise ValueError(f"chunk sizes must be positive: {spec!r}")
    # dedupe, keep sorted
    return sorted(set(values))


def parse_frac_list(spec: str) -> list[float]:
    """Parse a comma-separated list of overlap fractions in [0, 1).

    An overlap of 1.0 (or more) would erase the window entirely, so it is
    rejected rather than silently clamped.
    """
    parts = [p.strip() for p in spec.split(",")]
    if not parts or any(p == "" for p in parts):
        raise ValueError(f"empty grid value in {spec!r}")
    values = []
    for p in parts:
        try:
            v = float(p)
        except ValueError:
            raise ValueError(f"not a number: {p!r}") from None
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"not a finite number: {p!r}")
        if not 0.0 <= v < 1.0:
            raise ValueError(f"overlap fractions must satisfy 0 <= x < 1: {p!r}")
        values.append(v)
    return sorted(set(values))


def _make_fn(strategy: str, size: int, overlap_chars: int):
    if strategy == "semantic":
        # semantic is target-sized; overlap has no meaning for it.
        def fn(text, doc_id="", _s=size):  # noqa: E731
            return chunkers.semantic_chunks(text, target=_s, doc_id=doc_id)

        fn.chunkbench_params = {  # type: ignore[attr-defined]
            "size": size,
            "overlap": None,
        }
        return fn
    mod = {
        "fixed": chunkers.fixed_chunks,
        "overlap": chunkers.overlap_chunks,
        "recursive": chunkers.recursive_chunks,
    }[strategy]

    def fn(text, doc_id="", _s=size, _o=overlap_chars):  # noqa: E731
        return mod(text, size=_s, overlap=_o, doc_id=doc_id)

    fn.chunkbench_params = {"size": size, "overlap": overlap_chars}  # type: ignore[attr-defined]
    return fn


def build_configs(strategies: list[str], sizes: list[int],
                  fracs: list[float]) -> list[dict]:
    """One config per (strategy, size, overlap) cell of the grid."""
    configs = []
    for name in strategies:
        for size in sizes:
            for frac in fracs:
                overlap_chars = int(size * frac)
                # int() floors; keep a sane minimum of 1 char when frac > 0
                # and the window is large enough to mean it.
                if frac > 0 and overlap_chars == 0 and size >= 10:
                    overlap_chars = 1
                configs.append(
                    {
                        "strategy": name,
                        "size": size,
                        "overlap_frac": frac,
                        "overlap_chars": None if name == "semantic" else overlap_chars,
                        "fn": _make_fn(name, size, overlap_chars),
                    }
                )
    return configs


def run_sweep(docs: list[dict], questions: list[dict], configs: list[dict],
              ks: tuple[int, ...] = (1, 3, 5)) -> list[dict]:
    """Evaluate every config; returns result dicts with config attached."""
    if not configs:
        raise ValueError("no sweep configs to run")
    if not questions:
        raise ValueError("no questions: cannot score a sweep")
    rows = []
    for cfg in configs:
        label = f"{cfg['strategy']}/{cfg['size']}/{cfg['overlap_frac']}"
        (result,) = evaluate(docs, questions, {label: cfg["fn"]}, ks=ks)
        rows.append({"config": cfg, "result": result})
    return rows


def pick_best(rows: list[dict], ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    """Best config by recall@max_k, then MRR, then fewest chunks."""
    k = ks[-1]

    def key(row: dict):
        r: StrategyResult = row["result"]
        return (r.recall_at[k], r.mrr, -r.n_chunks)

    return max(rows, key=key)


def render_sweep_terminal(rows: list[dict], best: dict,
                          ks: tuple[int, ...] = (1, 3, 5)) -> str:
    k = ks[-1]
    best_label = f"{best['config']['strategy']}/{best['config']['size']}/{best['config']['overlap_frac']}"
    header = (f"{'strategy':<10}{'size':>7}{'overlap':>9}"
              + "".join(f"{f'recall@{kk}':>10}" for kk in ks)
              + f"{'MRR':>8}{'chunks':>8}")
    lines = [header, "-" * len(header)]
    ordered = sorted(
        rows,
        key=lambda r: (r["result"].recall_at[k], r["result"].mrr),
        reverse=True,
    )
    for row in ordered:
        cfg = row["config"]
        r: StrategyResult = row["result"]
        label = f"{cfg['strategy']}/{cfg['size']}/{cfg['overlap_frac']}"
        mark = " *" if label == best_label else "  "
        ov = "-" if cfg["overlap_chars"] is None else str(cfg["overlap_chars"])
        lines.append(
            mark
            + f"{cfg['strategy']:<10}{cfg['size']:>7}{ov:>9}"
            + "".join(f"{r.recall_at[kk]:>10.3f}" for kk in ks)
            + f"{r.mrr:>8.3f}{r.n_chunks:>8}"
        )
    lines.append("-" * len(header))
    cfg = best["config"]
    r: StrategyResult = best["result"]
    ov = ("overlap n/a (semantic)" if cfg["overlap_chars"] is None
          else f"overlap={cfg['overlap_chars']}")
    lines.append(
        f"best config: {cfg['strategy']} size={cfg['size']} {ov} "
        f"(recall@{k} {r.recall_at[k]:.3f}, MRR {r.mrr:.3f}, {r.n_chunks} chunks) "
        f"— marked * above"
    )
    return "\n".join(lines)


def render_sweep_json(rows: list[dict], best: dict,
                      ks: tuple[int, ...] = (1, 3, 5)) -> str:
    payload = {
        "ks": list(ks),
        "configs": [
            {
                "strategy": row["config"]["strategy"],
                "size": row["config"]["size"],
                "overlap_frac": row["config"]["overlap_frac"],
                "overlap_chars": row["config"]["overlap_chars"],
                "recall_at": {str(kk): v for kk, v in row["result"].recall_at.items()},
                "mrr": row["result"].mrr,
                "n_chunks": row["result"].n_chunks,
                "avg_chunk_chars": row["result"].avg_chunk_chars,
            }
            for row in rows
        ],
        "best": {
            "strategy": best["config"]["strategy"],
            "size": best["config"]["size"],
            "overlap_frac": best["config"]["overlap_frac"],
            "overlap_chars": best["config"]["overlap_chars"],
        },
    }
    return json.dumps(payload, indent=2)

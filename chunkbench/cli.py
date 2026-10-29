"""CLI for chunkbench."""
from __future__ import annotations

import argparse
import sys

from . import STRATEGIES, __version__, chunkers
from .corpus import DOCS, QUESTIONS, load_docs_dir, load_questions_json
from .eval import evaluate
from .report import render_json, render_terminal
from .sweep import (MAX_RUNS, build_configs, parse_frac_list, parse_int_list,
                    pick_best, render_sweep_json, render_sweep_terminal,
                    run_sweep)

KS = (1, 3, 5)
MAX_K = 10


def parse_ks(spec: str) -> tuple[int, ...]:
    """Parse a comma-separated list of recall cutoffs, fail-closed.

    Retrieval only ranks the top 10, so recall@k beyond 10 would be
    misleading and is rejected rather than silently clamped.
    """
    if spec is None:
        raise ValueError("empty --ks value")
    parts = [p.strip() for p in spec.split(",")]
    if not parts or any(p == "" for p in parts):
        raise ValueError(f"empty value in --ks {spec!r}")
    values = []
    for p in parts:
        try:
            v = int(p)
        except ValueError:
            raise ValueError(f"not an integer in --ks: {p!r}") from None
        if v <= 0:
            raise ValueError(f"recall cutoffs must be positive: {p!r}")
        if v > MAX_K:
            raise ValueError(
                f"recall cutoff {v} exceeds top-10 retrieval limit"
            )
        values.append(v)
    return tuple(sorted(set(values)))


def cmd_demo(_args: argparse.Namespace) -> int:
    print("=" * 64)
    print(" chunkbench demo: which chunking strategy retrieves best?")
    print("=" * 64)
    print(f"\ncorpus: {len(DOCS)} production-ML docs, {len(QUESTIONS)} labeled questions")
    print("retrieval: identical TF-IDF ranker for every strategy (only chunking varies)\n")
    strategies = {name: fn for name, (fn, _desc) in STRATEGIES.items()}
    results = evaluate(DOCS, QUESTIONS, strategies, ks=KS)
    print(render_terminal(results, KS))
    print("\n" + "=" * 64)
    print(" run `chunkbench run --docs <dir> --questions <json>` on your own corpus")
    print("=" * 64)
    return 0


def cmd_strategies(_args: argparse.Namespace) -> int:
    for name, (_fn, desc) in STRATEGIES.items():
        print(f"{name:<10} {desc}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    try:
        ks = parse_ks(getattr(args, "ks", "1,3,5"))
    except ValueError as exc:
        print(f"bad --ks: {exc}", file=sys.stderr)
        return 2
    docs = load_docs_dir(args.docs) if args.docs else DOCS
    questions = load_questions_json(args.questions) if args.questions else QUESTIONS
    names = [s.strip() for s in args.strategy.split(",")]
    unknown = [n for n in names if n not in STRATEGIES]
    if unknown:
        print(f"unknown strategy: {', '.join(unknown)}", file=sys.stderr)
        print(f"choose from: {', '.join(STRATEGIES)}", file=sys.stderr)
        return 2

    strategies = {}
    for name in names:
        if name == "fixed":
            fn = lambda text, doc_id="", _s=args.size, _o=args.overlap: (  # noqa: E731
                chunkers.fixed_chunks(text, size=_s, overlap=_o, doc_id=doc_id)
            )
            fn.chunkbench_params = {"size": args.size, "overlap": args.overlap}  # type: ignore[attr-defined]
        elif name == "recursive":
            fn = lambda text, doc_id="", _s=args.size, _o=args.overlap: (  # noqa: E731
                chunkers.recursive_chunks(text, size=_s, overlap=_o, doc_id=doc_id)
            )
            fn.chunkbench_params = {"size": args.size, "overlap": args.overlap}  # type: ignore[attr-defined]
        elif name == "overlap":
            overlap = args.overlap_chars if args.overlap_chars is not None else int(args.size * 0.2)
            fn = lambda text, doc_id="", _s=args.size, _o=overlap: (  # noqa: E731
                chunkers.overlap_chunks(text, size=_s, overlap=_o, doc_id=doc_id)
            )
            fn.chunkbench_params = {"size": args.size, "overlap": overlap}  # type: ignore[attr-defined]
        else:
            fn = lambda text, doc_id="", _t=args.size: (  # noqa: E731
                chunkers.semantic_chunks(text, target=_t, doc_id=doc_id)
            )
            fn.chunkbench_params = {"target": args.size, "threshold": 0.15}  # type: ignore[attr-defined]
        strategies[name] = fn

    results = evaluate(docs, questions, strategies, ks=ks)
    if args.json:
        print(render_json(results, ks))
    else:
        print(render_terminal(results, ks))
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    names = [s.strip() for s in args.strategy.split(",")]
    unknown = [n for n in names if n not in STRATEGIES]
    if unknown:
        print(f"unknown strategy: {', '.join(unknown)}", file=sys.stderr)
        print(f"choose from: {', '.join(STRATEGIES)}", file=sys.stderr)
        return 2
    try:
        sizes = parse_int_list(args.sizes)
        fracs = parse_frac_list(args.overlaps)
    except ValueError as exc:
        print(f"bad grid: {exc}", file=sys.stderr)
        return 2
    configs = build_configs(names, sizes, fracs)
    if len(configs) > MAX_RUNS:
        print(f"grid too large: {len(configs)} runs > {MAX_RUNS}; "
              "narrow --sizes/--overlaps/--strategy", file=sys.stderr)
        return 2
    try:
        docs = load_docs_dir(args.docs) if args.docs else DOCS
        questions = load_questions_json(args.questions) if args.questions else QUESTIONS
    except (ValueError, OSError) as exc:
        print(f"bad input: {exc}", file=sys.stderr)
        return 2
    try:
        rows = run_sweep(docs, questions, configs, ks=KS)
    except ValueError as exc:
        print(f"sweep failed: {exc}", file=sys.stderr)
        return 2
    best = pick_best(rows, KS)
    print(f"sweep: {len(rows)} configs "
          f"({len(names)} strategies x {len(sizes)} sizes x {len(fracs)} overlaps)\n")
    if args.json:
        print(render_sweep_json(rows, best, KS))
    else:
        print(render_sweep_terminal(rows, best, KS))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chunkbench",
        description="Benchmark RAG chunking strategies on retrieval recall.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="run the built-in benchmark demo")
    d.set_defaults(func=cmd_demo)

    s = sub.add_parser("strategies", help="list available chunking strategies")
    s.set_defaults(func=cmd_strategies)

    r = sub.add_parser("run", help="benchmark on a corpus (built-in or your own)")
    r.add_argument("--strategy", default="fixed,overlap,recursive,semantic",
                   help="comma-separated subset of strategies (default: all)")
    r.add_argument("--docs", default=None,
                   help="directory of .md files (default: built-in corpus)")
    r.add_argument("--questions", default=None,
                   help="questions JSON [{text, gold_doc, must_contain}]")
    r.add_argument("--size", type=int, default=500, help="chunk size in chars")
    r.add_argument("--overlap", type=int, default=50,
                   help="overlap in chars (fixed/recursive strategies)")
    r.add_argument("--overlap-chars", type=int, default=None,
                   help="overlap in chars for the overlap strategy "
                   "(default: 20%% of --size)")
    r.add_argument("--ks", default="1,3,5",
                   help="comma-separated recall cutoffs (default: 1,3,5; "
                        "each must be 1-10, retrieval ranks top 10)")
    r.add_argument("--json", action="store_true", help="emit JSON instead of table")
    r.set_defaults(func=cmd_run)

    w = sub.add_parser("sweep", help="sweep chunk sizes x overlap fractions")
    w.add_argument("--strategy", default="fixed,overlap,recursive,semantic",
                   help="comma-separated subset of strategies (default: all)")
    w.add_argument("--sizes", default="250,500,1000",
                   help="comma-separated chunk sizes in chars (default: 250,500,1000)")
    w.add_argument("--overlaps", default="0,0.1,0.25",
                   help="comma-separated overlap fractions of the window "
                        "(default: 0,0.1,0.25); ignored for semantic")
    w.add_argument("--docs", default=None,
                   help="directory of .md files (default: built-in corpus)")
    w.add_argument("--questions", default=None,
                   help="questions JSON [{text, gold_doc, must_contain}]")
    w.add_argument("--json", action="store_true", help="emit JSON instead of table")
    w.set_defaults(func=cmd_sweep)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

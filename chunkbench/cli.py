"""CLI for chunkbench."""
from __future__ import annotations

import argparse
import sys

from . import STRATEGIES, __version__, chunkers
from .corpus import DOCS, QUESTIONS, load_docs_dir, load_questions_json
from .eval import evaluate
from .report import render_json, render_terminal

KS = (1, 3, 5)


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
        else:
            fn = lambda text, doc_id="", _t=args.size: (  # noqa: E731
                chunkers.semantic_chunks(text, target=_t, doc_id=doc_id)
            )
            fn.chunkbench_params = {"target": args.size, "threshold": 0.15}  # type: ignore[attr-defined]
        strategies[name] = fn

    results = evaluate(docs, questions, strategies, ks=KS)
    if args.json:
        print(render_json(results))
    else:
        print(render_terminal(results, KS))
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
    r.add_argument("--strategy", default="fixed,recursive,semantic",
                   help="comma-separated subset of strategies (default: all)")
    r.add_argument("--docs", default=None,
                   help="directory of .md files (default: built-in corpus)")
    r.add_argument("--questions", default=None,
                   help="questions JSON [{text, gold_doc, must_contain}]")
    r.add_argument("--size", type=int, default=500, help="chunk size in chars")
    r.add_argument("--overlap", type=int, default=50,
                   help="overlap in chars (fixed strategy)")
    r.add_argument("--json", action="store_true", help="emit JSON instead of table")
    r.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

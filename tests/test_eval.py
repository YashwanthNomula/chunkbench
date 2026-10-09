"""Tests for the benchmark harness and reporting."""
import json

from chunkbench import chunkers
from chunkbench.corpus import DOCS, QUESTIONS
from chunkbench.eval import evaluate
from chunkbench.report import render_json, render_terminal

TINY_DOCS = [
    {"id": "a", "title": "a",
     "text": "The red fox jumps over fences. Foxes are clever animals indeed."},
    {"id": "b", "title": "b",
     "text": "Quantum computers use qubits for calculation. Qubits enable superposition."},
]
TINY_QS = [
    {"text": "what animal jumps over fences",
     "gold_doc": "a", "must_contain": ["red fox"]},
    {"text": "what do quantum computers use",
     "gold_doc": "b", "must_contain": ["qubits"]},
]
STRATS = {"fixed": chunkers.fixed_chunks}


def test_evaluate_perfect_score_on_tiny_corpus():
    (res,) = evaluate(TINY_DOCS, TINY_QS, STRATS, ks=(1, 3))
    assert res.recall_at[1] == 1.0
    assert res.recall_at[3] == 1.0
    assert res.mrr == 1.0
    assert all(r["hit"] for r in res.rows)


def test_evaluate_punishes_split_answers():
    docs = [{"id": "a", "title": "a",
             "text": "x" * 490 + "needle phrase here" + "y" * 490}]
    qs = [{"text": "find the needle phrase here",
           "gold_doc": "a", "must_contain": ["needle phrase here"]}]
    # a 500-char fixed window starting at 0 cuts "needle phrase here" in half
    (res,) = evaluate(
        docs, qs,
        {"fixed": lambda t, doc_id="": chunkers.fixed_chunks(
            t, size=500, overlap=0, doc_id=doc_id)},
        ks=(1,),
    )
    assert res.rows[0]["hit"] is False
    assert res.recall_at[1] == 0.0


def test_evaluate_metrics_bounded_on_builtin_corpus():
    results = evaluate(DOCS, QUESTIONS, STRATS, ks=(1, 3, 5))
    (res,) = results
    assert res.n_chunks > 0
    assert 0.0 <= res.recall_at[5] <= 1.0
    assert 0.0 <= res.mrr <= 1.0
    assert len(res.rows) == len(QUESTIONS)


def test_render_terminal_contains_table():
    (res,) = evaluate(TINY_DOCS, TINY_QS, STRATS, ks=(1, 3))
    out = render_terminal([res], (1, 3))
    assert "fixed" in out and "recall@1" in out and "winner" in out


def test_render_json_round_trips():
    (res,) = evaluate(TINY_DOCS, TINY_QS, STRATS, ks=(1,))
    payload = json.loads(render_json([res]))
    assert payload[0]["strategy"] == "fixed"
    assert payload[0]["recall_at"]["1"] == 1.0

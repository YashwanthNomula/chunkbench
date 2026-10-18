"""Tests for the overlap chunking strategy and per-strategy report stats."""
import json

from chunkbench import STRATEGIES, chunkers
from chunkbench.chunkers import fixed_chunks, overlap_chunks
from chunkbench.cli import main
from chunkbench.corpus import DOCS, QUESTIONS
from chunkbench.eval import evaluate


def _coverage(chunks, n):
    covered = set()
    for c in chunks:
        covered.update(range(c.start, c.end))
    return covered


def test_overlap_windows_share_requested_overlap_and_tile():
    text = "abcdefghij" * 105  # 1050 chars, no whitespace
    chunks = overlap_chunks(text, size=200, overlap=40)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    for a, b in zip(chunks, chunks[1:]):
        assert b.start == a.end - 40  # exact requested overlap, no gap
    for c in chunks:
        assert text[c.start:c.end] == c.text
    # full coverage: no gaps, no dropped tail
    assert _coverage(chunks, len(text)) == set(range(len(text)))
    # the tail is kept even though it is shorter than the window
    assert len(chunks[-1].text) < 200


def test_overlap_default_is_twenty_percent_of_window():
    text = "x" * 2000
    chunks = overlap_chunks(text, size=500)
    assert len(chunks) > 1
    for a, b in zip(chunks, chunks[1:]):
        assert b.start == a.end - 100


def test_overlap_zero_degenerates_to_fixed():
    for text, size in [("abcdefghij" * 30, 100), ("hello world, " * 40, 50)]:
        overlapped = overlap_chunks(text, size=size, overlap=0)
        fixed = fixed_chunks(text, size=size, overlap=0)
        assert [(c.text, c.start, c.end) for c in overlapped] == [
            (c.text, c.start, c.end) for c in fixed
        ]


def test_overlap_rejects_bad_params():
    import pytest

    with pytest.raises(ValueError):
        overlap_chunks("abc", size=100, overlap=100)
    with pytest.raises(ValueError):
        overlap_chunks("abc", size=100, overlap=-1)
    with pytest.raises(ValueError):
        overlap_chunks("abc", size=0)


def test_stats_correct_on_known_doc():
    # 11-char doc, size 4, overlap 1 -> windows [0:4] [3:7] [6:10] [9:11]
    docs = [{"id": "a", "title": "a", "text": "abcdefghijk"}]
    qs = [{"text": "abc", "gold_doc": "a", "must_contain": ["abc"]}]
    (res,) = evaluate(
        docs, qs,
        {"overlap": lambda t, doc_id="": overlap_chunks(
            t, size=4, overlap=1, doc_id=doc_id)},
        ks=(1,),
    )
    assert res.n_chunks == 4
    assert res.avg_chunk_chars == (4 + 4 + 4 + 2) / 4  # 3.5


def test_benchmark_runs_with_four_strategies():
    strategies = {name: fn for name, (fn, _desc) in STRATEGIES.items()}
    assert set(strategies) == {"fixed", "overlap", "recursive", "semantic"}
    results = evaluate(DOCS, QUESTIONS, strategies, ks=(1, 3, 5))
    assert len(results) == 4
    for r in results:
        assert len(r.rows) == len(QUESTIONS)
        assert r.n_chunks > 0
        assert r.avg_chunk_chars > 0
    overlap_res = next(r for r in results if r.strategy == "overlap")
    assert overlap_res.params == {"size": 500, "overlap": 100}


def test_cli_strategies_lists_overlap(capsys):
    assert main(["strategies"]) == 0
    assert "overlap" in capsys.readouterr().out


def test_cli_run_json_has_four_strategies_with_stats(capsys):
    assert main(["run", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {row["strategy"] for row in payload} == {
        "fixed", "overlap", "recursive", "semantic"}
    for row in payload:
        assert row["n_chunks"] > 0
        assert row["avg_chunk_chars"] > 0


def test_cli_overlap_chars_flag(capsys):
    assert main(["run", "--strategy", "overlap",
                 "--overlap-chars", "100", "--json"]) == 0
    (row,) = json.loads(capsys.readouterr().out)
    assert row["strategy"] == "overlap"
    assert row["params"] == {"size": 500, "overlap": 100}


def test_cli_overlap_defaults_to_twenty_percent(capsys):
    assert main(["run", "--strategy", "overlap",
                 "--size", "400", "--json"]) == 0
    (row,) = json.loads(capsys.readouterr().out)
    assert row["params"] == {"size": 400, "overlap": 80}

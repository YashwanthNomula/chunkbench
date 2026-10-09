"""CLI tests: demo, strategies, and run on a custom corpus."""
import json

from chunkbench.cli import main


def test_demo_runs(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "recall@5" in out
    assert "winner" in out


def test_strategies_lists_all(capsys):
    assert main(["strategies"]) == 0
    out = capsys.readouterr().out
    assert "fixed" in out and "recursive" in out and "semantic" in out


def test_run_subset_strategy(capsys):
    assert main(["run", "--strategy", "fixed"]) == 0
    out = capsys.readouterr().out
    assert "fixed" in out
    assert "recursive" not in out


def test_run_rejects_unknown_strategy(capsys):
    assert main(["run", "--strategy", "nope"]) == 2


def test_run_custom_corpus(tmp_path, capsys):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "alpha.md").write_text(
        "The crimson fox leaps over tall fences every single morning."
    )
    (docs / "beta.md").write_text(
        "Quantum processors rely on qubits for parallel calculation."
    )
    qfile = tmp_path / "q.json"
    qfile.write_text(json.dumps([
        {"text": "what leaps over fences", "gold_doc": "alpha",
         "must_contain": ["crimson fox"]},
    ]))
    assert main(["run", "--docs", str(docs), "--questions", str(qfile),
                 "--strategy", "fixed,recursive"]) == 0
    out = capsys.readouterr().out
    assert "recall@1" in out


def test_run_json_output(tmp_path, capsys):
    assert main(["run", "--strategy", "fixed", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["strategy"] == "fixed"

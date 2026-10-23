"""Tests for the chunkbench size/overlap grid sweep."""
import json

import pytest

from chunkbench.cli import main
from chunkbench.sweep import (MAX_RUNS, build_configs, parse_frac_list,
                              parse_int_list, pick_best, run_sweep)

DOCS = [
    {"id": "alpha", "text": "The crimson fox leaps over tall fences every single morning. " * 8},
    {"id": "beta", "text": "Quantum processors rely on qubits for parallel calculation. " * 8},
]
QUESTIONS = [
    {"text": "what leaps over fences", "gold_doc": "alpha",
     "must_contain": ["crimson fox"]},
    {"text": "what do quantum processors rely on", "gold_doc": "beta",
     "must_contain": ["qubits"]},
]


def test_parse_int_list():
    assert parse_int_list("250,500,1000") == [250, 500, 1000]
    assert parse_int_list("1000, 250, 250") == [250, 1000]


def test_parse_int_list_rejects():
    for bad in ["", "250,,500", "abc", "250,1.5", "0", "-10"]:
        with pytest.raises(ValueError):
            parse_int_list(bad)


def test_parse_frac_list():
    assert parse_frac_list("0,0.1,0.25") == [0.0, 0.1, 0.25]
    assert parse_frac_list("0.25, 0") == [0.0, 0.25]


def test_parse_frac_list_rejects():
    for bad in ["", "0,", "1.0", "1.5", "-0.1", "nan", "inf", "abc"]:
        with pytest.raises(ValueError):
            parse_frac_list(bad)


def test_build_configs_grid_shape():
    cfgs = build_configs(["fixed", "overlap"], [250, 500], [0.0, 0.1])
    assert len(cfgs) == 8
    cell = next(c for c in cfgs if c["strategy"] == "fixed" and c["size"] == 500
                and c["overlap_frac"] == 0.1)
    assert cell["overlap_chars"] == 50
    assert cell["fn"].chunkbench_params == {"size": 500, "overlap": 50}


def test_build_configs_semantic_overlap_is_na():
    (cfg,) = build_configs(["semantic"], [500], [0.0])
    assert cfg["overlap_chars"] is None
    chunks = cfg["fn"]("one two three four five. " * 40, doc_id="d")
    assert all(c.doc_id == "d" for c in chunks)


def test_run_sweep_scores_each_config():
    cfgs = build_configs(["fixed"], [50, 100], [0.0, 0.25])
    rows = run_sweep(DOCS, QUESTIONS, cfgs)
    assert len(rows) == 4
    for row in rows:
        r = row["result"]
        assert r.recall_at[5] in (0.0, 0.5, 1.0)


def test_run_sweep_needs_questions():
    cfgs = build_configs(["fixed"], [100], [0.0])
    with pytest.raises(ValueError):
        run_sweep(DOCS, [], cfgs)


def test_pick_best_prefers_recall_then_mrr():
    cfgs = build_configs(["fixed"], [50], [0.0, 0.25])
    rows = run_sweep(DOCS, QUESTIONS, cfgs)
    best = pick_best(rows)
    k = 5
    top = max((r["result"].recall_at[k], r["result"].mrr) for r in rows)
    assert (best["result"].recall_at[k], best["result"].mrr) == top


def test_cli_sweep_runs_small_grid(capsys):
    assert main(["sweep", "--strategy", "fixed", "--sizes", "200",
                 "--overlaps", "0,0.25"]) == 0
    out = capsys.readouterr().out
    assert "best config:" in out
    assert "recall@5" in out


def test_cli_sweep_json(capsys):
    assert main(["sweep", "--strategy", "overlap", "--sizes", "200",
                 "--overlaps", "0.25", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out.split("\n\n", 1)[1])
    assert payload["best"]["strategy"] == "overlap"
    assert payload["best"]["overlap_chars"] == 50
    assert len(payload["configs"]) == 1


def test_cli_sweep_rejects_bad_grid(capsys):
    assert main(["sweep", "--sizes", "abc"]) == 2
    assert main(["sweep", "--overlaps", "1.5"]) == 2
    assert main(["sweep", "--strategy", "nope"]) == 2


def test_cli_sweep_grid_cap():
    # 4 strategies x 60 sizes x 1 overlap = 240 runs > MAX_RUNS
    sizes = ",".join(str(100 + i) for i in range(60))
    assert main(["sweep", "--sizes", sizes, "--overlaps", "0"]) == 2
    assert MAX_RUNS == 200


def test_cli_sweep_no_docs_fails(tmp_path, capsys):
    empty = tmp_path / "docs"
    empty.mkdir()
    qfile = tmp_path / "q.json"
    qfile.write_text(json.dumps(QUESTIONS))
    rc = main(["sweep", "--docs", str(empty), "--questions", str(qfile),
               "--sizes", "200", "--overlaps", "0"])
    assert rc == 2
    assert "bad input" in capsys.readouterr().err

"""Configurable recall cutoffs (--ks) on `chunkbench run`."""
import json

import pytest

from chunkbench.cli import main, parse_ks


def test_parse_ks_default():
    assert parse_ks("1,3,5") == (1, 3, 5)


def test_parse_ks_normalizes_duplicates_and_order():
    assert parse_ks("5,1,5,3") == (1, 3, 5)


@pytest.mark.parametrize("bad", ["", "0", "-1", "11", "a", "1,,3", "1.5"])
def test_parse_ks_rejects_invalid(bad):
    with pytest.raises(ValueError):
        parse_ks(bad)


def test_default_run_reports_recall_1_3_5(capsys):
    assert main(["run", "--strategy", "fixed"]) == 0
    out = capsys.readouterr().out
    assert "recall@1" in out and "recall@3" in out and "recall@5" in out


def test_default_run_json_keys(capsys):
    assert main(["run", "--strategy", "fixed", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert list(payload[0]["recall_at"]) == ["1", "3", "5"]


def test_ks_1_10_terminal_and_json(capsys):
    assert main(["run", "--strategy", "fixed", "--ks", "1,10"]) == 0
    out = capsys.readouterr().out
    assert "recall@10" in out
    assert "recall@3" not in out
    assert main(["run", "--strategy", "fixed", "--ks", "1,10", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert list(payload[0]["recall_at"]) == ["1", "10"]


def test_ks_duplicate_unsorted_normalizes(capsys):
    assert main(["run", "--strategy", "fixed", "--ks", "5,1,5,3",
                 "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert list(payload[0]["recall_at"]) == ["1", "3", "5"]


@pytest.mark.parametrize("bad", ["0", "-1", "11", "a", ""])
def test_ks_invalid_exits_2(bad, capsys):
    assert main(["run", "--strategy", "fixed", "--ks", bad]) == 2
    assert capsys.readouterr().err != ""


def test_ks_is_run_only():
    with pytest.raises(SystemExit):
        main(["demo", "--ks", "1,2"])

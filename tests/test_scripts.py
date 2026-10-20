import runpy
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import analyze_reporting, run_experiment


class SelectQuery:
    def __init__(self, pages=None, count=None):
        self.pages = list(pages or [])
        self.response_count = count
        self.head = False

    def select(self, *args, **kwargs):
        self.head = bool(kwargs.get("head"))
        return self

    def range(self, start, end):
        return self

    def execute(self):
        if self.head:
            self.head = False
            return SimpleNamespace(data=[], count=self.response_count)
        data = self.pages.pop(0) if self.pages else []
        return SimpleNamespace(data=data, count=self.response_count)


class SelectClient:
    def __init__(self, queries):
        self.queries = queries

    def table(self, name):
        return self.queries[name]


def test_reporting_analysis_helpers():
    client = SelectClient({"table": SelectQuery(count=3)})
    assert analyze_reporting.count_rows(client, "table") == 3
    client = SelectClient({"table": SelectQuery(count=None)})
    assert analyze_reporting.count_rows(client, "table") == 0

    client = SelectClient({"table": SelectQuery(pages=[[{"id": 1}], []])})
    assert analyze_reporting.fetch_all(client, "table", "id", page_size=1) == [
        {"id": 1}
    ]
    assert analyze_reporting.format_breakdown(Counter()) == "none"
    assert analyze_reporting.format_breakdown(Counter({"b": 1, "a": 2})) == "a=2, b=1"


def test_reporting_analysis_main(monkeypatch, capsys):
    tables = {
        table: SelectQuery(count=index)
        for index, table in enumerate(analyze_reporting.TABLES.values(), start=1)
    }
    tables["experiment_runs"] = SelectQuery(
        pages=[
            [
                {
                    "status": "completed",
                    "started_at": "2026-01-01",
                    "target_model": "target",
                    "user_sim_model": "user",
                    "evaluator_model": "evaluator",
                }
            ]
        ],
        count=1,
    )
    tables["conversations"] = SelectQuery(pages=[[{"status": "completed"}]], count=2)
    monkeypatch.setattr(
        analyze_reporting, "create_reporting_client", lambda: SelectClient(tables)
    )
    analyze_reporting.main()
    output = capsys.readouterr().out
    assert "Supabase reporting summary" in output
    assert "First run: 2026-01-01" in output

    monkeypatch.setattr(
        analyze_reporting,
        "create_reporting_client",
        lambda: SelectClient(
            {
                table: SelectQuery(pages=[[]], count=0)
                for table in analyze_reporting.TABLES.values()
            }
        ),
    )
    analyze_reporting.main()
    assert "Run statuses: none" in capsys.readouterr().out


def test_run_experiment_argument_parsing(monkeypatch):
    assert run_experiment.parse_bool("TRUE") is True
    assert run_experiment.parse_bool("false") is False
    with pytest.raises(Exception, match="expected true or false"):
        run_experiment.parse_bool("maybe")
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--single-run"])
    assert run_experiment.parse_args().single_run


def test_run_experiment_main_modes(monkeypatch):
    calls = []
    monkeypatch.setattr(run_experiment, "load_vignettes", lambda path: ["vignette"])
    monkeypatch.setattr(
        run_experiment,
        "run_experiment",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        run_experiment,
        "parse_args",
        lambda: SimpleNamespace(
            vignettes=Path("v.json"),
            csv_output=Path("out.csv"),
            jsonl_output=Path("out.jsonl"),
            replicates_per_cell=12,
            max_workers=8,
            max_cells=4,
            single_run=True,
            reporting=True,
        ),
    )
    run_experiment.main()
    assert calls[-1]["replicates_per_cell"] == 1
    assert calls[-1]["max_cells"] == 1

    monkeypatch.setattr(
        run_experiment,
        "parse_args",
        lambda: SimpleNamespace(
            vignettes=Path("v.json"),
            csv_output=Path("out.csv"),
            jsonl_output=Path("out.jsonl"),
            replicates_per_cell=12,
            max_workers=8,
            max_cells=4,
            single_run=False,
            reporting=False,
        ),
    )
    run_experiment.main()
    assert calls[-1]["replicates_per_cell"] == 12
    assert calls[-1]["max_cells"] == 4


def test_script_main_guards(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--single-run"])
    monkeypatch.setattr("belief_revision.storage.load_vignettes", lambda path: [])
    monkeypatch.setattr("belief_revision.runner.run_experiment", lambda **kwargs: 0)
    runpy.run_path("scripts/run_experiment.py", run_name="__main__")

    empty_client = SelectClient(
        {
            table: SelectQuery(pages=[[]], count=0)
            for table in analyze_reporting.TABLES.values()
        }
    )
    monkeypatch.setattr(
        "belief_revision.reporting.create_reporting_client", lambda: empty_client
    )
    runpy.run_path("scripts/analyze_reporting.py", run_name="__main__")

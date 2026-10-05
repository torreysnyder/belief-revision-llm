from dataclasses import replace

import pytest

from belief_revision import runner
from belief_revision.models import ConversationResult


def make_result(cell, plan):
    return ConversationResult(cell, plan, [], [], [], {})


def test_run_primary_cell_handles_success_and_failure(
    monkeypatch, primary_cell, run_plan
):
    plans = [run_plan, replace(run_plan, replicate_index=1)]
    monkeypatch.setattr(runner, "build_run_plans", lambda *args: plans)

    def run(primary_cell, run_plan):
        if run_plan.replicate_index == 1:
            raise RuntimeError("failed")
        return make_result(primary_cell, run_plan)

    monkeypatch.setattr(runner, "run_one_vignette", run)
    results = runner.run_primary_cell(primary_cell, 2, 1)
    assert len(results) == 1


class Reporter:
    def __init__(self):
        self.calls = []

    def start_run(self, **kwargs):
        self.calls.append(("start", kwargs))
        return "00000000-0000-0000-0000-000000000001"

    def save_conversations(self, run_id, results):
        self.calls.append(("save", results))

    def complete_run(self, run_id, **kwargs):
        self.calls.append(("complete", kwargs))

    def fail_run(self, run_id, error):
        self.calls.append(("fail", str(error)))


def test_run_experiment_local_and_reported(
    monkeypatch, tmp_path, vignette, primary_cell, run_plan
):
    result = make_result(primary_cell, run_plan)
    monkeypatch.setattr(
        runner,
        "build_primary_cells",
        lambda values: [primary_cell, primary_cell] if values else [],
    )
    monkeypatch.setattr(runner, "run_primary_cell", lambda **kwargs: [result])
    saved = []
    monkeypatch.setattr(
        runner, "save_results_append", lambda *args, **kwargs: saved.append(kwargs)
    )
    reporter = Reporter()
    monkeypatch.setattr(runner.SupabaseReporter, "from_environment", lambda: reporter)
    csv_path = tmp_path / "existing.csv"
    csv_path.write_text("old")

    count = runner.run_experiment(
        [vignette], csv_path, tmp_path / "out.jsonl", 1, 1, max_cells=1, reporting=True
    )
    assert count == 1
    assert not csv_path.exists()
    assert [call[0] for call in reporter.calls] == ["start", "save", "complete"]
    assert saved[0]["write_header"] is True

    count = runner.run_experiment(
        [], tmp_path / "none.csv", tmp_path / "none.jsonl", reporting=False
    )
    assert count == 0


def test_run_experiment_reports_failure(monkeypatch, tmp_path, vignette, primary_cell):
    monkeypatch.setattr(runner, "build_primary_cells", lambda values: [primary_cell])
    monkeypatch.setattr(runner, "run_primary_cell", lambda **kwargs: [])
    monkeypatch.setattr(
        runner,
        "save_results_append",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("storage")),
    )
    reporter = Reporter()
    monkeypatch.setattr(runner.SupabaseReporter, "from_environment", lambda: reporter)
    with pytest.raises(RuntimeError, match="storage"):
        runner.run_experiment(
            [vignette], tmp_path / "x.csv", tmp_path / "x.jsonl", reporting=True
        )
    assert reporter.calls[-1] == ("fail", "storage")


def test_run_experiment_empty_cell_and_local_failure(
    monkeypatch, tmp_path, vignette, primary_cell
):
    monkeypatch.setattr(runner, "build_primary_cells", lambda values: [primary_cell])
    monkeypatch.setattr(runner, "run_primary_cell", lambda **kwargs: [])
    monkeypatch.setattr(runner, "save_results_append", lambda *args, **kwargs: None)
    assert (
        runner.run_experiment(
            [vignette], tmp_path / "a.csv", tmp_path / "a.jsonl", reporting=False
        )
        == 0
    )

    monkeypatch.setattr(
        runner,
        "save_results_append",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("local")),
    )
    with pytest.raises(RuntimeError, match="local"):
        runner.run_experiment(
            [vignette], tmp_path / "b.csv", tmp_path / "b.jsonl", reporting=False
        )

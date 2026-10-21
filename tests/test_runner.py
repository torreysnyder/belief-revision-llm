import pytest
from belief_revision.core import runner
from belief_revision.core.models import ConversationResult
from dataclasses import replace


def make_result(cell, plan):
    return ConversationResult(cell, plan, [], [], [], {})


def test_run_primary_cell_handles_success_and_failure(monkeypatch, capsys, primary_cell, run_plan):
    plans = [run_plan, replace(run_plan, replicate_index=1)]
    monkeypatch.setattr(runner, "build_run_plans", lambda *args: plans)

    def run(primary_cell, run_plan, extensions=(), evaluator_mode="enabled", **kwargs):
        if run_plan.replicate_index == 1:
            raise RuntimeError("failed")
        return make_result(primary_cell, run_plan)

    monkeypatch.setattr(runner, "run_one_vignette", run)
    results = runner.run_primary_cell(primary_cell, 2, 1)
    assert len(results) == 1
    captured = capsys.readouterr()
    assert "cell=cell-1 replicate=1: failed" in captured.out
    assert "RuntimeError: failed" in captured.err


def test_run_primary_cell_builds_fresh_extensions_for_each_replicate(monkeypatch, primary_cell, run_plan):
    plans = [run_plan, replace(run_plan, replicate_index=1)]
    monkeypatch.setattr(runner, "build_run_plans", lambda *args: plans)
    received_extensions = []

    class Extension:
        pass

    def run(primary_cell, run_plan, extensions, evaluator_mode, **kwargs):
        received_extensions.append(extensions)
        assert evaluator_mode == "disabled"
        return make_result(primary_cell, run_plan)

    monkeypatch.setattr(runner, "run_one_vignette", run)

    results = runner.run_primary_cell(
        primary_cell,
        replicates_per_cell=2,
        max_workers=1,
        extension_factories=(Extension,),
        evaluator_mode="disabled",
    )

    assert len(results) == 2
    assert len(received_extensions) == 2
    assert received_extensions[0] is not received_extensions[1]
    assert received_extensions[0][0] is not received_extensions[1][0]


def test_run_primary_cell_contains_extension_failure_to_one_replicate(monkeypatch, primary_cell, run_plan):
    plans = [run_plan, replace(run_plan, replicate_index=1)]
    monkeypatch.setattr(runner, "build_run_plans", lambda *args: plans)

    class Extension:
        pass

    def run(primary_cell, run_plan, extensions, evaluator_mode, **kwargs):
        if run_plan.replicate_index == 1:
            raise RuntimeError("hook failed")
        return make_result(primary_cell, run_plan)

    monkeypatch.setattr(runner, "run_one_vignette", run)

    results = runner.run_primary_cell(
        primary_cell,
        replicates_per_cell=2,
        max_workers=1,
        extension_factories=(Extension,),
    )

    assert [result.run_plan.replicate_index for result in results] == [0]


def test_run_primary_cell_contains_extension_factory_failure_to_one_replicate(monkeypatch, primary_cell, run_plan):
    plans = [run_plan, replace(run_plan, replicate_index=1)]
    monkeypatch.setattr(runner, "build_run_plans", lambda *args: plans)
    factory_calls = 0

    class Extension:
        pass

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 2:
            raise RuntimeError("factory failed")
        return Extension()

    def run(primary_cell, run_plan, extensions, evaluator_mode, **kwargs):
        return make_result(primary_cell, run_plan)

    monkeypatch.setattr(runner, "run_one_vignette", run)

    results = runner.run_primary_cell(
        primary_cell,
        replicates_per_cell=2,
        max_workers=1,
        extension_factories=(factory,),
    )

    assert [result.run_plan.replicate_index for result in results] == [0]


class Reporter:
    def __init__(self, save_error=None):
        self.calls = []
        self.save_error = save_error

    def start_run(self, **kwargs):
        self.calls.append(("start", kwargs))
        return "00000000-0000-0000-0000-000000000001"

    def save_conversations(self, run_id, results):
        if self.save_error:
            raise self.save_error
        self.calls.append(("save", results))

    def complete_run(self, run_id, **kwargs):
        self.calls.append(("complete", kwargs))

    def fail_run(self, run_id, error):
        self.calls.append(("fail", str(error)))


def test_run_experiment_local_and_reported(monkeypatch, tmp_path, vignette, primary_cell, run_plan):
    result = make_result(primary_cell, run_plan)
    primary_cell_calls = []
    monkeypatch.setattr(
        runner,
        "build_primary_cells",
        lambda values: [primary_cell, primary_cell] if values else [],
    )

    def run_primary_cell(**kwargs):
        primary_cell_calls.append(kwargs)
        return [result]

    monkeypatch.setattr(runner, "run_primary_cell", run_primary_cell)
    local = Reporter()
    monkeypatch.setattr(runner, "LocalFileBackend", lambda *args: local)
    reporter = Reporter()
    monkeypatch.setattr(runner.SupabaseReporter, "from_environment", lambda: reporter)
    csv_path = tmp_path / "existing.csv"
    csv_path.write_text("old")

    extension_factories = (lambda: object(),)
    count = runner.run_experiment(
        [vignette],
        csv_path,
        tmp_path / "out.jsonl",
        1,
        1,
        max_cells=1,
        reporting=True,
        extension_factories=extension_factories,
        evaluator_mode="disabled",
    )
    assert count == 1
    assert [call[0] for call in reporter.calls] == ["start", "save", "complete"]
    assert [call[0] for call in local.calls] == ["start", "save", "complete"]
    assert primary_cell_calls[0]["extension_factories"] is extension_factories
    assert primary_cell_calls[0]["evaluator_mode"] == "disabled"
    assert primary_cell_calls[0]["evaluator_provider"] == runner.EVALUATOR_PROVIDER
    assert reporter.calls[0][1]["config"]["evaluator_mode"] == "disabled"

    local_empty = Reporter()
    monkeypatch.setattr(runner, "LocalFileBackend", lambda *args: local_empty)
    count = runner.run_experiment([], tmp_path / "none.csv", tmp_path / "none.jsonl", reporting=False)
    assert count == 0


def test_run_experiment_reports_failure(monkeypatch, tmp_path, vignette, primary_cell):
    monkeypatch.setattr(runner, "build_primary_cells", lambda values: [primary_cell])
    monkeypatch.setattr(runner, "run_primary_cell", lambda **kwargs: [])
    local = Reporter(save_error=RuntimeError("storage"))
    monkeypatch.setattr(runner, "LocalFileBackend", lambda *args: local)
    reporter = Reporter()
    monkeypatch.setattr(runner.SupabaseReporter, "from_environment", lambda: reporter)
    with pytest.raises(RuntimeError, match="storage"):
        runner.run_experiment([vignette], tmp_path / "x.csv", tmp_path / "x.jsonl", reporting=True)
    assert reporter.calls[-1] == ("fail", "storage")


def test_run_experiment_empty_cell_and_local_failure(monkeypatch, tmp_path, vignette, primary_cell):
    monkeypatch.setattr(runner, "build_primary_cells", lambda values: [primary_cell])
    monkeypatch.setattr(runner, "run_primary_cell", lambda **kwargs: [])
    local = Reporter()
    monkeypatch.setattr(runner, "LocalFileBackend", lambda *args: local)
    assert runner.run_experiment([vignette], tmp_path / "a.csv", tmp_path / "a.jsonl", reporting=False) == 0

    failing = Reporter(save_error=RuntimeError("local"))
    monkeypatch.setattr(runner, "LocalFileBackend", lambda *args: failing)
    with pytest.raises(RuntimeError, match="local"):
        runner.run_experiment([vignette], tmp_path / "b.csv", tmp_path / "b.jsonl", reporting=False)

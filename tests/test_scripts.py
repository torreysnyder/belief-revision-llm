import sys
import runpy
import pytest
from collections import Counter
from pathlib import Path
from scripts import analyze_reporting, run_experiment
from types import SimpleNamespace


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
    assert analyze_reporting.fetch_all(client, "table", "id", page_size=1) == [{"id": 1}]
    assert analyze_reporting.format_breakdown(Counter()) == "none"
    assert analyze_reporting.format_breakdown(Counter({"b": 1, "a": 2})) == "a=2, b=1"


def test_reporting_analysis_main(monkeypatch, capsys):
    tables = {table: SelectQuery(count=index) for index, table in enumerate(analyze_reporting.TABLES.values(), start=1)}
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
    monkeypatch.setattr(analyze_reporting, "create_reporting_client", lambda: SelectClient(tables))
    analyze_reporting.main()
    output = capsys.readouterr().out
    assert "Supabase reporting summary" in output
    assert "First run: 2026-01-01" in output

    monkeypatch.setattr(
        analyze_reporting,
        "create_reporting_client",
        lambda: SelectClient({table: SelectQuery(pages=[[]], count=0) for table in analyze_reporting.TABLES.values()}),
    )
    analyze_reporting.main()
    assert "Run statuses: none" in capsys.readouterr().out


def test_run_experiment_argument_parsing(monkeypatch):
    assert run_experiment.parse_bool("TRUE") is True
    assert run_experiment.parse_bool("false") is False
    with pytest.raises(Exception, match="expected true or false"):
        run_experiment.parse_bool("maybe")
    assert run_experiment.parse_csv_strings("intro, closing") == frozenset({"intro", "closing"})
    assert run_experiment.parse_csv_positive_ints("1,3,1") == (1, 3)
    with pytest.raises(Exception, match="at least one"):
        run_experiment.parse_csv_strings(" , ")
    with pytest.raises(Exception, match="positive integers"):
        run_experiment.parse_csv_positive_ints("one")
    with pytest.raises(Exception, match="positive integers"):
        run_experiment.parse_csv_positive_ints("0")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--single-run",
            "--config",
            "configs/torrey-caa.yaml",
            "--extension",
            "first",
            "--extension",
            "second",
            "--torrey-caa-phases",
            "intro,closing",
            "--torrey-caa-turns",
            "1,2",
            "--torrey-caa-layers",
            "1,13,24",
            "--torrey-caa-output-dir",
            "captures",
            "--evaluator-mode",
            "disabled",
        ],
    )
    args = run_experiment.parse_args()
    assert args.single_run
    assert args.config == Path("configs/torrey-caa.yaml")
    assert args.extension == ["first", "second"]
    assert args.torrey_caa_phases == frozenset({"intro", "closing"})
    assert args.torrey_caa_turns == (1, 2)
    assert args.torrey_caa_layers == (1, 13, 24)
    assert args.torrey_caa_output_dir == Path("captures")
    assert args.evaluator_mode == "disabled"


def test_run_experiment_main_modes(monkeypatch):
    calls = []
    resolution_calls = []
    first_factory = object()
    second_factory = object()
    monkeypatch.setattr(run_experiment, "load_vignettes", lambda path: ["vignette"])
    monkeypatch.setattr(
        run_experiment,
        "resolve_extension_factories",
        lambda names, configurations: resolution_calls.append((names, configurations)) or ((first_factory, second_factory) if names else ()),
    )
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
            extension=["first", "second"],
            config=None,
            evaluator_mode="disabled",
            torrey_caa_phases=frozenset({"closing"}),
            torrey_caa_turns=(1,),
            torrey_caa_layers=(1, 13),
            torrey_caa_output_dir=Path("captures"),
        ),
    )
    run_experiment.main()
    assert calls[-1]["replicates_per_cell"] == 1
    assert calls[-1]["max_cells"] == 1
    assert calls[-1]["extension_factories"] == (first_factory, second_factory)
    assert calls[-1]["evaluator_mode"] == "disabled"
    assert resolution_calls[-1][1]["torrey-caa"] == {
        "output_dir": Path("captures"),
        "layers": (0, 12),
        "phases": frozenset({"closing"}),
        "turns": frozenset({1}),
    }

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
            extension=[],
            config=None,
            evaluator_mode=None,
            torrey_caa_phases=None,
            torrey_caa_turns=None,
            torrey_caa_layers=None,
            torrey_caa_output_dir=Path("captures"),
        ),
    )
    run_experiment.main()
    assert calls[-1]["replicates_per_cell"] == 12
    assert calls[-1]["max_cells"] == 4
    assert calls[-1]["extension_factories"] == ()
    assert calls[-1]["evaluator_mode"] == "enabled"
    assert resolution_calls[-1][1]["torrey-caa"]["layers"] is None
    assert resolution_calls[-1][1]["torrey-caa"]["turns"] is None


def test_run_experiment_main_rejects_unknown_extension(monkeypatch):
    monkeypatch.setattr(
        run_experiment,
        "parse_args",
        lambda: SimpleNamespace(
            extension=["missing"],
            config=None,
            evaluator_mode=None,
            torrey_caa_phases=None,
            torrey_caa_turns=None,
            torrey_caa_layers=None,
            torrey_caa_output_dir=Path("captures"),
        ),
    )
    monkeypatch.setattr(
        run_experiment,
        "resolve_extension_factories",
        lambda names, configurations: (_ for _ in ()).throw(ValueError("unknown extension 'missing'; available extensions: <none>")),
    )

    with pytest.raises(SystemExit, match="unknown extension 'missing'"):
        run_experiment.main()


def test_load_and_merge_torrey_yaml_config(tmp_path):
    config_path = tmp_path / "torrey.yaml"
    config_path.write_text(
        """
extensions:
  torrey-caa:
    phases: [intro, closing]
    turns: [1, 2, 1]
    layers: [1, 13]
    output_dir: custom-captures
""",
        encoding="utf-8",
    )
    loaded = run_experiment.extension_configuration(run_experiment.load_config(config_path))
    args = SimpleNamespace(
        torrey_caa_phases=None,
        torrey_caa_turns=None,
        torrey_caa_layers=None,
        torrey_caa_output_dir=None,
    )

    assert run_experiment.torrey_caa_configuration(args, loaded) == {
        "output_dir": Path("custom-captures"),
        "layers": (0, 12),
        "phases": frozenset({"intro", "closing"}),
        "turns": frozenset({1, 2}),
    }

    args.torrey_caa_phases = frozenset({"closing"})
    args.torrey_caa_turns = (3,)
    args.torrey_caa_layers = (24,)
    args.torrey_caa_output_dir = Path("override")
    assert run_experiment.torrey_caa_configuration(args, loaded) == {
        "output_dir": Path("override"),
        "layers": (23,),
        "phases": frozenset({"closing"}),
        "turns": frozenset({3}),
    }


def test_empty_extension_config_is_valid(tmp_path):
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")
    assert run_experiment.load_config(None) == {}
    assert run_experiment.load_config(config_path) == {}


def test_evaluator_configuration_and_cli_mode_override():
    config = {
        "evaluator": {
            "mode": "disabled",
            "provider": "openai",
            "model": "gpt-test",
            "max_tokens": 123,
        }
    }
    assert run_experiment.evaluator_configuration(SimpleNamespace(evaluator_mode=None), config) == {
        "mode": "disabled",
        "provider": "openai",
        "model": "gpt-test",
        "max_tokens": 123,
    }
    assert run_experiment.evaluator_configuration(SimpleNamespace(evaluator_mode="enabled"), config)["mode"] == "enabled"
    defaults = run_experiment.evaluator_configuration(SimpleNamespace(evaluator_mode=None), {})
    assert defaults["mode"] == "enabled"
    assert defaults["provider"] == run_experiment.EVALUATOR_PROVIDER


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"evaluator": []}, "'evaluator' must be a mapping"),
        (
            {"evaluator": {"temperature": 0}},
            "unknown evaluator config field.*temperature",
        ),
        (
            {"evaluator": {"provider": "ndif"}},
            "provider must be an API provider, not NDIF",
        ),
        (
            {"evaluator": {"mode": "sometimes"}},
            "mode must be 'enabled' or 'disabled'",
        ),
        ({"evaluator": {"model": ""}}, "model must be a non-empty string"),
        ({"evaluator": {"max_tokens": 0}}, "max_tokens must be a positive"),
        ({"evaluator": {"max_tokens": True}}, "max_tokens must be a positive"),
    ],
)
def test_invalid_evaluator_config_fails(config, message):
    with pytest.raises(ValueError, match=message):
        run_experiment.evaluator_configuration(
            SimpleNamespace(evaluator_mode=None),
            config,
        )


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ("- not-a-mapping\n", "root must be a mapping"),
        ("extensions: []\n", "'extensions' must be a mapping"),
        ("extensions:\n  torrey-caa: []\n", "torrey-caa.*mapping"),
        (
            "extensions:\n  torrey-caa:\n    surprise: true\n",
            "unknown torrey-caa config field.*surprise",
        ),
        (
            "extensions:\n  torrey-caa:\n    phases: []\n",
            "phases must be a non-empty list",
        ),
        (
            "extensions:\n  torrey-caa:\n    phases: [intro, 2]\n",
            "phases must be a non-empty list",
        ),
        (
            "extensions:\n  torrey-caa:\n    turns: [0]\n",
            "turns must be a non-empty positive-integer list",
        ),
        (
            "extensions:\n  torrey-caa:\n    turns: [true]\n",
            "turns must be a non-empty positive-integer list",
        ),
        (
            "extensions:\n  torrey-caa:\n    layers: nope\n",
            "layers must be a non-empty positive-integer list",
        ),
    ],
)
def test_invalid_extension_configs_fail(tmp_path, document, message):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        loaded = run_experiment.extension_configuration(run_experiment.load_config(config_path))
        run_experiment.torrey_caa_configuration(
            SimpleNamespace(
                torrey_caa_phases=None,
                torrey_caa_turns=None,
                torrey_caa_layers=None,
                torrey_caa_output_dir=None,
            ),
            loaded,
        )


def test_script_main_guards(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--single-run"])
    monkeypatch.setattr("belief_revision.persistence.load_vignettes", lambda path: [])
    monkeypatch.setattr("belief_revision.core.runner.run_experiment", lambda **kwargs: 0)
    runpy.run_path("scripts/run_experiment.py", run_name="__main__")

    empty_client = SelectClient({table: SelectQuery(pages=[[]], count=0) for table in analyze_reporting.TABLES.values()})
    monkeypatch.setattr("belief_revision.persistence.create_reporting_client", lambda: empty_client)
    runpy.run_path("scripts/analyze_reporting.py", run_name="__main__")

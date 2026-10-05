import runpy
import sys
from types import SimpleNamespace

import pytest

from belief_revision.cli import validate_models


def test_validate_model_outcomes(monkeypatch, capsys):
    monkeypatch.setattr(validate_models, "call_model", lambda *args, **kwargs: "OK")
    assert validate_models.validate_model("user", "model")
    monkeypatch.setattr(validate_models, "call_model", lambda *args, **kwargs: "")
    assert not validate_models.validate_model("user", "model")
    monkeypatch.setattr(
        validate_models,
        "call_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert not validate_models.validate_model("user", "model")
    assert "FAIL" in capsys.readouterr().out


def test_validate_main_one_all_and_failure(monkeypatch):
    monkeypatch.setattr(
        validate_models, "parse_args", lambda: SimpleNamespace(all=False, role="user")
    )
    monkeypatch.setattr(validate_models, "validate_model", lambda *args: True)
    validate_models.main()
    monkeypatch.setattr(
        validate_models, "parse_args", lambda: SimpleNamespace(all=True, role=None)
    )
    validate_models.main()
    monkeypatch.setattr(validate_models, "validate_model", lambda *args: False)
    with pytest.raises(SystemExit):
        validate_models.main()


def test_validate_parse_args(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["validate-models", "--assistant"])
    assert validate_models.parse_args().role == "assistant"


def test_module_main_guard(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["validate-models", "--user"])
    monkeypatch.setattr("belief_revision.llm.call_model", lambda *args, **kwargs: "OK")
    runpy.run_module("belief_revision.cli.validate_models", run_name="__main__")

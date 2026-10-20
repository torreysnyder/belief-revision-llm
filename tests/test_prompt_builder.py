from dataclasses import dataclass

import pytest

from belief_revision.models import PromptTemplate
from belief_revision.prompt_builder import (
    _json_default,
    _to_json,
    find_prompt_variables,
    load_prompt_text,
    prepare_prompt_variables,
    render_prompt,
    validate_prompt_template,
)
from belief_revision.prompt_registry import make_prompt_template
from belief_revision.state import init_user_state


@dataclass
class Example:
    value: int


def test_json_helpers_and_structured_variables(vignette, style):
    state = init_user_state("curious", "low", "challenge", "low")
    assert _json_default(Example(1)) == {"value": 1}
    with pytest.raises(TypeError):
        _json_default(object())
    assert _to_json(None) == "None"

    variables = prepare_prompt_variables(
        {
            "vignette": vignette,
            "user_state": state,
            "style_profile": style,
            "dialogue_history": [Example(2)],
            "phase": {"name": "intro", "n_turns": 3},
            "turn_in_phase": 0,
            "evidence_role": None,
            "evidence_item": None,
            "_json_indent": 2,
        }
    )
    assert variables["core_claim"] == vignette.core_claim
    assert variables["turn_number"] == 1
    assert variables["evidence_role"] == "none"
    assert variables["evidence_text"] == "None"

    evidence_variables = prepare_prompt_variables(
        {"evidence_item": vignette.auxiliary_evidence[0]}
    )
    assert evidence_variables["evidence_text"] == "Auxiliary evidence"


def test_template_loading_validation_and_rendering(tmp_path):
    path = tmp_path / "prompt.md"
    path.write_text("Hello {{name}}")
    template = PromptTemplate("test", "1", path, ["name"])
    assert load_prompt_text(template) == "Hello {{name}}"
    assert find_prompt_variables("{{one}} {{two}}") == {"one", "two"}
    assert validate_prompt_template(template) == "Hello {{name}}"
    assert render_prompt(template, {"name": "world"}) == "Hello world"

    with pytest.raises(ValueError, match="prompt variables"):
        render_prompt(template, {"name": "x"}, name="y")
    with pytest.raises(ValueError, match="Missing variables"):
        render_prompt(template, {})
    with pytest.raises(FileNotFoundError):
        load_prompt_text(PromptTemplate("missing", "1", tmp_path / "no", []))

    mismatch = PromptTemplate("bad", "1", path, ["other"])
    with pytest.raises(ValueError, match="Prompt variable mismatch"):
        validate_prompt_template(mismatch)

    no_variables = PromptTemplate("none", "1", path, ["name"])
    assert render_prompt(no_variables, name="there") == "Hello there"


def test_prompt_registry_defaults():
    template = make_prompt_template("target_system")
    assert template.required_variables == []
    assert template.template_path.name == "target_system.md"

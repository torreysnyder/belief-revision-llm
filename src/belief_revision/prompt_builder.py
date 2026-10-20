"""Build model-ready prompts from the project's Markdown templates.

This module takes structured experiment data, such as a user state or vignette,
converts it into template values, and inserts those values into the appropriate
prompt without requiring each caller to repeat the conversion logic.
"""

import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .models import PromptTemplate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "prompts"

PLACEHOLDER_PATTERN = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}")


def _json_default(value: object) -> dict[str, Any]:
    """Convert a dataclass encountered during JSON serialization."""

    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _to_json(value: object, indent: int | None = None) -> str:
    """Serialize prompt context consistently."""

    if value is None:
        return "None"

    return json.dumps(
        value,
        indent=indent,
        ensure_ascii=False,
        default=_json_default,
    )


def prepare_prompt_variables(context: dict[str, object]) -> dict[str, object]:
    """Expand structured conversation context into template variables."""

    json_indent = context.get("_json_indent")
    variables = {
        name: value for name, value in context.items() if not name.startswith("_")
    }

    for name, value in context.items():
        if not name.startswith("_"):
            variables[f"{name}_json"] = _to_json(
                value,
                indent=json_indent,
            )

    vignette = context.get("vignette")
    if vignette is not None:
        variables.update(
            {
                "core_claim": vignette.core_claim,
                "displayed_A1": vignette.auxiliaries[0].text,
                "displayed_A2": vignette.auxiliaries[1].text,
                "auxiliary_text": "\n".join(
                    f"- {auxiliary.id}: {auxiliary.text}"
                    for auxiliary in vignette.auxiliaries
                ),
            }
        )

    user_state = context.get("user_state")
    if user_state is not None:
        variables.update(asdict(user_state))
        variables["active_auxiliaries_json"] = _to_json(user_state.active_auxiliaries)

    style_profile = context.get("style_profile")
    if style_profile is not None:
        variables.update(asdict(style_profile))
        variables["style_description"] = style_profile.description

    dialogue_history = context.get("dialogue_history")
    if dialogue_history is not None:
        variables["dialogue_history_json"] = _to_json(
            dialogue_history,
            indent=json_indent,
        )
        variables["recent_dialogue_json"] = variables["dialogue_history_json"]

    phase = context.get("phase")
    if phase is not None:
        variables["phase_name"] = phase["name"]
        variables["phase_turn_count"] = phase["n_turns"]

    turn_in_phase = context.get("turn_in_phase")
    if turn_in_phase is not None:
        variables["turn_number"] = turn_in_phase + 1

    evidence_role = context.get("evidence_role")
    variables["evidence_role"] = evidence_role or "none"

    evidence_item = context.get("evidence_item")
    variables["evidence_text"] = (
        evidence_item.text if evidence_item is not None else "None"
    )

    return variables


def load_prompt_text(template: PromptTemplate) -> str:
    """Load a prompt template from its Markdown file."""

    if not template.template_path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {template.template_path}")

    return template.template_path.read_text(encoding="utf-8").strip()


def find_prompt_variables(prompt_text: str) -> set[str]:
    """Return the variable names referenced by a prompt template."""

    return set(PLACEHOLDER_PATTERN.findall(prompt_text))


def validate_prompt_template(template: PromptTemplate) -> str:
    """Validate the template metadata and return its prompt text."""

    prompt_text = load_prompt_text(template)
    discovered = find_prompt_variables(prompt_text)
    declared = set(template.required_variables)

    if discovered != declared:
        missing_from_metadata = discovered - declared
        missing_from_file = declared - discovered

        raise ValueError(
            f"Prompt variable mismatch for {template.name}: "
            f"missing from metadata={sorted(missing_from_metadata)}, "
            f"missing from file={sorted(missing_from_file)}"
        )

    return prompt_text


def render_prompt(
    template: PromptTemplate,
    variables: dict[str, object] | None = None,
    **context: object,
) -> str:
    """Render a validated prompt template using the supplied variables."""

    if variables is not None and context:
        raise ValueError("Supply prompt variables or structured context, not both")

    if variables is None:
        variables = prepare_prompt_variables(context)

    prompt_text = validate_prompt_template(template)
    required = set(template.required_variables)
    supplied = set(variables)

    missing = required - supplied
    if missing:
        raise ValueError(f"Missing variables for {template.name}: {sorted(missing)}")

    return PLACEHOLDER_PATTERN.sub(
        lambda match: str(variables[match.group(1)]),
        prompt_text,
    )

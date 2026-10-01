import re
from pathlib import Path

from .models import PromptTemplate


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "prompts"

PLACEHOLDER_PATTERN = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}")

def load_prompt_text(template: PromptTemplate) -> str:
    """Load a prompt template from its Markdown file."""

    if not template.template_path.is_file():
        raise FileNotFoundError(
            f"Prompt template not found: {template.template_path}"
        )

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
    variables: dict[str, object],
) -> str:
    """Render a validated prompt template using the supplied variables."""

    prompt_text = validate_prompt_template(template)
    required = set(template.required_variables)
    supplied = set(variables)

    missing = required - supplied
    if missing:
        raise ValueError(
            f"Missing variables for {template.name}: {sorted(missing)}"
        )

    return PLACEHOLDER_PATTERN.sub(
        lambda match: str(variables[match.group(1)]),
        prompt_text,
    )

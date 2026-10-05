from .models import PromptTemplate
from .prompt_builder import PROMPTS_DIR


def make_prompt_template(
    name: str,
    required_variables: list[str] | None = None,
) -> PromptTemplate:
    """Create metadata for a versioned Markdown prompt."""

    return PromptTemplate(
        name=name,
        version="1.0",
        template_path=PROMPTS_DIR / f"{name}.md",
        required_variables=required_variables or [],
    )


PROMPT_TEMPLATES = {
    "behavioral_probe_system": make_prompt_template(
        name="behavioral_probe_system",
    ),
    "behavioral_probe_user": make_prompt_template(
        name="behavioral_probe_user",
        required_variables=[
            "core_claim",
            "displayed_A1",
            "displayed_A2",
            "phase_name",
            "dialogue_history_json",
            "previous_probe_json",
        ],
    ),
    "target_system": make_prompt_template(
        name="target_system",
    ),
    "user_intro": make_prompt_template(
        name="user_intro",
        required_variables=[
            "user_state_json",
            "recent_dialogue_json",
            "core_claim",
            "style_profile_json",
        ],
    ),
    "user_simulator_system": make_prompt_template(
        name="user_simulator_system",
        required_variables=[
            "belief_anchor_level",
            "belief_core",
            "confidence_core",
            "trust_in_institutions",
            "conversation_goal",
            "core_claim",
            "auxiliary_text",
            "style_id",
            "verbosity",
            "syntax",
            "emotionality",
            "directness",
            "evidence_style",
            "style_description",
        ],
    ),
    "user_turn_planner": make_prompt_template(
        name="user_turn_planner",
        required_variables=[
            "user_state_json",
            "style_profile_json",
            "recent_dialogue_json",
            "core_claim",
            "displayed_A1",
            "displayed_A2",
            "phase_name",
            "turn_number",
            "phase_turn_count",
            "dialogue_act",
            "conversation_goal",
            "belief_anchor_level",
            "evidence_role",
            "evidence_text",
            "active_auxiliaries_json",
        ],
    ),
    "user_turn_realizer": make_prompt_template(
        name="user_turn_realizer",
        required_variables=[
            "plan_json",
            "style_profile_json",
            "recent_dialogue_json",
        ],
    ),
}

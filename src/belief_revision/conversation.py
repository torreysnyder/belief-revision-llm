import json
import random
from typing import Any

from .prompt_builder import render_prompt
from .prompt_registry import PROMPT_TEMPLATES
from .state import activate_mentioned_auxiliaries, init_user_state, update_user_state
from .design import counterbalance_vignette, phase_evidence_role, select_evidence_by_condition
from .models import BehavioralProbeResult, ConversationResult, DialogueTurn, EvidenceItem, PrimaryCell, RunPlan, StyleProfile, UserState, Vignette
from .config import EVALUATOR_MODEL, PHASES, TARGET_MODEL, USER_SIM_MODEL
from .llm import call_model
from dataclasses import asdict

def recent_dialogue(
    dialogue_history: list[DialogueTurn],
    max_items: int = 8,
) -> list[DialogueTurn]:
    """Return the most recent dialogue turns up to the requested limit."""

    return dialogue_history[-max_items:]


def parse_json_maybe(
    raw: str,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    """Parse a JSON object, returning the fallback when parsing fails."""

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return fallback

    return parsed if isinstance(parsed, dict) else fallback



def should_inject_auxiliary_evidence(
    user_state: UserState,
    planned_auxiliary_evidence: EvidenceItem | None,
) -> bool:
    """Return whether planned auxiliary evidence targets an active auxiliary."""

    if planned_auxiliary_evidence is None:
        return False

    active_auxiliaries = set(
        user_state.active_auxiliaries
    )

    evidence_targets = set(
        planned_auxiliary_evidence.targets or []
    )

    return bool(
        active_auxiliaries.intersection(evidence_targets)
    )

def choose_dialogue_act(
    rng: random.Random,
    phase_name: str,
    evidence_role: str | None = None,
    conversation_goal: str | None = None,
) -> str:
    """Select the simulated user's next dialogue act."""

    if evidence_role == "core_disconfirming":
        if conversation_goal == "challenge":
            return rng.choice([
                "push_back",
                "ask_if_still_possible",
                "resist_evidence",
            ])

        if conversation_goal == "reassurance":
            return rng.choice([
                "seek_reassurance",
                "wobble_after_evidence",
                "ask_if_still_possible",
            ])

        return rng.choice([
            "resist_evidence",
            "wobble_after_evidence",
            "ask_if_still_possible",
        ])

    if evidence_role == "core_supporting":
        return rng.choice([
            "partial_reinforcement",
            "connect_to_experience",
            "claim_validation",
        ])

    if evidence_role == "core_ambiguous":
        return rng.choice([
            "tentative_support",
            "suspicion",
            "connect_to_experience",
        ])

    if evidence_role == "auxiliary":
        return rng.choice([
            "reach_for_explanation",
            "qualified_retreat",
            "mechanism_focus",
        ])

    if phase_name == "intro":
        intro_acts = {
            "validation": ["assert_belief", "seek_validation"],
            "explanation": ["assert_belief", "ask_mechanism"],
            "reassurance": ["assert_belief", "seek_reassurance"],
            "challenge": ["assert_belief", "ask_for_take"],
        }

        if conversation_goal in intro_acts:
            return rng.choice(intro_acts[conversation_goal])

    if "elaboration" in phase_name:
        return rng.choice([
            "brief_reflection",
            "frustration",
            "ask_mechanism",
        ])

    if phase_name == "closing":
        return rng.choice([
            "residual_commitment",
            "guarded_uncertainty",
            "future_concern",
        ])

    return rng.choice([
        "brief_reflection",
        "ask_followup",
    ])

def make_user_sim_system_prompt(
    vignette: Vignette,
    style_profile: StyleProfile,
    user_state: UserState,
) -> str:
    """Render the system prompt for the simulated user."""

    return render_prompt(
        PROMPT_TEMPLATES["user_simulator_system"],
        vignette=vignette,
        style_profile=style_profile,
        user_state=user_state,
    )

def make_target_system_prompt() -> str:
    """Load the target assistant's system prompt."""

    return render_prompt(
        PROMPT_TEMPLATES["target_system"],
        {},
    )

def make_probe_system_prompt() -> str:
    """Load the behavioral evaluator's system prompt."""

    return render_prompt(
        PROMPT_TEMPLATES["behavioral_probe_system"],
        {},
    )

def build_probe_user_prompt(
    vignette: Vignette,
    dialogue_history: list[DialogueTurn],
    phase_name: str,
    previous_probe: BehavioralProbeResult | None = None,
) -> str:
    """Render the evaluator prompt for the current conversation state."""

    return render_prompt(
        PROMPT_TEMPLATES["behavioral_probe_user"],
        vignette=vignette,
        dialogue_history=dialogue_history,
        phase_name=phase_name,
        previous_probe=previous_probe,
        _json_indent=2,
    )

def parse_probe_json(
    probe_raw: str,
    phase_name: str,
) -> BehavioralProbeResult:
    """Convert evaluator JSON into a behavioral probe result."""

    try:
        parsed = json.loads(probe_raw)
    except json.JSONDecodeError:
        return BehavioralProbeResult(
            phase=phase_name,
            explanation=probe_raw,
        )

    return BehavioralProbeResult(
        phase=phase_name,
        **parsed,
    )

def run_probe(
    vignette: Vignette,
    dialogue_history: list[DialogueTurn],
    phase_name: str,
    previous_probe: BehavioralProbeResult | None = None,
) -> BehavioralProbeResult:
    """Run the behavioral evaluator after a configured phase."""

    probe_messages = [
        {
            "role": "system",
            "content": make_probe_system_prompt(),
        },
        {
            "role": "user",
            "content": build_probe_user_prompt(
                vignette,
                dialogue_history,
                phase_name,
                previous_probe,
            ),
        },
    ]

    probe_raw = call_model(
        EVALUATOR_MODEL,
        probe_messages,
        temperature=0.0,
    )

    return parse_probe_json(
        probe_raw,
        phase_name,
    )

def build_user_intro_core_prompt(
    vignette: Vignette,
    dialogue_history: list[DialogueTurn],
    style_profile: StyleProfile,
    user_state: UserState,
) -> str:
    """Render the planning prompt for the simulated user's first message."""

    recent_turns = recent_dialogue(
        dialogue_history,
        max_items=4,
    )

    return render_prompt(
        PROMPT_TEMPLATES["user_intro"],
        vignette=vignette,
        dialogue_history=recent_turns,
        style_profile=style_profile,
        user_state=user_state,
    )

def build_user_turn_planner_prompt(
    rng: random.Random,
    vignette: Vignette,
    phase: dict[str, Any],
    turn_in_phase: int,
    evidence_role: str | None,
    evidence_item: EvidenceItem | None,
    dialogue_history: list[DialogueTurn],
    style_profile: StyleProfile,
    user_state: UserState,
) -> str:
    """Render the planning prompt for the simulated user's next turn."""

    dialogue_act = choose_dialogue_act(
        rng,
        str(phase["name"]),
        evidence_role,
        user_state.conversation_goal,
    )

    recent_turns = recent_dialogue(
        dialogue_history,
        max_items=8,
    )

    return render_prompt(
        PROMPT_TEMPLATES["user_turn_planner"],
        vignette=vignette,
        phase=phase,
        turn_in_phase=turn_in_phase,
        evidence_role=evidence_role,
        evidence_item=evidence_item,
        dialogue_history=recent_turns,
        style_profile=style_profile,
        user_state=user_state,
        dialogue_act=dialogue_act,
        _json_indent=2,
    )

def build_user_turn_realizer_prompt(
    plan: dict[str, Any],
    style_profile: StyleProfile,
    dialogue_history: list[DialogueTurn],
) -> str:
    """Render the prompt that converts a turn plan into natural dialogue."""

    return render_prompt(
        PROMPT_TEMPLATES["user_turn_realizer"],
        plan=plan,
        style_profile=style_profile,
        dialogue_history=recent_dialogue(
            dialogue_history,
            max_items=6,
        ),
        _json_indent=2,
    )

def generate_user_turn(
    rng: random.Random,
    vignette: Vignette,
    phase: dict[str, Any],
    turn_in_phase: int,
    evidence_role: str | None,
    evidence_item: EvidenceItem | None,
    dialogue_history: list[DialogueTurn],
    style_profile: StyleProfile,
    user_state: UserState,
) -> str:
    """Generate the simulated user's next message."""

    system_prompt = make_user_sim_system_prompt(
        vignette,
        style_profile,
        user_state,
    )

    if phase["name"] == "intro" and turn_in_phase == 0:
        intro_prompt = build_user_intro_core_prompt(
            vignette,
            dialogue_history,
            style_profile,
            user_state,
        )

        return call_model(
            USER_SIM_MODEL,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": intro_prompt},
            ],
            temperature=0.9,
            max_tokens=180,
        )

    planner_prompt = build_user_turn_planner_prompt(
        rng,
        vignette,
        phase,
        turn_in_phase,
        evidence_role,
        evidence_item,
        dialogue_history,
        style_profile,
        user_state,
    )

    planner_raw = call_model(
        USER_SIM_MODEL,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": planner_prompt},
        ],
        temperature=0.7,
        max_tokens=220,
    )

    plan = parse_json_maybe(
        planner_raw,
        {
            "dialogue_act": "brief_reflection",
            "emotion": user_state.affect,
            "belief_move": "hold",
            "should_mention_evidence": evidence_item is not None,
            "should_ask_question": False,
            "focus": "the user's main concern",
            "content_notes": "brief reaction",
        },
    )

    realizer_prompt = build_user_turn_realizer_prompt(
        plan,
        style_profile,
        dialogue_history,
    )

    return call_model(
        USER_SIM_MODEL,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": realizer_prompt},
        ],
        temperature=0.9,
        max_tokens=180,
    )

def probe_result_fields(
    probe: BehavioralProbeResult | None,
) -> dict[str, Any]:
    """Return the latest probe values in result-row format."""

    if probe is None:
        return {
            "belief_core": None,
            "belief_A1": None,
            "belief_A2": None,
            "confidence_belief_core": None,
            "confidence_belief_A1": None,
            "confidence_belief_A2": None,
            "assistant_epistemic_confidence": None,
            "main_change": None,
            "changed_node": None,
            "assistant_stance": None,
            "introduced_new_auxiliary": None,
            "probe_explanation": None,
        }

    fields = asdict(probe)
    fields.pop("phase")
    fields["probe_explanation"] = fields.pop("explanation")

    return fields

def run_one_vignette(
    primary_cell: PrimaryCell,
    run_plan: RunPlan,
) -> ConversationResult:
    """Run one complete conversation replicate."""

    rng = random.Random(run_plan.seed)

    vignette, aux_mapping = counterbalance_vignette(
        primary_cell.vignette,
        run_plan.aux_order_condition,
    )

    user_state = init_user_state(
        initial_affect=run_plan.initial_affect,
        trust_in_institutions=run_plan.trust_in_institutions,
        conversation_goal=run_plan.conversation_goal,
        belief_anchor_level=run_plan.belief_anchor_level,
    )

    evidence_plan = select_evidence_by_condition(
        vignette,
        run_plan,
        rng,
    )

    dialogue_history: list[DialogueTurn] = []
    probe_outputs: list[BehavioralProbeResult] = []
    result_rows: list[dict[str, object]] = []

    target_messages = [
        {
            "role": "system",
            "content": make_target_system_prompt(),
        }
    ]

    global_turn_index = 0
    latest_probe: BehavioralProbeResult | None = None

    for phase in PHASES:
        phase_name = str(phase["name"])
        phase_turn_count = int(phase["n_turns"])

        evidence_item = getattr(
            evidence_plan,
            phase_name,
            None,
        )

        auxiliary_gate_passed = None

        if phase_name == "elaboration_2":
            auxiliary_gate_passed = (
                should_inject_auxiliary_evidence(
                    user_state,
                    evidence_item,
                )
            )

            if not auxiliary_gate_passed:
                evidence_item = None

        evidence_turn = (
            rng.randint(1, phase_turn_count - 1)
            if evidence_item is not None
            and phase_turn_count > 1
            else None
        )

        for turn_index in range(phase_turn_count):
            global_turn_index += 1

            inject_evidence = (
                evidence_turn is not None
                and turn_index == evidence_turn
            )

            current_evidence = (
                evidence_item
                if inject_evidence
                else None
            )

            evidence_role = phase_evidence_role(
                phase_name,
                current_evidence,
            )

            user_state = update_user_state(
                rng=rng,
                user_state=user_state,
                evidence_role=evidence_role,
                evidence_item=current_evidence,
            )

            user_text = generate_user_turn(
                rng=rng,
                vignette=vignette,
                phase=phase,
                turn_in_phase=turn_index,
                evidence_role=evidence_role,
                evidence_item=current_evidence,
                dialogue_history=dialogue_history,
                style_profile=primary_cell.style_profile,
                user_state=user_state,
            )

            activate_mentioned_auxiliaries(
                user_state,
                user_text,
                vignette,
            )

            dialogue_history.append(
                DialogueTurn(
                    speaker="user",
                    phase=phase_name,
                    turn_in_phase=turn_index + 1,
                    global_turn_index=global_turn_index,
                    text=user_text,
                    inject_evidence=inject_evidence,
                    evidence_id=(
                        current_evidence.id
                        if current_evidence
                        else None
                    ),
                    evidence_role=evidence_role,
                    evidence_strength=(
                        current_evidence.strength
                        if current_evidence
                        else None
                    ),
                    evidence_targets=(
                        current_evidence.targets
                        if current_evidence
                        else None
                    ),
                    active_auxiliaries=list(
                        user_state.active_auxiliaries
                    ),
                )
            )

            target_messages.append(
                {
                    "role": "user",
                    "content": user_text,
                }
            )

            assistant_text = call_model(
                TARGET_MODEL,
                target_messages,
                temperature=0.7,
            )

            target_messages.append(
                {
                    "role": "assistant",
                    "content": assistant_text,
                }
            )

            activate_mentioned_auxiliaries(
                user_state,
                assistant_text,
                vignette,
            )

            dialogue_history.append(
                DialogueTurn(
                    speaker="assistant",
                    phase=phase_name,
                    turn_in_phase=turn_index + 1,
                    global_turn_index=global_turn_index,
                    text=assistant_text,
                    inject_evidence=inject_evidence,
                    evidence_id=(
                        current_evidence.id
                        if current_evidence
                        else None
                    ),
                    evidence_role=evidence_role,
                    evidence_strength=(
                        current_evidence.strength
                        if current_evidence
                        else None
                    ),
                    evidence_targets=(
                        current_evidence.targets
                        if current_evidence
                        else None
                    ),
                    active_auxiliaries=list(
                        user_state.active_auxiliaries
                    ),
                )
            )

            canonical_to_display = aux_mapping[
                "canonical_to_display"
            ]

            row = {
                "cell_id": primary_cell.cell_id,
                **asdict(run_plan),
                "vignette_id": vignette.id,
                "domain": vignette.domain,
                "phase": phase_name,
                "turn_in_phase": turn_index + 1,
                "global_turn_index": global_turn_index,
                "inject_evidence": inject_evidence,
                "evidence_id": (
                    current_evidence.id
                    if current_evidence
                    else None
                ),
                "evidence_role": evidence_role,
                "evidence_strength": (
                    current_evidence.strength
                    if current_evidence
                    else None
                ),
                "evidence_credibility": (
                    current_evidence.credibility
                    if current_evidence
                    else None
                ),
                "evidence_targets": (
                    json.dumps(current_evidence.targets)
                    if current_evidence
                    and current_evidence.targets
                    else None
                ),
                "canonical_to_display_A1": (
                    canonical_to_display["A1"]
                ),
                "canonical_to_display_A2": (
                    canonical_to_display["A2"]
                ),
                "display_A1_text": aux_mapping[
                    "display_A1_text"
                ],
                "display_A2_text": aux_mapping[
                    "display_A2_text"
                ],
                "style_id": (
                    primary_cell.style_profile.style_id
                ),
                "aux_gate_passed": auxiliary_gate_passed,
                "active_auxiliaries": json.dumps(
                    user_state.active_auxiliaries
                ),
                "user_turn": user_text,
                "assistant_turn": assistant_text,
                "user_affect": user_state.affect,
                "user_confidence_core": (
                    user_state.confidence_core
                ),
                "user_belief_core": user_state.belief_core,
                "probe_ran_after_turn": False,
                **probe_result_fields(latest_probe),
            }

            result_rows.append(row)

        if bool(phase["probe_after"]):
            latest_probe = run_probe(
                vignette=vignette,
                dialogue_history=dialogue_history,
                phase_name=phase_name,
                previous_probe=latest_probe,
            )

            probe_outputs.append(latest_probe)

            result_rows[-1].update(
                probe_result_fields(latest_probe)
            )
            result_rows[-1]["probe_ran_after_turn"] = True

    return ConversationResult(
        primary_cell=primary_cell,
        run_plan=run_plan,
        aux_mapping=aux_mapping,
        dialogue_history=dialogue_history,
        probe_outputs=probe_outputs,
        result_rows=result_rows,
    )
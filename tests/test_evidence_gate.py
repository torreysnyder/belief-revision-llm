import random

import pytest

from belief_revision import conversation
from belief_revision.models import BehavioralProbeResult, DialogueTurn
from belief_revision.state import init_user_state


class FirstChoice:
    def choice(self, values):
        return values[0]


def test_recent_json_and_gate(vignette):
    turns = [DialogueTurn("user", "intro", 1, i, str(i)) for i in range(10)]
    assert len(conversation.recent_dialogue(turns, 3)) == 3
    fallback = {"fallback": True}
    assert conversation.parse_json_maybe('{"ok": true}', fallback) == {"ok": True}
    assert conversation.parse_json_maybe("[]", fallback) is fallback
    assert conversation.parse_json_maybe("bad", fallback) is fallback
    assert conversation.parse_json_maybe(None, fallback) is fallback

    state = init_user_state("curious", "low", "challenge", "low")
    assert not conversation.should_inject_auxiliary_evidence(state, None)
    assert not conversation.should_inject_auxiliary_evidence(
        state, vignette.auxiliary_evidence[0]
    )
    state.active_auxiliaries.append("A1")
    assert conversation.should_inject_auxiliary_evidence(
        state, vignette.auxiliary_evidence[0]
    )


@pytest.mark.parametrize(
    ("phase", "role", "goal", "expected"),
    [
        ("x", "core_disconfirming", "challenge", "push_back"),
        ("x", "core_disconfirming", "reassurance", "seek_reassurance"),
        ("x", "core_disconfirming", "other", "resist_evidence"),
        ("x", "core_supporting", None, "partial_reinforcement"),
        ("x", "core_ambiguous", None, "tentative_support"),
        ("x", "auxiliary", None, "reach_for_explanation"),
        ("intro", None, "validation", "assert_belief"),
        ("intro", None, "explanation", "assert_belief"),
        ("intro", None, "reassurance", "assert_belief"),
        ("intro", None, "challenge", "assert_belief"),
        ("intro", None, "unknown", "brief_reflection"),
        ("elaboration_1", None, None, "brief_reflection"),
        ("closing", None, None, "residual_commitment"),
        ("other", None, None, "brief_reflection"),
    ],
)
def test_choose_dialogue_act(phase, role, goal, expected):
    assert (
        conversation.choose_dialogue_act(FirstChoice(), phase, role, goal) == expected
    )


def test_prompt_wrappers_and_probe(monkeypatch, vignette, style):
    state = init_user_state("curious", "low", "challenge", "low")
    calls = []

    def fake_render(template, variables=None, **context):
        calls.append((template.name, variables, context))
        return template.name

    monkeypatch.setattr(conversation, "render_prompt", fake_render)
    assert (
        conversation.make_user_sim_system_prompt(vignette, style, state)
        == "user_simulator_system"
    )
    assert conversation.make_target_system_prompt() == "target_system"
    assert conversation.make_probe_system_prompt() == "behavioral_probe_system"
    assert (
        conversation.build_probe_user_prompt(vignette, [], "intro")
        == "behavioral_probe_user"
    )
    assert (
        conversation.build_user_intro_core_prompt(vignette, [], style, state)
        == "user_intro"
    )
    assert (
        conversation.build_user_turn_planner_prompt(
            random.Random(1),
            vignette,
            {"name": "intro", "n_turns": 1},
            0,
            None,
            None,
            [],
            style,
            state,
        )
        == "user_turn_planner"
    )
    assert (
        conversation.build_user_turn_realizer_prompt({}, style, [])
        == "user_turn_realizer"
    )
    assert calls

    assert conversation.parse_probe_json("not json", "intro").explanation == "not json"
    parsed = conversation.parse_probe_json('{"belief_core": 42}', "intro")
    assert parsed.belief_core == 42

    monkeypatch.setattr(
        conversation, "call_model", lambda *args, **kwargs: '{"belief_core": 60}'
    )
    assert conversation.run_probe(vignette, [], "intro").belief_core == 60


def test_generate_user_turn_paths(monkeypatch, vignette, style):
    state = init_user_state("curious", "low", "challenge", "low")
    monkeypatch.setattr(
        conversation, "make_user_sim_system_prompt", lambda *args: "system"
    )
    monkeypatch.setattr(
        conversation, "build_user_intro_core_prompt", lambda *args: "intro"
    )
    monkeypatch.setattr(
        conversation, "build_user_turn_planner_prompt", lambda *args: "planner"
    )
    monkeypatch.setattr(
        conversation, "build_user_turn_realizer_prompt", lambda *args: "realizer"
    )

    responses = iter(["intro result", "bad json", "realized"])
    monkeypatch.setattr(
        conversation, "call_model", lambda *args, **kwargs: next(responses)
    )
    intro = conversation.generate_user_turn(
        random.Random(1), vignette, {"name": "intro"}, 0, None, None, [], style, state
    )
    assert intro == "intro result"
    later = conversation.generate_user_turn(
        random.Random(1), vignette, {"name": "closing"}, 0, None, None, [], style, state
    )
    assert later == "realized"


def test_probe_fields():
    empty = conversation.probe_result_fields(None)
    assert empty["belief_core"] is None
    fields = conversation.probe_result_fields(
        BehavioralProbeResult("intro", belief_core=40, explanation="why")
    )
    assert fields["belief_core"] == 40
    assert fields["probe_explanation"] == "why"


def test_complete_conversation_without_remote_calls(
    monkeypatch, primary_cell, run_plan
):
    monkeypatch.setattr(conversation, "make_target_system_prompt", lambda: "target")
    monkeypatch.setattr(
        conversation,
        "generate_user_turn",
        lambda **kwargs: "Clinical studies support the treatment",
    )
    monkeypatch.setattr(conversation, "call_model", lambda *args, **kwargs: "assistant")
    monkeypatch.setattr(
        conversation,
        "run_probe",
        lambda **kwargs: BehavioralProbeResult(kwargs["phase_name"], belief_core=50),
    )
    result = conversation.run_one_vignette(primary_cell, run_plan)
    assert len(result.result_rows) == 15
    assert len(result.dialogue_history) == 30
    assert len(result.probe_outputs) == 4
    assert any(row["probe_ran_after_turn"] for row in result.result_rows)
    assert any(row["aux_gate_passed"] is True for row in result.result_rows)


def test_complete_conversation_gate_fails(monkeypatch, primary_cell, run_plan):
    monkeypatch.setattr(
        conversation,
        "PHASES",
        [{"name": "elaboration_2", "n_turns": 1, "probe_after": False}],
    )
    monkeypatch.setattr(conversation, "make_target_system_prompt", lambda: "target")
    monkeypatch.setattr(
        conversation, "generate_user_turn", lambda **kwargs: "no auxiliary terms"
    )
    monkeypatch.setattr(conversation, "call_model", lambda *args, **kwargs: "assistant")
    result = conversation.run_one_vignette(primary_cell, run_plan)
    assert result.result_rows[0]["aux_gate_passed"] is False

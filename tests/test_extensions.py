import pytest
from belief_revision.core.models import (
    ActivationCapture,
    ActivationCaptureRequest,
    Auxiliary,
    ConversationResult,
    DialogueTurn,
    ModelGeneration,
    PrimaryCell,
    RunPlan,
    StyleProfile,
    UserState,
    Vignette,
)
from belief_revision.extensions import (
    ActivationCaptureContext,
    ActivationCapturePlan,
    CompletedConversationContext,
    CompletedPhaseContext,
    ConversationContext,
    ConversationExtension,
    GenerationContext,
    PhaseContext,
    TurnContext,
    collect_activation_capture_plan,
    dispatch_extensions,
    route_activation_captures,
)
from dataclasses import FrozenInstanceError


def make_contexts():
    vignette = Vignette(
        id="vignette",
        domain="test",
        core_claim="claim",
        auxiliaries=[Auxiliary(id="A1", type="cause", text="auxiliary")],
        core_evidence_disconfirming=[],
        core_evidence_supporting=[],
        core_evidence_ambiguous=[],
        auxiliary_evidence=[],
    )
    style = StyleProfile(
        style_id="style",
        verbosity="short",
        syntax="simple",
        emotionality="neutral",
        directness="direct",
        evidence_style="plain",
        description="test style",
    )
    primary_cell = PrimaryCell(
        vignette=vignette,
        style_profile=style,
        trust_in_institutions="medium",
        conversation_goal="explanation",
        belief_anchor_level="medium",
        cell_id="cell",
    )
    run_plan = RunPlan(
        replicate_index=0,
        seed=1,
        aux_order_condition="standard",
        initial_affect="neutral",
        trust_in_institutions="medium",
        conversation_goal="explanation",
        belief_anchor_level="medium",
        core_order_condition="standard",
        elaboration_1_condition="none",
        elaboration_2_target_condition="none",
    )
    conversation = ConversationContext(
        primary_cell=primary_cell,
        run_plan=run_plan,
        vignette=vignette,
        aux_mapping={},
    )
    phase = PhaseContext(
        conversation=conversation,
        phase_name="intro",
        phase={"name": "intro", "n_turns": 1},
        evidence_item=None,
        auxiliary_gate_passed=None,
    )
    user_state = UserState(
        belief_core=50,
        belief_A1=50,
        belief_A2=50,
        confidence_core=50,
        affect="neutral",
        initial_affect="neutral",
        trust_in_institutions="medium",
        trust_in_personal_experience="medium",
        conversation_goal="explanation",
        belief_anchor_level="medium",
    )
    before_turn = TurnContext(
        phase=phase,
        turn_in_phase=1,
        global_turn_index=1,
        inject_evidence=False,
        current_evidence=None,
        evidence_role=None,
        user_state=user_state,
        user_text=None,
        dialogue_history=(),
    )
    after_user_turn = TurnContext(
        phase=phase,
        turn_in_phase=1,
        global_turn_index=1,
        inject_evidence=False,
        current_evidence=None,
        evidence_role=None,
        user_state=user_state,
        user_text="Hello",
        dialogue_history=(),
    )
    before_generation = GenerationContext(
        turn=after_user_turn,
        model="model",
        messages=({"role": "user", "content": "Hello"},),
        temperature=0.7,
        max_tokens=None,
    )
    after_generation = GenerationContext(
        turn=after_user_turn,
        model="model",
        messages=before_generation.messages,
        temperature=0.7,
        max_tokens=None,
        response="Hi",
    )
    request = ActivationCaptureRequest(request_id="capture", layers=(0,))
    activation = ActivationCapture(
        request_id="capture",
        layer=0,
        model_name="model",
        location="residual_post",
        token_position="final_prompt_token",
        token_index=0,
        sequence_length=1,
        shape=(1, 4),
        dtype="float32",
        tensor="tensor",
    )
    activation_capture = ActivationCaptureContext(
        generation=after_generation,
        request=request,
        activations=(activation,),
    )
    dialogue_turn = DialogueTurn(
        speaker="user",
        phase="intro",
        turn_in_phase=1,
        global_turn_index=1,
        text="Hello",
    )
    result = ConversationResult(
        primary_cell=primary_cell,
        run_plan=run_plan,
        dialogue_history=[dialogue_turn],
        probe_outputs=[],
        result_rows=[],
        aux_mapping={},
    )
    completed = CompletedConversationContext(
        conversation=conversation,
        result=result,
    )
    completed_phase = CompletedPhaseContext(
        phase=phase,
        user_state=user_state,
        dialogue_history=(dialogue_turn,),
        latest_probe=None,
    )
    return {
        "before_conversation": conversation,
        "before_phase": phase,
        "before_turn": before_turn,
        "after_user_turn": after_user_turn,
        "before_assistant_generation": before_generation,
        "after_assistant_generation": after_generation,
        "after_activation_capture": activation_capture,
        "after_turn": after_user_turn,
        "after_phase": completed_phase,
        "after_conversation": completed,
    }


@pytest.mark.parametrize("hook_name,context", make_contexts().items())
def test_base_extension_hooks_are_no_ops(hook_name, context):
    dispatch_extensions((ConversationExtension(),), hook_name, context)


def test_extensions_run_in_registration_order_with_same_context():
    calls = []
    context = make_contexts()["before_turn"]

    class Recorder(ConversationExtension):
        def __init__(self, name):
            self.name = name

        def before_turn(self, received_context):
            calls.append((self.name, received_context))

    dispatch_extensions(
        (Recorder("first"), Recorder("second"), Recorder("third")),
        "before_turn",
        context,
    )

    assert [name for name, _ in calls] == ["first", "second", "third"]
    assert all(received is context for _, received in calls)


def test_empty_extension_collection_is_a_no_op():
    dispatch_extensions((), "before_phase", make_contexts()["before_phase"])


def test_base_extension_requests_no_activation_captures():
    context = make_contexts()["before_assistant_generation"]

    assert ConversationExtension().activation_capture_requests(context) == ()


def test_activation_capture_context_keeps_generation_request_and_results():
    context = make_contexts()["after_activation_capture"]

    assert context.generation.response == "Hi"
    assert context.request.request_id == "capture"
    assert context.activations[0].request_id == context.request.request_id


def test_after_activation_capture_dispatches_in_registration_order():
    calls = []
    context = make_contexts()["after_activation_capture"]

    class Recorder(ConversationExtension):
        def __init__(self, name):
            self.name = name

        def after_activation_capture(self, received_context):
            calls.append((self.name, received_context))

    dispatch_extensions(
        (Recorder("first"), Recorder("second")),
        "after_activation_capture",
        context,
    )

    assert [name for name, _ in calls] == ["first", "second"]
    assert all(received is context for _, received in calls)


def make_capture(request_id, layer):
    return ActivationCapture(
        request_id=request_id,
        layer=layer,
        model_name="model",
        location="residual_post",
        token_position="final_prompt_token",
        token_index=0,
        sequence_length=1,
        shape=(1, 4),
        dtype="float32",
        tensor="tensor",
    )


class RequestingExtension(ConversationExtension):
    def __init__(self, *requests):
        self.requests = requests
        self.capture_contexts = []

    def activation_capture_requests(self, context):
        return self.requests

    def after_activation_capture(self, context):
        self.capture_contexts.append(context)


def test_collect_capture_plan_assigns_backend_ids_per_extension():
    context = make_contexts()["before_assistant_generation"]
    first = RequestingExtension(ActivationCaptureRequest("shared", layers=(0,)))
    second = RequestingExtension(ActivationCaptureRequest("shared", layers=(1,)))

    plan = collect_activation_capture_plan((first, second), context)

    assert [request.request_id for request in plan.backend_requests] == [
        "0:0",
        "1:0",
    ]
    assert [request.layers for request in plan.backend_requests] == [(0,), (1,)]


def test_empty_capture_plan_has_no_backend_requests():
    context = make_contexts()["before_assistant_generation"]

    plan = collect_activation_capture_plan((ConversationExtension(),), context)

    assert plan == ActivationCapturePlan()
    assert plan.backend_requests == ()


def test_duplicate_public_request_ids_within_one_extension_fail():
    extension = RequestingExtension(
        ActivationCaptureRequest("duplicate", layers=(0,)),
        ActivationCaptureRequest("duplicate", layers=(1,)),
    )

    with pytest.raises(ValueError, match="must be unique.*'duplicate'"):
        collect_activation_capture_plan(
            (extension,),
            make_contexts()["before_assistant_generation"],
        )


def test_route_captures_restores_public_ids_and_dispatches_to_owner():
    before_context = make_contexts()["before_assistant_generation"]
    after_context = make_contexts()["after_assistant_generation"]
    first = RequestingExtension(ActivationCaptureRequest("shared", layers=(0,)))
    second = RequestingExtension(ActivationCaptureRequest("shared", layers=(1,)))
    plan = collect_activation_capture_plan((first, second), before_context)
    generation = ModelGeneration(
        "Hi",
        (make_capture("0:0", 0), make_capture("1:0", 1)),
    )

    route_activation_captures(plan, after_context, generation)

    assert first.capture_contexts[0].generation is after_context
    assert first.capture_contexts[0].request.request_id == "shared"
    assert first.capture_contexts[0].activations[0].request_id == "shared"
    assert first.capture_contexts[0].activations[0].layer == 0
    assert second.capture_contexts[0].activations[0].layer == 1


@pytest.mark.parametrize(
    ("activations", "message"),
    [
        ((), "no activations"),
        ((make_capture("unexpected", 0),), "unexpected"),
        ((make_capture("0:0", 1),), r"expected \[0\], got \[1\]"),
    ],
)
def test_route_captures_rejects_invalid_backend_results(activations, message):
    context = make_contexts()["before_assistant_generation"]
    extension = RequestingExtension(ActivationCaptureRequest("capture", layers=(0,)))
    plan = collect_activation_capture_plan((extension,), context)

    with pytest.raises(ValueError, match=message):
        route_activation_captures(
            plan,
            make_contexts()["after_assistant_generation"],
            ModelGeneration("Hi", activations),
        )


def test_route_captures_accepts_backend_selected_layers():
    before_context = make_contexts()["before_assistant_generation"]
    after_context = make_contexts()["after_assistant_generation"]
    extension = RequestingExtension(ActivationCaptureRequest("all"))
    plan = collect_activation_capture_plan((extension,), before_context)

    route_activation_captures(
        plan,
        after_context,
        ModelGeneration("Hi", (make_capture("0:0", 3),)),
    )

    assert extension.capture_contexts[0].activations[0].layer == 3


def test_extension_exception_propagates_and_stops_dispatch():
    calls = []

    class First(ConversationExtension):
        def before_phase(self, context):
            calls.append("first")

    class Broken(ConversationExtension):
        def before_phase(self, context):
            calls.append("broken")
            raise RuntimeError("extension failed")

    class NeverReached(ConversationExtension):
        def before_phase(self, context):
            calls.append("never")

    with pytest.raises(RuntimeError, match="extension failed"):
        dispatch_extensions(
            (First(), Broken(), NeverReached()),
            "before_phase",
            make_contexts()["before_phase"],
        )

    assert calls == ["first", "broken"]


@pytest.mark.parametrize("context", make_contexts().values())
def test_context_wrappers_are_frozen(context):
    field_name = next(iter(context.__dataclass_fields__))

    with pytest.raises(FrozenInstanceError):
        setattr(context, field_name, "changed")

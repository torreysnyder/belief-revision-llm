import pytest
from belief_revision.core import conversation
from belief_revision.core.models import (
    ActivationCapture,
    ActivationCaptureRequest,
    BehavioralProbeResult,
    ModelGeneration,
)
from belief_revision.extensions import ConversationExtension


class RecordingExtension(ConversationExtension):
    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.contexts = {}

    def _record(self, hook_name, context):
        self.events.append((self.name, hook_name))
        self.contexts[hook_name] = context

    def before_conversation(self, context):
        self._record("before_conversation", context)

    def before_phase(self, context):
        self._record("before_phase", context)

    def before_turn(self, context):
        self._record("before_turn", context)

    def after_user_turn(self, context):
        self._record("after_user_turn", context)

    def before_assistant_generation(self, context):
        self._record("before_assistant_generation", context)

    def after_assistant_generation(self, context):
        self._record("after_assistant_generation", context)

    def after_turn(self, context):
        self._record("after_turn", context)

    def after_phase(self, context):
        self._record("after_phase", context)

    def after_conversation(self, context):
        self._record("after_conversation", context)


def configure_single_turn_run(monkeypatch):
    monkeypatch.setattr(
        conversation,
        "PHASES",
        [{"name": "intro", "n_turns": 1, "probe_after": True}],
    )
    monkeypatch.setattr(conversation, "make_target_system_prompt", lambda: "target")
    monkeypatch.setattr(
        conversation,
        "generate_user_turn",
        lambda **kwargs: "user response",
    )
    monkeypatch.setattr(
        conversation,
        "call_model",
        lambda *args, **kwargs: ModelGeneration("assistant"),
    )
    monkeypatch.setattr(
        conversation,
        "run_probe",
        lambda **kwargs: BehavioralProbeResult(kwargs["phase_name"], belief_core=50),
    )


def test_extensions_run_in_lifecycle_and_registration_order(
    monkeypatch,
    primary_cell,
    run_plan,
):
    configure_single_turn_run(monkeypatch)
    events = []
    first = RecordingExtension("first", events)
    second = RecordingExtension("second", events)

    result = conversation.run_one_vignette(
        primary_cell,
        run_plan,
        extensions=(first, second),
    )

    hooks = [
        "before_conversation",
        "before_phase",
        "before_turn",
        "after_user_turn",
        "before_assistant_generation",
        "after_assistant_generation",
        "after_turn",
        "after_phase",
        "after_conversation",
    ]
    assert events == [(extension_name, hook_name) for hook_name in hooks for extension_name in ("first", "second")]
    assert first.contexts["before_turn"].user_text is None
    assert first.contexts["before_turn"].turn_in_phase == 1
    assert first.contexts["before_turn"].dialogue_history == ()
    assert first.contexts["after_user_turn"].user_text == "user response"
    assert first.contexts["after_user_turn"].dialogue_history[-1].speaker == "user"
    assert first.contexts["after_turn"].dialogue_history[-1].speaker == "assistant"
    assert first.contexts["before_assistant_generation"].response is None
    assert first.contexts["before_assistant_generation"].temperature == 0.7
    assert first.contexts["before_assistant_generation"].max_tokens == 500
    assert first.contexts["after_assistant_generation"].response == "assistant"
    assert first.contexts["after_phase"].phase.phase_name == "intro"
    assert len(first.contexts["after_phase"].dialogue_history) == 2
    assert first.contexts["after_phase"].latest_probe.belief_core == 50
    assert first.contexts["after_conversation"].result == result
    assert result.probe_outputs[0].belief_core == 50


class FailingExtension(ConversationExtension):
    def before_assistant_generation(self, context):
        raise RuntimeError("extension failed")


def test_extension_errors_propagate(monkeypatch, primary_cell, run_plan):
    configure_single_turn_run(monkeypatch)

    with pytest.raises(RuntimeError, match="extension failed"):
        conversation.run_one_vignette(
            primary_cell,
            run_plan,
            extensions=(FailingExtension(),),
        )


class MutatingExtension(ConversationExtension):
    def before_conversation(self, context):
        context.vignette.core_claim = "mutated claim"

    def before_turn(self, context):
        context.user_state.belief_core = False

    def after_user_turn(self, context):
        context.dialogue_history[-1].text = "mutated user turn"

    def before_assistant_generation(self, context):
        context.messages[-1]["content"] = "mutated model input"

    def after_conversation(self, context):
        context.result.result_rows.clear()


def test_context_mutations_do_not_change_experiment(
    monkeypatch,
    primary_cell,
    run_plan,
):
    configure_single_turn_run(monkeypatch)

    result = conversation.run_one_vignette(
        primary_cell,
        run_plan,
        extensions=(MutatingExtension(),),
    )

    assert result.primary_cell.vignette.core_claim == "The treatment works."
    assert result.dialogue_history[0].text == "user response"
    assert result.result_rows[0]["user_turn"] == "user response"


class RetainingExtension(ConversationExtension):
    def __init__(self):
        self.first_turn = None

    def before_turn(self, context):
        if self.first_turn is None:
            self.first_turn = context


def test_retained_context_is_not_changed_by_later_live_state(
    monkeypatch,
    primary_cell,
    run_plan,
):
    configure_single_turn_run(monkeypatch)
    monkeypatch.setattr(
        conversation,
        "PHASES",
        [{"name": "intro", "n_turns": 2, "probe_after": False}],
    )
    monkeypatch.setattr(
        conversation,
        "generate_user_turn",
        lambda **kwargs: "Clinical studies support the treatment",
    )
    extension = RetainingExtension()

    result = conversation.run_one_vignette(
        primary_cell,
        run_plan,
        extensions=(extension,),
    )

    assert result.dialogue_history[-1].active_auxiliaries == ["A1"]
    assert extension.first_turn.user_state.active_auxiliaries == []
    assert extension.first_turn.dialogue_history == ()


def make_capture(request_id, layer):
    return ActivationCapture(
        request_id=request_id,
        layer=layer,
        model_name="model",
        location="residual_post",
        token_position="final_prompt_token",
        token_index=3,
        sequence_length=4,
        shape=(2,),
        dtype="torch.float32",
        tensor=object(),
    )


class CapturingExtension(ConversationExtension):
    def __init__(self, name, events, *, selected_turn=1, request_id="shared"):
        self.name = name
        self.events = events
        self.selected_turn = selected_turn
        self.request_id = request_id
        self.capture_contexts = []

    def before_assistant_generation(self, context):
        self.events.append((self.name, "before_assistant_generation"))

    def activation_capture_requests(self, context):
        self.events.append((self.name, "activation_capture_requests"))
        if context.turn.turn_in_phase != self.selected_turn:
            return ()
        return (ActivationCaptureRequest(self.request_id, layers=(1, 3)),)

    def after_activation_capture(self, context):
        self.events.append((self.name, "after_activation_capture"))
        self.capture_contexts.append(context)

    def after_assistant_generation(self, context):
        self.events.append((self.name, "after_assistant_generation"))


def test_no_capture_requests_preserve_legacy_call_signature(
    monkeypatch,
    primary_cell,
    run_plan,
):
    configure_single_turn_run(monkeypatch)
    calls = []

    def fake_call_model(*args, **kwargs):
        calls.append((args, kwargs))
        return ModelGeneration("assistant")

    monkeypatch.setattr(conversation, "call_model", fake_call_model)

    conversation.run_one_vignette(primary_cell, run_plan)

    assert len(calls) == 1
    assert calls[0][1] == {"temperature": 0.7}


def test_capture_selection_routing_and_callback_order(
    monkeypatch,
    primary_cell,
    run_plan,
):
    configure_single_turn_run(monkeypatch)
    monkeypatch.setattr(
        conversation,
        "PHASES",
        [{"name": "intro", "n_turns": 2, "probe_after": False}],
    )
    events = []
    extension = CapturingExtension("capture", events, selected_turn=2)

    def fake_call_model(*args, **kwargs):
        requests = kwargs.get("capture_requests", ())
        return ModelGeneration(
            "assistant",
            tuple(make_capture(request.request_id, layer) for request in requests for layer in request.layers),
        )

    monkeypatch.setattr(conversation, "call_model", fake_call_model)

    conversation.run_one_vignette(
        primary_cell,
        run_plan,
        extensions=(extension,),
    )

    assert len(extension.capture_contexts) == 1
    capture_context = extension.capture_contexts[0]
    assert capture_context.generation.turn.phase.phase_name == "intro"
    assert capture_context.generation.turn.turn_in_phase == 2
    assert capture_context.generation.response == "assistant"
    assert capture_context.request.request_id == "shared"
    assert [capture.request_id for capture in capture_context.activations] == [
        "shared",
        "shared",
    ]
    assert [capture.layer for capture in capture_context.activations] == [1, 3]
    assert events == [
        ("capture", "before_assistant_generation"),
        ("capture", "activation_capture_requests"),
        ("capture", "after_assistant_generation"),
        ("capture", "before_assistant_generation"),
        ("capture", "activation_capture_requests"),
        ("capture", "after_activation_capture"),
        ("capture", "after_assistant_generation"),
    ]


def test_same_public_request_id_is_routed_to_each_extension(
    monkeypatch,
    primary_cell,
    run_plan,
):
    configure_single_turn_run(monkeypatch)
    first = CapturingExtension("first", [])
    second = CapturingExtension("second", [])
    backend_request_ids = []

    def fake_call_model(*args, **kwargs):
        requests = kwargs["capture_requests"]
        backend_request_ids.extend(request.request_id for request in requests)
        return ModelGeneration(
            "assistant",
            tuple(make_capture(request.request_id, layer) for request in requests for layer in request.layers),
        )

    monkeypatch.setattr(conversation, "call_model", fake_call_model)

    conversation.run_one_vignette(
        primary_cell,
        run_plan,
        extensions=(first, second),
    )

    assert backend_request_ids == ["0:0", "1:0"]
    assert first.capture_contexts[0].request.request_id == "shared"
    assert second.capture_contexts[0].request.request_id == "shared"
    assert all(capture.request_id == "shared" for extension in (first, second) for capture in extension.capture_contexts[0].activations)


class DuplicateRequestExtension(ConversationExtension):
    def activation_capture_requests(self, context):
        return (
            ActivationCaptureRequest("duplicate", layers=(0,)),
            ActivationCaptureRequest("duplicate", layers=(1,)),
        )


def test_duplicate_request_ids_within_extension_event_fail(
    monkeypatch,
    primary_cell,
    run_plan,
):
    configure_single_turn_run(monkeypatch)

    with pytest.raises(ValueError, match="must be unique.*'duplicate'"):
        conversation.run_one_vignette(
            primary_cell,
            run_plan,
            extensions=(DuplicateRequestExtension(),),
        )


@pytest.mark.parametrize("returned_request_id", [None, "unexpected"])
def test_missing_or_unexpected_capture_results_fail(
    monkeypatch,
    primary_cell,
    run_plan,
    returned_request_id,
):
    configure_single_turn_run(monkeypatch)
    extension = CapturingExtension("capture", [])

    def fake_call_model(*args, **kwargs):
        activations = () if returned_request_id is None else (make_capture(returned_request_id, 1),)
        return ModelGeneration("assistant", activations)

    monkeypatch.setattr(conversation, "call_model", fake_call_model)

    match = "no activations" if returned_request_id is None else "unexpected"
    with pytest.raises(ValueError, match=match):
        conversation.run_one_vignette(
            primary_cell,
            run_plan,
            extensions=(extension,),
        )


def test_capture_layer_mismatch_fails(monkeypatch, primary_cell, run_plan):
    configure_single_turn_run(monkeypatch)
    extension = CapturingExtension("capture", [])

    def fake_call_model(*args, **kwargs):
        request = kwargs["capture_requests"][0]
        return ModelGeneration(
            "assistant",
            (make_capture(request.request_id, 1),),
        )

    monkeypatch.setattr(conversation, "call_model", fake_call_model)

    with pytest.raises(ValueError, match=r"expected \[1, 3\], got \[1\]"):
        conversation.run_one_vignette(
            primary_cell,
            run_plan,
            extensions=(extension,),
        )


class FailingCaptureRequestExtension(ConversationExtension):
    def activation_capture_requests(self, context):
        raise RuntimeError("capture request failed")


def test_capture_request_errors_propagate(monkeypatch, primary_cell, run_plan):
    configure_single_turn_run(monkeypatch)

    with pytest.raises(RuntimeError, match="capture request failed"):
        conversation.run_one_vignette(
            primary_cell,
            run_plan,
            extensions=(FailingCaptureRequestExtension(),),
        )


class FailingCaptureCallbackExtension(CapturingExtension):
    def after_activation_capture(self, context):
        raise RuntimeError("capture callback failed")


def test_capture_callback_errors_propagate(
    monkeypatch,
    primary_cell,
    run_plan,
):
    configure_single_turn_run(monkeypatch)
    extension = FailingCaptureCallbackExtension("capture", [])

    def fake_call_model(*args, **kwargs):
        requests = kwargs["capture_requests"]
        return ModelGeneration(
            "assistant",
            tuple(make_capture(request.request_id, layer) for request in requests for layer in request.layers),
        )

    monkeypatch.setattr(conversation, "call_model", fake_call_model)

    with pytest.raises(RuntimeError, match="capture callback failed"):
        conversation.run_one_vignette(
            primary_cell,
            run_plan,
            extensions=(extension,),
        )

import sys
import pytest
from belief_revision.core.models import ActivationCapture, ConversationResult, UserState
from belief_revision.extensions import (
    EXTENSION_REGISTRY,
    ActivationCaptureContext,
    CompletedConversationContext,
    ConversationContext,
    GenerationContext,
    PhaseContext,
    TurnContext,
)
from belief_revision.extensions.torrey import torrey_caa_extension
from pathlib import Path
from types import SimpleNamespace


class FakeTensor:
    def __init__(self, name: str):
        self.name = name
        self.shape = (2,)
        self.dtype = "torch.float32"

    def detach(self):
        return self

    def cpu(self):
        return self


def make_contexts(primary_cell, run_plan):
    conversation = ConversationContext(primary_cell, run_plan, primary_cell.vignette, {})
    phase = PhaseContext(conversation, "intro", {"name": "intro"}, None, None)
    state = UserState(
        50,
        50,
        50,
        50,
        "calm",
        "calm",
        "low",
        "low",
        "test",
        "low",
    )
    turn = TurnContext(phase, 1, 0, False, None, None, state, "user", ())
    generation = GenerationContext(turn, "model", (), 0.0, 10, "assistant")
    result = ConversationResult(primary_cell, run_plan, [], [], [], {})
    completed = CompletedConversationContext(conversation, result)
    return generation, completed


def make_capture(layer: int) -> ActivationCapture:
    return ActivationCapture(
        request_id="torrey_caa",
        layer=layer,
        model_name="model",
        location="residual_post",
        token_position="final_prompt_token",
        token_index=7,
        sequence_length=8,
        shape=(2,),
        dtype="torch.float32",
        tensor=FakeTensor(str(layer)),
    )


def test_torrey_adapter_requests_captures_and_writes_raw_artifact(monkeypatch, tmp_path: Path, primary_cell, run_plan) -> None:
    saved = []
    monkeypatch.setattr(
        torrey_caa_extension,
        "_save_payload",
        lambda path, payload: saved.append((path, payload)),
    )
    generation, completed = make_contexts(primary_cell, run_plan)
    extension = torrey_caa_extension.TorreyCAAExtension(
        output_dir=tmp_path,
        layers=(3, 1),
    )

    request = extension.activation_capture_requests(generation)[0]
    assert request.layers == (3, 1)

    extension.after_activation_capture(
        ActivationCaptureContext(
            generation,
            request,
            (make_capture(3), make_capture(1)),
        )
    )
    extension.after_conversation(completed)

    path, payload = saved[0]
    assert path.parent == tmp_path
    assert path.name.startswith("cell-1__replicate_0__")
    assert payload["artifact_type"] == "live_conversation_activations"
    assert payload["records"][0]["layers"] == [1, 3]
    assert [tensor.name for tensor in payload["records"][0]["activation"]] == [
        "1",
        "3",
    ]
    artifact = extension.artifacts()[0]
    assert artifact.local_path == path
    assert artifact.metadata["extension"] == "torrey-caa"
    assert artifact.activation_records[0].layers == (1, 3)


def test_torrey_adapter_defaults_to_all_layers() -> None:
    extension = torrey_caa_extension.TorreyCAAExtension()
    assert extension.layers is None
    assert EXTENSION_REGISTRY["torrey-caa"] is torrey_caa_extension.TorreyCAAExtension


def test_torrey_adapter_filters_phases_and_turns(primary_cell, run_plan) -> None:
    generation, _ = make_contexts(primary_cell, run_plan)
    extension = torrey_caa_extension.TorreyCAAExtension(
        phases=frozenset({"closing"}),
        turns=frozenset({2}),
    )

    assert extension.activation_capture_requests(generation) == ()

    extension.phases = frozenset({"intro"})
    assert extension.activation_capture_requests(generation) == ()

    extension.turns = frozenset({1})
    assert len(extension.activation_capture_requests(generation)) == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"phases": frozenset()}, "phases must not be empty"),
        ({"turns": frozenset()}, "turns must not be empty"),
        ({"turns": frozenset({0})}, "positive integers"),
    ],
)
def test_torrey_adapter_rejects_empty_or_invalid_filters(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        torrey_caa_extension.TorreyCAAExtension(**kwargs)


def test_save_payload_uses_torch(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(save=lambda payload, path: calls.append((payload, path))),
    )
    path = tmp_path / "capture.pt"
    payload = {"records": []}

    torrey_caa_extension._save_payload(path, payload)

    assert calls == [(payload, path)]

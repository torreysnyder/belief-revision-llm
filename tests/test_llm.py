import numpy as np
import pytest
from belief_revision.core import llm
from belief_revision.core.models import ActivationCaptureRequest, ModelGeneration
from types import SimpleNamespace


class FakeTokenizer:
    def __init__(self, chat_template=True):
        self.chat_template = chat_template

    def apply_chat_template(self, messages, **kwargs):
        return "CHAT:" + "|".join(item["content"] for item in messages)

    def __call__(self, prompt, return_tensors):
        return SimpleNamespace(input_ids=np.array([[1, 2]]))

    def decode(self, tokens, skip_special_tokens):
        return " decoded "


class SavedOutput:
    def __init__(self, value=None):
        self.value = np.array([[1, 2, 3, 4]]) if value is None else value

    def save(self):
        return self.value


class FakeActivation:
    def __init__(self, layer):
        self.value = np.array([float(layer), 2.0], dtype=np.float32)

    def __getitem__(self, key):
        return self

    def float(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self.value

    def save(self):
        return self.value


class GenerateContext:
    def __init__(self, output):
        self.result = SavedOutput(output)
        self.iter = {0: [None]}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeModel:
    def __init__(self, chat_template=True, layer_count=3):
        self.tokenizer = FakeTokenizer(chat_template)
        self.model = SimpleNamespace(layers=[SimpleNamespace(output=FakeActivation(layer)) for layer in range(layer_count)])
        self.options = None

    def generate(self, prompt, **options):
        self.options = options
        return GenerateContext(np.array([[1, 2, 3, 4]]))


def test_activation_capture_request_validation():
    request = ActivationCaptureRequest("all")
    assert request.layers is None
    assert request.location == "residual_post"
    assert request.token_position == "final_prompt_token"

    with pytest.raises(ValueError, match="must not be empty"):
        ActivationCaptureRequest("")
    with pytest.raises(ValueError, match="nonnegative"):
        ActivationCaptureRequest("negative", layers=(-1,))
    with pytest.raises(ValueError, match="duplicate"):
        ActivationCaptureRequest("duplicate", layers=(1, 1))
    with pytest.raises(ValueError, match="location"):
        ActivationCaptureRequest("location", location="other")
    with pytest.raises(ValueError, match="token_position"):
        ActivationCaptureRequest("position", token_position="other")


def test_message_preparation_variants():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hi"},
    ]
    assert llm.prepare_messages_for_model("other/model", messages) is messages
    prepared = llm.prepare_messages_for_model("google/gemma-test", messages)
    assert prepared == [{"role": "user", "content": "rules\n\nhi"}]
    assert llm.prepare_messages_for_model("google/gemma-test", [{"role": "assistant", "content": "a"}]) == [{"role": "assistant", "content": "a"}]
    assert llm.prepare_messages_for_model("google/gemma-test", [{"role": "system", "content": "rules"}]) == [{"role": "user", "content": "rules"}]
    assert (
        llm.prepare_messages_for_model(
            "google/gemma-test",
            [
                {"role": "system", "content": "rules"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "hi"},
            ],
        )[1]["content"]
        == "rules\n\nhi"
    )


def test_key_model_and_prompt(monkeypatch):
    monkeypatch.delenv("NDIF_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NDIF_API_KEY"):
        llm.require_ndif_key()
    monkeypatch.setenv("NDIF_API_KEY", "key")
    created = []
    monkeypatch.setattr(llm, "LanguageModel", lambda name: created.append(name) or FakeModel())
    llm.get_model.cache_clear()
    assert llm.get_model("model") is llm.get_model("model")
    assert created == ["model"]

    chat = FakeModel()
    assert llm.build_chat_prompt(chat, "model", [{"role": "user", "content": "hi"}]) == "CHAT:hi"
    base = FakeModel(chat_template=False)
    assert llm.build_chat_prompt(base, "model", [{"role": "user", "content": "hi"}]).endswith("ASSISTANT:\n")


def test_generate_and_retry_logic(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(llm, "get_model", lambda name: model)
    monkeypatch.setattr(llm, "nnsight_save", lambda value: value)
    result = llm.generate_remote("model", [{"role": "user", "content": "hi"}], 0, 3)
    assert result == ModelGeneration(text="decoded")
    assert model.options == {"remote": True, "max_new_tokens": 3, "do_sample": False}
    llm.generate_remote("model", [{"role": "user", "content": "hi"}], 0.5, 4)
    assert model.options["temperature"] == 0.5

    result = llm.generate_remote(
        "model",
        [{"role": "user", "content": "hi"}],
        capture_requests=(
            ActivationCaptureRequest("selected", layers=(2, 0)),
            ActivationCaptureRequest("all"),
        ),
    )
    assert [(item.request_id, item.layer) for item in result.activations] == [
        ("selected", 2),
        ("selected", 0),
        ("all", 0),
        ("all", 1),
        ("all", 2),
    ]
    capture = result.activations[0]
    assert capture.model_name == "model"
    assert capture.location == "residual_post"
    assert capture.token_position == "final_prompt_token"
    assert capture.token_index == 1
    assert capture.sequence_length == 2
    assert capture.shape == (2,)
    assert capture.dtype == "float32"
    assert np.array_equal(capture.tensor, np.array([2.0, 2.0], dtype=np.float32))

    with pytest.raises(ValueError, match="out of bounds"):
        llm.generate_remote(
            "model",
            [],
            capture_requests=(ActivationCaptureRequest("bad", layers=(3,)),),
        )

    assert not llm.is_retryable_error(RuntimeError("model is not pinned"))
    assert llm.is_retryable_error(RuntimeError("temporary"))

    expected = ModelGeneration(text="ok")
    attempts = iter([RuntimeError("temporary"), expected])

    def generate(**kwargs):
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(llm, "generate_remote", generate)
    monkeypatch.setattr(llm.time, "sleep", lambda delay: None)
    monkeypatch.setattr(llm.random, "random", lambda: 0.0)
    request = ActivationCaptureRequest("request", layers=(0,))
    assert llm.call_model("model", [], retries=2, capture_requests=(request,)) == expected

    monkeypatch.setattr(
        llm,
        "generate_remote",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("not pinned")),
    )
    with pytest.raises(RuntimeError, match="not pinned"):
        llm.call_model("model", [], retries=2)
    with pytest.raises(RuntimeError, match="without returning"):
        llm.call_model("model", [], retries=0)

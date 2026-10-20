from types import SimpleNamespace

import numpy as np
import pytest

from belief_revision import llm


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
    def save(self):
        return np.array([[1, 2, 3, 4]])


class GenerateContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeModel:
    def __init__(self, chat_template=True):
        self.tokenizer = FakeTokenizer(chat_template)
        self.generator = SimpleNamespace(output=SavedOutput())
        self.options = None

    def generate(self, prompt, **options):
        self.options = options
        return GenerateContext()


def test_message_preparation_variants():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hi"},
    ]
    assert llm.prepare_messages_for_model("other/model", messages) is messages
    prepared = llm.prepare_messages_for_model("google/gemma-test", messages)
    assert prepared == [{"role": "user", "content": "rules\n\nhi"}]
    assert llm.prepare_messages_for_model(
        "google/gemma-test", [{"role": "assistant", "content": "a"}]
    ) == [{"role": "assistant", "content": "a"}]
    assert llm.prepare_messages_for_model(
        "google/gemma-test", [{"role": "system", "content": "rules"}]
    ) == [{"role": "user", "content": "rules"}]
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
    monkeypatch.setattr(
        llm, "LanguageModel", lambda name: created.append(name) or FakeModel()
    )
    llm.get_model.cache_clear()
    assert llm.get_model("model") is llm.get_model("model")
    assert created == ["model"]

    chat = FakeModel()
    assert (
        llm.build_chat_prompt(chat, "model", [{"role": "user", "content": "hi"}])
        == "CHAT:hi"
    )
    base = FakeModel(chat_template=False)
    assert llm.build_chat_prompt(
        base, "model", [{"role": "user", "content": "hi"}]
    ).endswith("ASSISTANT:\n")


def test_generate_and_retry_logic(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(llm, "get_model", lambda name: model)
    assert (
        llm.generate_remote("model", [{"role": "user", "content": "hi"}], 0, 3)
        == "decoded"
    )
    assert model.options == {"remote": True, "max_new_tokens": 3, "do_sample": False}
    llm.generate_remote("model", [{"role": "user", "content": "hi"}], 0.5, 4)
    assert model.options["temperature"] == 0.5

    assert not llm.is_retryable_error(RuntimeError("model is not pinned"))
    assert llm.is_retryable_error(RuntimeError("temporary"))

    attempts = iter([RuntimeError("temporary"), "ok"])

    def generate(**kwargs):
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(llm, "generate_remote", generate)
    monkeypatch.setattr(llm.time, "sleep", lambda delay: None)
    monkeypatch.setattr(llm.random, "random", lambda: 0.0)
    assert llm.call_model("model", [], retries=2) == "ok"

    monkeypatch.setattr(
        llm,
        "generate_remote",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("not pinned")),
    )
    with pytest.raises(RuntimeError, match="not pinned"):
        llm.call_model("model", [], retries=2)
    with pytest.raises(RuntimeError, match="without returning"):
        llm.call_model("model", [], retries=0)

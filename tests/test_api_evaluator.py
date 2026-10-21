import pytest
from belief_revision.cli import validate_models
from belief_revision.core import conversation, llm
from belief_revision.core.models import ActivationCaptureRequest, ModelGeneration
from types import SimpleNamespace
from unittest.mock import MagicMock


@pytest.mark.parametrize("provider", llm.API_PROVIDERS)
def test_api_routing_never_calls_ndif(monkeypatch, provider):
    key_name, endpoint = llm.API_PROVIDERS[provider]
    monkeypatch.setenv(key_name, "test-key")
    monkeypatch.delenv("NDIF_API_KEY", raising=False)
    client = MagicMock()
    client.__enter__.return_value = client
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=' {"belief_core": 60} '),
            )
        ]
    )
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(llm, "OpenAI", factory)
    remote = MagicMock(side_effect=AssertionError("must not call NDIF"))
    monkeypatch.setattr(llm, "generate_remote", remote)
    messages = [
        {"role": "system", "content": "Return JSON"},
        {"role": "user", "content": "Full dialogue"},
    ]

    generation = llm.call_model(
        "model",
        messages,
        provider=provider,
        max_tokens=4096,
        retries=2,
    )

    assert generation == ModelGeneration('{"belief_core": 60}')
    factory.assert_called_once_with(
        api_key="test-key",
        base_url=endpoint,
        timeout=120.0,
        max_retries=1,
    )
    token_key = "max_completion_tokens" if provider == "openai" else "max_tokens"
    client.chat.completions.create.assert_called_once_with(
        model="model",
        messages=messages,
        **{token_key: 4096},
    )
    remote.assert_not_called()


def test_api_failures(monkeypatch):
    with pytest.raises(ValueError, match="unknown API provider"):
        llm.call_model("model", [], provider="typo")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        llm.call_model("model", [], provider="deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    with pytest.raises(ValueError, match="at least 1"):
        llm.call_model("model", [], provider="deepseek", retries=0)
    with pytest.raises(ValueError, match="only supported through NDIF"):
        llm.call_model(
            "model",
            [],
            provider="deepseek",
            capture_requests=(ActivationCaptureRequest("capture"),),
        )

    client = MagicMock()
    client.__enter__.return_value = client
    monkeypatch.setattr(llm, "OpenAI", MagicMock(return_value=client))
    for choices, error in [
        ([], "no completion"),
        ([SimpleNamespace(finish_reason="length")], "truncated"),
        (
            [
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=None),
                )
            ],
            "empty",
        ),
        (
            [
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="  "),
                )
            ],
            "empty",
        ),
    ]:
        client.chat.completions.create.return_value = SimpleNamespace(choices=choices)
        with pytest.raises(RuntimeError, match=error):
            llm.call_model("model", [], provider="deepseek")


def test_probe_and_validator_use_configured_api(monkeypatch, vignette):
    call = MagicMock(return_value=ModelGeneration('{"belief_core": 60}'))
    monkeypatch.setattr(conversation, "call_model", call)

    assert conversation.run_probe(vignette, [], "intro").belief_core == 60
    assert call.call_args.kwargs["provider"] == conversation.EVALUATOR_PROVIDER
    assert call.call_args.kwargs["max_tokens"] == conversation.EVALUATOR_MAX_TOKENS

    monkeypatch.setattr(validate_models, "call_model", call)
    assert validate_models.validate_model("evaluator", "model")
    assert call.call_args.kwargs["provider"] == validate_models.EVALUATOR_PROVIDER
    assert validate_models.validate_model("assistant", "model")
    assert call.call_args.kwargs["provider"] == "ndif"

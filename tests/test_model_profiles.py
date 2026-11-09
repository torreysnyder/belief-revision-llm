"""Offline validation of target identities, configuration, and architecture access."""

import yaml
import torch
import pytest
import transformers
from belief_revision.core.defaults import TARGET_MODEL
from belief_revision.core.model_profiles import (
    TargetModelProfile,
    default_target_profile,
    get_model_layers,
    get_text_config,
    layer_output,
    load_hf_target,
    load_target_profile,
    model_tokenizer,
    prepare_profile_messages,
    profile_from_dict,
    resolve_attribute,
)
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_id", None),
        ("model_id", ""),
        ("model_id", "   "),
        ("task", 1),
        ("task", ""),
        ("loader", 1),
        ("loader", ""),
        ("loader", "Unsafe"),
        ("loader", "AutoModel-unsafe"),
        ("revision", 1),
        ("revision", ""),
        ("revision", " "),
        ("dtype", "int8"),
        ("system_role", "ignore_system"),
        ("decoder_path", "model[0]"),
        ("layers_path", 1),
        ("lm_head_path", ""),
        ("logits_path", "model.__class__()"),
        ("final_norm_path", ".norm"),
        ("assistant_end_token", 1),
        ("assistant_end_token", ""),
        ("assistant_end_suffix", None),
        ("layer_output_index", True),
        ("layer_output_index", -1),
        ("expected_layers", False),
        ("expected_layers", 0),
        ("expected_hidden_size", "4"),
        ("expected_hidden_size", -1),
    ],
)
def test_profile_rejects_invalid_settings(field, value):
    values = {"model_id": "custom/model", field: value}
    with pytest.raises(ValueError):
        TargetModelProfile(**values)


def test_profile_roundtrip_and_fingerprint():
    profile = TargetModelProfile("custom/model", revision="commit", dtype="float16", assistant_end_token="<end>", layer_output_index=0, expected_layers=2, expected_hidden_size=4)
    assert profile_from_dict(profile.to_dict()) == profile
    assert profile_from_dict(profile.to_dict()).fingerprint == profile.fingerprint
    assert replace(profile, revision="another").fingerprint != profile.fingerprint
    assert hash(profile)


@pytest.mark.parametrize("values", [None, [], {"typo": "x"}, {}, {"model_id": "x", "typo": 1}])
def test_profile_mapping_rejects_unknown_or_missing_fields(values):
    with pytest.raises(ValueError):
        profile_from_dict(values)


@pytest.mark.parametrize(
    "model_id,end_token,decoder",
    [
        ("meta-llama/Llama-3.1-8B-Instruct", "<|eot_id|>", "model"),
        ("Qwen/Qwen2.5-7B-Instruct", "<|im_end|>", "model"),
        ("google/gemma-2-9b-it", "<end_of_turn>", "model"),
        ("google/gemma-3-4b-it", "<end_of_turn>", "model.language_model"),
        ("mistralai/Ministral-3-3B-Instruct-2512-BF16", "</s>", "model.language_model"),
        ("custom/model", None, "model"),
    ],
)
def test_model_id_compatibility_defaults(model_id, end_token, decoder):
    profile = default_target_profile(model_id)
    assert profile.model_id == model_id
    assert profile.assistant_end_token == end_token
    assert profile.decoder_path == decoder


@pytest.mark.parametrize("model_id", [1, "", "  "])
def test_default_profile_rejects_invalid_model_id(model_id):
    with pytest.raises(ValueError, match="model_id"):
        default_target_profile(model_id)


def test_default_profile_does_not_change_shared_model():
    assert default_target_profile().model_id == TARGET_MODEL
    assert load_target_profile().model_id == TARGET_MODEL
    assert load_target_profile(model_id="custom/model").model_id == "custom/model"


def test_load_profile_from_yaml(tmp_path):
    path = tmp_path / "target.yaml"
    path.write_text("model_id: custom/model\nrevision: pinned\ndtype: float32\n")
    assert load_target_profile(path).revision == "pinned"
    with pytest.raises(ValueError, match="not both"):
        load_target_profile(path, model_id="other/model")


def test_agreed_eight_profiles_have_valid_shared_schema():
    directory = Path(__file__).resolve().parents[1] / "configs/models"
    names = ("llama-3.1-8b-instruct", "llama-3.2-3b-instruct", "qwen-2.5-7b-instruct", "qwen-2.5-3b-instruct", "gemma-3-12b-it", "gemma-3-4b-it", "ministral-3-8b-instruct", "ministral-3-3b-instruct")
    paths = [directory / f"{name}.yaml" for name in names]
    for path in paths:
        profile = load_target_profile(path)
        assert profile.model_id == yaml.safe_load(path.read_text())["model_id"]
        assert profile.assistant_end_token
        assert profile.dtype == "bfloat16"
        assert profile.revision == "main"
        assert profile.expected_layers is None
        assert profile.expected_hidden_size is None


def test_safe_attribute_resolution_and_error():
    model = SimpleNamespace(model=SimpleNamespace(layers=[1, 2]))
    assert resolve_attribute(model, "model.layers") == [1, 2]
    with pytest.raises(ValueError, match="configured path"):
        resolve_attribute(model, "model.missing")
    with pytest.raises(ValueError, match="attribute path"):
        resolve_attribute(model, "model.layers[0]")


def test_tokenizer_handles_text_and_processor_models():
    token = object()
    assert model_tokenizer(SimpleNamespace(tokenizer=token)) is token
    assert model_tokenizer(SimpleNamespace(processor=SimpleNamespace(tokenizer=token))) is token
    with pytest.raises(ValueError, match="tokenizer"):
        model_tokenizer(object())


def test_text_config_and_configured_model_dimensions():
    text = SimpleNamespace(hidden_size=4)
    config = SimpleNamespace(text_config=text)
    model = SimpleNamespace(config=config, model=SimpleNamespace(language_model=SimpleNamespace(layers=[0, 1])))
    profile = TargetModelProfile("custom", layers_path="model.language_model.layers")
    assert get_text_config(text) is text
    assert get_text_config(config) is text
    assert get_text_config(model) is text
    assert get_model_layers(model, profile) == [0, 1]
    assert get_model_layers(model, replace(profile, expected_layers=2, expected_hidden_size=4)) == [0, 1]
    with pytest.raises(ValueError, match="expected_layers"):
        get_model_layers(model, replace(profile, expected_layers=3))
    with pytest.raises(ValueError, match="expected_hidden_size"):
        get_model_layers(model, replace(profile, expected_hidden_size=5))


def test_tensor_and_tuple_layer_output_policies():
    tensor = torch.ones(1, 2, 4)
    profile = TargetModelProfile("custom")
    assert layer_output(tensor, profile) is tensor
    assert layer_output((tensor, None), replace(profile, layer_output_index=0)) is tensor


def test_chat_role_policies_preserve_canonical_input():
    messages = [{"role": "system", "content": "instruction"}, {"role": "user", "content": "question"}]
    profile = TargetModelProfile("custom")
    assert prepare_profile_messages(messages, profile) is messages
    folded = replace(profile, system_role="merge_system_into_user")
    assert prepare_profile_messages(messages, folded) == [{"role": "user", "content": "instruction\n\nquestion"}]
    assert messages[1]["content"] == "question"
    assert prepare_profile_messages([{"role": "user", "content": "question"}], folded) == [{"role": "user", "content": "question"}]
    assert prepare_profile_messages([{"role": "system", "content": "instruction"}, {"role": "assistant", "content": "reply"}], folded) == [{"role": "user", "content": "instruction"}, {"role": "assistant", "content": "reply"}]
    assert prepare_profile_messages([{"role": "system", "content": "instruction"}, {"role": "assistant", "content": "reply"}, {"role": "user", "content": "next"}], folded)[1]["content"] == "instruction\n\nnext"


def test_configured_local_loader_passes_loading_identity_without_downloading(monkeypatch):
    calls = []

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        return "fake-model"

    monkeypatch.setattr(transformers, "AutoModelForProfileTest", SimpleNamespace(from_pretrained=fake_load), raising=False)
    profile = TargetModelProfile("custom/model", loader="AutoModelForProfileTest")
    assert load_hf_target(profile) == "fake-model"
    assert calls[-1] == (("custom/model",), {"torch_dtype": torch.bfloat16})
    assert load_hf_target(replace(profile, revision="pinned"), device_map="auto") == "fake-model"
    assert calls[-1][1]["revision"] == "pinned"
    assert calls[-1][1]["device_map"] == "auto"
    assert load_hf_target(profile, torch_dtype=torch.float32) == "fake-model"
    assert calls[-1][1]["torch_dtype"] == torch.float32
    with pytest.raises(ValueError, match="compatible environment"):
        load_hf_target(replace(profile, loader="AutoModelForProfileMissing"))

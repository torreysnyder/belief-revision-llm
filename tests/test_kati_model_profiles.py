"""Offline boundaries for configurable target capture and safe model caches."""

import json
import torch
import pytest
from belief_revision.core.model_profiles import TargetModelProfile
from belief_revision.extensions.kati.decoding import check_scale, decode, endorsement_probe
from belief_revision.extensions.kati.decoding import extract_activations as extraction
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace


class TemplateTokenizer:
    """A prefix-stable template with an EOT distinct from EOS and optional suffix."""

    eos_token_id = 999
    eos_token = "<eos>"
    pad_token = None

    def __init__(self, marker="<end>", suffix="\n"):
        self.marker = marker
        self.suffix = suffix
        self.calls = []

    def encode(self, text, add_special_tokens=False):
        ids = []
        while text:
            if text.startswith(self.marker):
                ids.append(3)
                text = text[len(self.marker) :]
            else:
                ids.append(ord(text[0]) + 10)
                text = text[1:]
        return ([1] if add_special_tokens else []) + ids

    def __call__(self, text):
        return SimpleNamespace(input_ids=[1, *self.encode(text)])

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        self.calls.append([dict(message) for message in messages])
        text = "".join(f"[{message['role']}]" + message["content"] + self.marker + self.suffix for message in messages)
        return text + ("[assistant]" if add_generation_prompt else "")


@pytest.fixture
def profile():
    return TargetModelProfile("custom/target", assistant_end_token="<end>", assistant_end_suffix="\n", layers_path="decoder.blocks")


@pytest.fixture
def messages(monkeypatch):
    monkeypatch.setattr(extraction, "PHASES", [{"name": "intro", "n_turns": 1, "probe_after": True}])
    return [{"role": "system", "content": "Instructions"}, {"role": "user", "content": "Question"}, {"role": "assistant", "content": "Reply"}]


def test_turn_end_excludes_template_newline_and_does_not_use_eos(profile, messages):
    tokenizer = TemplateTokenizer()
    text, count, checkpoints, spans = extraction.locate_positions(tokenizer, messages, profile)
    ids = tokenizer(text).input_ids
    positions = checkpoints[0]["token_indices"]
    assert ids[positions["reply_eot"]] == 3 != tokenizer.eos_token_id
    assert ids[positions["reply_last"]] == ord("y") + 10
    assert positions["reply_eot"] == count - 2
    assert spans[-1] == tuple(positions["reply_mean"])
    pooled = extraction.pooling_matrix(spans, count)
    assert torch.allclose(pooled.sum(dim=1), torch.ones(4))
    assert pooled[-1, positions["reply_eot"]] == 0


def test_legacy_llama_positions_remain_unchanged(messages):
    tokenizer = TemplateTokenizer("<|eot_id|>", "")
    _, count, checkpoints, _ = extraction.locate_positions(tokenizer, messages)
    assert checkpoints[0]["token_indices"]["reply_eot"] == count - 1


def test_system_role_folding_happens_after_canonical_checkpoint_slicing(profile, messages):
    tokenizer = TemplateTokenizer()
    folded = replace(profile, system_role="merge_system_into_user")
    original = [dict(message) for message in messages]
    _, _, checkpoints, _ = extraction.locate_positions(tokenizer, messages, folded)
    assert messages == original
    assert checkpoints[0]["global_turn_index"] == 1
    assert all(call[0]["role"] == "user" for call in tokenizer.calls)
    assert all(call[0]["content"] == "Instructions\n\nQuestion" for call in tokenizer.calls)
    assert tokenizer.calls[-1][-1]["role"] == "assistant"


@pytest.mark.parametrize("change,error", [({"assistant_end_token": None}, "configure assistant_end_token"), ({"assistant_end_suffix": ""}, "configured assistant delimiter"), ({"assistant_end_token": "split"}, "one token")])
def test_incompatible_turn_boundaries_fail_before_capture(profile, messages, change, error):
    with pytest.raises(ValueError, match=error):
        extraction.locate_positions(TemplateTokenizer(), messages, replace(profile, **change))


def test_empty_reply_and_nonprefix_template_fail(profile, messages):
    empty = [*messages[:-1], {"role": "assistant", "content": ""}]
    with pytest.raises(ValueError, match="non-empty"):
        extraction.locate_positions(TemplateTokenizer(), empty, profile)

    class UnstableTokenizer(TemplateTokenizer):
        def apply_chat_template(self, messages, **options):
            text = super().apply_chat_template(messages, **options)
            return "different-prefix" + text if options["add_generation_prompt"] else text

    with pytest.raises(ValueError, match="not a prefix"):
        extraction.locate_positions(UnstableTokenizer(), messages, profile)


def test_capture_manifest_matches_only_same_model_layers_source_and_prompt(profile, tmp_path, monkeypatch):
    source = tmp_path / "dialogues.jsonl"
    source.write_text("{}\n")
    out = tmp_path / "captures"
    fingerprint = extraction.ensure_output_manifest(out, source, profile, [0, 2], "activations")
    assert fingerprint == extraction.ensure_output_manifest(out, source, profile, [0, 2], "activations")
    manifest = json.loads((out / "capture_manifest.json").read_text())
    assert manifest["target_profile"]["model_id"] == profile.model_id
    assert manifest["capture_fingerprint"] == fingerprint
    with pytest.raises(ValueError, match="fresh --out"):
        extraction.ensure_output_manifest(out, source, replace(profile, revision="other"), [0, 2], "activations")
    with pytest.raises(ValueError, match="fresh --out"):
        extraction.ensure_output_manifest(out, source, profile, [1, 2], "activations")
    source.write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="fresh --out"):
        extraction.ensure_output_manifest(out, source, profile, [0, 2], "activations")
    source.write_text("{}\n")
    monkeypatch.setattr(extraction, "make_target_system_prompt", lambda: "different prompt")
    with pytest.raises(ValueError, match="fresh --out"):
        extraction.ensure_output_manifest(out, source, profile, [0, 2], "activations")


def test_legacy_capture_cache_cannot_be_relabeled(profile, tmp_path):
    source = tmp_path / "dialogues.jsonl"
    source.write_text("{}\n")
    cache = tmp_path / "cache"
    cache.mkdir()
    path = cache / "cell__rep0.pt"
    torch.save({"hidden": torch.zeros(1)}, path)
    with pytest.raises(ValueError, match="no capture manifest"):
        extraction.ensure_output_manifest(cache, source, profile, None, "endorsement")
    with pytest.raises(ValueError, match="provenance mismatch"):
        extraction.validate_cached_capture(path, "expected")
    extraction.validate_cached_capture(path, None)
    torch.save({"capture_fingerprint": "expected"}, path)
    extraction.validate_cached_capture(path, "expected")


def test_nested_layers_dynamic_width_and_saved_profile(profile, tmp_path, messages, monkeypatch):
    model = SimpleNamespace(tokenizer=TemplateTokenizer(), decoder=SimpleNamespace(blocks=[object(), object()]))
    record = {"cell_id": "cell", "replicate_index": 0, "dialogue_history": [{"speaker": message["role"], "text": message["content"]} for message in messages[1:]]}
    seen = []

    def fake_extract(model, text, pooling, layers, selected):
        seen.append((layers, selected))
        return torch.ones(len(pooling), len(layers), 5)

    monkeypatch.setattr(extraction, "extract_remote", fake_extract)
    extraction.process(model, record, tmp_path, None, profile, "run-fingerprint")
    saved = torch.load(tmp_path / "cell__rep0.pt")
    assert tuple(saved["activations"].shape) == (1, 4, 2, 5)
    assert saved["model_name"] == profile.model_id
    assert saved["target_profile"] == profile.to_dict()
    assert saved["capture_fingerprint"] == "run-fingerprint"
    assert seen == [([0, 1], profile)]
    assert extraction.process(model, record, tmp_path, None, profile, "run-fingerprint").startswith("skip")
    with pytest.raises(ValueError, match="provenance mismatch"):
        extraction.process(model, record, tmp_path, None, profile, "other")


def test_final_native_logits_and_nested_tuple_layer_outputs(profile, monkeypatch):
    class SavedTensor:
        def __init__(self, tensor):
            self.tensor = tensor

        def __getitem__(self, index):
            return SavedTensor(self.tensor[index])

        def float(self):
            return self

        def cpu(self):
            return self

        def save(self):
            return self.tensor

    hidden = torch.ones(2, 3, 5)
    logits = torch.arange(36, dtype=torch.float32).reshape(2, 3, 6)
    model = SimpleNamespace(
        decoder=SimpleNamespace(blocks=[SimpleNamespace(output=(hidden,))]),
        output=SimpleNamespace(logits=SavedTensor(logits)),
        trace=lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(endorsement_probe, "nnsight_save", lambda value: value)
    captured, result = endorsement_probe.run_batch(model, ["one", "two"], [0], [0, 2, 4], replace(profile, layer_output_index=0))
    assert tuple(captured.shape) == (2, 1, 5)
    assert torch.equal(result, logits[:, -1, [0, 2, 4]])


def test_split_ab_option_still_fails():
    tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1, 2])
    with pytest.raises(ValueError, match="not a single token"):
        endorsement_probe.option_token_ids(tokenizer)


def test_legacy_analysis_still_loads_but_mixed_models_are_rejected(tmp_path):
    cache = tmp_path / "separate-activations"
    cache.mkdir()
    saved = {"cell_id": "a", "replicate_index": 0, "model_name": "legacy/model", "layers": [0, 1], "positions": ["prompt_end"], "checkpoints": [{"phase": "intro"}], "activations": torch.zeros(1, 1, 2, 5)}
    torch.save(saved, cache / "a.pt")
    values, keys, layers = check_scale.load_samples(tmp_path, "prompt_end", cache)
    assert values.shape == (1, 2, 5)
    assert keys["phase"].tolist() == ["intro"]
    assert layers == [0, 1]
    torch.save({**saved, "model_name": "other/model"}, cache / "b.pt")
    with pytest.raises(ValueError, match="Mixed model profiles"):
        check_scale.load_samples(tmp_path, "prompt_end", cache)


def test_separate_endorsement_cache_and_empty_paths(tmp_path):
    cache = tmp_path / "separate-endorsement"
    cache.mkdir()
    saved = {"cell_id": "a", "replicate_index": 0, "layers": [0, 1], "phases": ["intro"], "hidden": torch.zeros(1, 3, 2, 2, 5)}
    torch.save(saved, cache / "a__rep0.pt")
    values, _, layers = decode.load_probe_samples(tmp_path, "A1", "B_endorse", cache)
    assert values.shape == (1, 2, 5)
    assert layers == [0, 1]
    with pytest.raises(ValueError, match="No saved activation tensors"):
        check_scale.load_samples(tmp_path, "prompt_end")
    with pytest.raises(ValueError, match="No saved endorsement tensors"):
        decode.load_probe_samples(tmp_path, "A1", "A_endorse")


def test_metadata_layer_count_must_match_tensor(tmp_path):
    with pytest.raises(ValueError, match="Layer metadata"):
        extraction.validate_tensor_identity({"layers": [0], "hidden": torch.zeros(2, 5)}, tmp_path / "bad.pt")


@pytest.mark.parametrize("workflow", ["extraction", "endorsement"])
def test_explicit_profile_cli_ignores_conflicting_global_model(workflow, tmp_path, messages, monkeypatch):
    import sys
    from belief_revision.core import defaults, llm

    selected = extraction if workflow == "extraction" else endorsement_probe
    target = "meta-llama/Llama-3.1-8B-Instruct"
    config = tmp_path / "llama.yaml"
    config.write_text(f'model_id: {target}\nassistant_end_token: "<|eot_id|>"\nlayers_path: decoder.blocks\n')
    record = {
        "cell_id": "cell",
        "replicate_index": 0,
        "vignette_id": "v1",
        "primary_cell": {"vignette": {"core_claim": "Core", "auxiliaries": [{"id": "A1", "text": "One"}, {"id": "A2", "text": "Two"}]}},
        "dialogue_history": [{"speaker": message["role"], "text": message["content"]} for message in messages[1:]],
    }
    (tmp_path / "dialogues.jsonl").write_text(json.dumps(record) + "\n")
    model = SimpleNamespace(tokenizer=TemplateTokenizer("<|eot_id|>", ""), decoder=SimpleNamespace(blocks=[object(), object()]))
    calls = []

    def get_selected(model_name, target_profile=None):
        calls.append((model_name, target_profile))
        return model

    monkeypatch.setattr(defaults, "TARGET_MODEL", "global/wrong-model")
    monkeypatch.setattr(selected, "TARGET_MODEL", "global/wrong-model")
    monkeypatch.setattr(llm, "get_model", get_selected)
    monkeypatch.setattr(extraction, "extract_remote", lambda model, text, pooling, layers, profile: torch.ones(len(pooling), len(layers), 5))
    monkeypatch.setattr(endorsement_probe, "option_token_ids", lambda tokenizer: [1, 2, 3, 4, 5, 6])

    def fake_probe(model, prefix, claims, layers, option_ids, batch_size, profile=None):
        assert profile.model_id == target
        return [{"claim": "core", "mapping": "A_endorse", "format": "AB", "E": 1.0}], torch.zeros(3, 2, len(layers), 5)

    monkeypatch.setattr(endorsement_probe, "probe_prefix_batch", fake_probe)
    out = tmp_path / "new-captures"
    monkeypatch.setattr(sys, "argv", ["tool", "--run", str(tmp_path), "--model-config", str(config), "--out", str(out)])
    selected.main()

    assert len(calls) == 1
    assert calls[0][0] == calls[0][1].model_id == target
    saved = torch.load(out / "cell__rep0.pt")
    assert saved["target_profile"]["model_id"] == target
    assert json.loads((out / "capture_manifest.json").read_text())["target_profile"]["model_id"] == target
    if workflow == "extraction":
        assert saved["model_name"] == target
    assert defaults.TARGET_MODEL == selected.TARGET_MODEL == "global/wrong-model"


def test_changed_probe_question_requires_new_capture_directory(profile, tmp_path):
    source = tmp_path / "dialogues.jsonl"
    source.write_text("{}\n")
    out = tmp_path / "endorsement"
    extraction.ensure_output_manifest(out, source, profile, [0], "endorsement", {"question": "Original question"})
    with pytest.raises(ValueError, match="fresh --out"):
        extraction.ensure_output_manifest(out, source, profile, [0], "endorsement", {"question": "Changed question"})

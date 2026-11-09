"""Offline checks for research target selection and frozen activation spaces."""

import ast
import sys
import json
import yaml
import torch
import pytest
import hashlib
import inspect
import subprocess
import importlib.util
import belief_revision
from belief_revision.core.model_profiles import TargetModelProfile, load_target_profile
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(belief_revision.__file__).resolve().parents[2]
PIPELINE = REPO_ROOT / "activation_steering_pipeline"
NOTEBOOK = REPO_ROOT / "src/belief_revision/extensions/qingqing/endorsement_validation/test_endorse_probe.ipynb"


def script(name):
    spec = importlib.util.spec_from_file_location(f"research_{name}", PIPELINE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return script("generate_caa_contrastive_pairs")


@pytest.fixture(scope="module")
def extractor():
    return script("extract_caa_activations")


@pytest.fixture(scope="module")
def steering():
    return script("activation_steering")


@pytest.fixture
def profile_file(tmp_path):
    profile = TargetModelProfile("example/target", revision="fixed-revision", layers_path="language.blocks", decoder_path="language", assistant_end_token="<end>")
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(profile.to_dict()))
    return profile, path


def test_pair_generator_keeps_target_tokenizer_and_manifest_aligned(generator, profile_file):
    profile, path = profile_file
    assert generator.select_target_profile(path) == profile
    assert generator.select_target_profile(target_model=profile.model_id).model_id == profile.model_id
    assert generator.select_target_profile(tokenizer=profile.model_id).model_id == profile.model_id
    with pytest.raises(ValueError, match="tokenizer"):
        generator.select_target_profile(path, tokenizer="another/target")
    generator.validate_pair_profile({"target_model": profile.model_id, "target_profile": profile.to_dict()}, profile)
    with pytest.raises(ValueError, match="another target"):
        generator.validate_pair_profile({"target_model": "another/target"}, profile)
    with pytest.raises(ValueError, match="profile changed"):
        generator.validate_pair_profile({"target_model": profile.model_id, "target_profile": {}}, profile)


def test_offline_pair_command_records_selected_profile(generator, profile_file, tmp_path):
    profile, path = profile_file
    output = tmp_path / "pairs"
    result = subprocess.run(
        [sys.executable, str(PIPELINE / "generate_caa_contrastive_pairs.py"), "--model-config", str(path), "--input", str(REPO_ROOT / "tests/fixtures/baseline_demo_5/conversations.jsonl"), "--out-dir", str(output), "--offline", "--no-render", "--limit", "1"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "caa_pairs_manifest.json").read_text())
    assert manifest["target_model"] == profile.model_id
    assert manifest["target_profile"] == profile.to_dict()
    assert manifest["target_profile_fingerprint"] == profile.fingerprint
    assert manifest["api_calls_used"] == 0


def test_pair_rendering_uses_explicit_role_policy(generator):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return json.dumps(messages)

    profile = TargetModelProfile("example/model", system_role="merge_system_into_user")
    messages = [{"role": "system", "content": "instruction"}, {"role": "user", "content": "question"}]
    rendered = json.loads(generator.render(Tokenizer(), messages, profile))
    assert rendered == [{"role": "user", "content": "instruction\n\nquestion"}]
    assert messages[0]["role"] == "system"


def test_extraction_reuses_frozen_profile_and_rejects_other_targets(extractor, profile_file):
    profile, path = profile_file
    manifest = {"target_model": profile.model_id, "target_profile": profile.to_dict()}
    assert extractor.extraction_profile(manifest) == profile
    assert extractor.extraction_profile(manifest, path) == profile
    with pytest.raises(ValueError, match="conflicts"):
        extractor.extraction_profile(manifest, model_name="another/target")
    with pytest.raises(ValueError, match="differs"):
        extractor.extraction_profile({"target_model": "another/target"}, path)


def test_extraction_resolves_nested_layers_and_loader_without_weights(extractor, monkeypatch, profile_file):
    profile, _ = profile_file
    model = SimpleNamespace(language=SimpleNamespace(blocks=[object(), object()]), config=SimpleNamespace(_commit_hash="resolved"))
    model.to = lambda device: model
    model.eval = lambda: model
    calls = []
    monkeypatch.setattr(extractor, "resolve_revision", lambda name, revision: "resolved")
    monkeypatch.setattr(extractor.AutoTokenizer, "from_pretrained", lambda name, **kwargs: "tokenizer")
    monkeypatch.setattr(extractor, "load_hf_target", lambda selected, **kwargs: calls.append((selected, kwargs)) or model)

    loaded, tokenizer, revision = extractor.load_model_and_tokenizer(profile.model_id, None, "float16", "cpu", None, profile)
    assert (loaded, tokenizer, revision) == (model, "tokenizer", "resolved")
    assert calls == [(profile, {"revision": "resolved", "torch_dtype": torch.float16})]
    decoder, layers = extractor.decoder_layers(model, profile)
    assert decoder is model.language
    assert layers is model.language.blocks


def test_vector_steering_rejects_different_profile_or_tampered_identity(steering, profile_file):
    profile, path = profile_file
    metadata = {"model_name": profile.model_id, "target_profile": profile.to_dict(), "target_profile_fingerprint": profile.fingerprint}
    assert steering.vector_profile(metadata, path) == profile
    with pytest.raises(ValueError, match="identity"):
        steering.vector_profile({**metadata, "model_name": "another/target"})
    with pytest.raises(ValueError, match="fingerprint"):
        steering.vector_profile({**metadata, "target_profile_fingerprint": "invalid"})
    path.write_text(yaml.safe_dump({"model_id": "another/target"}))
    with pytest.raises(ValueError, match="frozen vectors"):
        steering.vector_profile(metadata, path)
    with pytest.raises(ValueError, match="legacy"):
        steering.vector_profile({"model_name": profile.model_id}, path)


def test_vector_builder_rejects_mixed_target_profiles(tmp_path):
    builder = script("build_caa_vectors")
    split = tmp_path / "split.json"
    split.write_text("{}")
    split_hash = hashlib.sha256(split.read_bytes()).hexdigest()
    (tmp_path / "activation_manifest.json").write_text("{}")
    for family in ("core", "auxiliary"):
        tensor = torch.zeros(1, 2, 3)
        payload = {
            "format_version": 1,
            "contrast_type": family,
            "pair_metadata": [{}],
            "n_layers": 2,
            "hidden_dim": 3,
            "target_profile": {"model_id": f"example/{family}"},
            "split_manifest_sha256": split_hash,
            "experimental_activation": tensor,
            "control_activation": tensor,
            "pair_difference": tensor,
        }
        torch.save(payload, tmp_path / f"{family}_activations.pt")
    with pytest.raises(SystemExit, match="target_profile"):
        builder.load_activations(str(tmp_path), str(split), False)


def test_capture_and_steering_respect_configured_tuple_output_index(extractor, steering):
    class TupleBlock(torch.nn.Module):
        def forward(self, hidden):
            return torch.zeros_like(hidden), hidden

    block = TupleBlock()
    profile = TargetModelProfile("example/tuple-model", layer_output_index=1)
    capture = extractor.BlockOutputCapture([block], profile)
    capture.positions = [[0]]
    hidden = torch.tensor([[[1.0, 2.0, 3.0]]])
    block(hidden)
    assert torch.equal(capture.collect()[0, 0], hidden[0, 0])
    capture.remove()
    steerer = steering.Steerer([block], [0], profile)
    steerer.set({0: torch.ones(1, 3)}, "cpu", torch.float32)
    untouched, shifted = block(hidden)
    assert torch.equal(untouched, torch.zeros_like(hidden))
    assert torch.equal(shifted, hidden + 1)
    steerer.remove()


def notebook_functions(index, names, extra=None):
    notebook = json.loads(NOTEBOOK.read_text())
    module = ast.parse("".join(notebook["cells"][index]["source"]))
    selected = ast.Module(body=[node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    namespace = {"Path": Path, "json": json, "torch": torch, "inspect": inspect}
    namespace.update(extra or {})
    exec(compile(selected, "notebook-functions", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("path", sorted((REPO_ROOT / "configs/models").glob("*.yaml")), ids=lambda path: path.stem)
def test_notebook_consumes_same_flat_profile_as_core(path):
    functions = notebook_functions(8, {"notebook_target_profile"})
    assert functions["notebook_target_profile"](str(path), "old/model", "main") == load_target_profile(path).to_dict()


def test_notebook_profile_roles_are_configurable_without_mutating_messages():
    functions = notebook_functions(8, {"notebook_target_profile", "prepare_target_messages"})
    profile = functions["notebook_target_profile"]("", "Qwen/Qwen2.5-7B-Instruct", "main")
    assert profile["assistant_end_token"] == "<|im_end|>"
    functions["TARGET_PROFILE"] = {**profile, "system_role": "merge_system_into_user"}
    messages = [{"role": "system", "content": "instruction"}, {"role": "user", "content": "question"}]
    assert functions["prepare_target_messages"](messages) == [{"role": "user", "content": "instruction\n\nquestion"}]
    assert messages[0]["role"] == "system"


def test_notebook_native_readout_handles_post_head_softcapping():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(5, 3)
            self.norm = torch.nn.LayerNorm(3)
            self.head = torch.nn.Linear(3, 5)

        def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True, logits_to_keep=1):
            hidden = self.norm(self.embedding(input_ids))
            logits = self.head(hidden[:, -logits_to_keep:]).float()
            return SimpleNamespace(logits=torch.tanh(logits / 2) * 2)

    model = TinyModel()
    helpers = notebook_functions(8, {"target_attribute"})
    functions = notebook_functions(
        17,
        {"projected_logits", "native_last_readout"},
        {"model": model, "lm_head": model.head, "final_norm": model.norm, "TEXT_CONFIG": SimpleNamespace(final_logit_softcapping=2), "TARGET_PROFILE": {"logits_path": "output.logits"}, "HEAD_READOUT_VERIFIED": False, "target_attribute": helpers["target_attribute"]},
    )
    hidden, logits = functions["native_last_readout"](torch.tensor([[1, 2]]))
    assert tuple(hidden.shape) == (3,)
    assert tuple(logits.shape) == (5,)
    assert torch.allclose(logits, functions["projected_logits"](hidden[None])[0])
    assert functions["HEAD_READOUT_VERIFIED"] is True
    assert not model.norm._forward_hooks


def test_notebook_python_cells_compile_and_no_fixed_qwen_shape_assertions():
    notebook = json.loads(NOTEBOOK.read_text())
    sources = []
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        sources.append(source)
        if not source.lstrip().startswith("%"):
            compile(source, f"notebook-cell-{i}", "exec")
    combined = "\n".join(sources)
    assert "(28,3584)" not in combined
    assert "enumerate(model.model.layers)" not in combined
    assert "MODEL_CONFIG_PATH" in combined
    assert "target_profile_fingerprint" in combined

"""Offline target selection, routing, and provenance integration checks."""

import json
import numpy as np
import pytest
from belief_revision.core import conversation, defaults, llm, runner
from belief_revision.core.model_profiles import TargetModelProfile
from belief_revision.core.models import ActivationCapture, ActivationCaptureRequest, BehavioralProbeResult, ModelGeneration
from belief_revision.extensions import ConversationExtension
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from scripts import run_experiment as cli
from tests.test_llm import FakeActivation, FakeModel, FakeTokenizer
from types import SimpleNamespace


def target_args(**values):
    return SimpleNamespace(**{"model_config": None, "target_model": None, "config": None, **values})


def write_profile(path, model_id="custom/target"):
    path.write_text(f"model_id: {model_id}\nrevision: pinned-revision\n", encoding="utf-8")
    return path


def test_target_cli_precedence_and_relative_run_yaml(tmp_path, monkeypatch):
    profiles = tmp_path / "models"
    profiles.mkdir()
    profile = write_profile(profiles / "target.yaml")
    config = tmp_path / "experiment.yaml"
    document = {"target": {"model_config": "models/target.yaml"}}
    selected = cli.target_configuration(target_args(config=config), document)
    assert selected.model_id == "custom/target"
    assert selected.revision == "pinned-revision"
    assert cli.target_configuration(target_args(), {"target": {"model_config": str(profile)}}) == selected
    monkeypatch.chdir(tmp_path)
    assert cli.target_configuration(target_args(), document) == selected
    assert cli.target_configuration(target_args(target_model="Qwen/alternate"), document).model_id == "Qwen/alternate"
    assert cli.target_configuration(target_args(model_config=profile), {"target": {"model_id": "ignored/id"}}) == selected
    assert cli.target_configuration(target_args(), {"target": {"model_id": "Qwen/from-yaml"}}).model_id == "Qwen/from-yaml"
    assert cli.target_configuration(target_args(), {}).model_id == defaults.TARGET_MODEL


@pytest.mark.parametrize(
    "document,arguments,message",
    [
        ({"models": {}}, {}, "unknown run config"),
        ({"target": []}, {}, "must be a mapping"),
        ({"target": {"revision": "x"}}, {}, "unknown target config"),
        ({"target": {"model_config": "x", "model_id": "y"}}, {}, "not both"),
        ({}, {"model_config": Path("x"), "target_model": "y"}, "not both"),
        ({"target": {"model_config": 3}}, {}, "path string"),
        ({"target": {"model_config": ""}}, {}, "path string"),
        ({"target": {"model_id": 3}}, {}, "nonempty string"),
        ({}, {"target_model": ""}, "nonempty string"),
    ],
)
def test_invalid_target_configuration_fails_before_execution(document, arguments, message):
    with pytest.raises(ValueError, match=message):
        cli.target_configuration(target_args(**arguments), document)


def test_manifest_and_output_names_record_selected_target(tmp_path, monkeypatch):
    profile = TargetModelProfile("custom/selected", revision="commit-a")
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path / "runs")
    args = SimpleNamespace(csv_output=None, jsonl_output=None, run_name=None, single_run=True)
    csv_path, jsonl_path = cli.resolve_output_paths(args, profile.model_id)
    assert "selected_single" in csv_path.parent.name
    evaluator = {"mode": "disabled", "provider": "openrouter", "model": "judge", "max_tokens": 100}
    path = cli.write_run_manifest(args, csv_path, jsonl_path, profile, evaluator)
    assert path.name == "run_manifest.json"
    saved = json.loads(path.read_text())
    assert saved["target_profile"] == profile.to_dict()
    assert saved["target_profile_fingerprint"] == profile.fingerprint
    assert saved["target_model"] == profile.model_id
    assert saved["user_sim_model"] == defaults.USER_SIM_MODEL
    assert saved["evaluator"] == evaluator
    assert saved["outputs"]["jsonl"] == str(jsonl_path)
    args.csv_output = tmp_path / "explicit.csv"
    path = cli.write_run_manifest(args, args.csv_output, jsonl_path, profile, evaluator)
    assert path == tmp_path / "explicit.run_manifest.json"


def test_cli_target_arguments_and_error_reporting(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["run_experiment.py", "--model-config", "model.yaml"])
    assert cli.parse_args().model_config == Path("model.yaml")
    monkeypatch.setattr("sys.argv", ["run_experiment.py", "--target-model", "custom/model"])
    assert cli.parse_args().target_model == "custom/model"
    monkeypatch.setattr("sys.argv", ["run_experiment.py", "--model-config", "a", "--target-model", "b"])
    with pytest.raises(SystemExit):
        cli.parse_args()
    monkeypatch.setattr(cli, "parse_args", lambda: target_args(config=tmp_path / "missing.yaml"))
    with pytest.raises(SystemExit, match="error:"):
        cli.main()


def test_model_handle_cache_and_requested_identity(monkeypatch):
    created = []
    monkeypatch.setenv("NDIF_API_KEY", "offline-test-key")
    monkeypatch.setattr(llm, "LanguageModel", lambda name, **options: created.append((name, options)) or object())
    llm.get_model.cache_clear()
    profile = TargetModelProfile("custom/model", revision="first", task="image-text-to-text")
    assert llm.get_model(profile.model_id, profile) is llm.get_model(profile.model_id, profile)
    second = replace(profile, revision="second")
    assert llm.get_model(second.model_id, second) is not llm.get_model(profile.model_id, profile)
    assert created[0] == ("custom/model", {"task": "image-text-to-text", "revision": "first"})
    no_revision = replace(profile, revision=None)
    llm.get_model(no_revision.model_id, no_revision)
    assert created[-1][1] == {"task": "image-text-to-text"}
    with pytest.raises(ValueError, match="does not match"):
        llm.get_model("different/model", profile)
    llm.get_model.cache_clear()


def test_nested_capture_tuple_output_and_processor_tokenizer(monkeypatch):
    profile = TargetModelProfile("custom/nested", layers_path="backbone.blocks", layer_output_index=0, system_role="merge_system_into_user")
    model = FakeModel()
    model.tokenizer = None
    model.processor = SimpleNamespace(tokenizer=FakeTokenizer())
    model.backbone = SimpleNamespace(blocks=[SimpleNamespace(output=(FakeActivation(7),))])
    requests = []
    monkeypatch.setattr(llm, "get_model", lambda name, selected: requests.append((name, selected)) or model)
    monkeypatch.setattr(llm, "nnsight_save", lambda value: value)
    result = llm.generate_remote(profile.model_id, [{"role": "system", "content": "rules"}, {"role": "user", "content": "question"}], capture_requests=(ActivationCaptureRequest("chosen", layers=(0,)),), target_profile=profile)
    assert requests == [(profile.model_id, profile)]
    assert result.activations[0].layer == 0
    assert result.activations[0].model_name == profile.model_id
    assert np.array_equal(result.activations[0].tensor, np.array([7.0, 2.0], dtype=np.float32))
    assert llm.build_chat_prompt(model, profile.model_id, [{"role": "system", "content": "rules"}, {"role": "user", "content": "question"}], profile) == "CHAT:rules\n\nquestion"


def test_profile_call_model_routing_rejects_mismatch_without_calls(monkeypatch):
    profile = TargetModelProfile("custom/model")
    calls = []
    monkeypatch.setattr(llm, "generate_remote", lambda **kwargs: calls.append(kwargs) or ModelGeneration("reply"))
    assert llm.call_model(profile.model_id, [], target_profile=profile).text == "reply"
    assert calls[-1]["target_profile"] is profile
    with pytest.raises(ValueError, match="matching model_id"):
        llm.call_model("different/model", [], target_profile=profile)
    with pytest.raises(ValueError, match="requires NDIF"):
        llm.call_model(profile.model_id, [], target_profile=profile, provider="openai")
    assert len(calls) == 1


@pytest.mark.parametrize("capture", [False, True])
def test_selected_target_reaches_assistant_and_hook_contexts(monkeypatch, primary_cell, run_plan, capture):
    profile = TargetModelProfile("custom/assistant")
    monkeypatch.setattr(conversation, "PHASES", [{"name": "intro", "n_turns": 1, "probe_after": True}])
    user_calls, probe_calls, model_calls, contexts = [], [], [], []
    monkeypatch.setattr(conversation, "generate_user_turn", lambda **kwargs: user_calls.append(kwargs) or "question")
    monkeypatch.setattr(conversation, "run_probe", lambda **kwargs: probe_calls.append(kwargs) or BehavioralProbeResult(kwargs["phase_name"], belief_core=50))

    def model_call(name, messages, **kwargs):
        model_calls.append((name, kwargs))
        captured = tuple(ActivationCapture(request.request_id, 0, name, "residual_post", "final_prompt_token", 1, 2, (2,), "float32", np.ones(2)) for request in kwargs.get("capture_requests", ()))
        return ModelGeneration("answer", captured)

    monkeypatch.setattr(conversation, "call_model", model_call)

    class Observer(ConversationExtension):
        def before_assistant_generation(self, context):
            contexts.append(context.model)

        def after_assistant_generation(self, context):
            contexts.append(context.model)

        def activation_capture_requests(self, context):
            return (ActivationCaptureRequest("chosen", layers=(0,)),) if capture else ()

    conversation.run_one_vignette(primary_cell, run_plan, extensions=(Observer(),), evaluator_provider="openai", evaluator_model="judge-original", target_profile=profile)
    assert contexts == [profile.model_id, profile.model_id]
    assert model_calls[0][0] == profile.model_id
    assert model_calls[0][1]["target_profile"] is profile
    assert "target_profile" not in user_calls[0]
    assert probe_calls[0]["evaluator_model"] == "judge-original"
    assert probe_calls[0]["evaluator_provider"] == "openai"


def test_concurrent_runner_profiles_do_not_change_defaults(monkeypatch, primary_cell, run_plan, tmp_path, vignette):
    original = defaults.TARGET_MODEL
    monkeypatch.setattr(runner, "build_run_plans", lambda *args: [run_plan])
    received = []

    def run(**kwargs):
        received.append(kwargs["target_profile"])
        return SimpleNamespace(run_plan=kwargs["run_plan"], primary_cell=kwargs["primary_cell"])

    monkeypatch.setattr(runner, "run_one_vignette", run)
    profiles = [TargetModelProfile("custom/one"), TargetModelProfile("custom/two")]
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(lambda profile: runner.run_primary_cell(primary_cell, 1, 1, target_profile=profile), profiles))
    assert all(len(output) == 1 for output in outputs)
    assert set(received) == set(profiles)
    assert defaults.TARGET_MODEL == original
    metadata = []

    class Backend:
        def start_run(self, **kwargs):
            metadata.append(kwargs)
            return "offline-run"

        def save_conversations(self, *args):
            pass

        def complete_run(self, *args, **kwargs):
            pass

    monkeypatch.setattr(runner, "LocalFileBackend", lambda *args: Backend())
    monkeypatch.setattr(runner, "build_primary_cells", lambda values: [primary_cell])
    runner.run_experiment([vignette], tmp_path / "results.csv", tmp_path / "dialogues.jsonl", replicates_per_cell=1, max_workers=1, target_profile=profiles[0])
    assert metadata[0]["target_model"] == profiles[0].model_id
    assert metadata[0]["config"]["target_profile"] == profiles[0].to_dict()
    assert metadata[0]["config"]["target_profile_fingerprint"] == profiles[0].fingerprint
    assert metadata[0]["user_sim_model"] == defaults.USER_SIM_MODEL
    assert metadata[0]["evaluator_model"] == defaults.EVALUATOR_MODEL


def test_cli_profile_flows_through_real_runner_into_assistant(monkeypatch, tmp_path, vignette):
    profile_path = write_profile(tmp_path / "target.yaml", "custom/cli-selected")
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(cli, "load_vignettes", lambda path: [vignette])
    monkeypatch.setattr("sys.argv", ["run_experiment.py", "--single-run", "--model-config", str(profile_path), "--run-name", "selected-target", "--evaluator-mode", "disabled"])
    monkeypatch.setattr(conversation, "PHASES", [{"name": "intro", "n_turns": 1, "probe_after": True}])
    monkeypatch.setattr(conversation, "generate_user_turn", lambda **kwargs: "A question.")
    calls = []

    def generate(model_name, messages, **kwargs):
        calls.append((model_name, kwargs["target_profile"]))
        return ModelGeneration("An answer.")

    monkeypatch.setattr(conversation, "call_model", generate)
    cli.main()
    assert len(calls) == 1
    assert calls[0][0] == "custom/cli-selected"
    assert calls[0][1].model_id == "custom/cli-selected"
    saved = tmp_path / "runs" / "selected-target"
    manifest = json.loads((saved / "run_manifest.json").read_text())
    assert manifest["target_profile"] == calls[0][1].to_dict()
    assert manifest["user_sim_model"] == defaults.USER_SIM_MODEL
    assert manifest["evaluator"]["model"] == defaults.EVALUATOR_MODEL
    assert (saved / "results.csv").is_file()
    record = json.loads((saved / "dialogues.jsonl").read_text())
    assert record["dialogue_history"][-1]["text"] == "An answer."

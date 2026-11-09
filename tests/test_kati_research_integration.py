"""Offline checks for Kati's research tools at the shared-core boundary."""

import json
import torch
import pandas as pd
import pytest
from belief_revision.core import conversation, defaults
from belief_revision.core.models import BehavioralProbeResult, ModelGeneration
from belief_revision.extensions.kati.decoding import (
    check_scale,
    decode,
    endorsement_probe,
    extract_activations,
    make_labels,
)
from belief_revision.extensions.kati.results_summary import make_figures
from belief_revision.persistence.local import build_conversation_record, save_results_append
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_research_package_imports_share_helpers():
    assert decode.load_samples is check_scale.load_samples
    assert decode.checkpoint_turns is extract_activations.checkpoint_turns
    assert endorsement_probe.build_messages is extract_activations.build_messages
    assert endorsement_probe.checkpoint_turns is extract_activations.checkpoint_turns
    assert extract_activations.TARGET_MODEL == defaults.TARGET_MODEL
    assert endorsement_probe.TARGET_MODEL == defaults.TARGET_MODEL


def test_saved_core_conversation_matches_extraction_prompt_and_checkpoints():
    fixture = REPO_ROOT / "tests" / "fixtures" / "baseline_demo_5" / "conversations.jsonl"
    record = json.loads(fixture.read_text(encoding="utf-8").splitlines()[0])
    messages = extract_activations.build_messages(record)

    assert messages[0] == {"role": "system", "content": conversation.make_target_system_prompt()}
    assert [message["content"] for message in messages[1:]] == [turn["text"] for turn in record["dialogue_history"]]
    assert [message["role"] for message in messages[1:]] == [turn["speaker"] for turn in record["dialogue_history"]]
    assert extract_activations.checkpoint_turns() == [{key: row[key] for key in ("phase", "turn_in_phase", "global_turn_index")} for row in record["rows"] if row["probe_ran_after_turn"]]


@pytest.fixture
def generated_result(monkeypatch, primary_cell, run_plan):
    monkeypatch.setattr(conversation, "generate_user_turn", lambda **kwargs: "Tell me more.")
    monkeypatch.setattr(conversation, "call_model", lambda *args, **kwargs: ModelGeneration("An assistant reply."))
    monkeypatch.setattr(
        conversation,
        "run_probe",
        lambda **kwargs: BehavioralProbeResult(kwargs["phase_name"], belief_core=19, belief_A1=23, belief_A2=71, confidence_belief_A1=31, confidence_belief_A2=83),
    )

    def generate(order):
        return conversation.run_one_vignette(primary_cell, replace(run_plan, aux_order_condition=order))

    return generate


@pytest.mark.parametrize("order,beliefs,confidences", [("original", (23, 71), (31, 83)), ("reversed", (71, 23), (83, 31))])
def test_decoding_labels_consume_core_csv_for_both_auxiliary_orders(generated_result, tmp_path, order, beliefs, confidences):
    result = generated_result(order)
    csv_path = tmp_path / "results.csv"
    jsonl_path = tmp_path / "dialogues.jsonl"
    save_results_append([result], csv_path, jsonl_path, write_header=True)

    labels = make_labels.build_labels(pd.read_csv(csv_path))

    assert labels["phase"].tolist() == [probe.phase for probe in result.probe_outputs]
    assert len(labels) < len(result.result_rows)
    assert set(labels["belief_core"]) == {19}
    assert set(labels["belief_A1_canonical"]) == {beliefs[0]}
    assert set(labels["belief_A2_canonical"]) == {beliefs[1]}
    assert set(labels["confidence_belief_A1_canonical"]) == {confidences[0]}
    assert set(labels["confidence_belief_A2_canonical"]) == {confidences[1]}
    record = json.loads(jsonl_path.read_text(encoding="utf-8"))
    assert extract_activations.build_messages(record)[-1]["content"] == result.dialogue_history[-1].text


def test_endorsement_probes_use_core_checkpoint_prefixes_and_canonical_claims(generated_result, monkeypatch, tmp_path):
    result = generated_result("reversed")
    record = build_conversation_record(result)
    messages = extract_activations.build_messages(record)
    captured = []

    def fake_batch(model, prefix, claims, layers, option_ids, batch_size):
        captured.append((prefix, claims))
        return [{"claim": "core", "mapping": "A_endorse", "format": "AB", "E": 1.0}], torch.zeros(3, 2, len(layers), 4)

    monkeypatch.setattr(endorsement_probe, "probe_prefix_batch", fake_batch)
    endorsement_probe.process(object(), record, tmp_path, [0], [], 4)

    checkpoints = extract_activations.checkpoint_turns()
    assert len(captured) == len(result.probe_outputs)
    for (prefix, claims), checkpoint in zip(captured, checkpoints, strict=True):
        assert prefix == messages[: 1 + 2 * checkpoint["global_turn_index"]]
        assert prefix[-1]["role"] == "assistant"
        assert claims == {
            "core": result.primary_cell.vignette.core_claim,
            **{auxiliary.id: auxiliary.text for auxiliary in result.primary_cell.vignette.auxiliaries},
        }
    saved = torch.load(tmp_path / f"{record['cell_id']}__rep{record['replicate_index']}.pt")
    assert saved["phases"] == [probe.phase for probe in result.probe_outputs]
    assert tuple(saved["hidden"].shape) == (len(checkpoints), 3, 2, 1, 4)


def test_relocated_figures_read_repo_data_and_keep_figure_output_local():
    assert make_figures.ROOT == REPO_ROOT
    assert make_figures.EARLY == REPO_ROOT / "output_full_crossed" / "paper_v3"
    assert make_figures.RUN == REPO_ROOT / "runs" / "politics03-llama8b-full"
    assert make_figures.FIGURES == Path(make_figures.__file__).resolve().parent / "figures"

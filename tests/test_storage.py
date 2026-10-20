import csv
import json

from belief_revision.models import (
    BehavioralProbeResult,
    ConversationResult,
    DialogueTurn,
)
from belief_revision.storage import (
    build_conversation_record,
    load_vignettes,
    parse_vignette,
    save_results_append,
)


def raw_vignette():
    return {
        "id": "v1",
        "domain": "health",
        "core_claim": "claim",
        "auxiliaries": [
            {"id": "A1", "type": "causal", "text": "one"},
            {"id": "A2", "type": "social", "text": "two"},
        ],
        "core_evidence_disconfirming": [{"id": "d", "text": "d", "strength": "strong"}],
        "core_evidence_supporting": [{"id": "s", "text": "s", "strength": "weak"}],
        "core_evidence_ambiguous": [{"id": "a", "text": "a", "strength": "weak"}],
        "auxiliary_evidence": [
            {"id": "x", "text": "x", "strength": "weak", "targets": ["A1"]}
        ],
    }


def make_result(primary_cell, run_plan):
    return ConversationResult(
        primary_cell=primary_cell,
        run_plan=run_plan,
        dialogue_history=[DialogueTurn("assistant", "intro", 1, 1, "hello")],
        probe_outputs=[BehavioralProbeResult("intro", belief_core=50)],
        result_rows=[{"global_turn_index": 1, "value": 2}],
        aux_mapping={"A1": "A1"},
    )


def test_parse_load_record_and_save(tmp_path, primary_cell, run_plan):
    parsed = parse_vignette(raw_vignette())
    assert parsed.auxiliaries[0].id == "A1"
    source = tmp_path / "vignettes.json"
    source.write_text(json.dumps({"vignettes": [raw_vignette()]}))
    assert load_vignettes(source)[0].id == "v1"

    result = make_result(primary_cell, run_plan)
    record = build_conversation_record(result)
    assert record["probe_outputs"][0]["belief_core"] == 50

    csv_path = tmp_path / "out.csv"
    jsonl_path = tmp_path / "out.jsonl"
    save_results_append([], csv_path, jsonl_path, True)
    assert not csv_path.exists()
    save_results_append([result], csv_path, jsonl_path, True)
    save_results_append([result], csv_path, jsonl_path, False)
    assert len(jsonl_path.read_text().splitlines()) == 2
    with csv_path.open() as handle:
        assert len(list(csv.DictReader(handle))) == 2

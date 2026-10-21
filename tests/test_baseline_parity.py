"""Structural parity checks for the first completed experiment baseline."""

import csv
import json
import hashlib
from collections import Counter
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "baseline_demo_5"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"
CONVERSATIONS_PATH = FIXTURE_DIR / "conversations.jsonl"
RESULTS_PATH = FIXTURE_DIR / "results.csv"


def load_manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def load_conversations() -> list[dict[str, object]]:
    return [json.loads(line) for line in CONVERSATIONS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_rows() -> list[dict[str, str]]:
    with RESULTS_PATH.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_baseline_files_match_manifest() -> None:
    manifest = load_manifest()
    source_files = manifest["source_files"]

    assert sha256(CONVERSATIONS_PATH) == source_files["conversations.jsonl"]["sha256"]
    assert sha256(RESULTS_PATH) == source_files["results.csv"]["sha256"]


def test_baseline_conversation_structure() -> None:
    manifest = load_manifest()
    conversations = load_conversations()

    assert len(conversations) == manifest["source_files"]["conversations.jsonl"]["completed_conversations"]
    assert [record["replicate_index"] for record in conversations] == manifest["replicate_indices"]
    assert [record["seed"] for record in conversations] == manifest["seeds"]

    expected_turns = list(range(1, manifest["turns_per_conversation"] + 1))
    expected_phase_counts = manifest["phase_turn_counts"]
    expected_phase_sequence = [phase for phase, count in expected_phase_counts.items() for _ in range(count)]

    for record in conversations:
        assert record["cell_id"] == manifest["cell_id"]
        assert len(record["rows"]) == manifest["turns_per_conversation"]
        assert len(record["dialogue_history"]) == manifest["messages_per_conversation"]
        assert [probe["phase"] for probe in record["probe_outputs"]] == manifest["probe_phases"]

        history = record["dialogue_history"]
        user_turns = [turn for turn in history if turn["speaker"] == "user"]
        assistant_turns = [turn for turn in history if turn["speaker"] == "assistant"]

        assert [turn["speaker"] for turn in history] == [speaker for _ in expected_turns for speaker in ("user", "assistant")]
        assert [turn["global_turn_index"] for turn in user_turns] == expected_turns
        assert [turn["global_turn_index"] for turn in assistant_turns] == expected_turns
        assert [turn["phase"] for turn in user_turns] == expected_phase_sequence
        assert [turn["phase"] for turn in assistant_turns] == expected_phase_sequence
        assert Counter(turn["phase"] for turn in user_turns) == expected_phase_counts
        assert all(user["phase"] == assistant["phase"] and user["global_turn_index"] == assistant["global_turn_index"] for user, assistant in zip(user_turns, assistant_turns, strict=True))


def test_csv_and_jsonl_represent_the_same_completed_turns() -> None:
    manifest = load_manifest()
    conversations = load_conversations()
    rows = load_rows()

    assert len(rows) == manifest["source_files"]["results.csv"]["turn_rows"]
    assert len(rows[0]) == manifest["source_files"]["results.csv"]["columns"]

    rows_by_replicate: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_replicate.setdefault(int(row["replicate_index"]), []).append(row)

    for record in conversations:
        replicate_index = record["replicate_index"]
        csv_rows = rows_by_replicate[replicate_index]
        json_rows = record["rows"]

        assert len(csv_rows) == len(json_rows)
        assert [int(row["global_turn_index"]) for row in csv_rows] == [row["global_turn_index"] for row in json_rows]
        assert [row["phase"] for row in csv_rows] == [row["phase"] for row in json_rows]
        assert [row["evidence_id"] or None for row in csv_rows] == [row["evidence_id"] for row in json_rows]
        assert sum(row["probe_ran_after_turn"] == "True" for row in csv_rows) == len(record["probe_outputs"])

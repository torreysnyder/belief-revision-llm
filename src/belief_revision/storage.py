"""Serialize and store completed experiment conversations."""

import csv
import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import Auxiliary, ConversationResult, EvidenceItem, Vignette

WRITE_LOCK = threading.Lock()


def parse_vignette(raw: dict[str, Any]) -> Vignette:
    """Convert one raw vignette record into structured experiment data."""

    return Vignette(
        id=raw["id"],
        domain=raw["domain"],
        core_claim=raw["core_claim"],
        auxiliaries=[Auxiliary(**item) for item in raw["auxiliaries"]],
        core_evidence_disconfirming=[
            EvidenceItem(**item) for item in raw["core_evidence_disconfirming"]
        ],
        core_evidence_supporting=[
            EvidenceItem(**item) for item in raw["core_evidence_supporting"]
        ],
        core_evidence_ambiguous=[
            EvidenceItem(**item) for item in raw["core_evidence_ambiguous"]
        ],
        auxiliary_evidence=[EvidenceItem(**item) for item in raw["auxiliary_evidence"]],
    )


def load_vignettes(
    path: str | Path = "vignettes_revised.json",
) -> list[Vignette]:
    """Load the experiment vignettes from JSON."""

    with Path(path).open("r", encoding="utf-8") as file:
        raw_vignettes = json.load(file)["vignettes"]

    return [parse_vignette(raw) for raw in raw_vignettes]


def build_conversation_record(
    result: ConversationResult,
) -> dict[str, object]:
    """Convert a conversation result into the legacy JSONL record shape."""

    return {
        "cell_id": result.primary_cell.cell_id,
        "replicate_index": result.run_plan.replicate_index,
        "seed": result.run_plan.seed,
        "vignette_id": result.primary_cell.vignette.id,
        "style_profile": asdict(result.primary_cell.style_profile),
        "primary_cell": asdict(result.primary_cell),
        "run_plan": asdict(result.run_plan),
        "aux_mapping": result.aux_mapping,
        "dialogue_history": [asdict(turn) for turn in result.dialogue_history],
        "probe_outputs": [asdict(probe) for probe in result.probe_outputs],
        "rows": result.result_rows,
    }


def save_results_append(
    results: list[ConversationResult],
    csv_path: str | Path,
    jsonl_path: str | Path,
    write_header: bool,
) -> None:
    """Append completed conversations to JSONL and turn rows to CSV."""

    if not results:
        return

    csv_path = Path(csv_path)
    jsonl_path = Path(jsonl_path)

    records = [build_conversation_record(result) for result in results]

    with WRITE_LOCK:
        with jsonl_path.open(
            "a",
            encoding="utf-8",
        ) as jsonl_file:
            for record in records:
                jsonl_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        fieldnames = list(results[0].result_rows[0].keys())

        mode = "w" if write_header else "a"

        with csv_path.open(
            mode,
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
            )

            if write_header:
                writer.writeheader()

            for result in results:
                writer.writerows(result.result_rows)

"""Pluggable destinations for completed experiment results."""

from ..core.models import ConversationResult
from .local import save_results_append
from pathlib import Path
from typing import Any, Protocol


class ExperimentBackend(Protocol):
    """Lifecycle implemented by local, Supabase, or researcher-owned storage."""

    def start_run(
        self,
        *,
        target_model: str,
        user_sim_model: str,
        evaluator_model: str,
        config: dict[str, Any],
    ) -> str: ...

    def save_conversations(
        self,
        run_id: str,
        results: list[ConversationResult],
    ) -> None: ...

    def complete_run(
        self,
        run_id: str,
        *,
        completed_conversations: int,
        expected_conversations: int,
    ) -> None: ...

    def fail_run(self, run_id: str, error: Exception) -> None: ...


class LocalFileBackend:
    """Store conversations in the project's legacy CSV and JSONL formats."""

    def __init__(self, csv_path: str | Path, jsonl_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.jsonl_path = Path(jsonl_path)
        self.write_header = True

    def start_run(self, **_: Any) -> str:
        """Prepare clean local outputs for a new run."""

        self.jsonl_path.write_text("", encoding="utf-8")
        if self.csv_path.exists():
            self.csv_path.unlink()
        self.write_header = True
        return "local"

    def save_conversations(
        self,
        run_id: str,
        results: list[ConversationResult],
    ) -> None:
        """Append one completed cell and manage the CSV header."""

        del run_id
        save_results_append(
            results,
            csv_path=self.csv_path,
            jsonl_path=self.jsonl_path,
            write_header=self.write_header,
        )
        if results:
            self.write_header = False

    def complete_run(self, run_id: str, **_: Any) -> None:
        """Local files need no finalization beyond completed writes."""

        del run_id

    def fail_run(self, run_id: str, error: Exception) -> None:
        """Keep partial local output available when a run fails."""

        del run_id, error

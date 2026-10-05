"""Optional durable experiment reporting through Supabase."""

import os
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4, uuid5

from dotenv import load_dotenv
from supabase import Client, create_client

from .models import ConversationResult


def utc_now() -> str:
    """Return the current UTC time in an API-safe format."""

    return datetime.now(UTC).isoformat()


def create_reporting_client() -> Client:
    """Create the server-side Supabase client used by reporting tools."""

    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    secret_key = os.getenv("SUPABASE_SECRET_KEY")

    if not url or not secret_key:
        raise RuntimeError(
            "Supabase access requires SUPABASE_URL and "
            "SUPABASE_SECRET_KEY in .env."
        )

    return create_client(url, secret_key)


class SupabaseReporter:
    """Upload completed experiment records when reporting is enabled."""

    def __init__(self, client: Client) -> None:
        self.client = client

    @classmethod
    def from_environment(cls) -> "SupabaseReporter":
        """Build a reporter from the local server-side credentials."""

        return cls(create_reporting_client())

    def start_run(
        self,
        *,
        target_model: str,
        user_sim_model: str,
        evaluator_model: str,
        config: dict[str, Any],
    ) -> str:
        """Create one durable record for the experiment invocation."""

        run_id = str(uuid4())
        self.client.table("experiment_runs").insert(
            {
                "id": run_id,
                "status": "running",
                "target_model": target_model,
                "user_sim_model": user_sim_model,
                "evaluator_model": evaluator_model,
                "config": config,
            }
        ).execute()
        return run_id

    def save_conversations(
        self,
        run_id: str,
        results: list[ConversationResult],
    ) -> None:
        """Upsert completed conversations and their child records."""

        for result in results:
            conversation_id = str(
                uuid5(
                    UUID(run_id),
                    (
                        f"{result.primary_cell.cell_id}:"
                        f"{result.run_plan.replicate_index}"
                    ),
                )
            )
            result_rows = {
                int(row["global_turn_index"]): row
                for row in result.result_rows
            }

            conversation = {
                "id": conversation_id,
                "run_id": run_id,
                "cell_id": result.primary_cell.cell_id,
                "replicate_index": result.run_plan.replicate_index,
                "seed": result.run_plan.seed,
                "vignette_id": result.primary_cell.vignette.id,
                "status": "running",
                "style_profile": asdict(
                    result.primary_cell.style_profile
                ),
                "primary_cell": asdict(result.primary_cell),
                "run_plan": asdict(result.run_plan),
                "aux_mapping": result.aux_mapping,
                "metadata": {
                    "dialogue_turn_count": len(result.dialogue_history),
                    "probe_count": len(result.probe_outputs),
                },
            }
            self.client.table("conversations").upsert(
                conversation,
                on_conflict="id",
            ).execute()

            turns = []
            for turn in result.dialogue_history:
                payload = asdict(turn)
                if turn.speaker == "assistant":
                    payload["result_row"] = result_rows.get(
                        turn.global_turn_index
                    )

                turns.append(
                    {
                        "conversation_id": conversation_id,
                        "global_turn_index": turn.global_turn_index,
                        "speaker": turn.speaker,
                        "phase": turn.phase,
                        "turn_in_phase": turn.turn_in_phase,
                        "text": turn.text,
                        "payload": payload,
                    }
                )

            if turns:
                self.client.table("conversation_turns").upsert(
                    turns,
                    on_conflict=(
                        "conversation_id,global_turn_index,speaker"
                    ),
                ).execute()

            probes = [
                {
                    "conversation_id": conversation_id,
                    "probe_index": index,
                    "phase": probe.phase,
                    "payload": asdict(probe),
                }
                for index, probe in enumerate(result.probe_outputs)
            ]
            if probes:
                self.client.table("behavioral_probes").upsert(
                    probes,
                    on_conflict="conversation_id,probe_index",
                ).execute()

            self.client.table("conversations").update(
                {
                    "status": "completed",
                    "completed_at": utc_now(),
                }
            ).eq("id", conversation_id).execute()

    def complete_run(
        self,
        run_id: str,
        *,
        completed_conversations: int,
        expected_conversations: int,
    ) -> None:
        """Mark a reporting run complete with final counts."""

        self.client.table("experiment_runs").update(
            {
                "status": "completed",
                "completed_at": utc_now(),
                "metadata": {
                    "completed_conversations": completed_conversations,
                    "expected_conversations": expected_conversations,
                },
            }
        ).eq("id", run_id).execute()

    def fail_run(self, run_id: str, error: Exception) -> None:
        """Record a reporting or experiment failure before re-raising it."""

        self.client.table("experiment_runs").update(
            {
                "status": "failed",
                "completed_at": utc_now(),
                "error": str(error),
            }
        ).eq("id", run_id).execute()

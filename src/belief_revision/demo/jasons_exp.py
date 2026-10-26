"""Walkthrough: plug a researcher-owned experiment into the shared system.

This file is intentionally more verbose than a normal extension. It shows every
available conversation hook, activation capture, artifact reporting, extension
registration, and read-only Supabase access in one place. Delete hooks you do
not need when using this as a starting point for a real experiment.
"""

import json
from belief_revision.core.models import ActivationCaptureRequest, ExperimentArtifact
from belief_revision.extensions.extensions import (
    ActivationCaptureContext,
    CompletedConversationContext,
    CompletedPhaseContext,
    ConversationContext,
    ConversationExtension,
    GenerationContext,
    PhaseContext,
    TurnContext,
)
from belief_revision.persistence import ResearchData
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class JasonsExperiment(ConversationExtension):
    """Example observer that records lifecycle events and selected activations."""

    output_dir: Path = Path(".tmp/jasons-exp")
    capture_layers: tuple[int, ...] | None = (0, 12, 23)
    events: list[dict[str, object]] = field(default_factory=list, init=False)
    generated_artifacts: list[ExperimentArtifact] = field(default_factory=list, init=False)

    # Setup hook: called once after the run plan is fixed.
    def before_conversation(self, context: ConversationContext) -> None:
        self.events.append(
            {
                "event": "before_conversation",
                "cell_id": context.primary_cell.cell_id,
                "replicate": context.run_plan.replicate_index,
                "vignette": context.vignette.id,
            }
        )

    # Phase hook: inspect which phase and evidence the core selected.
    def before_phase(self, context: PhaseContext) -> None:
        self.events.append(
            {
                "event": "before_phase",
                "phase": context.phase_name,
                "evidence_id": context.evidence_item.id if context.evidence_item else None,
                "auxiliary_gate_passed": context.auxiliary_gate_passed,
            }
        )

    # Turn hook: inspect state immediately before the simulated user speaks.
    def before_turn(self, context: TurnContext) -> None:
        self.events.append(
            {
                "event": "before_turn",
                "turn": context.global_turn_index,
                "belief_core": context.user_state.belief_core,
                "inject_evidence": context.inject_evidence,
            }
        )

    # User hook: inspect the completed simulated-user message.
    def after_user_turn(self, context: TurnContext) -> None:
        self.events.append(
            {
                "event": "after_user_turn",
                "turn": context.global_turn_index,
                "user_text": context.user_text,
            }
        )

    # Generation hook: inspect the exact structured inputs before model rendering.
    def before_assistant_generation(self, context: GenerationContext) -> None:
        self.events.append(
            {
                "event": "before_assistant_generation",
                "turn": context.turn.global_turn_index,
                "model": context.model,
                "message_count": len(context.messages),
                "temperature": context.temperature,
                "max_tokens": context.max_tokens,
            }
        )

    # Capture interface: return no requests to skip capture, selected zero-based
    # layers as shown here, or layers=None to capture every decoder layer.
    def activation_capture_requests(
        self,
        context: GenerationContext,
    ) -> tuple[ActivationCaptureRequest, ...]:
        del context
        return (
            ActivationCaptureRequest(
                request_id="jasons_residual_stream",
                layers=self.capture_layers,
                location="residual_post",
                token_position="final_prompt_token",
            ),
        )

    # Generation result hook: inspect the assistant text after NDIF returns.
    def after_assistant_generation(self, context: GenerationContext) -> None:
        self.events.append(
            {
                "event": "after_assistant_generation",
                "turn": context.turn.global_turn_index,
                "assistant_text": context.response,
            }
        )

    # Activation hook: tensors are available here; this demo records metadata only.
    def after_activation_capture(self, context: ActivationCaptureContext) -> None:
        self.events.append(
            {
                "event": "after_activation_capture",
                "turn": context.generation.turn.global_turn_index,
                "request_id": context.request.request_id,
                "layers": [capture.layer for capture in context.activations],
                "shapes": [capture.shape for capture in context.activations],
            }
        )

    # Completed-turn hook: inspect the updated state and full dialogue so far.
    def after_turn(self, context: TurnContext) -> None:
        self.events.append(
            {
                "event": "after_turn",
                "turn": context.global_turn_index,
                "dialogue_messages": len(context.dialogue_history),
                "active_auxiliaries": list(context.user_state.active_auxiliaries),
            }
        )

    # Completed-phase hook: the optional behavioral probe is available here.
    def after_phase(self, context: CompletedPhaseContext) -> None:
        self.events.append(
            {
                "event": "after_phase",
                "phase": context.phase.phase_name,
                "probe_ran": context.latest_probe is not None,
            }
        )

    # Final hook: write one researcher-owned artifact after a successful conversation.
    def after_conversation(self, context: CompletedConversationContext) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_plan = context.conversation.run_plan
        filename = f"{context.conversation.primary_cell.cell_id}__replicate-{run_plan.replicate_index}.json"
        output_path = self.output_dir / filename
        output_path.write_text(json.dumps(self.events, indent=2, default=str), encoding="utf-8")
        self.generated_artifacts.append(
            ExperimentArtifact(
                artifact_type="jasons_walkthrough_events",
                local_path=output_path,
                metadata={"event_count": len(self.events)},
            )
        )

    # Artifact interface: returned files remain local by default and upload to
    # Supabase Storage, with searchable metadata, when --reporting true is used.
    def artifacts(self) -> tuple[ExperimentArtifact, ...]:
        return tuple(self.generated_artifacts)


def inspect_reported_data() -> None:
    """Example notebook/script access to data uploaded through Supabase."""

    data = ResearchData.from_environment()
    print("Tables:", data.list_tables())
    print("Named queries:", data.list_queries())

    runs, conversations = data.download_all("experiment_runs", "conversations")
    print(f"Downloaded {len(runs)} runs and {len(conversations)} conversations")

    # Equality filters use the ordinary Supabase Data API and do not require a
    # direct Postgres URL.
    if conversations:
        conversation_id = conversations[0]["id"]
        turns = data.download(
            "conversation_turns",
            filters={"conversation_id": conversation_id},
        )
        print(f"First conversation has {len(turns)} turns")

    # Named SQL is inspectable before execution. Running it additionally needs
    # SUPABASE_DB_URL because it uses a read-only Postgres transaction.
    print(data.show_query("run_summary"))


# This walkthrough is registered as ``jasons-exp``. Run it with:
#
#     uv run python scripts/run_experiment.py \
#       --single-run \
#       --extension jasons-exp \
#       --config configs/default.yaml \
#       --reporting true

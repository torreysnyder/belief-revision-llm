"""Thin adapter from live conversations to Torrey's CAA activation pipeline.

This adapter captures residual-stream activations during the existing assistant
generation.  It writes raw per-conversation tensors and metadata for Torrey's
pairing pipeline; it does not construct controls, differences, or CAA vectors.
"""

from ...core.models import (
    ActivationArtifactRecord,
    ActivationCaptureRequest,
    ExperimentArtifact,
)
from ..extensions import (
    ActivationCaptureContext,
    CompletedConversationContext,
    ConversationExtension,
    GenerationContext,
)
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

DEFAULT_OUTPUT_DIR = Path("activation_steering_pipeline/activations/live_captures")


def _save_payload(path: Path, payload: dict[str, object]) -> None:
    """Write one PyTorch artifact without importing Torch at CLI startup."""

    import torch

    torch.save(payload, path)


@dataclass
class TorreyCAAExtension(ConversationExtension):
    """Capture every assistant turn for later controlled-pair construction."""

    output_dir: Path = DEFAULT_OUTPUT_DIR
    layers: tuple[int, ...] | None = None
    phases: frozenset[str] | None = None
    turns: frozenset[int] | None = None
    _records: list[dict[str, object]] = field(default_factory=list, init=False)
    _artifacts: list[ExperimentArtifact] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.phases is not None and not self.phases:
            raise ValueError("phases must not be empty")
        if self.turns is not None:
            if not self.turns:
                raise ValueError("turns must not be empty")
            if any(turn < 1 for turn in self.turns):
                raise ValueError("turns must contain only positive integers")

    def activation_capture_requests(
        self,
        context: GenerationContext,
    ) -> tuple[ActivationCaptureRequest, ...]:
        """Request Torrey's prespecified residual-stream activation location."""

        phase_name = context.turn.phase.phase_name
        turn_in_phase = context.turn.turn_in_phase
        if self.phases is not None and phase_name not in self.phases:
            return ()
        if self.turns is not None and turn_in_phase not in self.turns:
            return ()

        return (
            ActivationCaptureRequest(
                request_id="torrey_caa",
                layers=self.layers,
                location="residual_post",
                token_position="final_prompt_token",
            ),
        )

    def after_activation_capture(self, context: ActivationCaptureContext) -> None:
        """Retain a detached CPU tensor and the metadata needed for pairing."""

        captures = tuple(sorted(context.activations, key=lambda capture: capture.layer))
        self._records.append(
            {
                "phase": context.generation.turn.phase.phase_name,
                "turn_in_phase": context.generation.turn.turn_in_phase,
                "global_turn_index": context.generation.turn.global_turn_index,
                "model_name": context.generation.model,
                "layers": [capture.layer for capture in captures],
                "location": context.request.location,
                "token_position": context.request.token_position,
                "token_indices": [capture.token_index for capture in captures],
                "sequence_lengths": [capture.sequence_length for capture in captures],
                "activation": [capture.tensor.detach().cpu() for capture in captures],
            }
        )

    def after_conversation(self, context: CompletedConversationContext) -> None:
        """Write one collision-safe artifact for the completed conversation."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        cell_id = context.conversation.primary_cell.cell_id
        run_plan = context.conversation.run_plan
        path = self.output_dir / (f"{cell_id}__replicate_{run_plan.replicate_index}__{uuid4().hex}.pt")
        payload = {
            "format_version": 1,
            "artifact_type": "live_conversation_activations",
            "cell_id": cell_id,
            "vignette_id": context.conversation.vignette.id,
            "replicate_index": run_plan.replicate_index,
            "seed": run_plan.seed,
            "activation_location": "residual_post",
            "token_position": "final_prompt_token",
            "records": self._records,
        }
        _save_payload(path, payload)
        activation_records = tuple(
            ActivationArtifactRecord(
                global_turn_index=int(record["global_turn_index"]),
                phase=str(record["phase"]),
                turn_in_phase=int(record["turn_in_phase"]),
                model_name=str(record["model_name"]),
                location=str(record["location"]),
                token_position=str(record["token_position"]),
                layers=tuple(record["layers"]),
                token_indices=tuple(record["token_indices"]),
                sequence_lengths=tuple(record["sequence_lengths"]),
                tensor_shapes=tuple(tuple(tensor.shape) for tensor in record["activation"]),
                tensor_dtypes=tuple(str(tensor.dtype) for tensor in record["activation"]),
            )
            for record in self._records
        )
        self._artifacts.append(
            ExperimentArtifact(
                artifact_type="live_conversation_activations",
                local_path=path,
                metadata={
                    "format_version": 1,
                    "extension": "torrey-caa",
                    "activation_location": "residual_post",
                    "token_position": "final_prompt_token",
                    "record_count": len(self._records),
                },
                activation_records=activation_records,
            )
        )

    def artifacts(self) -> tuple[ExperimentArtifact, ...]:
        """Expose completed local captures to configured reporting backends."""

        return tuple(self._artifacts)

"""Observational lifecycle hooks for conversation experiments.

Extensions receive frozen context wrappers and cannot alter the conversation
flow through return values. Hooks are synchronous, run in registration order,
and deliberately let exceptions propagate to the caller.
"""

from ..core.models import (
    ActivationCapture,
    ActivationCaptureRequest,
    BehavioralProbeResult,
    ConversationResult,
    DialogueTurn,
    EvidenceItem,
    ExperimentArtifact,
    ModelGeneration,
    PrimaryCell,
    RunPlan,
    UserState,
    Vignette,
)
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class ConversationContext:
    """Detached setup snapshot shared by one conversation's events."""

    primary_cell: PrimaryCell
    run_plan: RunPlan
    vignette: Vignette
    aux_mapping: Mapping[str, object]


@dataclass(frozen=True)
class PhaseContext:
    """Detached snapshot captured at the start of a conversation phase."""

    conversation: ConversationContext
    phase_name: str
    phase: Mapping[str, object]
    evidence_item: EvidenceItem | None
    auxiliary_gate_passed: bool | None


@dataclass(frozen=True)
class TurnContext:
    """Detached snapshot of one user/assistant exchange within a phase."""

    phase: PhaseContext
    turn_in_phase: int
    global_turn_index: int
    inject_evidence: bool
    current_evidence: EvidenceItem | None
    evidence_role: str | None
    user_state: UserState
    user_text: str | None
    dialogue_history: tuple[DialogueTurn, ...]


@dataclass(frozen=True)
class GenerationContext:
    """Exact inputs to ``call_model`` plus its response when available.

    These inputs are not necessarily the final prompt rendered by the remote
    model backend.
    """

    turn: TurnContext
    model: str
    messages: tuple[Mapping[str, str], ...]
    temperature: float
    max_tokens: int | None
    response: str | None = None


@dataclass(frozen=True)
class ActivationCaptureContext:
    """Captured activations paired with their request and generation."""

    generation: GenerationContext
    request: ActivationCaptureRequest
    activations: tuple[ActivationCapture, ...]


@dataclass(frozen=True)
class CompletedConversationContext:
    """Detached completed-result snapshot paired with conversation setup."""

    conversation: ConversationContext
    result: ConversationResult


@dataclass(frozen=True)
class CompletedPhaseContext:
    """Detached phase snapshot captured after its optional probe completes."""

    phase: PhaseContext
    user_state: UserState
    dialogue_history: tuple[DialogueTurn, ...]
    latest_probe: BehavioralProbeResult | None


class ConversationExtension:
    """No-op base class for observing conversation lifecycle events."""

    def before_conversation(self, context: ConversationContext) -> None:
        """Observe a conversation after setup and before its first phase."""

    def before_phase(self, context: PhaseContext) -> None:
        """Observe a phase after its evidence plan has been resolved."""

    def before_turn(self, context: TurnContext) -> None:
        """Observe a turn before the user message has been generated."""

    def after_user_turn(self, context: TurnContext) -> None:
        """Observe a turn after the user message has been generated."""

    def before_assistant_generation(self, context: GenerationContext) -> None:
        """Observe the exact inputs about to be sent to the assistant."""

    def activation_capture_requests(
        self,
        context: GenerationContext,
    ) -> tuple[ActivationCaptureRequest, ...]:
        """Request activations from the upcoming assistant generation."""

        return ()

    def after_assistant_generation(self, context: GenerationContext) -> None:
        """Observe the assistant response immediately after generation."""

    def after_activation_capture(self, context: ActivationCaptureContext) -> None:
        """Observe activations captured from an assistant generation."""

    def after_turn(self, context: TurnContext) -> None:
        """Observe a completed user/assistant exchange."""

    def after_phase(self, context: CompletedPhaseContext) -> None:
        """Observe a phase after its turns and optional probe complete."""

    def after_conversation(self, context: CompletedConversationContext) -> None:
        """Observe the completed conversation result."""

    def artifacts(self) -> tuple[ExperimentArtifact, ...]:
        """Return artifacts produced after the conversation completed."""

        return ()


@dataclass(frozen=True)
class _ActivationCaptureRoute:
    """Private routing metadata for one extension capture request."""

    extension: ConversationExtension
    public_request: ActivationCaptureRequest
    backend_request: ActivationCaptureRequest


@dataclass(frozen=True)
class ActivationCapturePlan:
    """Capture requests prepared for one model generation."""

    _routes: tuple[_ActivationCaptureRoute, ...] = ()

    @property
    def backend_requests(self) -> tuple[ActivationCaptureRequest, ...]:
        """Requests with IDs made unique for the model backend."""

        return tuple(route.backend_request for route in self._routes)


def collect_activation_capture_plan(
    extensions: Sequence[ConversationExtension],
    context: GenerationContext,
) -> ActivationCapturePlan:
    """Collect extension requests and assign collision-safe backend IDs."""

    routes: list[_ActivationCaptureRoute] = []
    for extension_index, extension in enumerate(extensions):
        requests = extension.activation_capture_requests(context)
        public_request_ids: set[str] = set()
        for request_index, request in enumerate(requests):
            if request.request_id in public_request_ids:
                raise ValueError(f"activation capture request IDs must be unique within one extension event: {request.request_id!r}")
            public_request_ids.add(request.request_id)

            backend_request = replace(
                request,
                request_id=f"{extension_index}:{request_index}",
            )
            routes.append(
                _ActivationCaptureRoute(
                    extension=extension,
                    public_request=request,
                    backend_request=backend_request,
                )
            )

    return ActivationCapturePlan(tuple(routes))


def route_activation_captures(
    plan: ActivationCapturePlan,
    generation_context: GenerationContext,
    generation: ModelGeneration,
) -> None:
    """Validate backend captures and return them to requesting extensions."""

    routes_by_id = {route.backend_request.request_id: route for route in plan._routes}
    captures_by_id: dict[str, list[ActivationCapture]] = {request_id: [] for request_id in routes_by_id}
    for capture in generation.activations:
        if capture.request_id not in routes_by_id:
            raise ValueError(f"model returned an unexpected activation capture request ID: {capture.request_id!r}")
        captures_by_id[capture.request_id].append(capture)

    for route in plan._routes:
        backend_request_id = route.backend_request.request_id
        captures = captures_by_id[backend_request_id]
        if not captures:
            raise ValueError(f"model returned no activations for capture request: {route.public_request.request_id!r}")

        if route.public_request.layers is not None:
            expected_layers = sorted(route.public_request.layers)
            captured_layers = sorted(capture.layer for capture in captures)
            if captured_layers != expected_layers:
                raise ValueError(f"model returned activation layers that do not match capture request {route.public_request.request_id!r}: expected {expected_layers}, got {captured_layers}")

        public_captures = tuple(replace(capture, request_id=route.public_request.request_id) for capture in captures)
        route.extension.after_activation_capture(
            ActivationCaptureContext(
                generation=generation_context,
                request=route.public_request,
                activations=public_captures,
            )
        )


ExtensionFactory = Callable[[], ConversationExtension]

HookName = Literal[
    "before_conversation",
    "before_phase",
    "before_turn",
    "after_user_turn",
    "before_assistant_generation",
    "after_assistant_generation",
    "after_activation_capture",
    "after_turn",
    "after_phase",
    "after_conversation",
]


def dispatch_extensions(
    extensions: Sequence[ConversationExtension],
    hook_name: HookName,
    context: object,
) -> None:
    """Invoke one hook on each extension in registration order."""

    for extension in extensions:
        hook = getattr(extension, hook_name)
        hook(context)

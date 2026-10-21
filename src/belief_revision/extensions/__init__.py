"""Research extension contracts and registration."""

from .extension_registry import (
    EXTENSION_REGISTRY,
    register_extension,
    resolve_extension_factories,
)
from .extensions import (
    ActivationCaptureContext,
    ActivationCapturePlan,
    CompletedConversationContext,
    CompletedPhaseContext,
    ConversationContext,
    ConversationExtension,
    ExtensionFactory,
    GenerationContext,
    PhaseContext,
    TurnContext,
    collect_activation_capture_plan,
    dispatch_extensions,
    route_activation_captures,
)

__all__ = [
    "EXTENSION_REGISTRY",
    "ActivationCaptureContext",
    "ActivationCapturePlan",
    "CompletedConversationContext",
    "CompletedPhaseContext",
    "ConversationContext",
    "ConversationExtension",
    "ExtensionFactory",
    "GenerationContext",
    "PhaseContext",
    "TurnContext",
    "collect_activation_capture_plan",
    "dispatch_extensions",
    "register_extension",
    "resolve_extension_factories",
    "route_activation_captures",
]

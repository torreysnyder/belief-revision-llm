from .models import EvidenceItem, UserState, Vignette

def should_inject_auxiliary_evidence(
    user_state: UserState,
    planned_auxiliary_evidence: EvidenceItem | None,
) -> bool:
    """Return whether planned auxiliary evidence targets an active auxiliary."""

    if planned_auxiliary_evidence is None:
        return False

    active_auxiliaries = set(
        user_state.active_auxiliaries
    )

    evidence_targets = set(
        planned_auxiliary_evidence.targets or []
    )

    return bool(
        active_auxiliaries.intersection(evidence_targets)
    )
import re
import random

from .config import BELIEF_ANCHOR_MAP
from .models import EvidenceItem, UserState, Vignette

def init_user_state(
    initial_affect: str,
    trust_in_institutions: str,
    conversation_goal: str,
    belief_anchor_level: str,
) -> UserState:
    """Create the simulated user's initial belief state."""

    anchor = BELIEF_ANCHOR_MAP[belief_anchor_level]

    return UserState(
        belief_core=anchor["belief_core"],
        belief_A1=55,
        belief_A2=50,
        confidence_core=anchor["confidence_core"],
        affect=initial_affect,
        initial_affect=initial_affect,
        trust_in_institutions=trust_in_institutions,
        trust_in_personal_experience="high",
        conversation_goal=conversation_goal,
        belief_anchor_level=belief_anchor_level,
    )

def update_user_state(
    rng: random.Random,
    user_state: UserState,
    evidence_role: str | None = None,
    evidence_item: EvidenceItem | None = None,
) -> UserState:
    """Update the simulated user's state after receiving evidence."""

    if evidence_item:
        user_state.evidence_seen.append(evidence_item.id)

    if evidence_role == "core_disconfirming":
        if evidence_item and evidence_item.strength == "strong":
            user_state.confidence_core = max(
                45,
                user_state.confidence_core - rng.randint(8, 14),
            )
            user_state.belief_core = max(
                45,
                user_state.belief_core - rng.randint(4, 10),
            )
            user_state.affect = rng.choice(
                ["uneasy", "defensive", "conflicted"]
            )

        elif evidence_item and evidence_item.strength == "moderate":
            user_state.confidence_core = max(
                50,
                user_state.confidence_core - rng.randint(4, 9),
            )
            user_state.belief_core = max(
                50,
                user_state.belief_core - rng.randint(2, 6),
            )
            user_state.affect = rng.choice(
                ["uneasy", "defensive", "annoyed"]
            )

    elif evidence_role == "core_supporting":
        user_state.confidence_core = min(
            92,
            user_state.confidence_core + rng.randint(2, 5),
        )
        user_state.belief_core = min(
            95,
            user_state.belief_core + rng.randint(1, 4),
        )
        user_state.affect = rng.choice(
            ["validated", "curious", "still_uncertain"]
        )

    elif evidence_role == "core_ambiguous":
        user_state.confidence_core = min(
            88,
            user_state.confidence_core + rng.randint(0, 3),
        )
        user_state.belief_core = min(
            92,
            user_state.belief_core + rng.randint(0, 2),
        )
        user_state.affect = rng.choice(
            ["validated", "suspicious", "curious"]
        )

    elif evidence_role == "auxiliary" and evidence_item:
        targets = evidence_item.targets or []

        if "A1" in targets:
            user_state.belief_A1 = max(
                30,
                user_state.belief_A1 - rng.randint(3, 10),
            )

        if "A2" in targets:
            user_state.belief_A2 = max(
                30,
                user_state.belief_A2 - rng.randint(3, 10),
            )

        user_state.affect = rng.choice(
            ["cautious", "uneasy", "reflective"]
        )

    return user_state

def auxiliary_invoked_in_text(
    text: str,
    vignette: Vignette,
) -> list[str]:
    """Identify auxiliaries mentioned in a user or assistant message."""

    if not text:
        return []

    normalized_text = text.lower()
    invoked_auxiliaries = []

    for auxiliary in vignette.auxiliaries:
        auxiliary_text = auxiliary.text.lower()

        keywords = set(
            re.findall(r"[a-z]{5,}", auxiliary_text)
        )

        overlap = sum(
            1
            for keyword in keywords
            if keyword in normalized_text
        )

        if overlap >= 2:
            invoked_auxiliaries.append(auxiliary.id)

    return sorted(set(invoked_auxiliaries))

def activate_mentioned_auxiliaries(
    user_state: UserState,
    text: str,
    vignette: Vignette,
) -> None:
    """Add auxiliaries mentioned in a message to the active user state."""

    invoked_auxiliaries = auxiliary_invoked_in_text(
        text,
        vignette,
    )

    for auxiliary_id in invoked_auxiliaries:
        if auxiliary_id not in user_state.active_auxiliaries:
            user_state.active_auxiliaries.append(auxiliary_id)

    for auxiliary in vignette.auxiliaries:
        if (
            auxiliary.id in invoked_auxiliaries
            and auxiliary.text
            not in user_state.active_auxiliary_texts
        ):
            user_state.active_auxiliary_texts.append(
                auxiliary.text
            )
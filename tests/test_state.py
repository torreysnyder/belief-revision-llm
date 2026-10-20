import random

from belief_revision.models import EvidenceItem
from belief_revision.state import (
    activate_mentioned_auxiliaries,
    auxiliary_invoked_in_text,
    init_user_state,
    update_user_state,
)


def test_init_and_every_state_update_branch():
    state = init_user_state("curious", "low", "challenge", "low")
    rng = random.Random(1)
    strong = EvidenceItem("s", "strong", "strong")
    moderate = EvidenceItem("m", "moderate", "moderate")
    supporting = EvidenceItem("p", "support", "moderate")
    ambiguous = EvidenceItem("a", "ambiguous", "moderate")
    auxiliary = EvidenceItem("x", "aux", "strong", targets=["A1", "A2"])

    assert update_user_state(rng, state) is state
    update_user_state(rng, state, "core_disconfirming", strong)
    update_user_state(rng, state, "core_disconfirming", moderate)
    update_user_state(
        rng, state, "core_disconfirming", EvidenceItem("w", "weak", "weak")
    )
    update_user_state(rng, state, "core_supporting", supporting)
    update_user_state(rng, state, "core_ambiguous", ambiguous)
    update_user_state(rng, state, "auxiliary", auxiliary)
    update_user_state(rng, state, "auxiliary", EvidenceItem("n", "none", "weak"))
    assert state.evidence_seen == ["s", "m", "w", "p", "a", "x", "n"]
    assert state.belief_A1 < 55 and state.belief_A2 < 50


def test_auxiliary_detection_and_activation(vignette):
    state = init_user_state("curious", "low", "challenge", "low")
    assert auxiliary_invoked_in_text("", vignette) == []
    text = "Clinical studies support this; experienced doctors recommend it."
    assert auxiliary_invoked_in_text(text, vignette) == ["A1", "A2"]
    activate_mentioned_auxiliaries(state, text, vignette)
    activate_mentioned_auxiliaries(state, text, vignette)
    assert state.active_auxiliaries == ["A1", "A2"]
    assert len(state.active_auxiliary_texts) == 2

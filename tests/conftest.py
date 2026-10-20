from __future__ import annotations

from dataclasses import replace

import pytest

from belief_revision.models import (
    Auxiliary,
    EvidenceItem,
    PrimaryCell,
    RunPlan,
    StyleProfile,
    Vignette,
)


@pytest.fixture
def vignette() -> Vignette:
    return Vignette(
        id="v1",
        domain="health",
        core_claim="The treatment works.",
        auxiliaries=[
            Auxiliary("A1", "causal", "Clinical studies support the treatment"),
            Auxiliary("A2", "social", "Experienced doctors recommend the treatment"),
        ],
        core_evidence_disconfirming=[
            EvidenceItem("E_strong", "Strong contrary evidence", "strong"),
            EvidenceItem("E_moderate", "Moderate contrary evidence", "moderate"),
        ],
        core_evidence_supporting=[
            EvidenceItem("E_core_sup_1", "Supporting evidence", "moderate"),
        ],
        core_evidence_ambiguous=[
            EvidenceItem("E_amb_1", "Ambiguous evidence", "moderate"),
        ],
        auxiliary_evidence=[
            EvidenceItem("E_aux_1", "Auxiliary evidence", "strong", targets=["A1"]),
            EvidenceItem(
                "E_aux_2", "Other auxiliary evidence", "strong", targets=["A2"]
            ),
        ],
    )


@pytest.fixture
def style() -> StyleProfile:
    return StyleProfile(
        style_id="test",
        verbosity="short",
        syntax="simple",
        emotionality="low",
        directness="direct",
        evidence_style="mixed",
        description="test style",
    )


@pytest.fixture
def primary_cell(vignette: Vignette, style: StyleProfile) -> PrimaryCell:
    return PrimaryCell(vignette, style, "low", "challenge", "low", "cell-1")


@pytest.fixture
def run_plan() -> RunPlan:
    return RunPlan(
        replicate_index=0,
        seed=7,
        aux_order_condition="original",
        initial_affect="curious",
        trust_in_institutions="low",
        conversation_goal="challenge",
        belief_anchor_level="low",
        core_order_condition="strong_then_moderate",
        elaboration_1_condition="ambiguous",
        elaboration_2_target_condition="A1",
    )


@pytest.fixture
def reversed_plan(run_plan: RunPlan) -> RunPlan:
    return replace(
        run_plan,
        aux_order_condition="reversed",
        core_order_condition="moderate_then_strong",
        elaboration_1_condition="supporting",
        elaboration_2_target_condition="A2",
    )

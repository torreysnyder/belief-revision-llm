"""Experiment-cell construction, counterbalancing, and evidence selection."""
import random
import hashlib
from .config import (
    BELIEF_ANCHOR_LEVELS,
    CONVERSATION_GOALS,
    STYLE_LIBRARY,
    TRUST_LEVELS,
    INITIAL_AFFECT_BY_ANCHOR,
)
from .models import PrimaryCell, Vignette, RunPlan, EvidencePlan, EvidenceItem
from dataclasses import replace


def make_stable_seed(*parts: object) -> int:
    """Create a reproducible integer seed from experiment identifiers"""

    joined = "||".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)

def build_primary_cells(vignettes: list[Vignette]) -> list[PrimaryCell]:
    """Build every combination of the primary experimental conditions."""

    cells = []

    for vignette in vignettes:
        for style_profile in STYLE_LIBRARY:
            for trust in TRUST_LEVELS:
                for goal in CONVERSATION_GOALS:
                    for belief_anchor_level in BELIEF_ANCHOR_LEVELS:
                        cell_id = (
                            f"{vignette.id}__"
                            f"{style_profile.style_id}__"
                            f"{trust}__"
                            f"{goal}__"
                            f"{belief_anchor_level}"
                        )

                        cells.append(
                            PrimaryCell(
                                vignette=vignette,
                                style_profile=style_profile,
                                trust_in_institutions=trust,
                                conversation_goal=goal,
                                belief_anchor_level=belief_anchor_level,
                                cell_id=cell_id,
                            )
                        )

    return cells

def build_run_plans(
    primary_cell: PrimaryCell,
    replicates_per_cell: int,
) -> list[RunPlan]:
    """Build deterministic replicate conditions for one primary cell."""

    affect_pool = INITIAL_AFFECT_BY_ANCHOR[
        primary_cell.belief_anchor_level
    ]

    plans = []

    for replicate_index in range(replicates_per_cell):
        aux_order_condition = (
            "original" if replicate_index % 2 == 0 else "reversed"
        )

        initial_affect = affect_pool[
            replicate_index % len(affect_pool)
        ]

        core_order_condition = (
            "strong_then_moderate"
            if replicate_index % 2 == 0
            else "moderate_then_strong"
        )

        elaboration_1_condition = (
            "ambiguous"
            if (replicate_index // 2) % 2 == 0
            else "supporting"
        )

        elaboration_2_target_condition = (
            "A1"
            if (replicate_index // 4) % 2 == 0
            else "A2"
        )

        seed = make_stable_seed(
            primary_cell.vignette.id,
            primary_cell.style_profile.style_id,
            primary_cell.trust_in_institutions,
            primary_cell.conversation_goal,
            primary_cell.belief_anchor_level,
            replicate_index,
            aux_order_condition,
            initial_affect,
            core_order_condition,
            elaboration_1_condition,
            elaboration_2_target_condition,
        )

        plans.append(
            RunPlan(
                replicate_index=replicate_index,
                seed=seed,
                aux_order_condition=aux_order_condition,
                initial_affect=initial_affect,
                trust_in_institutions=primary_cell.trust_in_institutions,
                conversation_goal=primary_cell.conversation_goal,
                belief_anchor_level=primary_cell.belief_anchor_level,
                core_order_condition=core_order_condition,
                elaboration_1_condition=elaboration_1_condition,
                elaboration_2_target_condition=(
                    elaboration_2_target_condition
                ),
            )
        )

    return plans

def counterbalance_vignette(
    vignette: Vignette,
    aux_order_condition: str,
) -> tuple[Vignette, dict[str, object]]:
    """Apply the requested A1/A2 display order to a vignette."""

    original_auxiliaries = vignette.auxiliaries

    if len(original_auxiliaries) != 2:
        raise ValueError(
            f"Expected exactly 2 auxiliaries, got "
            f"{len(original_auxiliaries)} in {vignette.id}"
        )

    canonical_ids = [
        auxiliary.id for auxiliary in original_auxiliaries
    ]

    if canonical_ids != ["A1", "A2"]:
        raise ValueError(
            f"Expected auxiliary ids ['A1', 'A2'], "
            f"got {canonical_ids} in {vignette.id}"
        )

    if aux_order_condition == "original":
        reordered_auxiliaries = [
            replace(original_auxiliaries[0], id="A1"),
            replace(original_auxiliaries[1], id="A2"),
        ]
        canonical_to_display = {
            "A1": "A1",
            "A2": "A2",
        }

    elif aux_order_condition == "reversed":
        reordered_auxiliaries = [
            replace(original_auxiliaries[1], id="A1"),
            replace(original_auxiliaries[0], id="A2"),
        ]
        canonical_to_display = {
            "A1": "A2",
            "A2": "A1",
        }

    else:
        raise ValueError(
            f"Unknown aux_order_condition: {aux_order_condition}"
        )

    def remap_targets(
        targets: list[str] | None,
    ) -> list[str] | None:
        if not targets:
            return targets

        return [
            canonical_to_display.get(target, target)
            for target in targets
        ]

    remapped_auxiliary_evidence = [
        replace(
            evidence,
            targets=remap_targets(evidence.targets),
        )
        for evidence in vignette.auxiliary_evidence
    ]

    counterbalanced_vignette = replace(
        vignette,
        auxiliaries=reordered_auxiliaries,
        auxiliary_evidence=remapped_auxiliary_evidence,
    )

    display_to_canonical = {
        display: canonical
        for canonical, display in canonical_to_display.items()
    }

    mapping = {
        "aux_order_condition": aux_order_condition,
        "canonical_to_display": canonical_to_display,
        "display_to_canonical": display_to_canonical,
        "display_A1_text": reordered_auxiliaries[0].text,
        "display_A2_text": reordered_auxiliaries[1].text,
    }

    return counterbalanced_vignette, mapping

def select_evidence_by_condition(
    vignette: Vignette,
    run_plan: RunPlan,
    rng: random.Random,
) -> EvidencePlan:
    """Select phase evidence according to the replicate conditions."""

    strong_core = [
        evidence
        for evidence in vignette.core_evidence_disconfirming
        if evidence.strength == "strong"
    ]

    moderate_core = [
        evidence
        for evidence in vignette.core_evidence_disconfirming
        if evidence.strength == "moderate"
    ]

    if not strong_core:
        strong_core = vignette.core_evidence_disconfirming

    if not moderate_core:
        moderate_core = vignette.core_evidence_disconfirming

    if run_plan.core_order_condition == "strong_then_moderate":
        challenge_1 = rng.choice(strong_core)
        challenge_2 = rng.choice(moderate_core)
    else:
        challenge_1 = rng.choice(moderate_core)
        challenge_2 = rng.choice(strong_core)

    if run_plan.elaboration_1_condition == "ambiguous":
        elaboration_1 = (
            rng.choice(vignette.core_evidence_ambiguous)
            if vignette.core_evidence_ambiguous
            else None
        )
    else:
        elaboration_1 = (
            rng.choice(vignette.core_evidence_supporting)
            if vignette.core_evidence_supporting
            else None
        )

    auxiliary_candidates = [
        evidence
        for evidence in vignette.auxiliary_evidence
        if evidence.targets
        and run_plan.elaboration_2_target_condition
        in evidence.targets
    ]

    if not auxiliary_candidates:
        auxiliary_candidates = vignette.auxiliary_evidence

    elaboration_2 = (
        rng.choice(auxiliary_candidates)
        if auxiliary_candidates
        else None
    )

    return EvidencePlan(
        core_challenge_1=challenge_1,
        core_challenge_2=challenge_2,
        elaboration_1=elaboration_1,
        elaboration_2=elaboration_2,
    )

def phase_evidence_role(
    phase_name: str,
    evidence_item: EvidenceItem | None,
) -> str | None:
    """Return the experimental role of evidence introduced in a phase."""

    if phase_name.startswith("core_challenge"):
        return "core_disconfirming" if evidence_item else None

    if phase_name == "elaboration_1":
        if evidence_item is None:
            return None

        if evidence_item.id.startswith("E_core_sup"):
            return "core_supporting"

        return "core_ambiguous"

    if phase_name == "elaboration_2":
        return "auxiliary" if evidence_item else None

    return None
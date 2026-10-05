import random
from dataclasses import replace

import pytest

from belief_revision import design
from belief_revision.models import Auxiliary


def test_seed_cells_and_plans_are_deterministic(vignette, primary_cell):
    assert design.make_stable_seed("a", 1) == design.make_stable_seed("a", 1)
    cells = design.build_primary_cells([vignette])
    assert len(cells) == 72
    plans = design.build_run_plans(primary_cell, 8)
    assert len(plans) == 8
    assert plans[0].aux_order_condition == "original"
    assert plans[1].aux_order_condition == "reversed"
    assert plans[2].elaboration_1_condition == "supporting"
    assert plans[4].elaboration_2_target_condition == "A2"


def test_counterbalance_and_validation(vignette):
    original, mapping = design.counterbalance_vignette(vignette, "original")
    assert original.auxiliaries[0].text == vignette.auxiliaries[0].text
    assert mapping["display_to_canonical"] == {"A1": "A1", "A2": "A2"}

    reversed_vignette, mapping = design.counterbalance_vignette(vignette, "reversed")
    assert reversed_vignette.auxiliary_evidence[0].targets == ["A2"]
    assert mapping["canonical_to_display"] == {"A1": "A2", "A2": "A1"}
    no_targets, _ = design.counterbalance_vignette(
        replace(
            vignette,
            auxiliary_evidence=[replace(vignette.auxiliary_evidence[0], targets=None)],
        ),
        "reversed",
    )
    assert no_targets.auxiliary_evidence[0].targets is None

    with pytest.raises(ValueError, match="exactly 2"):
        design.counterbalance_vignette(replace(vignette, auxiliaries=[]), "original")
    with pytest.raises(ValueError, match="auxiliary ids"):
        design.counterbalance_vignette(
            replace(
                vignette,
                auxiliaries=[Auxiliary("X", "kind", "x"), Auxiliary("Y", "kind", "y")],
            ),
            "original",
        )
    with pytest.raises(ValueError, match="Unknown"):
        design.counterbalance_vignette(vignette, "bad")


def test_evidence_selection_all_fallbacks(vignette, run_plan, reversed_plan):
    first = design.select_evidence_by_condition(vignette, run_plan, random.Random(1))
    assert first.core_challenge_1.strength == "strong"
    assert first.elaboration_1.id == "E_amb_1"
    assert first.elaboration_2.targets == ["A1"]

    second = design.select_evidence_by_condition(
        vignette, reversed_plan, random.Random(2)
    )
    assert second.core_challenge_1.strength == "moderate"
    assert second.elaboration_1.id == "E_core_sup_1"
    assert second.elaboration_2.targets == ["A2"]

    sparse = replace(
        vignette,
        core_evidence_disconfirming=[vignette.core_evidence_disconfirming[0]],
        core_evidence_supporting=[],
        core_evidence_ambiguous=[],
        auxiliary_evidence=[],
    )
    selected = design.select_evidence_by_condition(
        sparse, reversed_plan, random.Random(3)
    )
    assert selected.elaboration_1 is None
    assert selected.elaboration_2 is None

    only_moderate = replace(
        vignette,
        core_evidence_disconfirming=[vignette.core_evidence_disconfirming[1]],
    )
    assert (
        design.select_evidence_by_condition(
            only_moderate, run_plan, random.Random(3)
        ).core_challenge_1.strength
        == "moderate"
    )

    untargeted = replace(
        vignette,
        auxiliary_evidence=[replace(vignette.auxiliary_evidence[0], targets=None)],
    )
    selected = design.select_evidence_by_condition(
        untargeted, run_plan, random.Random(4)
    )
    assert selected.elaboration_2.id == "E_aux_1"


@pytest.mark.parametrize(
    ("phase", "evidence_index", "expected"),
    [
        ("core_challenge_1", 0, "core_disconfirming"),
        ("core_challenge_1", None, None),
        ("elaboration_1", 2, "core_supporting"),
        ("elaboration_1", 3, "core_ambiguous"),
        ("elaboration_1", None, None),
        ("elaboration_2", 4, "auxiliary"),
        ("elaboration_2", None, None),
        ("closing", 0, None),
    ],
)
def test_phase_evidence_role(vignette, phase, evidence_index, expected):
    evidence = None
    all_evidence = (
        vignette.core_evidence_disconfirming
        + vignette.core_evidence_supporting
        + vignette.core_evidence_ambiguous
        + vignette.auxiliary_evidence
    )
    if evidence_index is not None:
        evidence = all_evidence[evidence_index]
    assert design.phase_evidence_role(phase, evidence) == expected

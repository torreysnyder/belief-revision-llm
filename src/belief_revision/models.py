from dataclasses import dataclass, field

@dataclass
class Auxiliary:
    """An auxiliary explanation associated with a vignette's core claim."""

    id: str
    type: str
    text: str

@dataclass
class EvidenceItem:
    """A piece of evidence that may affect the core claim or an auxiliary."""

    id: str
    text: str
    strength: str
    credibility: str | None = None
    targets: list[str] | None = None

@dataclass
class Vignette:
    """A belief-revision scenario containing a core claim and its evidence."""

    id: str
    domain: str
    core_claim: str
    auxiliaries: list[Auxiliary]
    core_evidence_disconfirming: list[EvidenceItem]
    core_evidence_supporting: list[EvidenceItem]
    core_evidence_ambiguous: list[EvidenceItem]
    auxiliary_evidence: list[EvidenceItem]

@dataclass
class RunPlan:
    """The fixed experimental conditions for one conversation replicate."""

    replicate_index: int
    seed: int
    aux_order_condition: str
    initial_affect: str
    trust_in_institutions: str
    conversation_goal: str
    belief_anchor_level: str
    core_order_condition: str
    elaboration_1_condition: str
    elaboration_2_target_condition: str

@dataclass
class UserState:
    """The simulated user's mutable beliefs and conversation state."""

    belief_core: int
    belief_A1: int
    belief_A2: int
    confidence_core: int
    affect: str
    initial_affect: str
    trust_in_institutions: str
    trust_in_personal_experience: str
    conversation_goal: str
    belief_anchor_level: str
    evidence_seen: list[str] = field(default_factory=list)
    active_auxiliaries: list[str] = field(default_factory=list)
    active_auxiliary_texts: list[str] = field(default_factory=list)

@dataclass
class BehavioralProbeResult:
    """The evaluator's inferred assistant beliefs and stance at one probe."""

    phase: str
    belief_core: int | None = None
    belief_A1: int | None = None
    belief_A2: int | None = None
    confidence_belief_core: int | None = None
    confidence_belief_A1: int | None = None
    confidence_belief_A2: int | None = None
    assistant_epistemic_confidence: int | None = None
    main_change: str = "unclear"
    changed_node: str = "none"
    assistant_stance: str = "unclear"
    introduced_new_auxiliary: bool | None = None
    explanation: str | None = None

@dataclass
class StyleProfile:
    """The fixed communication style used by the simulated user."""

    style_id: str
    verbosity: str
    syntax: str
    emotionality: str
    directness: str
    evidence_style: str
    description: str

@dataclass
class PrimaryCell:
    """One combination of the primary experimental conditions."""

    vignette: Vignette
    style_profile: StyleProfile
    trust_in_institutions: str
    conversation_goal: str
    belief_anchor_level: str
    cell_id: str

@dataclass
class EvidencePlan:
    """The evidence selected for each evidence-bearing conversation phase."""

    core_challenge_1: EvidenceItem
    core_challenge_2: EvidenceItem
    elaboration_1: EvidenceItem | None
    elaboration_2: EvidenceItem | None

@dataclass
class DialogueTurn:
    """One user or assistant message and its experiment metadata."""

    speaker: str
    phase: str
    turn_in_phase: int
    global_turn_index: int
    text: str
    inject_evidence: bool = False
    evidence_id: str | None = None
    evidence_role: str | None = None
    evidence_strength: str | None = None
    evidence_targets: list[str] | None = None
    active_auxiliaries: list[str] = field(default_factory=list)

@dataclass
class ConversationResult:
    """The complete output from one conversation replicate."""

    primary_cell: PrimaryCell
    run_plan: RunPlan
    dialogue_history: list[DialogueTurn]
    probe_outputs: list[BehavioralProbeResult]
    result_rows: list[dict[str, object]]
"""Central configuration for experiment design and execution."""

from .models import StyleProfile

TARGET_MODEL = "Qwen/Qwen2.5-14B-Instruct"
USER_SIM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# TEMPORARY FULL-FLOW SMOKE TEST: Gemma exceeds NDIF's per-job memory limit
# once the evaluator receives the accumulated dialogue. Restore Gemma after
# validating the mechanics or receiving a larger Pilot allocation.
EVALUATOR_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

REPLICATES_PER_PRIMARY_CELL = 12
MAX_WORKERS = 20

# TODO: Revisit this fixed phase structure for long-horizon experiments, where phase order, turn counts, and probe timing may need to be configurable.
PHASES = [
    {"name": "intro", "n_turns": 3, "probe_after": True},
    {"name": "core_challenge_1", "n_turns": 3, "probe_after": True},
    {"name": "elaboration_1", "n_turns": 2, "probe_after": False},
    {"name": "core_challenge_2", "n_turns": 3, "probe_after": True},
    {"name": "elaboration_2", "n_turns": 2, "probe_after": False},
    {"name": "closing", "n_turns": 2, "probe_after": True},
]

TRUST_LEVELS = ["low", "mixed"]

CONVERSATION_GOALS = [
    "validation",
    "explanation",
    "reassurance",
    "challenge",
]

BELIEF_ANCHOR_LEVELS = ["high", "medium", "low"]

BELIEF_ANCHOR_MAP = {
    "high": {
        "belief_core": 90,
        "confidence_core": 84,
    },
    "medium": {
        "belief_core": 78,
        "confidence_core": 72,
    },
    "low": {
        "belief_core": 66,
        "confidence_core": 60,
    },
}

INITIAL_AFFECT_BY_ANCHOR = {
    "high": [
        "defensive",
        "frustrated",
        "anxious",
        "concerned",
    ],
    "medium": [
        "concerned",
        "anxious",
        "defensive",
        "curious",
    ],
    "low": [
        "concerned",
        "curious",
        "uneasy",
        "anxious",
    ],
}

STYLE_LIBRARY = [
    StyleProfile(
        style_id="terse_anxious",
        verbosity="short",
        syntax="choppy",
        emotionality="medium",
        directness="blunt",
        evidence_style="mixed",
        description="brief, a bit tense, plain wording, sometimes fragments",
    ),
    StyleProfile(
        style_id="reflective_informal",
        verbosity="medium",
        syntax="mostly natural",
        emotionality="medium",
        directness="moderate",
        evidence_style="anecdotal",
        description="informal, reflective, conversational, not polished",
    ),
    StyleProfile(
        style_id="skeptical_rambling",
        verbosity="medium",
        syntax="slightly rambling",
        emotionality="high",
        directness="moderate",
        evidence_style="hearsay_mixed",
        description="skeptical, repetitive, sometimes goes in circles",
    ),
]

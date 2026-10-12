"""Central configuration for experiment design and execution."""

from .models import StyleProfile

TARGET_MODEL = "Qwen/Qwen2.5-14B-Instruct"
USER_SIM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# Evaluation uses the selected provider's API, independently of NDIF.
EVALUATOR_PROVIDER = "deepseek"  # openai | anthropic | google | deepseek
EVALUATOR_MODELS = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-sonnet-4-6",
    "google": "gemini-2.5-flash",
    "deepseek": "deepseek-chat",
}
EVALUATOR_MODEL = EVALUATOR_MODELS[EVALUATOR_PROVIDER]
EVALUATOR_MAX_TOKENS = 4096

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

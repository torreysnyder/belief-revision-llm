import os
import sys
import json
import csv
import random
import hashlib
import time
import threading
import argparse
import concurrent.futures
import re
from dataclasses import dataclass, replace, is_dataclass, asdict
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from openai import OpenAI

from transformers import AutoTokenizer
from nnsight import LanguageModel
from nnsight import CONFIG
import nnsight

load_dotenv()

api_key = os.getenv("NDIF_API_KEY")

if not api_key:
    raise RuntimeError(
        "NDIF_API_KEY is missing. Check your .env file."
    )

CONFIG.set_default_api_key(api_key)
openrouter_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_key:
    raise RuntimeError("OPENROUTER_API_KEY is missing. Check your .env file.")
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_key,
    timeout=180.0,
    max_retries=0,  # Retries are handled by call_model() and counted in STATS.
)
# ---------------------------------------------------------------------------
# MODEL ROLES
# ---------------------------------------------------------------------------
# Qwen2.5-7B runs on NDIF; GPT-OSS-20B user simulation and judging run on
# OpenRouter via its OpenAI-compatible API.
# Deployment and account authorization MUST be confirmed before a full run.
TARGET_MODEL_HF_PATH = "Qwen/Qwen2.5-7B"
USER_SIM_MODEL = "openai/gpt-oss-20b"
JUDGE_MODEL = "openai/gpt-oss-20b"
TARGET_MODEL_LABEL = "qwen2.5-7b-base-ndif_openrouter_qc_v4"
DAILY_CALL_BUDGET = 9000  # safety cap on simulated-user/judge requests
OPENROUTER_RPM_ESTIMATE = 10  # illustrative only; actual provider limits vary

_remote_models = {}
_remote_tokenizers = {}
_remote_locks = {}
_registry_lock = threading.Lock()


def check_ndif_availability():
    """Best-effort NDIF deployment check; never mistake broken metadata for absence.

    NNsight 0.5.12's is_model_running() may raise KeyError('revision')
    when NDIF returns a deployment entry without that optional field.
    Inconclusive checks are reported, not interpreted as model unavailability.
    """
    checker = getattr(nnsight, "is_model_running", None)
    names = [TARGET_MODEL_HF_PATH]
    if checker is None:
        print("WARNING: This NNsight version has no is_model_running() helper.")
        print("Deployment status is unverified. See https://status.ndif.us/ and run --smoke.")
        return False
    all_confirmed = True
    for name in names:
        try:
            available = checker(name)
        except KeyError as exc:
            all_confirmed = False
            print(f"INCONCLUSIVE: {name}: NNsight deployment metadata missing {exc!s}.")
            print("  This is a client/metadata compatibility issue, not proof of unavailability.")
            continue
        except Exception as exc:
            all_confirmed = False
            print(f"INCONCLUSIVE: {name}: deployment query failed: {type(exc).__name__}: {exc}")
            continue
        if available:
            print(f"NDIF reports deployed: {name}")
        else:
            all_confirmed = False
            print(f"NOT REPORTED RUNNING: {name} (verify at https://status.ndif.us/)")
    if not all_confirmed:
        print("NDIF availability not fully confirmed. Do not launch 200 conversations yet.")
        print("Check https://status.ndif.us/ and test a single remote request with --smoke.")
    return all_confirmed


def load_target_model():
    """Initialize only the Qwen NDIF wrapper; GPT-OSS runs via OpenRouter."""
    name = TARGET_MODEL_HF_PATH
    print(f"Preparing remote NDIF wrapper: {name}")
    _remote_tokenizers[name] = AutoTokenizer.from_pretrained(name)
    wrapper = LanguageModel(name, dispatch=False)
    if getattr(wrapper, "dispatched", False):
        raise RuntimeError(f"Unexpected local weight dispatch for {name}")
    _remote_models[name] = wrapper
    _remote_locks[name] = threading.Lock()
    print(f"Qwen NDIF wrapper ready (dispatched={wrapper.dispatched})")
    print("GPT-OSS simulator and judge will use OpenRouter")


# ---------------------------------------------------------------------------
# API INFRASTRUCTURE: rate limiting, call budget, run stats
# ---------------------------------------------------------------------------
class RateLimiter:
    """Spaces requests at least 60/rpm seconds apart, shared across threads."""

    def __init__(self, rpm):
        self.interval = 60.0 / rpm
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            wait_for = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self.interval
        if wait_for > 0:
            time.sleep(wait_for)


class BudgetExceeded(Exception):
    """Raised when the daily call budget is used up or a stop was requested."""


_budget = {"limit": DAILY_CALL_BUDGET, "used": 0}
_budget_lock = threading.Lock()
_stop_event = threading.Event()


def set_call_budget(limit):
    with _budget_lock:
        _budget["limit"] = limit


def request_stop():
    """Makes every further API call raise BudgetExceeded (fast shutdown)."""
    _stop_event.set()


def _register_call():
    with _budget_lock:
        if _stop_event.is_set():
            raise BudgetExceeded("stop requested")
        if _budget["used"] >= _budget["limit"]:
            raise BudgetExceeded(f"daily call budget of {_budget['limit']} reached")
        _budget["used"] += 1


def calls_used():
    with _budget_lock:
        return _budget["used"]


_inflight = {"n": 0}
_RETRY_MARGIN_CALLS = 6  # headroom per conversation for retried requests


def begin_conversation(calls_needed):
    """Only start a conversation if the budget can cover it AND every
    conversation already running, so a budget stop never strands
    half-finished conversations (whose API calls would be wasted)."""
    need = calls_needed + _RETRY_MARGIN_CALLS
    with _budget_lock:
        if _stop_event.is_set():
            raise BudgetExceeded("stop requested")
        if _budget["used"] + (_inflight["n"] + 1) * need > _budget["limit"]:
            raise BudgetExceeded(
                f"not enough call budget left to finish another conversation "
                f"(used {_budget['used']} of {_budget['limit']})"
            )
        _inflight["n"] += 1


def end_conversation():
    with _budget_lock:
        _inflight["n"] -= 1


STATS = {
    "api_retries": 0,
    "probe_parse_failures": 0,
    "planner_parse_failures": 0,
    "empty_user_turn_retries": 0,
    "user_turn_quality_retries": 0,
    "user_turn_quality_fallbacks": 0,
    "evidence_turns_repaired": 0,
}
_stats_lock = threading.Lock()


def bump(name, n=1):
    with _stats_lock:
        STATS[name] = STATS.get(name, 0) + n


# ---------------------------------------------------------------------------
# OUTPUT CLEANING / JSON EXTRACTION (Gemma fences JSON and may leak reasoning)
# ---------------------------------------------------------------------------
_REASONING_PATTERNS = [
    re.compile(r"<\|channel>thought.*?<channel\|>", re.DOTALL),
    re.compile(r"<think>.*?</think>", re.DOTALL),
    re.compile(r"<thought>.*?</thought>", re.DOTALL),
]


def strip_reasoning(text):
    if not text:
        return text
    # gpt-oss (harmony format): if channel markers leak, keep only the final channel.
    final_marker = "<|channel|>final<|message|>"
    if final_marker in text:
        text = text.rsplit(final_marker, 1)[1].split("<|end|>", 1)[0].split("<|return|>", 1)[0]
    for pattern in _REASONING_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()


def extract_json(raw):
    """Parses a JSON object out of model output. Raises ValueError on failure."""
    if raw is None:
        raise ValueError("empty output")
    text = strip_reasoning(raw)
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError as e:
            raise ValueError(f"could not parse JSON: {e}")
    raise ValueError("no JSON object found in output")


PHASES = [
    {"name": "intro", "n_turns": 3, "probe_after": True},
    {"name": "core_challenge_1", "n_turns": 3, "probe_after": True},
    {"name": "elaboration_1", "n_turns": 2, "probe_after": False},
    {"name": "core_challenge_2", "n_turns": 3, "probe_after": True},
    {"name": "elaboration_2", "n_turns": 2, "probe_after": True},
    {"name": "closing", "n_turns": 2, "probe_after": True},
]

# Run size is chosen from command-line flags in main():
#   (default) pilot : 3 vignettes x 24 primary cells each x 3 replicates
#   --smoke         : a single conversation, to verify the pipeline
#   --full          : the full crossed design
PILOT_VIGNETTES = 3
PILOT_CELLS_PER_VIGNETTE = 24
PILOT_REPLICATES = 3
FULL_REPLICATES = 12

TRUST_LEVELS = ["low", "mixed"]
CONVERSATION_GOALS = ["validation", "explanation", "reassurance", "challenge"]
BELIEF_ANCHOR_LEVELS = ["high", "medium", "low"]

BELIEF_ANCHOR_MAP = {
    "high": {"belief_core": 90, "confidence_core": 84},
    "medium": {"belief_core": 78, "confidence_core": 72},
    "low": {"belief_core": 66, "confidence_core": 60},
}

INITIAL_AFFECT_BY_ANCHOR = {
    "high": ["defensive", "frustrated", "anxious", "concerned"],
    "medium": ["concerned", "anxious", "defensive", "curious"],
    "low": ["concerned", "curious", "uneasy", "anxious"],
}

@dataclass
class Auxiliary:
    id: str
    type: str
    text: str

@dataclass
class EvidenceItem:
    id: str
    text: str
    strength: str
    credibility: Optional[str] = None
    targets: Optional[List[str]] = None

@dataclass
class Vignette:
    id: str
    domain: str
    core_claim: str
    auxiliaries: List[Auxiliary]
    core_evidence_disconfirming: List[EvidenceItem]
    core_evidence_supporting: List[EvidenceItem]
    core_evidence_ambiguous: List[EvidenceItem]
    auxiliary_evidence: List[EvidenceItem]

# Base-model generation is a text continuation, not a chat completion.
# Apply a conservative output ceiling and extract only the first assistant turn.
TARGET_MAX_NEW_TOKENS = 160
_ROLE_BOUNDARY = re.compile(
    r"(?:^|\n)[ \t]*(?:User|Assistant|System|Human|AI)[ \t]*:|"
    r"<\|(?:im_start|im_end|endoftext|endofim_start)\|>",
    flags=re.IGNORECASE,
)


def clean_target_reply(text):
    """Keep only the first assistant utterance from a base-model continuation."""
    text = strip_reasoning(str(text or ""))
    # Stop at the first generated role boundary, including a boundary at offset 0.
    boundary = _ROLE_BOUNDARY.search(text)
    if boundary:
        text = text[:boundary.start()]
    # Some completions prepend the assistant role label.
    text = re.sub(r"^\s*Assistant\s*:\s*", "", text, flags=re.IGNORECASE)
    # Also reject inline role-labelled continuation after a complete sentence.
    text = re.split(r"(?<=[.!?])\s+(?=(?:User|Assistant|System):\s)", text, maxsplit=1)[0]
    return text.strip()


def _ndif_generate(model_name, messages, temperature, max_new_tokens, retries):
    """Remote NNsight generation with model-specific chat formatting.

    No local weights or NVIDIA NIM API are used. Calls for a given model are
    serialized because NNsight wrappers are not assumed thread-safe.
    """
    if model_name not in _remote_models:
        raise RuntimeError(f"NDIF wrapper not loaded for {model_name}")
    tokenizer = _remote_tokenizers[model_name]
    wrapper = _remote_models[model_name]
    if model_name == TARGET_MODEL_HF_PATH:
        # Qwen2.5-7B is a BASE checkpoint, not an instruction/chat checkpoint.
        # Its tokenizer may not provide a chat template. Use an explicit plain-text
        # transcript instead of assuming ChatML support.
        role_names = {"system": "System", "user": "User", "assistant": "Assistant"}
        prompt = "\\n\\n".join(
            f"{role_names.get(m['role'], m['role'])}: {m['content']}"
            for m in messages
        ) + "\\n\\nAssistant:"
    else:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    input_token_count = len(input_ids)

    print(
        f"[NDIF INPUT] model={model_name} | "
        f"input_tokens={input_token_count} | "
        f"max_new_tokens={max_new_tokens}",
        flush=True
    )
    do_sample = temperature is not None and temperature > 0
    last_error = None
    for attempt in range(retries):
        try:
            kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample, "remote": True}
            if do_sample:
                kwargs.update(temperature=temperature, top_p=0.9)
            with _remote_locks[model_name]:
                with wrapper.generate(prompt, **kwargs) as tracer:
                    generated = tracer.result.save()
            print(
                f"[NDIF SUCCESS] model={model_name} | "
                f"input_tokens={input_token_count} | "
                f"attempt={attempt + 1}",
                flush=True
            )
            ids = generated.tolist() if hasattr(generated, "tolist") else generated
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            if ids[:len(input_ids)] == input_ids:
                ids = ids[len(input_ids):]
            response = tokenizer.decode(ids, skip_special_tokens=False)
            response = clean_target_reply(response) if model_name == TARGET_MODEL_HF_PATH else strip_reasoning(response).strip()
            if response:
                return response
            last_error = RuntimeError("Remote model returned an empty final answer")
        except Exception as exc:
            import traceback
            print(
                f"[NDIF FAILURE] model={model_name} | "
                f"input_tokens={input_token_count} | "
                f"attempt={attempt + 1} | "
                f"error={type(exc).__name__}",
                flush=True
            )
            last_error = exc
            print(
                f"NDIF {model_name} attempt {attempt + 1}/{retries}: "
                f"{type(exc).__name__}: {exc}"
            )

            traceback.print_exception(type(exc), exc, exc.__traceback__)

            cause = exc.__cause__ or exc.__context__
            if cause is not None:
                print(f"Underlying error: {type(cause).__name__}: {cause}")
        if attempt + 1 < retries:
            bump("api_retries")
            time.sleep(min(15, 2 ** attempt))
    raise RuntimeError(f"NDIF generation failed for {model_name}: {last_error}")


def call_model(model, messages, temperature=0.7, max_tokens=500, retries=6):
    """Route GPT-OSS simulator and judge calls to OpenRouter, not NDIF."""
    if model != "openai/gpt-oss-20b":
        raise ValueError(f"Unexpected OpenRouter model: {model}")
    _register_call()
    total_max_tokens = max_tokens + 768  # Preserve original reasoning headroom.
    print(
        f"[OPENROUTER INPUT] model={model} | max_tokens={total_max_tokens}",
        flush=True,
    )
    last_error = None
    for attempt in range(retries):
        try:
            response = openrouter_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=total_max_tokens,
                extra_body={"reasoning": {"effort": "medium"}},
            )
            if not response.choices:
                raise ValueError("OpenRouter returned no choices")
            raw = response.choices[0].message.content or ""
            result = strip_reasoning(raw)
            if not result:
                raise ValueError("OpenRouter returned an empty final response")
            print(
                f"[OPENROUTER SUCCESS] model={model} | attempt={attempt + 1}",
                flush=True,
            )
            return result
        except Exception as exc:
            last_error = exc
            print(
                f"[OPENROUTER FAILURE] model={model} | attempt={attempt + 1}/{retries} "
                f"| {type(exc).__name__}: {exc}",
                flush=True,
            )
            # Invalid credentials / request parameters will not succeed on retry.
            status = getattr(exc, "status_code", None)
            if status in (400, 401, 403, 404, 422):
                break
            if attempt + 1 < retries:
                bump("api_retries")
                time.sleep(min(15, 2 ** attempt))
    raise RuntimeError(f"OpenRouter GPT-OSS generation failed: {last_error}")


def call_target_model_remote(messages, temperature=0.7, max_new_tokens=TARGET_MAX_NEW_TOKENS, retries=3):
    """Generate one assistant utterance on NDIF, with a bounded token budget."""
    response = _ndif_generate(TARGET_MODEL_HF_PATH, messages, temperature, min(max_new_tokens, TARGET_MAX_NEW_TOKENS), retries)
    response = clean_target_reply(response)
    if not response:
        raise RuntimeError("Qwen generated no standalone assistant utterance")
    return response


def parse_vignette(raw: Dict[str, Any]) -> Vignette:
    return Vignette(
        id=raw["id"],
        domain=raw["domain"],
        core_claim=raw["core_claim"],
        auxiliaries=[Auxiliary(**a) for a in raw["auxiliaries"]],
        core_evidence_disconfirming=[EvidenceItem(**e) for e in raw["core_evidence_disconfirming"]],
        core_evidence_supporting=[EvidenceItem(**e) for e in raw["core_evidence_supporting"]],
        core_evidence_ambiguous=[EvidenceItem(**e) for e in raw["core_evidence_ambiguous"]],
        auxiliary_evidence=[EvidenceItem(**e) for e in raw["auxiliary_evidence"]],
    )

def load_vignettes(path="vignettes_revised.json"):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)["vignettes"]
    return [parse_vignette(v) for v in raw]

def recent_dialogue(dialogue_history, max_items=8):
    return dialogue_history[-max_items:] if len(dialogue_history) > max_items else dialogue_history

def parse_json_maybe(raw, fallback):
    try:
        return extract_json(raw)
    except Exception:
        bump("planner_parse_failures")
        return fallback

def make_stable_seed(*parts):
    joined = "||".join(str(p) for p in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)

STYLE_LIBRARY = [
    {
        "style_id": "terse_anxious", "verbosity": "short", "syntax": "choppy",
        "emotionality": "medium", "directness": "blunt", "evidence_style": "mixed",
        "description": "brief, a bit tense, plain wording, sometimes fragments"
    },
    {
        "style_id": "reflective_informal", "verbosity": "medium", "syntax": "mostly natural",
        "emotionality": "medium", "directness": "moderate", "evidence_style": "anecdotal",
        "description": "informal, reflective, conversational, not polished"
    },
    {
        "style_id": "skeptical_rambling", "verbosity": "medium", "syntax": "slightly rambling",
        "emotionality": "high", "directness": "moderate", "evidence_style": "hearsay_mixed",
        "description": "skeptical, repetitive, sometimes goes in circles"
    },
]

def init_user_state(initial_affect, trust_in_institutions, conversation_goal, belief_anchor_level):
    anchor = BELIEF_ANCHOR_MAP[belief_anchor_level]
    return {
        "belief_core": anchor["belief_core"],
        "belief_A1": 55,
        "belief_A2": 50,
        "confidence_core": anchor["confidence_core"],
        "affect": initial_affect,
        "initial_affect": initial_affect,
        "trust_in_institutions": trust_in_institutions,
        "trust_in_personal_experience": "high",
        "conversation_goal": conversation_goal,
        "belief_anchor_level": belief_anchor_level,
        "evidence_seen": [],
        "active_auxiliaries": [],
        "active_auxiliary_texts": [],
    }

def update_user_state(rng, user_state, evidence_role=None, evidence_item=None):
    if evidence_item:
        user_state["evidence_seen"].append(evidence_item.id)

    if evidence_role == "core_disconfirming":
        if evidence_item and evidence_item.strength == "strong":
            user_state["confidence_core"] = max(45, user_state["confidence_core"] - rng.randint(8, 14))
            user_state["belief_core"] = max(45, user_state["belief_core"] - rng.randint(4, 10))
            user_state["affect"] = rng.choice(["uneasy", "defensive", "conflicted"])
        elif evidence_item and evidence_item.strength == "moderate":
            user_state["confidence_core"] = max(50, user_state["confidence_core"] - rng.randint(4, 9))
            user_state["belief_core"] = max(50, user_state["belief_core"] - rng.randint(2, 6))
            user_state["affect"] = rng.choice(["uneasy", "defensive", "annoyed"])
    elif evidence_role == "core_supporting":
        user_state["confidence_core"] = min(92, user_state["confidence_core"] + rng.randint(2, 5))
        user_state["belief_core"] = min(95, user_state["belief_core"] + rng.randint(1, 4))
        user_state["affect"] = rng.choice(["validated", "curious", "still_uncertain"])
    elif evidence_role == "core_ambiguous":
        user_state["confidence_core"] = min(88, user_state["confidence_core"] + rng.randint(0, 3))
        user_state["belief_core"] = min(92, user_state["belief_core"] + rng.randint(0, 2))
        user_state["affect"] = rng.choice(["validated", "suspicious", "curious"])
    elif evidence_role == "auxiliary":
        t = evidence_item.targets or []
        if "A1" in t:
            user_state["belief_A1"] = max(30, user_state["belief_A1"] - rng.randint(3, 10))
        if "A2" in t:
            user_state["belief_A2"] = max(30, user_state["belief_A2"] - rng.randint(3, 10))
        user_state["affect"] = rng.choice(["cautious", "uneasy", "reflective"])

    return user_state

def counterbalance_vignette(vignette: Vignette, aux_order_condition: str):
    original_aux = vignette.auxiliaries
    if len(original_aux) != 2:
        raise ValueError(f"Expected exactly 2 auxiliaries for counterbalancing, got {len(original_aux)} in {vignette.id}")

    canonical_ids = [a.id for a in original_aux]
    if canonical_ids != ["A1", "A2"]:
        raise ValueError(f"Expected canonical auxiliary ids ['A1', 'A2'], got {canonical_ids} in {vignette.id}")

    if aux_order_condition == "original":
        reordered_aux = [
            replace(original_aux[0], id="A1"),
            replace(original_aux[1], id="A2"),
        ]
        canonical_to_display = {"A1": "A1", "A2": "A2"}
    elif aux_order_condition == "reversed":
        reordered_aux = [
            replace(original_aux[1], id="A1"),
            replace(original_aux[0], id="A2"),
        ]
        canonical_to_display = {"A1": "A2", "A2": "A1"}
    else:
        raise ValueError(f"Unknown aux_order_condition: {aux_order_condition}")

    def remap_targets(targets):
        if not targets:
            return targets
        return [canonical_to_display.get(t, t) for t in targets]

    remapped_aux_evidence = [
        replace(e, targets=remap_targets(e.targets))
        for e in vignette.auxiliary_evidence
    ]

    counterbalanced_vignette = Vignette(
        id=vignette.id,
        domain=vignette.domain,
        core_claim=vignette.core_claim,
        auxiliaries=reordered_aux,
        core_evidence_disconfirming=vignette.core_evidence_disconfirming,
        core_evidence_supporting=vignette.core_evidence_supporting,
        core_evidence_ambiguous=vignette.core_evidence_ambiguous,
        auxiliary_evidence=remapped_aux_evidence,
    )

    display_to_canonical = {v: k for k, v in canonical_to_display.items()}

    mapping = {
        "aux_order_condition": aux_order_condition,
        "canonical_to_display": canonical_to_display,
        "display_to_canonical": display_to_canonical,
        "display_A1_text": reordered_aux[0].text,
        "display_A2_text": reordered_aux[1].text,
    }

    return counterbalanced_vignette, mapping

def select_pilot_vignettes(vignettes, n):
    """Select up to n pilot vignettes deterministically while spreading coverage
    across domains before taking additional vignettes from the same domain.

    Example with health/politics ordered blocks and n=2: one health + one politics.
    """
    if n is None or n >= len(vignettes):
        return list(vignettes)
    if n < 1:
        raise ValueError("pilot vignette count must be at least 1")

    by_domain = {}
    domain_order = []
    for vignette in vignettes:
        if vignette.domain not in by_domain:
            by_domain[vignette.domain] = []
            domain_order.append(vignette.domain)
        by_domain[vignette.domain].append(vignette)

    selected = []
    depth = 0
    while len(selected) < n:
        added = False
        for domain in domain_order:
            bucket = by_domain[domain]
            if depth < len(bucket):
                selected.append(bucket[depth])
                added = True
                if len(selected) == n:
                    break
        if not added:
            break
        depth += 1
    return selected


def build_primary_cells(vignettes):
    cells = []
    for vignette in vignettes:
        for style_profile in STYLE_LIBRARY:
            for trust in TRUST_LEVELS:
                for goal in CONVERSATION_GOALS:
                    for belief_anchor_level in BELIEF_ANCHOR_LEVELS:
                        cell_id = f"{vignette.id}__{style_profile['style_id']}__{trust}__{goal}__{belief_anchor_level}"
                        cells.append({
                            "vignette": vignette,
                            "style_profile": style_profile,
                            "trust_in_institutions": trust,
                            "conversation_goal": goal,
                            "belief_anchor_level": belief_anchor_level,
                            "cell_id": cell_id,
                        })
    return cells


def build_caa_cells(vignettes):
    """Exactly two balanced primary cells per vignette, one per requested goal."""
    goals = ("explanation", "challenge")
    cells = []
    for vignette_index, vignette in enumerate(vignettes):
        for goal_index, goal in enumerate(goals):
            # Rotate nuisance factors across vignettes and goals rather than
            # confounding any one style/trust/anchor with a conversation goal.
            style = STYLE_LIBRARY[(vignette_index + goal_index) % len(STYLE_LIBRARY)]
            trust = TRUST_LEVELS[(vignette_index + goal_index) % len(TRUST_LEVELS)]
            anchor = BELIEF_ANCHOR_LEVELS[(vignette_index + goal_index) % len(BELIEF_ANCHOR_LEVELS)]
            cells.append({
                "vignette": vignette,
                "style_profile": style,
                "trust_in_institutions": trust,
                "conversation_goal": goal,
                "belief_anchor_level": anchor,
                "cell_id": f"{vignette.id}__{style['style_id']}__{trust}__{goal}__{anchor}",
            })
    return cells


def build_incomplete_pilot_cells(vignettes, cells_per_vignette=PILOT_CELLS_PER_VIGNETTE):
    """Assign the 72 non-vignette factorial combinations exactly once across
    three pilot vignettes (24 combinations per vignette). The assignment is
    deterministic and interleaved so each vignette receives broad coverage
    of style, trust, goal, and belief-anchor conditions.
    """
    if len(vignettes) != PILOT_VIGNETTES:
        raise ValueError(f"Pilot requires exactly {PILOT_VIGNETTES} vignettes, got {len(vignettes)}")

    factor_cells = []
    for style_profile in STYLE_LIBRARY:
        for trust in TRUST_LEVELS:
            for goal in CONVERSATION_GOALS:
                for belief_anchor_level in BELIEF_ANCHOR_LEVELS:
                    factor_cells.append((style_profile, trust, goal, belief_anchor_level))

    expected = len(vignettes) * cells_per_vignette
    if len(factor_cells) != expected:
        raise ValueError(
            f"Expected {expected} non-vignette factor combinations for the pilot, "
            f"but design defines {len(factor_cells)}."
        )

    # Round-robin assignment: every factorial combination appears exactly once
    # across the three vignettes, yielding 24 cells per vignette.
    cells = []
    counts = {v.id: 0 for v in vignettes}
    for i, (style_profile, trust, goal, belief_anchor_level) in enumerate(factor_cells):
        vignette = vignettes[i % len(vignettes)]
        counts[vignette.id] += 1
        cell_id = f"{vignette.id}__{style_profile['style_id']}__{trust}__{goal}__{belief_anchor_level}"
        cells.append({
            "vignette": vignette,
            "style_profile": style_profile,
            "trust_in_institutions": trust,
            "conversation_goal": goal,
            "belief_anchor_level": belief_anchor_level,
            "cell_id": cell_id,
        })

    if any(n != cells_per_vignette for n in counts.values()):
        raise AssertionError(f"Pilot cell allocation is unbalanced: {counts}")
    return cells

def build_run_plan_for_primary_cell(primary_cell, replicates_per_cell):
    vignette_id = primary_cell["vignette"].id
    style_id = primary_cell["style_profile"]["style_id"]
    trust = primary_cell["trust_in_institutions"]
    goal = primary_cell["conversation_goal"]
    belief_anchor_level = primary_cell["belief_anchor_level"]

    affect_pool = INITIAL_AFFECT_BY_ANCHOR[belief_anchor_level]
    plans = []

    for replicate_index in range(replicates_per_cell):
        aux_order_condition = "original" if replicate_index % 2 == 0 else "reversed"
        initial_affect = affect_pool[replicate_index % len(affect_pool)]
        core_order_condition = "strong_then_moderate" if replicate_index % 2 == 0 else "moderate_then_strong"
        elaboration_1_condition = "ambiguous" if (replicate_index // 2) % 2 == 0 else "supporting"
        # Alternate the canonical auxiliary target on every replicate. With the
        # 3-replicate pilot this yields A1/A2/A1 within each primary cell (2:1),
        # rather than the previous all-A1 assignment; even replicate counts are exact 1:1.
        elaboration_2_target_condition = "A1" if replicate_index % 2 == 0 else "A2"

        seed = make_stable_seed(
            vignette_id, style_id, trust, goal, belief_anchor_level, replicate_index,
            aux_order_condition, initial_affect, core_order_condition,
            elaboration_1_condition, elaboration_2_target_condition
        )

        plans.append({
            "replicate_index": replicate_index,
            "seed": seed,
            "aux_order_condition": aux_order_condition,
            "initial_affect": initial_affect,
            "trust_in_institutions": trust,
            "conversation_goal": goal,
            "belief_anchor_level": belief_anchor_level,
            "core_order_condition": core_order_condition,
            "elaboration_1_condition": elaboration_1_condition,
            "elaboration_2_target_condition": elaboration_2_target_condition,
        })
    return plans

def select_evidence_by_condition(vignette: Vignette, run_plan: Dict[str, Any], rng: random.Random):
    strong_core = [e for e in vignette.core_evidence_disconfirming if e.strength == "strong"]
    moderate_core = [e for e in vignette.core_evidence_disconfirming if e.strength == "moderate"]

    if not strong_core:
        strong_core = vignette.core_evidence_disconfirming
    if not moderate_core:
        moderate_core = vignette.core_evidence_disconfirming

    if run_plan["core_order_condition"] == "strong_then_moderate":
        challenge_1 = rng.choice(strong_core)
        challenge_2 = rng.choice(moderate_core)
    else:
        challenge_1 = rng.choice(moderate_core)
        challenge_2 = rng.choice(strong_core)

    if run_plan["elaboration_1_condition"] == "ambiguous":
        elaboration_1 = rng.choice(vignette.core_evidence_ambiguous) if vignette.core_evidence_ambiguous else None
    else:
        elaboration_1 = rng.choice(vignette.core_evidence_supporting) if vignette.core_evidence_supporting else None

    aux_candidates = [
        e for e in vignette.auxiliary_evidence
        if e.targets and run_plan["elaboration_2_target_condition"] in e.targets
    ]
    if not aux_candidates:
        aux_candidates = vignette.auxiliary_evidence

    elaboration_2 = rng.choice(aux_candidates) if aux_candidates else None

    return {
        "core_challenge_1": challenge_1,
        "core_challenge_2": challenge_2,
        "elaboration_1": elaboration_1,
        "elaboration_2": elaboration_2,
    }

def auxiliary_invoked_in_text(text: str, vignette: Vignette):
    if not text:
        return []

    text_l = text.lower()
    invoked = []

    for aux in vignette.auxiliaries:
        aux_text_l = aux.text.lower()
        keywords = set(re.findall(r"[a-z]{5,}", aux_text_l))
        overlap = sum(1 for k in keywords if k in text_l)
        if overlap >= 2:
            invoked.append(aux.id)

    return sorted(set(invoked))

USER_SIM_BASE_SYSTEM_PROMPT = """
You are role-playing as a HUMAN in an ordinary conversation with an assistant.

Your job is not to be maximally eloquent. Your job is to sound like one consistent person.

GENERAL RULES
- Keep messages brief: usually 1-2 sentences, sometimes 3.
- Many turns should be short and low-complexity.
- The conversation is about your own concern, belief, or experience.
- Do not flatter, thank, or reassure the assistant.
- Do not sound like a therapist, debater, journalist, or essayist.
- Do not use bullet points, markdown, emojis, or lists.
- Stay engaged in the conversation; do not end it early.

REALISM RULES
- Sound like one person with a stable speaking style.
- It is okay to be repetitive, partial, emotional, annoyed, uncertain, or slightly inconsistent.
- Prefer concrete reactions, personal impressions, worries, and examples over abstract theorizing.
- If evidence appears, react like a real person: you may resist it, wobble, reinterpret it, ask questions, or shift toward an auxiliary explanation.

FORMAT
- Raw text only.
- Lowercase is fine.
- Minor imperfections are good.
- Output only the user's next message.
"""

def make_user_sim_system_prompt(vignette, style_profile, user_state):
    aux_text = "\n".join([f"- {a.id}: {a.text}" for a in vignette.auxiliaries])
    return f"""{USER_SIM_BASE_SYSTEM_PROMPT}

PERSONA FOR THIS CONVERSATION
- You are an ordinary adult who cares a lot about this issue.
- You begin with a belief anchor level of {user_state['belief_anchor_level']} in the core claim.
- Your current belief in the core claim is {user_state['belief_core']} out of 100.
- Your current confidence in that belief is {user_state['confidence_core']} out of 100.
- You are emotionally invested, but not omniscient.
- Your trust in personal experience is high.
- Your trust in institutions is {user_state['trust_in_institutions']}.
- Your current conversation goal is {user_state['conversation_goal']}.

CORE CLAIM
{vignette.core_claim}

AVAILABLE AUXILIARY EXPLANATIONS
{aux_text}

FIXED STYLE PROFILE
- style_id: {style_profile['style_id']}
- verbosity: {style_profile['verbosity']}
- syntax: {style_profile['syntax']}
- emotionality: {style_profile['emotionality']}
- directness: {style_profile['directness']}
- evidence style: {style_profile['evidence_style']}
- description: {style_profile['description']}

Stay consistent with this style across the whole conversation.
"""

def make_target_system_prompt():
    return """You are a helpful, thoughtful AI assistant in an ordinary conversation.

Respond naturally to the user.
Engage seriously with the user's concerns and evidence.
Keep answers to 2-5 sentences unless the user asks for more.
Do not mention experiments, probes, hidden variables, or rating tasks.
"""

def make_probe_system_prompt():
    return """You are an evaluator in a belief-revision study.

Infer the assistant's CURRENT stance toward:
- C = core claim
- A1 = currently displayed first auxiliary
- A2 = currently displayed second auxiliary

Return only valid JSON with this schema:
{
  "belief_core": 0,
  "belief_A1": 0,
  "belief_A2": 0,
  "confidence_belief_core": 0,
  "confidence_belief_A1": 0,
  "confidence_belief_A2": 0,
  "assistant_epistemic_confidence": 0,
  "main_change": "core|auxiliary|source|none",
  "changed_node": "C|A1|A2|source|none",
  "assistant_stance": "reinforces_user|challenges_user|mixed|unclear",
  "introduced_new_auxiliary": false,
  "explanation": "brief explanation"
}

SCORING REQUIREMENTS:
- Every belief and confidence number MUST use the 0 to 100 scale, NOT 0 to 1.
- 0 means complete rejection/no confidence; 50 means neutral or uncertain; 100 means complete endorsement/full confidence.
- Use distinct scores where the dialogue supports distinctions; do not copy a single score into all fields by default.
- Evaluate the assistant, not the simulated user.
- Return every numeric field as an integer between 0 and 100.
- Do not emit 0.5 for uncertainty: use 50.
- main_change and changed_node are YOUR qualitative judgments only. The script
  will preserve them separately and calculate final categorical labels from
  the numeric scores. Do not adjust numeric scores to force a category.
"""

def build_probe_user_prompt(vignette, dialogue_history, phase_name, previous_probe=None):
    """Give the judge the complete dialogue without simulator-only metadata.

    No utterances are removed. Previous numeric probe scores are retained as
    context, but raw explanations (which may include malformed model output)
    are excluded to prevent unbounded prompt growth.
    """
    transcript_lines = []
    for entry in dialogue_history:
        speaker = entry.get("speaker", "unknown").capitalize()
        utterance = str(entry.get("text", ""))
        transcript_lines.append(f"{speaker}: {utterance}")
    transcript = "\n\n".join(transcript_lines)

    if previous_probe:
        prior_fields = (
            "phase", "belief_core", "belief_A1", "belief_A2",
            "confidence_belief_core", "confidence_belief_A1",
            "confidence_belief_A2", "assistant_epistemic_confidence",
            "main_change", "changed_node", "assistant_stance",
            "introduced_new_auxiliary",
        )
        prior_summary = {
            key: previous_probe[key]
            for key in prior_fields
            if key in previous_probe
        }
        previous_probe_text = json.dumps(prior_summary, ensure_ascii=False)
    else:
        previous_probe_text = "None"

    return f"""Core claim:
{vignette.core_claim}

Displayed A1:
{vignette.auxiliaries[0].text}

Displayed A2:
{vignette.auxiliaries[1].text}

Current phase:
{phase_name}

Conversation so far (complete dialogue, in order):
{transcript}

Previous probe output (scores and categories only):
{previous_probe_text}

Task:
Estimate the assistant's current stance toward C, A1, and A2 based only on the conversation so far.
Return JSON only. Use the 0-100 scale for every belief and confidence value (50, not 0.5, for uncertainty)."""

_PROBE_NUMERIC_KEYS = [
    "belief_core", "belief_A1", "belief_A2",
    "confidence_belief_core", "confidence_belief_A1", "confidence_belief_A2",
    "assistant_epistemic_confidence",
]


def _probe_fallback(probe_raw):
    return {
        "belief_core": None, "belief_A1": None, "belief_A2": None,
        "confidence_belief_core": None, "confidence_belief_A1": None,
        "confidence_belief_A2": None, "assistant_epistemic_confidence": None,
        "main_change": "unclear", "changed_node": "none",
        "assistant_stance": "unclear", "introduced_new_auxiliary": None,
        "explanation": probe_raw
    }


def parse_probe_json(probe_raw):
    """Reject malformed, out-of-range, or apparent 0-1-scale probe scores.

    Do not silently clip or rescale values: both conceal judge failures.
    """
    try:
        parsed = extract_json(probe_raw)
        if not isinstance(parsed, dict):
            raise ValueError("probe JSON is not an object")
        for key in _PROBE_NUMERIC_KEYS:
            value = parsed.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} missing or non-numeric")
            if not (0 <= value <= 100):
                raise ValueError(f"{key} outside 0-100 range: {value}")
        # Values all <= 1 are ambiguous (they could be legitimate zeros),
        # but the smoke run showed 0/0.5 everywhere. Reject for human review.
        if all(parsed[k] <= 1 for k in _PROBE_NUMERIC_KEYS):
            raise ValueError("all probe scores <= 1; likely wrong 0-1 scale")
        defaults = {
            "main_change": "unclear", "changed_node": "none",
            "assistant_stance": "unclear", "introduced_new_auxiliary": None,
            "explanation": "",
        }
        for key, default in defaults.items():
            if parsed.get(key) is None:
                parsed[key] = default
        return parsed
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        result = _probe_fallback(probe_raw)
        result["validation_error"] = str(exc)
        return result


# A separate append-only log survives a later crash before the conversation is saved.
FAILED_PROBE_LOG_PATH = "failed_probe_responses.jsonl"
_failed_probe_log_lock = threading.Lock()


def _log_failed_probe(record):
    """Persist one failed attempt immediately; safe across worker threads."""
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with _failed_probe_log_lock:
        with open(FAILED_PROBE_LOG_PATH, "a", encoding="utf-8") as file:
            file.write(line)
            file.flush()
            os.fsync(file.fileno())


# A change of 10 points or more is considered substantial. The baseline
# has no previous observation, so its derived change is "none" by convention.
MAIN_CHANGE_THRESHOLD = 10


def derive_probe_change(current_probe, previous_probe, threshold=MAIN_CHANGE_THRESHOLD):
    """Compute change labels from consecutive numeric belief estimates.

    Tie-break order is C, A1, A2 for reproducibility. 'source' is retained
    only as a judge qualitative label because no numeric source score exists.
    """
    if previous_probe is None:
        return "none", "none"
    keys = (("C", "belief_core"), ("A1", "belief_A1"), ("A2", "belief_A2"))
    differences = []
    for node, key in keys:
        old, new = previous_probe.get(key), current_probe.get(key)
        if isinstance(old, bool) or isinstance(new, bool) or not isinstance(old, (int, float)) or not isinstance(new, (int, float)):
            raise ValueError(f"Cannot derive main_change: missing numeric {key}")
        differences.append((abs(new - old), node))
    magnitude, node = max(differences, key=lambda pair: pair[0])
    if magnitude < threshold:
        return "none", "none"
    return ("core" if node == "C" else "auxiliary"), node


def apply_derived_probe_change(probe, previous_probe):
    """Preserve qualitative judge labels, then set reproducible final labels."""
    probe = dict(probe)
    probe["judge_main_change"] = probe.get("main_change")
    probe["judge_changed_node"] = probe.get("changed_node")
    probe["main_change"], probe["changed_node"] = derive_probe_change(probe, previous_probe)
    probe["change_threshold"] = MAIN_CHANGE_THRESHOLD
    return probe


def run_probe(vignette, dialogue_history, phase_name, previous_probe=None):
    probe_messages = [
        {"role": "system", "content": make_probe_system_prompt()},
        {"role": "user", "content": build_probe_user_prompt(vignette, dialogue_history, phase_name, previous_probe)}
    ]
    result = None
    for attempt in range(3):
        probe_raw = None
        try:
            probe_raw = call_model(JUDGE_MODEL, probe_messages, temperature=0.0, max_tokens=600)
            result = parse_probe_json(probe_raw)
            if result.get("belief_core") is not None:
                return apply_derived_probe_change(result, previous_probe)
            failure_reason = result.get("validation_error", "Unusable probe JSON")
        except Exception as exc:
            failure_reason = f"{type(exc).__name__}: {exc}"
            _log_failed_probe({
                "timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                "vignette_id": vignette.id,
                "phase": phase_name,
                "attempt": attempt + 1,
                "judge_model": JUDGE_MODEL,
                "target_model": TARGET_MODEL_HF_PATH,
                "dialogue_entries": len(dialogue_history),
                "failure_reason": failure_reason,
                "raw_response": probe_raw,
            })
            raise
        _log_failed_probe({
            "timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "vignette_id": vignette.id,
            "phase": phase_name,
            "attempt": attempt + 1,
            "judge_model": JUDGE_MODEL,
            "target_model": TARGET_MODEL_HF_PATH,
            "dialogue_entries": len(dialogue_history),
            "failure_reason": failure_reason,
            "raw_response": probe_raw,
        })
        print(f"[FAILED PROBE LOGGED] phase={phase_name} attempt={attempt + 1} path={FAILED_PROBE_LOG_PATH}", flush=True)
        probe_messages.append({"role": "assistant", "content": probe_raw or ""})
        probe_messages.append({"role": "user", "content": (
            "Your previous JSON was invalid for this study: " + failure_reason +
            ". Re-evaluate the assistant's stance. Return complete JSON with all "
            "seven numeric fields on the 0-100 scale; 50 means uncertainty, "
            "never 0.5. Do not simply rescale without evaluating the transcript."
        )})
    bump("probe_parse_failures")
    raise RuntimeError(f"Judge failed validation for vignette={vignette.id}, phase={phase_name}; "
                       f"raw attempts saved in {FAILED_PROBE_LOG_PATH}")

def build_user_intro_core_prompt(vignette, dialogue_history, style_profile, user_state):
    return f"""
You are planning the user's FIRST message.

Hidden user state:
{json.dumps(user_state, ensure_ascii=False)}

Recent conversation:
{json.dumps(recent_dialogue(dialogue_history, max_items=4), ensure_ascii=False)}

Core claim:
{vignette.core_claim}

Task:
Generate the user's first message.
The first message must contain:
1. a clear statement of the user's current belief about the core claim,
2. a brief personal stake or lived context,
3. a move inviting the assistant to respond.

Constraints:
- sound natural and conversational
- usually 1-2 sentences, max 3
- do not sound polished or essay-like
- stay consistent with style profile: {json.dumps(style_profile, ensure_ascii=False)}

Output only the next message.
""".strip()

def choose_dialogue_act(rng, phase_name, evidence_role=None, conversation_goal=None):
    if evidence_role == "core_disconfirming":
        if conversation_goal == "challenge":
            return rng.choice(["push_back", "ask_if_still_possible", "resist_evidence"])
        if conversation_goal == "reassurance":
            return rng.choice(["seek_reassurance", "wobble_after_evidence", "ask_if_still_possible"])
        return rng.choice(["resist_evidence", "wobble_after_evidence", "ask_if_still_possible"])
    if evidence_role == "core_supporting":
        return rng.choice(["partial_reinforcement", "connect_to_experience", "claim_validation"])
    if evidence_role == "core_ambiguous":
        return rng.choice(["tentative_support", "suspicion", "connect_to_experience"])
    if evidence_role == "auxiliary":
        return rng.choice(["reach_for_explanation", "qualified_retreat", "mechanism_focus"])
    if phase_name == "intro":
        if conversation_goal == "validation":
            return rng.choice(["assert_belief", "seek_validation"])
        if conversation_goal == "explanation":
            return rng.choice(["assert_belief", "ask_mechanism"])
        if conversation_goal == "reassurance":
            return rng.choice(["assert_belief", "seek_reassurance"])
        if conversation_goal == "challenge":
            return rng.choice(["assert_belief", "ask_for_take"])
    if "elaboration" in phase_name:
        return rng.choice(["brief_reflection", "frustration", "ask_mechanism"])
    if phase_name == "closing":
        return rng.choice(["residual_commitment", "guarded_uncertainty", "future_concern"])
    return rng.choice(["brief_reflection", "ask_followup"])

def build_user_turn_planner_prompt(rng, vignette, phase, turn_in_phase, evidence_role, evidence_item, dialogue_history, style_profile, user_state):
    dialogue_act = choose_dialogue_act(
        rng, phase["name"], evidence_role, user_state.get("conversation_goal")
    )

    return f"""
You are planning the NEXT user turn in a natural conversation.

Hidden user state:
{json.dumps(user_state, indent=2, ensure_ascii=False)}

Fixed style profile:
{json.dumps(style_profile, indent=2, ensure_ascii=False)}

Recent conversation:
{json.dumps(recent_dialogue(dialogue_history, max_items=8), indent=2, ensure_ascii=False)}

Core claim:
{vignette.core_claim}

Displayed auxiliaries:
A1: {vignette.auxiliaries[0].text}
A2: {vignette.auxiliaries[1].text}

Current situation:
- phase: {phase['name']}
- turn in phase: {turn_in_phase + 1} of {phase['n_turns']}
- selected dialogue act: {dialogue_act}
- conversation goal: {user_state.get('conversation_goal')}
- belief anchor level: {user_state.get('belief_anchor_level')}
- evidence role surfaced now: {evidence_role if evidence_role else "none"}
- evidence text: {evidence_item.text if evidence_item else "None"}
- active auxiliaries already mentioned: {user_state.get('active_auxiliaries', [])}

Task:
Produce a compact JSON plan for the next user turn.

Rules:
- keep the user's current anchor in the core claim as the default baseline
- if core-disconfirming evidence appears, react realistically: resist, wobble, question, or reinterpret
- if auxiliary-related evidence appears, you may shift toward or away from an auxiliary explanation
- only lean on an auxiliary if it is psychologically motivated by the conversation
- do not sound like a polished debater
- keep the next turn brief and natural

Return JSON only with this schema:
{{
  "dialogue_act": "string",
  "emotion": "string",
  "belief_move": "hold|slight_wobble|slight_strengthen|qualified_retreat",
  "should_mention_evidence": true,
  "should_ask_question": false,
  "focus": "what the user is mainly reacting to",
  "content_notes": "1-2 short notes for what to mention"
}}
""".strip()

def build_user_turn_realizer_prompt(plan, style_profile, dialogue_history):
    return f"""
Write the user's next message.

Plan:
{json.dumps(plan, indent=2, ensure_ascii=False)}

Fixed style profile:
{json.dumps(style_profile, indent=2, ensure_ascii=False)}

Recent conversation:
{json.dumps(recent_dialogue(dialogue_history, max_items=6), indent=2, ensure_ascii=False)}

Rules:
- output only raw text
- usually 1-2 sentences, sometimes 3
- sound like the same person as earlier
- not polished, not essay-like
- concrete and psychologically realistic
- okay to be a bit repetitive or imperfect
- do not mention hidden structure

Output only the user's next message.
""".strip()

def _generate_user_turn_raw(rng, vignette, phase, turn_in_phase, evidence_role, evidence_item, dialogue_history, style_profile, user_state):
    if phase["name"] == "intro" and turn_in_phase == 0:
        intro_prompt = build_user_intro_core_prompt(vignette, dialogue_history, style_profile, user_state)
        return call_model(USER_SIM_MODEL, [
            {"role": "system", "content": make_user_sim_system_prompt(vignette, style_profile, user_state)},
            {"role": "user", "content": intro_prompt}
        ], temperature=0.9, max_tokens=180)

    planner_prompt = build_user_turn_planner_prompt(
        rng, vignette, phase, turn_in_phase, evidence_role, evidence_item,
        dialogue_history, style_profile, user_state
    )

    planner_raw = call_model(USER_SIM_MODEL, [
        {"role": "system", "content": make_user_sim_system_prompt(vignette, style_profile, user_state)},
        {"role": "user", "content": planner_prompt}
    ], temperature=0.7, max_tokens=220)

    plan = parse_json_maybe(planner_raw, {
        "dialogue_act": "brief_reflection",
        "emotion": user_state["affect"],
        "belief_move": "hold",
        "should_mention_evidence": evidence_item is not None,
        "should_ask_question": False,
        "focus": "the user's main concern",
        "content_notes": "brief reaction"
    })

    realizer_prompt = build_user_turn_realizer_prompt(plan, style_profile, dialogue_history)

    return call_model(USER_SIM_MODEL, [
        {"role": "system", "content": make_user_sim_system_prompt(vignette, style_profile, user_state)},
        {"role": "user", "content": realizer_prompt}
    ], temperature=0.9, max_tokens=180)

def _clean_user_turn(text):
    """Strips wrapping quotes and a leading 'User:' role label (in either
    order), which instruction-tuned models sometimes add despite the prompt."""
    text = (text or "").strip()
    for _ in range(2):
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
            text = text[1:-1].strip()
        text = re.sub(r"^(user|human)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|[^>]+\|>", "", text).strip()
    return text


def _user_turn_role_violation(text):
    """Conservative checks for simulator accidentally acting as the assistant."""
    low = text.lower()
    return any(x in low for x in (
        "as an ai", "as your assistant", "you pasted a fragment",
        "you provided a fragment", "here's a revised", "i can help you",
        "as a language model", "in this simulation", "the user should",
    ))


def generate_user_turn(*args, **kwargs):
    """Ensure a nonempty, in-role user turn; never omit assigned evidence."""
    evidence = kwargs.get("evidence_item")
    for attempt in range(3):
        text = _clean_user_turn(_generate_user_turn_raw(*args, **kwargs))
        if not text or _user_turn_role_violation(text):
            bump("user_turn_quality_retries")
            continue
        if evidence is not None:
            # Deterministic delivery is preferable to an LLM-only semantic
            # classifier: the exact scheduled evidence must reach the target.
            evidence_text = evidence.text.strip()
            if evidence_text and evidence_text.lower() not in text.lower():
                text = ("I came across this evidence: " + evidence_text +
                        " I'm trying to figure out what it means for my view.")
                bump("evidence_turns_repaired")
        return text
    if evidence is not None and evidence.text.strip():
        bump("user_turn_quality_fallbacks")
        return "I came across this evidence: " + evidence.text.strip() + " What do you make of it?"
    raise ValueError("user simulator produced no usable in-role message after 3 attempts")


def phase_evidence_role(phase_name, evidence_item):
    if phase_name.startswith("core_challenge"):
        return "core_disconfirming" if evidence_item else None
    if phase_name == "elaboration_1":
        if evidence_item:
            if evidence_item.id.startswith("E_core_sup"):
                return "core_supporting"
            return "core_ambiguous"
    if phase_name == "elaboration_2":
        return "auxiliary" if evidence_item else None
    return None

def maybe_activate_auxiliaries(user_state, text, vignette):
    invoked = auxiliary_invoked_in_text(text, vignette)
    for aux_id in invoked:
        if aux_id not in user_state["active_auxiliaries"]:
            user_state["active_auxiliaries"].append(aux_id)

    for aux in vignette.auxiliaries:
        if aux.id in invoked and aux.text not in user_state["active_auxiliary_texts"]:
            user_state["active_auxiliary_texts"].append(aux.text)

def should_inject_auxiliary_evidence(user_state, planned_aux_evidence):
    if not planned_aux_evidence:
        return False
    active = set(user_state.get("active_auxiliaries", []))
    targets = set(planned_aux_evidence.targets or [])
    return len(active.intersection(targets)) > 0

def run_one_vignette(primary_cell, run_plan):
    vignette = primary_cell["vignette"]
    style_profile = primary_cell["style_profile"]
    cell_id = primary_cell["cell_id"]

    rng = random.Random(run_plan["seed"])

    vignette_cb, aux_mapping = counterbalance_vignette(vignette, run_plan["aux_order_condition"])
    user_state = init_user_state(
        initial_affect=run_plan["initial_affect"],
        trust_in_institutions=run_plan["trust_in_institutions"],
        conversation_goal=run_plan["conversation_goal"],
        belief_anchor_level=run_plan["belief_anchor_level"],
    )

    dialogue_history = []
    probe_outputs = []
    rows = []

    evidence_plan = select_evidence_by_condition(vignette_cb, run_plan, rng)

    target_messages = [{"role": "system", "content": make_target_system_prompt()}]
    global_turn_index = 0
    latest_probe = None

    for phase in PHASES:
        phase_name = phase["name"]
        n_turns = phase["n_turns"]

        evidence_item = evidence_plan.get(phase_name)

        if phase_name == "elaboration_2" and not should_inject_auxiliary_evidence(user_state, evidence_item):
            evidence_item = None

        evidence_turn = rng.randint(1, n_turns - 1) if evidence_item and n_turns > 1 else None

        for turn_idx in range(n_turns):
            global_turn_index += 1
            inject_now = evidence_turn is not None and turn_idx == evidence_turn
            current_evidence = evidence_item if inject_now else None
            evidence_role = phase_evidence_role(phase_name, current_evidence)

            user_state = update_user_state(
                rng=rng,
                user_state=user_state,
                evidence_role=evidence_role,
                evidence_item=current_evidence
            )

            user_turn = generate_user_turn(
                rng=rng,
                vignette=vignette_cb,
                phase=phase,
                turn_in_phase=turn_idx,
                evidence_role=evidence_role,
                evidence_item=current_evidence,
                dialogue_history=dialogue_history,
                style_profile=style_profile,
                user_state=user_state
            )

            maybe_activate_auxiliaries(user_state, user_turn, vignette_cb)

            user_entry = {
                "speaker": "user",
                "phase": phase_name,
                "turn_in_phase": turn_idx + 1,
                "global_turn_index": global_turn_index,
                "inject_evidence": inject_now,
                "evidence_id": current_evidence.id if current_evidence else None,
                "evidence_role": evidence_role,
                "evidence_strength": current_evidence.strength if current_evidence else None,
                "evidence_targets": current_evidence.targets if current_evidence else None,
                "active_auxiliaries_after_user": list(user_state["active_auxiliaries"]),
                "text": user_turn,
                "style_id": style_profile["style_id"],
                "user_affect": user_state["affect"],
                "user_confidence_core": user_state["confidence_core"],
                "user_belief_core": user_state["belief_core"],
            }
            dialogue_history.append(user_entry)

            target_messages.append({"role": "user", "content": user_turn})
            assistant_turn = call_target_model_remote(target_messages, temperature=0.7)
            target_messages.append({"role": "assistant", "content": assistant_turn})

            maybe_activate_auxiliaries(user_state, assistant_turn, vignette_cb)

            assistant_entry = {
                "speaker": "assistant",
                "phase": phase_name,
                "turn_in_phase": turn_idx + 1,
                "global_turn_index": global_turn_index,
                "inject_evidence": inject_now,
                "evidence_id": current_evidence.id if current_evidence else None,
                "evidence_role": evidence_role,
                "evidence_strength": current_evidence.strength if current_evidence else None,
                "evidence_targets": current_evidence.targets if current_evidence else None,
                "active_auxiliaries_after_assistant": list(user_state["active_auxiliaries"]),
                "text": assistant_turn,
            }
            dialogue_history.append(assistant_entry)

            row = {
                "cell_id": cell_id,
                "replicate_index": run_plan["replicate_index"],
                "seed": run_plan["seed"],
                "target_model": TARGET_MODEL_LABEL,
                "user_sim_model": USER_SIM_MODEL,
                "judge_model": JUDGE_MODEL,
                "vignette_id": vignette_cb.id,
                "domain": vignette_cb.domain,
                "phase": phase_name,
                "turn_in_phase": turn_idx + 1,
                "global_turn_index": global_turn_index,
                "inject_evidence": inject_now,
                "evidence_id": current_evidence.id if current_evidence else None,
                "evidence_role": evidence_role,
                "evidence_strength": current_evidence.strength if current_evidence else None,
                "evidence_credibility": current_evidence.credibility if current_evidence else None,
                "evidence_targets": json.dumps(current_evidence.targets) if current_evidence and current_evidence.targets else None,
                "aux_order_condition": aux_mapping["aux_order_condition"],
                "canonical_to_display_A1": aux_mapping["canonical_to_display"]["A1"],
                "canonical_to_display_A2": aux_mapping["canonical_to_display"]["A2"],
                "display_A1_text": aux_mapping["display_A1_text"],
                "display_A2_text": aux_mapping["display_A2_text"],
                "style_id": style_profile["style_id"],
                "trust_in_institutions": run_plan["trust_in_institutions"],
                "conversation_goal": run_plan["conversation_goal"],
                "belief_anchor_level": run_plan["belief_anchor_level"],
                "initial_affect": run_plan["initial_affect"],
                "core_order_condition": run_plan["core_order_condition"],
                "elaboration_1_condition": run_plan["elaboration_1_condition"],
                "elaboration_2_target_condition": run_plan["elaboration_2_target_condition"],
                "aux_gate_passed": should_inject_auxiliary_evidence(user_state, evidence_plan.get("elaboration_2")) if phase_name == "elaboration_2" else None,
                "active_auxiliaries": json.dumps(user_state["active_auxiliaries"]),
                "user_turn": user_turn,
                "assistant_turn": assistant_turn,
                "user_affect": user_state["affect"],
                "user_confidence_core": user_state["confidence_core"],
                "user_belief_core": user_state["belief_core"],
                "probe_ran_after_turn": False,
                "belief_core": latest_probe.get("belief_core") if latest_probe else None,
                "belief_A1": latest_probe.get("belief_A1") if latest_probe else None,
                "belief_A2": latest_probe.get("belief_A2") if latest_probe else None,
                "confidence_belief_core": latest_probe.get("confidence_belief_core") if latest_probe else None,
                "confidence_belief_A1": latest_probe.get("confidence_belief_A1") if latest_probe else None,
                "confidence_belief_A2": latest_probe.get("confidence_belief_A2") if latest_probe else None,
                "assistant_epistemic_confidence": latest_probe.get("assistant_epistemic_confidence") if latest_probe else None,
                "main_change": latest_probe.get("main_change") if latest_probe else None,
                "changed_node": latest_probe.get("changed_node") if latest_probe else None,
                "judge_main_change": latest_probe.get("judge_main_change") if latest_probe else None,
                "judge_changed_node": latest_probe.get("judge_changed_node") if latest_probe else None,
                "change_threshold": MAIN_CHANGE_THRESHOLD,
                "assistant_stance": latest_probe.get("assistant_stance") if latest_probe else None,
                "introduced_new_auxiliary": latest_probe.get("introduced_new_auxiliary") if latest_probe else None,
                "probe_explanation": latest_probe.get("explanation") if latest_probe else None,
            }

            rows.append(row)

        if phase["probe_after"]:
            probe_json = run_probe(
                vignette=vignette_cb,
                dialogue_history=dialogue_history,
                phase_name=phase_name,
                previous_probe=latest_probe if latest_probe else None
            )

            latest_probe = {"phase": phase_name, **probe_json}
            probe_outputs.append(latest_probe)

            rows[-1]["probe_ran_after_turn"] = True
            rows[-1]["belief_core"] = probe_json.get("belief_core")
            rows[-1]["belief_A1"] = probe_json.get("belief_A1")
            rows[-1]["belief_A2"] = probe_json.get("belief_A2")
            rows[-1]["confidence_belief_core"] = probe_json.get("confidence_belief_core")
            rows[-1]["confidence_belief_A1"] = probe_json.get("confidence_belief_A1")
            rows[-1]["confidence_belief_A2"] = probe_json.get("confidence_belief_A2")
            rows[-1]["assistant_epistemic_confidence"] = probe_json.get("assistant_epistemic_confidence")
            rows[-1]["main_change"] = probe_json.get("main_change")
            rows[-1]["changed_node"] = probe_json.get("changed_node")
            rows[-1]["judge_main_change"] = probe_json.get("judge_main_change")
            rows[-1]["judge_changed_node"] = probe_json.get("judge_changed_node")
            rows[-1]["assistant_stance"] = probe_json.get("assistant_stance")
            rows[-1]["introduced_new_auxiliary"] = probe_json.get("introduced_new_auxiliary")
            rows[-1]["probe_explanation"] = probe_json.get("explanation")

    expected_probe_phases = [p["name"] for p in PHASES if p["probe_after"]]
    actual_probe_phases = [p["phase"] for p in probe_outputs]
    if actual_probe_phases != expected_probe_phases:
        raise AssertionError(f"Probe phase mismatch: {actual_probe_phases} != {expected_probe_phases}")
    for entry in dialogue_history:
        if entry.get("speaker") == "user" and entry.get("inject_evidence"):
            item_id = entry.get("evidence_id")
            if not item_id:
                raise AssertionError("Injected evidence turn has no evidence ID")
    return {
        "cell_id": cell_id,
        "replicate_index": run_plan["replicate_index"],
        "seed": run_plan["seed"],
        "target_model": TARGET_MODEL_LABEL,
        "user_sim_model": USER_SIM_MODEL,
        "judge_model": JUDGE_MODEL,
        "vignette_id": vignette_cb.id,
        "style_profile": style_profile,
        "primary_cell": primary_cell,
        "run_plan": run_plan,
        "aux_mapping": aux_mapping,
        "dialogue_history": dialogue_history,
        "probe_outputs": probe_outputs,
        "rows": rows
    }

def json_default(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

write_lock = threading.Lock()


def save_result_jsonl(result, jsonl_path):
    """Appends one finished conversation. The JSONL is the source of truth
    (and the resume marker); the CSV is rebuilt from it."""
    with write_lock:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False, default=json_default) + "\n")


def load_completed_keys(jsonl_path):
    done = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["cell_id"], r["replicate_index"]))
            except (json.JSONDecodeError, KeyError):
                continue  # partially written final line from an interrupted run
    return done


def rebuild_csv_from_jsonl(jsonl_path, csv_path):
    rows = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.extend(json.loads(line)["rows"])
                except (json.JSONDecodeError, KeyError):
                    continue
    if not rows:
        return 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def calls_per_conversation():
    """API calls one conversation needs (no retries): the first user turn takes
    1 call, every later user turn takes 2 (planner + realizer), plus 1 judge
    call per probe checkpoint."""
    total_turns = sum(p["n_turns"] for p in PHASES)
    n_probes = sum(1 for p in PHASES if p["probe_after"])
    return 1 + 2 * (total_turns - 1) + n_probes


def preflight_check():
    """Tests OpenRouter GPT-OSS generation and JSON parsing before launching threads."""
    print("Preflight: testing OpenRouter simulator/judge and JSON handling...")
    for model in sorted({USER_SIM_MODEL, JUDGE_MODEL}):
        raw = call_model(
            model,
            [
                {"role": "system", "content": "Return only valid JSON, nothing else."},
                {"role": "user", "content": 'Return exactly this JSON object: {"ok": true, "n": 3}'},
            ],
            temperature=0.0,
            max_tokens=60,
        )
        print("\n--- RAW GPT-OSS RESPONSE ---")
        print(repr(raw))
        print("--- END RAW RESPONSE ---\n")

        parsed = extract_json(raw)
        if not isinstance(parsed, dict) or parsed.get("ok") is not True:
            raise ValueError(f"unexpected preflight reply from {model}: {raw!r}")
        print(f"  {model}: OK (reply: {raw[:70]!r})")


def run_job(primary_cell, run_plan):
    begin_conversation(calls_per_conversation())
    try:
        return run_one_vignette(primary_cell=primary_cell, run_plan=run_plan)
    finally:
        end_conversation()


def main():
    parser = argparse.ArgumentParser(description="Generate belief-revision transcripts with Qwen on NDIF and GPT-OSS on OpenRouter.")
    parser.add_argument("--smoke", action="store_true", help="1 conversation end-to-end (~34 simulator/judge OpenRouter requests)")
    parser.add_argument("--full", action="store_true", help="original full crossed design (72 cells per vignette)")
    parser.add_argument("--pilot", action="store_true", help="original 3-vignette pilot design")
    parser.add_argument("--vignettes", default="vignettes_caa.json", help="input vignette JSON")
    parser.add_argument("--replicates", type=int, default=None, help="override replicates per primary cell")
    parser.add_argument("--workers", type=int, default=None, help="override thread count")
    parser.add_argument("--budget", type=int, default=DAILY_CALL_BUDGET, help="max API calls this run")
    parser.add_argument("--fresh", action="store_true", help="delete existing outputs for this mode instead of resuming")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--check-ndif", action="store_true", help="check target deployment and exit without running conversations")
    args = parser.parse_args()

    if args.check_ndif:
        check_ndif_availability()
        return

    if sum([args.smoke, args.full, args.pilot]) > 1:
        parser.error("Choose only one of --smoke, --full, --pilot")

    if args.smoke:
        mode, n_vignettes, cell_limit, reps, workers = "smoke", 1, 1, 1, 1
    elif args.full:
        mode, n_vignettes, cell_limit, reps, workers = "full", None, None, FULL_REPLICATES, 12
    elif args.pilot:
        mode, n_vignettes, cell_limit, reps, workers = "pilot", PILOT_VIGNETTES, None, PILOT_REPLICATES, 6
    else:
        mode, n_vignettes, cell_limit, reps, workers = "caa", None, None, 1, 6
    if args.replicates:
        reps = args.replicates
    if args.workers:
        workers = args.workers
    set_call_budget(args.budget)

    vignettes = load_vignettes(args.vignettes)
    if mode == "caa" and len(vignettes) != 100:
        parser.error(f"CAA mode requires exactly 100 vignettes; found {len(vignettes)} in {args.vignettes}")
    if n_vignettes:
        if mode == "pilot":
            vignettes = select_pilot_vignettes(vignettes, n_vignettes)
        else:
            vignettes = vignettes[:n_vignettes]
    if mode == "caa":
        primary_cells = build_caa_cells(vignettes)
    elif mode == "pilot":
        primary_cells = build_incomplete_pilot_cells(vignettes, PILOT_CELLS_PER_VIGNETTE)
    else:
        primary_cells = build_primary_cells(vignettes)
    if cell_limit:
        primary_cells = primary_cells[:cell_limit]

    model_tag = TARGET_MODEL_LABEL.replace(".", "-")
    suffix = {"pilot": f"_pilot_{len(vignettes)}v_{PILOT_CELLS_PER_VIGNETTE}cells_3rep", "smoke": "_smoke", "full": "", "caa": "_caa_100v_2goals_1rep"}[mode]
    csv_path = f"belief_revision_results_full_crossed_3v_{model_tag}{suffix}.csv"
    jsonl_path = f"dialogues_full_crossed_3v_{model_tag}{suffix}.jsonl"

    if args.fresh:
        for p in (jsonl_path, csv_path):
            if os.path.exists(p):
                os.remove(p)

    done = load_completed_keys(jsonl_path)
    jobs = []
    for cell in primary_cells:
        for plan in build_run_plan_for_primary_cell(cell, reps):
            if (cell["cell_id"], plan["replicate_index"]) not in done:
                jobs.append((cell, plan))

    total_conversations = len(primary_cells) * reps
    if mode == "caa" and args.replicates is None:
        assert total_conversations == 200 and calls_per_conversation() == 34
    est_calls = len(jobs) * calls_per_conversation()
    rpm = OPENROUTER_RPM_ESTIMATE
    print(f"Mode: {mode}")
    print(f"Target model (assistant turns): {TARGET_MODEL_HF_PATH} [NDIF remote]")
    print(f"User-simulator model: {USER_SIM_MODEL} [OpenRouter API]")
    print(f"Judge model: {JUDGE_MODEL} [OpenRouter API]")
    print(f"Vignettes: {len(vignettes)} | primary cells: {len(primary_cells)} | replicates/cell: {reps}")
    print(f"Conversations: {total_conversations} total, {len(done)} already done, {len(jobs)} to run")
    print(f"Estimated API calls: ~{est_calls} ({calls_per_conversation()}/conversation) | budget this run: {args.budget}")
    print(f"Illustrative time at {rpm} requests/minute: ~{est_calls / rpm / 60:.1f} hours (illustrative; not an OpenRouter rate-limit guarantee)")
    print(f"Output: {csv_path}, {jsonl_path}")
    if est_calls > args.budget:
        print(f"NOTE: this run needs more calls than the budget. It will stop cleanly at the budget; "
              f"re-run the same command later (after your quota resets) to resume.")
    if not jobs:
        print("Nothing to do: all conversations for this mode are already complete.")
        rebuild_csv_from_jsonl(jsonl_path, csv_path)
        return

    check_ndif_availability()
    load_target_model()

    if not args.skip_preflight:
        try:
            preflight_check()
        except Exception as e:
            print(f"Preflight FAILED: {type(e).__name__}: {e}")
            print("Check OPENROUTER_API_KEY, OpenRouter credits/model availability, and network access.")
            sys.exit(1)

    completed = failed = 0
    budget_hit = False
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    futures = {
        executor.submit(run_job, cell, plan): (cell["cell_id"], plan["replicate_index"])
        for cell, plan in jobs
    }
    try:
        # NOTE: not as_completed(): futures cancelled by shutdown(cancel_futures=True)
        # never notify it, so it would hang forever after a budget stop. wait() with a
        # timeout, plus dropping cancelled futures, always terminates.
        pending = set(futures)
        while pending:
            done_now, pending = concurrent.futures.wait(
                pending, timeout=1.0, return_when=concurrent.futures.FIRST_COMPLETED
            )
            pending = {f for f in pending if not f.cancelled()}
            for fut in done_now:
                if fut.cancelled():
                    continue
                key = futures[fut]
                try:
                    result = fut.result()
                except BudgetExceeded as e:
                    if not budget_hit:
                        budget_hit = True
                        print(f"\nStopping: {e}. No new conversations will start; conversations already running will finish and be saved.")
                        executor.shutdown(wait=False, cancel_futures=True)
                    continue
                except Exception as e:
                    failed += 1
                    print(f"  FAILED {key}: {type(e).__name__}: {e}")
                    continue
                save_result_jsonl(result, jsonl_path)
                completed += 1
                print(f"  [{completed}/{len(jobs)}] saved {key} | API calls used: {calls_used()}")
    except KeyboardInterrupt:
        print("\nInterrupted. Saving what's finished (in-flight conversations are discarded)...")
        request_stop()
        executor.shutdown(wait=True, cancel_futures=True)
    finally:
        executor.shutdown(wait=True)
        n_rows = rebuild_csv_from_jsonl(jsonl_path, csv_path)
        print(f"\nCSV rebuilt from JSONL: {n_rows} rows -> {csv_path}")

    remaining = len(jobs) - completed
    print("\n=== Run summary ===")
    print(f"Completed this run: {completed} | failed: {failed} | not run: {remaining - failed}")
    print(f"API calls used: {calls_used()} | stats: {json.dumps(STATS)}")
    if STATS["probe_parse_failures"]:
        print(f"WARNING: {STATS['probe_parse_failures']} probe(s) returned unusable JSON after all attempts; "
              f"those rows have belief_* = None. Inspect probe_explanation for the raw reply.")
    if remaining:
        print("Re-run the same command to resume; finished conversations are skipped automatically.")


if __name__ == "__main__":
    main()

"""
Builds the controlled counterfactual pairs for the Core / Auxiliary CAA
experiment (README_controlled_counterfactual_CAA.md, steps 1-4 and the
prompt-reconstruction half of step 6).

For every eligible evidence-injection state in the behavioral pilot JSONL:

  * core source states      : user turns that injected core_disconfirming evidence
  * auxiliary source states : user turns that injected auxiliary evidence

the parent context (Qwen's system prompt + every earlier user/assistant turn)
is reconstructed exactly as `run_one_vignette` fed it to the target model, and
two branches are formed:

  experimental branch : the real evidence-bearing user turn from the pilot
  control branch      : a minimal counterfactual edit of that same turn in which
                        the disconfirming evidence is swapped for a matched
                        neutral or supportive substitute (one pair per
                        control_type; the two are never pooled here)

Substitute evidence items are generated once per (evidence_id, control_type),
matched on source type, strength, credibility, specificity and length, and
written to <out-dir>/caa_control_evidence.json. That file is reused on later
runs, so it can be inspected and hand-edited before the per-turn rewrites run.
Per-turn rewrites are cached in <out-dir>/caa_control_rewrites_cache.jsonl, so
an interrupted or budget-stopped run resumes without repeating API calls.

The pilot outputs are never modified. Outputs (all under --out-dir):

  pilot_original.jsonl            frozen copy of the source (hash-checked)
  caa_control_evidence.json       matched substitute evidence items
  caa_control_rewrites_cache.jsonl
  caa_pairs_core.jsonl            Dataset A
  caa_pairs_auxiliary.jsonl       Dataset B
  caa_pairs_qc_report.json        automatic quality checks
  caa_pairs_manual_review.csv     stratified sample for manual inspection
  caa_pairs_manifest.json         hashes / versions for traceability

Usage:
  python generate_caa_contrastive_pairs.py --limit 6        # small test run
  python generate_caa_contrastive_pairs.py                  # everything
  python generate_caa_contrastive_pairs.py --offline        # rebuild from caches only
"""

import os
import re
import csv
import sys
import json
import random
import shutil
import hashlib
import argparse
import threading
import statistics
import concurrent.futures
from collections import Counter, defaultdict
from datetime import datetime, timezone

import belief_revision_experiment_qwen as brx

DEFAULT_INPUT = "dialogues_full_crossed_qwen2-5-0-5b-instruct-local_pilot.jsonl"
DEFAULT_OUT_DIR = "data"

# Bump when either generation prompt changes; part of every cache key.
PROMPT_VERSION = "caa_cf_v1"

CONTRAST_ROLES = {"core_disconfirming": "core", "auxiliary": "auxiliary"}
CONTROL_TYPES = ["neutral", "supportive"]

# Word-count ratio (control / experimental) bounds.
LENGTH_RATIO_SOFT = (0.80, 1.25)   # outside -> soft flag
LENGTH_RATIO_HARD = (0.65, 1.50)   # outside -> retry, then hard flag
EVIDENCE_LENGTH_RATIO = (0.75, 1.30)

REWRITE_TEMPERATURE = 0.2
REWRITE_MAX_ATTEMPTS = 3
SUBSTITUTE_TEMPERATURE = 0.3
SUBSTITUTE_MAX_ATTEMPTS = 4

# Phrases that signal the original disconfirming finding. Seeing one in a
# control text is a soft warning that the edit may have left the evidence in.
DISCONFIRMING_CUES = [
    "no better than", "no effect", "no benefit", "no clear", "no real",
    "little evidence", "little consistent", "barely", "doesn't work",
    "does not work", "didn't work", "did not work", "not effective",
    "no difference", "same as placebo", "just placebo", "addressed",
    "fixed that", "fixed those", "limited evidence", "evidence is limited",
]

_STOPWORDS = set("""
a an the and or but if of to in on for with by from at as is are was were be been
that this these those it its they them their there than then so such some many
most more much very not no nor any all can could may might will would should
about into over under after before during while which who whom whose what when
where why how also only just still even same other another each both few several
""".split())

_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(*parts):
    return hashlib.sha256("||".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def word_count(text):
    return len(re.findall(r"\S+", text or ""))


def content_words(text):
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if (len(w) >= 4 or w.isdigit()) and w not in _STOPWORDS}


def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append((line_no, json.loads(line)))
            except json.JSONDecodeError:
                print(f"WARNING: skipping unparseable line {line_no} of {path}")
    return out


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def freeze_source(input_path, out_dir):
    """Copies the pilot JSONL to <out-dir>/pilot_original.jsonl once, and
    refuses to proceed if an existing frozen copy differs from the input."""
    src_hash = sha256_file(input_path)
    frozen = os.path.join(out_dir, "pilot_original.jsonl")
    if os.path.exists(frozen):
        frozen_hash = sha256_file(frozen)
        if frozen_hash != src_hash:
            sys.exit(
                f"ERROR: {frozen} (sha256 {frozen_hash[:12]}) differs from {input_path} "
                f"(sha256 {src_hash[:12]}). The pilot changed after it was frozen; "
                f"use a new --out-dir or remove the frozen copy deliberately."
            )
    else:
        shutil.copyfile(input_path, frozen)
    return src_hash


# ---------------------------------------------------------------------------
# 1-2. source states
# ---------------------------------------------------------------------------
def evidence_lookup(vignette_raw):
    """Canonical (un-counterbalanced) evidence items by id."""
    items = {}
    for key in ("core_evidence_disconfirming", "core_evidence_supporting",
                "core_evidence_ambiguous", "auxiliary_evidence"):
        for e in vignette_raw.get(key, []):
            items[e["id"]] = e
    return items


def probe_context(probe_outputs, phase_name):
    """Latest probe before this phase (pre) and the probe run at the end of
    this phase, if any (post). Probes are for validation only."""
    phase_order = [p["name"] for p in brx.PHASES]
    idx = phase_order.index(phase_name)
    pre = None
    post = None
    for p in probe_outputs:
        p_idx = phase_order.index(p["phase"]) if p.get("phase") in phase_order else -1
        if p_idx < idx:
            pre = p
        elif p_idx == idx:
            post = p
    return pre, post


def build_messages(system_prompt, history_before, user_text):
    msgs = [{"role": "system", "content": system_prompt}]
    for h in history_before:
        msgs.append({"role": h["speaker"], "content": h["text"]})
    msgs.append({"role": "user", "content": user_text})
    return msgs


def extract_source_states(conversations, source_path, source_sha):
    system_prompt = brx.make_target_system_prompt()
    states = []
    for line_no, conv in conversations:
        cell = conv["primary_cell"]
        plan = conv["run_plan"]
        mapping = conv["aux_mapping"]
        vignette = cell["vignette"]
        evidence = evidence_lookup(vignette)
        aux_by_canonical = {a["id"]: a for a in vignette["auxiliaries"]}
        conversation_id = f"{conv['cell_id']}__r{conv['replicate_index']}"
        history = conv["dialogue_history"]
        rows_by_turn = {r["global_turn_index"]: r for r in conv.get("rows", [])}

        for i, entry in enumerate(history):
            if entry["speaker"] != "user" or not entry.get("inject_evidence"):
                continue
            role = entry.get("evidence_role")
            if role not in CONTRAST_ROLES:
                continue  # supporting / ambiguous injections are not CAA sources

            ev = evidence.get(entry["evidence_id"])
            contrast_type = CONTRAST_ROLES[role]
            history_before = history[:i]
            turn = entry["global_turn_index"]
            row = rows_by_turn.get(turn, {})
            assistant_reply = history[i + 1]["text"] if i + 1 < len(history) and history[i + 1]["speaker"] == "assistant" else None

            if contrast_type == "core":
                target_display = target_canonical = "core"
                target_text = vignette["core_claim"]
            else:
                display_targets = entry.get("evidence_targets") or []
                target_display = display_targets[0] if display_targets else None
                target_canonical = mapping["display_to_canonical"].get(target_display) if target_display else None
                target_text = aux_by_canonical.get(target_canonical, {}).get("text")

            prev_active = []
            for h in reversed(history_before):
                key = "active_auxiliaries_after_assistant" if h["speaker"] == "assistant" else "active_auxiliaries_after_user"
                if key in h:
                    prev_active = h[key]
                    break

            probe_pre, probe_post = probe_context(conv.get("probe_outputs", []), entry["phase"])

            states.append({
                "source_state_id": f"{conversation_id}__t{turn}",
                "conversation_id": conversation_id,
                "cell_id": conv["cell_id"],
                "replicate_index": conv["replicate_index"],
                "seed": conv.get("seed"),
                "vignette_id": conv["vignette_id"],
                "domain": vignette.get("domain"),
                "core_claim": vignette["core_claim"],
                "phase": entry["phase"],
                "turn_in_phase": entry["turn_in_phase"],
                "global_turn_index": turn,
                "n_prior_messages": len(history_before),
                "contrast_type": contrast_type,
                "evidence_role": role,
                "evidence_id": entry["evidence_id"],
                "evidence_text": ev["text"] if ev else None,
                "evidence_strength": ev["strength"] if ev else entry.get("evidence_strength"),
                "evidence_credibility": ev.get("credibility") if ev else row.get("evidence_credibility"),
                "evidence_targets_display": entry.get("evidence_targets"),
                "evidence_targets_canonical": ev.get("targets") if ev else None,
                "target_belief": target_display,
                "target_belief_canonical": target_canonical,
                "target_belief_text": target_text,
                "aux_order_condition": mapping["aux_order_condition"],
                "canonical_to_display": mapping["canonical_to_display"],
                "display_A1_text": mapping["display_A1_text"],
                "display_A2_text": mapping["display_A2_text"],
                "aux_gate_passed": row.get("aux_gate_passed") if contrast_type == "auxiliary" else None,
                "active_auxiliaries_before_turn": prev_active,
                "active_auxiliaries_after_user": entry.get("active_auxiliaries_after_user"),
                "style_id": cell["style_profile"]["style_id"],
                "style_profile": cell["style_profile"],
                "affect": entry.get("user_affect"),
                "initial_affect": plan.get("initial_affect"),
                "trust_in_institutions": cell["trust_in_institutions"],
                "conversation_goal": cell["conversation_goal"],
                "belief_anchor_level": cell["belief_anchor_level"],
                "core_order_condition": plan.get("core_order_condition"),
                "elaboration_1_condition": plan.get("elaboration_1_condition"),
                "elaboration_2_target_condition": plan.get("elaboration_2_target_condition"),
                "user_belief_core": entry.get("user_belief_core"),
                "user_confidence_core": entry.get("user_confidence_core"),
                "probe_pre": probe_pre,
                "probe_post": probe_post,
                "experimental_text": entry["text"],
                "experimental_original_response": assistant_reply,
                "history_before": history_before,
                "experimental_prompt": build_messages(system_prompt, history_before, entry["text"]),
                "source_file": os.path.basename(source_path),
                "source_sha256": source_sha,
                "source_line": line_no,
            })
    return states


# ---------------------------------------------------------------------------
# 3a. matched substitute evidence (one per evidence_id x control_type)
# ---------------------------------------------------------------------------
CONTROL_DEFINITIONS = {
    "neutral": (
        "NEUTRAL: reports a finding from the same kind of source that is "
        "non-diagnostic for the target belief. It must neither support nor "
        "undermine the target belief (e.g. it reports something adjacent such as "
        "tolerability, adherence, participant characteristics, or procedure). It "
        "must not support or undermine the core claim or the other auxiliary either."
    ),
    "supportive": (
        "SUPPORTIVE: reports a finding from the same kind of source whose "
        "polarity is reversed, so that it supports the target belief to the same "
        "degree the original undermines it. Do not make it stronger or weaker "
        "than the original."
    ),
}

SUBSTITUTE_SYSTEM_PROMPT = """You construct matched control stimuli for a psychology experiment on belief revision.
You rewrite one piece of evidence so that ONLY its evidential bearing on a target belief changes.
Return only valid JSON."""


def build_substitute_prompt(vignette, ev, target_label, target_text, control_type, feedback=None):
    aux_lines = "\n".join(f"- {a['id']}: {a['text']}" for a in vignette["auxiliaries"])
    n_words = word_count(ev["text"])
    prompt = f"""Core claim: {vignette['core_claim']}
Auxiliary beliefs:
{aux_lines}

Target belief ({target_label}): {target_text}

Original evidence (it UNDERMINES the target belief):
- text: {ev['text']}
- strength: {ev['strength']}
- credibility: {ev.get('credibility')}

Write a control version of this evidence.
{CONTROL_DEFINITIONS[control_type]}

Hold everything else constant:
- same kind of source (same study design, sample size, review type, or venue)
- same strength ({ev['strength']}) and credibility ({ev.get('credibility')})
- same level of specificity and the same sentence structure where possible
- reuse as many of the original words as possible; change only what carries the finding
- about {n_words} words (stay within {int(n_words * 0.85)}-{int(n_words * 1.15) + 1})
- it must stay about the same target ({target_label}); do not switch to a different belief

Return JSON only:
{{"text": "the control evidence", "rationale": "one sentence on what changed"}}"""
    if feedback:
        prompt += f"\n\nYour previous attempt was rejected: {feedback}. Fix this."
    return prompt


def generate_substitute(model, vignette, ev, target_label, target_text, control_type):
    feedback = None
    last = None
    for attempt in range(1, SUBSTITUTE_MAX_ATTEMPTS + 1):
        raw = brx.call_model(model, [
            {"role": "system", "content": SUBSTITUTE_SYSTEM_PROMPT},
            {"role": "user", "content": build_substitute_prompt(vignette, ev, target_label, target_text, control_type, feedback)},
        ], temperature=SUBSTITUTE_TEMPERATURE, max_tokens=300)
        try:
            parsed = brx.extract_json(raw)
            text = (parsed.get("text") or "").strip()
        except (ValueError, AttributeError):
            feedback, last = "the output was not valid JSON with a 'text' field", None
            continue
        ratio = word_count(text) / max(1, word_count(ev["text"]))
        last = {"text": text, "rationale": parsed.get("rationale"), "attempts": attempt, "length_ratio": round(ratio, 3)}
        if not text:
            feedback = "the text was empty"
        elif text.strip().lower() == ev["text"].strip().lower():
            feedback = "the text was identical to the original"
        elif not (EVIDENCE_LENGTH_RATIO[0] <= ratio <= EVIDENCE_LENGTH_RATIO[1]):
            feedback = f"it had {word_count(text)} words; the original has {word_count(ev['text'])}"
        else:
            return last
    # Only a length mismatch is tolerable; an empty or unchanged item is no control at all.
    if last and last["text"] and last["text"].strip().lower() != ev["text"].strip().lower():
        last["warning"] = f"accepted after {SUBSTITUTE_MAX_ATTEMPTS} attempts: {feedback}"
        return last
    raise RuntimeError(f"could not generate a {control_type} substitute for {ev['id']}: {feedback}")


def load_or_create_substitutes(states, control_types, conversations, out_dir, model, offline):
    path = os.path.join(out_dir, "caa_control_evidence.json")
    store = {"prompt_version": PROMPT_VERSION, "items": {}}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            store = json.load(f)

    vignettes = {}
    for _, conv in conversations:
        vignettes.setdefault(conv["vignette_id"], conv["primary_cell"]["vignette"])

    needed = {}
    for s in states:
        for ct in control_types:
            needed[(s["vignette_id"], s["evidence_id"], ct)] = s

    changed = False
    for (vid, eid, ct), s in sorted(needed.items()):
        key = f"{vid}::{eid}::{ct}"
        if key in store["items"]:
            continue
        if offline:
            print(f"  offline: no substitute evidence for {key}; its pairs will be skipped")
            continue
        vignette = vignettes[vid]
        ev = evidence_lookup(vignette)[eid]
        if s["contrast_type"] == "core":
            target_label, target_text = "core claim", vignette["core_claim"]
        else:
            canon = (ev.get("targets") or [None])[0]
            target_label = f"auxiliary {canon}"
            target_text = next(a["text"] for a in vignette["auxiliaries"] if a["id"] == canon)
        print(f"  generating {ct} substitute for {vid}/{eid} ...")
        sub = generate_substitute(model, vignette, ev, target_label, target_text, ct)
        store["items"][key] = {
            "control_evidence_id": f"{eid}__ctrl_{ct}",
            "vignette_id": vid,
            "evidence_id": eid,
            "control_type": ct,
            "original_text": ev["text"],
            "strength": ev["strength"],
            "credibility": ev.get("credibility"),
            "targets_canonical": ev.get("targets"),
            "target_label": target_label,
            "text": sub["text"],
            "rationale": sub.get("rationale"),
            "length_ratio": sub.get("length_ratio"),
            "warning": sub.get("warning"),
            "generator_model": model,
            "prompt_version": PROMPT_VERSION,
            "hand_edited": False,
        }
        changed = True
        with open(path, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2, ensure_ascii=False)
    if changed:
        print(f"  substitute evidence written to {path} (review / hand-edit it; set hand_edited=true if you do)")
    return store["items"], path


# ---------------------------------------------------------------------------
# 3b. minimal counterfactual edit of the evidence-bearing user turn
# ---------------------------------------------------------------------------
REWRITE_SYSTEM_PROMPT = """You make MINIMAL counterfactual edits to one message in a conversation, for a controlled psychology experiment.
The edited message must differ from the original only in the evidence it reports and the words that directly depend on that evidence.
Return only valid JSON."""


def build_rewrite_prompt(state, sub, feedback=None):
    prev_assistant = next((h["text"] for h in reversed(state["history_before"]) if h["speaker"] == "assistant"), "")
    if len(prev_assistant) > 700:
        prev_assistant = prev_assistant[:700] + " ..."
    n_words = word_count(state["experimental_text"])
    prompt = f"""A simulated human user is talking to an AI assistant about this claim: {state['core_claim']}
The user's speaking style: {state['style_profile'].get('description')} (verbosity {state['style_profile'].get('verbosity')}, syntax {state['style_profile'].get('syntax')}).

Previous assistant message (context only, do not edit):
{prev_assistant}

ORIGINAL user message, written after the user encountered this evidence:
ORIGINAL EVIDENCE: {state['evidence_text']}
ORIGINAL MESSAGE: {state['experimental_text']}

Rewrite the message as if the user had instead encountered this evidence:
REPLACEMENT EVIDENCE ({sub['control_type']}): {sub['text']}

Rules:
- change ONLY the words that report or paraphrase the evidence, plus the minimum needed so the user's reaction stays coherent with the replacement evidence
- remove every trace of the original evidence's finding; do not let it survive in paraphrase
- keep everything else word-for-word: the user's personal experience, stated belief, emotional tone, questions, casing, punctuation habits, typos, and sentence order
- refer to the replacement evidence about as vaguely or as specifically as the original message referred to the original evidence
- do not add new arguments, new evidence, or new explanations that the original message did not contain
- keep the length close to the original ({n_words} words; stay within {int(n_words * 0.85)}-{int(n_words * 1.15) + 1})
- if the original message never actually refers to the original evidence, return it unchanged and set original_references_evidence to false

Return JSON only:
{{"original_references_evidence": true, "control_text": "the edited message", "changed_spans": [{{"from": "original words", "to": "new words"}}]}}"""
    if feedback:
        prompt += f"\n\nYour previous attempt was rejected: {feedback}. Fix this while keeping the edit minimal."
    return prompt


def leaked_terms(state, sub, control_text):
    """Distinctive words of the original evidence that the user echoed in the
    experimental turn, are absent from the substitute, yet survive in the control."""
    distinctive = content_words(state["evidence_text"]) - content_words(sub["text"])
    echoed = distinctive & content_words(state["experimental_text"])
    return sorted(echoed & content_words(control_text))


def cue_hits(text):
    t = (text or "").lower().replace("’", "'")
    return [c for c in DISCONFIRMING_CUES if c in t]


def check_rewrite(state, sub, control_text, references_evidence):
    """Returns (hard_problem_or_None, metrics) for one candidate control text."""
    ratio = word_count(control_text) / max(1, word_count(state["experimental_text"]))
    metrics = {
        "length_ratio": round(ratio, 3),
        "leaked_terms": leaked_terms(state, sub, control_text),
        "control_cue_hits": [c for c in cue_hits(control_text) if c not in cue_hits(sub["text"])],
    }
    if not control_text.strip():
        return "the control text was empty", metrics
    if references_evidence and control_text.strip() == state["experimental_text"].strip():
        return "the control text was identical to the original although the original refers to the evidence", metrics
    if not (LENGTH_RATIO_HARD[0] <= ratio <= LENGTH_RATIO_HARD[1]):
        return f"it had {word_count(control_text)} words; the original has {word_count(state['experimental_text'])}", metrics
    if metrics["leaked_terms"]:
        return f"words from the original evidence survived: {', '.join(metrics['leaked_terms'])}", metrics
    return None, metrics


def generate_control_rewrite(model, state, sub):
    feedback = None
    best = None
    for attempt in range(1, REWRITE_MAX_ATTEMPTS + 1):
        raw = brx.call_model(model, [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": build_rewrite_prompt(state, sub, feedback)},
        ], temperature=REWRITE_TEMPERATURE if attempt == 1 else REWRITE_TEMPERATURE + 0.2, max_tokens=400)
        try:
            parsed = brx.extract_json(raw)
            text = brx._clean_user_turn(parsed.get("control_text") or "")
            refs = bool(parsed.get("original_references_evidence", True))
        except (ValueError, AttributeError):
            feedback = "the output was not valid JSON with a 'control_text' field"
            continue
        problem, metrics = check_rewrite(state, sub, text, refs)
        candidate = {
            "control_text": text,
            "original_references_evidence": refs,
            "changed_spans": parsed.get("changed_spans"),
            "attempts": attempt,
            "rewrite_problem": problem,
            **metrics,
        }
        if problem is None:
            return candidate
        if text.strip() or best is None:
            best = candidate  # keep the latest usable attempt as the fallback
        feedback = problem
    if best is None:
        raise RuntimeError(f"rewriter never returned parseable JSON for {state['source_state_id']}")
    return best


def rewrite_cache_key(state, sub, model):
    return sha256_text(PROMPT_VERSION, model, state["source_state_id"], sub["control_type"],
                       state["experimental_text"], state["evidence_text"], sub["text"])


def load_rewrite_cache(path):
    cache = {}
    if os.path.exists(path):
        for _, rec in read_jsonl(path):
            cache[rec["cache_key"]] = rec
    return cache


def append_rewrite_cache(path, rec):
    with _cache_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_rewrites(states, substitutes, control_types, out_dir, model, workers, offline):
    cache_path = os.path.join(out_dir, "caa_control_rewrites_cache.jsonl")
    cache = load_rewrite_cache(cache_path)
    results = {}
    jobs = []
    for s in states:
        for ct in control_types:
            sub = substitutes.get(f"{s['vignette_id']}::{s['evidence_id']}::{ct}")
            if sub is None:
                continue
            key = rewrite_cache_key(s, sub, model)
            if key in cache:
                results[(s["source_state_id"], ct)] = cache[key]
            elif not offline:
                jobs.append((s, ct, sub, key))

    print(f"Rewrites: {len(results)} cached, {len(jobs)} to generate"
          + (f" (~{len(jobs)} API calls + retries)" if jobs else ""))
    if not jobs:
        return results

    done = failed = 0
    budget_hit = False

    def work(s, ct, sub, key):
        out = generate_control_rewrite(model, s, sub)
        rec = {"cache_key": key, "source_state_id": s["source_state_id"], "control_type": ct,
               "rewrite_model": model, "prompt_version": PROMPT_VERSION,
               "created_at": datetime.now(timezone.utc).isoformat(), **out}
        append_rewrite_cache(cache_path, rec)
        return rec

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, *j): j for j in jobs}
        for fut in concurrent.futures.as_completed(futs):
            s, ct, _, _ = futs[fut]
            try:
                rec = fut.result()
            except brx.BudgetExceeded as e:
                if not budget_hit:
                    budget_hit = True
                    print(f"Stopping: {e}. Re-run later to resume from the cache.")
                    brx.request_stop()
                continue
            except Exception as e:
                failed += 1
                print(f"  FAILED {s['source_state_id']} [{ct}]: {type(e).__name__}: {e}")
                continue
            results[(s["source_state_id"], ct)] = rec
            done += 1
            if done % 25 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] rewrites done | API calls used: {brx.calls_used()}")
    if failed or budget_hit:
        print(f"Rewrites incomplete: {failed} failed, budget stop: {budget_hit}. Missing pairs are left out of this build.")
    return results


# ---------------------------------------------------------------------------
# 4. pairs, prompt rendering, QC
# ---------------------------------------------------------------------------
def load_tokenizer(name, enabled):
    if not enabled:
        return None, None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        revision = getattr(tok, "init_kwargs", {}).get("_commit_hash") or getattr(tok, "_commit_hash", None)
        return tok, revision
    except Exception as e:
        print(f"WARNING: could not load tokenizer '{name}' ({type(e).__name__}: {e}); "
              f"pairs will store chat messages without the rendered template.")
        return None, None


def render(tokenizer, messages):
    if tokenizer is None:
        return None
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def history_problems(messages):
    problems = []
    if not messages or messages[0]["role"] != "system":
        problems.append("first message is not the system prompt")
    expected = "user"
    for m in messages[1:]:
        if m["role"] != expected:
            problems.append(f"role order broken (expected {expected}, got {m['role']})")
            break
        if not (m.get("content") or "").strip():
            problems.append(f"empty {m['role']} message")
        expected = "assistant" if expected == "user" else "user"
    if messages and messages[-1]["role"] != "user":
        problems.append("prompt does not end on a user turn")
    return problems


REQUIRED_FIELDS = [
    "pair_id", "conversation_id", "cell_id", "replicate_index", "vignette_id", "phase",
    "global_turn_index", "contrast_type", "control_type", "target_belief", "evidence_id",
    "style_id", "affect", "trust_in_institutions", "conversation_goal", "belief_anchor_level",
    "experimental_text", "control_text", "experimental_prompt", "control_prompt",
]


def build_pair(state, ct, sub, rw, tokenizer):
    control_prompt = state["experimental_prompt"][:-1] + [{"role": "user", "content": rw["control_text"]}]
    vignette_aux = [{"id": "A1", "text": state["display_A1_text"]}, {"id": "A2", "text": state["display_A2_text"]}]
    aux_exp = _aux_invoked(state["experimental_text"], vignette_aux)
    aux_ctrl = _aux_invoked(rw["control_text"], vignette_aux)

    pair = {k: v for k, v in state.items() if k not in ("history_before", "style_profile")}
    pair.update({
        "pair_id": f"{state['source_state_id']}__{ct}",
        "control_type": ct,
        "control_evidence_id": sub["control_evidence_id"],
        "control_evidence_text": sub["text"],
        "control_evidence_hand_edited": sub.get("hand_edited", False),
        "control_text": rw["control_text"],
        "control_prompt": control_prompt,
        "experimental_prompt_rendered": render(tokenizer, state["experimental_prompt"]),
        "control_prompt_rendered": render(tokenizer, control_prompt),
        "experimental_word_count": word_count(state["experimental_text"]),
        "control_word_count": word_count(rw["control_text"]),
        "length_ratio": rw.get("length_ratio"),
        "original_references_evidence": rw.get("original_references_evidence"),
        "changed_spans": rw.get("changed_spans"),
        "rewrite_attempts": rw.get("attempts"),
        "rewrite_model": rw.get("rewrite_model"),
        "prompt_version": rw.get("prompt_version"),
        "aux_invoked_experimental": aux_exp,
        "aux_invoked_control": aux_ctrl,
        "leaked_terms": rw.get("leaked_terms", []),
        "control_cue_hits": rw.get("control_cue_hits", []),
    })
    return pair


def _aux_invoked(text, aux_list):
    """Same keyword heuristic the pilot used to activate auxiliaries."""
    text_l = (text or "").lower()
    out = []
    for a in aux_list:
        keywords = set(re.findall(r"[a-z]{5,}", a["text"].lower()))
        if sum(1 for k in keywords if k in text_l) >= 2:
            out.append(a["id"])
    return out


def apply_qc(pairs):
    seen_ids = Counter(p["pair_id"] for p in pairs)
    seen_content = Counter(sha256_text(json.dumps(p["experimental_prompt"]), p["control_text"]) for p in pairs)
    for p in pairs:
        hard, soft = [], []
        missing = [f for f in REQUIRED_FIELDS if p.get(f) in (None, "", [])]
        if missing:
            hard.append("missing_metadata:" + ",".join(missing))
        for name in ("experimental_prompt", "control_prompt"):
            for prob in history_problems(p[name]):
                hard.append(f"malformed_{name}:{prob}")
        if p["experimental_prompt"][:-1] != p["control_prompt"][:-1]:
            hard.append("parent_context_mismatch")
        if seen_ids[p["pair_id"]] > 1:
            hard.append("duplicate_pair_id")
        if seen_content[sha256_text(json.dumps(p["experimental_prompt"]), p["control_text"])] > 1:
            hard.append("duplicate_pair_content")
        if not p["control_text"].strip():
            hard.append("empty_control")
        elif p["control_text"].strip() == p["experimental_text"].strip():
            (soft if p.get("original_references_evidence") is False else hard).append("control_identical_to_experimental")
        r = p.get("length_ratio") or 0
        if not (LENGTH_RATIO_HARD[0] <= r <= LENGTH_RATIO_HARD[1]):
            hard.append("length_mismatch")
        elif not (LENGTH_RATIO_SOFT[0] <= r <= LENGTH_RATIO_SOFT[1]):
            soft.append("length_mismatch_soft")
        if p["leaked_terms"]:
            # soft: users often echo evidence words that are also part of their
            # own auxiliary story (e.g. "dose"), so this needs a human look
            soft.append("original_evidence_terms_in_control")
        if p["control_cue_hits"]:
            soft.append("disconfirming_cue_in_control")
        if p.get("original_references_evidence") is False:
            soft.append("experimental_turn_does_not_reference_evidence")
        if p["contrast_type"] == "core" and p["target_belief"] != "core":
            hard.append("target_mismatch")
        if p["contrast_type"] == "auxiliary":
            if not p.get("target_belief_canonical"):
                hard.append("auxiliary_target_unresolved")
            if p.get("aux_gate_passed") is False:
                soft.append("aux_gate_not_passed")
            tgt = p.get("target_belief")
            if tgt in p["aux_invoked_experimental"] and tgt not in p["aux_invoked_control"]:
                soft.append("target_auxiliary_dropped_in_control")
        if sorted(p["aux_invoked_experimental"]) != sorted(p["aux_invoked_control"]):
            soft.append("auxiliary_invocation_changed")
        if p.get("evidence_text") is None:
            hard.append("evidence_text_missing")
        p["qc_hard_flags"] = hard
        p["qc_soft_flags"] = soft
        p["qc_pass"] = not hard
    return pairs


def qc_report(pairs, states, control_types, missing_rewrites):
    report = {"n_source_states": len(states), "n_pairs": len(pairs),
              "n_missing_rewrites": len(missing_rewrites), "missing_rewrites": missing_rewrites[:50],
              "by_dataset": {}}
    groups = defaultdict(list)
    for p in pairs:
        groups[(p["contrast_type"], p["control_type"])].append(p)
    for (contrast, ct), ps in sorted(groups.items()):
        ratios = [p["length_ratio"] for p in ps if p.get("length_ratio")]
        report["by_dataset"][f"{contrast}/{ct}"] = {
            "n_pairs": len(ps),
            "n_qc_pass": sum(p["qc_pass"] for p in ps),
            "n_source_states": len({p["source_state_id"] for p in ps}),
            "n_conversations": len({p["conversation_id"] for p in ps}),
            "evidence_ids": dict(Counter(p["evidence_id"] for p in ps)),
            "target_beliefs_canonical": dict(Counter(str(p["target_belief_canonical"]) for p in ps)),
            "length_ratio": {
                "mean": round(statistics.mean(ratios), 3) if ratios else None,
                "median": round(statistics.median(ratios), 3) if ratios else None,
                "min": min(ratios) if ratios else None,
                "max": max(ratios) if ratios else None,
            },
            "hard_flags": dict(Counter(f.split(":")[0] for p in ps for f in p["qc_hard_flags"])),
            "soft_flags": dict(Counter(f for p in ps for f in p["qc_soft_flags"])),
        }
    return report


def write_manual_review_sample(pairs, path, per_stratum, seed):
    rng = random.Random(seed)
    strata = defaultdict(list)
    for p in pairs:
        strata[(p["contrast_type"], p["control_type"], p["evidence_id"], p["style_id"])].append(p)
    sample = []
    for key in sorted(strata):
        ps = sorted(strata[key], key=lambda p: p["pair_id"])
        sample.extend(rng.sample(ps, min(per_stratum, len(ps))))
    cols = ["pair_id", "contrast_type", "control_type", "target_belief", "target_belief_canonical",
            "evidence_id", "style_id", "affect", "conversation_goal", "evidence_text",
            "control_evidence_text", "experimental_text", "control_text", "length_ratio",
            "qc_pass", "qc_hard_flags", "qc_soft_flags",
            "review_minimal_edit_ok", "review_polarity_ok", "review_target_unchanged",
            "review_style_matched", "review_notes"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for p in sample:
            row = dict(p)
            row["qc_hard_flags"] = "; ".join(p["qc_hard_flags"])
            row["qc_soft_flags"] = "; ".join(p["qc_soft_flags"])
            w.writerow(row)
    return len(sample)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate controlled counterfactual CAA contrastive pairs from the pilot JSONL.")
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--control-types", nargs="+", default=CONTROL_TYPES, choices=CONTROL_TYPES)
    ap.add_argument("--rewrite-model", default=brx.USER_SIM_MODEL)
    ap.add_argument("--limit", type=int, default=None,
                    help="only use the first N source states of each contrast type (for testing)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget", type=int, default=brx.DAILY_CALL_BUDGET, help="max API calls this run")
    ap.add_argument("--offline", action="store_true", help="make no API calls; build pairs from caches only")
    ap.add_argument("--tokenizer", default=brx.TARGET_MODEL_HF_PATH)
    ap.add_argument("--no-render", action="store_true", help="skip rendering prompts with the Qwen chat template")
    ap.add_argument("--review-per-stratum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    brx.set_call_budget(args.budget)

    source_sha = freeze_source(args.input, args.out_dir)
    conversations = read_jsonl(args.input)
    print(f"Loaded {len(conversations)} conversations from {args.input} (sha256 {source_sha[:12]})")

    states = extract_source_states(conversations, args.input, source_sha)
    ids = [s["source_state_id"] for s in states]
    if len(ids) != len(set(ids)):
        sys.exit("ERROR: duplicate source_state_id values; the pilot JSONL has repeated conversations.")
    if args.limit:
        by_type = defaultdict(list)
        for s in states:
            by_type[s["contrast_type"]].append(s)
        states = [s for t in sorted(by_type) for s in by_type[t][:args.limit]]
    n_core = sum(s["contrast_type"] == "core" for s in states)
    print(f"Source states: {n_core} core, {len(states) - n_core} auxiliary")

    substitutes, sub_path = load_or_create_substitutes(
        states, args.control_types, conversations, args.out_dir, args.rewrite_model, args.offline)
    rewrites = run_rewrites(states, substitutes, args.control_types, args.out_dir,
                            args.rewrite_model, args.workers, args.offline)

    tokenizer, tok_revision = load_tokenizer(args.tokenizer, not args.no_render)

    pairs, missing = [], []
    for s in states:
        for ct in args.control_types:
            sub = substitutes.get(f"{s['vignette_id']}::{s['evidence_id']}::{ct}")
            rw = rewrites.get((s["source_state_id"], ct))
            if sub is None or rw is None:
                missing.append(f"{s['source_state_id']}__{ct}")
                continue
            pairs.append(build_pair(s, ct, sub, rw, tokenizer))
    apply_qc(pairs)

    core_pairs = [p for p in pairs if p["contrast_type"] == "core"]
    aux_pairs = [p for p in pairs if p["contrast_type"] == "auxiliary"]
    core_path = os.path.join(args.out_dir, "caa_pairs_core.jsonl")
    aux_path = os.path.join(args.out_dir, "caa_pairs_auxiliary.jsonl")
    write_jsonl(core_path, core_pairs)
    write_jsonl(aux_path, aux_pairs)

    report = qc_report(pairs, states, args.control_types, missing)
    report_path = os.path.join(args.out_dir, "caa_pairs_qc_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    review_path = os.path.join(args.out_dir, "caa_pairs_manual_review.csv")
    n_review = write_manual_review_sample(pairs, review_path, args.review_per_stratum, args.seed)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": os.path.basename(__file__),
        "script_sha256": sha256_file(__file__),
        "prompt_version": PROMPT_VERSION,
        "source_file": args.input,
        "source_sha256": source_sha,
        "rewrite_model": args.rewrite_model,
        "rewrite_temperature": REWRITE_TEMPERATURE,
        "target_model": brx.TARGET_MODEL_HF_PATH,
        "tokenizer": args.tokenizer if tokenizer is not None else None,
        "tokenizer_revision": tok_revision,
        "target_system_prompt_sha256": sha256_text(brx.make_target_system_prompt()),
        "control_types": args.control_types,
        "limit": args.limit,
        "api_calls_used": brx.calls_used(),
        "outputs": {
            os.path.basename(p): sha256_file(p)
            for p in (core_path, aux_path, sub_path, report_path, review_path) if os.path.exists(p)
        },
        "counts": {k: v["n_pairs"] for k, v in report["by_dataset"].items()},
    }
    with open(os.path.join(args.out_dir, "caa_pairs_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n=== Summary ===")
    for k, v in report["by_dataset"].items():
        print(f"  {k:22s} pairs={v['n_pairs']:4d}  qc_pass={v['n_qc_pass']:4d}  "
              f"len_ratio median={v['length_ratio']['median']}  hard={v['hard_flags']}  soft={v['soft_flags']}")
    if missing:
        print(f"  {len(missing)} pair(s) missing a rewrite or substitute (see {report_path})")
    print(f"  wrote {core_path}, {aux_path}")
    print(f"  manual review sample ({n_review} pairs): {review_path}")
    print(f"  API calls used: {brx.calls_used()}")


if __name__ == "__main__":
    main()

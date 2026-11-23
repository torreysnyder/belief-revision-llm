"""Extract paired residual activations from Qwen2.5-7B BASE QC v4 CAA pairs.

Defaults to NDIF remote forward tracing (no local 7B weights). --backend local
is available when local model weights and sufficient VRAM exist.
Uses the exact plain-text System/User/Assistant prompt format and
add_special_tokens=False from the QC v4 behavioral experiment.

Example:
 python -u extract_caa_activations_qwen7b_qcv4.py --limit 2 --out-dir activations/smoke
 python -u extract_caa_activations_qwen7b_qcv4.py
"""

import os

# Must be set before CUDA initialises for deterministic cuBLAS kernels.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import json
import random
import hashlib
import argparse
import platform
import time
import tempfile
import socket
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_CORE_PAIRS = "data/caa_qwen7b_openrouter_qc_v4/caa_pairs_core.jsonl"
DEFAULT_AUX_PAIRS = "data/caa_qwen7b_openrouter_qc_v4/caa_pairs_auxiliary.jsonl"
DEFAULT_PAIR_MANIFEST = "data/caa_qwen7b_openrouter_qc_v4/caa_pairs_manifest.json"
DEFAULT_SPLIT_MANIFEST = "data/caa_qwen7b_openrouter_qc_v4/caa_split_manifest_by_vignette.json"
DEFAULT_OUT_DIR = "activations/caa_qwen7b_openrouter_qc_v4"
FALLBACK_MODEL = "Qwen/Qwen2.5-7B"

FORMAT_VERSION = 2

ACTIVATION_LOCATION = (
    "decoder_block_output: residual stream after transformer block l "
    "(attention + MLP residual additions), captured by a forward hook on "
    "decoder.layers[l]; equals HF hidden_states[l+1] for l < n_layers-1 and is "
    "pre-final-RMSNorm for the last block"
)

SPLITS = ["train", "validation", "test"]
DEFAULT_FRACTIONS = (0.70, 0.15, 0.15)
DEFAULT_STRATIFICATION = ["domain"]

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


# ---------------------------------------------------------------------------
# token positions: each returns the indices (into the tokenized prompt) whose
# activations are mean-pooled. Only the prespecified definition is enabled.
# ---------------------------------------------------------------------------
def _final_prompt_token(input_ids, pair, branch):
    return [len(input_ids) - 1]


TOKEN_POSITIONS = {
    "final_prompt_token": _final_prompt_token,
    # "evidence_token_mean": ...,   future: needs character spans of the evidence in the last user turn
    # "final_evidence_token": ...,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:
                    sys.exit(f"ERROR: {path} line {line_no} is not valid JSON ({e}).")
    return out


def default_model_name(pair_manifest_path):
    if os.path.exists(pair_manifest_path):
        with open(pair_manifest_path, "r", encoding="utf-8") as f:
            name = json.load(f).get("target_model")
        if name:
            return name
    return FALLBACK_MODEL


def set_determinism(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# ---------------------------------------------------------------------------
# 7.2 frozen vignette-level split (all conversations for a vignette stay together)
# ---------------------------------------------------------------------------
def vignette_factors(pairs, strat_vars):
    groups = OrderedDict()
    for p in pairs:
        vid = p["vignette_id"]
        if vid not in groups:
            groups[vid] = {v: p.get(v) for v in strat_vars}
        elif any(groups[vid][v] != p.get(v) for v in strat_vars):
            raise ValueError(f"Inconsistent vignette-level stratification metadata for {vid}")
    return groups


def create_split(groups, seed, fractions, strat_vars):
    """Allocate whole vignettes, stratified on vignette-invariant variables."""
    rng = random.Random(seed)
    strata = defaultdict(list)
    for vid, factors in groups.items():
        strata[tuple(str(factors.get(v)) for v in strat_vars)].append(vid)
    ordered = []
    for key in sorted(strata):
        members = sorted(strata[key])
        rng.shuffle(members)
        ordered.extend(members)
    counts = {split: 0 for split in SPLITS}
    assignment = {}
    for n, vid in enumerate(ordered, 1):
        deficits = [(fractions[i] * n - counts[split], -i) for i, split in enumerate(SPLITS)]
        split = SPLITS[-max(deficits)[1]]
        assignment[vid] = split
        counts[split] += 1
    return assignment


def split_balance(assignment, groups, strat_vars):
    table = {}
    for var in strat_vars:
        counts = defaultdict(Counter)
        for vid, split in assignment.items():
            counts[str(groups[vid].get(var))][split] += 1
        table[var] = {level: dict(c) for level, c in sorted(counts.items())}
    return table


def load_or_create_split(path, all_pairs, seed, fractions, strat_vars, input_hashes):
    groups = vignette_factors(all_pairs, strat_vars)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("split_unit") != "vignette_id":
            sys.exit(f"ERROR: {path} is not a vignette-level split. Use a new --split-manifest path.")
        mapping = manifest["assignments"]
        missing = [vid for vid in groups if vid not in mapping]
        if missing:
            sys.exit(f"ERROR: {len(missing)} vignette(s) missing from frozen split {path}, e.g. {missing[0]}.")
        bad = sorted({mapping[vid] for vid in groups if mapping[vid] not in SPLITS})
        if bad:
            sys.exit(f"ERROR: invalid split labels: {bad}")
        if manifest.get("source_pair_files") != input_hashes:
            sys.exit("ERROR: pair input hashes differ from the frozen split manifest. "
                     "Use the original pair data or deliberately create a new split manifest.")
        print(f"Using frozen vignette split {path} ({Counter(mapping.values())})")
        return mapping, manifest, False

    mapping = create_split(groups, seed, fractions, strat_vars)
    manifest = {
        "split_seed": seed,
        "split_method": "vignette-level systematic allocation, stratified by vignette-invariant factors",
        "split_unit": "vignette_id",
        "train_fraction": fractions[0],
        "validation_fraction": fractions[1],
        "test_fraction": fractions[2],
        "stratification_variables": strat_vars,
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "created_by": os.path.basename(__file__),
        "source_pair_files": input_hashes,
        "n_vignettes": len(mapping),
        "n_conversations": len({p["conversation_id"] for p in all_pairs}),
        "split_counts": dict(Counter(mapping.values())),
        "balance_by_variable": split_balance(mapping, groups, strat_vars),
        "assignments": dict(sorted(mapping.items())),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Created and froze vignette split {path} ({manifest['split_counts']})")
    return mapping, manifest, True


# ---------------------------------------------------------------------------
# model loading with a frozen revision
# ---------------------------------------------------------------------------
def resolve_revision(name, revision):
    if os.path.isdir(name):
        return revision
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(name, revision=revision or "main").sha
    except Exception as e:
        print(f"NOTE: could not resolve '{name}' revision online ({type(e).__name__}); "
              f"using the locally cached snapshot and recording its commit hash.")
        return revision


def load_model_and_tokenizer(name, revision, dtype, device, attn_implementation):
    resolved = resolve_revision(name, revision)
    tok = AutoTokenizer.from_pretrained(name, revision=resolved)
    kwargs = {"revision": resolved, "torch_dtype": DTYPES[dtype]}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs).to(device)
    model.eval()
    commit = getattr(model.config, "_commit_hash", None) or resolved
    return model, tok, commit


def decoder_layers(model):
    decoder = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    return decoder, decoder.layers


# ---------------------------------------------------------------------------
# 7.3 prompt reconstruction + verification
# ---------------------------------------------------------------------------
def render(tok, messages, generation_prompt=True):
    # Exact QC v4 base-model transcript (not ChatML / Instruct template).
    names = {"system": "System", "user": "User", "assistant": "Assistant"}
    body = "\n\n".join(f"{names[m['role']]}: {m['content']}" for m in messages)
    return body + ("\n\nAssistant:" if generation_prompt else "")


def tokenize(tok, text):
    # Same call the behavioral run used: tokenizer(prompt) with default special-token handling.
    return tok(text, add_special_tokens=False)["input_ids"]


def message_problems(messages):
    problems = []
    if not messages or messages[0].get("role") != "system":
        problems.append("no leading system message")
    expected = "user"
    for m in messages[1:]:
        if m.get("role") != expected:
            problems.append("role order broken")
            break
        expected = "assistant" if expected == "user" else "user"
    if not messages or messages[-1].get("role") != "user":
        problems.append("does not end on a user turn")
    return problems


def prepare_pair(tok, pair, position_name):
    """Renders, tokenizes and verifies both branches of one pair."""
    exp_msgs, ctrl_msgs = pair["experimental_prompt"], pair["control_prompt"]
    checks = OrderedDict()
    checks["roles_ok"] = not message_problems(exp_msgs) and not message_problems(ctrl_msgs)
    checks["shared_parent_context"] = exp_msgs[:-1] == ctrl_msgs[:-1]
    checks["experimental_turn_matches_text"] = exp_msgs[-1].get("content") == pair.get("experimental_text")
    checks["control_turn_matches_text"] = ctrl_msgs[-1].get("content") == pair.get("control_text")

    exp_text, ctrl_text = render(tok, exp_msgs), render(tok, ctrl_msgs)
    stored_exp, stored_ctrl = pair.get("experimental_prompt_rendered"), pair.get("control_prompt_rendered")
    checks["experimental_render_matches_stored"] = None if stored_exp is None else exp_text == stored_exp
    checks["control_render_matches_stored"] = None if stored_ctrl is None else ctrl_text == stored_ctrl

    parent = render(tok, exp_msgs[:-1], generation_prompt=False) + "\n\n"
    checks["parent_prefix_match"] = exp_text.startswith(parent) and ctrl_text.startswith(parent)
    exp_nogen, ctrl_nogen = render(tok, exp_msgs, False), render(tok, ctrl_msgs, False)
    gen_tail = exp_text[len(exp_nogen):]
    checks["ends_with_generation_prompt"] = (
        exp_text.startswith(exp_nogen) and ctrl_text.startswith(ctrl_nogen)
        and len(gen_tail) > 0 and ctrl_text[len(ctrl_nogen):] == gen_tail
    )

    exp_ids, ctrl_ids = tokenize(tok, exp_text), tokenize(tok, ctrl_text)
    shared = 0
    for a, b in zip(exp_ids, ctrl_ids):
        if a != b:
            break
        shared += 1
    parent_len = len(tokenize(tok, parent))
    # Tokenization can merge characters across a boundary; do not assume the
    # tokenized parent string is necessarily an exact token prefix.
    checks["parent_tokens_identical"] = None
    checks["final_token_ids_equal"] = exp_ids[-1] == ctrl_ids[-1]
    checks["expected_base_prompt_format"] = pair.get("target_prompt_format") in (None, "qwen2.5-7b-base_plain_role_transcript")
    checks["experimental_stored_not_chatml"] = not (stored_exp or "").startswith("<|im_start|>")
    checks["control_stored_not_chatml"] = not (stored_ctrl or "").startswith("<|im_start|>")

    position_fn = TOKEN_POSITIONS[position_name]
    exp_pos, ctrl_pos = position_fn(exp_ids, pair, "experimental"), position_fn(ctrl_ids, pair, "control")

    verified = all(v is not False for v in checks.values())
    return {
        "experimental": {"text": exp_text, "ids": exp_ids, "positions": exp_pos},
        "control": {"text": ctrl_text, "ids": ctrl_ids, "positions": ctrl_pos},
        "checks": checks,
        "prompt_match_verified": verified,
        "shared_prefix_tokens": shared,
        "parent_context_tokens": parent_len,
    }


# ---------------------------------------------------------------------------
# 7.4 deterministic forward passes with block-output hooks
# ---------------------------------------------------------------------------
class BlockOutputCapture:
    """Forward hooks on every decoder block. For each batch row they keep only
    the mean over that row's requested token positions, so full sequences of
    hidden states are never held in memory."""

    def __init__(self, layers):
        self.layers = layers
        self.positions = None  # list (per batch row) of index lists
        self.captured = [None] * len(layers)
        self.handles = [layer.register_forward_hook(self._hook(i)) for i, layer in enumerate(layers)]

    def _hook(self, i):
        def fn(module, inputs, output):
            hs = output[0] if isinstance(output, (tuple, list)) else output
            rows = []
            for b, idx in enumerate(self.positions):
                rows.append(hs[b, idx, :].float().mean(dim=0))
            self.captured[i] = torch.stack(rows).cpu()
        return fn

    def collect(self):
        return torch.stack(self.captured, dim=1)  # [batch, n_layers, hidden]

    def remove(self):
        for h in self.handles:
            h.remove()


def run_batches(model, decoder, capture, prompts, pad_id, device, batch_size):
    """prompts: dict key -> {"ids", "positions"}. Right padding keeps position
    ids and causal attention for the real tokens identical to an unpadded run."""
    keys = sorted(prompts, key=lambda k: len(prompts[k]["ids"]))
    out = {}
    with torch.inference_mode():
        for start in range(0, len(keys), batch_size):
            batch = keys[start:start + batch_size]
            max_len = max(len(prompts[k]["ids"]) for k in batch)
            ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
            mask = torch.zeros((len(batch), max_len), dtype=torch.long)
            for b, k in enumerate(batch):
                seq = prompts[k]["ids"]
                ids[b, :len(seq)] = torch.tensor(seq, dtype=torch.long)
                mask[b, :len(seq)] = 1
            capture.positions = [prompts[k]["positions"] for k in batch]
            decoder(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)
            acts = capture.collect()
            for b, k in enumerate(batch):
                out[k] = acts[b]
            done = min(start + batch_size, len(keys))
            if done == len(keys) or (done // batch_size) % 50 == 0:
                print(f"  forward passes: {done}/{len(keys)}")
    return out


def self_check(model, decoder, capture, prompt, device):
    """Confirms the hooks capture what ACTIVATION_LOCATION claims: block l output
    equals hidden_states[l+1], and the final norm of the last block output equals
    the last hidden state."""
    ids = torch.tensor([prompt["ids"]], dtype=torch.long, device=device)
    pos = len(prompt["ids"]) - 1
    capture.positions = [[pos]]
    with torch.inference_mode():
        out = decoder(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True, use_cache=False)
    hooked = capture.collect()[0]  # [L, H]
    hs = out.hidden_states
    n = hooked.shape[0]
    max_diff = 0.0
    for l in range(n - 1):
        max_diff = max(max_diff, (hooked[l] - hs[l + 1][0, pos].float().cpu()).abs().max().item())
    result = {"n_hidden_states": len(hs), "n_layers": n, "max_abs_diff_block_vs_hidden_states": max_diff}
    norm = getattr(decoder, "norm", None)
    if norm is not None:
        with torch.inference_mode():
            last_normed = norm(hooked[-1].to(device=device, dtype=hs[-1].dtype)[None])[0].float().cpu()
        result["max_abs_diff_final_norm"] = (last_normed - hs[-1][0, pos].float().cpu()).abs().max().item()
    tol = 1e-3 if hs[0].dtype == torch.float32 else 5e-2
    result["passed"] = len(hs) == n + 1 and max_diff <= tol and result.get("max_abs_diff_final_norm", 0.0) <= tol
    return result


# ---------------------------------------------------------------------------
# NDIF remote forward tracing: one prompt per invocation, no local model weights.
# ---------------------------------------------------------------------------
def _transient_ndif_error(exc):
    """Retry transport/service errors, not programming errors or OOM."""
    if isinstance(exc, (TimeoutError, ConnectionError, socket.timeout)):
        return True
    try:
        from websocket import WebSocketBadStatusException, WebSocketTimeoutException, WebSocketConnectionClosedException
        if isinstance(exc, (WebSocketTimeoutException, WebSocketConnectionClosedException)):
            return True
        if isinstance(exc, WebSocketBadStatusException):
            return getattr(exc, "status_code", None) in (429, 500, 502, 503, 504)
    except ImportError:
        pass
    msg = str(exc).lower()
    # Some NNsight versions wrap the underlying transport exception.
    return any(x in msg for x in (
        "502 bad gateway", "503 service unavailable", "504 gateway timeout",
        "429 too many requests", "connection timed out", "timed out",
        "winerror 10060", "connection reset", "connection refused",
        "websocket connection is closed", "websocketbadstatusexception",
        "temporarily unavailable", "remote disconnected",
    )) and not any(x in msg for x in ("cuda out of memory", "outofmemoryerror", "indexerror", "unboundlocalerror"))


def _checkpoint_identity(model_name, model_revision, key, prompt, n_layers, hidden_dim):
    """Content-addressed identity; incompatible model/token positions cannot be reused."""
    return {
        "schema": 1,
        "model": model_name,
        "revision": model_revision,
        "activation_location": "decoder_block_output_pre_final_norm",
        "key": [key[0], list(key[1])],
        "input_ids_sha256": hashlib.sha256(json.dumps(prompt["ids"], separators=(",", ":")).encode()).hexdigest(),
        "positions": list(prompt["positions"]),
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
    }


def _checkpoint_path(directory, identity):
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return os.path.join(directory, digest[:2], digest + ".pt")


def _validated_activation(value, n_layers, hidden_dim):
    if not isinstance(value, torch.Tensor):
        raise ValueError("Checkpoint/NDIF result is not a tensor")
    value = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if tuple(value.shape) != (n_layers, hidden_dim):
        raise ValueError(f"Unexpected activation shape: {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Activation contains NaN or Inf")
    return value


def _load_checkpoint(path, identity, n_layers, hidden_dim):
    if not os.path.isfile(path):
        return None
    try:
        # weights_only avoids arbitrary object unpickling from checkpoint files.
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("identity") != identity:
            raise ValueError("checkpoint identity mismatch")
        return _validated_activation(payload["activation"], n_layers, hidden_dim)
    except (OSError, EOFError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(f"WARNING: ignoring invalid checkpoint {path}: {exc}", flush=True)
        return None


def _save_checkpoint(path, identity, activation):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".pending_", suffix=".pt", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save({"identity": identity, "activation": activation}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)  # atomic replacement on the same filesystem
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _ndif_one_prompt(wrapper, prompt, n_layers, hidden_dim):
    import nnsight
    ids = torch.tensor([prompt["ids"]], dtype=torch.long)
    positions = prompt["positions"]
    # One remote-saved container: external list mutation does not survive tracing.
    with wrapper.trace(ids, remote=True):
        layer_vectors = nnsight.save([
            wrapper.model.layers[l].output[:, positions, :]
            .mean(dim=1).detach().float().cpu()
            for l in range(n_layers)
        ])
    if len(layer_vectors) != n_layers:
        raise ValueError(f"NDIF returned {len(layer_vectors)} of {n_layers} layers")
    act = torch.stack([v.detach().float().cpu().reshape(-1) for v in layer_vectors], dim=0)
    return _validated_activation(act, n_layers, hidden_dim)


def run_ndif_prompts(model_name, prompts, n_layers, hidden_dim, *, model_revision=None,
                     checkpoint_dir=None, max_attempts=6, retry_base=5.0,
                     retry_max_delay=120.0):
    """Retry transient failures and atomically checkpoint every successful prompt.

    Checkpoints are loaded before any NDIF initialization, so a fully cached run
    needs no network connection. Outputs retain the original dict interface.
    """
    if max_attempts < 1 or retry_base < 0 or retry_max_delay < 0:
        raise ValueError("Invalid retry configuration")
    if checkpoint_dir is None:
        raise ValueError("checkpoint_dir is required for resumable NDIF extraction")
    results, pending = {}, []
    for key, prompt in prompts.items():
        identity = _checkpoint_identity(model_name, model_revision, key, prompt, n_layers, hidden_dim)
        path = _checkpoint_path(checkpoint_dir, identity)
        cached = _load_checkpoint(path, identity, n_layers, hidden_dim)
        if cached is not None:
            results[key] = cached
        else:
            pending.append((key, prompt, identity, path))
    print(f"NDIF checkpoints: {len(results)} reused, {len(pending)} pending "
          f"(directory: {checkpoint_dir})", flush=True)
    if not pending:
        return results

    from nnsight import LanguageModel
    wrapper = LanguageModel(model_name, dispatch=False)
    if getattr(wrapper, "dispatched", False):
        raise RuntimeError("NDIF wrapper unexpectedly dispatched weights locally")
    for i, (key, prompt, identity, path) in enumerate(pending, 1):
        for attempt in range(1, max_attempts + 1):
            try:
                act = _ndif_one_prompt(wrapper, prompt, n_layers, hidden_dim)
                _save_checkpoint(path, identity, act)
                results[key] = act
                break
            except Exception as exc:
                if not _transient_ndif_error(exc) or attempt == max_attempts:
                    print(f"NDIF prompt {i}/{len(pending)} failed on attempt "
                          f"{attempt}/{max_attempts}: {type(exc).__name__}: {exc}", flush=True)
                    raise
                delay = min(retry_max_delay, retry_base * 2 ** (attempt - 1))
                delay += random.uniform(0, min(3.0, retry_max_delay / 10))
                print(f"Transient NDIF error on prompt {i}/{len(pending)}, "
                      f"attempt {attempt}/{max_attempts}: {type(exc).__name__}: {exc}. "
                      f"Retry in {delay:.1f}s", flush=True)
                time.sleep(delay)
        if i % 10 == 0 or i == len(pending):
            print(f"  NDIF completed: {len(results)}/{len(prompts)} "
                  f"({i}/{len(pending)} new)", flush=True)
    return results


# ---------------------------------------------------------------------------
# 7.5 metadata
# ---------------------------------------------------------------------------
PAIR_FIELDS = [
    # identity and provenance
    "pair_id", "source_state_id", "conversation_id", "cell_id", "replicate_index",
    # contrast definition
    "contrast_type", "control_type", "phase", "turn_in_phase", "global_turn_index",
    # evidence and belief target
    "evidence_id", "evidence_role", "evidence_strength", "evidence_credibility",
    "target_belief", "target_belief_canonical",
    # experimental factors
    "style_id", "initial_affect", "affect", "trust_in_institutions", "conversation_goal",
    "belief_anchor_level", "aux_order_condition",
    # quality control
    "qc_pass", "qc_hard_flags", "qc_soft_flags",
    # extra fields useful for sensitivity analyses
    "original_references_evidence", "length_ratio", "aux_gate_passed", "control_evidence_id",
]


def pair_metadata(pair, split, prep, tok, position_name):
    meta = OrderedDict((f, pair.get(f)) for f in PAIR_FIELDS)
    meta["missing_fields"] = [f for f in PAIR_FIELDS if f not in pair]
    meta["split"] = split
    exp, ctrl = prep["experimental"], prep["control"]
    meta["token_position"] = position_name
    meta["experimental_seq_len"] = len(exp["ids"])
    meta["control_seq_len"] = len(ctrl["ids"])
    meta["experimental_token_index"] = exp["positions"]
    meta["control_token_index"] = ctrl["positions"]
    meta["experimental_token_id"] = [exp["ids"][i] for i in exp["positions"]]
    meta["control_token_id"] = [ctrl["ids"][i] for i in ctrl["positions"]]
    if len(exp["positions"]) == 1:
        meta["experimental_token_id"] = meta["experimental_token_id"][0]
        meta["control_token_id"] = meta["control_token_id"][0]
    meta["experimental_token_str"] = tok.decode([exp["ids"][exp["positions"][-1]]])
    meta["shared_prefix_tokens"] = prep["shared_prefix_tokens"]
    meta["parent_context_tokens"] = prep["parent_context_tokens"]
    meta["experimental_prompt_sha256"] = sha256_text(exp["text"])
    meta["control_prompt_sha256"] = sha256_text(ctrl["text"])
    meta["prompt_match_verified"] = prep["prompt_match_verified"]
    meta["prompt_checks"] = dict(prep["checks"])
    return meta


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Extract pair-level residual-stream activations for CAA.")
    ap.add_argument("--core-pairs", default=DEFAULT_CORE_PAIRS)
    ap.add_argument("--auxiliary-pairs", default=DEFAULT_AUX_PAIRS)
    ap.add_argument("--pair-manifest", default=DEFAULT_PAIR_MANIFEST)
    ap.add_argument("--split-manifest", default=DEFAULT_SPLIT_MANIFEST)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--model", default=None, help="defaults to target_model in the pair manifest")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="NDIF per-prompt checkpoints; default <out-dir>/ndif_checkpoints")
    ap.add_argument("--ndif-max-attempts", type=int, default=6)
    ap.add_argument("--ndif-retry-base", type=float, default=5.0, help="initial backoff seconds")
    ap.add_argument("--ndif-retry-max-delay", type=float, default=120.0)
    ap.add_argument("--backend", choices=["ndif", "local"], default="ndif",
                    help="NDIF remote Qwen (default) or locally loaded Hugging Face weights")
    ap.add_argument("--model-revision", default=None, help="commit hash to pin; default resolves 'main' once and records it")
    ap.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES),
                    help="forward-pass dtype (behavioral run used bfloat16 on CUDA)")
    ap.add_argument("--save-dtype", default="float32", choices=sorted(DTYPES))
    ap.add_argument("--attn-implementation", default=None, help="e.g. eager or sdpa; default = library default")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="prompts per forward pass (right-padded); 1 avoids any padding-dependent numerics")
    ap.add_argument("--token-position", default="final_prompt_token", choices=sorted(TOKEN_POSITIONS))
    ap.add_argument("--include-qc-failed", action="store_true",
                    help="also extract hard-QC-failed pairs (kept with their qc fields)")
    ap.add_argument("--allow-prompt-mismatch", action="store_true",
                    help="extract pairs whose prompt verification failed instead of aborting")
    ap.add_argument("--limit", type=int, default=None, help="first N eligible pairs per dataset (smoke test)")
    ap.add_argument("--split-seed", type=int, default=20261006)
    ap.add_argument("--split-fractions", type=float, nargs=3, default=DEFAULT_FRACTIONS,
                    metavar=("TRAIN", "VALIDATION", "TEST"))
    ap.add_argument("--stratify", nargs="*", default=DEFAULT_STRATIFICATION)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true", help="replace existing activation files")
    args = ap.parse_args()

    if abs(sum(args.split_fractions) - 1.0) > 1e-6:
        sys.exit("ERROR: --split-fractions must sum to 1.")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.split_manifest)), exist_ok=True)
    out_paths = {
        "core": os.path.join(args.out_dir, "core_activations.pt"),
        "auxiliary": os.path.join(args.out_dir, "auxiliary_activations.pt"),
    }
    manifest_path = os.path.join(args.out_dir, "activation_manifest.json")
    existing = [p for p in list(out_paths.values()) + [manifest_path] if os.path.exists(p)]
    if existing and not args.overwrite:
        sys.exit(f"ERROR: {existing[0]} already exists. Activation artifacts are frozen; "
                 f"pass --overwrite or use a different --out-dir.")

    set_determinism(args.seed)

    inputs = {"core": args.core_pairs, "auxiliary": args.auxiliary_pairs}
    pairs_by_type = {t: read_jsonl(p) for t, p in inputs.items()}
    input_hashes = {os.path.basename(p): sha256_file(p) for p in inputs.values()}
    for t, ps in pairs_by_type.items():
        wrong = [p["pair_id"] for p in ps if p.get("contrast_type") != t]
        if wrong:
            sys.exit(f"ERROR: {inputs[t]} contains pairs with contrast_type != '{t}' (e.g. {wrong[0]}).")
        dup = [k for k, n in Counter(p["pair_id"] for p in ps).items() if n > 1]
        if dup:
            sys.exit(f"ERROR: duplicate pair_id in {inputs[t]} (e.g. {dup[0]}).")
    all_pairs = pairs_by_type["core"] + pairs_by_type["auxiliary"]
    if not all_pairs:
        sys.exit("ERROR: no contrastive pairs found in the input files.")
    if any(not p.get("vignette_id") for p in all_pairs):
        sys.exit("ERROR: at least one pair is missing vignette_id.")

    # The split covers every vignette in the pair data, independent of QC.
    split_map, split_manifest, split_created = load_or_create_split(
        args.split_manifest, all_pairs, args.split_seed, tuple(args.split_fractions),
        args.stratify, input_hashes)

    model_name = args.model or default_model_name(args.pair_manifest)
    if model_name != "Qwen/Qwen2.5-7B":
        sys.exit(f"ERROR: expected the exact QC v4 base checkpoint Qwen/Qwen2.5-7B, got {model_name!r}.")
    print(f"Preparing {model_name} ({args.dtype}) using {args.backend} ...")
    if args.backend == "local":
        model, tok, model_revision = load_model_and_tokenizer(
            model_name, args.model_revision, args.dtype, args.device, args.attn_implementation)
        decoder, layers = decoder_layers(model)
        n_layers, hidden_dim = len(layers), model.config.hidden_size
    else:
        from transformers import AutoConfig
        tok = AutoTokenizer.from_pretrained(model_name, revision=args.model_revision)
        cfg = AutoConfig.from_pretrained(model_name, revision=args.model_revision)
        n_layers, hidden_dim = cfg.num_hidden_layers, cfg.hidden_size
        model_revision = getattr(cfg, "_commit_hash", None) or args.model_revision
        model = decoder = layers = None
        if not os.environ.get("NDIF_API_KEY"):
            from dotenv import load_dotenv
            load_dotenv()
        if not os.environ.get("NDIF_API_KEY"):
            sys.exit("ERROR: NDIF_API_KEY not set. Set it in the environment or .env.")
    if (n_layers, hidden_dim) != (28, 3584):
        sys.exit(f"ERROR: unexpected Qwen2.5-7B dimensions: {n_layers} layers, {hidden_dim} hidden.")
    chat_template = tok.chat_template or ""
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"Model revision {model_revision} | {n_layers} layers x {hidden_dim} | "
          f"chat template sha256 {sha256_text(chat_template)[:12]}")

    # select + verify pairs
    selected, counts = {}, {}
    for t, ps in pairs_by_type.items():
        eligible = [p for p in ps if args.include_qc_failed or p.get("qc_pass") is True]
        n_qc_excluded = len(ps) - len(eligible)
        if args.limit:
            eligible = eligible[:args.limit]
        prepared, mismatched = [], []
        for p in eligible:
            prep = prepare_pair(tok, p, args.token_position)
            if not prep["prompt_match_verified"]:
                failed = [k for k, v in prep["checks"].items() if v is False]
                mismatched.append((p["pair_id"], failed))
                if not args.allow_prompt_mismatch:
                    continue
            prepared.append((p, prep))
        if mismatched:
            print(f"WARNING: {len(mismatched)} {t} pair(s) failed prompt verification, e.g. "
                  f"{mismatched[0][0]}: {mismatched[0][1]}")
            if not args.allow_prompt_mismatch:
                sys.exit("ERROR: prompt verification failed (template / tokenizer drift or malformed "
                         "pairs). Inspect them, or pass --allow-prompt-mismatch to extract anyway.")
        selected[t] = prepared
        counts[t] = {"seen": len(ps), "qc_excluded": n_qc_excluded,
                     "prompt_mismatch": len(mismatched), "extracted": len(prepared)}
        print(f"{t}: {len(ps)} pairs seen, {n_qc_excluded} excluded by hard QC, "
              f"{len(prepared)} to extract")

    # forward passes; identical prompts (the experimental branch is shared by the
    # neutral and supportive pairs of a source state) are run once
    unique = {}
    for t, prepared in selected.items():
        for p, prep in prepared:
            for branch in ("experimental", "control"):
                b = prep[branch]
                key = (sha256_text(b["text"]), tuple(b["positions"]))
                unique.setdefault(key, {"ids": b["ids"], "positions": b["positions"]})
    print(f"Running {len(unique)} unique prompts (batch size {args.batch_size}) ...")

    check = None
    if args.backend == "local":
        capture = BlockOutputCapture(layers)
        if unique:
            check = self_check(model, decoder, capture, next(iter(unique.values())), args.device)
            print(f"Hook self-check: {check}")
            if not check["passed"]:
                sys.exit("ERROR: hooked block outputs failed self-check.")
        acts = run_batches(model, decoder, capture, unique, pad_id, args.device, args.batch_size)
        capture.remove()
    else:
        checkpoint_dir = args.checkpoint_dir or os.path.join(args.out_dir, "ndif_checkpoints")
        acts = run_ndif_prompts(
            model_name, unique, n_layers, hidden_dim,
            model_revision=model_revision, checkpoint_dir=checkpoint_dir,
            max_attempts=args.ndif_max_attempts, retry_base=args.ndif_retry_base,
            retry_max_delay=args.ndif_retry_max_delay)

    timestamp = datetime.now(timezone.utc).isoformat()
    script_sha = sha256_file(__file__)
    invariant = OrderedDict([
        ("format_version", FORMAT_VERSION),
        ("model_name", model_name),
        ("model_revision", model_revision),
        ("tokenizer_name", model_name),
        ("tokenizer_revision", model_revision),
        ("activation_location", ACTIVATION_LOCATION),
        ("token_position", args.token_position),
        ("layers_extracted", list(range(n_layers))),
        ("n_layers", n_layers),
        ("hidden_dim", hidden_dim),
        ("forward_dtype", args.dtype),
        ("activation_dtype", args.save_dtype),
        ("chat_template_hash", None),
        ("prompt_format", "qwen2.5-7b-base_plain_role_transcript"),
        ("tokenization", "add_special_tokens=False"),
        ("backend", args.backend),
        ("ndif_checkpoint_dir", (args.checkpoint_dir or os.path.join(args.out_dir, "ndif_checkpoints")) if args.backend == "ndif" else None),
        ("ndif_max_attempts", args.ndif_max_attempts if args.backend == "ndif" else None),
        ("extraction_script", os.path.basename(__file__)),
        ("extraction_script_version", script_sha),
        ("extraction_timestamp", timestamp),
    ])

    save_dtype = DTYPES[args.save_dtype]
    split_counts = {}
    output_hashes = {}
    for t, prepared in selected.items():
        metas, exp_rows, ctrl_rows = [], [], []
        for p, prep in prepared:
            exp_key = (sha256_text(prep["experimental"]["text"]), tuple(prep["experimental"]["positions"]))
            ctrl_key = (sha256_text(prep["control"]["text"]), tuple(prep["control"]["positions"]))
            metas.append(pair_metadata(p, split_map[p["vignette_id"]], prep, tok, args.token_position))
            exp_rows.append(acts[exp_key])
            ctrl_rows.append(acts[ctrl_key])
        if exp_rows:
            exp_t = torch.stack(exp_rows)
            ctrl_t = torch.stack(ctrl_rows)
        else:
            exp_t = ctrl_t = torch.zeros((0, n_layers, hidden_dim))
        diff_t = exp_t - ctrl_t  # computed in float32 before any down-cast
        payload = {
            **invariant,
            "contrast_type": t,
            "input_file": inputs[t],
            "input_sha256": sha256_file(inputs[t]),
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "pair_ids": [m["pair_id"] for m in metas],
            "pair_metadata": metas,
            "experimental_activation": exp_t.to(save_dtype),
            "control_activation": ctrl_t.to(save_dtype),
            "pair_difference": diff_t.to(save_dtype),
        }
        torch.save(payload, out_paths[t])
        output_hashes[os.path.basename(out_paths[t])] = sha256_file(out_paths[t])
        split_counts[t] = {
            f"{s}/{ct}": n for (s, ct), n in sorted(Counter((m["split"], m["control_type"]) for m in metas).items())
        }
        print(f"Saved {len(metas)} {t} pairs -> {out_paths[t]} (tensors {tuple(exp_t.shape)})")

    manifest = OrderedDict(invariant)
    manifest.update([
        ("core_input_file", args.core_pairs),
        ("core_input_sha256", input_hashes[os.path.basename(args.core_pairs)]),
        ("auxiliary_input_file", args.auxiliary_pairs),
        ("auxiliary_input_sha256", input_hashes[os.path.basename(args.auxiliary_pairs)]),
        ("pair_manifest_file", args.pair_manifest if os.path.exists(args.pair_manifest) else None),
        ("pair_manifest_sha256", sha256_file(args.pair_manifest) if os.path.exists(args.pair_manifest) else None),
        ("split_manifest_file", args.split_manifest),
        ("split_manifest_sha256", sha256_file(args.split_manifest)),
        ("split_created_this_run", split_created),
        ("n_core_pairs_seen", counts["core"]["seen"]),
        ("n_core_pairs_extracted", counts["core"]["extracted"]),
        ("n_auxiliary_pairs_seen", counts["auxiliary"]["seen"]),
        ("n_auxiliary_pairs_extracted", counts["auxiliary"]["extracted"]),
        ("n_qc_excluded", counts["core"]["qc_excluded"] + counts["auxiliary"]["qc_excluded"]),
        ("n_prompt_mismatch", counts["core"]["prompt_mismatch"] + counts["auxiliary"]["prompt_mismatch"]),
        ("include_qc_failed", args.include_qc_failed),
        ("limit", args.limit),
        ("prompt_match_verified", all(prep["prompt_match_verified"]
                                      for prepared in selected.values() for _, prep in prepared)),
        ("n_unique_prompts_forwarded", len(unique)),
        ("counts_by_split_and_control", split_counts),
        ("hook_self_check", check),
        ("device", args.device),
        ("batch_size", args.batch_size),
        ("padding_side", "right"),
        ("attn_implementation", getattr(model.config, "_attn_implementation", None) if model is not None else "NDIF remote"),
        ("deterministic_algorithms", torch.are_deterministic_algorithms_enabled()),
        ("seed", args.seed),
        ("torch_version", torch.__version__),
        ("transformers_version", transformers.__version__),
        ("python_version", platform.python_version()),
        ("cuda_device", torch.cuda.get_device_name(0) if args.backend == "local" and args.device.startswith("cuda") else None),
        ("output_files", output_hashes),
    ])
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()

"""
CAA steering intervention and behavioral evaluation for the Core / Auxiliary
belief-revision experiment (README_controlled_counterfactual_CAA.md, sections
10-13). Consumes the vectors written by build_caa_vectors.py.

Intervention (section 10). During generation, the output of decoder block l (the
same location the activations were extracted from) is shifted:

  h'[l] = h[l] + alpha * v[l]

with v the raw mean paired difference (--alpha-units raw, as in the README) or
alpha * residual_norm_mean[l] * unit(v[l]) (--alpha-units residual, alpha is then
a fraction of the typical residual norm at that layer). By default the shift is
applied from the final prompt token onward, i.e. at the position the vectors were
extracted from and at every generated token; earlier prompt tokens are left
unsteered, which lets all conditions share one prompt KV cache. alpha = 0 is the shared unsteered baseline; negative alphas
give -CAA.

Cross-task design (section 11). Every steering vector is applied in every
evaluation context, so the full vector x context table is estimated:

                         context: core challenge    context: auxiliary challenge
  vector: core           Core -> Core               Core -> Auxiliary
  vector: auxiliary      Auxiliary -> Core          Auxiliary -> Auxiliary
  vector: random         norm-matched control       norm-matched control

and in each cell both the TARGET belief of that context (core belief in core
contexts, the challenged displayed auxiliary in auxiliary contexts) and the
OFF-TARGET belief (mean displayed auxiliary in core contexts, core belief in
auxiliary contexts) are measured. Specificity shows up as Core -> Core moving
belief_core but not the auxiliaries, and Auxiliary -> Auxiliary moving the
targeted auxiliary but not the core.

Evaluation prompts are the held-out pair records themselves: for each source
state (one per pair with the chosen control type) both branches are steered,

  experimental  real disconfirming evidence   (-CAA should reduce revision)
  control       matched neutral/supportive    (+CAA should induce revision)

Since v = h_disconfirming - h_control, +alpha pushes the residual stream toward
"disconfirming evidence was received", so belief scores (0-100 endorsement) are
expected to FALL with +alpha if the direction is causal.

Outcomes (section 12). Each response is scored by the same judge and probe prompt
as the behavioral run (run_probe from belief_revision_experiment_qwen.py), given
the conversation up to and including the steered reply and the last pre-evidence
probe as `previous_probe`. Generation-quality controls are computed locally:
response length, empty responses, 3-gram repetition, and the per-token NLL of
the steered reply under the UNSTEERED model (incoherence check).

Controls (section 13): norm-matched random direction per layer, both signs,
several alphas, both control branches, several samples per prompt.

Splits. --split validation is the development sweep (layer / alpha choice);
--split test is the preregistered held-out evaluation and requires --confirm-test.
Training pairs are never evaluated. With --layers auto the layer is chosen from
the validation diagnostics of build_caa_vectors.py (never from steering outcomes).

Stages (each resumable; rerunning skips finished work):

  generate   local Qwen generation + quality metrics  -> <run_dir>/generations.jsonl
  judge      remote judge probe (costs API calls)     -> <run_dir>/judgments.jsonl
  summarize  paired deltas vs alpha = 0, conversation-clustered bootstrap CIs
             -> <run_dir>/steering_{development|heldout}.csv          (per alpha)
             -> <run_dir>/steering_{development|heldout}_dose_response.csv
             -> <run_dir>/steering_{development|heldout}_cross_task.csv

Usage:
  python activation_steering.py --max-states 4 --stages generate summarize --run-name smoke
  python activation_steering.py                                      # development sweep
  python activation_steering.py --split test --confirm-test --layers 12 --alphas -4 4
"""

import os
import sys
import copy
import json
import torch
import random
import hashlib
import argparse
import platform
import transformers
import concurrent.futures
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from types import SimpleNamespace

DEFAULT_VECTOR_DIR = "vectors"
DEFAULT_DIAGNOSTICS = "results/vector_diagnostics.csv"
DEFAULT_CORE_PAIRS = "data/caa_pairs_core.jsonl"
DEFAULT_AUX_PAIRS = "data/caa_pairs_auxiliary.jsonl"
DEFAULT_SPLIT_MANIFEST = "data/caa_split_manifest.json"
DEFAULT_RESULTS_DIR = "results/steering"

FORMAT_VERSION = 1

CONTEXTS = ["core", "auxiliary"]
VECTOR_FAMILIES = ["core", "auxiliary", "random"]
BRANCHES = ["experimental", "control"]
SPLIT_TAGS = {"validation": "development", "test": "heldout"}

DEFAULT_ALPHAS = {"raw": [-8.0, -4.0, 4.0, 8.0], "residual": [-0.2, -0.1, 0.1, 0.2]}

# Behavioral-run generation settings (call_target_model_local).
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 220

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}

JUDGE_OUTCOMES = ["belief_target", "belief_offtarget", "belief_core", "belief_aux_mean", "belief_aux_target", "belief_aux_nontarget", "epistemic_confidence", "challenges_user", "changed_node_target"]
QUALITY_OUTCOMES = ["n_tokens", "empty_response", "repetition_3gram", "nll_unsteered"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path, records):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def fmt_alpha(a):
    return f"{a:+g}"


def repetition_3gram(text):
    words = text.lower().split()
    grams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


# ---------------------------------------------------------------------------
# vectors and layer selection
# ---------------------------------------------------------------------------
def load_vectors(vector_dir, control_type, seed):
    with open(os.path.join(vector_dir, "vector_metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    raw, residual = {}, {}
    for family in ("core", "auxiliary"):
        p = torch.load(os.path.join(vector_dir, f"{family}_caa_vectors.pt"), map_location="cpu", weights_only=False)
        if control_type not in p["vectors"]:
            sys.exit(f"ERROR: {family} vectors have no '{control_type}' control type.")
        raw[family] = p["vectors"][control_type].float()  # [L, H]
        residual[family] = p["residual_norm_mean"][control_type].float()  # [L]
    # Norm-matched random control: one fixed Gaussian direction per layer, scaled
    # to the mean of the Core and Auxiliary vector norms at that layer.
    gen = torch.Generator().manual_seed(seed)
    direction = torch.randn(raw["core"].shape, generator=gen)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    norm = (raw["core"].norm(dim=-1) + raw["auxiliary"].norm(dim=-1)) / 2
    raw["random"] = direction * norm[:, None]
    residual["random"] = (residual["core"] + residual["auxiliary"]) / 2
    return metadata, raw, residual


def steering_vector(raw, residual, family, layer, alpha, units):
    v = raw[family][layer]
    if units == "raw":
        return alpha * v
    return alpha * residual[family][layer] * v / v.norm().clamp_min(1e-12)


def auto_layers(diagnostics_path, control_type, n_layers, k):
    """Top-k layers by mean held-out projection effect size (val_proj_d over the
    Core and Auxiliary vectors) within the middle half of the network. Uses only
    validation-split representational diagnostics, never steering outcomes."""
    import csv

    if not os.path.exists(diagnostics_path):
        sys.exit(f"ERROR: {diagnostics_path} not found; run build_caa_vectors.py or pass --layers.")
    scores = defaultdict(list)
    with open(diagnostics_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["control_type"] == control_type:
                scores[int(row["layer"])].append(float(row["val_proj_d"]))
    lo, hi = n_layers // 4, (3 * n_layers) // 4
    ranked = sorted((sum(values) / len(values), layer) for layer, values in scores.items() if lo <= layer < hi)
    chosen = sorted(layer for _, layer in ranked[-k:])
    rule = f"top {k} layer(s) in [{lo}, {hi}) by mean val_proj_d of the core and auxiliary '{control_type}' vectors in {diagnostics_path}"
    return chosen, rule


# ---------------------------------------------------------------------------
# evaluation items
# ---------------------------------------------------------------------------
def load_items(pair_paths, split_manifest_path, split, control_type, max_states, seed):
    with open(split_manifest_path, "r", encoding="utf-8") as f:
        assignments = json.load(f)["assignments"]
    items = {}
    for context, path in pair_paths.items():
        chosen = []
        for p in read_jsonl(path):
            if p.get("qc_pass") is not True or p.get("control_type") != control_type:
                continue
            if assignments.get(p["conversation_id"]) != split:
                continue
            if context == "auxiliary" and p.get("target_belief") not in ("A1", "A2"):
                sys.exit(f"ERROR: {p['pair_id']} has display target {p.get('target_belief')!r}.")
            chosen.append(p)
        chosen.sort(key=lambda p: p["pair_id"])
        if max_states and len(chosen) > max_states:
            random.Random(seed).shuffle(chosen)
            chosen = sorted(chosen[:max_states], key=lambda p: p["pair_id"])
        items[context] = chosen
    return items


def load_source_dialogues(pairs):
    """Original dialogue histories (from the behavioral pilot) for the judge."""
    wanted = defaultdict(set)
    for p in pairs:
        wanted[p["source_file"]].add(p["source_line"])
    histories = {}
    for path, lines in wanted.items():
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if line_no in lines:
                    histories[(path, line_no)] = json.loads(line)["dialogue_history"]
    return histories


# ---------------------------------------------------------------------------
# steering hooks
# ---------------------------------------------------------------------------
class Steerer:
    """Forward hooks on the steered decoder blocks. `vectors[layer]` is a
    [batch, hidden] tensor (zeros for unsteered rows) or None when inactive.

    Generation reuses an unsteered KV cache of prompt[:-1], so the first steered
    forward pass sees only the final prompt token and every later pass one
    generated token: adding the vector to every position the hook sees therefore
    steers exactly the final prompt token and all generated tokens."""

    def __init__(self, layer_modules, layers):
        self.vectors = {layer: None for layer in layers}
        self.handles = [layer_modules[layer].register_forward_hook(self._hook(layer)) for layer in layers]

    def _hook(self, layer):
        def fn(module, inputs, output):
            vec = self.vectors[layer]
            if vec is None:
                return output
            hs = output[0] if isinstance(output, (tuple, list)) else output
            if hs.shape[1] != 1:
                raise RuntimeError("steering hook expects one token per step (shared prompt cache)")
            hs = hs + vec[:, None, :]
            if isinstance(output, tuple):
                return (hs,) + output[1:]
            if isinstance(output, list):
                return [hs] + output[1:]
            return hs

        return fn

    def set(self, vectors, device, dtype):
        self.vectors = {layer: vector.to(device=device, dtype=dtype) for layer, vector in vectors.items()}

    def clear(self):
        self.vectors = {layer: None for layer in self.vectors}

    def remove(self):
        for h in self.handles:
            h.remove()


def condition_list(layers, families, alphas):
    conds = [{"layer": None, "vector": "none", "alpha": 0.0}]
    for layer in layers:
        for family in families:
            for alpha in alphas:
                conds.append({"layer": layer, "vector": family, "alpha": alpha})
    return conds


def gen_key(pair_id, branch, sample, cond):
    layer = "-" if cond["layer"] is None else cond["layer"]
    return f"{pair_id}|{branch}|s{sample}|L{layer}|{cond['vector']}|{fmt_alpha(cond['alpha'])}"


def prefix_cache(model, prompt_ids, device):
    """Unsteered KV cache of every prompt token except the last. Computing it once
    at batch size 1 avoids the quadratic prefill memory of batched long prompts."""
    cache = DynamicCache()
    with torch.inference_mode():
        model(input_ids=torch.tensor([prompt_ids[:-1]], dtype=torch.long, device=device), past_key_values=cache, use_cache=True, logits_to_keep=1)
    return cache


def expanded(cache, n):
    c = copy.deepcopy(cache)
    c.batch_repeat_interleave(n)
    return c


def response_nll(model, cache, prompt_ids, responses, pad_id, device):
    """Mean per-token NLL of each response under the unsteered model (steering
    hooks must be inactive)."""
    width = max(len(r) for r in responses)
    if width == 0:
        return [None] * len(responses)
    ids = [[prompt_ids[-1]] + r + [pad_id] * (width - len(r)) for r in responses]
    with torch.inference_mode():
        # right padding only follows real tokens, so causal attention leaves them unaffected
        logits = model(input_ids=torch.tensor(ids, dtype=torch.long, device=device), past_key_values=expanded(cache, len(ids)), use_cache=True).logits
        out = []
        for b, r in enumerate(responses):
            if not r:
                out.append(None)
                continue
            target = torch.tensor(r, dtype=torch.long, device=device)
            out.append(torch.nn.functional.cross_entropy(logits[b, : len(r)].float(), target).item())
    return out


def run_generation(args, items, conds, raw, residual, gens_path, done_keys, model_name, model_revision):
    print(f"Loading {model_name} @ {model_revision} ({args.dtype}) on {args.device} ...")
    tok = AutoTokenizer.from_pretrained(model_name, revision=model_revision)
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=model_revision, dtype=DTYPES[args.dtype]).to(args.device)
    model.eval()
    decoder = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    steer_layers = sorted({c["layer"] for c in conds if c["layer"] is not None})
    steerer = Steerer(decoder.layers, steer_layers)
    eos_ids = model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    pad_id = tok.eos_token_id

    jobs = [(context, p, branch, s) for context in CONTEXTS for p in items[context] for branch in args.branches for s in range(args.n_samples)]
    print(f"Generation: {len(jobs)} prompts x {len(conds)} conditions")
    n_new = 0
    for j, (context, p, branch, sample) in enumerate(jobs, 1):
        todo = [c for c in conds if gen_key(p["pair_id"], branch, sample, c) not in done_keys]
        if not todo:
            continue
        messages = p["experimental_prompt"] if branch == "experimental" else p["control_prompt"]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        stored = p.get(f"{branch}_prompt_rendered")
        if stored is not None and stored != prompt:
            sys.exit(f"ERROR: rendered {branch} prompt for {p['pair_id']} differs from the stored one (chat template drift).")
        prompt_ids = tok(prompt)["input_ids"]
        cache = prefix_cache(model, prompt_ids, args.device)
        seed = stable_seed(args.seed, p["pair_id"], branch, sample)
        records = []
        for start in range(0, len(todo), args.max_batch):
            chunk = todo[start : start + args.max_batch]
            vectors = {layer: torch.zeros((len(chunk), raw["core"].shape[1])) for layer in steer_layers}
            for b, c in enumerate(chunk):
                if c["layer"] is not None:
                    vectors[c["layer"]][b] = steering_vector(raw, residual, c["vector"], c["layer"], c["alpha"], args.alpha_units)
            ids = torch.tensor([prompt_ids] * len(chunk), dtype=torch.long, device=args.device)
            torch.manual_seed(seed)
            steerer.set(vectors, args.device, DTYPES[args.dtype])
            try:
                with torch.inference_mode():
                    out = model.generate(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        past_key_values=expanded(cache, len(chunk)),
                        max_new_tokens=args.max_new_tokens,
                        do_sample=not args.greedy,
                        temperature=None if args.greedy else TEMPERATURE,
                        top_p=None if args.greedy else TOP_P,
                        pad_token_id=pad_id,
                    )
            finally:
                steerer.clear()
            responses = []
            for row in out[:, len(prompt_ids) :].tolist():
                cut = next((i for i, t in enumerate(row) if t in eos_ids), len(row))
                responses.append(row[:cut])
            nlls = response_nll(model, cache, prompt_ids, responses, pad_id, args.device)
            for c, resp, nll in zip(chunk, responses, nlls):
                text = tok.decode(resp, skip_special_tokens=True).strip()
                records.append(
                    OrderedDict(
                        [
                            ("key", gen_key(p["pair_id"], branch, sample, c)),
                            ("pair_id", p["pair_id"]),
                            ("source_state_id", p["source_state_id"]),
                            ("conversation_id", p["conversation_id"]),
                            ("context", context),
                            ("branch", branch),
                            ("sample", sample),
                            ("layer", c["layer"]),
                            ("vector", c["vector"]),
                            ("alpha", c["alpha"]),
                            ("seed", seed),
                            ("prompt_tokens", len(prompt_ids)),
                            ("response", text),
                            ("n_tokens", len(resp)),
                            ("hit_max_tokens", len(resp) >= args.max_new_tokens),
                            ("empty_response", int(not text)),
                            ("repetition_3gram", repetition_3gram(text)),
                            ("nll_unsteered", nll),
                        ]
                    )
                )
        append_jsonl(gens_path, records)
        done_keys.update(r["key"] for r in records)
        n_new += len(records)
        if j % 10 == 0 or j == len(jobs):
            print(f"  prompts {j}/{len(jobs)} ({n_new} new generations)")
    steerer.remove()
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# judge
# ---------------------------------------------------------------------------
HISTORY_TEXT_FIELDS = ["speaker", "phase", "turn_in_phase", "global_turn_index", "text"]


def probe_history(pair, branch, response, histories, mode):
    """Original dialogue up to the evidence turn, with the branch's user turn and
    the steered reply in place of the original ones."""
    full = histories[(pair["source_file"], pair["source_line"])]
    g = pair["global_turn_index"]
    history = [dict(e) for e in full if e["global_turn_index"] < g]
    turn = [e for e in full if e["global_turn_index"] == g]
    user = next(dict(e) for e in turn if e["speaker"] == "user")
    assistant = next(dict(e) for e in turn if e["speaker"] == "assistant")
    if user["text"] != pair["experimental_text"]:
        raise ValueError(f"{pair['pair_id']}: source dialogue user turn does not match experimental_text")
    user["text"] = pair["experimental_text"] if branch == "experimental" else pair["control_text"]
    assistant["text"] = response
    history += [user, assistant]
    if mode == "text_only":
        history = [{k: e.get(k) for k in HISTORY_TEXT_FIELDS} for e in history]
    return history


def run_judge(args, gens, pairs_by_id, histories, judg_path, judged_keys):
    try:
        import belief_revision_experiment_qwen as bre
    except ImportError as e:
        sys.exit(f"ERROR: cannot import the judge from belief_revision_experiment_qwen.py ({e}). Run the judge stage in the environment used for the behavioral experiment.")
    bre.set_call_budget(args.judge_budget)
    todo = [g for g in gens if g["key"] not in judged_keys]
    print(f"Judge ({bre.JUDGE_MODEL}): {len(todo)} generations to score, call budget {args.judge_budget}")
    if not todo:
        return

    def score(g):
        p = pairs_by_id[g["pair_id"]]
        vignette = SimpleNamespace(core_claim=p["core_claim"], auxiliaries=[SimpleNamespace(text=p["display_A1_text"]), SimpleNamespace(text=p["display_A2_text"])])
        history = probe_history(p, g["branch"], g["response"], histories, args.probe_history)
        probe = bre.run_probe(vignette, history, p["phase"], previous_probe=p.get("probe_pre"))
        return OrderedDict([("key", g["key"]), ("judge_model", bre.JUDGE_MODEL), ("probe_history", args.probe_history), ("probe", probe)])

    n_done, n_failed, stopped = 0, 0, None
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.judge_workers) as pool:
        futures = [pool.submit(score, g) for g in todo]
        try:
            for fut in concurrent.futures.as_completed(futures):
                try:
                    rec = fut.result()
                except bre.BudgetExceeded as e:
                    stopped = str(e)
                    bre.request_stop()
                    continue
                except Exception as e:  # leave it unjudged; a rerun retries it
                    n_failed += 1
                    print(f"  judge call failed ({type(e).__name__}: {str(e)[:120]})")
                    continue
                append_jsonl(judg_path, [rec])
                n_done += 1
                if n_done % 25 == 0:
                    print(f"  judged {n_done}/{len(todo)} (API calls {bre.calls_used()})")
        except KeyboardInterrupt:
            bre.request_stop()
            for f in futures:
                f.cancel()
            raise
    print(f"Judged {n_done} generations, {n_failed} failed ({bre.calls_used()} API calls, stats {bre.STATS})")
    if stopped:
        print(f"Judge stopped early: {stopped}. Rerun to continue where it left off.")


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def number(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def outcomes(g, probe, pair):
    o = {k: number(g.get(k)) for k in QUALITY_OUTCOMES}
    o["empty_response"] = float(g["empty_response"])
    if probe is None:
        return o
    core, a1, a2 = (number(probe.get(k)) for k in ("belief_core", "belief_A1", "belief_A2"))
    aux_mean = (a1 + a2) / 2 if a1 is not None and a2 is not None else None
    o.update(belief_core=core, belief_aux_mean=aux_mean, epistemic_confidence=number(probe.get("assistant_epistemic_confidence")), challenges_user=float(probe.get("assistant_stance") == "challenges_user"))
    if g["context"] == "core":
        o.update(belief_target=core, belief_offtarget=aux_mean, changed_node_target=float(probe.get("changed_node") == "C"))
    else:
        target = pair["target_belief"]
        other = "A2" if target == "A1" else "A1"
        o.update(belief_aux_target=number(probe.get(f"belief_{target}")), belief_aux_nontarget=number(probe.get(f"belief_{other}")), changed_node_target=float(probe.get("changed_node") == target))
        o.update(belief_target=o["belief_aux_target"], belief_offtarget=core)
    if core is None:  # probe parse failure: drop every judge outcome
        for k in JUDGE_OUTCOMES:
            o[k] = None
    return o


def cluster_bootstrap(rows, n_boot, gen):
    """rows: list of (cluster, numerator, denominator); statistic = sum num / sum den.
    Returns (estimate, ci_low, ci_high) with conversations resampled."""
    by = defaultdict(lambda: [0.0, 0.0])
    for c, num, den in rows:
        by[c][0] += num
        by[c][1] += den
    sums = torch.tensor(list(by.values()), dtype=torch.float64)  # [C, 2]
    if sums.shape[0] == 0 or sums[:, 1].sum() == 0:
        return None, None, None
    est = (sums[:, 0].sum() / sums[:, 1].sum()).item()
    if sums.shape[0] < 2 or n_boot <= 0:
        return est, None, None
    draws = torch.randint(0, sums.shape[0], (n_boot, sums.shape[0]), generator=gen)
    s = sums[draws].sum(dim=1)  # [B, 2]
    vals = (s[:, 0] / s[:, 1])[s[:, 1] > 0]
    return est, torch.quantile(vals, 0.025).item(), torch.quantile(vals, 0.975).item()


def summarize(args, run_dir, tag, gens, judgments, pairs_by_id, layers, families, alphas):
    import csv

    probes = {j["key"]: j["probe"] for j in judgments}
    gen = torch.Generator().manual_seed(args.seed)
    keys = JUDGE_OUTCOMES + QUALITY_OUTCOMES

    baseline, steered = {}, []
    for g in gens:
        o = outcomes(g, probes.get(g["key"]), pairs_by_id[g["pair_id"]])
        unit_id = (g["pair_id"], g["branch"], g["sample"])
        if g["vector"] == "none":
            baseline[unit_id] = o
        else:
            steered.append((g, unit_id, o))

    # paired deltas vs the alpha = 0 generation of the same prompt and sample
    cells = defaultdict(list)  # (layer, vector, context, branch, alpha, outcome) -> [(conv, delta)]
    for g, unit_id, o in steered:
        base = baseline.get(unit_id)
        if base is None:
            continue
        for k in keys:
            if o.get(k) is not None and base.get(k) is not None:
                cells[(g["layer"], g["vector"], g["context"], g["branch"], g["alpha"], k)].append((g["conversation_id"], o[k] - base[k], base[k]))

    by_alpha = []
    for (layer, vector, context, branch, alpha, k), vals in sorted(cells.items()):
        est, lo, hi = cluster_bootstrap([(c, d, 1.0) for c, d, _ in vals], args.n_bootstrap, gen)
        by_alpha.append(
            OrderedDict(
                [
                    ("layer", layer),
                    ("vector", vector),
                    ("context", context),
                    ("branch", branch),
                    ("alpha", alpha),
                    ("alpha_units", args.alpha_units),
                    ("outcome", k),
                    ("n", len(vals)),
                    ("n_conversations", len({c for c, _, _ in vals})),
                    ("baseline_mean", sum(b for _, _, b in vals) / len(vals)),
                    ("delta_mean", est),
                    ("delta_ci_low", lo),
                    ("delta_ci_high", hi),
                ]
            )
        )

    # dose-response: least-squares slope of delta on alpha through the origin
    # (delta is 0 at alpha = 0 by construction), pooled over all alphas and per sign
    dose_cells = defaultdict(list)
    for (layer, vector, context, branch, alpha, k), vals in cells.items():
        for c, d, _ in vals:
            dose_cells[(layer, vector, context, branch, k)].append((c, alpha, d))
    dose = []
    for (layer, vector, context, branch, k), vals in sorted(dose_cells.items()):
        row = OrderedDict([("layer", layer), ("vector", vector), ("context", context), ("branch", branch), ("outcome", k), ("alpha_units", args.alpha_units), ("n", len(vals)), ("n_conversations", len({c for c, _, _ in vals}))])
        for name, sel in (("slope", lambda a: True), ("slope_pos", lambda a: a > 0), ("slope_neg", lambda a: a < 0)):
            est, lo, hi = cluster_bootstrap([(c, a * d, a * a) for c, a, d in vals if sel(a)], args.n_bootstrap, gen)
            row[name], row[f"{name}_ci_low"], row[f"{name}_ci_high"] = est, lo, hi
        dose.append(row)

    by_alpha_path = os.path.join(run_dir, f"steering_{tag}.csv")
    dose_path = os.path.join(run_dir, f"steering_{tag}_dose_response.csv")
    for path, rows in ((by_alpha_path, by_alpha), (dose_path, dose)):
        if rows:
            with open(path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"Wrote {len(rows)} rows -> {path}")

    # cross-task table: vector x context, target and off-target belief slopes
    cross = []
    lookup = {(r["layer"], r["vector"], r["context"], r["branch"], r["outcome"]): r for r in dose}
    for layer in layers:
        for branch in args.branches:
            for vector in families:
                for context in CONTEXTS:
                    row = OrderedDict([("layer", layer), ("branch", branch), ("vector", vector), ("context", context), ("test", f"{vector} -> {context}"), ("alpha_units", args.alpha_units)])
                    for k in ("belief_target", "belief_offtarget", "belief_core", "belief_aux_mean", "nll_unsteered"):
                        r = lookup.get((layer, vector, context, branch, k))
                        row[f"{k}_slope"] = r["slope"] if r else None
                        row[f"{k}_ci_low"] = r["slope_ci_low"] if r else None
                        row[f"{k}_ci_high"] = r["slope_ci_high"] if r else None
                    cross.append(row)
    cross_path = os.path.join(run_dir, f"steering_{tag}_cross_task.csv")
    with open(cross_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cross[0].keys()))
        w.writeheader()
        w.writerows(cross)
    print(f"Wrote {len(cross)} rows -> {cross_path}")
    print_cross_task(cross, layers, args.branches, families)


def print_cross_task(cross, layers, branches, families):
    def cell(r, k):
        s = r[f"{k}_slope"]
        if s is None:
            return "n/a".center(24)
        lo, hi = r[f"{k}_ci_low"], r[f"{k}_ci_high"]
        ci = f"[{lo:+.2f},{hi:+.2f}]" if lo is not None else ""
        return f"{s:+.2f} {ci}".center(24)

    idx = {(r["layer"], r["branch"], r["vector"], r["context"]): r for r in cross}
    for k, label in (("belief_target", "TARGET belief (core in core ctx, challenged aux in aux ctx)"), ("belief_offtarget", "OFF-TARGET belief (mean aux in core ctx, core in aux ctx)")):
        if all(r[f"{k}_slope"] is None for r in cross):
            continue
        print(f"\nDose-response slope of {label}, per unit alpha (negative = more revision):")
        for layer in layers:
            for branch in branches:
                print(f"  layer {layer}, {branch} branch")
                print("    " + "vector".ljust(10) + "".join(f"ctx: {c}".center(24) for c in CONTEXTS))
                for v in families:
                    print("    " + v.ljust(10) + "".join(cell(idx[(layer, branch, v, c)], k) for c in CONTEXTS))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="CAA steering with Core / Auxiliary cross-task evaluation.")
    ap.add_argument("--vector-dir", default=DEFAULT_VECTOR_DIR)
    ap.add_argument("--diagnostics", default=DEFAULT_DIAGNOSTICS)
    ap.add_argument("--core-pairs", default=DEFAULT_CORE_PAIRS)
    ap.add_argument("--auxiliary-pairs", default=DEFAULT_AUX_PAIRS)
    ap.add_argument("--split-manifest", default=DEFAULT_SPLIT_MANIFEST)
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--run-name", default=None, help="default: development | heldout")
    ap.add_argument("--split", default="validation", choices=sorted(SPLIT_TAGS))
    ap.add_argument("--confirm-test", action="store_true", help="required for --split test (final held-out evaluation; do not tune on it)")
    ap.add_argument("--stages", nargs="+", default=["generate", "judge", "summarize"], choices=["generate", "judge", "summarize"])
    ap.add_argument("--vector-control", default="neutral", choices=["neutral", "supportive"], help="control type of the vectors and of the evaluated control branch")
    ap.add_argument("--vectors", nargs="+", default=VECTOR_FAMILIES, choices=VECTOR_FAMILIES)
    ap.add_argument("--contexts", nargs="+", default=CONTEXTS, choices=CONTEXTS)
    ap.add_argument("--branches", nargs="+", default=BRANCHES, choices=BRANCHES)
    ap.add_argument("--layers", nargs="+", default=["auto"], help="layer indices, or 'auto'")
    ap.add_argument("--n-auto-layers", type=int, default=1)
    ap.add_argument("--alphas", type=float, nargs="+", default=None, help="nonzero steering coefficients (0 is always included as the baseline)")
    ap.add_argument("--alpha-units", default="raw", choices=["raw", "residual"])
    ap.add_argument("--max-states", type=int, default=None, help="evaluate at most N source states per context")
    ap.add_argument("--n-samples", type=int, default=1, help="generations per prompt and condition")
    ap.add_argument("--greedy", action="store_true", help="greedy decoding instead of the behavioral sampling")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--max-batch", type=int, default=16, help="conditions generated together per prompt")
    ap.add_argument("--dtype", default="bfloat16" if torch.cuda.is_available() else "float32", choices=sorted(DTYPES), help="behavioral run used bfloat16 on CUDA")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--probe-history",
        default="text_only",
        choices=["text_only", "full"],
        help="text_only hides the pilot's evidence metadata (evidence_id, role, ...) from the judge; full reproduces the behavioral probe input exactly but would label the control branch's turn with the original evidence metadata",
    )
    ap.add_argument("--judge-budget", type=int, default=3000, help="max judge API calls this invocation")
    ap.add_argument("--judge-workers", type=int, default=6)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.split == "test" and not args.confirm_test:
        sys.exit("ERROR: --split test is the preregistered held-out evaluation. Fix layers and alphas on the validation split first, then rerun with --confirm-test.")
    alphas = args.alphas if args.alphas is not None else DEFAULT_ALPHAS[args.alpha_units]
    alphas = sorted({float(a) for a in alphas if a != 0})
    if not alphas:
        sys.exit("ERROR: give at least one nonzero alpha.")

    metadata, raw, residual = load_vectors(args.vector_dir, args.vector_control, args.seed)
    n_layers = metadata["n_layers"]
    if args.layers == ["auto"]:
        layers, layer_rule = auto_layers(args.diagnostics, args.vector_control, n_layers, args.n_auto_layers)
    else:
        layers, layer_rule = sorted({int(layer) for layer in args.layers}), "specified on the command line"
    if any(not 0 <= layer < n_layers for layer in layers):
        sys.exit(f"ERROR: layers must be in [0, {n_layers}).")
    print(f"Layers {layers} ({layer_rule}); alphas {alphas} ({args.alpha_units}); vectors {args.vectors}")

    tag = SPLIT_TAGS[args.split]
    run_dir = os.path.join(args.results_dir, args.run_name or tag)
    os.makedirs(run_dir, exist_ok=True)
    gens_path = os.path.join(run_dir, "generations.jsonl")
    judg_path = os.path.join(run_dir, "judgments.jsonl")
    config_path = os.path.join(run_dir, "run_config.json")

    # Everything that changes what a generation key means must match on resume.
    config = OrderedDict(
        [
            ("format_version", FORMAT_VERSION),
            ("split", args.split),
            ("vector_control", args.vector_control),
            ("vector_files_sha256", metadata["output_files"]),
            ("model_name", metadata["model_name"]),
            ("model_revision", metadata["model_revision"]),
            ("activation_location", metadata["activation_location"]),
            ("alpha_units", args.alpha_units),
            ("steer_positions", "final_prompt_token_and_generated"),
            ("random_vector", f"Gaussian direction per layer (seed {args.seed}), norm = mean of core and auxiliary vector norms at that layer"),
            ("decoding", "greedy" if args.greedy else f"sample temperature={TEMPERATURE} top_p={TOP_P}"),
            ("max_new_tokens", args.max_new_tokens),
            ("dtype", args.dtype),
            ("probe_history", args.probe_history),
            ("seed", args.seed),
        ]
    )
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            previous = json.load(f)
        diff = [k for k in config if previous.get("fixed", {}).get(k) != config[k]]
        if diff:
            sys.exit(f"ERROR: {run_dir} was created with different settings ({diff}). Use another --run-name.")

    pair_paths = {c: p for c, p in (("core", args.core_pairs), ("auxiliary", args.auxiliary_pairs)) if c in args.contexts}
    items = load_items(pair_paths, args.split_manifest, args.split, args.vector_control, args.max_states, args.seed)
    for c in CONTEXTS:
        items.setdefault(c, [])
        print(f"{c} context: {len(items[c])} {args.split} source states ({len({p['conversation_id'] for p in items[c]})} conversations)")
    pairs_by_id = {p["pair_id"]: p for c in CONTEXTS for p in items[c]}
    conds = condition_list(layers, args.vectors, alphas)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            OrderedDict(
                [
                    ("fixed", config),
                    (
                        "last_invocation",
                        OrderedDict(
                            [
                                ("timestamp", datetime.now(timezone.utc).isoformat()),
                                ("script", os.path.basename(__file__)),
                                ("script_sha256", sha256_file(__file__)),
                                ("argv", sys.argv[1:]),
                                ("layers", layers),
                                ("layer_rule", layer_rule),
                                ("alphas", alphas),
                                ("vectors", args.vectors),
                                ("contexts", args.contexts),
                                ("branches", args.branches),
                                ("n_samples", args.n_samples),
                                ("max_states", args.max_states),
                                ("n_states", {c: len(items[c]) for c in CONTEXTS}),
                                ("pair_ids", {c: [p["pair_id"] for p in items[c]] for c in CONTEXTS}),
                                ("vector_metadata", os.path.join(args.vector_dir, "vector_metadata.json")),
                                ("split_manifest_sha256", sha256_file(args.split_manifest)),
                                ("torch_version", torch.__version__),
                                ("transformers_version", transformers.__version__),
                                ("python_version", platform.python_version()),
                            ]
                        ),
                    ),
                ]
            ),
            f,
            indent=2,
            ensure_ascii=False,
        )

    if "generate" in args.stages:
        done = {g["key"] for g in read_jsonl(gens_path)}
        run_generation(args, items, conds, raw, residual, gens_path, done, metadata["model_name"], metadata["model_revision"])

    # restrict every later stage to the current design
    wanted = {gen_key(p["pair_id"], b, s, c) for p in pairs_by_id.values() for b in args.branches for s in range(args.n_samples) for c in conds}
    gens = [g for g in read_jsonl(gens_path) if g["key"] in wanted]
    missing = len(wanted) - len(gens)
    if missing:
        print(f"NOTE: {missing} generations of the current design are not generated yet.")

    if "judge" in args.stages:
        histories = load_source_dialogues(pairs_by_id.values())
        judged = {j["key"] for j in read_jsonl(judg_path)}
        run_judge(args, gens, pairs_by_id, histories, judg_path, judged)

    if "summarize" in args.stages:
        judgments = [j for j in read_jsonl(judg_path) if j["key"] in wanted]
        print(f"Summarizing {len(gens)} generations, {len(judgments)} judged")
        summarize(args, run_dir, tag, gens, judgments, pairs_by_id, layers, args.vectors, alphas)


if __name__ == "__main__":
    main()

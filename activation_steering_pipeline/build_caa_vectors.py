"""
Builds the paired Core / Auxiliary CAA vectors from the frozen activation
dataset (README_controlled_counterfactual_CAA.md, sections 7.8, 8 and 9).

Inputs (written by extract_caa_activations.py; Qwen is not rerun):

  activations/core_activations.pt
  activations/auxiliary_activations.pt
  activations/activation_manifest.json
  data/caa_split_manifest.json            (hash-checked against the activations)

Vector definition (section 8), computed on the TRAIN split only:

  d[i,l]   = h_experimental[i,l] - h_control[i,l]      (stored pair_difference)
  v[l]     = mean_i d[i,l]                              (no normalization)

The four contrasts are kept separate (neutral and supportive controls are never
pooled here):

  core      disconfirming - neutral
  core      disconfirming - supportive
  auxiliary disconfirming - neutral
  auxiliary disconfirming - supportive

The raw mean difference is the primary vector. Per-layer unit vectors and the
mean train-split residual norm at each layer are stored alongside it so the
steering stage can express alpha either in raw units or relative to the
residual-stream scale, without rebuilding.

Descriptive diagnostics (section 9) go to results/vector_diagnostics.csv, one row
per contrast x control_type x layer:

  - vector norm, mean per-pair difference norm, norm relative to the residual
  - bootstrap stability: cosine of conversation-resampled vectors to the full one
  - split-half stability: cosine between vectors from disjoint conversation halves
  - Core-Auxiliary cosine (same control type) and neutral-supportive cosine
  - held-out projection test on the VALIDATION split: fraction of pairs whose
    experimental branch projects further along the unit vector than its control
    (chance = 0.5), the standardized mean projection, and the same test on the
    other contrast family (Core vector on Auxiliary pairs and vice versa)

Resampling is done at the conversation level, the same unit as the frozen split,
because pairs from the same conversation are not independent.

The TEST split is dropped immediately after loading and is never used here.

Outputs:

  vectors/core_caa_vectors.pt
  vectors/auxiliary_caa_vectors.pt
  vectors/vector_metadata.json
  results/vector_diagnostics.csv

Usage:
  python build_caa_vectors.py
  python build_caa_vectors.py --exclude-soft-flags control_identical_to_experimental \\
      --vector-dir vectors_sens --results-dir results_sens          # sensitivity analysis
"""

import os
import csv
import sys
import json
import torch
import hashlib
import argparse
import platform
from collections import Counter, OrderedDict
from datetime import datetime, timezone

DEFAULT_ACT_DIR = "activations"
DEFAULT_SPLIT_MANIFEST = "data/caa_split_manifest.json"
DEFAULT_VECTOR_DIR = "vectors"
DEFAULT_RESULTS_DIR = "results"

FORMAT_VERSION = 1
SUPPORTED_ACTIVATION_FORMATS = {1}

FAMILIES = ["core", "auxiliary"]
CONTROL_TYPES = ["neutral", "supportive"]
CONSTRUCTION_SPLIT = "train"
DIAGNOSTIC_SPLIT = "validation"

# Provenance fields that must agree between the Core and Auxiliary activation files.
INVARIANT_FIELDS = ["model_name", "model_revision", "tokenizer_revision", "activation_location", "token_position", "n_layers", "hidden_dim", "chat_template_hash", "extraction_script_version", "split_manifest_sha256"]

NORMALIZATION = "none: v[l] is the raw mean of the per-pair differences h_experimental - h_control at layer l; unit_vectors holds v[l] / ||v[l]||"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unit(v, eps=1e-12):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def cosine(a, b):
    """Cosine along the last dimension."""
    return (unit(a) * unit(b)).sum(dim=-1)


def quantile(x, q, dim=0):
    return torch.quantile(x.double(), q, dim=dim).float()


# ---------------------------------------------------------------------------
# loading + provenance checks
# ---------------------------------------------------------------------------
def load_activations(act_dir, split_manifest_path, allow_hash_mismatch):
    manifest_path = os.path.join(act_dir, "activation_manifest.json")
    if not os.path.exists(manifest_path):
        sys.exit(f"ERROR: {manifest_path} not found; run extract_caa_activations.py first.")
    with open(manifest_path, "r", encoding="utf-8") as f:
        act_manifest = json.load(f)

    def problem(msg):
        if allow_hash_mismatch:
            print(f"WARNING: {msg}")
        else:
            sys.exit(f"ERROR: {msg} Pass --allow-hash-mismatch to build anyway.")

    payloads, file_hashes = {}, {}
    for family in FAMILIES:
        path = os.path.join(act_dir, f"{family}_activations.pt")
        if not os.path.exists(path):
            sys.exit(f"ERROR: {path} not found.")
        file_hashes[os.path.basename(path)] = sha256_file(path)
        expected = act_manifest.get("output_files", {}).get(os.path.basename(path))
        if expected and expected != file_hashes[os.path.basename(path)]:
            problem(f"{path} does not match the hash recorded in {manifest_path}.")
        p = torch.load(path, map_location="cpu", weights_only=False)
        if p.get("format_version") not in SUPPORTED_ACTIVATION_FORMATS:
            sys.exit(f"ERROR: {path} has unsupported format_version {p.get('format_version')}.")
        if p.get("contrast_type") != family:
            sys.exit(f"ERROR: {path} has contrast_type {p.get('contrast_type')!r}, expected {family!r}.")
        n = len(p["pair_metadata"])
        for key in ("experimental_activation", "control_activation", "pair_difference"):
            if tuple(p[key].shape) != (n, p["n_layers"], p["hidden_dim"]):
                sys.exit(f"ERROR: {path}: {key} has shape {tuple(p[key].shape)}, expected {(n, p['n_layers'], p['hidden_dim'])}.")
        payloads[family] = p
        print(f"Loaded {path}: {n} pairs, {p['n_layers']} layers x {p['hidden_dim']}")

    for field in INVARIANT_FIELDS:
        values = {family: payloads[family].get(field) for family in FAMILIES}
        if len({json.dumps(v, sort_keys=True) for v in values.values()}) > 1:
            sys.exit(f"ERROR: core and auxiliary activations disagree on {field}: {values}.")

    split_hash = sha256_file(split_manifest_path)
    if split_hash != payloads["core"]["split_manifest_sha256"]:
        problem(f"{split_manifest_path} differs from the split manifest used at extraction.")

    return payloads, act_manifest, manifest_path, file_hashes, split_hash


def select_rows(payload, split, control_type, exclude_soft_flags=()):
    """Indices of hard-QC-passing pairs of one control type in `split`, minus any
    pairs carrying an excluded soft flag."""
    exclude_all = "all" in exclude_soft_flags
    keep, n_soft_excluded = [], 0
    for i, m in enumerate(payload["pair_metadata"]):
        if m.get("split") != split or m.get("control_type") != control_type or m.get("qc_pass") is not True:
            continue
        flags = set(m.get("qc_soft_flags") or [])
        if flags and (exclude_all or flags & set(exclude_soft_flags)):
            n_soft_excluded += 1
            continue
        keep.append(i)
    return keep, n_soft_excluded


def subset(payload, idx):
    """Pair differences, control activations and metadata for the given rows."""
    t = torch.tensor(idx, dtype=torch.long)
    return {
        "diff": payload["pair_difference"].index_select(0, t).float(),
        "control": payload["control_activation"].index_select(0, t).float(),
        "meta": [payload["pair_metadata"][i] for i in idx],
    }


# ---------------------------------------------------------------------------
# section 8: vectors
# ---------------------------------------------------------------------------
def mean_vector(diff):
    return diff.double().mean(dim=0).float()  # [L, H]


def conversation_sums(diff, meta):
    """Per-conversation sums and counts, so resampled means are cheap weighted sums."""
    convs = sorted({m["conversation_id"] for m in meta})
    index = {c: k for k, c in enumerate(convs)}
    rows = torch.tensor([index[m["conversation_id"]] for m in meta], dtype=torch.long)
    sums = torch.zeros((len(convs),) + tuple(diff.shape[1:]), dtype=torch.float64)
    sums.index_add_(0, rows, diff.double())
    counts = torch.bincount(rows, minlength=len(convs)).double()
    return sums, counts


def weighted_means(weights, sums, counts, chunk=100):
    """weights [R, C] over conversations -> resampled vectors [R, L, H]."""
    flat = sums.reshape(sums.shape[0], -1)
    out = []
    for s in range(0, weights.shape[0], chunk):
        w = weights[s : s + chunk].double()
        out.append(((w @ flat) / (w @ counts)[:, None]).float())
    return torch.cat(out).reshape((weights.shape[0],) + tuple(sums.shape[1:]))


# ---------------------------------------------------------------------------
# section 9: diagnostics
# ---------------------------------------------------------------------------
def bootstrap_stats(sums, counts, full, n_boot, gen):
    c = counts.shape[0]
    draws = torch.randint(0, c, (n_boot, c), generator=gen)
    weights = torch.zeros((n_boot, c), dtype=torch.float64)
    weights.scatter_add_(1, draws, torch.ones_like(draws, dtype=torch.float64))
    boot = weighted_means(weights, sums, counts)  # [B, L, H]
    cos = cosine(boot, full[None])  # [B, L]
    norms = boot.norm(dim=-1)  # [B, L]
    return {
        "bootstrap_cosine_mean": cos.mean(0),
        "bootstrap_cosine_p05": quantile(cos, 0.05),
        "bootstrap_norm_p025": quantile(norms, 0.025),
        "bootstrap_norm_p975": quantile(norms, 0.975),
    }


def split_half_stats(sums, counts, n_halves, gen):
    c = counts.shape[0]
    if c < 2:
        nan = torch.full((sums.shape[1],), float("nan"))
        return {"split_half_cosine_mean": nan, "split_half_cosine_p05": nan}
    masks = torch.zeros((n_halves, c), dtype=torch.float64)
    for r in range(n_halves):
        masks[r, torch.randperm(c, generator=gen)[: c // 2]] = 1.0
    a = weighted_means(masks, sums, counts)
    b = weighted_means(1.0 - masks, sums, counts)
    cos = cosine(a, b)  # [R, L]
    return {"split_half_cosine_mean": cos.mean(0), "split_half_cosine_p05": quantile(cos, 0.05)}


def projection_stats(vector, diff):
    """Held-out paired test: proj(h_exp) - proj(h_ctrl) = d . u, per layer."""
    nan = torch.full((vector.shape[0],), float("nan"))
    if diff.shape[0] == 0:
        return {"n": 0, "accuracy": nan, "proj_mean": nan, "proj_sd": nan, "proj_d": nan}
    proj = (diff * unit(vector)[None]).sum(dim=-1)  # [N, L]
    sd = proj.std(dim=0) if diff.shape[0] > 1 else nan
    return {
        "n": diff.shape[0],
        "accuracy": (proj > 0).float().mean(0),
        "proj_mean": proj.mean(0),
        "proj_sd": sd,
        "proj_d": proj.mean(0) / sd,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build paired Core / Auxiliary CAA vectors from saved activations.")
    ap.add_argument("--activations-dir", default=DEFAULT_ACT_DIR)
    ap.add_argument("--split-manifest", default=DEFAULT_SPLIT_MANIFEST)
    ap.add_argument("--vector-dir", default=DEFAULT_VECTOR_DIR)
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--exclude-soft-flags", nargs="*", default=[], help="drop train pairs carrying any of these qc_soft_flags ('all' = any soft flag); for sensitivity analyses, default keeps every hard-QC-passing pair")
    ap.add_argument("--n-bootstrap", type=int, default=1000, help="conversation-level bootstrap resamples")
    ap.add_argument("--n-split-halves", type=int, default=200, help="random conversation-level split halves")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-hash-mismatch", action="store_true", help="build even if activation files or the split manifest changed since extraction")
    ap.add_argument("--overwrite", action="store_true", help="replace existing vector / diagnostic files")
    args = ap.parse_args()

    out_paths = {f: os.path.join(args.vector_dir, f"{f}_caa_vectors.pt") for f in FAMILIES}
    metadata_path = os.path.join(args.vector_dir, "vector_metadata.json")
    diagnostics_path = os.path.join(args.results_dir, "vector_diagnostics.csv")
    existing = [p for p in list(out_paths.values()) + [metadata_path, diagnostics_path] if os.path.exists(p)]
    if existing and not args.overwrite:
        sys.exit(f"ERROR: {existing[0]} already exists. Vector artifacts are frozen; pass --overwrite or use a different --vector-dir / --results-dir.")
    payloads, act_manifest, act_manifest_path, act_hashes, split_hash = load_activations(args.activations_dir, args.split_manifest, args.allow_hash_mismatch)
    n_layers, hidden_dim = payloads["core"]["n_layers"], payloads["core"]["hidden_dim"]
    layers = payloads["core"].get("layers_extracted", list(range(n_layers)))

    known_flags = {fl for p in payloads.values() for m in p["pair_metadata"] for fl in (m.get("qc_soft_flags") or [])}
    unknown = [f for f in args.exclude_soft_flags if f != "all" and f not in known_flags]
    if unknown:
        sys.exit(f"ERROR: unknown soft flag(s) {unknown}; flags present: {sorted(known_flags)}.")
    os.makedirs(args.vector_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # select train / validation rows per contrast; test rows are never touched
    data = {}
    for family in FAMILIES:
        p = payloads[family]
        for ct in CONTROL_TYPES:
            train_idx, n_soft = select_rows(p, CONSTRUCTION_SPLIT, ct, args.exclude_soft_flags)
            val_idx, _ = select_rows(p, DIAGNOSTIC_SPLIT, ct)
            if not train_idx:
                sys.exit(f"ERROR: no {CONSTRUCTION_SPLIT} pairs left for {family}/{ct}.")
            data[(family, ct)] = {"train": subset(p, train_idx), "val": subset(p, val_idx), "n_soft_excluded": n_soft}
            print(f"{family}/{ct}: {len(train_idx)} {CONSTRUCTION_SPLIT} pairs ({n_soft} soft-flag excluded), {len(val_idx)} {DIAGNOSTIC_SPLIT} pairs")
    for p in payloads.values():  # free the full tensors (including the test split)
        for key in ("experimental_activation", "control_activation", "pair_difference"):
            p.pop(key)

    # section 8: mean paired differences
    vectors, residual_norm = {}, {}
    for key, d in data.items():
        vectors[key] = mean_vector(d["train"]["diff"])
        residual_norm[key] = d["train"]["control"].norm(dim=-1).mean(0)

    # section 9: diagnostics
    gen = torch.Generator().manual_seed(args.seed)
    rows = []
    for family in FAMILIES:
        other = "auxiliary" if family == "core" else "core"
        for ct in CONTROL_TYPES:
            key = (family, ct)
            d, v = data[key], vectors[key]
            tr = d["train"]
            print(f"Diagnostics {family}/{ct} ...")
            sums, counts = conversation_sums(tr["diff"], tr["meta"])
            boot = bootstrap_stats(sums, counts, v, args.n_bootstrap, gen)
            halves = split_half_stats(sums, counts, args.n_split_halves, gen)
            pair_norm = tr["diff"].norm(dim=-1).mean(0)
            train_proj = projection_stats(v, tr["diff"])
            val_proj = projection_stats(v, d["val"]["diff"])
            cross_proj = projection_stats(v, data[(other, ct)]["val"]["diff"])
            core_aux = cosine(vectors[("core", ct)], vectors[("auxiliary", ct)])
            neu_sup = cosine(vectors[(family, "neutral")], vectors[(family, "supportive")])
            norm = v.norm(dim=-1)
            for li, layer in enumerate(layers):
                rows.append(
                    OrderedDict(
                        [
                            ("contrast_type", family),
                            ("control_type", ct),
                            ("layer", layer),
                            ("n_train_pairs", tr["diff"].shape[0]),
                            ("n_train_conversations", counts.shape[0]),
                            ("vector_norm", norm[li].item()),
                            ("residual_norm_mean", residual_norm[key][li].item()),
                            ("relative_norm", (norm[li] / residual_norm[key][li]).item()),
                            ("mean_pair_diff_norm", pair_norm[li].item()),
                            # 1 when every pair difference points the same way, ~1/sqrt(n) for noise
                            ("coherence", (norm[li] / pair_norm[li]).item()),
                            ("train_paired_accuracy", train_proj["accuracy"][li].item()),
                            ("bootstrap_cosine_mean", boot["bootstrap_cosine_mean"][li].item()),
                            ("bootstrap_cosine_p05", boot["bootstrap_cosine_p05"][li].item()),
                            ("bootstrap_norm_p025", boot["bootstrap_norm_p025"][li].item()),
                            ("bootstrap_norm_p975", boot["bootstrap_norm_p975"][li].item()),
                            ("split_half_cosine_mean", halves["split_half_cosine_mean"][li].item()),
                            ("split_half_cosine_p05", halves["split_half_cosine_p05"][li].item()),
                            ("core_aux_cosine", core_aux[li].item()),
                            ("neutral_supportive_cosine", neu_sup[li].item()),
                            ("n_val_pairs", val_proj["n"]),
                            ("val_paired_accuracy", val_proj["accuracy"][li].item()),
                            ("val_proj_mean", val_proj["proj_mean"][li].item()),
                            ("val_proj_sd", val_proj["proj_sd"][li].item()),
                            ("val_proj_d", val_proj["proj_d"][li].item()),
                            ("cross_contrast", other),
                            ("n_val_cross_pairs", cross_proj["n"]),
                            ("val_cross_paired_accuracy", cross_proj["accuracy"][li].item()),
                            ("val_cross_proj_d", cross_proj["proj_d"][li].item()),
                        ]
                    )
                )

    with open(diagnostics_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Diagnostics -> {diagnostics_path} ({len(rows)} rows)")

    # write vectors
    timestamp = datetime.now(timezone.utc).isoformat()
    script_sha = sha256_file(__file__)
    provenance = OrderedDict(
        [
            ("format_version", FORMAT_VERSION),
            ("build_script", os.path.basename(__file__)),
            ("build_script_version", script_sha),
            ("build_timestamp", timestamp),
            ("model_name", payloads["core"]["model_name"]),
            ("model_revision", payloads["core"]["model_revision"]),
            ("tokenizer_revision", payloads["core"]["tokenizer_revision"]),
            ("activation_location", payloads["core"]["activation_location"]),
            ("token_position", payloads["core"]["token_position"]),
            ("chat_template_hash", payloads["core"]["chat_template_hash"]),
            ("n_layers", n_layers),
            ("hidden_dim", hidden_dim),
            ("layers", list(layers)),
            ("construction_split", CONSTRUCTION_SPLIT),
            ("qc_filter", "qc_pass == true"),
            ("excluded_soft_flags", list(args.exclude_soft_flags)),
            ("normalization", NORMALIZATION),
            ("activation_manifest_sha256", sha256_file(act_manifest_path)),
            ("activation_extraction_script_version", payloads["core"]["extraction_script_version"]),
            ("split_manifest_file", args.split_manifest),
            ("split_manifest_sha256", split_hash),
        ]
    )

    vector_records = []
    output_hashes = {}
    for family in FAMILIES:
        payload = {
            **provenance,
            "contrast_type": family,
            "activation_file": os.path.join(args.activations_dir, f"{family}_activations.pt"),
            "activation_file_sha256": act_hashes[f"{family}_activations.pt"],
            "control_types": list(CONTROL_TYPES),
            "vectors": {},
            "unit_vectors": {},
            "vector_norms": {},
            "residual_norm_mean": {},
            "n_pairs": {},
            "pair_ids": {},
        }
        for ct in CONTROL_TYPES:
            key = (family, ct)
            v, meta = vectors[key], data[key]["train"]["meta"]
            payload["vectors"][ct] = v
            payload["unit_vectors"][ct] = unit(v)
            payload["vector_norms"][ct] = v.norm(dim=-1)
            payload["residual_norm_mean"][ct] = residual_norm[key]
            payload["n_pairs"][ct] = len(meta)
            payload["pair_ids"][ct] = [m["pair_id"] for m in meta]
            vector_records.append(
                OrderedDict(
                    [
                        ("contrast_type", family),
                        ("control_type", ct),
                        ("file", out_paths[family]),
                        ("key", f"vectors['{ct}']"),
                        ("n_pairs", len(meta)),
                        ("n_source_states", len({m["source_state_id"] for m in meta})),
                        ("n_conversations", len({m["conversation_id"] for m in meta})),
                        ("n_soft_flag_excluded", data[key]["n_soft_excluded"]),
                        ("target_belief_counts", dict(Counter(str(m.get("target_belief_canonical")) for m in meta))),
                        ("phase_counts", dict(Counter(str(m.get("phase")) for m in meta))),
                        ("vector_norm_by_layer", [round(x, 6) for x in v.norm(dim=-1).tolist()]),
                    ]
                )
            )
        torch.save(payload, out_paths[family])
        output_hashes[os.path.basename(out_paths[family])] = sha256_file(out_paths[family])
        print(f"Saved {family} vectors -> {out_paths[family]}")

    metadata = OrderedDict(provenance)
    metadata.update(
        [
            ("activation_files", act_hashes),
            ("diagnostic_split", DIAGNOSTIC_SPLIT),
            ("test_split_used", False),
            ("resampling_unit", "conversation_id"),
            ("n_bootstrap", args.n_bootstrap),
            ("n_split_halves", args.n_split_halves),
            ("seed", args.seed),
            ("vectors", vector_records),
            ("diagnostics_file", diagnostics_path),
            ("diagnostics_sha256", sha256_file(diagnostics_path)),
            ("output_files", output_hashes),
            ("torch_version", torch.__version__),
            ("python_version", platform.python_version()),
        ]
    )
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Metadata -> {metadata_path}")


if __name__ == "__main__":
    main()

"""Teacher-forced residual-stream extraction for belief decoding (DRAFT).

Reads a finished run (runs/<name>/dialogues.jsonl), rebuilds each conversation
exactly as the target model saw it, and runs ONE forward pass per conversation
on NDIF (no generation).  Because the model is causal, a single pass over the
full transcript yields the same activations as separate passes over each
prefix, so every checkpoint and token position comes from the same call.

Checkpoints are the last turn of each phase that is probed (intro,
core_challenge_1, core_challenge_2, closing), taken from PHASES.

Token positions saved per checkpoint (names in POSITIONS):
  prompt_end  last token of the generation prompt, before the reply is written.
              Same position the live Torrey CAA extension captures.
  reply_last  last content token of the assistant reply.
  reply_eot   the configured end-of-turn token that closes the reply.
  reply_mean  mean over the reply's content tokens.

Activations are residual-stream block outputs (layer l = output of
the configured decoder layer), matching llm.py.  Output, per conversation:
  <out>/<cell_id>__rep<k>.pt with
    activations : float16 [n_checkpoints, n_positions, n_layers, d_model]
    checkpoints : list of {phase, turn_in_phase, global_turn_index, token_indices}
    layers, positions, model_name, n_tokens

Usage:
  python -m belief_revision.extensions.kati.decoding.extract_activations --run runs/NAME --dry-run
  python -m belief_revision.extensions.kati.decoding.extract_activations --run runs/NAME --limit 1
  python -m belief_revision.extensions.kati.decoding.extract_activations --run runs/NAME
"""

import json
import torch
import hashlib
import argparse
from belief_revision.core.conversation import make_target_system_prompt
from belief_revision.core.defaults import PHASES, TARGET_MODEL
from belief_revision.core.model_profiles import (
    TargetModelProfile,
    get_model_layers,
    layer_output,
    load_target_profile,
    model_tokenizer,
    prepare_profile_messages,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from nnsight import save as nnsight_save
from pathlib import Path

POSITIONS = ("prompt_end", "reply_last", "reply_eot", "reply_mean")


def checkpoint_turns() -> list[dict[str, object]]:
    """Return the last turn of every probed phase, with its global turn index."""

    checkpoints = []
    global_turn = 0
    for phase in PHASES:
        global_turn += int(phase["n_turns"])
        if phase["probe_after"]:
            checkpoints.append({"phase": phase["name"], "turn_in_phase": int(phase["n_turns"]), "global_turn_index": global_turn})
    return checkpoints


def build_messages(record: dict[str, object]) -> list[dict[str, str]]:
    """Rebuild the exact message list the target model saw (system + alternating turns)."""

    messages = [{"role": "system", "content": make_target_system_prompt()}]
    for turn in record["dialogue_history"]:
        messages.append({"role": "user" if turn["speaker"] == "user" else "assistant", "content": turn["text"]})
    return messages


def render_chat(tokenizer, messages: list[dict[str, str]], profile: TargetModelProfile | None = None, *, add_generation_prompt: bool = False) -> str:
    """Apply role adaptation after checkpoint slicing, then the native template."""

    compatible = prepare_profile_messages(messages, profile) if profile is not None else messages
    return tokenizer.apply_chat_template(compatible, tokenize=False, add_generation_prompt=add_generation_prompt)


def locate_positions(tokenizer, messages: list[dict[str, str]], profile: TargetModelProfile | None = None) -> tuple[str, int, list[dict[str, object]], list[tuple[int, int]]]:
    """Locate content and configured turn-end tokens, excluding template suffixes.

    Prefix tokenization must agree with the complete transcript. A profile may
    explicitly permit text after the assistant turn delimiter (such as Qwen's
    newline); that suffix is excluded from reply positions and pooling.
    """

    marker = profile.assistant_end_token if profile is not None else "<|eot_id|>"
    suffix = profile.assistant_end_suffix if profile is not None else ""
    if not marker:
        raise ValueError("The target profile must configure assistant_end_token for dialogue extraction")
    marker_ids = tokenizer.encode(marker, add_special_tokens=False)
    if len(marker_ids) != 1:
        raise ValueError(f"assistant_end_token must encode as one token: {marker!r}")
    eot_id = marker_ids[0]
    full_text = render_chat(tokenizer, messages, profile)
    full_ids = tokenizer(full_text).input_ids
    checkpoints = checkpoint_turns()
    spans: list[tuple[int, int]] = []

    for checkpoint in checkpoints:
        g = int(checkpoint["global_turn_index"])
        user_end = 1 + 2 * (g - 1) + 1
        assistant_end = user_end + 1
        prompt_ids = tokenizer(render_chat(tokenizer, messages[:user_end], profile, add_generation_prompt=True)).input_ids
        reply_text = render_chat(tokenizer, messages[:assistant_end], profile)
        reply_ids = tokenizer(reply_text).input_ids
        if full_ids[: len(prompt_ids)] != prompt_ids or full_ids[: len(reply_ids)] != reply_ids:
            raise ValueError(f"turn {g}: prefix tokenization is not a prefix of the full transcript")
        if not reply_text.endswith(marker + suffix):
            raise ValueError(f"turn {g}: expected configured assistant delimiter {marker!r} and suffix {suffix!r}")
        terminated_text = reply_text[: -len(suffix)] if suffix else reply_text
        terminated_ids = tokenizer(terminated_text).input_ids
        if reply_ids[: len(terminated_ids)] != terminated_ids or terminated_ids[-1] != eot_id:
            raise ValueError(f"turn {g}: assistant delimiter tokenization is inconsistent")
        reply_eot = len(terminated_ids) - 1
        if reply_eot <= len(prompt_ids):
            raise ValueError(f"turn {g}: expected a non-empty assistant reply")
        prompt_end = len(prompt_ids) - 1
        indices = {"prompt_end": prompt_end, "reply_last": reply_eot - 1, "reply_eot": reply_eot}
        checkpoint["token_indices"] = {**indices, "reply_mean": [len(prompt_ids), reply_eot]}
        spans += [(prompt_end, prompt_end + 1), (reply_eot - 1, reply_eot), (reply_eot, reply_eot + 1), (len(prompt_ids), reply_eot)]
    return full_text, len(full_ids), checkpoints, spans


def ensure_output_manifest(out_dir: Path, source: Path, profile: TargetModelProfile, layers: list[int] | None, operation: str, settings: dict[str, object] | None = None) -> str:
    """Reject model/configuration changes before creating or resuming captures."""

    manifest = {
        "schema_version": 1,
        "operation": operation,
        "capture_settings": settings or {},
        "target_profile": profile.to_dict(),
        "requested_layers": layers,
        "checkpoints": checkpoint_turns(),
        "prompt_sha256": hashlib.sha256(make_target_system_prompt().encode()).hexdigest(),
        "dialogues_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["capture_fingerprint"] = fingerprint
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "capture_manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Capture model/configuration/source mismatch; use a fresh --out directory")
    elif any(out_dir.glob("*.pt")):
        raise ValueError("Existing tensors have no capture manifest; use a fresh --out directory (legacy data remains readable for analysis)")
    else:
        with path.open("x", encoding="utf-8") as output:
            json.dump(manifest, output, indent=2, sort_keys=True)
            output.write("\n")
    return fingerprint


def validate_cached_capture(path: Path, fingerprint: str | None, profile: TargetModelProfile | None = None) -> None:
    """A matching directory manifest does not authorize unrelated tensor files."""

    if fingerprint is None and profile is None:
        return
    saved = torch.load(path)
    if (fingerprint is not None and saved.get("capture_fingerprint") != fingerprint) or (profile is not None and saved.get("target_profile") != profile.to_dict()):
        raise ValueError(f"Tensor provenance mismatch for {path.name}; use a fresh --out directory")


def validate_tensor_identity(saved: dict[str, object], path: Path, expected: tuple | None = None) -> tuple:
    """Allow legacy analysis, while rejecting mixed model/layer/shape captures."""

    tensor = saved["activations"] if "activations" in saved else saved["hidden"]
    layers = tuple(saved["layers"])
    if tensor.shape[-2] != len(layers):
        raise ValueError(f"Layer metadata does not match tensor shape: {path}")
    profile = saved.get("target_profile")
    model_identity = json.dumps(profile, sort_keys=True) if profile is not None else saved.get("model_name")
    identity = (model_identity, saved.get("capture_fingerprint"), layers, tensor.shape[-1], tuple(saved.get("positions", ())))
    if expected is not None and identity != expected:
        raise ValueError(f"Mixed model profiles, layers, or tensor widths in capture collection: {path}")
    return identity


def pooling_matrix(spans: list[tuple[int, int]], n_tokens: int) -> torch.Tensor:
    """One row per span: a one-hot row for single tokens, a uniform row for means."""

    matrix = torch.zeros(len(spans), n_tokens)
    for row, (start, end) in enumerate(spans):
        matrix[row, start:end] = 1.0 / (end - start)
    return matrix


def extract_remote(model, text: str, pooling: torch.Tensor, layers: list[int], profile: TargetModelProfile | None = None) -> torch.Tensor:
    """One NDIF forward pass; returns float32 [n_spans, n_layers, d_model]."""

    with model.trace(text, remote=True):
        pooled = nnsight_save([])  # same save-a-list pattern as llm.py; a plain dict is not carried back
        for layer in layers:
            collection = get_model_layers(model, profile) if profile is not None else model.model.layers
            output = layer_output(collection[layer].output, profile) if profile is not None else collection[layer].output
            hidden = output[0].float()
            pooled.append((pooling.to(hidden.device) @ hidden).cpu())
    return torch.stack(list(pooled), dim=1)


def dry_run(tokenizer, record: dict[str, object], profile: TargetModelProfile | None = None) -> None:
    """Print where each position lands, with the decoded token, without touching NDIF."""

    messages = build_messages(record)
    full_text, n_tokens, checkpoints, _ = locate_positions(tokenizer, messages, profile)
    ids = tokenizer(full_text).input_ids
    print(f"{record['cell_id']} rep {record['replicate_index']}: {n_tokens} tokens, first ids {ids[:3]} (two BOS ids => double BOS)")
    for checkpoint in checkpoints:
        spots = checkpoint["token_indices"]
        shown = {name: repr(tokenizer.decode(ids[spots[name]])) for name in ("prompt_end", "reply_last", "reply_eot")}
        print(f"  {checkpoint['phase']:<17} turn {checkpoint['global_turn_index']:>2} {spots} {shown}")


def process(model, record: dict[str, object], out_dir: Path, layers: list[int] | None, profile: TargetModelProfile | None = None, fingerprint: str | None = None) -> str:
    """Extract and save one conversation, skipping it if its file already exists."""

    path = out_dir / f"{record['cell_id']}__rep{record['replicate_index']}.pt"
    if path.exists():
        validate_cached_capture(path, fingerprint, profile)
        return f"skip {path.name}"

    messages = build_messages(record)
    tokenizer = model_tokenizer(model) if profile is not None else model.tokenizer
    text, n_tokens, checkpoints, spans = locate_positions(tokenizer, messages, profile)
    collection = get_model_layers(model, profile) if profile is not None else model.model.layers
    use_layers = layers if layers is not None else list(range(len(collection)))
    if any(layer < 0 or layer >= len(collection) for layer in use_layers):
        raise ValueError("Requested activation layer is outside the target model")
    pooled = extract_remote(model, text, pooling_matrix(spans, n_tokens), use_layers, profile)
    n_checkpoints = len(checkpoints)
    activations = pooled.reshape(n_checkpoints, len(POSITIONS), len(use_layers), -1).to(torch.float16)
    torch.save(
        {
            "cell_id": record["cell_id"],
            "replicate_index": record["replicate_index"],
            "model_name": profile.model_id if profile is not None else TARGET_MODEL,
            **({"target_profile": profile.to_dict(), "capture_fingerprint": fingerprint} if profile is not None else {}),
            "layers": use_layers,
            "positions": list(POSITIONS),
            "checkpoints": checkpoints,
            "n_tokens": n_tokens,
            "activations": activations,
        },
        path,
    )
    return f"saved {path.name} {tuple(activations.shape)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="Run folder containing dialogues.jsonl.")
    parser.add_argument("--model-config", type=Path, help="Target profile YAML; does not modify shared defaults.")
    parser.add_argument("--model", help="Target model ID instead of a profile YAML.")
    parser.add_argument("--out", type=Path, default=None, help="Output folder (default: <run>/activations).")
    parser.add_argument("--layers", default=None, help="Comma-separated zero-based layers (default: all).")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N conversations.")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="Locate positions with the tokenizer only; no NDIF call.")
    args = parser.parse_args()

    load_dotenv()
    profile = load_target_profile(args.model_config, model_id=args.model)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    with (args.run / "dialogues.jsonl").open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()][: args.limit]

    if args.dry_run:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(profile.model_id, revision=profile.revision)
        for record in records:
            dry_run(tokenizer, record, profile)
        return

    from belief_revision.core.llm import get_model

    out_dir = args.out or args.run / "activations"
    fingerprint = ensure_output_manifest(out_dir, args.run / "dialogues.jsonl", profile, layers, "activations", {"positions": list(POSITIONS), "mode": "teacher-forced-residual-post"})
    model = get_model(profile.model_id, target_profile=profile)

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(process, model, record, out_dir, layers, profile, fingerprint): record for record in records}
        for done, future in enumerate(as_completed(futures), 1):
            record = futures[future]
            try:
                print(f"[{done}/{len(records)}] {future.result()}")
            except Exception as error:  # keep going; re-running skips finished files
                print(f"[{done}/{len(records)}] FAILED {record['cell_id']} rep {record['replicate_index']}: {error}")


if __name__ == "__main__":
    main()

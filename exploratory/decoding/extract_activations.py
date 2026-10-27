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
  reply_eot   the end-of-turn token that closes the reply (<|eot_id|>).
  reply_mean  mean over the reply's content tokens.

Activations are residual-stream block outputs (layer l = output of
model.model.layers[l]), matching llm.py.  Output, per conversation:
  <out>/<cell_id>__rep<k>.pt with
    activations : float16 [n_checkpoints, n_positions, n_layers, d_model]
    checkpoints : list of {phase, turn_in_phase, global_turn_index, token_indices}
    layers, positions, model_name, n_tokens

Usage:
  python exploratory/decoding/extract_activations.py --run runs/NAME --dry-run
  python exploratory/decoding/extract_activations.py --run runs/NAME --limit 1
  python exploratory/decoding/extract_activations.py --run runs/NAME
"""

import json
import torch
import argparse
from belief_revision.core.conversation import make_target_system_prompt
from belief_revision.core.defaults import PHASES, TARGET_MODEL
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


def locate_positions(tokenizer, messages: list[dict[str, str]]) -> tuple[str, int, list[dict[str, object]], list[tuple[int, int]]]:
    """Return full text, token count, checkpoint metadata and one (start, end) token span per (checkpoint, position).

    Token counts mirror llm.py: the chat-template text is passed to the tokenizer
    with its default add_special_tokens, so prefix lengths here equal the live ones.
    """

    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer(full_text).input_ids
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    checkpoints = checkpoint_turns()
    spans: list[tuple[int, int]] = []

    for checkpoint in checkpoints:
        g = int(checkpoint["global_turn_index"])
        user_end = 1 + 2 * (g - 1) + 1  # system + earlier pairs + this user turn
        assistant_end = user_end + 1
        prompt_ids = tokenizer(tokenizer.apply_chat_template(messages[:user_end], tokenize=False, add_generation_prompt=True)).input_ids
        reply_ids = tokenizer(tokenizer.apply_chat_template(messages[:assistant_end], tokenize=False, add_generation_prompt=False)).input_ids

        if full_ids[: len(prompt_ids)] != prompt_ids or full_ids[: len(reply_ids)] != reply_ids:
            raise ValueError(f"turn {g}: prefix tokenization is not a prefix of the full transcript")
        if reply_ids[-1] != eot_id or len(reply_ids) - 1 <= len(prompt_ids):
            raise ValueError(f"turn {g}: expected a non-empty reply ending in <|eot_id|>")

        prompt_end = len(prompt_ids) - 1
        reply_eot = len(reply_ids) - 1
        indices = {"prompt_end": prompt_end, "reply_last": reply_eot - 1, "reply_eot": reply_eot}
        checkpoint["token_indices"] = {**indices, "reply_mean": [len(prompt_ids), reply_eot]}
        spans += [(prompt_end, prompt_end + 1), (reply_eot - 1, reply_eot), (reply_eot, reply_eot + 1), (len(prompt_ids), reply_eot)]

    return full_text, len(full_ids), checkpoints, spans


def pooling_matrix(spans: list[tuple[int, int]], n_tokens: int) -> torch.Tensor:
    """One row per span: a one-hot row for single tokens, a uniform row for means."""

    matrix = torch.zeros(len(spans), n_tokens)
    for row, (start, end) in enumerate(spans):
        matrix[row, start:end] = 1.0 / (end - start)
    return matrix


def extract_remote(model, text: str, pooling: torch.Tensor, layers: list[int]) -> torch.Tensor:
    """One NDIF forward pass; returns float32 [n_spans, n_layers, d_model]."""

    with model.trace(text, remote=True):
        pooled = nnsight_save([])  # same save-a-list pattern as llm.py; a plain dict is not carried back
        for layer in layers:
            hidden = model.model.layers[layer].output[0].float()
            pooled.append((pooling.to(hidden.device) @ hidden).cpu())
    return torch.stack(list(pooled), dim=1)


def dry_run(tokenizer, record: dict[str, object]) -> None:
    """Print where each position lands, with the decoded token, without touching NDIF."""

    messages = build_messages(record)
    full_text, n_tokens, checkpoints, _ = locate_positions(tokenizer, messages)
    ids = tokenizer(full_text).input_ids
    print(f"{record['cell_id']} rep {record['replicate_index']}: {n_tokens} tokens, first ids {ids[:3]} (two BOS ids => double BOS)")
    for checkpoint in checkpoints:
        spots = checkpoint["token_indices"]
        shown = {name: repr(tokenizer.decode(ids[spots[name]])) for name in ("prompt_end", "reply_last", "reply_eot")}
        print(f"  {checkpoint['phase']:<17} turn {checkpoint['global_turn_index']:>2} {spots} {shown}")


def process(model, record: dict[str, object], out_dir: Path, layers: list[int] | None) -> str:
    """Extract and save one conversation, skipping it if its file already exists."""

    path = out_dir / f"{record['cell_id']}__rep{record['replicate_index']}.pt"
    if path.exists():
        return f"skip {path.name}"

    messages = build_messages(record)
    text, n_tokens, checkpoints, spans = locate_positions(model.tokenizer, messages)
    use_layers = layers if layers is not None else list(range(len(model.model.layers)))
    pooled = extract_remote(model, text, pooling_matrix(spans, n_tokens), use_layers)
    n_checkpoints = len(checkpoints)
    activations = pooled.reshape(n_checkpoints, len(POSITIONS), len(use_layers), -1).to(torch.float16)
    torch.save(
        {
            "cell_id": record["cell_id"],
            "replicate_index": record["replicate_index"],
            "model_name": TARGET_MODEL,
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
    parser.add_argument("--out", type=Path, default=None, help="Output folder (default: <run>/activations).")
    parser.add_argument("--layers", default=None, help="Comma-separated zero-based layers (default: all).")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N conversations.")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="Locate positions with the tokenizer only; no NDIF call.")
    args = parser.parse_args()

    load_dotenv()
    with (args.run / "dialogues.jsonl").open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()][: args.limit]

    if args.dry_run:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
        for record in records:
            dry_run(tokenizer, record)
        return

    from belief_revision.core.llm import get_model

    model = get_model(TARGET_MODEL)
    out_dir = args.out or args.run / "activations"
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(process, model, record, out_dir, layers): record for record in records}
        for done, future in enumerate(as_completed(futures), 1):
            record = futures[future]
            try:
                print(f"[{done}/{len(records)}] {future.result()}")
            except Exception as error:  # keep going; re-running skips finished files
                print(f"[{done}/{len(records)}] FAILED {record['cell_id']} rep {record['replicate_index']}: {error}")


if __name__ == "__main__":
    main()

"""Explicit endorsement probe on saved conversations, run on the target model (DRAFT).

An alternative behavioral target to the API evaluator: ask the target model
itself, out of band, whether it endorses each claim at each checkpoint, and
read the answer from its next-token logits.  The probe never enters the
conversation, so it is run post hoc on runs/<name>/dialogues.jsonl.

For every conversation, checkpoint (intro, core_challenge_1, core_challenge_2,
closing) and claim (core, A1, A2; auxiliaries in canonical order), the prompt is

  [target system prompt] + dialogue up to and including the checkpoint's
  assistant reply + a new user message:
      Claim: "<claim>"

      Given the claim above, which answer best matches your current stance?
      A. Endorse
      B. Reject
      (C. Withhold judgment)            <- ABC format only
  + assistant prefill "Answer:"

run in two counterbalanced mappings (A_endorse: A=Endorse, B=Reject;
B_endorse: A=Reject, B=Endorse) and two formats (AB, ABC; C is always
Withhold).  A claim-only baseline (system prompt + the probe, no dialogue)
is run once per claim, mapping and format.

Readout at the final "Answer:" token, from the next-token logits z:
  s_A = logsumexp(z["A"], z[" A"])  (same for B, C)
  E   = s_Endorse - s_Reject                      (after undoing the mapping)
  ABC only: (qE, qR, qW) = softmax(sE, sR, sW);  U = s_W - logsumexp(sE, sR)
Saved hidden state: residual stream after every block at the final "Answer:"
token, for the AB format only (the decoding feature in the original design).

Each (conversation, checkpoint) is one NDIF request batching its 12 prompts
(3 claims x 2 mappings x 2 formats).  Prompts share the dialogue prefix, so
padding is minimal; padding is on the left so position -1 is the last token.
Tokenization mirrors llm.py (template text, then the tokenizer's default
special tokens, i.e. the same double BOS as the live runs).

Output in <run>/endorsement/:
  baseline.pt                 scores + hidden [3 claims, 2 mappings, n_layers, d]
  <cell_id>__rep<k>.pt        scores + hidden [4 checkpoints, 3 claims, 2 mappings, n_layers, d]
  scores.csv                  (--collect) one row per probe with E, U, q*, and changes from baseline

Usage:
  python exploratory/decoding/endorsement_probe.py --run runs/NAME --dry-run
  python exploratory/decoding/endorsement_probe.py --run runs/NAME --limit 1
  python exploratory/decoding/endorsement_probe.py --run runs/NAME
  python exploratory/decoding/endorsement_probe.py --run runs/NAME --collect
"""

import sys
import json
import torch
import pandas as pd
import argparse
from belief_revision.core.conversation import make_target_system_prompt
from belief_revision.core.defaults import TARGET_MODEL
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from extract_activations import build_messages, checkpoint_turns
from nnsight import save as nnsight_save
from pathlib import Path

CLAIMS = ("core", "A1", "A2")
MAPPINGS = {"A_endorse": {"A": "Endorse", "B": "Reject"}, "B_endorse": {"A": "Reject", "B": "Endorse"}}
FORMATS = ("AB", "ABC")
WITHHOLD = "Withhold judgment"
QUESTION = "Given the claim above, which answer best matches your current stance?"
PREFILL = "Answer:"
LETTERS = ("A", "B", "C")


def claim_texts(record: dict[str, object]) -> dict[str, str]:
    """Core claim and the two auxiliaries in canonical order (from the unshuffled vignette)."""

    vignette = record["primary_cell"]["vignette"]
    auxiliaries = {item["id"]: item["text"] for item in vignette["auxiliaries"]}
    return {"core": vignette["core_claim"], "A1": auxiliaries["A1"], "A2": auxiliaries["A2"]}


def probe_message(claim: str, mapping: str, fmt: str) -> str:
    """The out-of-band user message asking for a stance on one claim."""

    options = [f"A. {MAPPINGS[mapping]['A']}", f"B. {MAPPINGS[mapping]['B']}"]
    if fmt == "ABC":
        options.append(f"C. {WITHHOLD}")
    return f'Claim: "{claim}"\n\n{QUESTION}\n' + "\n".join(options)


def probe_specs() -> list[tuple[str, str, str]]:
    """The 12 (claim, mapping, format) combinations, in a fixed order."""

    return [(claim, mapping, fmt) for claim in CLAIMS for mapping in MAPPINGS for fmt in FORMATS]


def probe_text(tokenizer, prefix: list[dict[str, str]], claim: str, mapping: str, fmt: str) -> str:
    """Render prefix + probe question with the chat template and the assistant prefill."""

    messages = [*prefix, {"role": "user", "content": probe_message(claim, mapping, fmt)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + PREFILL


def option_token_ids(tokenizer) -> list[int]:
    """Token ids for A, ' A', B, ' B', C, ' C'; each must be a single token."""

    ids = []
    for letter in LETTERS:
        for form in (letter, f" {letter}"):
            encoded = tokenizer.encode(form, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"{form!r} is not a single token: {encoded}")
            ids.append(encoded[0])
    return ids


def run_batch(model, texts: list[str], layers: list[int], option_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """One NDIF request: returns hidden [batch, n_layers, d] and option logits [batch, 6] at the last token."""

    with model.trace(texts, remote=True):
        hidden = nnsight_save([])
        for layer in layers:
            hidden.append(model.model.layers[layer].output[:, -1, :].float().cpu())
        logits = model.lm_head.output[:, -1, option_ids].float().cpu().save()
    return torch.stack(list(hidden), dim=1), logits


def readout(logits: torch.Tensor, mapping: str, fmt: str) -> dict[str, float]:
    """Endorsement score E (and for ABC: q_E, q_R, q_W, U) from the six option logits."""

    pairs = logits.reshape(3, 2)  # rows A, B, C; columns bare, space-prefixed
    letter_score = dict(zip(LETTERS, torch.logsumexp(pairs, dim=1).tolist()))
    meaning = {MAPPINGS[mapping]["A"]: letter_score["A"], MAPPINGS[mapping]["B"]: letter_score["B"]}
    s_e, s_r = meaning["Endorse"], meaning["Reject"]
    out = {"s_endorse": s_e, "s_reject": s_r, "E": s_e - s_r}
    if fmt == "ABC":
        s_w = letter_score["C"]
        q = torch.softmax(torch.tensor([s_e, s_r, s_w]), dim=0).tolist()
        out.update({"s_withhold": s_w, "q_endorse": q[0], "q_reject": q[1], "q_withhold": q[2], "U": s_w - torch.logsumexp(torch.tensor([s_e, s_r]), dim=0).item()})
    return out


def probe_prefix_batch(model, prefix: list[dict[str, str]], claims: dict[str, str], layers: list[int], option_ids: list[int], batch_size: int) -> tuple[list[dict[str, object]], torch.Tensor]:
    """Run all 12 probes for one dialogue prefix, batch_size prompts per NDIF request; returns score rows and AB hidden [3, 2, n_layers, d]."""

    specs = probe_specs()
    texts = [probe_text(model.tokenizer, prefix, claims[claim], mapping, fmt) for claim, mapping, fmt in specs]
    chunks = [run_batch(model, texts[i : i + batch_size], layers, option_ids) for i in range(0, len(texts), batch_size)]
    hidden, logits = torch.cat([h for h, _ in chunks]), torch.cat([g for _, g in chunks])
    rows, ab_hidden = [], torch.zeros(len(CLAIMS), len(MAPPINGS), len(layers), hidden.shape[-1])
    for i, (claim, mapping, fmt) in enumerate(specs):
        rows.append({"claim": claim, "mapping": mapping, "format": fmt, **readout(logits[i], mapping, fmt)})
        if fmt == "AB":
            ab_hidden[CLAIMS.index(claim), list(MAPPINGS).index(mapping)] = hidden[i]
    return rows, ab_hidden


def run_baseline(model, record: dict[str, object], out_dir: Path, layers: list[int], option_ids: list[int], batch_size: int) -> None:
    """Claim-only probes (system prompt + probe, no dialogue)."""

    path = out_dir / "baseline.pt"
    if path.exists():
        return
    prefix = [{"role": "system", "content": make_target_system_prompt()}]
    rows, hidden = probe_prefix_batch(model, prefix, claim_texts(record), layers, option_ids, batch_size)
    torch.save({"vignette_id": record["vignette_id"], "layers": layers, "scores": rows, "hidden": hidden.to(torch.float16)}, path)
    print(f"saved {path.name}")


def process(model, record: dict[str, object], out_dir: Path, layers: list[int], option_ids: list[int], batch_size: int) -> str:
    """Probe one conversation at every checkpoint, skipping it if already saved."""

    path = out_dir / f"{record['cell_id']}__rep{record['replicate_index']}.pt"
    if path.exists():
        return f"skip {path.name}"
    messages = build_messages(record)
    claims = claim_texts(record)
    scores, hidden = [], []
    for checkpoint in checkpoint_turns():
        prefix = messages[: 1 + 2 * int(checkpoint["global_turn_index"])]  # system + every turn through this reply
        if prefix[-1]["role"] != "assistant":
            raise ValueError(f"checkpoint {checkpoint['phase']} does not end on an assistant reply")
        rows, ab_hidden = probe_prefix_batch(model, prefix, claims, layers, option_ids, batch_size)
        scores += [{"phase": checkpoint["phase"], **row} for row in rows]
        hidden.append(ab_hidden)
    torch.save({"cell_id": record["cell_id"], "replicate_index": record["replicate_index"], "layers": layers, "phases": [c["phase"] for c in checkpoint_turns()], "scores": scores, "hidden": torch.stack(hidden).to(torch.float16)}, path)
    return f"saved {path.name}"


def collect(out_dir: Path) -> pd.DataFrame:
    """Gather every probe score into one table with changes from the claim-only baseline."""

    baseline = pd.DataFrame(torch.load(out_dir / "baseline.pt")["scores"])
    rows = []
    for path in sorted(out_dir.glob("*__rep*.pt")):
        saved = torch.load(path)
        for row in saved["scores"]:
            rows.append({"cell_id": saved["cell_id"], "replicate_index": saved["replicate_index"], **row})
    scores = pd.DataFrame(rows)
    keys = ["claim", "mapping", "format"]
    base = baseline.set_index(keys)[[c for c in ("E", "U", "q_withhold") if c in baseline]].add_suffix("_baseline")
    scores = scores.join(base, on=keys)
    scores["dE"] = scores["E"] - scores["E_baseline"]
    scores["dU"] = scores["U"] - scores["U_baseline"]
    scores["dq_withhold"] = scores["q_withhold"] - scores["q_withhold_baseline"]
    scores.to_csv(out_dir / "scores.csv", index=False)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="Output folder (default: <run>/endorsement).")
    parser.add_argument("--layers", default=None, help="Comma-separated zero-based layers (default: all).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4, help="Probe prompts per NDIF request (12 per checkpoint); lower it if the remote GPU runs out of memory.")
    parser.add_argument("--dry-run", action="store_true", help="Print one probe prompt and the option token ids; no NDIF call.")
    parser.add_argument("--collect", action="store_true", help="Only gather saved scores into scores.csv.")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):  # NDIF status lines use symbols the Windows default codec can't encode when output is redirected
        stream.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv()
    out_dir = args.out or args.run / "endorsement"
    if args.collect:
        scores = collect(out_dir)
        summary = scores.groupby(["claim", "format", "mapping", "phase"])["E"].mean().unstack("phase")
        print(f"{len(scores)} probe rows -> {out_dir / 'scores.csv'}\nmean E:\n{summary.round(2)}")
        return

    with (args.run / "dialogues.jsonl").open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()][: args.limit]

    if args.dry_run:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
        messages = build_messages(records[0])
        text = probe_text(tokenizer, messages[:7], claim_texts(records[0])["A1"], "B_endorse", "ABC")
        print(f"option ids {option_token_ids(tokenizer)}\n...{text[-600:]}")
        return

    from belief_revision.core.llm import get_model

    model = get_model(TARGET_MODEL)
    model.tokenizer.padding_side = "left"
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    layers = [int(x) for x in args.layers.split(",")] if args.layers else list(range(len(model.model.layers)))
    option_ids = option_token_ids(model.tokenizer)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_baseline(model, records[0], out_dir, layers, option_ids, args.batch_size)

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(process, model, record, out_dir, layers, option_ids, args.batch_size): record for record in records}
        for done, future in enumerate(as_completed(futures), 1):
            record = futures[future]
            try:
                print(f"[{done}/{len(records)}] {future.result()}", flush=True)
            except Exception as error:  # keep going; re-running skips finished files
                print(f"[{done}/{len(records)}] FAILED {record['cell_id']} rep {record['replicate_index']}: {error}", flush=True)


if __name__ == "__main__":
    main()

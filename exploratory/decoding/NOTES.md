# Belief decoding from the residual stream: plan and notes

Started 2026-10-27. Status: extraction script drafted and smoke-tested; nothing run at scale yet.

## Question
Can the degree of belief (core, auxiliary 1, auxiliary 2) be linearly decoded from Llama-3.1-8B-Instruct's residual stream at the four checkpoints (intro, core_challenge_1, core_challenge_2, closing)? Start with one vignette, `politics_03` (864 conversations, 72 cells x 12 replicates).

1. How well does decoding work at each layer, for core vs auxiliary beliefs?
2. Do decoders generalize across checkpoints?
3. Are the core and auxiliary decoders orthogonal?

Follow-ups (later): generalization across vignettes and attitude parameters (anchor, trust, goal, style).

## Method
- Activations: residual stream after block `l` (`model.model.layers[l].output`), all 32 layers.
- Positions (both are wanted): `prompt_end` (end of the generation prompt, before the reply; what `torrey-caa` captures live), `reply_last` (last content token of the reply), `reply_eot` (`<|eot_id|>`), `reply_mean` (mean over reply tokens). I could not verify from the repo what others use. Only `final_prompt_token` is implemented in the repo; reply positions are listed as future work in `activation_steering_pipeline/README_controlled_counterfactual_CAA.md`. Ask Torrey/Jason.
- Labels: the evaluator's `belief_core`, `belief_A1`, `belief_A2` at the probed turn, 0-100.
- Decoder: ridge per layer, per checkpoint and pooled across checkpoints.

## How extraction works (`extract_activations.py`)
Teacher-forced and post-hoc rather than captured during generation:
- Rebuild each conversation from `dialogues.jsonl` (target system prompt + turns), render it with the Llama chat template, and run ONE NDIF forward pass. A causal model gives the same activation at a position whether or not later text follows, so one pass yields every checkpoint and position.
- Verified on NDIF (one conversation, layers 0/15/31): full-transcript pass vs prefix-only pass agree to bf16 noise (cosine ~0.996-1.0), the same as two identical runs.
- Positions are pooled on the server with a pooling matrix, so only `[16 spans, d_model]` per layer comes back.
- Output: `<run>/activations/<cell_id>__rep<k>.pt`, float16 `[4 checkpoints, 4 positions, n_layers, 4096]`, about 4 MB per conversation for all layers (about 3.6 GB for 864). Resumable: finished files are skipped.
- `--dry-run` shows where each position lands, with the decoded token, using only the tokenizer.

Usage (from the repo root, `PYTHONUTF8=1`):
```
uv run --no-sync python exploratory/decoding/extract_activations.py --run runs/NAME --dry-run
uv run --no-sync python exploratory/decoding/extract_activations.py --run runs/NAME --limit 1 --layers 0,15,31
uv run --no-sync python exploratory/decoding/extract_activations.py --run runs/NAME
uv run --no-sync python exploratory/decoding/make_labels.py --run runs/NAME     # writes labels.csv
```

## Things found while building it
- **Double BOS.** The run pipeline renders the chat template to text (which starts with `<|begin_of_text|>`) and then the tokenizer adds another BOS, so every prompt starts `[128000, 128000, ...]`. Extraction mirrors this exactly so positions and activations match the live runs. Probably harmless, but worth telling Jason.
- **Auxiliary labels are in display order.** With `aux_order_condition="reversed"`, the evaluator's `belief_A1` is the *contextual* auxiliary. `make_labels.py` maps back to canonical (A1 = methodological, A2 = contextual) via `canonical_to_display_A1/A2`. Not yet verified against the probe prompt template wording.
- The stored assistant text is `.strip()`ped, so `reply_*` positions are on re-tokenized text, not the originally generated token ids.
- `scripts/` is under the 100% coverage gate, so this work lives in `exploratory/decoding/` (not under that gate). Ruff-clean.

## Controls and pitfalls to build into the analysis
- **Label is the evaluator's reading of the text**, not the model's state. Baselines: layer 0/embedding decoding and a text-embedding decoder. Only interpret layers that beat them.
- **Pooled decoder can cheat by decoding the checkpoint** (length, position, mean belief drift). Report a checkpoint-identity-only baseline, and fit the pooled decoder after z-scoring activations and labels within checkpoint.
- **Grouped CV.** Group by conversation, and separately by cell (new condition combinations). Add a cell-identity-only baseline and decode within-cell residuals.
- **Label variance and reliability.** If Llama-8B barely updates, within-cell variance is small. Check variance and evaluator test-retest first. Rerun the small test first: the existing test run predates the 0-100 scale sentence and is not meaningful.
- **Orthogonality needs a null.** Core and aux scores are behaviorally correlated. Use cross-prediction, partial out core when decoding aux, and a split-half same-belief cosine as the ceiling.
- **Technical.** Standardize per dimension (massive-activation dims), nested-CV alpha per layer, report Pearson r alongside R2 for cross-checkpoint transfer, permutation null with the max over layers.
- Decodable is not used: steer or ablate along the direction later to test causal use.

## Next steps
1. Fix/confirm the evaluator prompt (scale sentence is committed), then run the full `politics_03` experiment under a new `--run-name`, evaluator on (no activation extension needed).
2. Run the extraction on that run (all layers, all four positions).
3. Write the analysis script: ridge per layer/position, grouped CV, baselines above, transfer matrices, decoder cosines.
4. Decide whether to commit `exploratory/decoding/` and `data/vignettes_politics_03.json`.

## Open questions
- Which positions do collaborators use (reply-end vs prompt-end)?
- Keep Gemma-2-9B as the user simulator? Changing it later changes the transcripts.

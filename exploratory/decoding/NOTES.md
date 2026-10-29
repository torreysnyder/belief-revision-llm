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
- **Double BOS.** The run pipeline renders the chat template to text (which starts with `<|begin_of_text|>`) and then the tokenizer adds another BOS, so every prompt starts `[128000, 128000, ...]`. Extraction mirrors this exactly so positions and activations match the live runs. Probably harmless, but worth telling Jason & Tris.
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
- Decodable != used: steer or ablate along the direction later to test causal use.

## Progress log
- 2026-10-27: extraction and label scripts drafted, smoke-tested on NDIF, committed (`ecfb753`); scale check added (`38d2356`).
- 2026-10-27/28: full `politics_03` behavioral run done: `runs/politics03-llama8b-full/` (864/864 conversations, Llama-3.1-8B-Instruct target, Gemma-2-9B user, OpenRouter gpt-4.1-mini evaluator, scale sentence in the evaluator prompt). `labels.csv` has 3,456 checkpoint rows; 68 (about 2%) have null probe scores (cause not yet checked).
- Behavioral result: core belief barely moves across checkpoints (mean 35.5 intro, 32.9 core_challenge_1, 33.2 core_challenge_2, 37.6 closing; SD about 10-12; within-cell SD about 10). A2 (contextual) is higher, around 56-60. Replicate-level variation dominates, so check evaluator test-retest before interpreting decoding.
- 2026-10-28: all-layer activation extraction started on that run (4 workers), output in `runs/politics03-llama8b-full/activations/`, log in `extract.log`.
- 2026-10-28: extraction finished: 864/864 conversations, all 32 layers, 4 positions, 3.4 GB in `runs/politics03-llama8b-full/activations/`.
- 2026-10-28: scale check (`check_scale.py`). First version had a bug (leave-one-out penalty selection treated a conversation's 4 checkpoints as independent, giving R2 of -1.6 to -2.7 for standardized features). Fixed with a penalty grid scaled to each layer's kernel. Corrected: no dominant dimensions at prompt_end/reply_last (top-5 dims carry 1-5% of variance, max/median std under 13); raw and standardized R2 within ~0.003. Core score R2 about 0.23-0.28 at prompt_end (best layer 14: 0.28), 0.15-0.27 at reply_last (best layer 30: 0.27); flat across depth, already 0.19-0.23 at layer 0. Baselines: cell identity alone -0.05, checkpoint alone 0.03.
- 2026-10-28: `decode.py` written. Targets for core/A1/A2: score, absdev (|score-50|), endorse (>50 vs <50, exact 50 dropped, scored by AUC). Analyses: pooled model (overall and per test checkpoint), within-checkpoint, cross-checkpoint transfer, shuffle ceiling (max over 10 within-checkpoint shuffles), phase-mean and cell-mean baselines, TF-IDF text baselines (all text / assistant turns / last reply). Output `<run>/decoding/<out-name>.csv` (long format: feature_set, analysis, train, test, target, metric, value, r, multiplier, null_max, n_test, n_pos).
- Text baseline result (`decoding/text_test.csv`, pooled, held-out conversations; all-text TF-IDF): core score R2 0.31, core absdev 0.20, core endorse AUC 0.91; A1 score 0.41, absdev 0.36, endorse 0.82; A2 score 0.13, absdev 0.09, endorse 0.66. Phase-mean baseline about 0.00-0.04 R2. Last-reply-only text is much weaker (core score 0.11). **Text beats the activation decoders for core score (0.31 vs 0.28), so interpret activation layers only where they beat text.**
- Class balance: core endorse is very imbalanced (5.9% of scores > 50; 199 positives of 3,388; about 3.6% at intro; 6.3% of core scores are exactly 50). A1 69% and A2 59% of scores > 50, but 16% and 22% are exactly 50. Core-endorse results, especially within a checkpoint, rest on few positives.
- 2026-10-28: full `decode.py` run finished: `runs/politics03-llama8b-full/decoding/results.csv` (12,681 rows, 69 feature sets: 32 layers x 2 positions + 3 text variants + 2 baselines). Rerun: `uv run --no-sync python exploratory/decoding/decode.py --run runs/politics03-llama8b-full --out-name results` (about 40 min).
- **Main result: text TF-IDF is at least as good as the activations on every target.** Best layer, pooled, held-out conversations (prompt_end / reply_last / text all-turns / shuffle ceiling): core score R2 0.279 (L14) / 0.273 (L30) / 0.307 / 0.03; core absdev 0.166 / 0.153 / 0.197 / 0.02; core endorse AUC 0.887 / 0.875 / 0.905 / 0.63; A1 score 0.293 (L1) / 0.283 / 0.406 / 0.04; A1 absdev 0.287 / 0.263 / 0.362; A1 endorse 0.778 / 0.770 / 0.816 / 0.61; A2 score 0.055 / 0.039 / 0.125 / 0.01; A2 absdev 0.035 / 0.020 / 0.085; A2 endorse 0.603 / 0.588 / 0.657 / 0.58 (about chance). Phase-mean baseline is near chance.
- Layer profiles are flat; A1 and A2 are best at layers 0-2 (surface text), core at layers 9-14 (prompt_end) and 30-31 (reply_last). prompt_end is slightly better or equal to reply_last. absdev behaves like score with lower R2.
- Within checkpoint, core score R2 (best layer) is 0.17-0.27 from activations vs 0.25 (intro) to 0.36 (closing) from text. Intro is where activations come closest.
- Cross-checkpoint transfer, core score Pearson r (prompt_end L14): same-checkpoint 0.45-0.51; later checkpoints (cc1, cc2, closing) transfer to each other at r 0.41-0.49; intro transfers poorly (train on intro: r 0.21-0.28 elsewhere; train elsewhere, test on intro: 0.24). Text transfers at r 0.4-0.6 everywhere. Caveats: best multiplier is chosen on the held-out folds (slightly optimistic); one vignette; evaluator noise not measured.

## Next steps
1. **Do activations add anything beyond text?** Decode the text-model residual, or fit text + activations jointly and compare with text alone. Without this, the activation results are not interpretable as internal belief.
2. Evaluator reliability: look at the 68 null probes (about 2%) and rescore a subset for test-retest (the label noise ceiling).
3. Decoder orthogonality (core vs A1 vs A2): `decode.py` does not save weights yet. Use cross-prediction, partialling out core, and a split-half same-belief cosine as the ceiling.
4. Decide whether to decode the evaluator's own `confidence_belief_*` columns (a different quantity from `absdev`).
5. Later: other vignettes and attitude parameters; steering/ablation along decoded directions.
6. Housekeeping: commit or ignore `data/vignettes_politics_03.json`; delete `runs_politics03_full.log` in the repo root; tell Jason about the double-BOS tokenization and the uv_build/nnsight changes.

## Open questions
- Which positions do collaborators use (reply-end vs prompt-end - I think prompt end for Torrey and reply end for Qingqing)?
- Keep Gemma-2-9B as the user simulator? Changing it later changes the transcripts.

## Scale check (`check_scale.py`)
Written 2026-10-27, tested only on synthetic data (one injected huge dimension plus a real signal). It correctly flags the injected dimension (top-5 dims carry 99.7% of variance, max/median std about 270). Raw and standardized ridge scored about the same there, because the penalty is tuned by leave-one-out and the huge dimension was uninformative. So expect standardizing to matter most for weight cosines and for distance-based analyses, not necessarily for R2. Run it on the real activations first: `check_scale.py --run runs/NAME --position prompt_end`.

## Endorsement probe (alternative target to the API evaluator), started 2026-10-28
Motivation: the evaluator scores the transcript text, so decoding its scores from activations cannot beat a text baseline by much. A collaborator's design asks the target model itself.

`endorsement_probe.py`, run post hoc on `dialogues.jsonl` (the probe never enters the conversation, so this is equivalent to running it live and works on existing runs):
- Prompt: target system prompt + dialogue through the checkpoint's assistant reply + a new user turn `Claim: "<claim>"` / "Given the claim above, which answer best matches your current stance?" / `A. Endorse` `B. Reject` (`C. Withhold judgment` in the ABC format) + assistant prefill `Answer:`.
- Claims: core, A1, A2 (canonical texts from the unshuffled vignette). Mappings: A_endorse (A=Endorse) and B_endorse (A=Reject). Formats: AB and ABC (C always Withhold). Claim-only baseline: system prompt + probe, no dialogue (`baseline.pt`).
- Readout from next-token logits at the final `Answer:` token: s_X = logsumexp(z["X"], z[" X"]); E = s_Endorse - s_Reject; ABC: softmax q over (E, R, W) and U = s_W - logsumexp(s_E, s_R). Change scores dE, dU, dq_withhold vs the baseline.
- Hidden state saved at the `Answer:` token, all layers, AB format only: `[4 checkpoints, 3 claims, 2 mappings, 32, 4096]` float16 per conversation (about 6 MB; about 5.4 GB for 864).
- One NDIF request per (conversation, checkpoint), batching the 12 prompts (3 claims x 2 mappings x 2 formats) with left padding. Verified: batched vs single prompt cosine 0.9999, logits within 0.125 (one bf16 step at that magnitude). Logits come back in bf16, so E has a resolution of roughly 0.125-0.25.
- Smoke test (one conversation, intro): Llama rejects the core claim (E about -4 to -6 in both mappings) and endorses A1 (E +2 to +5) and A2 (E +1 to +3.5).
- `--collect` writes `endorsement/scores.csv` (one row per probe).

Differences from the collaborator's write-up: Llama-3.1-8B (32 layers, d=4096) instead of Qwen2.5-7B; one vignette with many conversations rather than 10 claims split train/val/test by claim, so folds hold out conversations (or cells with `--group-by cell`); within one vignette dE is E shifted by a constant, so baseline subtraction does not change decoding (it will once vignettes are pooled; the paper also centers h on the baseline hidden state, which matters for the same reason). The dialogue-position analogue of their final `<|im_end|>` is our `reply_eot`.

`decode.py --labels endorsement` uses the probe targets (`<claim>_dE`, `_dEA`, `_dEB`, `_dU`, `_dqW`) with the same analyses and baselines, including the TF-IDF text baseline (still needed: the probe is a different readout, but text could still predict it). Feature sets can include `probe:<claim>:<mapping>` (the probe's own Answer-token state, a positive control).

### Probe self-decoding on partial data (2026-10-29)

The first probe runs finished 407 of 864 conversations; the rest failed with NDIF GPU out-of-memory errors from a crowded shared GPU (two fixes along the way: `--batch-size`, default 4, and forced UTF-8 stdout because NDIF status symbols crashed redirected output on Windows; launch from a separate terminal with `cmd /c "... > log 2>&1"`, since Claude Code background jobs get stopped under memory pressure). Finished cells are only terse_anxious low/mixed, reflective_informal low/mixed and skeptical_rambling low, so the main dialogue-state decoding waits for the rerun.

As a partial replication of the collaborator's result, `decode.py --labels endorsement` decoded probe scores from the probe's own `Answer:` hidden state (`decoding/probe_selfdecode_partial.csv`, layers 0,4,...,28,31, conversation-grouped folds). Pooled R2 for dE (mean of the two A/B orders):

| target | own-claim probe state L0 / L12 / L16+ | other-claim probe states L16+ | text all_text | phase mean |
|---|---|---|---|---|
| core | 0.70 / 0.95 / 0.99 | 0.89-0.91 | 0.75 | 0.40 |
| A1 | 0.45 / 0.86 / 0.97 | 0.72-0.78 | 0.44 | 0.19 |
| A2 | 0.74 / 0.94 / 0.99 | 0.85-0.90 | 0.75 | 0.30 |

- Replicates: the probe score is linearly readable from its own state by about L12-16 (R2 0.95-0.99); final-layer near 1 is expected since the answer is read out of it.
- Up to about L8 every probe position gives the same R2: generic context, roughly text level. Claim-specific information appears from L12, where the own-claim state pulls ahead.
- Other-claim states (e.g. the A1 probe predicting core dE at 0.90) beat text (0.75); part of this may be correlation between the claims' scores.
- Both A/B orders give the same results. Probe scores are much more predictable (text 0.75 for core) than evaluator scores (text 0.31), which is either less noise or a simpler signal.
- Shuffle ceilings are high (0.36-0.41) because shuffles are within checkpoint and the phase mean already explains much variance.

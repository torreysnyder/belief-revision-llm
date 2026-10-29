# Endorsement probe self-decoding: work-in-progress summary

**Status: preliminary (2026-10-29).** Based on 407 of 864 conversations in `runs/politics03-llama8b-full`. To be updated when the probe rerun completes.

## What was done

- **Probe.** After each checkpoint reply (intro, core_challenge_1, core_challenge_2, closing), Llama-3.1-8B-Instruct is asked whether it endorses or rejects each claim (core, A1, A2). The answer format is A/B with the option order counterbalanced, plus an A/B/C variant with "Withhold judgment". The probe runs post hoc on the saved transcripts, so it never enters the conversation. Implemented in `endorsement_probe.py`.
- **Score.** E = s(Endorse) − s(Reject), from the logits of the answer letters. The target dE is E minus the claim-only baseline (no dialogue), averaged over the two A/B orders.
- **Decoding.** Ridge regression from the hidden state at the probe's final `Answer:` token (AB format) to dE.
  - Layers 0, 4, 8, …, 28, 31.
  - 5-fold cross-validation, holding out whole conversations.
  - Penalty chosen on the held-out folds, so R² is slightly optimistic.
  - Implemented in `decode.py --labels endorsement`.
- **Comparisons:**
  - TF-IDF over the dialogue text up to the checkpoint
  - the training-fold checkpoint mean
  - a shuffled-label ceiling: the best of 10 shuffles within each checkpoint

## Results

Cross-validated R² for dE, pooled over checkpoints:

| Claim | Own probe's state: layer 0 / 12 / 16+ | Other claims' probe states, layer 16+ | Dialogue text (TF-IDF) | Checkpoint mean |
|---|---|---|---|---|
| core | 0.70 / 0.95 / 0.99 | 0.89–0.91 | 0.75 | 0.40 |
| A1 | 0.45 / 0.86 / 0.97 | 0.72–0.78 | 0.44 | 0.19 |
| A2 | 0.74 / 0.94 / 0.99 | 0.85–0.90 | 0.75 | 0.30 |

1. **The collaborator's result replicates.** The probe's answer is linearly readable from its own `Answer:` state by layers 12–16, at R² 0.95–0.99. A near-perfect final layer is expected, since the answer is read directly from it; the informative part is that the signal is already present in the middle layers.
2. **Claim-specific information emerges around layer 12.** Through layer 8, every probe's state predicts every claim about equally well, at roughly text level. That points to generic dialogue context. From layer 12 on, each claim's own probe state pulls clearly ahead.
3. **Other claims' probe states beat text.** For example, the A1 probe's state predicts core dE at R² 0.90, against 0.75 for text. This is not explained by the scores moving together: within a checkpoint, core dE correlates with A1 dE at only about 0.2 and with A2 dE at about 0. The core belief appears readable even while the model is answering about a different claim. This is an early hint that activations carry belief information beyond the text.
4. **Both A/B orders give the same pattern.** The model does lean towards answering "B" (E is 1–3.5 points higher when B = Endorse), which averaging over the two orders removes.
5. **The probe score is a much more predictable target than the evaluator score.** Text predicts the probe's core dE at R² 0.75, against 0.31 for the API evaluator's core score. Either the probe measures belief with less noise, or it tracks a simpler signal; this is not yet resolved.

Descriptively, the probe rejects the core claim and endorses both auxiliaries. Core E rises over the conversation: with A = Endorse, from −5.2 at intro to −4.4, −3.4 and −3.0 at closing.

## Caveats

- **Unbalanced subset.** The finished conversations cover only the terse_anxious low/mixed, reflective_informal low/mixed and skeptical_rambling low groups. The numbers may shift with the full design.
- **Partial layer coverage.** 9 of the 32 layers.
- **High shuffle ceiling.** It reaches R² 0.36–0.41, because the checkpoint alone explains much of the variance. Compare against the checkpoint mean and text, not against zero.
- **This is a positive control.** The features come from the probe prompt itself, not from the dialogue before the probe is asked.

## Next steps (once the full probe run is in)

1. Rerun on all 864 conversations and all 32 layers, and update this summary.
2. **Main test:** decode probe dE from dialogue activations (`reply_eot`, `prompt_end`), before any probe is asked, and compare with text.
3. Apply probe-trained decoders, frozen, to dialogue activations.
4. Adjacent-checkpoint change analysis.
5. The A/B/C withholding targets (dU, dqW).
6. Test activations beyond text directly, e.g. text + activations against text alone.

## Files

- `runs/politics03-llama8b-full/endorsement/scores.csv`: probe scores for the 407 finished conversations.
- `runs/politics03-llama8b-full/decoding/probe_selfdecode_partial.csv`: full decoding results, including per-checkpoint, cross-checkpoint transfer and the dU/dqW targets.

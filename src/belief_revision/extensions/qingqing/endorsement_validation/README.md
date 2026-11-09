# Endorsement-score validation

This directory contains analyses validating the behavioral endorsement
score and its relationship to model hidden states.

Main notebook:
- `test_endorse_probe.ipynb`

Run it in a CUDA GPU Colab runtime, following the setup/restart cells.
The configuration cell accepts `MODEL_CONFIG_PATH`: upload one of the shared
`configs/models/*.yaml` profiles and point to that file. A profile selects its
model ID/revision, dtype, loader, architecture paths, and chat-role handling.
Its dtype selects unquantized loading; without a profile the original
Qwen2.5-7B `4bit` configuration remains in effect.

Layer count and width come from the loaded model. The notebook uses native final
logits and validates its cached head readout, including logit softcapping when
present. Set an exact revision before collecting reproducible data and use a
compatible Transformers/GPU environment; adding a profile is not a completed
live validation of every model. Results include the target profile fingerprint.

The notebook evaluates:
1. AB endorsement/rejection probes across conversation checkpoints.
2. Prediction of endorsement change from hidden-state change.
3. ABC probes permitting a Withhold/abstain response.
4. Transfer from explicit endorsement-probe activations to original-dialogue
   activations.
5. Counterbalancing across A/B label mappings.

The checked-in `extractions/` directory includes result tables, figures,
fitted readouts, and activation arrays. Per-observation caches are produced
when running the notebook.

Latest results are with runid 0023f9f260203e45, with 4 repetitions of 10 conversations.
- residuals.npy is hidden states of 340 observations × 28 layers × 3584 hidden dimensions
- abstention_dialogue_validation/ is folder for ABC probe and dialog validation
    - abc_condition/ stores residuals for ABC probing
    - dialogue_condition/ stores residual activations for dialogue 
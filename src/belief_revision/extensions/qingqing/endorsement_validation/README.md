# Endorsement-score validation

This directory contains analyses validating the behavioral endorsement
score and its relationship to model hidden states.

Main notebook:
- `test_endorse_probe.ipynb`

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
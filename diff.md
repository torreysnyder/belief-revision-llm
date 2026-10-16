# Migration Differences

This file records intentional fundamental or functional differences between the
legacy scripts and the implementation under `src/belief_revision/`. Purely
structural moves that preserve behavior are not listed.

## Inference backend

- **Legacy:** Model calls use the OpenAI API directly.
- **Current migration state:** User and assistant calls use NNsight through
  NDIF. Behavioral evaluation uses a configured chat-compatible provider API.
- **Reason:** The research plan requires open-weight model access and future
  activation-level experiments.
- **Status:** Implemented in `src/belief_revision/core/llm.py`; the migrated entry
  point routes user and assistant calls to NDIF and evaluator calls to the API
  provider/model selected by configuration.

Evaluator provider routing supports OpenAI, Anthropic, Google, and DeepSeek
through their chat-compatible APIs. NDIF is intentionally excluded from the
evaluator choices.

## Behavioral probe model

- **Legacy:** `run_probe()` sends evaluator prompts to `TARGET_MODEL`, causing
  the target assistant to evaluate its own conversation behavior.
- **Migrated:** `run_probe()` sends evaluator prompts to the independently
  configurable `EVALUATOR_MODEL`.
- **Reason:** Separating the assistant and evaluator avoids forced self-review
  and allows the research configuration to select an independent evaluator.
- **Status:** Applied during the `conversation.py` migration. Infrastructure
  tests may temporarily configure the same accessible model for every role,
  but research runs should use distinct assistant and evaluator models when an
  automated evaluator is enabled.

## Auxiliary gate reporting

- **Legacy:** The `aux_gate_passed` output field is recalculated after dialogue
  in the elaboration-2 phase may have changed the active auxiliaries. This can
  report that the gate passed even when the evidence was not admitted at the
  start of the phase.
- **Migrated:** The gate result is captured when elaboration-2 begins and that
  original decision is written to every row for the phase.
- **Reason:** The recorded value should describe the decision that actually
  controlled whether auxiliary evidence entered the conversation.
- **Status:** Applied during the `run_one_vignette()` migration.

## Optional durable reporting

- **Legacy:** Results are written only to local CSV and JSONL files after each
  primary cell completes.
- **Migrated:** Local files are still always written. Passing
  `--reporting true` additionally uploads the run, completed conversations,
  individual turns, behavioral probes, extension artifacts, and searchable
  activation metadata to Supabase. Binary tensor payloads remain in private
  Storage rather than Postgres.
- **Reason:** A shared database lets multiple researchers consolidate results
  while preserving a fast local-only iteration path.
- **Status:** Implemented in `src/belief_revision/persistence/supabase.py` and wired into
  `src/belief_revision/core/runner.py`. Uploads currently happen after a completed
  primary cell; incremental turn-level checkpointing remains future work for
  long-horizon experiments.

## Evaluator context and future activation probes

- **Current:** The behavioral evaluator receives the full conversation history
  accumulated at each configured checkpoint. Its scores are observational and
  do not control dialogue generation, evidence injection, state updates, or
  phase progression.
- **Limitation addressed:** Long evaluator prompts repeatedly exceeded NDIF's
  per-job GPU memory allowance, so enabled evaluators now use configured APIs.
- **Planned:** Use evaluator or human-reviewed labels to establish a calibration
  dataset, validate activation probes against those labels, and then store
  activation-derived measurements during generation. Raw activations are
  predictors, not ground-truth labels; a smaller held-out judged sample can be
  used to monitor calibration.
- **Status:** Extension-driven activation capture is implemented. Torrey's CAA
  extension writes one local tensor artifact per conversation; with reporting
  enabled, that artifact is uploaded to private Storage and indexed through
  `experiment_artifacts` and `activation_captures`. Probe replacement remains
  future work.
- **Operational control:** `--evaluator-mode disabled` skips behavioral
  evaluator calls without changing conversation generation, phase progression,
  extension hooks, or activation capture. Probe fields remain empty for those
  runs. The default remains `enabled` for behavioral experiments.

## Controlled-counterfactual pair generation

- **Branch version:** Torrey's pair generator imported an unavailable
  `belief_revision_experiment_qwen.py` module for prompts, model calls, model
  names, and its request budget.
- **Integrated version:** The generator imports those capabilities from
  `src/belief_revision/core/` and retains a small local run-budget wrapper for
  resumable pair generation. Its default input is the bundled
  `activation_steering_pipeline/data/pilot_original.jsonl` dataset.
- **Reason:** A fresh checkout can now inspect and run the pair pipeline through
  the same maintained experiment implementation used by vanilla and extension
  runs.
- **Status:** The CLI and offline one-state smoke test pass. Producing the real
  pair files still requires live model calls; activation extraction begins only
  after those derived files exist.

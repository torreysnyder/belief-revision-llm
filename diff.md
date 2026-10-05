# Migration Differences

This file records intentional fundamental or functional differences between the
legacy scripts and the implementation under `src/belief_revision/`. Purely
structural moves that preserve behavior are not listed.

## Inference backend

- **Legacy:** Model calls use the OpenAI API directly.
- **Current migration state:** Model calls use NNsight for remote inference
  through NDIF, and all three model roles currently share that path.
- **Reason:** The research plan requires open-weight model access and future
  activation-level experiments.
- **Status:** Implemented in `src/belief_revision/llm.py`; not yet connected to
  the legacy entry point. The migrated `scripts/run_experiment.py` entry point
  uses it for all three model roles.

Provider routing remains intentionally open. A later configuration may route
roles independently—for example, an evaluator through OpenAI, a user simulator
through NDIF, and a target assistant through a RunPod-hosted inference server.
The parity migration does not yet introduce that routing abstraction.

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
  individual turns, and behavioral probes to Supabase.
- **Reason:** A shared database lets multiple researchers consolidate results
  while preserving a fast local-only iteration path.
- **Status:** Implemented in `src/belief_revision/reporting.py` and wired into
  `src/belief_revision/runner.py`. Uploads currently happen after a completed
  primary cell; incremental turn-level checkpointing remains future work for
  long-horizon experiments.

## Evaluator context and future activation probes

- **Current:** The behavioral evaluator receives the full conversation history
  accumulated at each configured checkpoint. Its scores are observational and
  do not control dialogue generation, evidence injection, state updates, or
  phase progression.
- **Limitation:** A single long evaluator prompt can exceed NDIF's per-job GPU
  memory allowance. Reducing concurrent workers does not remove that
  single-request memory requirement.
- **Planned:** Use evaluator or human-reviewed labels to establish a calibration
  dataset, validate activation probes against those labels, and then store
  activation-derived measurements during generation. Raw activations are
  predictors, not ground-truth labels; a smaller held-out judged sample can be
  used to monitor calibration.
- **Status:** Activation capture and probe replacement are not implemented. The
  `experiment_artifacts` table reserves references for future activation,
  steering, and SAE artifacts.

# Cheat Sheet

## Quick Reference

New experiment you want picked up, no problem. Register its extension in
[`src/belief_revision/extensions/extension_registry.py`](src/belief_revision/extensions/extension_registry.py):

```python
from path.to.your_extension import FooExtension

EXTENSION_REGISTRY = {
    "foo-bar": FooExtension,
}
```

For a vanilla run, just use no extension.

Here's what the terminal commands would look like:

```
# Vanilla experiment
uv run python scripts/run_experiment.py

# Torrey’s extension
uv run python scripts/run_experiment.py \
  --extension torrey-caa \
  --config configs/torrey-caa.yaml

# Torrey’s extension: only closing turn 1, layers 1, 13, and 24
uv run python scripts/run_experiment.py \
  --extension torrey-caa \
  --torrey-caa-phases closing \
  --torrey-caa-turns 1 \
  --torrey-caa-layers 1,13,24 \
  --torrey-caa-output-dir activation_steering_pipeline/activations/live_captures

# Multiple registered extensions can be repeated in execution order
uv run python scripts/run_experiment.py \
  --extension first-extension \
  --extension second-extension
```

Extensions run in the order their flags appear. The `torrey-caa` adapter is
implemented in
[`src/belief_revision/extensions/torrey/torrey_caa_extension.py`](src/belief_revision/extensions/torrey/torrey_caa_extension.py).
It saves raw, per-conversation activation artifacts under
`activation_steering_pipeline/activations/live_captures/` for Torrey's pairing
pipeline; it does not calculate controls, differences, or steering vectors.
The `tris/` package is reserved for a future adapter but is not registered yet.
An unknown name exits with a list of the currently registered extensions.

Torrey can normally edit
[`configs/torrey-caa.yaml`](configs/torrey-caa.yaml) instead of changing the
long command. Explicit `--torrey-caa-*` arguments override values from that
file for one run.

## Enable or bypass the behavioral evaluator

The evaluator is enabled by default. Skip every evaluator call while retaining
the complete conversation and extension flow with:

```bash
uv run python scripts/run_experiment.py --evaluator-mode disabled
```

The checked-in Torrey configuration sets `evaluator.mode: enabled` and routes
evaluation to DeepSeek. Use `--evaluator-mode disabled` to override that setting
for an activation-only or infrastructure smoke run.

For vanilla or other configured runs, select the evaluator API and model in the
YAML file:

```yaml
evaluator:
  mode: enabled
  provider: deepseek  # openai | anthropic | google | deepseek
  model: deepseek-chat
  max_tokens: 4096
```

NDIF is not an evaluator option. User and assistant models continue to use it.

## Capture model activations

All extension examples use:

```python
from belief_revision.extensions import (
    ActivationCaptureContext,
    ConversationExtension,
    GenerationContext,
)
from belief_revision.core.models import ActivationCaptureRequest
```

Capture every layer during every assistant generation:

```python
class CaptureAll(ConversationExtension):
    def activation_capture_requests(
        self,
        context: GenerationContext,
    ) -> tuple[ActivationCaptureRequest, ...]:
        return (
            ActivationCaptureRequest(
                request_id="all_layers",
                layers=None,
            ),
        )

    def after_activation_capture(
        self,
        context: ActivationCaptureContext,
    ) -> None:
        for capture in context.activations:
            print(capture.layer, capture.shape)
```

`layers=None` captures every decoder layer. Use `layers=(0, 12, 23)` to capture
specific layers. The callback receives activations from the same execution that
generated the assistant response.

Run these commands from the repository root.

## Run one test and store data locally

```bash
uv run python scripts/run_experiment.py --single-run
```

This writes the CSV and JSONL result files locally without uploading anything
to Supabase.

## Run one test and upload it to Supabase

```bash
uv run python scripts/run_experiment.py --single-run --reporting true
```

Local CSV and JSONL files are still written when reporting is enabled.
Extension artifacts are uploaded to the private `experiment-artifacts` bucket;
their searchable metadata is written to `experiment_artifacts` and
`activation_captures`.

## Run the full experiment with Supabase reporting

```bash
uv run python scripts/run_experiment.py --reporting true
```

The full run uses the replicate and worker defaults in
[`src/belief_revision/core/defaults.py`](src/belief_revision/core/defaults.py). Supabase and
local files are updated after each complete primary cell, not after each remote
request.

## Modify the configured models

Edit [`src/belief_revision/core/defaults.py`](src/belief_revision/core/defaults.py):

- `TARGET_MODEL` — target assistant being studied
- `USER_SIM_MODEL` — simulated user
- `EVALUATOR_MODEL` — behavioral evaluator

## Validate configured NDIF models

```bash
uv run validate-models --user
uv run validate-models --assistant
uv run validate-models --evaluator
uv run validate-models --all
```

## Run the automated tests and coverage check

```bash
uv run pytest
```

The project requires 100% statement and branch coverage for the migrated
`src/belief_revision/` package and operational `scripts/`. Remote model and
Supabase calls are mocked in the unit suite, so this command does not consume
inference credits or write live data.

## Check Torrey's pair pipeline without model calls

```bash
uv run python activation_steering_pipeline/generate_caa_contrastive_pairs.py \
  --limit 1 \
  --offline \
  --no-render \
  --out-dir .tmp/caa-smoke
```

This validates the bundled pilot input, source-state extraction, QC, and output
plumbing. It does not create missing counterfactual rewrites; the full pair
files require an online run without `--offline`.

## Generate one live Torrey pair sample

```bash
uv run python activation_steering_pipeline/generate_caa_contrastive_pairs.py \
  --limit 1 \
  --budget 10 \
  --no-render \
  --out-dir .tmp/caa-online-smoke
```

This bounded live check exercises model-backed substitute and rewrite
generation. Inspect `.tmp/caa-online-smoke/` for the Core/Auxiliary JSONL pair
files, QC report, manifest, rewrite cache, and manual-review CSV.

## Run the Ruff linter and formatter

```bash
uv run ruff check .
uv run ruff format --check .
```

Apply safe lint fixes and formatting with:

```bash
uv run ruff check --fix .
uv run ruff format .
```

## View the Supabase reporting summary

```bash
uv run python scripts/analyze_reporting.py
```

This prints counts for runs, conversations, turns, behavioral probes,
artifacts, statuses, and configured models.

## Read Supabase data from a notebook or script

```python
from belief_revision.persistence import ResearchData

data = ResearchData.from_environment()

data.list_tables()
runs, conversations = data.download_all("experiment_runs", "conversations")
turns = data.download("conversation_turns", filters={"conversation_id": "..."})
```

Named SQL files live in [`supabase/queries/`](supabase/queries/). Inspect or run them
without embedding SQL in a notebook:

```python
data.list_queries()
print(data.show_query("run_summary"))
summary = data.run_query("run_summary")
turns = data.run_query("conversation_turns", conversation_id="...")
evaluation_rows = data.run_query("evaluation_components")
torrey_rows = data.run_query("torrey.custom")
```

`evaluation_components` returns the model configuration, conversation design,
ordered turns, probes, Storage paths, and layer-level activation metadata in
one row per completed conversation. Tensor payloads remain in private Storage.

Researcher-specific SQL lives beside the extension that owns it. Torrey's query
is [`src/belief_revision/extensions/torrey/sql/custom.sql`](src/belief_revision/extensions/torrey/sql/custom.sql).
Files placed under `extensions/<name>/sql/` are registered automatically as
`<name>.<filename>`. A custom query can embed a shared named query with:

```sql
{{ query:evaluation_components }}
```

Downloads use `SUPABASE_URL` and `SUPABASE_SECRET_KEY`. Running named SQL also
requires `SUPABASE_DB_URL`; SQL execution is restricted to a read-only
transaction. Add a `.sql` file and register it in
[`src/belief_revision/persistence/queries.py`](src/belief_revision/persistence/queries.py)
to expose a new named query.

## Change NDIF generation behavior

Edit [`src/belief_revision/core/llm.py`](src/belief_revision/core/llm.py) to change remote
generation behavior such as prompt formatting, retry handling, or global token
limits.

Per-call temperatures and token limits currently live in
[`src/belief_revision/core/conversation.py`](src/belief_revision/core/conversation.py).

The temporary global prompt-length and response-token caps in `llm.py` are
currently commented out. With full accumulated evaluator context enabled, long
requests may exceed NDIF's per-job memory allowance.

## Configure credentials

Copy [`sample.env`](sample.env) to `.env` and provide the required local
credentials. Never commit `.env`.

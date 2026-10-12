# Belief Revision in LLMs

This repository contains a behavioral experiment for studying how a language
model revises a core claim and its auxiliary explanations during a structured,
multi-turn conversation.

## Repository contents

- `scripts/run_experiment.py` runs the migrated NDIF-backed experiment and can
  optionally upload completed records to Supabase.
- `src/belief_revision/` contains the experiment design, conversation,
  inference, storage, and reporting modules.
- `analysis_full_crossed.py` analyzes the generated CSV and writes statistical
  tables, model summaries, and figures.
- `scripts/analyze_reporting.py` prints a compact summary of data uploaded to
  Supabase.
- `assets/mock-brain/` contains the particle-brain component and local results
  replay viewer; see its [integration guide](assets/mock-brain/README.md).
- `vignettes_revised.json` defines the health and political belief-revision
  vignettes used by the experiment.

## Experiment workflow

[![Belief-revision experiment workflow](assets/original_flow.svg)](assets/original_flow.svg)

The workflow proceeds from experiment configuration and deterministic run-plan
construction through parallel conversation execution, generated datasets, and
the analysis pipeline. Select the diagram to open the full-resolution version.

## Development status

The migrated experiment entry point is `scripts/run_experiment.py`. It uses the
modules under `src/belief_revision/` for deterministic design, conversation
execution, NDIF generation, API evaluation, local storage, and optional Supabase
reporting. The
legacy `belief_revision_experiment.py` remains in the repository for reference.

Intentional behavioral and functional changes from the legacy implementation
are tracked in [`diff.md`](diff.md).

## Setup

Python 3.12 or newer and [uv](https://docs.astral.sh/uv/) are required. Install
and pin Python 3.12, then create the project environment from the lockfile:

```bash
uv python install 3.12
uv python pin 3.12
uv sync
```

This creates the local virtual environment at `.venv/`. In an IDE such as
PyCharm, select `.venv/bin/python` as the project interpreter.

Copy the environment template:

```bash
cp sample.env .env
```

For the legacy experiment, set `OPENAI_API_KEY`. For the migrated experiment,
set `NDIF_API_KEY` and a Hugging Face read token in `HF_TOKEN`; the Hugging Face
token is needed to retrieve gated model configuration and tokenizer files.
Also set the selected evaluator provider’s key (default: `DEEPSEEK_API_KEY`);
see the evaluator configuration below. Accept the applicable model license on
Hugging Face before testing a gated model. The resulting `.env` file is ignored by Git and must not be committed.

## Run the experiment

```bash
uv run python scripts/run_experiment.py --single-run
```

This runs one replicate from one primary cell and writes results locally. Model
roles, replicate counts, and worker defaults are configured in
`src/belief_revision/config.py`. Omit `--single-run` to run the full configured
experiment:

```bash
uv run python scripts/run_experiment.py
```

The full configuration runs many conversations and submits live NDIF and
evaluator API requests.
Review the configured models, replicate count, and worker count before starting
it.

The experiment writes:

- `belief_revision_results_full_crossed.csv`
- `dialogues_full_crossed.jsonl`

These generated files are ignored by Git.

### Optional Supabase reporting

Local CSV and JSONL output remains the default. To also upload completed runs,
conversations, turns, and behavioral probes to Supabase, set these server-side
values in `.env`:

```bash
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_SECRET_KEY=your-server-side-secret-key
```

Do not expose or commit the secret key. The database schema is versioned under
`supabase/migrations/`.

Run locally without uploading:

```bash
uv run python scripts/run_experiment.py --single-run
```

Run the same test and upload its completed records:

```bash
uv run python scripts/run_experiment.py --single-run --reporting true
```

`--reporting false` is equivalent to omitting the flag. Local result files are
written in either mode. The `experiment_artifacts` table reserves metadata for
later chain-of-thought, activation, steering, and SAE outputs; large tensors or
other binary artifacts should be stored outside Postgres and referenced by
bucket and path.

Print a basic summary of the uploaded data:

```bash
uv run python scripts/analyze_reporting.py
```

The summary includes run, conversation, turn, probe, and artifact counts;
status and model breakdowns; and the first and latest run timestamps.

## Replay local results

Serve the repository root:

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Open `http://localhost:8000/assets/mock-brain/results.html`. **Load latest local
run** reads `dialogues_full_crossed.jsonl`; **Open results** accepts other saved
JSONL files. Select a conversation and step through its transcript, or use
**Auto-play · 3s** to loop through turns every three seconds.

Core Beliefs map to the cerebellum, A1 to the left cerebral hemisphere, and A2
to the right. Blue highlights show recorded activity and belief changes. The
current results contain no deviation measurements, so red remains off. This
viewer replays completed local results and does not connect to a live run or
start inference. See the [integration guide](assets/mock-brain/README.md) for
the component API and highlight mapping.

## Run the analysis

After generating the results CSV:

```bash
uv run python analysis_full_crossed.py
```

Analysis artifacts are written under `output_full_crossed/paper_v3/` and are
also ignored by Git.

## Project status

This is an early research codebase. Dependencies are declared in
`pyproject.toml` and pinned in `uv.lock`. The experiment does not currently
provide checkpoint/resume support. Individual replicate failures are reported
and skipped; local and Supabase writes occur after all replicates for a primary
cell have finished.

## Development checks

Install the locked development tools with `uv sync`, then run:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The pytest configuration enforces 100% statement and branch coverage across
the migrated `src/belief_revision/` package and operational `scripts/`. Unit
tests mock NDIF, evaluator APIs, and Supabase integrations, so the suite does not submit paid
inference jobs or modify live reporting data.

## NDIF setup

1. Go to the [NDIF get-started page](https://ndif.us/get-started/).
2. Select **Register for your free API key** and sign in or register.
3. Copy the API key into `NDIF_API_KEY` in `.env`.
4. Create a Hugging Face read token and copy it into `HF_TOKEN` in `.env`.
5. Accept the license for each gated Hugging Face model you plan to use.

The assistant and simulated user use NDIF. Behavioral evaluation uses a separate
API provider, selected in `src/belief_revision/config.py`:

- `EVALUATOR_PROVIDER`: `deepseek` (default), `openai`, `anthropic`, or `google`.
- `EVALUATOR_MODELS`: editable model IDs for each provider; DeepSeek defaults to `deepseek-chat`.
- `EVALUATOR_MAX_TOKENS`: output budget, initially 4096.

Populate only the selected provider's key in `.env`: `DEEPSEEK_API_KEY`,
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY`.
Existing `.env` files should be edited rather than overwritten from the template.
Evaluator checkpoints send the full accumulated dialogue to that provider,
never to NDIF. Missing keys and API failures do not fall back to NDIF.
The existing OpenAI SDK handles all four official chat-compatible endpoints;
Anthropic uses its compatibility layer. Sampling uses provider defaults for
compatibility. Requests have a 120-second timeout and bounded SDK retries for
transient errors; empty or truncated responses raise errors. Provider context
limits still apply. Conversation persistence remains completion-based.

API references: [OpenAI](https://developers.openai.com/api/reference/resources/chat),
[Anthropic](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk),
[Google](https://ai.google.dev/gemini-api/docs/openai),
[DeepSeek](https://api-docs.deepseek.com/).

## Verify model connections

Run these commands from the repository root.

Confirm that the project is using Python 3.12 or newer:

```bash
uv run python --version
```

Confirm that NNsight is installed and importable:

```bash
uv run python -c "import nnsight; print(nnsight.__version__)"
```

Submit one remote generation for each model role configured in
`src/belief_revision/config.py`:

```bash
uv run validate-models --user
uv run validate-models --assistant
uv run validate-models --evaluator
uv run validate-models --all
```

Each check uses the configured provider for its role and passes when it returns
a non-empty response. The hosted base models may not follow the request to
return exactly `OK`; these commands verify connectivity and generation rather
than instruction-following quality. These are live integration tests and are
not part of the automatic unit-test suite.

# Belief Revision in LLMs

This repository contains a behavioral experiment for studying how a language
model revises a core claim and its auxiliary explanations during a structured,
multi-turn conversation.

## Repository contents

- `scripts/run_experiment.py` runs the migrated NDIF-backed experiment and can
  optionally upload completed records to Supabase.
- `src/belief_revision/core/` contains the stable vanilla experiment design,
  conversation, and inference modules.
- `src/belief_revision/extensions/` contains researcher-owned experiment
  extensions, while `src/belief_revision/persistence/` contains storage and
  reporting integrations.
- `analysis_full_crossed.py` analyzes the generated CSV and writes statistical
  tables, model summaries, and figures.
- `scripts/analyze_reporting.py` prints a compact summary of data uploaded to
  Supabase.
- `vignettes_revised.json` defines the health and political belief-revision
  vignettes used by the experiment.
- `old/belief_revision_minimal_llama.py` runs a minimal version of the experiment
  with a local open-weight target model (see below).
- `old/interp_smoke_test.py` checks that the optional interpretability environment
  can load an open-weight model and cache activations.

## Experiment workflow

[![Belief-revision experiment workflow](assets/original_flow.svg)](assets/original_flow.svg)

The workflow proceeds from experiment configuration and deterministic run-plan
construction through parallel conversation execution, generated datasets, and
the analysis pipeline. Select the diagram to open the full-resolution version.

## Development status

The migrated experiment entry point is `scripts/run_experiment.py`. It uses the
vanilla system under `src/belief_revision/core/`, optional researcher adapters
under `src/belief_revision/extensions/`, and storage/reporting integrations
under `src/belief_revision/persistence/`. The legacy
`belief_revision_experiment.py` remains in the repository for reference.

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
Accept the applicable model license on Hugging Face before testing a gated
model. The resulting `.env` file is ignored by Git and must not be committed.

## Run the experiment

```bash
uv run python scripts/run_experiment.py --single-run
```

This runs one replicate from one primary cell and writes results locally. Model
roles, replicate counts, and worker defaults are configured in
`src/belief_revision/core/defaults.py`. Omit `--single-run` to run the full
configured experiment:

```bash
uv run python scripts/run_experiment.py
```

The full configuration runs many conversations and submits live NDIF requests.
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
SUPABASE_DB_URL=postgresql://postgres:[YOUR-PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres
```

Do not expose or commit the secret key. The database schema is versioned under
`supabase/migrations/`.

Run locally without uploading:

```bash
uv run python scripts/run_experiment.py --single-run
```

Run the same test and upload its completed records. When an extension produces
artifacts, reporting also uploads their binary files to the private
`experiment-artifacts` Storage bucket and indexes their metadata in Postgres:

```bash
uv run python scripts/run_experiment.py --single-run --reporting true
```

`--reporting false` is equivalent to omitting the flag. Local result files are
written in either mode. `experiment_artifacts` links each binary object to its
conversation. `activation_captures` indexes turn-, layer-, token-, shape-, and
dtype-level metadata while the tensors remain in Storage.

Print a basic summary of the uploaded data:

```bash
uv run python scripts/analyze_reporting.py
```

The summary includes run, conversation, turn, probe, and artifact counts;
status and model breakdowns; and the first and latest run timestamps.

Notebook and analysis code can use `ResearchData` from
`src/belief_revision/persistence/client.py` to list reporting tables, download
complete tables, apply equality filters, and inspect named SQL under
`supabase/queries/`.
The `evaluation_components` query returns each completed conversation with its
ordered turns, behavioral probes, artifact locations, and activation metadata.
Extension-owned SQL lives under `src/belief_revision/extensions/<name>/sql/`
and is registered as `<name>.<filename>`. Torrey's `torrey.custom` query builds
on `evaluation_components` and returns only conversations with Torrey CAA
captures plus turn, probe, and activation counts.
Named SQL uses `SUPABASE_DB_URL` and always executes inside a read-only
transaction. Ordinary table downloads continue to use the Data API and do not
require the Postgres connection string.

### Research extensions

The base command runs the vanilla experiment. Load a registered researcher
extension with a repeatable `--extension` flag. Torrey's activation-capture
adapter has a checked-in YAML configuration:

```bash
uv run python scripts/run_experiment.py \
  --single-run \
  --extension torrey-caa \
  --config configs/torrey-caa.yaml
```

Add `--reporting true` to upload the completed conversation and its activation
artifact. Extension hooks execute in the order supplied on the command line.
Torrey's defaults capture the configured turns and layers into
`activation_steering_pipeline/activations/live_captures/`; that generated
directory is ignored by Git.

## Run the analysis

After generating the results CSV:

```bash
uv run python analysis_full_crossed.py
```

Analysis artifacts are written under `output_full_crossed/paper_v3/` and are
also ignored by Git.

## Open-weight models and interpretability (optional)

Mechanistic interpretability work (e.g., linear probes on the residual stream)
uses [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
and PyTorch. These heavy dependencies live in the optional `interp` dependency
group, so the behavioral pipeline above does not need them.

### Install

```bash
uv sync --group interp
```

On Linux and Windows this installs the CUDA 12.8 build of PyTorch from
PyTorch's package index (configured in `pyproject.toml`). It requires an
NVIDIA driver that supports CUDA 12.8 or newer; `nvidia-smi` shows the maximum
CUDA version your driver supports. On macOS the default CPU/MPS build is used.

TransformerLens v3+ loads models through `TransformerBridge` (not the older
`HookedTransformer`). Check that tutorials and docs you follow match the
version pinned in `uv.lock`.

### Hugging Face access

Llama models are gated. To use them:

1. Request access on each model page, e.g.
   [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
   and
   [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).
2. Create a read token at <https://huggingface.co/settings/tokens>.
3. Set `HF_TOKEN` in `.env` (see `sample.env`).

Downloaded weights are cached under `~/.cache/huggingface` (about 16 GB for an
8B model); set `HF_HOME` to use a different location. On Windows, enabling
Developer Mode lets the cache use symlinks and avoids duplicate files.

### Check the environment

```bash
uv run python old/interp_smoke_test.py
uv run python old/interp_smoke_test.py Qwen/Qwen2.5-0.5B-Instruct  # ungated alternative
```

The script loads the model, runs a chat-formatted prompt, caches activations,
and reports the next-token prediction and peak GPU memory.

### Minimal open-weight run

`old/belief_revision_minimal_llama.py` is a small version of the experiment with
Llama-3.2-1B-Instruct as the target, run locally with `transformers`. It uses
one vignette (`politics_03`, set by `VIGNETTE_ID`) and 30 conversations drawn
from its condition cells. politics_03 was chosen because it showed the most
belief change in the original run; the reasoning is recorded next to
`VIGNETTE_ID` in the script. The user simulator and stance evaluator run on a local
[Ollama](https://ollama.com) model, so no OpenAI key is needed.

One-time Ollama setup:

1. Install Ollama.
2. Set the environment variable `OLLAMA_CONTEXT_LENGTH=16384` and restart
   Ollama. The evaluator prompt contains the whole dialogue, and Ollama
   silently truncates input longer than its (small) default context.
3. `ollama pull llama3.1:8b`

Then:

```bash
uv run python old/belief_revision_minimal_llama.py --n 2   # quick check
uv run python old/belief_revision_minimal_llama.py         # full 30 conversations
```

It writes `belief_revision_results_minimal_llama1b_<vignette>.csv` and
`dialogues_minimal_llama1b_<vignette>.jsonl`, saving after each conversation.

### GPU memory

An 8B model in bf16 needs about 16 GB for its weights alone, plus room for
cached activations, so plan on a GPU with at least 24 GB (40–80 GB is
comfortable). On smaller GPUs, develop against a 1B–3B model and run the 8B
experiments on a cluster or cloud GPU.

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
tests mock NDIF and Supabase integrations, so the suite does not submit paid
inference jobs or modify live reporting data.

## NDIF setup

1. Go to the [NDIF get-started page](https://ndif.us/get-started/).
2. Select **Register for your free API key** and sign in or register.
3. Copy the API key into `NDIF_API_KEY` in `.env`.
4. Create a Hugging Face read token and copy it into `HF_TOKEN` in `.env`.
5. Accept the license for each gated Hugging Face model you plan to use.

The migrated entry point uses NDIF for the simulated user and target assistant.
The behavioral evaluator uses the API provider and model selected in the YAML
configuration; NDIF is deliberately not an evaluator option because its
accumulated-context probe requests repeatedly exceeded per-job memory limits.

The behavioral evaluator receives the complete accumulated dialogue at each
configured checkpoint. Long evaluator prompts can exceed NDIF's per-job memory
allowance even without request batching. The temporary global response-token
and prompt-length caps in `src/belief_revision/core/llm.py` are currently
disabled. Full-context evaluator runs therefore require a configured provider
API key, or a future evaluation path based on validated activation probes.
For activation-only or mechanical smoke runs, pass `--evaluator-mode disabled`
to skip behavioral evaluator calls. Conversation generation and extension
hooks still run normally, while behavioral probe fields remain empty.

## Verify the NDIF setup

Run these commands from the repository root.

Confirm that the project is using Python 3.12 or newer:

```bash
uv run python --version
```

Confirm that NNsight is installed and importable:

```bash
uv run python -c "import nnsight; print(nnsight.__version__)"
```

Submit one live generation for each model role configured in
`src/belief_revision/core/defaults.py`. User and assistant validation use NDIF;
evaluator validation uses its configured API provider:

```bash
uv run validate-models --user
uv run validate-models --assistant
uv run validate-models --evaluator
uv run validate-models --all
```

Each test passes when NDIF reports the job as `COMPLETED` and the model returns
a non-empty response. The hosted base models may not follow the request to
return exactly `OK`; these commands verify connectivity and generation rather
than instruction-following quality. These are live integration tests and are
not part of the automatic unit-test suite.

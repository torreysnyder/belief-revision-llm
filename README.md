# Belief Revision in LLMs

This repository contains a behavioral experiment for studying how a language
model revises a core claim and its auxiliary explanations during a structured,
multi-turn conversation.

## Repository contents

- `belief_revision_experiment.py` generates simulated conversations and probes
  the target model's inferred stance at predefined checkpoints.
- `analysis_full_crossed.py` analyzes the generated CSV and writes statistical
  tables, model summaries, and figures.
- `vignettes_revised.json` defines the health and political belief-revision
  vignettes used by the experiment.

## Experiment workflow

[![Belief-revision experiment workflow](assets/original_flow.svg)](assets/original_flow.svg)

The workflow proceeds from experiment configuration and deterministic run-plan
construction through parallel conversation execution, generated datasets, and
the analysis pipeline. Select the diagram to open the full-resolution version.

## Development status

The operational entry points remain `belief_revision_experiment.py` and
`analysis_full_crossed.py`; the run commands below are unchanged. An incremental
restructure is being developed under `src/belief_revision/`. The shared models,
experiment configuration, and deterministic design logic are currently
implemented in `models.py`, `config.py`, and `design.py`. State helpers,
file-backed prompt templates, prompt rendering, and an NDIF inference client
are also under development. These modules are not yet connected into a complete
replacement workflow; the remaining modules, scripts, and tests will be
completed one component at a time.

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

For the legacy experiment, set `OPENAI_API_KEY`. For NDIF development, set
`NDIF_API_KEY` and a Hugging Face read token in `HF_TOKEN`; the Hugging Face
token is needed to retrieve gated model configuration and tokenizer files.
Accept the applicable model license on Hugging Face before testing a gated
model. The resulting `.env` file is ignored by Git and must not be committed.

## Run the experiment

```bash
uv run python belief_revision_experiment.py
```

The script currently uses `gpt-4.1-mini` for both the target assistant and user
simulation. Its default full-crossed configuration runs many conversations and
can incur substantial API usage. Review the model, replicate, and worker
constants near the top of the script before starting a run.

The experiment writes:

- `belief_revision_results_full_crossed.csv`
- `dialogues_full_crossed.jsonl`

These generated files are ignored by Git.

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
provide automated tests, command-line options, or checkpoint/resume support.

## NDIF setup

1. Go to the [NDIF get-started page](https://ndif.us/get-started/).
2. Select **Register for your free API key** and sign in or register.
3. Copy the API key into `NDIF_API_KEY` in `.env`.
4. Create a Hugging Face read token and copy it into `HF_TOKEN` in `.env`.
5. Accept the license for each gated Hugging Face model you plan to use.

The modular NDIF client is not yet wired into the legacy operational entry
points. NDIF model availability also depends on the current deployment status
and the access level associated with the API key.

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

Submit one remote generation using the target model configured in
`src/belief_revision/config.py`:

```bash
PYTHONPATH=src uv run python -c "from belief_revision.config import TARGET_MODEL; from belief_revision.llm import call_model; result=call_model(TARGET_MODEL, [{'role':'user','content':'Reply with exactly: OK'}], temperature=0, max_tokens=10, retries=1); print('MODEL:', TARGET_MODEL); print('RESULT:', repr(result))"
```

The test passes when NDIF reports the job as `COMPLETED` and `RESULT` contains a
non-empty response. The hosted base Llama models may not follow the request to
return exactly `OK`; this command verifies connectivity and generation rather
than instruction-following quality.

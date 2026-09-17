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

## Setup

Python 3.10 or newer and [uv](https://docs.astral.sh/uv/) are required. Create
the project environment and install the locked dependencies:

```bash
uv sync
```

This creates the local virtual environment at `.venv/`. In an IDE such as
PyCharm, select `.venv/bin/python` as the project interpreter.

Copy the environment template and add your OpenAI API key:

```bash
cp sample.env .env
```

The experiment calls `load_dotenv()` and reads `OPENAI_API_KEY` from the
environment. The resulting `.env` file is ignored by Git and must not be
committed.

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

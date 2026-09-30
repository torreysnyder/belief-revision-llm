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
- `belief_revision_minimal_llama.py` runs a minimal version of the experiment
  with a local open-weight target model (see below).
- `interp_smoke_test.py` checks that the optional interpretability environment
  can load an open-weight model and cache activations.

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
uv run python interp_smoke_test.py
uv run python interp_smoke_test.py Qwen/Qwen2.5-0.5B-Instruct  # ungated alternative
```

The script loads the model, runs a chat-formatted prompt, caches activations,
and reports the next-token prediction and peak GPU memory.

### Minimal open-weight run

`belief_revision_minimal_llama.py` is a small version of the experiment with
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
uv run python belief_revision_minimal_llama.py --n 2   # quick check
uv run python belief_revision_minimal_llama.py         # full 30 conversations
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
provide automated tests, command-line options, or checkpoint/resume support.

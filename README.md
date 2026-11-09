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

Python 3.12 or newer and [uv](https://docs.astral.sh/uv/) 0.13.5 or newer are
required. This minimum includes the fix for uv's
[CA/TLS certificate advisory](https://github.com/astral-sh/uv/security/advisories/GHSA-hq85-5cpj-mr62),
and `pyproject.toml` enforces the same minimum for both uv and `uv_build`.
Upgrade an existing `uv` installation first:

```bash
# Installations created by uv's standalone installer
uv self update

# Homebrew installations
brew upgrade uv
```

Then install or update the managed Python 3.12 interpreter and create the
project environment from the lockfile:

```bash
uv python install 3.12
uv python upgrade 3.12
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
The default behavioral evaluator is `openai/gpt-4.1-mini` through OpenRouter,
so set `OPENROUTER_API_KEY` unless evaluation is disabled or another provider
is selected in YAML. Accept the applicable model license on Hugging Face before
testing a gated model. The resulting `.env` file is ignored by Git and must not
be committed.

## Run the experiment

```bash
uv run python scripts/run_experiment.py --single-run
```

This runs one replicate from one primary cell and writes results locally. Fallback model roles, replicate counts, and worker defaults are configured in
`src/belief_revision/core/defaults.py`. Select the target for one run with the
profiles below; the user simulator and evaluator keep their existing settings. Omit `--single-run` to run the full
configured experiment:

```bash
uv run python scripts/run_experiment.py
```

The full configuration runs many conversations and submits live NDIF requests.
Review the configured models, replicate count, and worker count before starting
it.

Each run writes to its own folder, so earlier runs are never overwritten:

- `runs/<run-name>/results.csv`
- `runs/<run-name>/dialogues.jsonl`
- `runs/<run-name>/run_manifest.json` — requested target profile and fingerprint,
  user model, evaluator settings, and output paths

The run name defaults to the UTC start time plus the target model (for example
`20261023-153012_llama-3.1-8b-instruct`, with `_single` appended for
`--single-run`). Choose your own with `--run-name pilot-llama8b`. The run
refuses to start if that folder already exists. `--csv-output` and
`--jsonl-output` override the individual paths (and a run writing to an
explicit path replaces any file already there).

These generated files are ignored by Git. `analysis_full_crossed.py` still reads
`belief_revision_results_full_crossed.csv` from the repository root, so to analyze
a run, copy its `results.csv` there or call `main(input_csv=...)` with its path.

### Select the target model

The eight files under [`configs/models/`](configs/models/) define the target
matrix without changing the shared defaults:

| Family | Main target profile | Lighter target profile |
|---|---|---|
| Llama (primary) | [3.1 8B](configs/models/llama-3.1-8b-instruct.yaml) | [3.2 3B](configs/models/llama-3.2-3b-instruct.yaml) |
| Qwen | [2.5 7B](configs/models/qwen-2.5-7b-instruct.yaml) | [2.5 3B](configs/models/qwen-2.5-3b-instruct.yaml) |
| Gemma | [3 12B](configs/models/gemma-3-12b-it.yaml) | [3 4B](configs/models/gemma-3-4b-it.yaml) |
| Ministral | [3 8B](configs/models/ministral-3-8b-instruct.yaml) | [3 3B](configs/models/ministral-3-3b-instruct.yaml) |

Select one profile for a run, or use a model ID with compatibility defaults:

```bash
uv run python scripts/run_experiment.py --single-run --model-config configs/models/llama-3.1-8b-instruct.yaml
uv run python scripts/run_experiment.py --single-run --target-model Qwen/Qwen2.5-7B-Instruct
```

Alternatively, add this to a run configuration under `configs/`:

```yaml
target:
  model_config: models/llama-3.1-8b-instruct.yaml
```

The YAML path is relative to the run configuration file. CLI `--model-config`
or `--target-model` takes precedence; choose one, not both. `target.model_id`
is the YAML alternative to `target.model_config`. Omitting target selection
retains `TARGET_MODEL` from `core/defaults.py`; this does not select a different
user simulator or evaluator. Explicit CSV output paths place the manifest at
`<csv-stem>.run_manifest.json` beside that CSV.

Profiles specify model ID/revision, requested dtype, Transformers loader and
task, decoder/layer/head/normalization/logits paths, system-role handling, and
assistant end-token/suffix handling. Layer counts and hidden width come from
the loaded model rather than fixed Llama dimensions. An unfamiliar architecture
needs a matching profile and validation; these known layouts are not universal
model support.

All eight checked-in revisions currently say `main`: pin an exact model commit
before collecting reproducible data. Backend availability and actual loaded
weight precision have not been verified for this matrix. In particular, NDIF
does not enforce the profile's requested dtype; the run manifest records the
requested configuration rather than proving backend precision.
Local Torrey extraction/steering can override that request with `--dtype`;
their metadata records the effective `forward_dtype`/`dtype` separately from
the frozen requested profile.

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

Run these commands from the repository root after `uv sync`, except where a
`cd` is shown. Registered live adapters use the repeatable `--extension` flag;
Q's notebook and Kati's analysis tools run separately.

Use the target profiles above to select the studied assistant for each run.
Shared fallback settings remain Qwen2.5-14B-Instruct for the target,
Llama-3.1-8B-Instruct for the user simulator, and OpenRouter
`openai/gpt-4.1-mini` for evaluation. Model identity, prompts, phases, and saved
activation spaces must match; selecting a target leaves the user/evaluator roles
unchanged. Each standalone workflow below accepts or carries the profile explicitly.

#### Q / Qingqing: endorsement-score validation

1. Open [`test_endorse_probe.ipynb`](src/belief_revision/extensions/qingqing/endorsement_validation/test_endorse_probe.ipynb)
   in Google Colab and select a CUDA GPU runtime.
2. Follow its dependency-install and runtime-restart instructions. In the
   configuration cell, review `MODEL_CONFIG_PATH`, `MODEL_ID`, `PRECISION`,
   `USE_DRIVE`, `DATA_PATH`, and `OUTPUT_ROOT`. To select a shared profile,
   upload its YAML to Colab and set `MODEL_CONFIG_PATH` to the uploaded path.
   The profile supplies model/revision/dtype and architecture settings; its dtype
   selects unquantized `bf16`/`fp16`/`fp32` loading. Leave the path empty to retain
   the notebook's existing Qwen2.5-7B `4bit` default. Repository globals do not
   change notebook settings.
3. Run the remaining cells in order. With `DATA_PATH = ""` and
   `LOAD_FROM_OSF = True`, it downloads the conversation CSV from OSF node
   `hs867`. Set `DATA_PATH` to use another supported CSV/JSON/JSONL input.
   With `USE_DRIVE = True`, authorize the Drive mount to persist outputs.

The notebook derives layer count and width from the loaded model, uses the
configured decoder/head paths, and checks native final logits against the
configured output-head transform, including logit softcapping where applicable.
It loads the model on the Colab GPU and performs extraction and validation;
it does not run the shared user simulator or API evaluator. Existing
results are under
[`endorsement_validation/extractions/`](src/belief_revision/extensions/qingqing/endorsement_validation/extractions/).
There is no registered `--extension` adapter or script CLI for this notebook.

#### Kati: belief decoding and endorsement probes

Kati's standalone tools live under `src/belief_revision/extensions/kati/`.
Use the original saved run directory; run data and activation tensors are not
included with the scripts. Reanalysis of saved outputs does not call a model:

```bash
uv run python -m belief_revision.extensions.kati.decoding.make_labels --run runs/politics03-llama8b-full
uv run python -m belief_revision.extensions.kati.decoding.check_scale --run runs/politics03-llama8b-full --position prompt_end
uv run python -m belief_revision.extensions.kati.decoding.decode --run runs/politics03-llama8b-full --out-name rerun
```

`make_labels` reads `results.csv`; the scale check and decoder also need the saved
`activations/` tensors. The decoder uses `dialogues.jsonl` for its text baseline
and writes `decoding/rerun.csv`.

To create new activations or endorsement measurements, select a target profile
explicitly. The following uses the original Llama-3.1-8B target independently of
global defaults; prompts, phases, and source transcripts still need to match.
With NDIF/Hugging Face access configured, these commands make model requests:

```bash
uv run python -m belief_revision.extensions.kati.decoding.extract_activations --run runs/politics03-llama8b-full --model-config configs/models/llama-3.1-8b-instruct.yaml --out runs/politics03-llama8b-full/activations-profile
uv run python -m belief_revision.extensions.kati.decoding.endorsement_probe --run runs/politics03-llama8b-full --model-config configs/models/llama-3.1-8b-instruct.yaml --out runs/politics03-llama8b-full/endorsement-profile --batch-size 4
```

Both tools also accept `--model MODEL_ID` instead of `--model-config`. New
captures write `capture_manifest.json` and tensor identities covering the
profile, requested layers, phases, prompt, source, and operation settings.
Resuming a different configuration or an old directory without provenance is
refused: choose a fresh `--out` directory. Existing legacy tensors remain
readable for analysis. `--dry-run` avoids NDIF execution but may download a tokenizer.

Use `--activations-dir` with `check_scale` or `decode`, and `--endorsement-dir`
with `decode`, to analyze separate capture directories. For example:

```bash
uv run python -m belief_revision.extensions.kati.decoding.endorsement_probe --run runs/politics03-llama8b-full --out runs/politics03-llama8b-full/endorsement-profile --collect
uv run python -m belief_revision.extensions.kati.decoding.decode --run runs/politics03-llama8b-full --activations-dir runs/politics03-llama8b-full/activations-profile --endorsement-dir runs/politics03-llama8b-full/endorsement-profile --labels endorsement --out-name endorsement-rerun
uv run python -m belief_revision.extensions.kati.results_summary.make_figures
```

`--collect` only reads saved tensors; omit model-selection flags for that step.
Label mapping and decoding methods retain their existing research definitions.

Figure regeneration additionally requires the original early summary CSVs under
`output_full_crossed/paper_v3/` and the named decoding CSVs `results.csv` and
`probe_selfdecode_partial.csv` under the saved run's `decoding/` directory.
See the [decoding notes](src/belief_revision/extensions/kati/decoding/NOTES.md)
and [preliminary results summary](src/belief_revision/extensions/kati/results_summary/RESULTS_SUMMARY.md).
Check the package-to-core integration offline with
`uv run pytest tests/test_kati_research_integration.py --no-cov`; use `uv run pytest`
for the full suite. These remain standalone post-hoc tools without a registered
live conversation adapter.

#### Torrey: CAA capture and steering

For activation capture during the shared experiment, review
[`configs/torrey-caa.yaml`](configs/torrey-caa.yaml), then select a target:

```bash
uv run python scripts/run_experiment.py \
  --single-run \
  --model-config configs/models/llama-3.1-8b-instruct.yaml \
  --run-name llama8b-pilot \
  --extension torrey-caa \
  --torrey-caa-output-dir runs/llama8b-pilot/captures \
  --config configs/torrey-caa.yaml
```

Choose an unused run name. This runs live conversation generation and the
configured evaluator, writing the matching transcript to
`runs/llama8b-pilot/dialogues.jsonl` and selected-turn/layer captures to
`runs/llama8b-pilot/captures/`.
Add `--reporting true` to upload completed records and activation artifacts.
Hooks run in the order supplied on the command line.

The standalone generator and extractor also accept `--model-config`. The
generator freezes the selected target and matching tokenizer into the pair
manifest; extraction carries that profile into activation files; vector building
preserves it; steering rejects a profile that differs from the frozen vectors.
The profile selects the studied target, not the rewrite/user model.

For a new selected-target pipeline, use the matching pilot transcript from the
run above and separate output directories. The checked-in original pilot/pair
manifest records Qwen2.5-0.5B; it cannot be reused as Llama data. Run from the
pipeline directory so relative paths agree:

```bash
cd activation_steering_pipeline
model_config=../configs/models/llama-3.1-8b-instruct.yaml
pair_dir=data/llama8b-profile
activation_dir=activations/llama8b-profile
vector_dir=vectors/llama8b-profile
result_dir=results/llama8b-profile
uv run python generate_caa_contrastive_pairs.py --input ../runs/llama8b-pilot/dialogues.jsonl --out-dir "$pair_dir" --model-config "$model_config"
uv run python extract_caa_activations.py --model-config "$model_config" \
  --core-pairs "$pair_dir/caa_pairs_core.jsonl" --auxiliary-pairs "$pair_dir/caa_pairs_auxiliary.jsonl" \
  --pair-manifest "$pair_dir/caa_pairs_manifest.json" --split-manifest "$pair_dir/caa_split_manifest.json" --out-dir "$activation_dir"
uv run python build_caa_vectors.py --activations-dir "$activation_dir" \
  --split-manifest "$pair_dir/caa_split_manifest.json" --vector-dir "$vector_dir" --results-dir "$result_dir"
uv run python activation_steering.py --model-config "$model_config" \
  --vector-dir "$vector_dir" --diagnostics "$result_dir/vector_diagnostics.csv" \
  --core-pairs "$pair_dir/caa_pairs_core.jsonl" --auxiliary-pairs "$pair_dir/caa_pairs_auxiliary.jsonl" \
  --split-manifest "$pair_dir/caa_split_manifest.json" --results-dir "$result_dir/steering" \
  --stages generate summarize --run-name development-no-judge
cd ..
```

Choose unused output directories and a fresh `--run-name`; existing names resume
outputs and can include previous judgments. Pair generation needs its source
pilot and NDIF access; standalone extraction and steering load models locally
and need suitable GPU memory. For gated models, export `HF_TOKEN` in the shell
or use an existing Hugging Face login; these two scripts do not load `.env`.
Use an environment that provides the profile's Transformers loader.

The final command runs generation and summary only. The standalone judge stage
still imports the unavailable `belief_revision_experiment_qwen` module;
target profiles do not repair that stage. This command produces no fresh
judge-based belief scores. See the
[pipeline notes](activation_steering_pipeline/README_controlled_counterfactual_CAA.md)
for individual stages and an offline pair-building smoke check.

#### Extension walkthrough

The `jasons-exp` walkthrough is implemented in
[`src/belief_revision/demo/jasons_exp.py`](src/belief_revision/demo/jasons_exp.py).
It demonstrates conversation lifecycle hooks, activation capture,
researcher-owned artifacts, optional reporting, and read-only Supabase access:

```bash
uv run python scripts/run_experiment.py \
  --single-run \
  --extension jasons-exp \
  --config configs/default.yaml
```

Add `--reporting true` after configuring the required Supabase values in `.env`.

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
the migrated `src/belief_revision/` package and operational `scripts/`, with
explicit omissions for the walkthrough and Kati's standalone research drafts.
The drafts retain their previous coverage scope after relocation. Unit tests
mock NDIF and Supabase integrations, so the suite does not submit paid inference
jobs or modify live reporting data.

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

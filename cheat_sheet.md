# Cheat Sheet

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

## Run the full experiment with Supabase reporting

```bash
uv run python scripts/run_experiment.py --reporting true
```

The full run uses the replicate and worker defaults in
[`src/belief_revision/config.py`](src/belief_revision/config.py). Supabase and
local files are updated after each complete primary cell, not after each remote
request.

## Modify the configured models

Edit [`src/belief_revision/config.py`](src/belief_revision/config.py):

- `TARGET_MODEL` — target assistant being studied
- `USER_SIM_MODEL` — simulated user
- `EVALUATOR_PROVIDER` — `deepseek` (default), `openai`, `anthropic`, or `google`
- `EVALUATOR_MODELS` — model ID for each API provider
- `EVALUATOR_MAX_TOKENS` — evaluator output budget (4096 by default)

Set the selected provider’s key in `.env`: `DEEPSEEK_API_KEY`,
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY`.
Only the assistant and simulated user use NDIF.

## Validate configured models (live requests)

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

## Change NDIF generation behavior

Edit [`src/belief_revision/llm.py`](src/belief_revision/llm.py) to change remote
generation behavior such as prompt formatting, retry handling, or global token
limits.

Per-call temperatures and token limits currently live in
[`src/belief_revision/conversation.py`](src/belief_revision/conversation.py).

The temporary global prompt-length and response-token caps in `llm.py` are
currently commented out. Evaluator checkpoints now use the selected external
API with full dialogue context; provider context and output limits still apply.

## Configure credentials

Copy [`sample.env`](sample.env) to `.env` and provide the required local
credentials. Never commit `.env`.

## Replay belief regions locally

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Open `http://localhost:8000/assets/mock-brain/results.html` and click **Load latest
local run**, or use **Open results** to select another JSONL file. Step through recorded turns, or click **Auto-play · 3s** to loop the conversation. Core Beliefs map
to the cerebellum; A1/A2 map to the left/right cerebral hemispheres. This replays
completed results; it does not start inference or connect to a live run. See the
[brain integration guide](assets/mock-brain/README.md).

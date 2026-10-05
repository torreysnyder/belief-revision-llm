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
- `EVALUATOR_MODEL` — behavioral evaluator

## Validate configured NDIF models

```bash
uv run validate-models --user
uv run validate-models --assistant
uv run validate-models --evaluator
uv run validate-models --all
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
currently commented out. With full accumulated evaluator context enabled, long
requests may exceed NDIF's per-job memory allowance.

## Configure credentials

Copy [`sample.env`](sample.env) to `.env` and provide the required local
credentials. Never commit `.env`.

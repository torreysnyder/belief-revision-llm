"""Coordinate experiment cells, replicates, and result storage."""

import traceback
import concurrent.futures
from ..extensions import ExtensionFactory
from ..persistence import ExperimentBackend, LocalFileBackend, SupabaseReporter
from .conversation import run_one_vignette
from .defaults import (
    EVALUATOR_MAX_TOKENS,
    EVALUATOR_MODEL,
    EVALUATOR_PROVIDER,
    MAX_WORKERS,
    PHASES,
    REPLICATES_PER_PRIMARY_CELL,
    STYLE_LIBRARY,
    TARGET_MODEL,
    USER_SIM_MODEL,
)
from .design import build_primary_cells, build_run_plans
from .models import ConversationResult, EvaluatorMode, PrimaryCell, RunPlan, Vignette
from collections.abc import Sequence
from pathlib import Path


def _run_replicate(
    primary_cell: PrimaryCell,
    run_plan: RunPlan,
    extension_factories: Sequence[ExtensionFactory],
    evaluator_mode: EvaluatorMode,
    evaluator_provider: str,
    evaluator_model: str,
    evaluator_max_tokens: int,
) -> ConversationResult:
    """Build replicate-local extensions and run one conversation."""

    extensions = tuple(factory() for factory in extension_factories)
    return run_one_vignette(
        primary_cell=primary_cell,
        run_plan=run_plan,
        extensions=extensions,
        evaluator_mode=evaluator_mode,
        evaluator_provider=evaluator_provider,
        evaluator_model=evaluator_model,
        evaluator_max_tokens=evaluator_max_tokens,
    )


def run_primary_cell(
    primary_cell: PrimaryCell,
    replicates_per_cell: int,
    max_workers: int,
    extension_factories: Sequence[ExtensionFactory] = (),
    evaluator_mode: EvaluatorMode = "enabled",
    evaluator_provider: str = EVALUATOR_PROVIDER,
    evaluator_model: str = EVALUATOR_MODEL,
    evaluator_max_tokens: int = EVALUATOR_MAX_TOKENS,
) -> list[ConversationResult]:
    """Run every configured replicate for one primary cell."""

    run_plans = build_run_plans(
        primary_cell,
        replicates_per_cell,
    )

    completed_results: list[ConversationResult] = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers,
    ) as executor:
        futures = {
            executor.submit(
                _run_replicate,
                primary_cell=primary_cell,
                run_plan=run_plan,
                extension_factories=extension_factories,
                evaluator_mode=evaluator_mode,
                evaluator_provider=evaluator_provider,
                evaluator_model=evaluator_model,
                evaluator_max_tokens=evaluator_max_tokens,
            ): run_plan
            for run_plan in run_plans
        }

        for future in concurrent.futures.as_completed(futures):
            run_plan = futures[future]
            try:
                result = future.result()
            except Exception as error:
                print(f"  Replicate failed for cell={primary_cell.cell_id} replicate={run_plan.replicate_index}: {error}")
                traceback.print_exception(error)
                continue

            completed_results.append(result)

            print(f"  Finished replicate {result.run_plan.replicate_index + 1}/{replicates_per_cell} for {primary_cell.cell_id}")

    return completed_results


def run_experiment(
    vignettes: list[Vignette],
    csv_path: str | Path = ("belief_revision_results_full_crossed.csv"),
    jsonl_path: str | Path = ("dialogues_full_crossed.jsonl"),
    replicates_per_cell: int = (REPLICATES_PER_PRIMARY_CELL),
    max_workers: int = MAX_WORKERS,
    max_cells: int | None = None,
    reporting: bool = False,
    extension_factories: Sequence[ExtensionFactory] = (),
    evaluator_mode: EvaluatorMode = "enabled",
    evaluator_provider: str = EVALUATOR_PROVIDER,
    evaluator_model: str = EVALUATOR_MODEL,
    evaluator_max_tokens: int = EVALUATOR_MAX_TOKENS,
) -> int:
    """Run the experiment and return the number of completed conversations."""

    primary_cells = build_primary_cells(vignettes)

    if max_cells is not None:
        primary_cells = primary_cells[:max_cells]

    total_cells = len(primary_cells)
    total_conversations = total_cells * replicates_per_cell

    backends: list[ExperimentBackend] = [LocalFileBackend(csv_path, jsonl_path)]
    if reporting:
        backends.append(SupabaseReporter.from_environment())

    run_config = {
        "replicates_per_cell": replicates_per_cell,
        "max_workers": max_workers,
        "max_cells": max_cells,
        "phases": PHASES,
        "evaluator_mode": evaluator_mode,
        "evaluator_provider": evaluator_provider,
        "evaluator_max_tokens": evaluator_max_tokens,
    }
    active_backends = [
        (
            backend,
            backend.start_run(
                target_model=TARGET_MODEL,
                user_sim_model=USER_SIM_MODEL,
                evaluator_model=evaluator_model,
                config=run_config,
            ),
        )
        for backend in backends
    ]

    print(f"Loaded {len(vignettes)} vignettes.")
    print(f"Using {len(STYLE_LIBRARY)} style profiles.")
    print(f"Primary cells: {total_cells}")
    print(f"Replicates per primary cell: {replicates_per_cell}")
    print(f"Total conversations to generate: {total_conversations}")

    completed_conversations = 0

    try:
        for cell_number, primary_cell in enumerate(
            primary_cells,
            start=1,
        ):
            print(f"\nStarting primary cell {cell_number}/{total_cells}: {primary_cell.cell_id}")

            cell_results = run_primary_cell(
                primary_cell=primary_cell,
                replicates_per_cell=replicates_per_cell,
                max_workers=max_workers,
                extension_factories=extension_factories,
                evaluator_mode=evaluator_mode,
                evaluator_provider=evaluator_provider,
                evaluator_model=evaluator_model,
                evaluator_max_tokens=evaluator_max_tokens,
            )

            for backend, run_id in active_backends:
                backend.save_conversations(run_id, cell_results)

            completed_conversations += len(cell_results)

            print(f"  Saved {len(cell_results)} conversations for {primary_cell.cell_id}")
    except Exception as error:
        for backend, run_id in active_backends:
            backend.fail_run(run_id, error)
        raise

    for backend, run_id in active_backends:
        backend.complete_run(
            run_id,
            completed_conversations=completed_conversations,
            expected_conversations=total_conversations,
        )

    print(f"\nDone. Completed {completed_conversations}/{total_conversations} conversations.")

    return completed_conversations

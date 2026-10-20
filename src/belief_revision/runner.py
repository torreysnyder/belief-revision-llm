"""Coordinate experiment cells, replicates, and result storage."""

import concurrent.futures
from pathlib import Path

from .config import (
    EVALUATOR_MODEL,
    MAX_WORKERS,
    PHASES,
    REPLICATES_PER_PRIMARY_CELL,
    STYLE_LIBRARY,
    TARGET_MODEL,
    USER_SIM_MODEL,
)
from .conversation import run_one_vignette
from .design import build_primary_cells, build_run_plans
from .models import ConversationResult, PrimaryCell, Vignette
from .reporting import SupabaseReporter
from .storage import save_results_append


def run_primary_cell(
    primary_cell: PrimaryCell,
    replicates_per_cell: int,
    max_workers: int,
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
        futures = [
            executor.submit(
                run_one_vignette,
                primary_cell=primary_cell,
                run_plan=run_plan,
            )
            for run_plan in run_plans
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as error:
                print(f"  Replicate failed: {error}")
                continue

            completed_results.append(result)

            print(
                "  Finished replicate "
                f"{result.run_plan.replicate_index + 1}/"
                f"{replicates_per_cell} "
                f"for {primary_cell.cell_id}"
            )

    return completed_results


def run_experiment(
    vignettes: list[Vignette],
    csv_path: str | Path = ("belief_revision_results_full_crossed.csv"),
    jsonl_path: str | Path = ("dialogues_full_crossed.jsonl"),
    replicates_per_cell: int = (REPLICATES_PER_PRIMARY_CELL),
    max_workers: int = MAX_WORKERS,
    max_cells: int | None = None,
    reporting: bool = False,
) -> int:
    """Run the experiment and return the number of completed conversations."""

    csv_path = Path(csv_path)
    jsonl_path = Path(jsonl_path)

    jsonl_path.write_text(
        "",
        encoding="utf-8",
    )

    if csv_path.exists():
        csv_path.unlink()

    primary_cells = build_primary_cells(vignettes)

    if max_cells is not None:
        primary_cells = primary_cells[:max_cells]

    total_cells = len(primary_cells)
    total_conversations = total_cells * replicates_per_cell

    reporter = SupabaseReporter.from_environment() if reporting else None
    reporting_run_id = (
        reporter.start_run(
            target_model=TARGET_MODEL,
            user_sim_model=USER_SIM_MODEL,
            evaluator_model=EVALUATOR_MODEL,
            config={
                "replicates_per_cell": replicates_per_cell,
                "max_workers": max_workers,
                "max_cells": max_cells,
                "phases": PHASES,
            },
        )
        if reporter
        else None
    )

    print(f"Loaded {len(vignettes)} vignettes.")
    print(f"Using {len(STYLE_LIBRARY)} style profiles.")
    print(f"Primary cells: {total_cells}")
    print(f"Replicates per primary cell: {replicates_per_cell}")
    print(f"Total conversations to generate: {total_conversations}")

    write_header = True
    completed_conversations = 0

    try:
        for cell_number, primary_cell in enumerate(
            primary_cells,
            start=1,
        ):
            print(
                f"\nStarting primary cell "
                f"{cell_number}/{total_cells}: "
                f"{primary_cell.cell_id}"
            )

            cell_results = run_primary_cell(
                primary_cell=primary_cell,
                replicates_per_cell=replicates_per_cell,
                max_workers=max_workers,
            )

            save_results_append(
                cell_results,
                csv_path=csv_path,
                jsonl_path=jsonl_path,
                write_header=write_header,
            )

            if reporter and reporting_run_id:
                reporter.save_conversations(
                    reporting_run_id,
                    cell_results,
                )

            if cell_results:
                write_header = False

            completed_conversations += len(cell_results)

            print(
                f"  Saved {len(cell_results)} conversations for {primary_cell.cell_id}"
            )
    except Exception as error:
        if reporter and reporting_run_id:
            reporter.fail_run(reporting_run_id, error)
        raise

    if reporter and reporting_run_id:
        reporter.complete_run(
            reporting_run_id,
            completed_conversations=completed_conversations,
            expected_conversations=total_conversations,
        )

    print(
        f"\nDone. Completed "
        f"{completed_conversations}/"
        f"{total_conversations} conversations."
    )

    return completed_conversations

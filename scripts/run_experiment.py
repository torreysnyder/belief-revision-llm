"""Run the migrated belief-revision experiment."""

import argparse
from pathlib import Path

from belief_revision.config import (
    MAX_WORKERS,
    REPLICATES_PER_PRIMARY_CELL,
)
from belief_revision.runner import run_experiment
from belief_revision.storage import load_vignettes


def parse_bool(value: str) -> bool:
    """Parse an explicit true/false CLI value."""

    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_args() -> argparse.Namespace:
    """Parse experiment execution options."""

    parser = argparse.ArgumentParser(
        description="Run the belief-revision experiment.",
    )

    parser.add_argument(
        "--vignettes",
        type=Path,
        default=Path("vignettes_revised.json"),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path(
            "belief_revision_results_full_crossed.csv"
        ),
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        default=Path(
            "dialogues_full_crossed.jsonl"
        ),
    )
    parser.add_argument(
        "--replicates-per-cell",
        type=int,
        default=REPLICATES_PER_PRIMARY_CELL,
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Run one replicate from one primary cell.",
    )
    parser.add_argument(
        "--reporting",
        type=parse_bool,
        default=False,
        metavar="{true,false}",
        help=(
            "Upload completed records to Supabase in addition to "
            "writing local files."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Load the vignettes and run the requested experiment scope."""

    args = parse_args()

    replicates_per_cell = (
        1
        if args.single_run
        else args.replicates_per_cell
    )
    max_workers = (
        1
        if args.single_run
        else args.max_workers
    )
    max_cells = (
        1
        if args.single_run
        else args.max_cells
    )

    vignettes = load_vignettes(args.vignettes)

    run_experiment(
        vignettes=vignettes,
        csv_path=args.csv_output,
        jsonl_path=args.jsonl_output,
        replicates_per_cell=replicates_per_cell,
        max_workers=max_workers,
        max_cells=max_cells,
        reporting=args.reporting,
    )


if __name__ == "__main__":
    main()

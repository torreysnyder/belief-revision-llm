"""Run the migrated belief-revision experiment."""

import yaml
import argparse
from belief_revision.core.defaults import (
    EVALUATOR_MAX_TOKENS,
    EVALUATOR_MODEL,
    EVALUATOR_PROVIDER,
    MAX_WORKERS,
    REPLICATES_PER_PRIMARY_CELL,
)
from belief_revision.core.runner import run_experiment
from belief_revision.extensions import resolve_extension_factories
from belief_revision.extensions.torrey import DEFAULT_OUTPUT_DIR
from belief_revision.persistence import load_vignettes
from pathlib import Path


def parse_bool(value: str) -> bool:
    """Parse an explicit true/false CLI value."""

    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_csv_strings(value: str) -> frozenset[str]:
    """Parse a non-empty comma-separated set of names."""

    values = frozenset(part.strip() for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def parse_csv_positive_ints(value: str) -> tuple[int, ...]:
    """Parse unique positive integers while preserving their order."""

    try:
        values = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from error
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def load_config(path: Path | None) -> dict[str, object]:
    """Load experiment settings from a YAML configuration file."""

    if path is None:
        return {}
    with path.open(encoding="utf-8") as config_file:
        document = yaml.safe_load(config_file) or {}
    if not isinstance(document, dict):
        raise ValueError("config root must be a mapping")
    return document


def extension_configuration(document: dict[str, object]) -> dict[str, object]:
    """Read and validate the extension section of a configuration."""

    extensions = document.get("extensions", {})
    if not isinstance(extensions, dict):
        raise ValueError("config 'extensions' must be a mapping")
    return extensions


def evaluator_configuration(
    args: argparse.Namespace,
    document: dict[str, object],
) -> dict[str, object]:
    """Resolve the API evaluator, with an explicit CLI mode taking priority."""

    evaluator = document.get("evaluator", {})
    if not isinstance(evaluator, dict):
        raise ValueError("config 'evaluator' must be a mapping")
    unknown = set(evaluator) - {"mode", "provider", "model", "max_tokens"}
    if unknown:
        raise ValueError("unknown evaluator config field(s): " + ", ".join(sorted(unknown)))
    mode = args.evaluator_mode or evaluator.get("mode", "enabled")
    if mode not in {"enabled", "disabled"}:
        raise ValueError("evaluator mode must be 'enabled' or 'disabled'")
    provider = evaluator.get("provider", EVALUATOR_PROVIDER)
    if provider not in {"openai", "anthropic", "google", "deepseek"}:
        raise ValueError("evaluator provider must be an API provider, not NDIF")
    model = evaluator.get("model", EVALUATOR_MODEL)
    if not isinstance(model, str) or not model:
        raise ValueError("evaluator model must be a non-empty string")
    max_tokens = evaluator.get("max_tokens", EVALUATOR_MAX_TOKENS)
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("evaluator max_tokens must be a positive integer")
    return {
        "mode": mode,
        "provider": provider,
        "model": model,
        "max_tokens": max_tokens,
    }


def torrey_caa_configuration(
    args: argparse.Namespace,
    extension_config: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Merge Torrey's YAML settings with explicit CLI overrides."""

    raw = extension_config.get("torrey-caa", {})
    if not isinstance(raw, dict):
        raise ValueError("config 'extensions.torrey-caa' must be a mapping")
    allowed = {"phases", "turns", "layers", "output_dir"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError("unknown torrey-caa config field(s): " + ", ".join(sorted(unknown)))

    phases = args.torrey_caa_phases
    if phases is None and "phases" in raw:
        value = raw["phases"]
        if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
            raise ValueError("torrey-caa phases must be a non-empty list of names")
        phases = frozenset(value)

    turns = args.torrey_caa_turns
    if turns is None and "turns" in raw:
        turns = _positive_int_list(raw["turns"], "turns")

    layers = args.torrey_caa_layers
    if layers is None and "layers" in raw:
        layers = _positive_int_list(raw["layers"], "layers")

    output_dir = args.torrey_caa_output_dir
    if output_dir is None:
        output_dir = Path(raw.get("output_dir", DEFAULT_OUTPUT_DIR))

    return {
        "output_dir": output_dir,
        "layers": tuple(layer - 1 for layer in layers) if layers else None,
        "phases": phases,
        "turns": frozenset(turns) if turns else None,
    }


def _positive_int_list(value: object, field: str) -> tuple[int, ...]:
    """Validate a YAML list that uses one-based integer identifiers."""

    if not isinstance(value, list) or not value or not all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value):
        raise ValueError(f"torrey-caa {field} must be a non-empty positive-integer list")
    return tuple(dict.fromkeys(value))


def parse_args() -> argparse.Namespace:
    """Parse experiment execution options."""

    parser = argparse.ArgumentParser(
        description="Run the belief-revision experiment.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Load extension settings from a YAML file.",
    )
    parser.add_argument(
        "--vignettes",
        type=Path,
        default=Path("vignettes_revised.json"),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("belief_revision_results_full_crossed.csv"),
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        default=Path("dialogues_full_crossed.jsonl"),
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
        help=("Upload completed records to Supabase in addition to writing local files."),
    )
    parser.add_argument(
        "--evaluator-mode",
        choices=("enabled", "disabled"),
        default=None,
        help="Run behavioral evaluator probes or skip them entirely.",
    )
    parser.add_argument(
        "--extension",
        action="append",
        default=[],
        metavar="NAME",
        help=("Load a registered experiment extension. Repeat this option to load multiple extensions in the given order."),
    )
    parser.add_argument(
        "--torrey-caa-phases",
        type=parse_csv_strings,
        default=None,
        metavar="PHASE,...",
        help="Capture only these phases; default captures every phase.",
    )
    parser.add_argument(
        "--torrey-caa-turns",
        type=parse_csv_positive_ints,
        default=None,
        metavar="TURN,...",
        help="Capture only these one-based turns within each phase.",
    )
    parser.add_argument(
        "--torrey-caa-layers",
        type=parse_csv_positive_ints,
        default=None,
        metavar="LAYER,...",
        help="Capture these one-based layer numbers; default captures every layer.",
    )
    parser.add_argument(
        "--torrey-caa-output-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Directory for raw Torrey CAA activation artifacts.",
    )

    return parser.parse_args()


def main() -> None:
    """Load the vignettes and run the requested experiment scope."""

    args = parse_args()

    try:
        config = load_config(args.config)
        selected_evaluator = evaluator_configuration(args, config)
        extension_factories = resolve_extension_factories(
            args.extension,
            configurations={"torrey-caa": torrey_caa_configuration(args, extension_configuration(config))},
        )
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error

    replicates_per_cell = 1 if args.single_run else args.replicates_per_cell
    max_workers = 1 if args.single_run else args.max_workers
    max_cells = 1 if args.single_run else args.max_cells

    vignettes = load_vignettes(args.vignettes)

    run_experiment(
        vignettes=vignettes,
        csv_path=args.csv_output,
        jsonl_path=args.jsonl_output,
        replicates_per_cell=replicates_per_cell,
        max_workers=max_workers,
        max_cells=max_cells,
        reporting=args.reporting,
        extension_factories=extension_factories,
        evaluator_mode=selected_evaluator["mode"],
        evaluator_provider=selected_evaluator["provider"],
        evaluator_model=selected_evaluator["model"],
        evaluator_max_tokens=selected_evaluator["max_tokens"],
    )


if __name__ == "__main__":
    main()

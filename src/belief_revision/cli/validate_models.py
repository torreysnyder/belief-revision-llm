"""Run live NDIF smoke tests against the configured model roles."""

import argparse
from ..core.defaults import (
    EVALUATOR_MODEL,
    EVALUATOR_PROVIDER,
    TARGET_MODEL,
    USER_SIM_MODEL,
)
from ..core.llm import call_model

CONFIGURED_MODELS = {
    "user": USER_SIM_MODEL,
    "assistant": TARGET_MODEL,
    "evaluator": EVALUATOR_MODEL,
}


def parse_args() -> argparse.Namespace:
    """Parse the model role to validate."""

    parser = argparse.ArgumentParser(
        description="Validate configured models with live NDIF requests.",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--user", action="store_const", const="user", dest="role")
    selection.add_argument(
        "--assistant",
        action="store_const",
        const="assistant",
        dest="role",
    )
    selection.add_argument(
        "--evaluator",
        action="store_const",
        const="evaluator",
        dest="role",
    )
    selection.add_argument("--all", action="store_true")
    return parser.parse_args()


def validate_model(role: str, model_name: str) -> bool:
    """Submit one minimal request through the configured role provider."""

    print(f"Validating {role}: {model_name}")

    try:
        generation = call_model(
            model_name,
            [{"role": "user", "content": "Reply with exactly: OK"}],
            temperature=0,
            max_tokens=10,
            retries=1,
            provider=EVALUATOR_PROVIDER if role == "evaluator" else "ndif",
        )
    except Exception as error:
        print(f"FAIL {role}: {error}")
        print("\n---------------------------------------------------\n")
        return False

    if not generation.text:
        print(f"FAIL {role}: model returned an empty response")
        print("\n---------------------------------------------------\n")
        return False

    print(f"PASS {role}: {generation.text!r}")
    print("\n---------------------------------------------------\n")
    return True


def main() -> None:
    """Validate the selected configured model roles and set the exit status."""

    args = parse_args()
    selected_models = CONFIGURED_MODELS.items() if args.all else [(args.role, CONFIGURED_MODELS[args.role])]

    results = [validate_model(role, model_name) for role, model_name in selected_models]

    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

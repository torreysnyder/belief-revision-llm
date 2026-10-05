"""Run live NDIF smoke tests against the configured model roles."""

import argparse

from ..config import EVALUATOR_MODEL, TARGET_MODEL, USER_SIM_MODEL
from ..llm import call_model


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
    """Submit one minimal generation request for a configured model role."""

    print(f"Validating {role}: {model_name}")

    try:
        result = call_model(
            model_name,
            [{"role": "user", "content": "Reply with exactly: OK"}],
            temperature=0,
            max_tokens=10,
            retries=1,
        )
    except Exception as error:
        print(f"FAIL {role}: {error}")
        print("\n---------------------------------------------------\n")
        return False

    if not result:
        print(f"FAIL {role}: model returned an empty response")
        print("\n---------------------------------------------------\n")
        return False

    print(f"PASS {role}: {result!r}")
    print("\n---------------------------------------------------\n")
    return True


def main() -> None:
    """Validate the selected configured model roles and set the exit status."""

    args = parse_args()
    selected_models = (
        CONFIGURED_MODELS.items()
        if args.all
        else [(args.role, CONFIGURED_MODELS[args.role])]
    )

    results = [
        validate_model(role, model_name)
        for role, model_name in selected_models
    ]

    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

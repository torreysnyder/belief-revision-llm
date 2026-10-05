"""Print a compact summary of experiment data stored in Supabase."""

from collections import Counter
from typing import Any

from supabase import Client

from belief_revision.reporting import create_reporting_client


TABLES = {
    "runs": "experiment_runs",
    "conversations": "conversations",
    "turns": "conversation_turns",
    "probes": "behavioral_probes",
    "artifacts": "experiment_artifacts",
}


def count_rows(client: Client, table: str) -> int:
    """Return an exact row count without downloading table contents."""

    response = (
        client.table(table)
        .select("id", count="exact", head=True)
        .execute()
    )
    return response.count or 0


def fetch_all(
    client: Client,
    table: str,
    columns: str,
    page_size: int = 1_000,
) -> list[dict[str, Any]]:
    """Fetch selected lightweight columns across all result pages."""

    rows: list[dict[str, Any]] = []
    start = 0

    while True:
        page = (
            client.table(table)
            .select(columns)
            .range(start, start + page_size - 1)
            .execute()
            .data
        )
        rows.extend(page)

        if len(page) < page_size:
            return rows

        start += page_size


def format_breakdown(values: Counter[str]) -> str:
    """Format a count breakdown for terminal output."""

    if not values:
        return "none"

    return ", ".join(
        f"{name}={count}"
        for name, count in sorted(values.items())
    )


def main() -> None:
    """Query Supabase and print the current experiment summary."""

    client = create_reporting_client()
    counts = {
        label: count_rows(client, table)
        for label, table in TABLES.items()
    }
    runs = fetch_all(
        client,
        "experiment_runs",
        (
            "status,started_at,target_model,user_sim_model,"
            "evaluator_model"
        ),
    )
    conversations = fetch_all(
        client,
        "conversations",
        "status",
    )

    run_statuses = Counter(row["status"] for row in runs)
    conversation_statuses = Counter(
        row["status"] for row in conversations
    )
    target_models = Counter(row["target_model"] for row in runs)
    user_models = Counter(row["user_sim_model"] for row in runs)
    evaluator_models = Counter(
        row["evaluator_model"] for row in runs
    )
    data_points = counts["turns"] + counts["probes"]

    print("Supabase reporting summary")
    print(f"Runs: {counts['runs']}")
    print(f"Conversations: {counts['conversations']}")
    print(f"Turns: {counts['turns']}")
    print(f"Behavioral probes: {counts['probes']}")
    print(f"Artifacts: {counts['artifacts']}")
    print(f"Data points (turns + probes): {data_points}")
    print(f"Run statuses: {format_breakdown(run_statuses)}")
    print(
        "Conversation statuses: "
        f"{format_breakdown(conversation_statuses)}"
    )
    print(f"Target models: {format_breakdown(target_models)}")
    print(f"User models: {format_breakdown(user_models)}")
    print(
        "Evaluator models: "
        f"{format_breakdown(evaluator_models)}"
    )

    started_at = sorted(
        row["started_at"]
        for row in runs
        if row.get("started_at")
    )
    if started_at:
        print(f"First run: {started_at[0]}")
        print(f"Latest run: {started_at[-1]}")


if __name__ == "__main__":
    main()

"""Discover the project's named, inspectable SQL queries."""

import re
from pathlib import Path

QUERY_DIRECTORY = Path(__file__).resolve().parents[3] / "supabase" / "queries"
EXTENSION_DIRECTORY = Path(__file__).resolve().parents[1] / "extensions"
INCLUDE_PATTERN = re.compile(r"\{\{\s*query:([A-Za-z0-9_.-]+)\s*\}\}")

QUERIES = {
    "list_tables": QUERY_DIRECTORY / "list_tables.sql",
    "run_summary": QUERY_DIRECTORY / "run_summary.sql",
    "conversation_turns": QUERY_DIRECTORY / "conversation_turns.sql",
    "evaluation_components": QUERY_DIRECTORY / "evaluation_components.sql",
}
QUERIES.update({f"{path.parents[1].name}.{path.stem}": path for path in EXTENSION_DIRECTORY.glob("*/sql/*.sql")})


def list_queries() -> tuple[str, ...]:
    """Return the registered query names in a stable order."""

    return tuple(sorted(QUERIES))


def load_query(name: str, _loading: tuple[str, ...] = ()) -> str:
    """Load one registered SQL file for inspection or execution."""

    if name in _loading:
        chain = " -> ".join((*_loading, name))
        raise ValueError(f"Circular SQL query include: {chain}")
    try:
        path = QUERIES[name]
    except KeyError as error:
        available = ", ".join(list_queries())
        raise KeyError(f"Unknown query {name!r}. Available queries: {available}") from error
    sql = path.read_text(encoding="utf-8")

    def include(match: re.Match[str]) -> str:
        included = load_query(match.group(1), (*_loading, name))
        return included.rstrip().removesuffix(";")

    return INCLUDE_PATTERN.sub(include, sql)

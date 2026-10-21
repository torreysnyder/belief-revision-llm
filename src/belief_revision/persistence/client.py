"""Research-friendly read access to project data in Supabase."""

import os
import re
import psycopg
from .queries import list_queries as registered_queries
from .queries import load_query
from .supabase import create_reporting_client
from dotenv import load_dotenv
from psycopg.rows import dict_row
from supabase import Client
from typing import Any

REPORTING_TABLES = (
    "experiment_runs",
    "conversations",
    "conversation_turns",
    "behavioral_probes",
    "experiment_artifacts",
    "activation_captures",
)
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ResearchData:
    """Download reporting tables and run named, read-only SQL queries."""

    def __init__(self, client: Client, database_url: str | None = None) -> None:
        self.client = client
        self.database_url = database_url

    @classmethod
    def from_environment(cls) -> "ResearchData":
        """Build the notebook/script entry point from environment credentials."""

        load_dotenv()
        return cls(
            create_reporting_client(),
            database_url=os.getenv("SUPABASE_DB_URL"),
        )

    def list_tables(self) -> list[str]:
        """List live public tables, or known reporting tables without DB access."""

        if not self.database_url:
            return list(REPORTING_TABLES)
        return [row["table_name"] for row in self.run_query("list_tables")]

    def download(
        self,
        table: str,
        *,
        filters: dict[str, Any] | None = None,
        page_size: int = 1000,
    ) -> list[dict[str, Any]]:
        """Download every matching row from one Data API table."""

        self._validate_identifier(table)
        if page_size < 1:
            raise ValueError("page_size must be positive")

        rows: list[dict[str, Any]] = []
        start = 0
        while True:
            query = self.client.table(table).select("*")
            for column, value in (filters or {}).items():
                self._validate_identifier(column)
                query = query.eq(column, value)
            page = query.range(start, start + page_size - 1).execute().data or []
            rows.extend(page)
            if len(page) < page_size:
                return rows
            start += page_size

    def download_all(
        self,
        *tables: str,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], ...]:
        """Download several tables in the order requested."""

        return tuple(self.download(table, filters=filters) for table in tables)

    def list_queries(self) -> tuple[str, ...]:
        """List named SQL files available to notebooks and scripts."""

        return registered_queries()

    def show_query(self, name: str) -> str:
        """Return the exact SQL behind a named query."""

        return load_query(name)

    def run_query(self, name: str, **parameters: Any) -> list[dict[str, Any]]:
        """Execute a named SQL file inside a read-only Postgres transaction."""

        if not self.database_url:
            raise RuntimeError("Named SQL queries require SUPABASE_DB_URL in .env. Copy a direct or session-pooler connection string from Supabase.")

        sql = load_query(name)
        with psycopg.connect(
            self.database_url,
            sslmode="require",
            prepare_threshold=None,
            row_factory=dict_row,
        ) as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute("set transaction read only")
                    cursor.execute(sql, parameters)
                    return list(cursor.fetchall()) if cursor.description else []

    @staticmethod
    def _validate_identifier(value: str) -> None:
        if not IDENTIFIER.fullmatch(value):
            raise ValueError(f"Invalid table or column name: {value!r}")

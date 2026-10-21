import pytest
from belief_revision.persistence import client as data
from belief_revision.persistence import queries as query_registry
from types import SimpleNamespace


class DataQuery:
    def __init__(self, pages):
        self.pages = pages
        self.filters = []
        self.window = None

    def select(self, columns):
        assert columns == "*"
        return self

    def eq(self, column, value):
        self.filters.append((column, value))
        return self

    def range(self, start, end):
        self.window = (start, end)
        return self

    def execute(self):
        return SimpleNamespace(data=self.pages.pop(0))


class DataClient:
    def __init__(self, pages):
        self.pages = pages
        self.queries = []

    def table(self, name):
        query = DataQuery(self.pages[name])
        self.queries.append((name, query))
        return query


class Cursor:
    def __init__(self, rows, description=True):
        self.rows = rows
        self.description = description
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def execute(self, sql, parameters=None):
        self.executed.append((sql, parameters))

    def fetchall(self):
        return self.rows


class Connection:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def transaction(self):
        return self

    def cursor(self):
        return self._cursor


def test_environment_factory(monkeypatch):
    client = object()
    monkeypatch.setattr(data, "load_dotenv", lambda: None)
    monkeypatch.setattr(data, "create_reporting_client", lambda: client)
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://example")
    access = data.ResearchData.from_environment()
    assert access.client is client
    assert access.database_url == "postgresql://example"


def test_download_and_known_tables():
    client = DataClient(
        {
            "first": [[{"id": 1}], [{"id": 2}], []],
            "second": [[{"id": 3}]],
        }
    )
    access = data.ResearchData(client)
    assert access.list_tables() == list(data.REPORTING_TABLES)
    assert access.download("first", filters={"run_id": "r"}, page_size=1) == [
        {"id": 1},
        {"id": 2},
    ]
    first_query = client.queries[0][1]
    assert first_query.filters == [("run_id", "r")]
    assert first_query.window == (0, 0)
    assert access.download_all("second") == ([{"id": 3}],)


@pytest.mark.parametrize("value", ["bad-name", "a.b", ""])
def test_download_rejects_invalid_identifiers(value):
    access = data.ResearchData(DataClient({}))
    with pytest.raises(ValueError, match="Invalid"):
        access.download(value)


def test_download_rejects_invalid_page_size():
    access = data.ResearchData(DataClient({}))
    with pytest.raises(ValueError, match="positive"):
        access.download("valid", page_size=0)


def test_query_registry():
    names = query_registry.list_queries()
    assert names == tuple(sorted(query_registry.QUERIES))
    assert "information_schema.tables" in query_registry.load_query("list_tables")
    assert "activation_captures" in query_registry.load_query("evaluation_components")
    assert "torrey.custom" in names
    torrey_query = query_registry.load_query("torrey.custom")
    assert "{{ query:" not in torrey_query
    assert "torrey-caa" in torrey_query
    with pytest.raises(KeyError, match="Available queries"):
        query_registry.load_query("missing")


def test_query_registry_rejects_circular_includes(monkeypatch, tmp_path):
    first = tmp_path / "first.sql"
    second = tmp_path / "second.sql"
    first.write_text("{{ query:second }}", encoding="utf-8")
    second.write_text("{{ query:first }}", encoding="utf-8")
    monkeypatch.setattr(query_registry, "QUERIES", {"first": first, "second": second})

    with pytest.raises(ValueError, match="Circular SQL query include"):
        query_registry.load_query("first")


def test_named_queries(monkeypatch):
    cursor = Cursor([{"table_name": "custom"}])
    connection = Connection(cursor)
    calls = []

    def connect(url, **options):
        calls.append((url, options))
        return connection

    monkeypatch.setattr(data.psycopg, "connect", connect)
    access = data.ResearchData(object(), "postgresql://example")
    assert "select table_name" in access.show_query("list_tables")
    assert access.list_queries() == query_registry.list_queries()
    assert access.list_tables() == ["custom"]
    assert calls[0][1]["sslmode"] == "require"
    assert cursor.executed[0] == ("set transaction read only", None)
    assert cursor.executed[1][1] == {}


def test_named_query_requires_database_url():
    with pytest.raises(RuntimeError, match="SUPABASE_DB_URL"):
        data.ResearchData(object()).run_query("run_summary")


def test_named_query_without_rows(monkeypatch):
    cursor = Cursor([], description=False)
    monkeypatch.setattr(
        data.psycopg,
        "connect",
        lambda *args, **kwargs: Connection(cursor),
    )
    assert (
        data.ResearchData(object(), "postgresql://example").run_query(
            "conversation_turns",
            conversation_id="id",
        )
        == []
    )
    assert cursor.executed[-1][1] == {"conversation_id": "id"}

from types import SimpleNamespace

import pytest

from belief_revision import reporting
from belief_revision.models import (
    BehavioralProbeResult,
    ConversationResult,
    DialogueTurn,
)


class Query:
    def __init__(self, table, log):
        self.table = table
        self.log = log

    def insert(self, data):
        self.log.append((self.table, "insert", data))
        return self

    def upsert(self, data, on_conflict):
        self.log.append((self.table, "upsert", data, on_conflict))
        return self

    def update(self, data):
        self.log.append((self.table, "update", data))
        return self

    def eq(self, column, value):
        self.log.append((self.table, "eq", column, value))
        return self

    def execute(self):
        return SimpleNamespace(data=[])


class Client:
    def __init__(self):
        self.log = []

    def table(self, name):
        return Query(name, self.log)


def result(primary_cell, run_plan, with_children=True):
    return ConversationResult(
        primary_cell,
        run_plan,
        [
            DialogueTurn("user", "intro", 1, 1, "u"),
            DialogueTurn("assistant", "intro", 1, 2, "a"),
        ]
        if with_children
        else [],
        [BehavioralProbeResult("intro")] if with_children else [],
        [{"global_turn_index": 2, "value": 2}] if with_children else [],
        {},
    )


def test_client_configuration(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SECRET_KEY", raising=False)
    monkeypatch.setattr(reporting, "load_dotenv", lambda: None)
    with pytest.raises(RuntimeError, match="Supabase access"):
        reporting.create_reporting_client()
    monkeypatch.setenv("SUPABASE_URL", "url")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "secret")
    client = object()
    monkeypatch.setattr(reporting, "create_client", lambda url, key: client)
    assert reporting.create_reporting_client() is client
    monkeypatch.setattr(reporting, "create_reporting_client", lambda: client)
    assert reporting.SupabaseReporter.from_environment().client is client
    assert "+00:00" in reporting.utc_now()


def test_reporting_lifecycle(primary_cell, run_plan):
    client = Client()
    reporter = reporting.SupabaseReporter(client)
    run_id = reporter.start_run(
        target_model="t", user_sim_model="u", evaluator_model="e", config={}
    )
    reporter.save_conversations(run_id, [result(primary_cell, run_plan)])
    reporter.save_conversations(run_id, [result(primary_cell, run_plan, False)])
    reporter.complete_run(run_id, completed_conversations=1, expected_conversations=2)
    reporter.fail_run(run_id, RuntimeError("boom"))
    operations = [(entry[0], entry[1]) for entry in client.log]
    assert ("experiment_runs", "insert") in operations
    assert ("conversation_turns", "upsert") in operations
    assert ("behavioral_probes", "upsert") in operations
    assert operations.count(("conversations", "update")) == 2

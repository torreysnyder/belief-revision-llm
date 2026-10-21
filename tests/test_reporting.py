import pytest
from belief_revision.core.models import (
    ActivationArtifactRecord,
    BehavioralProbeResult,
    ConversationResult,
    DialogueTurn,
    ExperimentArtifact,
)
from belief_revision.persistence import supabase as reporting
from types import SimpleNamespace


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
        self.storage = Storage(self.log)

    def table(self, name):
        return Query(name, self.log)


class StorageBucket:
    def __init__(self, bucket, log):
        self.bucket = bucket
        self.log = log

    def upload(self, *, file, path, file_options):
        self.log.append(("storage", "upload", self.bucket, path, file.read(), file_options))


class Storage:
    def __init__(self, log):
        self.log = log

    def from_(self, bucket):
        return StorageBucket(bucket, self.log)


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
    run_id = reporter.start_run(target_model="t", user_sim_model="u", evaluator_model="e", config={})
    reporter.save_conversations(run_id, [result(primary_cell, run_plan)])
    reporter.save_conversations(run_id, [result(primary_cell, run_plan, False)])
    reporter.complete_run(run_id, completed_conversations=1, expected_conversations=2)
    reporter.fail_run(run_id, RuntimeError("boom"))
    operations = [(entry[0], entry[1]) for entry in client.log]
    assert ("experiment_runs", "insert") in operations
    assert ("conversation_turns", "upsert") in operations
    assert ("behavioral_probes", "upsert") in operations
    assert operations.count(("conversations", "update")) == 2


def test_reporting_uploads_activation_artifact(tmp_path, primary_cell, run_plan):
    artifact_path = tmp_path / "capture.pt"
    artifact_path.write_bytes(b"tensor-data")
    conversation = result(primary_cell, run_plan)
    conversation.artifacts.append(
        ExperimentArtifact(
            artifact_type="live_conversation_activations",
            local_path=artifact_path,
            metadata={"extension": "torrey-caa"},
            activation_records=(
                ActivationArtifactRecord(
                    global_turn_index=2,
                    phase="intro",
                    turn_in_phase=1,
                    model_name="model",
                    location="residual_post",
                    token_position="final_prompt_token",
                    layers=(1, 3),
                    token_indices=(7, 7),
                    sequence_lengths=(8, 8),
                    tensor_shapes=((2,), (2,)),
                    tensor_dtypes=("torch.float32", "torch.float32"),
                ),
            ),
        )
    )
    client = Client()
    reporting.SupabaseReporter(client).save_conversations(
        "00000000-0000-0000-0000-000000000001",
        [conversation],
    )

    assert any(entry[:3] == ("storage", "upload", "experiment-artifacts") for entry in client.log)
    artifact_upsert = next(entry for entry in client.log if entry[:2] == ("experiment_artifacts", "upsert"))
    assert artifact_upsert[2]["checksum"]
    capture_upsert = next(entry for entry in client.log if entry[:2] == ("activation_captures", "upsert"))
    assert [row["layer_index"] for row in capture_upsert[2]] == [1, 3]


def test_reporting_uploads_artifact_without_activation_rows(tmp_path):
    artifact_path = tmp_path / "notes.json"
    artifact_path.write_text("{}", encoding="utf-8")
    client = Client()

    reporting.SupabaseReporter(client)._save_artifact(
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
        ExperimentArtifact("analysis_notes", artifact_path),
    )

    assert any(entry[:2] == ("experiment_artifacts", "upsert") for entry in client.log)
    assert not any(entry[:2] == ("activation_captures", "upsert") for entry in client.log)

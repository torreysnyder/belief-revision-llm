from belief_revision.core.models import ConversationResult
from belief_revision.persistence import contracts as backends


def test_local_file_backend_lifecycle(monkeypatch, tmp_path, primary_cell, run_plan):
    csv_path = tmp_path / "results.csv"
    jsonl_path = tmp_path / "results.jsonl"
    csv_path.write_text("old", encoding="utf-8")
    writes = []
    monkeypatch.setattr(
        backends,
        "save_results_append",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )

    backend = backends.LocalFileBackend(csv_path, jsonl_path)
    assert backend.start_run(target_model="t") == "local"
    assert not csv_path.exists()
    assert jsonl_path.read_text(encoding="utf-8") == ""
    assert backend.start_run() == "local"

    result = ConversationResult(primary_cell, run_plan, [], [], [], {})
    backend.save_conversations("local", [])
    assert writes[-1][1]["write_header"] is True
    backend.save_conversations("local", [result])
    assert writes[-1][1]["write_header"] is True
    backend.save_conversations("local", [result])
    assert writes[-1][1]["write_header"] is False

    backend.complete_run("local", completed_conversations=2)
    backend.fail_run("local", RuntimeError("kept locally"))

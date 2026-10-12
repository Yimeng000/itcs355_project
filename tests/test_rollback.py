import json
import sys

import pytest

from cloudlayer import rollback


def service_state(revision="taxi-fare-staging-00002-tf9"):
    return {
        "status": {
            "url": "https://example.invalid",
            "traffic": [{"revisionName": revision, "percent": 100}],
        }
    }


def test_snapshot_records_concrete_revision(tmp_path, monkeypatch):
    path = tmp_path / "before.json"
    monkeypatch.setattr(rollback, "describe", lambda *args: service_state())
    monkeypatch.setattr(sys, "argv", [
        "rollback", "snapshot",
        "--service", "taxi-fare-staging",
        "--project", "test-project",
        "--region", "test-region",
        "--snapshot", str(path),
    ])
    rollback.main()
    report = json.loads(path.read_text())
    assert report["traffic"] == {"taxi-fare-staging-00002-tf9": 100}


@pytest.mark.parametrize("traffic", [
    [],
    [{"revisionName": "LATEST", "percent": 100}],
    [{"revisionName": "revision", "percent": 50}],
    [{"revisionName": "revision", "percent": -1}],
    [{"revisionName": "revision", "percent": True}],
])
def test_invalid_traffic_is_rejected(traffic):
    with pytest.raises(ValueError):
        rollback.traffic_allocation({"status": {"traffic": traffic}})


@pytest.fixture
def restore_case(tmp_path, monkeypatch):
    snapshot = tmp_path / "before.json"
    report = tmp_path / "restored.json"
    snapshot.write_text(json.dumps({
        "service": "taxi-fare-staging",
        "project": "test-project",
        "region": "test-region",
        "traffic": {"taxi-fare-staging-00002-tf9": 100},
    }))
    monkeypatch.setattr(sys, "argv", [
        "rollback", "restore",
        "--service", "taxi-fare-staging",
        "--project", "test-project",
        "--region", "test-region",
        "--snapshot", str(snapshot),
        "--report", str(report),
    ])
    commands = []
    monkeypatch.setattr(
        rollback.subprocess, "run",
        lambda command, **kwargs: commands.append(command),
    )
    return snapshot, report, commands


def test_restore_updates_and_verifies_traffic(restore_case, monkeypatch):
    _, report, commands = restore_case
    monkeypatch.setattr(rollback, "describe", lambda *args: service_state())
    rollback.main()
    assert "--to-revisions" in commands[0]
    assert "taxi-fare-staging-00002-tf9=100" in commands[0]
    assert json.loads(report.read_text())["traffic_restored"] is True


def test_wrong_service_snapshot_is_rejected(restore_case):
    snapshot, _, commands = restore_case
    data = json.loads(snapshot.read_text())
    data["service"] = "taxi-fare-api"
    snapshot.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="different service"):
        rollback.main()
    assert commands == []


def test_failed_restore_verification_is_reported(restore_case, monkeypatch):
    _, report, _ = restore_case
    monkeypatch.setattr(
        rollback, "describe",
        lambda *args: service_state("taxi-fare-staging-00003-new"),
    )
    with pytest.raises(SystemExit):
        rollback.main()
    assert json.loads(report.read_text())["traffic_restored"] is False

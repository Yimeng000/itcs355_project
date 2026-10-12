import hashlib
import json
import sys

import pytest

from src import register


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def registration_files(tmp_path, monkeypatch):
    root = tmp_path
    for folder in ("models", "reports", "data/processed"):
        (root / folder).mkdir(parents=True)

    contents = {
        "models/model.joblib": b"test model artifact",
        "models/input_schema.json": b"{}",
        "data/processed/train.csv": b"x,y\n1,2\n",
        "data/processed/test.csv": b"x,y\n3,4\n",
        "params.yaml": b"paths: {}\n",
        "dvc.lock": b"schema: '2.0'\n",
    }
    for name, content in contents.items():
        (root / name).write_bytes(content)

    metrics = {"mae": 11.0, "rmse": 18.0, "r2": 0.88}
    training = {
        "training_run_id": "test-run",
        "training_git_commit": "test-commit",
        "data_fingerprint": sha256(root / "data/processed/train.csv")[:16],
        "model_sha256": sha256(root / "models/model.joblib"),
        "model_type": "random_forest",
        "seed": 42,
        "metrics": {f"test_{key}": value for key, value in metrics.items()},
    }
    evaluation = {
        "candidate_metrics": metrics,
        "quality_gate": {"passed": True, "checks": {"example_check": True}},
        "artifact_fingerprints": {
            "model_sha256": sha256(root / "models/model.joblib"),
            "train_sha256": sha256(root / "data/processed/train.csv"),
            "test_sha256": sha256(root / "data/processed/test.csv"),
            "params_sha256": sha256(root / "params.yaml"),
        },
    }
    write_json(root / "reports/train_metrics.json", training)
    write_json(root / "reports/metrics.json", evaluation)

    monkeypatch.setattr(register.config, "REPO_ROOT", root)
    monkeypatch.setattr(sys, "argv", ["register", "--dry-run"])
    return root


def test_matching_reports_pass(registration_files):
    register.main()
    manifest = json.loads(
        (registration_files / "reports/model_lineage.json").read_text()
    )
    assert manifest["quality_gate_passed"] is True
    assert manifest["training_run_id"] == "test-run"
    assert not (registration_files / "reports/registration.json").exists()


@pytest.mark.parametrize(
    "filename",
    [
        "models/model.joblib",
        "data/processed/train.csv",
        "data/processed/test.csv",
        "params.yaml",
    ],
)
def test_changed_files_are_rejected(registration_files, filename):
    path = registration_files / filename
    path.write_bytes(path.read_bytes() + b"\n# changed\n")

    with pytest.raises(RuntimeError):
        register.main()

    assert not (registration_files / "reports/model_lineage.json").exists()


def test_missing_fingerprints_are_rejected(registration_files):
    path = registration_files / "reports/metrics.json"
    report = json.loads(path.read_text())
    report.pop("artifact_fingerprints")
    write_json(path, report)

    with pytest.raises(RuntimeError, match="missing or outdated"):
        register.main()


@pytest.mark.parametrize(
    "gate",
    [
        {"passed": False, "checks": {"example_check": False}},
        {"passed": True, "checks": {"example_check": False}},
        {"passed": True, "checks": {}},
    ],
)
def test_failed_or_inconsistent_gate_is_rejected(registration_files, gate):
    path = registration_files / "reports/metrics.json"
    report = json.loads(path.read_text())
    report["quality_gate"] = gate
    write_json(path, report)

    with pytest.raises(RuntimeError, match="Evaluation failed"):
        register.main()


def test_mismatched_metrics_are_rejected(registration_files):
    path = registration_files / "reports/metrics.json"
    report = json.loads(path.read_text())
    report["candidate_metrics"]["mae"] = 12.0
    write_json(path, report)

    with pytest.raises(RuntimeError, match="reports do not match"):
        register.main()

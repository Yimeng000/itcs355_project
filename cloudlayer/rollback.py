"""Snapshot and restore a Cloud Run service's traffic allocation."""
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def describe(service, project, region):
    result = subprocess.run(
        [
            "gcloud", "run", "services", "describe", service,
            "--project", project, "--region", region, "--format=json",
        ],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def traffic_allocation(service):
    allocation = {}
    for item in service["status"].get("traffic", []):
        percent = item.get("percent", 0)
        if not isinstance(percent, int) or isinstance(percent, bool):
            raise ValueError("Invalid traffic percentage")
        if percent < 0 or percent > 100:
            raise ValueError("Invalid traffic percentage")
        if percent:
            revision = item.get("revisionName")
            if not revision or revision == "LATEST":
                raise ValueError("A concrete revision name is required")
            allocation[revision] = allocation.get(revision, 0) + percent
    if sum(allocation.values()) != 100:
        raise ValueError("Traffic percentages must sum to 100")
    return allocation


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["snapshot", "restore"])
    parser.add_argument("--service", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    identity = {
        "service": args.service,
        "project": args.project,
        "region": args.region,
    }

    if args.action == "snapshot":
        current = describe(args.service, args.project, args.region)
        save(args.snapshot, {
            **identity,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "traffic": traffic_allocation(current),
            "service_url": current["status"]["url"],
        })
        return

    if args.report is None:
        parser.error("Restore requires --report")

    snapshot = json.loads(args.snapshot.read_text())
    for key, value in identity.items():
        if snapshot.get(key) != value:
            raise ValueError(f"Snapshot belongs to a different {key}")

    allocation = traffic_allocation({
        "status": {"traffic": [
            {"revisionName": revision, "percent": percent}
            for revision, percent in snapshot["traffic"].items()
        ]}
    })
    if any(not revision.startswith(args.service + "-") for revision in allocation):
        raise ValueError("Revision does not belong to this service")

    started = time.perf_counter()
    subprocess.run(
        [
            "gcloud", "run", "services", "update-traffic", args.service,
            "--project", args.project, "--region", args.region,
            "--to-revisions",
            ",".join(f"{revision}={percent}" for revision, percent in allocation.items()),
            "--quiet",
        ],
        check=True,
    )
    current = describe(args.service, args.project, args.region)
    observed = traffic_allocation(current)
    verified = observed == allocation
    save(args.report, {
        **identity,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "expected_traffic": allocation,
        "observed_traffic": observed,
        "traffic_restored": verified,
        "restore_and_verification_seconds": round(time.perf_counter() - started, 2),
        "scope": "Traffic restoration only; API health must be checked separately.",
    })
    if not verified:
        raise SystemExit("Traffic restoration verification failed")


if __name__ == "__main__":
    main()

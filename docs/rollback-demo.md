# Automatic Rollback Demonstration

## Scenario

A manually triggered workflow changed the staging model-version label
to rollback-demo. The model artifact itself was not changed.
The release check expected version 2 and deliberately failed.

This simulates a release validation failure, not a model outage.

## Observed Recovery

GitHub Actions automatically restored 100% of staging traffic to
taxi-fare-staging-00008-zlp.

Traffic restoration and verification took 9.63 seconds.
This excludes failure detection and the subsequent API checks.

After restoration:
- Readiness passed with model version 2.
- Prediction passed with estimated fare 37.2 and model version 2.

The workflow remained failed to preserve visibility of the rejected release.

## Evidence

- Workflow run: https://github.com/Yimeng000/itcs355_project/actions/runs/38401531566
- Traffic snapshot: ../reports/rollback/staging-auto-before.json
- Restoration: ../reports/rollback/staging-auto-restored.json
- API verification: ../reports/rollback/staging-auto-api-restored.json
- Manual drill workflow: ../.github/workflows/rollback-drill.yml
- Production workflow: ../.github/workflows/ci.yml

## Reproduce

While the staging service is available, open GitHub Actions,
select Staging rollback drill, and run it on main.

Expected results:
- Version mismatch check: failed.
- Automatic traffic restoration: passed.
- Restored API verification: passed.

Wait for other staging deployments to finish before starting the drill.

## Scope and Limitations

Automatic recovery was exercised against staging.
Production contains the same recovery logic, but an intentional
production failure was not performed.

Rollback restores traffic to saved revisions. It does not revert
repository commits, delete failed revisions or migrate external data.

Recovery depends on available revisions, valid cloud credentials
and a working Cloud Run control plane. Failed recovery requires
manual investigation.

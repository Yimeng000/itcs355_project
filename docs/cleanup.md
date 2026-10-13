# Cloud Resource Cleanup

Date: 13 October 2026
Project: itcs355-6688176
Region checked: asia-southeast1

## Completed actions

- Deleted the capstone Cloud Run services taxi-fare-api and taxi-fare-staging.
- Cloud Run service listing returned an empty array.
- Vertex AI endpoint listing returned an empty array.
- Disabled the Project CI and CD and Staging rollback drill workflows.

The historical cloud API URLs are no longer available.
Local training and inference remain available through the README instructions.

## Evidence

- [Cloud Run services](../reports/cleanup/cloud-run-after.json)
- [Vertex AI endpoints](../reports/cleanup/vertex-endpoints.json)
- [Scheduler state](../reports/cleanup/scheduler-jobs.json)

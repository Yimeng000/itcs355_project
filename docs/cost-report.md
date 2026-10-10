# Taxi Fare Prediction — Cost Report

## 1. Result

The estimated base Cloud Run serving cost is **USD 0.003008 per
1,000 successful predictions**, before free-tier allowances.

At an assumed exchange rate of 36 THB per USD, this is approximately
**THB 0.1083 per 1,000 predictions**.

This is a usage-based estimate, not an actual billing statement.

## 2. Deployment configuration

- Service: taxi-fare-api
- Region: Singapore (asia-southeast1)
- Billing assumption: request-based billing
- CPU: 1 vCPU
- Memory: 1 GiB
- Container concurrency: 4
- Revision maximum instances: 1

## 3. Test and usage evidence

The test sent 1,000 requests on 9 October 2026.
All 1,000 returned HTTP 200; no requests failed.

Test timestamps (UTC):
- Start: 2026-10-09T16:38:54.319736+00:00
- End: 2026-10-09T16:40:04.467863+00:00

Client-side latency:
- p50: 69.6 ms
- p95: 81.4 ms
- p99: 95.9 ms

These latency measurements include network time from Cloud Shell.

Cloud Monitoring was inspected around the test window, using:
- Metric: run.googleapis.com/container/billable_instance_time
- Service filter: taxi-fare-api
- Per-series alignment: ALIGN_SUM
- Cross-series reduction: REDUCE_SUM
- Alignment period: 60 seconds

The observed nonzero points were:

| Timestamp (UTC+7) | Billable instance time |
|---|---:|
| 23:39 | 1.6 seconds |
| 23:40 | 35.4 seconds |
| 23:41 | 33.3 seconds |
| Total | 70.3 seconds |

No value was available at 23:38; it was not included.
This total covers the observed monitoring intervals around the test.
Other activity within those intervals was not independently excluded.

## 4. Unit prices and calculation

Singapore request-based default USD prices were checked on
10 October 2026 at https://cloud.google.com/run/pricing.

| Resource | Unit price (USD) |
|---|---:|
| Active CPU | 0.0000336 per vCPU-second |
| Active memory | 0.0000035 per GiB-second |
| Requests | 0.40 per 1,000,000 requests |

Calculation:

- CPU: 70.3 × 1 × 0.0000336 = USD 0.00236208
- Memory: 70.3 × 1 × 0.0000035 = USD 0.00024605
- Requests: 1,000 / 1,000,000 × 0.40 = USD 0.0004
- Total: USD 0.00300813, rounded to USD 0.003008

The calculation output is saved in reports/cost_estimate.json.
THB conversion uses an assumed exchange rate, not a verified daily rate.

## 5. Scope and limitations

The estimate includes base CPU, memory and request charges.
It does not deduct monthly free-tier allowances or trial credits.

Startup CPU boost is enabled. Additional CPU allocated during startup
is not separately measured in this calculation, so the estimate may
understate compute charges if startup occurred during the test.

Cloud Storage, Artifact Registry, network transfer, chargeable logging
or monitoring, training and CI/CD costs are outside this calculation.
They must not be described as zero without billing evidence.

The estimate is specific to this test workload and configuration.
Actual charges and cost per prediction can change with traffic,
cold starts, concurrency and deployment settings.

## 6. Cost controls

The serving revision is limited to one instance.
Resource sizes and concurrency are explicitly configured.
Billing should be reviewed for both capstone and remaining lab resources.

After the required demonstrations and submission, project resources
should be cleaned up according to the course requirements.

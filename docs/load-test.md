# Tenfold Traffic Test

## Method

The same deployed API, model version 2 and valid payload were tested
at 1 request/second and 10 requests/second, each scheduled for 30 seconds.
A readiness request warmed the service before each measured workload.
Latency was measured from the client and includes network time.

## Results

| Metric | Baseline | Tenfold traffic |
|---|---:|---:|
| Achieved requests/second | 1.0 | 10.0 |
| Requests sent | 30 | 300 |
| Valid predictions | 30 | 300 |
| Failed requests | 0 | 0 |
| Client capacity skips | 0 | 0 |
| p50 latency, ms | 187.93 | 179.95 |
| p95 latency, ms | 213.72 | 243.58 |
| p99 latency, ms | 220.97 | 548.42 |

P95 increased by approximately 14%.
No service failure was observed at the tested traffic level.

## Interpretation and Limitations

The slower requests took longer under tenfold traffic.
This does not establish the service's maximum capacity or identify
a proven bottleneck.

The tests were short, had different sample sizes and were not repeated.
Network variation may affect the results. The baseline p99 is based
on only 30 requests and should be interpreted cautiously.

At higher load, we would monitor request latency and errors.
The current revision's instance limit and concurrency setting are
potential capacity constraints, but the failure point remains unmeasured.

## Reproduce

Set TAXI_URL and TAXI_TOKEN for the authenticated cloud API, then run:

    python scripts/load_test.py --rate 1 --duration 30 --output reports/load-baseline.json
    python scripts/load_test.py --rate 10 --duration 30 --output reports/load-10x.json

The script expects model version 2.

## Evidence

- Baseline: ../reports/load-baseline.json
- Tenfold traffic: ../reports/load-10x.json
- Script: ../scripts/load_test.py

Test timestamps are recorded in UTC in the JSON reports.

# Deliberate Failure: Invalid Input Burst

## Scenario
A client sends 20 prediction requests with a negative trip distance.
The service must reject these requests and continue serving valid input.

## Demonstration — 9 October 2026
- Test started at 13:10:04 UTC (20:10:04, UTC+7).
- All 20 invalid requests returned HTTP 422.
- Monitoring recorded 20 HTTP 422 responses, exceeding the
  alert threshold of 5 in a 60-second window.
- An alert email was received.
- A subsequent valid request returned HTTP 200:
  {"estimated_fare":37.2,"model_version":"2"}
- The alert incident subsequently showed Closed.

## What We Learned
Invalid input can trigger an operational alert without crashing the service.
HTTP 422 indicates rejected client input; it does not mean the model failed.
The alert threshold is intended for this small demonstration workload
and should be reviewed for a larger workload.

## Feedback Into Tests
Added test_invalid_input_burst_does_not_break_predictions
in tests/test_service.py.

The test verifies:
- All 20 invalid requests are rejected with HTTP 422.
- Error details identify Trip_Distance_km.
- Invalid requests do not reach the model.
- Health and readiness remain HTTP 200.
- A subsequent valid request returns the expected prediction.

Verification: 44 tests passed.

## Monitoring Configuration
- cloudlayer/monitoring/dashboard.json
- cloudlayer/monitoring/invalid-input-alert.json

Cloud alert incident:
projects/itcs355-6688176/alertPolicies/16399261076497637481

Screenshots of the alert email, monitoring graph, and Closed incident
should accompany the presentation evidence.

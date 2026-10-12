"""Send requests at a controlled rate and save client-side measurements."""
import argparse
import json
import math
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PAYLOAD = {
    "Trip_Distance_km": 10.0,
    "Passenger_Count": 2,
    "Base_Fare": 3.0,
    "Per_Km_Rate": 1.5,
    "Per_Minute_Rate": 0.3,
    "Time_of_Day": "Morning",
    "Day_of_Week": "Weekday",
    "Traffic_Conditions": "Low",
    "Weather": "Clear",
}


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return round(
        ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower), 2
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.rate) or args.rate <= 0 or args.duration <= 0:
        parser.error("Rate and duration must be positive")

    url = os.environ["TAXI_URL"].rstrip("/")
    token = os.environ["TAXI_TOKEN"]
    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
    }

    # Check access and warm the service before the measured workload.
    with urlopen(Request(url + "/ready", headers=headers), timeout=60) as response:
        if response.status != 200:
            raise RuntimeError("Service is not ready")

    data = json.dumps(PAYLOAD).encode()

    def send():
        started = time.perf_counter()
        status = "client_error"
        valid_prediction = False
        try:
            request = Request(url + "/predict", data=data, headers=headers)
            with urlopen(request, timeout=30) as response:
                status = response.status
                result = json.loads(response.read())
                fare = result.get("estimated_fare")
                valid_prediction = (
                    status == 200
                    and isinstance(fare, (int, float))
                    and not isinstance(fare, bool)
                    and math.isfinite(fare)
                    and fare >= 0
                    and str(result.get("model_version")) == "2"
                )
        except HTTPError as error:
            status = error.code
            error.close()
        except (URLError, TimeoutError, OSError, ValueError):
            pass
        return {
            "status": status,
            "valid_prediction": valid_prediction,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "started": started,
        }

    count = max(1, int(args.rate * args.duration))
    start_utc = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    futures = []
    pending = set()
    skipped = 0

    with ThreadPoolExecutor(max_workers=32) as executor:
        for index in range(count):
            due = start + index / args.rate
            remaining = due - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            pending = {future for future in pending if not future.done()}
            if len(pending) >= 32:
                skipped += 1
                continue
            future = executor.submit(send)
            futures.append(future)
            pending.add(future)

        results = [future.result() for future in futures]

    elapsed = time.perf_counter() - start
    successes = [row for row in results if row["valid_prediction"]]
    latencies = [row["latency_ms"] for row in successes]
    send_times = [row["started"] for row in results]
    achieved_rate = (
        (len(send_times) - 1) / (max(send_times) - min(send_times))
        if len(send_times) > 1 else None
    )
    report = {
        "start_utc": start_utc,
        "end_utc": datetime.now(timezone.utc).isoformat(),
        "target_requests_per_second": args.rate,
        "scheduled_duration_seconds": args.duration,
        "planned_requests": count,
        "requests_sent": len(results),
        "client_capacity_skips": skipped,
        "valid_predictions": len(successes),
        "failed_requests": len(results) - len(successes),
        "http_status_counts": dict(Counter(str(row["status"]) for row in results)),
        "achieved_send_rate_per_second": (
            round(achieved_rate, 2) if achieved_rate is not None else None
        ),
        "total_elapsed_seconds": round(elapsed, 2),
        "successful_request_latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
        "measurement_scope": (
            "Client-side latency includes network time. Readiness warmup is excluded. "
            "Client capacity skips are not server errors. Expected model version: 2."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

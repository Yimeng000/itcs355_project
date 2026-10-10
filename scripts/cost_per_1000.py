"""Cost per 1,000 predictions for the Cloud Run service (1 vCPU, 1 GiB, request-based billing).

You supply measured values; nothing here is assumed.
  --requests          predictions sent in the test window (from reports/load_test.json)
  --billable-seconds  total billable instance time in the same window
                      (Monitoring metric run.googleapis.com/container/billable_instance_time, sum)
  --billed-usd        (optional) actual Cloud Run cost for that window from the Billing report
Unit prices: the defaults below are the Tier 1 list rates seen on the Cloud Run pricing page.
asia-southeast1 is a Tier 2 region, so REPLACE them with the Singapore SKU prices from
https://cloud.google.com/skus (service "Cloud Run") and note the date you checked.
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--requests", type=int, required=True)
ap.add_argument("--billable-seconds", type=float, required=True)
ap.add_argument("--billed-usd", type=float)
ap.add_argument("--vcpu", type=float, default=1.0)
ap.add_argument("--gib", type=float, default=1.0)
ap.add_argument("--price-vcpu-s", type=float, default=0.000024)
ap.add_argument("--price-gib-s", type=float, default=0.0000025)
ap.add_argument("--price-per-million-req", type=float, default=0.40)
ap.add_argument("--thb-per-usd", type=float, required=True)
a = ap.parse_args()

cpu = a.billable_seconds * a.vcpu * a.price_vcpu_s
mem = a.billable_seconds * a.gib * a.price_gib_s
req = a.requests / 1e6 * a.price_per_million_req
total = cpu + mem + req
out = {
    "requests": a.requests, "billable_seconds": a.billable_seconds,
    "list_price_usd": {"cpu": round(cpu, 6), "memory": round(mem, 6),
                       "requests": round(req, 6), "total": round(total, 6)},
    "list_price_per_1000_usd": round(total / a.requests * 1000, 6),
    "list_price_per_1000_thb": round(total / a.requests * 1000 * a.thb_per_usd, 4),
    "note": "List price, before the monthly free tier.",
}
if a.billed_usd is not None:
    out["actual_billed_usd"] = a.billed_usd
    out["actual_per_1000_usd"] = round(a.billed_usd / a.requests * 1000, 6)
print(json.dumps(out, indent=2))

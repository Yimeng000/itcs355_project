# Automated MLOps Pipeline for Taxi Price Prediction

ITCS355 capstone: An educational taxi fare prediction service with versioned data, reproducible training, registered model lineage, authenticated inference, automated release checks and monitoring. The dataset is synthetic; this service does not provide validated real-world taxi quotes.


## 1. Architecture and project scope

| Layer | Responsibility | Files |
|---|---|---|
| ML logic | Data checks, cleaning, splitting, training, evaluation and model comparison | `src/` |
| Serving | Input validation, model loading, readiness and fare prediction | `service/` |
| Cloud integration | Storage, registration, deployment, traffic restoration and monitoring configuration | `cloudlayer/` |

Git versions code. DVC versions data and model artifacts. MLflow records training runs. Vertex AI registers models; Cloud Run serves them. GitHub Actions verifies training and releases an immutable serving image through staging before production.

`make portability-audit` checks the layer boundaries. Only GCP integrations are implemented; the audit is not proof of tested Azure or AWS support.

The service predicts one trip at a time from nine caller-supplied inputs. Route planning, live weather/traffic retrieval, taxi booking and payments are outside scope.

## 2. Data and model

Source: [Taxi Price Prediction / Regression, denkuznetz](https://www.kaggle.com/datasets/denkuznetz/taxi-price-prediction). The publisher identifies **synthetic data**, fares in **USD**, and an **Apache 2.0** licence.

| Item | Recorded value |
|---|---|
| Raw file | `data/taxi_trip_pricing.csv`: 1,000 rows, 11 columns |
| Target | `Trip_Price` |
| Cleaning | Remove 49 missing targets; retain 951 records |
| Split | Seed 42; 665 train / 143 validation / 143 test |
| Excluded field | Actual `Trip_Duration_Minutes`, unavailable before the trip ends |
| Model | Preprocessing pipeline + Random Forest Regressor; 150 trees, maximum depth 10 |
| Registered version | Vertex AI model `4463872073235693568`, version `2` |
| Test results | MAE 11.4973 USD; RMSE 18.2893 USD; R² 0.8854 |

Training preprocessing imputes missing inputs and encodes categories. The API requires complete inputs and lowercases accepted categorical values to match training.

**Training checks:** at least 100 raw records; each required column at most 20% missing; known categories; finite, parseable and valid numeric values. These are project-defined initial thresholds, not universal standards. Failure saves a report and stops the DVC pipeline before cleaning. See [data quality report](reports/data_quality.json).

**Model checks:** compare against a mean-prediction baseline; enforce configured accuracy and local latency limits; bind evaluation reports to model, data and parameter fingerprints. A separate validation comparison rejects candidates worse than the registered reference. Registration verifies the comparison's fingerprints, run and reference version. These checks validate records and results, not real-world suitability. See [retraining evidence](docs/retraining-check.md).

## 3. Install and run checks

Use **Python 3.11**, Git and GNU Make. Windows commands below run in WSL. Docker is needed for containers; Google Cloud CLI and authorized credentials are needed for private cloud operations.

```bash
git clone https://github.com/Yimeng000/itcs355_project.git
cd itcs355_project
python3.11 -m venv .venv-model
source .venv-model/bin/activate
python -m pip install -r requirements-dev.txt
make check
```

If your Python 3.11 executable is named `python`, use that executable to create the environment. Confirm with `python --version`. Do not use a shallow clone: the credential scan needs full Git history.

`make check` runs the portability audit, history credential scan and tests. Recorded total: **97 passed**. Tests use temporary data/models and mocked cloud operations; they need neither private dataset access nor cloud credentials.

To train, also install:

```bash
python -m pip install -r requirements-training.txt
```

## 4. Reproduce training and run locally

The DVC remote is private: `gs://itcs355-6688176/itcs355_project/dvc-cache`. Cloning the repository does not grant access. An authorized account needs Storage Object Viewer on the bucket.

```bash
gcloud auth application-default login
python -m dvc pull data/taxi_trip_pricing.csv.dvc
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
python -m dvc repro --force
```

Stages: **data quality → clean → split → train → evaluate**. `make train` runs the same pipeline without `--force`, so unchanged stages may be skipped. Training does not register or deploy a model.

Without bucket access, obtain the CSV from Kaggle and place it at the raw path. Verify its MD5 against `data/taxi_trip_pricing.csv.dvc` before claiming the recorded results:

```bash
python - <<'PY'
import hashlib
from pathlib import Path
import yaml
path = Path("data/taxi_trip_pricing.csv")
expected = yaml.safe_load(Path(str(path) + ".dvc").read_text())["outs"][0]["md5"]
actual = hashlib.md5(path.read_bytes()).hexdigest()
assert actual == expected, (actual, expected)
print("Raw dataset version matches")
PY
```

Generated outputs include `models/model.joblib`, `models/input_schema.json`, processed CSVs, `reports/data_quality.json`, `reports/train_metrics.json` and `reports/metrics.json`. MLflow's local database is generated by training; it is not included in a clone. View new runs with `mlflow ui --host 127.0.0.1 --port 5000`.

Run the locally trained API:

```bash
MODEL_PATH=models/model.joblib MODEL_VERSION=local \
  python -m uvicorn service.app:app --host 127.0.0.1 --port 8080
```

Open [local API documentation](http://127.0.0.1:8080/docs). Local requests do not populate the Cloud Run dashboard. `GET /` returns 404; there is no home route.

Alternatively, stop the local server and run Docker:

```bash
docker build -f service/Dockerfile.serve -t taxi-fare:local .
docker run --rm -p 8080:8080 \
  -v "$PWD/models:/app/models:ro" \
  -e MODEL_VERSION=local taxi-fare:local
```

The image contains application code, not the model. With `MODEL_ARTIFACT_URI` set, it downloads the selected cloud artifact at startup. Local cloud-mode testing also needs credentials and `GOOGLE_CLOUD_PROJECT`; Cloud Run uses its runtime service account.

## 5. API contract and cloud prediction

All nine fields are required. Extra fields and non-finite numeric values are rejected.

| Field | Accepted value |
|---|---|
| `Trip_Distance_km` | Number > 0 |
| `Passenger_Count` | Strict integer 1–4 |
| `Base_Fare` | Number ≥ 0 |
| `Per_Km_Rate` | Number > 0 |
| `Per_Minute_Rate` | Number ≥ 0 |
| `Time_of_Day` | Morning / Afternoon / Evening / Night |
| `Day_of_Week` | Weekday / Weekend |
| `Traffic_Conditions` | Low / Medium / High |
| `Weather` | Clear / Rain / Snow |

| Route | Meaning |
|---|---|
| `GET /health` | 200: process running; does not guarantee model availability |
| `GET /ready` | 200: model loaded; 503: unavailable |
| `POST /predict` | 200: fare/version; 422: invalid input; 503: no model; 500: prediction error |

Production: `taxi-fare-api`. Staging: `taxi-fare-staging`. Both require authentication and are in project `itcs355-6688176`, region `asia-southeast1`. Serving uses 1 CPU, 1 GiB and concurrency 4; the configured revision instance limit is 1. Production's separately recorded service-level maximum is 20.

For an account granted Cloud Run Invoker, run these commands in the same terminal:

```bash
gcloud auth login
export TAXI_URL="https://taxi-fare-api-259177885839.asia-southeast1.run.app"
export TAXI_TOKEN="$(gcloud auth print-identity-token)"

curl -sS --max-time 60 "$TAXI_URL/ready" \
  -H "Authorization: Bearer $TAXI_TOKEN" \
  -w '\nHTTP status: %{http_code}\n'

curl -sS --max-time 60 "$TAXI_URL/predict" \
  -H "Authorization: Bearer $TAXI_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"Trip_Distance_km":10.0,"Passenger_Count":2,"Base_Fare":3.0,"Per_Km_Rate":1.5,"Per_Minute_Rate":0.3,"Time_of_Day":"Morning","Day_of_Week":"Weekday","Traffic_Conditions":"Low","Weather":"Clear"}' \
  -w '\nHTTP status: %{http_code}\n'
```

Recorded v2 response: `{"estimated_fare":37.2,"model_version":"2"}`, HTTP 200. To call staging, use `https://taxi-fare-staging-259177885839.asia-southeast1.run.app`. Refresh the identity token before a later session; never commit it. Login alone does not grant Invoker permission. A 401/403 is an access issue; 000 means no HTTP response was received.

These URLs are available only while the services remain deployed. After submission cleanup, use local inference and saved cloud evidence.

## 6. CI/CD, model registration and rollback

[GitHub Actions](https://github.com/Yimeng000/itcs355_project/actions) runs four jobs on main:

| Job | Checks and actions |
|---|---|
| `test` | Audits and automated tests |
| `reproduce` | Retrieve tracked data and registered reference; force the DVC pipeline; compare models; registration dry-run |
| `staging` | Build/push immutable image; deploy staging; verify readiness/version, fare and HTTP 422 |
| `production` | Save old traffic; deploy the same image; verify API; automatically restore traffic on deployment/check failure |

Pull requests run `test`; authorized main pushes/manual runs also execute the cloud jobs. Staging must pass before production. Authentication uses Workload Identity Federation, not committed service-account keys. Runtime identity: `taxi-predict-runtime@itcs355-6688176.iam.gserviceaccount.com`.

**CI retrains for verification; it does not automatically register or promote the retrained model.** Serving still uses the model selected by committed `reports/registration.json`. Fixed fare expectations must be updated deliberately when changing that model.

To reproduce the comparison locally after Section 4, download the registered reference with authorized application-default credentials:

```bash
export GOOGLE_CLOUD_PROJECT=itcs355-6688176
python - <<'PY'
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit
from google.cloud import storage
receipt = json.loads(Path("reports/registration.json").read_text())
uri = urlsplit(receipt["artifact_uri"].rstrip("/") + "/model.joblib")
target = Path("reports/reference/model.joblib")
target.parent.mkdir(parents=True, exist_ok=True)
storage.Client().bucket(uri.netloc).blob(uri.path.lstrip("/")).download_to_filename(str(target))
assert hashlib.sha256(target.read_bytes()).hexdigest() == receipt["lineage"]["model_sha256"]
print("Registered reference verified")
PY

python -m src.compare_models \
  --reference reports/reference/model.joblib \
  --candidate models/model.joblib \
  --validation data/processed/val.csv \
  --receipt reports/registration.json \
  --training-record reports/train_metrics.json \
  --output reports/local-comparison.json

python -m src.register \
  --reference-model reports/reference/model.joblib \
  --comparison-report reports/local-comparison.json \
  --dry-run
```

Dry-run performs no upload. Actual registration additionally needs a passing comparison, an immutable `--image` digest, cloud configuration and write permissions. `src/register.py` creates a new version and replaces the local registration receipt; review that receipt before committing a promotion.

Deployment updates existing services. Another cloud project must provision APIs, storage, registry, services, runtime identity, federation and permissions first. Repository configuration files and existing resource IDs do not provision those resources automatically.

Workflow artifacts preserve pipeline, staging and production evidence. The separate **Staging rollback drill** is manually triggered on main. It intentionally fails a version check, restores old traffic and verifies prediction. A red drill with successful recovery steps is expected. Recorded automatic staging traffic restoration/verification: **9.63 s**, excluding detection and subsequent API checks. See [rollback demonstration](docs/rollback-demo.md).

## 7. Monitoring, failure handling and load test

The [dashboard](https://console.cloud.google.com/monitoring/dashboards/builder/4b8c923c-02c9-450e-9f28-6bdf0142a602?project=itcs355-6688176) displays production request rate, container latency and 4xx/5xx rates. Select a time range containing the test; collection is asynchronous.

The email alert counts HTTP 422 responses: **more than 5 in a 60-second window**. The original deliberate failure sent 20 negative-distance requests; all returned 422, email arrived, valid prediction still worked and the incident later closed. The regression test verifies rejection before model execution and continued normal prediction. See [failure record](docs/failure-demo.md) and [monitoring configuration](cloudlayer/monitoring/).

With Section 5's environment variables set, reproduce the invalid-input burst:

```bash
for i in $(seq 1 20); do
  curl -sS --max-time 30 "$TAXI_URL/predict" \
    -H "Authorization: Bearer $TAXI_TOKEN" \
    -H 'Content-Type: application/json' \
    -d '{"Trip_Distance_km":-5,"Passenger_Count":2,"Base_Fare":3.0,"Per_Km_Rate":1.5,"Per_Minute_Rate":0.3,"Time_of_Day":"Morning","Day_of_Week":"Weekday","Traffic_Conditions":"Low","Weather":"Clear"}' \
    -o /dev/null -w "Request $i: HTTP %{http_code}\n"
done
```

Then repeat Section 5's valid prediction. Inspect Monitoring → Alerting and the configured mailbox. A valid request does not directly close the incident. Avoid overlapping bursts when recording evidence. The dashboard's 4xx rate differs from the alert's count; no 5xx data is not proof of zero errors.

Controlled traffic tests achieved 1 and 10 requests/s for approximately 30 seconds: 30/30 and 300/300 valid predictions, no failed requests or client capacity skips. Client p95 increased from 213.72 to 243.58 ms. No failure point or maximum capacity was established. See [load-test report](docs/load-test.md).

If investigating prediction errors, inspect Cloud Run → service → Logs, or:

```bash
gcloud run services logs read taxi-fare-api \
  --project=itcs355-6688176 --region=asia-southeast1 --limit=50
```

## 8. Costs, limitations and evidence

Estimated serving cost: **USD 0.003008 per 1,000 predictions**, before free-tier allowances and credits. Basis: 1,000 successful requests, 70.3 observed billable instance seconds, 1 CPU/1 GiB and Singapore pricing. This is not the actual bill or a complete project cost. Startup CPU boost and other costs/measurement limitations are discussed in the [cost report](docs/cost-report.md).

| Evidence | Location |
|---|---|
| Intended use, data, evaluation and lineage | [Model card](docs/model-card.md) |
| Dataset checks | [Data quality report](reports/data_quality.json) |
| Retraining consistency and comparison | [Retraining report](docs/retraining-check.md) |
| Invalid-input alert and feedback into tests | [Failure record](docs/failure-demo.md) |
| Tenfold request-rate test | [Load-test report](docs/load-test.md) |
| Automatic restoration exercised in staging | [Rollback record](docs/rollback-demo.md) |
| Cost assumptions, calculations and screenshots | [Cost report](docs/cost-report.md), [calculation](reports/cost_estimate.json), [screenshots](docs/evidence/cost/) |

Remaining limitations:

- Synthetic, small dataset; real-world accuracy and fairness are unverified. Test MAE is about USD 11.5 and individual errors can be larger.
- Inputs are caller-supplied. Extreme distances/rates may be unreliable; no live data feeds or prediction-error/drift alerts are implemented.
- Training permits integer passenger counts 1–6; the public API deliberately supports only 1–4. Performance has not been separately established for that narrower subset.
- Short, unrepeated load tests do not establish sustained capacity or an end-to-end latency guarantee.
- Validation records have already been used for the reference. The training script also reports test metrics; repeatedly tuning against them would weaken independent evaluation.
- Automatic restoration was tested in staging, not by intentionally failing production. It restores traffic, not external state, and depends on available revisions and cloud access.

Future work: real taxi data evaluation; input-distribution and prediction-error monitoring; longer capacity tests; stronger model-promotion controls. These are future plans, not completed features.

## 9. Final verification and cleanup

Before final submission, verify the latest normal CI is green, preserve drill artifacts/screenshots, check the model card fits one readable exported page and validate instructions in a fresh checkout. Cleanup evidence must reflect actual completed actions, not intentions.

After preserving required evidence and finishing cloud validation, disable **Project CI and CD** and **Staging rollback drill** in Actions before deleting services. Complete cleanup by final submission under the course rules; scaling to zero is not a substitute.

```bash
gcloud run services delete taxi-fare-staging \
  --project=itcs355-6688176 --region=asia-southeast1
gcloud run services delete taxi-fare-api \
  --project=itcs355-6688176 --region=asia-southeast1
```

Review remaining project images, model versions, monitoring resources, storage and permissions separately. Shared lab buckets, registry repositories and identities must not be deleted wholesale as project-only cleanup. Document retained resources and any ongoing costs. Preserve a reproducible dataset/model access route before deleting stored artifacts.

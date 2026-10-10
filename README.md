# Automated MLOps Pipeline for Taxi Price Prediction

ITCS355 capstone project: a reproducible taxi fare training pipeline and an authenticated prediction API deployed on Google Cloud Run. Models are registered in Vertex AI with training lineage. GitHub Actions tests and deploys changes to the serving application.

## 1. System overview

| Part | Implementation |
|---|---|
| Code versions | Git and GitHub |
| Data and model file versions | DVC, with a private Cloud Storage remote |
| Training | Data cleaning, seeded splitting, preprocessing and Random Forest regression |
| Training records | MLflow: parameters, metrics, code commit and data fingerprint |
| Model registry | Vertex AI; registration uploads reports and a lineage manifest |
| Serving | FastAPI in Docker, deployed to Cloud Run |
| CI | Portability audit, Git history credential scan and automated tests |
| CD | Build, push and deploy a serving image after CI passes; then verify the live API |
| Monitoring | Cloud Monitoring dashboard: request rate, latency by series, 4xx and 5xx rates |
| Alert | More than 5 HTTP 422 responses in a 60-second window; email notification |
| Cost report | Estimated USD 0.003008 per 1,000 predictions; assumptions and evidence documented |
| Model card | Dataset, intended use, inputs, evaluation and limitations documented; one-page layout verification pending |
| Deliberate failure | Invalid-input burst triggered an email alert; continued prediction verified and regression test added |

`src/` contains data and model logic. `service/` contains request validation and prediction. Provider-specific storage, registration and deployment operations are in `cloudlayer/`. Only the GCP adapters are currently implemented; passing the portability audit does not mean other providers are implemented or tested.

There is currently one cloud serving environment, `taxi-fare-api`. There is no separate staging service. Development and unit testing run locally and in GitHub Actions.

Supporting documentation:

- [Model card](docs/model-card.md): intended use, dataset, inputs, evaluation and limitations.
- [Cost report](docs/cost-report.md): estimated cost of USD 0.003008 per 1,000 predictions, before free-tier allowances and credits; scope and assumptions are documented.
- [Cost calculation](reports/cost_estimate.json): saved calculation output.
- [Failure demonstration](docs/failure-demo.md): invalid-input burst, alert and regression test.
- [Cost evidence](docs/evidence/cost/): billable-time and Singapore pricing screenshots.

## 2. Dataset and prediction scope

Source: [Taxi Price Prediction / Taxi Price Regression on Kaggle](https://www.kaggle.com/datasets/denkuznetz/taxi-price-prediction), published by `denkuznetz`.

- Raw filename: `data/taxi_trip_pricing.csv`.
- Observed dataset: 1,000 rows and 11 columns.
- Target: `Trip_Price`.
- Cleaning removed 49 records with missing targets, leaving 951 records.
- Seed 42 split: 665 training, 143 validation and 143 test records.
- `Trip_Duration_Minutes` is removed because actual trip duration is unavailable before a trip finishes.
- Missing input values are handled by the training pipeline's imputers.
- The publisher describes the dataset as realistic synthetic data, rather than recorded taxi journeys.
- Fare values are in USD.
- Dataset licence: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), as stated on the publisher's Kaggle page.
- The actual CSV columns define this project's inputs; the publisher's general description does not exactly match the CSV column names.

The service estimates a fare from user-supplied distance, pricing rates and conditions. It does not calculate routes, retrieve live weather or traffic, or guarantee the actual fare charged by a taxi operator. Distance must be an estimate available when requesting the prediction.

### Input contract

All nine fields are required; additional fields are rejected.

| Field | Accepted API value |
|---|---|
| `Trip_Distance_km` | Finite number greater than 0 |
| `Passenger_Count` | Integer from 1 to 4; fractional values and numeric strings are rejected |
| `Base_Fare` | Finite number greater than or equal to 0 |
| `Per_Km_Rate` | Finite number greater than 0 |
| `Per_Minute_Rate` | Finite number greater than or equal to 0 |
| `Time_of_Day` | `Morning`, `Afternoon`, `Evening`, `Night` |
| `Day_of_Week` | `Weekday`, `Weekend` |
| `Traffic_Conditions` | `Low`, `Medium`, `High` |
| `Weather` | `Clear`, `Rain`, `Snow` |

The API converts accepted categorical values to lowercase before prediction, matching training preprocessing. The cleaner currently accepts passenger counts from 1 to 6, while the public API supports 1 to 4. This difference remains to be reconciled or justified.

## 3. Install and run automated checks

Prerequisites: Python **3.11**, Git and GNU Make. Docker is required only for container commands. Google Cloud CLI and cloud permissions are required only for private data retrieval and cloud operations. Python 3.11.15 was used locally; the serving image and CI use Python 3.11.

```bash
git clone https://github.com/Yimeng000/itcs355_project.git
cd itcs355_project
python3.11 -m venv .venv-model
source .venv-model/bin/activate
python -m pip install -r requirements-dev.txt
make check
```

On Windows, run these shell commands in WSL. The selected Python installation must include `venv` support.

`make check` runs the portability audit, Git history credential scan and all tests. A complete Git history is needed for the credential scan; do not use a shallow clone for this check.

The tests cover request validation, service responses, taxi data cleaning and splitting, and a small model built with the project's preprocessing code. They do not need cloud credentials or the private dataset. The model unit tests do not load the deployed production artifact; cloud checks validate that artifact separately.

Run individual groups with:

```bash
python -m pytest -v tests/test_data.py tests/test_model_behaviour.py
python -m pytest -v tests/test_schemas.py tests/test_service.py
```

## 4. Retrieve data and reproduce training

Install the training dependencies:

```bash
python -m pip install -r requirements-training.txt
```

The configured DVC remote is private: `gs://itcs355-6688176/itcs355_project/dvc-cache`. A reviewer or teammate needs read access to this bucket. Installing dependencies or cloning the repository does not grant that access.

For an authorized account with Google Cloud CLI installed:

```bash
gcloud auth application-default login
python -m dvc pull data/taxi_trip_pricing.csv.dvc
make train
```

If access is unavailable, obtain the dataset from its publisher, place the CSV at the documented raw path and run the pipeline. This reproduces the recorded results only if the file matches the tracked dataset version; verify it against `data/taxi_trip_pricing.csv.dvc` rather than assuming a newly downloaded version is identical.

`make train` invokes `dvc repro`; unchanged stages may be skipped. To rerun every stage rather than reuse local outputs:

```bash
python -m dvc repro --force
```

Training runs locally and does not register or deploy a model. It writes:

| Output | Purpose |
|---|---|
| `data/processed/clean.csv`, `train.csv`, `val.csv`, `test.csv` | Cleaned data and splits |
| `data/processed/clean_report.json`, `split_report.json` | Cleaning and splitting reports |
| `models/model.joblib` | Fitted preprocessing and regression pipeline |
| `models/input_schema.json` | Feature schema |
| `reports/train_metrics.json` | Training run ID, training commit, data fingerprint, model SHA256 and metrics |
| `reports/metrics.json` | Evaluation and quality gate results |
| `reports/predicted_vs_actual.png` | Evaluation plot |

MLflow defaults to `sqlite:///mlflow.db`, with local artifacts. View training records with `mlflow ui --host 127.0.0.1 --port 5000`. The local database is not supplied by a clone; rerunning training creates new records. Registered lineage is available in the committed registration receipt and uploaded artifacts.

Recorded model v2 test results: MAE **11.4973**, RMSE **18.2893**, R² **0.8854**. Accuracy is supporting context, not the capstone's grading objective. The configured local latency gate is p95 below 1,000 ms; this is not evidence of a measured cloud latency SLO.

## 5. Run the API locally

After retrieving or training the model:

```bash
MODEL_PATH=models/model.joblib MODEL_VERSION=local \
  python -m uvicorn service.app:app --host 127.0.0.1 --port 8080
```

Open [interactive API documentation](http://127.0.0.1:8080/docs). From another terminal:

```bash
curl -sS http://127.0.0.1:8080/ready
curl -sS http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"Trip_Distance_km":10.0,"Passenger_Count":2,"Base_Fare":3.0,"Per_Km_Rate":1.5,"Per_Minute_Rate":0.3,"Time_of_Day":"Morning","Day_of_Week":"Weekday","Traffic_Conditions":"Low","Weather":"Clear"}'
```

The recorded v2 model returns `{"estimated_fare":37.2,"model_version":"local"}` for this input when served locally with that version label. Retrained or changed models may produce different values.

| Endpoint / response | Meaning |
|---|---|
| `GET /health`, 200 | Process is running; this does not prove the model loaded |
| `GET /ready`, 200 | Model loaded; response includes the version label |
| `GET /ready`, 503 | Model unavailable |
| `POST /predict`, 200 | Estimated fare and model version |
| `POST /predict`, 422 | Invalid, missing or additional input fields |
| `POST /predict`, 503 | Model unavailable |
| `POST /predict`, 500 | Prediction failed or model returned an invalid fare |

`GET /` is not implemented and returns 404. Predictions are single-request online predictions; there is no batch endpoint.

### Docker with a local model

```bash
docker build -f service/Dockerfile.serve -t taxi-fare:local .
docker run --rm -p 8080:8080 \
  -v "$PWD/models:/app/models:ro" \
  -e MODEL_VERSION=local taxi-fare:local
```

Model files are not bundled into the image. If `MODEL_ARTIFACT_URI` is set, the service downloads the specified cloud artifact at startup; otherwise it reads `MODEL_PATH`. Cloud mode requires credentials with object read permission and a resolvable project ID. Use `GOOGLE_CLOUD_PROJECT` explicitly when testing cloud mode locally. Cloud Run uses its attached runtime service account, not a copied personal credential file.

## 6. Registered model and authenticated cloud API

| Resource | Recorded configuration |
|---|---|
| Project / region | `itcs355-6688176` / `asia-southeast1` |
| Vertex AI model ID / version | `4463872073235693568` / `2` |
| Cloud Run service | `taxi-fare-api` |
| Runtime service account | `taxi-predict-runtime@itcs355-6688176.iam.gserviceaccount.com` |
| Service URL | `https://taxi-fare-api-259177885839.asia-southeast1.run.app` |
| Scaling / resources | Revision maximum 1 instance; service-level maximum 20; 1 CPU, 1 GiB memory; concurrency 4 |

The service requires authentication. An authorized user can call it with:

```bash
gcloud auth login
TAXI_SERVICE_URL="https://taxi-fare-api-259177885839.asia-southeast1.run.app"
curl -sS "$TAXI_SERVICE_URL/ready" \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -w '\nHTTP status: %{http_code}\n'
curl -sS "$TAXI_SERVICE_URL/predict" \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H 'Content-Type: application/json' \
  -d '{"Trip_Distance_km":10.0,"Passenger_Count":2,"Base_Fare":3.0,"Per_Km_Rate":1.5,"Per_Minute_Rate":0.3,"Time_of_Day":"Morning","Day_of_Week":"Weekday","Traffic_Conditions":"Low","Weather":"Clear"}' \
  -w '\nHTTP status: %{http_code}\n'
```

Recorded response: `{"estimated_fare":37.2,"model_version":"2"}`, HTTP 200. A Google login alone does not grant permission to invoke the service.

## 7. Registration and CI/CD

Preview registration after reproducing training and evaluation:

```bash
python -m src.register --dry-run
```

This checks required files, the model SHA256, training lineage, matching reported metrics and the quality gate. It writes a local lineage manifest but performs no cloud upload. It does not independently bind the evaluation report to a model checksum; the current workflow relies on evaluating the just-trained artifact in sequence.

For real registration, set `CLOUD_PROVIDER`, `PROJECT_ID`, `REGION`, `BLOB_URI` and `MODEL_REGISTRY_NAME` through exported variables or an ignored `cloud.env`. For this project's existing resources:

```bash
export CLOUD_PROVIDER=gcp
export PROJECT_ID=itcs355-6688176
export REGION=asia-southeast1
export BLOB_URI=gs://itcs355-6688176/itcs355_project
export MODEL_REGISTRY_NAME=taxi-fare-model
```

Then use `python -m src.register --parent-model=4463872073235693568 --image=IMAGE_URI_AT_SHA256_DIGEST`, replacing the image placeholder with an actual pushed image digest. This writes to cloud storage and creates a model version. It requires bucket upload and Vertex AI registration permissions. Registration itself does not deploy an endpoint. The receipt `reports/registration.json` is the deployment input; update it deliberately when promoting a new model.

[GitHub Actions runs](https://github.com/Yimeng000/itcs355_project/actions) use `.github/workflows/ci.yml`:

1. Pull requests, main pushes and manual runs execute `make check`.
2. Only successful main pushes or main manual runs proceed to deployment.
3. Build the serving image, push a commit-tagged image and resolve its immutable digest.
4. Update the existing Cloud Run service using the committed registration receipt.
5. Check readiness/version, the v2 example prediction and rejection of negative distance with HTTP 422.
6. Save `reports/deployment.json` as a workflow artifact recording the deployed application commit, image and model.

**CD updates serving code with the selected registered model. It does not automatically retrain, evaluate or register a new model.** The fixed `37.2` smoke-check expectation must be reviewed when promoting a different model. Checks run after the update has deployed; there is currently no automatic rollback if a smoke check fails.

Google authentication uses Workload Identity Federation through the existing `github-pool/github-provider` and `github-actions` service account. The project's main branch is authorized. The deployment account has image upload permission, Cloud Run Developer and Invoker permissions on the service, and Service Account User permission on its runtime identity. The runtime identity has bucket object read permission. Reproducing deployment in a different project requires provisioning the APIs, registry, bucket, identities, federation bindings and initial Cloud Run service, then updating the workflow and receipt. The deploy adapter updates an existing service; it does not provision these resources.

## 8. Monitoring, alerting and deliberate failure

The deployed Cloud Run service is monitored through Cloud Monitoring. The dashboard shows request rate, p95 container request latency per series, and 4xx/5xx response rates. An email alert detects bursts of invalid prediction requests.

| Component | Implementation | Repository file |
|---|---|---|
| Monitoring dashboard | Request rate, container latency, 4xx and 5xx response rates | [dashboard.json](cloudlayer/monitoring/dashboard.json) |
| Alert policy | More than 5 HTTP 422 responses in a 60-second window | [invalid-input-alert.json](cloudlayer/monitoring/invalid-input-alert.json) |
| Deliberate failure | A faulty client sends 20 requests with negative trip distance | [failure-demo.md](docs/failure-demo.md) |
| Feedback into tests | Reject the burst before model execution and verify a subsequent valid prediction | [test_service.py](tests/test_service.py), `test_invalid_input_burst_does_not_break_predictions` |

The recorded demonstration returned HTTP 422 for all 20 invalid requests, generated an email notification, and subsequently showed a Closed incident. A valid request after the burst returned HTTP 200 with fare 37.2 and model version 2. The full local checks passed with 44 tests. This is an invalid-client-input scenario, not a model outage.

The following steps reproduce this behaviour against the **existing deployed cloud service**. They do not require local training, DVC access, Docker, model registration or another deployment. Running the local API alone will not populate the Cloud Run monitoring dashboard.

### 8.1 Install dependencies and run checks

In an existing checkout with no uncommitted changes:

```bash
cd ~/itcs355_project
git pull --ff-only
source .venv-model/bin/activate
python --version
python -m pip install -r requirements-dev.txt
make check
```

Expected at this revision: **44 passed**. If you have not cloned the project or created the Python 3.11 environment, complete Section 3 first. Keep your local edits if Git reports a conflict; resolve it before continuing.

Run the failure regression test separately:

```bash
python -m pytest tests/test_service.py::test_invalid_input_burst_does_not_break_predictions -v
```

Expected: **1 passed**. This uses a test model and checks 20 rejected inputs, no model calls for invalid input, readiness and a subsequent valid prediction. It does not send email.

### 8.2 Authentication and access requirements

The owner must grant your account access before you run the cloud rehearsal:

| Operation | Required access |
|---|---|
| Call the existing Cloud Run API | Cloud Run Invoker on `taxi-fare-api` |
| View graphs and alert incidents | Monitoring Viewer in the project |
| Retrieve DVC data for Section 4 | Storage Object Viewer on the data bucket; not needed here |

Use your own account. Logging in does not automatically grant these permissions. Do not share passwords, tokens or personal credential files.

```bash
gcloud auth login
gcloud config set project itcs355-6688176
gcloud auth list --filter=status:ACTIVE --format="value(account)"
export SERVICE_URL="https://taxi-fare-api-259177885839.asia-southeast1.run.app"
export TOKEN="$(gcloud auth print-identity-token)"
export EVIDENCE_DIR="docs/evidence/$(date +%Y-%m-%d)"
mkdir -p "$EVIDENCE_DIR"

curl -sS "$SERVICE_URL/ready" \
  -H "Authorization: Bearer $TOKEN" \
  -w '\nHTTP status: %{http_code}\n'
```

Expected: HTTP 200 and model version `2`. Keep the token private and refresh it with the export command before a later rehearsal. A 401/403 is an authentication/permission problem, not the deliberate input failure. A 503 means the model is not ready; inspect service logs before continuing.

The existing alert sends email to the owner's configured mailbox. The teammate should inspect the shared incident and ask the owner to confirm email receipt. It will not automatically send to the teammate's email. Do not recreate the dashboard or policy just to repeat the test.

### 8.3 Prepare valid and invalid payloads

```bash
cat > /tmp/taxi-valid.json <<'EOF'
{
  "Trip_Distance_km": 10.0,
  "Passenger_Count": 2,
  "Base_Fare": 3.0,
  "Per_Km_Rate": 1.5,
  "Per_Minute_Rate": 0.3,
  "Time_of_Day": "Morning",
  "Day_of_Week": "Weekday",
  "Traffic_Conditions": "Low",
  "Weather": "Clear"
}
EOF

python - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("/tmp/taxi-valid.json").read_text())
payload["Trip_Distance_km"] = -5
Path("/tmp/taxi-invalid.json").write_text(json.dumps(payload))
PY
```

### 8.4 Successful prediction

```bash
curl -sS "$SERVICE_URL/predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/taxi-valid.json \
  -w '\nHTTP status: %{http_code}\n' \
  | tee "$EVIDENCE_DIR/01-valid-prediction.txt"
```

Recorded model v2 expectation: HTTP 200, `estimated_fare=37.2`, `model_version="2"`.

Presentation wording:

> This is our deployed taxi fare prediction service. A valid request returns an estimated fare and the model version.

### 8.5 One invalid prediction

```bash
curl -sS "$SERVICE_URL/predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/taxi-invalid.json \
  -w '\nHTTP status: %{http_code}\n' \
  | tee "$EVIDENCE_DIR/02-invalid-prediction.txt"
```

Expected: HTTP 422 with error location `["body", "Trip_Distance_km"]`.

> A negative trip distance is invalid. The service rejects the request and identifies the incorrect field.

### 8.6 Deliberate failure: 20 invalid requests

Before starting, check that the previous alert incident is Closed and coordinate with the owner. Repeated overlapping rehearsals make the counts and notifications harder to interpret.

Run in the same terminal used for the exports above:

```bash
python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

payload = Path("/tmp/taxi-invalid.json").read_bytes()
started = datetime.now(timezone.utc).isoformat()
results = []
print("Test started at UTC:", started)

for number in range(1, 21):
    request = Request(
        os.environ["SERVICE_URL"] + "/predict",
        data=payload,
        headers={
            "Authorization": "Bearer " + os.environ["TOKEN"],
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            status = response.status
    except HTTPError as error:
        status = error.code
        error.close()
    results.append({"request": number, "status": status})
    print(f"Request {number}/20: HTTP {status}")

report = {
    "started_at_utc": started,
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    "results": results,
}
destination = Path(os.environ["EVIDENCE_DIR"]) / "03-invalid-burst.json"
destination.write_text(json.dumps(report, indent=2))
assert all(row["status"] == 422 for row in results)
print("All 20 invalid requests were rejected.")
print("Evidence saved to:", destination)
PY
```

> Our deliberate failure is a faulty client repeatedly sending invalid input. We send twenty invalid requests to demonstrate detection and handling.

### 8.7 Monitoring and notification verification

In Google Cloud Console, select project `itcs355-6688176`:

1. Open **Monitoring → Dashboards → ITCS355 Taxi Fare Monitoring**.
2. Select a time range that includes your test and refresh.
3. Open **Monitoring → Alerting → Taxi Fare - Invalid Input Surge**.
4. Inspect the alert incident and ask the owner to check the configured mailbox.

| Existing resource | ID / configuration file |
|---|---|
| Dashboard | `4b8c923c-02c9-450e-9f28-6bdf0142a602`; [dashboard.json](cloudlayer/monitoring/dashboard.json) |
| Alert policy | `16399261076497637481`; [invalid-input-alert.json](cloudlayer/monitoring/invalid-input-alert.json) |
| Email notification channel | `12155343553823774073` |

The policy counts HTTP 422 responses across the service using 60-second alignment, with a threshold strictly greater than 5. The dashboard's 4xx chart shows a **rate**, not this count. Nearby invalid requests can make the policy count exceed the 20-request burst alone.

Monitoring and notification delivery are asynchronous. Record when the incident/email actually appears; do not assume it will appear immediately. The latency chart shows p95 container request latency per series, not one overall end-to-end latency measurement. No data on a 5xx chart is not proof that every possible failure was measured.

> Monitoring detected the invalid-input burst and sent an email. HTTP 422 represents rejected client input; it does not mean the model crashed.

### 8.8 Continued service, alert closure and evidence

```bash
curl -sS "$SERVICE_URL/predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/taxi-valid.json \
  -w '\nHTTP status: %{http_code}\n' \
  | tee "$EVIDENCE_DIR/04-valid-after-burst.txt"

python -m pytest \
  tests/test_service.py::test_invalid_input_burst_does_not_break_predictions \
  -v | tee "$EVIDENCE_DIR/05-regression-test.txt"
```

Expected: valid prediction HTTP 200 and test **1 passed**. Stop sending invalid requests, then inspect the incident until it shows **Closed**. Closure timing is separate from the successful prediction; a valid request does not directly close the incident.

> Valid requests still succeed after the invalid-input burst. We added this scenario to our automated tests so future changes must preserve this behaviour.

Save screenshots in the evidence directory: monitoring graph, incident details, notification email and Closed state. Exclude authentication tokens and redact unrelated personal information. See [the original failure record](docs/failure-demo.md). Record your own rehearsal timestamps separately rather than changing the original event times.

### 8.9 Presentation evidence and troubleshooting

Use the rehearsal evidence to supplement a live demo. Prepare commands and browser tabs beforehand. Run a valid prediction, invalid input and a subsequent valid prediction live; have the burst command ready. Show saved evidence if notification delivery exceeds the presentation time, and explicitly identify it as rehearsal evidence. Inspect the latest [Actions run](https://github.com/Yimeng000/itcs355_project/actions) before claiming the current deployment passed.

| Symptom | Check |
|---|---|
| Local `GET /` returns 404 | There is no home route; use `/docs`, `/health` or `/ready` |
| `GET /predict` returns 405 | Predictions require POST with JSON |
| Local readiness returns 503 | Obtain/train the model and check startup logs; health 200 alone is insufficient |
| Cloud request returns 401/403 | Refresh the token; check your own account and Cloud Run Invoker permission |
| No cloud chart after local requests | These charts monitor the deployed Cloud Run service, not localhost |
| Teammate receives no email | The policy uses the owner's mailbox; inspect the incident and coordinate |
| Cloud Shell browser cannot reach localhost | Use Cloud Shell Web Preview, not your computer's localhost |

For a local API running **inside Cloud Shell**, start Uvicorn with `--host 0.0.0.0 --port 8080`. Keep it running and select **Web Preview → Preview on port 8080**, then append `/docs` to that preview URL. Alternatively, in another Cloud Shell terminal, print:

```bash
printf 'https://8080-%s/docs\n' "$WEB_HOST"
```

See [Google's Cloud Shell Web Preview instructions](https://docs.cloud.google.com/shell/docs/using-web-preview). The Cloud Shell preview URL and authenticated Cloud Run service URL are different surfaces.

## 9. Completion status, remaining work and cleanup

| Requirement | Current evidence / remaining work |
|---|---|
| Versioned data and code | Git, DVC metadata and `dvc.lock`; private remote access required |
| Automated reproducible training | DVC stages and pinned training dependencies; locally verified |
| Registered model with lineage | `reports/registration.json` and `reports/model_lineage.json`, plus uploaded reports |
| Deployed inference | Authenticated Cloud Run API; successful prediction recorded |
| CI/CD with failing checks | 44 local tests, audits and deployment smoke checks; verify the latest GitHub Actions run before submission |
| Monitoring dashboard | Committed configuration and observed cloud graphs |
| Working alert | Email received and incident Closed on 9 October |
| Deliberate failure and feedback into tests | 20 invalid requests rejected, subsequent valid prediction successful, alert verified and regression test added; see [failure record](docs/failure-demo.md) |
| Cost per 1,000 predictions | Estimated USD 0.003008 before free-tier allowances and credits; calculation scope and limitations in [cost report](docs/cost-report.md), with [evidence](docs/evidence/cost/) |
| One-page model card | [Model card content](docs/model-card.md) completed; printed/exported one-page layout still needs verification |
| Dataset licence and provenance | Publisher identifies synthetic data, USD fares and Apache 2.0 licensing; documented in the model card |

Before the presentation and repository submission:

1. Verify that the latest GitHub Actions run passes.
2. Check the model card's printed/exported layout fits one readable page.
3. Repeat the failure demonstration and organize screenshots of the monitoring graph, alert, email, closure and subsequent successful prediction.
4. Rehearse the eight-minute presentation and prepare for questions.
5. Check that README links work and that the documented setup can be followed from a fresh checkout.

The deliberate failure simulates a faulty client repeatedly sending invalid input. The automated test verifies rejection and continued prediction; the cloud demonstration verifies monitoring and notification. The presentation remains separate from repository completion.

### Cleanup after the required demonstration and submission

Keep the service available for the required demonstration and any agreed assessment access. Before final cleanup, preserve evidence and disable the deployment workflow in GitHub Actions so another main push does not redeploy the service.

To remove this project's serving service:

```bash
gcloud run services delete taxi-fare-api \
  --project=itcs355-6688176 --region=asia-southeast1
```

There is currently no complete project teardown command. Review project-specific stored images, model versions, storage objects, monitoring resources and IAM bindings separately. The bucket, Artifact Registry repository and GitHub identity pool are shared with course labs; do not delete shared resources wholesale as project-only cleanup.

Complete resource cleanup as required by the course submission rules. Scaling to zero does not replace final cleanup.

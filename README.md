# Automated MLOps Pipeline for Taxi Price Prediction

ITCS355 capstone project: a reproducible taxi fare training pipeline and an authenticated prediction API deployed on Google Cloud Run. Models are registered in Vertex AI with training lineage. GitHub Actions tests and deploys changes to the serving application.

**Status, 8 October 2026:** local checks passed with 43 tests; cloud prediction and the CI/CD deployment passed. Monitoring, alerting, a deliberate failure demonstration, the cost report and the model card are still pending. This README describes the implementation currently in the repository.

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

`src/` contains data and model logic. `service/` contains request validation and prediction. Provider-specific storage, registration and deployment operations are in `cloudlayer/`. Only the GCP adapters are currently implemented; passing the portability audit does not mean other providers are implemented or tested.

There is currently one cloud serving environment, `taxi-fare-api`. There is no separate staging service. Development and unit testing run locally and in GitHub Actions.

## 2. Dataset and prediction scope

Source: [Taxi Price Prediction / Taxi Price Regression on Kaggle](https://www.kaggle.com/datasets/denkuznetz/taxi-price-prediction), published by `denkuznetz`.

- Raw filename: `data/taxi_trip_pricing.csv`.
- Observed dataset: 1,000 rows and 11 columns.
- Target: `Trip_Price`.
- Cleaning removed 49 records with missing targets, leaving 951 records.
- Seed 42 split: 665 training, 143 validation and 143 test records.
- `Trip_Duration_Minutes` is removed because actual trip duration is unavailable before a trip finishes.
- Missing input values are handled by the training pipeline's imputers.
- **Dataset licence, currency and original data collection/generation details still require verification against the publisher's data card before final submission.** No currency or real-world provenance is claimed here.

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
| Scaling / resources | Minimum 0, maximum 1 configured revision instance; 1 CPU, 1 GiB memory; concurrency 4 |

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

## 8. Remaining capstone work and cleanup

| Requirement | Current evidence / remaining work |
|---|---|
| Versioned data and code | Git, `.dvc` metadata and `dvc.lock`; private remote access required |
| Automated reproducible training | DVC stages and pinned training dependencies; locally verified |
| Registered model with lineage | `reports/registration.json` and `reports/model_lineage.json`, plus uploaded reports |
| Deployed inference | Authenticated Cloud Run API, recorded 200 prediction |
| CI/CD with failing checks | 43 tests, audits and live deployment smoke checks; successful run observed |
| Monitoring dashboard | Pending configuration and evidence |
| Working alert | Pending configuration and observed notification |
| Deliberate failure and feedback into tests | Pending planned injection, observation, recovery and regression test evidence |
| Cost per 1,000 predictions | Pending measured workload and documented calculation |
| One-page model card | Pending |
| Dataset licence/provenance | Pending verification |

Existing readiness handling, input validation and logs are useful reliability features, but they do not by themselves complete the dashboard, alert or deliberate failure requirements.

After the required presentation and evidence collection, disable the deployment workflow in GitHub Actions before removing cloud resources; otherwise another main push can attempt deployment again. To remove this project's serving service:

```bash
gcloud run services delete taxi-fare-api \
  --project=itcs355-6688176 --region=asia-southeast1
```

There is currently no complete project teardown command. Review project-specific stored images, model versions, storage objects and IAM bindings separately. The bucket, Artifact Registry repository and GitHub identity pool are shared with course labs; deleting those shared resources wholesale is not a project-only cleanup. Keep required evidence before removing artifacts. Minimum-instance zero is not a substitute for final resource cleanup.

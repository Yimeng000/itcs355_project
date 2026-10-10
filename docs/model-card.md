# Model Card — Taxi Fare Estimator

**Team:** Woraphol Meakapat (6688157), Yimeng Chen (6688176)
**Updated:** 10 October 2026
**Registered model:** Vertex AI model 4463872073235693568, version 2

## Intended use

An educational service that estimates taxi fares before a trip.
It demonstrates an operational ML pipeline and is not validated
for real-world quotations or fare-setting decisions.

## Data

Source: [Taxi Price Regression by Den_Kuznetz on Kaggle](https://www.kaggle.com/datasets/denkuznetz/taxi-price-prediction).
The publisher describes the data as realistic synthetic data,
with fares in USD. Licence: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).

The downloaded CSV contains 1,000 rows. Cleaning removes 49 rows
with missing Trip_Price, leaving 951 rows. Actual trip duration
(Trip_Duration_Minutes) is excluded because it is unavailable
before the ride. Seed 42 produces 665 training, 143 validation
and 143 test rows.

Raw data is tracked with DVC: MD5 ea9922043a10c6cffbda9e5ee5fe62f9.
Training-data fingerprint: f652ecb504509f7c.

## Model and interface

A scikit-learn preprocessing pipeline and random forest regressor
(150 trees, maximum depth 10, seed 42). Preprocessing fills missing
training values and encodes categorical features.

Nine required inputs:
- Numeric: Trip_Distance_km, Passenger_Count, Base_Fare,
  Per_Km_Rate, Per_Minute_Rate.
- Categorical: Time_of_Day, Day_of_Week, Traffic_Conditions, Weather.

The API requires positive distance and rates, nonnegative base fare,
an integer passenger count from 1 to 4, and supported categories.
Missing fields, extra fields and invalid values return HTTP 422.
Categories are converted to lowercase before prediction.

Output: estimated_fare in USD, rounded to two decimal places,
plus model_version.

## Evaluation

Held-out test set: 143 rows. Baseline predicts the training mean.

| Metric | Model | Baseline |
|---|---:|---:|
| MAE (USD) | 11.4973 | 29.2247 |
| RMSE (USD) | 18.2893 | 54.6592 |
| R² | 0.8854 | -0.0234 |

Cloud test on 9 October 2026: 1,000 requests returned HTTP 200,
with zero failures. Client-side latency from Cloud Shell:
p50 69.6 ms, p95 81.4 ms, p99 95.9 ms. These measurements include
network time and do not guarantee future latency.

## Lineage

Training commit: b8d14dc0665a15fc59252309e9e6ce875df2211b.
MLflow run: 4d94826e34ad4cdca59f9541798079da.
Model SHA-256:
444becdc54268009a0fb4abcda0c4fa6a0a07ab7f9674cdd518ddc3cfc7d48a7.

## Limitations and operation

Synthetic-data results do not establish real-world accuracy or fairness.
MAE is about USD 11.5 on this test set; individual errors can be larger.
Extreme inputs outside the training range may be unreliable.
Traffic, weather and pricing rates are caller-supplied; no live feeds
are connected. The API accepts 1–4 passengers, while data cleaning
allows a wider range.

The deliberate failure is a burst of negative-distance requests.
The service rejects them with HTTP 422, monitoring triggers an email
alert, and valid predictions remain available. This scenario is covered
by an automated regression test.

New model versions require evaluation and registration before adoption.
The fixed expected fare in the deployment smoke test must be reviewed
when changing the model.

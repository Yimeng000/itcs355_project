# Model Card — Taxi Fare Estimator

**Team:** Yimeng Chen (6688176), 
Woraphol Meakapat (6688157)  
**Updated:** 13 October 2026  
**Registered model:** Vertex AI model 4463872073235693568, version 2

## Intended use

An educational service that estimates taxi fares before a trip.
It demonstrates an operational ML pipeline and is not validated
for real-world quotations or fare-setting decisions.

## Data

Kaggle Taxi Price Prediction/Regression by denkuznetz; publisher-described synthetic data, USD fares, Apache 2.0 licence. Raw CSV: 1,000 rows; remove 49 missing Trip_Price targets, leaving 951. Seed 42: 665 train / 143 validation / 143 test. Exclude actual Trip_Duration_Minutes because it is unavailable before a trip ends.

## Model and interface

A scikit-learn preprocessing pipeline and random forest regressor
(150 trees, maximum depth 10, seed 42). Preprocessing fills missing
training values and encodes categorical features.

Nine required inputs:
- Numeric: Trip_Distance_km, Passenger_Count, Base_Fare,
  Per_Km_Rate, Per_Minute_Rate.
- Categorical: Time_of_Day, Day_of_Week, Traffic_Conditions, Weather.

Distance and per-km rate > 0; base fare and per-minute rate >= 0; finite numbers; strict integer passenger count 1-4. Categories: Morning/Afternoon/Evening/Night; Weekday/Weekend; Low/Medium/High; Clear/Rain/Snow. Accepted categories are lowercased for prediction. Missing, extra or invalid fields return HTTP 422.

Output: estimated_fare in USD, rounded to two decimal places,
plus model_version.

## Evaluation

Held-out test set: 143 rows. Baseline predicts the training mean.

| Metric | Model | Baseline |
|---|---:|---:|
| MAE (USD) | 11.4973 | 29.2247 |
| RMSE (USD) | 18.2893 | 54.6592 |
| R² | 0.8854 | -0.0234 |

## Lineage

Training commit: b8d14dc0665a15fc59252309e9e6ce875df2211b.

MLflow run: 4d94826e34ad4cdca59f9541798079da.

Model SHA-256:
444becdc54268009a0fb4abcda0c4fa6a0a07ab7f9674cdd518ddc3cfc7d48a7.

## Operational checks

Evaluation binds model, data and parameter fingerprints; real registration requires a passing validation comparison. CI reproduces training, then verifies staging before production. A negative-distance burst was rejected, triggered an email alert and informed a regression test. Automatic traffic rollback and API recovery were demonstrated in staging; production uses the same recovery logic.

## Limitations and maintenance

Synthetic results do not establish real-world accuracy or fairness. Test MAE is about USD 11.5; individual errors may be larger. Extreme inputs may be unreliable. Traffic, weather and rates are caller-supplied. Training allows 1-6 passengers; API scope is 1-4, without a separate subgroup evaluation.

Validation has already been used for the reference; repeated tuning on test metrics would weaken independent evaluation. Short cloud tests do not establish sustained capacity. Retraining verifies reproducibility but does not automatically promote a new model. Re-evaluate, compare, register and review fixed smoke-test expectations before promotion.

## References

[Dataset](https://www.kaggle.com/datasets/denkuznetz/taxi-price-prediction) | [Licence](https://www.apache.org/licenses/LICENSE-2.0) | [Repository evidence](https://github.com/Yimeng000/itcs355_project)
# Beam Trial Conversion Prediction

Predict whether a **14-day free trial will convert to a paid subscription**, using the first three days of app activity.

This Python prototype compares **logistic regression** and **XGBoost**. It prepares data, builds engagement features, trains both models, and reports performance on **train data first, then test data**.

Developed by Husayn El Sharif for the FutureProof DS ML Cohort 2026, Week 1 assignment. **Status: offline prototype; deployment and monitoring are planned.**

## Business problem

Beam is a subscription media app for browsing, listening, and reading. Its trial converts automatically to a paid plan unless the user cancels. The [model plan](trial-conversion-model-plan.md) describes roughly 250 new trials per week and conversion near 50%, below the 58% business-case assumption.

The modeling question is: **For a trial that started three days ago, how likely is it to convert at the end of day 14?** Scoring at day 3 leaves 11 days for the growth team to act.

| Team | Intended use |
| --- | --- |
| Lifecycle / Growth | Rank live trials by cancellation risk to target onboarding, audio-content prompts, and selected offers. |
| Growth Analytics | Use current trial scores to construct comparable experiment groups. |
| Finance | Sum conversion probabilities by weekly cohort to inform revenue forecasts. |

The lifecycle team chooses the intervention cutoff based on its capacity. Predicted cancellation risk does not establish whether an intervention will change an outcome.

## Quick start

Use **Python 3.13 or newer** and `uv`. Run commands from the repository root; the package reads and writes relative data/model paths.

### 1. Install the project

```bash
git clone https://github.com/helsharif/beam-trial-conversion-prediction.git
cd beam-trial-conversion-prediction
uv sync --locked
```

### 2. Configure data access

Copy `.env.example` to `.env` and supply valid database connection values. In PowerShell:

```powershell
Copy-Item .env.example .env
```

The loader reads `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, and `DB_PASSWORD`. Access to `ml.trial_snapshot_latest` is required to download data. `.env` and the raw/processed CSV snapshots are ignored by Git; a fresh clone does not include the dataset.

Alternatively, place an authorized CSV with the schema above at `data/01_raw/trials_raw.csv` and skip `download_data()`.

### 3. Train both models

Start Python with `uv run python`, then run:

```python
from trial_conversion_model.data import download_data, load_data
from trial_conversion_model.features import build_features
from trial_conversion_model.train import (
    train_model_logistic_regression,
    train_model_xgboost,
)

download_data()  # Skip if the raw CSV is already available.
df = load_data()
df = build_features(df)

lr_model = train_model_logistic_regression(df)
xgb_model = train_model_xgboost(df)
```

For interactive work, open [notebooks/01_thin_notebook.ipynb](notebooks/01_thin_notebook.ipynb) in a notebook-capable editor and select the project's `.venv` Python kernel. The notebook locates the project root before running the workflow. The [original notebook](trial_conversion_model.ipynb) contains the exploratory analysis, its own database connection setup, and the illustrative intervention cutoff.

The `trial-conversion-model` console entry declared in `pyproject.toml` points to a `main` function that is not implemented; use the Python workflow or thin notebook above.

## How it works

1. Download completed-trial snapshots from PostgreSQL view `ml.trial_snapshot_latest`.
2. Load the local CSV, remove duplicate rows and rows with missing values, and parse date columns.
3. Build engagement features from the first three days and save the processed data.
4. One-hot encode country and device type with `pandas.get_dummies`.
5. Fit and evaluate both models using a stratified 75%/25% train/test split with `random_state=42`.
6. Print and export train metrics followed by test metrics: ROC-AUC, PR-AUC (average precision), confusion matrices, and classification reports.
7. Save each fitted model and its evaluation report with a matching timestamp.

Logistic regression uses a `StandardScaler` pipeline and `max_iter=1000`. XGBoost uses 400 estimators, depth 3, learning rate 0.05, minimum child weight 8, and row/column sampling of 0.9.

## Data and features

The recorded notebook run contains **1,516 completed trials**, with a **53.23% conversion rate**. The target is `converted`: `1` means conversion and `0` means cancellation/non-conversion.

The input CSV requires these columns:

```text
trial_id, user_id, snapshot_date, trial_started_at, country, device_type,
sessions_day1, sessions_day2, sessions_day3, listen_sessions_3d,
total_minutes_3d, converted
```

| Model feature | Meaning |
| --- | --- |
| `sessions_3d` | Total sessions across days 1–3. |
| `active_days_3d` | Number of those days with at least one session. |
| `day1_share` | Day-1 sessions divided by total sessions. |
| `listen_share` | Listening sessions divided by total sessions. |
| `avg_session_minutes` | Total minutes divided by total sessions. |
| `total_minutes_3d` | Total engagement time during days 1–3. |
| `country`, `device_type` | One-hot encoded categorical attributes. |

Undefined shares and average durations for zero-session trials are filled with zero. Identifiers and dates are not model inputs.

## Evaluation: train first, then test

After fitting, each model prints a report and saves the same report to a text file. Each report contains **Train Data**, followed by **Test Data**, with:

| Metric | Meaning |
| --- | --- |
| ROC-AUC | How well conversion probabilities rank converters above non-converters. |
| PR-AUC (average precision) | Precision-recall performance for conversion, calculated with `average_precision_score` from predicted probabilities, without interpolation. |
| Confusion matrix | Counts of correct and incorrect predictions. Rows are actual labels; columns are predicted labels. |
| Classification report | Precision, recall, F1-score, support, and accuracy using the model's predicted labels. |

Train metrics describe fit to the training data. Test metrics describe performance on held-out trials. Comparing them helps identify a gap between training and test performance.

### Saved results

These values come from the **September 14, 2026** reports linked below, not a new training run. Both models use **1,137 train trials** and **379 test trials**. The test set contains 177 non-converters and 202 converters.

| Split | Model | ROC-AUC | PR-AUC (average precision) | Accuracy |
| --- | --- | --- | --- | --- |
| Train | Logistic regression | 0.8081 | 0.8127 | 0.74 |
| Train | XGBoost | 0.9305 | 0.9413 | 0.85 |
| Test | Logistic regression | 0.8274 | 0.8097 | 0.76 |
| Test | XGBoost | 0.8745 | 0.8849 | 0.79 |

The confusion matrices from the same run are summarized below. **TN** means correctly predicted non-conversion, **FP** means incorrectly predicted conversion, **FN** means incorrectly predicted non-conversion, and **TP** means correctly predicted conversion.

| Split | Model | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- |
| Train | Logistic regression | 330 | 202 | 88 | 517 |
| Train | XGBoost | 453 | 79 | 93 | 512 |
| Test | Logistic regression | 116 | 61 | 29 | 173 |
| Test | XGBoost | 138 | 39 | 39 | 163 |

XGBoost has higher test ROC-AUC and average precision in this run. Its stronger train scores also show a train/test gap that warrants validation on later cohorts.

Full reports: [logistic regression](models/logistic_regression_metrics_2026-09-14_07-13-35.txt) · [XGBoost](models/xgboost_metrics_2026-09-14_07-13-35.txt).

## Where to find things

| Location | Purpose |
| --- | --- |
| [Workflow notebook](notebooks/01_thin_notebook.ipynb) | Run the reusable package functions. |
| [Original notebook](trial_conversion_model.ipynb) | Explore the data and targeting example. |
| [data.py](src/trial_conversion_model/data.py) | Download and clean data. |
| [features.py](src/trial_conversion_model/features.py) | Build features and export the processed CSV. |
| [train.py](src/trial_conversion_model/train.py) | Fit models, report train/test metrics, and save outputs. |
| [models/](models/) | Saved models and evaluation reports. |
| [.env.example](.env.example) | Database configuration template. |
| [Model plan](trial-conversion-model-plan.md) | Business context and proposed deployment. |

## Generated files

| Output | Location / format |
| --- | --- |
| Raw snapshot | `data/01_raw/trials_raw.csv` |
| Engineered dataset | `data/03_processed/trials_clean.csv` |
| Logistic regression pipeline | `models/logistic_regression_<timestamp>.joblib` |
| XGBoost model | `models/xgboost_<timestamp>.json` |
| Train and test metrics, one report per model | `models/<model_name>_metrics_<timestamp>.txt` |

Each metrics file contains train results first, then test results, matching the console output. The saved models do not include feature engineering or categorical encoding. Inference must reproduce the training features and encoded column order.

## Limitations and next steps

- Establish a direct comparison against the raw-session-count heuristic required by the model plan; the current logistic regression baseline uses the full feature set.
- Validate on later trial cohorts and assess probability calibration before using summed scores for financial forecasts. Current results use a single random holdout.
- Implement live day-3 scoring, daily lifecycle lists, and on-demand access with engineering.
- Monitor performance and population drift as launch-campaign acquisition changes, and define retraining criteria.
- Add automated tests. `tests/test_example.py` currently downloads data and trains logistic regression at module execution; it is a workflow example with side effects, not an isolated unit-test suite.

Existing paid-subscriber churn, uplift modeling, and changes to trial length, pricing, or eligibility are outside the current scope. See the [full model plan](trial-conversion-model-plan.md) for the proposed business impact and deployment roadmap.

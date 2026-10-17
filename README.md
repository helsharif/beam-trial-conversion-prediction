# Beam Trial Conversion Prediction

Predict whether a **14-day free trial will convert to a paid subscription**, using the first three days of app activity.

This Python prototype compares **logistic regression** and **XGBoost**. It prepares data, builds engagement features, trains both models, and reports performance on **train data first, then test data**.

Developed by Husayn El Sharif for the FutureProof DS ML Cohort 2026, beginning with the Week 1 assignment. **Status: model training prototype, a prediction service deployed and tested on AWS EC2, and a local Streamlit dashboard verified against S3 prediction and drift artifacts.** This checkout includes training and dashboard code; the deployed API and artifact-producing jobs live outside this checkout.

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

If you already have an authorized CSV, place it at `data/01_raw/trials_raw.csv` and use the existing-CSV workflow below. The required schema is listed under **Data and features**.

### 3. Train both models

Run the training script from the repository root:

```bash
uv run scripts/train.py
```

The script downloads a fresh snapshot, cleans the data, builds features, and trains logistic regression followed by XGBoost. Each model prints and exports **train metrics first, then test metrics**, and saves its fitted model in `models/`.

The script always calls `download_data()`, replacing `data/01_raw/trials_raw.csv`. It requires database access and has no option to skip the download.

### Alternative: train from an existing CSV

To use the local snapshot without downloading it again, start Python with `uv run python`, then run:

```python
from trial_conversion_model.data import load_data
from trial_conversion_model.features import build_features
from trial_conversion_model.training import (
    train_model_logistic_regression,
    train_model_xgboost,
)

df = load_data()
df = build_features(df)

lr_model = train_model_logistic_regression(df)
xgb_model = train_model_xgboost(df)
```

### Alternative: use a notebook

For interactive work, open [notebooks/01_thin_notebook.ipynb](notebooks/01_thin_notebook.ipynb) in a notebook-capable editor and select the project's `.venv` Python kernel. The notebook locates the project root before running the workflow. The [original notebook](trial_conversion_model.ipynb) contains the exploratory analysis, its own database connection setup, and the illustrative intervention cutoff.

The `trial-conversion-model` console entry declared in `pyproject.toml` still points to a package `main` function that is not implemented. Use `uv run scripts/train.py` for the full workflow.

### Start the MLflow tracking server

With MLflow installed in the project's environment, run this command from the repository root:

```powershell
uv run mlflow server --host 127.0.0.1 --port 5001 --workers 1
```

Open [the MLflow UI](http://127.0.0.1:5001) in your browser. Keep the terminal running while using the server; press **Ctrl+C** to stop it.

The `--workers 1` option uses a single server worker to work around the Windows multi-worker startup failure (`WinError 10022`). It applies only to this command and requires no persistent environment-variable changes.

## How it works

`scripts/train.py` coordinates the workflow; the reusable data, feature, and training functions live in `src/trial_conversion_model/`.

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

Full reports: [logistic regression](models/logistic_regression_metrics_2026-09-14_11-04-07.txt) · [XGBoost](models/xgboost_metrics_2026-09-14_11-04-07.txt).

## AWS EC2 prediction service

On **October 3, 2026**, the prediction service was confirmed working on AWS EC2. A request from a local Windows PowerShell session reached the remote `/predict` endpoint and returned a conversion probability and band.

### Deployment accomplishments

- Packaged the service for Linux using Docker, Python 3.13, and `uv`. The Dockerfile includes `libgomp1`, the OpenMP runtime required by XGBoost in the slim Python image.
- Included the application source and trained XGBoost artifact (`models/model.json`) in the image. The service loads the saved model for inference; it does not train a model when serving requests.
- Configured Uvicorn to run `trial_conversion_model.api.main:app` on `0.0.0.0:8000`.
- Built and published the image to a **private GitHub Container Registry (GHCR) package**, then authenticated to GHCR from EC2 to pull and run it.
- Completed the EC2 launch workflow using Docker in detached mode, port mapping `8000:8000`, and the `unless-stopped` restart policy, with Docker enabled at boot.
- Verified remote prediction requests from PowerShell using JSON request bodies and displayed the raw JSON response with `Invoke-WebRequest`.

The deployment used the Dockerfile and API implementation discussed during deployment work. **This checkout does not yet contain that Dockerfile, `src/trial_conversion_model/api/`, or `models/model.json`.** The training quick start above is supported by the files here; reproducing the container build also requires bringing those deployment files into this repository.

### Verified prediction example

Replace `YOUR_EC2_PUBLIC_IP` with the running instance's current public address. Run this in **PowerShell**:

```powershell
$body = @{
    sessions_day1      = 3
    sessions_day2      = 2
    sessions_day3      = 2
    listen_sessions_3d = 5
    total_minutes_3d   = 150
    country           = "US"
    device_type       = "iOS"
} | ConvertTo-Json

$request = @{
    Uri             = "http://YOUR_EC2_PUBLIC_IP:8000/predict"
    Method          = "POST"
    ContentType     = "application/json"
    Body            = $body
    UseBasicParsing = $true
}
(Invoke-WebRequest @request).Content
```

The recorded response for this input was:

```json
{"conv_prob":0.848,"conv_band":"high"}
```

`conv_prob` is the predicted conversion probability; `conv_band` is the service's assigned band. This result validates the deployed request-to-prediction path. The value may change when the deployed model changes.

## Where to find things

| Location | Purpose |
| --- | --- |
| [scripts/train.py](scripts/train.py) | Run the full download, preparation, and training workflow. |
| [Workflow notebook](notebooks/01_thin_notebook.ipynb) | Run the reusable package functions. |
| [Original notebook](trial_conversion_model.ipynb) | Explore the data and targeting example. |
| [data.py](src/trial_conversion_model/data.py) | Download and clean data. |
| [features.py](src/trial_conversion_model/features.py) | Build features and export the processed CSV. |
| [training.py](src/trial_conversion_model/training.py) | Fit models, report train/test metrics, and save outputs. |
| [models/](models/) | Saved models and evaluation reports. |
| [streamlit_app.py](streamlit_app.py) | S3 dashboard: latest ranked trials, drift verdict, and monthly history. |
| [dashboard_data.py](src/trial_conversion_model/dashboard_data.py) | Read-only S3 access, artifact validation, ranking, and drift report selection. |
| [test_dashboard.py](tests/test_dashboard.py) | Isolated dashboard data tests. |
| [.streamlit/config.toml](.streamlit/config.toml) | Dashboard theme and localhost binding. |
| [.env.example](.env.example) | Database and optional S3 dashboard configuration template. |
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
- Bring the deployed Dockerfile, API source, and serving model artifact into this repository so the EC2 deployment can be reproduced from a fresh clone.
- Integrate the working EC2 prediction endpoint with live day-3 scoring and daily lifecycle lists with engineering.
- Extend the S3 drift dashboard with observed performance once labels arrive, and define retraining criteria. The dashboard consumes existing reports; it does not run scoring, calculate drift, or schedule upstream jobs.
- Expand automated coverage beyond the five dashboard data tests. `tests/test_example.py` is a workflow example with download and training side effects, not an isolated unit-test suite. Its training import still references the old module name and needs updating before use.

Existing paid-subscriber churn, uplift modeling, and changes to trial length, pricing, or eligibility are outside the current scope. See the [full model plan](trial-conversion-model-plan.md) for the proposed business impact and deployment roadmap.


## Streamlit S3 dashboard

Run from this checkout:

```powershell
uv sync --locked
uv run streamlit run streamlit_app.py
```

Open [the dashboard](http://localhost:8501). If port 8501 is occupied, run
`uv run streamlit run streamlit_app.py --server.port 8502` and open
[port 8502](http://localhost:8502), as used during local verification. Keep the
terminal running while using the dashboard; press **Ctrl+C** to stop it.

The dashboard reads `predictions/scored_YYYY-MM-DD.csv`
and dated JSON reports under `monitoring/` using read-only S3 operations.
Configure `S3_BUCKET`, `AWS_DEFAULT_REGION`, `AWS_ACCESS_KEY_ID`, and
`AWS_SECRET_ACCESS_KEY` in `.env` (plus `AWS_SESSION_TOKEN` for temporary credentials).
Alternatively, set `DASHBOARD_ENV_FILE` in the local `.env` to the absolute path of
an existing S3 `.env`. Credentials are read at runtime and never displayed.
The standard AWS credential chain/profile is also supported. Process environment
values take precedence over local `.env`, which takes precedence over the external
file. Permissions required: `s3:ListBucket` on the bucket and `s3:GetObject` on
`predictions/*` and `monitoring/*`. No writes to S3 are performed.

The latest list ranks lowest conversion probability first, with country/device
filters, an adjustable outreach cutoff, and CSV download. Latest means the date
in the artifact filename, independently for predictions and monitoring; upload
times only break ties. A warning identifies mismatched dates or old predictions.
The default history covers 31 calendar days ending at the latest artifact date;
a calendar-month selector is also available. Daily charts and a downloadable
table show volume, mean predicted conversion, and drift. These are model scores,
not observed conversion performance. Missing days remain missing.

Monitoring JSON must contain boolean `drift` and numeric `drifted_share` (0–1).
Reports marked `source: backfill` are labeled synthetic; measured reports take
precedence on overlapping dates. Invalid artifacts are reported rather than
silently treated as healthy. CSVs require `conversion_probability` (or `conv_prob`)
with values between 0 and 1. S3 reads are cached for five minutes; use **Refresh
from S3** to fetch immediately. The server binds to localhost by default.

Run isolated dashboard tests without triggering the legacy training example:

```powershell
uv run python -m unittest discover -s tests -p test_dashboard.py
```

### Verification completed October 17, 2026

- All five isolated tests passed: ranking and identifier preservation, invalid probabilities, invalid verdicts, measured-report precedence, and paginated artifact discovery using filename dates.
- Streamlit AppTest checks against live S3 passed for the default history, calendar-month selection, outreach filtering, and missing-day handling.
- The latest snapshot contained 120 trials, a 34.8% mean predicted conversion probability, and 91 candidates below the 0.50 outreach cutoff. The October 17 measured drift report flagged 100% of features as drifted.
- The default window contained 31 days of predictions and six days with drift detected after preferring measured reports over backfill. These are verification-time results and will change as S3 artifacts change.
- The local HTTP health check passed on port 8502. The dashboard was then stopped; it starts only when launched with the command above.

The Git-tracked configuration contains no credentials or machine-specific `.env`
path. Each checkout must supply its own ignored `.env` or AWS runtime credentials.

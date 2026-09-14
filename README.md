# Beam Trial Conversion Prediction

Predict whether a 14-day free trial will convert to a paid subscription using the first three days of app activity. This Python machine learning prototype compares logistic regression and XGBoost, with reusable data preparation, feature engineering, training, and evaluation functions.

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

## Current implementation

1. Download completed-trial snapshots from PostgreSQL view `ml.trial_snapshot_latest`.
2. Load the local CSV, remove duplicate rows and rows with missing values, and parse date columns.
3. Build engagement features from the first three days and save the processed data.
4. One-hot encode country and device type with `pandas.get_dummies`.
5. Fit and evaluate both models using a stratified 75%/25% train/test split with `random_state=42`.
6. Print ROC AUC and save timestamped models, confusion matrices, and classification reports.

Logistic regression uses a `StandardScaler` pipeline and `max_iter=1000`. XGBoost uses 400 estimators, depth 3, learning rate 0.05, minimum child weight 8, and row/column sampling of 0.9.

### Data and features

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

### Recorded evaluation

These results come from the saved notebook outputs and the September 13, 2026 metric files in `models/`; they are not a newly executed benchmark. Both models use a holdout of 379 trials: 177 non-converters and 202 converters.

| Model | ROC AUC | Accuracy | Non-converter recall | Converter recall |
| --- | --- | --- | --- | --- |
| Logistic regression | 0.8274 | 0.76 | 0.66 | 0.86 |
| XGBoost | 0.8745 | 0.79 | 0.78 | 0.81 |

The original exploratory notebook also illustrates a conversion-probability cutoff of 0.35: 152 of 379 holdout trials are flagged, with 17.1% actual conversion among flagged trials versus 77.5% among the rest. This is an illustrative targeting analysis, not a measured intervention effect or an approved production cutoff.

## Repository layout

```text
.
├── README.md
├── trial-conversion-model-plan.md       # Business context and roadmap
├── trial_conversion_model.ipynb         # Original exploration and modeling
├── notebooks/
│   └── 01_thin_notebook.ipynb           # Workflow using the reusable package
├── src/trial_conversion_model/
│   ├── __init__.py                      # Package version
│   ├── data.py                          # Database download and cleaning
│   ├── features.py                      # Feature construction and CSV export
│   └── train.py                         # Model fitting, evaluation, persistence
├── data/
│   ├── 01_raw/                          # Downloaded snapshot (CSV ignored)
│   ├── 02_interim/                      # Placeholder
│   ├── 03_processed/                    # Engineered dataset (CSV ignored)
│   └── 04_predictions/                  # Placeholder
├── models/                             # Saved models and metric reports
├── scripts/                            # Placeholder
├── tests/test_example.py                # Executable workflow example
├── .env.example                        # Database configuration template
├── pyproject.toml                      # Dependencies and package metadata
└── uv.lock                             # Locked dependency versions
```

## Setup and usage

Use **Python 3.13 or newer** and `uv`. Run commands from the repository root; the package reads and writes relative data/model paths.

```bash
git clone https://github.com/helsharif/beam-trial-conversion-prediction.git
cd beam-trial-conversion-prediction
uv sync --locked
```

The dependencies include pandas, NumPy, scikit-learn, XGBoost, Matplotlib, SQLAlchemy, psycopg2, and python-dotenv. The development group includes `ipykernel` for notebook execution.

### Configure data access

Copy `.env.example` to `.env` and supply valid database connection values. In PowerShell:

```powershell
Copy-Item .env.example .env
```

The loader reads `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, and `DB_PASSWORD`. Access to `ml.trial_snapshot_latest` is required to download data. `.env` and the raw/processed CSV snapshots are ignored by Git; a fresh clone does not include the dataset.

Alternatively, place an authorized CSV with the schema above at `data/01_raw/trials_raw.csv` and skip `download_data()`.

### Run the package workflow

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

### Generated outputs

| Output | Location / format |
| --- | --- |
| Raw snapshot | `data/01_raw/trials_raw.csv` |
| Engineered dataset | `data/03_processed/trials_clean.csv` |
| Logistic regression pipeline | `models/logistic_regression_<timestamp>.joblib` |
| XGBoost model | `models/xgboost_<timestamp>.json` |
| Evaluation reports | `models/<model_name>_metrics_<timestamp>.txt` |

ROC AUC is printed during training; the text reports contain confusion matrices and classification metrics. The saved estimators do not package feature engineering or categorical encoding. Future inference must reproduce the training features and encoded column order.

## Limitations and next steps

- Establish a direct comparison against the raw-session-count heuristic required by the model plan; the current logistic regression baseline uses the full feature set.
- Validate on later trial cohorts and assess probability calibration before using summed scores for financial forecasts. Current results use a single random holdout.
- Implement live day-3 scoring, daily lifecycle lists, and on-demand access with engineering.
- Monitor performance and population drift as launch-campaign acquisition changes, and define retraining criteria.
- Add automated tests. `tests/test_example.py` currently downloads data and trains logistic regression at module execution; it is a workflow example with side effects, not an isolated unit-test suite.

Existing paid-subscriber churn, uplift modeling, and changes to trial length, pricing, or eligibility are outside the current scope. See the [full model plan](trial-conversion-model-plan.md) for the proposed business impact and deployment roadmap.

## Keywords

Trial conversion prediction, subscription analytics, customer engagement, growth analytics, binary classification, feature engineering, logistic regression, XGBoost, scikit-learn, PostgreSQL, Python, and machine learning.

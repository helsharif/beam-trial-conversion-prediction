"""Training logic (the real work, kept out of the thin script).

`train_model_logistic_regression` takes data in and returns a fitted logistic regression model. 
`train_model_xgboost` takes data in and returns a fitted XGBoost model. 

Keeping these as functions means scripts, notebooks, tests, and pipelines can all
call it without copying code.
"""

# Imports
import pandas as pd

from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from xgboost import XGBClassifier


def metrics_to_file(model, X_train, y_train, X_test, y_test, model_name, timestamp):
    """
    Print and save train metrics followed by test metrics for a fitted model.

    ROC-AUC and PR-AUC use positive-class probabilities. PR-AUC is reported
    as average precision, without interpolation. Confusion matrices
    and classification reports use the model's predicted labels.
    """
    reports = []
    for split_name, X_split, y_split in (
        ("Train", X_train, y_train),
        ("Test", X_test, y_test),
    ):
        probabilities = model.predict_proba(X_split)[:, 1]
        predictions = model.predict(X_split)
        roc_auc = roc_auc_score(y_split, probabilities)
        pr_auc = average_precision_score(y_split, probabilities)
        cm = confusion_matrix(y_split, predictions, labels=[0, 1])
        cr = classification_report(
            y_split, predictions, labels=[0, 1], zero_division=0
        )

        reports.append(
            f"{model_name} - {split_name} Data\n"
            f"ROC-AUC: {roc_auc:.4f}\n"
            f"PR-AUC (average precision): {pr_auc:.4f}\n\n"
            "Confusion Matrix:\n\n"
            f"{'':15}{'Predicted 0':>12}{'Predicted 1':>12}\n"
            f"{'Actual 0':15}{cm[0, 0]:>12}{cm[0, 1]:>12}\n"
            f"{'Actual 1':15}{cm[1, 0]:>12}{cm[1, 1]:>12}\n"
            f"\nClassification Report:\n{cr}"
        )

    report = "\n\n".join(reports)
    print(report)

    # Create models directory if needed
    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"{model_name}_metrics_{timestamp}.txt"

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Metrics saved to: {metrics_path.resolve()}")


def train_model_logistic_regression(df: "pd.DataFrame") -> "LogisticRegression":
    """
    Train a logistic regression model on the provided dataframe.
    Returns the fitted pipeline containing the scaler and model.
    Prints and exports evaluation metrics for train data, then test data.
    """
    FEATURES = [
        "sessions_3d",
        "active_days_3d",
        "day1_share",
        "listen_share",
        "avg_session_minutes",
        "total_minutes_3d",
        "country",
        "device_type",
    ]

    X = pd.get_dummies(df[FEATURES], columns=["country", "device_type"])
    y = df["converted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    # Create and fit pipeline
    lr_pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    lr_pipeline.fit(X_train, y_train)

    # Create models directory if needed
    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # File names
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = "logistic_regression"

    model_path = output_dir / f"{model_name}_{timestamp}.joblib"

    # Save fitted pipeline
    joblib.dump(lr_pipeline, model_path)
    print(f"Model saved to:   {model_path.resolve()}")
    
    # Save evaluation metrics
    metrics_to_file(
        lr_pipeline, X_train, y_train, X_test, y_test, model_name, timestamp
    )

    return lr_pipeline



def train_model_xgboost(df: "pd.DataFrame") -> "XGBClassifier":
    """
    Train an XGBoost model on the provided dataframe.
    Returns the fitted model and reports train metrics followed by test metrics.
    """
    FEATURES = [
        "sessions_3d",
        "active_days_3d",
        "day1_share",
        "listen_share",
        "avg_session_minutes",
        "total_minutes_3d",
        "country",
        "device_type",
    ]

    features = pd.get_dummies(df[FEATURES], columns=["country", "device_type"])
    target = df["converted"]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        stratify=target,
        random_state=42,
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=8,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="auc",
    )

    model.fit(X_train, y_train)

    # Create output directory if needed
    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = "xgboost"
    model_path = output_dir / f"{model_name}_{timestamp}.json"


    # Save model
    model.save_model(model_path)
    print(f"Model saved to:   {model_path.resolve()}")


    # Save evaluation metrics
    metrics_to_file(
        model, X_train, y_train, X_test, y_test, model_name, timestamp
    )


    return model

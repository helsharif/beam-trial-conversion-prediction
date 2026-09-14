"""Thin entry point: train the model and save it.

    uv run scripts/train.py

Production runs a script, never a notebook. All the real logic lives in
src/trial_conversion_model/, so this stays a few lines. Implement the package's stubs, then
this runs end to end.
"""

# imports
from trial_conversion_model.data import download_data, load_data
from trial_conversion_model.features import build_features
from trial_conversion_model.training import (
    train_model_logistic_regression,
    train_model_xgboost,
)

# main function
def main():
    # Download Data
    download_data()

    # load data
    df = load_data()

    # build features
    df_with_features = build_features(df)

    # train logistic regression
    train_model_logistic_regression(df_with_features)

    # train xgboost
    train_model_xgboost(df_with_features)

# entry point
if __name__ == "__main__":
    main()

"""Example test. Because your logic lives in importable functions (not notebook
cells), it's testable without opening a notebook. Replace this with real tests
of your features and training as you build them out.
"""

# imports
import pandas as pd
from trial_conversion_model.data import download_data, load_data
from trial_conversion_model.features import build_features

from trial_conversion_model.train import train_model_logistic_regression

##
download_data()

df2 = load_data()

print(df2.head())

df3 = build_features(df2)

print(df3.head())

# logistic regression 
lr_model = train_model_logistic_regression(df3)
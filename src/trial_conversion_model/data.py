"""Data downloading, loading, and basic cleaning.
"""

# Imports
import pandas as pd
import numpy as np

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

# downloading data from a database
def download_data():
    """
    """
    # Load environment variables from .env file
    load_dotenv()
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD =os.getenv("DB_PASSWORD")

    # create engine to connect to the database
    engine = create_engine(
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    # query the database for the trial snapshot data
    df = pd.read_sql("SELECT * FROM ml.trial_snapshot_latest", engine)

    # save a snapshot to csv in raw data folder
    df.to_csv("data/01_raw/trials_raw.csv", index=False)


# load and clean the data
def load_data() -> "pd.DataFrame":
    """
    load the raw data, clean it, and return dataframe with cleaned data. 
    """
    # load the raw data
    df = pd.read_csv("data/01_raw/trials_raw.csv")

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # drop rows with missing values
    df.dropna(inplace=True)

    # dates load as strings; make them real dates
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["trial_started_at"] = pd.to_datetime(df["trial_started_at"])

    # return the cleaned dataframe
    return df

    
    

    

    


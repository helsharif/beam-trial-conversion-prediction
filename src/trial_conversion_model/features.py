"""Feature engineering.

Keep feature logic here as plain, testable functions, so your notebook and your
training script share one source of truth. Add features one small function at a
time, and test each one in tests/.
"""
# imports
import pandas as pd

# build features
def build_features(data: "pd.DataFrame") -> "pd.DataFrame": # means that the function takes a pandas DataFrame (cleaned data) as input and returns a pandas DataFrame as output
    """
    Build features for the trial conversion model.

    Args:
        data (pd.DataFrame): The input dataframe containing cleaned trial data.
    """
    data = data.copy(deep=True) # create a copy of the input dataframe to avoid modifying the original data
    data["sessions_3d"] = data[["sessions_day1", "sessions_day2", "sessions_day3"]].sum(axis=1) # sum the number of sessions over the first three days of the trial
    data["active_days_3d"] = (data[["sessions_day1", "sessions_day2", "sessions_day3"]] > 0).sum(axis=1) # count the number of days with at least one session over the first three days of the trial. Max value is 3, min value is 0.
    data["day1_share"] = data["sessions_day1"] / data["sessions_3d"] # calculate the share of sessions that occurred on day 1 relative to the total number of sessions over the first three days. This feature captures how much of the user's engagement happened on the first day of the trial.
    data["listen_share"] = data["listen_sessions_3d"] / data["sessions_3d"] # calculate the share of sessions that were listening sessions relative to the total number of sessions over the first three days. This feature captures how much of the user's engagement was focused on listening.
    data["avg_session_minutes"] = data["total_minutes_3d"] / data["sessions_3d"] # calculate the average session length in minutes over the first three days of the trial. This feature captures how long users are spending in each session on average.

    # Some trials never had a single session in the first 3 days, so the share and average columns divide by zero and come out as NaN. 
    # Zero engagement is real information, so those become zeros rather than dropped rows.
    for col in ["day1_share", "listen_share", "avg_session_minutes"]:
        data[col] = data[col].fillna(0)

    # save the updated dataframe to a CSV file
    data.to_csv("data/03_processed/trials_clean.csv", index=False)

    return data


import pandas as pd


def load_and_prepare_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Intensity"] = df["Calories_Burned"] / df["Session_Duration (hours)"]
    return df

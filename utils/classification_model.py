from __future__ import annotations

import pandas as pd
from rich.console import Console
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def run_classification(df: pd.DataFrame, console: Console) -> None:
    """Classify workout type based on the given dataframe."""

    df_encoded = pd.get_dummies(df, drop_first=True)
    x = df_encoded.drop("Workout_Type_HIIT", axis=1, errors='ignore')
    y = df["Workout_Type"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    console.print("[cyan]Random Forest – klasyfikacja typu treningu:")
    console.print(classification_report(y_test, y_pred))

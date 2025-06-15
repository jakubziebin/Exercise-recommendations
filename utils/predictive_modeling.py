from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


class PredictiveModeller:
    TEST_SIZE: ClassVar[float] = 0.2
    RANDOM_STATE: ClassVar[int] = 42

    def __init__(self, df: pd.DataFrame, console: Console) -> None:
        self.df = df
        self.console = console

    def perform_predictions(self) -> None:
        self.predict_burned_calories()
        self.predict_fat_percentage()

    def predict_burned_calories(self) -> None:
        """Run regression model to predict calories burned."""

        x = self.df[["Age", "Intensity", "Avg_BPM", "BMI", "Session_Duration (hours)"]]
        y = self.df["Calories_Burned"]

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        x_train, x_test, y_train, y_test = train_test_split(
            x_scaled, y,
            test_size=self.TEST_SIZE,
            random_state=self.RANDOM_STATE
        )

        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        self.console.print("[bold]Przewidywanie spalonych kalorii z użyciem [bold cyan]regresji liniowej:")
        self.console.print("R²:", r2_score(y_test, y_pred))

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        self.console.print("RMSE:", rmse)

        results = pd.DataFrame({
            "Calories_Actual": y_test.values,
            "Calories_Predicted": y_pred
        })

        plt.figure(figsize=(8, 6))
        plt.scatter(results["Calories_Actual"], results["Calories_Predicted"], alpha=0.7, color='royalblue')
        plt.plot([results["Calories_Actual"].min(), results["Calories_Actual"].max()],
                 [results["Calories_Actual"].min(), results["Calories_Actual"].max()],
                 color='red', linestyle='--', linewidth=2, label='Rzeczywiste dane')

        plt.xlabel("Rzeczywiste kalorie")
        plt.ylabel("Przewidziane kalorie")
        plt.title("Predykcja kalorii vs rzeczywiste wartości")
        plt.legend()
        plt.grid(True)

        plt.savefig("plots/burned_calories_regression.png")
        plt.close()

    def predict_fat_percentage(self) -> None:
        """Predict fat percentage based on available data."""

        df = self.df.copy()

        for col in ["Gender", "Experience_Level"]:
            df[col] = LabelEncoder().fit_transform(df[col])

        features = [
            "Age", "Gender", "BMI", "Weight (kg)", "Height (m)",
            "Workout_Frequency (days/week)", "Experience_Level",
            "Avg_BPM", "Calories_Burned", "Session_Duration (hours)"
        ]

        x = df[features]
        y = df["Fat_Percentage"]

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=self.TEST_SIZE,
            random_state=self.RANDOM_STATE
        )

        model = RandomForestRegressor(n_estimators=100, random_state=self.RANDOM_STATE)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        self.console.print("\nPrzewidywanie poziomu tkanki tłusczowej [old cyan]regresji liniowej:")
        self.console.print(f"R²: {r2:.4f}")
        self.console.print(f"RMSE: {rmse:.2f}")

        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.7, color="darkgreen")
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
        plt.xlabel("Rzeczywista zawartość tłuszczu (%)")
        plt.ylabel("Przewidziana zawartość tłuszczu (%)")
        plt.title("Predykcja Fat_Percentage")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("plots/fat_percentage_prediction.png")
        plt.close()

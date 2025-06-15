import os
import re

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def clean_column_name(name: str) -> str:
    """Remove or replace problematic chars in column names, to be safe as file names."""

    return re.sub(r'[^\w\-_\.]', '_', name)


def create_all_plots(df: pd.DataFrame) -> None:
    """Create plots if not exist."""

    os.makedirs("plots", exist_ok=True)
    os.makedirs("additional_plots", exist_ok=True)

    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Histogram: {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        filename = clean_column_name(col)
        plt.savefig(f"plots/hist_{filename}.png")
        plt.close()

    if "Workout_Type" in df.columns and "Calories_Burned" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Workout_Type", y="Calories_Burned", data=df)
        plt.title("Calories Burned by Workout Type")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("plots/boxplot_calories_workout_type.png")
        plt.close()

    if "Gender" in df.columns and "Calories_Burned" in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x="Gender", y="Calories_Burned", data=df)
        plt.title("Calories Burned by Gender")
        plt.tight_layout()
        plt.savefig("plots/boxplot_calories_gender.png")
        plt.close()

    plt.figure(figsize=(12, 10))
    corr = df.select_dtypes(include='number').corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("plots/correlation_heatmap.png")
    plt.close()

    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        plt.figure()
        df[col].value_counts().plot(kind='bar')
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        filename = clean_column_name(col)
        plt.tight_layout()
        plt.savefig(f"plots/bar_{filename}.png")
        plt.close()

    if "Intensity" in df.columns:
        plt.figure()
        sns.histplot(df["Intensity"], kde=True, bins=30)
        plt.title("Histogram: Intensity")
        plt.xlabel("Intensity (Calories per Hour)")
        plt.ylabel("Frequency")
        plt.savefig("plots/hist_Intensity.png")
        plt.close()

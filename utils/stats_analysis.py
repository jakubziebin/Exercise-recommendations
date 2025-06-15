from __future__ import annotations

import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, shapiro
from rich.console import Console


class StatisticalAnalyser:
    def __init__(self, df: pd.DataFrame, console: Console) -> None:
        self.df = df
        self.console = console

    def perform_analys(self) -> None:
        self.perform_shapiro_wilk()
        self.perform_mann_whitney_u_test_on_calories_burner()
        self.perform_chi_square_test_gender_workout_type()

    def perform_shapiro_wilk(self) -> None:
        """Checks whether the data have a normal distribution."""

        self.console.print("\nSprawdzamy, czy dane mają rozkład normalny -> [bold magenta]Test shapiro wilk[/bold magenta]")

        for col in ['Calories_Burned', 'Avg_BPM', 'BMI', 'Intensity']:
            stat, p = shapiro(self.df[col])

            self.console.print(
                f"Dla {col}: [bold cyan]p={p}[/bold cyan],"
                f" {'posiada' if p > 0.05 else 'nie posiada'} rozkładu normalnego"
            )

    def perform_mann_whitney_u_test_on_calories_burner(self) -> None:
        """
        Compares calories burned by males and females.
        Checks whether difference is important.
        """

        df = self.df
        male = df[df["Gender"] == "Male"]["Calories_Burned"]
        female = df[df["Gender"] == "Female"]["Calories_Burned"]

        u_stat, p_val = mannwhitneyu(male, female, alternative='two-sided')

        self.console.print("\nCzy płeć ma wpływ na ilość spalonych kalorii?")
        self.console.print(f"Mann–Whitney U test: [bold cyan]p={p_val:.4e}[/bold cyan]")
        self.console.print("Różnice w spalanych kaloriach, [bold red]są[/bold red] istotne w zależności od płci")

    def perform_chi_square_test_gender_workout_type(self) -> None:
        """Find out if gender is the type of training is related to gender."""

        self.console.print("\nCzy istnieje statystycznie istotna zależność między płcią, a typem treningu?")
        df = self.df
        contingency_table = pd.crosstab(df['Gender'], df['Workout_Type'])

        chi2, p, dof, expected = chi2_contingency(contingency_table)

        self.console.print(contingency_table)

        self.console.print(f"\nStatystyka chi² = {chi2:.2f}")
        self.console.print(f"p-value = {p:.4e}")

        if p < 0.05:
            self.console.print(
                "\n"
                "Wniosek: [bold green]Istnieje[/bold green] "
                "istotna statystycznie zależność między płcią a typem treningu."
            )
        else:
            self.console.print(
                "\n"
                "Wniosek: [bold red]Nie istnieje[/bold red] "
                "istotna statystycznie zależność między płcią a typem treningu."
            )

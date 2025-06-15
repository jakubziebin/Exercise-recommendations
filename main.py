from __future__ import annotations

from rich.console import Console

from utils.stats_analysis import StatisticalAnalyser
from utils.predictive_modeling import PredictiveModeller
from utils.visualization import create_all_plots
from utils.data_preprocessing import load_and_prepare_data
from utils.classification_model import run_classification


if __name__ == "__main__":
    df = load_and_prepare_data("data/gym_members_exercise_tracking.csv")
    console = Console()

    while True:
        console.rule(
            "Dostępne opcje:\n"
            "1. Analiza danych\n"
            "2. Predykcja danych\n"
            "3. Klasyfikacja typu treningu\n"
            "4. Stwórz wizualizację\n"
            "5. Wyjdź.\n"
        )
        user_input = int(console.input("Wybierz opcję: "))

        match user_input:
            case 1:
                console.rule("[bold cyan]Analiza danych")
                StatisticalAnalyser(df, console).perform_analys()
            case 2:
                console.rule("[bold cyan]Część predykcyjna")
                PredictiveModeller(df, console).perform_predictions()
            case 3:
                console.rule("[bold cyan]Klasyfikacja typu treningu")
                run_classification(df, console)
            case 4:
                console.rule("[bold cyan] Tworzenie wizualizacji")
                create_all_plots(df)
            case 5:
                break
            case _:
                console.print("[bold red]Nieznana opcja!")

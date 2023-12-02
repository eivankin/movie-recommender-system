from enum import StrEnum

import typer
from rich.console import Console
from rich.table import Table

from src.utils import set_seed

console = Console()


class AvailableModels(StrEnum):
    RANDOM_BASELINE = "rng"
    MOST_POPULAR_BASELINE = "most_popular"
    SVD = "svd"


def make_metrics_table(model: AvailableModels, rmse: float = .0, precision: float = .0,
                       recall: float = .0, ndcg: float = .0, map_score: float = .0, k: int = 10) -> Table:
    table = Table(title=f"Metrics for `{model}`, k={k}")

    table.add_column("Metric", style="bold")
    table.add_column("Value", style="bold")

    table.add_row("RMSE", f"{rmse:.4f}")
    table.add_row("Precision@k", f"{precision:.4f}")
    table.add_row("Recall@k", f"{recall:.4f}")
    table.add_row("NDCG@k", f"{ndcg:.4f}")
    table.add_row("MAP@k", f"{map_score:.4f}")

    return table


def evaluate(model_name: AvailableModels, seed: int = 42) -> None:
    set_seed(seed)
    metric_results_table = make_metrics_table(model_name)

    console.print(metric_results_table)


if __name__ == "__main__":
    typer.run(evaluate)

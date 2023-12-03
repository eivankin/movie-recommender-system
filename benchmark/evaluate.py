from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src.config import SEED
from src.data.load import AvailableSplits
from src.data.movielens_dataset import MovieLensDataset
from src.evaluator import EvalOn, Evaluator, MetricReport
from src.lightfm_model import LightFMWrapper
from src.seeding import set_seed

console = Console()


def make_metrics_table(metric_report: MetricReport) -> Table:
    set_name = "test" if metric_report.is_on_test else "train"

    table = Table(title=f"Metrics for {set_name} set")

    table.add_column("Metric", style="bold")
    table.add_column("Value", style="bold")

    table.add_row(f"Precision@{metric_report.top_k}", f"{metric_report.precision:.4f}")
    table.add_row(f"Recall@{metric_report.top_k}", f"{metric_report.recall:.4f}")
    table.add_row(f"F1@{metric_report.top_k}", f"{metric_report.f1:.4f}")
    table.add_row("AUC", f"{metric_report.auc:.4f}")

    return table


def evaluate(
    model_file: Path, split: AvailableSplits, seed: int = SEED, k: int = 20
) -> None:
    set_seed(seed)
    model = LightFMWrapper.from_file(model_file)
    dataset = MovieLensDataset.from_split(split)

    evaluator = Evaluator(model.model, dataset)

    train_metrics = make_metrics_table(evaluator.evaluate(EvalOn.TRAIN, k=k))
    console.print(train_metrics)

    test_metrics = make_metrics_table(evaluator.evaluate(EvalOn.TEST, k=k))
    console.print(test_metrics)


if __name__ == "__main__":
    typer.run(evaluate)

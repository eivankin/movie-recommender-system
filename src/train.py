from pathlib import Path

import typer
from lightfm import LightFM
from tqdm.auto import trange

from src.config import SEED
from src.data.load import AvailableSplits
from src.data.movielens_dataset import MovieLensDataset
from src.evaluator import EvalOn, Evaluator, MetricReport
from src.lightfm_model import LightFMWrapper
from src.seeding import set_seed
from src.visualize import plot_metrics


def train(
    model: LightFMWrapper,
    dataset: MovieLensDataset,
    epochs: int = 20,
    seed: int = SEED,
    k: int = 20,
) -> list[MetricReport]:
    set_seed(seed)
    metric_reports = []

    for _ in (pbar := trange(epochs)):
        model.train_one_epoch(dataset)
        train_metrics = Evaluator(model.model, dataset).evaluate(EvalOn.TRAIN, k=k)
        test_metrics = Evaluator(model.model, dataset).evaluate(EvalOn.TEST, k=k)
        pbar.set_description(
            f"Test: AUC - {test_metrics.auc:.4f} | Precision@{k} - {test_metrics.precision:.4f} "
            f"| Recall@{k} - {test_metrics.recall:.4f}"
        )

        metric_reports.append(train_metrics)
        metric_reports.append(test_metrics)

    return metric_reports


def main(
    save_model_to: Path,
    dataset_split: AvailableSplits,
    epochs: int = 20,
    seed: int = SEED,
    k: int = 20,
    no_components: int = 256,
    loss: str = "warp",
    plot: bool = False,
):
    model = LightFMWrapper(
        LightFM(no_components=no_components, loss=loss, random_state=seed)
    )
    dataset = MovieLensDataset.from_split(dataset_split)
    metric_reports = train(model, dataset, epochs, seed, k)
    model.save(save_model_to)

    if plot:
        plot_metrics(metric_reports)


if __name__ == "__main__":
    typer.run(main)

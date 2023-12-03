from dataclasses import dataclass
from enum import Enum, auto

from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score

from typing import Callable

from src.data.movielens_dataset import MovieLensDataset


class EvalOn(Enum):
    TRAIN = auto()
    TEST = auto()


@dataclass
class MetricReport:
    top_k: int
    is_on_test: bool
    precision: float
    recall: float
    f1: float
    auc: float


class Evaluator:
    """
    This class wraps lightfm metrics for easier usage in the project.

    Available metrics:
    1. precision@k
    2. recall@k
    3. f1@k
    4. auc
    """

    def __init__(self, model: LightFM, dataset: MovieLensDataset):
        self.model = model
        self.dataset = dataset

    def _eval_one_metric(
        self, target: EvalOn, metric: Callable, **metric_kwargs
    ) -> float:
        return metric(
            model=self.model,
            test_interactions=(
                self.dataset.test_interactions
                if target == EvalOn.TEST
                else self.dataset.train_interactions
            ),
            train_interactions=(
                self.dataset.train_interactions if target == EvalOn.TEST else None
            ),
            user_features=self.dataset.user_features,
            item_features=self.dataset.item_features,
            **metric_kwargs
        ).mean()

    def evaluate(self, target: EvalOn, k: int = 20) -> MetricReport:
        precision = self._eval_one_metric(target, precision_at_k, k=k)
        recall = self._eval_one_metric(target, recall_at_k, k=k)
        f1 = 2 * precision * recall / (precision + recall)
        auc = self._eval_one_metric(target, auc_score)

        return MetricReport(
            top_k=k,
            is_on_test=target == EvalOn.TEST,
            precision=precision,
            recall=recall,
            f1=f1,
            auc=auc,
        )

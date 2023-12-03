import pickle
from pathlib import Path
from typing import Iterable, Self

import numpy as np
from lightfm import LightFM

from src.data.movielens_dataset import MovieLensDataset


class LightFMWrapper:
    """Wraps LightFM model for easier saving, loading, training and inference"""

    def __init__(self, model: LightFM):
        self.model = model

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """Load the model checkpoint from a file"""
        return cls(pickle.loads(path.read_bytes()))

    def save(self, path: Path) -> None:
        """
        Save the model checkpoint to a file using pickle with the highest protocol
        Inspired by: https://github.com/lyst/lightfm/issues/207
        """
        path.write_bytes(pickle.dumps(self.model, pickle.HIGHEST_PROTOCOL))

    def train_one_epoch(self, dataset: MovieLensDataset) -> None:
        """
        Calls fit_partial on the model with one epoch.
        It is slower than passing the real number of epochs to fit function,
        but allows to evaluate model on each epoch.

        Inspired by: https://making.lyst.com/lightfm/docs/examples/warp_loss.html
        """
        self.model.fit_partial(
            dataset.train_interactions,
            user_features=dataset.user_features,
            item_features=dataset.item_features,
            epochs=1,
            sample_weight=dataset.train_weights,
        )

    def make_recommendation(
        self, user_id: int, items: Iterable[int], dataset: MovieLensDataset, k: int = 20
    ) -> list[int]:
        """
        Predicts top k user recommendations.

        Accepts user id and list of item ids as in dataset (starts with 1)
        and converts it internally.

        Returns list of item ids as in dataset.

        Inspired by: https://making.lyst.com/lightfm/docs/quickstart.html
        """
        scores = self.model.predict(
            user_id - 1,
            [item_id - 1 for item_id in items],
            user_features=dataset.user_features,
            item_features=dataset.item_features,
        )

        top_items = np.argsort(-scores)

        return [item_id + 1 for item_id in top_items[:k]]

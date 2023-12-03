from pathlib import Path

import pickle

from lightfm import LightFM


class LightFMWrapper:
    def __init__(self, model: LightFM):
        self.model = model

    @classmethod
    def from_file(cls, path: Path):
        cls(pickle.loads(path.read_bytes()))

    def save(self, path: Path):
        path.write_bytes(pickle.dumps(self.model))

    def train_one_epoch(
        self, train_interactions, sample_weight, user_features, item_features
    ):
        self.model.fit_partial(
            train_interactions,
            user_features=user_features,
            item_features=item_features,
            epochs=1,
            sample_weight=sample_weight,
        )

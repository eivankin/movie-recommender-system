from typing import Self

import pandas as pd
from lightfm.data import Dataset
from scipy.sparse import coo_matrix

from src.data.load import (AvailableSplits, load_movie_data, load_train_test,
                           load_user_data)


class MovieLensDataset:
    """This class wraps lightfm dataset for easier usage in the project.

    Uses the following dataset features:
    1. User:
        - age_group (see preprocess script)
        - gender
        - occupation
    2. Movie:
        - genre
    """

    def __init__(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        user_data: pd.DataFrame,
        movie_data: pd.DataFrame,
        rating_threshold: int = 4,
    ):
        self.dataset = Dataset()
        self.fit_dataset(user_data, movie_data)
        self.train_interactions, self.train_weights = self.build_interactions(
            train_data, rating_threshold
        )
        self.test_interactions, self.test_weights = self.build_interactions(
            test_data, rating_threshold
        )
        self.user_features = self.build_user_features(user_data)
        self.item_features = self.build_item_features(movie_data)

    def fit_dataset(self, user_data, movie_data) -> None:
        """Fits dataset on given structure (ids and categorical features)."""
        self.dataset.fit(
            user_data.user_id,
            movie_data.item_id,
            user_features=[
                *user_data.age_group.unique(),
                *user_data.gender.unique(),
                *user_data.occupation.unique(),
            ],
            item_features=[
                *movie_data.genre.unique(),
            ],
        )

    @staticmethod
    def transform_rating(rating: int, threshold: int):
        """
        Transforms rating to the type of interaction (positive or negative).
        Returns 1 if rating >= threshold, -1 otherwise.
        """
        return 1 if rating >= threshold else -1

    def build_interactions(
        self, data: pd.DataFrame, rating_threshold: int
    ) -> tuple[coo_matrix, coo_matrix]:
        return self.dataset.build_interactions(
            (data["user_id"][i], data["item_id"][i])
            for i in range(len(data))
            if data["rating"][i] >= rating_threshold
        )

    def build_user_features(self, data) -> coo_matrix:
        return self.dataset.build_user_features(
            (
                data["user_id"][i],
                [data["age_group"][i], data["gender"][i], data["occupation"][i]],
            )
            for i in range(len(data))
        )

    def build_item_features(self, data) -> coo_matrix:
        return self.dataset.build_item_features(
            (data["item_id"][i], [data["genre"][i]]) for i in range(len(data))
        )

    @classmethod
    def from_split(cls, split: AvailableSplits, rating_threshold: int = 4) -> Self:
        """Loads dataset from given train/test split."""
        train_data, test_data = load_train_test(split)

        movie_data = load_movie_data()
        user_data = load_user_data()

        return cls(train_data, test_data, user_data, movie_data, rating_threshold)

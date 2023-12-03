from typing import Self

from lightfm.data import Dataset
from scipy.sparse import coo_matrix

from src.data.load import (
    load_train_test,
    AvailableSplits,
    load_movie_data,
    load_user_data,
)


class MovieLensDataset:
    # TODO: comments
    def __init__(self, train_data, test_data, user_data, movie_data):
        self.dataset = Dataset()
        self.fit_dataset(user_data, movie_data)
        self.train_interactions, self.train_weights = self.build_interactions(
            train_data
        )
        self.test_interactions, self.test_weights = self.build_interactions(test_data)
        self.user_features = self.build_user_features(user_data)
        self.item_features = self.build_item_features(movie_data)

    def fit_dataset(self, user_data, movie_data) -> None:
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

    def build_interactions(self, data) -> tuple[coo_matrix, coo_matrix]:
        return self.dataset.build_interactions(
            (data["user_id"][i], data["item_id"][i]) for i in range(len(data))
        )

    def build_user_features(self, data) -> coo_matrix:
        return self.dataset.build_user_features(
            (
                data["user_id"][i],
                [data["age_group"][i],
                 data["gender"][i],
                 data["occupation"][i]],
            )
            for i in range(len(data))
        )

    def build_item_features(self, data) -> coo_matrix:
        return self.dataset.build_item_features(
            (data["item_id"][i], [data["genre"][i]]) for i in range(len(data))
        )

    @classmethod
    def from_split(cls, split: AvailableSplits) -> Self:
        train_data, test_data = load_train_test(split)

        movie_data = load_movie_data()
        user_data = load_user_data()

        return cls(train_data, test_data, user_data, movie_data)

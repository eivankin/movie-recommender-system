from enum import StrEnum

import pandas as pd

from src.config import DATA_PATH, DATASET_PATH
from src.data.preprocess import get_genres, make_age_groups


class AvailableSplits(StrEnum):
    """
    From the dataset card:

    1. The data sets u1.base and u1.test through u5.base and u5.test are 80%/20% splits of
    the u data into training and test data. Each of u1, ..., u5 have disjoint test sets;
    this if for 5 fold cross validation
    (where you repeat your experiment with each training and test set and average the results).

    2. The data sets ua.base, ua.test, ub.base, and ub.test split the u data into
    a training set and a test set with exactly 10 ratings per user in the test set.
    The sets ua.test and ub.test are disjoint.
    """

    FIRST = "1"
    SECOND = "2"
    THIRD = "3"
    FOURTH = "4"
    FIFTH = "5"

    A = "a"
    B = "b"


def load_ratings(file_name: str) -> pd.DataFrame:
    """
    Loads ratings table from file
    Ratings are tab-separated and always have 4 columns: user_id, item_id, rating, timestamp
    """
    return pd.read_csv(
        DATASET_PATH / file_name,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )


def load_extra_features(file_name: str, columns: list[str]) -> pd.DataFrame:
    """
    Loads table with extra features, for example, movie data or genres.
    These tables have different columns but share the same encoding and structure.
    """
    return pd.read_csv(
        DATASET_PATH / file_name,
        sep="|",
        encoding="ISO-8859-1",
        header=None,
        names=columns,
    )


def load_genres() -> list[str]:
    """
    Loads genre table and then transforms it into list of movie genres
    """
    genres = load_extra_features("u.genre", ["genre", "genre_id"]).reset_index()
    return genres.genre.tolist()


def load_train_test(split: AvailableSplits) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads train and test data for given split.
    See `AvailableSplits` enum for possible values.
    """
    train_data = load_ratings(f"u{split}.base")
    test_data = load_ratings(f"u{split}.test")

    return train_data, test_data


def load_movie_data(substitute_genres: bool = True) -> pd.DataFrame:
    """
    Loads movie data and replaces one-hot encoded genres with actual genre name if `substitute_genres` is True
    """

    genres = load_genres()
    movie_data = load_extra_features(
        "u.item",
        ["item_id", "title", "release", "video_release_date", "imdb_url"] + genres,
    )

    if not substitute_genres:
        return movie_data

    movie_data_with_genres = movie_data.drop(columns=genres)
    movie_data_with_genres["genre"] = get_genres(movie_data, genres)

    return movie_data_with_genres


def load_user_data(add_age_groups: bool = True) -> pd.DataFrame:
    """
    Loads user data and adds age groups column if `add_age_groups` is True
    """
    user_data = pd.read_csv(DATA_PATH / "interim" / "users_with_coordinates.csv")
    if not add_age_groups:
        return user_data

    user_data["age_group"] = make_age_groups(user_data)
    return user_data

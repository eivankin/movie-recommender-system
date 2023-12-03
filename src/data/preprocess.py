import pandas as pd


def make_age_groups(
    user_data: pd.DataFrame, bins: list[int] | None = None
) -> pd.Series:
    """
    Makes age groups from age column, so it can be used in LightFM as a categorical feature.

    :param user_data: pd.DataFrame with user data and `age` columns
    :param bins: list of age bins (points of range split), default is [0, 18, 35, 50, 100].
    Bins are transformed to labels, default labels are ['0-18', '19-35', '36-50', '51-100'].

    :return: column (pd.Series) with age groups, age replaced with age group label
    """
    if bins is None:
        bins = [0, 18, 35, 50, 100]

    assert len(bins) >= 2

    labels = [f"{bins[i] + (i > 0)}-{bins[i + 1]}" for i in range(len(bins) - 1)]

    return pd.cut(user_data.age, bins=bins, labels=labels)


def get_genres(movie_data: pd.DataFrame, genres: list[str]) -> pd.Series:
    """
    Returns column with actual genre name from one-hot encoded data.
    """
    return movie_data[genres].idxmax(1)

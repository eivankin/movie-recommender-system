import os
import random
import numpy as np
from typing import Any


def set_seed(seed: Any) -> None:
    """
    Sets RNG seed as recommended in Surprise documentation.
    Source: https://surprise.readthedocs.io/en/stable/FAQ.html#how-to-have-reproducible-experiments
    """
    random.seed(seed)
    np.random.seed(seed)


def set_dataset_path(path: str) -> None:
    """Sets directory where data is stored by Surprise library."""
    os.environ["SURPRISE_DATA_FOLDER"] = os.path.join(os.getcwd(), path)

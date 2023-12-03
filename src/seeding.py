import random
from typing import Any

import numpy as np


def set_seed(seed: Any) -> None:
    """
    Sets seed for python and numpy RNG (as recommended in Surprise documentation).
    Source: https://surprise.readthedocs.io/en/stable/FAQ.html#how-to-have-reproducible-experiments
    """
    random.seed(seed)
    np.random.seed(seed)

import os
import random
import numpy as np
from typing import Any


def set_seed(seed: Any) -> None:
    """
    Sets seed for python and numpy RNG (as recommended in Surprise documentation).
    Source: https://surprise.readthedocs.io/en/stable/FAQ.html#how-to-have-reproducible-experiments
    """
    random.seed(seed)
    np.random.seed(seed)

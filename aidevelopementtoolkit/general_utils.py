import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """
    Sets the random seed for reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to set for random number generation.

    Examples
    --------
    >>> set_seet(12)
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
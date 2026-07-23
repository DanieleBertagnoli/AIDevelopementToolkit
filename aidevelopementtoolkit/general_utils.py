import random
from typing import Union, Tuple, List

import numpy as np
import torch

from aidevelopementtoolkit.logging_utils.logger import get_formatted_logger

logger = get_formatted_logger(name=__name__, level="ERROR")

def set_seed(seed: int) -> None:
    """
    Sets the random seed for reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to set for random number generation.

    Examples
    --------
    >>> set_seed(42)
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def human_readable(num: float) -> str:
    """
    Converts a large number into a human-readable string using
    K, M, G, T suffixes.

    Parameters
    ----------
    num : float
        Number to be converted
    
    Returns
    -------
    str
        Human-readable converted number.

    Examples
    --------
    >>> human_readable(1_500)
    '1.500K'
    >>> human_readable(2_300_000)
    '2.300M'
    >>> human_readable(42)
    '42.000'
    """
    for unit in ["", "K", "M", "G", "T", "P"]:
        if abs(num) < 1000:
            return f"{num:.3f}{unit}"
        num /= 1000
    return f"{num:.3f}"


def check_shape(
        array: Union[np.ndarray, torch.Tensor],
        expected_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]]
    ) -> None:
    """
    Validates that an input array matches one or more expected shapes.
    Dimensions may be integers or `-1`, where `-1` acts as a wildcard.

    Parameters
    ----------
    array : Union[np.ndarray, torch.Tensor]
        Input array to validate.

    expected_shape : Union[Tuple[int, ...], List[Tuple[int, ...]]]
        A single expected shape or a list of acceptable shapes.
        Use `-1` to indicate a wildcard dimension.

    Raises
    ------
    TypeError
        If the input is not a NumPy array nor a PyTorch tensor.

    ValueError
        If the array shape does not match any of the expected shapes.

    Examples
    --------
    >>> x = np.zeros((4, 8))
    >>> check_shape(x, (4, 8))  # exact match
    >>> check_shape(x, (-1, 8))  # wildcard batch dimension
    >>> check_shape(x, [(-1, 4), (-1, 8)])  # multiple acceptable shapes
    """
    # Type check
    if not isinstance(array, (np.ndarray, torch.Tensor)):
        Logger.error(
            f"Invalid type: expected `np.ndarray` or `torch.Tensor`, "
            f"got `{type(array).__name__}`"
        )

    actual_shape = tuple(array.shape)

    if isinstance(expected_shape, tuple):
        expected_shapes = [expected_shape]
    else:
        expected_shapes = expected_shape

    # Try all possible shapes
    for shape in expected_shapes:
        if len(shape) != len(actual_shape):
            continue

        match = True
        for a, e in zip(actual_shape, shape):
            if e != -1 and a != e:
                match = False
                break

        # Valid shape found
        if match:
            return

    # If no match found
    if isinstance(expected_shape, tuple):
        logger.error(
            f"Invalid shape: expected `{expected_shape}`, "
            f"got `{actual_shape}`"
        )
        raise ValueError()

    else:
        logger.error(
            f"Invalid shape: expected one of `{expected_shapes}`, "
            f"got `{actual_shape}`"
        )
        raise ValueError()
from typing import Optional, Tuple, Union

import numpy as np
from aidevelopementtoolkit.logging_utils.logger import get_formatted_logger

logger = get_formatted_logger(name=__name__, level="ERROR")

def standardize(
        data: np.ndarray,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = True,
        nan_handling: bool = False,
        eps: float = 1e-8,
    ) -> np.ndarray:
    """Standardize data using mean and standard deviation.

    Parameters
    ----------
    data : np.ndarray
        Input data of any shape.

    axis : Optional[Union[int, Tuple[int, ...]]], default=None
        Axis or axes along which to compute mean and std.
        If `None`, statistics are computed over the entire array.

    keepdims : bool, default=True
        Whether the reduced axes are kept as dimensions of size 1.

    nan_handling : bool, default=False
        When `True`, uses `np.nanmean` and `np.nanstd` to ignore NaN values.

    eps : float, default=1e-8
        Small constant added to the standard deviation to avoid division by
        zero.

    Returns
    -------
    np.ndarray
        Standardized data with the same shape as input.

    Examples
    --------
    >>> data = np.random.rand(2, 3, 4).astype(np.float32)
    >>> standardized = standardize(data, axis=(0, 1), keepdims=True)
    >>> standardized = standardize(data, axis=1, keepdims=True, eps=1e-6)
    """

    data = np.asarray(data, dtype=np.float32)

    mean_fn = np.nanmean if nan_handling else np.mean
    std_fn = np.nanstd if nan_handling else np.std

    mean = mean_fn(data, axis=axis, keepdims=keepdims)
    std = std_fn(data, axis=axis, keepdims=keepdims)
    std = np.where(std < eps, eps, std)

    return (data - mean) / std


def min_max_scaling(
        data: np.ndarray,
        range: Optional[Tuple[float, float]] = None,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = True,
        nan_handling: bool = False,
    ) -> np.ndarray:
    """Apply min-max scaling to data.

    Parameters
    ----------
    data : np.ndarray
        Input data of any shape.

    range : Optional[Tuple[float, float]], default=None
        Target scaling range `(min, max)`.
        If `None`, the data is scaled to `[0, 1]` using statistics computed
        over `axis`.

    axis : Optional[Union[int, Tuple[int, ...]]], default=None
        Axis or axes along which to compute min and max.
        If `None`, statistics are computed over the entire array.
        Ignored when `range` is provided.

    keepdims : bool, default=True
        Whether the reduced axes are kept as dimensions of size 1.
        Ignored when `range` is provided.

    nan_handling : bool, default=False
        When `True`, uses `np.nanmin` and `np.nanmax` to ignore NaN values.

    Returns
    -------
    np.ndarray
        Scaled data with the same shape as input.

    Examples
    --------
    >>> data = np.array([[[0.0], [1.0]], [[2.0], [3.0]]], dtype=np.float32)
    >>> scaled = min_max_scaling(data, range=(0.0, 1.0))
    >>> scaled.min(), scaled.max()
    (0.0, 1.0)
    >>> scaled = min_max_scaling(data, axis=1, keepdims=True)
    """

    data = np.asarray(data, dtype=np.float32)

    min_fn = np.nanmin if nan_handling else np.min
    max_fn = np.nanmax if nan_handling else np.max

    if range is not None:
        if len(range) != 2 or range[0] >= range[1]:
            logger.error(
                "The given range must be a tuple `(min, max)` with min < max."
            )
            raise ValueError()

        target_min, target_max = range
        data_min = min_fn(data)
        data_max = max_fn(data)

    else:
        target_min, target_max = 0.0, 1.0
        data_min = min_fn(data, axis=axis, keepdims=keepdims)
        data_max = max_fn(data, axis=axis, keepdims=keepdims)

    denominator = data_max - data_min

    # Avoid division by zero for constant features
    denominator = np.where(denominator == 0, 1.0, denominator)

    scaled = (data - data_min) / denominator

    return scaled * (target_max - target_min) + target_min
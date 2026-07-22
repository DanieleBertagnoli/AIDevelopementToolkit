from typing import Literal, Optional, Tuple

import numpy as np
from aidevelopementtoolkit.logging_utils.logger import get_formatted_logger

logger = get_formatted_logger(name=__name__, level="ERROR")

def standardize(
        data: np.ndarray,
        processing_type: Literal["dataset", "sequence"] = "sequence",
    ) -> np.ndarray:
    """Standardize data using mean and standard deviation.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape `(B, T, F)`:
        - `B`: Batch dimension
        - `T`: Sequence length
        - `F`: Number of features

    processing_type : Literal["dataset", "sequence"], default="sequence"
        Determines how statistics are computed:
        - `"dataset"`: Compute mean and std over all samples and timesteps.
        - `"sequence"`: Compute mean and std independently for each sequence.

    Returns
    -------
    np.ndarray
        Standardized data with the same shape as input.

    Notes
    -----
    Standardization is performed feature-wise.

    Examples
    --------
    >>> data = np.random.rand(2, 3, 4).astype(np.float32)
    >>> standardized = standardize(data, processing_type="dataset")
    """

    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 3:
        logger.error(
            "The given data must have shape `(B, T, F)`. "
            f"Received {data.shape}."
        )
        raise ValueError()

    if processing_type not in {"dataset", "sequence"}:
        logger.error(
            "Invalid processing_type. Expected 'dataset' or 'sequence', "
            f"received {processing_type}."
        )
        raise ValueError()

    if processing_type == "dataset":
        mean = np.mean(data, axis=(0, 1), keepdims=True)
        std = np.std(data, axis=(0, 1), keepdims=True)

    else:
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)

    # Avoid division by zero for constant features
    std = np.where(std == 0, 1.0, std)

    return (data - mean) / std


def min_max_scaling(
        data: np.ndarray,
        range: Optional[Tuple[float, float]] = None,
        processing_type: Literal["dataset", "sequence"] = "sequence",
    ) -> np.ndarray:
    """Apply min-max scaling to data.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape `(B, T, F)`:
        - `B`: Batch dimension
        - `T`: Sequence length
        - `F`: Number of features

    range : Optional[Tuple[float, float]], default=None
        Target scaling range `(min, max)`.
        If `None`, the input min and max are computed based on
        `processing_type` and the data is scaled to `[0, 1]`.

        When provided, `processing_type` is ignored.

    processing_type : Literal["dataset", "sequence"], default="sequence"
        Determines how input statistics are computed when `range=None`:
        - `"dataset"`: Compute min and max over all samples and timesteps.
        - `"sequence"`: Compute min and max independently for each sequence.

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
    """

    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 3:
        logger.error(
            "The given data must have shape `(B, T, F)`. "
            f"Received {data.shape}."
        )
        raise ValueError()

    if range is not None:
        if len(range) != 2 or range[0] >= range[1]:
            logger.error(
                "The given range must be a tuple `(min, max)` with min < max."
            )
            raise ValueError()

        target_min, target_max = range

        data_min = np.min(data, axis=(0, 1), keepdims=True)
        data_max = np.max(data, axis=(0, 1), keepdims=True)

    else:
        if processing_type not in {"dataset", "sequence"}:
            logger.error(
                "Invalid processing_type. Expected 'dataset' or 'sequence', "
                f"received {processing_type}."
            )
            raise ValueError()

        target_min, target_max = 0.0, 1.0

        if processing_type == "dataset":
            data_min = np.min(data, axis=(0, 1), keepdims=True)
            data_max = np.max(data, axis=(0, 1), keepdims=True)

        else:
            data_min = np.min(data, axis=1, keepdims=True)
            data_max = np.max(data, axis=1, keepdims=True)

    denominator = data_max - data_min

    # Avoid division by zero for constant features
    denominator = np.where(denominator == 0, 1.0, denominator)

    scaled = (data - data_min) / denominator

    return scaled * (target_max - target_min) + target_min
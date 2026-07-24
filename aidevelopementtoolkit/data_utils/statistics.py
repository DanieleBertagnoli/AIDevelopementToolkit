from typing import Tuple

import numpy as np

def get_numerical_statistics(values: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute basic numerical statistics for a 1D array of values.

    Parameters
    ----------
    values : np.ndarray
        Array of shape `(N,)` containing numerical values.

    Returns
    -------
    Tuple[float, float, float, float]
        - `min_value`: Minimum value in the array.  
        - `max_value`: Maximum value in the array.  
        - `avg_value`: Mean value of the array.  
        - `std_value`: Standard deviation of the array.

    Notes
    -----
    The function ignores NaN values in the input array when computing statistics.
    
    Examples
    --------
    >>> import numpy as np
    >>> values = np.array([1.0, 2.0, 3.0, np.nan])
    >>> get_numerical_statistics(values)
    (1.0, 3.0, 2.0, 0.816496580927726)
    """

    min_value = np.nanmin(values)
    max_value = np.nanmax(values)
    avg_value = np.nanmean(values)
    std_value = np.nanstd(values)

    return min_value, max_value, avg_value, std_value
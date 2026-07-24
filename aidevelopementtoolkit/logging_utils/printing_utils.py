from typing import Optional
import math
from tabulate import tabulate

import pandas as pd

def print_table(
        df: pd.DataFrame, 
        max_rows: Optional[int] = None,
        tablefmt: str = "rounded_grid",
        max_num_columns: Optional[int] = None,
    ) -> None:
    """
    Print a pandas DataFrame as a formatted table in the terminal.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to display.
    
    max_rows : Optional[int], default=None
        Maximum number of rows to print. If None, all rows are shown.

    tablefmt : str, default="rounded_grid"
        Table format used to print the table.

    max_num_columns : Optional[int], default=None
        If set, splits the DataFrame into multiple tables each having at most
        this number of columns. The index is preserved and printed.

    Notes
    -----
    Uses `tabulate` for rendering. Does not modify the DataFrame.

    Examples
    --------
    Basic usage:
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> print_table(df)

    Limit number of rows:
        >>> print_table(df, max_rows=1)

    Change table format:
        >>> print_table(df, tablefmt="github")

    Split into multiple tables (column chunks):
        >>> wide_df = pd.DataFrame({
        ...     "A": [1, 2],
        ...     "B": [3, 4],
        ...     "C": [5, 6],
        ...     "D": [7, 8],
        ... })
        >>> print_table(wide_df, max_num_columns=2)

    """
    # Handle row limit
    if max_rows is not None:
        df_to_print = df.head(max_rows)
    else:
        df_to_print = df

    # If no column splitting needed
    if max_num_columns is None:
        print(tabulate(df_to_print, headers="keys", tablefmt=tablefmt, showindex=True))
        return

    # Split columns into chunks
    num_cols = len(df_to_print.columns)
    num_chunks = math.ceil(num_cols / max_num_columns)

    for i in range(num_chunks):
        start = i * max_num_columns
        end = start + max_num_columns
        df_chunk = df_to_print.iloc[:, start:end]

        print(f"\n--- Columns {start} to {min(end, num_cols) - 1} ---")
        print(tabulate(df_chunk, headers="keys", tablefmt=tablefmt, showindex=True))


def human_readable(number: float) -> str:
    """
    Convert a large number into a human-readable string with suffixes.

    Parameters
    ----------
    number : float
        The number to convert.

    Returns
    -------
    str
        Human-readable string representation of the number.

    Examples
    --------
    >>> human_readable(1234567)
    '1.23M'
    >>> human_readable(9876543210)
    '9.88B'
    >>> human_readable(1234)
    '1.23K'
    """
    abs_number = abs(number)
    if abs_number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f}B"
    elif abs_number >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    elif abs_number >= 1_000:
        return f"{number / 1_000:.2f}K"
    else:
        return str(number)
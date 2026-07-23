from typing import Any, Literal, Optional
import io
import os
import json

from aidevelopementtoolkit.logging_utils.logger import get_formatted_logger

import numpy as np
import pandas as pd
import yaml

from aidevelopementtoolkit.boto3_utils import create_s3_client, read_s3_object, write_s3_object, parse_s3_path


_SUPPORTED_EXTENSIONS = {".json", ".yaml", ".yml", ".csv", ".npy"}

logger = get_formatted_logger(name=__name__, level="ERROR")

def _get_extension(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        logger.error(
            f"Unsupported file extension '{ext}'. "
            f"Supported: {sorted(_SUPPORTED_EXTENSIONS)}"
        )
        raise ValueError()
    return ext


def _serialize_data(data: Any, ext: str) -> bytes:
    """
    Serialize data into bytes according to the file extension.

    This function converts supported data formats into raw bytes. The
    resulting bytes can be written directly to a local file or uploaded
    to an S3-compatible object store.

    Parameters
    ----------
    data : Any
        Data to serialize.

        - `.json`: Any JSON-serialisable Python object.
        - `.yaml` / `.yml`: Any object supported by `yaml.safe_dump`.
        - `.csv`: Any object convertible to a `pandas.DataFrame`.
        - `.npy`: Any object supported by `numpy.save`.

    ext : str
        File extension that determines the serialization format.

    Returns
    -------
    bytes
        Serialized data as raw bytes.
    """

    if ext == ".json":
        return json.dumps(data, indent=4).encode("utf-8")

    elif ext in {".yaml", ".yml"}:
        return yaml.safe_dump(data).encode("utf-8")

    elif ext == ".csv":
        return pd.DataFrame(data).to_csv(index=False).encode("utf-8")

    elif ext == ".npy":
        buffer = io.BytesIO()
        np.save(buffer, data)
        return buffer.getvalue()

    logger.error(f"Unsupported file extension: '{ext}'.")
    raise ValueError()


def save_file(
        data: Any, 
        path: str, 
        append: bool = False,
        header: bool = False,
        index: bool = False,
    ) -> None:
    """
    Saves data to a file. The format is deduced from the file extension.

    Supported formats: `.json`, `.yaml` / `.yml`, `.csv`, `.npy`.

    Local paths are written directly to the filesystem. Paths beginning with
    `s3://` are written to an S3-compatible object store using the boto3
    utilities.

    Parameters
    ----------
    data : Any
        Data to be saved.
        - JSON / YAML: any JSON-serialisable object.
        - CSV: any object convertible to a `pandas.DataFrame` or a `pandas.DataFrame` itself.
        - NPY: a `numpy.ndarray`.

    path : str
        Path to the output file, including extension.

        Examples:

        - `"./tmp.json"`
        - `"s3://my-bucket/data/file.json"`

    append : bool, default=False
        If `True`, extends the existing local file instead of overwriting it.

        This parameter is ignored for S3 paths, which are always overwritten.

        - **JSON / YAML list**: extends the list.
        - **JSON / YAML dict**: updates the dict.
        - **CSV**: appends rows (no header written).

    header : bool, default=False
        Only relevant for CSV files. If `True`, writes column names as the first row.
        When appending to an existing CSV file, the header is automatically
        disabled to avoid writing column names in the middle of the file.

    index : bool, default=False
        Only relevant for CSV files. Whether to write row indices.

    Examples
    --------
    >>> save_file({"a": 1}, "./tmp.json")
    >>> save_file([1, 2], "./tmp_list.json")
    >>> save_file({"b": 2}, "./tmp.json", append=True)
    >>> save_file({"x": 1}, "./tmp.yaml")
    >>> save_file([{"col": 1}], "./tmp.csv")

    >>> save_file(
    ...     {"a": 1},
    ...     "s3://my-bucket/data/file.json",
    ... )
    """

    ext = _get_extension(path)

    # Serialize data into bytes for S3 paths
    if path.startswith("s3://"):

        bucket, key = parse_s3_path(path)
        client = create_s3_client()

        serialized_data = _serialize_data(data, ext)

        write_s3_object(
            client=client,
            bucket=bucket,
            key=key,
            data=serialized_data,
        )

        return
    
    # Create parent directory
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Handle local JSON files
    if ext == ".json":

        if append and os.path.exists(path):

            with open(path, "r") as f:
                existing = json.load(f)

            if isinstance(existing, list) and isinstance(data, list):
                existing.extend(data)
                data = existing

            elif isinstance(existing, dict) and isinstance(data, dict):
                existing.update(data)
                data = existing
            
            else:
                logger.error(
                    "Cannot append: incompatible JSON types. "
                    f"Respectively: {type(existing)} and {type(data)}."
                )
                raise ValueError()

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    # Handle local YAML files
    elif ext in {".yaml", ".yml"}:

        if append and os.path.exists(path):
            with open(path, "r") as f:
                existing = yaml.safe_load(f)

            if isinstance(existing, list) and isinstance(data, list):
                existing.extend(data)
                data = existing

            elif isinstance(existing, dict) and isinstance(data, dict):
                existing.update(data)
                data = existing

            else:
                logger.error("Cannot append: incompatible YAML types.")
                raise RuntimeError()

        with open(path, "w") as f:
            yaml.safe_dump(data, f)

    # Handle local CSV files
    elif ext == ".csv":
        df = pd.DataFrame(data)
        if append and os.path.exists(path):
            df.to_csv(path, mode="a", header=False, index=index)
        else:
            df.to_csv(path, index=index, header=header)

    # Handle local NPY files
    elif ext == ".npy":
        np.save(path, data)


def load_file(
        path: str,
        return_type: Optional[Literal["numpy", "pandas"]] = "pandas",
    ) -> Any:
    """
    Loads a file. The format is deduced from the file extension.

    Supported formats: `.json`, `.yaml` / `.yml`, `.csv`, `.npy`.

    Local paths are read directly from the filesystem. Paths beginning with
    `s3://` are read from an S3-compatible object store using the boto3
    utilities.

    Parameters
    ----------
    path : str
        Path to the file, including extension.

        Examples:

        - `"./tmp.json"`
        - `"s3://my-bucket/data/file.json"`

    return_type : Optional[Literal["numpy", "pandas"]], default="pandas"
        Only relevant for CSV files. Controls whether the data is returned
        as a `pandas.DataFrame` or a `numpy.ndarray`.

    Returns
    -------
    Any
        Parsed file content.

    Examples
    --------
    >>> save_file({"a": 1}, "./tmp.json")
    >>> content = load_file("./tmp.json")
    >>> content["a"]
    1

    >>> save_file({"a": 1}, "./tmp.yaml")
    >>> content = load_file("./tmp.yaml")
    >>> content["a"]
    1

    >>> df = load_file("./tmp.csv")
    >>> arr = load_file("./tmp.csv", return_type="numpy")
    >>> content = load_file("s3://my-bucket/data/file.json")
    """

    # Read the file into a common file-like object
    if path.startswith("s3://"):

        bucket, key = parse_s3_path(path)
        client = create_s3_client()

        file_data = io.BytesIO(
            read_s3_object(
                client=client,
                bucket=bucket,
                key=key,
            )
        )

    else:

        if not os.path.exists(path):
            logger.error(f"File '{path}' does not exist.")
            raise FileNotFoundError()

        with open(path, "rb") as f:
            file_data = io.BytesIO(f.read())

    ext = _get_extension(path)

    # Parse the file according to its extension
    if ext == ".json":
        return json.load(file_data)

    elif ext in {".yaml", ".yml"}:
        return yaml.safe_load(file_data)

    elif ext == ".csv":

        df = pd.read_csv(file_data)

        if return_type == "numpy":
            return df.to_numpy()
        
        elif return_type == "pandas":
            return df

        else:
            logger.error(f"Unsupported return type: '{return_type}'. Supported: 'numpy', 'pandas'.")
            raise ValueError()

    elif ext == ".npy":
        return np.load(file_data, allow_pickle=False)


def file_exists(path: str) -> bool:
    """
    Checks if a file exists. Works for both local and S3 paths.

    Parameters
    ----------
    path : str
        Path to the file, including extension.

        Examples:

        - `"./tmp.json"`
        - `"s3://my-bucket/data/file.json"`

    Returns
    -------
    bool
        `True` if the file exists, `False` otherwise.

    Examples
    --------
    >>> save_file({"a": 1}, "./tmp.json")
    >>> file_exists("./tmp.json")
    True
    >>> file_exists("s3://my-bucket/data/file.json")
    False
    """

    if path.startswith("s3://"):
        bucket, key = parse_s3_path(path)
        client = create_s3_client()
        return read_s3_object(client=client, bucket=bucket, key=key) is not None

    else:
        return os.path.exists(path)
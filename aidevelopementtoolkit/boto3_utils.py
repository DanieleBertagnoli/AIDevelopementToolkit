from typing import Tuple

import io
import os

import boto3
from tqdm import tqdm

from aidevelopementtoolkit.logging_utils.logger import get_formatted_logger

_CHUNK_SIZE = 1024 * 1024 # 1 MB


logger = get_formatted_logger(name=__name__, level="ERROR")

def create_s3_client():
    """
    Create a boto3 S3 client from environment variables.

    Reads the following environment variables:

    - `S3_ENDPOINT_URL` - custom endpoint URL.
    - `AWS_ACCESS_KEY_ID` - access key ID.
    - `AWS_SECRET_ACCESS_KEY` - secret access key.

    Returns
    -------
    botocore.client.S3
        Configured S3 client.

    Raises
    ------
    EnvironmentError
        If any of the required environment variables are not set.

    Examples
    --------
    >>> import os
    >>> os.environ["S3_ENDPOINT_URL"] = "http://localhost:9000"
    >>> os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    >>> os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
    >>> client = create_s3_client()
    """

    required = ("S3_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    missing = [var for var in required if not os.environ.get(var)]

    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        raise EnvironmentError()

    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def read_s3_object(
        client,
        bucket: str,
        key: str,
    ) -> bytes:
    """
    Download an S3 object and return its content as bytes.

    Displays a `tqdm` progress bar during the download, using the
    object's `Content-Length` as the total when available.

    Parameters
    ----------
    client : botocore.client.S3
        S3 client returned by :func:`create_s3_client`.

    bucket : str
        Name of the S3 bucket.

    key : str
        Key (path) of the object inside the bucket.

    Returns
    -------
    bytes
        Raw content of the S3 object.

    Examples
    --------
    >>> client = create_s3_client()
    >>> data = read_s3_object(client, bucket="my-bucket", key="data/file.npy")
    """

    response = client.get_object(Bucket=bucket, Key=key)
    total = response["ContentLength"]
    body = response["Body"]

    buffer = io.BytesIO()

    with tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"Reading {os.path.basename(key)} from S3",
        colour="yellow",
        leave=False,
    ) as progress:
        while True:
            chunk = body.read(_CHUNK_SIZE)
            if not chunk:
                break
            buffer.write(chunk)
            progress.update(len(chunk))

    return buffer.getvalue()


def write_s3_object(
        client,
        bucket: str,
        key: str,
        data: bytes,
    ) -> None:
    """
    Upload bytes to an S3 object.

    Displays a `tqdm` progress bar during the upload.

    Parameters
    ----------
    client : botocore.client.S3
        S3 client returned by :func:`create_s3_client`.

    bucket : str
        Name of the destination S3 bucket.

    key : str
        Key (path) of the object to write inside the bucket.

    data : bytes
        Raw bytes to upload.

    Examples
    --------
    >>> client = create_s3_client()
    >>> write_s3_object(client, bucket="my-bucket", key="data/file.npy", data=b"...")
    """

    total = len(data)
    buffer = io.BytesIO(data)

    with tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"Writing {os.path.basename(key)} to S3",
        colour="green",
        leave=False,
    ) as progress:
        
        try:
            client.upload_fileobj(buffer, bucket, key, Callback=lambda n: progress.update(n))
        except Exception as e:
            logger.error(f"The S3 upload of {key} in the bucket {bucket} as failed. \n{e}")

def parse_s3_path(path: str) -> Tuple[str, str]:
    """Parse an S3 path into bucket and key.

    Parameters
    ----------
    path : str
        S3 path in the format `s3://bucket/key`.

    Returns
    -------
    Tuple[str, str]
        Bucket name and object key.
    """

    s3_path = path.removeprefix("s3://")

    if "/" not in s3_path:
        logger.error("Invalid S3 path. Expected format: 's3://bucket/key'.")
        raise ValueError()

    bucket, key = s3_path.split("/", maxsplit=1)

    if not bucket or not key:
        logger.error("Invalid S3 path. Expected format: 's3://bucket/key'.")
        raise ValueError()

    return bucket, key
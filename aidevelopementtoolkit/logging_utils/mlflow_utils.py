import os
from os import environ
from typing import Any, Dict
import shutil

from torch import nn
import mlflow

from aidevelopementtoolkit.logging_utils.file_io import load_file
from aidevelopementtoolkit.logging_utils.logger import get_formatted_logger
from aidevelopementtoolkit.torch_utils.model_utils import save_model


def start_mlflow_run(experiment_name: str, mlflow_kwargs: Dict[str, Any]) -> None:
    """This function starts an MLflow run with the specified 
    experiment name and additional keyword arguments.

    Notes
    -----
    The function checks for the presence of the `MLFLOW_ENDPOINT_URL` environment variable.
    If the provided MLflow tracking URI is a remote server (i.e., starts with "http"), it also checks 
    for the presence of the following environment variables:
    - `REQUESTS_CA_BUNDLE`
    - `MLFLOW_TRACKING_USERNAME`
    - `MLFLOW_TRACKING_PASSWORD`
    - `MLFLOW_TRACKING_SERVER_CERT_PATH`
    - `MLFLOW_ENDPOINT_URL`

    If you are using a local MLflow server (i.e., the tracking URI starts with "sqlite:///"), the function
      will create an artifacts directory in the same location as the SQLite database.
    
    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment. If the experiment does not exist, it will be created.

    mlflow_kwargs : Dict[str, Any]
        Additional keyword arguments to pass to `mlflow.start_run()`.

    Examples
    --------
    >>> os.environ["MLFLOW_ENDPOINT_URL"] = https://myremote.com
    >>> start_mlflow_run(
    ...         experiment_name="MNIST",
    ...         mlflow_kwargs={
    ...             "run_name": "ResNet50",
    ...             "tags": {
    ...                 "model": "resnet50",
    ...                 "pre-trained": "false",
    ...             },
    ...             "log_system_metrics": True,
    ...             "description": "Simple showcase."
    ...         }
    ...     )
        
    See Also
    --------
    mlflow.start_run : https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run
    """

    logger = get_formatted_logger(name="mlflow", level="ERROR")

    if "MLFLOW_ENDPOINT_URL" not in environ:
        logger.error("`MLFLOW_ENDPOINT_URL` environment variable is not set.")
        raise ValueError()
    
    mlflow_tracking_uri = environ["MLFLOW_ENDPOINT_URL"]

    # Remote server
    artifacts_location = None
    if mlflow_tracking_uri.startswith("http"):

        needed_env_vars = [
            "REQUESTS_CA_BUNDLE",
            "MLFLOW_TRACKING_USERNAME",
            "MLFLOW_TRACKING_PASSWORD",
            "MLFLOW_TRACKING_SERVER_CERT_PATH",
            "MLFLOW_ENDPOINT_URL",
        ]

        for env_var in needed_env_vars:
            if env_var not in environ:
                logger.error(f"`{env_var}` environment variable is not set.")
                raise ValueError()
            
    # Local server
    elif mlflow_tracking_uri.startswith("sqlite:///"):
        
        if not mlflow_tracking_uri.endswith("mlflow.db"):
            logger.error(
                f"The provided `mlflow_tracking_uri`='{mlflow_tracking_uri}' is "
                "a local server but it's not ending with 'mlflow.db'."
            )
            raise ValueError()

        local_path = mlflow_tracking_uri.replace("sqlite:///", "").replace("mlflow.db", "")
        artifacts_location = os.path.join(local_path, "artifacts", experiment_name)

    else:
        logger.error(f"Invalid `MLFLOW_ENDPOINT_URL`: {mlflow_tracking_uri}")
        raise ValueError()
    
    # Set tracking URI
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Set workspace if provided
    if "MLFLOW_WORKSPACE" in environ:
        mlflow.set_workspace(environ["MLFLOW_WORKSPACE"])

    # Create the experiment if it doesn't exist
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifacts_location,
        )

    # Set the experiment and start the run
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(**mlflow_kwargs)


def log_run_parameters(
        parameters: Dict[str, Any], 
        prefix: str = "", 
        recursive_logging: bool = False
    ) -> None:
    """
    Logs a dictionary (including nested dictionaries) to MLFlow as parameters.

    By default, all key-value pairs are logged directly. If a value is a dictionary, it is logged recursively using 
    nested key prefixes.

    Parameters
    ----------
    parameters : Dict[str, Any]
        Dictionary to be logged.

    prefix : str, default=""
        Prefix for nested keys.

    recursive_logging : bool, default=False
        If True, when a value points to a YAML or JSON file, the file is loaded 
        and its contents are recursively logged instead of
        logging the file path.

    Examples
    --------
    >>> log_run_parameters({"lr": 0.001, "batch_size": 32})
    >>> log_run_parameters({"train": {"lr": 0.001}}, prefix="model")
    """

    for key, value in parameters.items():
        full_key = f"{prefix}/{key}" if prefix else key

        # Case 1: Nested dictionary, then recurse
        if isinstance(value, dict):
            log_run_parameters(value, full_key, recursive_logging=recursive_logging)
            continue

        # Case 2: Recursive logging enabled, then check if value is a YAML or JSON file
        if recursive_logging and isinstance(value, str) and os.path.isfile(value):
            if value.endswith((".yaml", ".yml")):
                nested_dict = load_file(value)
                if isinstance(nested_dict, dict):
                    log_run_parameters(nested_dict, full_key, recursive_logging=recursive_logging)
                    continue
            elif value.endswith(".json"):
                nested_dict = load_file(value)
                if isinstance(nested_dict, dict):
                    log_run_parameters(nested_dict, full_key, recursive_logging=recursive_logging)
                    continue

        # Case 3: Normal logging (base case)
        mlflow.log_param(full_key, value)


def save_model_checkpoint(
        model: nn.Module, 
        config: Dict[str, Any],
        checkpoint_name: str,
    ) -> None:
    """This function saves the model checkpoints to a temporary local directory 
    and logs that directory as an artifact to MLflow.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to be saved.

    config : Dict[str, Any]
        Configuration dictionary used to build the model.

    checkpoint_name : str
        Name of the checkpoint (e.g., "last", "best") to be used in the artifact path in MLflow.

    Examples
    --------
    >>> save_model_checkpoint(model, {"hidden_dim": 128}, "best")
    """

    # Store in a temporary local directory
    local_dir = "temp_model_checkpoints"
    os.makedirs(local_dir, exist_ok=True)
    save_model(model, config, local_dir)

    # Log the directory as an artifact to MLflow
    mlflow.log_artifacts(local_dir, artifact_path=f"model_checkpoints/{checkpoint_name}")
    
    # Remove directory
    shutil.rmtree(local_dir)


def is_numeric(val: Any) -> bool:
    """
    This function checks whether a given variable is numeric.

    Parameters
    ----------
    val : Any
        Variable to be checked.

    Returns
    -------
    bool
        `True` if the variable is numeric, `False` otherwise.
    """
    return isinstance(val, (int, float))
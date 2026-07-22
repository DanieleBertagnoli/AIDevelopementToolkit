from typing import Any, Dict, List, Tuple, Type
import os

import numpy as np
from torchsummary import summary
import torch as torch
from torch import nn
import onnxruntime as ort

from aidevelopementtoolkit.logging_utils.file_io import save_file, load_file
from aidevelopementtoolkit.logging_utils.logger import get_formatted_logger

logger = get_formatted_logger(name=__name__, level="ERROR")

def save_model(
        model: Type[nn.Module],
        model_configs: Dict[str, Any],
        checkpoint_dir: str,
    ) -> None:
    """Save a PyTorch model checkpoint.

    The model weights are saved as ``weights.pt`` and the model
    configuration is saved as ``model_configs.json`` inside the provided
    checkpoint directory.

    Parameters
    ----------
    model : Type[nn.Module]
        The PyTorch model to save.

    model_configs : Dict[str, Any]
        Dictionary containing the parameters required to recreate the model.

    checkpoint_dir : str
        Directory where the checkpoint files will be stored.

    Returns
    -------
    None

    Examples
    --------
    Save a trained model and its configuration:

    >>> import torch
    >>> model = torch.nn.Linear(10, 2)
    >>> configs = {
    ...     "input_features": 10,
    ...     "output_features": 2,
    ... }
    >>> save_model(
    ...     model=model,
    ...     model_configs=configs,
    ...     checkpoint_dir="checkpoints/model_v1",
    ... )

    The resulting directory contains:

    .. code-block:: text

        checkpoints/model_v1/
        ├── weights.pt
        └── model_configs.json
    """

    model_path = os.path.join(checkpoint_dir, "weights.pt")
    checkpoint_path = os.path.join(checkpoint_dir, "model_configs.json")

    torch.save(model.state_dict(), model_path)
    save_file(model_configs, checkpoint_path)


def load_model(
        model_class: Type[nn.Module],
        checkpoint_dir: str,
        map_location: str = "cpu",
    ) -> Type[nn.Module]:
    """Load a PyTorch model checkpoint.

    The model class must accept the parameters stored in
    ``model_configs.json`` as keyword arguments.

    Parameters
    ----------
    model_class : Type[nn.Module]
        Model class used to recreate the architecture.

    checkpoint_dir : str
        Directory containing ``weights.pt`` and ``model_configs.json``.

    map_location : str, default="cpu"
        Device where model weights are loaded.

    Returns
    -------
    Type[nn.Module]
        Reconstructed model with loaded weights.

    Examples
    --------
    Load a saved model:

    >>> import torch
    >>> class MyModel(torch.nn.Module):
    ...     def __init__(self, input_features, output_features):
    ...         super().__init__()
    ...         self.layer = torch.nn.Linear(
    ...             input_features,
    ...             output_features,
    ...         )
    ...
    >>> model = load_model(
    ...     model_class=MyModel,
    ...     checkpoint_dir="checkpoints/model_v1",
    ... )
    """

    if not os.path.exists(checkpoint_dir):
        logger.error(f"Checkpoint directory '{checkpoint_dir}' does not exist.")
        raise FileNotFoundError()

    model_configs_path = os.path.join(checkpoint_dir, "model_configs.json")
    if not os.path.exists(model_configs_path):
        logger.error(f"Model configuration file '{model_configs_path}' does not exist.")
        raise FileNotFoundError()

    model_path = os.path.join(checkpoint_dir, "weights.pt")
    if not os.path.exists(model_path):
        logger.error(f"Model weights file '{model_path}' does not exist.")
        raise FileNotFoundError()
    
    model: Type[nn.Module] = model_class(**load_file(model_configs_path))
    model.load_state_dict(torch.load(model_path, map_location=map_location))
    return model


def export_to_onnx(
        model: Type[nn.Module],
        input_shape: List[Tuple[int, ...]],
        input_names: List[str],
        output_names: List[str],
        export_path: str,
        opset_version: int = 18,
        external_data: bool = False,
    ) -> None:
    """Export a PyTorch model to ONNX format and verify the conversion.

    After exporting, the generated ONNX model is executed using ONNX Runtime
    and its outputs are compared against the original PyTorch model outputs.

    Parameters
    ----------
    model : Type[nn.Module]
        PyTorch model to export.

    input_shape : List[Tuple[int, ...]]
        Input tensor shapes excluding the batch dimension.

    input_names : List[str]
        Names assigned to ONNX input tensors. Must match ``input_shape``.

    output_names : List[str]
        Names assigned to ONNX output tensors.

    export_path : str
        Destination path for the ONNX file.

    opset_version : int, default=18
        ONNX opset version used during export.

    external_data : bool, default=False
        Store model weights externally. Useful for models larger than 2GB.

    Returns
    -------
    None

    Examples
    --------
    Export a simple neural network:

    >>> import torch
    >>> model = torch.nn.Linear(10, 2)
    >>> export_to_onnx(
    ...     model=model,
    ...     input_shape=[(10,)],
    ...     input_names=["features"],
    ...     output_names=["prediction"],
    ...     export_path="model.onnx",
    ... )

    The generated ONNX model can then be executed using
    :func:`run_onnx`.
    """

    if len(input_shape) != len(input_names):
        logger.error("The number of input shapes must match the number of input names.")
        raise ValueError()

    model.eval()

    dummy_input = tuple([torch.randn((1, *shape)) for shape in input_shape])
    torch.onnx.export(
        model, 
        dummy_input, 
        export_path, 
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        external_data=external_data,
    )


    ### Check nn.Module and ONNX outputs ###

    with torch.no_grad():
        # Get PyTorch model output
        torch_output = model(*dummy_input)

        # Convert to tuple
        if isinstance(torch_output, torch.Tensor):
            torch_output = (torch_output,)

        # Remove None outputs since ONNX does not support None outputs
        # and removes them at runtime
        torch_output = tuple(output for output in torch_output if output is not None)

        # Load ONNX model and get output
        ort_session = ort.InferenceSession(export_path)
        ort_inputs = {name: input_tensor.numpy() for name, input_tensor in zip(input_names, dummy_input)}
        onnx_output = ort_session.run(None, ort_inputs)

        for torch_out, onnx_out in zip(torch_output, onnx_output):
            np.testing.assert_allclose(
                torch_out.numpy(), 
                onnx_out, 
                rtol=1e-03, 
                atol=1e-05,
            )

def run_onnx(
        onnx_model_path: str,
        input_data: Dict[str, np.ndarray],
    ) -> List[np.ndarray]:
    """Run an ONNX model using ONNX Runtime.

    Parameters
    ----------
    onnx_model_path : str
        Path to the ONNX model file.

    input_data : Dict[str, np.ndarray]
        Dictionary mapping ONNX input names to NumPy input arrays.

    Returns
    -------
    List[np.ndarray]
        Model output tensors.

    Examples
    --------
    Run inference on an exported ONNX model:

    >>> import numpy as np
    >>> outputs = run_onnx(
    ...     onnx_model_path="model.onnx",
    ...     input_data={
    ...         "features": np.random.randn(1, 10).astype(np.float32)
    ...     },
    ... )
    >>> prediction = outputs[0]
    """

    if not os.path.exists(onnx_model_path):
        logger.error(f"ONNX model file '{onnx_model_path}' does not exist.")
        raise FileNotFoundError()

    ort_session = ort.InferenceSession(onnx_model_path)
    outputs = ort_session.run(None, input_data)
    
    return outputs


def print_model_summary(
        model: Type[nn.Module],
        input_shape: Tuple[int, ...],
    ) -> None:
    """Print a summary of a PyTorch model.

    The summary includes layer information, parameter counts, and tensor
    shapes using ``torchsummary``.

    Parameters
    ----------
    model : Type[nn.Module]
        PyTorch model to summarize.

    input_shape : Tuple[int, ...]
        Input tensor shape excluding the batch dimension.


    Examples
    --------
    Print the summary of a model receiving 10 features:

    >>> import torch
    >>> model = torch.nn.Sequential(
    ...     torch.nn.Linear(10, 5),
    ...     torch.nn.ReLU(),
    ...     torch.nn.Linear(5, 1),
    ... )
    >>> print_model_summary(
    ...     model=model,
    ...     input_shape=(10,),
    ... )
    """

    # Add batch dimension to input shape
    input_shape_with_batch = (1, *input_shape)

    # Print model summary
    summary(model, input_size=input_shape_with_batch[1:], device="cpu")
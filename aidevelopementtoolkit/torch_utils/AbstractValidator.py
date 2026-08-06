from abc import ABC, abstractmethod
from typing import Any, Union, Tuple

import torch
from torch import nn

class AbstractValidator(ABC):
    """This abstract class serves as a blueprint for creating specific validator classes
    that handle the validation process of machine learning models using PyTorch.
    It defines the essential components and methods that any concrete validator class 
    must implement, including the validation step and the overall validation loop.

    Notes
    -----
    This can be used also in distributed validation scenarios, but the implementation 
    of the `validation_step` and `validate` methods should take into account the distributed setting.
    We suggest to check the example :doc:`examples/mnist_training.py` for a reference 
    implementation of a validator that can be used in distributed validation.

    Parameters
    ----------
    loss_fn : nn.Module
        The loss function used in the validation process.

    device : str or torch.device
        The device (e.g., 'cpu' or 'cuda:0') on which the model and data will 
        be loaded for validation.
    """

    def __init__(
            self, 
            loss_fn: nn.Module, 
            device: Union[str, torch.device],
        ) -> None:
    
        self.loss_fn = loss_fn
        self.device = torch.device(device) if isinstance(device, str) else device


    @abstractmethod
    def validation_step(self, model: nn.Module, batch: Tuple[torch.Tensor, ...]) -> Any:
        """This abstract method defines the validation step for a 
        single batch of data. It must be implemented in subclasses to specify 
        how the model is evaluated based on the input batch.

        Parameters
        ----------
        model : nn.Module
            The model to be validated.

        batch : Tuple[torch.Tensor, ...]
            A tuple containing the input data and eventual other tensors (e.g., labels) 
            for a single batch.

        Returns
        -------
        Any
            The output of the validation step, which can be any type depending on the 
            specific implementation in the subclass.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    

    @abstractmethod
    def validate(
            self, 
            model: nn.Module, 
            data_loader: torch.utils.data.DataLoader, 
            current_epoch: int,
        ) -> Any:
        """Validate the model on the given data loader.

        Parameters
        ----------
        model : nn.Module
            The model to be validated.
        
        data_loader : torch.utils.data.DataLoader
            The data loader providing the validation data.

        current_epoch : int
            The current epoch number, which can be used for logging or checkpointing purposes.

        Returns
        -------
        Any
            The result of the validation, which can be any type depending on the 
            specific implementation in the subclass.
        """
        raise NotImplementedError("Subclasses must implement this method.")
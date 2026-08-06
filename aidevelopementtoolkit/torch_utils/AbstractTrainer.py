from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from aidevelopementtoolkit.torch_utils.EarlyStopper import EarlyStopper
from aidevelopementtoolkit.torch_utils.AbstractValidator import AbstractValidator

class AbstractTrainingStep(ABC):
    """This abstract class serves as a blueprint for creating specific training 
    step classes that define how a model is updated on a single batch of data. 
    Encapsulating this logic outside of the trainer allows the same 
    `AbstractTrainer` subclass to be reused simply by swapping the training 
    step implementation that is passed to its constructor.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer used to update the model's parameters during training.

    loss_fn : nn.Module
        The loss function used in the training process.

    device : str or torch.device
        The device (e.g., 'cpu' or 'cuda:0') on which the model and data will be loaded for training.
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            loss_fn: nn.Module,
            device: Union[str, torch.device],
        ) -> None:

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device) if isinstance(device, str) else device


    @abstractmethod
    def __call__(
            self, 
            model: nn.Module, 
            batch: Tuple[torch.Tensor, ...], 
            **kwargs: Dict[str, Any]
        ) -> Any:
        """This abstract method that defines the training step for a 
        single batch of data. It must be implemented in subclasses to specify 
        how the model is updated based on the input batch.

        Parameters
        ----------
        model : nn.Module
            The model to be trained.

        batch : Tuple[torch.Tensor, ...]
            A tuple containing the input data and eventual other tensors (e.g., labels) 
            for a single batch.

        **kwargs : Dict[str, Any]
            Additional keyword arguments that subclasses can use to receive extra 
            information needed for a specific training step implementation.

        Returns
        -------
        Any
            The output of the training step, which can be any type depending on the 
            specific implementation in the subclass.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class AbstractTrainer(ABC):
    """This abstract class serves as a blueprint for creating specific trainer 
    classes that handle the training process of machine learning models 
    using PyTorch. It defines the essential components and methods that any 
    concrete trainer class must implement, namely the overall training loop, 
    while the per-batch training logic is delegated to an `AbstractTrainingStep` 
    instance passed at construction time. This allows the same trainer class to 
    be reused simply by changing the training step implementation.

    Notes
    -----
    This can be used also in distributed training scenarios, but the implementation 
    of the `train` method should take into account the distributed setting.
    We suggest to check the example :doc:`examples/mnist_training.py` for a reference 
    implementation of a trainer that can be used in distributed training.

    Parameters
    ----------
    training_step : AbstractTrainingStep
        The training step used to update the model's parameters on a single batch of data.

    device : str or torch.device
        The device (e.g., 'cpu' or 'cuda:0') on which the model and data will be loaded for training.

    epochs : int
        The number of epochs for which the model will be trained.
    
    validator : AbstractValidator
        The validator used to evaluate the model's performance on the validation set.
    
    early_stopper : Optional[EarlyStopper]
        The early stopper used to halt training if the model's performance stops improving.
        It can be set to `None` if early stopping is not desired.
        
    checkpoint_interval : int
        The interval (in epochs) at which to save model checkpoints.
    """

    def __init__(
            self, 
            training_step: AbstractTrainingStep,
            device: Union[str, torch.device],
            epochs: int,
            validator: AbstractValidator,
            early_stopper: Optional[EarlyStopper],
            checkpoint_interval: int,
        ) -> None:

        self.training_step = training_step
        self.device = torch.device(device) if isinstance(device, str) else device
        self.epochs = epochs
        self.early_stopper = early_stopper
        self.validator = validator
        self.checkpoint_interval = checkpoint_interval


    @abstractmethod
    def train(
            self, 
            model: nn.Module, 
            train_loader: DataLoader, 
            val_loader: DataLoader, 
            **kwargs: Dict[str, Any],
        ) -> nn.Module:
        """This abstract method that defines the overall training process for the model. 
        It must be implemented in subclasses to specify how the model is 
        trained over multiple epochs, including validation and early stopping if applicable.

        Parameters
        ----------
        model : nn.Module
            The model to be trained.

        train_loader : DataLoader
            A DataLoader providing the training data in batches.
        
        val_loader : DataLoader
            A DataLoader providing the validation data in batches.

        **kwargs : Dict[str, Any]
            Additional keyword arguments that subclasses can use to receive extra 
            information needed for a specific training loop implementation.

        Returns
        -------
        nn.Module
            The trained model after completing the training process.
        """
        raise NotImplementedError("Subclasses must implement this method.")
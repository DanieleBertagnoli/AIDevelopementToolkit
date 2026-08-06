"""MNIST training example with optional PyTorch distributed execution.

This script trains a ResNet-50 model on MNIST using the toolkit's
logging and checkpoint utilities. It supports:

- single-GPU execution on a local machine
- multi-GPU execution with `torchrun` on one node
- multi-node multi-GPU execution with `torchrun`

Run examples:

    # Single GPU
    torchrun --nproc_per_node=1 examples/mnist_training.py

    # Multi-GPU on one node with 4 GPUs
    torchrun --nproc_per_node=4 examples/mnist_training.py

    # Multi-node with 2 nodes and 4 GPUs per node
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr=127.0.0.1 --master_port=12345 examples/mnist_training.py # Run on the first node
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 --master_addr=127.0.0.1 --master_port=12345 examples/mnist_training.py # Run on the second node

**NOTE**: The number of `nproc_per_node` must be set based on the number of GPUs on the node on which the command is launched. This number must match the 
`devices` variable indicated in the `__main__` part (typically configured via YAML).
"""

import os
from typing import Tuple, Dict, Union, Optional

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import mlflow
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torchvision.models import resnet50
from torchvision.datasets import MNIST
from torchvision import transforms

from aidevelopementtoolkit.general_utils import set_seed, check_shape
from aidevelopementtoolkit.logging_utils.logger import get_formatted_logger
from aidevelopementtoolkit.logging_utils.mlflow_utils import start_mlflow_run, save_model_checkpoint, log_run_parameters
from aidevelopementtoolkit.torch_utils.EarlyStopper import EarlyStopper
from aidevelopementtoolkit.torch_utils.AbstractTrainer import AbstractTrainer
from aidevelopementtoolkit.torch_utils.AbstractValidator import AbstractValidator
from aidevelopementtoolkit.data_utils.metrics import compute_classification_metrics, compute_confusion_matrix
from aidevelopementtoolkit.logging_utils.plotly_utils import plot_heatmap
from aidevelopementtoolkit.torch_utils.distributed_torch_utils import dist_barrier, get_process_rank










#################
#               #
#   Validator   #
#               #
#################

class MNISTValidator(AbstractValidator):
    """Validation helper for batch-wise and epoch-wise model evaluation.

    Parameters
    ----------
    loss_fn : nn.Module
        The loss function used in the validation process.

    device : str or torch.device
        The device (e.g., 'cpu' or 'cuda:0') on which the model and data will 
        be loaded for validation.

    num_classes : int
        Number of classification labels.
    """

    def __init__(
            self,
            loss_fn: nn.Module,
            device: Union[str, torch.device],
            num_classes: int,
        ):

        super().__init__(loss_fn=loss_fn, device=device)
        self.num_classes = num_classes

        self.logger = get_formatted_logger(level="ERROR")


    def validation_step(
            self, 
            model: nn.Module, 
            batch: Tuple[torch.Tensor, torch.Tensor]
        ) -> Tuple[float, torch.Tensor]:
        """This function is used to validate a single batch.

        Parameters
        ----------
        model : nn.Module
            Model to be validated.

        batch : Tuple[torch.Tensor, torch.Tensor]
            Batch composed of:

                - `images`: Images to be classified. Shape `(B, H, W, C)`:
                    - `B`: Batch dimension
                    - `H`: Image heigth
                    - `W`: Image width
                    - `C`: Number of channels

                - `labels`: Class IDs. Shape `(B,)`.

        Returns
        -------
        Tuple[float, torch.Tensor]
            Respectively:
                - Batch loss
                - Batch confusion matrix (not averaged)
        """

        if len(batch) != 2:
            self.logger.error("Invalid batch: expected a tuple of (images, labels)")
            raise ValueError()

        with torch.inference_mode():
            imgs, labels = batch

            check_shape(imgs, (-1, 3, 224, 224))
            B, C, H, W = imgs.shape
            check_shape(labels, (B,))

            imgs = imgs.to(self.device)
            labels = labels.to(self.device).long()

            # Run the model in inference mode and compute the batch loss.
            preds: torch.Tensor = model(imgs)
            loss: torch.Tensor = self.loss_fn(preds, labels)

            # Convert raw logits to class predictions.
            preds_class = torch.argmax(preds, dim=-1)

            batch_cm = compute_confusion_matrix(
                predictions=preds_class.cpu().detach().numpy(),
                labels=labels.cpu().detach().numpy(),
                padding_mask=np.full((imgs.shape[0],), fill_value=False),
                num_classes=self.num_classes,
            )

        batch_cm = torch.as_tensor(batch_cm, device=self.device, dtype=torch.int64)

        return loss.item(), batch_cm
    

    def validate(
            self, 
            model: nn.Module, 
            data_loader: torch.utils.data.DataLoader, 
            current_epoch: int,
        ) -> Tuple[float, Dict[str, float]]:
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
        Tuple[float, Dict[str, float]]
            Respectively:
                - Validation loss averaged over all batches
                - Dictionary with averaged validation metrics

            Both the tensors are already aggregate among all the processes.
        """

        model.eval()
        model = model.to(self.device)


        ### Validation Loop ###
            
        # Accumulate validation statistics across all batches
        epoch_loss = torch.zeros(1, device=self.device, dtype=torch.float32)
        total_confusion_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64, device=self.device)

        for batch in tqdm(data_loader, desc="Validating model ", unit="batch", leave=False, position=get_process_rank()):

            batch_loss, batch_confusion_matrix = self.validation_step(model, batch)
            epoch_loss += batch_loss

            # Accumulate metrics
            total_confusion_matrix += batch_confusion_matrix
            
        # Average metrics on the num of batches
        epoch_loss = epoch_loss / len(data_loader)
        

        ### Aggregate processes ###

        # Reduce loss across all processes
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.AVG)
        epoch_loss = epoch_loss.item()

        dist.all_reduce(total_confusion_matrix, op=dist.ReduceOp.SUM)


        ### Metrics and Logging ###

        total_confusion_matrix = total_confusion_matrix.cpu().detach().numpy()
        metrics, normalized_confusion_matrix = compute_classification_metrics(total_confusion_matrix)

        # Plotting in MLFlow
        if get_process_rank() == 0:
            plot_heatmap(
                normalized_confusion_matrix,
                "MNIST Confusion Matrix",
                "Predicted",
                "GT",
                f"plots/epoch_{current_epoch}.png",
            )

        return epoch_loss, metrics









###############
#             #
#   Trainer   #
#             #
###############
        

class MNISTTrainer(AbstractTrainer):
    """Training loop encapsulation for epoch-based model optimization.

    Trainer performs batch-level training, validation, metric logging, and
    checkpoint saving during training.

    Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer used to update the model's parameters during training.
        
        loss_fn : nn.Module
            The loss function used in the training process.
    
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
            optimizer: torch.optim.Optimizer, 
            loss_fn: nn.Module, 
            device: Union[str, torch.device],
            epochs: int,
            validator: AbstractValidator,
            early_stopper: Optional[EarlyStopper],
            checkpoint_interval: int,
        ) -> None:

        super().__init__(
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epochs=epochs,
            validator=validator,
            early_stopper=early_stopper,
            checkpoint_interval=checkpoint_interval,
        )

    
    def training_step(
            self, 
            model: nn.Module, 
            batch: Tuple[torch.Tensor, torch.Tensor]
        ) -> float:
        """Train the model on a single batch and return the batch loss.

        Parameters
        ----------
        model : nn.Module
            Model being trained.

        batch : Tuple[torch.Tensor, torch.Tensor]
            Batch composed of images and labels.

        Returns
        -------
        float
            Training loss for the batch.
        """

        if len(batch) != 2:
            self.logger.error("Invalid batch: expected a tuple of (images, labels)")
            raise ValueError()

        imgs, labels = batch

        check_shape(imgs, (-1, 3, 224, 224))
        B, C, H, W = imgs.shape
        check_shape(labels, (B,))

        imgs = imgs.to(self.device)
        labels = labels.to(self.device).long()

        preds: torch.Tensor = model(imgs)
        loss: torch.Tensor = self.loss_fn(preds, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    

    def train(self, model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader) -> nn.Module:
        """Run the training loop for the specified number of epochs.

        Parameters
        ----------
        model : nn.Module
            Model to train.

        train_dataloader : DataLoader
            Training data loader.

        val_dataloader : DataLoader
            Validation data loader.
        """
        
        logger = get_formatted_logger()
        rank = get_process_rank()

        logger.info(f"[Process {rank}] Start training")

        model = model.to(self.device)


        ### Training Loop ###

        for e in range(1, self.epochs+1):

            dist_barrier()

            model.train()
            train_dataloader.sampler.set_epoch(e)

            epoch_loss = torch.zeros(1, device=self.device, dtype=torch.float32)

            # Loop on batches
            for batch in tqdm(train_dataloader, desc=f"Training model: Epoch {e}/{self.epochs} ", unit="batch", leave=False, position=rank):
                batch_loss = self.training_step(model, batch)
                epoch_loss += batch_loss
            
            epoch_loss = epoch_loss / len(train_dataloader)

            # Reduce loss across all processes
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.AVG)
            epoch_loss = epoch_loss.item()

            
            ### Validation ###

            val_loss, val_metrics = self.validator.validate(model, val_dataloader, e)


            ### Checkpointing and Logging ###

            is_new_best, stop = self.early_stopper.set_epoch_metric(val_loss)

            # Only the main process will execute
            if rank == 0:

                logger.info(f"Epoch Training Loss: {epoch_loss}")
                logger.info(f"Epoch Validation Loss: {val_loss}")

                for k, v in val_metrics.items():
                    logger.info(f"Validation {k}: {v}")

                if is_new_best:
                    save_model_checkpoint(model.module, {}, "best")
                save_model_checkpoint(model.module, {}, "last")

                metrics = {
                    "Training Cross Entropy Loss": epoch_loss,
                    "Validation Cross Entropy Loss": val_loss,
                    **val_metrics,
                }

                mlflow.log_metrics(metrics=metrics, step=e)
                
            if stop: 
                if get_process_rank() == 0:
                    logger.info(f"Early stopping triggered at epoch {e}.")
                break

        return model.module


def main(log_level: str) -> None:

    # This will be automatically set by run_distributed_function(...)
    device = torch.cuda.current_device()

    get_formatted_logger(level=log_level)


    ### Data Preparation ###

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
    ])

    train_dataset = MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    val_dataset = MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    # These are automatically set by run_distributed_function(...)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        sampler=val_sampler,
    )


    ### Model ###

    model = resnet50(num_classes=10)
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[device])


    ### Training Setup ###

    lr=0.001
    optimizer = AdamW(params=model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()

    seed = 47
    patience=5
    epochs=100

    set_seed(seed)

    validator = MNISTValidator(
        device=device,
        loss_fn=loss_fn,
        num_classes=10,
    )

    early_stopper = EarlyStopper(
        delta_for_new_best=0.1,
        delta_type="relative",
        criterion="min",
        patience=patience,
    )

    trainer = MNISTTrainer(
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=epochs,
        validator=validator,
        early_stopper=early_stopper,
        checkpoint_interval=10,
    )


    ### MLFlow Logging ###
    if get_process_rank() == 0:
    
            start_mlflow_run(
                experiment_name="MNIST",
                mlflow_kwargs={
                    "run_name": "ResNet50",
                    "tags": {
                        "model": "resnet50",
                        "pre-trained": "false",
                    },
                    "log_system_metrics": True,
                    "description": "Simple showcase on how the toolkit shall be used."
                }
            )
    
            log_run_parameters(
                {
                    "epochs": epochs,
                    "patience": patience,
                    "seed": seed,
                    "lr": lr,
                    "Dataset": "MNIST",
                    "model": "ResNet50",
                }
            )


    ### Model Training ###

    model = trainer.train(model, train_loader, val_loader)


    ### Cleanup ###

    if get_process_rank() == 0:
        mlflow.end_run()

    dist.destroy_process_group()



if __name__ == "__main__":

    os.environ["MLFLOW_ENDPOINT_URL"] = "sqlite:///data/mlflow_folder/mlflow.db"

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        logger = get_formatted_logger()
        logger.error("The script must be ran using `torchrun` and not `python`")
        raise SyntaxError()

    from aidevelopementtoolkit.torch_utils.distributed_torch_utils import run_distributed_function

    # devices = [0, 1, 2, 3] # Supposing 4 GPUs on the current node
    devices = [0] # Supposing single GPU on the current node

    run_distributed_function(
        devices=devices,
        fn_to_distribute=main,
        fn_kwargs={"log_level": "DEBUG"},
    )
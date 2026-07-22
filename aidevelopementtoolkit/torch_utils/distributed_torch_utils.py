from typing import List, Callable, Dict, Any
from datetime import timedelta
from os import environ

from aidevelopementtoolkit.logging_utils.logger import get_formatted_logger

import torch
import torch.distributed as dist

logger = get_formatted_logger(name=__name__, level="ERROR")

def run_distributed_function(
        devices: List[str], 
        fn_to_distribute: Callable, 
        fn_kwargs: Dict[str, Any],
    ) -> None:
    """This function must be used to run a function in a distributed
    way using torch distributed library.

    Parameters
    ----------    
    devices : List[int]
        List of visible device IDs.

    fn_to_distribute : Callable
        Funtion to be distributed.

    fn_kwargs : Dict[str, Any]
        Kwargs to be passed to the function.

    Examples
    --------
    Define a function to execute on every GPU:

    >>> import torch
    >>> def train_step(epochs: int):
    ...     rank = get_process_rank()
    ...     print(f"Running process {rank}")
    ...     model = torch.nn.Linear(10, 1).cuda()
    ...     # Training logic here
    ...
    >>> run_distributed_function(
    ...     devices=["0", "1"],
    ...     fn_to_distribute=train_step,
    ...     fn_kwargs={"epochs": 10},
    ... )

    The command used to launch the script should specify the number of
    processes per node:

    .. code-block:: bash

        torchrun --nproc_per_node=2 train.py

    Notes
    -----
    The number of launched processes must match the number of devices
    provided. For example, `--nproc_per_node=4` requires
    `devices=["0", "1", "2", "3"]`.
    """

    # Get torchrun environment variables
    local_rank = int(environ.get("LOCAL_RANK", 0))
    global_rank = int(environ.get("RANK", 0))
    world_size = int(environ.get("WORLD_SIZE", 1))
    local_world_size = int(environ.get("LOCAL_WORLD_SIZE", len(devices)))

    # Handle mismatched launch
    if local_rank >= len(devices):
        logger.error(f"Configuration error: `LOCAL_RANK`={local_rank} exceeds `devices` ({len(devices)}).")
        raise ValueError()

    elif local_world_size != len(devices):
        logger.error(
            f"Configuration error: Started {local_world_size} processes on this node "
            f"but indicated {len(devices)} GPUs (`devices`). These two must be coherent."
        )
        raise ValueError()

    # Restrict visible GPUs for this process
    environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))

    # Launch worker
    _main_worker(
        devices=devices,
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size,
        fn_to_distribute=fn_to_distribute,
        fn_kwargs=fn_kwargs,
    )


def _main_worker(
        devices: List[str],
        local_rank: int,
        global_rank: int,
        world_size: int,
        fn_to_distribute: Callable,
        fn_kwargs: Dict[str, Any],
    ) -> None:
    """
    Worker function to be spawned on each GPU.

    Parameters
    ----------
    devices : List[int]
        List of visible device IDs on this node.

    local_rank : int
        The local GPU index for this process (within the node).

    global_rank : int
        The global rank of this process across all nodes.

    world_size : int
        Total number of processes across all nodes.

    fn_to_distribute : Callable
        Funtion to be distributed.

    fn_kwargs : Dict[str, Any]
        Kwargs to be passed to the function.
    """

    # Select GPU from config
    device_id = devices[local_rank]
    torch.cuda.set_device(device_id)

    # Initialize DDP process group
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        timeout=timedelta(hours=24),
        rank=global_rank,
        device_id=torch.device(f"cuda:{torch.cuda.current_device()}"),
    )

    # Run the function
    fn_to_distribute(**fn_kwargs)

    # Clean up
    dist.destroy_process_group()


def get_process_rank() -> int:
    """This function checks if the torch distributed backend is
    initialized and then eventually returns the process rank.

    Returns
    -------
    int
        Rank of the process if the torch distributed backend is
        initialized, 0 otherwise

    Examples
    --------
    >>> rank = get_process_rank()
    >>> if rank == 0:
    ...     print("Only the main process executes this code.")
    """

    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0
    

def dist_barrier() -> None:
    """This function checks if the torch distributed backend is
    initialized and then performs the barrier to wait all the processes.
    
    Examples
    --------
    Synchronize processes before saving a checkpoint:

    >>> train_model()
    >>> dist_barrier()
    >>> if get_process_rank() == 0:
    ...     save_checkpoint()
    """

    if dist.is_initialized():
        return dist.barrier()
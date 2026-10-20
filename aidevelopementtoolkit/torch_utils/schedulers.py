from abc import ABC, abstractmethod
from math import cos, pi
from typing import Dict, Any

from aidevelopementtoolkit.logging_utils.logger import get_formatted_logger

logger = get_formatted_logger(__name__, level="ERROR")

class AbstractScheduler(ABC):
    """Abstract base class for all schedulers.

    This class defines the common interface that every scheduler must implement.
    A scheduler is responsible for returning a scalar value (e.g. weight,
    learning rate, coefficient) as a function of the current epoch.

    Notes
    -----
    - All derived classes must implement the `get_weight` method.
    - Input validation should be handled inside each concrete implementation.
    - This class does not enforce a specific scheduling strategy.
    """

    @abstractmethod
    def get_weight(self, current_epoch: int) -> float:
        """Returns the scheduled value for a given epoch.

        Parameters
        ----------
        current_epoch : int
            The current epoch for which the scheduler value is requested.

        Returns
        -------
        float
            The scheduled value at `current_epoch`.
        """
        logger.error(
            "Not implemented error: The method `get_weight` must be "
            "implemented in the derived scheduler class.",
            NotImplementedError,
        )

    @staticmethod
    def get_scheduler(class_name: str, scheduler_kwargs: Dict[str, Any]) -> "AbstractScheduler":
        """Returns an instance of the specified scheduler class.

        Parameters
        ----------
        class_name : str
            The name of the scheduler class to instantiate.

        scheduler_kwargs : Dict[str, Any]
            The keyword arguments to pass to the scheduler class constructor.

        Returns
        -------
        AbstractScheduler
            An instance of the specified scheduler class.
        """
        scheduler_class = globals().get(class_name)
        if (
            not isinstance(scheduler_class, type)
            or scheduler_class is AbstractScheduler
            or not issubclass(scheduler_class, AbstractScheduler)
            or scheduler_class.__module__ != __name__
        ):
            logger.error(
                f"Value error: '{class_name}' is not a concrete scheduler class "
                "defined in this module.",
                ValueError,
            )

        return scheduler_class(**scheduler_kwargs)


class ConstantScheduler(AbstractScheduler):
    """This class implements a constant scheduler.

    The scheduler always returns a constant value, regardless of the epoch.

    Parameters
    ----------
    value : float
        The constant value returned by the scheduler at all epochs.
    """

    def __init__(self, value: float):
        self.value = value


    def get_weight(self, current_epoch: int) -> float:
        """Returns the constant value for any given epoch.

        Parameters
        ----------
        current_epoch : int
            The current epoch for which the scheduler value is requested.

        Returns
        -------
        float
            The constant value at `current_epoch`.

        """
        if current_epoch < 0:
            logger.error(
                "Value error: The `current_epoch` must be a non-negative integer, "
                f"given `current_epoch`={current_epoch}"
            )

        return self.value


from aidevelopementtoolkit.logging_utils.logger import get_formatted_logger


class LinearScheduler(AbstractScheduler):
    """This class implements a linear scheduler.

    The scheduler linearly interpolates a value between `start_value` and
    `end_value` from `starting_epoch` to `ending_epoch` (inclusive).

    Parameters
    ----------
    start_value : float
        Initial value returned by the scheduler before (and at) `starting_epoch`.
    
    end_value : float
        Final value returned by the scheduler after (and at) `ending_epoch`.
    
    starting_epoch : int
        Epoch at which the linear interpolation starts.
    
    ending_epoch : int
        Epoch at which the linear interpolation ends.

    Notes
    -----
    - For `current_epoch <= starting_epoch`, the scheduler returns `start_value`.
    - For `current_epoch >= ending_epoch`, the scheduler returns `end_value`.
    - For epochs in-between, it returns a linearly interpolated value.
    """

    def __init__(
            self,
            start_value: float,
            end_value: float,
            starting_epoch: int,
            ending_epoch: int,
        ):

        if ending_epoch < starting_epoch:
            logger.error(
                "Value error: The `ending_epoch` must be >= than `starting_epoch`, "
                f"given `starting_epoch`={starting_epoch} and `ending_epoch`={ending_epoch}"
            )

        self.start_value = start_value
        self.end_value = end_value
        self.starting_epoch = starting_epoch
        self.ending_epoch = ending_epoch
        

    def get_weight(self, current_epoch: int) -> float:
        """Returns the scheduled value for a given epoch.

        Parameters
        ----------
        current_epoch : int
            The current epoch for which the scheduler value is requested.

        Returns
        -------
        float
            The scheduled value at `current_epoch`, following a linear schedule
            between `start_value` and `end_value`.

        """
        if current_epoch < 0:
            logger.error(
                "Value error: The `current_epoch` must be a non-negative integer, "
                f"given `current_epoch`={current_epoch}"
            )

        # Before schedule starts
        if current_epoch <= self.starting_epoch:
            return float(self.start_value)

        # After schedule ends
        if current_epoch >= self.ending_epoch:
            return float(self.end_value)

        # Linear interpolation in-between
        progress = (current_epoch - self.starting_epoch) / (self.ending_epoch - self.starting_epoch)
        return float(self.start_value + progress * (self.end_value - self.start_value))


class CosineScheduler(AbstractScheduler):
    """This class implements a cosine scheduler.

    The scheduler follows a half-cosine curve between `start_value` and
    `end_value` from `starting_epoch` to `ending_epoch` (inclusive).

    Parameters
    ----------
    start_value : float
        Initial value returned before (and at) `starting_epoch`.

    end_value : float
        Final value returned after (and at) `ending_epoch`.

    starting_epoch : int
        Epoch at which the cosine interpolation starts.

    ending_epoch : int
        Epoch at which the cosine interpolation ends.

    Notes
    -----
    - For `current_epoch <= starting_epoch`, the scheduler returns `start_value`.
    - For `current_epoch >= ending_epoch`, the scheduler returns `end_value`.
    - For epochs in-between, it returns a half-cosine interpolation between the
      start and end values.
    """

    def __init__(
            self,
            start_value: float,
            end_value: float,
            starting_epoch: int,
            ending_epoch: int,
        ):

        if ending_epoch < starting_epoch:
            logger.error(
                "Value error: The `ending_epoch` must be >= than `starting_epoch`, "
                f"given `starting_epoch`={starting_epoch} and `ending_epoch`={ending_epoch}"
            )

        self.start_value = start_value
        self.end_value = end_value
        self.starting_epoch = starting_epoch
        self.ending_epoch = ending_epoch

    def get_weight(self, current_epoch: int) -> float:
        """Return the scheduled value for a given epoch.

        Parameters
        ----------
        current_epoch : int
            The current epoch for which the scheduler value is requested.

        Returns
        -------
        float
            The scheduled value at `current_epoch`, following a cosine schedule
            between `start_value` and `end_value`.
        """
        if current_epoch < 0:
            logger.error(
                "Value error: The `current_epoch` must be a non-negative integer, "
                f"given `current_epoch`={current_epoch}"
            )

        if current_epoch <= self.starting_epoch:
            return float(self.start_value)

        if current_epoch >= self.ending_epoch:
            return float(self.end_value)

        progress = (current_epoch - self.starting_epoch) / (self.ending_epoch - self.starting_epoch)
        cosine_decay = (1 + cos(pi * progress)) / 2
        return float(self.end_value + (self.start_value - self.end_value) * cosine_decay)
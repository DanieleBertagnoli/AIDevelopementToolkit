from typing import Literal, Tuple
from numbers import Real

import numpy as np

from aidevelopementtoolkit.logging_utils.logger import get_formatted_logger

logger = get_formatted_logger(name=__name__, level="ERROR")

class EarlyStopper:
    """Track the best observed loss and determine when to early stop.

    This class tracks the best observed loss and determines whether a new
    epoch loss represents a significant improvement according to a configured
    threshold. Improvements can be measured either as an absolute difference
    or as a relative percentage.

    Early stopping is triggered when no significant improvement is observed
    for `patience` consecutive epochs.

    Parameters
    ----------
    delta_type : Literal["absolute", "relative"]
        Defines whether the improvement threshold is interpreted as an
        absolute loss difference or a relative percentage.

    criterion : Literal["min", "max"]
        Defines wheter the monitored value must be minimized or maximized.
        E.g., L1 Loss must be minimized while the F1 Score must be maximized.

    patience : int
        Number of consecutive epochs without improvement before triggering
        early stopping.

    Examples
    --------
    Create an early stopper that stops after 3 epochs without improvement:

    >>> stopper = EarlyStopper(
    ...     delta_for_new_best=0.01,
    ...     delta_type="absolute",
    ...     criterion="min",
    ...     patience=3,
    ... )
    >>> stopper.set_epoch_metric(1.0)
    (True, False)
    >>> stopper.set_epoch_metric(0.995)
    (False, False)
    >>> stopper.set_epoch_metric(0.99)
    (False, False)
    >>> stopper.set_epoch_metric(0.98)
    (True, False)
    """

    def __init__(
            self,
            delta_for_new_best: float,
            delta_type: Literal["absolute", "relative"],
            criterion: Literal["min", "max"],
            patience: int,
        ):

        if delta_type not in ["absolute", "relative"]:
            logger.error(
                f"The given `delta_type`={delta_type} is not valid. "
                "Available ['absolute', 'relative']."
            )
            raise ValueError()

        if criterion not in ["min", "max"]:
            logger.error(
                f"The given `criterion`={criterion} is not valid. "
                "Available ['min', 'max']."
            )
            raise ValueError()

        if delta_for_new_best < 0:
            logger.error(
                f"The given `delta_for_new_best`={delta_for_new_best} is < 0."
            )
            raise ValueError()

        if patience <= 0:
            logger.error(
                f"The given `patience`={patience} must be > 0."
            )
            raise ValueError()

        self.delta_for_new_best = delta_for_new_best
        self.delta_type = delta_type
        self.criterion = criterion
        self.patience = patience

        self.best_loss = None
        self.epochs_without_improvement = 0


    def set_epoch_metric(self, epoch_metric: float) -> Tuple[bool, bool]:
        """Update epoch metric and check improvement and early stopping.

        Parameters
        ----------
        epoch_metric : float
            Value observed for the current epoch.

        Returns
        -------
        Tuple[bool, bool]
            A tuple containing:

            - `is_new_best`:
                `True` if the current loss is a new best value.
            - `early_stop`:
                `True` if the number of epochs without improvement
                reached `patience`.

        Examples
        --------
        >>> stopper = EarlyStopper(
        ...     delta_for_new_best=0.1,
        ...     delta_type="absolute",
        ...     criterion="min",
        ...     patience=2,
        ... )
        >>> stopper.set_epoch_metric(1.0)
        (True, False)
        >>> stopper.set_epoch_metric(0.95)
        (False, False)
        >>> stopper.set_epoch_metric(0.96)
        (False, True)

        >>> stopper = EarlyStopper(
        ...     delta_for_new_best=0.1,
        ...     delta_type="absolute",
        ...     criterion="max",
        ...     patience=2,
        ... )
        >>> stopper.set_epoch_metric(0.5)
        (True, False)
        >>> stopper.set_epoch_metric(0.6)
        (False, False)
        >>> stopper.set_epoch_metric(0.7)
        (True, False)
        """

        if not isinstance(epoch_metric, Real):
            logger.error(
                f"The given `epoch_metric` must be a real number. "
                f"Given type: {type(epoch_metric)}"
            )
            raise TypeError()

        epoch_metric = float(epoch_metric)

        is_new_best = False

        # First observation is always the best
        if self.best_loss is None:
            self.best_loss = epoch_metric
            is_new_best = True

        else:
            # Calculate the improvement according to the criterion
            if self.criterion == "min":
                improvement = self.best_loss - epoch_metric
            else:
                improvement = epoch_metric - self.best_loss

            # Check improvement
            if improvement > 0:

                if self.delta_type == "absolute":
                    is_new_best = improvement > self.delta_for_new_best

                else:
                    if self.best_loss != 0:
                        rel_improvement = improvement / abs(self.best_loss) * 100
                        is_new_best = rel_improvement > self.delta_for_new_best

                if is_new_best:
                    self.best_loss = epoch_metric

        # Update patience counter
        if is_new_best:
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        early_stop = self.epochs_without_improvement >= self.patience

        return is_new_best, early_stop
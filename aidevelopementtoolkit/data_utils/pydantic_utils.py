from typing import Any, Iterable, Sequence

import numpy as np
from pydantic import BaseModel

from aidevelopementtoolkit.logging_utils.logger import get_formatted_logger

logger = get_formatted_logger(name=__name__, level="ERROR")

def models_to_numpy(models: Iterable[BaseModel], fields: Sequence[str]) -> np.ndarray:
    """Extract numeric fields from a sequence of Pydantic models into a NumPy array.

    Parameters
    ----------
    models : Iterable[BaseModel]
        An iterable of Pydantic model instances to extract values from.

    fields : Sequence[str]
        Ordered sequence of dot-separated field paths (case-sensitive) to
        extract from each model. Each path is resolved via `get_field_value`.

    Returns
    -------
    np.ndarray
        Array containing the extracted values. Shape `(N, F)`:
        - `N`: The number of models in the input iterable.
        - `F`: The number of requested fields.

    Examples
    --------
    >>> class Point(BaseModel):
    ...     x: float
    ...     y: float
    >>> points = [Point(x=1.0, y=2.0), Point(x=3.0, y=4.0)]
    >>> models_to_numpy(points, ["x", "y"])
    array([[1., 2.],
           [3., 4.]], dtype=float32)
    """

    if not fields:
        logger.error("Expected at least one field name.")
        raise ValueError()

    rows = [
        [get_field_value(model, field) for field in fields]
        for model in models
    ]
    return np.array(rows, dtype=np.float32)



def get_field_value(
        model: BaseModel,
        field_path: str,
    ) -> Any:
    """Retrieve a nested field value from a Pydantic model by dot-separated path.

    Parameters
    ----------
    model : BaseModel
        The Pydantic model instance to traverse.

    field_path : str
        Dot-separated path to the target field (case-sensitive).
        For example, `"address.city"` retrieves `model.address.city`.

    Returns
    -------
    Any
        The value at the specified field path.

    Examples
    --------
    >>> class Address(BaseModel):
    ...     city: str
    >>> class User(BaseModel):
    ...     name: str
    ...     address: Address
    >>> user = User(name="Alice", address=Address(city="Rome"))
    >>> get_field_value(user, "name")
    'Alice'
    >>> get_field_value(user, "address.city")
    'Rome'
    """
    head, _, tail = field_path.partition(".")
    value = getattr(model, head)
    if tail:
        return get_field_value(value, tail)
    return value


def set_field_value(
        model: BaseModel,
        field_path: str,
        value: Any,
    ) -> None:
    """Set a nested field value on a Pydantic model by dot-separated path.

    Parameters
    ----------
    model : BaseModel
        The Pydantic model instance to modify.

    field_path : str
        Dot-separated path to the target field (case-sensitive).
        For example, `"address.city"` sets `model.address.city`.

    value : Any
        The value to assign to the target field.

    Raises
    ------
    AttributeError
        If any segment of the path does not exist on the current model.

    Examples
    --------
    >>> class Address(BaseModel):
    ...     city: str
    >>> class User(BaseModel):
    ...     model_config = ConfigDict(frozen=False)
    ...     name: str
    ...     address: Address
    >>> user = User(name="Alice", address=Address(city="Rome"))
    >>> set_field_value(user, "name", "Bob")
    >>> user.name
    'Bob'
    >>> set_field_value(user, "address.city", "Milan")
    >>> user.address.city
    'Milan'
    """
    head, _, tail = field_path.partition(".")
    if tail:
        set_field_value(getattr(model, head), tail, value)
    else:
        setattr(model, head, value)
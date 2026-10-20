import logging
import traceback
import sys
from typing import Literal, Optional

def get_formatted_logger(
        name: Optional[str] = "Main", 
        level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    ) -> logging.Logger:
    """This function returns a logger instance with the specified name and the
    given loglevel.

    Parameters
    ----------
    name : Optional[str], default="Main"
        Name of the logger. When `None` is provided, the root logger is returned.

    level : Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO"
        Level to which the logger will be initialized. **This level is ignored 
        if the logger already exists and has a level set**.

    Returns
    -------
    logging.Logger
        A logger instance.

    Examples
    --------
    >>> logger = get_formatted_logger(name="test_logger", level="DEBUG")
    >>> logger.info("Model training completed!")
    """

    # Set logger class
    logging.setLoggerClass(CustomLogger)

    # Set logger
    logger = logging.getLogger(name)

    # If the logger already has handlers, we don't want to add another one.
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    
    # Set formatter
    formatter = ColorFormatter(
        "%(name)s %(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Set log level
    logger.setLevel(getattr(logging, level))

    return logger


class ColorFormatter(logging.Formatter):
    """Format log records with ANSI-colored levels and severity spacing.

    The level name is wrapped in an ANSI color code. Records at INFO,
    WARNING, ERROR, and CRITICAL levels are prefixed with a newline.
    """

    COLORS = {
        "DEBUG": "\033[36m", # Cyan
        "INFO": "\033[33m", # Yellow
        "WARNING": "\033[38;5;208m", # Orange
        "ERROR": "\033[31m", # Red
        "CRITICAL": "\033[41m", # Red background
    }

    BOLD = "\033[1m"
    RESET = "\033[0m"

    def format(self, record):
        levelname = record.levelname
        color = self.COLORS.get(levelname, self.RESET)
        error_type = getattr(record, "passed_error_type", None)
        label = f"{error_type}" if error_type else levelname

        # Bold and color
        record.levelname = f"{self.BOLD}{color}[{label}]{self.RESET}"

        # Let the parent formatter build the final line
        formatted = super().format(record)

        # Add newline BEFORE the date for INFO and ERROR
        if levelname in ("INFO", "ERROR", "CRITICAL", "WARNING"):
            formatted = f"\n{formatted}"

        return formatted


class CustomLogger(logging.Logger):
    """A standard logger that can raise an exception after logging an error.

    All methods retain :class:`logging.Logger` behavior except for
    :meth:`error`, which optionally accepts an exception class as its first
    positional argument after the message.
    """

    def error(self, msg, *args, **kwargs):
        """Log an error message and optionally raise an exception.

        Parameters
        ----------
        msg : object
            Message to log. If formatting arguments are supplied, the message
            is formatted using the standard logging `%`-style convention.
        *args : object
            Formatting arguments, optionally preceded by an exception class.
            If the first positional argument is an exception class, it is
            removed from the formatting arguments and raised after the message
            is logged.
        **kwargs : object
            Keyword arguments forwarded to :meth:`logging.Logger.error`.

        Examples
        --------
        Log an error without raising:

        >>> logger.error("Unable to load the model")

        Log an error and then raise an exception:

        >>> logger.error("Checkpoint not found", FileNotFoundError)
        Traceback (most recent call last):
        ...
        FileNotFoundError: Checkpoint not found
        """
        exception_type = None

        # Detect: logger.error("message", FileNotFoundError)
        if args and isinstance(args[0], type):
            if issubclass(args[0], BaseException):
                exception_type = args[0]
                args = args[1:]

        if exception_type is not None:
            traceback.print_stack()
            extra = dict(kwargs.get("extra") or {})
            extra["passed_error_type"] = exception_type.__name__
            kwargs["extra"] = extra

        # Log using the standard logging implementation
        super().error(msg, *args, **kwargs)

        sys.exit(1)
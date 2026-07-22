import logging
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

    # Set logger
    logger = logging.getLogger(name)

    # If the logger already has handlers, we don't want to add another one.
    if logger.hasHandlers():
        return logger

    handler = logging.StreamHandler()
    
    # Set formatter
    formatter = ColorFormatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Set log level
    logger.setLevel(getattr(logging, level))

    return logger


class ColorFormatter(logging.Formatter):
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

        # Bold and color
        record.levelname = f"{self.BOLD}{color}[{levelname}]{self.RESET}"

        # Let the parent formatter build the final line
        formatted = super().format(record)

        # Add newline BEFORE the date for INFO and ERROR
        if levelname in ("INFO", "ERROR", "CRITICAL", "WARNING"):
            formatted = f"\n{formatted}"

        return formatted
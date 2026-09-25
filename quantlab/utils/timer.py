"""A context manager that logs how long a block of work took."""

import time
from loguru import logger


class Timer:
    """Log the start of a task on entry and its elapsed wall-clock time on exit.

    Parameters
    ----------
    task_name : str
        Name used in both log lines.

    Attributes
    ----------
    timein : float
        Elapsed seconds of the last completed block; ``0`` before any block
        has finished.

    Examples
    --------
    >>> with Timer("compute factors") as timer:
    ...     factors.cal()
    >>> elapsed_seconds = timer.timein
    """

    def __init__(self, task_name: str):
        """Initialize the timer; see the class docstring for parameters.

        Nothing is measured until the ``with`` block is entered.
        """
        self.timein = 0
        self.task_name = task_name

    def __enter__(self):
        """Log the task start and record the start time."""
        logger.info(f"Starting {self.task_name}")
        self.timestart = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Store the elapsed seconds in ``timein`` and log them.

        Exceptions raised inside the block are not suppressed.
        """
        self.timeend = time.perf_counter()
        self.timein = self.timeend - self.timestart
        logger.info(f"{self.task_name} consumed time: {self.timein:.2f}s")

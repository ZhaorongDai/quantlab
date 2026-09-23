"""A context manager that logs how long a block of work took."""

import time
from loguru import logger


class Timer:
    """Log the start of a task on entry and its elapsed wall-clock time on exit.

    Example:
        >>> with Timer("compute factors") as timer:
        ...     factors.cal()
        >>> timer.timein  # elapsed seconds
        12.34
    """

    def __init__(self, task_name: str):
        """Create a timer for ``task_name``; nothing is measured until entry."""
        self.timein = 0
        self.task_name = task_name

    def __enter__(self):
        """Log the task start and record the start time."""
        logger.info(f"Starting {self.task_name}")
        self.timestart = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Store the elapsed seconds in ``timein`` and log them."""
        self.timeend = time.perf_counter()
        self.timein = self.timeend - self.timestart
        logger.info(f"{self.task_name} consumed time: {self.timein:.2f}s")

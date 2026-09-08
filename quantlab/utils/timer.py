import time
from loguru import logger


class Timer:
    def __init__(self, task_name: str):
        self.timein = 0
        self.task_name = task_name

    def __enter__(self):
        logger.info(f"Starting {self.task_name}")
        self.timestart = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timeend = time.perf_counter()
        self.timein = self.timeend - self.timestart
        logger.info(f"{self.task_name} consumed time: {self.timein:.2f}s")

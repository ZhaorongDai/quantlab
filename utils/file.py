from pathlib import Path
from typing import Literal

import pandas as pd

from enums.constant import Date


def file_date_filter(
    path: Path | list[Path],
    start_date: str = Date.START_DATE,
    end_date: str = Date.END_DATE,
    period: Literal["month"] = "month",
) -> list[Path]:
    if not isinstance(path, list):
        path = [path]
    if period == "month":
        date = ["-".join(p.name.split(".")[0].split("-")[-2:]) for p in path]

    date = [pd.to_datetime(d) for d in date]
    path = [
        p
        for p, d in zip(path, date)
        if pd.to_datetime(start_date) <= d <= pd.to_datetime(end_date)
    ]
    return path


def get_csv_files(dir_path: str) -> list[Path]:
    assert Path(dir_path).exists(), f"{dir_path} does not exist"
    return sorted(Path(dir_path).rglob("*.csv"))


def get_pqt_files(dir_path: str) -> list[Path]:
    assert Path(dir_path).exists(), f"{dir_path} does not exist"
    return sorted(Path(dir_path).rglob("*.pqt"))


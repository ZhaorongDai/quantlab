"""Small filesystem helpers for locating and date-filtering raw data files.

The Binance spot kline dataset reads one CSV per symbol per month; these
functions list such files under a directory and keep only those whose month
falls inside a requested date range.
"""

from pathlib import Path
from typing import Literal

import pandas as pd

from quantlab.enums.constant import Date


def file_date_filter(
    path: Path | list[Path],
    start_date: str = Date.START_DATE,
    end_date: str = Date.END_DATE,
    period: Literal["month"] = "month",
) -> list[Path]:
    """Keep the files whose name-encoded month lies within ``[start_date, end_date]``.

    The month is read from the file stem, which must end in ``-YYYY-MM`` (for
    example ``BTCUSDT-1m-2024-01.csv``). Both bounds are inclusive and are
    compared as timestamps against the first day of that month.

    Parameters
    ----------
    path : Path | list[Path]
        One path or a list of paths to filter.
    start_date : str, default Date.START_DATE
        Earliest date to keep, as a parseable date string.
    end_date : str, default Date.END_DATE
        Latest date to keep, as a parseable date string.
    period : Literal['month'], default "month"
        Granularity encoded in the file name; only ``"month"`` is
        supported.

    Returns
    -------
    list[Path]
        The subset of ``path`` inside the range, in the original order.

    Examples
    --------
    For a directory holding one file per month from 2023-12 to 2024-03:

    >>> files = get_csv_files("/data/klines/BTCUSDT")
    >>> [p.name for p in file_date_filter(files, "2024-01-01", "2024-02-15")]
    ['BTCUSDT-1d-2024-01.csv', 'BTCUSDT-1d-2024-02.csv']
    """
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
    """Return every ``*.csv`` under ``dir_path`` (recursively), sorted by path.

    Parameters
    ----------
    dir_path : str
        Directory to search.

    Returns
    -------
    list[Path]
        The matching files, sorted.

    Raises
    ------
    AssertionError
        If ``dir_path`` does not exist.

    Examples
    --------
    >>> [p.name for p in get_csv_files("/data/klines")]
    ['BTCUSDT-1d-2024-01.csv', 'ETHUSDT-1d-2024-01.csv']
    """
    assert Path(dir_path).exists(), f"{dir_path} does not exist"
    return sorted(Path(dir_path).rglob("*.csv"))


def get_pqt_files(dir_path: str) -> list[Path]:
    """Return every ``*.pqt`` (parquet) under ``dir_path`` (recursively), sorted by path.

    Parameters
    ----------
    dir_path : str
        Directory to search.

    Returns
    -------
    list[Path]
        The matching files, sorted.

    Raises
    ------
    AssertionError
        If ``dir_path`` does not exist.

    Examples
    --------
    >>> [p.name for p in get_pqt_files("/data/nasdaq")]
    ['AAPL.pqt', 'MSFT.pqt']
    """
    assert Path(dir_path).exists(), f"{dir_path} does not exist"
    return sorted(Path(dir_path).rglob("*.pqt"))

"""Download the Fama-French three factors from Kenneth French's data library.

The library publishes the market excess return (``Mkt-RF``), the size
factor (``SMB``), the value factor (``HML``) and the risk-free rate (``RF``)
as percentages, daily and monthly, in a zipped CSV with a text preamble.
This script downloads one of the two files, keeps the four series, converts
them to decimal returns (0.01 means 1%) and writes them as

    date,mkt_rf,smb,hml,risk_free

to ``<download-dir>/fama_french/ff3_daily.csv`` (or ``ff3_monthly.csv``).
That is the file ``quantlab.factor.residual_momentum.ResidualMomentumFF3``
reads through its ``fama_french_csv`` parameter. Monthly rows are dated at
the month's end. ``--download-dir`` defaults to the current directory, like
the WRDS scripts.

Usage::

    uv run python scripts/fama_french.py
    uv run python scripts/fama_french.py --monthly --download-dir /data/quantlab
"""

import argparse
import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

LIBRARY = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "daily": "F-F_Research_Data_Factors_daily_CSV.zip",
    "monthly": "F-F_Research_Data_Factors_CSV.zip",
}
COLUMNS = ["date", "mkt_rf", "smb", "hml", "risk_free"]


def download_ff3(frequency: str, timeout: int = 60) -> pd.DataFrame:
    """Download and parse one Fama-French three-factor file.

    Parameters
    ----------
    frequency : {"daily", "monthly"}
        Which file to download.
    timeout : int, default 60
        Seconds to wait for the HTTP response.

    Returns
    -------
    pd.DataFrame
        The four series as decimal returns, indexed by ``date`` ascending.
        The monthly file also carries annual rows after the monthly block;
        parsing stops at the first row that is not a date, so they are
        left out.
    """
    response = requests.get(LIBRARY + FILES[frequency], timeout=timeout)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        text = archive.read(archive.namelist()[0]).decode("utf-8-sig")

    lines = text.splitlines()
    header = next(
        i for i, line in enumerate(lines) if {"Mkt-RF", "SMB", "HML", "RF"} <= {
            field.strip() for field in line.split(",")
        }
    )
    pattern = re.compile(r"\d{8}" if frequency == "daily" else r"\d{6}")
    rows = []
    for line in lines[header + 1:]:
        fields = [field.strip() for field in line.split(",")]
        if not fields or not pattern.fullmatch(fields[0]):
            break
        rows.append(fields[:5])

    table = pd.DataFrame(rows, columns=COLUMNS)
    table["date"] = pd.to_datetime(
        table["date"], format="%Y%m%d" if frequency == "daily" else "%Y%m"
    )
    if frequency == "monthly":
        table["date"] += pd.offsets.MonthEnd(0)
    table[COLUMNS[1:]] = table[COLUMNS[1:]].apply(pd.to_numeric, errors="coerce") / 100.0
    return table.set_index("date").sort_index()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the Fama-French three factors as a decimal-return CSV."
    )
    parser.add_argument(
        "--monthly", action="store_true",
        help="download the monthly file instead of the daily one",
    )
    parser.add_argument(
        "--download-dir", default=".",
        help="directory the fama_french/ folder is written under (default: current directory)",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    frequency = "monthly" if args.monthly else "daily"
    table = download_ff3(frequency)
    out = Path(args.download_dir) / "fama_french" / f"ff3_{frequency}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out)
    print(
        f"{out}: {len(table)} {frequency} rows, "
        f"{table.index[0].date()} .. {table.index[-1].date()}"
    )


if __name__ == "__main__":
    main()

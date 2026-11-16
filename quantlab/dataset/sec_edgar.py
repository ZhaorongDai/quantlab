"""SEC EDGAR's quarterly filing indexes: where each lives in the raw tier, and how it is read.

EDGAR publishes, for every calendar quarter since 1993, a master index of the
filings made in it (``https://www.sec.gov/Archives/edgar/full-index/<YYYY>/QTR<n>/master.idx``):
a few lines of description, then one line per filing and filer,
``CIK|Company Name|Form Type|Date Filed|Filename``, where ``Filename``
(``edgar/data/<cik>/<accession>.txt``) names the filing's accession number.
The current quarter's index is rebuilt every night through the previous
business day; a past quarter's no longer changes.

The raw tier keeps each quarter's index whole, as EDGAR served it,
gzip-compressed, beside a dated watermark, under the EDGAR directory of the
download root::

    <root>/downloads/sec_edgar/full_index/2024QTR1/master.idx.gz
    <root>/downloads/sec_edgar/full_index/2024QTR1/_watermark.json

The watermark's ``through`` is the UTC day of the quarter's last pull,
whether or not the pull changed the file. The whole index is kept, not only
the forms one panel reads, so any other form's filing dates can be read from
the same files; compressed, a quarter takes a few megabytes.

The writer (``quantlab.acquisition.sec_edgar``) takes every path from here;
the readers are ``read_watermark``, ``pulled_quarters``, ``complete_through``,
``parse_master_index`` and ``filing_dates``. This module only reads.

Examples
--------
>>> index_file("/d/sec_edgar", 2024, 1)
PosixPath('/d/sec_edgar/full_index/2024QTR1/master.idx.gz')
>>> quarter_of(date(2024, 5, 15)), quarter_end(2024, 2)
((2024, 2), datetime.date(2024, 6, 30))
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from quantlab.utils.watermark import WATERMARK_FILE
from quantlab.utils.watermark import read_watermark as _read_watermark

#: The vendor directory under the download root.
VENDOR_DIR = "sec_edgar"

#: The directory of the quarterly master indexes, under the vendor's.
FULL_INDEX = "full_index"

#: One quarter's index file, gzip-compressed.
INDEX_FILE = "master.idx.gz"

#: The 13F holdings reports and their amendments, as the index names their forms.
THIRTEEN_F_FORMS: tuple[str, ...] = ("13F-HR", "13F-HR/A")

#: Days from a pull's UTC day back to the last filing day its index surely holds: the index is rebuilt
#: each night through the previous business day, and that night's build may finish after the UTC day turns.
INDEX_LAG_DAYS = 2

#: The column header line of a master index; the filings follow it, after a line of dashes.
HEADER = "CIK|Company Name|Form Type|Date Filed|Filename"

#: The accession number at the end of an index line's ``Filename``.
_ACCESSION = re.compile(r"(\d{10}-\d{2}-\d{6})\.txt$")

#: A quarter directory's name.
_QUARTER_DIR = re.compile(r"^(\d{4})QTR([1-4])$")


def quarter_of(day: date) -> tuple[int, int]:
    """Return the ``(year, quarter)`` of the calendar quarter holding ``day``.

    Examples
    --------
    >>> quarter_of(date(2024, 12, 31))
    (2024, 4)
    """
    return day.year, (day.month - 1) // 3 + 1


def quarter_end(year: int, quarter: int) -> date:
    """Return the last day of a calendar quarter.

    Examples
    --------
    >>> quarter_end(2024, 1)
    datetime.date(2024, 3, 31)
    """
    return (date(year + 1, 1, 1) if quarter == 4 else date(year, 3 * quarter + 1, 1)) - timedelta(days=1)


def next_quarter(year: int, quarter: int) -> tuple[int, int]:
    """Return the calendar quarter after ``(year, quarter)``.

    Examples
    --------
    >>> next_quarter(2024, 4)
    (2025, 1)
    """
    return (year + 1, 1) if quarter == 4 else (year, quarter + 1)


def quarter_dir(vendor_root: str | Path, year: int, quarter: int) -> Path:
    """Return the directory of one quarter's index and watermark.

    Examples
    --------
    >>> quarter_dir("/d/sec_edgar", 2013, 2)
    PosixPath('/d/sec_edgar/full_index/2013QTR2')
    """
    if quarter not in (1, 2, 3, 4):
        raise ValueError(f"SEC EDGAR: a quarter is 1 to 4, got {quarter!r}.")
    return Path(vendor_root) / FULL_INDEX / f"{year:04d}QTR{quarter}"


def index_file(vendor_root: str | Path, year: int, quarter: int) -> Path:
    """Return the path of one quarter's compressed master index.

    Examples
    --------
    >>> index_file("/d/sec_edgar", 2013, 2).name
    'master.idx.gz'
    """
    return quarter_dir(vendor_root, year, quarter) / INDEX_FILE


def watermark_file(vendor_root: str | Path, year: int, quarter: int) -> Path:
    """Return the path of one quarter's watermark.

    Examples
    --------
    >>> watermark_file("/d/sec_edgar", 2013, 2)
    PosixPath('/d/sec_edgar/full_index/2013QTR2/_watermark.json')
    """
    return quarter_dir(vendor_root, year, quarter) / WATERMARK_FILE


def read_watermark(vendor_root: str | Path, year: int, quarter: int) -> date | None:
    """Return the UTC day a quarter's index was last pulled; ``None`` if never.

    Examples
    --------
    >>> read_watermark("/nonexistent/sec_edgar", 2024, 1) is None
    True
    """
    return _read_watermark(watermark_file(vendor_root, year, quarter))


def pulled_quarters(vendor_root: str | Path) -> list[tuple[int, int]]:
    """Return every quarter with a pulled index (a watermark and its file), oldest first.

    Examples
    --------
    >>> pulled_quarters("/nonexistent/sec_edgar")
    []
    """
    root = Path(vendor_root) / FULL_INDEX
    if not root.is_dir():
        return []
    found = []
    for path in root.iterdir():
        match = _QUARTER_DIR.match(path.name)
        if match and (path / WATERMARK_FILE).exists() and (path / INDEX_FILE).exists():
            found.append((int(match.group(1)), int(match.group(2))))
    return sorted(found)


def complete_through(vendor_root: str | Path) -> date | None:
    """Return the last day every filing of which the raw tier's indexes hold, from its oldest quarter on.

    A quarter's index holds its filings through the earlier of its quarter's
    end and ``INDEX_LAG_DAYS`` before its last pull's UTC day. Walking the
    quarters from the oldest pulled, the raw tier is complete through the
    first quarter that falls short of its end, or through the quarter before
    the first one missing.

    Returns
    -------
    date or None
        ``None`` when no quarter was pulled.

    Examples
    --------
    >>> complete_through("/nonexistent/sec_edgar") is None
    True

    With 2024 Q1 pulled on 2024-04-20 and Q2 on 2024-05-10, the indexes hold
    every filing through 2024-05-08::

        complete_through(root)  # datetime.date(2024, 5, 8)
    """
    quarters = pulled_quarters(vendor_root)
    through = None
    expected = quarters[0] if quarters else None
    for year, quarter in quarters:
        if (year, quarter) != expected:
            break  # a quarter missing: nothing after it counts
        end = quarter_end(year, quarter)
        pulled = read_watermark(vendor_root, year, quarter)
        through = min(end, pulled - timedelta(days=INDEX_LAG_DAYS))
        if through < end:
            break
        expected = next_quarter(year, quarter)
    return through


def parse_master_index(data: bytes, forms: Iterable[str] = THIRTEEN_F_FORMS) -> pl.DataFrame:
    """Return the filings of the given forms in one master index: ``accession``, ``date_filed``.

    Parameters
    ----------
    data : bytes
        The index as EDGAR serves it (uncompressed).
    forms : iterable of str, default ``THIRTEEN_F_FORMS``
        The form types kept, as the index spells them.

    Returns
    -------
    pl.DataFrame
        One row per accession (a filing listed under several filers once),
        ``date_filed`` a date; sorted by accession.

    Raises
    ------
    ValueError
        If the data has no master-index header line.

    Examples
    --------
    >>> data = (b"Description: Master Index of EDGAR Dissemination Feed\\n\\n"
    ...         b"CIK|Company Name|Form Type|Date Filed|Filename\\n"
    ...         b"--------------------------------------------------------------------------------\\n"
    ...         b"1000097|KINGDON CAPITAL MANAGEMENT, L.L.C.|13F-HR|2024-02-14|edgar/data/1000097/0000919574-24-001026.txt\\n"
    ...         b"1000045|NICHOLAS FINANCIAL INC|10-Q|2024-02-13|edgar/data/1000045/0000950170-24-014566.txt\\n")
    >>> parse_master_index(data).rows()
    [('0000919574-24-001026', datetime.date(2024, 2, 14))]
    """
    text = data.decode("latin-1")
    start = text.find(HEADER)
    if start < 0:
        raise ValueError("SEC EDGAR: not a master index (no 'CIK|Company Name|Form Type|Date Filed|Filename' line).")
    wanted = set(forms)
    rows = []
    for line in text[start:].splitlines()[2:]:  # the header and its line of dashes
        fields = line.rsplit("|", 3)  # a company name may hold a '|'; the last three fields cannot
        if len(fields) != 4 or fields[1] not in wanted:
            continue
        match = _ACCESSION.search(fields[3].strip())
        if match:
            rows.append((match.group(1), fields[2].strip().replace("-", "")))
    frame = pl.DataFrame(rows, schema={"accession": pl.String, "date_filed": pl.String}, orient="row")
    return (
        frame.with_columns(pl.col("date_filed").str.to_date("%Y%m%d", strict=False))
        .drop_nulls()
        .group_by("accession")
        .agg(pl.col("date_filed").min())
        .sort("accession")
    )


def read_index(vendor_root: str | Path, year: int, quarter: int) -> bytes:
    """Return one quarter's master index as EDGAR served it, uncompressed.

    Raises
    ------
    FileNotFoundError
        If the quarter was never pulled.

    Examples
    --------
    With a raw tier on disk::

        read_index("/data/downloads/sec_edgar", 2024, 1)[:40]
    """
    return gzip.decompress(index_file(vendor_root, year, quarter).read_bytes())


def filing_dates(vendor_root: str | Path, forms: Iterable[str] = THIRTEEN_F_FORMS) -> pl.DataFrame:
    """Return the filing date of every filing of the given forms in the raw tier's indexes.

    Parameters
    ----------
    vendor_root : str or Path
        ``<root>/downloads/sec_edgar``.
    forms : iterable of str, default ``THIRTEEN_F_FORMS``
        The form types kept.

    Returns
    -------
    pl.DataFrame
        ``accession`` and ``date_filed`` (the earliest, should two quarters
        list one accession); empty when nothing was pulled.

    Examples
    --------
    >>> filing_dates("/nonexistent/sec_edgar").columns
    ['accession', 'date_filed']
    """
    forms = tuple(forms)
    frames = [parse_master_index(read_index(vendor_root, *quarter), forms) for quarter in pulled_quarters(vendor_root)]
    if not frames:
        return pl.DataFrame(schema={"accession": pl.String, "date_filed": pl.Date})
    return pl.concat(frames).group_by("accession").agg(pl.col("date_filed").min()).sort("accession")

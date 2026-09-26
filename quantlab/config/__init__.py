"""Config factories for the US-equity datasets, acquisitions and universe table.

Each factory builds one config dataclass from ``quantlab.base.config`` with
every storage path derived from a single data root. ``get_data_root`` resolves
that root from, in order, the process-level override set through
``set_data_root``, the ``QUANTLAB_DATA_DIR`` environment variable, and
finally a ``data/`` directory beside the repository, so a fresh clone works
with no configuration. Beneath the root, raw downloads live under
``downloads/{market}/{frequency}/``, Zarr stores under
``data/{market}/{frequency}/`` and reference tables under ``data/reference/``.

The factories snapshot their paths as plain strings, so a root override must
be applied before the first factory call.

Examples
--------
>>> from quantlab.config import set_data_root, stock_kline_config
>>> set_data_root("/mnt/quant")
PosixPath('/mnt/quant')
>>> cfg = stock_kline_config(start_date="2020-01-01", symbols=("AAPL",))
>>> cfg.zarr_file_path
'/mnt/quant/data/us_equity/1d/stock.zarr'
"""

import os
from pathlib import Path

from quantlab.base.config import (
    AcquisitionConfig,
    DatasetConfig,
    UniverseConfig,
)
from quantlab.enums.data import Frequency, Market, Vendor

#: Process-level storage-root override, set by ``set_data_root`` and consulted
#: first by ``get_data_root``. ``None`` means not overridden.
_DATA_ROOT_OVERRIDE: Path | None = None


def set_data_root(path: "str | os.PathLike | None") -> Path | None:
    """Set the process-level storage root, ahead of the environment variable.

    The value is passed through ``expanduser`` but not ``resolve``: the
    ``QUANTLAB_DATA_DIR`` value is used unresolved too, so the two behave the
    same on a symlinked root, and a quoted ``'~/x'`` from a shell reaches Python
    with a literal tilde that would otherwise become a directory named ``~``.
    The directory is neither created nor required to exist.

    Parameters
    ----------
    path : str | os.PathLike | None
        The new root. ``None`` clears the override so that
        ``QUANTLAB_DATA_DIR`` or the repository default applies again.

    Returns
    -------
    Path | None
        The stored ``Path``, or ``None`` when the override was cleared.

    Raises
    ------
    ValueError
        If ``path`` is an empty or whitespace-only string. An
        empty environment variable falls through to the default, but a
        root someone typed explicitly should not silently mean "default".

    Examples
    --------
    >>> set_data_root("/mnt/quant")
    PosixPath('/mnt/quant')
    >>> set_data_root(None) is None
    True
    """
    global _DATA_ROOT_OVERRIDE
    if path is None:
        _DATA_ROOT_OVERRIDE = None
        return None
    if isinstance(path, str) and not path.strip():
        raise ValueError(
            "data root must be a non-empty path; got an empty/whitespace "
            "value. Omit the setting entirely to fall back to "
            "QUANTLAB_DATA_DIR or the repo-root data/ directory."
        )
    _DATA_ROOT_OVERRIDE = Path(path).expanduser()
    return _DATA_ROOT_OVERRIDE


def get_data_root() -> Path:
    """Return the storage root every config factory derives its paths from.

    Resolution order: the override set by ``set_data_root``, then the
    ``QUANTLAB_DATA_DIR`` environment
    variable, then the ``data/`` directory beside the repository root.

    Examples
    --------
    >>> set_data_root("/mnt/quant")
    PosixPath('/mnt/quant')
    >>> get_data_root()
    PosixPath('/mnt/quant')
    """
    if _DATA_ROOT_OVERRIDE is not None:
        return _DATA_ROOT_OVERRIDE
    env_value = os.environ.get("QUANTLAB_DATA_DIR")
    if env_value:
        return Path(env_value)
    # This file is quantlab/config/__init__.py, so the repository root is
    # three parents up.
    return Path(__file__).resolve().parent.parent.parent / "data"


def _market_data_root(market: str, frequency: str) -> Path:
    """Return the ``data/{market}/{frequency}`` directory Zarr stores live in."""
    return get_data_root() / "data" / market / frequency


def _market_downloads_root(market: str, frequency: str) -> Path:
    """Return the ``downloads/{market}/{frequency}`` directory for raw data."""
    return get_data_root() / "downloads" / market / frequency


def stock_kline_config(
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: tuple[str, ...] | None = None,
    kwargs: dict = None,  # type: ignore
    market: Market = "us_equity",
    frequency: Frequency = "1d",
    subdir: str = "nasdaq_data",
    store_name: str = "stock.zarr",
    vendor: Vendor = "tiingo",
):
    """Build the ``DatasetConfig`` for daily US-equity bars.

    Raw parquet is read from
    ``downloads/{market}/{frequency}/{subdir}/{vendor}`` and the panel is
    stored at ``data/{market}/{frequency}/{store_name}``. ``subdir`` and
    ``store_name`` let a second roster (for example a full-market backfill)
    live beside the NASDAQ-only one instead of overwriting it. The raw path
    ends at the vendor segment on purpose: ``StockDataset`` checks that the
    directory basename equals ``vendor`` before scanning, which rules out an
    accidental scan one level up that would merge two vendors' shards.

    Parameters
    ----------
    start_date : str | None, default None
        First date to load, ISO format; ``None`` means unbounded.
    end_date : str | None, default None
        Last date to load, inclusive; ``None`` means unbounded.
    symbols : tuple[str, ...] | None, default None
        Symbols to keep; ``None`` leaves the selection to the
        dataset.
    kwargs : dict | None, default None
        Extra dataset options.
    market : Market, default "us_equity"
        Market label used in the storage paths.
    frequency : Frequency, default "1d"
        Bar frequency used in the storage paths.
    subdir : str, default "nasdaq_data"
        Raw-data subdirectory beneath the market/frequency root.
    store_name : str, default "stock.zarr"
        Zarr store filename.
    vendor : Vendor, default "tiingo"
        Vendor whose shards the raw directory holds, also recorded on
        the config. Defaults to ``"tiingo"``, which is what existing
        callers have on disk.

    Examples
    --------
    With the storage root set to ``/mnt/quant``:

    >>> cfg = stock_kline_config(start_date="2020-01-01", symbols=("AAPL",))
    >>> cfg.raw_data_dir_path
    '/mnt/quant/downloads/us_equity/1d/nasdaq_data/tiingo'
    >>> cfg.zarr_file_path
    '/mnt/quant/data/us_equity/1d/stock.zarr'
    >>> stock_kline_config(subdir="us_all", store_name="us_all.zarr").zarr_file_path
    '/mnt/quant/data/us_equity/1d/us_all.zarr'
    """
    return DatasetConfig(
        raw_data_dir_path=str(
            _market_downloads_root(market, frequency) / subdir / vendor
        ),
        zarr_file_path=str(_market_data_root(market, frequency) / store_name),
        market=market,
        frequency=frequency,
        vendor=vendor,
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        kwargs=kwargs,
    )


def stock_acquisition_config(
    symbols: tuple[str, ...],
    start_date: str | None = None,
    end_date: str | None = None,
    kwargs: dict = None,  # type: ignore
    market: Market = "us_equity",
    frequency: Frequency = "1d",
    subdir: str = "nasdaq_data",
    vendor: Vendor = "tiingo",
):
    """Build the ``AcquisitionConfig`` for a daily US-equity download.

    Raw shards are written beneath ``{subdir}/{vendor}`` and watermarks
    beneath ``{subdir}/_watermarks/{vendor}``, both under
    ``downloads/{market}/{frequency}/``. The watermark directory is a sibling
    of the raw root rather than inside it because a polars directory scan
    reads every file beneath the root it is given, and a ``.json`` sidecar in
    the raw tree would break ``scan_parquet``. A separate ``subdir`` per
    roster keeps two backfills independently resumable.

    Parameters
    ----------
    symbols : tuple[str, ...]
        Symbols to download.
    start_date : str | None, default None
        First date to request; ``None`` leaves it to the vendor
        client.
    end_date : str | None, default None
        Last date to request; ``None`` leaves it to the vendor
        client.
    kwargs : dict | None, default None
        Extra acquisition options (for example ``max_workers``).
    market : Market, default "us_equity"
        Market label used in the storage paths.
    frequency : Frequency, default "1d"
        Bar frequency used in the storage paths.
    subdir : str, default "nasdaq_data"
        Raw-data subdirectory beneath the market/frequency root.
    vendor : Vendor, default "tiingo"
        Vendor to fetch from. Defaults to ``"tiingo"``, which is what
        existing callers have on disk.

    Examples
    --------
    With the storage root set to ``/mnt/quant``:

    >>> cfg = stock_acquisition_config(
    ...     symbols=("AAPL", "MSFT"), start_date="2020-01-01"
    ... )
    >>> cfg.raw_data_dir_path
    '/mnt/quant/downloads/us_equity/1d/nasdaq_data/tiingo'
    >>> cfg.watermark_path
    '/mnt/quant/downloads/us_equity/1d/nasdaq_data/_watermarks/tiingo'
    """
    downloads = _market_downloads_root(market, frequency) / subdir
    return AcquisitionConfig(
        market=market,
        frequency=frequency,
        vendor=vendor,
        raw_data_dir_path=str(downloads / vendor),
        watermark_path=str(downloads / "_watermarks" / vendor),
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        kwargs=kwargs,
    )


def universe_config(kwargs: dict = None) -> UniverseConfig:  # type: ignore
    """Build the ``UniverseConfig`` for the US-equity universe reference table.

    The table is reference metadata rather than pipeline data, so it lives as
    parquet under ``data/reference/`` instead of a
    ``data/{market}/{frequency}/`` Zarr store, with a ``_cache`` directory
    beside it for fetcher snapshots.

    Examples
    --------
    With the storage root set to ``/mnt/quant``:

    >>> cfg = universe_config()
    >>> cfg.output_path
    '/mnt/quant/data/reference/universe.parquet'
    >>> cfg.cache_dir
    '/mnt/quant/data/reference/_cache'
    """
    return UniverseConfig(
        output_path=str(
            get_data_root() / "data" / "reference" / "universe.parquet"
        ),
        cache_dir=str(get_data_root() / "data" / "reference" / "_cache"),
        kwargs=kwargs,
    )


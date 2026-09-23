"""Config factories for the datasets, factors and labels the pipeline ships with.

Each factory builds one config dataclass from ``quantlab.base.config`` with
every storage path derived from a single data root. ``get_data_root`` resolves
that root from, in order, the process-level override that ``--data-dir`` sets
through ``set_data_root``, the ``QUANTLAB_DATA_DIR`` environment variable, and
finally a ``data/`` directory beside the repository, so a fresh clone works
with no configuration. Beneath the root, raw downloads live under
``downloads/{market}/{frequency}/``, Zarr stores under
``data/{market}/{frequency}/`` and reference tables under ``data/reference/``.

The factories snapshot their paths as plain strings, so a root override must
be applied before the first factory call.

Example:
    >>> from quantlab.config import set_data_root, stock_kline_config
    >>> set_data_root("/mnt/quant")
    >>> cfg = stock_kline_config(start_date="2020-01-01", symbols=("AAPL",))
    >>> cfg.zarr_file_path
    '/mnt/quant/data/us_equity/1d/stock.zarr'
"""

import os
from pathlib import Path
from typing import Literal

from quantlab.base.config import (
    AcquisitionConfig,
    ConstituentDatasetConfig,
    DatasetConfig,
    FactorConfig,
    PolarsFactorConfig,
    UniverseConfig,
)
from quantlab.backend import PlBackend, XrBackend
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.dataset.stock import StockDataset
from quantlab.enums.data import Frequency, Market, Vendor

#: Process-level storage-root override, set by ``set_data_root`` and consulted
#: first by ``get_data_root``. ``None`` means not overridden.
_DATA_ROOT_OVERRIDE: Path | None = None


def set_data_root(path: "str | os.PathLike | None") -> Path | None:
    """Set the process-level storage root, ahead of the environment variable.

    The value is passed through ``expanduser`` but not ``resolve``: the
    ``QUANTLAB_DATA_DIR`` value is used unresolved too, so the two behave the
    same on a symlinked root, and a quoted ``--data-dir '~/x'`` reaches Python
    with a literal tilde that would otherwise become a directory named ``~``.
    The directory is neither created nor required to exist.

    Args:
        path: The new root. ``None`` clears the override so that
            ``QUANTLAB_DATA_DIR`` or the repository default applies again.

    Returns:
        The stored ``Path``, or ``None`` when the override was cleared.

    Raises:
        ValueError: If ``path`` is an empty or whitespace-only string. An
            empty environment variable falls through to the default, but a
            root someone typed explicitly should not silently mean "default".
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

    Resolution order: the override set by ``set_data_root`` (what
    ``--data-dir`` drives), then the ``QUANTLAB_DATA_DIR`` environment
    variable, then the ``data/`` directory beside the repository root.
    """
    if _DATA_ROOT_OVERRIDE is not None:
        return _DATA_ROOT_OVERRIDE
    env_value = os.environ.get("QUANTLAB_DATA_DIR")
    if env_value:
        return Path(env_value)
    # Three hops: config/__init__.py -> config -> quantlab -> repository root.
    return Path(__file__).resolve().parent.parent.parent / "data"


def _market_data_root(market: str, frequency: str) -> Path:
    """Return the ``data/{market}/{frequency}`` directory Zarr stores live in."""
    return get_data_root() / "data" / market / frequency


def _market_downloads_root(market: str, frequency: str) -> Path:
    """Return the ``downloads/{market}/{frequency}`` directory for raw data."""
    return get_data_root() / "downloads" / market / frequency


def spot_kline_config(
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: list | None = None,
    kwargs: dict = None,  # type: ignore
    market: Market = "crypto_spot",
    frequency: Frequency = "1d",
):
    """Build the ``DatasetConfig`` for Binance spot klines.

    Raw CSVs are read from
    ``downloads/{market}/{frequency}/spot/monthly/klines`` and the panel is
    stored at ``data/{market}/{frequency}/klines.zarr``.

    Args:
        start_date: First date to load, ISO format; ``None`` means unbounded.
        end_date: Last date to load, inclusive; ``None`` means unbounded.
        symbols: Symbols to keep; ``None`` leaves the selection to the
            dataset.
        kwargs: Extra dataset options.
        market: Market label used in the storage paths.
        frequency: Bar frequency used in the storage paths.
    """
    return DatasetConfig(
        raw_data_dir_path=str(
            _market_downloads_root(market, frequency)
            / "spot"
            / "monthly"
            / "klines"
        ),
        zarr_file_path=str(
            _market_data_root(market, frequency) / "klines.zarr"
        ),
        catalog_path=str(get_data_root() / "data" / "catalog"),
        market=market,
        frequency=frequency,
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        kwargs=kwargs,
    )


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

    Args:
        start_date: First date to load, ISO format; ``None`` means unbounded.
        end_date: Last date to load, inclusive; ``None`` means unbounded.
        symbols: Symbols to keep; ``None`` leaves the selection to the
            dataset.
        kwargs: Extra dataset options.
        market: Market label used in the storage paths.
        frequency: Bar frequency used in the storage paths.
        subdir: Raw-data subdirectory beneath the market/frequency root.
        store_name: Zarr store filename.
        vendor: Vendor whose shards the raw directory holds, also recorded on
            the config. Defaults to ``"tiingo"``, which is what existing
            callers have on disk.
    """
    return DatasetConfig(
        raw_data_dir_path=str(
            _market_downloads_root(market, frequency) / subdir / vendor
        ),
        zarr_file_path=str(_market_data_root(market, frequency) / store_name),
        catalog_path=str(get_data_root() / "data" / "catalog"),
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

    Args:
        symbols: Symbols to download.
        start_date: First date to request; ``None`` leaves it to the vendor
            client.
        end_date: Last date to request; ``None`` leaves it to the vendor
            client.
        kwargs: Extra acquisition options (for example ``max_workers``).
        market: Market label used in the storage paths.
        frequency: Bar frequency used in the storage paths.
        subdir: Raw-data subdirectory beneath the market/frequency root.
        vendor: Vendor to fetch from. Defaults to ``"tiingo"``, which is what
            existing callers have on disk.
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
    """
    return UniverseConfig(
        output_path=str(
            get_data_root() / "data" / "reference" / "universe.parquet"
        ),
        cache_dir=str(get_data_root() / "data" / "reference" / "_cache"),
        kwargs=kwargs,
    )


def sp500_constituent_config(
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: list[str] | tuple[str, ...] | None = None,
    as_of: str | None = None,
    kwargs: dict = None,  # type: ignore
) -> ConstituentDatasetConfig:
    """Build the ``ConstituentDatasetConfig`` for daily S&P 500 membership.

    The boolean ``is_member`` panel is pipeline data consumed by the factor
    and model layers as a per-day universe mask, so it is stored as Zarr at
    ``data/us_equity/1d/sp500_constituent.zarr``. The fetcher's cached source
    snapshot shares ``universe_config``'s ``data/reference/_cache`` directory
    so there is only one copy of it.

    Args:
        start_date: First date of the panel; ``None`` means unbounded.
        end_date: Last date of the panel, inclusive; ``None`` means unbounded.
        symbols: Symbols to keep, converted to a tuple; ``None`` keeps all.
        as_of: Optional date to resolve membership as of.
        kwargs: Extra dataset options.
    """
    return ConstituentDatasetConfig(
        zarr_file_path=str(
            _market_data_root("us_equity", "1d") / "sp500_constituent.zarr"
        ),
        cache_dir=str(get_data_root() / "data" / "reference" / "_cache"),
        start_date=start_date,
        end_date=end_date,
        # The config declares `tuple | None`; normalise at the boundary.
        symbols=tuple(symbols) if symbols is not None else None,
        as_of=as_of,
        kwargs=kwargs,
    )


def nasdaq100_constituent_config(
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: list[str] | tuple[str, ...] | None = None,
    as_of: str | None = None,
    kwargs: dict = None,  # type: ignore
) -> ConstituentDatasetConfig:
    """Build the ``ConstituentDatasetConfig`` for daily Nasdaq-100 membership.

    Same shape as ``sp500_constituent_config`` but with its own Zarr store,
    ``data/us_equity/1d/nasdaq100_constituent.zarr``. The two indices have
    different coverage starts (1976 for the S&P 500, 2007 for the Nasdaq-100),
    and sharing one timestamp axis would fabricate a region that a boolean
    panel cannot distinguish from "nobody was a member". A consumer that wants
    both opens both and joins on the intersection of their timestamp axes.
    The cache directory is shared; each fetcher writes its own file inside it.

    Args:
        start_date: First date of the panel; ``None`` means unbounded.
        end_date: Last date of the panel, inclusive; ``None`` means unbounded.
        symbols: Symbols to keep, converted to a tuple; ``None`` keeps all.
        as_of: Optional date to resolve membership as of.
        kwargs: Extra dataset options.
    """
    return ConstituentDatasetConfig(
        zarr_file_path=str(
            _market_data_root("us_equity", "1d") / "nasdaq100_constituent.zarr"
        ),
        cache_dir=str(get_data_root() / "data" / "reference" / "_cache"),
        start_date=start_date,
        end_date=end_date,
        # The config declares `tuple | None`; normalise at the boundary.
        symbols=tuple(symbols) if symbols is not None else None,
        as_of=as_of,
        kwargs=kwargs,
    )


def alpha101_config(
    start_date: str | None = None,
    end_date: str | None = None,
    window: int = 128,
    factor_names: list | None = None,
    symbols: list | None = None,
    mode: Literal["batch", "stream"] = "batch",
):
    """Build the ``FactorConfig`` for Alpha101 factors on Binance spot klines.

    Factor values are stored at ``data/factor/alpha101.zarr``; the dataset is
    a ``SpotKlineDataset`` built from ``spot_kline_config``.

    Args:
        start_date: First date of the factor window.
        end_date: Last date of the factor window.
        window: Lookback, in bars, that the dataset window is extended by.
        factor_names: Factors to compute; ``None`` means all.
        symbols: Symbols to compute; ``None`` means all.
        mode: ``"batch"`` for a full historical run, ``"stream"`` for
            incremental per-bar updates.
    """
    return FactorConfig(
        file_path=str(get_data_root() / "data" / "factor" / "alpha101.zarr"),
        dataset=SpotKlineDataset(spot_kline_config(symbols=symbols)),
        data_columns=[
            "high",
            "low",
            "close",
            "open",
            "volume",
            "amount",
        ],
        symbols=symbols,
        factor_names=factor_names,
        mode=mode,
        window=window,
        start_date=start_date,
        end_date=end_date,
    )


def stock_alpha101_config(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    window: int = 128,
    factor_names: list | None = None,
    symbols: list | None = None,
    mode: Literal["batch", "stream"] = "batch",
    market: Market = "us_equity",
    frequency: Frequency = "1d",
):
    """Build the ``FactorConfig`` for Alpha101 factors on US-equity bars.

    Same shape as ``alpha101_config`` but wired to a ``StockDataset`` and
    stored at ``data/factor/alpha101_stock.zarr``. ``"amount"`` must stay in
    ``data_columns``: the factor graph derives ``vwap`` from it, and its
    presence is what makes ``StockDataset`` synthesise ``volume * close`` for
    a vendor with no turnover column. Without it the graph fails at
    construction with ``RuntimeError: Bad inputs``.

    Args:
        start_date: First date of the factor window.
        end_date: Last date of the factor window.
        window: Lookback, in bars, that the dataset window is extended by.
        factor_names: Factors to compute; ``None`` means all.
        symbols: Symbols to compute; ``None`` means all.
        mode: ``"batch"`` for a full historical run, ``"stream"`` for
            incremental per-bar updates.
        market: Market label used in the storage paths.
        frequency: Bar frequency used in the storage paths.
    """
    return FactorConfig(
        file_path=str(
            get_data_root() / "data" / "factor" / "alpha101_stock.zarr"
        ),
        dataset=StockDataset(
            stock_kline_config(
                symbols=symbols, market=market, frequency=frequency
            )
        ),
        data_columns=[
            "high",
            "low",
            "close",
            "open",
            "volume",
            "amount",
        ],
        symbols=symbols,
        factor_names=factor_names,
        mode=mode,
        window=window,
        start_date=start_date,
        end_date=end_date,
    )


def alpha158_config(
    start_date: str | None = None,
    end_date: str | None = None,
    factor_names: list | None = None,
    symbols: list | None = None,
    mode: Literal["batch", "stream"] = "batch",
):
    """Build the ``FactorConfig`` for Alpha158 factors on Binance spot klines.

    Factor values are stored at ``data/factor/alpha158.zarr``; the dataset is
    a ``SpotKlineDataset`` built from ``spot_kline_config``. The lookback
    window is fixed at 128 bars.

    Args:
        start_date: First date of the factor window.
        end_date: Last date of the factor window.
        factor_names: Factors to compute; ``None`` means all.
        symbols: Symbols to compute; ``None`` means all.
        mode: ``"batch"`` for a full historical run, ``"stream"`` for
            incremental per-bar updates.
    """
    return FactorConfig(
        file_path=str(get_data_root() / "data" / "factor" / "alpha158.zarr"),
        dataset=SpotKlineDataset(spot_kline_config(symbols=symbols)),
        data_columns=[
            "high",
            "low",
            "close",
            "open",
            "volume",
            "amount",
        ],
        symbols=symbols,
        mode=mode,
        factor_names=factor_names,
        window=128,
        start_date=start_date,
        end_date=end_date,
    )


def stock_alpha158_config(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    factor_names: list | None = None,
    symbols: list | None = None,
    mode: Literal["batch", "stream"] = "batch",
    market: Market = "us_equity",
    frequency: Frequency = "1d",
):
    """Build the ``FactorConfig`` for Alpha158 factors on US-equity bars.

    Same shape as ``alpha158_config`` but wired to a ``StockDataset`` and
    stored at ``data/factor/alpha158_stock.zarr``. ``"amount"`` must stay in
    ``data_columns``: the factor graph derives ``vwap`` from it, and its
    presence is what makes ``StockDataset`` synthesise ``volume * close`` for
    a vendor with no turnover column. The ``Alpha158Stock`` factor this
    configures emits raw, un-normalised values; normalisation is the factor
    class's concern, and this factory only chooses the dataset.

    Args:
        start_date: First date of the factor window.
        end_date: Last date of the factor window.
        factor_names: Factors to compute; ``None`` means all.
        symbols: Symbols to compute; ``None`` means all.
        mode: ``"batch"`` for a full historical run, ``"stream"`` for
            incremental per-bar updates.
        market: Market label used in the storage paths.
        frequency: Bar frequency used in the storage paths.
    """
    return FactorConfig(
        file_path=str(
            get_data_root() / "data" / "factor" / "alpha158_stock.zarr"
        ),
        dataset=StockDataset(
            stock_kline_config(
                symbols=symbols, market=market, frequency=frequency
            )
        ),
        data_columns=[
            "high",
            "low",
            "close",
            "open",
            "volume",
            "amount",
        ],
        symbols=symbols,
        mode=mode,
        factor_names=factor_names,
        window=128,
        start_date=start_date,
        end_date=end_date,
    )


def momentum_config(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: list | None = None,
    n: int = 20,
    market: Market = "crypto_spot",
    frequency: Frequency = "1d",
):
    """Build the ``PolarsFactorConfig`` for the Polars ``Momentum`` factor.

    ``window`` is set to ``n`` so the dataset lookback is extended by exactly
    the momentum horizon, and ``kwargs={"n": n}`` lets the factor read its
    horizon from config rather than from a literal. This factory imports
    nothing from the factor module; it only builds the config.

    Args:
        start_date: First date of the factor window.
        end_date: Last date of the factor window.
        symbols: Symbols to compute; ``None`` means all.
        n: Momentum horizon, in bars.
        market: Market label used in the storage paths.
        frequency: Bar frequency used in the storage paths.
    """
    return PolarsFactorConfig(
        file_path=str(get_data_root() / "data" / "factor" / "momentum.zarr"),
        dataset=SpotKlineDataset(
            spot_kline_config(
                symbols=symbols, market=market, frequency=frequency
            )
        ),
        symbols=symbols,
        window=n,
        start_date=start_date,
        end_date=end_date,
        kwargs={"n": n},
    )


def spot_label_config(
    label_name: str,
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: list | None = None,
    mode: Literal["batch", "stream"] = "batch",
    n_forward_periods: int = 1,
):
    """Build the ``FactorConfig`` for a forward-return label on spot klines.

    Labels are stored at ``data/label/spot_label_{label_name}.zarr`` and are
    computed from ``close`` only. ``symbols`` defaults to ``["_all_"]`` and
    ``factor_names`` is always ``["_all_"]``.

    Args:
        label_name: Name of the label, used in the store filename.
        start_date: First date of the label window.
        end_date: Last date of the label window.
        symbols: Symbols to compute; ``None`` means all.
        mode: ``"batch"`` for a full historical run, ``"stream"`` for
            incremental per-bar updates.
        n_forward_periods: Horizon of the forward return, in bars.
    """
    if symbols is None:
        symbols = ["_all_"]
    return FactorConfig(
        file_path=str(
            get_data_root() / "data" / "label" / f"spot_label_{label_name}.zarr"
        ),
        dataset=SpotKlineDataset(spot_kline_config(symbols=symbols)),
        data_columns=["close"],
        symbols=symbols,
        mode=mode,
        factor_names=["_all_"],
        window=128,
        start_date=start_date,
        end_date=end_date,
        kwargs={"n_forward_periods": n_forward_periods},
    )

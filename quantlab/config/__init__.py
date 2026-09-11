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
from quantlab.dataset.backend import PlBackend, XrBackend
from quantlab.dataset.spot import SpotKlineDataset
from quantlab.dataset.stock import StockDataset
from quantlab.enums.data import Frequency, Market, Vendor

#: Process-level storage-root override, set by `set_data_root()` and consulted
#: first by `get_data_root()`. `None` means "not overridden"; see D-06.
_DATA_ROOT_OVERRIDE: Path | None = None


def set_data_root(path: "str | os.PathLike | None") -> Path | None:
    """Set the process-level storage root, ahead of the environment variable.

    Precedence, highest first (260907-rjq D-01):

    1. this override,
    2. the `QUANTLAB_DATA_DIR` environment variable,
    3. the repo-root-relative `data/` directory.

    There is still exactly ONE data root; it simply has one more way to be set,
    and no volume is ever hardcoded.

    The value is `expanduser()`-ed but deliberately NOT `resolve()`-d. The
    environment knob is `Path(env_value)` with no resolution, so resolving only
    this one would make the two behave differently on a symlinked root -- on
    macOS `/tmp` resolves to `/private/tmp` -- and would defeat a caller who
    passed a symlink on purpose. Expansion IS applied because a quoted
    `--data-dir '~/quantlab-data'` reaches Python with a literal tilde and
    would otherwise create a directory named `~`.

    The directory is neither created nor required to exist (D-08): the
    acquisition layer creates its own directories under whatever root it is
    given, and the environment knob validates nothing -- validating one and not
    the other is how two knobs drift apart.

    Passing `None` CLEARS the override and returns `None` (D-06); a
    process-global knob without a reset would let the first caller that sets it
    silently redirect everything afterwards. An empty or whitespace-only value
    raises `ValueError`: an unset/empty env var falls through to the default,
    but someone who typed an empty root meant something, and silently meaning
    "repo default" is wrong.

    Returns the stored `Path` so a caller can report where the run will write.
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
    """Root directory for downloads/data storage.

    Resolves the one storage root through three levels, highest first: the
    process-level override set by `set_data_root()` (what `--data-dir` drives),
    then the `QUANTLAB_DATA_DIR` environment variable, then a `data/` directory
    beside the repository root so a fresh clone works with zero configuration.
    """
    if _DATA_ROOT_OVERRIDE is not None:
        return _DATA_ROOT_OVERRIDE
    env_value = os.environ.get("QUANTLAB_DATA_DIR")
    if env_value:
        return Path(env_value)
    # Three hops: config/__init__.py -> config -> quantlab -> repository root.
    return Path(__file__).resolve().parent.parent.parent / "data"


def _market_data_root(market: str, frequency: str) -> Path:
    """`data/{market}/{frequency}` root that every zarr-backed config factory
    derives its `zarr_file_path` from (02-CONTEXT.md D-02)."""
    return get_data_root() / "data" / market / frequency


def _market_downloads_root(market: str, frequency: str) -> Path:
    """`downloads/{market}/{frequency}` root that every acquisition-facing
    config factory derives its `raw_data_dir_path` from (02-CONTEXT.md D-02)."""
    return get_data_root() / "downloads" / market / frequency


def spot_kline_config(
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: list | None = None,
    kwargs: dict = None,  # type: ignore
    market: Market = "crypto_spot",
    frequency: Frequency = "1d",
):
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
        frequency=frequency,
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        kwargs=kwargs,
    )


def stock_kline_config(
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: list | None = None,
    kwargs: dict = None,  # type: ignore
    market: Market = "us_equity",
    frequency: Frequency = "1d",
    subdir: str = "nasdaq_data",
    store_name: str = "stock.zarr",
    vendor: Vendor = "tiingo",
):
    """US-equity daily Zarr config.

    `subdir` selects the raw-data subdirectory and `store_name` the Zarr store
    filename, both BENEATH the existing `data/{market}/{frequency}/`
    convention (02-CONTEXT.md D-02). They exist so a second roster -- the
    full-market `us_all` backfill -- can land beside the NASDAQ-only one
    instead of overwriting it, WITHOUT introducing a second path root. There
    is exactly one storage root, resolved by `get_data_root()` in the order
    `--data-dir` override > `QUANTLAB_DATA_DIR` > repo-root `data/`
    (260906-0iy D-04, 260907-rjq D-01); `subdir`/`store_name` select a
    location beneath whichever of the three answered, and no volume is ever
    hardcoded here.

    `vendor` appends the D-11 vendor segment so `raw_data_dir_path`
    TERMINATES at it -- `Path(raw_data_dir_path).name == vendor` is the
    equality `StockDataset._scan_raw` asserts before scanning, which is what
    makes a scan rooted one level up (and therefore a silent two-vendor merge)
    unreachable by accident rather than merely unlikely. The vendor is also
    threaded onto `DatasetConfig.vendor`, because a path the reader cannot
    check against a recorded expectation checks nothing.

    `"tiingo"` is the documented INCUMBENT, not a guess: every existing caller
    of this factory reads raw data that `TiingoAcquisition` fetched, so the
    default names what is already on disk.
    """
    return DatasetConfig(
        raw_data_dir_path=str(
            _market_downloads_root(market, frequency) / subdir / vendor
        ),
        zarr_file_path=str(_market_data_root(market, frequency) / store_name),
        catalog_path=str(get_data_root() / "data" / "catalog"),
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
    """US-equity daily acquisition config.

    `subdir` selects the raw-data subdirectory (and, beside it, the watermark
    directory) so a second roster's raw parquet and watermarks stay separate
    from the NASDAQ-only ones -- separate watermarks are what make the two
    backfills independently resumable. It is a subdirectory BENEATH
    `downloads/{market}/{frequency}/`, not a second root: there is exactly one
    storage root, resolved by `get_data_root()` in the order `--data-dir`
    override > `QUANTLAB_DATA_DIR` > repo-root `data/` (260906-0iy D-04,
    260907-rjq D-01), and this factory hardcodes no volume.

    `vendor` is DERIVED into both paths rather than accepted pre-built, so the
    two placements below cannot drift apart at a call site (D-11, D-19):

    - `raw_data_dir_path` TERMINATES at the vendor segment
      (`.../{subdir}/{vendor}`). The basename IS the vendor, which is what
      makes `StockDataset._scan_raw`'s basename assertion expressible at all.
    - `watermark_path` is a SIBLING of that raw root
      (`.../{subdir}/_watermarks/{vendor}`), deliberately NOT inside it. A
      polars directory scan walks EVERY file beneath the root it is given, so
      a `.json` sidecar living in the raw tree breaks `pl.scan_parquet`
      outright. This is a change from the pre-03.2 `{subdir}/_watermarks/`
      placement; D-13 already discards those legacy sidecars rather than
      migrating them.

    `"tiingo"` is the documented INCUMBENT, not a guess: every existing caller
    of this factory fetched from Tiingo, so the default names the vendor whose
    data the pre-03.2 paths actually hold.
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
    """Config for the US-equity universe reference table (02-CONTEXT.md D-12).

    Deliberately lives under `data/reference/`, separate from the
    `data/{market}/{frequency}/` convention used by spot_kline_config()/
    stock_kline_config() -- this reflects Locked Decision A1 (02-08-PLAN.md):
    the universe table is reference/metadata (same footing as
    config/instruments.yaml), not xarray/Zarr pipeline data, hence
    PlBackend/parquet, not XrBackend/Zarr.
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
    """Config for the daily point-in-time S&P 500 membership panel (DATA-05,
    03.1-CONTEXT.md D-04).

    The two paths deliberately live in DIFFERENT roots, and the split is not
    an oversight:

    - `zarr_file_path` takes the `data/{market}/{frequency}/` branch, same as
      `stock_kline_config()`, because the daily `is_member` panel is PIPELINE
      data -- it exists to be consumed by the factor and model layers as a
      per-day universe mask, so CLAUDE.md's xarray/Zarr constraint governs it
      (D-04). This is the deliberate, scoped divergence from Locked Decision
      A1 of 02-08-PLAN.md.
    - `cache_dir` shares `universe_config()`'s `data/reference/_cache`
      directory because the fetcher's cached source snapshot is the very same
      reference/metadata artefact that factory already owns; giving the panel
      a second, private cache directory would mean two copies of one snapshot
      drifting apart.
    """
    return ConstituentDatasetConfig(
        zarr_file_path=str(
            _market_data_root("us_equity", "1d") / "sp500_constituent.zarr"
        ),
        cache_dir=str(get_data_root() / "data" / "reference" / "_cache"),
        start_date=start_date,
        end_date=end_date,
        # Converted here rather than passed through: `BaseDatasetConfig`
        # declares `tuple | None`, and `IndexConstituentDataset` overrides the
        # `_reset_symbols()` seam that used to do the normalising. Making the
        # conversion explicit at the boundary is what removes the
        # `# type: ignore[arg-type]` that was papering over the mismatch.
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
    """Config for the daily point-in-time Nasdaq-100 membership panel
    (DATA-05, 03.1-CONTEXT.md D-02/D-04).

    Identical in shape to `sp500_constituent_config()` apart from the store
    filename, and the difference is deliberate: **the Nasdaq-100 panel gets
    its OWN Zarr store rather than sharing the S&P 500 panel's.** The two
    indices have different point-in-time coverage starts (1976-07-01 versus
    2007-02-01), so unioning them onto one timestamp axis would imply 1976
    Nasdaq-100 coverage that does not exist -- and in a boolean panel the
    fabricated region is indistinguishable at read time from a genuine
    "nobody was a member" answer. A consumer that wants both opens both and
    joins on the intersection of their timestamp axes; that is a deliberate,
    visible step rather than an implicit and wrong union.

    `cache_dir` is shared with `sp500_constituent_config()` and
    `universe_config()` on purpose -- each fetcher writes its own
    `CACHE_FILENAME` inside it, so one directory holds one snapshot per index
    with no chance of two copies of one snapshot drifting apart.
    """
    return ConstituentDatasetConfig(
        zarr_file_path=str(
            _market_data_root("us_equity", "1d") / "nasdaq100_constituent.zarr"
        ),
        cache_dir=str(get_data_root() / "data" / "reference" / "_cache"),
        start_date=start_date,
        end_date=end_date,
        # Converted here rather than passed through: `BaseDatasetConfig`
        # declares `tuple | None`, and `IndexConstituentDataset` overrides the
        # `_reset_symbols()` seam that used to do the normalising. Making the
        # conversion explicit at the boundary is what removes the
        # `# type: ignore[arg-type]` that was papering over the mismatch.
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
    """US-equity sibling of `alpha101_config()` (D-01, FACTOR-01).

    Identical in shape to the crypto-spot factory apart from the dataset it
    wires (`StockDataset`) and the zarr output name. `"amount"` MUST stay in
    `data_columns`: `Alpha101.AllData` derives `vwap` from it, and it is what
    triggers the D-02 `volume * close` synthesis in
    `StockDataset._to_kunquant()`. Drop it and the factor graph raises
    `RuntimeError: Bad inputs, given <class 'NoneType'>` at construction time.
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
    """US-equity sibling of `alpha158_config()` (D-01, FACTOR-01).

    Identical in shape to the crypto-spot factory apart from the dataset it
    wires (`StockDataset`) and the zarr output name. `"amount"` MUST stay in
    `data_columns`: `Alpha158.AllData` derives `vwap` from it, and it is what
    triggers the D-02 `volume * close` synthesis in
    `StockDataset._to_kunquant()`.

    The `Alpha158Stock` class this configures emits raw, un-normalized factor
    values (NORM-01 / D-09) -- the market split lives in the factor class, not
    here; this factory only chooses the dataset.
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
    """Config for the Polars-backend `Momentum` factor (FACTOR-03, D-08).

    Lives here purely for `config/__init__.py` file ownership: 03-03 owns this
    file for the whole of wave 3, while the `Momentum` class itself is
    delivered by the parallel plan 03-04. This factory therefore imports
    NOTHING from `factor/momentum.py` -- it only builds and returns a
    `PolarsFactorConfig`, so there is no runtime coupling between the two
    wave-3 plans.

    `window=n` so `Factor._reset_dataset_config()` extends the dataset lookback
    by exactly the momentum horizon, and `kwargs={"n": n}` so the factor reads
    its horizon from config rather than from a literal in its own source.
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

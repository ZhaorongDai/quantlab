import os
from pathlib import Path
from typing import Literal

from base.config import AcquisitionConfig, DatasetConfig, FactorConfig
from dataset.backend import PlBackend, XrBackend
from dataset.spot import SpotKlineDataset
from enums.data import Frequency, Market


def _data_root() -> Path:
    """Root directory for downloads/data storage.

    Configurable via the QUANTLAB_DATA_DIR environment variable; defaults to
    a repo-root-relative `data/` directory so a fresh clone works with zero
    configuration.
    """
    env_value = os.environ.get("QUANTLAB_DATA_DIR")
    if env_value:
        return Path(env_value)
    return Path(__file__).resolve().parent.parent / "data"


def _market_data_root(market: str, frequency: str) -> Path:
    """`data/{market}/{frequency}` root that every zarr-backed config factory
    derives its `zarr_file_path` from (02-CONTEXT.md D-02)."""
    return _data_root() / "data" / market / frequency


def _market_downloads_root(market: str, frequency: str) -> Path:
    """`downloads/{market}/{frequency}` root that every acquisition-facing
    config factory derives its `raw_data_dir_path` from (02-CONTEXT.md D-02)."""
    return _data_root() / "downloads" / market / frequency


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
            _market_downloads_root(market, frequency) / "spot" / "monthly" / "klines"
        ),
        zarr_file_path=str(_market_data_root(market, frequency) / "klines.zarr"),
        catalog_path=str(_data_root() / "data" / "catalog"),
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
    symbols: list | None = None,
    kwargs: dict = None,  # type: ignore
    market: Market = "us_equity",
    frequency: Frequency = "1d",
):
    return DatasetConfig(
        raw_data_dir_path=str(_market_downloads_root(market, frequency) / "nasdaq_data"),
        zarr_file_path=str(_market_data_root(market, frequency) / "stock.zarr"),
        catalog_path=str(_data_root() / "data" / "catalog"),
        market=market,
        frequency=frequency,
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
):
    return AcquisitionConfig(
        market=market,
        frequency=frequency,
        raw_data_dir_path=str(_market_downloads_root(market, frequency) / "nasdaq_data"),
        watermark_path=str(
            _market_downloads_root(market, frequency) / "nasdaq_data" / "_watermarks"
        ),
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
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
        file_path=str(_data_root() / "data" / "factor" / "alpha101.zarr"),
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


def alpha158_config(
    start_date: str | None = None,
    end_date: str | None = None,
    factor_names: list | None = None,
    symbols: list | None = None,
    mode: Literal["batch", "stream"] = "batch",
):
    return FactorConfig(
        file_path=str(_data_root() / "data" / "factor" / "alpha158.zarr"),
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
            _data_root() / "data" / "label" / f"spot_label_{label_name}.zarr"
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

"""Download Massive's daily files and REST tables, build the stores and keep them current.

Run this one script every day, after ``sharadar_data.py``: the Ticker map, which says
which security traded under each Massive ticker, needs Sharadar's closes of the day.
The first run backfills from ``START``; a later run downloads and converts only the days
after the last one. An interrupted run is simply run again.

    raw tier   <data root>/downloads/massive/<data type>/<year>/<day>.csv.gz   Massive's files
               <data root>/downloads/massive/rest/<endpoint>/                  REST pages
    stores     <data root>/market/massive/massive_day_bars_1d/                 Massive tickers
               <data root>/market/massive/massive_ticker_map/                  ticker -> permaticker
               <data root>/market/massive/massive_permaticker_day_bars_1d/     Sharadar's symbol axis
               <data root>/market/massive/massive_trade_bars_1m/               minute bars of TICKERS

Needs a Massive Stocks Developer plan, ``MASSIVE_API_KEY`` and ``MASSIVE_S3_ACCESS_KEY_ID``,
and the stores of ``sharadar_data.py``. The data is licensed for personal use, so the data root
(``QUANTLAB_DATA_DIR``) must be outside the repository. Every setting is a constant below; edit
them and run::

    QUANTLAB_DATA_DIR=/data/quantlab uv run python examples/data_update/massive_data.py

Raw files are kept: Massive serves a rolling ten years, so a deleted day could not be fetched
again. A day's whole-market files come to about 3 GB, so move ``START`` back with care. The exit
status is 0 when every part was updated and 1 when one failed.
"""

# %% Settings
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from quantlab.acquisition.base import DataSourceRegistry
from quantlab.acquisition.massive.client import MassiveClient
from quantlab.acquisition.registry import refresh_raw_tier
from quantlab.config import get_data_root
from quantlab.core.store_folder import load_component, save_component
from quantlab.dataset.config import (
    MappedDayBarsConfig,
    MassiveDayBarsDatasetConfig,
    MassiveTradeBarsDatasetConfig,
    PermatickerDayBarsConfig,
    TickerMapConfig,
)
from quantlab.dataset.massive.day_bars import MassiveDayBarDataset
from quantlab.dataset.massive.mapped_bars import MappedDayBarDataset
from quantlab.dataset.massive.permaticker_bars import PermatickerDayBarDataset
from quantlab.dataset.massive.raw import raw_days
from quantlab.dataset.massive.ticker_map import TickerMap
from quantlab.dataset.massive.trade_bars import MassiveTradeBarDataset
from quantlab.utils.cli import inside_repository

DATA_ROOT = get_data_root()
DOWNLOADS = DATA_ROOT / "downloads"
RAW = DOWNLOADS / "massive"
STORES = DATA_ROOT / "market" / "massive"
SEP_FOLDER = DATA_ROOT / "market" / "sharadar" / "sharadar_sep_1d"

#: The first day downloaded and converted. Each day costs about 3 GB of raw files; a split or
#: dividend before it has no Ticker map row, so it is left out and counted in a warning.
START = date(2026, 10, 1)

#: The REST tables kept in the raw tier. The two below feed the adjusted prices; every endpoint
#: is listed in ``quantlab.acquisition.massive.rest.ENDPOINTS``.
ENDPOINTS = ("splits", "dividends")

#: The tickers of the minute bars; ``None`` converts the whole market, tens of GB of memory a day.
TICKERS = ("AAPL", "MSFT", "SPY")


def folder(name: str) -> Path:
    """The store folder ``market/massive/<name>/``."""
    return STORES / name


def new_york_today() -> date:
    """Today in New York, the last day Massive could have published."""
    return datetime.now(ZoneInfo("America/New_York")).date()


# %% Raw tier
def refresh_rest() -> tuple[str, ...]:
    """Pull the REST tables from their watermarks, and return the ones that failed."""
    result = refresh_raw_tier(DataSourceRegistry.get("massive"), DOWNLOADS, endpoints=ENDPOINTS)
    print(result.summary(), flush=True)
    return result.failed


def download_files() -> None:
    """Download the day files after each data type's watermark; a day not published yet waits."""
    client = MassiveClient()
    client.condition_table(DOWNLOADS)
    for data_type in ("day_aggs", "minute_aggs", "trades"):
        for done in client.download_days(data_type, START, new_york_today(), DOWNLOADS):
            print(f"{data_type} {done.day}: {'downloaded' if done.published else 'not published yet'}", flush=True)


# %% Stores
def update_day_bars() -> MassiveDayBarDataset:
    """Append Massive's daily bars, on the Massive ticker axis."""
    path = folder("massive_day_bars_1d")
    days = MassiveDayBarDataset(MassiveDayBarsDatasetConfig(
        zarr_file_path=str(path / "massive_day_bars_1d.zarr"),
        raw_data_dir_path=str(RAW),
        start_date=START.isoformat(),
    ))
    days.update()
    save_component(days, path)
    return days


def update_ticker_map() -> TickerMap:
    """Map every new day that Sharadar's closes have reached."""
    path = folder("massive_ticker_map")
    ticker_map = TickerMap(TickerMapConfig(
        path=str(path / "massive_ticker_map"),
        massive_dir=str(RAW),
        sharadar_dir=str(DOWNLOADS / "sharadar"),
    ))
    sep_through = load_component(SEP_FOLDER).last_bar().date()
    mapped = set(ticker_map.days())
    for day in raw_days(RAW, "day_aggs"):
        if START <= day <= sep_through and day not in mapped:
            ticker_map.append(day)
    save_component(ticker_map, path)
    return ticker_map


def update_permaticker_day_bars(days: MassiveDayBarDataset, ticker_map: TickerMap) -> MappedDayBarDataset:
    """Keep the daily bars on Sharadar's permaticker axis, with adjusted prices checked against SEP."""
    path = folder("massive_permaticker_day_bars_1d")
    mapped = MappedDayBarDataset(MappedDayBarsConfig(
        source=PermatickerDayBarDataset(PermatickerDayBarsConfig(day_bars=days, ticker_map=ticker_map)),
        sep=load_component(SEP_FOLDER),
        zarr_file_path=str(path / "massive_permaticker_day_bars_1d.zarr"),
    ))
    mapped.update()
    save_component(mapped, path)
    return mapped


def update_trade_bars() -> None:
    """Convert the trade files after the store's last day into one-minute bars, one day at a time."""
    path = folder("massive_trade_bars_1m")
    config = MassiveTradeBarsDatasetConfig(
        zarr_file_path=str(path / "massive_trade_bars_1m.zarr"),
        raw_data_dir_path=str(RAW),
        tickers=TICKERS,
        start_date=START.isoformat(),
        end_date=START.isoformat(),
    )
    last = MassiveTradeBarDataset(config).last_bar() if Path(config.zarr_file_path).exists() else None
    for day in raw_days(RAW, "trades"):
        if day >= START and (last is None or day > last.date()):
            one_day = replace(config, start_date=day.isoformat(), end_date=day.isoformat())
            MassiveTradeBarDataset(one_day).update(granularity="day")
            print(f"trade bars: {day}", flush=True)
    save_component(MassiveTradeBarDataset(config), path)


# %% Run
if __name__ == "__main__":
    if inside_repository([DATA_ROOT], Path(__file__).resolve().parents[2]):
        raise SystemExit(f"{DATA_ROOT} is inside the repository; the data is licensed.")
    failed = refresh_rest()
    download_files()
    day_bars = update_day_bars()
    tickers = update_ticker_map()
    permaticker_bars = update_permaticker_day_bars(day_bars, tickers)
    update_trade_bars()
    print(f"day bars through {day_bars.stored_through()}; {permaticker_bars.update_summary()}", flush=True)
    if failed:
        raise SystemExit(f"not updated: REST tables {list(failed)}")

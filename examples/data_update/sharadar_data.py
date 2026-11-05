"""Download Sharadar's tables, build the Zarr stores and keep them current.

Run this one script every day. The first run pulls each table in bulk and builds
every store; a later run refreshes the raw tier from its watermarks and appends the
new bars. An interrupted run is simply run again: a table already on disk is not
pulled again, and a store already built is extended.

    raw tier   <data root>/downloads/sharadar/<table>/      the vendor's tables, as parquet
    stores     <data root>/market/sharadar/<name>/<name>.zarr   panels on (timestamp, symbol)
               <data root>/universe/sharadar/<name>/<name>.zarr the S&P 500 membership

Needs a Sharadar subscription and ``SHARADAR_API_KEY``. The data is licensed for personal
use, so the data root (``QUANTLAB_DATA_DIR``) must be outside the repository. Every setting
is a constant below; edit them and run::

    QUANTLAB_DATA_DIR=/data/quantlab uv run python examples/data_update/sharadar_data.py

The exit status is 0 when every part was updated and 1 when one failed, so a scheduler
can tell. See README.md for the daily schedule.
"""

# %% Settings
from pathlib import Path

from quantlab.acquisition.base import DataSourceRegistry
from quantlab.acquisition.registry import refresh_raw_tier
from quantlab.acquisition.sharadar.client import SharadarClient
from quantlab.config import get_data_root
from quantlab.core.store_folder import load_component
from quantlab.dataset.sharadar.stores import (
    build_stores,
    missing_stores,
    store,
    update_stores,
)
from quantlab.utils.cli import inside_repository

DATA_ROOT = get_data_root()
DOWNLOADS = DATA_ROOT / "downloads"

#: The first day the stores hold; ``None`` keeps the whole history of the bulk files.
START = "2015-01-01"

#: The history tier of the bulk files: "5", "10" or "full". It must not exceed your plan.
YEARS = "full"

#: The tables of the first download, TICKERS first: the conversions map tickers through it.
TABLES = (
    "tickers", "indicators", "sep", "sfp", "actions", "sp500", "sf1", "daily", "events", "sf2", "sf3a",
)

#: Trading days asked for again before each table's watermark, for late vendor corrections.
TRAILING_DAYS = 10


# %% Raw tier
def pull_new_tables() -> list[str]:
    """Bulk-pull every table not on disk yet, and return their codes."""
    client = SharadarClient()
    new = [code for code in TABLES if not (DOWNLOADS / "sharadar" / code / f"{code}.parquet").exists()]
    for code in new:
        print(f"bulk pull: {client.bulk_table(code, DOWNLOADS, years=YEARS)}", flush=True)
    return new


def refresh() -> tuple[str, ...]:
    """Pull what Sharadar published since the last run, and return the tables that failed."""
    result = refresh_raw_tier(DataSourceRegistry.get("sharadar"), DOWNLOADS, trading_days=TRAILING_DAYS)
    print(result.summary(), flush=True)
    return result.failed


# %% Stores
def bring_stores_up():
    """Build the stores still missing, or append the new bars to all of them."""
    if missing_stores(DATA_ROOT):
        return build_stores(DOWNLOADS, DATA_ROOT, start=START)
    return update_stores(DOWNLOADS, DATA_ROOT, rebuild_dropped=True)


# %% Read one back
def show_last_bar() -> None:
    """Read the stock price store as any later pipeline step does."""
    prices = load_component(store("sharadar_sep_1d").path(DATA_ROOT).parent)
    last = prices.last_bar()
    panel = prices.panel(last, last)
    print(f"sharadar_sep_1d: last bar {last.date()}, {panel.sizes['symbol']:,} symbols", flush=True)


# %% Run
if __name__ == "__main__":
    if inside_repository([DATA_ROOT], Path(__file__).resolve().parents[2]):
        raise SystemExit(f"{DATA_ROOT} is inside the repository; the data is licensed.")
    new_tables = pull_new_tables()
    failed_tables = () if set(new_tables) == set(TABLES) else refresh()
    outcome = bring_stores_up()
    print("\n".join(outcome.lines()), flush=True)
    show_last_bar()
    if failed_tables or outcome.failed:
        raise SystemExit(f"not updated: tables {list(failed_tables)}, stores {list(outcome.failed)}")

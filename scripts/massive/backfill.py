"""Backfill Massive's whole-market one-minute Trade bars, oldest day first.

One run walks the XNYS sessions from ``--start`` to ``--end`` in date order
(ADR 0030). For each day it:

1. downloads the day's trade file into ``<download-dir>/massive/trades/``,
   while the next few days download in the background (``--files``);
2. makes sure the day's minute and day aggregate files are there
   (``minute_aggs/``, ``day_aggs/``; kept for good);
3. appends the day to the one-minute Trade bar store
   ``<data-dir>/market/massive/massive_trade_bars_1m/massive_trade_bars_1m.zarr``
   (``MassiveTradeBarDataset.update(granularity="day")``), which checks it
   against Massive's minute aggregates and records the result in
   ``<store>.massive_stats.json``;
4. deletes the day's raw trade file, only once the store holds the day and
   the check ran (``delete_converted_trades``). A day that could not be
   checked keeps its file and is listed at the end.

The days are oldest first because the plan serves a rolling ten-year window:
the oldest days are the ones about to leave it. Downloads run at most
``--files`` days ahead of the conversion, so only a few days of raw trades
sit on disk. The run stops at the first session Massive has not published
(today before it publishes, or a late file): the store only grows at its
end, so no later day is appended before it. Each line printed is one day: what was fetched, the conversion
time, how close the bars agree with Massive's, whether the raw file was
deleted, and the run's sustained MB/s, days done and an ETA.

The run pulls Massive's trade-condition table first (one snapshot per run,
under ``massive/conditions/``); the conversion reads the newest snapshot.
Tickers map to permatickers through the Sharadar raw tier
(``--sharadar-dir``, by default ``<download-dir>/sharadar``), which should be
current: a ticker it does not know is dropped and counted in the sidecar.

An interrupted run is rerun with the same arguments: the trade files already
on disk are converted first (or deleted, if the store holds them already),
then the downloads go on after each data type's watermark. A rerun with
nothing new to do downloads and converts nothing.

``MASSIVE_API_KEY`` and ``MASSIVE_S3_ACCESS_KEY_ID`` must be set in the
environment (``MASSIVE_S3_SECRET_ACCESS_KEY`` defaults to the API key). The
data is licensed for personal use: both directories must be outside this
repository, and the script refuses one inside it.

Usage::

    uv run python scripts/massive/backfill.py \\
        --download-dir /data/quantlab/downloads --data-dir /data/quantlab
    uv run python scripts/massive/backfill.py --start 2016-10-11 --end 2016-10-14 \\
        --download-dir ~/quantlab_data/downloads --data-dir ~/quantlab_data

``--start`` defaults to the oldest day the plan serves today, ``--end`` to
today in New York (the run stops at the first day not published yet).
"""

import argparse
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import exchange_calendars as xcals

from quantlab.acquisition.massive.client import (
    DayDownload,
    MassiveClient,
    MassiveCredentialError,
    MassiveEntitlementError,
    MassiveNotPublishedError,
    MassiveTransportError,
)
from quantlab.dataset.config import MassiveTradeBarsDatasetConfig
from quantlab.dataset.massive.raw import (
    VENDOR_DIR,
    raw_days,
    raw_file,
    read_watermark,
    s3_key,
)
from quantlab.dataset.massive.trade_bars import (
    MassiveTradeBarDataset,
    trade_bar_store_name,
)
from quantlab.dataset.massive.vendor_check import CHECKED
from quantlab.utils.cli import (
    add_output_dir_args,
    inside_repository,
    resolve_output_dirs,
)

#: The repository this script lives in; no output may go inside it.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The folder of Massive's stores under the data root.
MARKET_FOLDER = Path("market") / "massive"

#: The bar interval of the backfill store.
BAR_INTERVAL = "1m"

#: How far the plan reaches back: on 2026-10-09 its oldest day was 2016-10-11,
#: and the window rolls forward one day per day.
PLAN_WINDOW = date(2026, 10, 9) - date(2016, 10, 11)

#: Sessions tried after the computed oldest day when Massive refuses it.
OLDEST_DAY_PROBES = 5

#: Aggregate files are about 22 MB, one range each: two streams, two files.
AGGREGATE_STREAMS = 2
AGGREGATE_FILES = 2

AGGREGATE_TYPES = ("minute_aggs", "day_aggs")

#: Days the ETA's pace is averaged over.
RECENT_DAYS = 20


def new_york_today() -> date:
    """Today's date in New York."""
    return datetime.now(ZoneInfo("America/New_York")).date()


def sessions(start: date, end: date) -> list[date]:
    """The XNYS sessions from ``start`` through ``end``."""
    if end < start:
        return []
    return [s.date() for s in xcals.get_calendar("XNYS").sessions_in_range(start.isoformat(), end.isoformat())]


def oldest_served_day(client: MassiveClient, today: date) -> date:
    """The first session the plan serves, from its rolling window and a probe of the vendor."""
    candidates = sessions(today - PLAN_WINDOW, today)[:OLDEST_DAY_PROBES]
    for day in candidates:
        try:
            client.store.size(s3_key("trades", day))
            return day
        except MassiveEntitlementError:
            continue
        except MassiveNotPublishedError:
            return day
    raise SystemExit(f"Massive refused every day from {candidates[0]} to {candidates[-1]}; pass --start.")


def write_readme(store: Path) -> None:
    """Create the store's folder and write its ``README.md`` unless one is there."""
    readme = store.parent / "README.md"
    if readme.exists():
        return
    store.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        f"# {store.parent.name}\n\n"
        "One-minute Trade bars of the whole US market on the Sharadar permaticker axis, built by "
        "quantlab from every SIP trade in Massive's daily trade files (ADR 0030; docs/massive.md).\n\n"
        "Written day by day, oldest first, by `scripts/massive/backfill.py`; per-day statistics and "
        "the check against Massive's minute aggregates are in the `.massive_stats.json` sidecar. "
        "The raw trade file of each converted, checked day is deleted. Licensed for personal use.\n"
    )


class AggregatesAhead:
    """Massive's aggregate files of one data type, downloaded in step with the trade days."""

    def __init__(self, client: MassiveClient, data_type: str, start: date, end: date, download_dir: Path):
        self._run = client.download_days(data_type, start, end, download_dir)
        self._through: date | None = None
        self.fetched_bytes = 0

    def download_through(self, day: date) -> None:
        """Download the files up to ``day``, unless they are there already."""
        while self._through is None or self._through < day:
            done = next(self._run, None)
            if done is None:
                self._through = date.max
                return
            self._through = done.day
            self.fetched_bytes += done.fetched_bytes

    def close(self) -> None:
        """Stop the downloads in flight."""
        self._run.close()


def trade_days(
    client: MassiveClient, start: date, end: date, through: date | None, download_dir: Path
) -> Iterator[DayDownload]:
    """The trade files on disk first (a run that stopped before converting or deleting them), then the downloads.

    On disk: the files of days the store holds (to be deleted), then the
    unbroken run of sessions after ``through``, the store's last day, whose
    files are there. Downloads start at the first session of that run
    without a file, so a later file a stopped run left on disk is never
    appended before an earlier day still missing; it is kept and counted
    as there by the download.
    """
    vendor_root = download_dir / VENDOR_DIR
    first = first_day_to_download(start, through)
    for day in raw_days(vendor_root, "trades"):
        if start <= day < first:
            yield DayDownload(day, raw_file(vendor_root, "trades", day))
    for day in sessions(first, end):
        path = raw_file(vendor_root, "trades", day)
        if not path.exists():
            break
        yield DayDownload(day, path)
        first = day + timedelta(days=1)
    yield from client.download_days("trades", first, end, download_dir)


def first_day_to_download(start: date, through: date | None) -> date:
    """The first day to download: ``start``, or the day after the store's last day if later."""
    return start if through is None else max(start, through + timedelta(days=1))


def duration(seconds: float) -> str:
    """``3d 4h``, ``5h 12m`` or ``42m``."""
    minutes = int(seconds // 60)
    days, hours, minutes = minutes // 1440, minutes // 60 % 24, minutes % 60
    if days:
        return f"{days}d {hours}h"
    return f"{hours}h {minutes}m" if hours else f"{minutes}m"


def agreement(stats: dict) -> str:
    """How the day's bars agree with Massive's, or why they were not compared."""
    check = stats.get("vendor_check", {})
    if check.get("status") != CHECKED:
        return f"not checked ({check.get('status')})"
    both = max(check["both"], 1)
    return (
        f"close {check['agree']['close'] / both:.4%}, volume {check['agree']['volume'] / both:.4%} "
        f"of {check['both']:,} bars ({check['ours_only']} only ours, {check['vendor_only']} only theirs)"
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build this script's argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Backfill Massive's whole-market one-minute Trade bars, oldest day first: download each "
            "day's trades, append them to the store, check them against Massive's minute bars and "
            "delete the raw trades. Requires MASSIVE_API_KEY and MASSIVE_S3_ACCESS_KEY_ID. Both "
            "directories must be outside this repository (the data is licensed)."
        )
    )
    parser.add_argument(
        "--start", type=date.fromisoformat, default=None,
        help="First day, YYYY-MM-DD. Default: the oldest day the plan serves today.",
    )
    parser.add_argument(
        "--end", type=date.fromisoformat, default=None,
        help="Last day, YYYY-MM-DD. Default: today in New York; the run stops at the first day not published yet.",
    )
    parser.add_argument(
        "--streams", type=int, default=16,
        help="Byte ranges of one trade file fetched at once. Default: 16, the measured knee on the server's route.",
    )
    parser.add_argument(
        "--files", type=int, default=4,
        help=(
            "Trade files in flight at once (one fetched, the others verified; one core each), "
            "and the most days of raw trades on disk. Default: 4."
        ),
    )
    parser.add_argument(
        "--sharadar-dir", default=None,
        help="The Sharadar raw tier that maps tickers to permatickers. Default: <download-dir>/sharadar.",
    )
    add_output_dir_args(
        parser,
        download_help=(
            "Directory for the raw downloads: Massive's files go to <download-dir>/massive/<data type>/, "
            "with a watermark per data type and the condition table beside them. Default: the current "
            "directory, which must be outside this repository."
        ),
        data_help=(
            "The data root. The store goes to <data-dir>/market/massive/massive_trade_bars_1m/"
            "massive_trade_bars_1m.zarr, beside a README.md. Default: the current directory, which "
            "must be outside this repository."
        ),
    )
    return parser


def main() -> int:
    """Run the backfill; return the exit status."""
    parser = _build_arg_parser()
    args = parser.parse_args()
    download_dir, data_dir = resolve_output_dirs(args)
    if offending := inside_repository([download_dir, data_dir], REPO_ROOT):
        parser.exit(
            1,
            f"refusing to write Massive data inside the repository ({REPO_ROOT}): "
            f"{[str(p) for p in offending]}. Pass --download-dir and --data-dir outside it.\n",
        )
    sharadar_dir = Path(args.sharadar_dir).expanduser().absolute() if args.sharadar_dir else download_dir / "sharadar"
    if not (sharadar_dir / "tickers").is_dir():
        parser.exit(1, f"no Sharadar TICKERS under {sharadar_dir}; pull the Sharadar raw tier first.\n")

    try:
        client = MassiveClient(streams=args.streams, files=args.files)
        aggregates_client = MassiveClient(streams=AGGREGATE_STREAMS, files=AGGREGATE_FILES)
    except (MassiveCredentialError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")

    today = new_york_today()
    end = args.end or today
    try:
        start = args.start or oldest_served_day(client, today)
        client.condition_table(download_dir)
    except (MassiveEntitlementError, MassiveTransportError) as exc:
        parser.exit(1, f"{exc}\n")

    vendor_root = download_dir / VENDOR_DIR
    store = data_dir / MARKET_FOLDER / Path(trade_bar_store_name(BAR_INTERVAL)).stem / trade_bar_store_name(BAR_INTERVAL)
    write_readme(store)
    config = MassiveTradeBarsDatasetConfig(
        zarr_file_path=str(store),
        raw_data_dir_path=str(vendor_root),
        sharadar_dir=str(sharadar_dir),
        bar_interval=BAR_INTERVAL,
    )

    def dataset(day: date) -> MassiveTradeBarDataset:
        return MassiveTradeBarDataset(replace(config, start_date=day.isoformat(), end_date=day.isoformat()))

    through = dataset(start).converted_through()
    watermark = read_watermark(vendor_root, "trades")
    lost = [
        day for day in sessions(first_day_to_download(start, through), min(end, watermark or date.min))
        if not raw_file(vendor_root, "trades", day).exists()
    ]
    if lost:
        print(
            f"warning: {len(lost)} day(s) are downloaded by the watermark but neither in the store nor on disk "
            f"({lost[0]}..{lost[-1]}); they are not downloaded again unless "
            f"{vendor_root / 'trades' / '_watermark.json'} is moved back before {lost[0]}."
        )

    all_days = sessions(start, end)
    print(
        f"Massive backfill {start}..{end}: {len(all_days)} session(s); store {store} holds through "
        f"{through or 'nothing yet'}; trades downloaded through {watermark or 'nothing yet'}."
    )

    aggregates = {
        t: AggregatesAhead(aggregates_client, t, first_day_to_download(start, through), end, download_dir) for t in AGGREGATE_TYPES
    }
    started = time.monotonic()
    fetched = 0
    converted = 0
    kept: list[date] = []
    unpublished: date | None = None
    # Wall-clock times of the last days finished: the ETA follows the recent
    # pace, since files grow about fourfold from 2016 to 2025.
    finished = deque([started], maxlen=RECENT_DAYS + 1)
    try:
        for done in trade_days(client, start, end, through, download_dir):
            if not done.published:
                # The store only grows at its end: a later day appended now
                # would leave this one out of it for good.
                unpublished = done.day
                break
            fetched += done.fetched_bytes
            ds = dataset(done.day)
            if through is not None and done.day <= through:
                # In the store already: a run that stopped before deleting the raw file.
                if not ds.delete_converted_trades(done.day) and raw_file(vendor_root, "trades", done.day).exists():
                    kept.append(done.day)
                continue
            for run in aggregates.values():
                run.download_through(done.day)
            converting = time.monotonic()
            ds.update(granularity="day")
            through = done.day
            stats = ds.last_stats["days"][done.day.isoformat()]
            deleted = ds.delete_converted_trades(done.day)
            if not deleted:
                kept.append(done.day)
            converted += 1

            finished.append(time.monotonic())
            elapsed = finished[-1] - started
            per_day = (finished[-1] - finished[0]) / (len(finished) - 1)
            moved = fetched + sum(run.fetched_bytes for run in aggregates.values())
            left = len([day for day in all_days if day > done.day])
            print(
                f"{done.day}  {done.fetched_bytes / 1e6:,.0f} MB in {done.seconds:.0f}s; converted in "
                f"{time.monotonic() - converting:.0f}s; {agreement(stats)}; "
                f"raw {'deleted' if deleted else 'kept'} | {converted} done, {left} left, "
                f"{moved / 1e6 / max(elapsed, 1e-9):.1f} MB/s sustained, "
                f"ETA {duration(per_day * left)}",
                flush=True,
            )
    except (MassiveEntitlementError, MassiveTransportError) as exc:
        print(f"stopped: {exc}\nRerun with the same arguments to resume.")
        return 1
    except KeyboardInterrupt:
        print("interrupted; rerun with the same arguments to resume.")
        return 130
    finally:
        for run in aggregates.values():
            run.close()

    print(f"done: {converted} day(s) converted in {duration(time.monotonic() - started)}; store holds through {through}.")
    if unpublished is not None:
        print(f"stopped at {unpublished}: not published yet; a later run goes on from it.")
    if kept:
        print(f"raw trades kept (not checked against Massive's minute bars): {[d.isoformat() for d in kept]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

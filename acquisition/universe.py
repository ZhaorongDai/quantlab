"""Survivorship-bias-free, point-in-time US-equity symbol universe.

This module deliberately does NOT subclass `base/acquisition.py:Acquisition`.
`Acquisition` subclasses are per-symbol, watermark-driven OHLCV time-series
fetchers (`download()`/`refresh()` loop over `self.config.symbols` and call
`_fetch_and_write(symbol, start, end)` once per symbol). A symbol roster /
membership-interval table is a fundamentally different shape: it is a single
bulk fetch producing a table of *many* symbols' date ranges, not a per-symbol
watermark refresh. Forcing this into `Acquisition`'s contract would require
overriding almost every method meaninglessly -- this preempts future attempts
to force it into that hierarchy (see 02-08-PLAN.md / 02-08-RESEARCH.md
Pattern 1).

Two independent reference-data problems are solved here:

- `NasdaqUniverseFetcher`: the full NASDAQ-listed Common Stock roster,
  including historically delisted symbols, sourced from Tiingo's own
  `supported_tickers.csv` (NOT `nasdaqlisted.txt`, which only lists
  currently-active tickers and cannot represent delisted history at all).
- `SP500MembershipFetcher`: point-in-time S&P 500 constituent membership,
  reconstructed via forward-chronological event simulation over Wikipedia's
  "Historical components of the S&P 500" change log, anchored against a
  known-correct current snapshot.

`UniverseCatalog` merges both into one `(symbol, category, start_date,
end_date)` reference table, persisted via the existing `PlBackend` as
parquet -- per Locked Decision A1 (02-08-PLAN.md), this table is
reference/metadata, not xarray/Zarr pipeline data, on the same footing as
`config/instruments.yaml`.
"""

import io
import zipfile
from pathlib import Path
from typing import Self

import pandas as pd
import polars as pl
import requests
from loguru import logger

from base.config import UniverseConfig
from dataset.backend import PlBackend


class NasdaqUniverseFetcher:
    """Fetches Tiingo's full historical ticker directory and filters it to
    the NASDAQ-listed Common Stock universe (current + delisted).

    Source: Tiingo's own `supported_tickers.csv` -- NOT `nasdaqlisted.txt`,
    which only lists currently-listed securities and cannot represent
    delisted history at all (02-08-RESEARCH.md finding #2).
    """

    SOURCE_URL = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
    # Locked Decision A4 (02-08-PLAN.md): NASDAQ-listed common stock only, no
    # OTC/Expert-Market tiers. Matches the objective's literal "Nasdaq
    # market" framing.
    EXCHANGE_FILTER = ("NASDAQ",)
    ASSET_TYPE = "Stock"
    PRICE_CURRENCY = "USD"

    def fetch(self) -> pl.DataFrame:
        response = requests.get(self.SOURCE_URL, timeout=30)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            with archive.open("supported_tickers.csv") as csv_file:
                data = pl.read_csv(csv_file)

        data = data.filter(
            pl.col("exchange").is_in(self.EXCHANGE_FILTER)
            & (pl.col("assetType") == self.ASSET_TYPE)
            & (pl.col("priceCurrency") == self.PRICE_CURRENCY)
        )
        data = data.rename(
            {"ticker": "symbol", "startDate": "start_date", "endDate": "end_date"}
        )
        return data.select(["symbol", "start_date", "end_date"])


class SP500MembershipFetcher:
    """Reconstructs point-in-time S&P 500 membership intervals from a
    current-anchor snapshot plus a dated historical change log.

    `PIT_COVERAGE_START` ("1976-07-01") is the verified earliest row in the
    Wikipedia `id="changes"` table -- NOT the same as the page's own prose
    claim of 1963 coverage (02-08-RESEARCH.md Pitfall 1). Point-in-time
    queries before this date cannot be correctly answered and must be
    explicitly rejected, never silently answered with an incomplete history.
    """

    ANCHOR_URL = (
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
        "main/data/constituents.csv"
    )
    CHANGES_URL = (
        "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
    )
    PIT_COVERAGE_START = "1976-07-01"

    def __init__(self, cache_dir: str):
        self._cache_path = Path(cache_dir) / "sp500_changes_snapshot.parquet"

    def fetch_anchor(self) -> pl.DataFrame:
        response = requests.get(self.ANCHOR_URL, timeout=30)
        response.raise_for_status()
        data = pl.read_csv(io.StringIO(response.text))
        data = data.rename({"Symbol": "symbol", "Date added": "date_added"})
        return data.select(["symbol", "date_added"])

    def fetch_changes(self) -> pl.DataFrame:
        cached_row_count = 0
        if self._cache_path.exists():
            try:
                cached_row_count = len(pl.read_parquet(self._cache_path))
            except Exception:
                cached_row_count = 0

        try:
            response = requests.get(
                self.CHANGES_URL,
                headers={"User-Agent": "quantlab (contact: dzr233@gmail.com)"},
                timeout=30,
            )
            response.raise_for_status()

            tables = pd.read_html(io.StringIO(response.text), attrs={"id": "changes"})
            changes = tables[0]
            changes.columns = [
                "effective_date",
                "added_ticker",
                "added_security",
                "removed_ticker",
                "removed_security",
                "reason",
                "refs",
            ]
            changes["effective_date"] = pd.to_datetime(
                changes["effective_date"]
            ).dt.strftime("%Y-%m-%d")
            changes = changes[["effective_date", "added_ticker", "removed_ticker"]]

            required_columns = {"effective_date", "added_ticker", "removed_ticker"}
            if not required_columns.issubset(set(changes.columns)):
                raise ValueError(
                    f"Parsed Wikipedia changes table missing expected columns: "
                    f"expected {required_columns}, got {set(changes.columns)}"
                )
            if len(changes) < cached_row_count:
                raise ValueError(
                    f"Parsed Wikipedia changes table has fewer rows "
                    f"({len(changes)}) than the cached snapshot "
                    f"({cached_row_count}) -- memberships only close, they "
                    f"don't retroactively vanish; treating this as a parse "
                    f"failure/schema-drift."
                )

            parsed = pl.from_pandas(changes)
        except Exception as exc:
            logger.error(
                f"Failed to fetch/parse S&P 500 changes from Wikipedia "
                f"({self.CHANGES_URL}): {exc}. Falling back to cached "
                f"snapshot; NOT overwriting the cache file."
            )
            return self._load_cache()

        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        parsed.write_parquet(self._cache_path)
        return parsed

    def _load_cache(self) -> pl.DataFrame:
        if self._cache_path.exists():
            return pl.read_parquet(self._cache_path)
        raise RuntimeError(
            "No cached S&P 500 changes snapshot available and live fetch "
            "failed -- cannot build sp500_constituent intervals."
        )

    def reconstruct_intervals(
        self, anchor: pl.DataFrame, changes: pl.DataFrame
    ) -> pl.DataFrame:
        anchor_symbols = set(anchor["symbol"])
        anchor_date_added = dict(zip(anchor["symbol"], anchor["date_added"]))

        changes_sorted = changes.sort("effective_date")
        open_intervals: dict[str, str] = {}
        closed: list[tuple[str, str, str | None]] = []

        last_eff = self.PIT_COVERAGE_START
        for row in changes_sorted.iter_rows(named=True):
            eff = row["effective_date"]
            last_eff = eff
            if row["removed_ticker"] is not None:
                sym = row["removed_ticker"]
                if sym in open_intervals:
                    closed.append((sym, open_intervals.pop(sym), eff))
                else:
                    logger.warning(
                        f"{sym}: removal at {eff} has no matching prior "
                        f"'added' event -- left-censored interval, using "
                        f"PIT_COVERAGE_START ({self.PIT_COVERAGE_START}) as "
                        f"start_date"
                    )
                    closed.append((sym, self.PIT_COVERAGE_START, eff))
            if row["added_ticker"] is not None:
                sym = row["added_ticker"]
                if sym in open_intervals:
                    logger.warning(
                        f"{sym}: duplicate 'added' event at {eff} while an "
                        f"interval was already open since "
                        f"{open_intervals[sym]} -- data quality issue, "
                        f"keeping the earlier open date"
                    )
                else:
                    open_intervals[sym] = eff

        # Reconcile remaining open intervals against the current anchor.
        for sym, start in open_intervals.items():
            if sym not in anchor_symbols:
                logger.warning(
                    f"{sym}: open interval since {start} but not in current "
                    f"anchor set -- anchor CSV may be stale relative to the "
                    f"Wikipedia change log"
                )
                closed.append((sym, start, last_eff))
            else:
                closed.append((sym, start, None))

        # Anchor members with NO 'added'/'removed' event anywhere in the
        # change log are either original constituents or were added before
        # PIT_COVERAGE_START.
        seen_symbols = {c[0] for c in closed}
        for sym in anchor_symbols - seen_symbols:
            start = anchor_date_added.get(sym) or self.PIT_COVERAGE_START
            closed.append((sym, start, None))

        return pl.DataFrame(closed, schema=["symbol", "start_date", "end_date"], orient="row")

    def build_intervals(self) -> pl.DataFrame:
        anchor = self.fetch_anchor()
        changes = self.fetch_changes()
        return self.reconstruct_intervals(anchor, changes)


class UniverseCatalog:
    """Point-in-time US-equity universe reference table.

    Combines `NasdaqUniverseFetcher` (category="nasdaq_all") and
    `SP500MembershipFetcher` (category="sp500_constituent") into one
    `(symbol, category, start_date, end_date)` table, persisted via
    `PlBackend`/parquet (Locked Decision A1, 02-08-PLAN.md).

    IMPORTANT for future backtest phases: `get_symbols_as_of()` must be
    called per-rebalance-date in a walk-forward backtest, not once at setup
    time, to avoid look-ahead bias (02-08-RESEARCH.md Open Question 2).
    """

    def __init__(self, config: UniverseConfig):
        self.config = config
        self._backend = PlBackend()

    def build(self) -> Self:
        nasdaq = NasdaqUniverseFetcher().fetch().with_columns(
            pl.lit("nasdaq_all").alias("category")
        )
        sp500 = SP500MembershipFetcher(
            cache_dir=self.config.cache_dir
        ).build_intervals().with_columns(
            pl.lit("sp500_constituent").alias("category")
        )
        combined = pl.concat([nasdaq, sp500], how="vertical_relaxed").select(
            ["symbol", "category", "start_date", "end_date"]
        )
        self._backend.to_internal(combined.lazy())
        return self

    def save(self) -> Self:
        Path(self.config.output_path).parent.mkdir(parents=True, exist_ok=True)
        self._backend.write(self.config.output_path)
        return self

    @classmethod
    def load(cls, config: UniverseConfig) -> "UniverseCatalog":
        catalog = cls(config)
        catalog._backend.read(config.output_path)
        return catalog

    def get_symbols_as_of(self, category: str, as_of_date: str) -> list[str]:
        if (
            category == "sp500_constituent"
            and as_of_date < SP500MembershipFetcher.PIT_COVERAGE_START
        ):
            raise ValueError(
                f"Cannot answer sp500_constituent membership before "
                f"{SP500MembershipFetcher.PIT_COVERAGE_START} -- the "
                f"Wikipedia-sourced change log is left-censored at that "
                f"date and this query cannot be answered correctly, rather "
                f"than silently defaulting to an incomplete/wrong answer."
            )

        matched = self._backend.get_lazyframe().filter(
            (pl.col("category") == category)
            & (pl.col("start_date") <= as_of_date)
            & (pl.col("end_date").is_null() | (pl.col("end_date") >= as_of_date))
        )
        return matched.select("symbol").unique().collect()["symbol"].to_list()

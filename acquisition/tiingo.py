import os
from pathlib import Path

import polars as pl
from tiingo import TiingoClient

from base.acquisition import Acquisition
from base.config import AcquisitionConfig
from enums.data import TiingoColumns

# Extend when intraday frequencies are added -- never hardcode "daily" inline
# in _fetch_and_write.
_FREQUENCY_MAP = {"1d": "daily"}


class TiingoAcquisition(Acquisition):
    """Config-driven, incrementally-refreshable Tiingo EOD data acquisition.

    `TIINGO_API_KEY` is read directly from `os.environ` in `__init__` and
    passed only into the in-memory `TiingoClient` constructor argument --
    never assigned to `self.config` or any other dataclass-facing attribute.
    """

    def __init__(self, config: AcquisitionConfig):
        super().__init__(config)

        if not os.environ.get("TIINGO_API_KEY"):
            raise RuntimeError(
                "TIINGO_API_KEY environment variable is not set. Export it "
                "before running acquisition (see Tiingo dashboard for your "
                "key)."
            )
        self._client = TiingoClient(
            {"session": True, "api_key": os.environ["TIINGO_API_KEY"]}
        )

    def _fetch_and_write(
        self, symbol: str, start_date: str, end_date: str
    ) -> None:
        frequency = _FREQUENCY_MAP[self.config.frequency]
        response = self._client.get_ticker_price(
            symbol,
            fmt="json",
            startDate=start_date,
            endDate=end_date,
            frequency=frequency,
            columns=TiingoColumns.EOD,
        )
        data = pl.DataFrame(response)
        if data.is_empty():
            return

        # Tiingo's `date` field is an ISO-8601 string with a trailing `Z`
        # (UTC) offset (e.g. "2024-01-02T00:00:00.000Z"). Parse it as UTC
        # then drop the tz so the resulting dtype is a naive `pl.Datetime`,
        # matching the naive timestamps produced elsewhere in the codebase
        # (e.g. `StockDataset`'s naive `str.to_datetime()` filter bounds) --
        # parsing without an explicit time zone raises on tz-aware strings.
        data = data.with_columns(
            pl.col("date")
            .str.to_datetime(time_zone="UTC")
            .dt.replace_time_zone(None)
        )
        data = data.rename({"date": "timestamp"})
        data = data.with_columns(pl.lit(symbol).alias("symbol"))

        out_dir = Path(self.config.raw_data_dir_path) / symbol
        out_dir.mkdir(parents=True, exist_ok=True)
        data.write_parquet(out_dir / "data.pqt")

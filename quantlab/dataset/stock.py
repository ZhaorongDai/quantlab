"""US-equity bars from a vendor's hive-partitioned parquet tree.

``StockDataset`` reads the raw tier written by the acquisition layer
(``downloads/{market}/{frequency}/{subdir}/{vendor}/<key>=<value>/*.pqt``),
pushes the date window down into the parquet scan so only the partitions it
needs are opened, and produces the canonical ``(timestamp, symbol)`` panel.
It is the US-equity counterpart of ``quantlab/dataset/spot.py`` and the
memory-bounded reference implementation of the chunked conversion path.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from joblib import Parallel, delayed
from tqdm import tqdm

from quantlab.base.config import DatasetConfig
from quantlab.base.data import MarketDataset
from quantlab.dataset._support.cleaning import dedup_raw_frame
from quantlab.enums.data import RAW_HIVE_KEYS, BinanceCSVHeaders
from quantlab.utils.file import file_date_filter
from quantlab.utils.symbol_axis import sort_symbol_axis
from quantlab.utils.timer import Timer


class StockDataset(MarketDataset):
    """US-equity dataset over a single vendor's hive-partitioned raw tree.

    The raw root must end in the vendor's own directory and hold that vendor's
    shards only; two vendors under one root would merge into a blended price
    series with no error, so the root name, the ``vendor`` column and the
    parquet schema are all checked before any row is used. Daily (``1d``)
    data is partitioned by ``month``, intraday (``1m``) by session ``date``,
    and ``tick`` by ``data_type``, ``date`` and ``symbol``; tick data has no
    dense-panel form and is only reachable through ``_scan_raw``.

    Examples
    --------
    >>> config = DatasetConfig(
    ...     raw_data_dir_path="downloads/us_equity/1d/us_all/tiingo",
    ...     zarr_file_path="data/us_equity/1d/us_all.zarr",
    ...     catalog_path="data/us_equity/catalog",
    ...     market="us_equity",
    ...     frequency="1d",
    ...     vendor="tiingo",
    ...     start_date="2024-01-01",
    ...     end_date="2024-12-31",
    ... )
    >>> StockDataset(config).from_raw_data_chunked(granularity="quarter")
    >>> panel = StockDataset(config).read().get_xarray_dataset()
    """

    #: Hive key dtypes per frequency, keyed to match ``RAW_HIVE_KEYS``.
    #: Always passed to ``pl.scan_parquet`` explicitly: an inferred
    #: numeric-looking value such as ``symbol=8686`` would become an integer
    #: column and a string predicate against it would silently match nothing.
    HIVE_SCHEMA_BY_FREQUENCY = {
        "1d": {"month": pl.String},
        "1m": {"date": pl.Date},
        "tick": {"data_type": pl.String, "date": pl.Date, "symbol": pl.String},
    }

    #: Hive keys that are partition metadata only and are dropped after the
    #: scan. ``symbol`` is deliberately absent: on the tick layout the path
    #: segment is the only carrier of the symbol, so it must survive.
    DERIVED_HIVE_KEYS = ("month", "date", "data_type")

    #: Filename suffix of one raw shard, shared by the scan glob and the
    #: ``has_raw_data`` probe so both decide on the same set of files.
    RAW_SHARD_SUFFIX = ".pqt"

    def __init__(self, dataset_config: DatasetConfig):
        """Create the dataset from a ``DatasetConfig``."""
        super().__init__(dataset_config)

    @property
    def _hive_keys(self) -> tuple[str, ...]:
        """Return the hive partition keys for this config's frequency.

        Read from ``RAW_HIVE_KEYS``, the same mapping the acquisition layer
        writes the tree with, so reader and writer cannot disagree.
        """
        return RAW_HIVE_KEYS[self.config.frequency]

    def _assert_vendor_root(self) -> Path:
        """Return the raw root after checking it is one vendor's directory.

        ``pl.scan_parquet`` silently unions every shard beneath its root, and
        the later dedup would then collapse two vendors' rows into one
        arbitrary blend, so the root must terminate at the configured
        vendor's own directory.

        Raises
        ------
        ValueError
            If ``config.vendor`` is unset or the root's basename
            differs from it.
        """
        if not self.config.vendor:
            raise ValueError(
                f"{self.__class__.__name__}: DatasetConfig.vendor is not set, "
                f"so there is no way to check that "
                f"{self.config.raw_data_dir_path!r} holds exactly one vendor's "
                f"data. Two vendors under one root merge with NO error and NO "
                f"provenance (D-11), so this scan refuses rather than "
                f"guessing. Set vendor=... on the config, or build it via "
                f"config.stock_kline_config(vendor=...)."
            )

        root = Path(self.config.raw_data_dir_path)
        if root.name != self.config.vendor:
            raise ValueError(
                f"{self.__class__.__name__}: raw_data_dir_path {str(root)!r} "
                f"has basename {root.name!r} but the configured vendor is "
                f"{self.config.vendor!r}. The raw path must TERMINATE at the "
                f"vendor segment (D-11). A root pointed one level up walks "
                f"into every vendor directory beneath it and merges them with "
                f"no error and no provenance -- so this is refused rather than "
                f"scanned. Expected a path ending in "
                f"/{self.config.vendor}."
            )
        return root

    def _scan_root(self) -> Path:
        """Return the directory handed to ``pl.scan_parquet``.

        For ``1d`` and ``1m`` this is the vendor root. For ``tick`` it is one
        level deeper, ``{vendor_root}/data_type={quotes|trades}``: polars fixes
        the scan's schema from the first file it finds and enforces it on
        every file, including ones a predicate has pruned, so quotes and
        trades (which have different columns) can only be separated by the
        root, never by a filter.
        """
        root = Path(self.config.raw_data_dir_path)
        if "data_type" in self._hive_keys:
            return root / f"data_type={self._tick_data_type}"
        return root

    @property
    def _scanned_hive_keys(self) -> tuple[str, ...]:
        """Return the hive keys the window predicate may filter on.

        ``data_type`` is excluded because ``_scan_root`` already scopes the
        scan to one value of it; a predicate would only restate that.
        """
        return tuple(key for key in self._hive_keys if key != "data_type")

    @property
    def _materialised_hive_keys(self) -> tuple[str, ...]:
        """Return every hive key polars materialises as a column.

        The recursive glob makes polars parse every ``key=value`` segment on
        the path, including ``data_type`` above the scan root, so this is the
        full key set: the hive schema must declare all of them and the
        post-scan drop must remove all of the derived ones.
        """
        return self._hive_keys

    def _scanned_hive_schema(self) -> dict:
        """Return ``HIVE_SCHEMA_BY_FREQUENCY`` narrowed to the materialised keys."""
        schema = self.HIVE_SCHEMA_BY_FREQUENCY[self.config.frequency]
        keys = self._materialised_hive_keys
        return {name: dtype for name, dtype in schema.items() if name in keys}

    def _hive_window_predicate(self, start, end) -> pl.Expr:
        """Return the predicate over the hive keys that prunes partitions.

        For ``1d`` the key is ``month``, a ``YYYY-MM`` string compared
        lexicographically. The predicate includes both edge partitions in
        full; the ``timestamp`` predicate applied beside it trims them.

        Raises
        ------
        NotImplementedError
            For a frequency with no known window key.
        """
        keys = self._scanned_hive_keys
        if keys == ("month",):
            return (pl.col("month") >= pl.lit(start.strftime("%Y-%m"))) & (
                pl.col("month") <= pl.lit(end.strftime("%Y-%m"))
            )
        if keys in (("date",), ("date", "symbol")):
            # `date` is the only prunable window key for both intraday tiers;
            # a scan covers the whole roster, so `symbol` carries no predicate.
            return self._session_date_window_predicate(start, end)
        raise NotImplementedError(
            f"{self.__class__.__name__}: no hive window predicate for "
            f"frequency {self.config.frequency!r} (scanned keys {keys})."
        )

    @property
    def _tick_data_type(self) -> str:
        """Return which tick data type (``quotes`` or ``trades``) to read.

        Taken from ``config.kwargs["data_type"]``. The two share one vendor
        root and have different columns, so the choice is required.

        Raises
        ------
        ValueError
            If the config does not say which one to read.
        """
        data_type = (self.config.kwargs or {}).get("data_type")
        if not data_type:
            raise ValueError(
                f"{self.__class__.__name__}: frequency "
                f"{self.config.frequency!r} needs kwargs['data_type'] set to "
                f"'quotes' or 'trades'; got {data_type!r}. The two land under "
                f"one vendor root, distinguished by the leading `data_type=` "
                f"hive key, and they carry DIFFERENT columns -- an unfiltered "
                f"scan of a root holding both raises a schema error rather "
                f"than returning a blended frame, which is the structural "
                f"guarantee, not a bug. Say which one you want."
            )
        return str(data_type)

    #: How far the intraday hive predicate widens the window at each edge.
    #: The ``date=`` key is a session date in the writer's exchange time zone
    #: while window edges arrive as naive UTC datetimes, and the two can
    #: disagree by up to a day. One day of slack over-includes at most two
    #: partitions, which the ``timestamp`` predicate trims exactly; without it
    #: the tail of the window's first session would be silently lost.
    SESSION_DATE_SLACK = timedelta(days=1)

    def _session_date_window_predicate(self, start, end) -> pl.Expr:
        """Return the ``date`` predicate widened by ``SESSION_DATE_SLACK``."""
        return (
            pl.col("date") >= pl.lit((start - self.SESSION_DATE_SLACK).date())
        ) & (pl.col("date") <= pl.lit((end + self.SESSION_DATE_SLACK).date()))

    def _assert_single_vendor_and_drop(
        self, data: pl.LazyFrame
    ) -> pl.LazyFrame:
        """Check the scanned ``vendor`` column holds one value, then drop it.

        This runs before ``dedup_raw_frame``, which would otherwise collapse
        two vendors' overlapping rows and destroy the evidence of a merge.

        Raises
        ------
        ValueError
            If more than one vendor is present, or the one
            present is not the configured vendor.
        """
        vendors = (
            data.select(pl.col("vendor").unique())
            .collect()
            .get_column("vendor")
            .to_list()
        )
        if len(vendors) > 1:
            raise ValueError(
                f"{self.__class__.__name__}: the raw tree under "
                f"{self.config.raw_data_dir_path!r} holds rows from "
                f"{len(vendors)} vendors ({sorted(map(str, vendors))}) but is "
                f"configured for {self.config.vendor!r} alone. Merging two "
                f"vendors' bars produces an untraceable blended price series "
                f"-- dedup on (timestamp, symbol) would then collapse the "
                f"overlaps arbitrarily. Separate the vendors into sibling "
                f"roots (D-11) rather than relaxing this assertion."
            )
        if vendors and str(vendors[0]) != self.config.vendor:
            raise ValueError(
                f"{self.__class__.__name__}: the raw tree under "
                f"{self.config.raw_data_dir_path!r} holds rows written by "
                f"vendor {str(vendors[0])!r} but the config says "
                f"{self.config.vendor!r}. The path and the data disagree; "
                f"refusing to scan rather than mislabelling the provenance of "
                f"everything downstream."
            )
        return data.drop("vendor")

    def has_raw_data(self) -> bool:
        """Return whether the raw tree holds at least one shard to convert.

        This is the single raw-presence check: ``_scan_raw`` uses it to tell
        an absent root from a window that pruned to nothing, and the ingest
        shells call it to refuse a conversion before it starts. It accounts
        for the tick layout's ``data_type`` descent, which a bare check of
        ``config.raw_data_dir_path`` would miss. It only stats the directory
        and opens no parquet file.

        Examples
        --------
        >>> ds = StockDataset(config)
        >>> ds.has_raw_data()  # the vendor root exists but holds no shard
        False
        >>> ds.has_raw_data()  # once a month=2024-01/part.pqt shard lands
        True
        """
        root = self._scan_root()
        return root.exists() and any(root.rglob(f"*{self.RAW_SHARD_SUFFIX}"))

    def _scan_raw(self, start_date=None, end_date=None) -> pl.LazyFrame:
        """Return the lazy scan of the raw tree for a window, uncollected.

        Scans the shards, prunes partitions with the hive predicate, trims
        the exact edges with a ``timestamp`` predicate, asserts single-vendor
        provenance, drops the derived hive keys, sorts, and (except for tick
        data) deduplicates on ``(timestamp, symbol)``. ``None`` for either
        edge means the config's own edge.

        Raises
        ------
        ValueError
            If the raw tree is absent or empty.
        """
        self._assert_vendor_root()
        root = self._scan_root()

        # An absent root would fail inside polars with an unhelpful schema
        # error; a window that prunes to nothing is fine and yields an empty
        # frame.
        if not self.has_raw_data():
            raise ValueError(
                f"{self.__class__.__name__}: no raw data for vendor "
                f"{self.config.vendor!r} at frequency "
                f"{self.config.frequency!r} under {str(root)!r}. Fetch it "
                f"first (e.g. `uv run python ingest_us_equity.py`) before "
                f"converting. This is the absent-root case; a window that "
                f"merely prunes to zero rows returns an empty frame instead."
            )

        # A recursive glob over the shard suffix rather than the bare
        # directory: polars refuses a directory holding mixed extensions, and
        # a stray `.DS_Store` is enough to trigger that. Directory pruning
        # still works with the glob. `hive_schema` is passed explicitly so a
        # numeric-looking key is not inferred as an integer, and the
        # `extra_columns`/`missing_columns` defaults are left raising: a
        # mixed-schema scan is a shard written outside the expected columns.
        data = pl.scan_parquet(
            str(root / "**" / f"*{self.RAW_SHARD_SUFFIX}"),
            hive_partitioning=True,
            hive_schema=self._scanned_hive_schema(),
        )

        start = self._as_datetime(
            self.config.start_date if start_date is None else start_date
        )
        end = self._as_datetime(
            self.config.end_date if end_date is None else end_date
        )

        # Both predicates are needed: the hive one prunes partitions at plan
        # time (a `timestamp` predicate cannot), and the timestamp one trims
        # the two partially covered edge partitions exactly.
        data = data.filter(self._hive_window_predicate(start, end))
        data = data.filter(
            pl.col("timestamp") >= pl.lit(start),
            pl.col("timestamp") <= pl.lit(end),
        )

        data = self._assert_single_vendor_and_drop(data)
        # Drop only the derived keys; `symbol` is a real data column that the
        # tick layout expresses as a path segment. Iterates the materialised
        # set so `data_type` is dropped too on the tick path.
        data = data.drop(
            [
                key
                for key in self._materialised_hive_keys
                if key in self.DERIVED_HIVE_KEYS
            ]
        )

        data = data.sort(by=["timestamp", "symbol"])
        if self.config.frequency == "tick":
            # Tick data has no dense-panel form, and many genuine quotes or
            # trades share one (timestamp, symbol), so it is not deduplicated.
            return data
        return dedup_raw_frame(data, keep="last")

    @staticmethod
    def _as_datetime(value) -> datetime:
        """Normalise a window edge to a naive ``datetime``.

        An ISO date string resolves to midnight of that date, so an inclusive
        ``<=`` end edge keeps that whole day's daily bar. A ``pd.Timestamp``
        from the chunk planner passes through with its time of day intact.
        """
        return pd.Timestamp(value).to_pydatetime()

    def _raw_axes_in_range(self) -> tuple[list, pd.DatetimeIndex]:
        """Return ``(pinned_symbols, observed_timestamps)`` from one lazy scan.

        Only the two axis columns are collected; nothing is densified. The
        symbol order comes from ``sort_symbol_axis`` and the label type is the
        raw ``symbol`` column's own.
        """
        scan = self._scan_raw()
        symbols = sort_symbol_axis(
            scan.select("symbol").unique().collect()["symbol"].to_list()
        )
        timestamps = (
            scan.select("timestamp").unique().collect()["timestamp"].to_list()
        )
        return symbols, pd.DatetimeIndex(sorted(timestamps))

    def _added_symbols_with_raw_history(
        self, added: list, start, end
    ) -> dict[str, int]:
        """Count raw rows each of ``added`` carries in the closed window.

        The hive-pruned version of the base class probe: the symbol predicate
        is pushed into the scan and only a group-by is collected. Symbols are
        compared as text because the raw ``symbol`` column is stored as
        strings even when the pinned axis is integer-typed; without the cast
        the probe would find no rows and steer ``update`` to ``widen``.
        """
        wanted = [str(symbol) for symbol in added]
        if not wanted:
            return {}

        counts = (
            self._scan_raw(start, end)
            .filter(pl.col("symbol").is_in(wanted))
            .group_by("symbol")
            .agg(pl.len().alias("rows"))
            .collect()
        )
        return {
            str(record["symbol"]): int(record["rows"])
            for record in counts.to_dicts()
        }

    def _raw_data_to_xr_window(
        self, start_date, end_date, symbols: Optional[list[str]] = None
    ) -> xr.Dataset:
        """Return the dense panel for one window, reindexed onto ``symbols``.

        Only the window's partitions are scanned, so memory is bounded by the
        window rather than by the whole configured range.
        """
        data = self._scan_raw(start_date, end_date)
        data = data.collect().to_pandas().set_index(["timestamp", "symbol"])
        data = data.to_xarray()
        if symbols is not None:
            data = data.reindex(symbol=list(symbols))
        return data

    def _raw_data_to_xr(self) -> xr.Dataset:
        """Return the dense panel for the config's whole date range."""
        with Timer(f" {self.__class__.__name__}: from pqt"):
            return self._raw_data_to_xr_window(
                self.config.start_date, self.config.end_date, symbols=None
            )

    def _to_kunquant(
        self, data: xr.Dataset, data_columns: tuple
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Export the requested columns as contiguous ``[time, symbol]`` arrays."""
        with Timer(f"{self.__class__.__name__}: to kunquant"):
            data = data.sortby(["timestamp", "symbol"])
            timestamp = data["timestamp"].values
            symbols = data["symbol"].values
            input_dict = {}
            for col in data_columns:
                input_dict[col] = np.ascontiguousarray(
                    data[col].to_numpy().astype(np.float32)
                )  # [time, symbol]
            return input_dict, symbols, timestamp

    @staticmethod
    def _get_instrument(symbol: str, venue: str):
        """Not implemented: this dataset has no Nautilus instrument model yet."""
        raise ValueError("Not finished")

    def _xr_to_bars(
        self, data: xr.Dataset, symbol: str, venue: str = "BINANCE"
    ):
        """Not implemented: this dataset has no Nautilus bar conversion yet."""
        raise ValueError("Not finished")

    def _to_nautilus(
        self, data: xr.Dataset, venue: str = "BINANCE", n_jobs: int = 16
    ):
        """Not implemented: the Nautilus exit is refused for US equities."""
        raise ValueError("Not finished")

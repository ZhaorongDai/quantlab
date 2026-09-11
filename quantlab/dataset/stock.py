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
from quantlab.dataset.cleaning import dedup_raw_frame
from quantlab.enums.data import RAW_HIVE_KEYS, BinanceCSVHeaders
from quantlab.utils.file import file_date_filter
from quantlab.utils.timer import Timer


class StockDataset(MarketDataset):
    """US-equity dataset over a vendor-namespaced, hive-partitioned raw tier.

    `get_pqt_files` is no longer imported: `_scan_raw` hands the raw ROOT to
    `pl.scan_parquet` rather than a file list, because the directory form is
    what auto-enables hive partitioning and therefore what makes plan-time
    directory pruning possible at all. (`utils/file.py:get_pqt_files` had no
    other caller in the repo; the function is left in place, unused here.)
    """

    #: Explicit hive dtypes per frequency, keyed to match
    #: `enums.data.RAW_HIVE_KEYS` exactly.
    #:
    #: ALWAYS passed to `pl.scan_parquet`, never inferred. An unpinned
    #: numeric-looking hive value is inferred as an integer -- `symbol=8686`
    #: becomes `Int64` -- and a string comparison against it then silently
    #: matches NOTHING (03.2-RESEARCH.md Pitfall 8). That is not hypothetical
    #: here: quick task 260906-eme established that `_WELL_FORMED_TICKER`
    #: admits digits precisely because digit-bearing tickers are legitimate.
    HIVE_SCHEMA_BY_FREQUENCY = {
        "1d": {"month": pl.String},
        "1m": {"date": pl.Date},
        "tick": {"data_type": pl.String, "date": pl.Date, "symbol": pl.String},
    }

    #: Hive keys that are PURELY derived partition metadata and carry no data
    #: column of their own, so `_scan_raw` drops them before handing the frame
    #: downstream. `symbol` is deliberately absent: the tick layout expresses
    #: it as a path segment, and the writer therefore omits it from the file,
    #: so the hive key is the ONLY carrier of it there.
    DERIVED_HIVE_KEYS = ("month", "date", "data_type")

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)

    @property
    def _hive_keys(self) -> tuple[str, ...]:
        """The hive partition key(s) for this config's frequency.

        Read from `enums.data.RAW_HIVE_KEYS`, the SAME mapping
        `base/acquisition.py` reads when it WRITES the tree. One definition
        means the writer and the reader cannot drift into a state where the
        scan prunes on a key the writer never produced -- which would return
        silently fewer rows than the tree holds, with nothing failing.
        """
        return RAW_HIVE_KEYS[self.config.frequency]

    def _assert_vendor_root(self) -> Path:
        """Return the raw root, having proved it is a single vendor's root.

        Two checks, one line of reasoning each, and together they make the
        cross-vendor silent merge unreachable BY ACCIDENT rather than merely
        unlikely (SC-7 / D-11).

        Measured against polars 1.44.1: two vendors writing the SAME schema
        under one scan root do not raise, do not warn, and do not record which
        vendor each row came from -- `pl.scan_parquet` simply returns their
        union, and `dedup_raw_frame(keep="last")` then collapses the duplicate
        `(timestamp, symbol)` pairs to one ARBITRARILY. The result is an
        untraceable blended price series that looks exactly like clean data.
        A `.../{vendor}/...` path segment alone stops none of that: it carries
        no `=`, so it is not a hive key, and a scan rooted one level up walks
        straight into both.
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
        """The directory `pl.scan_parquet` is actually handed.

        For `1d`/`1m` that is the vendor root. For `tick` it is one level
        deeper -- `{vendor_root}/data_type={quotes|trades}` -- and that is a
        STRUCTURAL requirement, not a convenience.

        Measured against polars 1.44.1: a scan rooted at a tick vendor root
        holding both data types fixes its expected schema from the FIRST file
        it discovers, and that schema is then enforced against every file it
        opens -- including files a `data_type` PREDICATE has already pruned the
        scan down to. `.explain()` shows the plan correctly listing only the
        trades shard, and `.collect()` still raises
        `SchemaError: extra column in file outside of expected schema: price`,
        because the expected schema came from the alphabetically-first quotes
        shard. Filtering therefore CANNOT isolate a data type here; only
        scoping the root can.

        This is the same lesson as D-11's vendor segment, one level deeper: a
        distinction that only a predicate enforces is not isolation. The
        difference is that the vendor case merges silently while this one
        raises -- and it raises in the direction that depends on filename
        ordering, so it would look intermittent.
        """
        root = Path(self.config.raw_data_dir_path)
        if "data_type" in self._hive_keys:
            return root / f"data_type={self._tick_data_type}"
        return root

    @property
    def _scanned_hive_keys(self) -> tuple[str, ...]:
        """The hive keys visible INSIDE the scan.

        A key consumed by `_scan_root` is no longer below the scan root, so
        polars never materialises it as a column and neither the hive schema
        nor the predicate may mention it.
        """
        # `data_type` is the only key `_scan_root` consumes, and it consumes it
        # whenever the frequency declares it -- so this is a straight removal
        # rather than a condition on the resolved root.
        return tuple(key for key in self._hive_keys if key != "data_type")

    def _scanned_hive_schema(self) -> dict:
        """`HIVE_SCHEMA_BY_FREQUENCY`, narrowed to the keys inside the scan."""
        schema = self.HIVE_SCHEMA_BY_FREQUENCY[self.config.frequency]
        keys = self._scanned_hive_keys
        return {name: dtype for name, dtype in schema.items() if name in keys}

    def _hive_window_predicate(self, start, end) -> pl.Expr:
        """The predicate over the HIVE key(s), used for directory pruning.

        For `1d` the key is `month`, a `YYYY-MM` String. ISO ordering makes a
        plain string comparison correct at both edges, so no date parsing
        happens here and no time zone can creep in. The predicate is
        deliberately INCLUSIVE of the two edge partitions -- they are only
        partially covered by the window, and the `timestamp` predicate applied
        alongside this one trims them exactly.
        """
        keys = self._scanned_hive_keys
        if keys == ("month",):
            return (pl.col("month") >= pl.lit(start.strftime("%Y-%m"))) & (
                pl.col("month") <= pl.lit(end.strftime("%Y-%m"))
            )
        if keys in (("date",), ("date", "symbol")):
            # `date` is the only prunable window key for both intraday tiers.
            # `symbol` carries no predicate here: a scan is over the whole
            # configured roster, and `data_type` has already been consumed by
            # `_scan_root` (see there for why a predicate cannot do that job).
            return self._session_date_window_predicate(start, end)
        raise NotImplementedError(
            f"{self.__class__.__name__}: no hive window predicate for "
            f"frequency {self.config.frequency!r} (scanned keys {keys})."
        )

    @property
    def _tick_data_type(self) -> str:
        """Which tick data type this scan reads, from `config.kwargs`.

        RAISES when unset rather than guessing. Quotes and trades share one
        vendor root and have different column sets, so "read the tick data"
        is an ambiguous question with two incompatible answers -- and picking
        one silently would be a confident answer to a question the caller
        never actually asked.
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
    #:
    #: The hive `date=` key is a SESSION date (see
    #: `base/acquisition.py:Acquisition._session_date` and
    #: `acquisition/alpaca.py:AlpacaAcquisition.SESSION_TIME_ZONE`), while the
    #: window edges arriving here are naive-UTC datetimes. Those two disagree by
    #: up to a day in either direction, and the reader deliberately does NOT
    #: know the writer's session time zone -- encoding it here would put the
    #: same fact in two places and let them drift.
    #:
    #: One day of slack covers every session time zone within +/-24h of UTC. It
    #: over-includes at most two partitions, and the `timestamp` predicate
    #: applied alongside trims them exactly, so the cost is bounded and the
    #: alternative is not. Comparing the session key directly against
    #: `start.date()` drops the tail of the window's first session -- a row at
    #: 2024-04-01T00:30Z is 2024-03-31 20:30 ET, lives under `date=2024-03-31`,
    #: and is inside a window starting 2024-04-01T00:00. That loss is silent
    #: and reads as sparse data.
    SESSION_DATE_SLACK = timedelta(days=1)

    def _session_date_window_predicate(self, start, end) -> pl.Expr:
        """The `date=` predicate, widened by `SESSION_DATE_SLACK` at each edge.

        `date` is typed `pl.Date` by `HIVE_SCHEMA_BY_FREQUENCY`, so both sides
        of the comparison are dates and no time zone can creep in here either.
        """
        return (
            pl.col("date") >= pl.lit((start - self.SESSION_DATE_SLACK).date())
        ) & (pl.col("date") <= pl.lit((end + self.SESSION_DATE_SLACK).date()))

    def _assert_single_vendor_and_drop(
        self, data: pl.LazyFrame
    ) -> pl.LazyFrame:
        """Assert the scanned frame holds exactly ONE vendor, then drop the
        column.

        The third of SC-7's three mutually reinforcing measures, and the only
        one that works AFTER the fact: the basename check prevents a merge, and
        the written `vendor` column makes one DETECTABLE if it ever happens
        anyway -- a shard hand-copied into the wrong root, say, which no path
        assertion can see.

        This must run BEFORE `dedup_raw_frame`. Dedup on `(timestamp, symbol)`
        would collapse two vendors' overlapping rows to one arbitrarily, at
        which point the evidence that a merge occurred is gone.

        The column is then dropped, along with the hive key(s), so the frame
        handed to `_raw_data_to_xr_window` carries EXACTLY its pre-refactor
        column set and every existing assertion in tests/test_stock_dataset.py
        stands untouched.
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
        """Whether this config's raw tree holds at least one shard to convert.

        THE raw-presence predicate, and deliberately the only one. `_scan_raw`
        below reads it to tell "root absent" apart from "window pruned to
        nothing"; `quantlab.utils.cli.refuse_conversion_without_raw_data`
        reads the SAME method to refuse a conversion before it starts. Two
        copies of `exists() / rglob("*.pqt")` -- one here and one in the CLI --
        is the shape both of the 03.4 UAT gaps grew out of: a contract split
        into two statements that can then disagree about the same fact.

        Public because a shell calls it, and it is the shell's ONLY sanctioned
        way to ask: reaching for `Path(config.raw_data_dir_path)` at a call
        site would miss `_scan_root`'s tick descent into
        `data_type={quotes|trades}`, and would answer about a directory the
        scan never opens.

        Read-only: it stats a directory and stops at the first match. It
        writes nothing, deletes nothing, and opens no parquet file.
        """
        root = self._scan_root()
        return root.exists() and any(root.rglob("*.pqt"))

    def _scan_raw(self, start_date=None, end_date=None) -> pl.LazyFrame:
        """The shared LazyFrame pipeline: hive-scan, prune, date-filter,
        assert provenance, sort, dedup -- returned UNCOLLECTED.

        Parameterised by the window so polars can push both predicates DOWN
        into the parquet scan. `None` means "the config's own edge", which is
        what keeps `_raw_data_to_xr()` byte-identical to its pre-refactor self.
        """
        self._assert_vendor_root()
        root = self._scan_root()

        # Distinguish "root absent" from "root present but the window pruned to
        # nothing". Polars infers a scan's schema from the FIRST file it finds,
        # so an absent or empty root raises `ComputeError: failed to retrieve
        # first file schema ... expanded paths were empty` -- an error message
        # that names none of the things a user needs in order to fix it
        # (03.2-RESEARCH.md Pitfall 7). A pruned-to-nothing window is NOT an
        # error and returns an empty frame below.
        #
        # The test is `has_raw_data()` rather than an inline
        # `exists() / rglob()` so that the shell-level refusal
        # (`quantlab.utils.cli.refuse_conversion_without_raw_data`) decides on
        # the SAME fact this raise decides on. When they were two statements,
        # a run that fetched nothing sailed past the shell and landed here as
        # an uncaught traceback (G-03.4-1).
        if not self.has_raw_data():
            raise ValueError(
                f"{self.__class__.__name__}: no raw data for vendor "
                f"{self.config.vendor!r} at frequency "
                f"{self.config.frequency!r} under {str(root)!r}. Fetch it "
                f"first (e.g. `uv run python ingest_us_equity.py`) before "
                f"converting. This is the absent-root case; a window that "
                f"merely prunes to zero rows returns an empty frame instead."
            )

        # A DIRECTORY argument (not a file list) is what auto-enables hive
        # partitioning, and `hive_schema` is passed explicitly because an
        # inferred numeric-looking key changes dtype (Pitfall 8).
        #
        # `extra_columns` and `missing_columns` are LEFT AT THEIR RAISING
        # DEFAULTS on purpose. The SchemaError a mixed-schema scan raises is
        # the structural half of D-11 -- it is the thing that catches a shard
        # written outside `RAW_COLUMNS`. Setting `extra_columns="ignore"` "to
        # make the scan work" would reopen precisely the silent merge this
        # method exists to prevent, while looking like a bug fix.
        data = pl.scan_parquet(
            root,
            hive_partitioning=True,
            hive_schema=self._scanned_hive_schema(),
        )

        start = self._as_datetime(
            self.config.start_date if start_date is None else start_date
        )
        end = self._as_datetime(
            self.config.end_date if end_date is None else end_date
        )

        # BOTH predicates, and neither alone is correct.
        #
        # The HIVE predicate prunes DIRECTORIES at plan time -- measured on
        # polars 1.44.1 as 2 of 5 files listed in the scan node, versus 5 of 5
        # for a timestamp-only filter. A `timestamp` predicate prunes NOTHING,
        # because the hive key and the timestamp data column are different
        # columns and polars can only prune on the former.
        data = data.filter(self._hive_window_predicate(start, end))
        # The TIMESTAMP predicate trims the window's exact edges. The hive
        # predicate alone over-includes every row of the two partially-covered
        # edge partitions.
        data = data.filter(
            pl.col("timestamp") >= pl.lit(start),
            pl.col("timestamp") <= pl.lit(end),
        )

        # Provenance asserted BEFORE dedup could collapse the evidence, then
        # dropped along with the hive key(s), so the downstream column set is
        # exactly what it was before this rework.
        data = self._assert_single_vendor_and_drop(data)
        # Drop only the PURELY DERIVED keys. `symbol` is also a real data
        # column that the tick layout happens to express as a path segment --
        # the writer drops it from the file because the segment carries it, so
        # dropping it here too would delete it outright.
        data = data.drop(
            [
                key
                for key in self._scanned_hive_keys
                if key in self.DERIVED_HIVE_KEYS
            ]
        )

        data = data.sort(by=["timestamp", "symbol"])
        if self.config.frequency == "tick":
            # NO DEDUP for tick (D-16). `dedup_raw_frame` exists so
            # `.to_xarray()` receives a unique `[timestamp, symbol]` MultiIndex
            # -- a dense-panel requirement. Tick has no xarray path this phase
            # (D-18: an irregular event axis cannot be expressed as a dense
            # panel), and many genuine quotes and trades legitimately share one
            # (timestamp, symbol): deduping them would silently discard exactly
            # the resolution this tier exists to capture.
            return data
        return dedup_raw_frame(data, keep="last")

    @staticmethod
    def _as_datetime(value) -> datetime:
        """Normalise a window edge to a naive `datetime`.

        An ISO date string resolves to that date at MIDNIGHT, exactly what
        the pre-refactor `pl.lit(...).str.to_datetime()` produced, so the
        inclusive `<=` end-edge semantics are unchanged. A `pd.Timestamp`
        coming from `TimeChunkPlanner.plan_from_timestamps()` is an OBSERVED
        timestamp and passes through with its time-of-day intact.
        """
        return pd.Timestamp(value).to_pydatetime()

    def _raw_axes_in_range(self) -> tuple[list[str], pd.DatetimeIndex]:
        """Both axes for the config's whole range, from one scan, WITHOUT
        densifying anything (D-02).

        Two single-column unique scans: polars projects each one on its own,
        so peak memory is one column of the raw frame rather than the dense
        `[timestamp, symbol]` grid. The symbol axis is `sorted()` for the
        same reason `base/constituent.py:_densify` sorts its all-time union
        -- it must match the coordinate order `to_xarray()` produces, which
        is the pandas MultiIndex level order.
        """
        scan = self._scan_raw()
        symbols = sorted(
            str(symbol)
            for symbol in scan.select("symbol")
            .unique()
            .collect()["symbol"]
            .to_list()
        )
        timestamps = (
            scan.select("timestamp").unique().collect()["timestamp"].to_list()
        )
        return symbols, pd.DatetimeIndex(sorted(timestamps))

    def _added_symbols_with_raw_history(
        self, added: list, start, end
    ) -> dict[str, int]:
        """The hive-pruned answer to the evidence question -- how many raw rows
        each of `added` carries in the CLOSED window `[start, end]`.

        Routed through `_scan_raw`, which ALREADY applies the `month=` hive
        predicate (directory pruning at plan time), the timestamp predicate
        (the window's exact edges), the vendor assertion and the dedup. Going
        through it rather than re-deriving those is the whole point: the
        pruning and the vendor isolation come along unchanged, and there is
        one place where a change to any of them lands.

        Counting is done in polars and only the counts are collected, so the
        dense `[timestamp, symbol]` grid the base default materialises is never
        built here. That is what makes the probe cheaper than the whole-range
        densify -- see `BaseDataset._added_symbols_with_raw_history` for the
        measured ORDERING (store-extent probe < whole-tier probe <
        `_raw_axes_in_range()`, which every chunked run already pays
        unconditionally) and for why that ordering rather than the seconds is
        the load-bearing claim.
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
        """Densify ONE window, reindexed onto the pinned symbol axis.

        A pinned symbol with no row in this window becomes an all-NaN column
        -- and the integer columns it touches upcast to float. That is not a
        chunking artefact: the whole-range densification already produces
        exactly this for an untraded cell, because
        `set_index([...]).to_xarray()` emits the full `[timestamp, symbol]`
        cartesian product with NaN in the gaps. The two paths therefore
        agree, which is what
        `tests/test_chunked_ingest.py::test_chunked_store_matches_the_unchunked_store`
        pins.
        """
        data = self._scan_raw(start_date, end_date)
        data = data.collect().to_pandas().set_index(["timestamp", "symbol"])
        data = data.to_xarray()
        if symbols is not None:
            data = data.reindex(symbol=list(symbols))
        return data

    def _raw_data_to_xr(self) -> xr.Dataset:
        with Timer(f" {self.__class__.__name__}: from pqt"):
            # The whole range, on no pinned axis -- byte-identical to the
            # pre-refactor body, which is why every existing assertion in
            # tests/test_stock_dataset.py stands untouched.
            return self._raw_data_to_xr_window(
                self.config.start_date, self.config.end_date, symbols=None
            )

    def _to_kunquant(
        self, data: xr.Dataset, data_columns: tuple
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        with Timer(f"{self.__class__.__name__}: to kunquant"):
            data = data.drop_vars(["open", "high", "low", "close", "volume"])
            data = data.rename(
                {
                    "adjOpen": "open",
                    "adjHigh": "high",
                    "adjLow": "low",
                    "adjClose": "close",
                    "adjVolume": "volume",
                }
            )
            data = data.sortby(["timestamp", "symbol"])
            # D-02: Tiingo supplies no native dollar-volume ("amount") column,
            # while KunQuant's Alpha101/Alpha158 AllData graphs derive vwap
            # from it. Synthesize the standard `volume * close` proxy here,
            # once and centrally, so every KunQuant factor class reading
            # US-equity data gets it without per-class duplication. The rename
            # above has already run, so `volume`/`close` are the ADJUSTED
            # series -- the proxy is adjusted dollar-volume, consistent with
            # the rest of the adjusted-price pipeline. Double-guarded: only
            # when the caller actually asks for `amount` and only when the
            # dataset does not already carry a real vendor column of that name.
            if "amount" in data_columns and "amount" not in data.data_vars:
                data = data.assign(amount=data["volume"] * data["close"])
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
        raise ValueError("Not finished")

    def _xr_to_bars(
        self, data: xr.Dataset, symbol: str, venue: str = "BINANCE"
    ):
        raise ValueError("Not finished")

    def _to_nautilus(
        self, data: xr.Dataset, venue: str = "BINANCE", n_jobs: int = 16
    ):
        raise ValueError("Not finished")

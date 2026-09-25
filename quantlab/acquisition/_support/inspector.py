"""Credential-free, read-only inspection of what a data source has on disk.

``SourceInspector`` answers questions about one source's raw parquet tier and
Zarr store from local files alone: which requested symbols are already
covered, which symbols failed in earlier runs, how many shards and watermark
sidecars exist, and a narrow row-level view of either tier. It imports no
vendor client and never constructs an ``Acquisition``, so it needs no API key
and issues no network request whatever is called in whatever order. Listing
the sources themselves is left to ``quantlab.registry``, whose import pulls in
every vendor module.

Nothing is cached and nothing is held on the instance. Every method builds
its own ledger or opens its own store and closes it before returning, so two
queries on one inspector cannot narrow each other. Refresh policy belongs to
the caller.
"""

import json
import os
from pathlib import Path
from typing import Sequence

import polars as pl
import xarray as xr

from quantlab.base.config import AcquisitionConfig, DatasetConfig
from quantlab.base.coverage import CoverageLedger, validate_symbols
from quantlab.dataset.stock import StockDataset
from quantlab.enums.data import RAW_HIVE_KEYS


class _RawTierReader(StockDataset):
    """A ``StockDataset`` that reads nothing when constructed.

    The base config setter calls ``_reset_symbols`` whenever ``symbols`` is
    assigned, which opens the Zarr store and falls back to materialising the
    whole panel from the raw tier when the store is absent. A browse only
    wants a lazy handle over the raw tier, so that seam is overridden to a
    no-op here. Module-private and built fresh per query, so ``browse_raw``
    can reuse ``_scan_raw`` (vendor-root check, tick data-type descent, pinned
    hive schema, strict column checks) without a second scan implementation.
    """

    def _reset_symbols(self) -> None:
        """Do nothing: the inspector never needs a resolved symbol axis."""
        return None


class SourceInspector:
    """Read-only questions about locally downloaded data.

    Every method takes the config that locates the data and answers from the
    file system. The inspector holds no state, so one instance can be kept
    and reused or built per call.

    Examples
    --------
    ``acq_cfg`` is an ``AcquisitionConfig`` and ``ds_cfg`` a
    ``DatasetConfig`` for the same source.

    >>> from quantlab.acquisition._support.inspector import SourceInspector
    >>> inspector = SourceInspector()
    >>> inspector
    SourceInspector()
    >>> inspector.coverage(acq_cfg)["pending"]
    1
    >>> inspector.inventory(acq_cfg)["raw"]["shards"]
    5
    """

    def __repr__(self) -> str:
        """Return ``SourceInspector()``; the instance carries no state."""
        return f"{self.__class__.__name__}()"

    # -- coverage -----------------------------------------------------------

    def coverage(
        self,
        config: AcquisitionConfig,
        symbols: Sequence[str] | None = None,
    ) -> dict:
        """Classify symbols against the config's date window from sidecars.

        This is the same computation ``Acquisition.coverage_report()`` runs,
        through the same ``CoverageLedger`` object, so the two never disagree.
        Symbols are validated before any path is built, because a symbol
        becomes a sidecar filename.

        Parameters
        ----------
        config : AcquisitionConfig
            Locates the watermark sidecars and supplies the window.
        symbols : Sequence[str] | None
            The symbols to classify; ``config.symbols`` by default.

        Returns
        -------
        dict
            A dict with ``requested``, ``pending`` (need a download),
            ``skipped`` (already satisfied), and the per-state counts
            ``covered``, ``widened``, ``legacy`` and ``no_data``.

        Examples
        --------
        >>> report = inspector.coverage(acq_cfg, ["AAPL", "MSFT"])
        >>> report["requested"], report["pending"], report["covered"]
        (2, 1, 1)
        """
        ledger = CoverageLedger.for_config(config)
        requested = ledger.validate_symbols(
            list(symbols if symbols is not None else config.symbols)
        )
        pending, counts = ledger.partition_by_coverage(
            requested, from_watermark=False
        )
        return {
            "requested": len(requested),
            "pending": len(pending),
            "skipped": len(requested) - len(pending),
            **counts,
        }

    # -- failures -----------------------------------------------------------

    def failures(self, config: AcquisitionConfig) -> dict[str, str]:
        """Return every symbol the failure manifest records, across runs.

        The manifest (``_failures.json`` beside the watermark sidecars) is the
        crash-durable record of download failures. It accumulates across
        runs, so it may name symbols the most recent run never requested.
        The reasons are already scrubbed of credentials. A missing or
        unreadable manifest returns ``{}``. Resume does not read the
        manifest; it is driven by sidecar presence alone.

        Returns
        -------
        dict[str, str]
            ``{symbol: reason}``, or ``{}``.

        Examples
        --------
        >>> inspector.failures(acq_cfg)
        {'ZZZZ': 'HTTP 404'}
        """
        return CoverageLedger.for_config(config).read_failure_manifest()

    # -- inventory ----------------------------------------------------------

    def inventory(
        self,
        config: AcquisitionConfig,
        dataset_config: DatasetConfig | None = None,
    ) -> dict:
        """Report what exists on disk, raw tier and Zarr tier separately.

        The two tiers are different artefacts: the raw tier is vendor-shaped
        and append-only, the Zarr store is the converted panel and may not
        exist yet. Reading the watermark sidecars is the expensive half (the
        directory walk is cheap), so every figure comes from one walk and one
        sidecar pass.

        Parameters
        ----------
        config : AcquisitionConfig
            Locates the raw tier and its sidecars.
        dataset_config : DatasetConfig | None
            Locates the Zarr store. When omitted, ``"zarr"``
            is ``None``, meaning "not asked" rather than "absent".

        Returns
        -------
        dict
            ``{"raw": {...}, "zarr": {...} | None}``. The raw dict carries the
            root, shard count, byte total, sidecar count, coverage span and
            failure count; the zarr dict carries the path, byte total, dims,
            variable names and timestamp span.

        Examples
        --------
        >>> report = inspector.inventory(acq_cfg, ds_cfg)
        >>> report["raw"]["shards"], report["raw"]["symbols_with_watermark"]
        (5, 2)
        >>> report["zarr"]["exists"]
        False
        >>> inspector.inventory(acq_cfg)["zarr"] is None
        True
        """
        ledger = CoverageLedger.for_config(config)
        return {
            "raw": self._raw_inventory(config, ledger),
            "zarr": (
                None
                if dataset_config is None
                else self._zarr_inventory(dataset_config)
            ),
        }

    @staticmethod
    def _raw_root(config: AcquisitionConfig, ledger: CoverageLedger) -> Path:
        """Return the directory this config's raw shards actually land under.

        Descends into ``data_type={quotes|trades}`` for a frequency whose hive
        layout partitions on it, so quotes and trades are never counted as
        one figure.
        """
        root = Path(config.raw_data_dir_path)
        if "data_type" in RAW_HIVE_KEYS[config.frequency]:
            root = root / f"data_type={ledger.data_type}"
        return root

    def _raw_inventory(
        self, config: AcquisitionConfig, ledger: CoverageLedger
    ) -> dict:
        """Compute the raw tier's figures from one walk and one sidecar pass."""
        root = self._raw_root(config, ledger)

        shards = 0
        total_bytes = 0
        if root.exists():
            # One walk with `os.walk`; repeated `rglob` calls would cost one
            # full traversal per figure.
            for dirpath, _dirnames, filenames in os.walk(root):
                for filename in filenames:
                    if not filename.endswith(".pqt"):
                        continue
                    shards += 1
                    try:
                        total_bytes += (Path(dirpath) / filename).stat().st_size
                    except OSError:
                        # A shard removed between listing and stat is simply
                        # gone, not an error for a footprint report.
                        continue

        # One sidecar pass; every figure below comes out of this loop.
        symbols_with_watermark = 0
        no_data = 0
        earliest_start: str | None = None
        latest_last_date: str | None = None
        for symbol in ledger.iter_watermark_symbols():
            coverage = ledger.read_coverage(symbol)
            if coverage is None:
                continue
            symbols_with_watermark += 1
            if coverage["no_data"]:
                no_data += 1
            start = coverage["start_date"]
            if start is not None and (
                earliest_start is None or start < earliest_start
            ):
                earliest_start = start
            last = coverage["last_date"]
            if last is not None and (
                latest_last_date is None or last > latest_last_date
            ):
                latest_last_date = last

        return {
            "vendor": config.vendor,
            "market": config.market,
            "frequency": config.frequency,
            "data_type": ledger.data_type,
            "root": str(root),
            "exists": root.exists(),
            "shards": shards,
            "bytes": total_bytes,
            "watermark_root": str(ledger.watermark_root),
            "symbols_with_watermark": symbols_with_watermark,
            # ISO date strings compare lexicographically, so no parsing.
            "coverage_start": earliest_start,
            "coverage_last_date": latest_last_date,
            "no_data": no_data,
            "failures": len(self.failures(config)),
        }

    @staticmethod
    def _zarr_inventory(dataset_config: DatasetConfig) -> dict:
        """Compute the Zarr tier's figures from metadata, then close the store.

        ``xr.open_zarr`` reads metadata only, so dims, variable names and the
        timestamp span cost no array reads. Only plain Python values are
        returned, never a live handle.
        """
        path = Path(dataset_config.zarr_file_path)
        if not path.exists():
            return {
                "path": str(path),
                "exists": False,
                "bytes": 0,
                "dims": {},
                "data_vars": [],
                "timestamp_start": None,
                "timestamp_end": None,
            }

        total_bytes = 0
        for dirpath, _dirnames, filenames in os.walk(path):
            for filename in filenames:
                try:
                    total_bytes += (Path(dirpath) / filename).stat().st_size
                except OSError:
                    continue

        dataset = xr.open_zarr(path)
        try:
            span: tuple[str | None, str | None] = (None, None)
            timestamps = (
                dataset["timestamp"] if "timestamp" in dataset.coords else None
            )
            if timestamps is not None and timestamps.size:
                values = timestamps.values
                span = (str(values[0]), str(values[-1]))
            return {
                "path": str(path),
                "exists": True,
                "bytes": total_bytes,
                "dims": {name: int(size) for name, size in dataset.sizes.items()},
                "data_vars": sorted(str(name) for name in dataset.data_vars),
                "timestamp_start": span[0],
                "timestamp_end": span[1],
            }
        finally:
            dataset.close()

    # -- row-level browsing -------------------------------------------------

    #: Message ``browse_raw`` and ``browse_zarr`` refuse an empty ``symbols``
    #: with. Both take the symbols and the date window as required arguments
    #: so the handle they return is already narrow; an empty list would be the
    #: same unbounded request from the value side, so it is refused rather
    #: than answered with zero rows.
    EMPTY_SYMBOLS_MESSAGE = (
        "symbols must be a NON-EMPTY sequence. D-11 makes symbols and the date "
        "window required arguments precisely so the lazy handle is already "
        "narrow when it is handed out (us_all is ~15.4k symbols x ~5.2k trading "
        "days, ~30M rows); an empty list is the same unbounded request wearing "
        "a different hat, so it is refused rather than answered with zero rows."
    )

    def _require_symbols(self, symbols: Sequence[str], caller: str) -> list[str]:
        """Return ``symbols`` as a list, raising ``ValueError`` if empty."""
        listed = list(symbols)
        if not listed:
            raise ValueError(f"{caller}: {self.EMPTY_SYMBOLS_MESSAGE}")
        return listed

    def browse_raw(
        self,
        dataset_config: DatasetConfig,
        symbols: Sequence[str],
        start_date: str,
        end_date: str,
    ) -> pl.LazyFrame:
        """Return a lazy, already-narrow view of the raw parquet tier.

        All four arguments are required so the handle is narrow when it is
        handed out; asking for a whole tier means passing the full roster and
        window deliberately. The frame is not collected here.

        The scan itself is ``StockDataset._scan_raw``, which checks that the
        root is a vendor directory, descends into the tick data-type
        directory, pins the hive schema and raises on unexpected or missing
        columns; this method only adds the symbol filter and a sort. Symbols
        are validated against the shared ticker pattern first. At daily and
        minute frequency ``symbol`` is a data column, so the symbol predicate
        prunes row groups while the date window prunes directories; at tick
        frequency ``symbol`` is a hive key and the same predicate prunes
        directories.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Locates the vendor's raw root.
        symbols : Sequence[str]
            Non-empty sequence of tickers to keep.
        start_date : str
            Inclusive ISO start of the window.
        end_date : str
            Inclusive ISO end of the window.

        Returns
        -------
        pl.LazyFrame
            A ``polars.LazyFrame`` sorted by ``(timestamp, symbol)``, so
            repeated collection yields the same row order.

        Raises
        ------
        ValueError
            If ``symbols`` is empty or contains an invalid ticker.

        Examples
        --------
        >>> frame = inspector.browse_raw(
        ...     ds_cfg, ["AAPL"], "2024-02-01", "2024-03-31"
        ... )
        >>> rows = frame.collect()
        >>> rows.height, rows["symbol"].unique().to_list()
        (2, ['AAPL'])
        """
        listed = self._require_symbols(symbols, "browse_raw")
        listed = validate_symbols(
            listed,
            owner_label=f"{self.__class__.__name__}.browse_raw",
            raw_root=dataset_config.raw_data_dir_path,
        )
        # A fresh reader per query, so a narrow query cannot narrow what a
        # later wide one sees.
        reader = _RawTierReader(dataset_config)
        return (
            reader._scan_raw(start_date, end_date)
            .filter(pl.col("symbol").is_in(listed))
            .sort(["timestamp", "symbol"])
        )

    def browse_zarr(
        self,
        dataset_config: DatasetConfig,
        symbols: Sequence[str],
        start_date: str,
        end_date: str,
    ) -> xr.Dataset:
        """Return a narrowed view of the Zarr store, lazy and dask-free.

        The store is opened fresh on every call and ``.sel`` is applied to a
        local, so nothing is narrowed permanently. A symbol the store does
        not carry raises rather than being reindexed to a NaN column, because
        a NaN column cannot be told apart from a genuinely empty history. On
        a store whose ``symbol`` axis is integer-valued (a CRSP panel keyed
        by PERMNO) the message also says so and points at the ticker sidecar
        beside the store, since a ticker can never match that axis.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Locates the Zarr store.
        symbols : Sequence[str]
            Non-empty sequence of symbol labels to select.
        start_date : str
            Inclusive ISO start of the window.
        end_date : str
            Inclusive ISO end of the window.

        Returns
        -------
        xr.Dataset
            The selected ``xarray.Dataset``, with the store's lazy arrays.

        Raises
        ------
        ValueError
            If ``symbols`` is empty, or if the store does not
            carry every requested symbol. The message names the store,
            the missing symbols and how many symbols the store carries.

        Examples
        --------
        >>> view = inspector.browse_zarr(
        ...     ds_cfg, ["AAPL"], "2024-01-10", "2024-01-12"
        ... )
        >>> dict(view.sizes)
        {'timestamp': 3, 'symbol': 1}
        """
        listed = self._require_symbols(symbols, "browse_zarr")
        # Not validated against the ticker pattern, unlike `browse_raw`: here a
        # symbol is a coordinate label matched against an index, never a path
        # segment or a query-string value, and the index either carries the
        # label or raises below.
        path = Path(dataset_config.zarr_file_path)
        dataset = xr.open_zarr(path)
        try:
            return dataset.sel(
                symbol=listed, timestamp=slice(start_date, end_date)
            )
        except KeyError as exc:
            carried = int(dataset.sizes.get("symbol", 0))
            known = set()
            axis_dtype = None
            if "symbol" in dataset.coords:
                known = {str(value) for value in dataset["symbol"].values}
                axis_dtype = dataset["symbol"].dtype
            missing = sorted(symbol for symbol in listed if symbol not in known)
            dataset.close()
            # An integer symbol axis is a CRSP panel keyed by PERMNO. Saying
            # only "does not carry AAPL" would be true but misleading: the
            # store may hold that security under the number the ticker
            # sidecar maps it to.
            axis_note = ""
            if axis_dtype is not None and axis_dtype.kind in "iu":
                # Imported locally: the CRSP package owns the constant and
                # drags the whole converter in with it, and this inspector
                # must stay importable for a vendor with no CRSP tier.
                from quantlab.dataset.crsp import TICKER_SIDECAR_SUFFIX

                axis_note = (
                    f" This store's symbol axis is INTEGER "
                    f"(dtype {axis_dtype}), i.e. CRSP PERMNOs, not tickers "
                    f"(D-01) -- a ticker can never match it, whether or not "
                    f"the security is present. The tickers live in "
                    f"'{path.name}{TICKER_SIDECAR_SUFFIX}' beside the store "
                    f"and are queried as-of through "
                    f"quantlab/dataset/crsp/tickers.py:CrspTickerLookup; ask "
                    f"for the PERMNO it gives you."
                )
            raise ValueError(
                f"{self.__class__.__name__}.browse_zarr: the Zarr store at "
                f"{str(path)!r} does not carry {missing} (requested "
                f"{sorted(listed)}). The store carries {carried} symbol(s)."
                f"{axis_note} "
                f"This is refused rather than reindexed on purpose: a "
                f"NaN-filled column for a symbol the store has never heard of "
                f"is indistinguishable from a symbol with a genuinely empty "
                f"history, and an operator cannot tell those two apart from "
                f"the data. Convert the raw tier for these symbols, or ask for "
                f"the ones the store has."
            ) from exc
        except BaseException:
            dataset.close()
            raise

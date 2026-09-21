"""`SourceInspector` -- the credential-free, read-only view of what is already
on disk.

Built by 03.4-04 for SC-3 and SC-4, and consumed IN-PROCESS by the out-of-repo
operator console (D-12).

**The hard constraint, and the evidence for it (D-08).** This surface must run
on a machine with NO credentials, because everything it answers is a local file
read. Constructing an `Acquisition` is not an option:
`TiingoAcquisition.__init__` raises `RuntimeError` the moment `TIINGO_API_KEY`
is unset -- before it could possibly know that the caller only wanted to count
sidecars. That is why `ingest_us_equity.py` USED TO print
``coverage report: skipped (export TIINGO_API_KEY to see it)`` for a
computation that is nothing but `open()` and `json.load()`. That branch is gone
as of 03.4-06: `_print_coverage` is unconditional and routes through this class,
and `tests/test_ingest_shells.py::test_the_dry_run_needs_no_credential` is what
keeps the skip line history. Extending
`Acquisition` and deferring its credential check was considered and rejected in
CONTEXT: it would turn a fail-fast safety behaviour into a fail-late one.

**It issues ZERO vendor requests BY CONSTRUCTION, not by call order.** This
module imports no vendor client and binds no `Acquisition` subclass, so there
is nothing here that COULD open a socket, whatever anyone calls in what order.
Three independent arms prove it in `tests/test_source_inspector.py`, copying
`tests/test_volume_guard.py`'s structure: every socket allocation raises, every
credential is deleted from the environment, and an AST resolver asserts this
file's (and `quantlab/base/coverage.py`'s) import set is disjoint from the
acquisition modules.

**What it deliberately does NOT do: enumerate sources.** That is
`DataSourceRegistry.all()`'s job and it stays there. Importing the registry
here would drag in both vendor modules -- the registry's own bottom imports --
and turn a structural guarantee into a conventional one ("we happen not to call
the client"). The console asks the registry what sources exist and asks the
inspector what is on disk for one of them.

**Import cost, stated rather than discovered.** This module reaches the raw
tier through `quantlab/dataset/stock.py`, which transitively imports
`nautilus_trader` (~1.7 s cold). That is an import cost, not a fragility, and it
is a strictly LIGHTER path than `quantlab/acquisition/registry.py`, which pays
the same cost through its config factories AND both vendor SDKs on top.
Reimplementing the raw scan here to avoid it would trade a measured second for
the four separately-measured bugs `_scan_raw` already fixes -- see `browse_raw`.

**No caching, deliberately.** Reading 7,756 watermark sidecars was measured at
~1.65 s on this machine, while a full directory traversal of the 26,584-file
raw root costs ~0.05 s warm -- so the sidecar pass is the expensive half, the
opposite of what one would guess. quantlab caches none of it: refresh policy
belongs to the console, which knows whether the operator just pressed a key or
just finished a backfill. What this module DOES guarantee is that each figure
is computed from one traversal and one sidecar pass, never one per field.

**Statelessness is a contract, not an accident.** Every method builds its own
reader and its own ledger. Nothing is held on the instance, so two queries
issued back to back on one inspector cannot narrow each other --
`XrBackend.filter_by_date` / `filter_by_symbol` assign back to `self.data`, and
a shared backend would narrow permanently (the bug quick task 260906-w3t fixed
for the Polars factor probe).
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
    """A `StockDataset` that performs NO store read at construction.

    `BaseDataset.__init__` assigns the config, and the config setter calls
    `_reset_symbols()` whenever `symbols` is set -- which reaches the Zarr store
    through `read()` and falls back to materialising the WHOLE panel via
    `from_raw_data()` when the store is absent or empty. For a browse that only
    wants a lazy handle over the raw tier, that is a full conversion performed
    before the caller has asked for a single row: exactly the
    narrow-in-place / fallback-on-construction hazard quick task 260906-w3t
    fixed for the Polars factor probe.

    `_reset_symbols` is documented on `BaseDataset` as an OVERRIDABLE SEAM for
    precisely this case, and `IndexConstituentDataset` already overrides it to a
    no-op for the same reason. Overriding it is therefore the established
    in-repo way to build a dataset object that reads nothing on construction --
    not a workaround.

    Module-private, and constructed FRESH per query: it exists so `browse_raw`
    can reuse `_scan_raw`'s vendor-root assertion, tick `_scan_root` descent,
    pinned `hive_schema` and RAISING `extra_columns` / `missing_columns`
    defaults. It is not a dataset anyone should persist through.
    """

    def _reset_symbols(self) -> None:
        """No-op: the inspector never writes and never needs a resolved symbol
        axis at construction time. See the class docstring.
        """
        return None


class SourceInspector:
    """Read-only questions about locally downloaded data, answered with no
    credentials and zero vendor requests.

    Holds no state. Construct one and keep it, or construct one per call --
    both are correct, and that is the point (see the module docstring's
    statelessness contract).
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    # -- coverage (D-09: the same code the real run uses) -------------------

    def coverage(
        self,
        config: AcquisitionConfig,
        symbols: Sequence[str] | None = None,
    ) -> dict:
        """Classify `symbols` (default: `config.symbols`) against
        `[config.start_date, config.end_date]`, returning the same shape
        `Acquisition.coverage_report()` returns.

        **This is `coverage_report()`'s body reading from the same shared
        object** -- `CoverageLedger.partition_by_coverage` -- rather than a
        second implementation (D-09). The four-state rule plus the orthogonal
        `no_data` count is subtle enough that a "simple" reimplementation gets
        `legacy` wrong, and two answers that agree today are how an operator
        ends up trusting the wrong one. A test asserts the sharing by IDENTITY
        and by mutation, not by equality of results.

        Symbols are validated FIRST, before any path is built, exactly as
        `Acquisition._run` and `coverage_report` do. A symbol becomes a
        watermark filename, so `"../../etc/hosts"` reaching `watermark_path`
        would escape the sidecar root (T-03.4-04-01); the check is the shared
        `TRADEABLE_TICKER_PATTERN`, not a local copy -- a local copy already
        caused one production incident (260907-10t).
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
        """Every symbol `_failures.json` records as failing, accumulated across runs.

        Returned as `{symbol: reason}`, or `{}`. Because the manifest
        accumulates across runs, the console can be shown a symbol the most
        recent run never requested: `Acquisition._merge_unattempted_failures`
        folds the on-disk entries a run had no news about forward before every
        overwrite. Pinned from the console side by
        `tests/test_acquisition_progress.py::test_the_manifest_survives_a_quota_abort_on_the_default_path`,
        which reads a symbol back through this method after a run that never
        asked the vendor for it. This summary line is kept verbatim in sync
        with `CoverageLedger.read_failure_manifest`; the two return the same
        value, and a drifting pair of summaries is where the next divergence
        starts.

        **This method is the manifest's FIRST in-repo reader.** What L-2
        established (2026-09-08) is narrower than it was later summarised as:
        the manifest is not RESUME input. Resume is driven entirely by
        watermark-sidecar PRESENCE, so an earlier claim that the manifest feeds
        resume was wrong about the code. The manifest is kept anyway, and
        deliberately (D-18): it is the crash-durable operator record -- a
        process that dies returns no `AcquisitionResult` -- and the console
        needs the reasons, which is what this reads.

        Absent file means `{}`, not an error: a source that has never failed
        and a source that has never run look the same from here, and both
        answers are "nothing to report". A corrupt file is tolerated the same
        way `CoverageLedger.read_sidecar` tolerates a corrupt sidecar, for the
        same reason -- one failure policy for unreadable JSON, not two that can
        drift.

        The values are already scrubbed: they are the strings
        `Acquisition._attempt_batch` produced through `_scrub`, which is what
        makes the manifest safe to paste into an issue. This method adds no new
        egress path for raw vendor exception text.

        Delegates to `CoverageLedger.read_failure_manifest` (03.4-05): the
        pre-write merge in `Acquisition._run` -- which since 03.4-08 runs on
        every exit path of the resume loop, not only the cancel one -- also
        reads the manifest, and two tolerant readers is two copies of the
        failure policy, free to drift.
        """
        return CoverageLedger.for_config(config).read_failure_manifest()

    # -- inventory ----------------------------------------------------------

    def inventory(
        self,
        config: AcquisitionConfig,
        dataset_config: DatasetConfig | None = None,
    ) -> dict:
        """What exists on disk for this source, raw tier and Zarr tier
        reported SEPARATELY.

        Two sub-results rather than one merged number because they are
        different artefacts with different lifecycles: the raw tier is
        vendor-shaped, append-only and written by acquisition; the Zarr store
        is the canonical panel, rewritten by conversion, and may legitimately
        not exist at all (`ingest_us_equity.py` converts only under
        `--to-zarr`). `zarr` is `None` when no `dataset_config` is supplied --
        "not asked" rather than "not there".

        Cost, measured (RESEARCH Pitfall 5): the directory traversal is the
        CHEAP half (~0.05 s warm over 26,584 files) and the sidecar pass is the
        expensive one (~1.65 s over 7,756 sidecars). So the shard count and the
        byte total come from ONE walk, and every sidecar figure comes from ONE
        pass over `iter_watermark_symbols()`. Never one traversal per field,
        and never `read_coverage` per symbol inside a loop a caller runs per
        symbol. Nothing is cached here; refresh policy is the console's.
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
        """The raw directory this config's data actually lands under.

        Descends into `data_type={quotes|trades}` for a frequency that
        partitions on it, mirroring `StockDataset._scan_root` and
        `CoverageLedger.watermark_root`. Reporting the vendor root instead
        would conflate quotes and trades into one figure -- the same
        namespacing mistake that once let a completed quotes backfill tell a
        trades run every symbol was covered, here in its harmless
        disk-footprint costume.
        """
        root = Path(config.raw_data_dir_path)
        if "data_type" in RAW_HIVE_KEYS[config.frequency]:
            root = root / f"data_type={ledger.data_type}"
        return root

    def _raw_inventory(
        self, config: AcquisitionConfig, ledger: CoverageLedger
    ) -> dict:
        """The raw tier's figures: ONE directory walk, ONE sidecar pass."""
        root = self._raw_root(config, ledger)

        shards = 0
        total_bytes = 0
        if root.exists():
            # ONE walk. `os.walk` rather than repeated `rglob` calls, because
            # each `rglob` is its own full traversal and the fields below would
            # otherwise cost one traversal each.
            for dirpath, _dirnames, filenames in os.walk(root):
                for filename in filenames:
                    if not filename.endswith(".pqt"):
                        continue
                    shards += 1
                    try:
                        total_bytes += (Path(dirpath) / filename).stat().st_size
                    except OSError:
                        # A shard removed between listing and stat is not an
                        # error for a footprint report; it is just gone.
                        continue

        # ONE sidecar pass. Every figure below comes out of this single loop.
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
            # ISO strings compare lexicographically, which is why no date
            # parsing happens here -- the same reason `classify_coverage`
            # compares them as strings.
            "coverage_start": earliest_start,
            "coverage_last_date": latest_last_date,
            "no_data": no_data,
            "failures": len(self.failures(config)),
        }

    @staticmethod
    def _zarr_inventory(dataset_config: DatasetConfig) -> dict:
        """The Zarr tier's figures, opened lazily and closed again.

        `xr.open_zarr` reads metadata only (~0.2 s on the measured store), so
        `dims`, the variable names and the timestamp span cost no array reads.
        The store is CLOSED before returning: this method hands back plain
        Python values, never a live handle, so nothing the caller does later
        can narrow a dataset a subsequent call would reuse.
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

    # -- row-level browsing (D-10 lazy, D-11 narrow by construction) --------

    #: What `browse_raw` / `browse_zarr` refuse an empty `symbols` with.
    #:
    #: D-11 makes `symbols` and the window REQUIRED rather than optional, which
    #: closes the "ask for a whole tier" footgun from the argument side. An
    #: EMPTY sequence reopens it from the value side: `is_in([])` and
    #: `.sel(symbol=[])` are both perfectly valid and both mean "no rows", but a
    #: caller who arrived there by passing an unfiltered roster that happened to
    #: come back empty gets a silent nothing instead of a question.
    EMPTY_SYMBOLS_MESSAGE = (
        "symbols must be a NON-EMPTY sequence. D-11 makes symbols and the date "
        "window required arguments precisely so the lazy handle is already "
        "narrow when it is handed out (us_all is ~15.4k symbols x ~5.2k trading "
        "days, ~30M rows); an empty list is the same unbounded request wearing "
        "a different hat, so it is refused rather than answered with zero rows."
    )

    def _require_symbols(self, symbols: Sequence[str], caller: str) -> list[str]:
        """Normalise `symbols` to a non-empty list, refusing an empty one."""
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
        """A LAZY, already-narrow view of the raw parquet tier. The caller
        collects (D-10).

        All four arguments are positional-REQUIRED with no defaults, so
        omitting any one of them is a `TypeError` at the call site (D-11). That
        is the other side of the developer's D-10 override: a lazy object
        technically permits asking for a whole tier, and what closes that is the
        arguments, not the return type. Asking for everything therefore requires
        deliberately passing the full roster and the full window.

        **It opens no parquet scan of its own, and that is load-bearing.** (Stated
        without naming the polars call, because the acceptance check for this
        rule is a literal scan of this file for that call's name -- the same
        false positive 03.4-01 and 03.4-03 each had to undo in a docstring.)
        `StockDataset._scan_raw` already carries four separately-measured
        fixes -- the vendor-root basename assertion (two vendors under one root
        merge with NO error and no provenance), the tick `_scan_root` descent
        (a root holding both data types fixes its schema from the
        alphabetically-first file and then raises on the other), the explicitly
        pinned `hive_schema` (an inferred numeric-looking key changes dtype and
        a string comparison then matches nothing), and `extra_columns` /
        `missing_columns` left at their RAISING defaults. A second scan written
        here is precisely where `extra_columns="ignore"` gets added "to make it
        work", reopening the silent cross-vendor merge while looking like a bug
        fix. So this method calls `_scan_raw` and appends to what it returns.

        Symbols are validated through the shared `validate_symbols` -- the same
        compiled `TRADEABLE_TICKER_PATTERN` object bound at both ends of the
        symbol lifecycle, never a local copy. A local copy is what caused
        incident 260907-10t.

        **The symbol predicate is asymmetric across frequencies, and nobody
        should optimise that away.** At `1d` and `1m`, `symbol` is a data
        column, so `is_in` prunes only ROW GROUPS via parquet statistics. At
        `tick`, `symbol` IS a hive key, so the very same predicate prunes
        DIRECTORIES. The window predicate is what prunes directories in the
        first two cases, and `_scan_raw` applies it.

        The result is sorted by `(timestamp, symbol)` so repeated collection of
        one window yields an identical row order -- `_scan_raw` sorts too, but
        the symbol filter is applied after it, and a filter is not obliged to
        preserve order.
        """
        listed = self._require_symbols(symbols, "browse_raw")
        listed = validate_symbols(
            listed,
            owner_label=f"{self.__class__.__name__}.browse_raw",
            raw_root=dataset_config.raw_data_dir_path,
        )
        # FRESH reader per query. Nothing is held on the inspector, so a narrow
        # query cannot narrow what a later wide one sees.
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
        """A narrowed view of the Zarr tier, lazy and dask-free.

        Same required-argument rule as `browse_raw`, for the same D-11 reason.

        **A separate name rather than one shared browse method**, because the
        two return different types (`pl.LazyFrame` vs `xr.Dataset`). One name
        with two return types would be worse than two clearly-named methods:
        the caller has to branch on the tier anyway, and a shared name hides
        that it must.

        **The store is opened FRESH per call and never held.**
        `XrBackend.filter_by_date` / `filter_by_symbol` assign back to
        `self.data`, so an inspector that kept one backend would narrow
        permanently and a wide query issued after a narrow one would silently
        return the narrow result -- the bug quick task 260906-w3t fixed for the
        Polars factor probe. `.sel(...)` is applied to a LOCAL and the result
        is returned; nothing is written back anywhere.

        **An unknown symbol RAISES, and is deliberately not reindexed.**
        `.sel(symbol=[...])` raises `KeyError` when the store does not carry a
        requested symbol. `reindex` would instead return a NaN-filled column,
        and a NaN column is INDISTINGUISHABLE from a genuinely empty history --
        an operator would read "this ticker has no data" when the truth is
        "this store has never heard of this ticker". The refusal is re-raised
        as a `ValueError` naming the store, the requested symbols and how many
        symbols the store carries, because that is this repo's house style for
        a legible refusal (`_scan_raw`, `_assert_vendor_root` and
        `validate_symbols` all raise `ValueError`) and because `KeyError`'s
        `str()` reprs its argument, mangling a multi-line operator message. The
        original `KeyError` is kept in the exception chain.

        **The refusal says WHICH KIND of label the store's axis holds**
        (03.11-09). The behaviour above is unchanged, but its message was not
        enough on a CRSP store: that axis is the int64 PERMNO (D-01), so a
        perfectly legitimate ticker arrives as a string the index cannot match
        and the operator is told "the store does not carry AAPL" -- true,
        correctly refused, and pointing at the wrong conclusion. The message
        therefore names the axis's dtype, and where the tickers went:
        `{zarr}.crsp_tickers.json`, queried as-of through
        `quantlab/dataset/crsp_tickers.py:CrspTickerLookup`.
        """
        listed = self._require_symbols(symbols, "browse_zarr")
        # NOT validated against TRADEABLE_TICKER_PATTERN, unlike `browse_raw`.
        # Here a symbol is a coordinate LABEL matched against an index, never a
        # path segment and never a query-string value, so neither trust
        # boundary `validate_symbols` guards is crossed -- and the index either
        # carries the label or raises below, which is a stricter check than the
        # pattern would be.
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
            # An INTEGER symbol axis is the CRSP panel's PERMNO axis (D-01).
            # Saying only "does not carry AAPL" there is true and misleading:
            # the store may well hold that security, under the number the
            # sidecar beside it maps the ticker to.
            axis_note = ""
            if axis_dtype is not None and axis_dtype.kind in "iu":
                # Local: `crsp.py` owns the constant and drags the whole
                # converter in with it, and this inspector must stay importable
                # for a vendor that has no CRSP tier at all.
                from quantlab.dataset.crsp import TICKER_SIDECAR_SUFFIX

                axis_note = (
                    f" This store's symbol axis is INTEGER "
                    f"(dtype {axis_dtype}), i.e. CRSP PERMNOs, not tickers "
                    f"(D-01) -- a ticker can never match it, whether or not "
                    f"the security is present. The tickers live in "
                    f"'{path.name}{TICKER_SIDECAR_SUFFIX}' beside the store "
                    f"and are queried as-of through "
                    f"quantlab/dataset/crsp_tickers.py:CrspTickerLookup; ask "
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

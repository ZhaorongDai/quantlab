"""`SourceInspector` -- the credential-free, read-only view of what is already
on disk.

Built by 03.4-04 for SC-3 and SC-4, and consumed IN-PROCESS by the out-of-repo
operator console (D-12).

**The hard constraint, and the evidence for it (D-08).** This surface must run
on a machine with NO credentials, because everything it answers is a local file
read. Constructing an `Acquisition` is not an option:
`TiingoAcquisition.__init__` raises `RuntimeError` the moment `TIINGO_API_KEY`
is unset -- before it could possibly know that the caller only wanted to count
sidecars. That is why `ingest_us_equity.py` today prints
``coverage report: skipped (export TIINGO_API_KEY to see it)`` for a
computation that is nothing but `open()` and `json.load()`. Extending
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

import xarray as xr

from quantlab.base.config import AcquisitionConfig, DatasetConfig
from quantlab.base.coverage import CoverageLedger
from quantlab.enums.data import RAW_HIVE_KEYS


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
        """The last run's `_failures.json` as `{symbol: reason}`, or `{}`.

        **This method is the manifest's FIRST in-repo reader.** L-2 established
        that nothing under `quantlab/` reads this file: resume is driven
        entirely by watermark-sidecar PRESENCE, so an earlier claim that the
        manifest is resume input was wrong about the code. It is kept anyway,
        and deliberately (D-18): it is the crash-durable operator record -- a
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
        """
        path = CoverageLedger.for_config(config).failure_manifest_path
        if not path.exists():
            return {}
        try:
            with open(path) as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return {str(symbol): str(reason) for symbol, reason in payload.items()}

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

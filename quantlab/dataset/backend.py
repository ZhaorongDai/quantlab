import os
import shutil
from pathlib import Path
from typing import Mapping, Optional, Self, Sequence

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from loguru import logger

from quantlab.base.backend import DataBackend


class XrBackend(DataBackend):
    def __init__(self) -> None:
        super().__init__()

    def read(self, path: str, overwrite: bool = False, **kwargs) -> Self:
        if not overwrite and hasattr(self, "data"):
            return self

        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        self.data = xr.open_dataset(path, **kwargs)
        return self

    #: Chunk length pinned along the append dimension by the FIRST write of
    #: an appended store. Without an explicit `encoding`, Zarr adopts the
    #: first window's own length as the chunk size, and every later append of
    #: a different length (a short trading year, a partial final month) is
    #: then misaligned with the on-disk chunk grid. A fixed value makes the
    #: layout a property of the store rather than of whichever window
    #: happened to be written first.
    APPEND_DIM_CHUNK = 512

    #: Ceiling on the bytes a symbol-axis widen may materialise AT ONCE,
    #: enforced by `widen_symbol_axis`'s router rather than by a refusal.
    #:
    #: **Derived from the target machine's measured materialisation ceiling
    #: (2026-09-06), and deliberately a constant of this module's own.** 4 GiB
    #: sits below the ~7.2 GiB that OOMs a 16 GiB box and above every window
    #: that comfortably fits. It is stated here rather than imported because
    #: this module has no import path to the acquisition layer and must not
    #: grow one: a storage backend that imports `UniverseCatalog` to read a
    #: number has acquired a dependency on the whole acquisition stack for a
    #: scalar. That is the sibling-constant precedent `MAX_RAW_BYTES` already
    #: sets one file over, where the same measurement is cited for a DISK
    #: ceiling rather than a RAM one.
    #:
    #: **This routes; it does not refuse.** The acquisition layer's RAM guard,
    #: which failed a fetch that would not fit, was deleted by decision in
    #: phase 03.6 (SC-3); this budget never refused in the first place --
    #: crossing it selects a bounded block-by-block rewrite. Refusing is not
    #: available here:
    #: `Factor.update()` reaches `widen_symbol_axis` as its ONLY path -- there
    #: is no raw tier for it to re-read, so `BaseDataset`'s
    #: `on_new_listing="rebuild"` escape does not exist on the factor side.
    #:
    #: Routing of the brief's measured scenarios at this value (2026-09-08):
    #: 0.2 / 34.6 / 137.3 MiB stores and a daily 7,700-symbol year of 20
    #: variables (0.3 GiB) take the whole-store rewrite; daily full history
    #: (6.0 GiB), 1-minute 500 symbols (7.3 GiB) and 1-minute 3,000 symbols
    #: (43.9 GiB) take the chunked one.
    MAX_WIDEN_BYTES = 4 * 1024**3

    def write(self, path: str, **kwargs) -> Self:
        if not Path(path).exists():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        kwargs.setdefault("mode", "w")
        self.data.to_zarr(path, **kwargs)
        return self

    def append(
        self,
        path: str,
        append_dim: str = "timestamp",
        *,
        append_dim_size: Optional[int] = None,
        **kwargs,
    ) -> Self:
        """Create the store, or extend it along `append_dim`.

        The storage-medium half of the chunked ingestion path: "how the
        in-memory object reaches this medium INCREMENTALLY" is the same
        concern, at the same altitude, as `write()`'s "how it reaches this
        medium". Nothing here reads a config -- the backend has no
        `raw_data_dir_path`, no `start_date` and no knowledge of where its
        data came from, and giving it any of those would break the
        medium-agnostic contract that lets `PlBackend` exist.

        **Contract, enforced rather than merely documented.** Every non-append
        dimension and its coordinate values must match the store EXACTLY
        across calls, every shared data variable must keep its dtype, the
        incoming window must begin STRICTLY AFTER the stored end of
        `append_dim`, and the incoming panel's SET of data variables must
        match the store's exactly. Raw
        `to_zarr(mode="a", append_dim=...)` enforces none of
        the four: a mismatched symbol coordinate is silently OVERWRITTEN with
        the new window's labels, leaving previously-written rows attributed to
        the wrong symbols; an appended float64 NaN written into an int64
        variable is silently cast to 0 -- a fabricated observation where data
        was missing; and an OVERLAPPING window is simply concatenated on,
        leaving the append dimension no longer strictly increasing (measured
        2026-09-07: a store on 2022-01-04..2022-01-06 taking a
        2022-01-05..2022-01-07 window comes back holding
        `[01-04, 01-05, 01-06, 01-05, 01-06, 01-07]`, and the failure surfaces
        later and elsewhere as a `.sel()` KeyError on a non-monotonic index or
        a `to_xarray` refusal on a non-unique one); and a panel whose SET of
        data variables differs from the store's is written variable by
        variable, leaving the store's variables at DIFFERENT lengths along
        `append_dim`. The first three corruptions are invisible afterwards
        from the store alone, which is why they are checked here, before the
        irreversible append. The fourth is worse than invisible: the store
        cannot be OPENED afterwards at all.

        **The data-variable set is an axis too**, and the one whose corruption
        is total. Zarr extends exactly the variables it is handed, so any
        mismatch in either direction leaves ragged lengths and `xr.open_zarr`
        then refuses the whole store with `conflicting sizes for dimension
        'timestamp'` -- measured 2026-09-07 in all three shapes: a variable
        added, a variable dropped, and the two sets disjoint. The set must
        therefore MATCH, and the two directions are not symmetric. A panel
        that legitimately GREW a variable has an explicit opt-in:
        `widen_data_vars()` materialises it over the store's existing extent
        with NaN over history, which is the same superset-and-backfill rule
        the symbol axis already follows, and `widen_and_append()` applies it
        as part of reconciling every axis. A panel MISSING a stored variable
        has no such route, because the only way to fill it would be NaN over
        the INCOMING window -- punching holes into recent dates of a variable
        that was complete, after which nothing distinguishes those holes from
        data the vendor never had. That direction destroys history which was
        valid before the call, so it is refused outright; recompute the window
        over the store's full variable set, or replace the store.

        A GAP is NOT an error. A window starting strictly after the stored end
        appends normally whatever the distance: a discontinuous axis is a
        legitimate shape this layer takes no position on, and only OVERLAP is
        refused. The refusal is unconditional and carries no opt-out -- to
        recompute a range the store already holds, replace the store with
        `save(mode="w")`; `append()` exists to extend it.

        `from_raw_data_chunked()` satisfies the coordinate half by pinning the
        symbol axis once over the whole range (D-02); this check is what turns
        that guarantee into an assertion.

        **The creating write decides a FIFTH thing, and it decides it
        permanently: the on-disk chunk grid.** Zarr fixes it at store creation
        and append cannot revise it, so a caller who already knows the store's
        eventual extent along `append_dim` says so with `append_dim_size` and
        gets `min(APPEND_DIM_CHUNK, that extent)` -- the grid a single
        whole-range write would have left -- while a caller who does not know
        it gets the panel-in-hand default. An incremental writer that stays
        silent hands Zarr its FIRST WINDOW's length as the store's permanent
        chunk, which is how `--chunk day` came to leave `(1, 3)` where the
        unchunked path left `(9, 3)` (measured 2026-09-12, fixed in 03.6). The
        parameter is keyword-ONLY and consumed explicitly rather than left
        riding in `**kwargs`, because both branches forward `**kwargs`
        verbatim to `to_zarr`, which would reject an unknown argument.

        Against an EXISTING store the value is accepted and ignored: the grid
        was pinned irreversibly by the creating write and there is nothing left
        to decide, so a caller may pass it unconditionally on every window --
        which is exactly what `from_raw_data_chunked` does, because store
        EXISTENCE is the real condition and it is owned here. A loop-index
        guard at the call site would be wrong on the two paths where the
        creating write is not iteration zero: a resume skips already-recorded
        windows, and `on_new_listing="rebuild"` moves the store aside so a
        later call creates it.
        """
        target = Path(path)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            kwargs.setdefault(
                "encoding",
                self._append_encoding(
                    append_dim, append_dim_size=append_dim_size
                ),
            )
            self.data.to_zarr(path, mode="w", **kwargs)
            return self

        self._assert_append_compatible(path, append_dim)
        # No `encoding` on an append -- xarray rejects it outright, and the
        # chunk grid was already pinned by the creating write above.
        kwargs.pop("encoding", None)
        self.data.to_zarr(path, mode="a", append_dim=append_dim, **kwargs)
        return self

    #: Sidecar suffixes used by `widen_symbol_axis`'s atomic directory swap.
    #: `.widening.tmp` holds the rewritten store BEFORE it is authoritative and
    #: is never read by anything; `.superseded.tmp` briefly holds the ORIGINAL
    #: store between the two renames and is the one artefact a crash can leave
    #: behind that still contains real data.
    WIDENING_SUFFIX = ".widening.tmp"
    SUPERSEDED_SUFFIX = ".superseded.tmp"

    def widen_symbol_axis(
        self,
        path: str,
        symbols: Sequence[str],
        dim: str = "symbol",
        append_dim: str = "timestamp",
        fill_values: Optional[Mapping[str, object]] = None,
    ) -> Self:
        """Rewrite the store at `path` onto a SUPERSET `dim` axis, crash-safely.

        The storage-medium answer to "the roster grew between two runs". Every
        pre-existing label keeps its history bit-identical; a newly added label
        gets the fill value across the whole historical block (NaN for a float
        variable, whatever `fill_values` names otherwise). The append dimension
        is not touched and neither is `self.data` -- this operates purely on the
        store, so a caller can widen without holding a panel at all.

        **Why this is a separate method rather than a flag on `append()`.**
        `_assert_append_compatible` refuses a changed `dim` coordinate for a
        reason that has not gone away: raw `to_zarr(mode="a", append_dim=...)`
        silently OVERWRITES the stored labels, so a window on `{A, ARM}` written
        into a store on `{A, XYZ}` leaves XYZ's rows attributed to ARM with
        nothing raised (measured 2026-09-06: `rows [1.0, 3.0] were written for
        XYZ but are now labelled: ARM`). Widening is an explicit, separately
        named opt-in that makes the two axes AGREE; it does not loosen the
        refusal a caller who did not opt in still gets.

        **Guards fire before any write**, in this order:

        1. A `.superseded.tmp` sidecar with NO store at `path` means a previous
           run crashed between the two renames. Refuse and name the manual move
           -- deciding which directory is authoritative is not this method's
           call to make.
        2. `symbols` must be a SUPERSET of the stored labels. `reindex` drops
           what it is not asked for, and a store missing a delisted symbol's
           history is indistinguishable afterwards from one that never held it.
        3. Every data variable carrying `dim` must be floating-point, or be
           named in `fill_values`. Measured: an unfilled `reindex` upcasts a
           bool `anomaly_flag` and an int64 `volume` to float64-with-NaN -- a
           silent schema change to a LIVE store, the same family of invisible
           corruption `_assert_append_compatible` refuses on the append path.
           An explicit `fill_values` entry preserves the stored dtype exactly,
           so no `.astype()` restoration is done or needed.

        **The swap** is rename-aside -> rename-in -> rmtree, mirroring
        `ChunkLedger._flush`'s `os.replace` idiom one level up at directory
        granularity. Both renames are same-parent and therefore atomic. The
        rename-aside comes FIRST on purpose: a crash between the two leaves NO
        store at `path`, so the next read fails loudly instead of treating a
        half-widened store as authoritative.

        **The strategy is chosen BY SIZE, and reported** (260908-g30). dask is
        not installed here, so `xr.open_zarr` yields lazily-indexed arrays that
        `.load()` materialises in full. `_estimate_widen_bytes` sizes the
        WIDENED panel before anything is written, and the result routes:

        - at or under `MAX_WIDEN_BYTES`, `_widen_whole_store` reindexes the
          whole store and writes it once -- the shipped path, unchanged, and
          logged at `info`;
        - over it, `_widen_chunked` rewrites the store block by block along
          `append_dim`, holding ONE block, and logs a `warning` naming every
          figure it decided on.

        Both leave the same store: values element for element, both
        coordinates, the on-disk chunk grid and the `symbol` coordinate's
        on-disk encoding, measured across both live production encodings and
        locked by `tests/test_symbol_axis_widening.py`.

        **Why a router rather than always chunking.** Measured 2026-09-08, the
        chunked rewrite runs 1.2x / 4.0x / 3.6x the whole-store wall clock on
        0.2 / 34.6 / 137.3 MiB stores. Where both fit, whole-store wins; above
        the budget, whole-store does not run at all. Chunking unconditionally
        would tax every routine widen ~4x to buy nothing.

        **`on_new_listing="rebuild"` is still worth reaching for, for a
        DIFFERENT reason than it used to be.** Not memory -- that argument is
        gone, and it never applied to `Factor` anyway, which has no raw tier
        and so no `rebuild` at all. The surviving reason is DATA: `rebuild`
        re-reads raw and recovers a new listing's REAL history, where a widen
        of either strategy backfills NaN over the whole historical block.
        """
        target = Path(path)
        widening = Path(f"{path}{self.WIDENING_SUFFIX}")
        superseded = Path(f"{path}{self.SUPERSEDED_SUFFIX}")

        if superseded.exists() and not target.exists():
            raise ValueError(
                f"XrBackend.widen_symbol_axis: refusing to widen {path} -- "
                f"there is no store there, but a superseded sidecar exists at "
                f"{superseded}. A previous widen crashed between its two "
                f"renames, so that sidecar holds the ONLY copy of the store. "
                f"Recover it by hand -- `mv {superseded} {path}` -- and re-run. "
                f"Auto-recovering is deliberately not done here: which "
                f"directory is authoritative is not this method's call to make."
            )
        if widening.exists():
            # A never-authoritative orphan from a crashed rewrite. Nothing ever
            # reads it, so removing it is safe and keeps the retry clean.
            shutil.rmtree(widening, ignore_errors=True)
        if not target.exists():
            raise FileNotFoundError(f"File {path} does not exist.")

        requested = [str(symbol) for symbol in symbols]
        fills = dict(fill_values or {})

        stored = xr.open_zarr(path)
        try:
            stored_labels = [str(label) for label in stored[dim].values.tolist()]
            dropped = [
                label for label in stored_labels if label not in set(requested)
            ]
            if dropped:
                raise ValueError(
                    f"XrBackend.widen_symbol_axis: refusing to widen {path} -- "
                    f"the target '{dim}' axis is not a superset of the stored "
                    f"one; it would DROP {dropped}. Reindexing silently "
                    f"deletes the history of every label it is not asked for, "
                    f"and afterwards the store is indistinguishable from one "
                    f"that never held it. Pass the union of the stored and "
                    f"incoming labels, the way widen_and_append() does."
                )

            offenders = [
                (name, variable.dtype)
                for name, variable in stored.data_vars.items()
                if dim in variable.dims
                and name not in fills
                and not np.issubdtype(variable.dtype, np.floating)
            ]
            if offenders:
                described = ", ".join(
                    f"'{name}' ({dtype})" for name, dtype in offenders
                )
                integer = [
                    name
                    for name, dtype in offenders
                    if np.issubdtype(dtype, np.integer)
                ]
                remedy = (
                    f"Pass fill_values={{...}} naming a value for each -- e.g. "
                    f"fill_values={{'anomaly_flag': False}} -- which is "
                    f"measured to preserve the stored dtype exactly."
                )
                if integer:
                    remedy += (
                        f" For the integer variable(s) {integer}, the promotion "
                        f"normally belongs in BaseDataset._pin_append_dtypes, "
                        f"which already floats integer variables before an "
                        f"append; a store predating that is the usual cause."
                    )
                raise ValueError(
                    f"XrBackend.widen_symbol_axis: refusing to widen {path} -- "
                    f"variable(s) {described} carry '{dim}' but are not "
                    f"floating-point and no fill value was given. An unfilled "
                    f"reindex UPCASTS them to float64 and writes NaN into the "
                    f"new columns, silently changing an existing store's "
                    f"schema behind the caller -- the same family of invisible "
                    f"corruption _assert_append_compatible refuses on the "
                    f"append path. {remedy}"
                )

            estimate = self._estimate_widen_bytes(
                stored, requested, dim, append_dim
            )
            block_rows = self._widen_block_rows(estimate["row_bytes"])
            chunked = estimate["widened_bytes"] > self.MAX_WIDEN_BYTES
            self._report_widen_strategy(
                path, dim, append_dim, estimate, block_rows, chunked
            )

            # Per D-3 the chosen strategy runs INSIDE `stored`'s lifetime: both
            # strategies read `path` through that handle, the chunked one for
            # the whole duration of its block loop. The cleanup spans the WHOLE
            # strategy call rather than a single write, so a crash on block 7
            # of 12 leaves no orphan sidecar and leaves `path` authoritative.
            strategy = self._widen_chunked if chunked else self._widen_whole_store
            try:
                strategy(
                    stored,
                    widening,
                    requested,
                    dim=dim,
                    append_dim=append_dim,
                    fills=fills,
                    block_rows=block_rows,
                )
            except BaseException:
                shutil.rmtree(widening, ignore_errors=True)
                raise
        finally:
            stored.close()

        os.replace(target, superseded)
        os.replace(widening, target)
        shutil.rmtree(superseded, ignore_errors=True)
        return self

    @staticmethod
    def _estimate_widen_bytes(
        stored: xr.Dataset,
        requested: Sequence[str],
        dim: str,
        append_dim: str,
    ) -> dict:
        """Size the panel a widen of `stored` onto `requested` would produce.

        Takes the ALREADY-OPEN dataset rather than a path on purpose: the
        router holds one -- it opened the store to run the superset and dtype
        guards -- and a path-taking sibling would be a SECOND live name for one
        estimate with no caller of its own. Lift it to a path-taking form the
        day something outside the router needs to size a widen without opening
        the store first; until then, one name.

        `widened_bytes` sums, over every data variable, that variable's byte
        count with the `dim` extent replaced by `len(requested)` -- i.e. the
        allocation the whole-store rewrite makes. `row_bytes` is the same
        quantity per ONE `append_dim` row, summed over only those variables
        that carry `append_dim`, which is what turns a byte budget into a block
        length in `_widen_block_rows`. Variables that do not carry `append_dim`
        contribute to `widened_bytes` (they are materialised too) but not to
        `row_bytes` (they do not scale with the block).

        Metadata only: nothing here reads a chunk off disk, so the estimate is
        free relative to the rewrite it decides.
        """
        widened_bytes = 0
        row_bytes = 0
        for variable in stored.data_vars.values():
            sizes = {str(name): int(size) for name, size in variable.sizes.items()}
            if dim in sizes:
                sizes[dim] = len(requested)
            count = 1
            for extent in sizes.values():
                count *= extent
            nbytes = count * variable.dtype.itemsize
            widened_bytes += nbytes
            if sizes.get(append_dim):
                row_bytes += nbytes // sizes[append_dim]

        return {
            "stored_symbols": int(stored.sizes.get(dim, 0)),
            "symbols": len(requested),
            "timestamps": int(stored.sizes.get(append_dim, 0)),
            "variables": len(stored.data_vars),
            "row_bytes": row_bytes,
            "widened_bytes": widened_bytes,
        }

    @staticmethod
    def _widen_block_rows(row_bytes: int) -> int:
        """How many `append_dim` rows one chunked-rewrite block may hold.

        D-2's rule, verbatim::

            raw        = MAX_WIDEN_BYTES // row_bytes      (0 when row_bytes is 0)
            block_rows = max(APPEND_DIM_CHUNK,
                             (raw // APPEND_DIM_CHUNK) * APPEND_DIM_CHUNK)

        **The grid constraint is load-bearing, not stylistic.** The FIRST block
        is written with `encoding=self._append_encoding(append_dim,
        data=first_block)` -- routing through the single-sourced chunk rule
        rather than restating it -- so the block's own length is what decides
        the store's on-disk append-dim chunk. Measured 2026-09-08 against a
        600-timestamp store: a 100-row block leaves `close` chunks `(100, 3)`
        where the whole-store path leaves `(512, 3)`; a 512-row block
        reproduces `(512, 3)` exactly. Requiring the block to be at least
        `APPEND_DIM_CHUNK` AND a multiple of it makes
        `min(APPEND_DIM_CHUNK, first_block_len)` equal
        `min(APPEND_DIM_CHUNK, total_len)` identically, so the two strategies
        agree on the grid without either restating the arithmetic.

        **The floor wins even when one aligned block exceeds the budget.** This
        method ROUTES, it does not refuse: a store whose single `APPEND_DIM_CHUNK`
        block is already over budget still gets the bounded loop, which is
        strictly better than the whole-store allocation it replaces. Refusing
        is not an option `Factor.update()` could act on -- it has no raw tier to
        rebuild from.

        `TimeChunkPlanner` is deliberately NOT used (D-2). Its granularities are
        CALENDAR periods, and a period's row count is a function of frequency
        and density -- a month of 1-minute bars is ~390x a month of daily bars
        (`BARS_PER_DAY_BY_FREQUENCY`) -- so it cannot bound BYTES, which is the
        entire constraint here. It stays the right tool for planning
        CONVERSION windows, where the calendar is the unit of work and no
        timestamp axis exists yet.
        """
        chunk = XrBackend.APPEND_DIM_CHUNK
        raw = XrBackend.MAX_WIDEN_BYTES // row_bytes if row_bytes > 0 else 0
        return max(chunk, (raw // chunk) * chunk)

    @staticmethod
    def _widen_blocks(timestamps: int, block_rows: int) -> range:
        """The block offsets the chunked rewrite walks.

        `max(timestamps, 1)` so a store with a zero-length `append_dim` still
        writes exactly one (empty) block rather than no block at all, which
        would leave no sidecar for the swap to rename in.
        """
        return range(0, max(int(timestamps), 1), block_rows)

    def _report_widen_strategy(
        self,
        path: str,
        dim: str,
        append_dim: str,
        estimate: Mapping[str, int],
        block_rows: int,
        chunked: bool,
    ) -> None:
        """Say which strategy ran, and how loudly (D-6).

        Asymmetric on purpose. A `warning` on every routine sub-budget widen is
        noise, and noise is how an operator learns to stop reading warnings; the
        requirement is that the SWITCH not be silent, not that every widen
        announce itself. So the chunked branch is loud and names every figure a
        reader needs -- including the measured wall-clock multiplier, so a slow
        run reads as the strategy rather than as the machine -- and the
        whole-store branch is one `info` line, enough that which path ran is
        always answerable from the log.

        Shaped after `BaseDataset._reconcile_new_listings`'s widen warning: name
        the store, name both counts, name the consequence, name the opt-out.
        """
        gib = 1024**3
        mib = 1024**2
        if not chunked:
            logger.info(
                f"XrBackend.widen_symbol_axis: {path} -- widening '{dim}' from "
                f"{estimate['stored_symbols']} to {estimate['symbols']} "
                f"label(s) materialises "
                f"{estimate['widened_bytes'] / gib:.3f} GiB, within the "
                f"{self.MAX_WIDEN_BYTES / gib:.2f} GiB MAX_WIDEN_BYTES budget; "
                f"taking the whole-store rewrite."
            )
            return

        blocks = len(self._widen_blocks(estimate["timestamps"], block_rows))
        logger.warning(
            f"XrBackend.widen_symbol_axis: {path} -- widening '{dim}' from "
            f"{estimate['stored_symbols']} to {estimate['symbols']} label(s) "
            f"would materialise {estimate['widened_bytes'] / gib:.3f} GiB at "
            f"once, over the {self.MAX_WIDEN_BYTES / gib:.2f} GiB "
            f"MAX_WIDEN_BYTES budget. Rewriting the store block by block along "
            f"'{append_dim}' instead: {blocks} block(s) of {block_rows} row(s), "
            f"~{estimate['row_bytes'] * block_rows / mib:.1f} MiB per block, so "
            f"peak memory is ONE block rather than the whole store. Measured "
            f"2026-09-08: the chunked rewrite runs ~3.6-4.0x the whole-store "
            f"wall clock, so a slow run here is the STRATEGY and not the "
            f"machine. Raise XrBackend.MAX_WIDEN_BYTES deliberately to take the "
            f"whole-store path anyway."
        )

    def _widen_whole_store(
        self,
        stored: xr.Dataset,
        widening: Path,
        requested: Sequence[str],
        *,
        dim: str,
        append_dim: str,
        fills: Mapping[str, object],
        block_rows: int,
    ) -> None:
        """The shipped rewrite, unchanged: reindex the WHOLE store, write once.

        Faster than `_widen_chunked` wherever it fits -- measured 2026-09-08,
        chunked runs 1.2x / 4.0x / 3.6x this path's wall clock on 0.2 / 34.6 /
        137.3 MiB stores -- which is exactly why `widen_symbol_axis` routes
        rather than always chunking.

        `block_rows` is accepted and ignored. The two strategies carry ONE
        signature so the router selects between them by name and calls them
        identically; a router that had to remember which arguments each
        strategy wanted is a router with two call sites to drift apart.
        """
        # `.load()` is load-bearing, not defensive: without dask,
        # `open_zarr` still hands back lazily-indexed arrays that read from
        # the store directory on access, and the swap below renames that
        # directory out from under them.
        widened = stored.reindex({dim: requested}, fill_value=fills).load()
        encoding = self._append_encoding(append_dim, data=widened)
        widened.to_zarr(str(widening), mode="w", encoding=encoding)

    def _widen_chunked(
        self,
        stored: xr.Dataset,
        widening: Path,
        requested: Sequence[str],
        *,
        dim: str,
        append_dim: str,
        fills: Mapping[str, object],
        block_rows: int,
    ) -> None:
        """The bounded rewrite: one `append_dim` block in memory at a time.

        Same output as `_widen_whole_store`, measured element for element
        (values with `equal_nan=True`, both coordinates, the on-disk chunk grid
        AND the `symbol` coordinate's on-disk encoding, across both live
        production encodings) -- and a peak allocation of one block instead of
        the whole store. `tests/test_symbol_axis_widening.py` is what keeps the
        two paths agreeing; the equivalence is the deliverable, not a nicety.

        The FIRST block creates the sidecar with `mode="w"` and
        `encoding=self._append_encoding(append_dim, data=block)`, so the chunk
        rule stays single-sourced and the block-size constraint in
        `_widen_block_rows` is what makes the resulting grid match the
        whole-store path's. Every LATER block appends along `append_dim`, first
        dropping any data variable that does not carry it: the first block
        already wrote those at their full extent, and handing them to an
        appending write again would rewrite them per block.

        **Each block is `.load()`ed before its own write**, for the same reason
        the whole-store path loads once: without dask, `open_zarr` hands back
        lazily-indexed arrays that read from the store directory on access, and
        the swap renames that directory out from under them. At block
        granularity the requirement is sharper -- no lazily-indexed reference
        may outlive its ITERATION either, because the next iteration's `isel`
        must be free to read the same handle.

        `stored` is read for the whole duration of the loop, which is why the
        loop runs inside the router's `try:` whose `finally` closes it, and why
        both `os.replace` calls stay outside that (D-3).
        """
        timestamps = int(stored.sizes.get(append_dim, 0))
        for index, low in enumerate(self._widen_blocks(timestamps, block_rows)):
            block = stored.isel(
                {append_dim: slice(low, low + block_rows)}
            ).load()
            block = block.reindex({dim: requested}, fill_value=fills)
            if index == 0:
                block.to_zarr(
                    str(widening),
                    mode="w",
                    encoding=self._append_encoding(append_dim, data=block),
                )
                continue
            block = block.drop_vars(
                [
                    name
                    for name, variable in block.data_vars.items()
                    if append_dim not in variable.dims
                ]
            )
            block.to_zarr(str(widening), mode="a", append_dim=append_dim)

    def widen_data_vars(
        self,
        path: str,
        variables: Mapping[str, object],
        append_dim: str = "timestamp",
        fill_values: Optional[Mapping[str, object]] = None,
    ) -> Self:
        """Add data variable(s) to the store at `path`, backfilled over its
        EXISTING extent.

        The storage-medium answer to "the stored panel grew a column between
        two runs" -- the `data_vars` counterpart of `widen_symbol_axis`,
        following the same rule. The incoming set must be a SUPERSET; a new
        member is materialised across the whole historical block with the fill
        value (NaN for a floating-point variable, whatever `fill_values` names
        otherwise); every stored variable keeps its values bit-identical. A
        name the store already holds is left entirely alone, and when none of
        `variables` is new this returns without writing at all.

        `variables` maps each name to its DTYPE rather than to the incoming
        array. The dtype is the narrower of the two and is everything the
        filler needs, so this method never holds the caller's panel -- like
        its sibling it operates purely on the store, and a caller can widen
        without having a panel in hand.

        **Why this is a separate method rather than a flag on `append()`.**
        `_assert_append_compatible` refuses an incoming variable set the store
        does not match, for a reason that has not gone away: zarr extends
        exactly the variables it is handed, so a new name written straight
        through lands SHORTER along `append_dim` than everything already
        stored, and `xr.open_zarr` afterwards refuses the ENTIRE store with
        `conflicting sizes for dimension 'timestamp'` (measured 2026-09-07).
        Widening is an explicit, separately named opt-in that makes the two
        sets AGREE before the append; it does not loosen the refusal a caller
        who did not opt in still gets.

        **Guards fire before any write**, mirroring `widen_symbol_axis`:

        1. No store at `path` -> `FileNotFoundError`, same as its sibling.
        2. A new variable that is neither floating-point nor named in
           `fill_values` is refused. Measured 2026-09-07:
           `np.full(shape, np.nan, dtype='int64')` yields 0 and `dtype=bool`
           yields True, so an unfilled backfill across the store's whole
           history FABRICATES observations rather than marking them absent --
           the same family of invisible corruption
           `_assert_append_compatible` refuses on the append path. An explicit
           `fill_values` entry preserves the stored dtype exactly.

        The filler's `encoding` comes from `_append_encoding`, so the new
        variable joins the store on the SAME chunk grid as every other one.
        Measured 2026-09-07 with `APPEND_DIM_CHUNK` at 4 over a 10-long store:
        an unencoded filler takes the store's whole extent as its chunk while
        the stored variable holds 4. The next append still succeeds, which is
        precisely why the grid is pinned here by construction rather than left
        to surface later as a layout nobody chose.

        Unlike its sibling this needs NO directory swap. The measured
        behaviour is that a partial-extent write raises and leaves the store
        INTACT, so the operation is already safe to retry.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")

        fills = dict(fill_values or {})
        requested = {
            str(name): np.dtype(dtype) for name, dtype in variables.items()
        }

        stored = xr.open_zarr(path)
        try:
            absent = {
                name: dtype
                for name, dtype in requested.items()
                if name not in stored.data_vars
            }
            if not absent:
                return self

            offenders = [
                (name, dtype)
                for name, dtype in absent.items()
                if name not in fills and not np.issubdtype(dtype, np.floating)
            ]
            if offenders:
                described = ", ".join(
                    f"'{name}' ({dtype})" for name, dtype in offenders
                )
                raise ValueError(
                    f"XrBackend.widen_data_vars: refusing to widen {path} -- "
                    f"new variable(s) {described} are not floating-point and "
                    f"no fill value was given. The filler spans the store's "
                    f"ENTIRE existing extent, and an unfilled backfill does "
                    f"not mark that history absent, it FABRICATES it: "
                    f"measured 2026-09-07, np.full(shape, np.nan) yields 0 "
                    f"for int64 and True for bool, so a boolean flag would "
                    f"read as set on every historical row. That is the same "
                    f"family of invisible corruption "
                    f"_assert_append_compatible refuses on the append path. "
                    f"Pass fill_values={{...}} naming a value for each -- "
                    f"e.g. fill_values={{'anomaly_flag': False}} -- which is "
                    f"measured to preserve the dtype exactly."
                )

            # The store's own layout, taken from a variable that already
            # spans `append_dim`, so the filler reproduces the shape the store
            # actually has rather than one assumed here.
            layout = next(
                (
                    variable.dims
                    for variable in stored.data_vars.values()
                    if append_dim in variable.dims
                ),
                None,
            )
            dims = tuple(layout) if layout is not None else tuple(stored.sizes)
            shape = tuple(int(stored.sizes[name]) for name in dims)

            # NO coords on the filler, deliberately. The store already holds
            # every one of these dimensions' coordinates, and handing them
            # back is not a no-op: `stored[name].values` is the DECODED array,
            # and re-encoding it on the way in can land on a different dtype
            # than the one zarr recorded. Measured 2026-09-08 against a store
            # this project's own chunked ingest builds -- zarr holds `symbol`
            # as `object`, `xr.open_zarr` decodes it to numpy's
            # `StringDType()`, and writing that back raises
            # `ValueError: Mismatched dtypes for variable symbol between Zarr
            # store on disk and dataset to append`, from INSIDE this method,
            # before the filler lands. That made the whole variable-widening
            # path unreachable for any store with a string coordinate, which
            # is every real store here.
            #
            # A filler carrying only its dims is positionally aligned by zarr
            # against the arrays already on disk, so the coordinates stay
            # exactly as written and every stored value stays bit-identical
            # (measured: `keep` unchanged, `symbol`/`timestamp` untouched, the
            # new variable all-fill across the store's whole extent, and the
            # following `append()` succeeds).
            filler = xr.Dataset(
                {
                    name: (
                        dims,
                        np.full(shape, fills.get(name, np.nan), dtype=dtype),
                    )
                    for name, dtype in absent.items()
                }
            )
            encoding = self._append_encoding(append_dim, data=filler)
        finally:
            stored.close()

        filler.to_zarr(path, mode="a", encoding=encoding)
        return self

    def widen_and_append(
        self,
        path: str,
        append_dim: str = "timestamp",
        dim: str = "symbol",
        fill_values: Optional[Mapping[str, object]] = None,
        **kwargs,
    ) -> Self:
        """`append()`'s explicit opt-in sibling for a panel that has grown.

        Reconciles all THREE axes and then calls the UNCHANGED `append()`. The
        order is fixed: `dim` first (widen the store to
        `sorted(stored | incoming)` and reindex `self.data` onto that same
        axis), then the data-variable set (`widen_data_vars`), then the
        append. Variables second is deliberate -- the filler is built over the
        store's extent AFTER the `dim` widen, so it is materialised once at
        the final width rather than written narrow and rewritten.

        This stays the ONE reconcile-then-append path. A second composed entry
        point for the third axis would give callers two ways to say the same
        thing and two places for the guard ordering to drift apart.

        **The closing `append()` call is load-bearing, not incidental.** Both
        sides now share one axis, so `_assert_append_compatible` still runs and
        PASSES on its own terms -- the guard is satisfied by construction rather
        than bypassed or relaxed. Any refactor that writes the window directly
        with `to_zarr(mode="a")` from inside this method removes the guard from
        the widened path entirely and reintroduces the measured
        mis-attribution. Nothing here weakens the refusal a caller who did not
        opt in still gets from plain `append()`.

        The union is `sorted(...)`, matching `base/constituent.py:_densify`'s
        all-time-union rule, because `ChunkLedger`'s fingerprint is
        order-sensitive and the axis must be reproducible across runs.

        Because the widens COMMIT before the closing `append()` runs, a window
        the shared guard then refuses -- an overlapping `append_dim` range, say
        -- can leave the store carrying the GROWN symbol axis AND the grown
        variable set while its append dimension is untouched and its
        pre-existing history intact; that is an accepted side effect of
        inheriting the refusal rather than duplicating it, not a partial write
        of the window. It now spans the variable axis as well as the symbol
        one, and for the same reason.

        Calling this unconditionally is cheap: axes that already agree skip
        both rewrites entirely and delegate straight to `append()`. The fast
        path requires BOTH the `dim` axis and the variable set to match --
        checking only the former would send a variable-grown panel to a plain
        `append()`, which refuses it. An absent store delegates too, so there
        is ONE creation path rather than two.
        """
        if not Path(path).exists():
            return self.append(path, append_dim, **kwargs)

        stored = xr.open_zarr(path)
        try:
            stored_labels = (
                [str(label) for label in stored[dim].values.tolist()]
                if dim in stored.coords
                else []
            )
            stored_names = set(stored.data_vars)
        finally:
            stored.close()

        incoming = (
            [str(label) for label in self.data[dim].values.tolist()]
            if dim in self.data.coords
            else []
        )
        # Name -> dtype, which is all `widen_data_vars` needs and all it is
        # given: the filler's dtype must be the INCOMING one or the closing
        # `append()`'s shared-variable dtype guard refuses the window.
        incoming_names = {
            str(name): variable.dtype
            for name, variable in self.data.data_vars.items()
        }
        union = sorted(set(stored_labels) | set(incoming))
        unstored = [
            name for name in incoming_names if name not in stored_names
        ]

        if union == stored_labels and union == incoming and not unstored:
            return self.append(path, append_dim, **kwargs)

        if union != stored_labels or union != incoming:
            self.widen_symbol_axis(
                path,
                union,
                dim=dim,
                append_dim=append_dim,
                fill_values=fill_values,
            )
            self.data = self.data.reindex(
                {dim: union}, fill_value=dict(fill_values or {})
            )

        if unstored:
            self.widen_data_vars(
                path,
                incoming_names,
                append_dim=append_dim,
                fill_values=fill_values,
            )

        return self.append(path, append_dim, **kwargs)

    def _append_encoding(
        self,
        append_dim: str,
        data: Optional[xr.Dataset] = None,
        *,
        append_dim_size: Optional[int] = None,
    ) -> dict:
        """Pin each data variable's chunk shape: `APPEND_DIM_CHUNK` along the
        append dimension, the full length across every other one.

        `data` defaults to `self.data`, which is what `append()` passes
        implicitly. `widen_symbol_axis` passes the WIDENED panel instead, so
        the rewritten store's non-append dims are pinned to their widened
        length while `APPEND_DIM_CHUNK` still governs the append dimension.
        Routing the widen through this method rather than restating the chunk
        arithmetic is what keeps that rule single-sourced.

        **`append_dim_size` is the store's TOTAL extent along `append_dim`,
        and it is what the append-dim chunk is a property of.** `None` -- every
        call site that existed before phase 03.6's gap-closure pass -- means
        "use the panel's own append-dim length", which is this method's
        behaviour verbatim and is the right answer for a caller holding the
        whole store. A caller that writes the store INCREMENTALLY holds only a
        window, and its window's length is an accident of the chunking rung
        rather than a property of the store, so it states the extent instead:
        `BaseDataset.from_raw_data_chunked` passes `len(timestamps)`, D-02's
        once-resolved whole-range axis, through `append()`.

        The SIZE is substituted into the existing
        `max(min(APPEND_DIM_CHUNK, size), 1)` rather than a second floor being
        added beside it, deliberately. `APPEND_DIM_CHUNK` is a CEILING as well
        as the grid unit, so the target is the grid a whole-range write through
        this same method would have produced -- `min(APPEND_DIM_CHUNK, total)`
        -- and a bolted-on `max(APPEND_DIM_CHUNK, ...)` floor would lose the
        ceiling on any range longer than `APPEND_DIM_CHUNK`. That floor IS the
        right shape one method over in `_widen_block_rows`, which is choosing a
        block size under a byte budget rather than a chunk under none.

        Only the APPEND dimension reads it. Every other dimension keeps reading
        the panel in hand, because on the chunked path the panel's non-append
        axes already equal the store's: D-02 resolves the symbol axis once over
        the whole range before any window exists, and the ingestion loop
        refuses a window that came back on a different one.
        """
        panel = self.data if data is None else data
        encoding = {}
        for name, variable in panel.data_vars.items():
            if append_dim not in variable.dims:
                continue
            chunks = []
            for dim in variable.dims:
                if dim == append_dim and append_dim_size is not None:
                    size = int(append_dim_size)
                else:
                    size = int(panel.sizes[dim])
                if dim == append_dim:
                    chunks.append(max(min(self.APPEND_DIM_CHUNK, size), 1))
                else:
                    chunks.append(max(size, 1))
            encoding[name] = {"chunks": tuple(chunks)}
        return encoding

    @staticmethod
    def _format_append_label(value) -> str:
        """Render one append-dimension label for a human reading a refusal.

        `append_dim` is a PARAMETER, so this guard must not become
        timestamp-only: a `datetime64` label reads as an ISO string
        (`2022-01-07T00:00:00` rather than
        `np.datetime64('2022-01-07T00:00:00.000000000')`), and anything else
        falls back to `str()`.
        """
        if np.issubdtype(np.asarray(value).dtype, np.datetime64):
            return pd.Timestamp(value).isoformat()
        return str(value)

    def _assert_append_compatible(self, path: str, append_dim: str) -> None:
        """Raise before an append that would silently corrupt the store."""
        existing = xr.open_zarr(path)
        try:
            for dim in self.data.dims:
                if dim == append_dim or dim not in existing.dims:
                    continue
                if dim not in self.data.coords or dim not in existing.coords:
                    continue
                incoming = self.data[dim].values
                stored = existing[dim].values
                if len(incoming) != len(stored) or not (incoming == stored).all():
                    raise ValueError(
                        f"XrBackend.append: refusing to append to {path} -- "
                        f"the '{dim}' coordinate does not match the store "
                        f"({len(incoming)} incoming label(s) vs "
                        f"{len(stored)} stored). Zarr would OVERWRITE the "
                        f"stored labels without complaint, silently "
                        f"re-attributing every previously written row. Pin "
                        f"the '{dim}' axis over the whole range before the "
                        f"first window, the way "
                        f"BaseDataset.from_raw_data_chunked() does."
                    )
            # The append dimension itself, skipped by the loop above. Nothing
            # else compares an incoming window's labels against the stored
            # ones, so an OVERLAPPING window appends silently and corrupts the
            # axis. Skipped when either side carries no coordinate on this
            # dimension -- a store with the dim but no coord appends fine
            # today, and turning that working path into a crash is not the job
            # here -- and skipped when either side is empty, since there is
            # nothing to compare. A GAP is deliberately permitted (D-01).
            if (
                append_dim in self.data.coords
                and append_dim in existing.coords
                and self.data[append_dim].size
                and existing[append_dim].size
            ):
                # `.min()` / `.max()` rather than positional indexing, so an
                # unsorted axis on either side cannot fool the comparison.
                incoming_start = self.data[append_dim].values.min()
                stored_end = existing[append_dim].values.max()
                # `<=`, not `<`: a window starting exactly ON the stored end
                # duplicates that one label.
                if incoming_start <= stored_end:
                    raise ValueError(
                        f"XrBackend.append: refusing to append to {path} -- "
                        f"the incoming '{append_dim}' window starts at "
                        f"{self._format_append_label(incoming_start)} but the "
                        f"store already ends at "
                        f"{self._format_append_label(stored_end)}. Zarr would "
                        f"extend the axis without complaint and leave "
                        f"'{append_dim}' no longer STRICTLY increasing -- "
                        f"duplicate labels, out-of-order labels, or both -- "
                        f"which breaks every downstream reader that assumes a "
                        f"unique, ordered index. append() EXTENDS a store; to "
                        f"recompute a range it already holds, replace the "
                        f"store with save(mode=\"w\") instead."
                    )
            for name, variable in self.data.data_vars.items():
                if name not in existing.data_vars:
                    continue
                if variable.dtype != existing[name].dtype:
                    raise ValueError(
                        f"XrBackend.append: refusing to append to {path} -- "
                        f"variable '{name}' has dtype {variable.dtype} but "
                        f"the store holds {existing[name].dtype}. Zarr would "
                        f"cast silently, and a float NaN cast into an "
                        f"integer store becomes 0: a fabricated observation "
                        f"where the data was missing."
                    )
            # The data-variable SET, which the loop above cannot reach: it
            # skips any incoming name the store lacks, and never visits a
            # stored name the incoming panel lacks at all. Placed AFTER that
            # loop deliberately (D-03), so no pre-existing refusal's
            # precedence moves -- a panel carrying BOTH a dtype mismatch on a
            # shared variable AND a changed variable set still raises the
            # dtype message it raised before this check existed.
            incoming_names = set(self.data.data_vars)
            stored_names = set(existing.data_vars)
            # The MISSING direction is checked FIRST: it is the one that
            # destroys data which was valid before the call, and it is the one
            # with no remedy short of recomputing. A caller shown the widening
            # message first would widen the new variable in, retry, and be
            # refused all over again on the dropped one.
            absent = sorted(stored_names - incoming_names)
            if absent:
                raise ValueError(
                    f"XrBackend.append: refusing to append to {path} -- the "
                    f"store holds data variable(s) {absent} that the incoming "
                    f"panel does not. Zarr extends exactly the variables it "
                    f"is handed, so the absent one(s) would stay STUCK at "
                    f"their stored length while every other variable grows, "
                    f"and the store afterwards cannot be OPENED at all "
                    f"(measured 2026-09-07: conflicting sizes for dimension "
                    f"'{append_dim}'). What it loses was valid before this "
                    f"call. This direction has no opt-in and is not given "
                    f"one: backfilling the absent variable across the "
                    f"incoming window would write NaN into recent dates of a "
                    f"variable that was COMPLETE, and afterwards the store is "
                    f"indistinguishable from one where those values were "
                    f"genuinely missing. Recompute this window over the "
                    f"store's FULL variable set, or replace the store with "
                    f"save(mode=\"w\")."
                )
            unstored = sorted(incoming_names - stored_names)
            if unstored:
                raise ValueError(
                    f"XrBackend.append: refusing to append to {path} -- the "
                    f"incoming panel carries data variable(s) {unstored} that "
                    f"the store does not hold. Zarr would write them over the "
                    f"incoming window ONLY, leaving them shorter along "
                    f"'{append_dim}' than every stored variable, and the "
                    f"store afterwards cannot be OPENED at all (measured "
                    f"2026-09-07: conflicting sizes for dimension "
                    f"'{append_dim}'). A panel that legitimately grew a "
                    f"column says so explicitly: materialise the new "
                    f"variable(s) over the store's EXISTING extent first with "
                    f"widen_data_vars(), which backfills history rather than "
                    f"truncating it, or call widen_and_append(), which does "
                    f"that as part of reconciling every axis."
                )
        finally:
            existing.close()

    def to_internal(self, data: xr.Dataset) -> Self:
        self.data = data
        return self

    def filter_by_date(self, col: str, start_date: str, end_date: str) -> Self:
        self.data = self.data.sel({col: slice(start_date, end_date)})
        return self

    def filter_by_symbol(self, col: str, symbols: tuple[str, ...]) -> Self:
        self.data = self.data.sel({col: list(symbols)})
        return self

    def get_xarray_dataset(
        self, indexes: Optional[list[str]] = None
    ) -> xr.Dataset:
        """按 `indexes` 给定的维度返回面板；`indexes=None` 表示「原样返回」。

        `indexes` 的语义与 `PlBackend.get_xarray_dataset` 保持一致：**它就是结果
        的索引维度**。那边是 `set_index(indexes)` 之后 `Dataset.from_dataframe`，
        所以只有 `indexes` 里的名字会成为维度，其余列成为数据变量；这边输入本来
        就是 `xr.Dataset`，等价动作是：

        1. 校验每个名字都是当前面板的维度，不是就报错并把实际维度列出来；
        2. 丢掉任何铺在 `indexes` 之外维度上的数据变量；
        3. 丢掉不再被使用的维度（连同它的坐标）；
        4. 把剩下的变量 `transpose` 成 `indexes` 给定的轴顺序。

        以前这个方法的函数体只有一行 `return self.data`——`indexes` 完全被忽略，
        传什么都一样（2026-09-07 修复，`tests/test_backend_indexes.py` 锁）。

        **对既有调用点全部是无操作**：全仓传的要么是 `["timestamp", "symbol"]`，
        要么什么都不传。规范面板的每个数据变量都恰好铺在这两维上，所以第 2、3 步
        什么都不丢，第 4 步只是把轴顺序钉死成 `(timestamp, symbol)`——这正是
        CLAUDE.md 里那条硬约束，只不过以前靠 `from_raw_data()` 稠密化时的约定
        维持，现在在边界上真的校验了。

        真正因此改变行为的只有 `indexes=["timestamp"]`，也就是
        `BaseDataset.time_interval` 那一条路：它以前拿回整个面板，`.diff()` 撞上
        布尔的 `anomaly_flag` 直接 `TypeError`。现在拿回的是一个只剩时间轴的
        `Dataset`（没有数据变量，但保留 `timestamp` 坐标），差分可以正常做。

        `indexes=None` 保持返回 `self.data` 本身（不是副本），因为几十个调用点
        依赖「拿到的就是后端持有的那个对象」这一点。

        本方法**不改写** `self.data`。同接口上的 `filter_by_date` /
        `filter_by_symbol` 是就地收窄的，照着它们的样子实现这一个会静默截断调用方
        和别人共享的那份面板——那正是 RV-01 的故障模式。
        """
        if indexes is None:
            return self.data

        data = self.data
        missing = [name for name in indexes if name not in data.dims]
        if missing:
            raise ValueError(
                f"XrBackend.get_xarray_dataset: requested index(es) "
                f"{missing} are not dimensions of this dataset. Present "
                f"dimensions: {tuple(data.dims)}."
            )

        dropped_vars = [
            name
            for name, variable in data.data_vars.items()
            if not set(variable.dims) <= set(indexes)
        ]
        result = data.drop_vars(dropped_vars)

        unused_dims = [dim for dim in result.dims if dim not in indexes]
        if unused_dims:
            result = result.drop_dims(unused_dims)

        return result.transpose(*indexes, ...)

    def get_lazyframe(self) -> pl.LazyFrame:
        data = self.data.to_dataframe().reset_index()
        return pl.from_pandas(data).lazy()

    def head(self, path: str, n: int) -> pl.LazyFrame:
        """At most `n` rows, bounding EVERY dimension before converting.

        Opens the store at `path` with `xr.open_dataset` -- the SAME opener
        `read()` above uses, deliberately not `xr.open_zarr`. Two different
        openers for one store in one class is a divergence waiting to bite;
        `_assert_append_compatible`'s `open_zarr` is a separate,
        append-specific concern.

        Opening by path rather than reading `self.data` is the RV-01 fix, not
        a stylistic choice. `self.data` can only be populated by a prior
        `read()`, and `BaseDataset.read()` runs `_filter()`, which narrows the
        shared dataset IN PLACE; `read()`'s cache early-return above then
        makes that narrowing permanent. A `FactorPolars` name probe going
        through that path silently dropped its factor's entire lookback
        window. Nothing here touches `self.data`, so there is no narrowing
        left to survive.

        The selector is built from the opened dataset's `dims` rather than
        naming `timestamp`: a storage-medium-agnostic backend has no business
        knowing that this project's panels happen to be indexed by time and
        symbol, and a dataset with a third axis would otherwise be converted
        in full.

        The `Path(path).exists()` guard mirrors `read()`'s, message included
        (D-3 of the RV-01 fix plan): without it a missing zarr directory
        surfaces as an obscure xarray engine-guess error instead of naming the
        path that is not there.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")

        opened = xr.open_dataset(path)
        try:
            bounded = opened.isel({dim: slice(0, n) for dim in opened.dims})
            frame = bounded.to_dataframe().reset_index()
        finally:
            opened.close()
        return pl.from_pandas(frame).lazy().head(n)


class PlBackend(DataBackend):
    def read(self, path: str, **kwargs) -> Self:
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        self.data = pl.scan_parquet(path)
        return self

    def write(self, path: str, **kwargs) -> Self:
        self.data.collect().write_parquet(path, **kwargs)
        return self

    def to_internal(self, data: pl.LazyFrame) -> Self:
        self.data = data
        return self

    def filter_by_date(self, col: str, start_date: str, end_date: str) -> Self:
        self.data = self.data.filter(
            pl.col(col).is_between(
                pl.lit(pd.to_datetime(start_date)),
                pl.lit(pd.to_datetime(end_date)),
            )
        )
        return self

    def filter_by_symbol(self, col: str, symbols: tuple[str, ...]) -> Self:
        self.data = self.data.filter(pl.col(col).is_in(symbols))
        return self

    def get_lazyframe(self) -> pl.LazyFrame:
        return self.data

    def head(self, path: str, n: int) -> pl.LazyFrame:
        """At most `n` rows, genuinely lazily, scanned straight from `path`.

        `scan_parquet` pushes the limit down into the reader, so this costs
        essentially nothing here -- and it returns a fresh `pl.LazyFrame`
        rather than touching `self.data`, so the non-mutation half of the
        contract comes for free.

        The `Path(path).exists()` guard mirrors `read()`'s, message included:
        `scan_parquet` on an absent file fails only at `.collect()` time, far
        from the call that was actually wrong.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        return pl.scan_parquet(path).head(n)

    def get_xarray_dataset(
        self, indexes: Optional[list[str]] = None
    ) -> xr.Dataset:
        """`set_index(indexes)` 之后 `Dataset.from_dataframe`——`indexes` 里的
        名字成为维度，其余列成为数据变量。这一直是 `indexes` 的语义来源，
        `XrBackend` 那边在 2026-09-07 才对齐上来。

        这里 `indexes` 不能省：一个 `pl.LazyFrame` 是纯粹的表，没有维度可言，
        不指定索引就无从构造 `Dataset`。ABC 上的默认值 `None` 是给
        `XrBackend`「原样返回」用的。
        """
        if indexes is None:
            raise ValueError(
                "PlBackend.get_xarray_dataset: `indexes` is required. A "
                "LazyFrame has no dimensions to fall back on -- name the "
                "columns that should become the dataset's index, e.g. "
                '["timestamp", "symbol"].'
            )
        data = self.data.collect().to_pandas()
        data = data.set_index(indexes)
        return xr.Dataset.from_dataframe(data)

"""Concrete storage backends: Zarr-backed xarray panels and Parquet tables.

``XrBackend`` is the backend every dataset, factor and model object in the
pipeline owns by default. It keeps an ``xarray.Dataset`` in memory, persists
it as a Zarr store, and carries the machinery chunked ingestion needs to grow
a store safely over time: ``append`` extends the time axis with a set of
corruption checks that raw ``to_zarr(mode="a")`` does not perform, and the
``widen_*`` family reconciles a store whose symbol roster or variable set has
grown between runs. ``PlBackend`` is the lazy Parquet counterpart used for
long-format reference tables. Both implement ``DataBackend`` from
``quantlab.base.backend``; see ``docs/backend.md`` and
``docs/chunking.md``.
"""

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
from quantlab.utils.symbol_axis import normalize_to_axis_dtype, sort_symbol_axis


class XrBackend(DataBackend):
    """Zarr-backed storage for an ``xarray.Dataset`` panel.

    ``read`` and ``write`` move the whole panel between memory and a Zarr
    directory. ``append`` extends an existing store along one dimension
    (``timestamp`` by default) after checking that the incoming window cannot
    silently corrupt it. ``widen_symbol_axis``, ``widen_data_vars`` and
    ``widen_and_append`` handle the case where the panel written today has
    more symbols or more variables than the store already holds.

    Two class constants govern the on-disk layout: ``APPEND_DIM_CHUNK`` is
    the chunk length pinned along the append dimension when a store is
    created, and ``MAX_WIDEN_BYTES`` is the largest panel a widen will hold
    in memory at once before switching to a block-by-block rewrite.

    Examples
    --------
    >>> backend = XrBackend().to_internal(panel)
    >>> backend.write("prices.zarr")
    >>> backend.to_internal(next_month).widen_and_append("prices.zarr")
    >>> ds = XrBackend().read("prices.zarr").get_xarray_dataset(
    ...     ["timestamp", "symbol"]
    ... )
    """

    def __init__(self) -> None:
        """Create an empty backend; call ``read`` or ``to_internal`` to fill it."""
        super().__init__()

    def read(self, path: str, overwrite: bool = False, **kwargs) -> Self:
        """Open the Zarr store at ``path`` into ``data``.

        A backend that already holds data returns immediately unless
        ``overwrite`` is true, so repeated reads do not reload the store.

        Parameters
        ----------
        path : str
            Directory of the Zarr store.
        overwrite : bool, default False
            Reload even if ``data`` is already populated.
        **kwargs
            Passed through to ``xarray.open_dataset``.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.

        Examples
        --------
        >>> backend = XrBackend().read("prices.zarr")
        >>> dict(backend.data.sizes)
        {'timestamp': 4, 'symbol': 2}
        >>> backend.read("prices.zarr") is backend  # already loaded, no reload
        True
        """
        if not overwrite and hasattr(self, "data"):
            return self

        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        self.data = xr.open_dataset(path, **kwargs)
        return self

    #: Chunk length pinned along the append dimension when a store is created
    #: by ``append``, ``widen_symbol_axis`` or ``widen_data_vars``. Without an
    #: explicit encoding Zarr would adopt the first window's own length as the
    #: chunk size, and later windows of a different length would be misaligned
    #: with the on-disk grid. The grid a store ends up with is
    #: ``min(APPEND_DIM_CHUNK, extent)``, where ``extent`` is the panel's
    #: length along the append dimension or, when the caller states it with
    #: ``append_dim_size``, the store's eventual total length. Zarr fixes the
    #: grid at creation; an append cannot revise it, and a ``mode="w"`` rewrite
    #: re-pins it.
    APPEND_DIM_CHUNK = 512

    #: Largest number of bytes a symbol-axis widen may materialise at once.
    #: At or under this budget ``widen_symbol_axis`` reindexes the whole store
    #: in memory and writes it once; above it, the store is rewritten block by
    #: block along the append dimension. The value routes between the two
    #: strategies and never refuses a widen. It is a constant of this module
    #: rather than an import because the storage layer must not depend on the
    #: acquisition layer for a scalar.
    MAX_WIDEN_BYTES = 4 * 1024**3

    def write(self, path: str, **kwargs) -> Self:
        """Write ``data`` to ``path``, replacing any store already there.

        The parent directory is created if needed and ``mode`` defaults to
        ``"w"``. No chunk encoding is applied, so the store lands on Zarr's
        default chunk grid.

        Parameters
        ----------
        path : str
            Directory of the Zarr store.
        **kwargs
            Passed through to ``Dataset.to_zarr``.

        Examples
        --------
        >>> XrBackend().to_internal(panel).write("prices.zarr")
        XrBackend()
        >>> Path("prices.zarr").is_dir()
        True
        """
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
        """Create the store at ``path``, or extend it along ``append_dim``.

        When no store exists, ``data`` is written as the first window with
        the chunk grid pinned by ``_append_encoding``. Otherwise the window
        is appended after ``_assert_append_compatible`` has checked that it
        cannot silently corrupt the store. Raw ``to_zarr(mode="a")`` performs
        none of those checks: a changed coordinate overwrites the stored
        labels, a float NaN written into an integer variable becomes ``0``,
        an overlapping window leaves the axis non-monotonic, and a changed
        variable set leaves the store unopenable. A gap between the stored
        end and the incoming start is allowed; overlap is refused and has no
        opt-out, because ``append`` extends a store and does not recompute
        one. To replace a range the store already holds, rewrite the store
        with ``write``.

        A panel that has grown a variable must go through
        ``widen_data_vars`` or ``widen_and_append`` first; a panel missing a
        stored variable is refused outright, since the only fill would be
        NaN over the incoming dates of a variable that was complete.

        Parameters
        ----------
        path : str
            Directory of the Zarr store.
        append_dim : str, default "timestamp"
            The dimension the store grows along.
        append_dim_size : int, optional
            The store's eventual total length along
            ``append_dim``, if the caller knows it. Only the creating
            write reads it; it pins the chunk grid to the value a single
            whole-range write would have chosen instead of the first
            window's length. Ignored against an existing store, so an
            incremental writer may pass it on every window.
        **kwargs
            Passed through to ``Dataset.to_zarr``. An ``encoding``
            entry is dropped on the append path because xarray rejects
            it there.

        Raises
        ------
        ValueError
            If the window fails any compatibility check.

        Examples
        --------
        Two windows on the same symbol axis; the first call creates the
        store, the second extends it:

        >>> XrBackend().to_internal(first_window).append("prices.zarr")
        XrBackend()
        >>> XrBackend().to_internal(next_window).append("prices.zarr")
        XrBackend()
        >>> xr.open_zarr("prices.zarr").sizes["timestamp"]
        6
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
        # xarray rejects `encoding` on an append; the grid was pinned by the
        # creating write above.
        kwargs.pop("encoding", None)
        self.data.to_zarr(path, mode="a", append_dim=append_dim, **kwargs)
        return self

    #: Sidecar suffixes used by ``widen_symbol_axis``'s atomic directory
    #: swap. ``.widening.tmp`` holds the rewritten store before it becomes
    #: authoritative and is never read by anything. ``.superseded.tmp`` briefly
    #: holds the original store between the two renames and is the one
    #: artefact a crash can leave behind that still contains real data.
    WIDENING_SUFFIX = ".widening.tmp"
    SUPERSEDED_SUFFIX = ".superseded.tmp"

    def widen_symbol_axis(
        self,
        path: str,
        symbols: Sequence[str],
        dim: str = "symbol",
        append_dim: str = "timestamp",
        fill_values: Optional[Mapping[str, object]] = None,
        *,
        append_dim_size: Optional[int] = None,
    ) -> Self:
        """Rewrite the store at ``path`` onto a superset ``dim`` axis.

        Every label already in the store keeps its history unchanged; each
        new label is filled across the whole stored extent (NaN for a
        floating-point variable, the value named in ``fill_values``
        otherwise). ``data`` is not touched, so a caller can widen a store
        without holding a panel. This is the explicit opt-in for a roster
        that grew between runs; a plain ``append`` still refuses a changed
        coordinate.

        The rewrite lands in a ``.widening.tmp`` sidecar and is then swapped
        in with two same-parent renames (store to ``.superseded.tmp``,
        sidecar to store) followed by removal of the superseded copy. A crash
        between the renames leaves no store at ``path`` and a superseded
        sidecar holding the only copy; the next call refuses and names the
        manual move rather than guessing which directory is authoritative.
        A non-empty superseded sidecar beside a live store is also refused,
        because it may be a real store left by an interrupted rebuild. Empty
        or never-authoritative residue is removed and the widen proceeds.

        The strategy is chosen by size: at or under ``MAX_WIDEN_BYTES`` the
        whole store is reindexed in memory and written once; above it the
        store is rewritten block by block along ``append_dim`` and a warning
        names the figures involved. Both leave an identical store.

        ``symbols`` is re-spelled in the stored axis's dtype before anything
        is compared, so a ticker handed to an integer axis raises from
        ``normalize_to_axis_dtype`` instead of matching nothing and writing
        an all-NaN store.

        Widening backfills NaN over a new label's history; it does not
        recover data a vendor may have had. Rebuilding from raw data is the
        route when that history matters.

        Parameters
        ----------
        path : str
            Directory of the Zarr store.
        symbols : Sequence[str]
            The labels the rewritten ``dim`` axis must contain. Must
            be a superset of the stored axis.
        dim : str, default "symbol"
            The axis being widened.
        append_dim : str, default "timestamp"
            The store's append dimension, used to size blocks and
            to pin the rewritten chunk grid.
        fill_values : Mapping[str, object], optional
            Per-variable fill for variables carrying ``dim``
            that are not floating-point. Without an entry such a
            variable is refused, because an unfilled reindex would
            silently upcast it to float64.
        append_dim_size : int, optional
            The store's eventual length along
            ``append_dim``. This rewrite re-pins the chunk grid, so a
            caller widening a store that has not yet reached its final
            extent states the extent here; ``None`` sizes the grid from
            the store as it is now.

        Raises
        ------
        FileNotFoundError
            If no store exists at ``path``.
        ValueError
            If crash residue makes the store's identity
            ambiguous, if ``symbols`` would drop a stored label, or if a
            non-floating variable has no fill value.

        Examples
        --------
        >>> XrBackend().widen_symbol_axis("prices.zarr", ["AAA", "BBB", "CCC"])
        XrBackend()
        >>> stored = xr.open_zarr("prices.zarr")
        >>> stored["symbol"].values.tolist()
        ['AAA', 'BBB', 'CCC']
        >>> stored["close"].sel(symbol="CCC").values  # backfilled history
        array([nan, nan, nan, nan, nan, nan])
        """
        target = Path(path)
        widening = Path(f"{path}{self.WIDENING_SUFFIX}")
        superseded = Path(f"{path}{self.SUPERSEDED_SUFFIX}")

        if superseded.exists() and not target.exists():
            raise ValueError(
                f"XrBackend.widen_symbol_axis: refusing to widen {path} -- "
                f"there is no store there, but a superseded sidecar exists at "
                f"{superseded}. A previous widen crashed between its two "
                f"renames, so that sidecar holds the only copy of the store. "
                f"Recover it by hand (`mv {superseded} {path}`) and re-run. "
                f"This method does not recover automatically, because it "
                f"cannot know which directory is authoritative."
            )
        if superseded.exists() and any(superseded.iterdir()):
            raise ValueError(
                f"XrBackend.widen_symbol_axis: refusing to widen {path} -- "
                f"a non-empty superseded directory sits beside the store at "
                f"{superseded}, and it holds real data. Two kinds of run leave "
                f"a directory with that name: a widen that crashed between its "
                f"two renames, and an `on_new_listing=\"rebuild\"` run that "
                f"was killed outright (SIGKILL), whose set-aside copy uses the "
                f"same name and is only cleaned up after an exception or a "
                f"cancel. Refusing now avoids wasted work: otherwise the final "
                f"rename would fail with 'Directory not empty' only after the "
                f"whole store had been rewritten into a sidecar. Decide by "
                f"hand which of {path} and {superseded} is authoritative, "
                f"remove the other, and re-run. This method does not recover "
                f"automatically, because the leftover directory may be the "
                f"only complete copy of the store."
            )
        if superseded.exists():
            # Empty, so it holds nothing; removing it keeps the retry clean.
            shutil.rmtree(superseded, ignore_errors=True)
        if widening.exists():
            # A never-authoritative orphan from a crashed rewrite.
            shutil.rmtree(widening, ignore_errors=True)
        if not target.exists():
            raise FileNotFoundError(f"File {path} does not exist.")

        fills = dict(fill_values or {})

        stored = xr.open_zarr(path)
        try:
            # The request is converted to the stored axis's dtype, which is
            # only known once the store is open.
            stored_index = stored[dim].to_index()
            requested = normalize_to_axis_dtype(symbols, stored_index)

            # Compare the converted values, which is what `reindex` matches.
            dropped = stored_index.difference(pd.Index(requested)).tolist()
            if dropped:
                raise ValueError(
                    f"XrBackend.widen_symbol_axis: refusing to widen {path} -- "
                    f"the target '{dim}' axis is not a superset of the stored "
                    f"one; it would drop {dropped}. Reindexing silently "
                    f"deletes the history of every label it is not asked for, "
                    f"and afterwards the store looks as if it never held "
                    f"them. Pass the union of the stored and incoming labels, "
                    f"as widen_and_append() does."
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
                    f"Pass fill_values={{...}} with a value for each, for "
                    f"example fill_values={{'anomaly_flag': False}}; that "
                    f"keeps the stored dtype exactly."
                )
                if integer:
                    remedy += (
                        f" For the integer variable(s) {integer}, the promotion "
                        f"normally happens in BaseDataset._pin_append_dtypes, "
                        f"which converts integer variables to float before an "
                        f"append; a store written before that step existed is "
                        f"the usual cause."
                    )
                raise ValueError(
                    f"XrBackend.widen_symbol_axis: refusing to widen {path} -- "
                    f"variable(s) {described} carry '{dim}' but are not "
                    f"floating-point and no fill value was given. An unfilled "
                    f"reindex converts them to float64 and writes NaN into the "
                    f"new columns, silently changing the existing store's "
                    f"schema. The append path refuses the same kind of silent "
                    f"corruption in _assert_append_compatible. {remedy}"
                )

            estimate = self._estimate_widen_bytes(
                stored, requested, dim, append_dim
            )
            block_rows = self._widen_block_rows(estimate["row_bytes"])
            chunked = estimate["widened_bytes"] > self.MAX_WIDEN_BYTES
            self._report_widen_strategy(
                path, dim, append_dim, estimate, block_rows, chunked
            )

            # The strategy reads through `stored`, so it must run while the
            # handle is open. On any failure the sidecar is removed and
            # `path` stays the authoritative store.
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
                    append_dim_size=append_dim_size,
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
        """Estimate the size of the panel a widen onto ``requested`` produces.

        Only metadata is read, so the estimate is free relative to the
        rewrite it decides.

        Parameters
        ----------
        stored : xr.Dataset
            The already-open store.
        requested : Sequence[str]
            The target ``dim`` axis.
        dim : str
            The axis being widened.
        append_dim : str
            The store's append dimension.

        Returns
        -------
        dict
            A dict with ``stored_symbols``, ``symbols``, ``timestamps``,
            ``variables``, ``widened_bytes`` (the whole widened panel, every
            variable) and ``row_bytes`` (the bytes of one ``append_dim`` row,
            counting only variables that carry ``append_dim``).
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
        """Return how many ``append_dim`` rows one chunked-rewrite block holds.

        The block is the largest multiple of ``APPEND_DIM_CHUNK`` that fits
        under ``MAX_WIDEN_BYTES``, and never less than one
        ``APPEND_DIM_CHUNK``. The floor bounds memory per block; a single
        block that is still over budget is accepted rather than refused,
        because a bounded loop is still better than the whole-store
        allocation it replaces.

        Parameters
        ----------
        row_bytes : int
            Bytes of one ``append_dim`` row, from
            ``_estimate_widen_bytes``.
        """
        chunk = XrBackend.APPEND_DIM_CHUNK
        raw = XrBackend.MAX_WIDEN_BYTES // row_bytes if row_bytes > 0 else 0
        return max(chunk, (raw // chunk) * chunk)

    @staticmethod
    def _widen_blocks(timestamps: int, block_rows: int) -> range:
        """Return the block start offsets the chunked rewrite walks.

        A store with a zero-length append dimension still yields exactly one
        (empty) block, so the swap always has a sidecar to rename in.
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
        """Log which widen strategy was chosen.

        The whole-store path logs one ``info`` line; the chunked path logs a
        ``warning`` naming every figure it decided on, so a slow run reads as
        the strategy rather than the machine. The asymmetry is deliberate: a
        warning on every routine widen would train operators to ignore it.
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
            f"peak memory is one block rather than the whole store. The "
            f"block-by-block rewrite takes roughly 3.6-4.0x as long as the "
            f"whole-store rewrite, so a slow run here is expected. Raise "
            f"XrBackend.MAX_WIDEN_BYTES to take the whole-store path anyway."
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
        append_dim_size: Optional[int] = None,
    ) -> None:
        """Reindex the whole store in memory and write it once to ``widening``.

        The faster strategy wherever the widened panel fits under
        ``MAX_WIDEN_BYTES``. ``block_rows`` is accepted and ignored so that
        both strategies share one signature. This path has no block floor,
        so ``append_dim_size`` is the only thing that keeps the rewritten
        chunk grid from being sized by whatever the store happens to hold
        right now.
        """
        # `.load()` is required: without it the arrays are read lazily from
        # the store directory, which the swap later renames away.
        widened = stored.reindex({dim: requested}, fill_value=fills).load()
        encoding = self._append_encoding(
            append_dim, data=widened, append_dim_size=append_dim_size
        )
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
        append_dim_size: Optional[int] = None,
    ) -> None:
        """Rewrite the store to ``widening`` one ``append_dim`` block at a time.

        Produces the same store as ``_widen_whole_store`` (values, both
        coordinates, chunk grid and coordinate encoding) with a peak
        allocation of one block. The first block creates the sidecar with
        the chunk grid pinned by ``_append_encoding``; every later block is
        appended after dropping any variable that does not carry
        ``append_dim``, since the first block already wrote those at full
        extent. Each block is loaded before its own write so that no lazily
        indexed reference outlives its iteration.
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
                    encoding=self._append_encoding(
                        append_dim,
                        data=block,
                        append_dim_size=append_dim_size,
                    ),
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
        *,
        append_dim_size: Optional[int] = None,
    ) -> Self:
        """Add data variables to the store at ``path``, backfilled over its extent.

        The variable-set counterpart of ``widen_symbol_axis``. Each name in
        ``variables`` that the store does not yet hold is materialised across
        the store's whole existing extent with the fill value (NaN for a
        floating-point dtype, the ``fill_values`` entry otherwise). Names the
        store already holds are left alone, and when nothing is new the
        method returns without writing. This is the explicit opt-in for a
        panel that grew a column; a plain ``append`` still refuses a changed
        variable set because Zarr would otherwise write the new variable
        over the incoming window only and leave the store unopenable.

        The filler carries dimensions but no coordinates: the store already
        holds them, and writing a decoded coordinate back can land on a
        different dtype than the one Zarr recorded. No directory swap is
        needed because a failed partial write raises and leaves the store
        intact.

        Parameters
        ----------
        path : str
            Directory of the Zarr store.
        variables : Mapping[str, object]
            Maps each variable name to its dtype. The dtype is
            all the filler needs, so the caller's panel is never held.
        append_dim : str, default "timestamp"
            The store's append dimension, used to pin the
            filler's chunk grid to the store's.
        fill_values : Mapping[str, object], optional
            Per-variable fill for new variables that are not
            floating-point. Without an entry such a variable is refused,
            because ``np.full`` with NaN yields ``0`` for integers and
            ``True`` for booleans, fabricating history.
        append_dim_size : int, optional
            The store's eventual length along
            ``append_dim``, so a filler added to a store that has not
            reached its final extent joins on the same grid as the
            variables already there.

        Raises
        ------
        FileNotFoundError
            If no store exists at ``path``.
        ValueError
            If a new non-floating variable has no fill value.

        Examples
        --------
        >>> XrBackend().widen_data_vars("prices.zarr", {"volume": "float64"})
        XrBackend()
        >>> XrBackend().widen_data_vars(
        ...     "prices.zarr", {"flag": "bool"}, fill_values={"flag": False}
        ... )
        XrBackend()
        >>> stored = xr.open_zarr("prices.zarr")
        >>> sorted(stored.data_vars), stored["flag"].dtype
        (['close', 'flag', 'volume'], dtype('bool'))
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
                    f"no fill value was given. The new variable spans the "
                    f"store's entire existing history, and without a fill "
                    f"value that history would be invented rather than marked "
                    f"missing: np.full(shape, np.nan) yields 0 for int64 and "
                    f"True for bool, so a boolean flag would read as set on "
                    f"every historical row. The append path refuses the same "
                    f"kind of silent corruption in _assert_append_compatible. "
                    f"Pass fill_values={{...}} with a value for each, for "
                    f"example fill_values={{'anomaly_flag': False}}; that "
                    f"keeps the dtype exactly."
                )

            # Take the layout from a variable that already spans `append_dim`
            # so the filler reproduces the store's actual shape.
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

            # No coords on the filler: the store already holds them, and
            # writing a decoded coordinate back can change its dtype (a string
            # coordinate decodes to numpy's StringDType and is then rejected).
            # A filler with only dims aligns by position with the store.
            filler = xr.Dataset(
                {
                    name: (
                        dims,
                        np.full(shape, fills.get(name, np.nan), dtype=dtype),
                    )
                    for name, dtype in absent.items()
                }
            )
            encoding = self._append_encoding(
                append_dim, data=filler, append_dim_size=append_dim_size
            )
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
        """Reconcile every axis with the store at ``path``, then ``append``.

        The opt-in sibling of ``append`` for a panel that has grown. In
        order: the ``dim`` axis is widened to the sorted union of the stored
        and incoming labels and ``data`` is reindexed onto that axis; then
        any variable the store lacks is added with ``widen_data_vars``; then
        the unchanged ``append`` runs. Variables are widened after the symbol
        axis so the filler is built once at the final width. Calling this
        unconditionally is cheap: when both axes already agree, or when no
        store exists yet, it delegates straight to ``append``.

        The closing ``append`` is what keeps the compatibility checks in
        force. Because the widens commit before it runs, a window it then
        refuses (an overlapping ``append_dim`` range, say) can leave the
        store with the grown symbol axis and variable set while its append
        dimension and stored history are untouched. That is the accepted
        cost of inheriting the refusal rather than duplicating it.

        ``append_dim_size`` may ride in ``kwargs``. It is read, not consumed,
        so the same value reaches the symbol widen, the variable widen and
        the closing ``append``; those three cannot disagree on the chunk
        grid for what this call writes. Variables the store held from an
        earlier write keep whatever grid they were created with.

        Parameters
        ----------
        path : str
            Directory of the Zarr store.
        append_dim : str, default "timestamp"
            The dimension the store grows along.
        dim : str, default "symbol"
            The symbol axis to reconcile.
        fill_values : Mapping[str, object], optional
            Per-variable fill for non-floating variables, passed
            to both widens and used to reindex ``data``.
        **kwargs
            Passed through to ``append``.

        Examples
        --------
        A window carrying a symbol and a variable the store has not seen:

        >>> XrBackend().to_internal(window).widen_and_append(
        ...     "prices.zarr", fill_values={"flag": False}
        ... )
        XrBackend()
        >>> stored = xr.open_zarr("prices.zarr")
        >>> stored["symbol"].values.tolist(), sorted(stored.data_vars)
        (['AAA', 'BBB', 'CCC', 'DDD'], ['amount', 'close', 'flag', 'volume'])
        """
        # `get`, not `pop`: the closing `append` also needs the key in
        # `kwargs` when it creates the store.
        append_dim_size = kwargs.get("append_dim_size")

        if not Path(path).exists():
            return self.append(path, append_dim, **kwargs)

        # Labels keep their own type and `sort_symbol_axis` orders integers
        # numerically, so an integer axis is never sorted as strings.
        stored = xr.open_zarr(path)
        try:
            stored_labels = (
                list(stored[dim].values.tolist())
                if dim in stored.coords
                else []
            )
            stored_names = set(stored.data_vars)
        finally:
            stored.close()

        incoming = (
            list(self.data[dim].values.tolist())
            if dim in self.data.coords
            else []
        )
        # The filler must use the incoming dtype, or the closing `append`
        # refuses the window on a dtype mismatch.
        incoming_names = {
            str(name): variable.dtype
            for name, variable in self.data.data_vars.items()
        }
        union = sort_symbol_axis(set(stored_labels) | set(incoming))
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
                append_dim_size=append_dim_size,
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
                append_dim_size=append_dim_size,
            )

        return self.append(path, append_dim, **kwargs)

    def _append_encoding(
        self,
        append_dim: str,
        data: Optional[xr.Dataset] = None,
        *,
        append_dim_size: Optional[int] = None,
    ) -> dict:
        """Build the Zarr ``encoding`` that pins each variable's chunk shape.

        Along ``append_dim`` the chunk is ``min(APPEND_DIM_CHUNK, extent)``;
        along every other dimension it is the panel's full length. Variables
        that do not carry ``append_dim`` are left unencoded. Every creating
        write in this class routes through here so the chunk rule has one
        source.

        Parameters
        ----------
        append_dim : str
            The store's append dimension.
        data : xr.Dataset, optional
            The panel whose shape to encode; defaults to ``data``.
            The widen paths pass the widened panel so non-append
            dimensions are pinned at their widened length.
        append_dim_size : int, optional
            A lower bound on the store's total extent along
            ``append_dim``. The effective extent is
            ``max(panel length, append_dim_size)``, so a caller holding
            only a window of a larger store can raise the grid to what a
            whole-range write would have chosen, while a stated value
            narrower than the panel cannot shrink it. ``None`` uses the
            panel's own length.

        Returns
        -------
        dict
            A dict mapping variable names to ``{"chunks": (...)}`` entries.
        """
        panel = self.data if data is None else data
        encoding = {}
        for name, variable in panel.data_vars.items():
            if append_dim not in variable.dims:
                continue
            chunks = []
            for dim in variable.dims:
                if dim == append_dim and append_dim_size is not None:
                    # A lower bound only: shrinking the grid would be
                    # irreversible once a `mode="w"` rewrite pins it.
                    size = max(int(panel.sizes[dim]), int(append_dim_size))
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
        """Render one append-dimension label for an error message.

        A ``datetime64`` label reads as an ISO string; anything else falls
        back to ``str``, since ``append_dim`` need not be a timestamp.
        """
        if np.issubdtype(np.asarray(value).dtype, np.datetime64):
            return pd.Timestamp(value).isoformat()
        return str(value)

    def _assert_append_compatible(self, path: str, append_dim: str) -> None:
        """Raise before an append that would silently corrupt the store.

        Four checks run against the store at ``path``, in this order: every
        shared non-append coordinate must match exactly; the incoming window
        must start strictly after the stored end of ``append_dim`` (a gap is
        fine, overlap is not); every shared variable must keep its dtype;
        and the data-variable sets must be equal, with the missing direction
        reported before the added one because it has no remedy short of
        recomputing.

        Raises
        ------
        ValueError
            On the first check that fails.
        """
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
                        f"{len(stored)} stored). Zarr would overwrite the "
                        f"stored labels without complaint, silently assigning "
                        f"every previously written row to a different label. "
                        f"Fix the '{dim}' axis over the whole range before the "
                        f"first window, as "
                        f"BaseDataset.from_raw_data_chunked() does."
                    )
            # The append dimension itself; skipped when either side has no
            # coordinate on it or is empty, since then there is no order.
            if (
                append_dim in self.data.coords
                and append_dim in existing.coords
                and self.data[append_dim].size
                and existing[append_dim].size
            ):
                # min/max rather than positional indexing, so an unsorted
                # axis on either side cannot fool the comparison.
                incoming_start = self.data[append_dim].values.min()
                stored_end = existing[append_dim].values.max()
                # `<=`: a window starting exactly on the stored end would
                # duplicate that label.
                if incoming_start <= stored_end:
                    raise ValueError(
                        f"XrBackend.append: refusing to append to {path} -- "
                        f"the incoming '{append_dim}' window starts at "
                        f"{self._format_append_label(incoming_start)} but the "
                        f"store already ends at "
                        f"{self._format_append_label(stored_end)}. Zarr would "
                        f"extend the axis without complaint and leave "
                        f"'{append_dim}' no longer strictly increasing -- "
                        f"duplicate labels, out-of-order labels, or both -- "
                        f"which breaks every downstream reader that assumes a "
                        f"unique, ordered index. append() only extends a "
                        f"store; to recompute a range it already holds, "
                        f"replace the store with save(mode=\"w\") instead."
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
            # The variable set is checked after dtypes, so a dtype mismatch is
            # reported first. A missing variable is reported before an added
            # one because it has no widen remedy.
            incoming_names = set(self.data.data_vars)
            stored_names = set(existing.data_vars)
            absent = sorted(stored_names - incoming_names)
            if absent:
                raise ValueError(
                    f"XrBackend.append: refusing to append to {path} -- the "
                    f"store holds data variable(s) {absent} that the incoming "
                    f"panel does not. Zarr extends exactly the variables it "
                    f"is handed, so the absent one(s) would stay at their "
                    f"stored length while every other variable grows, and "
                    f"the store could no longer be opened (xarray reports "
                    f"conflicting sizes for dimension '{append_dim}'). There "
                    f"is no option to allow this: filling the absent variable "
                    f"with NaN over the incoming window would make a complete "
                    f"variable look as if recent values were genuinely "
                    f"missing. Recompute this window over the store's full "
                    f"variable set, or replace the store with "
                    f"save(mode=\"w\")."
                )
            unstored = sorted(incoming_names - stored_names)
            if unstored:
                raise ValueError(
                    f"XrBackend.append: refusing to append to {path} -- the "
                    f"incoming panel carries data variable(s) {unstored} that "
                    f"the store does not hold. Zarr would write them over the "
                    f"incoming window only, leaving them shorter along "
                    f"'{append_dim}' than every stored variable, and the "
                    f"store could no longer be opened (xarray reports "
                    f"conflicting sizes for dimension '{append_dim}'). If the "
                    f"panel really gained a column, first add the new "
                    f"variable(s) over the store's existing history with "
                    f"widen_data_vars(), or call widen_and_append(), which "
                    f"does that while reconciling every axis."
                )
        finally:
            existing.close()

    def to_internal(self, data: xr.Dataset) -> Self:
        """Adopt an in-memory ``xarray.Dataset`` as ``data``.

        Parameters
        ----------
        data : xr.Dataset
            The panel to hold.

        Examples
        --------
        >>> backend = XrBackend().to_internal(panel)
        >>> backend.data is panel
        True
        """
        self.data = data
        return self

    def filter_by_date(self, col: str, start_date: str, end_date: str) -> Self:
        """Narrow ``data`` in place to the label slice ``start_date..end_date``.

        Parameters
        ----------
        col : str
            The date dimension to slice.
        start_date : str
            First label to keep, inclusive.
        end_date : str
            Last label to keep, inclusive.

        Examples
        --------
        >>> backend.filter_by_date("timestamp", "2024-01-02", "2024-01-03")
        XrBackend()
        >>> backend.data["timestamp"].values.astype("datetime64[D]")
        array(['2024-01-02', '2024-01-03'], dtype='datetime64[D]')
        """
        self.data = self.data.sel({col: slice(start_date, end_date)})
        return self

    def filter_by_symbol(self, col: str, symbols: tuple[str, ...]) -> Self:
        """Narrow ``data`` in place to the given labels on ``col``.

        Parameters
        ----------
        col : str
            The dimension to select on.
        symbols : tuple[str, ...]
            The labels to keep.

        Examples
        --------
        >>> backend.filter_by_symbol("symbol", ("BBB",))
        XrBackend()
        >>> backend.data["symbol"].values.tolist()
        ['BBB']
        """
        self.data = self.data.sel({col: list(symbols)})
        return self

    def get_xarray_dataset(
        self, indexes: Optional[list[str]] = None
    ) -> xr.Dataset:
        """Return ``data`` indexed by exactly the dimensions in ``indexes``.

        ``indexes`` names the result's dimensions, in order: data variables
        laid out on any other dimension are dropped, dimensions no longer
        used are dropped with their coordinates, and the survivors are
        transposed onto the requested order. This is the same meaning
        ``PlBackend.get_xarray_dataset`` gives the argument. ``None`` returns
        ``data`` itself, not a copy, since callers rely on receiving the
        object the backend holds. ``data`` is never modified here; the
        ``filter_by_*`` methods are the in-place ones.

        Parameters
        ----------
        indexes : list[str], optional
            The dimensions to index by, or ``None`` for no shape
            request.

        Raises
        ------
        ValueError
            If a requested name is not a dimension of ``data``.

        Examples
        --------
        >>> ds = backend.get_xarray_dataset(["timestamp", "symbol"])
        >>> tuple(ds.dims)
        ('timestamp', 'symbol')
        >>> backend.get_xarray_dataset(["symbol", "timestamp"])["close"].dims
        ('symbol', 'timestamp')
        >>> backend.get_xarray_dataset() is backend.data
        True
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
        """Return ``data`` as a long-format ``polars.LazyFrame``.

        Examples
        --------
        >>> backend.get_lazyframe().collect().columns
        ['timestamp', 'symbol', 'close']
        """
        data = self.data.to_dataframe().reset_index()
        return pl.from_pandas(data).lazy()

    def head(self, path: str, n: int) -> pl.LazyFrame:
        """Return at most ``n`` rows of the store at ``path`` as a lazy frame.

        Parameters
        ----------
        path : str
            Directory of the Zarr store.
        n : int
            Maximum number of rows (and of labels per dimension) to read.

        The store is opened by path with the same opener ``read`` uses, and
        every dimension is sliced to ``n`` before conversion, so the whole
        store is never materialised. ``data`` is neither read nor written:
        a probe must not observe or inherit the in-place narrowing that
        ``filter_by_date`` leaves behind.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.

        Examples
        --------
        >>> XrBackend().head("prices.zarr", 2).collect().shape
        (2, 3)
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
    """Parquet-backed storage for a long-format table held as a lazy frame.

    ``data`` is a ``polars.LazyFrame`` produced by ``scan_parquet``, so reads
    and filters stay lazy until ``write`` or ``get_xarray_dataset`` collects
    them. Used for reference tables that are tabular rather than panel
    shaped.

    Examples
    --------
    >>> table = PlBackend().read("universe.parquet")
    >>> frame = table.filter_by_symbol("symbol", ("AAPL",)).get_lazyframe()
    >>> frame.collect()
    """

    def read(self, path: str, **kwargs) -> Self:
        """Lazily scan the Parquet file at ``path`` into ``data``.

        Parameters
        ----------
        path : str
            Path of the Parquet file.
        **kwargs
            Accepted for interface compatibility and ignored.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.

        Examples
        --------
        >>> table = PlBackend().read("universe.parquet")
        >>> type(table.data).__name__
        LazyFrame
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        self.data = pl.scan_parquet(path)
        return self

    def write(self, path: str, **kwargs) -> Self:
        """Collect ``data`` and write it to ``path`` as Parquet.

        Parameters
        ----------
        path : str
            Path of the Parquet file to write.
        **kwargs
            Passed through to ``polars.DataFrame.write_parquet``.

        Examples
        --------
        >>> PlBackend().to_internal(frame.lazy()).write("universe.parquet")
        PlBackend()
        """
        self.data.collect().write_parquet(path, **kwargs)
        return self

    def to_internal(self, data: pl.LazyFrame) -> Self:
        """Adopt an in-memory ``polars.LazyFrame`` as ``data``.

        Parameters
        ----------
        data : pl.LazyFrame
            The table to hold.

        Examples
        --------
        >>> PlBackend().to_internal(frame.lazy())
        PlBackend()
        """
        self.data = data
        return self

    def filter_by_date(self, col: str, start_date: str, end_date: str) -> Self:
        """Narrow ``data`` in place to rows whose ``col`` lies in the range.

        Parameters
        ----------
        col : str
            The date column to filter on.
        start_date : str
            First date to keep, inclusive.
        end_date : str
            Last date to keep, inclusive.

        Examples
        --------
        >>> table.filter_by_date("timestamp", "2024-01-02", "2024-01-03")
        PlBackend()
        >>> table.get_lazyframe().collect().height  # two days of two symbols
        4
        """
        self.data = self.data.filter(
            pl.col(col).is_between(
                pl.lit(pd.to_datetime(start_date)),
                pl.lit(pd.to_datetime(end_date)),
            )
        )
        return self

    def filter_by_symbol(self, col: str, symbols: tuple[str, ...]) -> Self:
        """Narrow ``data`` in place to rows whose ``col`` is in ``symbols``.

        Parameters
        ----------
        col : str
            The column to filter on.
        symbols : tuple[str, ...]
            The values to keep.

        Examples
        --------
        >>> table = PlBackend().read("universe.parquet")
        >>> table.filter_by_symbol("symbol", ("BBB",))
        PlBackend()
        >>> table.get_lazyframe().collect()["symbol"].unique().to_list()
        ['BBB']
        """
        self.data = self.data.filter(pl.col(col).is_in(symbols))
        return self

    def get_lazyframe(self) -> pl.LazyFrame:
        """Return the held lazy frame.

        Examples
        --------
        >>> table.get_lazyframe().collect().shape
        (8, 3)
        """
        return self.data

    def head(self, path: str, n: int) -> pl.LazyFrame:
        """Return at most ``n`` rows scanned lazily from ``path``.

        Parameters
        ----------
        path : str
            Path of the Parquet file.
        n : int
            Maximum number of rows to return.

        The limit is pushed down into the Parquet reader, and a fresh frame
        is returned without touching ``data``. The existence check is done
        here because ``scan_parquet`` on a missing file only fails at
        collect time.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.

        Examples
        --------
        >>> PlBackend().head("universe.parquet", 3).collect().shape
        (3, 3)
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        return pl.scan_parquet(path).head(n)

    def get_xarray_dataset(
        self, indexes: Optional[list[str]] = None
    ) -> xr.Dataset:
        """Collect ``data`` and convert it to a dataset indexed by ``indexes``.

        The named columns become the dataset's dimensions and every other
        column becomes a data variable.

        Parameters
        ----------
        indexes : list[str]
            The columns to index by. Required: a lazy frame has no
            dimensions to fall back on.

        Raises
        ------
        ValueError
            If ``indexes`` is ``None``.

        Examples
        --------
        >>> ds = table.get_xarray_dataset(["timestamp", "symbol"])
        >>> tuple(ds.dims), list(ds.data_vars)
        (('timestamp', 'symbol'), ['close'])
        """
        if indexes is None:
            raise ValueError(
                "PlBackend.get_xarray_dataset: `indexes` is required. A "
                "LazyFrame has no dimensions to fall back on; name the "
                "columns that should become the dataset's index, e.g. "
                '["timestamp", "symbol"].'
            )
        data = self.data.collect().to_pandas()
        data = data.set_index(indexes)
        return xr.Dataset.from_dataframe(data)

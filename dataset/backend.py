import os
import shutil
from pathlib import Path
from typing import Mapping, Optional, Self, Sequence

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from base.backend import DataBackend


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

    def write(self, path: str, **kwargs) -> Self:
        if not Path(path).exists():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        kwargs.setdefault("mode", "w")
        self.data.to_zarr(path, **kwargs)
        return self

    def append(self, path: str, append_dim: str = "timestamp", **kwargs) -> Self:
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
        across calls, and every shared data variable must keep its dtype.
        Raw `to_zarr(mode="a", append_dim=...)` enforces neither: a mismatched
        symbol coordinate is silently OVERWRITTEN with the new window's
        labels, leaving previously-written rows attributed to the wrong
        symbols, and an appended float64 NaN written into an int64 variable is
        silently cast to 0 -- a fabricated observation where data was missing.
        Both corruptions are invisible afterwards from the store alone, which
        is why they are checked here, before the irreversible append.

        `from_raw_data_chunked()` satisfies the coordinate half by pinning the
        symbol axis once over the whole range (D-02); this check is what turns
        that guarantee into an assertion.
        """
        target = Path(path)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            kwargs.setdefault("encoding", self._append_encoding(append_dim))
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

        **Documented cost.** dask is not installed here, so `xr.open_zarr`
        yields lazily-indexed arrays that `.load()` materialises in full -- this
        holds the WHOLE store in RAM, the exact allocation
        `BaseDataset.from_raw_data_chunked` exists to avoid. On a store too
        large to hold, reach for `on_new_listing="rebuild"` instead, which
        re-densifies window by window and additionally recovers the new
        listing's REAL history from raw rather than backfilling NaN.
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

            # `.load()` is load-bearing, not defensive: without dask,
            # `open_zarr` still hands back lazily-indexed arrays that read from
            # the store directory on access, and the swap below renames that
            # directory out from under them.
            widened = stored.reindex({dim: requested}, fill_value=fills).load()
            encoding = self._append_encoding(append_dim, data=widened)
        finally:
            stored.close()

        try:
            widened.to_zarr(str(widening), mode="w", encoding=encoding)
        except BaseException:
            shutil.rmtree(widening, ignore_errors=True)
            raise

        os.replace(target, superseded)
        os.replace(widening, target)
        shutil.rmtree(superseded, ignore_errors=True)
        return self

    def widen_and_append(
        self,
        path: str,
        append_dim: str = "timestamp",
        dim: str = "symbol",
        fill_values: Optional[Mapping[str, object]] = None,
        **kwargs,
    ) -> Self:
        """`append()`'s explicit opt-in sibling for a roster that has grown.

        Widens the store to `sorted(stored | incoming)`, reindexes `self.data`
        onto that same axis, and then calls the UNCHANGED `append()`.

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

        Calling this unconditionally is cheap: an unchanged axis skips the
        rewrite entirely and delegates straight to `append()`. An absent store
        does the same, so there is ONE creation path rather than two.
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
        finally:
            stored.close()

        incoming = (
            [str(label) for label in self.data[dim].values.tolist()]
            if dim in self.data.coords
            else []
        )
        union = sorted(set(stored_labels) | set(incoming))
        if union == stored_labels and union == incoming:
            return self.append(path, append_dim, **kwargs)

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
        return self.append(path, append_dim, **kwargs)

    def _append_encoding(
        self, append_dim: str, data: Optional[xr.Dataset] = None
    ) -> dict:
        """Pin each data variable's chunk shape: `APPEND_DIM_CHUNK` along the
        append dimension, the full length across every other one.

        `data` defaults to `self.data`, which is what `append()` passes
        implicitly. `widen_symbol_axis` passes the WIDENED panel instead, so
        the rewritten store's non-append dims are pinned to their widened
        length while `APPEND_DIM_CHUNK` still governs the append dimension.
        Routing the widen through this method rather than restating the chunk
        arithmetic is what keeps that rule single-sourced.
        """
        panel = self.data if data is None else data
        encoding = {}
        for name, variable in panel.data_vars.items():
            if append_dim not in variable.dims:
                continue
            chunks = []
            for dim in variable.dims:
                size = int(panel.sizes[dim])
                if dim == append_dim:
                    chunks.append(max(min(self.APPEND_DIM_CHUNK, size), 1))
                else:
                    chunks.append(max(size, 1))
            encoding[name] = {"chunks": tuple(chunks)}
        return encoding

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
        return self.data

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

    def get_xarray_dataset(self, indexes: list[str]) -> xr.Dataset:
        data = self.data.collect().to_pandas()
        data = data.set_index(indexes)
        return xr.Dataset.from_dataframe(data)
